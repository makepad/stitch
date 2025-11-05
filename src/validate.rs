use {
    crate::{
        instr::{BlockType, InstrDecoder, InstrDecoderAllocs, InstrVisitor, MemArg,},
        decode::DecodeError,
        func::{FuncType, UncompiledFuncBody},
        runtime::global::Mutability,
        module::ModuleBuilder,
        ops::*,
        ref_::RefType,
        val::{ValType, ValTypeOf}
    },
    std::{mem, ops::Deref},
};

#[derive(Clone, Debug)]
pub(crate) struct Validator {
    instr_decoder_allocs: InstrDecoderAllocs,
    br_table_label_idxs: Vec<u32>,
    typed_select_val_types: Vec<ValType>,
    locals: Vec<ValType>,
    blocks: Vec<Block>,
    opds: Vec<OpdType>,
    aux_opds: Vec<OpdType>,
}

impl Validator {
    pub(crate) fn new() -> Validator {
        Validator {
            instr_decoder_allocs: InstrDecoderAllocs::default(),
            br_table_label_idxs: Vec::new(),
            typed_select_val_types: Vec::new(),
            locals: Vec::new(),
            blocks: Vec::new(),
            opds: Vec::new(),
            aux_opds: Vec::new(),
        }
    }

    pub(crate) fn validate(
        &mut self,
        type_: &FuncType,
        module: &ModuleBuilder,
        code: &UncompiledFuncBody,
    ) -> Result<(), DecodeError> {
        self.locals.clear();
        self.blocks.clear();
        self.opds.clear();
        let mut validation = Validation {
            module,
            br_table_label_idxs: &mut self.br_table_label_idxs,
            typed_select_val_types: &mut self.typed_select_val_types,
            locals: &mut self.locals,
            blocks: &mut self.blocks,
            opds: &mut self.opds,
            aux_opds: &mut self.aux_opds,
        };
        validation.locals.extend(type_.params().iter().copied());
        validation.locals.extend(code.locals.iter().copied());
        validation.push_block(
            BlockKind::Block,
            FuncType::new([], type_.results().iter().copied()),
        );
        let mut decoder: InstrDecoder<'_> = InstrDecoder::new_with_allocs(&code.expr, mem::take(&mut self.instr_decoder_allocs));
        while !decoder.is_at_end() {
            decoder.decode(&mut validation)?;
        }
        self.instr_decoder_allocs = decoder.into_allocs();
        Ok(())
    }
}

impl Default for Validator {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
struct Validation<'a> {
    module: &'a ModuleBuilder,
    br_table_label_idxs: &'a mut Vec<u32>,
    typed_select_val_types: &'a mut Vec<ValType>,
    locals: &'a mut Vec<ValType>,
    blocks: &'a mut Vec<Block>,
    opds: &'a mut Vec<OpdType>,
    aux_opds: &'a mut Vec<OpdType>,
}

impl<'a> Validation<'a> {
    fn validate_select(&mut self, type_: Option<ValType>) -> Result<(), DecodeError> {
        if let Some(type_) = type_ {
            self.pop_opd()?.check(ValType::I32)?;
            self.pop_opd()?.check(type_)?;
            self.pop_opd()?.check(type_)?;
            self.push_opd(type_);
        } else {
            self.pop_opd()?.check(ValType::I32)?;
            let input_type_1 = self.pop_opd()?;
            let input_type_0 = self.pop_opd()?;
            if !(input_type_0.is_num() && input_type_1.is_num()) {
                return Err(DecodeError::new("type mismatch"));
            }
            if let OpdType::ValType(input_type_1) = input_type_1 {
                input_type_0.check(input_type_1)?;
            }
            self.push_opd(if input_type_0.is_unknown() {
                input_type_1
            } else {
                input_type_0
            });
        }
        Ok(())
    }

    fn validate_load<T>(&mut self, arg: MemArg) -> Result<(), DecodeError>
    where
        T: ValTypeOf,
    {
        self.validate_load_inner(arg, T::val_type_of(), align_of::<T>().ilog(2))
    }

    fn validate_load_n<Dst, Src>(&mut self, arg: MemArg) -> Result<(), DecodeError>
    where
        Dst: ValTypeOf,
    {
        self.validate_load_inner(arg, Dst::val_type_of(), align_of::<Src>().ilog(2))
    }

    fn validate_load_inner(&mut self, arg: MemArg, output_type: ValType, max_align: u32) -> Result<(), DecodeError> {
        if arg.align > max_align {
            return Err(DecodeError::new("alignment too large"));
        }
        self.module.memory(0)?;
        self.pop_opd()?.check(ValType::I32)?;
        self.push_opd(output_type);
        Ok(())
    }

    fn validate_store<T>(&mut self, arg: MemArg) -> Result<(), DecodeError>
    where
        T: ValTypeOf,
    {
        self.validate_store_inner(arg, T::val_type_of(), align_of::<T>().ilog(2))
    }

    fn validate_store_n<Src, Dst>(&mut self, arg: MemArg) -> Result<(), DecodeError>
    where
        Src: ValTypeOf
    {
        self.validate_store_inner(
            arg,
            Src::val_type_of(),
            align_of::<Dst>().ilog(2),
        )
    }

    fn validate_store_inner(&mut self, arg: MemArg, input_type: ValType, max_align: u32) -> Result<(), DecodeError> {
        if arg.align > max_align {
            return Err(DecodeError::new("alignment too large"));
        }
        self.module.memory(0)?;
        self.pop_opd()?.check(input_type)?;
        self.pop_opd()?.check(ValType::I32)?;
        Ok(())
    }

    fn validate_un_op<T, U>(&mut self) -> Result<(), DecodeError>
    where
        T: ValTypeOf,
        U: UnOp<T>,
        U::Output: ValTypeOf,
    {
        self.validate_un_op_inner(T::val_type_of(), U::Output::val_type_of())
    }

    fn validate_un_op_inner(&mut self, input_type: ValType, output_type: ValType) -> Result<(), DecodeError> {
        self.pop_opd()?.check(input_type)?;
        self.push_opd(output_type);
        Ok(())
    }

    fn validate_bin_op<T, B>(&mut self) -> Result<(), DecodeError>
    where
        T: ValTypeOf,
        B: BinOp<T>,
        B::Output: ValTypeOf,
    {
        self.validate_bin_op_inner(T::val_type_of(), T::val_type_of(), B::Output::val_type_of())
    }

    fn validate_bin_op_inner(&mut self, input_type_0: ValType, input_type_1: ValType, output_type: ValType) -> Result<(), DecodeError> {
        self.pop_opd()?.check(input_type_1)?;
        self.pop_opd()?.check(input_type_0)?;
        self.push_opd(output_type);
        Ok(())
    }

    fn resolve_block_type(&self, type_: BlockType) -> Result<FuncType, DecodeError> {
        match type_ {
            BlockType::TypeIdx(idx) => self.module.type_(idx).cloned(),
            BlockType::ValType(val_type) => Ok(FuncType::from_val_type(val_type)),
        }
    }

    fn local(&self, idx: u32) -> Result<ValType, DecodeError> {
        self.locals
            .get(idx as usize)
            .copied()
            .ok_or_else(|| DecodeError::new("unknown local"))
    }

    fn label(&self, idx: u32) -> Result<(), DecodeError> {
        let idx = usize::try_from(idx).unwrap();
        if idx >= self.blocks.len() {
            return Err(DecodeError::new("unknown label"));
        }
        Ok(())
    }

    fn block(&self, idx: u32) -> &Block {
        let idx = usize::try_from(idx).unwrap();
        &self.blocks[self.blocks.len() - 1 - idx]
    }

    fn push_block(&mut self, kind: BlockKind, type_: FuncType) {
        self.blocks.push(Block {
            kind,
            type_,
            is_unreachable: false,
            height: self.opds.len(),
        });
        for start_type in self.block(0).type_.clone().params().iter().copied() {
            self.push_opd(start_type);
        }
    }

    fn pop_block(&mut self) -> Result<Block, DecodeError> {
        for end_type in self.block(0).type_.clone().results().iter().rev().copied() {
            self.pop_opd()?.check(end_type)?;
        }
        if self.opds.len() != self.block(0).height {
            return Err(DecodeError::new("type mismatch"));
        }
        Ok(self.blocks.pop().unwrap())
    }

    fn set_unreachable(&mut self) {
        self.opds.truncate(self.block(0).height);
        self.blocks.last_mut().unwrap().is_unreachable = true;
    }

    fn push_opd(&mut self, type_: impl Into<OpdType>) {
        let type_ = type_.into();
        self.opds.push(type_);
    }

    fn pop_opd(&mut self) -> Result<OpdType, DecodeError> {
        if self.opds.len() == self.block(0).height {
            if !self.block(0).is_unreachable {
                return Err(DecodeError::new("type mismatch"));
            }
            Ok(OpdType::Unknown)
        } else {
            Ok(self.opds.pop().unwrap())
        }
    }
}

impl<'a> InstrVisitor for Validation<'a> {
    type Ok = ();
    type Error = DecodeError;

    // Control instructions
    fn visit_nop(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }

    fn visit_unreachable(&mut self) -> Result<(), Self::Error> {
        self.set_unreachable();
        Ok(())
    }

    fn visit_block(&mut self, type_: BlockType) -> Result<(), Self::Error> {
        let type_ = self.resolve_block_type(type_)?;
        for start_type in type_.params().iter().rev().copied() {
            self.pop_opd()?.check(start_type)?;
        }
        self.push_block(BlockKind::Block, type_);
        Ok(())
    }

    fn visit_loop(&mut self, type_: BlockType) -> Result<(), Self::Error> {
        let type_ = self.resolve_block_type(type_)?;
        for start_type in type_.params().iter().rev().copied() {
            self.pop_opd()?.check(start_type)?;
        }
        self.push_block(BlockKind::Loop, type_);
        Ok(())
    }

    fn visit_if(&mut self, type_: BlockType) -> Result<(), Self::Error> {
        let type_ = self.resolve_block_type(type_)?;
        self.pop_opd()?.check(ValType::I32)?;
        for start_type in type_.params().iter().rev().copied() {
            self.pop_opd()?.check(start_type)?;
        }
        self.push_block(BlockKind::If, type_);
        Ok(())
    }

    fn visit_else(&mut self) -> Result<(), Self::Error> {
        let block = self.pop_block()?;
        if block.kind != BlockKind::If {
            return Err(DecodeError::new("unexpected else opcode"));
        }
        self.push_block(BlockKind::Else, block.type_);
        Ok(())
    }

    fn visit_end(&mut self) -> Result<(), Self::Error> {
        let block = self.pop_block()?;
        let block = if block.kind == BlockKind::If {
            self.push_block(BlockKind::Else, block.type_);
            self.pop_block()?
        } else {
            block
        };
        for end_type in block.type_.results().iter().copied() {
            self.push_opd(end_type);
        }
        Ok(())
    }

    fn visit_br(&mut self, label_idx: u32) -> Result<(), Self::Error> {
        self.label(label_idx)?;
        for label_type in self.block(label_idx).label_types().iter().rev().copied() {
            self.pop_opd()?.check(label_type)?;
        }
        self.set_unreachable();
        Ok(())
    }

    fn visit_br_if(&mut self, label_idx: u32) -> Result<(), Self::Error> {
        self.pop_opd()?.check(ValType::I32)?;
        self.label(label_idx)?;
        for &label_type in self.block(label_idx).label_types().iter().rev() {
            self.pop_opd()?.check(label_type)?;
        }
        for &label_type in self.block(label_idx).label_types().iter() {
            self.push_opd(label_type);
        }
        Ok(())
    }
    
    fn visit_br_table_start(&mut self) -> Result<(), Self::Error> {
        self.br_table_label_idxs.clear();
        Ok(())
    }

    fn visit_br_table_label(&mut self, label_idx: u32) -> Result<(), Self::Error> {
        self.br_table_label_idxs.push(label_idx);
        Ok(())
    }

    fn visit_br_table_end(
        &mut self,
        default_label_idx: u32,
    ) -> Result<(), Self::Error> {
        let label_idxs = mem::take(self.br_table_label_idxs);

        self.pop_opd()?.check(ValType::I32)?;
        self.label(default_label_idx)?;
        let arity = self.block(default_label_idx).label_types().len();
        for label_idx in label_idxs.iter().copied() {
            self.label(label_idx)?;
            if self.block(label_idx).label_types().len() != arity {
                return Err(DecodeError::new("arity mismatch"));
            }
            let mut aux_opds = mem::take(self.aux_opds);
            for label_type in self.block(label_idx).label_types().iter().rev().copied() {
                let opd = self.pop_opd()?;
                opd.check(label_type)?;
                aux_opds.push(opd);
            }
            while let Some(opd) = aux_opds.pop() {
                self.push_opd(opd);
            }
            *self.aux_opds = aux_opds;
        }
        for label_type in self
            .block(default_label_idx)
            .label_types()
            .iter()
            .rev()
            .copied()
        {
            self.pop_opd()?.check(label_type)?;
        }
        self.set_unreachable();
        *self.br_table_label_idxs = label_idxs;
        Ok(())
    }

    fn visit_return(&mut self) -> Result<(), Self::Error> {
        self.visit_br(self.blocks.len() as u32 - 1)
    }

    fn visit_call(&mut self, func_idx: u32) -> Result<(), Self::Error> {
        let type_ = self.module.func(func_idx)?;
        for param_type in type_.params().iter().rev().copied() {
            self.pop_opd()?.check(param_type)?;
        }
        for result_type in type_.results().iter().copied() {
            self.push_opd(result_type);
        }
        Ok(())
    }

    fn visit_call_indirect(&mut self, table_idx: u32, type_idx: u32) -> Result<(), Self::Error> {
        let table_type = self.module.table(table_idx)?;
        if table_type.elem != RefType::FuncRef {
            return Err(DecodeError::new("type mismatch"));
        }
        let type_ = self.module.type_(type_idx)?;
        self.pop_opd()?.check(ValType::I32)?;
        for param_type in type_.params().iter().rev().copied() {
            self.pop_opd()?.check(param_type)?;
        }
        for result_type in type_.results().iter().copied() {
            self.push_opd(result_type);
        }
        Ok(())
    }

    // Reference instructions
    fn visit_ref_null(&mut self, type_: RefType) -> Result<(), Self::Error> {
        self.push_opd(type_);
        Ok(())
    }

    fn visit_ref_is_null(&mut self) -> Result<(), Self::Error> {
        if !self.pop_opd()?.is_ref() {
            return Err(DecodeError::new("type mismatch"));
        };
        self.push_opd(ValType::I32);
        Ok(())
    }

    fn visit_ref_func(&mut self, func_idx: u32) -> Result<(), Self::Error> {
        self.module.ref_(func_idx)?;
        self.push_opd(ValType::FuncRef);
        Ok(())
    }

    // Parametric instructions
    fn visit_drop(&mut self) -> Result<(), Self::Error> {
        self.pop_opd()?;
        Ok(())
    }

    fn visit_select(&mut self) -> Result<(), Self::Error> {
        self.validate_select(None)
    }

    fn visit_typed_select_start(&mut self) -> Result<(), Self::Error> {
        self.typed_select_val_types.clear();
        Ok(())
    }

    fn visit_typed_select_val_type(&mut self, val_type: ValType) -> Result<(), Self::Error> {
        self.typed_select_val_types.push(val_type);
        Ok(())
    }

    fn visit_typed_select_end(&mut self) -> Result<(), Self::Error> {
        let val_type = self.typed_select_val_types.pop().unwrap();
        self.validate_select(Some(val_type))
    }

    // Variable instructions
    fn visit_local_get(&mut self, local_idx: u32) -> Result<(), Self::Error> {
        let type_ = self.local(local_idx)?;
        self.push_opd(type_);
        Ok(())
    }

    fn visit_local_set(&mut self, local_idx: u32) -> Result<(), Self::Error> {
        let type_ = self.local(local_idx)?;
        self.pop_opd()?.check(type_)?;
        Ok(())
    }

    fn visit_local_tee(&mut self, local_idx: u32) -> Result<(), Self::Error> {
        let type_ = self.local(local_idx)?;
        self.pop_opd()?.check(type_)?;
        self.push_opd(type_);
        Ok(())
    }

    fn visit_global_get(&mut self, global_idx: u32) -> Result<(), Self::Error> {
        let type_ = self.module.global(global_idx)?;
        self.push_opd(type_.content());
        Ok(())
    }

    fn visit_global_set(&mut self, global_idx: u32) -> Result<(), Self::Error> {
        let type_ = self.module.global(global_idx)?;
        if type_.mutability() != Mutability::Var {
            return Err(DecodeError::new("type mismatch"));
        }
        self.pop_opd()?.check(type_.content())?;
        Ok(())
    }

    // Table instructions
    fn visit_table_get(&mut self, table_idx: u32) -> Result<(), Self::Error> {
        let type_ = self.module.table(table_idx)?;
        self.pop_opd()?.check(ValType::I32)?;
        self.push_opd(type_.elem);
        Ok(())
    }

    fn visit_table_set(&mut self, table_idx: u32) -> Result<(), Self::Error> {
        let type_ = self.module.table(table_idx)?;
        self.pop_opd()?.check(type_.elem)?;
        self.pop_opd()?.check(ValType::I32)?;
        Ok(())
    }

    fn visit_table_size(&mut self, table_idx: u32) -> Result<(), Self::Error> {
        self.module.table(table_idx)?;
        self.push_opd(ValType::I32);
        Ok(())
    }

    fn visit_table_grow(&mut self, table_idx: u32) -> Result<(), Self::Error> {
        let type_ = self.module.table(table_idx)?;
        self.pop_opd()?.check(ValType::I32)?;
        self.pop_opd()?.check(type_.elem)?;
        self.push_opd(ValType::I32);
        Ok(())
    }

    fn visit_table_fill(&mut self, table_idx: u32) -> Result<(), Self::Error> {
        let type_ = self.module.table(table_idx)?;
        self.pop_opd()?.check(ValType::I32)?;
        self.pop_opd()?.check(type_.elem)?;
        self.pop_opd()?.check(ValType::I32)?;
        Ok(())
    }

    fn visit_table_copy(
        &mut self,
        dst_table_idx: u32,
        src_table_idx: u32,
    ) -> Result<(), Self::Error> {
        let dst_type = self.module.table(dst_table_idx)?;
        let src_type = self.module.table(src_table_idx)?;
        if dst_type.elem != src_type.elem {
            return Err(DecodeError::new("type mismatch"));
        }
        self.pop_opd()?.check(ValType::I32)?;
        self.pop_opd()?.check(ValType::I32)?;
        self.pop_opd()?.check(ValType::I32)?;
        Ok(())
    }

    fn visit_table_init(&mut self, table_idx: u32, elem_idx: u32) -> Result<(), Self::Error> {
        let dst_type = self.module.table(table_idx)?;
        let src_type = self.module.elem(elem_idx)?;
        if dst_type.elem != src_type {
            return Err(DecodeError::new("type mismatch"));
        }
        self.pop_opd()?.check(ValType::I32)?;
        self.pop_opd()?.check(ValType::I32)?;
        self.pop_opd()?.check(ValType::I32)?;
        Ok(())
    }

    fn visit_elem_drop(&mut self, elem_idx: u32) -> Result<(), Self::Error> {
        self.module.elem(elem_idx)?;
        Ok(())
    }

    // Memory instructions
    fn visit_i32_load(&mut self, arg: MemArg) -> Result<(), Self::Error> {
        self.validate_load::<i32>(arg)
    }

    fn visit_i64_load(&mut self, arg: MemArg) -> Result<(), Self::Error> {
        self.validate_load::<i64>(arg)
    }

    fn visit_f32_load(&mut self, arg: MemArg) -> Result<(), Self::Error> {
        self.validate_load::<f32>(arg)
    }

    fn visit_f64_load(&mut self, arg: MemArg) -> Result<(), Self::Error> {
        self.validate_load::<f64>(arg)
    }

    fn visit_i32_load8_s(&mut self, arg: MemArg) -> Result<(), Self::Error> {
        self.validate_load_n::<i32, i8>(arg)
    }

    fn visit_i32_load8_u(&mut self, arg: MemArg) -> Result<(), Self::Error> {
        self.validate_load_n::<u32, u8>(arg)
    }

    fn visit_i32_load16_s(&mut self, arg: MemArg) -> Result<(), Self::Error> {
        self.validate_load_n::<i32, i16>(arg)
    }

    fn visit_i32_load16_u(&mut self, arg: MemArg) -> Result<(), Self::Error> {
        self.validate_load_n::<u32, u16>(arg)
    }

    fn visit_i64_load8_s(&mut self, arg: MemArg) -> Result<(), Self::Error> {
        self.validate_load_n::<i64, i8>(arg)
    }

    fn visit_i64_load8_u(&mut self, arg: MemArg) -> Result<(), Self::Error> {
        self.validate_load_n::<u64, u8>(arg)
    }

    fn visit_i64_load16_s(&mut self, arg: MemArg) -> Result<(), Self::Error> {
        self.validate_load_n::<i64, i16>(arg)
    }

    fn visit_i64_load16_u(&mut self, arg: MemArg) -> Result<(), Self::Error> {
        self.validate_load_n::<u64, u16>(arg)
    }

    fn visit_i64_load32_s(&mut self, arg: MemArg) -> Result<(), Self::Error> {
        self.validate_load_n::<i64, i32>(arg)
    }

    fn visit_i64_load32_u(&mut self, arg: MemArg) -> Result<(), Self::Error> {
        self.validate_load_n::<u64, u32>(arg)
    }

    fn visit_i32_store(&mut self, arg: MemArg) -> Result<(), Self::Error> {
        self.validate_store::<i32>(arg)
    }

    fn visit_i64_store(&mut self, arg: MemArg) -> Result<(), Self::Error> {
        self.validate_store::<i64>(arg)
    }

    fn visit_f32_store(&mut self, arg: MemArg) -> Result<(), Self::Error> {
        self.validate_store::<f32>(arg)
    }

    fn visit_f64_store(&mut self, arg: MemArg) -> Result<(), Self::Error> {
        self.validate_store::<f64>(arg)
    }

    fn visit_i32_store8(&mut self, arg: MemArg) -> Result<(), Self::Error> {
        self.validate_store_n::<i32, i8>(arg)
    }

    fn visit_i32_store16(&mut self, arg: MemArg) -> Result<(), Self::Error> {
        self.validate_store_n::<i32, i16>(arg)
    }

    fn visit_i64_store8(&mut self, arg: MemArg) -> Result<(), Self::Error> {
        self.validate_store_n::<i64, i8>(arg)
    }

    fn visit_i64_store16(&mut self, arg: MemArg) -> Result<(), Self::Error> {
        self.validate_store_n::<i64, i16>(arg)
    }

    fn visit_i64_store32(&mut self, arg: MemArg) -> Result<(), Self::Error> {
        self.validate_store_n::<i64, i32>(arg)
    }

    fn visit_memory_size(&mut self) -> Result<(), Self::Error> {
        self.module.memory(0)?;
        self.push_opd(ValType::I32);
        Ok(())
    }

    fn visit_memory_grow(&mut self) -> Result<(), Self::Error> {
        self.module.memory(0)?;
        self.pop_opd()?.check(ValType::I32)?;
        self.push_opd(ValType::I32);
        Ok(())
    }

    fn visit_memory_fill(&mut self) -> Result<(), Self::Error> {
        self.module.memory(0)?;
        self.pop_opd()?.check(ValType::I32)?;
        self.pop_opd()?.check(ValType::I32)?;
        self.pop_opd()?.check(ValType::I32)?;
        Ok(())
    }

    fn visit_memory_copy(&mut self) -> Result<(), Self::Error> {
        self.module.memory(0)?;
        self.pop_opd()?.check(ValType::I32)?;
        self.pop_opd()?.check(ValType::I32)?;
        self.pop_opd()?.check(ValType::I32)?;
        Ok(())
    }

    fn visit_memory_init(&mut self, data_idx: u32) -> Result<(), Self::Error> {
        self.module.memory(0)?;
        self.module.data(data_idx)?;
        self.pop_opd()?.check(ValType::I32)?;
        self.pop_opd()?.check(ValType::I32)?;
        self.pop_opd()?.check(ValType::I32)?;
        Ok(())
    }

    fn visit_data_drop(&mut self, data_idx: u32) -> Result<(), Self::Error> {
        self.module.data(data_idx)?;
        Ok(())
    }

    // Numeric instructions
    fn visit_i32_const(&mut self, _val: i32) -> Result<(), Self::Error> {
        self.push_opd(ValType::I32);
        Ok(())
    }

    fn visit_i64_const(&mut self, _val: i64) -> Result<(), Self::Error> {
        self.push_opd(ValType::I64);
        Ok(())
    }

    fn visit_f32_const(&mut self, _val: f32) -> Result<(), Self::Error> {
        self.push_opd(ValType::F32);
        Ok(())
    }

    fn visit_f64_const(&mut self, _val: f64) -> Result<(), Self::Error> {
        self.push_opd(ValType::F64);
        Ok(())
    }

    fn visit_i32_eqz(&mut self) -> Result<(), DecodeError> {
        self.validate_un_op::<i32, Eqz>()
    }

    fn visit_i32_eq(&mut self) -> Result<(), DecodeError> {
        self.validate_bin_op::<i32, Eq>()
    }

    fn visit_i32_ne(&mut self) -> Result<(), DecodeError> {
        self.validate_bin_op::<i32, Ne>()
    }

    fn visit_i32_lt_s(&mut self) -> Result<(), DecodeError> {
        self.validate_bin_op::<i32, Lt>()
    }

    fn visit_i32_lt_u(&mut self) -> Result<(), DecodeError> {
        self.validate_bin_op::<u32, Lt>()
    }

    fn visit_i32_gt_s(&mut self) -> Result<(), DecodeError> {
        self.validate_bin_op::<i32, Gt>()
    }

    fn visit_i32_gt_u(&mut self) -> Result<(), DecodeError> {
        self.validate_bin_op::<u32, Gt>()
    }

    fn visit_i32_le_s(&mut self) -> Result<(), DecodeError> {
        self.validate_bin_op::<i32, Le>()
    }

    fn visit_i32_le_u(&mut self) -> Result<(), DecodeError> {
        self.validate_bin_op::<u32, Le>()
    }

    fn visit_i32_ge_s(&mut self) -> Result<(), DecodeError> {
        self.validate_bin_op::<i32, Ge>()
    }

    fn visit_i32_ge_u(&mut self) -> Result<(), DecodeError> {
        self.validate_bin_op::<u32, Ge>()
    }

    fn visit_i64_eqz(&mut self) -> Result<(), DecodeError> {
        self.validate_un_op::<i64, Eqz>()
    }

    fn visit_i64_eq(&mut self) -> Result<(), DecodeError> {
        self.validate_bin_op::<i64, Eq>()
    }

    fn visit_i64_ne(&mut self) -> Result<(), DecodeError> {
        self.validate_bin_op::<i64, Ne>()
    }

    fn visit_i64_lt_s(&mut self) -> Result<(), DecodeError> {
        self.validate_bin_op::<i64, Lt>()
    }

    fn visit_i64_lt_u(&mut self) -> Result<(), DecodeError> {
        self.validate_bin_op::<u64, Lt>()
    }

    fn visit_i64_gt_s(&mut self) -> Result<(), DecodeError> {
        self.validate_bin_op::<i64, Gt>()
    }

    fn visit_i64_gt_u(&mut self) -> Result<(), DecodeError> {
        self.validate_bin_op::<u64, Gt>()
    }

    fn visit_i64_le_s(&mut self) -> Result<(), DecodeError> {
        self.validate_bin_op::<i64, Le>()
    }

    fn visit_i64_le_u(&mut self) -> Result<(), DecodeError> {
        self.validate_bin_op::<u64, Le>()
    }

    fn visit_i64_ge_s(&mut self) -> Result<(), DecodeError> {
        self.validate_bin_op::<i64, Ge>()
    }

    fn visit_i64_ge_u(&mut self) -> Result<(), DecodeError> {
        self.validate_bin_op::<u64, Ge>()
    }

    fn visit_f32_eq(&mut self) -> Result<(), DecodeError> {
        self.validate_bin_op::<f32, Eq>()
    }

    fn visit_f32_ne(&mut self) -> Result<(), DecodeError> {
        self.validate_bin_op::<f32, Ne>()
    }

    fn visit_f32_lt(&mut self) -> Result<(), DecodeError> {
        self.validate_bin_op::<f32, Lt>()
    }

    fn visit_f32_gt(&mut self) -> Result<(), DecodeError> {
        self.validate_bin_op::<f32, Gt>()
    }

    fn visit_f32_le(&mut self) -> Result<(), DecodeError> {
        self.validate_bin_op::<f32, Le>()
    }

    fn visit_f32_ge(&mut self) -> Result<(), DecodeError> {
        self.validate_bin_op::<f32, Ge>()
    }

    fn visit_f64_eq(&mut self) -> Result<(), DecodeError> {
        self.validate_bin_op::<f64, Eq>()
    }

    fn visit_f64_ne(&mut self) -> Result<(), DecodeError> {
        self.validate_bin_op::<f64, Ne>()
    }

    fn visit_f64_lt(&mut self) -> Result<(), DecodeError> {
        self.validate_bin_op::<f64, Lt>()
    }

    fn visit_f64_gt(&mut self) -> Result<(), DecodeError> {
        self.validate_bin_op::<f64, Gt>()
    }

    fn visit_f64_le(&mut self) -> Result<(), DecodeError> {
        self.validate_bin_op::<f64, Le>()
    }

    fn visit_f64_ge(&mut self) -> Result<(), DecodeError> {
        self.validate_bin_op::<f64, Ge>()
    }

    fn visit_i32_clz(&mut self) -> Result<(), Self::Error> {
        self.validate_un_op::<i32, Clz>()
    }

    fn visit_i32_ctz(&mut self) -> Result<(), Self::Error> {
        self.validate_un_op::<i32, Ctz>()
    }

    fn visit_i32_popcnt(&mut self) -> Result<(), Self::Error> {
        self.validate_un_op::<i32, Popcnt>()
    }

    fn visit_i32_add(&mut self) -> Result<(), Self::Error> {
        self.validate_bin_op::<i32, Add>()
    }

    fn visit_i32_sub(&mut self) -> Result<(), Self::Error> {
        self.validate_bin_op::<i32, Sub>()
    }

    fn visit_i32_mul(&mut self) -> Result<(), Self::Error> {
        self.validate_bin_op::<i32, Mul>()
    }

    fn visit_i32_div_s(&mut self) -> Result<(), Self::Error> {
        self.validate_bin_op::<i32, Div>()
    }

    fn visit_i32_div_u(&mut self) -> Result<(), Self::Error> {
        self.validate_bin_op::<u32, Div>()
    }

    fn visit_i32_rem_s(&mut self) -> Result<(), Self::Error> {
        self.validate_bin_op::<i32, Rem>()
    }

    fn visit_i32_rem_u(&mut self) -> Result<(), Self::Error> {
        self.validate_bin_op::<u32, Rem>()
    }

    fn visit_i32_and(&mut self) -> Result<(), Self::Error> {
        self.validate_bin_op::<i32, And>()
    }

    fn visit_i32_or(&mut self) -> Result<(), Self::Error> {
        self.validate_bin_op::<i32, Or>()
    }

    fn visit_i32_xor(&mut self) -> Result<(), Self::Error> {
        self.validate_bin_op::<i32, Xor>()
    }

    fn visit_i32_shl(&mut self) -> Result<(), Self::Error> {
        self.validate_bin_op::<i32, Shl>()
    }

    fn visit_i32_shr_s(&mut self) -> Result<(), Self::Error> {
        self.validate_bin_op::<i32, Shr>()
    }

    fn visit_i32_shr_u(&mut self) -> Result<(), Self::Error> {
        self.validate_bin_op::<u32, Shr>()
    }

    fn visit_i32_rotl(&mut self) -> Result<(), Self::Error> {
        self.validate_bin_op::<i32, Rotl>()
    }

    fn visit_i32_rotr(&mut self) -> Result<(), Self::Error> {
        self.validate_bin_op::<i32, Rotr>()
    }

    fn visit_i64_clz(&mut self) -> Result<(), Self::Error> {
        self.validate_un_op::<i64, Clz>()
    }

    fn visit_i64_ctz(&mut self) -> Result<(), Self::Error> {
        self.validate_un_op::<i64, Ctz>()
    }

    fn visit_i64_popcnt(&mut self) -> Result<(), Self::Error> {
        self.validate_un_op::<i64, Popcnt>()
    }

    fn visit_i64_add(&mut self) -> Result<(), Self::Error> {
        self.validate_bin_op::<i64, Add>()
    }

    fn visit_i64_sub(&mut self) -> Result<(), Self::Error> {
        self.validate_bin_op::<i64, Sub>()
    }

    fn visit_i64_mul(&mut self) -> Result<(), Self::Error> {
        self.validate_bin_op::<i64, Mul>()
    }

    fn visit_i64_div_s(&mut self) -> Result<(), Self::Error> {
        self.validate_bin_op::<i64, Div>()
    }

    fn visit_i64_div_u(&mut self) -> Result<(), Self::Error> {
        self.validate_bin_op::<u64, Div>()
    }

    fn visit_i64_rem_s(&mut self) -> Result<(), Self::Error> {
        self.validate_bin_op::<i64, Rem>()
    }

    fn visit_i64_rem_u(&mut self) -> Result<(), Self::Error> {
        self.validate_bin_op::<u64, Rem>()
    }

    fn visit_i64_and(&mut self) -> Result<(), Self::Error> {
        self.validate_bin_op::<i64, And>()
    }

    fn visit_i64_or(&mut self) -> Result<(), Self::Error> {
        self.validate_bin_op::<i64, Or>()
    }

    fn visit_i64_xor(&mut self) -> Result<(), Self::Error> {
        self.validate_bin_op::<i64, Xor>()
    }

    fn visit_i64_shl(&mut self) -> Result<(), Self::Error> {
        self.validate_bin_op::<i64, Shl>()
    }

    fn visit_i64_shr_s(&mut self) -> Result<(), Self::Error> {
        self.validate_bin_op::<i64, Shr>()
    }

    fn visit_i64_shr_u(&mut self) -> Result<(), Self::Error> {
        self.validate_bin_op::<u64, Shr>()
    }

    fn visit_i64_rotl(&mut self) -> Result<(), Self::Error> {
        self.validate_bin_op::<i64, Rotl>()
    }

    fn visit_i64_rotr(&mut self) -> Result<(), Self::Error> {
        self.validate_bin_op::<i64, Rotr>()
    }

    fn visit_f32_abs(&mut self) -> Result<(), Self::Error> {
        self.validate_un_op::<f32, Abs>()
    }

    fn visit_f32_neg(&mut self) -> Result<(), Self::Error> {
        self.validate_un_op::<f32, Neg>()
    }

    fn visit_f32_ceil(&mut self) -> Result<(), Self::Error> {
        self.validate_un_op::<f32, Ceil>()
    }

    fn visit_f32_floor(&mut self) -> Result<(), Self::Error> {
        self.validate_un_op::<f32, Floor>()
    }

    fn visit_f32_trunc(&mut self) -> Result<(), Self::Error> {
        self.validate_un_op::<f32, Trunc>()
    }

    fn visit_f32_nearest(&mut self) -> Result<(), Self::Error> {
        self.validate_un_op::<f32, Nearest>()
    }

    fn visit_f32_sqrt(&mut self) -> Result<(), Self::Error> {
        self.validate_un_op::<f32, Sqrt>()
    }

    fn visit_f32_add(&mut self) -> Result<(), Self::Error> {
        self.validate_bin_op::<f32, Add>()
    }

    fn visit_f32_sub(&mut self) -> Result<(), Self::Error> {
        self.validate_bin_op::<f32, Sub>()
    }

    fn visit_f32_mul(&mut self) -> Result<(), Self::Error> {
        self.validate_bin_op::<f32, Mul>()
    }

    fn visit_f32_div(&mut self) -> Result<(), Self::Error> {
        self.validate_bin_op::<f32, Div>()
    }

    fn visit_f32_min(&mut self) -> Result<(), Self::Error> {
        self.validate_bin_op::<f32, Min>()
    }

    fn visit_f32_max(&mut self) -> Result<(), Self::Error> {
        self.validate_bin_op::<f32, Max>()
    }

    fn visit_f32_copysign(&mut self) -> Result<(), Self::Error> {
        self.validate_bin_op::<f32, Copysign>()
    }

    fn visit_f64_abs(&mut self) -> Result<(), Self::Error> {
        self.validate_un_op::<f64, Abs>()
    }

    fn visit_f64_neg(&mut self) -> Result<(), Self::Error> {
        self.validate_un_op::<f64, Neg>()
    }

    fn visit_f64_ceil(&mut self) -> Result<(), Self::Error> {
        self.validate_un_op::<f64, Ceil>()
    }

    fn visit_f64_floor(&mut self) -> Result<(), Self::Error> {
        self.validate_un_op::<f64, Floor>()
    }

    fn visit_f64_trunc(&mut self) -> Result<(), Self::Error> {
        self.validate_un_op::<f64, Trunc>()
    }

    fn visit_f64_nearest(&mut self) -> Result<(), Self::Error> {
        self.validate_un_op::<f64, Nearest>()
    }

    fn visit_f64_sqrt(&mut self) -> Result<(), Self::Error> {
        self.validate_un_op::<f64, Sqrt>()
    }

    fn visit_f64_add(&mut self) -> Result<(), Self::Error> {
        self.validate_bin_op::<f64, Add>()
    }

    fn visit_f64_sub(&mut self) -> Result<(), Self::Error> {
        self.validate_bin_op::<f64, Sub>()
    }

    fn visit_f64_mul(&mut self) -> Result<(), Self::Error> {
        self.validate_bin_op::<f64, Mul>()
    }

    fn visit_f64_div(&mut self) -> Result<(), Self::Error> {
        self.validate_bin_op::<f64, Div>()
    }

    fn visit_f64_min(&mut self) -> Result<(), Self::Error> {
        self.validate_bin_op::<f64, Min>()
    }

    fn visit_f64_max(&mut self) -> Result<(), Self::Error> {
        self.validate_bin_op::<f64, Max>()
    }

    fn visit_f64_copysign(&mut self) -> Result<(), Self::Error> {
        self.validate_bin_op::<f64, Copysign>()
    }

    fn visit_i32_wrap_i64(&mut self) -> Result<(), Self::Error> {
        self.validate_un_op::<i64, WrapTo<i32>>()
    }

    fn visit_i32_trunc_f32_s(&mut self) -> Result<(), Self::Error> {
        self.validate_un_op::<f32, TruncTo<i32>>()
    }

    fn visit_i32_trunc_f32_u(&mut self) -> Result<(), Self::Error> {
        self.validate_un_op::<f32, TruncTo<u32>>()
    }

    fn visit_i32_trunc_f64_s(&mut self) -> Result<(), Self::Error> {
        self.validate_un_op::<f64, TruncTo<i32>>()
    }

    fn visit_i32_trunc_f64_u(&mut self) -> Result<(), Self::Error> {
        self.validate_un_op::<f64, TruncTo<u32>>()
    }

    fn visit_i64_extend_i32_s(&mut self) -> Result<(), Self::Error> {
        self.validate_un_op::<i32, ExtendTo<i64>>()
    }

    fn visit_i64_extend_i32_u(&mut self) -> Result<(), Self::Error> {
        self.validate_un_op::<u32, ExtendTo<u64>>()
    }

    fn visit_i64_trunc_f32_s(&mut self) -> Result<(), Self::Error> {
        self.validate_un_op::<f32, TruncTo<i64>>()
    }

    fn visit_i64_trunc_f32_u(&mut self) -> Result<(), Self::Error> {
        self.validate_un_op::<f32, TruncTo<u64>>()
    }

    fn visit_i64_trunc_f64_s(&mut self) -> Result<(), Self::Error> {
        self.validate_un_op::<f64, TruncTo<i64>>()
    }

    fn visit_i64_trunc_f64_u(&mut self) -> Result<(), Self::Error> {
        self.validate_un_op::<f64, TruncTo<u64>>()
    }

    fn visit_f32_convert_i32_s(&mut self) -> Result<(), Self::Error> {
        self.validate_un_op::<i32, ConvertTo<f32>>()
    }

    fn visit_f32_convert_i32_u(&mut self) -> Result<(), Self::Error> {
        self.validate_un_op::<u32, ConvertTo<f32>>()
    }

    fn visit_f32_convert_i64_s(&mut self) -> Result<(), Self::Error> {
        self.validate_un_op::<i64, ConvertTo<f32>>()
    }

    fn visit_f32_convert_i64_u(&mut self) -> Result<(), Self::Error> {
        self.validate_un_op::<u64, ConvertTo<f32>>()
    }

    fn visit_f32_demote_f64(&mut self) -> Result<(), Self::Error> {
        self.validate_un_op::<f64, DemoteTo<f32>>()
    }

    fn visit_f64_convert_i32_s(&mut self) -> Result<(), Self::Error> {
        self.validate_un_op::<i32, ConvertTo<f64>>()
    }

    fn visit_f64_convert_i32_u(&mut self) -> Result<(), Self::Error> {
        self.validate_un_op::<u32, ConvertTo<f64>>()
    }

    fn visit_f64_convert_i64_s(&mut self) -> Result<(), Self::Error> {
        self.validate_un_op::<i64, ConvertTo<f64>>()
    }

    fn visit_f64_convert_i64_u(&mut self) -> Result<(), Self::Error> {
        self.validate_un_op::<u64, ConvertTo<f64>>()
    }

    fn visit_f64_promote_f32(&mut self) -> Result<(), Self::Error> {
        self.validate_un_op::<f32, PromoteTo<f64>>()
    }

    fn visit_i32_reinterpret_f32(&mut self) -> Result<(), Self::Error> {
        self.validate_un_op::<f32, ReinterpretTo<i32>>()
    }

    fn visit_i64_reinterpret_f64(&mut self) -> Result<(), Self::Error> {
        self.validate_un_op::<f64, ReinterpretTo<i64>>()
    }

    fn visit_f32_reinterpret_i32(&mut self) -> Result<(), Self::Error> {
        self.validate_un_op::<i32, ReinterpretTo<f32>>()
    }

    fn visit_f64_reinterpret_i64(&mut self) -> Result<(), Self::Error> {
        self.validate_un_op::<i64, ReinterpretTo<f64>>()
    }

    fn visit_i32_extend8_s(&mut self) -> Result<(), Self::Error> {
        self.validate_un_op::<i32, ExtendFrom<i8>>()
    }

    fn visit_i32_extend16_s(&mut self) -> Result<(), Self::Error> {
        self.validate_un_op::<i32, ExtendFrom<i16>>()
    }

    fn visit_i64_extend8_s(&mut self) -> Result<(), Self::Error> {
        self.validate_un_op::<i64, ExtendFrom<i8>>()
    }

    fn visit_i64_extend16_s(&mut self) -> Result<(), Self::Error> {
        self.validate_un_op::<i64, ExtendFrom<i16>>()
    }

    fn visit_i64_extend32_s(&mut self) -> Result<(), Self::Error> {
        self.validate_un_op::<i64, ExtendFrom<i32>>()
    }

    fn visit_i32_trunc_sat_f32_s(&mut self) -> Result<(), Self::Error> {
        self.validate_un_op::<f32, TruncSatTo<i32>>()
    }

    fn visit_i32_trunc_sat_f32_u(&mut self) -> Result<(), Self::Error> {
        self.validate_un_op::<f32, TruncSatTo<u32>>()
    }

    fn visit_i32_trunc_sat_f64_s(&mut self) -> Result<(), Self::Error> {
        self.validate_un_op::<f64, TruncSatTo<i32>>()
    }

    fn visit_i32_trunc_sat_f64_u(&mut self) -> Result<(), Self::Error> {
        self.validate_un_op::<f64, TruncSatTo<u32>>()
    }

    fn visit_i64_trunc_sat_f32_s(&mut self) -> Result<(), Self::Error> {
        self.validate_un_op::<f32, TruncSatTo<i64>>()
    }

    fn visit_i64_trunc_sat_f32_u(&mut self) -> Result<(), Self::Error> {
        self.validate_un_op::<f32, TruncSatTo<u64>>()
    }

    fn visit_i64_trunc_sat_f64_s(&mut self) -> Result<(), Self::Error> {
        self.validate_un_op::<f64, TruncSatTo<i64>>()
    }

    fn visit_i64_trunc_sat_f64_u(&mut self) -> Result<(), Self::Error> {
        self.validate_un_op::<f64, TruncSatTo<u64>>()
    }
}

#[derive(Clone, Debug)]
struct Block {
    kind: BlockKind,
    type_: FuncType,
    is_unreachable: bool,
    height: usize,
}

impl Block {
    fn label_types(&self) -> LabelTypes {
        LabelTypes {
            kind: self.kind,
            type_: self.type_.clone(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BlockKind {
    Block,
    Loop,
    If,
    Else,
}

#[derive(Clone, Debug)]
struct LabelTypes {
    kind: BlockKind,
    type_: FuncType,
}

impl Deref for LabelTypes {
    type Target = [ValType];

    fn deref(&self) -> &Self::Target {
        match self.kind {
            BlockKind::Block | BlockKind::If | BlockKind::Else => self.type_.results(),
            BlockKind::Loop => self.type_.params(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum OpdType {
    ValType(ValType),
    Unknown,
}

impl OpdType {
    fn is_num(self) -> bool {
        match self {
            OpdType::ValType(type_) => type_.is_num(),
            _ => true,
        }
    }

    fn is_ref(self) -> bool {
        match self {
            OpdType::ValType(type_) => type_.is_ref(),
            _ => true,
        }
    }

    fn is_unknown(self) -> bool {
        match self {
            OpdType::Unknown => true,
            _ => false,
        }
    }

    fn check(self, expected_type: impl Into<ValType>) -> Result<(), DecodeError> {
        let expected_type = expected_type.into();
        match self {
            OpdType::ValType(actual_type) if actual_type != expected_type => {
                Err(DecodeError::new("type mismatch"))
            }
            _ => Ok(()),
        }
    }
}

impl From<RefType> for OpdType {
    fn from(type_: RefType) -> Self {
        OpdType::ValType(type_.into())
    }
}

impl From<ValType> for OpdType {
    fn from(type_: ValType) -> Self {
        OpdType::ValType(type_)
    }
}

impl From<Unknown> for OpdType {
    fn from(_: Unknown) -> Self {
        OpdType::Unknown
    }
}

#[derive(Debug)]
struct Unknown;