use {
    crate::{
        cast::{ExtendingCast, ExtendingCastFrom, WrappingCast, WrappingCastFrom},
        code,
        code::CodeBuilder,
        instr::{
            BlockType, InstrDecoder, InstrVisitor, MemArg,
        },
        decode::DecodeError,
        downcast::{DowncastRef, DowncastMut},
        exec,
        exec::{Imm, ReadReg, ReadFromPtr, Reg, Stk, ThreadedInstr, WriteReg, WriteToPtr},
        extern_ref::{ExternRef, UnguardedExternRef},
        func::{CompiledFuncBody, Func, FuncEntity, FuncType, InstrSlot, UncompiledFuncBody},
        func_ref::{FuncRef, UnguardedFuncRef},
        global::{GlobalEntity, GlobalEntityT},
        instance::Instance,
        ops::*,
        ref_::RefType,
        stack::StackSlot,
        store::Store,
        table::{TableEntity, TableEntityT},
        val::{UnguardedVal, ValType, ValTypeOf},
    },
    std::{mem, ops::Deref, ptr},
};

#[derive(Clone, Debug)]
pub(crate) struct Compiler {
    br_table_label_idxs: Vec<u32>,
    locals: Vec<Local>,
    blocks: Vec<Block>,
    opds: Vec<Opd>,
    fixup_offsets: Vec<usize>,
}

impl Compiler {
    pub(crate) fn new() -> Self {
        Self {
            br_table_label_idxs: Vec::new(),
            locals: Vec::new(),
            blocks: Vec::new(),
            opds: Vec::new(),
            fixup_offsets: Vec::new(),
        }
    }

    pub(crate) fn compile(
        &mut self,
        store: &mut Store,
        func: Func,
        instance: &Instance,
        code: &UncompiledFuncBody,
    ) -> CompiledFuncBody {
        use crate::decode::Decoder;

        self.locals.clear();
        self.blocks.clear();
        self.opds.clear();
        self.fixup_offsets.clear();

        let type_ = func.type_(store);
        let locals = &mut self.locals;
        for type_ in type_
            .params()
            .iter()
            .copied()
            .chain(code.locals.iter().copied())
        {
            locals.push(Local {
                type_,
                first_opd_idx: None,
            });
        }
        let local_count = locals.len() - type_.params().len();

        let mut compile = Compile {
            store,
            type_: type_.clone(),
            instance,
            br_table_label_idxs: &mut self.br_table_label_idxs,
            locals,
            blocks: &mut self.blocks,
            opds: &mut self.opds,
            fixup_offsets: &mut self.fixup_offsets,
            first_param_result_stack_idx: -(type_.call_frame_size() as isize),
            first_temp_stack_idx: local_count,
            max_stack_height: local_count,
            regs: [None; 2],
            code: CodeBuilder::new(),
        };
        compile.push_block(
            BlockKind::Block,
            FuncType::new([], type_.results().iter().copied()),
        );

        compile.emit_instr(exec::enter as ThreadedInstr);
        compile.emit(func.to_unguarded(store.id()));
        compile.emit(
            compile
                .instance
                .mem(0)
                .map(|mem| mem.to_unguarded(store.id())),
        );

        let mut decoder = Decoder::new(&code.expr);
        let mut instr_decoder = InstrDecoder::new(&mut decoder);
        while !compile.blocks.is_empty() {
            instr_decoder.decode(&mut compile).unwrap();
        }

        for (result_idx, result_type) in type_.clone().results().iter().copied().enumerate().rev() {
            compile.emit_instr(select_copy(result_type, OpdKind::Stk));
            compile.emit_stack_offset(compile.temp_stack_idx(result_idx));
            compile.emit_stack_offset(compile.param_result_stack_idx(result_idx));
        }
        compile.emit_instr(exec::return_ as ThreadedInstr);

        let mut code = compile.code.finish();
        for fixup_offset in compile.fixup_offsets.drain(..) {
            unsafe {
                let fixup_ptr = code.as_mut_ptr().add(fixup_offset);
                let instr_offset = ptr::read(fixup_ptr.cast());
                let instr_ptr = code.as_mut_ptr().add(instr_offset);
                ptr::write(fixup_ptr.cast(), instr_ptr);
            }
        }

        CompiledFuncBody {
            max_stack_height: compile.max_stack_height,
            local_count,
            code,
        }
    }
}

impl Default for Compiler {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
struct Compile<'a> {
    store: &'a Store,
    type_: FuncType,
    instance: &'a Instance,
    br_table_label_idxs: &'a mut Vec<u32>,
    locals: &'a mut [Local],
    blocks: &'a mut Vec<Block>,
    opds: &'a mut Vec<Opd>,
    fixup_offsets: &'a mut Vec<usize>,
    first_param_result_stack_idx: isize,
    first_temp_stack_idx: usize,
    max_stack_height: usize,
    regs: [Option<usize>; 2],
    code: CodeBuilder
}

impl<'a> Compile<'a> {
    fn compile_load<T>(&mut self, arg: MemArg) -> Result<(), DecodeError>
    where
        T: ValTypeOf + ReadFromPtr + WriteReg,
    {
        if self.block(0).is_unreachable {
            return Ok(());
        }
        self.compile_load_inner(
            arg,
            select_load::<T>(self.opd(0).kind()),
            T::val_type_of(),
        )
    }

    fn compile_load_n<Dst, Src>(&mut self, arg: MemArg) -> Result<(), DecodeError>
    where
        Dst: ValTypeOf + ExtendingCastFrom<Src> + WriteReg,
        Src: ReadFromPtr + ExtendingCast,
    {
        if self.block(0).is_unreachable {
            return Ok(());
        }
        self.compile_load_inner(
            arg,
            select_load_n::<Dst, Src>(self.opd(0).kind()),
            Dst::val_type_of(),
        )
    }

    fn compile_load_inner(&mut self, arg: MemArg, load: ThreadedInstr, output_type: ValType) -> Result<(), DecodeError> {
        // Loads write their output to a register, so we need to ensure that the output register is
        // available for the load to use.
        //
        // If the output register is already occupied, then we need to preserve the register on the
        // stack. Otherwise, the load will overwrite the register while it's already occupied.
        //
        // The only exception is if the input occupies the output register. In that case, the
        // operation can safely overwrite the register, since the input will be consumed by the
        // operation anyway.
        let output_reg_idx = output_type.reg_idx();
        if self.is_reg_occupied(output_reg_idx) && !self.opd(0).occupies_reg(output_reg_idx) {
            self.preserve_reg(output_reg_idx);
        }

        // Emit the instruction.
        self.emit_instr(load);

        // Emit and pop the inputs from the stack.
        self.emit_opd(0);
        self.pop_opd();

        // Push the output onto the stack and allocate a register for it.
        self.push_opd(output_type);
        self.alloc_reg();

        self.emit(arg.offset);

        Ok(())
    }

    fn compile_store<T>(&mut self, arg: MemArg) -> Result<(), DecodeError>
    where
        T: ReadReg + WriteToPtr,
    {
        if self.block(0).is_unreachable {
            return Ok(());
        }
        self.compile_store_inner(
            arg,
            select_store::<T>(self.opd(1).kind(), self.opd(0).kind()),
        )
    }

    fn compile_store_n<Src, Dst>(&mut self, arg: MemArg) -> Result<(), DecodeError>
    where
        Src: ReadReg + WrappingCast,
        Dst: WrappingCastFrom<Src> + WriteToPtr,
    {
        if self.block(0).is_unreachable {
            return Ok(());
        }
        self.compile_store_inner(
            arg,
            select_store_n::<Src, Dst>(self.opd(1).kind(), self.opd(0).kind()),
        )
    }

    fn compile_store_inner(&mut self, arg: MemArg, store: ThreadedInstr) -> Result<(), DecodeError> {
        // Emit the instruction.
        self.emit_instr(store);

        // Emit the inputs and pop them from the stack.
        self.emit_opd(0);
        self.pop_opd();
        self.emit_opd(0);
        self.pop_opd();

        // Emit the static offset.
        self.emit(arg.offset);
        
        Ok(())
    }

    fn compile_un_op<T, U>(&mut self) -> Result<(), DecodeError>
    where
        T: ReadReg,
        U: UnOp<T>,
        U::Output: ValTypeOf + WriteReg,
    {
        if self.block(0).is_unreachable {
            return Ok(());
        }
        self.compile_un_op_inner(
            select_un_op::<T, U>(self.opd(0).kind()),
            U::Output::val_type_of(),
        )
    }
    
    fn compile_un_op_inner(&mut self, un_op: ThreadedInstr, output_type: ValType) -> Result<(), DecodeError> {
        // Unary operations write their output to a register, so we need to ensure that the output
        // register is available for the operation to use.
        //
        // If the output register is already occupied, then we need to preserve the register on the
        // stack. Otherwise, the operation will overwrite the register while it's already occupied.
        //
        // The only exception is if the input occupies the output register. In that case, the
        // operation can safely overwrite the register, since the input will be consumed by the
        // operation anyway.
        let output_reg_idx = output_type.reg_idx();
        if self.is_reg_occupied(output_reg_idx) && !self.opd(0).occupies_reg(output_reg_idx) {
            self.preserve_reg(output_reg_idx);
        }

        // Emit the instruction.
        self.emit_instr(un_op);

        // Emit and pop the inputs from the stack.
        self.emit_opd(0);
        self.pop_opd();

        // Push the output onto the stack and allocate a register for it.
        self.push_opd(output_type);
        self.alloc_reg();

        Ok(())
    }

    fn compile_bin_op<T, B>(&mut self) -> Result<(), DecodeError>
    where
        T: ReadReg,
        B: BinOp<T>,
        B::Output: ValTypeOf + WriteReg,
    {
        if self.block(0).is_unreachable {
            return Ok(());
        }
        self.compile_bin_op_inner(
            select_bin_op::<T, B>(
                self.opd(1).kind(),
                self.opd(0).kind(),
            ),
            B::Output::val_type_of(),
        )
    }

    fn compile_bin_op_inner(&mut self, bin_op: ThreadedInstr, output_type: ValType) -> Result<(), DecodeError> {
        // Binary operations write their output to a register, so we need to ensure that the output
        // register is available for the operation to use.
        //
        // If the output register is already occupied, then we need to preserve the register on the
        // stack. Otherwise, the operation will the register while it's already occupied.
        //
        // The only exception is if one of the inputs occupies the output register. In that case,
        // the operation can safely overwrite the register, since the input will be consumed by the
        // operation anyway.
        let output_reg_idx = output_type.reg_idx();
        if self.is_reg_occupied(output_reg_idx)
            && !self.opd(1).occupies_reg(output_reg_idx)
            && !self.opd(0).occupies_reg(output_reg_idx)
        {
            self.preserve_reg(output_reg_idx);
        }

        // Emit the instruction.
        self.emit_instr(bin_op);

        // Emit the inputs and pop them from the stack.
        self.emit_opd(0);
        self.pop_opd();
        self.emit_opd(0);
        self.pop_opd();

        // Push the output onto the stack and allocate a register for it.
        self.push_opd(output_type);
        self.alloc_reg();

        Ok(())
    }

    // Methods for operating on types.

    /// Resolves a [`BlockType`] to its corresponding [`FuncType`].
    fn resolve_block_type(&self, type_: BlockType) -> FuncType {
        match type_ {
            BlockType::TypeIdx(idx) => self
                .store
                .resolve_type(self.instance.type_(idx).unwrap())
                .clone(),
            BlockType::ValType(val_type) => FuncType::from_val_type(val_type),
        }
    }

    // Methods for operating on locals.

    /// Appends the top operand to the list of operands that refer to the local with the given
    /// index.
    ///
    /// This marks the operand as a local operand.
    fn push_local_opd(&mut self, local_idx: usize) {
        let opd_idx = self.opds.len() - 1;

        debug_assert!(self.opds[opd_idx].local_idx.is_none());
        self.opds[opd_idx].local_idx = Some(local_idx);
        self.opds[opd_idx].next_opd_idx = self.locals[local_idx].first_opd_idx;
        if let Some(first_opd_idx) = self.locals[local_idx].first_opd_idx {
            self.opds[first_opd_idx].prev_opd_idx = Some(opd_idx);
        }

        self.locals[local_idx].first_opd_idx = Some(opd_idx);
    }

    /// Removes the operand with the given index from the list of operands that refer to the local
    /// with the given index.
    fn remove_local_opd(&mut self, opd_idx: usize) {
        let local_idx = self.opds[opd_idx].local_idx.unwrap();

        if let Some(prev_opd_idx) = self.opds[opd_idx].prev_opd_idx {
            self.opds[prev_opd_idx].next_opd_idx = self.opds[opd_idx].next_opd_idx;
        } else {
            self.locals[local_idx].first_opd_idx = self.opds[opd_idx].next_opd_idx;
        }
        if let Some(next_opd_idx) = self.opds[opd_idx].next_opd_idx {
            self.opds[next_opd_idx].prev_opd_idx = self.opds[opd_idx].prev_opd_idx;
        }

        self.opds[opd_idx].local_idx = None;
        self.opds[opd_idx].prev_opd_idx = None;
        self.opds[opd_idx].next_opd_idx = None;
    }

    /// Preserve the local with the given index by preserving every local operand that refers to it.
    fn preserve_local(&mut self, local_idx: usize) {
        while let Some(opd_idx) = self.locals[local_idx].first_opd_idx {
            self.preserve_local_opd(opd_idx);
            self.locals[local_idx].first_opd_idx = self.opds[opd_idx].next_opd_idx;
            self.opds[opd_idx].local_idx = None;
        }
    }

    // Methods for operating on blocks.

    /// Returns a reference to the block with the given index.
    fn block(&self, idx: usize) -> &Block {
        &self.blocks[self.blocks.len() - 1 - idx]
    }

    /// Returns a mutable reference to the block with the given index
    fn block_mut(&mut self, idx: usize) -> &mut Block {
        let len = self.blocks.len();
        &mut self.blocks[len - 1 - idx]
    }

    /// Marks the current block as unreachable.
    fn set_unreachable(&mut self) {
        // Unwind the operand stack to the height of the current block.
        while self.opds.len() > self.block(0).height {
            self.pop_opd();
        }
        self.block_mut(0).is_unreachable = true;
    }

    /// Pushes the hole with the given index onto the block with the given index.
    fn push_hole(&mut self, block_idx: usize, hole_offset: usize) {
        let first_hole_offset = self.block(block_idx).first_hole_offset;
        let first_hole_offset = first_hole_offset.unwrap_or(usize::MAX);
        unsafe {
            ptr::write(self.code.as_mut_ptr().add(hole_offset).cast(), first_hole_offset)
        }
        self.block_mut(block_idx).first_hole_offset = Some(hole_offset);
    }

    /// Pops a hole from the block with the given index.
    fn pop_hole(&mut self, block_idx: usize) -> Option<usize> {
        if let Some(hole_offset) = self.block(block_idx).first_hole_offset {
            let next_hole_offset = unsafe {
                ptr::read(self.code.as_ptr().add(hole_offset).cast())
            };
            let next_hole_offset = if next_hole_offset == usize::MAX {
                None
            } else {
                Some(next_hole_offset)
            };
            self.block_mut(block_idx).first_hole_offset = next_hole_offset;
            Some(hole_offset)
        } else {
            None
        }
    }

    /// Pushes a block with the given kind and type on stack.
    fn push_block(&mut self, kind: BlockKind, type_: FuncType) {
        self.code.pad_to_align(code::ALIGN);

        self.blocks.push(Block {
            kind,
            type_,
            is_unreachable: false,
            height: self.opds.len(),
            first_instr_offset: self.code.len(),
            first_hole_offset: None,
            else_hole_offset: None,
        });

        // Push the inputs of the block on the stack.
        for input_type in self.block(0).type_.clone().params().iter().copied() {
            self.push_opd(input_type);
        }
    }

    /// Pops a block from the stack.
    fn pop_block(&mut self) -> Block {
        // Unwind the operand stack to the height of the block.
        while self.opds.len() > self.block(0).height {
            self.pop_opd();
        }

        self.blocks.pop().unwrap()
    }

    // Methods for operating on operands.

    /// Returns a reference to the [`Opd`] at the given depth.
    fn opd(&self, depth: usize) -> &Opd {
        &self.opds[self.opds.len() - 1 - depth]
    }

    /// Returns a mutable reference to the [`Opd`] at the given depth.
    fn opd_mut(&mut self, depth: usize) -> &mut Opd {
        let len = self.opds.len();
        &mut self.opds[len - 1 - depth]
    }

    /// Ensures that the operand at the given depth is not a immediate operand, by preserving the
    /// constant on the stack if necessary.
    fn ensure_opd_not_imm(&mut self, opd_depth: usize) {
        if self.opd(opd_depth).is_imm() {
            self.preserve_imm_opd(opd_depth);
        }
    }

    /// Ensures that the operand at the given depth is not a local operand, by preserving the local
    /// on the stack if necessary.
    fn ensure_opd_not_local(&mut self, opd_depth: usize) {
        if self.opd(opd_depth).is_local() {
            self.preserve_local_opd(self.opds.len() - 1 - opd_depth);
        }
    }

    // Ensures that the operand at the given depth is not a register operand, by preserving the
    // register on the stack if necessary.
    fn ensure_opd_not_reg(&mut self, opd_depth: usize) {
        if self.opd(opd_depth).is_reg {
            self.preserve_reg(self.opd(opd_depth).type_.reg_idx());
        }
    }

    /// Preserves an immediate operand by copying its value to the stack, if necessary.
    fn preserve_imm_opd(&mut self, opd_depth: usize) {
        let opd_idx = self.opds.len() - 1 - opd_depth;
        self.emit_instr(select_copy(self.opds[opd_idx].type_, OpdKind::Imm));
        self.emit_val(self.opds[opd_idx].val.unwrap());
        self.emit_stack_offset(self.temp_stack_idx(opd_idx));
        self.opd_mut(opd_depth).val = None;
    }

    /// Preserve a local operand by copying the local it refers to to the stack, if necessary.
    fn preserve_local_opd(&mut self, opd_idx: usize) {
        let local_idx = self.opds[opd_idx].local_idx.unwrap();
        self.emit_instr(select_copy(self.locals[local_idx].type_, OpdKind::Stk));
        self.emit_stack_offset(self.local_stack_idx(local_idx));
        self.emit_stack_offset(self.temp_stack_idx(opd_idx));
        self.remove_local_opd(opd_idx);
    }

    /// Pushes an operand of the given type on the stack.
    fn push_opd(&mut self, type_: impl Into<ValType>) {
        self.opds.push(Opd {
            type_: type_.into(),
            val: None,
            local_idx: None,
            prev_opd_idx: None,
            next_opd_idx: None,
            is_reg: false,
        });
        let stack_height = self.first_temp_stack_idx as usize + (self.opds.len() - 1);
        self.max_stack_height = self.max_stack_height.max(stack_height);
    }

    /// Pops an operand from the stack.
    fn pop_opd(&mut self) -> ValType {
        if self.opd(0).is_reg {
            self.dealloc_reg(self.opd(0).type_.reg_idx());
        }
        let opd_idx = self.opds.len() - 1;
        if let Some(local_idx) = self.opds[opd_idx].local_idx {
            self.locals[local_idx].first_opd_idx = self.opds[opd_idx].next_opd_idx;
        }
        self.opds.pop().unwrap().type_
    }

    /// Emits an operand and then pops it from the stack.
    fn emit_and_pop_opd(&mut self) {
        self.emit_opd(0);
        self.pop_opd();
    }

    // Methods for operating on the stack.

    /// Returns the stack index of the parameter/result with the given index.
    fn param_result_stack_idx(&self, param_result_idx: usize) -> isize {
        self.first_param_result_stack_idx + param_result_idx as isize
    }

    /// Returns the stack index of the local with the given index.
    fn local_stack_idx(&self, local_idx: usize) -> isize {
        if local_idx < self.type_.params().len() {
            self.param_result_stack_idx(local_idx)
        } else {
            (local_idx - self.type_.params().len()) as isize
        }
    }

    /// Returns the stack index of the temporary with the given index.
    fn temp_stack_idx(&self, temp_idx: usize) -> isize {
        (self.first_temp_stack_idx + temp_idx) as isize
    }

    /// Returns the stack index of the operand at the given depth.
    fn opd_stack_idx(&self, opd_depth: usize) -> isize {
        let opd_idx = self.opds.len() - 1 - opd_depth;
        if let Some(local_idx) = self.opds[opd_idx].local_idx {
            self.local_stack_idx(local_idx)
        } else {
            self.temp_stack_idx(opd_idx)
        }
    }

    // Methods for operating on registers.

    /// Returns `true` if the register with the given index is occupied.
    fn is_reg_occupied(&self, reg_idx: usize) -> bool {
        self.regs[reg_idx].is_some()
    }

    /// Allocates a register to the top operand.
    fn alloc_reg(&mut self) {
        debug_assert!(!self.opd(0).is_reg);
        let reg_idx = self.opd(0).type_.reg_idx();
        debug_assert!(!self.is_reg_occupied(reg_idx));
        let opd_idx = self.opds.len() - 1;
        self.opds[opd_idx].is_reg = true;
        self.regs[reg_idx] = Some(opd_idx);
    }

    /// Deallocates the register with the given index.
    fn dealloc_reg(&mut self, reg_idx: usize) {
        let opd_idx = self.regs[reg_idx].unwrap();
        self.opds[opd_idx].is_reg = false;
        self.regs[reg_idx] = None;
    }

    /// Preserves the register with the given index by preserving the register operand that occupies
    /// it.
    fn preserve_reg(&mut self, reg_idx: usize) {
        let opd_idx = self.regs[reg_idx].unwrap();
        let opd_type = self.opds[opd_idx].type_;
        self.emit_instr(select_copy(opd_type, OpdKind::Reg));
        self.emit_stack_offset(self.temp_stack_idx(opd_idx));
        self.dealloc_reg(reg_idx);
    }

    /// Preserves all registers by preserving the register operands that occupy them.
    fn preserve_all_regs(&mut self) {
        for reg_idx in 0..self.regs.len() {
            if self.is_reg_occupied(reg_idx) {
                self.preserve_reg(reg_idx);
            }
        }
    }

    /// Copies the values for the label with the given index to their expected locations
    /// on the stack, and pop them from the stack.
    fn resolve_label_vals(&mut self, label_idx: usize) {
        for (label_val_idx, label_type) in self
            .block(label_idx)
            .label_types()
            .iter()
            .copied()
            .enumerate()
            .rev()
        {
            self.emit_instr(select_copy(label_type, OpdKind::Stk));
            self.emit_stack_offset(self.opd_stack_idx(0));
            self.pop_opd();
            self.emit_stack_offset(
                self.temp_stack_idx(self.block(label_idx).height + label_val_idx),
            );
        }
    }

    // Methods for emitting code.

    // Emits the given value.
    fn emit<T>(&mut self, val: T)
    where
        T: Copy,
    {
        debug_assert!(mem::size_of::<T>() <= mem::size_of::<InstrSlot>());
        self.code.push(val);
        self.code.pad_to_align(code::ALIGN);
    }

    fn emit_instr(&mut self, instr: ThreadedInstr) {
        self.code.pad_to_align(code::ALIGN);
        unsafe {
            self.code.push_aligned(instr);
        }
    }

    // Emits an operand.
    //
    // For local and temporary operands, which can be read from the stack, this emits the offset of
    // the stack slot for the operand. For immediate operands, which carry their own value, this
    // emits the value of the operand. For register operands, we don't need to anything.
    fn emit_opd(&mut self, opd_depth: usize) {
        match self.opd(opd_depth).kind() {
            OpdKind::Stk => self.emit_stack_offset(self.opd_stack_idx(opd_depth)),
            OpdKind::Imm => {
                self.emit_val(self.opd(opd_depth).val.unwrap());
            }
            OpdKind::Reg => {}
        }
    }

    // Emits an immediate value.
    fn emit_val(&mut self, val: UnguardedVal) {
        match val {
            UnguardedVal::I32(val) => self.emit(val),
            UnguardedVal::I64(val) => self.emit(val),
            UnguardedVal::F32(val) => self.emit(val),
            UnguardedVal::F64(val) => self.emit(val),
            UnguardedVal::FuncRef(val) => self.emit(val),
            UnguardedVal::ExternRef(val) => self.emit(val),
        }
    }

    /// Emits the offset of the stack slot with the given index.
    fn emit_stack_offset(&mut self, stack_idx: isize) {
        self.emit(stack_idx as i32 * mem::size_of::<StackSlot>() as i32);
    }

    /// Emits the label for the block with the given index.
    ///
    /// If the block is of kind [`BlockKind::Loop`], this emits the offset of the first instruction
    /// in the block. [`BlockKind::Block`], we don't yet know where the first instruction after the
    /// end of the block is, so we emit a hole instead.
    fn emit_label(&mut self, block_idx: usize) {
        match self.block(block_idx).kind {
            BlockKind::Block => {
                let hole_offset = self.emit_hole();
                self.push_hole(block_idx, hole_offset);
            }
            BlockKind::Loop => {
                self.emit_instr_offset(self.block(block_idx).first_instr_offset);
            }
        }
    }

    /// Emits a hole and returns its index.
    ///
    /// A hole is a placeholder for an instruction offset that is not yet known.
    fn emit_hole(&mut self) -> usize {
        self.code.push(0usize)
    }

    /// Patches the hole with the given index with the offset of the current instruction.
    fn patch_hole(&mut self, hole_offset: usize) {
        self.code.pad_to_align(code::ALIGN);
        let instr_offset = self.code.len();
        unsafe { ptr::write(self.code.as_mut_ptr().add(hole_offset).cast(), instr_offset) }
        self.fixup_offsets.push(hole_offset);
    }

    /// Emits the offset of the instruction with the given index.
    fn emit_instr_offset(&mut self, instr_offset: usize) {
        let fixup_offset = self.code.push(instr_offset);
        self.fixup_offsets.push(fixup_offset);
    }
}

impl<'a> InstrVisitor for Compile<'a> {
    type Ok = ();
    type Error = DecodeError;

    // Control instructions

    /// Compiles a `nop` instruction.
    fn visit_nop(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }

    /// Compiles an `unreachable` instruction.
    fn visit_unreachable(&mut self) -> Result<(), Self::Error> {
        // Emit the instruction.
        self.emit_instr(exec::unreachable as ThreadedInstr);

        // After an `unreachable` instruction, the rest of the block is unreachable.
        self.set_unreachable();

        Ok(())
    }

    /// Compiles a `block` instruction.
    fn visit_block(&mut self, type_: BlockType) -> Result<(), Self::Error> {
        // Resolve the type of the block.
        let type_ = self.resolve_block_type(type_);

        // Skip this instruction if it is unreachable.
        if !self.block(0).is_unreachable {
            for opd_depth in 0..type_.params().len() {
                self.ensure_opd_not_imm(opd_depth);
                self.ensure_opd_not_local(opd_depth);
                self.ensure_opd_not_reg(opd_depth);
            }

            // Pop the inputs of the block from the stack.
            for _ in 0..type_.params().len() {
                self.pop_opd();
            }
        }

        // Push the `block` block onto the stack.
        //
        // We do this even if the rest of the current block is unreachable, so we can match each
        // `end` instruction with its corresponding block.
        self.push_block(BlockKind::Block, type_);

        Ok(())
    }

    /// Compiles a `loop` instruction.
    fn visit_loop(&mut self, type_: BlockType) -> Result<(), Self::Error> {
        // Resolve the type of the block.
        let type_ = self.resolve_block_type(type_);

        // Skip this instruction if it is unreachable.
        if !self.block(0).is_unreachable {
            // This is a branch target. We need to ensure that each block input is stored in a
            // known location here, so that if we encounter a branch to this target elsewhere, we
            // can ensure that each block input is stored in the location expected by the target
            // before the branch is taken.
            //
            // We do this by ensuring that each block input is a temporary operand, which is stored
            // in a known location on the stack.
            for opd_depth in 0..type_.params().len() {
                self.ensure_opd_not_imm(opd_depth);
                self.ensure_opd_not_local(opd_depth);
                self.ensure_opd_not_reg(opd_depth);
            }

            // Pop the inputs of the block from the stack.
            for _ in 0..type_.params().len() {
                self.pop_opd();
            }
        }

        // Push the `loop` block onto the stack.
        //
        // We do this even if the rest of the current block is unreachable, so we can match each
        // `end` instruction with its corresponding block.
        self.push_block(BlockKind::Loop, type_);

        Ok(())
    }

    /// Compiles an `if` instruction.
    fn visit_if(&mut self, type_: BlockType) -> Result<(), Self::Error> {
        // Resolve the type of the block.
        let type_ = self.resolve_block_type(type_);

        let else_hole_offset = if !self.block(0).is_unreachable {
            // The `br_if_z` instruction does not have an _i variant, so ensure that the condition
            // is not an immediate operand.
            self.ensure_opd_not_imm(0);

            // This is a branch. We need to ensure that each block input is stored in the location
            // expected by the target before the branch is taken.
            //
            // We do this by ensuring that each block input is a temporary operand, which is stored
            // in a known location on the stack. We don't need to copy the block inputs to their
            // expected locations, because they are already in the correct locations.
            for opd_depth in 1..type_.params().len() + 1 {
                self.ensure_opd_not_imm(opd_depth);
                self.ensure_opd_not_local(opd_depth);
                self.ensure_opd_not_reg(opd_depth);
            }

            // Emit the instruction.
            self.emit_instr(select_br_if_z(self.opd(0).kind()));

            // Emit the condition and pop it from the stack.
            self.emit_and_pop_opd();

            // We don't yet know where the start of the `else` block is, so we emit a hole for it instead.
            let else_hole_offset = self.emit_hole();

            // Pop the inputs of the block from the stack.
            for _ in 0..type_.params().len() {
                self.pop_opd();
            }

            Some(else_hole_offset)
        } else {
            None
        };

        // Push the `if` block onto the stack.
        //
        // We do this even if the rest of the current block is unreachable, so we can match each
        // `end` instruction with its corresponding block.
        self.push_block(BlockKind::Block, type_);

        // Store the hole for the start of the `else` block with the `if` block so we can patch it
        // when we reach the start of the `else` block.
        self.block_mut(0).else_hole_offset = else_hole_offset;

        Ok(())
    }

    /// Compiles an `else` instruction.
    fn visit_else(&mut self) -> Result<(), Self::Error> {
        // Skip this instruction if it is unreachable.
        if !self.block(0).is_unreachable {
            // This is a branch target. We need to ensure that each block input is stored in a
            // known location here, so that if we encounter a branch to this target elsewhere, we
            // can ensure that each block input is stored in the location expected by the target
            // before the branch is taken.
            //
            // We do this by ensuring that each block input is a temporary operand, which is stored
            // in a known location on the stack.
            for opd_depth in 0..self.block(0).type_.results().len() {
                self.ensure_opd_not_imm(opd_depth);
                self.ensure_opd_not_local(opd_depth);
                self.ensure_opd_not_reg(opd_depth);
            }

            // We are now at the end of the `if` block, so we want to branch to the first
            // instruction after the end of the `else` block. We don't know where that is yet, so
            // we emit a hole instead, and append it to the list of holes for the `if` block.
            self.emit_instr(exec::br as ThreadedInstr);
            let hole_offset = self.emit_hole();
            self.push_hole(0, hole_offset);
        }

        // We are now at the start of the `else` block, so we can patch the hole for the start of
        // the `else` block.
        //
        // We do this even if rest of the the current `if` block is unreachable, since the hole
        // itself could still be reachable.
        if let Some(else_hole_offset) = self.block_mut(0).else_hole_offset.take() {
            self.patch_hole(else_hole_offset);
        }

        // Pop the `if` block from the stack.
        let block = self.pop_block();

        // Push the `else` block on the stack.
        //
        // We do this even if the rest of the `if` block was unreachable, so we can match each `end`
        // instruction with its corresponding block.
        self.push_block(BlockKind::Block, block.type_);

        // Copy the list of holes for the `if` block to that of the `else` block, so that they will
        // be patched when we reach the first instruction after the end of the `else` block.
        self.block_mut(0).first_hole_offset = block.first_hole_offset;

        Ok(())
    }

    /// Compiles an `end` instruction.
    fn visit_end(&mut self) -> Result<(), Self::Error> {
        // Skip this instruction if it is unreachable.
        if !self.block(0).is_unreachable {
            // This is a branch target. We need to ensure that each block input is stored in a
            // known location here, so that if we encounter a branch to this target elsewhere, we
            // can ensure that each block input is stored in the location expected by the target
            // before the branch is taken.
            //
            // We do this by ensuring that each block input is a temporary operand, which is stored
            // in a known location on the stack.
            for opd_depth in 0..self.block(0).type_.results().len() {
                self.ensure_opd_not_imm(opd_depth);
                self.ensure_opd_not_local(opd_depth);
                self.ensure_opd_not_reg(opd_depth);
            }
        }

        // If the current block has hole for the start of the `else` block, then it is an `if` block
        // without a corresponding `else` block. In this case, we should patch the hole for the
        // start of the `else` block with the offset of the first instruction after the end of the
        // `if` block.
        //
        // We do this even if the rest of the current block is unreachable, since the hole itself
        // could still be reachable.
        if let Some(else_hole_offset) = self.block_mut(0).else_hole_offset.take() {
            self.patch_hole(else_hole_offset);
        }

        // We are now at the first instruction after the end of the block, so we can patch the list
        // of holes for the block.
        //
        // We do this even if the rest of the current block is unreachable, since the holes themselves could
        // still be reachable.
        while let Some(hole_offset) = self.pop_hole(0) {
            self.patch_hole(hole_offset);
        }

        // Pop the block from the stack.
        //
        // We do this even if the rest of the current block is unreachable, so we can match each `end`
        // instruction with its corresponding block.
        let block = self.pop_block();

        // Push the outputs of the block onto the stack.
        for result_type in block.type_.results().iter().copied() {
            self.push_opd(result_type);
        }

        Ok(())
    }

    /// Compiles a `br` instruction.
    fn visit_br(&mut self, label_idx: u32) -> Result<(), Self::Error> {
        // Skip this instruction if it is unreachable.
        if self.block(0).is_unreachable {
            return Ok(());
        }

        // Wasm uses `u32` indices for labels, but we use `usize` indices.
        let label_idx = label_idx as usize;

        // This is a branch. We need to ensure that each block input is stored in the location
        // expected by the target before the branch is taken.
        //
        // We do this by ensuring that each block input is a temporary operand, which is stored
        // in a known location on the stack, and then copying them to their expected locations in
        // the code emitted below.
        for opd_depth in 0..self.block(label_idx).label_types().len() {
            self.ensure_opd_not_imm(opd_depth);
            self.ensure_opd_not_local(opd_depth);
            self.ensure_opd_not_reg(opd_depth);
        }

        self.resolve_label_vals(label_idx);
        self.emit_instr(exec::br as ThreadedInstr);
        self.emit_label(label_idx);
        self.set_unreachable();

        Ok(())
    }

    /// Compiles a `br_if` instruction.
    fn visit_br_if(&mut self, label_idx: u32) -> Result<(), Self::Error> {
        // Skip this instruction if it is unreachable.
        if self.block(0).is_unreachable {
            return Ok(());
        }

        // Wasm uses `u32` indices for labels, but we use `usize` indices.
        let label_idx = label_idx as usize;

        // The `br_if_nz` instruction does not have an _i variant, so ensure that the condition
        // is not an immediate operand.
        self.ensure_opd_not_imm(0);

        // This is a branch. We need to ensure that each block input is stored in the location
        // expected by the target before the branch is taken.
        //
        // We do this by ensuring that each block input is a temporary operand, which is stored
        // in a known location on the stack, and then copying them to their expected locations in
        // the code emitted below.
        for opd_depth in 1..self.block(label_idx).label_types().len() + 1 {
            self.ensure_opd_not_imm(opd_depth);
            self.ensure_opd_not_local(opd_depth);
            self.ensure_opd_not_reg(opd_depth);
        }

        if self.block(label_idx).label_types().is_empty() {
            // If the branch target has an empty type, we don't need to copy any block inputs to
            // their expected locations, so we can generate more efficient code.
            self.emit_instr(select_br_if_nz(self.opd(0).kind()));
            self.emit_and_pop_opd();
            self.emit_label(label_idx);
        } else {
            // If the branch target has a non-empty type, we need to copy all block inputs to their
            // expected locations.
            //
            // This is more expensive, because we cannot branch to the target directly. Instead, we
            // have to branch to code that first copies the block inputs to their expected
            // locations, and then branches to the target.
            self.emit_instr(select_br_if_z(self.opd(0).kind()));
            self.emit_and_pop_opd();
            let hole_offset = self.emit_hole();
            self.resolve_label_vals(label_idx);
            self.emit_instr(exec::br as ThreadedInstr);
            self.emit_label(label_idx);
            self.patch_hole(hole_offset);
        }

        for label_type in self.block(label_idx).label_types().iter().copied() {
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

    /// Compiles a `br_table` instruction.
    fn visit_br_table_end(&mut self, default_label_idx: u32) -> Result<(), Self::Error> {
        let label_idxs = mem::take(self.br_table_label_idxs);

        // Skip this instruction if it is unreachable.
        if self.block(0).is_unreachable {
            return Ok(());
        }

        // Wasm uses `u32` indices for labels, but we use `usize` indices.
        let default_label_idx = default_label_idx as usize;

        // The `br_table` instruction does not have an _i variant, so ensure that the index is not
        // an immediate operand.
        self.ensure_opd_not_imm(0);

        // This is a branch. We need to ensure that each block input is stored in the location
        // expected by the target before the branch is taken.
        //
        // We do this by ensuring that each block input is a temporary operand, which is stored
        // in a known location on the stack, and then copying them to their expected locations in
        // the code emitted below.
        for opd_depth in 1..self.block(default_label_idx).label_types().len() + 1 {
            self.ensure_opd_not_imm(opd_depth);
            self.ensure_opd_not_local(opd_depth);
            self.ensure_opd_not_reg(opd_depth);
        }

        if self.block(default_label_idx).label_types().is_empty() {
            // If the branch target has an empty type, we don't need to copy any block inputs to
            // their expected locations, so we can generate more efficient code.
            self.emit_instr(select_br_table(self.opd(0).kind()));
            self.emit_and_pop_opd();
            self.emit(label_idxs.len() as u32);
            for label_idx in label_idxs.iter().copied() {
                let label_idx = label_idx as usize;
                self.emit_label(label_idx);
                for label_type in self.block(0).label_types().iter().copied() {
                    self.push_opd(label_type);
                }
            }
            self.emit_label(default_label_idx);
        } else {
            // If the branch target has a non-empty type, we need to copy all block inputs to their
            // expected locations.
            //
            // This is more expensive, because we cannot branch to each target directly. Instead, for
            // each target, we have to branch to code that first copies the block inputs to their
            // expected locations, and then branches to the target.
            self.emit_instr(select_br_table(self.opd(0).kind()));
            self.emit_and_pop_opd();
            self.emit(label_idxs.len() as u32);
            let mut hole_offsets = Vec::new();
            for _ in 0..label_idxs.len() {
                let hole_offset = self.emit_hole();
                hole_offsets.push(hole_offset);
            }
            let default_hole_offset = self.emit_hole();
            for (label_idx, hole_offset) in label_idxs.iter().copied().zip(hole_offsets) {
                let label_idx = label_idx as usize;
                self.patch_hole(hole_offset);
                self.resolve_label_vals(label_idx);
                self.emit_instr(exec::br as ThreadedInstr);
                self.emit_label(label_idx);
                for label_type in self.block(label_idx).label_types().iter().copied() {
                    self.push_opd(label_type);
                }
            }
            self.patch_hole(default_hole_offset);
            self.resolve_label_vals(default_label_idx);
            self.emit_instr(exec::br as ThreadedInstr);
            self.emit_label(default_label_idx);
        }

        // After a `br_table` instruction, the rest of the block is unreachable.
        self.set_unreachable();

        *self.br_table_label_idxs = label_idxs;

        Ok(())
    }

    /// Compiles a `return` instruction.
    fn visit_return(&mut self) -> Result<(), Self::Error> {
        // Skip this instruction if it is unreachable.
        if self.block(0).is_unreachable {
            return Ok(());
        }

        // Copy the return values to their expected locations on the stack, and pop them from the
        // stack.
        for (result_idx, result_type) in self
            .type_
            .clone()
            .results()
            .iter()
            .copied()
            .enumerate()
            .rev()
        {
            self.ensure_opd_not_imm(0);
            self.emit_instr(if self.opd(0).is_reg {
                select_copy(result_type, OpdKind::Reg)
            } else {
                select_copy(result_type, OpdKind::Stk)
            });
            self.emit_and_pop_opd();
            self.emit_stack_offset(self.param_result_stack_idx(result_idx));
        }

        // Emit the instruction.
        self.emit_instr(exec::return_ as ThreadedInstr);

        // After a `return`` instruction, the rest of the block is unreachable.
        self.set_unreachable();

        Ok(())
    }

    /// Compiles a `call` instruction.
    fn visit_call(&mut self, func_idx: u32) -> Result<(), Self::Error> {
        // Skip this instruction if it is unreachable.
        if self.block(0).is_unreachable {
            return Ok(());
        }

        // Obtain the [`Func`] to be called for this instruction.
        let func = self.instance.func(func_idx).unwrap();

        // Obtain the type of the [`Func`] to be called.
        let type_ = func.type_(&self.store).clone();

        // Functions expect all their arguments to be temporary operands, and all registers to be
        // available.
        for opd_depth in 0..type_.params().len() {
            self.ensure_opd_not_imm(opd_depth);
            self.ensure_opd_not_local(opd_depth);
        }
        self.preserve_all_regs();

        // Emit the instruction.
        self.emit_instr(match func.0.as_ref(&self.store) {
            FuncEntity::Wasm(_) => exec::compile as ThreadedInstr,
            FuncEntity::Host(_) => exec::call_host as ThreadedInstr,
        });

        // Pop the arguments from the stack.
        for _ in 0..type_.params().len() {
            self.pop_opd();
        }

        // Emit an unguarded handle to the [`Func`] to be called.
        self.emit(func.0.to_unguarded(self.store.id()));

        // Compute the start and end of the call frame, and update the maximum stack height
        // attained by the [`Func`] being compiled.
        let call_frame_stack_start = self.first_temp_stack_idx + self.opds.len();
        let call_frame_stack_end = call_frame_stack_start + type_.call_frame_size();
        self.max_stack_height = self.max_stack_height.max(call_frame_stack_end);

        // Emit the stack offset of the end of the call frame.
        self.emit_stack_offset(call_frame_stack_end as isize);

        if let FuncEntity::Host(_) = func.0.as_ref(&self.store) {
            self.emit(
                self.instance
                    .mem(0)
                    .map(|mem| mem.0.to_unguarded(self.store.id())),
            );
        }

        // Push the results onto the stack.
        for result_type in type_.results().iter().copied() {
            self.push_opd(result_type);
        }
        Ok(())
    }

    /// Compiles a `call_indirect` instruction.
    fn visit_call_indirect(&mut self, table_idx: u32, type_idx: u32) -> Result<(), Self::Error> {
        // Skip this instruction if it is unreachable.
        if self.block(0).is_unreachable {
            return Ok(());
        }

        // Obtain the [`Table`] for this instruction.
        let table = self.instance.table(table_idx).unwrap();

        // Obtain the (interned) type of the elements in the [`Table`].
        let interned_type = self.instance.type_(type_idx).unwrap();
        let type_ = self.store.resolve_type(interned_type).clone();

        // The `call_indirect` instruction does not have an _i variant, so ensure that the index is
        // not an immediate operand.
        self.ensure_opd_not_imm(0);

        // Functions expect all their arguments to be temporary operands, and all registers to be
        // available.
        for opd_depth in 1..type_.params().len() + 1 {
            self.ensure_opd_not_imm(opd_depth);
            self.ensure_opd_not_local(opd_depth);
        }
        self.preserve_all_regs();

        // Emit the instruction.
        self.emit_instr(exec::call_indirect as ThreadedInstr);

        // Emit the function index and pop it from the stack.
        self.emit_and_pop_opd();

        // Pop the arguments from the stack.
        for _ in 0..type_.params().len() {
            self.pop_opd();
        }

        // Emit an unguarded handle to the [`Table`].
        self.emit(table.0.to_unguarded(self.store.id()));

        // Emit the interned type.
        self.emit(interned_type.to_unguarded(self.store.id()));

        // Compute the start and end of the call frame, and update the maximum stack height
        // attained by the [`Func`] being compiled.
        let call_frame_stack_start = self.first_temp_stack_idx + self.opds.len();
        let call_frame_stack_end = call_frame_stack_start + type_.call_frame_size();
        self.max_stack_height = self.max_stack_height.max(call_frame_stack_end as usize);

        // Emit the stack offset of the end of the call frame.
        self.emit_stack_offset(call_frame_stack_end as isize);

        // Emit an unguarded handle to the active [`Memory`] for the [`Func`] being compiled.
        self.emit(
            self.instance
                .mem(0)
                .map(|mem| mem.0.to_unguarded(self.store.id())),
        );

        // Push the results onto the stack.
        for result_type in type_.results().iter().copied() {
            self.push_opd(result_type);
        }

        Ok(())
    }

    // Reference instructions

    /// Compiles a `ref.null` instruction.
    fn visit_ref_null(&mut self, type_: RefType) -> Result<(), DecodeError> {
        // Skip this instruction if it is unreachable.
        if self.block(0).is_unreachable {
            return Ok(());
        }

        // Emit the instruction.
        self.emit_instr(select_ref_null(type_));

        match type_ {
            RefType::FuncRef => self.emit(FuncRef::null().to_unguarded(self.store.id())),
            RefType::ExternRef => self.emit(ExternRef::null().to_unguarded(self.store.id())),
        };

        // Push the output onto the stack and emit its stack offset.
        self.push_opd(ValType::FuncRef);
        self.emit_stack_offset(self.opd_stack_idx(0));

        Ok(())
    }

    /// Compiles a `ref.is_null` instruction.
    fn visit_ref_is_null(&mut self) -> Result<(), DecodeError> {
        if self.block(0).is_unreachable {
            return Ok(());
        }

        // Emit the instruction.
        self.emit_instr(select_ref_is_null(
            self.opd(0).type_.to_ref().unwrap(),
            self.opd(0).kind(),
        ));

        // Emit the input and pop it from the stack.
        self.emit_opd(0);
        self.pop_opd();

        // Push the output onto the stack and allocate a register for it.
        self.push_opd(ValType::I32);
        self.alloc_reg();

        Ok(())
    }

    /// Compiles a `ref.func` instruction.
    fn visit_ref_func(&mut self, func_idx: u32) -> Result<(), DecodeError> {
        // Skip this instruction if it is unreachable.
        if self.block(0).is_unreachable {
            return Ok(());
        }

        // Obtain the [`Func`] for this instruction.
        let func = self.instance.func(func_idx).unwrap();

        // Emit the instruction.
        self.emit_instr(exec::copy::<UnguardedFuncRef, Imm, Stk> as ThreadedInstr);
        
        // Emit an unguarded handle to the [`Func`].
        self.emit(func.to_unguarded(self.store.id()));

        // Push the output onto the stack and emit its stack offset.
        self.push_opd(ValType::FuncRef);
        self.emit_stack_offset(self.opd_stack_idx(0));

        Ok(())
    }

    // Parametric instructions

    /// Compiles a `drop` instruction
    fn visit_drop(&mut self) -> Result<(), DecodeError> {
        // Skip this instruction if it is unreachable.
        if self.block(0).is_unreachable {
            return Ok(());
        }

        // Pop the input from the stack.
        self.pop_opd();

        Ok(())
    }

    /// Compiles a `select` instruction.
    fn visit_select(&mut self, type_: Option<ValType>) -> Result<(), DecodeError> {
        // Skip this instruction if it is unreachable.
        if self.block(0).is_unreachable {
            return Ok(());
        }

        let type_ = type_.unwrap_or_else(|| self.opd(1).type_);

        // The `select` instruction does not have any _{sri}{sri}i variants.
        //
        // For instance, the following sequence of instructions:
        // local.get 0
        // local.get 1
        // i32.const 1
        //
        // will likely be constant folded by most Wasm compilers, so we expect it to occur very
        // rarely in real Wasm code. Therefore, we do not implement a select_i32_ssi instruction.
        //
        // Conversely, the following sequence of instructions:
        //
        // local.get 0
        // local.get 1
        // local.get 2
        //
        // cannot be constant folded, since the value of the condition cannot be known at compile
        // time. Therefore, we do implement a select_sss instruction.
        //
        // However, sequences like the first one above are still valid Wasm code, so we need to
        // handle them. We ensure that the condition is not an immediate operand, so that we can
        // use the _{sri}{sri}s variant instead (which is always available).
        self.ensure_opd_not_imm(0);

        // The select instruction writes it output to a register, so we need to ensure that the
        // register is available for the instruction to use.
        //
        // If the output register is already occupied, then we need to preserve the register on
        // the stack. Otherwise, the instruction will overwrite the register while it's already
        // occupied.
        //
        // The only exception is if one of the inputs occupies the output register. In that case,
        // the select instruction can safely overwrite the register, since the input will be
        // consumed by the instruction anyway.
        let output_reg_idx = type_.reg_idx();
        if self.is_reg_occupied(output_reg_idx)
            && !self.opd(2).occupies_reg(output_reg_idx)
            && !self.opd(1).occupies_reg(output_reg_idx)
            && !self.opd(0).occupies_reg(output_reg_idx)
        {
            self.preserve_reg(output_reg_idx);
        }

        // Emit the instruction.
        self.emit_instr(select_select(
            type_,
            self.opd(2).kind(),
            self.opd(1).kind(),
            self.opd(0).kind(),
        ));

        // Emit the inputs and pop them from the stack.
        for _ in 0..3 {
            self.emit_opd(0);
            self.pop_opd();
        }

        // Push the output onto the stack and allocate a register for it.
        self.push_opd(type_);
        self.alloc_reg();

        Ok(())
    }

    // Variable instructions

    /// Compiles a `local.get` instruction.
    fn visit_local_get(&mut self, local_idx: u32) -> Result<(), DecodeError> {
        // Skip this instruction if it is unreachable.
        if self.block(0).is_unreachable {
            return Ok(());
        }

        // Wasm uses `u32` indices for locals, but we use `usize` indices.
        let local_idx = local_idx as usize;

        // Push the output onto the stack and append it to the list of operands that refer to this
        // local.
        self.push_opd(self.locals[local_idx].type_);
        self.push_local_opd(local_idx);

        Ok(())
    }

    /// Compiles a `local.set` instruction.
    fn visit_local_set(&mut self, local_idx: u32) -> Result<(), DecodeError> {
        // Skip this instruction if it is unreachable.
        if self.block(0).is_unreachable {
            return Ok(());
        }

        // We compile local.set by delegating to the code for compiling local.tee. This works
        // because local.set is identical to local.tee, except that it pops its input from the
        // stack.
        self.visit_local_tee(local_idx)?;

        // Pop the input from the stack.
        self.pop_opd();

        Ok(())
    }

    /// Compiles a `local.tee` instruction.
    fn visit_local_tee(&mut self, local_idx: u32) -> Result<(), DecodeError> {
        // Skip this instruction if it is unreachable.
        if self.block(0).is_unreachable {
            return Ok(());
        }

        // Wasm uses `u32` indices for locals, but we use `usize` indices.
        let local_idx = local_idx as usize;

        // Obtain the type of the local.
        let local_type = self.locals[local_idx].type_;

        // The local.tee instruction overwrites the local, so we need to preserve all operands that
        // refer to the local on the stack.
        self.preserve_local(local_idx);

        // Emit the instruction.
        self.emit_instr(select_copy(local_type, self.opd(0).kind()));

        // Emit the input.
        self.emit_opd(0);

        // Emit the stack offset of the local.
        self.emit_stack_offset(self.local_stack_idx(local_idx));

        Ok(())
    }

    /// Compiles a `global.get` instruction.
    fn visit_global_get(&mut self, global_idx: u32) -> Result<(), DecodeError> {
        // Skip this instruction if it is unreachable.
        if self.block(0).is_unreachable {
            return Ok(());
        }

        // Obtain the [`Global`] for this instruction.
        let global = self.instance.global(global_idx).unwrap();

        // Obtain the type of the [`Global`].
        let val_type = global.type_(&self.store).val;

        // Emit the instruction.
        self.emit_instr(select_global_get(val_type));

        // Emit an unguarded handle to the [`Global`].
        self.emit(global.to_unguarded(self.store.id()));

        // Push the output onto the stack and emit its stack offset.
        self.push_opd(val_type);
        self.emit_stack_offset(self.opd_stack_idx(0));

        Ok(())
    }

    /// Compiles a `global.set` instruction.
    fn visit_global_set(&mut self, global_idx: u32) -> Result<(), DecodeError> {
        // Skip this instruction if it is unreachable.
        if self.block(0).is_unreachable {
            return Ok(());
        }

        // Obtain the [`Global`] for this instruction.
        let global = self.instance.global(global_idx).unwrap();

        // Obtain the type of the [`Global`].
        let val_type = global.type_(&self.store).val;

        // Emit the instruction.
        self.emit_instr(select_global_set(val_type, self.opd(0).kind()));

        // Emit the input and pop it from the stack.
        self.emit_opd(0);
        self.pop_opd();

        // Emit an unguarded handle to the [`Global`].
        self.emit(global.to_unguarded(self.store.id()));

        Ok(())
    }

    // Table instructions

    /// Compiles a `table.get` instruction.
    fn visit_table_get(&mut self, table_idx: u32) -> Result<(), Self::Error> {
        // Skip this instruction if it is unreachable.
        if self.block(0).is_unreachable {
            return Ok(());
        }

        // Obtain the [`Table`] for this instruction.
        let table = self.instance.table(table_idx).unwrap();

        // Obtain the type of the elements in the [`Table`].
        let elem_type = table.type_(&self.store).elem;

        // Emit the instruction.
        self.emit_instr(select_table_get(elem_type, self.opd(0).kind()));

        // Emit the input and pop it from the stack.
        self.emit_opd(0);
        self.pop_opd();

        // Emit an unguarded handle to the [`Table`].
        self.emit(table.to_unguarded(self.store.id()));

        // Push the output onto the stack and emit its stack offset.
        self.push_opd(ValType::I32);
        self.emit_stack_offset(self.opd_stack_idx(0));

        Ok(())
    }

    /// Compiles a `table.set` instruction.
    fn visit_table_set(&mut self, table_idx: u32) -> Result<(), Self::Error> {
        if self.block(0).is_unreachable {
            return Ok(());
        }

        // Obtain the [`Table`] for this instruction.
        let table = self.instance.table(table_idx).unwrap();

        // Obtain the type of the elements in the [`Table`].
        let elem_type = table.type_(&self.store).elem;

        // Emit the instruction.
        self.emit_instr(select_table_set(
            elem_type,
            self.opd(1).kind(),
            self.opd(0).kind(),
        ));

        // Emit the inputs and pop them from the stack.
        for _ in 0..2 {
            self.emit_opd(0);
            self.pop_opd();
        }

        // Emit an unguarded handle to the [`Table`].
        self.emit(table.to_unguarded(self.store.id()));

        Ok(())
    }

    /// Compiles a `table.size` instruction.
    fn visit_table_size(&mut self, table_idx: u32) -> Result<(), Self::Error> {
        // Skip this instruction if it is unreachable.
        if self.block(0).is_unreachable {
            return Ok(());
        }

        // Obtain the [`Table`] for this instruction.
        let table = self.instance.table(table_idx).unwrap();

        // Obtain the type of the elements in the [`Table`].
        let elem_type = table.type_(&self.store).elem;

        // Emit the instruction.
        self.emit_instr(select_table_size(elem_type));

        // Emit an unguarded handle to the [`Table`].
        self.emit(table.to_unguarded(self.store.id()));

        // Push the output onto the stack and emit its stack offset.
        self.push_opd(ValType::I32);
        self.emit_stack_offset(self.opd_stack_idx(0));

        Ok(())
    }

    /// Compiles a `table.grow` instruction.
    fn visit_table_grow(&mut self, table_idx: u32) -> Result<(), Self::Error> {
        // Skip this instruction if it is unreachable.
        if self.block(0).is_unreachable {
            return Ok(());
        }

        // Obtain the [`Table`] for this instruction.
        let table = self.instance.table(table_idx).unwrap();

        // Obtain the type of the elements in the [`Table`].
        let elem_type = table.type_(&self.store).elem;

        // This instruction has only one variant for each type, which reads all its operands from
        // the stack, so we need to ensure that all operands are neither constant nor register
        // operands.
        for opd_depth in 0..2 {
            self.ensure_opd_not_imm(opd_depth);
            self.ensure_opd_not_reg(opd_depth);
        }

        // Emit the instruction.
        self.emit_instr(select_table_grow(elem_type));

        // Emit the inputs and pop them from the stack.
        for _ in 0..2 {
            self.emit_opd(0);
            self.pop_opd();
        }

        // Emit an unguarded handle to the [`Table`].
        self.emit(table.to_unguarded(self.store.id()));

        // Push the output onto the stack and emit its stack offset.
        self.push_opd(ValType::I32);
        self.emit_stack_offset(self.opd_stack_idx(0));

        Ok(())
    }

    /// Compiles a `table.fill` instruction.
    fn visit_table_fill(&mut self, table_idx: u32) -> Result<(), Self::Error> {
        // Skip this instruction if it is unreachable.
        if self.block(0).is_unreachable {
            return Ok(());
        }

        // Obtain the [`Table`] for this instruction.
        let table = self.instance.table(table_idx).unwrap();

        // Obtain the type of the elements in the [`Table`].
        let elem_type = table.type_(&self.store).elem;

        // This instruction has only one variant for each type, which reads all its operands from
        // the stack, so we need to ensure that all operands are neither constants nor stored in a
        // register.
        for opd_depth in 0..3 {
            self.ensure_opd_not_imm(opd_depth);
            self.ensure_opd_not_reg(opd_depth);
        }

        // Emit the instruction.
        self.emit_instr(select_table_fill(elem_type));

        // Emit the inputs and pop them from the stack.
        for _ in 0..3 {
            self.emit_opd(0);
            self.pop_opd();
        }

        // Emit an unguarded handle to the [`Table`].
        self.emit(table.to_unguarded(self.store.id()));

        Ok(())
    }

    /// Compiles a `table.copy` instruction.
    fn visit_table_copy(
        &mut self,
        dst_table_idx: u32,
        src_table_idx: u32,
    ) -> Result<(), Self::Error> {
        // Skip this instruction if it is unreachable.
        if self.block(0).is_unreachable {
            return Ok(());
        }

        // Obtain the destination and source [`Table`] for this instruction.
        let dst_table = self.instance.table(dst_table_idx).unwrap();
        let src_table = self.instance.table(src_table_idx).unwrap();

        // Obtain the type of the elements in the destination [`Table`].
        let elem_type = dst_table.type_(&self.store).elem;

        // This instruction has only one variant for each type, which reads all its operands from
        // the stack, so we need to ensure that all operands are neither constant nor register
        // operands.
        for opd_depth in 0..3 {
            self.ensure_opd_not_imm(opd_depth);
            self.ensure_opd_not_reg(opd_depth);
        }

        // Emit the instruction.
        self.emit_instr(select_table_copy(elem_type));

        // Emit the inputs and pop them from the stack.
        for _ in 0..3 {
            self.emit_opd(0);
            self.pop_opd();
        }

        // Emit unguarded handles to the destination and source [`Table`].
        self.emit(dst_table.to_unguarded(self.store.id()));
        self.emit(src_table.to_unguarded(self.store.id()));

        Ok(())
    }

    /// Compiles a `table.init` instruction.
    fn visit_table_init(
        &mut self,
        dst_table_idx: u32,
        src_elem_idx: u32,
    ) -> Result<(), Self::Error> {
        // Skip this instruction if it is unreachable.
        if self.block(0).is_unreachable {
            return Ok(());
        }

        // Obtain the destination [`Table`] and source [`Elem`] for this instruction.
        let dst_table = self.instance.table(dst_table_idx).unwrap();
        let src_elem = self.instance.elem(src_elem_idx).unwrap();

        // Obtain the type of the elements in the destination [`Table`].
        let elem_type = dst_table.type_(&self.store).elem;

        // This instruction has only one variant for each type, which reads all its operands from
        // the stack, so we need to ensure that all operands are neither constant nor register
        // operands.
        for opd_depth in 0..3 {
            self.ensure_opd_not_imm(opd_depth);
            self.ensure_opd_not_reg(opd_depth);
        }

        // Emit the instruction.
        self.emit_instr(select_table_init(elem_type));

        // Emit the inputs and pop them from the stack.
        for _ in 0..3 {
            self.emit_opd(0);
            self.pop_opd();
        }

        // Emit unguarded handles to the destination [`Table`] and source [`Elem`].
        self.emit(dst_table.0.to_unguarded(self.store.id()));
        self.emit(src_elem.0.to_unguarded(self.store.id()));

        Ok(())
    }

    /// Compiles an `elem.drop` instruction.
    fn visit_elem_drop(&mut self, elem_idx: u32) -> Result<(), Self::Error> {
        // Skip this instruction if it is unreachable.
        if self.block(0).is_unreachable {
            return Ok(());
        }

        // Obtain the [`Elem`] for this instruction.
        let elem = self.instance.elem(elem_idx).unwrap();

        // Obtain the type of the elements in the [`Elem`].
        let elem_type = elem.type_(&self.store);

        // Emit the instruction.
        self.emit_instr(select_elem_drop(elem_type));

        // Emit an unguarded handle to the [`Elem`].
        self.emit(elem.to_unguarded(self.store.id()));

        Ok(())
    }

    // Memory instructions

    fn visit_i32_load(&mut self, arg: MemArg) -> Result<(), DecodeError> {
        self.compile_load::<i32>(arg)
    }

    fn visit_i64_load(&mut self, arg: MemArg) -> Result<(), DecodeError> {
        self.compile_load::<i64>(arg)
    }

    fn visit_f32_load(&mut self, arg: MemArg) -> Result<(), DecodeError> {
        self.compile_load::<f32>(arg)
    }

    fn visit_f64_load(&mut self, arg: MemArg) -> Result<(), DecodeError> {
        self.compile_load::<f64>(arg)
    }

    fn visit_i32_load8_s(&mut self, arg: MemArg) -> Result<(), DecodeError> {
        self.compile_load_n::<i32, i8>(arg)
    }

    fn visit_i32_load8_u(&mut self, arg: MemArg) -> Result<(), DecodeError> {
        self.compile_load_n::<u32, u8>(arg)
    }

    fn visit_i32_load16_s(&mut self, arg: MemArg) -> Result<(), DecodeError> {
        self.compile_load_n::<i32, i16>(arg)
    }

    fn visit_i32_load16_u(&mut self, arg: MemArg) -> Result<(), DecodeError> {
        self.compile_load_n::<u32, u16>(arg)
    }

    fn visit_i64_load8_s(&mut self, arg: MemArg) -> Result<(), DecodeError> {
        self.compile_load_n::<i64, i8>(arg)
    }

    fn visit_i64_load8_u(&mut self, arg: MemArg) -> Result<(), DecodeError> {
        self.compile_load_n::<u64, u8>(arg)
    }

    fn visit_i64_load16_s(&mut self, arg: MemArg) -> Result<(), DecodeError> {
        self.compile_load_n::<i64, i16>(arg)
    }

    fn visit_i64_load16_u(&mut self, arg: MemArg) -> Result<(), DecodeError> {
        self.compile_load_n::<u64, u16>(arg)
    }

    fn visit_i64_load32_s(&mut self, arg: MemArg) -> Result<(), DecodeError> {
        self.compile_load_n::<i64, i32>(arg)
    }

    fn visit_i64_load32_u(&mut self, arg: MemArg) -> Result<(), DecodeError> {
        self.compile_load_n::<u64, u32>(arg)
    }

    fn visit_i32_store(&mut self, arg: MemArg) -> Result<(), DecodeError> {
        self.compile_store::<i32>(arg)
    }

    fn visit_i64_store(&mut self, arg: MemArg) -> Result<(), DecodeError> {
        self.compile_store::<i64>(arg)
    }

    fn visit_f32_store(&mut self, arg: MemArg) -> Result<(), DecodeError> {
        self.compile_store::<f32>(arg)
    }

    fn visit_f64_store(&mut self, arg: MemArg) -> Result<(), DecodeError> {
        self.compile_store::<f64>(arg)
    }

    fn visit_i32_store8(&mut self, arg: MemArg) -> Result<(), DecodeError> {
        self.compile_store_n::<i32, i8>(arg)
    }

    fn visit_i32_store16(&mut self, arg: MemArg) -> Result<(), DecodeError> {
        self.compile_store_n::<i32, i16>(arg)
    }

    fn visit_i64_store8(&mut self, arg: MemArg) -> Result<(), DecodeError> {
        self.compile_store_n::<i64, i8>(arg)
    }

    fn visit_i64_store16(&mut self, arg: MemArg) -> Result<(), DecodeError> {
        self.compile_store_n::<i64, i16>(arg)
    }

    fn visit_i64_store32(&mut self, arg: MemArg) -> Result<(), DecodeError> {
        self.compile_store_n::<i64, i32>(arg)
    }

    /// Compiles a `memory.fill` instruction.
    fn visit_memory_size(&mut self) -> Result<(), Self::Error> {
        // Skip this instruction if it is unreachable.
        if self.block(0).is_unreachable {
            return Ok(());
        }

        // Obtain the [`Mem`] for this instruction.
        let mem = self.instance.mem(0).unwrap();

        // Emit the instruction.
        //
        // The cast to [`ThreadedInstr`] is necessary here, because otherwise we would emit a
        // function item instead of a function pointer.
        self.emit_instr(exec::memory_size::<Stk> as ThreadedInstr);

        // Emit an unguarded handle to the [`Mem`].
        self.emit(mem.to_unguarded(self.store.id()));

        // Push the output onto the stack and emit its stack offset.
        self.push_opd(ValType::I32);
        self.emit_stack_offset(self.opd_stack_idx(0));

        Ok(())
    }

    /// Compiles a `memory.grow` instruction.
    fn visit_memory_grow(&mut self) -> Result<(), Self::Error> {
        // Skip this instruction if it is unreachable.
        if self.block(0).is_unreachable {
            return Ok(());
        }

        // Obtain the [`Mem`] for this instruction.
        let mem = self.instance.mem(0).unwrap();

        // This instruction has only one variant, which reads all its operands from the stack, so we
        // need to ensure that all operands are neither constant nor register operands.
        self.ensure_opd_not_imm(0);
        self.ensure_opd_not_reg(0);

        // Emit the instruction.
        //
        // The cast to [`ThreadedInstr`] is necessary here, because otherwise we would emit a
        // function item instead of a function pointer.
        self.emit_instr(exec::memory_grow::<Stk, Stk> as ThreadedInstr);

        // Emit the input and pop it from the stack.
        self.emit_and_pop_opd();

        // Emit an unguarded handle to the memory instance.
        self.emit(mem.to_unguarded(self.store.id()));

        // Push the output onto the stack and emit its stack offset.
        self.push_opd(ValType::I32);
        self.emit_stack_offset(self.opd_stack_idx(0));

        Ok(())
    }

    /// Compiles a `memory.fill` instruction.
    fn visit_memory_fill(&mut self) -> Result<(), Self::Error> {
        // Skip this instruction if it is unreachable.
        if self.block(0).is_unreachable {
            return Ok(());
        }

        // Obtain the [`Mem`] for this instruction.
        let mem = self.instance.mem(0).unwrap();

        // This instruction has only one variant, which reads all its operands from the stack, so we
        // need to ensure that all operands are neither constant nor register operands.
        for opd_depth in 0..3 {
            self.ensure_opd_not_imm(opd_depth);
            self.ensure_opd_not_reg(opd_depth);
        }

        // Emit the instruction.
        //
        // The cast to [`ThreadedInstr`] is necessary here, because otherwise we would emit a
        // function item instead of a function pointer.
        self.emit_instr(exec::memory_fill::<Stk, Stk, Stk> as ThreadedInstr);

        // Emit the inputs and pop them from the stack.
        for _ in 0..3 {
            self.emit_opd(0);
            self.pop_opd();
        }

        // Emit an unguarded handle to the [`Mem`].
        self.emit(mem.to_unguarded(self.store.id()));

        Ok(())
    }

    /// Compiles a `memory.copy` instruction.
    fn visit_memory_copy(&mut self) -> Result<(), Self::Error> {
        // Skip this instruction if it is unreachable.
        if self.block(0).is_unreachable {
            return Ok(());
        }

        // Obtain the [`Mem`] for this instruction.
        let mem = self.instance.mem(0).unwrap();

        // This instruction has only one variant, which reads all its operands from the stack, so we
        // need to ensure that all operands are neither constant nor register operands.
        for opd_depth in 0..3 {
            self.ensure_opd_not_imm(opd_depth);
            self.ensure_opd_not_reg(opd_depth);
        }

        // Emit the instruction.
        //
        // The cast to [`ThreadedInstr`] is necessary here, because otherwise we would emit a
        // function item instead of a function pointer.
        self.emit_instr(exec::memory_copy::<Stk, Stk, Stk> as ThreadedInstr);

        // Emit the inputs and pop them from the stack.
        for _ in 0..3 {
            self.emit_opd(0);
            self.pop_opd();
        }

        // Emit an unguarded handle to the [`Mem`].
        self.emit(mem.to_unguarded(self.store.id()));

        Ok(())
    }

    /// Compiles a `memory.init` instruction.
    fn visit_memory_init(&mut self, data_idx: u32) -> Result<(), Self::Error> {
        // Skip this instruction if it is unreachable.
        if self.block(0).is_unreachable {
            return Ok(());
        }

        // Obtain the destination [`Mem`] and source [`Data`] for this instruction.
        let dst_mem = self.instance.mem(0).unwrap();
        let src_data = self.instance.data(data_idx).unwrap();

        // This instruction has only one variant, which reads all its operands from the stack, so we
        // need to ensure that all operands are neither constant nor register operands.
        for opd_depth in 0..3 {
            self.ensure_opd_not_imm(opd_depth);
            self.ensure_opd_not_reg(opd_depth);
        }

        // Emit the instruction.
        //
        // The cast to [`ThreadedInstr`] is necessary here, because otherwise we would emit a
        // function item instead of a function pointer.
        self.emit_instr(exec::memory_init::<Stk, Stk, Stk> as ThreadedInstr);

        // Emit the inputs and pop them from the stack.
        for _ in 0..3 {
            self.emit_opd(0);
            self.pop_opd();
        }

        // Emit unguarded handles to the destination [`Mem`] and source [`Data`] instance.
        self.emit(dst_mem.to_unguarded(self.store.id()));
        self.emit(src_data.to_unguarded(self.store.id()));

        Ok(())
    }

    /// Compiles a `data.drop` instruction.
    fn visit_data_drop(&mut self, data_idx: u32) -> Result<(), Self::Error> {
        // Skip this instruction if it is unreachable.
        if self.block(0).is_unreachable {
            return Ok(());
        }

        // Obtain the [`Data`] for this instruction.
        let data = self.instance.data(data_idx).unwrap();

        // Emit the instruction.
        self.emit_instr(exec::data_drop as ThreadedInstr);

        // Emit an unguarded handle to the [`Data`].
        self.emit(data.to_unguarded(self.store.id()));

        Ok(())
    }

    // Numeric instructions

    /// Compiles an `i32.const` instruction.
    fn visit_i32_const(&mut self, val: i32) -> Result<(), DecodeError> {
        // Skip this instruction if it is unreachable.
        if self.block(0).is_unreachable {
            return Ok(());
        }

        // Push the output onto the stack and set its value.
        //
        // Setting its value will mark the operand as a immediate operand.
        self.push_opd(ValType::I32);
        self.opd_mut(0).val = Some(UnguardedVal::I32(val));
        Ok(())
    }

    /// Compiles an `i64.const` instruction.
    fn visit_i64_const(&mut self, val: i64) -> Result<(), DecodeError> {
        // Skip this instruction if it is unreachable.
        if self.block(0).is_unreachable {
            return Ok(());
        }

        // Push the output onto the stack and set its value.
        //
        // Setting its value will mark the operand as a immediate operand.
        self.push_opd(ValType::I64);
        self.opd_mut(0).val = Some(UnguardedVal::I64(val));

        Ok(())
    }

    /// Compiles an `f32.const` instruction.
    fn visit_f32_const(&mut self, val: f32) -> Result<(), DecodeError> {
        // Skip this instruction if it is unreachable.
        if self.block(0).is_unreachable {
            return Ok(());
        }

        // Push the output onto the stack and set its value.
        //
        // Setting its value will mark the operand as a immediate operand.
        self.push_opd(ValType::F32);
        self.opd_mut(0).val = Some(UnguardedVal::F32(val));

        Ok(())
    }

    /// Compiles a `f64.const` instruction.
    fn visit_f64_const(&mut self, val: f64) -> Result<(), DecodeError> {
        // Skip this instruction if it is unreachable.
        if self.block(0).is_unreachable {
            return Ok(());
        }

        // Push the output onto the stack and set its value.
        //
        // Setting its value will mark the operand as a constant.
        self.push_opd(ValType::F64);
        self.opd_mut(0).val = Some(UnguardedVal::F64(val));

        Ok(())
    }

    fn visit_i32_eqz(&mut self) -> Result<(), DecodeError> {
        self.compile_un_op::<i32, Eqz>()
    }

    fn visit_i32_eq(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<i32, Eq>()
    }

    fn visit_i32_ne(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<i32, Ne>()
    }

    fn visit_i32_lt_s(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<i32, Lt>()
    }

    fn visit_i32_lt_u(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<u32, Lt>()
    }

    fn visit_i32_gt_s(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<i32, Gt>()
    }

    fn visit_i32_gt_u(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<u32, Gt>()
    }

    fn visit_i32_le_s(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<i32, Le>()
    }

    fn visit_i32_le_u(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<u32, Le>()
    }

    fn visit_i32_ge_s(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<i32, Ge>()
    }

    fn visit_i32_ge_u(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<u32, Ge>()
    }

    fn visit_i64_eqz(&mut self) -> Result<(), DecodeError> {
        self.compile_un_op::<i64, Eqz>()
    }

    fn visit_i64_eq(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<i64, Eq>()
    }

    fn visit_i64_ne(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<i64, Ne>()
    }

    fn visit_i64_lt_s(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<i64, Lt>()
    }

    fn visit_i64_lt_u(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<u64, Lt>()
    }

    fn visit_i64_gt_s(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<i64, Gt>()
    }

    fn visit_i64_gt_u(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<u64, Gt>()
    }

    fn visit_i64_le_s(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<i64, Le>()
    }

    fn visit_i64_le_u(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<u64, Le>()
    }

    fn visit_i64_ge_s(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<i64, Ge>()
    }

    fn visit_i64_ge_u(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<u64, Ge>()
    }

    fn visit_f32_eq(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<f32, Eq>()
    }

    fn visit_f32_ne(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<f32, Ne>()
    }

    fn visit_f32_lt(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<f32, Lt>()
    }

    fn visit_f32_gt(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<f32, Gt>()
    }

    fn visit_f32_le(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<f32, Le>()
    }

    fn visit_f32_ge(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<f32, Ge>()
    }

    fn visit_f64_eq(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<f64, Eq>()
    }

    fn visit_f64_ne(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<f64, Ne>()
    }

    fn visit_f64_lt(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<f64, Lt>()
    }

    fn visit_f64_gt(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<f64, Gt>()
    }

    fn visit_f64_le(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<f64, Le>()
    }

    fn visit_f64_ge(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<f64, Ge>()
    }

    fn visit_i32_clz(&mut self) -> Result<(), DecodeError> {
        self.compile_un_op::<i32, Clz>()
    }

    fn visit_i32_ctz(&mut self) -> Result<(), DecodeError> {
        self.compile_un_op::<i32, Ctz>()
    }

    fn visit_i32_popcnt(&mut self) -> Result<(), DecodeError> {
        self.compile_un_op::<i32, Popcnt>()
    }
    
    fn visit_i32_add(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<i32, Add>()
    }

    fn visit_i32_sub(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<i32, Sub>()
    }

    fn visit_i32_mul(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<i32, Mul>()
    }

    fn visit_i32_div_s(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<i32, Div>()
    }

    fn visit_i32_div_u(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<u32, Div>()
    }

    fn visit_i32_rem_s(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<i32, Rem>()
    }

    fn visit_i32_rem_u(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<u32, Rem>()
    }

    fn visit_i32_and(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<i32, And>()
    }

    fn visit_i32_or(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<i32, Or>()
    }

    fn visit_i32_xor(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<i32, Xor>()
    }

    fn visit_i32_shl(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<i32, Shl>()
    }

    fn visit_i32_shr_s(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<i32, Shr>()
    }

    fn visit_i32_shr_u(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<u32, Shr>()
    }

    fn visit_i32_rotl(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<i32, Rotl>()
    }

    fn visit_i32_rotr(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<i32, Rotr>()
    }

    fn visit_i64_clz(&mut self) -> Result<(), DecodeError> {
        self.compile_un_op::<i64, Clz>()
    }

    fn visit_i64_ctz(&mut self) -> Result<(), DecodeError> {
        self.compile_un_op::<i64, Ctz>()
    }

    fn visit_i64_popcnt(&mut self) -> Result<(), DecodeError> {
        self.compile_un_op::<i64, Popcnt>()
    }

    fn visit_i64_add(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<i64, Add>()
    }

    fn visit_i64_sub(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<i64, Sub>()
    }

    fn visit_i64_mul(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<i64, Mul>()
    }

    fn visit_i64_div_s(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<i64, Div>()
    }

    fn visit_i64_div_u(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<u64, Div>()
    }

    fn visit_i64_rem_s(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<i64, Rem>()
    }

    fn visit_i64_rem_u(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<u64, Rem>()
    }

    fn visit_i64_and(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<i64, And>()
    }

    fn visit_i64_or(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<i64, Or>()
    }

    fn visit_i64_xor(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<i64, Xor>()
    }

    fn visit_i64_shl(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<i64, Shl>()
    }

    fn visit_i64_shr_s(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<i64, Shr>()
    }

    fn visit_i64_shr_u(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<u64, Shr>()
    }

    fn visit_i64_rotl(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<i64, Rotl>()
    }

    fn visit_i64_rotr(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<i64, Rotr>()
    }

    fn visit_f32_abs(&mut self) -> Result<(), DecodeError> {
        self.compile_un_op::<f32, Abs>()
    }

    fn visit_f32_neg(&mut self) -> Result<(), DecodeError> {
        self.compile_un_op::<f32, Neg>()
    }

    fn visit_f32_ceil(&mut self) -> Result<(), DecodeError> {
        self.compile_un_op::<f32, Ceil>()
    }

    fn visit_f32_floor(&mut self) -> Result<(), DecodeError> {
        self.compile_un_op::<f32, Floor>()
    }

    fn visit_f32_trunc(&mut self) -> Result<(), DecodeError> {
        self.compile_un_op::<f32, Trunc>()
    }

    fn visit_f32_nearest(&mut self) -> Result<(), DecodeError> {
        self.compile_un_op::<f32, Nearest>()
    }

    fn visit_f32_sqrt(&mut self) -> Result<(), DecodeError> {
        self.compile_un_op::<f32, Sqrt>()
    }

    fn visit_f32_add(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<f32, Add>()
    }

    fn visit_f32_sub(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<f32, Sub>()
    }

    fn visit_f32_mul(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<f32, Mul>()
    }

    fn visit_f32_div(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<f32, Div>()
    }

    fn visit_f32_min(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<f32, Min>()
    }

    fn visit_f32_max(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<f32, Max>()
    }

    fn visit_f32_copysign(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<f32, Copysign>()
    }

    fn visit_f64_abs(&mut self) -> Result<(), DecodeError> {
        self.compile_un_op::<f64, Abs>()
    }

    fn visit_f64_neg(&mut self) -> Result<(), DecodeError> {
        self.compile_un_op::<f64, Neg>()
    }

    fn visit_f64_ceil(&mut self) -> Result<(), DecodeError> {
        self.compile_un_op::<f64, Ceil>()
    }

    fn visit_f64_floor(&mut self) -> Result<(), DecodeError> {
        self.compile_un_op::<f64, Floor>()
    }

    fn visit_f64_trunc(&mut self) -> Result<(), DecodeError> {
        self.compile_un_op::<f64, Trunc>()
    }

    fn visit_f64_nearest(&mut self) -> Result<(), DecodeError> {
        self.compile_un_op::<f64, Nearest>()
    }

    fn visit_f64_sqrt(&mut self) -> Result<(), DecodeError> {
        self.compile_un_op::<f64, Sqrt>()
    }

    fn visit_f64_add(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<f64, Add>()
    }

    fn visit_f64_sub(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<f64, Sub>()
    }

    fn visit_f64_mul(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<f64, Mul>()
    }

    fn visit_f64_div(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<f64, Div>()
    }

    fn visit_f64_min(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<f64, Min>()
    }

    fn visit_f64_max(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<f64, Max>()
    }

    fn visit_f64_copysign(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<f64, Copysign>()
    }

    fn visit_i32_wrap_i64(&mut self) -> Result<(), DecodeError> {
        self.compile_un_op::<i64, WrapTo<i32>>()
    }

    fn visit_i32_trunc_f32_s(&mut self) -> Result<(), DecodeError> {
        self.compile_un_op::<f32, TruncTo<i32>>()
    }

    fn visit_i32_trunc_f32_u(&mut self) -> Result<(), DecodeError> {
        self.compile_un_op::<f32, TruncTo<u32>>()
    }

    fn visit_i32_trunc_f64_s(&mut self) -> Result<(), DecodeError> {
        self.compile_un_op::<f64, TruncTo<i32>>()
    }

    fn visit_i32_trunc_f64_u(&mut self) -> Result<(), DecodeError> {
        self.compile_un_op::<f64, TruncTo<u32>>()
    }

    fn visit_i64_extend_i32_s(&mut self) -> Result<(), DecodeError> {
        self.compile_un_op::<i32, ExtendTo<i64>>()
    }

    fn visit_i64_extend_i32_u(&mut self) -> Result<(), DecodeError> {
        self.compile_un_op::<u32, ExtendTo<u64>>()
    }

    fn visit_i64_trunc_f32_s(&mut self) -> Result<(), DecodeError> {
        self.compile_un_op::<f32, TruncTo<i64>>()
    }

    fn visit_i64_trunc_f32_u(&mut self) -> Result<(), DecodeError> {
        self.compile_un_op::<f32, TruncTo<u64>>()
    }

    fn visit_i64_trunc_f64_s(&mut self) -> Result<(), DecodeError> {
        self.compile_un_op::<f64, TruncTo<i64>>()
    }

    fn visit_i64_trunc_f64_u(&mut self) -> Result<(), DecodeError> {
        self.compile_un_op::<f64, TruncTo<u64>>()
    }

    fn visit_f32_convert_i32_s(&mut self) -> Result<(), DecodeError> {
        self.compile_un_op::<i32, ConvertTo<f32>>()
    }

    fn visit_f32_convert_i32_u(&mut self) -> Result<(), DecodeError> {
        self.compile_un_op::<u32, ConvertTo<f32>>()
    }

    fn visit_f32_convert_i64_s(&mut self) -> Result<(), DecodeError> {
        self.compile_un_op::<i64, ConvertTo<f32>>()
    }

    fn visit_f32_convert_i64_u(&mut self) -> Result<(), DecodeError> {
        self.compile_un_op::<u64, ConvertTo<f32>>()
    }

    fn visit_f32_demote_f64(&mut self) -> Result<(), DecodeError> {
        self.compile_un_op::<f64, DemoteTo<f32>>()
    }

    fn visit_f64_convert_i32_s(&mut self) -> Result<(), DecodeError> {
        self.compile_un_op::<i32, ConvertTo<f64>>()
    }

    fn visit_f64_convert_i32_u(&mut self) -> Result<(), DecodeError> {
        self.compile_un_op::<u32, ConvertTo<f64>>()
    }

    fn visit_f64_convert_i64_s(&mut self) -> Result<(), DecodeError> {
        self.compile_un_op::<i64, ConvertTo<f64>>()
    }

    fn visit_f64_convert_i64_u(&mut self) -> Result<(), DecodeError> {
        self.compile_un_op::<u64, ConvertTo<f64>>()
    }

    fn visit_f64_promote_f32(&mut self) -> Result<(), DecodeError> {
        self.compile_un_op::<f32, PromoteTo<f64>>()
    }

    fn visit_i32_reinterpret_f32(&mut self) -> Result<(), DecodeError> {
        self.compile_un_op::<f32, ReinterpretTo<i32>>()
    }

    fn visit_i64_reinterpret_f64(&mut self) -> Result<(), DecodeError> {
        self.compile_un_op::<f64, ReinterpretTo<i64>>()
    }

    fn visit_f32_reinterpret_i32(&mut self) -> Result<(), DecodeError> {
        self.compile_un_op::<i32, ReinterpretTo<f32>>()
    }

    fn visit_f64_reinterpret_i64(&mut self) -> Result<(), DecodeError> {
        self.compile_un_op::<i64, ReinterpretTo<f64>>()
    }

    fn visit_i32_extend8_s(&mut self) -> Result<(), DecodeError> {
        self.compile_un_op::<i32, ExtendFrom<i8>>()
    }

    fn visit_i32_extend16_s(&mut self) -> Result<(), DecodeError> {
        self.compile_un_op::<i32, ExtendFrom<i16>>()
    }

    fn visit_i64_extend8_s(&mut self) -> Result<(), DecodeError> {
        self.compile_un_op::<i64, ExtendFrom<i8>>()
    }

    fn visit_i64_extend16_s(&mut self) -> Result<(), DecodeError> {
        self.compile_un_op::<i64, ExtendFrom<i16>>()
    }

    fn visit_i64_extend32_s(&mut self) -> Result<(), DecodeError> {
        self.compile_un_op::<i64, ExtendFrom<i32>>()
    }

    fn visit_i32_trunc_sat_f32_s(&mut self) -> Result<(), Self::Error> {
        self.compile_un_op::<f32, TruncSatTo<i32>>()
    }

    fn visit_i32_trunc_sat_f32_u(&mut self) -> Result<(), Self::Error> {
        self.compile_un_op::<f32, TruncSatTo<u32>>()
    }

    fn visit_i32_trunc_sat_f64_s(&mut self) -> Result<(), Self::Error> {
        self.compile_un_op::<f64, TruncSatTo<i32>>()
    }

    fn visit_i32_trunc_sat_f64_u(&mut self) -> Result<(), Self::Error> {
        self.compile_un_op::<f64, TruncSatTo<u32>>()
    }

    fn visit_i64_trunc_sat_f32_s(&mut self) -> Result<(), Self::Error> {
        self.compile_un_op::<f32, TruncSatTo<i64>>()
    }

    fn visit_i64_trunc_sat_f32_u(&mut self) -> Result<(), Self::Error> {
        self.compile_un_op::<f32, TruncSatTo<u64>>()
    }

    fn visit_i64_trunc_sat_f64_s(&mut self) -> Result<(), Self::Error> {
        self.compile_un_op::<f64, TruncSatTo<i64>>()
    }

    fn visit_i64_trunc_sat_f64_u(&mut self) -> Result<(), Self::Error> {
        self.compile_un_op::<f64, TruncSatTo<u64>>()
    }
}

/// A local on the stack.
#[derive(Clone, Copy, Debug)]
struct Local {
    // The type of this local.
    type_: ValType,
    // The index of the first operand in the list of operands for this local.
    first_opd_idx: Option<usize>,
}

/// A block on the stack.
#[derive(Clone, Debug)]
struct Block {
    // The [`BlockKind`] of this block.
    kind: BlockKind,
    // The type of this block.
    type_: FuncType,
    // Whether the rest of this block is unreachable.
    is_unreachable: bool,
    // The height of the operand stack at the start of this block.
    height: usize,
    // The index of the first instruction for this block.
    first_instr_offset: usize,
    // The index of the hole for the start of the `else` block. This is only used for `if` blocks.
    else_hole_offset: Option<usize>,
    // The index of the first hole for this block.
    first_hole_offset: Option<usize>,
}

impl Block {
    /// Returns the type of the label of this [`Block`].
    fn label_types(&self) -> LabelTypes {
        LabelTypes {
            kind: self.kind,
            type_: self.type_.clone(),
        }
    }
}

/// The kind of a [`Block`].
///
/// This determines whether the label of the [`Block`] is at the start or the end.
///
/// Blocks introduced by a `block``, `if``, or `else`` instruction are considered to be of kind
/// [`BlockKind::Block`], since their label is at the start of the block, and there there is no
/// need to otherwise distinguish between them.
///  
/// Blocks introduced by a `loop` instruction are considered to be of kind [`BlockKind::Loop`],
/// since their label is at the end of the block.
#[derive(Clone, Copy, Debug)]
enum BlockKind {
    Block,
    Loop,
}

/// The type of the label of a [`Block`].
///
/// This is either the type of the inputs of the block, or the type of the outputs of the block,
/// depending on the [`BlockKind`] of the block.
#[derive(Clone, Debug)]
struct LabelTypes {
    kind: BlockKind,
    type_: FuncType,
}

impl Deref for LabelTypes {
    type Target = [ValType];

    fn deref(&self) -> &Self::Target {
        match self.kind {
            BlockKind::Block => self.type_.results(),
            BlockKind::Loop => self.type_.params(),
        }
    }
}

/// An operand on the stack.
///
/// Every operand carries:
/// - Its type
///
/// - An implicit stack index
///   This is the index of the stack slot to be used for the operand, if it is stored on the stack.
///   It is determined by the position of the operand on the operand stack. Note that we reserve a
///   stack slot for an operand even if it is not stored on the stack.
///
/// - An implicit register index
///   This is the index of the register to be used for the operand, if it is stored in a register.
///   It is determined by the type of the operand. Note that an operand has a register index even
///   if it is not stored in a register.
///
/// An operand can be either:
///
/// - A local operand
///   These operands are created by instructions that write their output to a register, such as
///   i32.add. They are not stored on the stack, but instead carry the index of the local they refer
///   to.
///
///   When a local is overwritten, all local operands that refer to it should be preserved on the
///   stack. To keep track of which operands refer to a given local, we maintain a linked list of
///   operands for each local. Each local operand carries the index of the previous and next operand
///   in the list.
///
/// - A register operand
///   These operands are created by instructions that write their output to a register, such as
///   i32.add.
///
/// - A immediate operand
///   These operands are created by constant instructions, such as i32.const. They are not stored
///   on the stack, but instead carry their value with them.
///
/// - A temporary operand
///   These operands are neither immediate, local, nor register operands. They are created by
///   instructions that write their output to the stack, or when an immediate, local, or register
///   operand is preserved on the stack.
#[derive(Clone, Copy, Debug)]
struct Opd {
    // The type of this operand.
    type_: ValType,
    // The value of this operand, if it is a immediate operand.
    val: Option<UnguardedVal>,
    // The index of the local this operand refers to, if it is a local operand.
    local_idx: Option<usize>,
    // The index of the previous operand in the list of operands for the local that this operand
    // refers to, if it is a local operand.
    prev_opd_idx: Option<usize>,
    // The index of the next operand in the list of operands for the local this this operand refers
    // to, if it is a local operand.
    next_opd_idx: Option<usize>,
    // Whether this operand is stored in a register.
    is_reg: bool,
}

impl Opd {
    /// Returns `true` if this operand is a immediate operand.
    fn is_imm(&self) -> bool {
        self.val.is_some()
    }

    /// Returns `true` if this operand is a local operand.
    fn is_local(&self) -> bool {
        self.local_idx.is_some()
    }

    /// Returns `true` if this operand occupies the register with the given index.
    fn occupies_reg(&self, reg_idx: usize) -> bool {
        self.is_reg && self.type_.reg_idx() == reg_idx
    }

    /// Returns the kind of this operand (see [`OpdKind`]).
    fn kind(&self) -> OpdKind {
        if self.is_imm() {
            OpdKind::Imm
        } else if self.is_reg {
            OpdKind::Reg
        } else {
            OpdKind::Stk
        }
    }
}

/// The kind of an [`Opd`].
///
/// This indicates whether the value of the operand can be read from the stack, a register, or as
/// an immediate.
///
/// Local operands are considered to be of kind [`OpdKind::Stack`], since even though they are not
/// stored on the stack, the locals they refer to are, so their value can still be read from the
/// stack.
#[derive(Clone, Copy, Debug)]
enum OpdKind {
    Stk,
    Reg,
    Imm,
}

// Instruction selection
//
// Most instructions come in multiple variants, depending on the types of their operands, and
// whether their operands are stored as an immediate, on the stack, or in a register. These
// functions are used to select a suitable variant of an instruction based on the types and
// kinds of its operands.

fn select_br_if_z(kind: OpdKind) -> ThreadedInstr {
    match kind {
        OpdKind::Imm => exec::br_if_z::<Imm>,
        OpdKind::Stk => exec::br_if_z::<Stk>,
        OpdKind::Reg => exec::br_if_z::<Reg>,
    }
}

fn select_br_if_nz(kind: OpdKind) -> ThreadedInstr {
    match kind {
        OpdKind::Imm => exec::br_if_nz::<Imm>,
        OpdKind::Stk => exec::br_if_nz::<Stk>,
        OpdKind::Reg => exec::br_if_nz::<Reg>,
    }
}

fn select_br_table(kind: OpdKind) -> ThreadedInstr {
    match kind {
        OpdKind::Imm => exec::br_table::<Imm>,
        OpdKind::Stk => exec::br_table::<Stk>,
        OpdKind::Reg => exec::br_table::<Reg>,
    }
}

fn select_ref_null(type_: RefType) -> ThreadedInstr {
    match type_ {
        RefType::FuncRef => exec::copy::<UnguardedFuncRef, Imm, Stk>,
        RefType::ExternRef => exec::copy::<UnguardedExternRef, Imm, Stk>,
    }
}

fn select_ref_is_null(type_: RefType, input: OpdKind) -> ThreadedInstr {
    match type_ {
        RefType::FuncRef => select_un_op::<UnguardedFuncRef, IsNull>(input),
        RefType::ExternRef => select_un_op::<UnguardedExternRef, IsNull>(input),
    }
}

fn select_select(type_: ValType, input_0: OpdKind, input_1: OpdKind, input_2: OpdKind) -> ThreadedInstr {
    match type_ {
        ValType::I32 => select_select_inner::<i32>(input_0, input_1, input_2),
        ValType::I64 => select_select_inner::<i64>(input_0, input_1, input_2),
        ValType::F32 => select_select_inner::<f32>(input_0, input_1, input_2),
        ValType::F64 => select_select_inner::<f64>(input_0, input_1, input_2),
        ValType::FuncRef => select_select_inner::<UnguardedFuncRef>(input_0, input_1, input_2),
        ValType::ExternRef => select_select_inner::<UnguardedExternRef>(input_0, input_1, input_2),
    }
}

fn select_select_inner<T>(input_0: OpdKind, input_1: OpdKind, input_2: OpdKind) -> ThreadedInstr
where
    T: ReadReg + WriteReg
{
    match (input_0, input_1, input_2) {
        (OpdKind::Imm, OpdKind::Imm, OpdKind::Imm) => exec::select::<T, Imm, Imm, Imm, Reg>,
        (OpdKind::Stk, OpdKind::Imm, OpdKind::Imm) => exec::select::<T, Stk, Imm, Imm, Reg>,
        (OpdKind::Reg, OpdKind::Imm, OpdKind::Imm) => exec::select::<T, Reg, Imm, Imm, Reg>,
        (OpdKind::Imm, OpdKind::Stk, OpdKind::Imm) => exec::select::<T, Imm, Stk, Imm, Reg>,
        (OpdKind::Stk, OpdKind::Stk, OpdKind::Imm) => exec::select::<T, Stk, Stk, Imm, Reg>,
        (OpdKind::Reg, OpdKind::Stk, OpdKind::Imm) => exec::select::<T, Reg, Stk, Imm, Reg>,
        (OpdKind::Imm, OpdKind::Reg, OpdKind::Imm) => exec::select::<T, Imm, Reg, Imm, Reg>,
        (OpdKind::Stk, OpdKind::Reg, OpdKind::Imm) => exec::select::<T, Stk, Reg, Imm, Reg>,
        (OpdKind::Reg, OpdKind::Reg, OpdKind::Imm) => exec::select::<T, Reg, Reg, Imm, Reg>,
        (OpdKind::Imm, OpdKind::Imm, OpdKind::Stk) => exec::select::<T, Imm, Imm, Stk, Reg>,
        (OpdKind::Stk, OpdKind::Imm, OpdKind::Stk) => exec::select::<T, Stk, Imm, Stk, Reg>,
        (OpdKind::Reg, OpdKind::Imm, OpdKind::Stk) => exec::select::<T, Reg, Imm, Stk, Reg>,
        (OpdKind::Imm, OpdKind::Stk, OpdKind::Stk) => exec::select::<T, Imm, Stk, Stk, Reg>,
        (OpdKind::Stk, OpdKind::Stk, OpdKind::Stk) => exec::select::<T, Stk, Stk, Stk, Reg>,
        (OpdKind::Reg, OpdKind::Stk, OpdKind::Stk) => exec::select::<T, Reg, Stk, Stk, Reg>,
        (OpdKind::Imm, OpdKind::Reg, OpdKind::Stk) => exec::select::<T, Imm, Reg, Stk, Reg>,
        (OpdKind::Stk, OpdKind::Reg, OpdKind::Stk) => exec::select::<T, Stk, Reg, Stk, Reg>,
        (OpdKind::Reg, OpdKind::Reg, OpdKind::Stk) => exec::select::<T, Reg, Reg, Stk, Reg>,
        (OpdKind::Imm, OpdKind::Imm, OpdKind::Reg) => exec::select::<T, Imm, Imm, Reg, Reg>,
        (OpdKind::Stk, OpdKind::Imm, OpdKind::Reg) => exec::select::<T, Stk, Imm, Reg, Reg>,
        (OpdKind::Reg, OpdKind::Imm, OpdKind::Reg) => exec::select::<T, Reg, Imm, Reg, Reg>,
        (OpdKind::Imm, OpdKind::Stk, OpdKind::Reg) => exec::select::<T, Imm, Stk, Reg, Reg>,
        (OpdKind::Stk, OpdKind::Stk, OpdKind::Reg) => exec::select::<T, Stk, Stk, Reg, Reg>,
        (OpdKind::Reg, OpdKind::Stk, OpdKind::Reg) => exec::select::<T, Reg, Stk, Reg, Reg>,
        (OpdKind::Imm, OpdKind::Reg, OpdKind::Reg) => exec::select::<T, Imm, Reg, Reg, Reg>,
        (OpdKind::Stk, OpdKind::Reg, OpdKind::Reg) => exec::select::<T, Stk, Reg, Reg, Reg>,
        (OpdKind::Reg, OpdKind::Reg, OpdKind::Reg) => exec::select::<T, Reg, Reg, Reg, Reg>,
    }
}

fn select_global_get(type_: ValType) -> ThreadedInstr {
    match type_ {
        ValType::I32 => exec::global_get::<i32, Stk>,
        ValType::I64 => exec::global_get::<i64, Stk>,
        ValType::F32 => exec::global_get::<f32, Stk>,
        ValType::F64 => exec::global_get::<f64, Stk>,
        ValType::FuncRef => exec::global_get::<UnguardedFuncRef, Stk>,
        ValType::ExternRef => exec::global_get::<UnguardedExternRef, Stk>,
    }
}

fn select_global_set(type_: ValType, input: OpdKind) -> ThreadedInstr {
    match type_ {
        ValType::I32 => select_global_set_inner::<i32>(input),
        ValType::I64 => select_global_set_inner::<i64>(input),
        ValType::F32 => select_global_set_inner::<f32>(input),
        ValType::F64 => select_global_set_inner::<f64>(input),
        ValType::FuncRef => select_global_set_inner::<UnguardedFuncRef>(input),
        ValType::ExternRef => select_global_set_inner::<UnguardedExternRef>(input),
    }
}

fn select_global_set_inner<T>(input: OpdKind) -> ThreadedInstr
where
    GlobalEntityT<T>: DowncastMut<GlobalEntity>,
    T: Copy + ReadReg
{
    match input {
        OpdKind::Imm => exec::global_set::<T, Imm>,
        OpdKind::Stk => exec::global_set::<T, Stk>,
        OpdKind::Reg => exec::global_set::<T, Reg>,
    }
}

fn select_table_get(type_: RefType, input: OpdKind) -> ThreadedInstr {
    match type_ {
        RefType::FuncRef => select_table_get_inner::<UnguardedFuncRef>(input),
        RefType::ExternRef => select_table_get_inner::<UnguardedExternRef>(input),
    }
}

fn select_table_get_inner<T>(input: OpdKind) -> ThreadedInstr
where
    TableEntityT<T>: DowncastRef<TableEntity>,
    T: Copy + ReadReg + WriteReg
{
    match input {
        OpdKind::Imm => exec::table_get::<T, Imm, Stk>,
        OpdKind::Stk => exec::table_get::<T, Stk, Stk>,
        OpdKind::Reg => exec::table_get::<T, Reg, Stk>,
    }
}

fn select_table_set(type_: RefType, input_0: OpdKind, input_1: OpdKind) -> ThreadedInstr {
    match type_ {
        RefType::FuncRef => select_table_set_inner::<UnguardedFuncRef>(input_0, input_1),
        RefType::ExternRef => select_table_set_inner::<UnguardedExternRef>(input_0, input_1),
    }
}

fn select_table_set_inner<T>(input_0: OpdKind, input_1: OpdKind) -> ThreadedInstr
where
    TableEntityT<T>: DowncastMut<TableEntity>,
    T: Copy + ReadReg
{
    match (input_0, input_1) {
        (OpdKind::Imm, OpdKind::Imm) => exec::table_set::<T, Imm, Imm>,
        (OpdKind::Stk, OpdKind::Imm) => exec::table_set::<T, Stk, Imm>,
        (OpdKind::Reg, OpdKind::Imm) => exec::table_set::<T, Reg, Imm>,
        (OpdKind::Imm, OpdKind::Stk) => exec::table_set::<T, Imm, Stk>,
        (OpdKind::Stk, OpdKind::Stk) => exec::table_set::<T, Stk, Stk>,
        (OpdKind::Reg, OpdKind::Stk) => exec::table_set::<T, Reg, Stk>,
        (OpdKind::Imm, OpdKind::Reg) => exec::table_set::<T, Imm, Reg>,
        (OpdKind::Stk, OpdKind::Reg) => exec::table_set::<T, Stk, Reg>,
        (OpdKind::Reg, OpdKind::Reg) => exec::table_set::<T, Reg, Reg>,
    }
}

fn select_table_size(type_: RefType) -> ThreadedInstr {
    match type_ {
        RefType::FuncRef => exec::table_size::<UnguardedFuncRef, Stk>,
        RefType::ExternRef => exec::table_size::<UnguardedExternRef, Stk>,
    }
}

fn select_table_grow(type_: RefType) -> ThreadedInstr {
    match type_ {
        RefType::FuncRef => exec::table_grow::<UnguardedFuncRef, Stk, Stk, Stk>,
        RefType::ExternRef => exec::table_grow::<UnguardedExternRef, Stk, Stk, Stk>,
    }
}

fn select_table_fill(type_: RefType) -> ThreadedInstr {
    match type_ {
        RefType::FuncRef => exec::table_fill::<UnguardedFuncRef, Stk, Stk, Stk>,
        RefType::ExternRef => exec::table_fill::<UnguardedExternRef, Stk, Stk, Stk>,
    }
}

fn select_table_copy(type_: RefType) -> ThreadedInstr {
    match type_ {
        RefType::FuncRef => exec::table_copy::<UnguardedFuncRef, Stk, Stk, Stk>,
        RefType::ExternRef => exec::table_copy::<UnguardedExternRef, Stk, Stk, Stk>,
    }
}

fn select_table_init(type_: RefType) -> ThreadedInstr {
    match type_ {
        RefType::FuncRef => exec::table_init::<UnguardedFuncRef, Stk, Stk, Stk>,
        RefType::ExternRef => exec::table_init::<UnguardedExternRef, Stk, Stk, Stk>,
    }
}

fn select_elem_drop(type_: RefType) -> ThreadedInstr {
    match type_ {
        RefType::FuncRef => exec::elem_drop::<UnguardedFuncRef>,
        RefType::ExternRef => exec::elem_drop::<UnguardedExternRef>,
    }
}

fn select_load<T>(input: OpdKind) -> ThreadedInstr
where
    T: ReadFromPtr + WriteReg
{
    match input {
        OpdKind::Imm => exec::load::<T, Imm, Reg>,
        OpdKind::Stk => exec::load::<T, Stk, Reg>,
        OpdKind::Reg => exec::load::<T, Reg, Reg>,
    }
}

fn select_load_n<Dst, Src>(input: OpdKind) -> ThreadedInstr
where
    Dst: ExtendingCastFrom<Src> + WriteReg,
    Src: ReadFromPtr + ExtendingCast,
{
    match input {
        OpdKind::Imm => exec::load_n::<Dst, Src, Imm, Reg>,
        OpdKind::Stk => exec::load_n::<Dst, Src, Stk, Reg>,
        OpdKind::Reg => exec::load_n::<Dst, Src, Reg, Reg>,
    }
}

fn select_store<T>(input_0: OpdKind, input_1: OpdKind) -> ThreadedInstr
where
    T: ReadReg + WriteToPtr
{
    match (input_0, input_1) {
        (OpdKind::Imm, OpdKind::Imm) => exec::store::<T, Imm, Imm>,
        (OpdKind::Stk, OpdKind::Imm) => exec::store::<T, Stk, Imm>,
        (OpdKind::Reg, OpdKind::Imm) => exec::store::<T, Reg, Imm>,
        (OpdKind::Imm, OpdKind::Stk) => exec::store::<T, Imm, Stk>,
        (OpdKind::Stk, OpdKind::Stk) => exec::store::<T, Stk, Stk>,
        (OpdKind::Reg, OpdKind::Stk) => exec::store::<T, Reg, Stk>,
        (OpdKind::Imm, OpdKind::Reg) => exec::store::<T, Imm, Reg>,
        (OpdKind::Stk, OpdKind::Reg) => exec::store::<T, Stk, Reg>,
        (OpdKind::Reg, OpdKind::Reg) => exec::store::<T, Reg, Reg>,
    }
}

fn select_store_n<Src, Dst>(input_0: OpdKind, input_1: OpdKind) -> ThreadedInstr
where
    Src: ReadReg + WrappingCast,
    Dst: WrappingCastFrom<Src> + WriteToPtr,
{
    match (input_0, input_1) {
        (OpdKind::Imm, OpdKind::Imm) => exec::store_n::<Src, Dst, Imm, Imm>,
        (OpdKind::Stk, OpdKind::Imm) => exec::store_n::<Src, Dst, Stk, Imm>,
        (OpdKind::Reg, OpdKind::Imm) => exec::store_n::<Src, Dst, Reg, Imm>,
        (OpdKind::Imm, OpdKind::Stk) => exec::store_n::<Src, Dst, Imm, Stk>,
        (OpdKind::Stk, OpdKind::Stk) => exec::store_n::<Src, Dst, Stk, Stk>,
        (OpdKind::Reg, OpdKind::Stk) => exec::store_n::<Src, Dst, Reg, Stk>,
        (OpdKind::Imm, OpdKind::Reg) => exec::store_n::<Src, Dst, Imm, Reg>,
        (OpdKind::Stk, OpdKind::Reg) => exec::store_n::<Src, Dst, Stk, Reg>,
        (OpdKind::Reg, OpdKind::Reg) => exec::store_n::<Src, Dst, Reg, Reg>,
    }
}

fn select_un_op<T, U>(input: OpdKind) -> ThreadedInstr
where
    T: ReadReg,
    U: UnOp<T>,
    U::Output: WriteReg
{
    match input {
        OpdKind::Imm => exec::un_op::<T, U, Imm, Reg>,
        OpdKind::Stk => exec::un_op::<T, U, Stk, Reg>,
        OpdKind::Reg => exec::un_op::<T, U, Reg, Reg>,
    }
}

fn select_bin_op<T, B>(input_0: OpdKind, input_1: OpdKind) -> ThreadedInstr
where
    T: ReadReg,
    B: BinOp<T>,
    B::Output: WriteReg
{
    match (input_0, input_1) {
        (OpdKind::Imm, OpdKind::Imm) => exec::bin_op::<T, B, Imm, Imm, Reg>,
        (OpdKind::Stk, OpdKind::Imm) => exec::bin_op::<T, B, Stk, Imm, Reg>,
        (OpdKind::Reg, OpdKind::Imm) => exec::bin_op::<T, B, Reg, Imm, Reg>,
        (OpdKind::Imm, OpdKind::Stk) => exec::bin_op::<T, B, Imm, Stk, Reg>,
        (OpdKind::Stk, OpdKind::Stk) => exec::bin_op::<T, B, Stk, Stk, Reg>,
        (OpdKind::Reg, OpdKind::Stk) => exec::bin_op::<T, B, Reg, Stk, Reg>,
        (OpdKind::Imm, OpdKind::Reg) => exec::bin_op::<T, B, Imm, Reg, Reg>,
        (OpdKind::Stk, OpdKind::Reg) => exec::bin_op::<T, B, Stk, Reg, Reg>,
        (OpdKind::Reg, OpdKind::Reg) => exec::bin_op::<T, B, Reg, Reg, Reg>,
    }
}

fn select_copy(type_: ValType, kind: OpdKind) -> ThreadedInstr {
    match type_ {
        ValType::I32 => select_copy_inner::<i32>(kind),
        ValType::I64 => select_copy_inner::<i64>(kind),
        ValType::F32 => select_copy_inner::<f32>(kind),
        ValType::F64 => select_copy_inner::<f64>(kind),
        ValType::FuncRef => select_copy_inner::<UnguardedFuncRef>(kind),
        ValType::ExternRef => select_copy_inner::<UnguardedExternRef>(kind),
    }
}

fn select_copy_inner<T>(kind: OpdKind) -> ThreadedInstr
where
    T: ReadReg + WriteReg
{
    match kind {
        OpdKind::Imm => exec::copy::<T, Imm, Stk>,
        OpdKind::Stk => exec::copy::<T, Stk, Stk>,
        OpdKind::Reg => exec::copy::<T, Reg, Stk>,
    }
}
