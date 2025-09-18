use {
    crate::{
        aliasable_box::AliasableBox,
        code,
        code::{
            BinOpInfo, BlockType, CompiledCode, InstrSlot, InstrVisitor, LoadInfo, MemArg,
            StoreInfo, UnOpInfo, UncompiledCode,
        },
        decode::DecodeError,
        exec,
        exec::ThreadedInstr,
        extern_ref::ExternRef,
        func::{Func, FuncEntity, FuncType},
        func_ref::FuncRef,
        instance::Instance,
        ref_::RefType,
        stack::StackSlot,
        store::Store,
        val::{UnguardedVal, ValType},
    },
    std::{mem, ops::Deref},
};

#[derive(Clone, Debug)]
pub(crate) struct Compiler {
    label_idxs: Vec<u32>,
    locals: Vec<Local>,
    blocks: Vec<Block>,
    opds: Vec<Opd>,
    fixup_idxs: Vec<usize>,
}

impl Compiler {
    pub(crate) fn new() -> Self {
        Self {
            label_idxs: Vec::new(),
            locals: Vec::new(),
            blocks: Vec::new(),
            opds: Vec::new(),
            fixup_idxs: Vec::new(),
        }
    }

    pub(crate) fn compile(
        &mut self,
        store: &mut Store,
        func: Func,
        instance: &Instance,
        code: &UncompiledCode,
    ) -> CompiledCode {
        use crate::decode::Decoder;

        self.locals.clear();
        self.blocks.clear();
        self.opds.clear();
        self.fixup_idxs.clear();

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
            locals,
            blocks: &mut self.blocks,
            opds: &mut self.opds,
            fixup_idxs: &mut self.fixup_idxs,
            first_param_result_stack_idx: -(type_.call_frame_size() as isize),
            first_temp_stack_idx: local_count,
            max_stack_height: local_count,
            regs: [None; 2],
            code: Vec::new(),
        };
        compile.push_block(
            BlockKind::Block,
            FuncType::new([], type_.results().iter().copied()),
        );

        compile.emit(exec::enter as ThreadedInstr);
        compile.emit(func.to_unguarded(store.id()));
        compile.emit(
            compile
                .instance
                .mem(0)
                .map(|mem| mem.to_unguarded(store.id())),
        );

        let mut decoder = Decoder::new(&code.expr);
        while !compile.blocks.is_empty() {
            code::decode_instr(&mut decoder, &mut self.label_idxs, &mut compile).unwrap();
        }

        for (result_idx, result_type) in type_.clone().results().iter().copied().enumerate().rev() {
            compile.emit(select_copy_stack(result_type));
            compile.emit_stack_offset(compile.temp_stack_idx(result_idx));
            compile.emit_stack_offset(compile.param_result_stack_idx(result_idx));
        }
        compile.emit(exec::return_ as ThreadedInstr);

        let mut code: AliasableBox<[InstrSlot]> = AliasableBox::from_box(Box::from(compile.code));
        for fixup_idx in compile.fixup_idxs.drain(..) {
            code[fixup_idx] += code.as_ptr() as usize;
        }

        CompiledCode {
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
    locals: &'a mut [Local],
    blocks: &'a mut Vec<Block>,
    opds: &'a mut Vec<Opd>,
    fixup_idxs: &'a mut Vec<usize>,
    first_param_result_stack_idx: isize,
    first_temp_stack_idx: usize,
    max_stack_height: usize,
    regs: [Option<usize>; 2],
    code: Vec<InstrSlot>,
}

impl<'a> Compile<'a> {
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
    fn push_hole(&mut self, block_idx: usize, hole_idx: usize) {
        // We use the hole itself to store the index of the next hole. The value `usize::MAX` is
        // used to indicate the absence of a next hole.
        self.code[hole_idx] = self.block(block_idx).first_hole_idx.unwrap_or(usize::MAX);
        self.block_mut(block_idx).first_hole_idx = Some(hole_idx);
    }

    /// Pops a hole from the block with the given index.
    fn pop_hole(&mut self, block_idx: usize) -> Option<usize> {
        if let Some(hole_idx) = self.block(block_idx).first_hole_idx {
            // We use the hole itself to store the index of the next hole. The value `usize::MAX` is
            // used to indicate the absence of a next hole.
            self.block_mut(block_idx).first_hole_idx = if self.code[hole_idx] == usize::MAX {
                None
            } else {
                Some(self.code[hole_idx])
            };
            Some(hole_idx)
        } else {
            None
        }
    }

    /// Pushes a block with the given kind and type on stack.
    fn push_block(&mut self, kind: BlockKind, type_: FuncType) {
        self.blocks.push(Block {
            kind,
            type_,
            is_unreachable: false,
            height: self.opds.len(),
            first_instr_idx: self.code.len(),
            first_hole_idx: None,
            else_hole_idx: None,
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
        self.emit(select_copy_imm_to_stack(self.opds[opd_idx].type_));
        self.emit_val(self.opds[opd_idx].val.unwrap());
        self.emit_stack_offset(self.temp_stack_idx(opd_idx));
        self.opd_mut(opd_depth).val = None;
    }

    /// Preserve a local operand by copying the local it refers to to the stack, if necessary.
    fn preserve_local_opd(&mut self, opd_idx: usize) {
        let local_idx = self.opds[opd_idx].local_idx.unwrap();
        self.emit(select_copy_stack(self.locals[local_idx].type_));
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
        self.emit(select_copy_reg_to_stack(opd_type));
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
            self.emit(select_copy_stack(label_type));
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
        self.code.push(InstrSlot::default());
        unsafe { *(self.code.last_mut().unwrap() as *mut _ as *mut T) = val };
    }

    // Emits an operand.
    //
    // For local and temporary operands, which can be read from the stack, this emits the offset of
    // the stack slot for the operand. For immediate operands, which carry their own value, this
    // emits the value of the operand. For register operands, we don't need to anything.
    fn emit_opd(&mut self, opd_depth: usize) {
        match self.opd(opd_depth).kind() {
            OpdKind::Stack => self.emit_stack_offset(self.opd_stack_idx(opd_depth)),
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
        self.emit(stack_idx * mem::size_of::<StackSlot>() as isize);
    }

    /// Emits the label for the block with the given index.
    ///
    /// If the block is of kind [`BlockKind::Loop`], this emits the offset of the first instruction
    /// in the block. [`BlockKind::Block`], we don't yet know where the first instruction after the
    /// end of the block is, so we emit a hole instead.
    fn emit_label(&mut self, block_idx: usize) {
        match self.block(block_idx).kind {
            BlockKind::Block => {
                let hole_idx = self.emit_hole();
                self.push_hole(block_idx, hole_idx);
            }
            BlockKind::Loop => {
                self.emit_instr_offset(self.block(block_idx).first_instr_idx);
            }
        }
    }

    /// Emits a hole and returns its index.
    ///
    /// A hole is a placeholder for an instruction offset that is not yet known.
    fn emit_hole(&mut self) -> usize {
        let hole_idx = self.code.len();
        self.code.push(0);
        hole_idx
    }

    /// Patches the hole with the given index with the offset of the current instruction.
    fn patch_hole(&mut self, hole_idx: usize) {
        self.fixup_idxs.push(hole_idx);
        self.code[hole_idx] = self.code.len() * mem::size_of::<usize>();
    }

    /// Emits the offset of the instruction with the given index.
    fn emit_instr_offset(&mut self, instr_idx: usize) {
        self.fixup_idxs.push(self.code.len());
        self.emit(instr_idx * mem::size_of::<InstrSlot>());
    }
}

impl<'a> InstrVisitor for Compile<'a> {
    type Error = DecodeError;

    // Control instructions

    /// Compiles a `nop` instruction.
    fn visit_nop(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }

    /// Compiles an `unreachable` instruction.
    fn visit_unreachable(&mut self) -> Result<(), Self::Error> {
        // Emit the instruction.
        self.emit(exec::unreachable as ThreadedInstr);

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

        let else_hole_idx = if !self.block(0).is_unreachable {
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
            self.emit(select_br_if_z(self.opd(0).kind()));

            // Emit the condition and pop it from the stack.
            self.emit_and_pop_opd();

            // We don't yet know where the start of the `else` block is, so we emit a hole for it instead.
            let else_hole_idx = self.emit_hole();

            // Pop the inputs of the block from the stack.
            for _ in 0..type_.params().len() {
                self.pop_opd();
            }

            Some(else_hole_idx)
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
        self.block_mut(0).else_hole_idx = else_hole_idx;

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
            self.emit(exec::br as ThreadedInstr);
            let hole_idx = self.emit_hole();
            self.push_hole(0, hole_idx);
        }

        // We are now at the start of the `else` block, so we can patch the hole for the start of
        // the `else` block.
        //
        // We do this even if rest of the the current `if` block is unreachable, since the hole
        // itself could still be reachable.
        if let Some(else_hole_idx) = self.block_mut(0).else_hole_idx.take() {
            self.patch_hole(else_hole_idx);
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
        self.block_mut(0).first_hole_idx = block.first_hole_idx;

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
        if let Some(else_hole_idx) = self.block_mut(0).else_hole_idx.take() {
            self.patch_hole(else_hole_idx);
        }

        // We are now at the first instruction after the end of the block, so we can patch the list
        // of holes for the block.
        //
        // We do this even if the rest of the current block is unreachable, since the holes themselves could
        // still be reachable.
        while let Some(hole_idx) = self.pop_hole(0) {
            self.patch_hole(hole_idx);
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
        self.emit(exec::br as ThreadedInstr);
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
            self.emit(select_br_if_nz(self.opd(0).kind()));
            self.emit_and_pop_opd();
            self.emit_label(label_idx);
        } else {
            // If the branch target has a non-empty type, we need to copy all block inputs to their
            // expected locations.
            //
            // This is more expensive, because we cannot branch to the target directly. Instead, we
            // have to branch to code that first copies the block inputs to their expected
            // locations, and then branches to the target.
            self.emit(select_br_if_z(self.opd(0).kind()));
            self.emit_and_pop_opd();
            let hole_idx = self.emit_hole();
            self.resolve_label_vals(label_idx);
            self.emit(exec::br as ThreadedInstr);
            self.emit_label(label_idx);
            self.patch_hole(hole_idx);
        }

        for label_type in self.block(label_idx).label_types().iter().copied() {
            self.push_opd(label_type);
        }

        Ok(())
    }

    /// Compiles a `br_table` instruction.
    fn visit_br_table(
        &mut self,
        label_idxs: &[u32],
        default_label_idx: u32,
    ) -> Result<(), Self::Error> {
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
            self.emit(select_br_table(self.opd(0).kind()));
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
            self.emit(select_br_table(self.opd(0).kind()));
            self.emit_and_pop_opd();
            self.emit(label_idxs.len() as u32);
            let mut hole_idxs = Vec::new();
            for _ in 0..label_idxs.len() {
                let hole_idx = self.emit_hole();
                hole_idxs.push(hole_idx);
            }
            let default_hole_idx = self.emit_hole();
            for (label_idx, hole_idx) in label_idxs.iter().copied().zip(hole_idxs) {
                let label_idx = label_idx as usize;
                self.patch_hole(hole_idx);
                self.resolve_label_vals(label_idx);
                self.emit(exec::br as ThreadedInstr);
                self.emit_label(label_idx);
                for label_type in self.block(label_idx).label_types().iter().copied() {
                    self.push_opd(label_type);
                }
            }
            self.patch_hole(default_hole_idx);
            self.resolve_label_vals(default_label_idx);
            self.emit(exec::br as ThreadedInstr);
            self.emit_label(default_label_idx);
        }

        // After a `br_table` instruction, the rest of the block is unreachable.
        self.set_unreachable();

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
            self.emit(if self.opd(0).is_reg {
                select_copy_reg_to_stack(result_type)
            } else {
                select_copy_stack(result_type)
            });
            self.emit_and_pop_opd();
            self.emit_stack_offset(self.param_result_stack_idx(result_idx));
        }

        // Emit the instruction.
        self.emit(exec::return_ as ThreadedInstr);

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
        self.emit(match func.0.as_ref(&self.store) {
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
        self.emit(exec::call_indirect as ThreadedInstr);

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
        self.emit(select_ref_null(type_));

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
        self.emit(select_ref_is_null(
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
        self.emit(exec::copy_imm_to_stack_func_ref as ThreadedInstr);

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
        self.emit(select_select(
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
        self.emit(selecy_copy_opd_to_stack(local_type, self.opd(0).kind()));

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
        self.emit(select_global_get(val_type));

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
        self.emit(select_global_set(val_type, self.opd(0).kind()));

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
        self.emit(select_table_get(elem_type, self.opd(0).kind()));

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
        self.emit(select_table_set(
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
        self.emit(select_table_size(elem_type));

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
        self.emit(select_table_grow(elem_type));

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
        self.emit(select_table_fill(elem_type));

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
        self.emit(select_table_copy(elem_type));

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
        self.emit(select_table_init(elem_type));

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
        self.emit(select_elem_drop(elem_type));

        // Emit an unguarded handle to the [`Elem`].
        self.emit(elem.to_unguarded(self.store.id()));

        Ok(())
    }

    // Memory instructions

    /// Compiles a load instruction.
    fn visit_load(&mut self, arg: MemArg, info: LoadInfo) -> Result<(), DecodeError> {
        // Skip this instruction if it is unreachable.
        if self.block(0).is_unreachable {
            return Ok(());
        }

        // We compile load instructions by delegating to the code for compiling unary operations.
        // This works because load instructions are essentially unary operations with an extra
        // immediate operand.
        self.visit_un_op(info.op)?;

        // Emit the static offset.
        self.emit(arg.offset);

        Ok(())
    }

    /// Compiles a store instruction.
    fn visit_store(&mut self, arg: MemArg, info: StoreInfo) -> Result<(), DecodeError> {
        // Skip this instruction if it is unreachable.
        if self.block(0).is_unreachable {
            return Ok(());
        }

        // We compile store instructions by delegating to the code for compiling binary operations.
        // This works because store instructions are essentially binary operations with an extra
        // immediate operand.
        self.visit_bin_op(info.op)?;

        // Emit the static offset.
        self.emit(arg.offset);

        Ok(())
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
        self.emit(exec::memory_size as ThreadedInstr);

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
        self.emit(exec::memory_grow as ThreadedInstr);

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
        self.emit(exec::memory_fill as ThreadedInstr);

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
        self.emit(exec::memory_copy as ThreadedInstr);

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
        self.emit(exec::memory_init as ThreadedInstr);

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
        self.emit(exec::data_drop as ThreadedInstr);

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

    /// Compiles a unary operation.
    fn visit_un_op(&mut self, info: UnOpInfo) -> Result<(), DecodeError> {
        // Skip this instruction if it is unreachable.
        if self.block(0).is_unreachable {
            return Ok(());
        }

        // Not all unary operation have an _i variant.
        //
        // For instance, the following sequence of instructions:
        //
        // i32.const 21
        // i32.neg
        //
        // will likely be constant folded by most Wasm compilers, so we expect it to occur very
        // rarely in real Wasm code. Therefore, we do not implement an i32_neg_i instruction.
        //
        // Conversely, the following sequence of instructions:
        // i32.const 1
        // i32.load
        //
        // cannot be constant folded, since i32.load has side effects. Therefore, we do implement
        // an i32_load_i instruction.
        //
        // However, sequences like the first one above are still valid Wasm code, so we need to
        // handle them. If the operation does not have an _i variant, we ensure that the operand is
        // not an immediate operand, so that we can use the _s variant instead (which is always
        // available).
        if info.instr_i.is_none() {
            self.ensure_opd_not_imm(0);
        }

        // Unary operations write their output to a register, so we need to ensure that the output
        // register is available for the operation to use.
        //
        // If this operation has an output, and the output register is already occupied, then we
        // need to preserve the register on the stack. Otherwise, the operation will overwrite the
        // register while it's already occupied.
        //
        // The only exception is if the input occupies the output register. In that case, the
        // operation can safely overwrite the register, since the input will be consumed by the
        // operation anyway.
        if let Some(output_type) = info.output_type {
            let output_reg_idx = output_type.reg_idx();
            if self.is_reg_occupied(output_reg_idx) && !self.opd(0).occupies_reg(output_reg_idx) {
                self.preserve_reg(output_reg_idx);
            }
        }

        // Emit the instruction.
        self.emit(select_un_op(info, self.opd(0).kind()));

        // Emit and pop the inputs from the stack.
        self.emit_opd(0);
        self.pop_opd();

        // If the operation has an output, push the output onto the stack and allocate a register
        // for it.
        if let Some(output_type) = info.output_type {
            self.push_opd(output_type);
            self.alloc_reg();
        }

        Ok(())
    }

    /// Compiles a binary operation
    fn visit_bin_op(&mut self, info: BinOpInfo) -> Result<(), DecodeError> {
        // Skip this instruction if it is unreachable.
        if self.block(0).is_unreachable {
            return Ok(());
        }

        // Not all binary operations have an _ii variant.
        //
        // For instance, the following sequence of instructions:
        // i32.const 1
        // i32.const 2
        // i32.add
        //
        // will likely be constant folded by most Wasm compilers, so we expect it to occur very
        // rarely in real Wasm code. Therefore, we do not implement an i32_add_ii instruction.
        //
        // Conversely, the following sequence of instructions:
        // i32.const 1
        // i32.const 2
        // i32.store
        //
        // cannot be constant folded, since i32.store has side effects. Therefore, we do implement
        // an i32_store_ii instruction.
        //
        // However, sequences like the first one above are still valid Wasm code, so we need to
        // handle them. If the first operand is an immediate operand, and the operation does not
        // have an _ii variant, we ensure that the second operand is not an immediate operand, so
        // that we can use the _is variant instead (which is always available).
        if self.opd(1).is_imm() && info.instr_ii.is_none() {
            self.ensure_opd_not_imm(0);
        }

        // Binary operations write their output to a register, so we need to ensure that the output
        // register is available for the operation to use.
        //
        // If this operation has an output, and the output register is already occupied, then we
        // need to preserve the register on the stack. Otherwise, the operation will the register
        // while it's already occupied.
        //
        // The only exception is if one of the inputs occupies the output register. In that case,
        // the operation can safely overwrite the register, since the input will be consumed by the
        // operation anyway.
        if let Some(output_type) = info.output_type {
            let output_reg_idx = output_type.reg_idx();
            if self.is_reg_occupied(output_reg_idx)
                && !self.opd(1).occupies_reg(output_reg_idx)
                && !self.opd(0).occupies_reg(output_reg_idx)
            {
                self.preserve_reg(output_reg_idx);
            }
        }

        // Emit the instruction.
        self.emit(select_bin_op(info, self.opd(1).kind(), self.opd(0).kind()));

        // Emit the inputs and pop them from the stack.
        //
        // Commutative binary operations do not have an _sr, _si, or _ri variant. Since the order
        // of the operands does not matter for these operations, we can implement these variants
        // by swapping the operands, and forwarding to the _rs, _is, or _ir variant, respectively
        // (which are always available).
        //
        // We only need to swap the order in which the operands are emitted for the _si variant,
        // since we never emit anything for register operands.
        match (self.opd(1).kind(), self.opd(0).kind()) {
            (OpdKind::Stack, OpdKind::Imm) if info.instr_is == info.instr_si => {
                self.emit_opd(1);
                self.emit_opd(0);
                self.pop_opd();
                self.pop_opd();
            }
            _ => {
                self.emit_opd(0);
                self.pop_opd();
                self.emit_opd(0);
                self.pop_opd();
            }
        }

        // If the operation has an output, push the output onto the stack and allocate a register
        // for it.
        if let Some(output_type) = info.output_type {
            self.push_opd(output_type);
            self.alloc_reg();
        }

        Ok(())
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
    first_instr_idx: usize,
    // The index of the hole for the start of the `else` block. This is only used for `if` blocks.
    else_hole_idx: Option<usize>,
    // The index of the first hole for this block.
    first_hole_idx: Option<usize>,
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
            OpdKind::Stack
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
    Stack,
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
        OpdKind::Stack => exec::br_if_z_s,
        OpdKind::Reg => exec::br_if_z_r,
        OpdKind::Imm => panic!("no suitable instruction found"),
    }
}

fn select_br_if_nz(kind: OpdKind) -> ThreadedInstr {
    match kind {
        OpdKind::Stack => exec::br_if_nz_s,
        OpdKind::Reg => exec::br_if_nz_r,
        OpdKind::Imm => panic!("no suitable instruction found"),
    }
}

fn select_br_table(kind: OpdKind) -> ThreadedInstr {
    match kind {
        OpdKind::Stack => exec::br_table_s,
        OpdKind::Reg => exec::br_table_r,
        OpdKind::Imm => panic!("no suitable instruction found"),
    }
}

fn select_ref_null(type_: RefType) -> ThreadedInstr {
    match type_ {
        RefType::FuncRef => exec::copy_imm_to_stack_func_ref,
        RefType::ExternRef => exec::copy_imm_to_stack_extern_ref,
    }
}

fn select_ref_is_null(type_: RefType, kind: OpdKind) -> ThreadedInstr {
    match (type_, kind) {
        (RefType::FuncRef, OpdKind::Stack) => exec::ref_is_null_func_ref_s,
        (RefType::FuncRef, OpdKind::Reg) => exec::ref_is_null_func_ref_r,

        (RefType::ExternRef, OpdKind::Stack) => exec::ref_is_null_extern_ref_s,
        (RefType::ExternRef, OpdKind::Reg) => exec::ref_is_null_extern_ref_r,

        (_, OpdKind::Imm) => panic!("no suitable instruction found"),
    }
}

fn select_select(
    type_: ValType,
    kind_0: OpdKind,
    kind_1: OpdKind,
    kind_2: OpdKind,
) -> ThreadedInstr {
    match (type_, kind_0, kind_1, kind_2) {
        (ValType::I32, OpdKind::Stack, OpdKind::Stack, OpdKind::Stack) => exec::select_i32_sss,
        (ValType::I32, OpdKind::Reg, OpdKind::Stack, OpdKind::Stack) => exec::select_i32_rss,
        (ValType::I32, OpdKind::Imm, OpdKind::Stack, OpdKind::Stack) => exec::select_i32_iss,
        (ValType::I32, OpdKind::Stack, OpdKind::Reg, OpdKind::Stack) => exec::select_i32_srs,
        (ValType::I32, OpdKind::Imm, OpdKind::Reg, OpdKind::Stack) => exec::select_i32_irs,
        (ValType::I32, OpdKind::Stack, OpdKind::Imm, OpdKind::Stack) => exec::select_i32_sis,
        (ValType::I32, OpdKind::Reg, OpdKind::Imm, OpdKind::Stack) => exec::select_i32_ris,
        (ValType::I32, OpdKind::Imm, OpdKind::Imm, OpdKind::Stack) => exec::select_i32_iis,
        (ValType::I32, OpdKind::Stack, OpdKind::Stack, OpdKind::Reg) => exec::select_i32_ssr,
        (ValType::I32, OpdKind::Imm, OpdKind::Stack, OpdKind::Reg) => exec::select_i32_isr,
        (ValType::I32, OpdKind::Stack, OpdKind::Imm, OpdKind::Reg) => exec::select_i32_sir,
        (ValType::I32, OpdKind::Imm, OpdKind::Imm, OpdKind::Reg) => exec::select_i32_iir,

        (ValType::I64, OpdKind::Stack, OpdKind::Stack, OpdKind::Stack) => exec::select_i64_sss,
        (ValType::I64, OpdKind::Reg, OpdKind::Stack, OpdKind::Stack) => exec::select_i64_rss,
        (ValType::I64, OpdKind::Imm, OpdKind::Stack, OpdKind::Stack) => exec::select_i64_iss,
        (ValType::I64, OpdKind::Stack, OpdKind::Reg, OpdKind::Stack) => exec::select_i64_srs,
        (ValType::I64, OpdKind::Imm, OpdKind::Reg, OpdKind::Stack) => exec::select_i64_irs,
        (ValType::I64, OpdKind::Stack, OpdKind::Imm, OpdKind::Stack) => exec::select_i64_sis,
        (ValType::I64, OpdKind::Reg, OpdKind::Imm, OpdKind::Stack) => exec::select_i64_ris,
        (ValType::I64, OpdKind::Imm, OpdKind::Imm, OpdKind::Stack) => exec::select_i64_iis,
        (ValType::I64, OpdKind::Stack, OpdKind::Stack, OpdKind::Reg) => exec::select_i64_ssr,
        (ValType::I64, OpdKind::Imm, OpdKind::Stack, OpdKind::Reg) => exec::select_i64_isr,
        (ValType::I64, OpdKind::Stack, OpdKind::Imm, OpdKind::Reg) => exec::select_i64_sir,
        (ValType::I64, OpdKind::Imm, OpdKind::Imm, OpdKind::Reg) => exec::select_i64_iir,

        (ValType::F32, OpdKind::Stack, OpdKind::Stack, OpdKind::Stack) => exec::select_f32_sss,
        (ValType::F32, OpdKind::Reg, OpdKind::Stack, OpdKind::Stack) => exec::select_f32_rss,
        (ValType::F32, OpdKind::Imm, OpdKind::Stack, OpdKind::Stack) => exec::select_f32_iss,
        (ValType::F32, OpdKind::Stack, OpdKind::Reg, OpdKind::Stack) => exec::select_f32_srs,
        (ValType::F32, OpdKind::Imm, OpdKind::Reg, OpdKind::Stack) => exec::select_f32_irs,
        (ValType::F32, OpdKind::Stack, OpdKind::Imm, OpdKind::Stack) => exec::select_f32_sis,
        (ValType::F32, OpdKind::Reg, OpdKind::Imm, OpdKind::Stack) => exec::select_f32_ris,
        (ValType::F32, OpdKind::Imm, OpdKind::Imm, OpdKind::Stack) => exec::select_f32_iis,
        (ValType::F32, OpdKind::Stack, OpdKind::Stack, OpdKind::Reg) => exec::select_f32_ssr,
        (ValType::F32, OpdKind::Imm, OpdKind::Stack, OpdKind::Reg) => exec::select_f32_isr,
        (ValType::F32, OpdKind::Stack, OpdKind::Imm, OpdKind::Reg) => exec::select_f32_sir,
        (ValType::F32, OpdKind::Imm, OpdKind::Imm, OpdKind::Reg) => exec::select_f32_iir,
        (ValType::F32, OpdKind::Reg, OpdKind::Stack, OpdKind::Reg) => exec::select_f32_rsr,
        (ValType::F32, OpdKind::Stack, OpdKind::Reg, OpdKind::Reg) => exec::select_f32_srr,
        (ValType::F32, OpdKind::Imm, OpdKind::Reg, OpdKind::Reg) => exec::select_f32_irr,
        (ValType::F32, OpdKind::Reg, OpdKind::Imm, OpdKind::Reg) => exec::select_f32_rir,

        (ValType::F64, OpdKind::Stack, OpdKind::Stack, OpdKind::Stack) => exec::select_f64_sss,
        (ValType::F64, OpdKind::Reg, OpdKind::Stack, OpdKind::Stack) => exec::select_f64_rss,
        (ValType::F64, OpdKind::Imm, OpdKind::Stack, OpdKind::Stack) => exec::select_f64_iss,
        (ValType::F64, OpdKind::Stack, OpdKind::Reg, OpdKind::Stack) => exec::select_f64_srs,
        (ValType::F64, OpdKind::Imm, OpdKind::Reg, OpdKind::Stack) => exec::select_f64_irs,
        (ValType::F64, OpdKind::Stack, OpdKind::Imm, OpdKind::Stack) => exec::select_f64_sis,
        (ValType::F64, OpdKind::Reg, OpdKind::Imm, OpdKind::Stack) => exec::select_f64_ris,
        (ValType::F64, OpdKind::Imm, OpdKind::Imm, OpdKind::Stack) => exec::select_f64_iis,
        (ValType::F64, OpdKind::Stack, OpdKind::Stack, OpdKind::Reg) => exec::select_f64_ssr,
        (ValType::F64, OpdKind::Imm, OpdKind::Stack, OpdKind::Reg) => exec::select_f64_isr,
        (ValType::F64, OpdKind::Stack, OpdKind::Imm, OpdKind::Reg) => exec::select_f64_sir,
        (ValType::F64, OpdKind::Imm, OpdKind::Imm, OpdKind::Reg) => exec::select_f64_iir,
        (ValType::F64, OpdKind::Reg, OpdKind::Stack, OpdKind::Reg) => exec::select_f64_rsr,
        (ValType::F64, OpdKind::Stack, OpdKind::Reg, OpdKind::Reg) => exec::select_f64_srr,
        (ValType::F64, OpdKind::Imm, OpdKind::Reg, OpdKind::Reg) => exec::select_f64_irr,
        (ValType::F64, OpdKind::Reg, OpdKind::Imm, OpdKind::Reg) => exec::select_f64_rir,

        (ValType::FuncRef, OpdKind::Stack, OpdKind::Stack, OpdKind::Stack) => {
            exec::select_func_ref_sss
        }
        (ValType::FuncRef, OpdKind::Reg, OpdKind::Stack, OpdKind::Stack) => {
            exec::select_func_ref_rss
        }
        (ValType::FuncRef, OpdKind::Imm, OpdKind::Stack, OpdKind::Stack) => {
            exec::select_func_ref_iss
        }
        (ValType::FuncRef, OpdKind::Stack, OpdKind::Reg, OpdKind::Stack) => {
            exec::select_func_ref_srs
        }
        (ValType::FuncRef, OpdKind::Imm, OpdKind::Reg, OpdKind::Stack) => exec::select_func_ref_irs,
        (ValType::FuncRef, OpdKind::Stack, OpdKind::Imm, OpdKind::Stack) => {
            exec::select_func_ref_sis
        }
        (ValType::FuncRef, OpdKind::Reg, OpdKind::Imm, OpdKind::Stack) => exec::select_func_ref_ris,
        (ValType::FuncRef, OpdKind::Imm, OpdKind::Imm, OpdKind::Stack) => exec::select_func_ref_iis,
        (ValType::FuncRef, OpdKind::Stack, OpdKind::Stack, OpdKind::Reg) => {
            exec::select_func_ref_ssr
        }
        (ValType::FuncRef, OpdKind::Imm, OpdKind::Stack, OpdKind::Reg) => exec::select_func_ref_isr,
        (ValType::FuncRef, OpdKind::Stack, OpdKind::Imm, OpdKind::Reg) => exec::select_func_ref_sir,
        (ValType::FuncRef, OpdKind::Imm, OpdKind::Imm, OpdKind::Reg) => exec::select_func_ref_iir,

        (ValType::ExternRef, OpdKind::Stack, OpdKind::Stack, OpdKind::Stack) => {
            exec::select_extern_ref_sss
        }
        (ValType::ExternRef, OpdKind::Reg, OpdKind::Stack, OpdKind::Stack) => {
            exec::select_extern_ref_rss
        }
        (ValType::ExternRef, OpdKind::Imm, OpdKind::Stack, OpdKind::Stack) => {
            exec::select_extern_ref_iss
        }
        (ValType::ExternRef, OpdKind::Stack, OpdKind::Reg, OpdKind::Stack) => {
            exec::select_extern_ref_srs
        }
        (ValType::ExternRef, OpdKind::Imm, OpdKind::Reg, OpdKind::Stack) => {
            exec::select_extern_ref_irs
        }
        (ValType::ExternRef, OpdKind::Stack, OpdKind::Imm, OpdKind::Stack) => {
            exec::select_extern_ref_sis
        }
        (ValType::ExternRef, OpdKind::Reg, OpdKind::Imm, OpdKind::Stack) => {
            exec::select_extern_ref_ris
        }
        (ValType::ExternRef, OpdKind::Imm, OpdKind::Imm, OpdKind::Stack) => {
            exec::select_extern_ref_iis
        }
        (ValType::ExternRef, OpdKind::Stack, OpdKind::Stack, OpdKind::Reg) => {
            exec::select_extern_ref_ssr
        }
        (ValType::ExternRef, OpdKind::Imm, OpdKind::Stack, OpdKind::Reg) => {
            exec::select_extern_ref_isr
        }
        (ValType::ExternRef, OpdKind::Stack, OpdKind::Imm, OpdKind::Reg) => {
            exec::select_extern_ref_sir
        }
        (ValType::ExternRef, OpdKind::Imm, OpdKind::Imm, OpdKind::Reg) => {
            exec::select_extern_ref_iir
        }

        // The first operand is an integer or a reference, and the third operand is an integer,
        // both of which are stored in a register. Since we only have one integer register
        // available, there is no variant of this instruction that can handle this case.
        (
            ValType::I32 | ValType::I64 | ValType::FuncRef | ValType::ExternRef,
            OpdKind::Reg,
            _,
            OpdKind::Reg,
        )
        | (
            ValType::I32 | ValType::I64 | ValType::FuncRef | ValType::ExternRef,
            _,
            OpdKind::Reg,
            OpdKind::Reg,
        )
        // The first and the second operand have the same type, which means they are stored in the
        // same register. Since we only have one register available for every type, there is no
        // variant of this instruction that can handle this case.
        | (_, OpdKind::Reg, OpdKind::Reg, _)
        | (_, _, _, OpdKind::Imm) => panic!("no suitable instruction found"),
    }
}

fn select_global_get(type_: ValType) -> ThreadedInstr {
    match type_ {
        ValType::I32 => exec::global_get_i32,
        ValType::I64 => exec::global_get_i64,
        ValType::F32 => exec::global_get_f32,
        ValType::F64 => exec::global_get_f64,
        ValType::FuncRef => exec::global_get_func_ref,
        ValType::ExternRef => exec::global_get_extern_ref,
    }
}

fn select_global_set(type_: ValType, kind: OpdKind) -> ThreadedInstr {
    match (type_, kind) {
        (ValType::I32, OpdKind::Stack) => exec::global_set_i32_s,
        (ValType::I32, OpdKind::Reg) => exec::global_set_i32_r,
        (ValType::I32, OpdKind::Imm) => exec::global_set_i32_i,
        (ValType::I64, OpdKind::Stack) => exec::global_set_i64_s,
        (ValType::I64, OpdKind::Reg) => exec::global_set_i64_r,
        (ValType::I64, OpdKind::Imm) => exec::global_set_i64_i,
        (ValType::F32, OpdKind::Stack) => exec::global_set_f32_s,
        (ValType::F32, OpdKind::Reg) => exec::global_set_f32_r,
        (ValType::F32, OpdKind::Imm) => exec::global_set_f32_i,
        (ValType::F64, OpdKind::Stack) => exec::global_set_f64_s,
        (ValType::F64, OpdKind::Reg) => exec::global_set_f64_r,
        (ValType::F64, OpdKind::Imm) => exec::global_set_f64_i,
        (ValType::FuncRef, OpdKind::Stack) => exec::global_set_func_ref_s,
        (ValType::FuncRef, OpdKind::Reg) => exec::global_set_func_ref_r,
        (ValType::FuncRef, OpdKind::Imm) => exec::global_set_func_ref_i,
        (ValType::ExternRef, OpdKind::Stack) => exec::global_set_extern_ref_s,
        (ValType::ExternRef, OpdKind::Reg) => exec::global_set_extern_ref_r,
        (ValType::ExternRef, OpdKind::Imm) => exec::global_set_extern_ref_i,
    }
}

fn select_table_get(type_: RefType, kind: OpdKind) -> ThreadedInstr {
    match (type_, kind) {
        (RefType::FuncRef, OpdKind::Stack) => exec::table_get_func_ref_s,
        (RefType::FuncRef, OpdKind::Reg) => exec::table_get_func_ref_r,
        (RefType::FuncRef, OpdKind::Imm) => exec::table_get_func_ref_i,

        (RefType::ExternRef, OpdKind::Stack) => exec::table_get_extern_ref_s,
        (RefType::ExternRef, OpdKind::Reg) => exec::table_get_extern_ref_r,
        (RefType::ExternRef, OpdKind::Imm) => exec::table_get_extern_ref_i,
    }
}

fn select_table_set(type_: RefType, kind_0: OpdKind, kind_1: OpdKind) -> ThreadedInstr {
    match (type_, kind_0, kind_1) {
        (RefType::FuncRef, OpdKind::Stack, OpdKind::Stack) => exec::table_set_func_ref_ss,
        (RefType::FuncRef, OpdKind::Reg, OpdKind::Stack) => exec::table_set_func_ref_rs,
        (RefType::FuncRef, OpdKind::Imm, OpdKind::Stack) => exec::table_set_func_ref_is,
        (RefType::FuncRef, OpdKind::Imm, OpdKind::Reg) => exec::table_set_func_ref_ir,
        (RefType::FuncRef, OpdKind::Imm, OpdKind::Imm) => exec::table_set_func_ref_ii,
        (RefType::FuncRef, OpdKind::Stack, OpdKind::Reg) => exec::table_set_func_ref_sr,
        (RefType::FuncRef, OpdKind::Stack, OpdKind::Imm) => exec::table_set_func_ref_si,
        (RefType::FuncRef, OpdKind::Reg, OpdKind::Imm) => exec::table_set_func_ref_ri,

        (RefType::ExternRef, OpdKind::Stack, OpdKind::Stack) => exec::table_set_extern_ref_ss,
        (RefType::ExternRef, OpdKind::Reg, OpdKind::Stack) => exec::table_set_extern_ref_rs,
        (RefType::ExternRef, OpdKind::Imm, OpdKind::Stack) => exec::table_set_extern_ref_is,
        (RefType::ExternRef, OpdKind::Imm, OpdKind::Reg) => exec::table_set_extern_ref_ir,
        (RefType::ExternRef, OpdKind::Imm, OpdKind::Imm) => exec::table_set_extern_ref_ii,
        (RefType::ExternRef, OpdKind::Stack, OpdKind::Reg) => exec::table_set_extern_ref_sr,
        (RefType::ExternRef, OpdKind::Stack, OpdKind::Imm) => exec::table_set_extern_ref_si,
        (RefType::ExternRef, OpdKind::Reg, OpdKind::Imm) => exec::table_set_extern_ref_ri,

        // The first operand is an integer, and the second operand is a reference, both of which
        // are stored in an integer register. Since we only have one integer register available,
        // there is no variant of table_set that can handle this case.
        (RefType::FuncRef | RefType::ExternRef, OpdKind::Reg, OpdKind::Reg) => {
            panic!("no suitable instruction found")
        }
    }
}

fn select_table_size(type_: RefType) -> ThreadedInstr {
    match type_ {
        RefType::FuncRef => exec::table_size_func_ref,
        RefType::ExternRef => exec::table_size_extern_ref,
    }
}

fn select_table_grow(type_: RefType) -> ThreadedInstr {
    match type_ {
        RefType::FuncRef => exec::table_grow_func_ref,
        RefType::ExternRef => exec::table_grow_extern_ref,
    }
}

fn select_table_fill(type_: RefType) -> ThreadedInstr {
    match type_ {
        RefType::FuncRef => exec::table_fill_func_ref,
        RefType::ExternRef => exec::table_fill_extern_ref,
    }
}

fn select_table_copy(type_: RefType) -> ThreadedInstr {
    match type_ {
        RefType::FuncRef => exec::table_copy_func_ref,
        RefType::ExternRef => exec::table_copy_extern_ref,
    }
}

fn select_table_init(type_: RefType) -> ThreadedInstr {
    match type_ {
        RefType::FuncRef => exec::table_init_func_ref,
        RefType::ExternRef => exec::table_init_extern_ref,
    }
}

fn select_elem_drop(type_: RefType) -> ThreadedInstr {
    match type_ {
        RefType::FuncRef => exec::elem_drop_func_ref,
        RefType::ExternRef => exec::elem_drop_extern_ref,
    }
}

fn select_un_op(info: UnOpInfo, kind: OpdKind) -> ThreadedInstr {
    match kind {
        OpdKind::Stack => Some(info.instr_s),
        OpdKind::Reg => Some(info.instr_r),
        OpdKind::Imm => info.instr_i,
    }
    .expect("no suitable instruction found")
}

fn select_bin_op(info: BinOpInfo, kind_0: OpdKind, kind_1: OpdKind) -> ThreadedInstr {
    match (kind_0, kind_1) {
        (OpdKind::Stack, OpdKind::Stack) => Some(info.instr_ss),
        (OpdKind::Reg, OpdKind::Stack) => Some(info.instr_rs),
        (OpdKind::Imm, OpdKind::Stack) => Some(info.instr_is),
        (OpdKind::Stack, OpdKind::Reg) => Some(info.instr_sr),
        (OpdKind::Reg, OpdKind::Reg) => info.instr_rr,
        (OpdKind::Imm, OpdKind::Reg) => Some(info.instr_ir),
        (OpdKind::Stack, OpdKind::Imm) => Some(info.instr_si),
        (OpdKind::Reg, OpdKind::Imm) => Some(info.instr_ri),
        (OpdKind::Imm, OpdKind::Imm) => info.instr_ii,
    }
    .expect("no suitable instruction found")
}

fn selecy_copy_opd_to_stack(type_: ValType, kind: OpdKind) -> ThreadedInstr {
    match kind {
        OpdKind::Stack => select_copy_stack(type_),
        OpdKind::Reg => select_copy_reg_to_stack(type_),
        OpdKind::Imm => select_copy_imm_to_stack(type_),
    }
}

fn select_copy_imm_to_stack(type_: ValType) -> ThreadedInstr {
    match type_ {
        ValType::I32 => exec::copy_imm_to_stack_i32,
        ValType::I64 => exec::copy_imm_to_stack_i64,
        ValType::F32 => exec::copy_imm_to_stack_f32,
        ValType::F64 => exec::copy_imm_to_stack_f64,
        ValType::FuncRef => exec::copy_imm_to_stack_func_ref,
        ValType::ExternRef => exec::copy_imm_to_stack_extern_ref,
    }
}

fn select_copy_stack(type_: ValType) -> ThreadedInstr {
    match type_.into() {
        ValType::I32 => exec::copy_stack_i32,
        ValType::I64 => exec::copy_stack_i64,
        ValType::F32 => exec::copy_stack_f32,
        ValType::F64 => exec::copy_stack_f64,
        ValType::FuncRef => exec::copy_stack_func_ref,
        ValType::ExternRef => exec::copy_stack_extern_ref,
    }
}

fn select_copy_reg_to_stack(type_: ValType) -> ThreadedInstr {
    match type_ {
        ValType::I32 => exec::copy_reg_to_stack_i32,
        ValType::I64 => exec::copy_reg_to_stack_i64,
        ValType::F32 => exec::copy_reg_to_stack_f32,
        ValType::F64 => exec::copy_reg_to_stack_f64,
        ValType::FuncRef => exec::copy_reg_to_stack_func_ref,
        ValType::ExternRef => exec::copy_reg_to_stack_extern_ref,
    }
}
