mod variable;
mod memory;
mod numeric;
mod table;

use {
    crate::{
        cast::{ExtendingCast, ExtendingCastFrom, WrappingCast, WrappingCastFrom},
        code,
        code::CodeBuilder,
        instr::{
            BlockType, Instr, InstrStream, InstrStreamAllocs, InstrVisitor, MemArg,
        },
        decode::DecodeError,
        downcast::{DowncastRef, DowncastMut},
        executor,
        executor::{ReadImm, ReadFromReg, ReadFromPtr, ReadReg, ReadStack, ThreadedInstr, WriteReg, WriteStack, WriteToReg, WriteToPtr},
        func::{CompiledFuncBody, Func, FuncEntity, FuncType, InstrSlot, UncompiledFuncBody},
        runtime::{
            global::{GlobalEntity, TypedGlobalEntity},
            table::{TableEntity, TypedTableEntity},
        },
        guarded::Guarded,
        instance::Instance,
        ops::*,
        ref_::{ExternRef, FuncRef, RefType, UnguardedExternRef, UnguardedFuncRef},
        store::Store,
        val::{UnguardedVal, ValType, ValTypeOf},
    },
    std::{mem, ops::{Deref, Index, IndexMut}, ptr},
};

#[derive(Clone, Debug)]
pub(crate) struct CompilerAllocs {
    instr_stream_allocs: InstrStreamAllocs,
    br_table_label_idxs: Vec<u32>,
    typed_select_val_types: Vec<ValType>,
    locals: Vec<Local>,
    blocks: Vec<Block>,
    opds: Vec<Opd>,
    fixup_offsets: Vec<usize>,
}

impl CompilerAllocs {
    pub(crate) fn new() -> Self {
        Self {
            instr_stream_allocs: InstrStreamAllocs::default(),
            br_table_label_idxs: Vec::new(),
            typed_select_val_types: Vec::new(),
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
        self.locals.clear();
        self.blocks.clear();
        self.opds.clear();
        self.fixup_offsets.clear();

        let type_ = func.type_(store);
        let mut compile = Compiler {
            store,
            type_: func.type_(store).clone(),
            instance,
            instr_stream: InstrStream::new_with_allocs(&code.expr, mem::take(&mut self.instr_stream_allocs)),
            br_table_label_idxs: &mut self.br_table_label_idxs,
            typed_select_val_types: &mut self.typed_select_val_types,
            locals: &mut self.locals,
            blocks: &mut self.blocks,
            opds: &mut self.opds,
            fixup_offsets: &mut self.fixup_offsets,
            stack: Stack::new(),
            regs: Regs::new(),
            code: CodeBuilder::new(),
        };
        let mut stack_offset = -(executor::call_frame_size(&compile.type_) as isize);
        for type_ in compile.type_.params().iter().copied() {
            compile.locals.push(Local {
                type_,
                stack_offset,
                first_opd_idx: None,
            });
            stack_offset += type_.padded_size_of() as isize;
        }
        for type_ in code.locals.iter().copied() {
            let stack_offset = compile.stack.alloc(type_.padded_size_of());
            compile.locals.push(Local {
                type_,
                stack_offset: stack_offset as isize,
                first_opd_idx: None,
            });
        }
        compile.push_block(
            BlockKind::Block,
            FuncType::new([], type_.results().iter().copied()),
        );

        compile.emit_instr(executor::enter as ThreadedInstr);
        compile.emit(func.to_unguarded(store.guard()));
        compile.emit(
            compile
                .instance
                .mem(0)
                .map(|mem| mem.to_unguarded(store.guard())),
        );

        while !compile.instr_stream.is_empty() {
            let instr = compile.instr_stream.next().unwrap();
            instr.visit(&mut compile).unwrap();
        }
        let mut result_stack_offset = -(executor::call_frame_size(&compile.type_) as isize);
        for (result_idx, result_type) in compile.type_.clone().results().iter().copied().enumerate() {
            let opd_depth = compile.type_.results().len() - 1 - result_idx;
            compile.emit_instr(select_copy(result_type, OpdKind::Stk));
            compile.emit_opd(opd_depth);
            compile.emit_stack_offset(result_stack_offset);
            result_stack_offset += result_type.padded_size_of() as isize;
        }
        compile.emit_instr(executor::return_ as ThreadedInstr);

        let mut code = compile.code.finish();
        for fixup_offset in compile.fixup_offsets.drain(..) {
            unsafe {
                let fixup_ptr = code.as_mut_ptr().add(fixup_offset);
                let instr_offset = ptr::read(fixup_ptr.cast());
                let instr_ptr = code.as_mut_ptr().add(instr_offset);
                ptr::write(fixup_ptr.cast(), instr_ptr);
            }
        }

        self.instr_stream_allocs = compile.instr_stream.into_allocs();

        CompiledFuncBody {
            max_stack_height: compile.stack.max_height(),
            local_count: compile.locals.len() - compile.type_.params().len(),
            code,
        }
    }
}

impl Default for CompilerAllocs {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
struct Compiler<'a> {
    store: &'a Store,
    type_: FuncType,
    instance: &'a Instance,
    instr_stream: InstrStream<'a>,
    br_table_label_idxs: &'a mut Vec<u32>,
    typed_select_val_types: &'a mut Vec<ValType>,
    locals: &'a mut Vec<Local>,
    blocks: &'a mut Vec<Block>,
    opds: &'a mut Vec<Opd>,
    fixup_offsets: &'a mut Vec<usize>,
    stack: Stack,
    regs: Regs,
    code: CodeBuilder
}

impl<'a> Compiler<'a> {
    fn compile_select(&mut self, type_: Option<ValType>) -> Result<(), DecodeError> {
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
        let output_reg_idx = type_.reg_name();
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

    fn compile_rel_op<T, B>(&mut self) -> Result<(), DecodeError>
    where
        T: ReadFromReg,
        B: BinOp<T, Output = i32>,
    {
        if self.block(0).is_unreachable {
            return Ok(());
        }
        if let Instr::BrIf { label_idx } = self.instr_stream.peek(0)? {
            self.instr_stream.skip(1)?;
            self.compile_br_if_rel_op::<T, B>(label_idx)
        } else {
            self.compile_bin_op::<T, B>()
        }
    }

    fn compile_br_if_rel_op<T, B>(&mut self, label_idx: u32) -> Result<(), DecodeError>
    where
        T: ReadFromReg,
        B: BinOp<T, Output = i32>,
    {
        self.compile_br_if_rel_op_inner(
            label_idx,
            select_br_if_rel_op::<T, B>(self.opd(1).kind(), self.opd(0).kind()),
            select_br_if_not_rel_op::<T, B>(self.opd(1).kind(), self.opd(0).kind()),
        )
    }

    fn compile_br_if_rel_op_inner(
        &mut self,
        label_idx: u32,
        br_if_rel_op: ThreadedInstr,
        br_if_not_rel_op: ThreadedInstr,
    ) -> Result<(), DecodeError> {
         // Wasm uses `u32` indices for labels, but we use `usize` indices.
        let label_idx = label_idx as usize;

        // This is a branch. We need to ensure that each block input is stored in the location
        // expected by the target before the branch is taken.
        //
        // We do this by ensuring that each block input is a temporary operand, which is stored
        // in a known location on the stack, and then copying them to their expected locations in
        // the code emitted below.
        for opd_depth in 2..self.block(label_idx).label_types().len() + 2 {
            self.ensure_opd_not_imm(opd_depth);
            self.ensure_opd_not_local(opd_depth);
            self.ensure_opd_not_reg(opd_depth);
        }

        if self.block(label_idx).label_types().is_empty() {
            // If the branch target has an empty type, we don't need to copy any block inputs to
            // their expected locations, so we can generate more efficient code.
            self.emit_instr(br_if_rel_op);
            self.emit_and_pop_opd();
            self.emit_and_pop_opd();
            self.emit_label(label_idx);
        } else {
            // If the branch target has a non-empty type, we need to copy all block inputs to their
            // expected locations.
            //
            // This is more expensive, because we cannot branch to the target directly. Instead, we
            // have to branch to code that first copies the block inputs to their expected
            // locations, and then branches to the target.
            self.emit_instr(br_if_not_rel_op);
            self.emit_and_pop_opd();
            self.emit_and_pop_opd();
            let hole_offset = self.emit_hole();
            self.resolve_label_vals(label_idx);
            self.emit_instr(executor::br as ThreadedInstr);
            self.emit_label(label_idx);
            self.patch_hole(hole_offset);
        }

        for label_type in self.block(label_idx).label_types().iter().copied() {
            self.push_opd(label_type);
        }

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
            stack_offset: self.stack.height(),
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
            self.preserve_reg(self.opd(opd_depth).type_.reg_name());
        }
    }

    /// Preserves an immediate operand by copying its value to the stack, if necessary.
    fn preserve_imm_opd(&mut self, opd_depth: usize) {
        let opd_idx = self.opds.len() - 1 - opd_depth;
        self.emit_instr(select_copy(self.opds[opd_idx].type_, OpdKind::Imm));
        self.emit_val(self.opds[opd_idx].val.unwrap());
        self.emit_stack_offset(self.opds[opd_idx].stack_offset as isize);
        self.opd_mut(opd_depth).val = None;
    }

    /// Preserve a local operand by copying the local it refers to to the stack, if necessary.
    fn preserve_local_opd(&mut self, opd_idx: usize) {
        let local_idx = self.opds[opd_idx].local_idx.unwrap();
        self.emit_instr(select_copy(self.locals[local_idx].type_, OpdKind::Stk));
        self.emit_stack_offset(self.locals[local_idx].stack_offset);
        self.emit_stack_offset(self.opds[opd_idx].stack_offset as isize);
        self.remove_local_opd(opd_idx);
    }

    /// Pushes an operand of the given type on the stack.
    fn push_opd(&mut self, type_: impl Into<ValType>) {
        let type_ = type_.into();
        let stack_offset = self.stack.alloc(type_.padded_size_of());
        self.opds.push(Opd {
            type_,
            stack_offset,
            val: None,
            local_idx: None,
            prev_opd_idx: None,
            next_opd_idx: None,
            is_reg: false,
        });
    }

    /// Pops an operand from the stack.
    fn pop_opd(&mut self) {
        if self.opd(0).is_reg {
            self.dealloc_reg(self.opd(0).type_.reg_name());
        }
        let opd_idx = self.opds.len() - 1;
        if let Some(local_idx) = self.opds[opd_idx].local_idx {
            self.locals[local_idx].first_opd_idx = self.opds[opd_idx].next_opd_idx;
        }
        let opd = self.opds.pop().unwrap();
        self.stack.dealloc(opd.stack_offset);
    }

    /// Emits an operand and then pops it from the stack.
    fn emit_and_pop_opd(&mut self) {
        self.emit_opd(0);
        self.pop_opd();
    }

    /// Returns the stack index of the operand at the given depth.
    fn opd_stack_offset(&self, opd_depth: usize) -> isize {
        let opd_idx = self.opds.len() - 1 - opd_depth;
        if let Some(local_idx) = self.opds[opd_idx].local_idx {
            self.locals[local_idx].stack_offset
        } else {
            self.opds[opd_idx].stack_offset as isize
        }
    }

    // Methods for operating on registers.

    /// Returns `true` if the register with the given index is occupied.
    fn is_reg_occupied(&self, reg_name: RegName) -> bool {
        self.regs[reg_name].is_used()
    }

    /// Allocates a register to the top operand.
    fn alloc_reg(&mut self) {
        debug_assert!(!self.opd(0).is_reg);
        let reg_name = self.opd(0).type_.reg_name();
        let reg = &mut self.regs[reg_name];
        debug_assert!(!reg.is_used());
        let opd_idx = self.opds.len() - 1;
        self.opds[opd_idx].is_reg = true;
        reg.alloc(opd_idx);
    }

    /// Deallocates the register with the given name.
    fn dealloc_reg(&mut self, reg_name: RegName) {
        let reg = &mut self.regs[reg_name];
        let opd_idx = reg.opd_idx().unwrap();
        self.opds[opd_idx].is_reg = false;
        reg.dealloc();
    }

    /// Preserves the register with the given index by preserving the register operand that occupies
    /// it.
    fn preserve_reg(&mut self, reg_name: RegName) {
        let opd_idx = self.regs[reg_name].opd_idx().unwrap();
        let opd_type = self.opds[opd_idx].type_;
        self.emit_instr(select_copy(opd_type, OpdKind::Reg));
        self.emit_stack_offset(self.opds[opd_idx].stack_offset as isize);
        self.dealloc_reg(reg_name);
    }

    /// Preserves all registers by preserving the register operands that occupy them.
    fn preserve_all_regs(&mut self) {
        for reg_name in RegName::iter() {
            if self.is_reg_occupied(reg_name) {
                self.preserve_reg(reg_name);
            }
        }
    }

    /// Copies the values for the label with the given index to their expected locations
    /// on the stack, and pop them from the stack.
    fn resolve_label_vals(&mut self, label_idx: usize) {
        let mut label_val_stack_offset = self.block(label_idx).stack_offset;
        for (label_val_idx, label_type) in self
            .block(label_idx)
            .label_types()
            .iter()
            .copied()
            .enumerate()
        {
            let opd_depth = self.block(label_idx).label_types().len() - 1 - label_val_idx;
            self.emit_instr(select_copy(label_type, OpdKind::Stk));
            self.emit_stack_offset(self.opd_stack_offset(opd_depth));
            self.emit_stack_offset(label_val_stack_offset as isize);
            label_val_stack_offset += label_type.padded_size_of();
        }
        for _ in 0..self.block(label_idx).label_types().len() {
            self.pop_opd();
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
            OpdKind::Stk => self.emit_stack_offset(self.opd_stack_offset(opd_depth)),
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
    fn emit_stack_offset(&mut self, stack_offset: isize) {
        self.emit(stack_offset as i32);
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

impl<'a> InstrVisitor for Compiler<'a> {
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
        self.emit_instr(executor::unreachable as ThreadedInstr);

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
            self.emit_instr(select_br_if_not(self.opd(0).kind()));

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
            self.emit_instr(executor::br as ThreadedInstr);
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
        self.emit_instr(executor::br as ThreadedInstr);
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
            self.emit_instr(select_br_if(self.opd(0).kind()));
            self.emit_and_pop_opd();
            self.emit_label(label_idx);
        } else {
            // If the branch target has a non-empty type, we need to copy all block inputs to their
            // expected locations.
            //
            // This is more expensive, because we cannot branch to the target directly. Instead, we
            // have to branch to code that first copies the block inputs to their expected
            // locations, and then branches to the target.
            self.emit_instr(select_br_if_not(self.opd(0).kind()));
            self.emit_and_pop_opd();
            let hole_offset = self.emit_hole();
            self.resolve_label_vals(label_idx);
            self.emit_instr(executor::br as ThreadedInstr);
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
                self.emit_instr(executor::br as ThreadedInstr);
                self.emit_label(label_idx);
                for label_type in self.block(label_idx).label_types().iter().copied() {
                    self.push_opd(label_type);
                }
            }
            self.patch_hole(default_hole_offset);
            self.resolve_label_vals(default_label_idx);
            self.emit_instr(executor::br as ThreadedInstr);
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
        let mut result_stack_offset = -(executor::call_frame_size(&self.type_) as isize);
        for (result_idx, result_type) in self
            .type_
            .clone()
            .results()
            .iter()
            .copied()
            .enumerate()
        {
            let opd_depth =  self.type_.results().len() - 1 - result_idx;
            self.ensure_opd_not_imm(opd_depth);
            self.emit_instr(if self.opd(opd_depth).is_reg {
                select_copy(result_type, OpdKind::Reg)
            } else {
                select_copy(result_type, OpdKind::Stk)
            });
            self.emit_opd(opd_depth);
            self.emit_stack_offset(result_stack_offset);
            result_stack_offset += result_type.padded_size_of() as isize;
        }
        for _ in 0..self.type_.results().len() {
            self.pop_opd();
        }

        // Emit the instruction.
        self.emit_instr(executor::return_ as ThreadedInstr);

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
            FuncEntity::Wasm(_) => executor::compile as ThreadedInstr,
            FuncEntity::Host(_) => executor::call_host as ThreadedInstr,
        });

        // Pop the arguments from the stack.
        for _ in 0..type_.params().len() {
            self.pop_opd();
        }

        // Emit an unguarded handle to the [`Func`] to be called.
        self.emit(func.0.to_unguarded(self.store.guard()));

        // Compute the start and end of the call frame, and update the maximum stack height
        // attained by the [`Func`] being compiled.
        let call_frame_stack_start = self.stack.alloc(executor::call_frame_size(&type_));
        let call_frame_stack_end = self.stack.height();
        self.stack.dealloc(call_frame_stack_start);

        // Emit the stack offset of the end of the call frame.
        self.emit_stack_offset(call_frame_stack_end as isize);

        if let FuncEntity::Host(_) = func.0.as_ref(&self.store) {
            self.emit(
                self.instance
                    .mem(0)
                    .map(|mem| mem.0.to_unguarded(self.store.guard())),
            );
        }

        // Push the outputs onto the stack.
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
        self.emit_instr(executor::call_indirect as ThreadedInstr);

        // Emit the function index and pop it from the stack.
        self.emit_and_pop_opd();

        // Pop the arguments from the stack.
        for _ in 0..type_.params().len() {
            self.pop_opd();
        }

        // Emit an unguarded handle to the [`Table`].
        self.emit(table.0.to_unguarded(self.store.guard()));

        // Emit the interned type.
        self.emit(interned_type.to_unguarded(self.store.guard()));

        // Compute the start and end of the call frame, and update the maximum stack height
        // attained by the [`Func`] being compiled.
        let call_frame_stack_start = self.stack.alloc(executor::call_frame_size(&type_));
        let call_frame_stack_end = self.stack.height();
        self.stack.dealloc(call_frame_stack_start);
        
        // Emit the stack offset of the end of the call frame.
        self.emit_stack_offset(call_frame_stack_end as isize);

        // Emit an unguarded handle to the active [`Memory`] for the [`Func`] being compiled.
        self.emit(
            self.instance
                .mem(0)
                .map(|mem| mem.0.to_unguarded(self.store.guard())),
        );

        // Push the outputs onto the stack.
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
            RefType::FuncRef => self.emit(FuncRef::None.to_unguarded(self.store.guard())),
            RefType::ExternRef => self.emit(ExternRef::None.to_unguarded(self.store.guard())),
        };

        // Push the output onto the stack and emit its stack offset.
        self.push_opd(ValType::FuncRef);
        self.emit_stack_offset(self.opd_stack_offset(0));

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
        self.emit_instr(executor::copy::<UnguardedFuncRef, ReadImm, WriteStack> as ThreadedInstr);
        
        // Emit an unguarded handle to the [`Func`].
        self.emit(func.to_unguarded(self.store.guard()));

        // Push the output onto the stack and emit its stack offset.
        self.push_opd(ValType::FuncRef);
        self.emit_stack_offset(self.opd_stack_offset(0));

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
    fn visit_select(&mut self) -> Result<(), DecodeError> {
        self.compile_select(None)
    }

    fn visit_typed_select_start(&mut self) -> Result<(), DecodeError> {
        self.typed_select_val_types.clear();
        Ok(())
    }

    fn visit_typed_select_val_type(&mut self, type_: ValType) -> Result<(), DecodeError> {
        self.typed_select_val_types.push(type_);
        Ok(())
    }

    fn visit_typed_select_end(&mut self) -> Result<(), DecodeError> {
        let val_type = self.typed_select_val_types.pop().unwrap();
        self.compile_select(Some(val_type))
    }

    fn visit_local_get(&mut self, local_idx: u32) -> Result<(), DecodeError> {
        self.compile_local_get(local_idx)
    }

    fn visit_local_set(&mut self, local_idx: u32) -> Result<(), DecodeError> {
       self.compile_local_set(local_idx)
    }

    fn visit_local_tee(&mut self, local_idx: u32) -> Result<(), DecodeError> {
        self.compile_local_tee(local_idx)
    }

    fn visit_global_get(&mut self, global_idx: u32) -> Result<(), DecodeError> {
        self.compile_global_get(global_idx)
    }

    fn visit_global_set(&mut self, global_idx: u32) -> Result<(), DecodeError> {
        self.compile_global_set(global_idx)
    }

    fn visit_table_get(&mut self, table_idx: u32) -> Result<(), Self::Error> {
        self.compile_table_get(table_idx)
    }

    fn visit_table_set(&mut self, table_idx: u32) -> Result<(), Self::Error> {
        self.compile_table_set(table_idx)
    }

    fn visit_table_size(&mut self, table_idx: u32) -> Result<(), Self::Error> {
        self.compile_table_size(table_idx)
    }

    fn visit_table_grow(&mut self, table_idx: u32) -> Result<(), Self::Error> {
        self.compile_table_grow(table_idx)
    }

    fn visit_table_fill(&mut self, table_idx: u32) -> Result<(), Self::Error> {
        self.compile_table_fill(table_idx)
    }

    fn visit_table_copy(
        &mut self,
        dst_table_idx: u32,
        src_table_idx: u32,
    ) -> Result<(), Self::Error> {
        self.compile_table_copy(dst_table_idx, src_table_idx)
    }

    fn visit_table_init(
        &mut self,
        dst_table_idx: u32,
        src_elem_idx: u32,
    ) -> Result<(), Self::Error> {
        self.compile_table_init(dst_table_idx, src_elem_idx)
    }

    fn visit_elem_drop(&mut self, elem_idx: u32) -> Result<(), Self::Error> {
        self.compile_elem_drop(elem_idx)
    }

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

    fn visit_memory_size(&mut self) -> Result<(), Self::Error> {
        self.compile_memory_size()
    }

    fn visit_memory_grow(&mut self) -> Result<(), Self::Error> {
        self.compile_memory_grow()
    }

    fn visit_memory_fill(&mut self) -> Result<(), Self::Error> {
        self.compile_memory_fill()
    }

    fn visit_memory_copy(&mut self) -> Result<(), Self::Error> {
        self.compile_memory_copy()
    }

    fn visit_memory_init(&mut self, data_idx: u32) -> Result<(), Self::Error> {
        self.compile_memory_init(data_idx)
    }

    fn visit_data_drop(&mut self, data_idx: u32) -> Result<(), Self::Error> {
        self.compile_data_drop(data_idx)
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
        self.compile_rel_op::<i32, Eq>()
    }

    fn visit_i32_ne(&mut self) -> Result<(), DecodeError> {
        self.compile_rel_op::<i32, Ne>()
    }

    fn visit_i32_lt_s(&mut self) -> Result<(), DecodeError> {
        self.compile_rel_op::<i32, Lt>()
    }

    fn visit_i32_lt_u(&mut self) -> Result<(), DecodeError> {
        self.compile_rel_op::<u32, Lt>()
    }

    fn visit_i32_gt_s(&mut self) -> Result<(), DecodeError> {
        self.compile_rel_op::<i32, Gt>()
    }

    fn visit_i32_gt_u(&mut self) -> Result<(), DecodeError> {
        self.compile_rel_op::<u32, Gt>()
    }

    fn visit_i32_le_s(&mut self) -> Result<(), DecodeError> {
        self.compile_rel_op::<i32, Le>()
    }

    fn visit_i32_le_u(&mut self) -> Result<(), DecodeError> {
        self.compile_rel_op::<u32, Le>()
    }

    fn visit_i32_ge_s(&mut self) -> Result<(), DecodeError> {
        self.compile_rel_op::<i32, Ge>()
    }

    fn visit_i32_ge_u(&mut self) -> Result<(), DecodeError> {
        self.compile_rel_op::<u32, Ge>()
    }

    fn visit_i64_eqz(&mut self) -> Result<(), DecodeError> {
        self.compile_un_op::<i64, Eqz>()
    }

    fn visit_i64_eq(&mut self) -> Result<(), DecodeError> {
        self.compile_bin_op::<i64, Eq>()
    }

    fn visit_i64_ne(&mut self) -> Result<(), DecodeError> {
        self.compile_rel_op::<i64, Ne>()
    }

    fn visit_i64_lt_s(&mut self) -> Result<(), DecodeError> {
        self.compile_rel_op::<i64, Lt>()
    }

    fn visit_i64_lt_u(&mut self) -> Result<(), DecodeError> {
        self.compile_rel_op::<u64, Lt>()
    }

    fn visit_i64_gt_s(&mut self) -> Result<(), DecodeError> {
        self.compile_rel_op::<i64, Gt>()
    }

    fn visit_i64_gt_u(&mut self) -> Result<(), DecodeError> {
        self.compile_rel_op::<u64, Gt>()
    }

    fn visit_i64_le_s(&mut self) -> Result<(), DecodeError> {
        self.compile_rel_op::<i64, Le>()
    }

    fn visit_i64_le_u(&mut self) -> Result<(), DecodeError> {
        self.compile_rel_op::<u64, Le>()
    }

    fn visit_i64_ge_s(&mut self) -> Result<(), DecodeError> {
        self.compile_rel_op::<i64, Ge>()
    }

    fn visit_i64_ge_u(&mut self) -> Result<(), DecodeError> {
        self.compile_rel_op::<u64, Ge>()
    }

    fn visit_f32_eq(&mut self) -> Result<(), DecodeError> {
        self.compile_rel_op::<f32, Eq>()
    }

    fn visit_f32_ne(&mut self) -> Result<(), DecodeError> {
        self.compile_rel_op::<f32, Ne>()
    }

    fn visit_f32_lt(&mut self) -> Result<(), DecodeError> {
        self.compile_rel_op::<f32, Lt>()
    }

    fn visit_f32_gt(&mut self) -> Result<(), DecodeError> {
        self.compile_rel_op::<f32, Gt>()
    }

    fn visit_f32_le(&mut self) -> Result<(), DecodeError> {
        self.compile_rel_op::<f32, Le>()
    }

    fn visit_f32_ge(&mut self) -> Result<(), DecodeError> {
        self.compile_rel_op::<f32, Ge>()
    }

    fn visit_f64_eq(&mut self) -> Result<(), DecodeError> {
        self.compile_rel_op::<f64, Eq>()
    }

    fn visit_f64_ne(&mut self) -> Result<(), DecodeError> {
        self.compile_rel_op::<f64, Ne>()
    }

    fn visit_f64_lt(&mut self) -> Result<(), DecodeError> {
        self.compile_rel_op::<f64, Lt>()
    }

    fn visit_f64_gt(&mut self) -> Result<(), DecodeError> {
        self.compile_rel_op::<f64, Gt>()
    }

    fn visit_f64_le(&mut self) -> Result<(), DecodeError> {
        self.compile_rel_op::<f64, Le>()
    }

    fn visit_f64_ge(&mut self) -> Result<(), DecodeError> {
        self.compile_rel_op::<f64, Ge>()
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
    // The offset of this local on the stack.
    stack_offset: isize,
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
    // The offset of the stack at the start of this block.
    stack_offset: usize,
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
    stack_offset: usize,
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

    /// Returns `true` if this operand occupies the register with the given name.
    fn occupies_reg(&self, reg_name: RegName) -> bool {
        self.is_reg && self.type_.reg_name() == reg_name
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

#[derive(Debug)]
struct Stack {
    height: usize,
    max_height: usize,
}

impl Stack {
    fn new() -> Self {
        Self {
            height: 0,
            max_height: 0,
        }
    }

    fn height(&self) -> usize {
        self.height
    }

    fn max_height(&self) -> usize {
        self.max_height
    }

    fn alloc(&mut self, size: usize) -> usize {
        let offset = self.height;
        self.height += size;
        self.max_height = self.max_height.max(self.height);
        offset
    }

    fn dealloc(&mut self, offset: usize) {
        self.height = offset;
    }
}

/// A set of registers.
#[derive(Debug)]
struct Regs {
    /// The integer accumulator register.
    ia: Reg,
    /// The single-precision floating-point accumulator register.
    sa: Reg,
    /// The double-precision floating-point accumulator register.
    da: Reg,
}

impl Regs {
    /// Creates a new set of registers.
    fn new() -> Self {
        Self {
            ia: Reg::new(),
            sa: Reg::new(),
            da: Reg::new(),
        }
    }
}

impl Index<RegName> for Regs {
    type Output = Reg;

    fn index(&self, index: RegName) -> &Self::Output {
        match index {
            RegName::Ia => &self.ia,
            RegName::Sa => &self.sa,
            RegName::Da => &self.da,
        }
    }
}

impl IndexMut<RegName> for Regs {
    fn index_mut(&mut self, index: RegName) -> &mut Self::Output {
        match index {
            RegName::Ia => &mut self.ia,
            RegName::Sa => &mut self.sa,
            RegName::Da => &mut self.da,
        }
    }
}

/// The name of a register.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RegName {
    Ia,
    Sa,
    Da,
}

impl RegName {
    /// Returns an iterator over all register names.
    fn iter() -> impl Iterator<Item = RegName> {
        [RegName::Ia, RegName::Sa, RegName::Da].into_iter()
    }
}

/// A register.
#[derive(Debug)]
struct Reg {
    opd_idx: Option<usize>,
}

impl Reg {
    /// Creates a new register.
    fn new() -> Self {
        Self {
            opd_idx: None,
        }
    }

    /// Returns `true` if this register is free.
    fn is_free(&self) -> bool {
        self.opd_idx.is_none()
    }

    /// Returns `true` if this register is used.
    fn is_used(&self) -> bool {
        self.opd_idx.is_some()
    }

    /// Returns the index of the operand this register is allocated to, if any.
    fn opd_idx(&self) -> Option<usize> {
        self.opd_idx
    }

    /// Allocates this register to the operand with the given index.
    fn alloc(&mut self, opd_idx: usize) {
        debug_assert!(self.is_free(), "register is already used");
        self.opd_idx = Some(opd_idx);
    }

    /// Deallocates this register.
    fn dealloc(&mut self) {
        debug_assert!(self.is_used(), "register is already free");
        self.opd_idx = None;
    }
}

impl ValType {
    /// Returns the name of the register to be used for [`Val`]s of this [`ValType`].
    fn reg_name(self) -> RegName {
        match self {
            ValType::I32 | ValType::I64 | ValType::FuncRef | ValType::ExternRef => RegName::Ia,
            ValType::F32 => RegName::Sa,
            ValType::F64 => RegName::Da,
        }
    }
}

// Instruction selection
//
// Most instructions come in multiple variants, depending on the types of their operands, and
// whether their operands are stored as an immediate, on the stack, or in a register. These
// functions are used to select a suitable variant of an instruction based on the types and
// kinds of its operands.

fn select_br_if(kind: OpdKind) -> ThreadedInstr {
    match kind {
        OpdKind::Imm => executor::br_if::<ReadImm>,
        OpdKind::Stk => executor::br_if::<ReadStack>,
        OpdKind::Reg => executor::br_if::<ReadReg>,
    }
}

fn select_br_if_not(kind: OpdKind) -> ThreadedInstr {
    match kind {
        OpdKind::Imm => executor::br_if_not::<ReadImm>,
        OpdKind::Stk => executor::br_if_not::<ReadStack>,
        OpdKind::Reg => executor::br_if_not::<ReadReg>,
    }
}

fn select_br_if_rel_op<T, B>(input_0: OpdKind, input_1: OpdKind) -> ThreadedInstr
where
    T: ReadFromReg,
    B: BinOp<T, Output = i32>,
    B::Output: WriteToReg
{
    match (input_0, input_1) {
        (OpdKind::Imm, OpdKind::Imm) => executor::br_if_rel_op::<T, B, ReadImm, ReadImm>,
        (OpdKind::Stk, OpdKind::Imm) => executor::br_if_rel_op::<T, B, ReadStack, ReadImm>,
        (OpdKind::Reg, OpdKind::Imm) => executor::br_if_rel_op::<T, B, ReadReg, ReadImm>,
        (OpdKind::Imm, OpdKind::Stk) => executor::br_if_rel_op::<T, B, ReadImm, ReadStack>,
        (OpdKind::Stk, OpdKind::Stk) => executor::br_if_rel_op::<T, B, ReadStack, ReadStack>,
        (OpdKind::Reg, OpdKind::Stk) => executor::br_if_rel_op::<T, B, ReadReg, ReadStack>,
        (OpdKind::Imm, OpdKind::Reg) => executor::br_if_rel_op::<T, B, ReadImm, ReadReg>,
        (OpdKind::Stk, OpdKind::Reg) => executor::br_if_rel_op::<T, B, ReadStack, ReadReg>,
        (OpdKind::Reg, OpdKind::Reg) => executor::br_if_rel_op::<T, B, ReadReg, ReadReg>,
    }
}

fn select_br_if_not_rel_op<T, B>(input_0: OpdKind, input_1: OpdKind) -> ThreadedInstr
where
    T: ReadFromReg,
    B: BinOp<T, Output = i32>,
    B::Output: WriteToReg
{
    match (input_0, input_1) {
        (OpdKind::Imm, OpdKind::Imm) => executor::br_if_not_rel_op::<T, B, ReadImm, ReadImm>,
        (OpdKind::Stk, OpdKind::Imm) => executor::br_if_not_rel_op::<T, B, ReadStack, ReadImm>,
        (OpdKind::Reg, OpdKind::Imm) => executor::br_if_not_rel_op::<T, B, ReadReg, ReadImm>,
        (OpdKind::Imm, OpdKind::Stk) => executor::br_if_not_rel_op::<T, B, ReadImm, ReadStack>,
        (OpdKind::Stk, OpdKind::Stk) => executor::br_if_not_rel_op::<T, B, ReadStack, ReadStack>,
        (OpdKind::Reg, OpdKind::Stk) => executor::br_if_not_rel_op::<T, B, ReadReg, ReadStack>,
        (OpdKind::Imm, OpdKind::Reg) => executor::br_if_not_rel_op::<T, B, ReadImm, ReadReg>,
        (OpdKind::Stk, OpdKind::Reg) => executor::br_if_not_rel_op::<T, B, ReadStack, ReadReg>,
        (OpdKind::Reg, OpdKind::Reg) => executor::br_if_not_rel_op::<T, B, ReadReg, ReadReg>,
    }
}

fn select_br_table(kind: OpdKind) -> ThreadedInstr {
    match kind {
        OpdKind::Imm => executor::br_table::<ReadImm>,
        OpdKind::Stk => executor::br_table::<ReadStack>,
        OpdKind::Reg => executor::br_table::<ReadReg>,
    }
}

fn select_ref_null(type_: RefType) -> ThreadedInstr {
    match type_ {
        RefType::FuncRef => executor::copy::<UnguardedFuncRef, ReadImm, WriteStack>,
        RefType::ExternRef => executor::copy::<UnguardedExternRef, ReadImm, WriteStack>,
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
    T: ReadFromReg + WriteToReg
{
    match (input_0, input_1, input_2) {
        (OpdKind::Imm, OpdKind::Imm, OpdKind::Imm) => executor::select::<T, ReadImm, ReadImm, ReadImm, WriteReg>,
        (OpdKind::Stk, OpdKind::Imm, OpdKind::Imm) => executor::select::<T, ReadStack, ReadImm, ReadImm, WriteReg>,
        (OpdKind::Reg, OpdKind::Imm, OpdKind::Imm) => executor::select::<T, ReadReg, ReadImm, ReadImm, WriteReg>,
        (OpdKind::Imm, OpdKind::Stk, OpdKind::Imm) => executor::select::<T, ReadImm, ReadStack, ReadImm, WriteReg>,
        (OpdKind::Stk, OpdKind::Stk, OpdKind::Imm) => executor::select::<T, ReadStack, ReadStack, ReadImm, WriteReg>,
        (OpdKind::Reg, OpdKind::Stk, OpdKind::Imm) => executor::select::<T, ReadReg, ReadStack, ReadImm, WriteReg>,
        (OpdKind::Imm, OpdKind::Reg, OpdKind::Imm) => executor::select::<T, ReadImm, ReadReg, ReadImm, WriteReg>,
        (OpdKind::Stk, OpdKind::Reg, OpdKind::Imm) => executor::select::<T, ReadStack, ReadReg, ReadImm, WriteReg>,
        (OpdKind::Reg, OpdKind::Reg, OpdKind::Imm) => executor::select::<T, ReadReg, ReadReg, ReadImm, WriteReg>,
        (OpdKind::Imm, OpdKind::Imm, OpdKind::Stk) => executor::select::<T, ReadImm, ReadImm, ReadStack, WriteReg>,
        (OpdKind::Stk, OpdKind::Imm, OpdKind::Stk) => executor::select::<T, ReadStack, ReadImm, ReadStack, WriteReg>,
        (OpdKind::Reg, OpdKind::Imm, OpdKind::Stk) => executor::select::<T, ReadReg, ReadImm, ReadStack, WriteReg>,
        (OpdKind::Imm, OpdKind::Stk, OpdKind::Stk) => executor::select::<T, ReadImm, ReadStack, ReadStack, WriteReg>,
        (OpdKind::Stk, OpdKind::Stk, OpdKind::Stk) => executor::select::<T, ReadStack, ReadStack, ReadStack, WriteReg>,
        (OpdKind::Reg, OpdKind::Stk, OpdKind::Stk) => executor::select::<T, ReadReg, ReadStack, ReadStack, WriteReg>,
        (OpdKind::Imm, OpdKind::Reg, OpdKind::Stk) => executor::select::<T, ReadImm, ReadReg, ReadStack, WriteReg>,
        (OpdKind::Stk, OpdKind::Reg, OpdKind::Stk) => executor::select::<T, ReadStack, ReadReg, ReadStack, WriteReg>,
        (OpdKind::Reg, OpdKind::Reg, OpdKind::Stk) => executor::select::<T, ReadReg, ReadReg, ReadStack, WriteReg>,
        (OpdKind::Imm, OpdKind::Imm, OpdKind::Reg) => executor::select::<T, ReadImm, ReadImm, ReadReg, WriteReg>,
        (OpdKind::Stk, OpdKind::Imm, OpdKind::Reg) => executor::select::<T, ReadStack, ReadImm, ReadReg, WriteReg>,
        (OpdKind::Reg, OpdKind::Imm, OpdKind::Reg) => executor::select::<T, ReadReg, ReadImm, ReadReg, WriteReg>,
        (OpdKind::Imm, OpdKind::Stk, OpdKind::Reg) => executor::select::<T, ReadImm, ReadStack, ReadReg, WriteReg>,
        (OpdKind::Stk, OpdKind::Stk, OpdKind::Reg) => executor::select::<T, ReadStack, ReadStack, ReadReg, WriteReg>,
        (OpdKind::Reg, OpdKind::Stk, OpdKind::Reg) => executor::select::<T, ReadReg, ReadStack, ReadReg, WriteReg>,
        (OpdKind::Imm, OpdKind::Reg, OpdKind::Reg) => executor::select::<T, ReadImm, ReadReg, ReadReg, WriteReg>,
        (OpdKind::Stk, OpdKind::Reg, OpdKind::Reg) => executor::select::<T, ReadStack, ReadReg, ReadReg, WriteReg>,
        (OpdKind::Reg, OpdKind::Reg, OpdKind::Reg) => executor::select::<T, ReadReg, ReadReg, ReadReg, WriteReg>,
    }
}

fn select_global_get(type_: ValType) -> ThreadedInstr {
    match type_ {
        ValType::I32 => executor::execute_global_get::<i32, WriteStack>,
        ValType::I64 => executor::execute_global_get::<i64, WriteStack>,
        ValType::F32 => executor::execute_global_get::<f32, WriteStack>,
        ValType::F64 => executor::execute_global_get::<f64, WriteStack>,
        ValType::FuncRef => executor::execute_global_get::<FuncRef, WriteStack>,
        ValType::ExternRef => executor::execute_global_get::<ExternRef, WriteStack>,
    }
}

fn select_global_set(type_: ValType, input: OpdKind) -> ThreadedInstr {
    match type_ {
        ValType::I32 => select_global_set_inner::<i32>(input),
        ValType::I64 => select_global_set_inner::<i64>(input),
        ValType::F32 => select_global_set_inner::<f32>(input),
        ValType::F64 => select_global_set_inner::<f64>(input),
        ValType::FuncRef => select_global_set_inner::<FuncRef>(input),
        ValType::ExternRef => select_global_set_inner::<ExternRef>(input),
    }
}

fn select_global_set_inner<T>(input: OpdKind) -> ThreadedInstr
where
    T: Guarded,
    T::Unguarded: ReadFromReg,
    TypedGlobalEntity<T>: DowncastMut<GlobalEntity>,
{
    match input {
        OpdKind::Imm => executor::execute_global_set::<T, ReadImm>,
        OpdKind::Stk => executor::execute_global_set::<T, ReadStack>,
        OpdKind::Reg => executor::execute_global_set::<T, ReadReg>,
    }
}

fn select_table_get(type_: RefType, input: OpdKind) -> ThreadedInstr {
    match type_ {
        RefType::FuncRef => select_table_get_inner::<FuncRef>(input),
        RefType::ExternRef => select_table_get_inner::<ExternRef>(input),
    }
}

fn select_table_get_inner<T>(input: OpdKind) -> ThreadedInstr
where
    T: Guarded,
    T::Unguarded: ReadFromReg + WriteToReg,
    TypedTableEntity<T>: DowncastRef<TableEntity>,
{
    match input {
        OpdKind::Imm => executor::execute_table_get::<T, ReadImm, WriteStack>,
        OpdKind::Stk => executor::execute_table_get::<T, ReadStack, WriteStack>,
        OpdKind::Reg => executor::execute_table_get::<T, ReadReg, WriteStack>,
    }
}

fn select_table_set(type_: RefType, input_0: OpdKind, input_1: OpdKind) -> ThreadedInstr {
    match type_ {
        RefType::FuncRef => select_table_set_inner::<FuncRef>(input_0, input_1),
        RefType::ExternRef => select_table_set_inner::<ExternRef>(input_0, input_1),
    }
}

fn select_table_set_inner<T>(input_0: OpdKind, input_1: OpdKind) -> ThreadedInstr
where
    T: Guarded,
    T::Unguarded: ReadFromReg,
    TypedTableEntity<T>: DowncastMut<TableEntity>,
{
    match (input_0, input_1) {
        (OpdKind::Imm, OpdKind::Imm) => executor::execute_table_set::<T, ReadImm, ReadImm>,
        (OpdKind::Stk, OpdKind::Imm) => executor::execute_table_set::<T, ReadStack, ReadImm>,
        (OpdKind::Reg, OpdKind::Imm) => executor::execute_table_set::<T, ReadReg, ReadImm>,
        (OpdKind::Imm, OpdKind::Stk) => executor::execute_table_set::<T, ReadImm, ReadStack>,
        (OpdKind::Stk, OpdKind::Stk) => executor::execute_table_set::<T, ReadStack, ReadStack>,
        (OpdKind::Reg, OpdKind::Stk) => executor::execute_table_set::<T, ReadReg, ReadStack>,
        (OpdKind::Imm, OpdKind::Reg) => executor::execute_table_set::<T, ReadImm, ReadReg>,
        (OpdKind::Stk, OpdKind::Reg) => executor::execute_table_set::<T, ReadStack, ReadReg>,
        (OpdKind::Reg, OpdKind::Reg) => executor::execute_table_set::<T, ReadReg, ReadReg>,
    }
}

fn select_table_size(type_: RefType) -> ThreadedInstr {
    match type_ {
        RefType::FuncRef => executor::execute_table_size::<FuncRef, WriteStack>,
        RefType::ExternRef => executor::execute_table_size::<ExternRef, WriteStack>,
    }
}

fn select_table_grow(type_: RefType) -> ThreadedInstr {
    match type_ {
        RefType::FuncRef => executor::execute_table_grow::<FuncRef, ReadStack, ReadStack, WriteStack>,
        RefType::ExternRef => executor::execute_table_grow::<ExternRef, ReadStack, ReadStack, WriteStack>,
    }
}

fn select_table_fill(type_: RefType) -> ThreadedInstr {
    match type_ {
        RefType::FuncRef => executor::execute_table_fill::<FuncRef, ReadStack, ReadStack, ReadStack>,
        RefType::ExternRef => executor::execute_table_fill::<ExternRef, ReadStack, ReadStack, ReadStack>,
    }
}

fn select_table_copy(type_: RefType) -> ThreadedInstr {
    match type_ {
        RefType::FuncRef => executor::execute_table_copy::<FuncRef, ReadStack, ReadStack, ReadStack>,
        RefType::ExternRef => executor::execute_table_copy::<ExternRef, ReadStack, ReadStack, ReadStack>,
    }
}

fn select_table_init(type_: RefType) -> ThreadedInstr {
    match type_ {
        RefType::FuncRef => executor::execute_table_init::<FuncRef, ReadStack, ReadStack, ReadStack>,
        RefType::ExternRef => executor::execute_table_init::<ExternRef, ReadStack, ReadStack, ReadStack>,
    }
}

fn select_elem_drop(type_: RefType) -> ThreadedInstr {
    match type_ {
        RefType::FuncRef => executor::execute_elem_drop::<FuncRef>,
        RefType::ExternRef => executor::execute_elem_drop::<ExternRef>,
    }
}

fn select_load<T>(input: OpdKind) -> ThreadedInstr
where
    T: ReadFromPtr + WriteToReg
{
    match input {
        OpdKind::Imm => executor::load::<T, ReadImm, WriteReg>,
        OpdKind::Stk => executor::load::<T, ReadStack, WriteReg>,
        OpdKind::Reg => executor::load::<T, ReadReg, WriteReg>,
    }
}

fn select_load_n<Dst, Src>(input: OpdKind) -> ThreadedInstr
where
    Dst: ExtendingCastFrom<Src> + WriteToReg,
    Src: ReadFromPtr + ExtendingCast,
{
    match input {
        OpdKind::Imm => executor::load_n::<Dst, Src, ReadImm, WriteReg>,
        OpdKind::Stk => executor::load_n::<Dst, Src, ReadStack, WriteReg>,
        OpdKind::Reg => executor::load_n::<Dst, Src, ReadReg, WriteReg>,
    }
}

fn select_store<T>(input_0: OpdKind, input_1: OpdKind) -> ThreadedInstr
where
    T: ReadFromReg + WriteToPtr
{
    match (input_0, input_1) {
        (OpdKind::Imm, OpdKind::Imm) => executor::store::<T, ReadImm, ReadImm>,
        (OpdKind::Stk, OpdKind::Imm) => executor::store::<T, ReadStack, ReadImm>,
        (OpdKind::Reg, OpdKind::Imm) => executor::store::<T, ReadReg, ReadImm>,
        (OpdKind::Imm, OpdKind::Stk) => executor::store::<T, ReadImm, ReadStack>,
        (OpdKind::Stk, OpdKind::Stk) => executor::store::<T, ReadStack, ReadStack>,
        (OpdKind::Reg, OpdKind::Stk) => executor::store::<T, ReadReg, ReadStack>,
        (OpdKind::Imm, OpdKind::Reg) => executor::store::<T, ReadImm, ReadReg>,
        (OpdKind::Stk, OpdKind::Reg) => executor::store::<T, ReadStack, ReadReg>,
        (OpdKind::Reg, OpdKind::Reg) => executor::store::<T, ReadReg, ReadReg>,
    }
}

fn select_store_n<Src, Dst>(input_0: OpdKind, input_1: OpdKind) -> ThreadedInstr
where
    Src: ReadFromReg + WrappingCast,
    Dst: WrappingCastFrom<Src> + WriteToPtr,
{
    match (input_0, input_1) {
        (OpdKind::Imm, OpdKind::Imm) => executor::store_n::<Src, Dst, ReadImm, ReadImm>,
        (OpdKind::Stk, OpdKind::Imm) => executor::store_n::<Src, Dst, ReadStack, ReadImm>,
        (OpdKind::Reg, OpdKind::Imm) => executor::store_n::<Src, Dst, ReadReg, ReadImm>,
        (OpdKind::Imm, OpdKind::Stk) => executor::store_n::<Src, Dst, ReadImm, ReadStack>,
        (OpdKind::Stk, OpdKind::Stk) => executor::store_n::<Src, Dst, ReadStack, ReadStack>,
        (OpdKind::Reg, OpdKind::Stk) => executor::store_n::<Src, Dst, ReadReg, ReadStack>,
        (OpdKind::Imm, OpdKind::Reg) => executor::store_n::<Src, Dst, ReadImm, ReadReg>,
        (OpdKind::Stk, OpdKind::Reg) => executor::store_n::<Src, Dst, ReadStack, ReadReg>,
        (OpdKind::Reg, OpdKind::Reg) => executor::store_n::<Src, Dst, ReadReg, ReadReg>,
    }
}

fn select_un_op<T, U>(input: OpdKind) -> ThreadedInstr
where
    T: ReadFromReg,
    U: UnOp<T>,
    U::Output: WriteToReg
{
    match input {
        OpdKind::Imm => executor::execute_un_op::<T, U, ReadImm, WriteReg>,
        OpdKind::Stk => executor::execute_un_op::<T, U, ReadStack, WriteReg>,
        OpdKind::Reg => executor::execute_un_op::<T, U, ReadReg, WriteReg>,
    }
}

fn select_bin_op<T, B>(input_0: OpdKind, input_1: OpdKind) -> ThreadedInstr
where
    T: ReadFromReg,
    B: BinOp<T>,
    B::Output: WriteToReg
{
    match (input_0, input_1) {
        (OpdKind::Imm, OpdKind::Imm) => executor::execute_bin_op::<T, B, ReadImm, ReadImm, WriteReg>,
        (OpdKind::Stk, OpdKind::Imm) => executor::execute_bin_op::<T, B, ReadStack, ReadImm, WriteReg>,
        (OpdKind::Reg, OpdKind::Imm) => executor::execute_bin_op::<T, B, ReadReg, ReadImm, WriteReg>,
        (OpdKind::Imm, OpdKind::Stk) => executor::execute_bin_op::<T, B, ReadImm, ReadStack, WriteReg>,
        (OpdKind::Stk, OpdKind::Stk) => executor::execute_bin_op::<T, B, ReadStack, ReadStack, WriteReg>,
        (OpdKind::Reg, OpdKind::Stk) => executor::execute_bin_op::<T, B, ReadReg, ReadStack, WriteReg>,
        (OpdKind::Imm, OpdKind::Reg) => executor::execute_bin_op::<T, B, ReadImm, ReadReg, WriteReg>,
        (OpdKind::Stk, OpdKind::Reg) => executor::execute_bin_op::<T, B, ReadStack, ReadReg, WriteReg>,
        (OpdKind::Reg, OpdKind::Reg) => executor::execute_bin_op::<T, B, ReadReg, ReadReg, WriteReg>,
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
    T: ReadFromReg + WriteToReg
{
    match kind {
        OpdKind::Imm => executor::copy::<T, ReadImm, WriteStack>,
        OpdKind::Stk => executor::copy::<T, ReadStack, WriteStack>,
        OpdKind::Reg => executor::copy::<T, ReadReg, WriteStack>,
    }
}
