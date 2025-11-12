//! Memory instructions

use super::*;

impl<'a> Compiler<'a> {
    /// Compiles a load operation.
    pub(super) fn compile_load<T>(&mut self, arg: MemArg) -> Result<(), DecodeError>
    where
        T: ValTypeOf + ReadFromPtr + WriteToReg,
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

    /// Compiles an extending load operation.
    pub(super) fn compile_load_n<Dst, Src>(&mut self, arg: MemArg) -> Result<(), DecodeError>
    where
        Dst: ValTypeOf + ExtendingCastFrom<Src> + WriteToReg,
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
        let output_reg_idx = output_type.reg_name();
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

    /// Compiles a store operation.
    pub(super) fn compile_store<T>(&mut self, arg: MemArg) -> Result<(), DecodeError>
    where
        T: ReadFromReg + WriteToPtr,
    {
        if self.block(0).is_unreachable {
            return Ok(());
        }
        self.compile_store_inner(
            arg,
            select_store::<T>(self.opd(1).kind(), self.opd(0).kind()),
        )
    }

    pub(super) fn compile_store_n<Src, Dst>(&mut self, arg: MemArg) -> Result<(), DecodeError>
    where
        Src: ReadFromReg + WrappingCast,
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

     /// Compiles a `memory.size` instruction.
    pub(super) fn compile_memory_size(&mut self) -> Result<(), DecodeError> {
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
        self.emit_instr(executor::memory_size::<WriteStack> as ThreadedInstr);

        // Emit an unguarded handle to the [`Mem`].
        self.emit(mem.to_unguarded(self.store.guard()));

        // Push the output onto the stack and emit its stack offset.
        self.push_opd(ValType::I32);
        self.emit_stack_offset(self.opd_stack_offset(0));

        Ok(())
    }

    /// Compiles a `memory.grow` instruction.
    pub(super) fn compile_memory_grow(&mut self) -> Result<(), DecodeError> {
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
        self.emit_instr(executor::memory_grow::<ReadStack, WriteStack> as ThreadedInstr);

        // Emit the input and pop it from the stack.
        self.emit_and_pop_opd();

        // Emit an unguarded handle to the memory instance.
        self.emit(mem.to_unguarded(self.store.guard()));

        // Push the output onto the stack and emit its stack offset.
        self.push_opd(ValType::I32);
        self.emit_stack_offset(self.opd_stack_offset(0));

        Ok(())
    }

    /// Compiles a `memory.fill` instruction.
    pub(super) fn compile_memory_fill(&mut self) -> Result<(), DecodeError> {
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
        self.emit_instr(executor::memory_fill::<ReadStack, ReadStack, ReadStack> as ThreadedInstr);

        // Emit the inputs and pop them from the stack.
        for _ in 0..3 {
            self.emit_opd(0);
            self.pop_opd();
        }

        // Emit an unguarded handle to the [`Mem`].
        self.emit(mem.to_unguarded(self.store.guard()));

        Ok(())
    }

    /// Compiles a `memory.copy` instruction.
    pub(super) fn compile_memory_copy(&mut self) -> Result<(), DecodeError> {
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
        self.emit_instr(executor::memory_copy::<ReadStack, ReadStack, ReadStack> as ThreadedInstr);

        // Emit the inputs and pop them from the stack.
        for _ in 0..3 {
            self.emit_opd(0);
            self.pop_opd();
        }

        // Emit an unguarded handle to the [`Mem`].
        self.emit(mem.to_unguarded(self.store.guard()));

        Ok(())
    }

    /// Compiles a `memory.init` instruction.
    pub(super) fn compile_memory_init(&mut self, data_idx: u32) -> Result<(), DecodeError> {
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
        self.emit_instr(executor::memory_init::<ReadStack, ReadStack, ReadStack> as ThreadedInstr);

        // Emit the inputs and pop them from the stack.
        for _ in 0..3 {
            self.emit_opd(0);
            self.pop_opd();
        }

        // Emit unguarded handles to the destination [`Mem`] and source [`Data`] instance.
        self.emit(dst_mem.to_unguarded(self.store.guard()));
        self.emit(src_data.to_unguarded(self.store.guard()));

        Ok(())
    }

    /// Compiles a `data.drop` instruction.
    pub(crate) fn compile_data_drop(&mut self, data_idx: u32) -> Result<(), DecodeError> {
        // Skip this instruction if it is unreachable.
        if self.block(0).is_unreachable {
            return Ok(());
        }

        // Obtain the [`Data`] for this instruction.
        let data = self.instance.data(data_idx).unwrap();

        // Emit the instruction.
        self.emit_instr(executor::data_drop as ThreadedInstr);

        // Emit an unguarded handle to the [`Data`].
        self.emit(data.to_unguarded(self.store.guard()));

        Ok(())
    }
}