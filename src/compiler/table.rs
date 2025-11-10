//! Table instructions

use super::*;

impl<'a> Compiler<'a> {
    /// Compiles a `table.get` instruction.
    pub(crate) fn compile_table_get(&mut self, table_idx: u32) -> Result<(), DecodeError> {
        // Skip this instruction if it is unreachable.
        if self.block(0).is_unreachable {
            return Ok(());
        }

        // Obtain the [`Table`] for this instruction.
        let table = self.instance.table(table_idx).unwrap();

        // Obtain the type of the elements in the [`Table`].
        let elem_type = table.ty(&self.store).elem;

        // Emit the instruction.
        self.emit_instr(select_table_get(elem_type, self.opd(0).kind()));

        // Emit the input and pop it from the stack.
        self.emit_opd(0);
        self.pop_opd();

        // Emit an unguarded handle to the [`Table`].
        self.emit(table.to_unguarded(self.store.guard()));

        // Push the output onto the stack and emit its stack offset.
        self.push_opd(ValType::I32);
        self.emit_stack_offset(self.opd_stack_offset(0));

        Ok(())
    }

    /// Compiles a `table.set` instruction.
    pub(crate) fn compile_table_set(&mut self, table_idx: u32) -> Result<(), DecodeError> {
        if self.block(0).is_unreachable {
            return Ok(());
        }

        // Obtain the [`Table`] for this instruction.
        let table = self.instance.table(table_idx).unwrap();

        // Obtain the type of the elements in the [`Table`].
        let elem_type = table.ty(&self.store).elem;

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
        self.emit(table.to_unguarded(self.store.guard()));

        Ok(())
    }

    /// Compiles a `table.size` instruction.
    pub(crate) fn compile_table_size(&mut self, table_idx: u32) -> Result<(), DecodeError> {
        // Skip this instruction if it is unreachable.
        if self.block(0).is_unreachable {
            return Ok(());
        }

        // Obtain the [`Table`] for this instruction.
        let table = self.instance.table(table_idx).unwrap();

        // Obtain the type of the elements in the [`Table`].
        let elem_type = table.ty(&self.store).elem;

        // Emit the instruction.
        self.emit_instr(select_table_size(elem_type));

        // Emit an unguarded handle to the [`Table`].
        self.emit(table.to_unguarded(self.store.guard()));

        // Push the output onto the stack and emit its stack offset.
        self.push_opd(ValType::I32);
        self.emit_stack_offset(self.opd_stack_offset(0));

        Ok(())
    }

    /// Compiles a `table.grow` instruction.
    pub(crate) fn compile_table_grow(&mut self, table_idx: u32) -> Result<(), DecodeError> {
        // Skip this instruction if it is unreachable.
        if self.block(0).is_unreachable {
            return Ok(());
        }

        // Obtain the [`Table`] for this instruction.
        let table = self.instance.table(table_idx).unwrap();

        // Obtain the type of the elements in the [`Table`].
        let elem_type = table.ty(&self.store).elem;

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
        self.emit(table.to_unguarded(self.store.guard()));

        // Push the output onto the stack and emit its stack offset.
        self.push_opd(ValType::I32);
        self.emit_stack_offset(self.opd_stack_offset(0));

        Ok(())
    }

    /// Compiles a `table.fill` instruction.
    pub(crate) fn compile_table_fill(&mut self, table_idx: u32) -> Result<(), DecodeError> {
        // Skip this instruction if it is unreachable.
        if self.block(0).is_unreachable {
            return Ok(());
        }

        // Obtain the [`Table`] for this instruction.
        let table = self.instance.table(table_idx).unwrap();

        // Obtain the type of the elements in the [`Table`].
        let elem_type = table.ty(&self.store).elem;

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
        self.emit(table.to_unguarded(self.store.guard()));

        Ok(())
    }

    /// Compiles a `table.copy` instruction.
    pub(crate) fn compile_table_copy(
        &mut self,
        dst_table_idx: u32,
        src_table_idx: u32,
    ) -> Result<(), DecodeError> {
        // Skip this instruction if it is unreachable.
        if self.block(0).is_unreachable {
            return Ok(());
        }

        // Obtain the destination and source [`Table`] for this instruction.
        let dst_table = self.instance.table(dst_table_idx).unwrap();
        let src_table = self.instance.table(src_table_idx).unwrap();

        // Obtain the type of the elements in the destination [`Table`].
        let elem_type = dst_table.ty(&self.store).elem;

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
        self.emit(dst_table.to_unguarded(self.store.guard()));
        self.emit(src_table.to_unguarded(self.store.guard()));

        Ok(())
    }

    /// Compiles a `table.init` instruction.
    pub(crate) fn compile_table_init(
        &mut self,
        dst_table_idx: u32,
        src_elem_idx: u32,
    ) -> Result<(), DecodeError> {
        // Skip this instruction if it is unreachable.
        if self.block(0).is_unreachable {
            return Ok(());
        }

        // Obtain the destination [`Table`] and source [`Elem`] for this instruction.
        let dst_table = self.instance.table(dst_table_idx).unwrap();
        let src_elem = self.instance.elem(src_elem_idx).unwrap();

        // Obtain the type of the elements in the destination [`Table`].
        let elem_type = dst_table.ty(&self.store).elem;

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
        self.emit(dst_table.0.to_unguarded(self.store.guard()));
        self.emit(src_elem.0.to_unguarded(self.store.guard()));

        Ok(())
    }

    /// Compiles an `elem.drop` instruction.
    pub(crate) fn compile_elem_drop(&mut self, elem_idx: u32) -> Result<(), DecodeError> {
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
        self.emit(elem.to_unguarded(self.store.guard()));

        Ok(())
    }
}