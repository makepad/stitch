//! Global instructions

use super::*;

impl<'a> Compiler<'a> {
    /// Compiles a `global.get` instruction.
    pub(super) fn compile_global_get(&mut self, global_idx: u32) -> Result<(), DecodeError> {
        // Skip this instruction if it is unreachable.
        if self.block(0).is_unreachable {
            return Ok(());
        }

        // Obtain the [`Global`] for this instruction.
        let global = self.instance.global(global_idx).unwrap();

        // Obtain the type of the [`Global`].
        let val_type = global.ty(&self.store).content();

        // Emit the instruction.
        self.emit_instr(select_global_get(val_type));

        // Emit an unguarded handle to the [`Global`].
        self.emit(global.to_unguarded(self.store.guard()));

        // Push the output onto the stack and emit its stack offset.
        self.push_opd(val_type);
        self.emit_stack_offset(self.opd_stack_offset(0));

        Ok(())
    }

    /// Compiles a `global.set` instruction.
    pub(super) fn compile_global_set(&mut self, global_idx: u32) -> Result<(), DecodeError> {
        // Skip this instruction if it is unreachable.
        if self.block(0).is_unreachable {
            return Ok(());
        }

        // Obtain the [`Global`] for this instruction.
        let global = self.instance.global(global_idx).unwrap();

        // Obtain the type of the [`Global`].
        let val_type = global.ty(&self.store).content();

        // Emit the instruction.
        self.emit_instr(select_global_set(val_type, self.opd(0).kind()));

        // Emit the input and pop it from the stack.
        self.emit_opd(0);
        self.pop_opd();

        // Emit an unguarded handle to the [`Global`].
        self.emit(global.to_unguarded(self.store.guard()));

        Ok(())
    }
}