//! Variable instructions

use super::*;

impl<'a> Compiler<'a> {
    /// Compiles a `local.get` instruction.
    pub(super) fn compile_local_get(&mut self, local_idx: u32) -> Result<(), DecodeError> {
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
    pub(super) fn compile_local_set(&mut self, local_idx: u32) -> Result<(), DecodeError> {
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
    pub(super) fn compile_local_tee(&mut self, local_idx: u32) -> Result<(), DecodeError> {
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
        self.emit_stack_offset(self.locals[local_idx].stack_offset);
     
        Ok(())
    }

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