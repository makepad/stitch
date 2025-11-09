//! Numeric compilers

use super::*;

impl<'a> Compiler<'a> {
    /// Compiles a unary operation.
    pub(crate) fn compile_un_op<T, U>(&mut self) -> Result<(), DecodeError>
    where
        T: ReadFromReg,
        U: UnOp<T>,
        U::Output: ValTypeOf + WriteToReg,
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
        let output_reg_idx = output_type.reg_name();
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

    /// Compiles a binary operation.
    pub(crate) fn compile_bin_op<T, B>(&mut self) -> Result<(), DecodeError>
    where
        T: ReadFromReg,
        B: BinOp<T>,
        B::Output: ValTypeOf + WriteToReg,
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
        let output_reg_idx = output_type.reg_name();
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
}
