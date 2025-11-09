//! Numeric validators

use super::*;

impl<'a> Validator<'a> {
    /// Validates a unary operation.
    pub(super) fn validate_un_op<T, U>(&mut self) -> Result<(), DecodeError>
    where
        T: ValTypeOf,
        U: UnOp<T>,
        U::Output: ValTypeOf,
    {
        self.validate_un_op_inner(T::val_type_of(), U::Output::val_type_of())
    }

    fn validate_un_op_inner(&mut self, input: ValType, output: ValType) -> Result<(), DecodeError> {
        self.pop_opd()?.check(input)?;
        self.push_opd(output);
        Ok(())
    }

    /// Validates a binary operation.
    pub(super) fn validate_bin_op<T, B>(&mut self) -> Result<(), DecodeError>
    where
        T: ValTypeOf,
        B: BinOp<T>,
        B::Output: ValTypeOf,
    {
        self.validate_bin_op_inner(T::val_type_of(), T::val_type_of(), B::Output::val_type_of())
    }

    fn validate_bin_op_inner(&mut self, input_0: ValType, input_1: ValType, output: ValType) -> Result<(), DecodeError> {
        self.pop_opd()?.check(input_1)?;
        self.pop_opd()?.check(input_0)?;
        self.push_opd(output);
        Ok(())
    }
}