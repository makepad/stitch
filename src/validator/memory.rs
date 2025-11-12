//! Memory instructions
//! 
use super::*;

impl<'a> Validator<'a> {
    /// Validates a load operation.
    pub(super) fn validate_load<T>(&mut self, arg: MemArg) -> Result<(), DecodeError>
    where
        T: ValTypeOf,
    {
        self.validate_load_inner(arg, T::val_type_of(), align_of::<T>().ilog(2))
    }

    /// Validates an extending load operation.
    pub(super) fn validate_load_n<Dst, Src>(&mut self, arg: MemArg) -> Result<(), DecodeError>
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

    /// Validates a store operation.
    pub(super) fn validate_store<T>(&mut self, arg: MemArg) -> Result<(), DecodeError>
    where
        T: ValTypeOf,
    {
        self.validate_store_inner(arg, T::val_type_of(), align_of::<T>().ilog(2))
    }

    /// Validates an extending store operation.
    pub(super) fn validate_store_n<Src, Dst>(&mut self, arg: MemArg) -> Result<(), DecodeError>
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

    /// Validates a `memory.size` instruction.
    pub(super) fn validate_memory_size(&mut self) -> Result<(), DecodeError> {
        self.module.memory(0)?;
        self.push_opd(ValType::I32);
        Ok(())
    }

    /// Validates a `memory.grow` instruction.
    pub(super) fn validate_memory_grow(&mut self) -> Result<(), DecodeError> {
        self.module.memory(0)?;
        self.pop_opd()?.check(ValType::I32)?;
        self.push_opd(ValType::I32);
        Ok(())
    }

    /// Validates a `memory.fill` instruction.
    pub(super) fn validate_memory_fill(&mut self) -> Result<(), DecodeError> {
        self.module.memory(0)?;
        self.pop_opd()?.check(ValType::I32)?;
        self.pop_opd()?.check(ValType::I32)?;
        self.pop_opd()?.check(ValType::I32)?;
        Ok(())
    }

    /// Validates a `memory.copy` instruction.
    pub(super) fn validate_memory_copy(&mut self) -> Result<(), DecodeError> {
        self.module.memory(0)?;
        self.pop_opd()?.check(ValType::I32)?;
        self.pop_opd()?.check(ValType::I32)?;
        self.pop_opd()?.check(ValType::I32)?;
        Ok(())
    }

    /// Validates a `memory.init` instruction.
    pub(super) fn validate_memory_init(&mut self, data_idx: u32) -> Result<(), DecodeError> {
        self.module.memory(0)?;
        self.module.data(data_idx)?;
        self.pop_opd()?.check(ValType::I32)?;
        self.pop_opd()?.check(ValType::I32)?;
        self.pop_opd()?.check(ValType::I32)?;
        Ok(())
    }

    /// Validates a `data.drop` instruction.
    pub(super) fn validate_data_drop(&mut self, data_idx: u32) -> Result<(), DecodeError> {
        self.module.data(data_idx)?;
        Ok(())
    }
}