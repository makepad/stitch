//! Table validators

use super::*;

impl<'a> Validator<'a> {
    /// Validates a `table.get` instruction.
    pub(crate) fn validate_table_get(&mut self, table_idx: u32) -> Result<(), DecodeError> {
        let ty = self.module.table(table_idx)?;
        self.pop_opd()?.check(ValType::I32)?;
        self.push_opd(ty.elem);
        Ok(())
    }

    /// Validates a `table.set` instruction.
    pub(crate) fn validate_table_set(&mut self, table_idx: u32) -> Result<(), DecodeError> {
        let ty = self.module.table(table_idx)?;
        self.pop_opd()?.check(ty.elem)?;
        self.pop_opd()?.check(ValType::I32)?;
        Ok(())
    }

    /// Validates a `table.size` instruction.
    pub(crate) fn validate_table_size(&mut self, table_idx: u32) -> Result<(), DecodeError> {
        self.module.table(table_idx)?;
        self.push_opd(ValType::I32);
        Ok(())
    }

    /// Validates a `table.grow` instruction.
    pub(crate) fn validate_table_grow(&mut self, table_idx: u32) -> Result<(), DecodeError> {
        let ty = self.module.table(table_idx)?;
        self.pop_opd()?.check(ValType::I32)?;
        self.pop_opd()?.check(ty.elem)?;
        self.push_opd(ValType::I32);
        Ok(())
    }

    /// Validates a `table.fill` instruction.
    pub(crate) fn validate_table_fill(&mut self, table_idx: u32) -> Result<(), DecodeError> {
        let ty = self.module.table(table_idx)?;
        self.pop_opd()?.check(ValType::I32)?;
        self.pop_opd()?.check(ty.elem)?;
        self.pop_opd()?.check(ValType::I32)?;
        Ok(())
    }

    /// Validates a `table.copy` instruction.
    pub(crate) fn validate_table_copy(
        &mut self,
        dst_table_idx: u32,
        src_table_idx: u32,
    ) -> Result<(), DecodeError> {
        let dst_type = self.module.table(dst_table_idx)?;
        let src_type = self.module.table(src_table_idx)?;
        if dst_type.elem != src_type.elem {
            return Err(DecodeError::new("type mismatch"));
        }
        self.pop_opd()?.check(ValType::I32)?;
        self.pop_opd()?.check(ValType::I32)?;
        self.pop_opd()?.check(ValType::I32)?;
        Ok(())
    }

    /// Validates a `table.drop` instruction.
    pub(crate) fn validate_table_init(&mut self, table_idx: u32, elem_idx: u32) -> Result<(), DecodeError> {
        let dst_type = self.module.table(table_idx)?;
        let src_type = self.module.elem(elem_idx)?;
        if dst_type.elem != src_type {
            return Err(DecodeError::new("type mismatch"));
        }
        self.pop_opd()?.check(ValType::I32)?;
        self.pop_opd()?.check(ValType::I32)?;
        self.pop_opd()?.check(ValType::I32)?;
        Ok(())
    }

    /// Validates an `elem.drop` instruction.
    pub(crate) fn validate_elem_drop(&mut self, elem_idx: u32) -> Result<(), DecodeError> {
        self.module.elem(elem_idx)?;
        Ok(())
    }
}