use super::*;

impl<'a> Validator<'a> {
    pub(crate) fn validate_global_get(&mut self, global_idx: u32) -> Result<(), DecodeError> {
        let type_ = self.module.global(global_idx)?;
        self.push_opd(type_.content());
        Ok(())
    }

    pub(crate) fn validate_global_set(&mut self, global_idx: u32) -> Result<(), DecodeError> {
        let type_ = self.module.global(global_idx)?;
        if type_.mutability() != Mutability::Var {
            return Err(DecodeError::new("type mismatch"));
        }
        self.pop_opd()?.check(type_.content())?;
        Ok(())
    }
}