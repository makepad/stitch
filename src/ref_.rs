use crate::{
    decode::{Decode, DecodeError, Decoder},
    extern_ref::{ExternRef, UnguardedExternRef},
    func_ref::{FuncRef, UnguardedFuncRef},
    guarded::Guarded,
    store::StoreGuard,
};

/// A Wasm reference.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Ref {
    FuncRef(FuncRef),
    ExternRef(ExternRef),
}

impl Ref {
    /// Returns a null [`Ref`] of the given [`RefType`].
    pub fn null(type_: RefType) -> Self {
        match type_ {
            RefType::FuncRef => FuncRef::null().into(),
            RefType::ExternRef => ExternRef::null().into(),
        }
    }

    /// Returns the [`RefType`] of this [`Ref`].
    pub fn type_(self) -> RefType {
        match self {
            Ref::FuncRef(_) => RefType::FuncRef,
            Ref::ExternRef(_) => RefType::ExternRef,
        }
    }

    /// Returns `true` if this [`Ref`] is a [`FuncRef`].
    pub fn is_func_ref(self) -> bool {
        self.to_func_ref().is_some()
    }

    /// Returns `true` if this [`Ref`] is an [`ExternRef`].
    pub fn is_extern_ref(self) -> bool {
        self.to_extern_ref().is_some()
    }

    /// Converts this [`Ref`] to a [`FuncRef`], if it is one.
    pub fn to_func_ref(self) -> Option<FuncRef> {
        match self {
            Ref::FuncRef(val) => Some(val),
            _ => None,
        }
    }

    /// Converts this [`Ref`] to an [`ExternRef`], if it is one.
    pub fn to_extern_ref(self) -> Option<ExternRef> {
        match self {
            Ref::ExternRef(val) => Some(val),
            _ => None,
        }
    }
}

impl Guarded for Ref {
    type Unguarded = UnguardedRef;
    type Guard = StoreGuard;

    unsafe fn from_unguarded(unguarded: Self::Unguarded, guard: Self::Guard) -> Self {
        match unguarded {
            UnguardedRef::FuncRef(unguarded) => FuncRef::from_unguarded(unguarded, guard).into(),
            UnguardedRef::ExternRef(unguarded) => ExternRef::from_unguarded(unguarded, guard).into(),
        }
    }

    fn to_unguarded(self, guard: Self::Guard) -> Self::Unguarded {
        match self {
            Ref::FuncRef(guarded) => guarded.to_unguarded(guard).into(),
            Ref::ExternRef(guarded) => guarded.to_unguarded(guard).into(),
        }
    }
}

impl From<FuncRef> for Ref {
    fn from(val: FuncRef) -> Self {
        Ref::FuncRef(val)
    }
}

impl From<ExternRef> for Ref {
    fn from(val: ExternRef) -> Self {
        Ref::ExternRef(val)
    }
}

/// An unguarded [`Ref`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum UnguardedRef {
    FuncRef(UnguardedFuncRef),
    ExternRef(UnguardedExternRef),
}

impl From<UnguardedFuncRef> for UnguardedRef {
    fn from(val: UnguardedFuncRef) -> Self {
        UnguardedRef::FuncRef(val)
    }
}

impl From<UnguardedExternRef> for UnguardedRef {
    fn from(val: UnguardedExternRef) -> Self {
        UnguardedRef::ExternRef(val)
    }
}

/// The type of a [`Ref`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RefType {
    FuncRef,
    ExternRef,
}

impl Decode for RefType {
    fn decode(decoder: &mut Decoder<'_>) -> Result<Self, DecodeError> {
        match decoder.read_byte()? {
            0x6F => Ok(Self::ExternRef),
            0x70 => Ok(Self::FuncRef),
            _ => Err(DecodeError::new("malformed reference type")),
        }
    }
}
