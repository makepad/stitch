use crate::{
    func::{Func, UnguardedFunc},
    guarded::Guarded,
    store::{Handle, StoreGuard},
};

/// A nullable reference to a [`Func`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct FuncRef(pub(crate) Option<Func>);

impl FuncRef {
    /// Creates a new [`FuncRef`].
    pub fn new(func: impl Into<Option<Func>>) -> Self {
        Self(func.into())
    }

    /// Creates a null [`FuncRef`].
    pub fn null() -> Self {
        Self(None)
    }

    /// Returns `true` if this [`FuncRef`] is null.
    pub fn is_null(self) -> bool {
        self.0.is_none()
    }

    /// Returns the underlying [`Func`] if this [`FuncRef`] is not null.
    pub fn get(self) -> Option<Func> {
        self.0
    }
}

impl Guarded for FuncRef {
    type Unguarded = UnguardedFuncRef;
    type Guard = StoreGuard;

    unsafe fn from_unguarded(unguarded: Self::Unguarded, guard: Self::Guard) -> Self {
        Self(unguarded.map(|unguarded| unsafe { Func(Handle::from_unguarded(unguarded, guard)) }))
    }

    fn to_unguarded(self, guard: StoreGuard) -> Self::Unguarded {
        self.0.map(|guarded| guarded.0.to_unguarded(guard))
    }
}

impl From<Func> for FuncRef {
    fn from(func: Func) -> Self {
        Self::new(func)
    }
}

/// An unguarded [`FuncRef`].
pub(crate) type UnguardedFuncRef = Option<UnguardedFunc>;
