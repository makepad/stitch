use {
    crate::{
        extern_::{Extern, UnguardedExtern},
        guarded::Guarded,
        store::{Store, StoreGuard},
    },
    std::any::Any,
};

/// A nullable reference to an external object.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ExternRef(Option<Extern>);

impl ExternRef {
    /// Creates a new [`ExternRef`] wrapping the given underlying object.
    pub fn new<T>(store: &mut Store, object: impl Into<Option<T>>) -> Self
    where
        T: Any + Send + Sync + 'static,
    {
        Self(object.into().map(|object| Extern::new(store, object)))
    }

    /// Creates a null [`ExternRef`].
    pub fn null() -> Self {
        Self(None)
    }

    /// Returns `true` if this [`ExternRef`] is null.
    pub fn is_null(self) -> bool {
        self.0.is_none()
    }

    /// Returns a reference to the underlying object if this `ExternRef` is not null.
    pub fn get(self, store: &Store) -> Option<&dyn Any> {
        self.0.as_ref().map(|extern_| extern_.get(store))
    }
}

impl Guarded for ExternRef {
    type Unguarded = UnguardedExternRef;
    type Guard = StoreGuard;
    
    unsafe fn from_unguarded(unguarded: Self::Unguarded, guard: Self::Guard) -> Self {
        Self(unguarded.map(|unguarded| unsafe { Extern::from_unguarded(unguarded, guard) }))
    }

    fn to_unguarded(self, guard: Self::Guard) -> Self::Unguarded {
        self.0.map(|guarded| guarded.to_unguarded(guard))
    }
}

/// An unguarded [`ExternRef`].
pub(crate) type UnguardedExternRef = Option<UnguardedExtern>;
