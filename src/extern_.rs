use {
    crate::{
        guarded::Guarded,
        store::{Handle, Store, StoreGuard, UnguardedHandle},
    },
    std::any::Any,
};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Extern(Handle<ExternEntity>);

impl Extern {
    pub fn new(store: &mut Store, object: impl Any + Send + Sync + 'static) -> Self {
        Self(store.insert_extern(ExternEntity::new(object)))
    }

    pub fn get(self, store: &Store) -> &dyn Any {
        self.0.as_ref(store).get()
    }
}

impl Guarded for Extern {
    type Unguarded = UnguardedExtern;
    type Guard = StoreGuard;

    unsafe fn from_unguarded(unguarded: Self::Unguarded, guard: Self::Guard) -> Self {
        Self(Handle::from_unguarded(unguarded, guard))
    }

    fn to_unguarded(self, guard: Self::Guard) -> Self::Unguarded {
        self.0.to_unguarded(guard)
    }
}

pub(crate) type UnguardedExtern = UnguardedHandle<ExternEntity>;

#[derive(Debug)]
pub(crate) struct ExternEntity {
    object: Box<dyn Any + Send + Sync + 'static>,
}

impl ExternEntity {
    fn new(object: impl Any + Send + Sync + 'static) -> Self {
        Self {
            object: Box::new(object),
        }
    }

    fn get(&self) -> &dyn Any {
        &*self.object
    }
}
