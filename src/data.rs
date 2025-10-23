use {
    crate::{
        guarded::Guarded,
        store::{Handle, Store, StoreGuard, UnguardedHandle},
    },
    std::sync::Arc,
};

/// A Wasm data segment.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub(crate) struct Data(pub(crate) Handle<DataEntity>);

impl Data {
    pub(crate) fn new(store: &mut Store, bytes: Arc<[u8]>) -> Self {
        Self(store.insert_data(DataEntity::new(bytes)))
    }

    pub(crate) fn drop_bytes(self, store: &mut Store) {
        self.0.as_mut(store).drop_bytes();
    }
}

impl Guarded for Data {
    type Unguarded = UnguardedData;
    type Guard = StoreGuard;

    unsafe fn from_unguarded(unguarded: UnguardedData, guard: StoreGuard) -> Self {
        Self(Handle::from_unguarded(unguarded, guard))
    }

    fn to_unguarded(self, guard: Self::Guard) -> UnguardedData {
        self.0.to_unguarded(guard)
    }
}

/// An unguarded [`Data`].
pub(crate) type UnguardedData = UnguardedHandle<DataEntity>;

/// The representation of a [`Data`] in the store.
#[derive(Debug)]
pub(crate) struct DataEntity {
    bytes: Option<Arc<[u8]>>,
}

impl DataEntity {
    fn new(bytes: Arc<[u8]>) -> Self {
        Self { bytes: Some(bytes) }
    }

    pub(crate) fn bytes(&self) -> &[u8] {
        self.bytes.as_ref().map_or(&[], |bytes| &bytes)
    }

    pub(crate) fn drop_bytes(&mut self) {
        self.bytes = None;
    }
}
