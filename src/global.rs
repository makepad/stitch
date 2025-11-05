use {
    crate::{
        decode::{Decode, DecodeError, Decoder},
        downcast::{DowncastMut, DowncastRef},
        guarded::Guarded,
        ref_::{ExternRef, FuncRef},
        store::{Handle, Store, StoreGuard, UnguardedHandle},
        val::{Val, ValType},
    },
    std::{error::Error, fmt},
};

/// A Wasm global.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct Global(pub(crate) Handle<GlobalEntity>);

impl Global {
    /// Creates a new [`Global`] with the given [`GlobalType`] and initialization [`Val`] in the
    /// given [`Store`].
    ///
    /// # Errors
    ///
    /// If the [`ValType`] of the initialiation [`Val`] does not match the [`ValType`] of the
    /// [`Global`] to be created.
    ///
    /// # Panics
    ///
    /// If the initialization [`Val`] is not owned by the given [`Store`].
    pub fn new(store: &mut Store, type_: GlobalType, val: Val) -> Result<Self, GlobalError> {
        match (type_.val, val) {
            (ValType::I32, Val::I32(val)) => Ok(Self(
                store.insert_global(GlobalEntity::I32(TypedGlobalEntity::new(type_.mut_, val, store.id()))),
            )),
            (ValType::I64, Val::I64(val)) => Ok(Self(
                store.insert_global(GlobalEntity::I64(TypedGlobalEntity::new(type_.mut_, val, store.id()))),
            )),
            (ValType::F32, Val::F32(val)) => Ok(Self(
                store.insert_global(GlobalEntity::F32(TypedGlobalEntity::new(type_.mut_, val, store.id()))),
            )),
            (ValType::F64, Val::F64(val)) => Ok(Self(
                store.insert_global(GlobalEntity::F64(TypedGlobalEntity::new(type_.mut_, val, store.id()))),
            )),
            (ValType::FuncRef, Val::FuncRef(val)) => Ok(Self(
                store.insert_global(GlobalEntity::FuncRef(TypedGlobalEntity::new(type_.mut_, val, store.id()))),
            )),
            (ValType::ExternRef, Val::ExternRef(val)) => Ok(Self(
                store.insert_global(GlobalEntity::ExternRef(TypedGlobalEntity::new(type_.mut_, val, store.id()))),
            )),
            _ => Err(GlobalError::ValTypeMismatch),
        }
    }

    /// Returns the [`GlobalType`] of this [`Global`].
    pub fn type_(self, store: &Store) -> GlobalType {
        match self.0.as_ref(store) {
            GlobalEntity::I32(global) => GlobalType {
                mut_: global.mut_(),
                val: ValType::I32,
            },
            GlobalEntity::I64(global) => GlobalType {
                mut_: global.mut_(),
                val: ValType::I64,
            },
            GlobalEntity::F32(global) => GlobalType {
                mut_: global.mut_(),
                val: ValType::F32,
            },
            GlobalEntity::F64(global) => GlobalType {
                mut_: global.mut_(),
                val: ValType::F64,
            },
            GlobalEntity::FuncRef(global) => GlobalType {
                mut_: global.mut_(),
                val: ValType::FuncRef,
            },
            GlobalEntity::ExternRef(global) => GlobalType {
                mut_: global.mut_(),
                val: ValType::ExternRef,
            },
        }
    }

    /// Returns the value of this [`Global`].
    pub fn get(self, store: &Store) -> Val {
         match self.0.as_ref(store) {
            GlobalEntity::I32(global) => global.get().into(),
            GlobalEntity::I64(global) => global.get().into(),
            GlobalEntity::F32(global) => global.get().into(),
            GlobalEntity::F64(global) => global.get().into(),
            GlobalEntity::FuncRef(global) => global.get().into(),
            GlobalEntity::ExternRef(global) => global.get().into(),
        }
    }
    /// Sets the value of this [`Global`] to the given [`Val`].
    ///
    /// # Errors
    ///
    /// - If the global is immutable.
    /// - If the [`ValType`] of the given [`Val`] does not match the [`ValType`] of this [`Global`].
    ///
    /// # Panics
    ///
    /// If the given [`Val`] is not owned by the given [`Store`].
    pub fn set(self, store: &mut Store, val: Val) -> Result<(), GlobalError> {
        if self.type_(store).mut_ != Mut::Var {
            return Err(GlobalError::Immutable);
        }
        match (self.0.as_mut(store), val) {
            (GlobalEntity::I32(global), Val::I32(val)) => Ok(global.set(val)),
            (GlobalEntity::I64(global), Val::I64(val)) => Ok(global.set(val)),
            (GlobalEntity::F32(global), Val::F32(val)) => Ok(global.set(val)),
            (GlobalEntity::F64(global), Val::F64(val)) => Ok(global.set(val)),
            (GlobalEntity::FuncRef(global), Val::FuncRef(val)) => Ok(global.set(val)),
            (GlobalEntity::ExternRef(global), Val::ExternRef(val)) => Ok(global.set(val)),
            _ => Err(GlobalError::ValTypeMismatch),
        }
    }
}

impl Guarded for Global {
    type Unguarded = UnguardedGlobal;
    type Guard = StoreGuard;

    unsafe fn from_unguarded(global: UnguardedGlobal, guard: Self::Guard) -> Self {
        Self(Handle::from_unguarded(global, guard))
    }

    fn to_unguarded(self, guard: Self::Guard) -> UnguardedGlobal {
        self.0.to_unguarded(guard).into()
    }
}

/// An unguarded version of [`Global`].
pub(crate) type UnguardedGlobal = UnguardedHandle<GlobalEntity>;

/// The type of a [`Global`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GlobalType {
    /// The [`Mut`] of the [`Global`]
    pub mut_: Mut,
    /// The [`ValType`] of the [`Global`].
    pub val: ValType,
}

impl Decode for GlobalType {
    fn decode(decoder: &mut Decoder<'_>) -> Result<Self, DecodeError> {
        let val = decoder.decode()?;
        let mut_ = decoder.decode()?;
        Ok(Self { val, mut_ })
    }
}

/// The mutability of a `Global`.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Mut {
    /// The global is a constant.
    Const,
    /// The global is a variable.
    Var,
}

impl Decode for Mut {
    fn decode(decoder: &mut Decoder<'_>) -> Result<Self, DecodeError> {
        match decoder.read_byte()? {
            0x00 => Ok(Self::Const),
            0x01 => Ok(Self::Var),
            _ => Err(DecodeError::new("malformed mutability")),
        }
    }
}

/// An error which can occur when operating on a [`Global`].
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum GlobalError {
    Immutable,
    ValTypeMismatch,
}

impl fmt::Display for GlobalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GlobalError::Immutable => write!(f, "global is immutable"),
            GlobalError::ValTypeMismatch => write!(f, "value type mismatch"),
        }
    }
}

impl Error for GlobalError {}

/// The representation of a [`Global`] in a [`Store`].
#[derive(Debug)]
pub(crate) enum GlobalEntity {
    I32(TypedGlobalEntity<i32>),
    I64(TypedGlobalEntity<i64>),
    F32(TypedGlobalEntity<f32>),
    F64(TypedGlobalEntity<f64>),
    FuncRef(TypedGlobalEntity<FuncRef>),
    ExternRef(TypedGlobalEntity<ExternRef>),
}

impl GlobalEntity {
    /// Returns a reference to the inner value of this [`GlobalEntity`] if it is a
    /// [`GlobalEntityT<T>`].
    pub(crate) fn downcast_ref<T>(&self) -> Option<&TypedGlobalEntity<T>>
    where
        T: Guarded,
        TypedGlobalEntity<T>: DowncastRef<Self>,
    {
        TypedGlobalEntity::downcast_ref(self)
    }

    /// Returns a mutable reference to the inner value of this [`GlobalEntity`] if it is a
    /// [`GlobalEntityT<T>`].
    pub(crate) fn downcast_mut<T>(&mut self) -> Option<&mut TypedGlobalEntity<T>>
    where
        T: Guarded,
        TypedGlobalEntity<T>: DowncastMut<Self>,
    {
        TypedGlobalEntity::downcast_mut(self)
    }
}

/// A typed [`GlobalEntity`].
#[derive(Debug)]
pub(crate) struct TypedGlobalEntity<T>
where
    T: Guarded
{
    mut_: Mut,
    val: T::Unguarded,
    guard: T::Guard,
}

impl<T> TypedGlobalEntity<T>
where
    T: Guarded,
{
    /// Creates a new [`GlobalEntityT`] with the given [`Mut`] and value.
    fn new(mut_: Mut, val: T, guard: T::Guard) -> Self {
        let val = val.to_unguarded(guard);
        unsafe { Self::new_unguarded(mut_, val, guard) }
    }

    unsafe fn new_unguarded(mut_: Mut, val: T::Unguarded, guard: T::Guard) -> Self {
        Self { mut_, val, guard }
    }

    /// Returns the [`Mut`] of this [`GlobalEntityT`].
    fn mut_(&self) -> Mut {
        self.mut_
    }

    /// Returns the value of this [`GlobalEntityT`].
    pub(crate) fn get(&self) -> T {
        let val = self.get_unguarded();
        unsafe { T::from_unguarded(val, self.guard) }
    }

    pub(crate) fn get_unguarded(&self) -> T::Unguarded {
        self.val
    }

    /// Sets the value of this [`GlobalEntityT`] to the given value.
    pub(crate) fn set(&mut self, val: T) {
        let val = val.to_unguarded(self.guard);
        unsafe { self.set_unguarded(val) }
    }

    pub(crate) unsafe fn set_unguarded(&mut self, val: T::Unguarded) {
        self.val = val;
    }
}

impl DowncastRef<GlobalEntity> for TypedGlobalEntity<i32> {
    fn downcast_ref(global: &GlobalEntity) -> Option<&TypedGlobalEntity<i32>> {
        match global {
            GlobalEntity::I32(global) => Some(global),
            _ => None,
        }
    }
}

impl DowncastMut<GlobalEntity> for TypedGlobalEntity<i32> {
    fn downcast_mut(global: &mut GlobalEntity) -> Option<&mut TypedGlobalEntity<i32>> {
        match global {
            GlobalEntity::I32(global) => Some(global),
            _ => None,
        }
    }
}

impl DowncastRef<GlobalEntity> for TypedGlobalEntity<i64> {
    fn downcast_ref(global: &GlobalEntity) -> Option<&TypedGlobalEntity<i64>> {
        match global {
            GlobalEntity::I64(global) => Some(global),
            _ => None,
        }
    }
}

impl DowncastMut<GlobalEntity> for TypedGlobalEntity<i64> {
    fn downcast_mut(global: &mut GlobalEntity) -> Option<&mut TypedGlobalEntity<i64>> {
        match global {
            GlobalEntity::I64(global) => Some(global),
            _ => None,
        }
    }
}

impl DowncastRef<GlobalEntity> for TypedGlobalEntity<f32> {
    fn downcast_ref(global: &GlobalEntity) -> Option<&TypedGlobalEntity<f32>> {
        match global {
            GlobalEntity::F32(global) => Some(global),
            _ => None,
        }
    }
}

impl DowncastMut<GlobalEntity> for TypedGlobalEntity<f32> {
    fn downcast_mut(global: &mut GlobalEntity) -> Option<&mut TypedGlobalEntity<f32>> {
        match global {
            GlobalEntity::F32(global) => Some(global),
            _ => None,
        }
    }
}

impl DowncastRef<GlobalEntity> for TypedGlobalEntity<f64> {
    fn downcast_ref(global: &GlobalEntity) -> Option<&TypedGlobalEntity<f64>> {
        match global {
            GlobalEntity::F64(global) => Some(global),
            _ => None,
        }
    }
}

impl DowncastMut<GlobalEntity> for TypedGlobalEntity<f64> {
    fn downcast_mut(global: &mut GlobalEntity) -> Option<&mut TypedGlobalEntity<f64>> {
        match global {
            GlobalEntity::F64(global) => Some(global),
            _ => None,
        }
    }
}

impl DowncastRef<GlobalEntity> for TypedGlobalEntity<FuncRef> {
    fn downcast_ref(global: &GlobalEntity) -> Option<&TypedGlobalEntity<FuncRef>> {
        match global {
            GlobalEntity::FuncRef(global) => Some(global),
            _ => None,
        }
    }
}

impl DowncastMut<GlobalEntity> for TypedGlobalEntity<FuncRef> {
    fn downcast_mut(global: &mut GlobalEntity) -> Option<&mut TypedGlobalEntity<FuncRef>> {
        match global {
            GlobalEntity::FuncRef(global) => Some(global),
            _ => None,
        }
    }
}

impl DowncastRef<GlobalEntity> for TypedGlobalEntity<ExternRef> {
    fn downcast_ref(global: &GlobalEntity) -> Option<&TypedGlobalEntity<ExternRef>> {
        match global {
            GlobalEntity::ExternRef(global) => Some(global),
            _ => None,
        }
    }
}

impl DowncastMut<GlobalEntity> for TypedGlobalEntity<ExternRef> {
    fn downcast_mut(global: &mut GlobalEntity) -> Option<&mut TypedGlobalEntity<ExternRef>> {
        match global {
            GlobalEntity::ExternRef(global) => Some(global),
            _ => None,
        }
    }
}
