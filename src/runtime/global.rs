use {
    crate::{
        decode::{Decode, DecodeError, Decoder},
        downcast::{DowncastMut, DowncastRef},
        guarded::Guarded,
        ref_::{ExternRef, FuncRef},
        store::{Handle, Store, StoreGuard, UnguardedHandle},
        val::{Val, ValType, ValTypeOf},
    },
    std::{error::Error, fmt},
};

/// A WebAssembly global variable.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[repr(transparent)]
pub struct Global(pub(crate) Handle<GlobalEntity>);

impl Global {
    /// Creates a new [`Global`] with the following parameters:
    /// 
    /// * `store` - the [`Store`] in which to create the new [`Global`].
    /// * `ty` - the [`GlobalType`] of the new [`Global`].
    /// * `init_val` - the initial value of the new [`Global`]'s content.
    ///
    /// # Errors
    /// 
    /// If the [`ValType`] of `init_val` does not match that of the new [`Global`]'s content.
    /// 
    /// # Panics
    ///
    /// If `init_val` does not originate from `store`.
    pub fn new(store: &mut Store, ty: GlobalType, init_val: Val) -> Result<Self, GlobalError> {
        if init_val.type_() != ty.content() {
            return Err(GlobalError::TypeMismatch);
        }
        let global = match init_val {
            Val::I32(init_val) => {
                GlobalEntity::I32(TypedGlobalEntity::new(init_val, ty.mutability(), store.guard()))
            }
            Val::I64(init_val) => {
                GlobalEntity::I64(TypedGlobalEntity::new(init_val, ty.mutability(), store.guard()))
            }
            Val::F32(init_val) => {
                GlobalEntity::F32(TypedGlobalEntity::new(init_val, ty.mutability(), store.guard()))
            }
            Val::F64(init_val) => {
                GlobalEntity::F64(TypedGlobalEntity::new(init_val, ty.mutability(), store.guard()))
            }
            Val::FuncRef(init_val) => {
                GlobalEntity::FuncRef(TypedGlobalEntity::new(init_val, ty.mutability(), store.guard()))
            }
            Val::ExternRef(init_val) => {
                GlobalEntity::ExternRef(TypedGlobalEntity::new(init_val, ty.mutability(), store.guard()))
            }
        };
        Ok(Self(store.insert_global(global)))
    }

    /// Returns the [`GlobalType`] of `self` in the given `store`.
    ///
    /// # Panics
    ///
    /// If `self` does not originate from `store`.
    pub fn ty(self, store: &Store) -> GlobalType {
        match self.0.as_ref(store) {
            GlobalEntity::I32(global) => global.ty(),
            GlobalEntity::I64(global) => global.ty(),
            GlobalEntity::F32(global) => global.ty(),
            GlobalEntity::F64(global) => global.ty(),
            GlobalEntity::FuncRef(global) => global.ty(),
            GlobalEntity::ExternRef(global) => global.ty(),
        }
    }

    /// Returns the current value of `self`'s content in the given `store`.
    ///
    /// # Panics
    ///
    /// If `self` does not originate from [`Store`].
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

    /// Sets the value of `self`'s content to `new_val` in the given `store`.
    ///
    /// # Errors
    ///
    /// * If `self` is a constant.
    /// * If the [`ValType`] of `new_val` does not match that of `self`'s content.
    ///
    /// # Panics
    ///
    /// * If `self` does not originate from `store`.
    /// * If `new_val` does not originate from `store`.
    pub fn set(self, store: &mut Store, new_val: Val) -> Result<(), GlobalError> {
        if self.ty(store).mutability() != Mutability::Var {
            return Err(GlobalError::Immutable);
        }
        match (self.0.as_mut(store), new_val) {
            (GlobalEntity::I32(global), Val::I32(new_val)) => Ok(global.set(new_val)),
            (GlobalEntity::I64(global), Val::I64(new_val)) => Ok(global.set(new_val)),
            (GlobalEntity::F32(global), Val::F32(new_val)) => Ok(global.set(new_val)),
            (GlobalEntity::F64(global), Val::F64(new_val)) => Ok(global.set(new_val)),
            (GlobalEntity::FuncRef(global), Val::FuncRef(new_val)) => Ok(global.set(new_val)),
            (GlobalEntity::ExternRef(global), Val::ExternRef(new_val)) => Ok(global.set(new_val)),
            _ => Err(GlobalError::TypeMismatch),
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

/// An unguarded handle to a [`Global`].
pub(crate) type UnguardedGlobal = UnguardedHandle<GlobalEntity>;

/// The type of a [`Global`].
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct GlobalType {
    content: ValType,
    mutability: Mutability,
}

impl GlobalType {
    /// Creates a new [`GlobalType`] with the following parameters:
    /// 
    /// * `content` -  the [`ValType`] of the [`Global`]'s content.
    /// * `mutability` - the [`Mutability`] of the [`Global`].
    pub fn new(content: ValType, mutability: Mutability) -> Self {
        Self {
            content,
            mutability,
        }
    }

    /// Returns the [`ValType`] of the [`Global`]'s content.
    pub fn content(self) -> ValType {
        self.content
    }

    /// Returns the [`Mutability`] of the [`Global`].
    pub fn mutability(self) -> Mutability {
        self.mutability
    }
}

impl Decode for GlobalType {
    fn decode(decoder: &mut Decoder<'_>) -> Result<Self, DecodeError> {
        Ok(Self {
            content: decoder.decode()?,
            mutability: decoder.decode()?,
        })
    }
}

/// The mutability of a [`Global`].
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum Mutability {
    /// The [`Global`] is a constant.
    Const,
    /// The [`Global`] is a variable.
    Var,
}

impl Decode for Mutability {
    fn decode(decoder: &mut Decoder<'_>) -> Result<Self, DecodeError> {
        match decoder.read_byte()? {
            0x00 => Ok(Self::Const),
            0x01 => Ok(Self::Var),
            _ => Err(DecodeError::new("invalid mutability")),
        }
    }
}

/// An error returned by operations on a [`Global`].
#[derive(Debug)]
#[non_exhaustive]
pub enum GlobalError {
    /// The [`Global`] is immutable.
    Immutable,
    /// The [`ValType`] does not match that of the [`Global`]'s content.
    TypeMismatch,
}

impl fmt::Display for GlobalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GlobalError::Immutable => write!(f, "global is immutable"),
            GlobalError::TypeMismatch => write!(f, "type does not match that of global's content"),
        }
    }
}

impl Error for GlobalError {}

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
    pub(crate) fn downcast_ref<T>(&self) -> Option<&TypedGlobalEntity<T>>
    where
        T: Guarded,
        TypedGlobalEntity<T>: DowncastRef<Self>,
    {
        TypedGlobalEntity::downcast_ref(self)
    }

    pub(crate) fn downcast_mut<T>(&mut self) -> Option<&mut TypedGlobalEntity<T>>
    where
        T: Guarded,
        TypedGlobalEntity<T>: DowncastMut<Self>,
    {
        TypedGlobalEntity::downcast_mut(self)
    }
}

#[derive(Debug)]
pub(crate) struct TypedGlobalEntity<T>
where
    T: Guarded,
{
    content: T::Unguarded,
    mutability: Mutability,
    guard: T::Guard,
}

impl<T> TypedGlobalEntity<T>
where
    T: Guarded,
{
    fn new(init_val: T, mutability: Mutability, guard: T::Guard) -> Self {
        let init_val = init_val.to_unguarded(guard);
        unsafe { Self::new_unguarded(init_val, mutability, guard) }
    }

    unsafe fn new_unguarded(val: T::Unguarded, mutability: Mutability, guard: T::Guard) -> Self {
        Self {
            content: val,
            mutability,
            guard,
        }
    }

    fn get(&self) -> T {
        let val = self.get_unguarded();
        unsafe { T::from_unguarded(val, self.guard) }
    }

    pub(crate) fn get_unguarded(&self) -> T::Unguarded {
        self.content
    }

    fn set(&mut self, new_val: T) {
        let new_val = new_val.to_unguarded(self.guard);
        unsafe { self.set_unguarded(new_val) }
    }

    pub(crate) unsafe fn set_unguarded(&mut self, new_val: T::Unguarded) {
        self.content = new_val;
    }
}

impl<T> TypedGlobalEntity<T>
where
    T: Guarded + ValTypeOf,
{
    fn ty(&self) -> GlobalType {
        GlobalType::new(T::val_type_of(), self.mutability)
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
