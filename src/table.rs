use {
    crate::{
        decode::{Decode, DecodeError, Decoder},
        downcast::{DowncastMut, DowncastRef},
        elem::{Elem, ElemEntity, TypedElemEntity},
        guarded::Guarded,
        limits::Limits,
        ref_::{ExternRef, FuncRef, Ref, RefType, UnguardedRef},
        store::{Handle, HandlePair, Store, StoreGuard, UnguardedHandle},
        trap::Trap,
    },
    std::{error::Error, fmt},
};

/// A Wasm table.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct Table(pub(crate) Handle<TableEntity>);

impl Table {
    /// Creates a new [`Table`] with the given [`TableType`] and initialization [`Ref`] in the
    /// given [`Store`].
    ///
    /// # Errors
    ///
    /// - If the [`RefType`] of the initialization [`Ref`] does not match the [`RefType`] of the
    ///   elements in the [`Table`] to be created.
    ///
    /// # Panics
    ///
    /// - If the [`TableType`] is invalid.
    /// - If the initialization [`Ref`] is not owned by the given [`Store`].
    pub fn new(
        store: &mut Store,
        type_: TableType,
        val: Ref,
    ) -> Result<Self, TableError> {
        match (type_.elem, val) {
            (RefType::FuncRef, Ref::FuncRef(val)) => Ok(Self(
                store.insert_table(TableEntity::FuncRef(TypedTableEntity::new(type_.limits, val, store.id()))),
            )),
            (RefType::ExternRef, Ref::ExternRef(val)) => Ok(Self(
                store.insert_table(TableEntity::ExternRef(TypedTableEntity::new(type_.limits, val, store.id()))),
            )),
            _ => Err(TableError::ElemTypeMismatch),
        }
    }

    /// Returns the [`TableType`] of this [`Table`].
    pub fn type_(self, store: &Store) -> TableType {
        match self.0.as_ref(store) {
            TableEntity::FuncRef(table) => TableType {
                limits: table.limits(),
                elem: RefType::FuncRef,
            },
            TableEntity::ExternRef(table) => TableType {
                limits: table.limits(),
                elem: RefType::ExternRef,
            },
        }
    }

    /// Returns the element at the given index in this [`Table`].
    ///
    /// # Errors
    ///
    /// - If the access is out of bounds.
    pub fn get(self, store: &Store, idx: u32) -> Option<Ref> {
        match self.0.as_ref(store) {
            TableEntity::FuncRef(table) => table.get(idx).map(Into::into),
            TableEntity::ExternRef(table) => table.get(idx).map(Into::into),
        }
    }

    /// An unguarded version of [`Table::set`].
    pub fn set(
        self,
        store: &mut Store,
        idx: u32,
        val: Ref,
    ) -> Result<(), TableError> {
        match (self.0.as_mut(store), val) {
            (TableEntity::FuncRef(table), Ref::FuncRef(val)) => table.set(idx, val),
            (TableEntity::ExternRef(table), Ref::ExternRef(val)) => table.set(idx, val),
            _ => Err(TableError::ElemTypeMismatch),
        }
    }

    /// Returns the size of this [`Table`] in number of elements.
    pub fn size(&self, store: &Store) -> u32 {
        match self.0.as_ref(store) {
            TableEntity::FuncRef(table) => table.size(),
            TableEntity::ExternRef(table) => table.size(),
        }
    }

    /// Grows this [`Table`] by the given number of elements with the given initialization [`Ref`].
    ///
    /// Returns the previous size of this [`Table`] in number of elements.
    ///
    /// # Errors
    ///
    /// - If the [`RefType`] of the given initialization [`Ref`] does not match the [`RefType`] of
    ///   the elements in this [`Table`].
    /// - If this [`Table`] failed to grow.
    ///
    /// # Panics
    ///
    /// - If the given initialization [`Ref`] is not owned by the given [`Store`].
    pub fn grow(self, store: &mut Store, val: Ref, count: u32) -> Result<(), TableError> {
        unsafe { self.grow_unguarded(store, val.to_unguarded(store.id()), count) }
    }

    /// An unguarded version of [`Table::grow`].
    unsafe fn grow_unguarded(
        self,
        store: &mut Store,
        val: UnguardedRef,
        count: u32,
    ) -> Result<(), TableError> {
        match (self.0.as_mut(store), val) {
            (TableEntity::FuncRef(table), UnguardedRef::FuncRef(val)) => table.grow_unguarded(val, count),
            (TableEntity::ExternRef(table), UnguardedRef::ExternRef(val)) => table.grow_unguarded(val, count),
            _ => Err(TableError::ElemTypeMismatch),
        }
        .map(|_| ())
    }

    pub(crate) fn init(
        self,
        store: &mut Store,
        dst_idx: u32,
        src_elem: Elem,
        src_idx: u32,
        count: u32,
    ) -> Result<(), Trap> {
        let (dst_table, src_elem) = HandlePair(self.0, src_elem.0).as_mut_pair(store);
        match (dst_table, src_elem) {
            (TableEntity::FuncRef(table), ElemEntity::FuncRef(src_elem)) => {
                table.init(dst_idx, src_elem, src_idx, count)
            }
            (TableEntity::ExternRef(table), ElemEntity::ExternRef(src_elem)) => {
                table.init(dst_idx, src_elem, src_idx, count)
            }
            _ => panic!(),
        }
    }
}

impl Guarded for Table {
    type Unguarded = UnguardedTable;
    type Guard = StoreGuard;

    unsafe fn from_unguarded(unguarded: UnguardedTable, guard: Self::Guard) -> Self {
        Self(Handle::from_unguarded(unguarded, guard))
    }

    fn to_unguarded(self, guard: Self::Guard) -> Self::Unguarded {
        self.0.to_unguarded(guard)
    }
}

/// An unguarded version of [`Table`].
pub(crate) type UnguardedTable = UnguardedHandle<TableEntity>;

/// The type of a [`Table`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct TableType {
    /// The [`Limits`] of the [`Table`].
    pub limits: Limits,
    /// The [`RefType`] of the elements in the [`Table`].
    pub elem: RefType,
}

impl TableType {
    /// Returns `true` if this [`TableType`] is valid.
    ///
    /// A [`TableType`] is valid if its [`Limits`] are valid within range `u32::MAX`.
    pub fn is_valid(self) -> bool {
        if !self.limits.is_valid(u32::MAX) {
            return false;
        }
        true
    }

    /// Returns `true` if this [`TableType`] is a subtype of the given [`TableType`].
    ///
    /// A [`TableType`] is a subtype of another [`TableType`] if its [`Limits`] are a sublimit of
    /// the other's and the [`RefType`] of its elements is the same as the other's.
    pub fn is_subtype_of(self, other: Self) -> bool {
        if !self.limits.is_sublimit_of(other.limits) {
            return false;
        }
        if self.elem != other.elem {
            return false;
        }
        true
    }
}

impl Decode for TableType {
    fn decode(decoder: &mut Decoder<'_>) -> Result<Self, DecodeError> {
        let elem = decoder.decode()?;
        let limits = decoder.decode()?;
        Ok(Self { limits, elem })
    }
}

/// An error that can occur when operating on a [`Table`].
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum TableError {
    AccessOutOfBounds,
    ElemTypeMismatch,
    FailedToGrow,
}

impl fmt::Display for TableError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AccessOutOfBounds => write!(f, "table access out of bounds"),
            Self::ElemTypeMismatch => write!(f, "table element type mismatch"),
            Self::FailedToGrow => write!(f, "table failed to grow"),
        }
    }
}

impl Error for TableError {}

/// The representation of a [`Table`] in a [`Store`].
#[derive(Debug)]
pub(crate) enum TableEntity {
    FuncRef(TypedTableEntity<FuncRef>),
    ExternRef(TypedTableEntity<ExternRef>),
}

impl TableEntity {
    /// Returns a reference to the inner value of this [`TableEntity`] if it is a
    /// [`TableEntityT<T>`].
    pub(crate) fn downcast_ref<T>(&self) -> Option<&TypedTableEntity<T>>
    where
        T: Guarded,
        TypedTableEntity<T>: DowncastRef<Self>,
    {
        TypedTableEntity::downcast_ref(self)
    }

    /// Returns a mutable reference to the inner value of this [`TableEntity`] if it is a
    /// [`TableEntityT<T>`].
    pub(crate) fn downcast_mut<T>(&mut self) -> Option<&mut TypedTableEntity<T>>
    where
        T: Guarded,
        TypedTableEntity<T>: DowncastMut<Self>,
    {
        TypedTableEntity::downcast_mut(self)
    }
}

/// A typed [`TableEntity`].
#[derive(Debug)]
pub(crate) struct TypedTableEntity<T>
where 
    T: Guarded
{
    max: Option<u32>,
    elems: Vec<T::Unguarded>,
    guard: T::Guard,
}

impl<T> TypedTableEntity<T>
where
    T: Guarded,
{
    fn new(limits: Limits, val: T, guard: T::Guard) -> Self {
        let val = val.to_unguarded(guard);
        unsafe { Self::new_unguarded(limits, val, guard) }
    }

    unsafe fn new_unguarded(limits: Limits, val: T::Unguarded, guard: T::Guard) -> Self {
        let min = limits.min as usize;
        Self {
            max: limits.max,
            elems: vec![val; min],
            guard,
        }
    }

    /// Returns the [`Limits`] of this [`TableEntity`].
    fn limits(&self) -> Limits {
        Limits {
            min: u32::try_from(self.elems.len()).unwrap(),
            max: self.max,
        }
    }

    /// Returns the element at the given index in this [`TableEntity`].
    ///
    /// # Errors
    ///
    /// If the access is out of bounds.
    pub(crate) fn get(&self, idx: u32) -> Option<T> {
        let val = self.get_unguarded(idx)?;
        Some(unsafe { T::from_unguarded(val, self.guard) })
    }

    pub(crate) fn get_unguarded(&self, idx: u32) -> Option<T::Unguarded> {
        let idx = idx as usize;
        let elem = self.elems.get(idx)?;
        Some(*elem)
    }

    /// Sets the element at the given index in this [`TableEntity`] to the given value.
    ///
    /// # Errors
    ///
    /// If the access is out of bounds.
    pub(crate) fn set(&mut self, index: u32, val: T) -> Result<(), TableError> {
        let val = val.to_unguarded(self.guard);
        unsafe { self.set_unguarded(index, val) }
    }

    pub(crate) unsafe fn set_unguarded(&mut self, idx: u32, val: T::Unguarded) -> Result<(), TableError> {
        let idx = idx as usize;
        let elem = self
            .elems
            .get_mut(idx)
            .ok_or(TableError::AccessOutOfBounds)?;
        *elem = val;
        Ok(())
    }

    /// Returns the size of this [`TableEntity`] in number of elements.
    pub(crate) fn size(&self) -> u32 {
        self.elems.len() as u32
    }

    /// Grows this [`TableEntity`] by the given number of elements with the given initialization
    /// value.
    ///
    /// Returns the previous size of this [`TableEntity`] in number of elements.
    ///
    /// # Errors
    ///
    /// If this [`TableEntity`] failed to grow.
    pub(crate) unsafe fn grow_unguarded(&mut self, val: T::Unguarded, count: u32) -> Result<u32, TableError> {
        if count > self.max.unwrap_or(u32::MAX) - self.size() {
            return Err(TableError::FailedToGrow)?;
        }
        let count = count as usize;
        let size = self.size();
        self.elems.resize(self.elems.len() + count, val);
        Ok(size)
    }

    pub(crate) unsafe fn fill_unguarded(&mut self, idx: u32, val: T::Unguarded, count: u32) -> Result<(), Trap> {
        let elems = self
            .elems
            .get_mut(idx as usize..)
            .and_then(|elems| elems.get_mut(..count as usize))
            .ok_or(Trap::TableAccessOutOfBounds)?;
        elems.fill(val);
        Ok(())
    }

    pub(crate) fn copy(
        &mut self,
        dst_idx: u32,
        src_table: &TypedTableEntity<T>,
        src_idx: u32,
        count: u32,
    ) -> Result<(), Trap> {
        let dst_idx = dst_idx as usize;
        let src_idx = src_idx as usize;
        let count = count as usize;
        let dst_elems = self
            .elems
            .get_mut(dst_idx..)
            .and_then(|elems| elems.get_mut(..count))
            .ok_or(Trap::TableAccessOutOfBounds)?;
        let src_elems = src_table
            .elems
            .get(src_idx..)
            .and_then(|elems| elems.get(..count))
            .ok_or(Trap::TableAccessOutOfBounds)?;
        dst_elems.copy_from_slice(src_elems);
        Ok(())
    }

    pub(crate) fn copy_within(
        &mut self,
        dst_idx: u32,
        src_idx: u32,
        count: u32,
    ) -> Result<(), Trap> {
        let dst_idx = dst_idx as usize;
        let src_idx = src_idx as usize;
        let count = count as usize;
        if count > self.elems.len()
            || dst_idx > self.elems.len() - count
            || src_idx > self.elems.len() - count
        {
            return Err(Trap::TableAccessOutOfBounds)?;
        }
        self.elems.copy_within(src_idx..src_idx + count, dst_idx);
        Ok(())
    }

    pub(crate) fn init(
        &mut self,
        dst_idx: u32,
        src_elem: &TypedElemEntity<T>,
        src_idx: u32,
        count: u32,
    ) -> Result<(), Trap> {
        let dst_idx = dst_idx as usize;
        let src_idx = src_idx as usize;
        let count = count as usize;
        let dst_elems = self
            .elems
            .get_mut(dst_idx..)
            .and_then(|elems| elems.get_mut(..count))
            .ok_or(Trap::TableAccessOutOfBounds)?;
        let src_elems = src_elem
            .elems()
            .get(src_idx..)
            .and_then(|elems| elems.get(..count))
            .ok_or(Trap::TableAccessOutOfBounds)?;
        dst_elems.copy_from_slice(src_elems);
        Ok(())
    }
}

impl DowncastRef<TableEntity> for TypedTableEntity<FuncRef> {
    fn downcast_ref(table: &TableEntity) -> Option<&Self> {
        if let TableEntity::FuncRef(table) = table {
            Some(table)
        } else {
            None
        }
    }
}

impl DowncastMut<TableEntity> for TypedTableEntity<FuncRef> {
    fn downcast_mut(table: &mut TableEntity) -> Option<&mut Self> {
        if let TableEntity::FuncRef(table) = table {
            Some(table)
        } else {
            None
        }
    }
}

impl DowncastRef<TableEntity> for TypedTableEntity<ExternRef> {
    fn downcast_ref(table: &TableEntity) -> Option<&Self> {
        if let TableEntity::ExternRef(table) = table {
            Some(table)
        } else {
            None
        }
    }
}

impl DowncastMut<TableEntity> for TypedTableEntity<ExternRef> {
    fn downcast_mut(table: &mut TableEntity) -> Option<&mut Self> {
        if let TableEntity::ExternRef(table) = table {
            Some(table)
        } else {
            None
        }
    }
}
