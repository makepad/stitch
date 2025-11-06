use {
    crate::{
        decode::{Decode, DecodeError, Decoder},
        downcast::{DowncastMut, DowncastRef},
        elem::{Elem, ElemEntity, TypedElemEntity},
        guarded::Guarded,
        ref_::{ExternRef, FuncRef, Ref, RefType, RefTypeOf},
        store::{Handle, HandlePair, Store, StoreGuard, UnguardedHandle},
        trap::Trap,
    },
    std::{error::Error, fmt},
};

/// A WebAssembly table.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[repr(transparent)]
pub struct Table(pub(crate) Handle<TableEntity>);

impl Table {
    /// Creates a new [`Table`] with the following parameters:
    ///
    /// * `store` - the [`Store`] in which to create the new [`Table`].
    /// * `ty` - the [`TableType`] of the new [`Table`].
    /// * `init_val` - the initial value of the new [`Table`]'s elements.
    ///
    /// # Errors
    ///
    /// If the [`RefType`] of `init_val` does not match that of the new [`Table`].
    ///
    /// # Panics
    ///
    /// * If `ty` is invalid.
    /// * If `init_val` does not originate from `store`.
    pub fn new(store: &mut Store, ty: TableType, init_val: Ref) -> Result<Self, TableError> {
        assert!(ty.is_valid());
        if init_val.type_() != ty.element() {
            return Err(TableError::TypeMismatch);
        }
        let table = match init_val {
            Ref::FuncRef(init_val) => TableEntity::FuncRef(TypedTableEntity::new(
                init_val,
                ty.minimum(),
                ty.maximum(),
                store.guard(),
            )),
            Ref::ExternRef(init_val) => TableEntity::ExternRef(TypedTableEntity::new(
                init_val,
                ty.minimum(),
                ty.maximum(),
                store.guard(),
            )),
        };
        Ok(Self(store.insert_table(table)))
    }

    /// Returns the [`TableType`] of `self` in the given `store`.
    ///
    /// # Panics
    ///
    /// If `self` does not originate from `store`.
    pub fn ty(self, store: &Store) -> TableType {
        match self.0.as_ref(store) {
            TableEntity::FuncRef(table) => table.ty(),
            TableEntity::ExternRef(table) => table.ty(),
        }
    }

    /// Returns the current value of `self`s `idx`-th element in the given `store.`
    ///
    /// # Errors
    ///
    /// If `idx` is out of bounds.
    ///
    /// # Panics
    ///
    /// If `self` does not originate from [`Store`].
    pub fn get(self, store: &Store, idx: u32) -> Option<Ref> {
        match self.0.as_ref(store) {
            TableEntity::FuncRef(table) => table.get(idx).map(Into::into),
            TableEntity::ExternRef(table) => table.get(idx).map(Into::into),
        }
    }

    /// Sets the value of `self`'s `idx`-th element to `new_val` in the given `store`.
    ///
    /// # Errors
    ///
    /// * If `idx` is out of bounds.
    /// * If the [`RefType`] of `new_val` does not match that of `self`s elements.
    ///
    /// # Panics
    ///
    /// * If `self` does not originate from `store`.
    /// * If `new_val` does not originate from `store`.
    pub fn set(self, store: &mut Store, idx: u32, new_val: Ref) -> Result<(), TableError> {
        match (self.0.as_mut(store), new_val) {
            (TableEntity::FuncRef(table), Ref::FuncRef(val)) => table.set(idx, val),
            (TableEntity::ExternRef(table), Ref::ExternRef(val)) => table.set(idx, val),
            _ => Err(TableError::TypeMismatch),
        }
    }

    /// Returns `self`'s current size.
    pub fn size(&self, store: &Store) -> u32 {
        match self.0.as_ref(store) {
            TableEntity::FuncRef(table) => table.size(),
            TableEntity::ExternRef(table) => table.size(),
        }
    }

    /// Grows `self` by `num` elements with initial value `init_val`.
    ///
    /// Returns the previous size of this `self`.
    ///
    /// # Errors
    ///
    /// - If the [`RefType`] of `init_val` does not match that of `self`'s elements.
    /// - If `self` failed to grow.
    ///
    /// # Panics
    ///
    /// - If `init_val` does not originate from `store`.
    pub fn grow(self, store: &mut Store, val: Ref, num: u32) -> Result<(), TableError> {
        match (self.0.as_mut(store), val) {
            (TableEntity::FuncRef(table), Ref::FuncRef(val)) => table.grow(val, num),
            (TableEntity::ExternRef(table), Ref::ExternRef(val)) => table.grow(val, num),
            _ => Err(TableError::TypeMismatch),
        }
        .map(|_| ())
    }

    pub(crate) fn init(
        self,
        store: &mut Store,
        dst_idx: u32,
        src_elem: Elem,
        src_idx: u32,
        num: u32,
    ) -> Result<(), Trap> {
        let (dst_table, src_elem) = HandlePair(self.0, src_elem.0).as_mut_pair(store);
        match (dst_table, src_elem) {
            (TableEntity::FuncRef(table), ElemEntity::FuncRef(src_elem)) => {
                table.init(dst_idx, src_elem, src_idx, num)
            }
            (TableEntity::ExternRef(table), ElemEntity::ExternRef(src_elem)) => {
                table.init(dst_idx, src_elem, src_idx, num)
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
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct TableType {
    pub elem: RefType,
    pub min: u32,
    pub max: Option<u32>,
}

impl TableType {
    /// Creates a new [`TableType`] with the following parameters:
    ///
    /// * `elem` - The [`RefType`] of the [`Table`]'s elements.
    /// * `min` - The [`Table`]'s minimum size.
    /// * `max` - The [`Table`]'s maximum size, if any.
    pub fn new(elem: RefType, min: u32, max: Option<u32>) -> Self {
        Self { elem, min, max }
    }

    /// Returns the [`RefType`] of the [`Table`]'s elements.
    pub fn element(&self) -> RefType {
        self.elem
    }

    /// Returns the [`Table`]'s minimum size.
    pub fn minimum(&self) -> u32 {
        self.min
    }

    /// Returns the [`Table`]'s maximum size, if any.
    pub fn maximum(&self) -> Option<u32> {
        self.max
    }

    pub(crate) fn is_valid(self) -> bool {
        if self.min > u32::MAX {
            return false;
        }
        if let Some(max) = self.max {
            if self.min > max {
                return false;
            }
        }
        true
    }

    pub(crate) fn is_subtype_of(self, other: Self) -> bool {
        if self.elem != other.elem {
            return false;
        }
        if self.min < other.min {
            return false;
        }
        match (self.max, other.max) {
            (None, Some(_)) => return false,
            (Some(max), Some(other_max)) if max > other_max => return false,
            _ => {}
        }
        true
    }
}

impl Decode for TableType {
    fn decode(decoder: &mut Decoder<'_>) -> Result<Self, DecodeError> {
        let elem = decoder.decode()?;
        let has_max = match decoder.read_byte()? {
            0x00 => false,
            0x01 => true,
            _ => return Err(DecodeError::new("invalid table type")),
        };
        Ok(Self {
            elem,
            min: decoder.decode()?,
            max: if has_max {
                Some(decoder.decode()?)
            } else {
                None
            },
        })
    }
}

/// An error returned by operations on a [`Table`].
#[derive(Debug)]
#[non_exhaustive]
pub enum TableError {
    /// The index is out of bounds.
    IdxOutOfBounds,
    /// The [`RefType`] does not match that of the [`Table`]'s elements.
    TypeMismatch,
    /// The [`Table`] failed to grow.
    FailedToGrow,
}

impl fmt::Display for TableError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IdxOutOfBounds => write!(f, "index out of bounds"),
            Self::TypeMismatch => write!(f, "type does not match that of table's elements"),
            Self::FailedToGrow => write!(f, "table failed to grow"),
        }
    }
}

impl Error for TableError {}

#[derive(Debug)]
pub(crate) enum TableEntity {
    FuncRef(TypedTableEntity<FuncRef>),
    ExternRef(TypedTableEntity<ExternRef>),
}

impl TableEntity {
    pub(crate) fn downcast_ref<T>(&self) -> Option<&TypedTableEntity<T>>
    where
        T: Guarded,
        TypedTableEntity<T>: DowncastRef<Self>,
    {
        TypedTableEntity::downcast_ref(self)
    }

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
    T: Guarded,
{
    elems: Vec<T::Unguarded>,
    max: Option<u32>,
    guard: T::Guard,
}

impl<T> TypedTableEntity<T>
where
    T: Guarded,
{
    fn new(val: T, min: u32, max: Option<u32>, guard: T::Guard) -> Self {
        let val = val.to_unguarded(guard);
        unsafe { Self::new_unguarded(val, min, max, guard) }
    }

    unsafe fn new_unguarded(
        val: T::Unguarded,
        min: u32,
        max: Option<u32>,
        guard: T::Guard,
    ) -> Self {
        Self {
            elems: vec![val; min as usize],
            max,
            guard,
        }
    }

    fn get(&self, idx: u32) -> Option<T> {
        let val = self.get_unguarded(idx)?;
        Some(unsafe { T::from_unguarded(val, self.guard) })
    }

    pub(crate) fn get_unguarded(&self, idx: u32) -> Option<T::Unguarded> {
        let idx = idx as usize;
        let elem = self.elems.get(idx)?;
        Some(*elem)
    }

    fn set(&mut self, idx: u32, val: T) -> Result<(), TableError> {
        let val = val.to_unguarded(self.guard);
        unsafe { self.set_unguarded(idx, val) }
    }

    pub(crate) unsafe fn set_unguarded(
        &mut self,
        idx: u32,
        val: T::Unguarded,
    ) -> Result<(), TableError> {
        let elem = self
            .elems
            .get_mut(idx as usize)
            .ok_or(TableError::IdxOutOfBounds)?;
        *elem = val;
        Ok(())
    }

    pub(crate) fn size(&self) -> u32 {
        self.elems.len() as u32
    }

    fn grow(&mut self, val: T, num: u32) -> Result<u32, TableError> {
        let val = val.to_unguarded(self.guard);
        unsafe { self.grow_unguarded(val, num) }
    }

    pub(crate) unsafe fn grow_unguarded(
        &mut self,
        val: T::Unguarded,
        num: u32,
    ) -> Result<u32, TableError> {
        if num > self.max.unwrap_or(u32::MAX) - self.size() {
            return Err(TableError::FailedToGrow)?;
        }
        let num = num as usize;
        let size = self.size();
        self.elems.resize(self.elems.len() + num, val);
        Ok(size)
    }

    pub(crate) unsafe fn fill_unguarded(
        &mut self,
        idx: u32,
        val: T::Unguarded,
        num: u32,
    ) -> Result<(), Trap> {
        let elems = self
            .elems
            .get_mut(idx as usize..)
            .and_then(|elems| elems.get_mut(..num as usize))
            .ok_or(Trap::TableAccessOutOfBounds)?;
        elems.fill(val);
        Ok(())
    }

    pub(crate) fn copy(
        &mut self,
        dst_idx: u32,
        src_table: &TypedTableEntity<T>,
        src_idx: u32,
        num: u32,
    ) -> Result<(), Trap> {
        let dst_elems = self
            .elems
            .get_mut(dst_idx as usize..)
            .and_then(|elems| elems.get_mut(..num as usize))
            .ok_or(Trap::TableAccessOutOfBounds)?;
        let src_elems = src_table
            .elems
            .get(src_idx as usize..)
            .and_then(|elems| elems.get(..num as usize))
            .ok_or(Trap::TableAccessOutOfBounds)?;
        dst_elems.copy_from_slice(src_elems);
        Ok(())
    }

    pub(crate) fn copy_within(&mut self, dst_idx: u32, src_idx: u32, num: u32) -> Result<(), Trap> {
        if num > self.size() || dst_idx > self.size() - num || src_idx > self.size() - num {
            return Err(Trap::TableAccessOutOfBounds)?;
        }
        self.elems.copy_within(
            src_idx as usize..src_idx as usize + num as usize,
            dst_idx as usize,
        );
        Ok(())
    }

    pub(crate) fn init(
        &mut self,
        dst_idx: u32,
        src_elem: &TypedElemEntity<T>,
        src_idx: u32,
        num: u32,
    ) -> Result<(), Trap> {
        let dst_elems = self
            .elems
            .get_mut(dst_idx as usize..)
            .and_then(|elems| elems.get_mut(..num as usize))
            .ok_or(Trap::TableAccessOutOfBounds)?;
        let src_elems = src_elem
            .elems()
            .get(src_idx as usize..)
            .and_then(|elems| elems.get(..num as usize))
            .ok_or(Trap::TableAccessOutOfBounds)?;
        dst_elems.copy_from_slice(src_elems);
        Ok(())
    }
}

impl<T> TypedTableEntity<T>
where
    T: Guarded + RefTypeOf,
{
    fn ty(&self) -> TableType {
        TableType::new(T::ref_type_of(), self.size(), self.max)
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
