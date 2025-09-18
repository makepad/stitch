use std::{
    borrow::Borrow,
    collections::{btree_map::Iter as BTreeMapIter, BTreeMap},
    fmt,
    slice::Iter as SliceIter,
};

pub(crate) trait Guarded {
    type Unguarded: Copy;
    type Guard: Copy;

    unsafe fn from_unguarded(unguarded: Self::Unguarded, guard: Self::Guard) -> Self;

    fn to_unguarded(self, guard: Self::Guard) -> Self::Unguarded;
}

pub(crate) struct GuardedVec<T>
where
    T: Guarded,
{
    unguarded: Vec<T::Unguarded>,
    guard: T::Guard,
}

impl<T> GuardedVec<T>
where
    T: Guarded,
{
    pub(crate) fn new(guard: T::Guard) -> Self {
        Self {
            unguarded: Vec::new(),
            guard,
        }
    }

    pub(crate) fn get(&self, index: usize) -> Option<T> {
        self.unguarded
            .get(index)
            .copied()
            .map(|unguarded| unsafe { T::from_unguarded(unguarded, self.guard) })
    }

    pub(crate) fn iter(&self) -> GuardedSliceIter<'_, T> {
        GuardedSliceIter {
            unguarded: self.unguarded.iter(),
            guard: self.guard,
        }
    }

    pub(crate) fn push(&mut self, val: T) {
        self.unguarded.push(val.to_unguarded(self.guard));
    }
}

impl<T> fmt::Debug for GuardedVec<T>
where
    T: fmt::Debug + Guarded,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_list().entries(self).finish()
    }
}

impl<'a, T> IntoIterator for &'a GuardedVec<T>
where
    T: Guarded,
{
    type Item = T;
    type IntoIter = GuardedSliceIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

pub(crate) struct GuardedBoxSlice<T>
where
    T: Guarded,
{
    unguarded: Box<[T::Unguarded]>,
    guard: T::Guard,
}

impl<T> GuardedBoxSlice<T>
where
    T: Guarded,
{
    pub(crate) fn get(&self, index: usize) -> Option<T> {
        self.unguarded
            .get(index)
            .copied()
            .map(|unguarded| unsafe { T::from_unguarded(unguarded, self.guard) })
    }

    pub(crate) fn iter(&self) -> GuardedSliceIter<'_, T> {
        GuardedSliceIter {
            unguarded: self.unguarded.iter(),
            guard: self.guard,
        }
    }
}

impl<T> fmt::Debug for GuardedBoxSlice<T>
where
    T: fmt::Debug + Guarded,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_list().entries(self).finish()
    }
}

impl<T> From<GuardedVec<T>> for GuardedBoxSlice<T>
where
    T: Guarded,
{
    fn from(vec: GuardedVec<T>) -> Self {
        Self {
            unguarded: vec.unguarded.into(),
            guard: vec.guard,
        }
    }
}

impl<'a, T> IntoIterator for &'a GuardedBoxSlice<T>
where
    T: Guarded,
{
    type Item = T;
    type IntoIter = GuardedSliceIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

pub(crate) struct GuardedSliceIter<'a, T>
where
    T: Guarded,
{
    unguarded: SliceIter<'a, T::Unguarded>,
    guard: T::Guard,
}

impl<'a, T> Iterator for GuardedSliceIter<'a, T>
where
    T: Guarded,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.unguarded
            .next()
            .copied()
            .map(|unguarded| unsafe { T::from_unguarded(unguarded, self.guard) })
    }
}

pub(crate) struct GuardedBTreeMap<K, V>
where
    V: Guarded,
{
    unguarded: BTreeMap<K, V::Unguarded>,
    guard: V::Guard,
}

impl<K, V> GuardedBTreeMap<K, V>
where
    V: Guarded,
{
    pub(crate) fn new(guard: V::Guard) -> Self {
        Self {
            unguarded: BTreeMap::new(),
            guard,
        }
    }

    pub(crate) fn get<Q>(&self, key: &Q) -> Option<V>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        self.unguarded
            .get(key)
            .copied()
            .map(|unguarded| unsafe { V::from_unguarded(unguarded, self.guard) })
    }

    pub(crate) fn iter(&self) -> GuardedBTreeMapIter<'_, K, V>
    where
        K: Ord,
    {
        GuardedBTreeMapIter {
            unguarded: self.unguarded.iter(),
            guard: self.guard,
        }
    }

    pub(crate) fn insert(&mut self, key: K, val: V) -> Option<V>
    where
        K: Ord,
    {
        self.unguarded
            .insert(key, val.to_unguarded(self.guard))
            .map(|unguarded| unsafe { V::from_unguarded(unguarded, self.guard) })
    }
}

impl<K, V> fmt::Debug for GuardedBTreeMap<K, V>
where
    K: fmt::Debug + Ord,
    V: fmt::Debug + Guarded,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<'a, K, V> IntoIterator for &'a GuardedBTreeMap<K, V>
where
    K: Ord,
    V: Guarded,
{
    type Item = (&'a K, V);
    type IntoIter = GuardedBTreeMapIter<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

pub(crate) struct GuardedBTreeMapIter<'a, K, V>
where
    V: Guarded,
{
    unguarded: BTreeMapIter<'a, K, V::Unguarded>,
    guard: V::Guard,
}

impl<'a, K, V> Iterator for GuardedBTreeMapIter<'a, K, V>
where
    V: Guarded,
{
    type Item = (&'a K, V);

    fn next(&mut self) -> Option<Self::Item> {
        self.unguarded
            .next()
            .map(|(k, v)| (k, unsafe { V::from_unguarded(*v, self.guard) }))
    }
}
