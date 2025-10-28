use {
    crate::slice,
    std::{ops::RangeBounds, sync::Arc},
};

pub(crate) trait Guarded: Copy {
    type Unguarded: Copy;
    type Guard: Copy;

    unsafe fn from_unguarded(unguarded: Self::Unguarded, guard: Self::Guard) -> Self;

    fn to_unguarded(self, guard: Self::Guard) -> Self::Unguarded;
}

impl<T> Guarded for Option<T>
where
    T: Guarded,
{
    type Unguarded = Option<T::Unguarded>;
    type Guard = T::Guard;

    unsafe fn from_unguarded(unguarded: Self::Unguarded, guard: Self::Guard) -> Self {
        unguarded.map(|unguarded| T::from_unguarded(unguarded, guard))
    }

    fn to_unguarded(self, guard: Self::Guard) -> Self::Unguarded {
        self.map(|guarded| guarded.to_unguarded(guard))
    }
}

pub(crate) trait GuardedSlice<'a, T>: Sized
where 
    T: Guarded,
    T::Unguarded: 'a
{
    fn as_unguarded_slice(self) -> &'a [T::Unguarded];

    fn guard(&self) -> T::Guard;

    fn len(self) -> usize {
        self.as_unguarded_slice().len()
    }

    fn get(self, index: usize) -> Option<T> {
        let guard = self.guard();
        let unguarded = self.get_unguarded(index)?;
        unsafe { Some(T::from_unguarded(unguarded, guard)) }
    }

    fn get_unguarded(self, index: usize) -> Option<T::Unguarded> {
        self.as_unguarded_slice().get(index).copied()
    }

    fn slice<R>(self, range: R) -> Option<GuardedSliceRef<'a, T>>
    where 
        R: RangeBounds<usize>
    {
        let guard = self.guard();
        let unguarded = self.slice_unguarded(range)?;
        Some(GuardedSliceRef {
            unguarded,
            guard,
        })
    }

    fn slice_unguarded<R>(self, range: R) -> Option<&'a [T::Unguarded]>
    where 
        R: RangeBounds<usize>
    {
        let slice = self.as_unguarded_slice();
        let range = slice::try_range(range, ..slice.len())?;
        Some(&slice[range])
    }
}

pub(crate) trait GuardedSliceMut<'a, T>: GuardedSlice<'a, T>
where 
    T: Guarded,
    T::Unguarded: 'a,
{
    fn as_unguarded_slice_mut(self) -> &'a mut [T::Unguarded];

    fn set(self, index: usize, guarded: T) {
        let unguarded = guarded.to_unguarded(self.guard());
        unsafe { self.set_unguarded(index, unguarded) }
    }

    unsafe fn set_unguarded(self, index: usize, unguarded: T::Unguarded) {
        self.as_unguarded_slice_mut()[index] = unguarded;
    }

    fn slice_mut<R>(self, range: R) -> Option<GuardedSliceRefMut<'a, T>>
    where 
        R: RangeBounds<usize>
    {
        let guard = self.guard();
        let unguarded = self.slice_mut_unguarded(range)?;
        Some(GuardedSliceRefMut {
            unguarded,
            guard,
        })
    }

    fn slice_mut_unguarded<R>(self, range: R) -> Option<&'a mut [T::Unguarded]>
    where 
        R: RangeBounds<usize>
    {
        let slice = self.as_unguarded_slice_mut();
        let range = slice::try_range(range, ..slice.len())?;
        Some(&mut slice[range])
    }
}

pub(crate) struct GuardedVec<T>
where 
    T: Guarded
{
    unguarded: Vec<T::Unguarded>,
    guard: T::Guard,
}

impl<T> GuardedVec<T>
where 
    T: Guarded
{
    pub(crate) fn new(guard: T::Guard) -> Self {
        Self {
            unguarded: Vec::new(),
            guard,
        }
    }

    pub(crate) fn as_guarded_slice(&self) -> GuardedSliceRef<'_, T> {
        GuardedSliceRef {
            unguarded: &self.unguarded,
            guard: self.guard
        }
    }

    pub(crate) fn as_guarded_slice_mut(&mut self) -> GuardedSliceRefMut<'_, T> {
        GuardedSliceRefMut {
            unguarded: &mut self.unguarded,
            guard: self.guard,
        }
    }

    pub(crate) fn push(&mut self, guarded: T) {
        let unguarded = guarded.to_unguarded(self.guard);
        unsafe { self.push_unguarded(unguarded) }
    }

    pub(crate) unsafe fn push_unguarded(&mut self, unguarded: T::Unguarded) {
        self.unguarded.push(unguarded);
    }
}

impl<'a, T> GuardedSlice<'a, T> for &'a GuardedVec<T>
where 
    T: Guarded
{
    fn as_unguarded_slice(self) -> &'a [T::Unguarded] {
        &self.unguarded
    }

    fn guard(&self) -> T::Guard {
        self.guard
    }
}

impl<'a, T> GuardedSlice<'a, T> for &'a mut GuardedVec<T>
where 
    T: Guarded
{
    fn as_unguarded_slice(self) -> &'a [T::Unguarded] {
        &self.unguarded
    }

    fn guard(&self) -> T::Guard {
        self.guard
    }
}

impl<'a, T> GuardedSliceMut<'a, T> for &'a mut GuardedVec<T>
where 
    T: Guarded
{
    fn as_unguarded_slice_mut(self) -> &'a mut [T::Unguarded] {
        &mut self.unguarded
    }
}

pub(crate) struct GuardedBoxSlice<T>
where
    T: Guarded
{
    unguarded: Box<[T::Unguarded]>,
    guard: T::Guard,
}

impl<T> GuardedBoxSlice<T>
where 
    T: Guarded
{
    pub(crate) fn as_guarded_slice(&self) -> GuardedSliceRef<'_, T> {
        GuardedSliceRef {
            unguarded: &self.unguarded,
            guard: self.guard
        }
    }

    pub(crate) fn as_guarded_slice_mut(&mut self) -> GuardedSliceRefMut<'_, T> {
        GuardedSliceRefMut {
            unguarded: &mut self.unguarded,
            guard: self.guard,
        }
    }
}

impl<T> From<GuardedVec<T>> for GuardedBoxSlice<T>
where 
    T: Guarded
{
    fn from(vec: GuardedVec<T>) -> Self {
        Self {
            unguarded: vec.unguarded.into(),
            guard: vec.guard,
        }
    }
}

impl<'a, T> GuardedSlice<'a, T> for &'a GuardedBoxSlice<T>
where 
    T: Guarded
{
    fn as_unguarded_slice(self) -> &'a [T::Unguarded] {
        &self.unguarded
    }

    fn guard(&self) -> T::Guard {
        self.guard
    }
}

impl<'a, T> GuardedSlice<'a, T> for &'a mut GuardedBoxSlice<T>
where 
    T: Guarded
{
    fn as_unguarded_slice(self) -> &'a [T::Unguarded] {
        &self.unguarded
    }

    fn guard(&self) -> T::Guard {
        self.guard
    }
}

impl<'a, T> GuardedSliceMut<'a, T> for &'a mut GuardedBoxSlice<T>
where 
    T: Guarded
{
    fn as_unguarded_slice_mut(self) -> &'a mut [T::Unguarded] {
        &mut self.unguarded
    }
}

pub(crate) struct GuardedArcSlice<T>
where
    T: Guarded
{
    unguarded: Arc<[T::Unguarded]>,
    guard: T::Guard,
}

impl<T> GuardedArcSlice<T>
where 
    T: Guarded
{
    pub(crate) fn as_guarded_slice(&self) -> GuardedSliceRef<'_, T> {
        GuardedSliceRef {
            unguarded: &self.unguarded,
            guard: self.guard
        }
    }
}

impl<T> From<GuardedVec<T>> for GuardedArcSlice<T>
where 
    T: Guarded
{
    fn from(vec: GuardedVec<T>) -> Self {
        Self {
            unguarded: vec.unguarded.into(),
            guard: vec.guard,
        }
    }
}

impl<'a, T> GuardedSlice<'a, T> for &'a GuardedArcSlice<T>
where 
    T: Guarded
{
    fn as_unguarded_slice(self) -> &'a [T::Unguarded] {
        &self.unguarded
    }

    fn guard(&self) -> T::Guard {
        self.guard
    }
}

#[derive(Clone, Copy)]
pub(crate) struct GuardedSliceRef<'a, T>
where 
    T: Guarded
{
    unguarded: &'a [T::Unguarded],
    guard: T::Guard,
}

impl<'a, T> GuardedSlice<'a, T> for GuardedSliceRef<'a, T>
where 
    T: Guarded
{
    fn as_unguarded_slice(self) -> &'a [T::Unguarded] {
        self.unguarded
    }

    fn guard(&self) -> T::Guard {
        self.guard
    }
}

pub(crate) struct GuardedSliceRefMut<'a, T>
where 
    T: Guarded
{
    unguarded: &'a mut [T::Unguarded],
    guard: T::Guard,
}

impl<'a, T> GuardedSlice<'a, T> for GuardedSliceRefMut<'a, T>
where 
    T: Guarded
{
    fn as_unguarded_slice(self) -> &'a [T::Unguarded] {
        self.unguarded
    }

    fn guard(&self) -> T::Guard {
        self.guard
    }
}

impl<'a, T> GuardedSliceMut<'a, T> for GuardedSliceRefMut<'a, T>
where 
    T: Guarded
{
    fn as_unguarded_slice_mut(self) -> &'a mut [T::Unguarded] {
        self.unguarded
    }
}