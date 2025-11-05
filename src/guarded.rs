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