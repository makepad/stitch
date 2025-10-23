pub(crate) trait Guarded: Copy {
    type Unguarded: Copy;
    type Guard: Copy;

    unsafe fn from_unguarded(unguarded: Self::Unguarded, guard: Self::Guard) -> Self;

    fn to_unguarded(self, guard: Self::Guard) -> Self::Unguarded;
}