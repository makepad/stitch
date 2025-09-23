pub(crate) trait ExtendingCast: Sized {
    fn extending_cast<T>(self) -> T
    where
        T: ExtendingCastFrom<Self>
    {
        T::extending_cast_from(self)
    }
}

macro_rules! impl_extending_cast {
    ($($T:ty),*) => {
        $(
            impl ExtendingCast for $T {}
        )*
    }
}

impl_extending_cast! { u8, i8, u16, i16, u32, i32, u64, i64 }

pub(crate) trait ExtendingCastFrom<Src> {
    fn extending_cast_from(src: Src) -> Self;
}

macro_rules! impl_extending_cast_from {
    ($($Src:ty => $($Dst:ty)*,)*) => {
        $(
            $(
                impl ExtendingCastFrom<$Src> for $Dst {
                    fn extending_cast_from(src: $Src) -> Self {
                        src as $Dst
                    }
                }
            )*
        )*
    }
}

impl_extending_cast_from! {
    i8 => i16 i32 i64,
    u8 => u16 u32 u64,
    i16 => i32 i64,
    u16 => u32 u64,
    i32 => i64,
    u32 => u64,
}

pub(crate) trait WrappingCast: Sized {
    fn wrapping_cast<T>(self) -> T
    where
        T: WrappingCastFrom<Self>
    {
        T::wrapping_cast_from(self)
    }
}

macro_rules! impl_wrapping_cast {
    ($($T:ty),*) => {
        $(
            impl WrappingCast for $T {}
        )*
    }
}

impl_wrapping_cast! { u8, i8, u16, i16, u32, i32, u64, i64 }

pub(crate) trait WrappingCastFrom<Src> {
    fn wrapping_cast_from(src: Src) -> Self;
}

macro_rules! impl_wrapping_cast_from {
    ($($($Src:ty)* => $Dst:ty),*) => {
        $(
            $(
                impl WrappingCastFrom<$Src> for $Dst {
                    fn wrapping_cast_from(src: $Src) -> Self {
                        src as $Dst
                    }
                }
            )*
        )*
    }
}

impl_wrapping_cast_from! {
    i16 i32 i64 => i8,
    u16 u32 u64 => u8,
    i32 i64 => i16,
    u32 u64 => u16,
    i64 => i32,
    u64 => u32
}