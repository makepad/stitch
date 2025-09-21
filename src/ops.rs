use {crate::trap::Trap, std::{marker::PhantomData, mem}};

pub(crate) trait UnOp<T> {
    type Output;

    fn un_op(x: T) -> Result<Self::Output, Trap>;
}

pub(crate) trait BinOp<T> {
    type Output;

    fn bin_op(x0: T, x1: T) -> Result<Self::Output, Trap>;
}

pub(crate) struct Eqz;
pub(crate) struct Ne;
pub(crate) struct Eq;
pub(crate) struct Lt;
pub(crate) struct Gt;
pub(crate) struct Le;
pub(crate) struct Ge;
pub(crate) struct Clz;
pub(crate) struct Ctz;
pub(crate) struct Popcnt;
pub(crate) struct Add;
pub(crate) struct Sub;
pub(crate) struct Mul;
pub(crate) struct Div;
pub(crate) struct Rem;
pub(crate) struct And;
pub(crate) struct Or;
pub(crate) struct Xor;
pub(crate) struct Shl;
pub(crate) struct Shr;
pub(crate) struct Rotl;
pub(crate) struct Rotr;
pub(crate) struct Abs;
pub(crate) struct Neg;
pub(crate) struct Ceil;
pub(crate) struct Floor;
pub(crate) struct Trunc;
pub(crate) struct Nearest;
pub(crate) struct Sqrt;
pub(crate) struct Min;
pub(crate) struct Max;
pub(crate) struct Copysign;

pub(crate) struct WrapTo<T>(PhantomData<T>);
pub(crate) struct ExtendTo<T>(PhantomData<T>);
pub(crate) struct TruncTo<T>(PhantomData<T>);
pub(crate) struct TruncSatTo<T>(PhantomData<T>);

pub(crate) struct ConvertTo<T>(PhantomData<T>);
pub(crate) struct DemoteTo<T>(PhantomData<T>);
pub(crate) struct PromoteTo<T>(PhantomData<T>);
pub(crate) struct ReinterpretTo<T>(PhantomData<T>);
pub(crate) struct ExtendFrom<T>(PhantomData<T>);

macro_rules! impl_rel_ops {
    ($($T:ty)*) => {
        $(
            impl BinOp<$T> for Eq {
                type Output = i32;

                fn bin_op(x0: $T, x1: $T) -> Result<Self::Output, Trap> {
                    Ok((x0 == x1).into())
                }
            }

            impl BinOp<$T> for Ne {
                type Output = i32;

                fn bin_op(x0: $T, x1: $T) -> Result<Self::Output, Trap> {
                    Ok((x0 != x1).into())
                }
            }

            impl BinOp<$T> for Lt {
                type Output = i32;

                fn bin_op(x0: $T, x1: $T) -> Result<Self::Output, Trap> {
                    Ok((x0 < x1).into())
                }
            }

            impl BinOp<$T> for Gt {
                type Output = i32;

                fn bin_op(x0: $T, x1: $T) -> Result<Self::Output, Trap> {
                    Ok((x0 > x1).into())
                }
            }

            impl BinOp<$T> for Le {
                type Output = i32;

                fn bin_op(x0: $T, x1: $T) -> Result<Self::Output, Trap> {
                    Ok((x0 <= x1).into())
                }
            }

            impl BinOp<$T> for Ge {
                type Output = i32;

                fn bin_op(x0: $T, x1: $T) -> Result<Self::Output, Trap> {
                    Ok((x0 >= x1).into())
                }
            }
        )*
    }
}

impl_rel_ops! { i32 u32 i64 u64 f32 f64 }

macro_rules! impl_int_ops {
    ($($T:ty)*) => {
        $(
            impl UnOp<$T> for Eqz {
                type Output = i32;

                fn un_op(x: $T) -> Result<Self::Output, Trap> {
                    Ok((x == 0).into())
                }
            }

            impl UnOp<$T> for Clz {
                type Output = $T;

                fn un_op(x: $T) -> Result<Self::Output, Trap> {
                    Ok(x.leading_zeros() as $T)
                }
            }

            impl UnOp<$T> for Ctz {
                type Output = $T;

                fn un_op(x: $T) -> Result<Self::Output, Trap> {
                    Ok(x.trailing_zeros() as $T)
                }
            }

            impl UnOp<$T> for Popcnt {
                type Output = $T;

                fn un_op(x: $T) -> Result<Self::Output, Trap> {
                    Ok(x.count_ones() as $T)
                }
            }

            impl BinOp<$T> for Add {
                type Output = $T;

                fn bin_op(x0: $T, x1: $T) -> Result<Self::Output, Trap> {
                    Ok(x0.wrapping_add(x1))
                }
            }

            impl BinOp<$T> for Sub {
                type Output = $T;

                fn bin_op(x0: $T, x1: $T) -> Result<Self::Output, Trap> {
                    Ok(x0.wrapping_sub(x1))
                }
            }

            impl BinOp<$T> for Mul {
                type Output = $T;

                fn bin_op(x0: $T, x1: $T) -> Result<Self::Output, Trap> {
                    Ok(x0.wrapping_mul(x1))
                }
            }

            impl BinOp<$T> for Div {
                type Output = $T;

                fn bin_op(x0: $T, x1: $T) -> Result<Self::Output, Trap> {
                    if x1 == 0 {
                        return Err(Trap::IntDivByZero);
                    }
                    match x0.overflowing_div(x1) {
                        (result, false) => Ok(result),
                        (_, true) => Err(Trap::IntOverflow),
                    }
                }
            }

            impl BinOp<$T> for Rem {
                type Output = $T;

                fn bin_op(x0: $T, x1: $T) -> Result<Self::Output, Trap> {
                    if x1 == 0 {
                        return Err(Trap::IntDivByZero);
                    }
                    Ok(x0.wrapping_rem(x1))
                }
            }

            impl BinOp<$T> for And {
                type Output = $T;

                fn bin_op(x0: $T, x1: $T) -> Result<Self::Output, Trap> {
                    Ok(x0 & x1)
                }
            }

            impl BinOp<$T> for Or {
                type Output = $T;

                fn bin_op(x0: $T, x1: $T) -> Result<Self::Output, Trap> {
                    Ok(x0 | x1)
                }
            }

            impl BinOp<$T> for Xor {
                type Output = $T;

                fn bin_op(x0: $T, x1: $T) -> Result<Self::Output, Trap> {
                    Ok(x0 ^ x1)
                }
            }

            impl BinOp<$T> for Shl {
                type Output = $T;

                fn bin_op(x0: $T, x1: $T) -> Result<Self::Output, Trap> {
                    Ok(x0.wrapping_shl(x1 as u32))
                }
            }

            impl BinOp<$T> for Shr {
                type Output = $T;

                fn bin_op(x0: $T, x1: $T) -> Result<Self::Output, Trap> {
                    Ok(x0.wrapping_shr(x1 as u32))
                }
            }

            impl BinOp<$T> for Rotl {
                type Output = $T;

                fn bin_op(x0: $T, x1: $T) -> Result<Self::Output, Trap> {
                    Ok(x0.rotate_left(x1 as u32))
                }
            }

            impl BinOp<$T> for Rotr {
                type Output = $T;

                fn bin_op(x0: $T, x1: $T) -> Result<Self::Output, Trap> {
                    Ok(x0.rotate_right(x1 as u32))
                }
            }
        )*
    }
}

impl_int_ops! { i32 u32 i64 u64 }

macro_rules! impl_float_ops {
    ($($T:ty)*) => {
        $(
            impl UnOp<$T> for Abs {
                type Output = $T;

                fn un_op(x: $T) -> Result<Self::Output, Trap> {
                    Ok(x.abs())
                }
            }

            impl UnOp<$T> for Neg {
                type Output = $T;

                fn un_op(x: $T) -> Result<Self::Output, Trap> {
                    Ok(-x)
                }
            }

            impl UnOp<$T> for Ceil {
                type Output = $T;

                fn un_op(x: $T) -> Result<Self::Output, Trap> {
                    Ok(x.ceil())
                }
            }

            impl UnOp<$T> for Floor {
                type Output = $T;

                fn un_op(x: $T) -> Result<Self::Output, Trap> {
                    Ok(x.floor())
                }
            }

            impl UnOp<$T> for Trunc {
                type Output = $T;

                fn un_op(x: $T) -> Result<Self::Output, Trap> {
                    Ok(x.trunc())
                }
            }

            impl UnOp<$T> for Nearest {
                type Output = $T;

                fn un_op(x: $T) -> Result<Self::Output, Trap> {
                    let round = x.round();
                    if x.fract().abs() != 0.5 {
                        Ok(round)
                    } else {
                        let rem = round % 2.0;
                        if rem == 1.0 {
                            Ok(x.floor())
                        } else if rem == -1.0 {
                            Ok(x.ceil())
                        } else {
                            Ok(round)
                        }
                    }
                }
            }

            impl UnOp<$T> for Sqrt {
                type Output = $T;

                fn un_op(x: $T) -> Result<Self::Output, Trap> {
                    Ok(x.sqrt())
                }
            }

            impl BinOp<$T> for Add {
                type Output = $T;

                fn bin_op(x0: $T, x1: $T) -> Result<Self::Output, Trap> {
                    Ok(x0 + x1)
                }
            }

            impl BinOp<$T> for Sub {
                type Output = $T;

                fn bin_op(x0: $T, x1: $T) -> Result<Self::Output, Trap> {
                    Ok(x0 - x1)
                }
            }

            impl BinOp<$T> for Mul {
                type Output = $T;

                fn bin_op(x0: $T, x1: $T) -> Result<Self::Output, Trap> {
                    Ok(x0 * x1)
                }
            }

            impl BinOp<$T> for Div {
                type Output = $T;

                fn bin_op(x0: $T, x1: $T) -> Result<Self::Output, Trap> {
                    Ok(x0 / x1)
                }
            }

            impl BinOp<$T> for Min {
                type Output = $T;

                fn bin_op(x0: $T, x1: $T) -> Result<Self::Output, Trap> {
                    if x0 < x1 {
                        Ok(x0)
                    } else if x1 < x0 {
                        Ok(x1)
                    } else if x0 == x1 {
                        if x0.is_sign_negative() && x1.is_sign_positive() {
                            Ok(x0)
                        } else {
                            Ok(x1)
                        }
                    } else {
                        Ok(x0 + x1)
                    }
                }
            }

            impl BinOp<$T> for Max {
                type Output = $T;

                fn bin_op(x0: $T, x1: $T) -> Result<Self::Output, Trap> {
                    if x0 > x1 {
                        Ok(x0)
                    } else if x1 > x0 {
                        Ok(x1)
                    } else if x0 == x1 {
                        if x0.is_sign_positive() && x1.is_sign_negative() {
                            Ok(x0)
                        } else {
                            Ok(x1)
                        }
                    } else {
                        Ok(x0 + x1)
                    }
                }
            }

            impl BinOp<$T> for Copysign {
                type Output = $T;

                fn bin_op(x0: $T, x1: $T) -> Result<Self::Output, Trap> {
                    let sign_mask = 1 << (mem::size_of::<Self>() * 8) - 1;
                    let bits_0 = x0.to_bits();
                    let bits_1 = x1.to_bits();
                    let sign_0 = bits_0 & sign_mask != 0;
                    let sign_1 = bits_1 & sign_mask != 0;
                    if sign_0 == sign_1 {
                        Ok(x0)
                    } else if sign_1 {
                        Ok(Self::Output::from_bits(bits_0 | sign_mask))
                    } else {
                        Ok(Self::Output::from_bits(bits_0 & !sign_mask))
                    }
                }
            }
        )*
    }
}

impl_float_ops! { f32 f64 }

macro_rules! impl_wrap_to {
    ($($Src:ty => $Dst:ty)*) => {
        $(
            impl UnOp<$Src> for WrapTo<$Dst> {
                type Output = $Dst;

                fn un_op(x: $Src) -> Result<Self::Output, Trap> {
                    Ok(x as Self::Output)
                }
            }
        )*
    };
}

impl_wrap_to! {
    i64 => i32
}

macro_rules! impl_extend_to {
    ($($Src:ty => $Dst:ty)*) => {
        $(
            impl UnOp<$Src> for ExtendTo<$Dst> {
                type Output = $Dst;

                fn un_op(x: $Src) -> Result<Self::Output, Trap> {
                    Ok(x as Self::Output)
                }
            }
        )*
    };
}

impl_extend_to! {
    i32 => i64
    u32 => u64
}

macro_rules! impl_trunc_to {
    ($($Src:ty => $Dst:ty: ($MIN:literal, $MAX:literal))*) => {
        $(
            impl UnOp<$Src> for TruncTo<$Dst> {
                type Output = $Dst;

                fn un_op(x: $Src) -> Result<Self::Output, Trap> {
                    if x.is_nan() {
                        return Err(Trap::InvalidConversionToInt);
                    }
                    if x <= $MIN || x >= $MAX {
                        return Err(Trap::IntOverflow);
                    }
                    Ok(x as Self::Output)
                }
            }

            impl UnOp<$Src> for TruncSatTo<$Dst> {
                type Output = $Dst;

                fn un_op(x: $Src) -> Result<Self::Output, Trap> {
                    if x.is_nan() {
                        Ok(0)
                    } else if x <= $MIN {
                        Ok(Self::Output::MIN)
                    } else if x >= $MAX {
                        Ok(Self::Output::MAX)
                    } else {
                        Ok(x as Self::Output)
                    }
                }
            }
        )*
    };
}

impl_trunc_to! {
    f32 => i32: (-2147483904f32, 2147483648f32)
    f32 => u32: (-1f32, 4294967296f32)
    f64 => i32: (-2147483649f64, 2147483648f64)
    f64 => u32: (-1f64, 4294967296f64)
    f32 => i64: (-9223373136366403584f32, 9223372036854775808f32)
    f32 => u64: (-1f32, 18446744073709551616f32)
    f64 => i64: (-9223372036854777856f64, 9223372036854775808f64)
    f64 => u64: (-1f64, 18446744073709551616f64)
}

macro_rules! impl_convert_to {
    ($($Src:ty => $Dst:ty)*) => {
        $(
            impl UnOp<$Src> for ConvertTo<$Dst> {
                type Output = $Dst;

                fn un_op(x: $Src) -> Result<Self::Output, Trap> {
                    Ok(x as Self::Output)
                }
            }
        )*
    };
}

impl_convert_to! {
    i32 => f32
    u32 => f32
    i64 => f32
    u64 => f32
    i32 => f64
    u32 => f64
    i64 => f64
    u64 => f64
    f32 => i32
    f32 => u32
}

impl UnOp<f64> for DemoteTo<f32> {
    type Output = f32;

    fn un_op(x: f64) -> Result<Self::Output, Trap> {
        Ok(x as f32)
    }
}

impl UnOp<f32> for PromoteTo<f64> {
    type Output = f64;

    fn un_op(x: f32) -> Result<Self::Output, Trap> {
        Ok(x as f64)
    }
}

macro_rules! impl_reinterpret_to {
    ($($Src:ty => $Dst:ty: $f:expr)*) => {
        $(
            impl UnOp<$Src> for ReinterpretTo<$Dst> {
                type Output = $Dst;

                fn un_op(x: $Src) -> Result<Self::Output, Trap> {
                    $f(x)
                }
            }
        )*
    };
}

impl_reinterpret_to! {
    f32 => i32: |x| Ok(f32::to_bits(x) as i32)
    f64 => i64: |x| Ok(f64::to_bits(x) as i64)
    i32 => f32: |x| Ok(f32::from_bits(x as u32))
    i64 => f64: |x| Ok(f64::from_bits(x as u64))
}

macro_rules! impl_extend_from {
    ($($Src:ty => $Dst:ty)*) => {
        $(
            impl UnOp<$Dst> for ExtendFrom<$Src> {
                type Output = $Dst;

                fn un_op(x: $Dst) -> Result<Self::Output, Trap> {
                    Ok(x as $Src as Self::Output)
                }
            }
        )*
    };
}

impl_extend_from! {
    i8 => i32
    i16 => i32
    i8 => i64
    i16 => i64
    i32 => i64
}
