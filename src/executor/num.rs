//! Numeric executors

use super::*;

/// Executes a unary operation.
pub(crate) unsafe extern "C" fn execute_un_op<T, U, R, W>(
    ip: Ip,
    sp: Sp,
    md: Md,
    ms: Ms,
    ia: Ia,
    sa: Sa,
    da: Da,
    cx: Cx,
) -> ControlFlowBits
where
    U: UnOp<T>,
    R: Read<T>,
    W: Write<U::Output>,
{
    unsafe {
        let mut args = Args::from_parts(ip, sp, md, ms, ia, sa, da, cx);
        let x = R::read(&mut args);
        let y = r#try!(U::un_op(x));
        W::write(&mut args, y);
        args.next()
    }
}

/// Executes a binary operation.
pub(crate) unsafe extern "C" fn execute_bin_op<T, B, R0, R1, W>(
    ip: Ip,
    sp: Sp,
    md: Md,
    ms: Ms,
    ia: Ia,
    sa: Sa,
    da: Da,
    cx: Cx,
) -> ControlFlowBits
where
    B: BinOp<T>,
    R0: Read<T>,
    R1: Read<T>,
    W: Write<B::Output>,
{
    unsafe {
        let mut args = Args::from_parts(ip, sp, md, ms, ia, sa, da, cx);
        let x1 = R1::read(&mut args);
        let x0 = R0::read(&mut args);
        let y = r#try!(B::bin_op(x0, x1));
        W::write(&mut args, y);
        args.next()
    }
}