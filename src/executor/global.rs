//! Global executors

use {
    super::*,
    crate::runtime::global::{GlobalEntity, TypedGlobalEntity, UnguardedGlobal},
};

/// Executes a `global.set` instruction.
pub(crate) unsafe extern "C" fn execute_global_get<T, W>(
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
    T: Guarded,
    TypedGlobalEntity<T>: DowncastRef<GlobalEntity>,
    W: Write<T::Unguarded>,
{
    unsafe {
        let mut args = Args::from_parts(ip, sp, md, ms, ia, sa, da, cx);
        let global: UnguardedGlobal = args.read_imm();
        let global = global.as_ref().downcast_ref::<T>().unwrap_unchecked();
        let val = global.get_unguarded();
        W::write(&mut args, val);
        args.next()
    }
}

/// Executes a `global.set` instruction.
pub(crate) unsafe extern "C" fn execute_global_set<T, R>(
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
    T: Guarded,
    TypedGlobalEntity<T>: DowncastMut<GlobalEntity>,
    R: Read<T::Unguarded>,
{
    unsafe {
        let mut args = Args::from_parts(ip, sp, md, ms, ia, sa, da, cx);
        let val = R::read(&mut args);
        let mut global: UnguardedGlobal = args.read_imm();
        let global = global.as_mut().downcast_mut::<T>().unwrap_unchecked();
        global.set_unguarded(val);
        args.next()
    }
}
