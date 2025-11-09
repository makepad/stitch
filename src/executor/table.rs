//! Table executors

use {
    super::*,
    crate::runtime::table::{TableEntity, TypedTableEntity, UnguardedTable},
};

/// Executes a `table.get` instruction.
pub(crate) unsafe extern "C" fn execute_table_get<T, R, W> (
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
    TypedTableEntity<T>: DowncastRef<TableEntity>,
    R: Read<u32>,
    W: Write<T::Unguarded>,
{
    unsafe {
        let mut args = Args::from_parts(ip, sp, md, ms, ia, sa, da, cx);
        let idx = R::read(&mut args);
        let table: UnguardedTable = args.read_imm();
        let table = table.as_ref().downcast_ref::<T>().unwrap_unchecked();
        let val = r#try!(table.get_unguarded(idx).ok_or(Trap::TableAccessOutOfBounds));
        W::write(&mut args, val);
        args.next() 
    }
}

/// Executes a `table.set` instructions.
pub(crate) unsafe extern "C" fn execute_table_set<T, R0, R1> (
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
    TypedTableEntity<T>: DowncastMut<TableEntity>,
    R0: Read<u32>,
    R1: Read<T::Unguarded>,
{
    unsafe {
        let mut args = Args::from_parts(ip, sp, md, ms, ia, sa, da, cx);
        let val = R1::read(&mut args);
        let idx = R0::read(&mut args);
        let mut table: UnguardedTable = args.read_imm();
        let table = table.as_mut().downcast_mut::<T>().unwrap_unchecked();
        r#try!(table.set_unguarded(idx, val).map_err(|_| Trap::TableAccessOutOfBounds));
        args.next() 
    }
}

/// Executes a `table.size` instruction.
pub(crate) unsafe extern "C" fn execute_table_size<T, W>(
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
    TypedTableEntity<T>: DowncastRef<TableEntity>,
    W: Write<u32>,
{
    unsafe {
        let mut args = Args::from_parts(ip, sp, md, ms, ia, sa, da, cx);
        let table: UnguardedTable = args.read_imm();
        let table = table.as_ref().downcast_ref::<T>().unwrap_unchecked();
        let size = table.size();
        W::write(&mut args, size);
        args.next()
    }
}

/// Executes a `table.grow` instruction.
pub(crate) unsafe extern "C" fn execute_table_grow<T, R0, R1, W>(
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
    TypedTableEntity<T>: DowncastMut<TableEntity>,
    R0: Read<T::Unguarded>,
    R1: Read<u32>,
    W: Write<u32>,
{
    unsafe {
        let mut args = Args::from_parts(ip, sp, md, ms, ia, sa, da, cx);
        let num = R1::read(&mut args);
        let val = R0::read(&mut args);
        let mut table: UnguardedTable = args.read_imm();
        let table = table.as_mut().downcast_mut::<T>().unwrap_unchecked();
        let old_size = table.grow_unguarded(val, num).unwrap_or(u32::MAX);
        W::write(&mut args, old_size);
        args.next()
    }
}

/// Executes a `table.fill` instructions.
pub(crate) unsafe extern "C" fn execute_table_fill<T, R0, R1, R2>(
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
    TypedTableEntity<T>: DowncastMut<TableEntity>,
    R0: Read<u32>,
    R1: Read<T::Unguarded>,
    R2: Read<u32>,
{
    unsafe {
        let mut args = Args::from_parts(ip, sp, md, ms, ia, sa, da, cx);
        let num = R2::read(&mut args);
        let val = R1::read(&mut args);
        let idx = R0::read(&mut args);
        let mut table: UnguardedTable = args.read_imm();
        let table = table.as_mut().downcast_mut::<T>().unwrap_unchecked();
        r#try!(table.fill_unguarded(idx, val, num));
        args.next()
    }   
}

/// Executes a `table.copy` instruction.
pub(crate) unsafe extern "C" fn execute_table_copy<T, R0, R1, R2>(
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
    TypedTableEntity<T>: DowncastRef<TableEntity> + DowncastMut<TableEntity>,
    R0: Read<u32>,
    R1: Read<u32>,
    R2: Read<u32>,
{
    unsafe {
        let mut args = Args::from_parts(ip, sp, md, ms, ia, sa, da, cx);
        let num = R2::read(&mut args);
        let src_idx = R1::read(&mut args);
        let dst_idx = R0::read(&mut args);
        let mut dst_table: UnguardedTable = args.read_imm();
        let dst_table = dst_table.as_mut().downcast_mut::<T>().unwrap_unchecked();
        let src_table: UnguardedTable = args.read_imm();
        let src_table = src_table.as_ref().downcast_ref::<T>().unwrap_unchecked();
        r#try!(if dst_table as *const _ == src_table as *const _ {
            dst_table.copy_within(dst_idx, src_idx, num)
        } else {
            dst_table.copy(dst_idx, src_table, src_idx, num)
        });
        args.next()
    }
}

/// Executes a `table.init` instruction.
pub(crate) unsafe extern "C" fn execute_table_init<T, R0, R1, R2>(
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
    TypedTableEntity<T>: DowncastMut<TableEntity>,
    TypedElemEntity<T>: DowncastRef<ElemEntity>,
    R0: Read<u32>,
    R1: Read<u32>,
    R2: Read<u32>,
{
    unsafe {
        let mut args = Args::from_parts(ip, sp, md, ms, ia, sa, da, cx);
        let num = R2::read(&mut args);
        let src_idx = R1::read(&mut args);
        let dst_idx = R0::read(&mut args);
        let mut dst_table: UnguardedTable = args.read_imm();
        let dst_table = dst_table.as_mut().downcast_mut::<T>().unwrap_unchecked();
        let src_elem: UnguardedElem = args.read_imm();
        let src_elem = src_elem.as_ref().downcast_ref::<T>().unwrap_unchecked();
        r#try!(dst_table.init(dst_idx, src_elem, src_idx, num));
        args.next()
    }
}

/// Executes an `elem.drop` instruction.
pub(crate) unsafe extern "C" fn execute_elem_drop<T>(
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
    TypedElemEntity<T>: DowncastMut<ElemEntity>,
{
    unsafe {
        let mut args = Args::from_parts(ip, sp, md, ms, ia, sa, da, cx);
        let mut elem: UnguardedElem = args.read_imm();
        let elem = elem.as_mut().downcast_mut::<T>().unwrap_unchecked();
        elem.drop_elems();
        args.next()
    }
}
