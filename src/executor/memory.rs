//! Memory instructions

use {
    super::*,
    crate::mem::UnguardedMem,
};

pub(crate) unsafe extern "C" fn load<T, R, W>(
    ip: Ip,
    sp: Sp,
    md: Md,
    ms: Ms,
    ia: Ia,
    sa: Sa,
    da: Da,
    executor: &mut Executor<'_>,
) -> ControlFlowBits
where
    T: ReadFromPtr,
    R: Read<u32>,
    W: Write<T>,
{
    unsafe {
        let mut args = Args::from_parts(ip, sp, md, ms, ia, sa, da, executor);
        let offset = R::read(&mut args);
        let base = args.read_imm();
        let val = r#try!(args.load_mem(base, offset));
        W::write(&mut args, val);
        args.next()
    }
}

pub(crate) unsafe extern "C" fn load_n<Dst, Src, R, W>(
    ip: Ip,
    sp: Sp,
    md: Md,
    ms: Ms,
    ia: Ia,
    sa: Sa,
    da: Da,
    executor: &mut Executor<'_>,
) -> ControlFlowBits
where
    Dst: ExtendingCastFrom<Src>,
    Src: ReadFromPtr + ExtendingCast,
    R: Read<u32>,
    W: Write<Dst>,
{
    unsafe {
        let mut args = Args::from_parts(ip, sp, md, ms, ia, sa, da, executor);
        let offset = R::read(&mut args);
        let base = args.read_imm();
        let src: Src = r#try!(args.load_mem(base, offset));
        let dst = src.extending_cast();
        W::write(&mut args, dst);
        args.next()
    }
}

pub(crate) unsafe extern "C" fn store<T, R0, R1>(
    ip: Ip,
    sp: Sp,
    md: Md,
    ms: Ms,
    ia: Ia,
    sa: Sa,
    da: Da,
    executor: &mut Executor<'_>,
) -> ControlFlowBits
where
    T: WriteToPtr,
    R1: Read<T>,
    R0: Read<u32>,
{
    unsafe {
        let mut args = Args::from_parts(ip, sp, md, ms, ia, sa, da, executor);
        let val = R1::read(&mut args);
        let offset = R0::read(&mut args);
        let base: u32 = args.read_imm();
        r#try!(args.store_mem(base, offset, val));
        args.next()
    }
}

pub(crate) unsafe extern "C" fn store_n<Src, Dst, R0, R1>(
    ip: Ip,
    sp: Sp,
    md: Md,
    ms: Ms,
    ia: Ia,
    sa: Sa,
    da: Da,
    executor: &mut Executor<'_>,
) -> ControlFlowBits
where
    Src: WrappingCast,
    Dst: WrappingCastFrom<Src> + WriteToPtr,
    R1: Read<Src>,
    R0: Read<u32>,
{
    unsafe {
        let mut args = Args::from_parts(ip, sp, md, ms, ia, sa, da, executor);
        let src = R1::read(&mut args);
        let offset = R0::read(&mut args);
        let base: u32 = args.read_imm();
        let dst: Dst = src.wrapping_cast();
        r#try!(args.store_mem(base, offset, dst));
        args.next()
    }
}

pub(crate) unsafe extern "C" fn memory_size<W>(
    ip: Ip,
    sp: Sp,
    md: Md,
    ms: Ms,
    ia: Ia,
    sa: Sa,
    da: Da,
    executor: &mut Executor<'_>,
) -> ControlFlowBits
where 
    W: Write<u32>
{
    unsafe {
        let mut args = Args::from_parts(ip, sp, md, ms, ia, sa, da, executor);

        // Read operands
        let mem: UnguardedMem = args.read_imm();
        
        // Perform operation
        let size = mem.as_ref().size();

        // Write result
        W::write(&mut args, size);

        // Execute next instruction
        args.next()
    }
}

pub(crate) unsafe extern "C" fn memory_grow<R, W>(
    ip: Ip,
    sp: Sp,
    _md: Md,
    _ms: Ms,
    ia: Ia,
    sa: Sa,
    da: Da,
    executor: &mut Executor<'_>,
) -> ControlFlowBits
where
    R: Read<u32>,
    W: Write<u32>,
{
    unsafe {
        let mut args = Args::from_parts(ip, sp, ptr::null_mut(), 0, ia, sa, da, executor);
    
        // Read operands
        let count = R::read(&mut args);
        let mut mem: UnguardedMem = args.read_imm();

        // Perform operation
        let stack = &mut args.executor.stack;
        let stack_height = args.sp.offset_from(stack.as_ptr()) as usize;
        stack.set_len(stack_height);
        let old_size = mem
            .as_mut()
            .grow_with_stack(count, Some(stack))
            .unwrap_or(u32::MAX);
        let bytes = mem.as_mut().bytes_mut();
        args.md = bytes.as_mut_ptr();
        args.ms = bytes.len() as u32;

        // Write result
        W::write(&mut args, old_size);
        
        // Execute next instruction
        args.next()
    }
}

pub(crate) unsafe extern "C" fn memory_fill<R0, R1, R2>(
    ip: Ip,
    sp: Sp,
    md: Md,
    ms: Ms,
    ia: Ia,
    sa: Sa,
    da: Da,
    executor: &mut Executor<'_>,
) -> ControlFlowBits
where 
    R0: Read<u32>,
    R1: Read<u32>,
    R2: Read<u32>,
{
    unsafe {
        let mut args = Args::from_parts(ip, sp, md, ms, ia, sa, da, executor);
        let count = R2::read(&mut args);
        let val = R1::read(&mut args);
        let ida = R0::read(&mut args);
        let mut mem: UnguardedMem = args.read_imm();
        r#try!(mem.as_mut().fill(ida, val as u8, count));
        args.next()
    }
}

pub(crate) unsafe extern "C" fn memory_copy<R0, R1, R2>(
    ip: Ip,
    sp: Sp,
    md: Md,
    ms: Ms,
    ia: Ia,
    sa: Sa,
    da: Da,
    executor: &mut Executor<'_>,
) -> ControlFlowBits
where 
    R0: Read<u32>,
    R1: Read<u32>,
    R2: Read<u32>,
{
    unsafe {
        let mut args = Args::from_parts(ip, sp, md, ms, ia, sa, da, executor);
        let count = R2::read(&mut args);
        let src_ida = R1::read(&mut args);
        let dst_ida = R0::read(&mut args);
        let mut mem: UnguardedMem = args.read_imm();
        r#try!(mem.as_mut().copy_within(dst_ida, src_ida, count));
        args.next()
    }
}

pub(crate) unsafe extern "C" fn memory_init<R0, R1, R2>(
    ip: Ip,
    sp: Sp,
    md: Md,
    ms: Ms,
    ia: Ia,
    sa: Sa,
    da: Da,
    executor: &mut Executor<'_>,
) -> ControlFlowBits
where 
    R0: Read<u32>,
    R1: Read<u32>,
    R2: Read<u32>,
{
    unsafe {
        let mut args = Args::from_parts(ip, sp, md, ms, ia, sa, da, executor);
        let count = R2::read(&mut args);
        let src_ida = R1::read(&mut args);
        let dst_ida = R0::read(&mut args);
        let mut dst_mem: UnguardedMem = args.read_imm();
        let src_data: UnguardedData = args.read_imm();
        r#try!(dst_mem.as_mut().init(dst_ida, src_data.as_ref(), src_ida, count));
        args.next()
    }
}

pub(crate) unsafe extern "C" fn data_drop(
    ip: Ip,
    sp: Sp,
    md: Md,
    ms: Ms,
    ia: Ia,
    sa: Sa,
    da: Da,
    executor: &mut Executor<'_>,
) -> ControlFlowBits {
    unsafe {
        let mut args = Args::from_parts(ip, sp, md, ms, ia, sa, da, executor);
        let mut data: UnguardedData = args.read_imm();
        data.as_mut().drop_bytes();
        args.next()
    }
}
