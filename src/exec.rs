//! Types and functions for executing threaded code.

use {
    crate::{
        cast::{ExtendingCast, ExtendingCastFrom, WrappingCast, WrappingCastFrom},
        code,
        data::UnguardedData,
        downcast::{DowncastMut, DowncastRef},
        elem::{ElemEntity, ElemEntityT, UnguardedElem},
        error::Error,
        extern_::UnguardedExtern,
        extern_ref::UnguardedExternRef,
        func::{Func, FuncBody, FuncEntity, FuncType, InstrSlot, UnguardedFunc},
        func_ref::UnguardedFuncRef,
        global::{GlobalEntity, GlobalEntityT, UnguardedGlobal},
        mem::UnguardedMem,
        ops::*,
        stack,
        stack::{Stack, StackGuard, StackSlot},
        store::{Handle, Store, UnguardedInternedFuncType},
        table::{TableEntity, TableEntityT, UnguardedTable},
        trap::Trap,
        val::{UnguardedVal, Val},
    },
    std::{alloc::Layout, hint, ptr},
};

/// A `ThreadedInstr` is a subroutine that executes a single WebAssembly instruction.
///
/// The signature of a `ThreadedInstr` has been carefully designed so that LLVM can perform sibling
/// optimisation, and the most heavily used parts of the execution context are stored in hardware
/// registers, which is crucial for performance.
///
/// The idea is to pass a copy of the most heavily fields of the execution context as arguments to a
/// `ThreadedInstr`. These arguments form the "registers" of our virtual machine. We currently use 6
/// virtual integer registers and 2 virtual floating-point registers. Our goal is to make sure that
/// these virtual registers are mapped to actual hardware registers on the physical machine.
///
/// On 64-bit non-Windows platforms, we use the "C" ABI. This corresponds to the "aapcs" ABI on Mac,
/// and the "sysv64" ABI on Linux. Both ABIs allow at least 6 integer and 6 floating point arguments
/// to be passed in hardware registers, which is sufficient for our needs.
///
/// On 64-bit Windows platforms, the "C" ABI corresponds to the "win64" ABI. This ABI allows only
/// the first 4 arguments to be passed in hardware registers, regardless of their type. This is
/// insufficient for our needs, so on Windows platforms, we use the "sysv64" ABI instead.

#[cfg(not(windows))]
pub(crate) type ThreadedInstr = unsafe extern "C" fn(
    ip: Ip,
    sp: Sp,
    md: Md,
    ms: Ms,
    ia: Ia,
    sa: Sa,
    da: Da,
    cx: Cx,
) -> ControlFlowBits;

#[cfg(windows)]
pub(crate) type ThreadedInstr = unsafe extern "sysv64" fn(
    ip: Ip,
    sp: Sp,
    md: Md,
    ms: Ms,
    ia: Ia,
    sa: Sa,
    da: Da,
    cx: Cx,
) -> ControlFlowBits;

// Virtual registers

/// The instruction pointer register (`Ip`) stores a pointer to the current instruction.
pub(crate) type Ip = *mut u8;

/// The stack pointer register (`Sp`) stores a pointer to the end of the current call frame.
pub(crate) type Sp = *mut u8;

/// The memory data register (`Md`) stores a pointer to the start of the current [`Memory`].
pub(crate) type Md = *mut u8;

/// The memory size register (`Ms`) stores the size of the current [`Memory`].
pub(crate) type Ms = u32;

/// The integer register (`Ia`) stores temporary values of integral type.
pub(crate) type Ia = u64;

/// The single precision floating-point register (`Sa`) stores temporary values of type `f32`.
pub(crate) type Sa = f32;

/// The double precision floating-point register (`Da`) stores temporary values of type `f64`.
pub(crate) type Da = f64;

/// The context register (`Cx`) stores a pointer to a [`Context`].
///
/// This register is special because it's the only one that does not have a corresponding field in
/// the [`Context`], but instead stores a pointer to the [`Context`] itself.
pub(crate) type Cx<'a> = *mut Context<'a>;

/// An execution context for executing threaded code.
#[derive(Debug)]
pub(crate) struct Context<'a> {
    // Virtual registers
    pub(crate) ip: Ip,
    pub(crate) sp: Sp,
    pub(crate) md: Md,
    pub(crate) ms: Ms,
    pub(crate) ia: Ia,
    pub(crate) sa: Sa,
    pub(crate) da: Da,

    // A mutable reference to the store in which we're executing.
    pub(crate) store: &'a mut Store,
    // A scoped lock to the stack for the current thread.
    pub(crate) stack: Option<StackGuard>,
    // Used to store out-of-band error data.
    pub(crate) error: Option<Error>,
}

/// Used to tell the interpreter what to do next.
#[derive(Clone, Copy, Debug)]
pub(crate) enum ControlFlow {
    Stop,
    Trap(Trap),
    Error,
}

impl ControlFlow {
    /// Creates a `ControlFlow` from its raw bits.
    pub(crate) fn from_bits(bits: usize) -> Option<Self> {
        if bits == 0 {
            Some(Self::Stop)
        } else if bits & 0x03 == 2 {
            Trap::from_usize(bits >> 2).map(Self::Trap)
        } else if bits & 0x03 == 3 {
            Some(Self::Error)
        } else {
            None
        }
    }

    /// Converts a `ControlFlow` to its raw bits.
    pub(crate) fn to_bits(self) -> ControlFlowBits {
        match self {
            Self::Stop => 0,
            Self::Trap(trap) => trap.to_usize() << 2 | 2,
            Self::Error => 3,
        }
    }
}

/// The raw bit representation of a `ControlFlow`.
pub(crate) type ControlFlowBits = usize;

/// Executes the given [`Func`] with the given arguments.
///
/// The results are written to the `results` slice.
pub(crate) fn exec(
    store: &mut Store,
    func: Func,
    args: &[Val],
    results: &mut [Val],
) -> Result<(), Error> {
    // Lock the stack for the current thread.
    let mut stack = Stack::lock();

    // Obtain the type of the function.
    let type_ = func.type_(store).clone();

    // Check that the stack has enough space.
    let stack_height = unsafe { stack.ptr().offset_from(stack.base_ptr()) as usize };
    if call_frame_size(&type_) / size_of::<StackSlot>() > Stack::SIZE - stack_height {
        return Err(Trap::StackOverflow)?;
    }

    // Copy the arguments to the stack.
    let mut ptr = stack.ptr();
    for arg in args.iter().copied() {
        unsafe {
            arg.to_unguarded(store.id()).write_to_ptr(ptr);
            ptr = ptr.add(arg.type_().padded_size_of());
        };
    }

    // Ensure that the function is compiled before calling it.
    func.compile(store);

    // Store the start of the call frame so we can reset the stack to it later.
    let ptr = stack.ptr();

    match func.0.as_mut(store) {
        FuncEntity::Wasm(func) => {
            // Obtain the compiled code for this function.
            let FuncBody::Compiled(code) = func.code_mut() else {
                unreachable!();
            };

            // Create a trampoline for the [`WasmFuncEntity`].
            let mut trampoline = [
                call_wasm as InstrSlot,
                code.code.as_mut_ptr() as InstrSlot,
                call_frame_size(&type_),
                stop as InstrSlot,
            ];

            // Create an execution context.
            let mut context = Context {
                ip: trampoline.as_mut_ptr() as *mut u8,
                sp: stack.ptr() as Sp,
                md: ptr::null_mut(),
                ms: 0,
                ia: 0,
                sa: 0.0,
                da: 0.0,
                store,
                stack: Some(stack),
                error: None,
            };

            // Main interpreter loop
            loop {
                match ControlFlow::from_bits(unsafe {
                    let instr: ThreadedInstr = ptr::read(context.ip.cast());
                    (instr)(
                        context.ip,
                        context.sp,
                        context.md,
                        context.ms,
                        context.ia,
                        context.sa,
                        context.da,
                        &mut context as *mut _,
                    )
                })
                .unwrap()
                {
                    ControlFlow::Stop => {
                        stack = context.stack.take().unwrap();

                        // Reset the stack to the start of the call frame.
                        unsafe { stack.set_ptr(ptr) };

                        break;
                    }
                    ControlFlow::Trap(trap) => {
                        stack = context.stack.take().unwrap();

                        // Reset the stack to the start of the call frame.
                        unsafe { stack.set_ptr(ptr) };

                        return Err(trap)?;
                    }
                    ControlFlow::Error => {
                        stack = context.stack.take().unwrap();

                        // Reset the stack to the start of the call frame.
                        unsafe { stack.set_ptr(ptr) };

                        return Err(context.error.take().unwrap());
                    }
                }
            }
        }
        FuncEntity::Host(func) => {
            // Set the stack pointer to the end of the call frame.
            unsafe { stack.set_ptr(ptr.add(call_frame_size(&type_))) };

            // Call the [`HostTrampoline`] of the [`HostFuncEntity`].
            stack = func.trampoline().clone().call(store, stack)?;

            // Reset the stack to the start of the call frame.
            unsafe { stack.set_ptr(ptr) };
        }
    }

    // Copy the results from the stack.
    let mut ptr = stack.ptr();
    for result in results.iter_mut() {
        unsafe {
            *result = Val::from_unguarded(
                UnguardedVal::read_from_ptr(ptr, result.type_()),
                store.id(),
            );
            ptr = ptr.add(result.type_().padded_size_of());
        }
    }

    Ok(())
}

// Helper macros

/// A helper macro for unwrapping a result or propagating its trap.
macro_rules! r#try {
    ($expr:expr) => {
        match $expr {
            Ok(val) => val,
            Err(trap) => return ControlFlow::Trap(trap).to_bits(),
        }
    };
}

// Control instructions

pub(crate) unsafe extern "C" fn unreachable(
    _ip: Ip,
    _sp: Sp,
    _md: Md,
    _ms: Ms,
    _ia: Ia,
    _sa: Sa,
    _da: Da,
    _cx: Cx,
) -> ControlFlowBits {
    ControlFlow::Trap(Trap::Unreachable).to_bits()
}

pub(crate) unsafe extern "C" fn br(
    ip: Ip,
    sp: Sp,
    md: Md,
    ms: Ms,
    ia: Ia,
    sa: Sa,
    da: Da,
    cx: Cx,
) -> ControlFlowBits {
    unsafe {
        let mut args = Args::from_parts(ip, sp, md, ms, ia, sa, da, cx);
        let target = args.read_imm();
        args.set_ip(target);
        args.next()
    }
}

pub(crate) unsafe extern "C" fn br_if<R>(
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
    R: Read<i32>
{
    unsafe {
        let mut args = Args::from_parts(ip, sp, md, ms, ia, sa, da, cx);
        let cond: i32 = R::read(&mut args);
        let target = args.read_imm();
        if cond != 0 {
            args.set_ip(target);
            args.next()
        } else {
            args.next()
        }
    }
}

pub(crate) unsafe extern "C" fn br_if_not<R>(
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
    R: Read<i32>
{
    unsafe {
        let mut args = Args::from_parts(ip, sp, md, ms, ia, sa, da, cx);
        let cond: i32 = R::read(&mut args);
        let target = args.read_imm();
        if cond == 0 {
            args.set_ip(target);
            args.next()
        } else {
            args.next()
        }
    }
}

pub(crate) unsafe extern "C" fn br_if_rel_op<T, B, R0, R1>(
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
    B: BinOp<T, Output = i32>,
    R0: Read<T>,
    R1: Read<T>,
{
    unsafe {
        let mut args = Args::from_parts(ip, sp, md, ms, ia, sa, da, cx);
        let x1 = R1::read(&mut args);
        let x0 = R0::read(&mut args);
        let cond: i32 = r#try!(B::bin_op(x0, x1));
        let target = args.read_imm();
        if cond != 0 {
            args.set_ip(target);
            args.next()
        } else {
            args.next()
        }
    }
}

pub(crate) unsafe extern "C" fn br_if_not_rel_op<T, B, R0, R1>(
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
    B: BinOp<T, Output = i32>,
    R0: Read<T>,
    R1: Read<T>,
{
    unsafe {
        let mut args = Args::from_parts(ip, sp, md, ms, ia, sa, da, cx);
        let x1 = R1::read(&mut args);
        let x0 = R0::read(&mut args);
        let cond: i32 = r#try!(B::bin_op(x0, x1));
        let target = args.read_imm();
        if cond == 0 {
            args.set_ip(target);
            args.next()
        } else {
            args.next()
        }
    }
}


pub(crate) unsafe extern "C" fn br_table<R>(
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
    R: Read<u32>
{
    unsafe {
        let mut args = Args::from_parts(ip, sp, md, ms, ia, sa, da, cx);
        let target_ida: u32 = R::read(&mut args);
        let target_count: u32 = args.read_imm();
        args.align_ip(align_of::<Ip>());
        let targets: *mut Ip = args.ip().cast();
        args.set_ip(*targets.add(target_ida.min(target_count) as usize));
        args.next()
    }
}

pub(crate) unsafe extern "C" fn return_(
    _ip: Ip,
    sp: Sp,
    _md: Md,
    _ms: Ms,
    ia: Ia,
    sa: Sa,
    da: Da,
    cx: Cx,
) -> ControlFlowBits {
    unsafe {
        let mut args = Args::from_parts(_ip, sp, _md, _ms, ia, sa, da, cx);
        // Restore call frame from stack.
        let old_sp = args.sp ;
        let saved_regs: SavedRegs = ptr::read(
            old_sp.offset( -(size_of::<SavedRegs>() as isize)).cast(),
        );
        args.set_ip(saved_regs.ip);
        args.sp = saved_regs.sp as Sp;
        args.md = saved_regs.md;
        args.ms = saved_regs.ms;
        args.next()
    }
}

pub(crate) unsafe extern "C" fn call_wasm(
    ip: Ip,
    sp: Sp,
    md: Md,
    ms: Ms,
    ia: Ia,
    sa: Sa,
    da: Da,
    cx: Cx,
) -> ControlFlowBits {
    unsafe {
        let mut args = Args::from_parts(ip, sp, md, ms, ia, sa, da, cx);
        let target = args.read_imm();
        let offset: i32 = args.read_imm();
        // Store call frame on stack.
        let new_sp = args.sp.offset(offset as isize);
        args.align_ip(code::ALIGN);
        ptr::write(
        new_sp.offset( -(size_of::<SavedRegs>() as isize)).cast(),
            SavedRegs {
                ip: args.ip(),
                sp: args.sp,
                md: args.md,
                ms: args.ms,
            }
        );
        args.set_ip(target);
        args.sp = new_sp as Sp;
        args.next()
    }
}

pub(crate) unsafe extern "C" fn call_host(
    ip: Ip,
    sp: Sp,
    _md: Md,
    _ms: Ms,
    ia: Ia,
    sa: Sa,
    da: Da,
    cx: Cx,
) -> ControlFlowBits {
    unsafe {
        let mut args = Args::from_parts(ip, sp, _md, _ms, ia, sa, da, cx);
        let func: UnguardedFunc = args.read_imm();
        let offset: i32 = args.read_imm();
        let mem: Option<UnguardedMem> = args.read_imm();

        let mut stack = (*args.cx).stack.take().unwrap_unchecked();
        stack.set_ptr(args.sp.offset(offset as isize).cast());
        let FuncEntity::Host(func) = func.as_ref() else {
            hint::unreachable_unchecked();
        };
        let stack = match func.trampoline().clone().call((*args.cx).store, stack) {
            Ok(stack) => stack,
            Err(error) => {
                (*args.cx).error = Some(error);
                return ControlFlow::Error.to_bits();
            }
        };
        (*args.cx).stack = Some(stack);

        // If the host function called `memory.grow`, `md` and `ms` are out of date. To ensure that `md`
        // and `ms` are up to date, we reset them here.
        if let Some(mut mem) = mem {
            let data = mem.as_mut().bytes_mut();
            args.md = data.as_mut_ptr();
            args.ms = data.len() as u32;
        } else {
            args.md = ptr::null_mut();
            args.ms = 0;
        }
        
        // Execute next instruction
        args.next()
    }
}

pub(crate) unsafe extern "C" fn call_indirect(
    ip: Ip,
    sp: Sp,
    md: Md,
    ms: Ms,
    ia: Ia,
    sa: Sa,
    da: Da,
    cx: Cx,
) -> ControlFlowBits {
    unsafe {
        let mut args = Args::from_parts(ip, sp, md, ms, ia, sa, da, cx);
        let func_ida = args.read_stack();
        let table: UnguardedTable = args.read_imm();
        let type_: UnguardedInternedFuncType = args.read_imm();
        let stack_offset: i32 = args.read_imm();
        let mem: Option<UnguardedMem> = args.read_imm();

        let func = r#try!(table
            .as_ref()
            .downcast_ref::<UnguardedFuncRef>()
            .unwrap_unchecked()
            .get(func_ida)
            .ok_or(Trap::TableAccessOutOfBounds));
        let mut func = r#try!(func.ok_or(Trap::ElemUninited));
        if func
            .as_ref()
            .type_()
            .to_unguarded((*(*cx).store).id())
            != type_
        {
            return ControlFlow::Trap(Trap::TypeMismatch).to_bits();
        }
        let id = (*(*args.cx).store).id();
        Func(Handle::from_unguarded(func, id)).compile(&mut *(*args.cx).store);
        match func.as_mut() {
            FuncEntity::Wasm(func) => {
                let FuncBody::Compiled(code) = func.code_mut() else {
                    hint::unreachable_unchecked();
                };
                let target = code.code.as_mut_ptr() as *mut u8;

                // Store call frame on stack.
                let new_sp = args.sp.offset(stack_offset as isize);
                args.align_ip(code::ALIGN);
                ptr::write(
                    new_sp.offset( -(size_of::<SavedRegs>() as isize)).cast(),
                    SavedRegs {
                        ip: args.ip(),
                        sp: args.sp,
                        md: args.md,
                        ms: args.ms,
                    }
                );

                // Update stack pointer and branch to target.
                args.set_ip(target);
                args.sp = new_sp as Sp;
                
                // Execute next instruction
                args.next()
            }
            FuncEntity::Host(func) => {
                let mut stack = (*args.cx).stack.take().unwrap_unchecked();
                stack.set_ptr(args.sp.offset(stack_offset as isize).cast());
                let stack = match func.trampoline().clone().call((*args.cx).store, stack) {
                    Ok(stack) => stack,
                    Err(error) => {
                        (*cx).error = Some(error);
                        return ControlFlow::Error.to_bits();
                    }
                };
                (*args.cx).stack = Some(stack);

                // If the host function called `memory.grow`, `md` and `ms` are out of date. To ensure
                // that `md` and `ms` are up to date, we reset them here.
                if let Some(mut mem) = mem {
                    let data = mem.as_mut().bytes_mut();
                    args.md = data.as_mut_ptr();
                    args.ms = data.len() as u32;
                } else {
                    args.md = ptr::null_mut();
                    args.ms = 0;
                }

                // Execute next instruction
                args.next()
            }
        }
    }
}

// Parametric instructions

pub(crate) unsafe extern "C" fn select<T, R0, R1, R2, W>(
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
    R0: Read<T>,
    R1: Read<T>,
    R2: Read<i32>,
    W: Write<T>,
{
    unsafe {
        let mut args = Args::from_parts(ip, sp, md, ms, ia, sa, da, cx);
        let cond = R2::read(&mut args);
        let x1 = R1::read(&mut args);
        let x0 = R0::read(&mut args);
        let y = if cond != 0 {
            x0
        } else {
            x1
        };
        W::write(&mut args, y);
        args.next()
    }
}

// Variable instructions

pub(crate) unsafe extern "C" fn global_get<T, W>(
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
    GlobalEntityT<T>: DowncastRef<GlobalEntity>,
    T: Copy,
    W: Write<T>,
{
    unsafe {
        let mut args = Args::from_parts(ip, sp, md, ms, ia, sa, da, cx);
        let global: UnguardedGlobal = args.read_imm();
        let global = global.as_ref().downcast_ref::<T>().unwrap_unchecked();
        let val = global.get();
        W::write(&mut args, val);
        args.next()
    }
}

pub(crate) unsafe extern "C" fn global_set<T, R>(
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
    GlobalEntityT<T>: DowncastMut<GlobalEntity>,
    T: Copy,
    R: Read<T>,
{
    unsafe {
        let mut args = Args::from_parts(ip, sp, md, ms, ia, sa, da, cx);
        let val = R::read(&mut args);
        let mut global: UnguardedGlobal = args.read_imm();
        let global = global.as_mut().downcast_mut::<T>().unwrap_unchecked();
        global.set(val);
        args.next()
    }
}

// Table instructions

pub(crate) unsafe extern "C" fn table_get<T, R, W> (
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
    TableEntityT<T>: DowncastRef<TableEntity>,
    T: Copy,
    R: Read<u32>,
    W: Write<T>,
{
    unsafe {
        let mut args = Args::from_parts(ip, sp, md, ms, ia, sa, da, cx);
        let ida = R::read(&mut args);
        let table: UnguardedTable = args.read_imm();
        let table = table.as_ref().downcast_ref::<T>().unwrap_unchecked();
        let val = r#try!(table.get(ida).ok_or(Trap::TableAccessOutOfBounds));
        W::write(&mut args, val);
        args.next() 
    }
}

pub(crate) unsafe extern "C" fn table_set<T, R0, R1> (
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
    TableEntityT<T>: DowncastMut<TableEntity>,
    T: Copy,
    R0: Read<u32>,
    R1: Read<T>,
{
    unsafe {
        let mut args = Args::from_parts(ip, sp, md, ms, ia, sa, da, cx);
        let val = R1::read(&mut args);
        let ida = R0::read(&mut args);
        let mut table: UnguardedTable = args.read_imm();
        let table = table.as_mut().downcast_mut::<T>().unwrap_unchecked();
        r#try!(table.set(ida, val).map_err(|_| Trap::TableAccessOutOfBounds));
        args.next() 
    }
}

pub(crate) unsafe extern "C" fn table_size<T, W>(
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
    TableEntityT<T>: DowncastRef<TableEntity>,
    T: Copy,
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

pub(crate) unsafe extern "C" fn table_grow<T, R0, R1, W>(
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
    TableEntityT<T>: DowncastMut<TableEntity>,
    T: Copy,
    R0: Read<T>,
    R1: Read<u32>,
    W: Write<u32>,
{
    unsafe {
        let mut args = Args::from_parts(ip, sp, md, ms, ia, sa, da, cx);
        let count = R1::read(&mut args);
        let val = R0::read(&mut args);
        let mut table: UnguardedTable = args.read_imm();
        let table = table.as_mut().downcast_mut::<T>().unwrap_unchecked();
        let old_size = table.grow(val, count).unwrap_or(u32::MAX);
        W::write(&mut args, old_size);
        args.next()
    }
}

pub(crate) unsafe extern "C" fn table_fill<T, R0, R1, R2>(
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
    TableEntityT<T>: DowncastMut<TableEntity>,
    T: Copy,
    R0: Read<u32>,
    R1: Read<T>,
    R2: Read<u32>,
{
    unsafe {
        let mut args = Args::from_parts(ip, sp, md, ms, ia, sa, da, cx);
        let count = R2::read(&mut args);
        let val = R1::read(&mut args);
        let ida = R0::read(&mut args);
        let mut table: UnguardedTable = args.read_imm();
        let table = table.as_mut().downcast_mut::<T>().unwrap_unchecked();
        r#try!(table.fill(ida, val, count));
        args.next()
    }   
}

pub(crate) unsafe extern "C" fn table_copy<T, R0, R1, R2>(
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
    TableEntityT<T>: DowncastRef<TableEntity> + DowncastMut<TableEntity>,
    T: Copy,
    R0: Read<u32>,
    R1: Read<u32>,
    R2: Read<u32>,
{
    unsafe {
        let mut args = Args::from_parts(ip, sp, md, ms, ia, sa, da, cx);
        let count = R2::read(&mut args);
        let src_ida = R1::read(&mut args);
        let dst_ida = R0::read(&mut args);
        let mut dst_table: UnguardedTable = args.read_imm();
        let dst_table = dst_table.as_mut().downcast_mut::<T>().unwrap_unchecked();
        let src_table: UnguardedTable = args.read_imm();
        let src_table = src_table.as_ref().downcast_ref::<T>().unwrap_unchecked();
        r#try!(if dst_table as *const _ == src_table as *const _ {
            dst_table.copy_within(dst_ida, src_ida, count)
        } else {
            dst_table.copy(dst_ida, src_table, src_ida, count)
        });
        args.next()
    }
}

pub(crate) unsafe extern "C" fn table_init<T, R0, R1, R2>(
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
    TableEntityT<T>: DowncastMut<TableEntity>,
    ElemEntityT<T>: DowncastRef<ElemEntity>,
    T: Copy,
    R0: Read<u32>,
    R1: Read<u32>,
    R2: Read<u32>,
{
    unsafe {
        let mut args = Args::from_parts(ip, sp, md, ms, ia, sa, da, cx);
        let count = R2::read(&mut args);
        let src_offset = R1::read(&mut args);
        let dst_offset = R0::read(&mut args);
        let mut dst_table: UnguardedTable = args.read_imm();
        let dst_table = dst_table.as_mut().downcast_mut::<T>().unwrap_unchecked();
        let src_elem: UnguardedElem = args.read_imm();
        let src_elem = src_elem.as_ref().downcast_ref::<T>().unwrap_unchecked();
        r#try!(dst_table.init(dst_offset, src_elem, src_offset, count));
        args.next()
    }
}

pub(crate) unsafe extern "C" fn elem_drop<T>(
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
    ElemEntityT<T>: DowncastMut<ElemEntity>,
    T: Copy,
{
    unsafe {
        let mut args = Args::from_parts(ip, sp, md, ms, ia, sa, da, cx);
        let mut elem: UnguardedElem = args.read_imm();
        let elem = elem.as_mut().downcast_mut::<T>().unwrap_unchecked();
        elem.drop_elems();
        args.next()
    }
}

// Memory instructions

pub(crate) unsafe extern "C" fn load<T, R, W>(
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
    T: ReadFromPtr,
    R: Read<u32>,
    W: Write<T>,
{
    unsafe {
        let mut args = Args::from_parts(ip, sp, md, ms, ia, sa, da, cx);
        let offset = R::read(&mut args);
        let base = args.read_imm();
        let val = r#try!(args.load(base, offset));
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
    cx: Cx,
) -> ControlFlowBits
where
    Dst: ExtendingCastFrom<Src>,
    Src: ReadFromPtr + ExtendingCast,
    R: Read<u32>,
    W: Write<Dst>,
{
    unsafe {
        let mut args = Args::from_parts(ip, sp, md, ms, ia, sa, da, cx);
        let offset = R::read(&mut args);
        let base = args.read_imm();
        let src: Src = r#try!(args.load(base, offset));
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
    cx: Cx,
) -> ControlFlowBits
where
    T: WriteToPtr,
    R1: Read<T>,
    R0: Read<u32>,
{
    unsafe {
        let mut args = Args::from_parts(ip, sp, md, ms, ia, sa, da, cx);
        let val = R1::read(&mut args);
        let offset = R0::read(&mut args);
        let base: u32 = args.read_imm();
        r#try!(args.store(base, offset, val));
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
    cx: Cx,
) -> ControlFlowBits
where
    Src: WrappingCast,
    Dst: WrappingCastFrom<Src> + WriteToPtr,
    R1: Read<Src>,
    R0: Read<u32>,
{
    unsafe {
        let mut args = Args::from_parts(ip, sp, md, ms, ia, sa, da, cx);
        let src = R1::read(&mut args);
        let offset = R0::read(&mut args);
        let base: u32 = args.read_imm();
        let dst: Dst = src.wrapping_cast();
        r#try!(args.store(base, offset, dst));
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
    cx: Cx,
) -> ControlFlowBits
where 
    W: Write<u32>
{
    unsafe {
        let mut args = Args::from_parts(ip, sp, md, ms, ia, sa, da, cx);

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
    cx: Cx,
) -> ControlFlowBits
where
    R: Read<u32>,
    W: Write<u32>,
{
    unsafe {
        let mut args = Args::from_parts(ip, sp, ptr::null_mut(), 0, ia, sa, da, cx);
    
        // Read operands
        let count = R::read(&mut args);
        let mut mem: UnguardedMem = args.read_imm();

        // Perform operation
        (*args.cx).stack.as_mut().unwrap_unchecked().set_ptr(args.sp);
        let old_size = mem
            .as_mut()
            .grow_with_stack(count, (*cx).stack.as_mut().unwrap_unchecked())
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
    cx: Cx,
) -> ControlFlowBits
where 
    R0: Read<u32>,
    R1: Read<u32>,
    R2: Read<u32>,
{
    unsafe {
        let mut args = Args::from_parts(ip, sp, md, ms, ia, sa, da, cx);
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
    cx: Cx,
) -> ControlFlowBits
where 
    R0: Read<u32>,
    R1: Read<u32>,
    R2: Read<u32>,
{
    unsafe {
        let mut args = Args::from_parts(ip, sp, md, ms, ia, sa, da, cx);
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
    cx: Cx,
) -> ControlFlowBits
where 
    R0: Read<u32>,
    R1: Read<u32>,
    R2: Read<u32>,
{
    unsafe {
        let mut args = Args::from_parts(ip, sp, md, ms, ia, sa, da, cx);
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
    cx: Cx,
) -> ControlFlowBits {
    unsafe {
        let mut args = Args::from_parts(ip, sp, md, ms, ia, sa, da, cx);
        let mut data: UnguardedData = args.read_imm();
        data.as_mut().drop_bytes();
        args.next()
    }
}

// Numeric instructions

pub(crate) unsafe extern "C" fn un_op<T, U, R, W>(
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

pub(crate) unsafe extern "C" fn bin_op<T, B, R0, R1, W>(
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

// Miscellaneous instructions

pub(crate) unsafe extern "C" fn copy<T, R, W>(
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
    R: Read<T>,
    W: Write<T>,
{
    unsafe {
        let mut args = Args::from_parts(ip, sp, md, ms, ia, sa, da, cx);
        let val = R::read(&mut args);
        W::write(&mut args, val);
        args.next()
    }
}

pub(crate) unsafe extern "C" fn stop(
    _ip: Ip,
    _sp: Sp,
    _md: Md,
    _ms: Ms,
    _ia: Ia,
    _sa: Sa,
    _da: Da,
    _cx: Cx,
) -> ControlFlowBits {
    ControlFlow::Stop.to_bits()
}

pub(crate) unsafe extern "C" fn compile(
    ip: Ip,
    sp: Sp,
    md: Md,
    ms: Ms,
    ia: Ia,
    sa: Sa,
    da: Da,
    cx: Cx,
) -> ControlFlowBits {
    unsafe {
        let mut args = Args::from_parts(ip, sp, md, ms, ia, sa, da, cx);
        args.align_ip(align_of::<UnguardedFuncRef>());
        let mut func = ptr::read(args.ip().cast());
        Func(Handle::from_unguarded(func, (*(*cx).store).id())).compile((*cx).store);
        let FuncEntity::Wasm(func) = func.as_mut() else {
            hint::unreachable_unchecked();
        };
        let FuncBody::Compiled(state) = func.code_mut() else {
            hint::unreachable_unchecked();
        };
        ptr::write(args.ip().cast(), state.code.as_mut_ptr());
        args.reset_ip();
        ptr::write(args.ip().cast(), call_wasm as ThreadedInstr);
        args.next()
    }
}

pub(crate) unsafe extern "C" fn enter(
    ip: Ip,
    sp: Sp,
    _md: Md,
    _ms: Ms,
    ia: Ia,
    sa: Sa,
    da: Da,
    cx: Cx,
) -> ControlFlowBits {
    unsafe {
        let mut args = Args::from_parts(ip, sp, ptr::null_mut(), 0, ia, sa, da, cx);
        let func: UnguardedFunc = args.read_imm();
        let mem: Option<UnguardedMem> = args.read_imm();
        let FuncEntity::Wasm(func) = func.as_ref() else {
            hint::unreachable_unchecked();
        };
        let FuncBody::Compiled(code) = func.code() else {
            hint::unreachable_unchecked();
        };

        // Check that the stack has enough space.
        let stack_height = (args.sp).offset_from((*args.cx).stack.as_mut().unwrap_unchecked().base_ptr()) as usize;
        if code.max_stack_height > Stack::SIZE - stack_height {
            return ControlFlow::Trap(Trap::StackOverflow).to_bits();
        }

        // Initialize the locals for this function to their default values.
        ptr::write_bytes(args.sp as *mut StackSlot, 0, code.local_count);

        if let Some(mut mem) = mem {
            let data = mem.as_mut().bytes_mut();
            args.md = data.as_mut_ptr();
            args.ms = data.len() as u32;
        } else {
            args.md = ptr::null_mut();
            args.ms = 0;
        }

        // Execute the next instruction.
        args.next()
    }
}

// Helper functions

pub(crate) struct Args<'a> {
    ip_base: Ip,
    ip_offset: usize,
    sp: Sp,
    md: Md,
    ms: Ms,
    ia: Ia,
    sa: Sa,
    da: Da,
    cx: Cx<'a>,
}

impl<'a> Args<'a> {
    unsafe fn from_parts(
        ip: Ip,
        sp: Sp,
        md: Md,
        ms: Ms,
        ia: Ia,
        sa: Sa,
        da: Da,
        cx: Cx<'a>,
    ) -> Self {
        let mut args = Self {
            ip_base: ip,
            ip_offset: 0,
            sp,
            md,
            ms,
            ia,
            sa,
            da,
            cx,
        };
        args.advance_ip(size_of::<InstrSlot>());
        args
    }

    fn into_parts(self) -> (Ip, Sp, Md, Ms, Ia, Sa, Da, Cx<'a>) {
        (
            self.ip(),
            self.sp,
            self.md,
            self.ms,
            self.ia,
            self.sa,
            self.da,
            self.cx,
        )
    }

    fn ip(&self) -> Ip {
        unsafe { self.ip_base.add(self.ip_offset) }
    }

    unsafe fn advance_ip(&mut self, count: usize) {
        self.ip_offset += count;
    }

    unsafe fn align_ip(&mut self, align: usize) {
        debug_assert!(align.is_power_of_two());
        self.ip_offset = (self.ip_offset + align - 1) & !(align - 1);
    }

    unsafe fn reset_ip(&mut self) {
        self.ip_offset = 0;
    }

    fn set_ip(&mut self, ip: Ip) {
        self.ip_base = ip;
        self.ip_offset = 0;
    }

    unsafe fn read_imm<T>(&mut self) -> T {
        unsafe {
            self.align_ip(align_of::<T>());
            let val = ptr::read(self.ip().cast());
            self.advance_ip(size_of::<InstrSlot>());
            val
        }
    }

    unsafe fn read_stack<T>(&mut self) -> T {
        unsafe {
            let offset: i32 = self.read_imm();
            ptr::read(self.sp.offset(offset as isize).cast::<T>())
        }
    }

    unsafe fn load<T>(&self, base: u32, offset: u32) -> Result<T, Trap>
    where 
        T: ReadFromPtr
    {
        let start = base as u64 + offset as u64;
        let end = start + size_of::<T>() as u64;
        if end > self.ms as u64 {
            return Err(Trap::MemAccessOutOfBounds);
        }
        unsafe { Ok(T::read_from_ptr(self.md.add(start as usize).cast())) }
    }

    unsafe fn write_stack<T>(&mut self, val: T) {
        unsafe {
            let offset: i32 = self.read_imm();
            ptr::write(self.sp.cast::<u8>().offset(offset as isize).cast::<T>(), val)
        }
    }

    fn read_reg<T>(&self) -> T
    where
        T: ReadFromReg,
    {
        T::read_from_reg(self)
    }

    fn write_reg<T>(&mut self, val: T)
    where
        T: WriteToReg,
    {
        val.write_to_reg(self)
    }

    unsafe fn store<T>(&mut self, base: u32, offset: u32, val: T) -> Result<(), Trap>
    where 
        T: WriteToPtr
    {
        let start = base as u64 + offset as u64;
        let end = start + size_of::<T>() as u64;
        if end > self.ms as u64 {
            return Err(Trap::MemAccessOutOfBounds);
        }
        unsafe { Ok(val.write_to_ptr(self.md.add(start as usize).cast())) }
    }
    
    unsafe fn next(mut self) -> ControlFlowBits {
        unsafe {
            self.align_ip(code::ALIGN);
            let instr: ThreadedInstr = *self.ip().cast();
            let (ip, sp, md, ms, ia, sa, da, cx) = self.into_parts();
            (instr)(ip, sp, md, ms, ia, sa, da, cx)
        }
    }
}

pub(crate) trait ReadFromReg {
    fn read_from_reg(args: &Args) -> Self;
}

impl ReadFromReg for i32 {
    fn read_from_reg(args: &Args) -> Self {
        args.ia as i32
    }
}

impl ReadFromReg for u32 {
    fn read_from_reg(args: &Args) -> Self {
        args.ia as u32
    }
}

impl ReadFromReg for i64 {
    fn read_from_reg(args: &Args) -> Self {
        args.ia as i64
    }
}

impl ReadFromReg for u64 {
    fn read_from_reg(args: &Args) -> Self {
        args.ia as u64
    }
}

impl ReadFromReg for f32 {
    fn read_from_reg(args: &Args) -> Self {
        args.sa
    }
}

impl ReadFromReg for f64 {
    fn read_from_reg(args: &Args) -> Self {
        args.da
    }
}

impl ReadFromReg for UnguardedFuncRef {
    fn read_from_reg(args: &Args) -> Self {
        UnguardedFunc::new(args.ia as *mut _)
    }
}

impl ReadFromReg for UnguardedExternRef {
    fn read_from_reg(args: &Args) -> Self {
        UnguardedExtern::new(args.ia as *mut _)
    }
}

pub(crate) trait WriteToReg {
    fn write_to_reg(self, args: &mut Args);
}

impl WriteToReg for i32 {
    fn write_to_reg(self, args: &mut Args) {
        args.ia = self as u32 as Ia;
    }
}

impl WriteToReg for u32 {
    fn write_to_reg(self, args: &mut Args) {
        args.ia = self as Ia;
    }
}

impl WriteToReg for i64 {
    fn write_to_reg(self, args: &mut Args) {
        args.ia = self as Ia;
    }
}

impl WriteToReg for u64 {
    fn write_to_reg(self, args: &mut Args) {
        args.ia = self as Ia;
    }
}

impl WriteToReg for f32 {
    fn write_to_reg(self, args: &mut Args) {
        args.sa = self;
    }
}

impl WriteToReg for f64 {
    fn write_to_reg(self, args: &mut Args) {
        args.da = self;
    }
}

impl WriteToReg for UnguardedFuncRef {
    fn write_to_reg(self, args: &mut Args) {
        args.ia = self.map_or(ptr::null_mut(), |ptr| ptr.as_ptr()) as Ia;
    }
}

impl WriteToReg for UnguardedExternRef {
    fn write_to_reg(self, args: &mut Args) {
        args.ia = self.map_or(ptr::null_mut(), |ptr| ptr.as_ptr()) as Ia;
    }
}

pub(crate) trait ReadFromPtr: Copy {
    unsafe fn read_from_ptr(ptr: *const u8) -> Self;
}

macro_rules! impl_read_from_ptr {
    ($($T:ty),*) => {
        $(
            impl ReadFromPtr for $T {
                unsafe fn read_from_ptr(ptr: *const u8) -> Self {
                    let mut bytes = [0u8; size_of::<$T>()];
                    ptr::copy_nonoverlapping(ptr, bytes.as_mut_ptr(), bytes.len());
                    <$T>::from_le_bytes(bytes)
                }
            }
        )*
    };
}

impl_read_from_ptr! { i8, u8, i16, u16, i32, u32, i64, u64, f32, f64 }

pub(crate) trait WriteToPtr {
    fn write_to_ptr(self, ptr: *mut u8);
}

macro_rules! impl_write_to_ptr {
    ($($T:ty),*) => {
        $(
            impl WriteToPtr for $T {
                fn write_to_ptr(self, ptr: *mut u8) {
                    let bytes = self.to_le_bytes();
                    unsafe {
                        ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len());
                    }
                }
            }
        )*
    };
}

impl_write_to_ptr! { i8, u8, i16, u16, i32, u32, i64, u64, f32, f64 }

pub(crate) trait Read<T>: Sized {
    unsafe fn read(args: &mut Args) -> T;
}

pub(crate) trait Write<T>: Sized {
    unsafe fn write(args: &mut Args, val: T);
}

pub(crate) struct ReadImm;

impl<T> Read<T> for ReadImm {
    unsafe fn read(args: &mut Args) -> T {
        unsafe { args.read_imm() }
    }
}

pub(crate) struct ReadStack;

impl<T> Read<T> for ReadStack {
    unsafe fn read(args: &mut Args) -> T {
        unsafe { args.read_stack() }
    }
}

pub(crate) struct WriteStack;

impl<T> Write<T> for WriteStack {
    unsafe fn write(args: &mut Args, val: T) {
        unsafe { args.write_stack(val) }
    }
}

pub(crate) struct ReadReg;

impl<T> Read<T> for ReadReg
where
    T: ReadFromReg,
{
    unsafe fn read(args: &mut Args) -> T {
        args.read_reg()
    }
}

pub(crate) struct WriteReg;

impl<T> Write<T> for WriteReg
where 
    T: WriteToReg
{
    unsafe fn write(args: &mut Args, val: T) {
        args.write_reg(val)
    }
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub(crate) struct SavedRegs {
    pub(crate) ip: Ip,
    pub(crate) sp: Sp,
    pub(crate) md: Md,
    pub(crate) ms: Ms,
}

const _: () = assert!(size_of::<SavedRegs>() % stack::ALIGN == 0);
const _: () = assert!(align_of::<SavedRegs>() >= stack::ALIGN);

/// Returns the size of a call frame for a function of the given type.
pub(crate) fn call_frame_size(type_: &FuncType) -> usize {
    let mut params_layout = Layout::from_size_align(0, 1).unwrap();
    for param in type_.params() {
        let param_layout = Layout::from_size_align(param.size(), stack::ALIGN).unwrap();
        let (new_params_layout, _) = params_layout.extend(param_layout).unwrap();
        params_layout = new_params_layout;
    }
    let mut results_layout = Layout::from_size_align(0, 1).unwrap();
    for result in type_.results() {
        let result_layout = Layout::from_size_align(result.size(), stack::ALIGN).unwrap();
        let (new_results_layout, _) = results_layout.extend(result_layout).unwrap();
        results_layout = new_results_layout;
    }
    let layout = Layout::from_size_align(
        params_layout.size().max(results_layout.size()),
        params_layout.align().max(results_layout.align())
    ).unwrap();
    let (layout, _) = layout.extend(Layout::new::<SavedRegs>()).unwrap();
    layout.size()
}