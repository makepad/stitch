//! Types and functions for executing threaded code.

use {
    crate::{
        code::{Code, InstrSlot},
        data::UnguardedData,
        elem::UnguardedElem,
        error::Error,
        extern_::UnguardedExtern,
        extern_ref::UnguardedExternRef,
        func::{Func, FuncEntity, UnguardedFunc},
        func_ref::UnguardedFuncRef,
        global::UnguardedGlobal,
        mem::UnguardedMem,
        ops::*,
        stack::{Stack, StackGuard, StackSlot},
        store::{Handle, Store, UnguardedInternedFuncType},
        table::UnguardedTable,
        trap::Trap,
        val::{UnguardedVal, Val},
    },
    std::{hint, mem, ptr},
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
    ix: Ix,
    sx: Sx,
    dx: Dx,
    cx: Cx,
) -> ControlFlowBits;

#[cfg(windows)]
pub(crate) type ThreadedInstr = unsafe extern "sysv64" fn(
    ip: Ip,
    sp: Sp,
    md: Md,
    ms: Ms,
    ix: Ix,
    sx: Sx,
    dx: Dx,
    cx: Cx,
) -> ControlFlowBits;

// Virtual registers

/// The instruction pointer register (`Ip`) stores a pointer to the current instruction.
pub(crate) type Ip = *mut InstrSlot;

/// The stack pointer register (`Sp`) stores a pointer to the end of the current call frame.
pub(crate) type Sp = *mut StackSlot;

/// The memory data register (`Md`) stores a pointer to the start of the current [`Memory`].
pub(crate) type Md = *mut u8;

/// The memory size register (`Ms`) stores the size of the current [`Memory`].
pub(crate) type Ms = u32;

/// The integer register (`Ix`) stores temporary values of integral type.
pub(crate) type Ix = u64;

/// The single precision floating-point register (`Sx`) stores temporary values of type `f32`.
pub(crate) type Sx = f32;

/// The double precision floating-point register (`Dx`) stores temporary values of type `f64`.
pub(crate) type Dx = f64;

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
    pub(crate) ix: Ix,
    pub(crate) sx: Sx,
    pub(crate) dx: Dx,

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
    if type_.call_frame_size() > Stack::SIZE - stack_height {
        return Err(Trap::StackOverflow)?;
    }

    // Copy the arguments to the stack.
    let mut ptr = stack.ptr();
    for arg in args.iter().copied() {
        let arg = arg.to_unguarded(store.id());
        unsafe {
            arg.write_to_stack(ptr);
            ptr = ptr.add(1);
        };
    }

    // Ensure that the function is compiled before calling it.
    func.compile(store);

    // Store the start of the call frame so we can reset the stack to it later.
    let ptr = stack.ptr();

    match func.0.as_mut(store) {
        FuncEntity::Wasm(func) => {
            // Obtain the compiled code for this function.
            let Code::Compiled(code) = func.code_mut() else {
                unreachable!();
            };

            // Create a trampoline for the [`WasmFuncEntity`].
            let mut trampoline = [
                call_wasm as InstrSlot,
                code.code.as_mut_ptr() as InstrSlot,
                type_.call_frame_size() * mem::size_of::<StackSlot>(),
                stop as InstrSlot,
            ];

            // Create an execution context.
            let mut context = Context {
                ip: trampoline.as_mut_ptr(),
                sp: stack.ptr(),
                md: ptr::null_mut(),
                ms: 0,
                ix: 0,
                sx: 0.0,
                dx: 0.0,
                store,
                stack: Some(stack),
                error: None,
            };

            // Main interpreter loop
            loop {
                match ControlFlow::from_bits(unsafe {
                    next_instr(
                        context.ip,
                        context.sp,
                        context.md,
                        context.ms,
                        context.ix,
                        context.sx,
                        context.dx,
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
            unsafe { stack.set_ptr(ptr.add(type_.call_frame_size())) };

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
                UnguardedVal::read_from_stack(ptr, result.type_()),
                store.id(),
            );
            ptr = ptr.add(1);
        }
    }

    Ok(())
}

// Helper macros

/// A helper macro for defining a `ThreadedInstr` with the correct ABI.
#[cfg(windows)]
macro_rules! threaded_instr {
    ($name:ident(
        $ip:ident: Ip,
        $sp:ident: Sp,
        $md:ident: Md,
        $ms:ident: Ms,
        $ix:ident: Ix,
        $sx:ident: Sx,
        $dx:ident: Dx,
        $cx:ident: Cx,
    ) -> ControlFlowBits $body:block) => {
        pub(crate) unsafe extern "sysv64" fn $name(
            $ip: Ip,
            $sp: Sp,
            $md: Md,
            $ms: Ms,
            $ix: Ix,
            $sx: Sx,
            $dx: Dx,
            $cx: Cx,
        ) -> ControlFlowBits $body
    };
}
#[cfg(not(windows))]
macro_rules! threaded_instr {
    ($name:ident(
        $ip:ident: Ip,
        $sp:ident: Sp,
        $md:ident: Md,
        $ms:ident: Ms,
        $ix:ident: Ix,
        $sx:ident: Sx,
        $dx:ident: Dx,
        $cx:ident: Cx,
    ) -> ControlFlowBits $body:block) => {
        pub(crate) unsafe extern "C" fn $name(
            $ip: Ip,
            $sp: Sp,
            $md: Md,
            $ms: Ms,
            $ix: Ix,
            $sx: Sx,
            $dx: Dx,
            $cx: Cx,
        ) -> ControlFlowBits $body
    };
}

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

threaded_instr!(unreachable(
    _ip: Ip,
    _sp: Sp,
    _md: Md,
    _ms: Ms,
    _ix: Ix,
    _sx: Sx,
    _dx: Dx,
    _cx: Cx,
) -> ControlFlowBits {
    ControlFlow::Trap(Trap::Unreachable).to_bits()
});

pub(crate) unsafe extern "C" fn br(
    ip: Ip,
    sp: Sp,
    md: Md,
    ms: Ms,
    ix: Ix,
    sx: Sx,
    dx: Dx,
    cx: Cx,
) -> ControlFlowBits {
    let mut args = Args::from_parts(ip, sp, md, ms, ix, sx, dx, cx);
    unsafe {
        let target = args.read_imm();
        args.set_ip(target);
        args.next()
    }
}

pub(crate) unsafe extern "C" fn br_if_z<R>(
    ip: Ip,
    sp: Sp,
    md: Md,
    ms: Ms,
    ix: Ix,
    sx: Sx,
    dx: Dx,
    cx: Cx,
) -> ControlFlowBits
where
    R: Read<i32>
{
    let mut args = Args::from_parts(ip, sp, md, ms, ix, sx, dx, cx);
    unsafe {
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

pub(crate) unsafe extern "C" fn br_if_nz<R>(
    ip: Ip,
    sp: Sp,
    md: Md,
    ms: Ms,
    ix: Ix,
    sx: Sx,
    dx: Dx,
    cx: Cx,
) -> ControlFlowBits
where
    R: Read<i32>
{
    let mut args = Args::from_parts(ip, sp, md, ms, ix, sx, dx, cx);
    unsafe {
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

pub(crate) unsafe extern "C" fn br_table<R>(
    ip: Ip,
    sp: Sp,
    md: Md,
    ms: Ms,
    ix: Ix,
    sx: Sx,
    dx: Dx,
    cx: Cx,
) -> ControlFlowBits
where 
    R: Read<u32>
{
    let mut args = Args::from_parts(ip, sp, md, ms, ix, sx, dx, cx);
    unsafe {
        let target_idx: u32 = R::read(&mut args);
        let target_count: u32 = args.read_imm();
        let targets: *mut Ip = args.ip().cast();
        args.set_ip(*targets.add(target_idx.min(target_count) as usize));
        args.next()
    }
}

threaded_instr!(return_(
    _ip: Ip,
    sp: Sp,
    _md: Md,
    _ms: Ms,
    ix: Ix,
    sx: Sx,
    dx: Dx,
    cx: Cx,
) -> ControlFlowBits {
    // Restore call frame from stack.
    let old_sp = sp;
    let ip = *old_sp.offset(-4).cast();
    let sp = *old_sp.offset(-3).cast();
    let md = *old_sp.offset(-2).cast();
    let ms = *old_sp.offset(-1).cast();

    // Execute next instruction.
    next_instr(ip, sp, md, ms, ix, sx, dx, cx)
});

threaded_instr!(call_wasm(
    ip: Ip,
    sp: Sp,
    md: Md,
    ms: Ms,
    ix: Ix,
    sx: Sx,
    dx: Dx,
    cx: Cx,
) -> ControlFlowBits {
    // Read operands.
    let (target, ip) = read_imm(ip);
    let (offset, ip) = read_imm(ip);

    // Store call frame on stack.
    let new_sp: Sp = sp.cast::<u8>().add(offset).cast();
    *new_sp.offset(-4).cast() = ip;
    *new_sp.offset(-3).cast() = sp;
    *new_sp.offset(-2).cast() = md;
    *new_sp.offset(-1).cast() = ms;

    // Update stack pointer and branch to target.
    let ip = target;
    let sp = new_sp;

    // Execute next instruction.
    next_instr(ip, sp, md, ms, ix, sx, dx, cx)
});

threaded_instr!(call_host(
    ip: Ip,
    sp: Sp,
    _md: Md,
    _ms: Ms,
    ix: Ix,
    sx: Sx,
    dx: Dx,
    cx: Cx,
) -> ControlFlowBits {
    // Read operands
    let (func, ip): (UnguardedFunc, _) = read_imm(ip);
    let (offset, ip) = read_imm(ip);
    let (mem, ip): (Option<UnguardedMem>, _) = read_imm(ip);

    let mut stack = (*cx).stack.take().unwrap_unchecked();
    stack.set_ptr(sp.cast::<u8>().add(offset).cast());
    let FuncEntity::Host(func) = func.as_ref() else {
        hint::unreachable_unchecked();
    };
    let stack = match func.trampoline().clone().call((*cx).store, stack) {
        Ok(stack) => stack,
        Err(error) => {
            (*cx).error = Some(error);
            return ControlFlow::Error.to_bits();
        }
    };
    (*cx).stack = Some(stack);

    // If the host function called `memory.grow`, `md` and `ms` are out of date. To ensure that `md`
    // and `ms` are up to date, we reset them here.
    let md;
    let ms;
    if let Some(mut mem) = mem {
        let data = mem.as_mut().bytes_mut();
        md = data.as_mut_ptr();
        ms = data.len() as u32;
    } else {
        md = ptr::null_mut();
        ms = 0;
    }

    // Execute next instruction
    next_instr(ip, sp, md, ms, ix, sx, dx, cx)
});

threaded_instr!(call_indirect(
    ip: Ip,
    sp: Sp,
    md: Md,
    ms: Ms,
    ix: Ix,
    sx: Sx,
    dx: Dx,
    cx: Cx,
) -> ControlFlowBits {
    // Read operands
    let (func_idx, ip): (u32, _) = read_stack(ip, sp);
    let (table, ip): (UnguardedTable, _) = read_imm(ip);
    let (type_, ip): (UnguardedInternedFuncType, _) = read_imm(ip);
    let (stack_offset, ip) = read_imm(ip);
    let (mem, ip): (Option<UnguardedMem>, _) = read_imm(ip);

    let func = r#try!(table
        .as_ref()
        .downcast_ref::<UnguardedFuncRef>()
        .unwrap_unchecked()
        .get(func_idx)
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
    Func(Handle::from_unguarded(func, (*(*cx).store).id())).compile(&mut *(*cx).store);
    match func.as_mut() {
        FuncEntity::Wasm(func) => {
            let Code::Compiled(code) = func.code_mut() else {
                hint::unreachable_unchecked();
            };
            let target = code.code.as_mut_ptr();

            // Store call frame on stack.
            let new_sp: Sp = sp.cast::<u8>().add(stack_offset).cast();
            *new_sp.offset(-4).cast() = ip;
            *new_sp.offset(-3).cast() = sp;
            *new_sp.offset(-2).cast() = md;
            *new_sp.offset(-1).cast() = ms;

            // Update stack pointer and branch to target.
            let ip = target;
            let sp = new_sp;

            // Execute next instruction
            next_instr(ip, sp, md, ms, ix, sx, dx, cx)
        }
        FuncEntity::Host(func) => {
            let mut stack = (*cx).stack.take().unwrap_unchecked();
            stack.set_ptr(sp.cast::<u8>().add(stack_offset).cast());
            let stack = match func.trampoline().clone().call((*cx).store, stack) {
                Ok(stack) => stack,
                Err(error) => {
                    (*cx).error = Some(error);
                    return ControlFlow::Error.to_bits();
                }
            };
            (*cx).stack = Some(stack);

            // If the host function called `memory.grow`, `md` and `ms` are out of date. To ensure
            // that `md` and `ms` are up to date, we reset them here.
            let md;
            let ms;
            if let Some(mut mem) = mem {
                let data = mem.as_mut().bytes_mut();
                md = data.as_mut_ptr();
                ms = data.len() as u32;
            } else {
                md = ptr::null_mut();
                ms = 0;
            }

            // Execute next instruction
            next_instr(ip, sp, md, ms, ix, sx, dx, cx)
        }
    }
});

// Reference instructions

macro_rules! ref_is_null {
    ($ref_is_null_s:ident, $ref_is_null_r:ident, $T:ty) => {
        threaded_instr!($ref_is_null_s(
            ip: Ip,
            sp: Sp,
            md: Md,
            ms: Ms,
            ix: Ix,
            sx: Sx,
            dx: Dx,
            cx: Cx,
        ) -> ControlFlowBits {
            // Read operands
            let (x, ip): ($T, _) = read_stack(ip, sp);

            // Perform operation
            let y = x.is_none() as u32;

            // Write result
            let (ix, sx, dx) = write_reg(ix, sx, dx, y);

            // Execute next instruction
            next_instr(ip, sp, md, ms, ix, sx, dx, cx)
        });

        threaded_instr!($ref_is_null_r(
            ip: Ip,
            sp: Sp,
            md: Md,
            ms: Ms,
            ix: Ix,
            sx: Sx,
            dx: Dx,
            cx: Cx,
        ) -> ControlFlowBits {
            // Read operands
            let x: $T = read_reg(ix, sx, dx);

            // Perform operation
            let y = x.is_none() as u32;

            // Write result
            let (ix, sx, dx) = write_reg(ix, sx, dx, y);

            // Execute next instruction
            next_instr(ip, sp, md, ms, ix, sx, dx, cx)
        });
    };
}

ref_is_null!(
    ref_is_null_func_ref_s,
    ref_is_null_func_ref_r,
    UnguardedFuncRef
);
ref_is_null!(
    ref_is_null_extern_ref_s,
    ref_is_null_extern_ref_r,
    UnguardedExternRef
);

// Parametric instructions

pub(crate) unsafe extern "C" fn select<T, R0, R1, R2, W>(
    ip: Ip,
    sp: Sp,
    md: Md,
    ms: Ms,
    ix: Ix,
    sx: Sx,
    dx: Dx,
    cx: Cx,
) -> ControlFlowBits
where 
    R0: Read<T>,
    R1: Read<T>,
    R2: Read<i32>,
    W: Write<T>,
{
    let mut args = Args::from_parts(ip, sp, md, ms, ix, sx, dx, cx);
    unsafe {
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

macro_rules! global_get {
    ($global_get:ident, $T:ty) => {
        threaded_instr!($global_get(
            ip: Ip,
            sp: Sp,
            md: Md,
            ms: Ms,
            ix: Ix,
            sx: Sx,
            dx: Dx,
            cx: Cx,
        ) -> ControlFlowBits {
            // Read operands
            let (global, ip): (UnguardedGlobal, _) = read_imm(ip);

            // Perform operation
            let val = global
                .as_ref()
                .downcast_ref::<$T>()
                .unwrap_unchecked()
                .get();

            // Write result
            let ip = write_stack(ip, sp, val);

            // Execute next instruction
            next_instr(ip, sp, md, ms, ix, sx, dx, cx)
        });
    };
}

global_get!(global_get_i32, i32);
global_get!(global_get_i64, i64);
global_get!(global_get_f32, f32);
global_get!(global_get_f64, f64);
global_get!(global_get_func_ref, UnguardedFuncRef);
global_get!(global_get_extern_ref, UnguardedExternRef);

macro_rules! global_set {
    ($global_set_s:ident, $global_set_r:ident, $global_set_i:ident, $T:ty) => {
        threaded_instr!($global_set_s(
            ip: Ip,
            sp: Sp,
            md: Md,
            ms: Ms,
            ix: Ix,
            sx: Sx,
            dx: Dx,
            cx: Cx,
        ) -> ControlFlowBits {
            // Read operands
            let (val, ip) = read_stack(ip, sp);
            let (mut global, ip): (UnguardedGlobal, _) = read_imm(ip);

            // Perform operation
            global
                .as_mut()
                .downcast_mut::<$T>()
                .unwrap_unchecked()
                .set(val);

            // Execute next instruction
            next_instr(ip, sp, md, ms, ix, sx, dx, cx)
        });

        threaded_instr!($global_set_r(
            ip: Ip,
            sp: Sp,
            md: Md,
            ms: Ms,
            ix: Ix,
            sx: Sx,
            dx: Dx,
            cx: Cx,
        ) -> ControlFlowBits {
            // Read operands
            let val = read_reg(ix, sx, dx);
            let (mut global, ip): (UnguardedGlobal, _) = read_imm(ip);

            // Perform operation
            global
                .as_mut()
                .downcast_mut::<$T>()
                .unwrap_unchecked()
                .set(val);

            // Execute next instruction
            next_instr(ip, sp, md, ms, ix, sx, dx, cx)
        });

        threaded_instr!($global_set_i(
            ip: Ip,
            sp: Sp,
            md: Md,
            ms: Ms,
            ix: Ix,
            sx: Sx,
            dx: Dx,
            cx: Cx,
        ) -> ControlFlowBits {
            // Read operands
            let (val, ip) = read_imm(ip);
            let (mut global, ip): (UnguardedGlobal, _) = read_imm(ip);

            // Perform operation
            global
                .as_mut()
                .downcast_mut::<$T>()
                .unwrap_unchecked()
                .set(val);

            // Execute next instruction
            next_instr(ip, sp, md, ms, ix, sx, dx, cx)
        });
    };
}

global_set!(global_set_i32_s, global_set_i32_r, global_set_i32_i, i32);
global_set!(global_set_i64_s, global_set_i64_r, global_set_i64_i, i64);
global_set!(global_set_f32_s, global_set_f32_r, global_set_f32_i, f32);
global_set!(global_set_f64_s, global_set_f64_r, global_set_f64_i, f64);
global_set!(
    global_set_func_ref_s,
    global_set_func_ref_r,
    global_set_func_ref_i,
    UnguardedFuncRef
);
global_set!(
    global_set_extern_ref_s,
    global_set_extern_ref_r,
    global_set_extern_ref_i,
    UnguardedExternRef
);

// Table instructions

macro_rules! table_get {
    ($table_get_s:ident, $table_get_r:ident, $table_get_i:ident, $T:ty) => {
        threaded_instr!($table_get_s(
            ip: Ip,
            sp: Sp,
            md: Md,
            ms: Ms,
            ix: Ix,
            sx: Sx,
            dx: Dx,
            cx: Cx,
        ) -> ControlFlowBits {
            // Read operands
            let (idx, ip) = read_stack(ip, sp);
            let (table, ip): (UnguardedTable, _) = read_imm(ip);

            // Perform operation
            let val = r#try!(table
                .as_ref()
                .downcast_ref::<$T>()
                .unwrap_unchecked()
                .get(idx)
                .ok_or(Trap::TableAccessOutOfBounds));

            // Write result
            let ip = write_stack(ip, sp, val);

            // Execute next instruction
            next_instr(ip, sp, md, ms, ix, sx, dx, cx)
        });

        threaded_instr!($table_get_r(
            ip: Ip,
            sp: Sp,
            md: Md,
            ms: Ms,
            ix: Ix,
            sx: Sx,
            dx: Dx,
            cx: Cx,
        ) -> ControlFlowBits {
            // Read operands
            let idx = read_reg(ix, sx, dx);
            let (table, ip): (UnguardedTable, _) = read_imm(ip);

            // Perform operation
            let val = r#try!(table
                .as_ref()
                .downcast_ref::<$T>()
                .unwrap_unchecked()
                .get(idx)
                .ok_or(Trap::TableAccessOutOfBounds));

            // Write result
            let ip = write_stack(ip, sp, val);

            // Execute next instruction
            next_instr(ip, sp, md, ms, ix, sx, dx, cx)
        });

        threaded_instr!($table_get_i(
            ip: Ip,
            sp: Sp,
            md: Md,
            ms: Ms,
            ix: Ix,
            sx: Sx,
            dx: Dx,
            cx: Cx,
        ) -> ControlFlowBits {
            // Read operands
            let (idx, ip) = read_imm(ip);
            let (table, ip): (UnguardedTable, _) = read_imm(ip);

            // Perform operation
            let val = r#try!(table
                .as_ref()
                .downcast_ref::<$T>()
                .unwrap_unchecked()
                .get(idx)
                .ok_or(Trap::TableAccessOutOfBounds));

            // Write result
            let ip = write_stack(ip, sp, val);

            // Execute next instruction
            next_instr(ip, sp, md, ms, ix, sx, dx, cx)
        });
    };
}

table_get!(
    table_get_func_ref_s,
    table_get_func_ref_r,
    table_get_func_ref_i,
    UnguardedFuncRef
);
table_get!(
    table_get_extern_ref_s,
    table_get_extern_ref_r,
    table_get_extern_ref_i,
    UnguardedExternRef
);

macro_rules! table_set {
    (
        $table_set_ss:ident,
        $table_set_rs:ident,
        $table_set_is:ident,
        $table_set_ir:ident,
        $table_set_ii:ident,
        $table_set_sr:ident,
        $table_set_si:ident,
        $table_set_ri:ident,
        $T:ty
    ) => {
        threaded_instr!($table_set_ss(
            ip: Ip,
            sp: Sp,
            md: Md,
            ms: Ms,
            ix: Ix,
            sx: Sx,
            dx: Dx,
            cx: Cx,
        ) -> ControlFlowBits {
            // Read operands
            let (val, ip) = read_stack(ip, sp);
            let (idx, ip) = read_stack(ip, sp);
            let (mut table, ip): (UnguardedTable, _) = read_imm(ip);

            // Perform operation
            r#try!(table
                .as_mut()
                .downcast_mut::<$T>()
                .unwrap_unchecked()
                .set(idx, val)
                .map_err(|_| Trap::TableAccessOutOfBounds));

            // Execute next instruction
            next_instr(ip, sp, md, ms, ix, sx, dx, cx)
        });

        threaded_instr!($table_set_rs(
            ip: Ip,
            sp: Sp,
            md: Md,
            ms: Ms,
            ix: Ix,
            sx: Sx,
            dx: Dx,
            cx: Cx,
        ) -> ControlFlowBits {
            // Read operands
            let (val, ip) = read_stack(ip, sp);
            let idx = read_reg(ix, sx, dx);
            let (mut table, ip): (UnguardedTable, _) = read_imm(ip);

            // Perform operation
            r#try!(table
                .as_mut()
                .downcast_mut::<$T>()
                .unwrap_unchecked()
                .set(idx, val)
                .map_err(|_| Trap::TableAccessOutOfBounds));

            // Execute next instruction
            next_instr(ip, sp, md, ms, ix, sx, dx, cx)
        });

        threaded_instr!($table_set_is(
            ip: Ip,
            sp: Sp,
            md: Md,
            ms: Ms,
            ix: Ix,
            sx: Sx,
            dx: Dx,
            cx: Cx,
        ) -> ControlFlowBits {
            // Read operands
            let (val, ip) = read_stack(ip, sp);
            let (idx, ip) = read_imm(ip);
            let (mut table, ip): (UnguardedTable, _) = read_imm(ip);

            // Perform operation
            r#try!(table
                .as_mut()
                .downcast_mut::<$T>()
                .unwrap_unchecked()
                .set(idx, val)
                .map_err(|_| Trap::TableAccessOutOfBounds));

            // Execute next instruction
            next_instr(ip, sp, md, ms, ix, sx, dx, cx)
        });

        threaded_instr!($table_set_ir(
            ip: Ip,
            sp: Sp,
            md: Md,
            ms: Ms,
            ix: Ix,
            sx: Sx,
            dx: Dx,
            cx: Cx,
        ) -> ControlFlowBits {
            // Read operands
            let val = read_reg(ix, sx, dx);
            let (idx, ip) = read_imm(ip);
            let (mut table, ip): (UnguardedTable, _) = read_imm(ip);

            // Perform operation
            r#try!(table
                .as_mut()
                .downcast_mut::<$T>()
                .unwrap_unchecked()
                .set(idx, val)
                .map_err(|_| Trap::TableAccessOutOfBounds));

            // Execute next instruction
            next_instr(ip, sp, md, ms, ix, sx, dx, cx)
        });

        threaded_instr!($table_set_ii(
            ip: Ip,
            sp: Sp,
            md: Md,
            ms: Ms,
            ix: Ix,
            sx: Sx,
            dx: Dx,
            cx: Cx,
        ) -> ControlFlowBits {
            // Read operands
            let (val, ip) = read_imm(ip);
            let (idx, ip) = read_imm(ip);
            let (mut table, ip): (UnguardedTable, _) = read_imm(ip);

            // Perform operation
            r#try!(table
                .as_mut()
                .downcast_mut::<$T>()
                .unwrap_unchecked()
                .set(idx, val)
                .map_err(|_| Trap::TableAccessOutOfBounds));

            // Execute next instruction
            next_instr(ip, sp, md, ms, ix, sx, dx, cx)
        });

        threaded_instr!($table_set_sr(
            ip: Ip,
            sp: Sp,
            md: Md,
            ms: Ms,
            ix: Ix,
            sx: Sx,
            dx: Dx,
            cx: Cx,
        ) -> ControlFlowBits {
            // Read operands
            let val = read_reg(ix, sx, dx);
            let (idx, ip) = read_stack(ip, sp);
            let (mut table, ip): (UnguardedTable, _) = read_imm(ip);

            // Perform operation
            r#try!(table
                .as_mut()
                .downcast_mut::<$T>()
                .unwrap_unchecked()
                .set(idx, val)
                .map_err(|_| Trap::TableAccessOutOfBounds));

            // Execute next instruction
            next_instr(ip, sp, md, ms, ix, sx, dx, cx)
        });

        threaded_instr!($table_set_si(
            ip: Ip,
            sp: Sp,
            md: Md,
            ms: Ms,
            ix: Ix,
            sx: Sx,
            dx: Dx,
            cx: Cx,
        ) -> ControlFlowBits {
            // Read operands
            let (val, ip) = read_imm(ip);
            let (idx, ip) = read_stack(ip, sp);
            let (mut table, ip): (UnguardedTable, _) = read_imm(ip);

            // Perform operation
            r#try!(table
                .as_mut()
                .downcast_mut::<$T>()
                .unwrap_unchecked()
                .set(idx, val)
                .map_err(|_| Trap::TableAccessOutOfBounds));

            // Execute next instruction
            next_instr(ip, sp, md, ms, ix, sx, dx, cx)
        });

        threaded_instr!($table_set_ri(
            ip: Ip,
            sp: Sp,
            md: Md,
            ms: Ms,
            ix: Ix,
            sx: Sx,
            dx: Dx,
            cx: Cx,
        ) -> ControlFlowBits {
            // Read operands
            let (val, ip) = read_imm(ip);
            let idx = read_reg(ix, sx, dx);
            let (mut table, ip): (UnguardedTable, _) = read_imm(ip);

            // Perform operation
            r#try!(table
                .as_mut()
                .downcast_mut::<$T>()
                .unwrap_unchecked()
                .set(idx, val)
                .map_err(|_| Trap::TableAccessOutOfBounds));

            // Execute next instruction
            next_instr(ip, sp, md, ms, ix, sx, dx, cx)
        });
    };
}

table_set!(
    table_set_func_ref_ss,
    table_set_func_ref_rs,
    table_set_func_ref_is,
    table_set_func_ref_ir,
    table_set_func_ref_ii,
    table_set_func_ref_sr,
    table_set_func_ref_si,
    table_set_func_ref_ri,
    UnguardedFuncRef
);
table_set!(
    table_set_extern_ref_ss,
    table_set_extern_ref_rs,
    table_set_extern_ref_is,
    table_set_extern_ref_ir,
    table_set_extern_ref_ii,
    table_set_extern_ref_sr,
    table_set_extern_ref_si,
    table_set_extern_ref_ri,
    UnguardedExternRef
);

macro_rules! table_size {
    ($table_size:ident, $T:ty) => {
        threaded_instr!($table_size(
            ip: Ip,
            sp: Sp,
            md: Md,
            ms: Ms,
            ix: Ix,
            sx: Sx,
            dx: Dx,
            cx: Cx,
        ) -> ControlFlowBits {
            // Read operands
            let (table, ip): (UnguardedTable, _) = read_imm(ip);

            // Perform operation
            let size = table
                .as_ref()
                .downcast_ref::<$T>()
                .unwrap_unchecked()
                .size();

            // Write result
            let ip = write_stack(ip, sp, size);

            // Execute next instruction
            next_instr(ip, sp, md, ms, ix, sx, dx, cx)
        });
    };
}

table_size!(table_size_func_ref, UnguardedFuncRef);
table_size!(table_size_extern_ref, UnguardedExternRef);

macro_rules! table_grow {
    ($table_grow:ident, $T:ty) => {
        threaded_instr!($table_grow(
            ip: Ip,
            sp: Sp,
            md: Md,
            ms: Ms,
            ix: Ix,
            sx: Sx,
            dx: Dx,
            cx: Cx,
        ) -> ControlFlowBits {
            // Read operands
            let (count, ip): (u32, _) = read_stack(ip, sp);
            let (val, ip) = read_stack(ip, sp);

            // Perform operation
            let (mut table, ip): (UnguardedTable, _) = read_imm(ip);

            // Perform operation
            let old_size = table
                .as_mut()
                .downcast_mut::<$T>()
                .unwrap_unchecked()
                .grow(val, count)
                .unwrap_or(u32::MAX);

            // Write result
            let ip = write_stack(ip, sp, old_size);

            // Execute next instruction
            next_instr(ip, sp, md, ms, ix, sx, dx, cx)
        });
    };
}

table_grow!(table_grow_func_ref, UnguardedFuncRef);
table_grow!(table_grow_extern_ref, UnguardedExternRef);

macro_rules! table_fill {
    ($table_fill:ident, $T:ty) => {
        threaded_instr!($table_fill(
            ip: Ip,
            sp: Sp,
            md: Md,
            ms: Ms,
            ix: Ix,
            sx: Sx,
            dx: Dx,
            cx: Cx,
        ) -> ControlFlowBits {
            // Read operands
            let (count, ip): (u32, _) = read_stack(ip, sp);
            let (val, ip) = read_stack(ip, sp);
            let (idx, ip): (u32, _) = read_stack(ip, sp);
            let (mut table, ip): (UnguardedTable, _) = read_imm(ip);

            // Perform operation
            r#try!(table
                .as_mut()
                .downcast_mut::<$T>()
                .unwrap_unchecked()
                .fill(idx, val, count));

            // Execute next instruction
            next_instr(ip, sp, md, ms, ix, sx, dx, cx)
        });
    };
}

table_fill!(table_fill_func_ref, UnguardedFuncRef);
table_fill!(table_fill_extern_ref, UnguardedExternRef);

macro_rules! table_copy {
    ($table_copy:ident, $T:ty) => {
        threaded_instr!($table_copy(
            ip: Ip,
            sp: Sp,
            md: Md,
            ms: Ms,
            ix: Ix,
            sx: Sx,
            dx: Dx,
            cx: Cx,
        ) -> ControlFlowBits {
            // Read operands
            let (count, ip): (u32, _) = read_stack(ip, sp);
            let (src_offset, ip): (u32, _) = read_stack(ip, sp);
            let (dst_offset, ip): (u32, _) = read_stack(ip, sp);
            let (mut dst_table, ip): (UnguardedTable, _) = read_imm(ip);
            let (src_table, ip): (UnguardedTable, _) = read_imm(ip);

            // Perform operation
            r#try!(if dst_table == src_table {
                dst_table
                    .as_mut()
                    .downcast_mut::<$T>()
                    .unwrap_unchecked()
                    .copy_within(dst_offset, src_offset, count)
            } else {
                dst_table
                    .as_mut()
                    .downcast_mut::<$T>()
                    .unwrap_unchecked()
                    .copy(
                        dst_offset,
                        src_table.as_ref().downcast_ref::<$T>().unwrap_unchecked(),
                        src_offset,
                        count,
                    )
            });

            // Execute next instruction
            next_instr(ip, sp, md, ms, ix, sx, dx, cx)
        });
    };
}

table_copy!(table_copy_func_ref, UnguardedFuncRef);
table_copy!(table_copy_extern_ref, UnguardedExternRef);

macro_rules! table_init {
    ($table_init:ident, $T:ty) => {
        threaded_instr!($table_init(
            ip: Ip,
            sp: Sp,
            md: Md,
            ms: Ms,
            ix: Ix,
            sx: Sx,
            dx: Dx,
            cx: Cx,
        ) -> ControlFlowBits {
            // Read operands
            let (count, ip): (u32, _) = read_stack(ip, sp);
            let (src_offset, ip): (u32, _) = read_stack(ip, sp);
            let (dst_offset, ip): (u32, _) = read_stack(ip, sp);
            let (mut dst_table, ip): (UnguardedTable, _) = read_imm(ip);
            let (src_elem, ip): (UnguardedElem, _) = read_imm(ip);

            // Perform operation
            r#try!(dst_table
                .as_mut()
                .downcast_mut::<$T>()
                .unwrap_unchecked()
                .init(
                    dst_offset,
                    src_elem.as_ref().downcast_ref::<$T>().unwrap_unchecked(),
                    src_offset,
                    count
                ));
            next_instr(ip, sp, md, ms, ix, sx, dx, cx)
        });
    };
}

table_init!(table_init_func_ref, UnguardedFuncRef);
table_init!(table_init_extern_ref, UnguardedExternRef);

macro_rules! elem_drop {
    ($elem_drop:ident, $T:ty) => {
        threaded_instr!($elem_drop(
            ip: Ip,
            sp: Sp,
            md: Md,
            ms: Ms,
            ix: Ix,
            sx: Sx,
            dx: Dx,
            cx: Cx,
        ) -> ControlFlowBits {
            // Read operands
            let (mut elem, ip): (UnguardedElem, _) = read_imm(ip);

            // Perform operation
            elem.as_mut()
                .downcast_mut::<$T>()
                .unwrap_unchecked()
                .drop_elems();

            // Execute next instruction
            next_instr(ip, sp, md, ms, ix, sx, dx, cx)
        });
    };
}

elem_drop!(elem_drop_func_ref, UnguardedFuncRef);
elem_drop!(elem_drop_extern_ref, UnguardedExternRef);

// Memory instructions

macro_rules! load {
    ($load_s:ident, $load_r:ident, $load_i:ident, $T:ty, $U:ty) => {
        threaded_instr!($load_s(
            ip: Ip,
            sp: Sp,
            md: Md,
            ms: Ms,
            ix: Ix,
            sx: Sx,
            dx: Dx,
            cx: Cx,
        ) -> ControlFlowBits {
            // Read operands
            let (dyn_offset, ip): (u32, _) = read_stack(ip, sp);
            let (static_offset, ip): (u32, _) = read_imm(ip);

            // Perform operation
            let offset = dyn_offset as u64 + static_offset as u64;
            if offset + mem::size_of::<$T>() as u64 > ms as u64 {
                return ControlFlow::Trap(Trap::MemAccessOutOfBounds).to_bits();
            }
            let mut bytes = [0u8; mem::size_of::<$T>()];
            ptr::copy_nonoverlapping(md.add(offset as usize), bytes.as_mut_ptr(), bytes.len());
            let y = <$T>::from_le_bytes(bytes) as $U;

            // Write result
            let (ix, sx, dx) = write_reg(ix, sx, dx, y);

            // Execute next instruction
            next_instr(ip, sp, md, ms, ix, sx, dx, cx)
        });

        threaded_instr!($load_r(
            ip: Ip,
            sp: Sp,
            md: Md,
            ms: Ms,
            ix: Ix,
            sx: Sx,
            dx: Dx,
            cx: Cx,
        ) -> ControlFlowBits {
            // Read operands
            let dyn_offset: u32 = read_reg(ix, sx, dx);
            let (static_offset, ip): (u32, _) = read_imm(ip);

            // Perform operation
            let offset = dyn_offset as u64 + static_offset as u64;
            if offset + mem::size_of::<$T>() as u64 > ms as u64 {
                return ControlFlow::Trap(Trap::MemAccessOutOfBounds).to_bits();
            }
            let mut bytes = [0u8; mem::size_of::<$T>()];
            ptr::copy_nonoverlapping(md.add(offset as usize), bytes.as_mut_ptr(), bytes.len());
            let y = <$T>::from_le_bytes(bytes) as $U;

            // Write result
            let (ix, sx, dx) = write_reg(ix, sx, dx, y);

            // Execute next instruction
            next_instr(ip, sp, md, ms, ix, sx, dx, cx)
        });

        threaded_instr!($load_i(
            ip: Ip,
            sp: Sp,
            md: Md,
            ms: Ms,
            ix: Ix,
            sx: Sx,
            dx: Dx,
            cx: Cx,
        ) -> ControlFlowBits {
            // Read operands
            let (dyn_offset, ip): (u32, _) = read_imm(ip);
            let (static_offset, ip): (u32, _) = read_imm(ip);

            // Perform operation
            let offset = dyn_offset as u64 + static_offset as u64;
            if offset + mem::size_of::<$T>() as u64 > ms as u64 {
                return ControlFlow::Trap(Trap::MemAccessOutOfBounds).to_bits();
            }
            let mut bytes = [0u8; mem::size_of::<$T>()];
            ptr::copy_nonoverlapping(md.add(offset as usize), bytes.as_mut_ptr(), bytes.len());
            let y = <$T>::from_le_bytes(bytes) as $U;

            // Write result
            let (ix, sx, dx) = write_reg(ix, sx, dx, y);

            // Execute next instruction
            next_instr(ip, sp, md, ms, ix, sx, dx, cx)
        });
    };
}

macro_rules! store {
    (
        $store_ss:ident,
        $store_rs:ident,
        $store_is:ident,
        $store_ir:ident,
        $store_ii:ident,
        $store_sr:ident,
        $store_si:ident,
        $store_ri:ident,
        $T:ty,
        $U:ty
    ) => {
        threaded_instr!($store_ss(
            ip: Ip,
            sp: Sp,
            md: Md,
            ms: Ms,
            ix: Ix,
            sx: Sx,
            dx: Dx,
            cx: Cx,
        ) -> ControlFlowBits {
            // Read operands
            let (x, ip): ($T, _) = read_stack(ip, sp);
            let (dyn_offset, ip): (u32, _) = read_stack(ip, sp);
            let (static_offset, ip): (u32, _) = read_imm(ip);

            // Perform operation
            let offset = dyn_offset as u64 + static_offset as u64;
            if offset + mem::size_of::<$U>() as u64 > ms as u64 {
                return ControlFlow::Trap(Trap::MemAccessOutOfBounds).to_bits();
            }
            let bytes = (x as $U).to_le_bytes();
            ptr::copy_nonoverlapping(bytes.as_ptr(), md.add(offset as usize), bytes.len());

            // Execute next instruction
            next_instr(ip, sp, md, ms, ix, sx, dx, cx)
        });

        threaded_instr!($store_rs(
            ip: Ip,
            sp: Sp,
            md: Md,
            ms: Ms,
            ix: Ix,
            sx: Sx,
            dx: Dx,
            cx: Cx,
        ) -> ControlFlowBits {
            // Read operands
            let (x, ip): ($T, _) = read_stack(ip, sp);
            let dyn_offset: u32 = read_reg(ix, sx, dx);
            let (static_offset, ip): (u32, _) = read_imm(ip);

            // Perform operation
            let offset = dyn_offset as u64 + static_offset as u64;
            if offset + mem::size_of::<$U>() as u64 > ms as u64 {
                return ControlFlow::Trap(Trap::MemAccessOutOfBounds).to_bits();
            }
            let bytes = (x as $U).to_le_bytes();
            ptr::copy_nonoverlapping(bytes.as_ptr(), md.add(offset as usize), bytes.len());

            // Execute next instruction
            next_instr(ip, sp, md, ms, ix, sx, dx, cx)
        });

        threaded_instr!($store_is(
            ip: Ip,
            sp: Sp,
            md: Md,
            ms: Ms,
            ix: Ix,
            sx: Sx,
            dx: Dx,
            cx: Cx,
        ) -> ControlFlowBits {
            // Read operands
            let (x, ip): ($T, _) = read_stack(ip, sp);
            let (dyn_offset, ip): (u32, _) = read_imm(ip);
            let (static_offset, ip): (u32, _) = read_imm(ip);

            // Perform operation
            let offset = dyn_offset as u64 + static_offset as u64;
            if offset + mem::size_of::<$U>() as u64 > ms as u64 {
                return ControlFlow::Trap(Trap::MemAccessOutOfBounds).to_bits();
            }
            let bytes = (x as $U).to_le_bytes();
            ptr::copy_nonoverlapping(bytes.as_ptr(), md.add(offset as usize), bytes.len());

            // Execute next instruction
            next_instr(ip, sp, md, ms, ix, sx, dx, cx)
        });

        threaded_instr!($store_ir(
            ip: Ip,
            sp: Sp,
            md: Md,
            ms: Ms,
            ix: Ix,
            sx: Sx,
            dx: Dx,
            cx: Cx,
        ) -> ControlFlowBits {
            // Read operands
            let x: $T = read_reg(ix, sx, dx);
            let (dyn_offset, ip): (u32, _) = read_imm(ip);
            let (static_offset, ip): (u32, _) = read_imm(ip);

            // Perform operation
            let offset = dyn_offset as u64 + static_offset as u64;
            if offset + mem::size_of::<$U>() as u64 > ms as u64 {
                return ControlFlow::Trap(Trap::MemAccessOutOfBounds).to_bits();
            }
            let bytes = (x as $U).to_le_bytes();
            ptr::copy_nonoverlapping(bytes.as_ptr(), md.add(offset as usize), bytes.len());

            // Execute next instruction
            next_instr(ip, sp, md, ms, ix, sx, dx, cx)
        });

        threaded_instr!($store_ii(
            ip: Ip,
            sp: Sp,
            md: Md,
            ms: Ms,
            ix: Ix,
            sx: Sx,
            dx: Dx,
            cx: Cx,
        ) -> ControlFlowBits {
            // Read operands
            let (x, ip): ($T, _) = read_imm(ip);
            let (dyn_offset, ip): (u32, _) = read_imm(ip);
            let (static_offset, ip): (u32, _) = read_imm(ip);

            // Perform operation
            let offset = dyn_offset as u64 + static_offset as u64;
            if offset + mem::size_of::<$U>() as u64 > ms as u64 {
                return ControlFlow::Trap(Trap::MemAccessOutOfBounds).to_bits();
            }
            let bytes = (x as $U).to_le_bytes();
            ptr::copy_nonoverlapping(bytes.as_ptr(), md.add(offset as usize), bytes.len());

            // Execute next instruction
            next_instr(ip, sp, md, ms, ix, sx, dx, cx)
        });

        threaded_instr!($store_sr(
            ip: Ip,
            sp: Sp,
            md: Md,
            ms: Ms,
            ix: Ix,
            sx: Sx,
            dx: Dx,
            cx: Cx,
        ) -> ControlFlowBits {
            // Read operands
            let x: $T = read_reg(ix, sx, dx);
            let (dyn_offset, ip): (u32, _) = read_stack(ip, sp);
            let (static_offset, ip): (u32, _) = read_imm(ip);

            // Perform operation
            let offset = dyn_offset as u64 + static_offset as u64;
            if offset + mem::size_of::<$U>() as u64 > ms as u64 {
                return ControlFlow::Trap(Trap::MemAccessOutOfBounds).to_bits();
            }
            let bytes = (x as $U).to_le_bytes();
            ptr::copy_nonoverlapping(bytes.as_ptr(), md.add(offset as usize), bytes.len());

            // Execute next instruction
            next_instr(ip, sp, md, ms, ix, sx, dx, cx)
        });

        threaded_instr!($store_si(
            ip: Ip,
            sp: Sp,
            md: Md,
            ms: Ms,
            ix: Ix,
            sx: Sx,
            dx: Dx,
            cx: Cx,
        ) -> ControlFlowBits {
            // Read operands
            let (x, ip): ($T, _) = read_imm(ip);
            let (dyn_offset, ip): (u32, _) = read_stack(ip, sp);
            let (static_offset, ip): (u32, _) = read_imm(ip);

            // Perform operation
            let offset = dyn_offset as u64 + static_offset as u64;
            if offset + mem::size_of::<$U>() as u64 > ms as u64 {
                return ControlFlow::Trap(Trap::MemAccessOutOfBounds).to_bits();
            }
            let bytes = (x as $U).to_le_bytes();
            ptr::copy_nonoverlapping(bytes.as_ptr(), md.add(offset as usize), bytes.len());

            // Execute next instruction
            next_instr(ip, sp, md, ms, ix, sx, dx, cx)
        });

        threaded_instr!($store_ri(
            ip: Ip,
            sp: Sp,
            md: Md,
            ms: Ms,
            ix: Ix,
            sx: Sx,
            dx: Dx,
            cx: Cx,
        ) -> ControlFlowBits {
            // Read operands
            let (x, ip): ($T, _) = read_imm(ip);
            let dyn_offset: u32 = read_reg(ix, sx, dx);
            let (static_offset, ip): (u32, _) = read_imm(ip);

            // Perform operation
            let offset = dyn_offset as u64 + static_offset as u64;
            if offset + mem::size_of::<$U>() as u64 > ms as u64 {
                return ControlFlow::Trap(Trap::MemAccessOutOfBounds).to_bits();
            }
            let bytes = (x as $U).to_le_bytes();
            ptr::copy_nonoverlapping(bytes.as_ptr(), md.add(offset as usize), bytes.len());

            // Execute next instruction
            next_instr(ip, sp, md, ms, ix, sx, dx, cx)
        });
    };
}

macro_rules! store_float {
    (
        $store_ss:ident,
        $store_rs:ident,
        $store_is:ident,
        $store_ir:ident,
        $store_ii:ident,
        $store_sr:ident,
        $store_si:ident,
        $store_ri:ident,
        $store_rr:ident,
        $T:ty,
        $U:ty
    ) => {
        store!(
            $store_ss, $store_rs, $store_is, $store_ir, $store_ii, $store_sr, $store_si, $store_ri,
            $T, $U
        );

        threaded_instr!($store_rr(
            ip: Ip,
            sp: Sp,
            md: Md,
            ms: Ms,
            ix: Ix,
            sx: Sx,
            dx: Dx,
            cx: Cx,
        ) -> ControlFlowBits {
            // Read operands
            let x: $T = read_reg(ix, sx, dx);
            let dyn_offset: u32 = read_reg(ix, sx, dx);
            let (static_offset, ip): (u32, _) = read_imm(ip);

            // Perform operation
            let offset = dyn_offset as u64 + static_offset as u64;
            if offset + mem::size_of::<$U>() as u64 > ms as u64 {
                return ControlFlow::Trap(Trap::MemAccessOutOfBounds).to_bits();
            }
            let bytes = (x as $U).to_le_bytes();
            ptr::copy_nonoverlapping(bytes.as_ptr(), md.add(offset as usize), bytes.len());

            // Execute next instruction
            next_instr(ip, sp, md, ms, ix, sx, dx, cx)
        });
    };
}

load!(i32_load_s, i32_load_r, i32_load_i, i32, i32);
load!(i64_load_s, i64_load_r, i64_load_i, i64, i64);
load!(f32_load_s, f32_load_r, f32_load_i, f32, f32);
load!(f64_load_s, f64_load_r, f64_load_i, f64, f64);
load!(i32_load8_s_s, i32_load8_s_r, i32_load8_s_i, i8, i32);
load!(i32_load8_u_s, i32_load8_u_r, i32_load8_u_i, u8, u32);
load!(i32_load16_s_s, i32_load16_s_r, i32_load16_s_i, i16, i32);
load!(i32_load16_u_s, i32_load16_u_r, i32_load16_u_i, u16, u32);
load!(i64_load8_s_s, i64_load8_s_r, i64_load8_s_i, i8, i64);
load!(i64_load8_u_s, i64_load8_u_r, i64_load8_u_i, u8, u64);
load!(i64_load16_s_s, i64_load16_s_r, i64_load16_s_i, i16, i64);
load!(i64_load16_u_s, i64_load16_u_r, i64_load16_u_i, u16, u64);
load!(i64_load32_s_s, i64_load32_s_r, i64_load32_s_i, i32, i64);
load!(i64_load32_u_s, i64_load32_u_r, i64_load32_u_i, u32, u64);
store!(
    i32_store_ss,
    i32_store_rs,
    i32_store_is,
    i32_store_ir,
    i32_store_ii,
    i32_store_sr,
    i32_store_si,
    i32_store_ri,
    i32,
    i32
);
store!(
    i64_store_ss,
    i64_store_rs,
    i64_store_is,
    i64_store_ir,
    i64_store_ii,
    i64_store_sr,
    i64_store_si,
    i64_store_ri,
    i64,
    i64
);
store_float!(
    f32_store_ss,
    f32_store_rs,
    f32_store_is,
    f32_store_ir,
    f32_store_ii,
    f32_store_sr,
    f32_store_si,
    f32_store_ri,
    f32_store_rr,
    f32,
    f32
);
store_float!(
    f64_store_ss,
    f64_store_rs,
    f64_store_is,
    f64_store_ir,
    f64_store_ii,
    f64_store_sr,
    f64_store_si,
    f64_store_ri,
    f64_store_rr,
    f64,
    f64
);
store!(
    i32_store8_ss,
    i32_store8_rs,
    i32_store8_is,
    i32_store8_ir,
    i32_store8_ii,
    i32_store8_sr,
    i32_store8_si,
    i32_store8_ri,
    u32,
    u8
);
store!(
    i32_store16_ss,
    i32_store16_rs,
    i32_store16_is,
    i32_store16_ir,
    i32_store16_ii,
    i32_store16_sr,
    i32_store16_si,
    i32_store16_ri,
    u32,
    u16
);
store!(
    i64_store8_ss,
    i64_store8_rs,
    i64_store8_is,
    i64_store8_ir,
    i64_store8_ii,
    i64_store8_sr,
    i64_store8_si,
    i64_store8_ri,
    u64,
    u8
);
store!(
    i64_store16_ss,
    i64_store16_rs,
    i64_store16_is,
    i64_store16_ir,
    i64_store16_ii,
    i64_store16_sr,
    i64_store16_si,
    i64_store16_ri,
    u64,
    u16
);
store!(
    i64_store32_ss,
    i64_store32_rs,
    i64_store32_is,
    i64_store32_ir,
    i64_store32_ii,
    i64_store32_sr,
    i64_store32_si,
    i64_store32_ri,
    u64,
    u32
);

threaded_instr!(memory_size(
    ip: Ip,
    sp: Sp,
    md: Md,
    ms: Ms,
    ix: Ix,
    sx: Sx,
    dx: Dx,
    cx: Cx,
) -> ControlFlowBits {
    // Read operands
    let (mem, ip): (UnguardedMem, _) = read_imm(ip);

    // Perform operation
    let size = mem.as_ref().size();

    // Write result
    let ip = write_stack(ip, sp, size);

    // Execute next instruction
    next_instr(ip, sp, md, ms, ix, sx, dx, cx)
});

threaded_instr!(memory_grow(
    ip: Ip,
    sp: Sp,
    _md: Md,
    _ms: Ms,
    ix: Ix,
    sx: Sx,
    dx: Dx,
    cx: Cx,
) -> ControlFlowBits {
    // Read operands
    let (count, ip): (u32, _) = read_stack(ip, sp);
    let (mut mem, ip): (UnguardedMem, _) = read_imm(ip);

    // Perform operation
    (*cx).stack.as_mut().unwrap_unchecked().set_ptr(sp);
    let old_size = mem
        .as_mut()
        .grow_with_stack(count, (*cx).stack.as_mut().unwrap_unchecked())
        .unwrap_or(u32::MAX);
    let bytes = mem.as_mut().bytes_mut();
    let md = bytes.as_mut_ptr();
    let ms = bytes.len() as u32;

    // Write result
    let ip = write_stack(ip, sp, old_size);

    // Execute next instruction
    next_instr(ip, sp, md, ms, ix, sx, dx, cx)
});

threaded_instr!(memory_fill(
    ip: Ip,
    sp: Sp,
    md: Md,
    ms: Ms,
    ix: Ix,
    sx: Sx,
    dx: Dx,
    cx: Cx,
) -> ControlFlowBits {
    // Read operands
    let (count, ip) = read_stack(ip, sp);
    let (val, ip): (u32, _) = read_stack(ip, sp);
    let (idx, ip) = read_stack(ip, sp);
    let (mut mem, ip): (UnguardedMem, _) = read_imm(ip);

    // Perform operation
    r#try!(mem.as_mut().fill(idx, val as u8, count));

    // Execute next instruction
    next_instr(ip, sp, md, ms, ix, sx, dx, cx)
});

threaded_instr!(memory_copy(
    ip: Ip,
    sp: Sp,
    md: Md,
    ms: Ms,
    ix: Ix,
    sx: Sx,
    dx: Dx,
    cx: Cx,
) -> ControlFlowBits {
    // Read operands
    let (count, ip): (u32, _) = read_stack(ip, sp);
    let (src_idx, ip): (u32, _) = read_stack(ip, sp);
    let (dst_idx, ip): (u32, _) = read_stack(ip, sp);
    let (mut mem, ip): (UnguardedMem, _) = read_imm(ip);

    // Perform operation
    r#try!(mem.as_mut().copy_within(dst_idx, src_idx, count));

    // Execute next instruction
    next_instr(ip, sp, md, ms, ix, sx, dx, cx)
});

threaded_instr!(memory_init(
    ip: Ip,
    sp: Sp,
    md: Md,
    ms: Ms,
    ix: Ix,
    sx: Sx,
    dx: Dx,
    cx: Cx,
) -> ControlFlowBits {
    // Read operands
    let (count, ip): (u32, _) = read_stack(ip, sp);
    let (src_idx, ip): (u32, _) = read_stack(ip, sp);
    let (dst_idx, ip): (u32, _) = read_stack(ip, sp);
    let (mut dst_mem, ip): (UnguardedMem, _) = read_imm(ip);
    let (src_data, ip): (UnguardedData, _) = read_imm(ip);

    // Perform operation
    r#try!(dst_mem
        .as_mut()
        .init(dst_idx, src_data.as_ref(), src_idx, count));

    // Execute next instruction
    next_instr(ip, sp, md, ms, ix, sx, dx, cx)
});

threaded_instr!(data_drop(
    ip: Ip,
    sp: Sp,
    md: Md,
    ms: Ms,
    ix: Ix,
    sx: Sx,
    dx: Dx,
    cx: Cx,
) -> ControlFlowBits {
    // Read operands
    let (mut data, ip): (UnguardedData, _) = read_imm(ip);

    // Perform operation
    data.as_mut().drop_bytes();

    // Execute next instruction
    next_instr(ip, sp, md, ms, ix, sx, dx, cx)
});

// Numeric instructions

pub(crate) unsafe extern "C" fn un_op<T, U, R, W>(
    ip: Ip,
    sp: Sp,
    md: Md,
    ms: Ms,
    ix: Ix,
    sx: Sx,
    dx: Dx,
    cx: Cx,
) -> ControlFlowBits
where
    U: UnOp<T>,
    R: Read<T>,
    W: Write<U::Output>,
{
    let mut args = Args::from_parts(ip, sp, md, ms, ix, sx, dx, cx);
    unsafe {
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
    ix: Ix,
    sx: Sx,
    dx: Dx,
    cx: Cx,
) -> ControlFlowBits
where
    B: BinOp<T>,
    R0: Read<T>,
    R1: Read<T>,
    W: Write<B::Output>,
{
    let mut args = Args::from_parts(ip, sp, md, ms, ix, sx, dx, cx);
    unsafe {
        let x1 = R1::read(&mut args);
        let x0 = R0::read(&mut args);
        let y = r#try!(B::bin_op(x0, x1));
        W::write(&mut args, y);
        args.next()
    }
}

// Miscellaneous instructions

macro_rules! copy_imm_to_stack {
    ($copy_imm_to_stack:ident, $T:ty) => {
        threaded_instr!($copy_imm_to_stack(
            ip: Ip,
            sp: Sp,
            md: Md,
            ms: Ms,
            ix: Ix,
            sx: Sx,
            dx: Dx,
            cx: Cx,
        ) -> ControlFlowBits {
            // Read immediate value
            let (x, ip): ($T, _) = read_imm(ip);

            // Write value to stack
            let ip = write_stack(ip, sp, x);

            // Execute next instruction
            next_instr(ip, sp, md, ms, ix, sx, dx, cx)
        });
    };
}

copy_imm_to_stack!(copy_imm_to_stack_i32, i32);
copy_imm_to_stack!(copy_imm_to_stack_i64, i64);
copy_imm_to_stack!(copy_imm_to_stack_f32, f32);
copy_imm_to_stack!(copy_imm_to_stack_f64, f64);
copy_imm_to_stack!(copy_imm_to_stack_func_ref, UnguardedFuncRef);
copy_imm_to_stack!(copy_imm_to_stack_extern_ref, UnguardedExternRef);

macro_rules! copy_stack {
    ($copy_stack_t:ident, $T:ty) => {
        threaded_instr!($copy_stack_t(
            ip: Ip,
            sp: Sp,
            md: Md,
            ms: Ms,
            ix: Ix,
            sx: Sx,
            dx: Dx,
            cx: Cx,
        ) -> ControlFlowBits {
            // Read value from stack
            let (x, ip): ($T, _) = read_stack(ip, sp);

            // Write value to stack
            let ip = write_stack(ip, sp, x);

            // Execute next instruction
            next_instr(ip, sp, md, ms, ix, sx, dx, cx)
        });
    };
}

copy_stack!(copy_stack_i32, i32);
copy_stack!(copy_stack_i64, i64);
copy_stack!(copy_stack_f32, f32);
copy_stack!(copy_stack_f64, f64);
copy_stack!(copy_stack_func_ref, UnguardedFuncRef);
copy_stack!(copy_stack_extern_ref, UnguardedExternRef);

macro_rules! copy_reg_to_stack {
    ($copy_reg_to_stack_t:ident, $T:ty) => {
        threaded_instr!($copy_reg_to_stack_t(
            ip: Ip,
            sp: Sp,
            md: Md,
            ms: Ms,
            ix: Ix,
            sx: Sx,
            dx: Dx,
            cx: Cx,
        ) -> ControlFlowBits {
            // Read value from register
            let x: $T = read_reg(ix, sx, dx);

            // Write value to stack
            let ip = write_stack(ip, sp, x);

            // Execute next instruction
            next_instr(ip, sp, md, ms, ix, sx, dx, cx)
        });
    };
}

copy_reg_to_stack!(copy_reg_to_stack_i32, i32);
copy_reg_to_stack!(copy_reg_to_stack_i64, i64);
copy_reg_to_stack!(copy_reg_to_stack_f32, f32);
copy_reg_to_stack!(copy_reg_to_stack_f64, f64);
copy_reg_to_stack!(copy_reg_to_stack_func_ref, UnguardedFuncRef);
copy_reg_to_stack!(copy_reg_to_stack_extern_ref, UnguardedExternRef);

threaded_instr!(stop(
    _ip: Ip,
    _sp: Sp,
    _md: Md,
    _ms: Ms,
    _ix: Ix,
    _sx: Sx,
    _dx: Dx,
    _cx: Cx,
) -> ControlFlowBits {
    ControlFlow::Stop.to_bits()
});

threaded_instr!(compile(
    ip: Ip,
    sp: Sp,
    md: Md,
    ms: Ms,
    ix: Ix,
    sx: Sx,
    dx: Dx,
    cx: Cx,
) -> ControlFlowBits {
    let mut func: UnguardedFunc = *ip.cast();
    Func(Handle::from_unguarded(func, (*(*cx).store).id())).compile((*cx).store);
    let FuncEntity::Wasm(func) = func.as_mut() else {
        hint::unreachable_unchecked();
    };
    let Code::Compiled(state) = func.code_mut() else {
        hint::unreachable_unchecked();
    };
    *ip.cast() = state.code.as_mut_ptr();
    let ip = ip.offset(-1);
    *ip.cast() = call_wasm as ThreadedInstr;
    next_instr(ip, sp, md, ms, ix, sx, dx, cx)
});

threaded_instr!(enter(
    ip: Ip,
    sp: Sp,
    _md: Md,
    _ms: Ms,
    ix: Ix,
    sx: Sx,
    dx: Dx,
    cx: Cx,
) -> ControlFlowBits {
    let (func, ip): (UnguardedFunc, _) = read_imm(ip);
    let (mem, ip): (Option<UnguardedMem>, _) = read_imm(ip);
    let FuncEntity::Wasm(func) = func.as_ref() else {
        hint::unreachable_unchecked();
    };
    let Code::Compiled(code) = func.code() else {
        hint::unreachable_unchecked();
    };

    // Check that the stack has enough space.
    let stack_height = sp.offset_from((*cx).stack.as_mut().unwrap_unchecked().base_ptr()) as usize;
    if code.max_stack_height > Stack::SIZE - stack_height {
        return ControlFlow::Trap(Trap::StackOverflow).to_bits();
    }

    // Initialize the locals for this function to their default values.
    ptr::write_bytes(sp, 0, code.local_count);

    let md;
    let ms;
    if let Some(mut mem) = mem {
        let data = mem.as_mut().bytes_mut();
        md = data.as_mut_ptr();
        ms = data.len() as u32;
    } else {
        md = ptr::null_mut();
        ms = 0;
    }

    // Execute the next instruction.
    next_instr(ip, sp, md, ms, ix, sx, dx, cx)
});

// Helper functions

pub(crate) struct Args<'a> {
    ip_base: Ip,
    ip_offset: usize,
    sp: Sp,
    md: Md,
    ms: Ms,
    ix: Ix,
    sx: Sx,
    dx: Dx,
    cx: Cx<'a>,
}

impl<'a> Args<'a> {
    fn from_parts(
        ip: Ip,
        sp: Sp,
        md: Md,
        ms: Ms,
        ix: Ix,
        sx: Sx,
        dx: Dx,
        cx: Cx<'a>,
    ) -> Self {
        Self {
            ip_base: ip,
            ip_offset: 0,
            sp,
            md,
            ms,
            ix,
            sx,
            dx,
            cx,
        }
    }

    fn into_parts(self) -> (Ip, Sp, Md, Ms, Ix, Sx, Dx, Cx<'a>) {
        (
            self.ip(),
            self.sp,
            self.md,
            self.ms,
            self.ix,
            self.sx,
            self.dx,
            self.cx,
        )
    }

    fn ip(&self) -> Ip {
        unsafe { self.ip_base.add(self.ip_offset) }
    }

    unsafe fn advance_ip(&mut self, count: usize) {
        self.ip_offset += count;
    }

    fn set_ip(&mut self, ip: Ip) {
        self.ip_base = ip;
        self.ip_offset = 0;
    }

    unsafe fn read_imm<T>(&mut self) -> T {
        unsafe {
            let val = ptr::read(self.ip().cast());
            self.advance_ip(1);
            val
        }
    }

    unsafe fn read_stk<T>(&mut self) -> T {
        unsafe {
            let offset = self.read_imm();
            ptr::read(self.sp.cast::<u8>().offset(offset).cast::<T>())
        }
    }

    unsafe fn write_stk<T>(&mut self, val: T) {
        unsafe {
            let offset = self.read_imm();
            ptr::write(self.sp.cast::<u8>().offset(offset).cast::<T>(), val)
        }
    }

    fn read_reg<T>(&self) -> T
    where
        T: ReadReg,
    {
        read_reg(self.ix, self.sx, self.dx)
    }

    fn write_reg<T>(&mut self, val: T)
    where
        T: WriteReg,
    {
        let (ix, sx, dx) = write_reg(self.ix, self.sx, self.dx, val);
        self.ix = ix;
        self.sx = sx;
        self.dx = dx;
    }
    
    unsafe fn next(mut self) -> ControlFlowBits {
        unsafe {
            let instr: ThreadedInstr = self.read_imm();
            let (ip, sp, md, ms, ix, sx, dx, cx) = self.into_parts();
            (instr)(ip, sp, md, ms, ix, sx, dx, cx)
        }
    }
}

pub(crate) trait Read<T>: Sized {
    unsafe fn read(args: &mut Args) -> T;
}

pub(crate) trait Write<T>: Sized {
    unsafe fn write(args: &mut Args, val: T);
}

pub(crate) struct Imm;

impl<T> Read<T> for Imm {
    unsafe fn read(args: &mut Args) -> T {
        unsafe { args.read_imm() }
    }
}

pub(crate) struct Stk;

impl<T> Read<T> for Stk {
    unsafe fn read(args: &mut Args) -> T {
        unsafe { args.read_stk() }
    }
}

impl<T> Write<T> for Stk {
    unsafe fn write(args: &mut Args, val: T) {
        unsafe { args.write_stk(val) }
    }
}

pub(crate) struct Reg;

impl<T> Read<T> for Reg
where
    T: ReadReg,
{
    unsafe fn read(args: &mut Args) -> T {
        args.read_reg()
    }
}

impl<T> Write<T> for Reg
where 
    T: WriteReg
{
    unsafe fn write(args: &mut Args, val: T) {
        args.write_reg(val)
    }
}

/// Executes the next instruction.
pub(crate) unsafe fn next_instr(
    ip: Ip,
    sp: Sp,
    md: Md,
    ms: Ms,
    ix: Ix,
    sx: Sx,
    dx: Dx,
    cx: Cx,
) -> ControlFlowBits {
    let (instr, ip): (ThreadedInstr, _) = read_imm(ip);
    (instr)(ip, sp, md, ms, ix, sx, dx, cx)
}

/// Reads an immediate value.
unsafe fn read_imm<T>(ip: Ip) -> (T, Ip)
where
    T: Copy,
{
    let val = *ip.cast();
    let ip = ip.add(1);
    (val, ip)
}

/// Reads a value from the stack.
unsafe fn read_stack<T>(ip: Ip, sp: Sp) -> (T, Ip)
where
    T: Copy + std::fmt::Debug,
{
    let (offset, ip) = read_imm(ip);
    // The cast to `u8` is because stack offsets are premultiplied, which allows us to avoid
    // generating a shift instruction on some platforms.
    let x = *sp.cast::<u8>().offset(offset).cast::<T>();
    (x, ip)
}

/// Writes a value to the stack.
unsafe fn write_stack<T>(ip: Ip, sp: Sp, x: T) -> Ip
where
    T: Copy + std::fmt::Debug,
{
    let (offset, ip) = read_imm(ip);
    // The cast to `u8` is because stack offsets are premultiplied, which allows us to avoid
    // generating a shift instruction on some platforms.
    *sp.cast::<u8>().offset(offset).cast() = x;
    ip
}

/// Reads a value from a register.
fn read_reg<T>(ix: Ix, sx: Sx, dx: Dx) -> T
where
    T: ReadReg,
{
    T::read_reg(ix, sx, dx)
}

pub(crate) trait ReadReg {
    fn read_reg(ix: Ix, _sx: Sx, _dx: Dx) -> Self;
}

impl ReadReg for i32 {
    fn read_reg(ix: Ix, _sx: Sx, _dx: Dx) -> Self {
        ix as i32
    }
}

impl ReadReg for u32 {
    fn read_reg(ix: Ix, _sx: Sx, _dx: Dx) -> Self {
        ix as u32
    }
}

impl ReadReg for i64 {
    fn read_reg(ix: Ix, _sx: Sx, _dx: Dx) -> Self {
        ix as i64
    }
}

impl ReadReg for u64 {
    fn read_reg(ix: Ix, _sx: Sx, _dx: Dx) -> Self {
        ix as u64
    }
}

impl ReadReg for f32 {
    fn read_reg(_ix: Ix, sx: Sx, _dx: Dx) -> Self {
        sx
    }
}

impl ReadReg for f64 {
    fn read_reg(_ix: Ix, _sx: Sx, dx: Dx) -> Self {
        dx
    }
}

impl ReadReg for UnguardedFuncRef {
    fn read_reg(ix: Ix, _sx: Sx, _dx: Dx) -> Self {
        UnguardedFunc::new(ix as *mut _)
    }
}

impl ReadReg for UnguardedExternRef {
    fn read_reg(ix: Ix, _sx: Sx, _dx: Dx) -> Self {
        UnguardedExtern::new(ix as *mut _)
    }
}

// Writes a value to a register.
fn write_reg<T>(ix: Ix, sx: Sx, dx: Dx, x: T) -> (Ix, Sx, Dx)
where
    T: WriteReg,
{
    T::write_reg(ix, sx, dx, x)
}

pub(crate) trait WriteReg {
    fn write_reg(ix: Ix, sx: Sx, dx: Dx, x: Self) -> (Ix, Sx, Dx);
}

impl WriteReg for i32 {
    fn write_reg(_ix: Ix, sx: Sx, dx: Dx, x: Self) -> (Ix, Sx, Dx) {
        // Casting to `u32` first allows us to avoid generating a sign extension instruction on some
        // platforms.
        (x as u32 as Ix, sx, dx)
    }
}

impl WriteReg for u32 {
    fn write_reg(_ix: Ix, sx: Sx, dx: Dx, x: Self) -> (Ix, Sx, Dx) {
        (x as Ix, sx, dx)
    }
}

impl WriteReg for i64 {
    fn write_reg(_ix: Ix, sx: Sx, dx: Dx, x: Self) -> (Ix, Sx, Dx) {
        (x as Ix, sx, dx)
    }
}

impl WriteReg for u64 {
    fn write_reg(_ix: Ix, sx: Sx, dx: Dx, x: Self) -> (Ix, Sx, Dx) {
        (x as Ix, sx, dx)
    }
}

impl WriteReg for f32 {
    fn write_reg(ix: Ix, _sx: Sx, dx: Dx, x: Self) -> (Ix, Sx, Dx) {
        (ix, x, dx)
    }
}

impl WriteReg for f64 {
    fn write_reg(ix: Ix, sx: Sx, _dx: Dx, x: Self) -> (Ix, Sx, Dx) {
        (ix, sx, x)
    }
}

impl WriteReg for UnguardedFuncRef {
    fn write_reg(_ix: Ix, sx: Sx, dx: Dx, x: Self) -> (Ix, Sx, Dx) {
        (x.map_or(ptr::null_mut(), |ptr| ptr.as_ptr()) as Ix, sx, dx)
    }
}

impl WriteReg for UnguardedExternRef {
    fn write_reg(_ix: Ix, sx: Sx, dx: Dx, x: Self) -> (Ix, Sx, Dx) {
        (x.map_or(ptr::null_mut(), |ptr| ptr.as_ptr()) as Ix, sx, dx)
    }
}
