use {
    crate::aliasable_box::AliasableBox,
    std::{
        cell::Cell,
        mem::ManuallyDrop,
        ops::{Deref, DerefMut},
        ptr,
    },
};

/// A stack for executing threaded code.
///
/// There is exactly one stack per thread. To obtain a mutable reference to the stack for the
/// current thread, call [`Stack::lock`].
///
/// Each stack consists of an array of [`StackSlot`]s with a pointer to the top of the stack,
/// and contains both [`UnguardedVal`](crate::val::UnguardedVal)s and call frames. An
/// [`UnguardedVal`](crate::val::UnguardedVal) takes up a single [`StackSlot`], while a call frame
/// takes up multiple stack slots.
#[derive(Debug)]
pub struct Stack {
    slots: AliasableBox<[StackSlot]>,
    ptr: *mut StackSlot,
}

impl Stack {
    /// The size of the stack in bytes.
    pub(crate) const SIZE: usize = 1024 * 1024;

    /// Locks the [`Stack`] for the current thread, returning a [`StackGuard`].
    pub fn lock() -> StackGuard {
        StackGuard {
            stack: ManuallyDrop::new(STACK.take().unwrap()),
        }
    }

    /// Returns a pointer to the base of the stack.
    pub(crate) fn base_ptr(&mut self) -> *mut StackSlot {
        self.slots.as_mut_ptr() as *mut _
    }

    /// Returns a pointer to the top of the stack.
    pub fn ptr(&mut self) -> *mut StackSlot {
        self.ptr
    }

    /// Sets the pointer to the top of the stack.
    ///
    /// # Safety
    ///
    /// The pointer must be within bounds.
    pub(crate) unsafe fn set_ptr(&mut self, ptr: *mut StackSlot) {
        self.ptr = ptr;
    }

    /// Creates a new, zero-initialized [`Stack`].
    fn new() -> Self {
        let mut stack = Self {
            slots: AliasableBox::from_box(Box::from(vec![0; Self::SIZE])),
            ptr: ptr::null_mut(),
        };
        stack.ptr = stack.slots.as_mut_ptr();
        stack
    }
}

/// An RAII implementation of a "scoped lock" of a [`Stack`].
///
/// When this structure is dropped (falls out of scope), the lock will be unlocked.
#[derive(Debug)]
pub struct StackGuard {
    stack: ManuallyDrop<Stack>,
}

impl Deref for StackGuard {
    type Target = Stack;

    fn deref(&self) -> &Self::Target {
        &self.stack
    }
}

impl DerefMut for StackGuard {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.stack
    }
}

impl Drop for StackGuard {
    fn drop(&mut self) {
        STACK.set(Some(unsafe { ManuallyDrop::take(&mut self.stack) }));
    }
}

pub(crate) type StackSlot = u64;

thread_local! {
    static STACK: Cell<Option<Stack>> = Cell::new(Some(Stack::new()));
}
