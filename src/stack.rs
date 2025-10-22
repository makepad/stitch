use std::{
    alloc::{alloc_zeroed, dealloc, handle_alloc_error, Layout},
    cell::Cell,
    mem::ManuallyDrop,
    ops::{Deref, DerefMut},
    ptr::NonNull,
};

pub(crate) const ALIGN: usize = 8;

#[derive(Debug)]
pub struct Stack {
    ptr: NonNull<u8>,
    len: usize,
    capacity: usize,
}

impl Stack {
    /// Locks the [`Stack`] for the current thread, returning a [`StackGuard`].
    pub fn lock() -> StackGuard {
        StackGuard {
            stack: ManuallyDrop::new(STACK.take().unwrap()),
        }
    }

    pub(crate) fn new(capacity: usize) -> Self {
        let ptr = if capacity == 0 {
            NonNull::dangling()
        } else {
            let layout = Layout::from_size_align(capacity, ALIGN).unwrap();
            let ptr = unsafe { alloc_zeroed(layout) };
            NonNull::new(ptr).unwrap_or_else(|| handle_alloc_error(layout))
        };
        Self {
            ptr,
            len: 0,
            capacity,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }

    pub(crate) unsafe fn set_len(&mut self, new_len: usize) {
        self.len = new_len;
    }

    pub(crate) fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    pub(crate) fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }
}

impl Drop for Stack {
    fn drop(&mut self) {
        if self.capacity != 0 {
            unsafe {
                let layout = Layout::from_size_align_unchecked(self.capacity, ALIGN);
                dealloc(self.ptr.as_ptr(), layout)
            }
        }
    }
}

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

thread_local! {
    static STACK: Cell<Option<Stack>> = Cell::new(Some(Stack::new(8 * 1024 * 1024)));
}

pub(crate) fn padded_size_of<T>() -> usize {
    let layout = Layout::from_size_align(size_of::<T>(), ALIGN).unwrap();
    layout.pad_to_align().size()
}