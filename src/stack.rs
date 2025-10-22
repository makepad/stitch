use std::{
    alloc::{alloc_zeroed, dealloc, handle_alloc_error, Layout},
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

pub(crate) fn padded_size_of<T>() -> usize {
    let layout = Layout::from_size_align(size_of::<T>(), ALIGN).unwrap();
    layout.pad_to_align().size()
}