use std::{
    alloc::{alloc, dealloc, handle_alloc_error, realloc, Layout},
    fmt,
    mem::ManuallyDrop,
    ptr::NonNull,
};

pub(crate) const ALIGN: usize = align_of::<usize>();

pub(crate) struct Code {
    ptr: NonNull<u8>,
    len: usize,
}

impl Code {
    pub(crate) fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    pub(crate) fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }
}

impl fmt::Debug for Code {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Code").finish_non_exhaustive()
    }
}

impl Drop for Code {
    fn drop(&mut self) {
        if self.len != 0 {
            unsafe {
                let layout = Layout::from_size_align_unchecked(self.len, ALIGN);
                dealloc(self.ptr.as_ptr(), layout)
            }
        }
    }
}

pub(crate) struct CodeBuilder {
    ptr: NonNull<u8>,
    len: usize,
    capacity: usize,
}

impl CodeBuilder {
    pub(crate) fn new() -> Self {
        Self {
            ptr: NonNull::dangling(),
            len: 0,
            capacity: 0,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    pub(crate) fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    pub(crate) fn reserve(&mut self, additional: usize) {
        if additional > self.capacity - self.len {
            self.reserve_cold(additional);
        }
    }

    #[cold]
    fn reserve_cold(&mut self, additional: usize) {
        let new_capacity = self.len.checked_add(additional).unwrap();
        let new_capacity = new_capacity.max(self.capacity * 2);
        let new_capacity = new_capacity.max(8);
        let new_layout = Layout::from_size_align(new_capacity, ALIGN).unwrap();
        let new_ptr = if self.capacity == 0 {
            unsafe { alloc(new_layout) }
        } else {
            unsafe {
                let layout = Layout::from_size_align_unchecked(self.capacity, ALIGN);
                realloc(self.ptr.as_ptr(), layout, new_layout.size())
            }
        };
        let new_ptr = NonNull::new(new_ptr).unwrap_or_else(|| handle_alloc_error(new_layout));
        self.ptr = new_ptr;
        self.capacity = new_capacity;
    }

    pub(crate) fn push<T>(&mut self, val: T) -> usize {
        self.pad_to_align(align_of::<T>());
        unsafe { self.push_aligned(val) }
    }

    pub(crate) fn pad_to_align(&mut self, align: usize) -> usize {
        let layout = Layout::from_size_align(self.len, align).unwrap();
        let padded_len = layout.pad_to_align().size();
        self.reserve(padded_len - self.len);
        self.len = padded_len;
        padded_len
    }

    pub(crate) unsafe fn push_aligned<T>(&mut self, val: T) -> usize {
        let size = size_of::<T>();
        self.reserve(size);
        let offset = self.len;
        unsafe { self.ptr.as_ptr().add(offset).cast::<T>().write(val) }
        self.len += size;
        offset
    }

    pub(crate) fn shrink_to_fit(&mut self) {
        if self.len < self.capacity {
            unsafe { self.shrink_to_fit_cold() }
        }
    }

    #[cold]
    unsafe fn shrink_to_fit_cold(&mut self) {
        let new_capacity = self.len;
        let layout = unsafe { Layout::from_size_align_unchecked(self.capacity, ALIGN) };
        let new_ptr = if new_capacity == 0 {
            unsafe { dealloc(self.ptr.as_ptr(), layout) }
            NonNull::dangling()
        } else {
            unsafe {
                let new_layout = Layout::from_size_align_unchecked(new_capacity, ALIGN);
                let new_ptr = realloc(self.ptr.as_ptr(), layout, new_layout.size());
                NonNull::new(new_ptr).unwrap_or_else(|| handle_alloc_error(new_layout))
            }
        };
        self.ptr = new_ptr;
        self.capacity = new_capacity;
    }

    pub(crate) fn finish(mut self) -> Code {
        self.shrink_to_fit();
        let this = ManuallyDrop::new(self);
        Code {
            ptr: this.ptr,
            len: this.len,
        }
    }
}

impl Default for CodeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for CodeBuilder {
    fn drop(&mut self) {
        if self.capacity != 0 {
            unsafe {
                let layout = Layout::from_size_align_unchecked(self.capacity, ALIGN);
                dealloc(self.ptr.as_ptr(), layout)
            }
        }
    }
}

impl fmt::Debug for CodeBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CodeBuilder").finish_non_exhaustive()
    }
}
