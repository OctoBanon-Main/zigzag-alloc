use core::{
    alloc::Layout,
    marker::PhantomData,
    mem::MaybeUninit,
    ops::{Deref, DerefMut, Index, IndexMut},
    ptr::{self, NonNull},
    slice,
};

use crate::alloc::allocator::Allocator;
use crate::simd;

/// A contiguous growable array type using a manual allocator.
pub struct ExVec<'a, T> {
    ptr:     NonNull<T>,
    len:     usize,
    cap:     usize,
    alloc:   &'a dyn Allocator,
    _marker: PhantomData<T>,
}

impl<'a, T> ExVec<'a, T> {
    pub fn new(alloc: &'a dyn Allocator) -> Self {
        Self { ptr: NonNull::dangling(), len: 0, cap: 0, alloc, _marker: PhantomData }
    }

    #[inline] pub fn len(&self)      -> usize   { self.len }
    #[inline] pub fn capacity(&self) -> usize   { self.cap }
    #[inline] pub fn is_empty(&self) -> bool    { self.len == 0 }
    #[inline] pub fn as_ptr(&self)   -> *const T { self.ptr.as_ptr() }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Appends an element to the back of a collection.
    /// # Panics
    /// Panics if the new capacity exceeds `isize::MAX` bytes or if the allocator fails.
    pub fn push(&mut self, value: T) {
        if self.len == self.cap { self.grow(); }
        unsafe { self.ptr.as_ptr().add(self.len).write(value) };
        self.len += 1;
    }

    pub fn try_push(&mut self, value: T) -> Result<(), T> {
        if self.len == self.cap && !self.try_grow() { return Err(value); }
        unsafe { self.ptr.as_ptr().add(self.len).write(value) };
        self.len += 1;
        Ok(())
    }

    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 { return None; }
        self.len -= 1;
        Some(unsafe { self.ptr.as_ptr().add(self.len).read() })
    }

    pub fn truncate(&mut self, new_len: usize) {
        if new_len >= self.len { return; }
        let old_len = self.len;
        self.len = new_len;
        for i in new_len..old_len {
            unsafe { ptr::drop_in_place(self.ptr.as_ptr().add(i)) };
        }
    }

    pub fn clear(&mut self) { self.truncate(0); }

    pub unsafe fn set_len(&mut self, new_len: usize) {
        debug_assert!(new_len <= self.cap);
        self.len = new_len;
    }

    pub fn push_slice(&mut self, items: &[T]) where T: Copy {
        let n = items.len();
        if n == 0 { return; }

        while self.cap < self.len + n {
            self.grow();
        }

        let dst = unsafe { self.ptr.as_ptr().add(self.len) } as *mut u8;
        let src = items.as_ptr() as *const u8;
        let bytes = n * core::mem::size_of::<T>();

        unsafe { simd::copy_bytes(dst, src, bytes) };
        self.len += n;
    }

    #[cold] fn grow(&mut self) { assert!(self.try_grow(), "Vec: out of memory"); }

    fn try_grow(&mut self) -> bool {
        // SAFETY: If self.cap > 0, self.ptr is guaranteed to be a valid, 
        // allocated pointer with a layout matching `Layout::array::<T>(self.cap)`.
        // We use `simd::copy_bytes` for efficient data migration to the new block.
        let new_cap    = if self.cap == 0 { 4 } else { self.cap * 2 };
        let new_layout = match Layout::array::<T>(new_cap) { Ok(l) => l, Err(_) => return false };
        let new_ptr    = match unsafe { self.alloc.alloc(new_layout) } {
            Some(p) => p.cast::<T>(), None => return false,
        };
        if self.cap > 0 {
            unsafe {
                simd::copy_bytes(
                    new_ptr.as_ptr() as *mut u8,
                    self.ptr.as_ptr() as *const u8,
                    self.len * core::mem::size_of::<T>(),
                );
                let old_layout = Layout::array::<T>(self.cap).unwrap();
                self.alloc.dealloc(self.ptr.cast(), old_layout);
            }
        }
        self.ptr = new_ptr;
        self.cap = new_cap;
        true
    }
}

impl ExVec<'_, u8> {
    pub fn simd_fill(&mut self, val: u8) {
        if self.len == 0 { return; }
        unsafe { simd::fill_bytes(self.ptr.as_ptr(), val, self.len) };
    }

    pub fn find_byte(&self, val: u8) -> Option<usize> {
        if self.len == 0 { return None; }
        unsafe { simd::find_byte(self.ptr.as_ptr(), val, self.len) }
    }

    pub fn for_each_byte_match<F: FnMut(usize)>(&self, val: u8, mut f: F) {
        if self.len == 0 { return; }
        let ptr = self.ptr.as_ptr();
        let mut i = 0usize;
        while let Some(off) = unsafe { simd::find_byte(ptr.add(i), val, self.len - i) } {
            f(i + off);
            i += off + 1;
            if i >= self.len { break; }
        }
    }

    pub fn extend_filled(&mut self, val: u8, additional: usize) {
        if additional == 0 { return; }
        while self.cap < self.len + additional {
            self.grow();
        }
        unsafe {
            simd::fill_bytes(self.ptr.as_ptr().add(self.len), val, additional);
            self.len += additional;
        }
    }
}

impl<'a> ExVec<'a, MaybeUninit<u8>> {
    pub fn with_capacity_zeroed(alloc: &'a dyn Allocator, cap: usize) -> Option<Self> {
        if cap == 0 { return Some(Self::new(alloc)); }
        let layout = Layout::array::<MaybeUninit<u8>>(cap).ok()?;
        let ptr    = unsafe { alloc.alloc(layout)?.cast::<MaybeUninit<u8>>() };
        unsafe { simd::fill_bytes(ptr.as_ptr() as *mut u8, 0, cap) };
        Some(Self { ptr, len: cap, cap, alloc, _marker: PhantomData })
    }

    pub fn fill_range(&mut self, start: usize, len: usize, val: u8) {
        assert!(start + len <= self.len, "fill_range: out of bounds");
        unsafe { simd::fill_bytes(self.ptr.as_ptr().add(start) as *mut u8, val, len) };
    }

    pub fn find_byte(&self, val: u8) -> Option<usize> {
        unsafe { simd::find_byte(self.ptr.as_ptr() as *const u8, val, self.len) }
    }
}

impl<T> Drop for ExVec<'_, T> {
    fn drop(&mut self) {
        if self.cap == 0 { return; }
        for i in 0..self.len {
            unsafe { ptr::drop_in_place(self.ptr.as_ptr().add(i)) };
        }
        let layout = Layout::array::<T>(self.cap).unwrap();
        unsafe { self.alloc.dealloc(self.ptr.cast(), layout) };
    }
}

impl<T> Deref for ExVec<'_, T> {
    type Target = [T];
    fn deref(&self) -> &[T] { self.as_slice() }
}
impl<T> DerefMut for ExVec<'_, T> {
    fn deref_mut(&mut self) -> &mut [T] { self.as_mut_slice() }
}
impl<T> Index<usize> for ExVec<'_, T> {
    type Output = T;
    fn index(&self, i: usize) -> &T { &self.as_slice()[i] }
}
impl<T> IndexMut<usize> for ExVec<'_, T> {
    fn index_mut(&mut self, i: usize) -> &mut T { &mut self.as_mut_slice()[i] }
}