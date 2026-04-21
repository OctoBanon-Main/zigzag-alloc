use core::{
    mem::MaybeUninit,
    ops::{Deref, DerefMut, Index, IndexMut},
    ptr,
    slice
};

use crate::simd;

/// A fixed-capacity stack-allocated array.
///
/// This structure does not heap allocate and has a constant size determined 
/// by `N`. It is ideal for small, short-lived buffers where performance is 
/// critical and capacity is known at compile time.
pub struct ExBoundedArray<T, const N: usize> {
    data: [MaybeUninit<T>; N],
    len: usize,
}

impl<T, const N: usize> ExBoundedArray<T, N>  {
    #[inline]
    pub const fn new() -> Self {
        Self {
            data: unsafe {
                MaybeUninit::<[MaybeUninit<T>; N]>::uninit().assume_init()
            },
            len: 0,
        }
    }

    #[inline] pub fn len(&self)       -> usize { self.len }
    #[inline] pub fn capacity(&self)  -> usize { N }
    #[inline] pub fn remaining(&self) -> usize { N - self.len }
    #[inline] pub fn is_empty(&self)  -> bool  { self.len == 0 }
    #[inline] pub fn is_full(&self)   -> bool  { self.len == N }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.data.as_ptr() as *const T, self.len) }
    }
 
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.data.as_mut_ptr() as *mut T, self.len) }
    }

    #[inline]
    pub fn push(&mut self, value: T) -> Result<(), T> {
        if self.len == N { return Err(value); }
        self.data[self.len] = MaybeUninit::new(value);
        self.len += 1;
        Ok(())
    }

    #[inline]
    pub unsafe fn push_unchecked(&mut self, value: T) {
        debug_assert!(self.len < N);
        self.data[self.len] = MaybeUninit::new(value);
        self.len += 1;
    }

    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 { return None; }
        self.len -= 1;
        Some(unsafe { self.data[self.len].assume_init_read() })
    }

    pub fn insert(&mut self, idx: usize, value: T) -> Result<(), T> {
        if self.len == N { return Err(value); }
        assert!(idx <= self.len, "ExBoundedArray::insert: index out of bounds");
        unsafe {
            ptr::copy(
                self.data.as_ptr().add(idx),
                self.data.as_mut_ptr().add(idx + 1),
                self.len - idx,
            );
            self.data[idx] = MaybeUninit::new(value);
        }
        self.len += 1;
        Ok(())
    }

    pub fn remove(&mut self, idx: usize) -> T {
        assert!(idx < self.len, "ExBoundedArray::remove: index out of bounds");
        unsafe {
            let val = self.data[idx].assume_init_read();
            ptr::copy(
                self.data.as_ptr().add(idx + 1),
                self.data.as_mut_ptr().add(idx),
                self.len - idx - 1,
            );
            self.len -= 1;
            val
        }
    }

    /// Removes the element at `idx` and returns it, moving the last element 
    /// into its place. 
    /// 
    /// O(1) complexity, but does not preserve order.
    pub fn swap_remove(&mut self, idx: usize) -> T {
        // SAFETY: idx is checked to be within 0..len. `assume_init_read` is safe 
        // because we only read from initialized slots tracked by `self.len`.
        assert!(idx < self.len, "ExBoundedArray::swap_remove: index out of bounds");
        self.len -= 1;
        unsafe {
            if idx != self.len {
                let last = self.data[self.len].assume_init_read();
                let val = self.data[idx].assume_init_read();
                self.data[idx] = MaybeUninit::new(last);
                val
            } else {
                self.data[self.len].assume_init_read()
            }
        }
    }

    pub fn truncate(&mut self, new_len: usize) {
        if new_len >= self.len { return; }
        let old_len = self.len;
        self.len = new_len;
        for i in new_len..old_len {
            unsafe { ptr::drop_in_place(self.data[i].as_mut_ptr()); }
        }
    }

    pub fn clear(&mut self) { self.truncate(0); }

    pub fn push_slice(&mut self, items: &[T]) -> Result<(), usize> where T: Copy {
        let n = items.len();
        if n == 0 { return Ok(()); }
        if self.len + n > N {
            return Err(n - (N - self.len));
        }
        let dst = unsafe { self.data.as_mut_ptr().add(self.len) } as *mut u8;
        let src = items.as_ptr() as *const u8;
        unsafe { simd::copy_bytes(dst, src, n * core::mem::size_of::<T>()) };
        self.len += n;
        Ok(())
    }

    pub fn copy_from_slice(&mut self, items: &[T]) -> Result<(), ()> where T: Copy {
        if items.len() > N { return Err(()); }
        for i in 0..self.len {
            unsafe { ptr::drop_in_place(self.data[i].as_mut_ptr()) };
        }
        let n   = items.len();
        let dst = self.data.as_mut_ptr() as *mut u8;
        let src = items.as_ptr() as *const u8;
        unsafe { simd::copy_bytes(dst, src, n * core::mem::size_of::<T>()) };
        self.len = n;
        Ok(())
    }
}

impl<const N: usize> ExBoundedArray<u8, N> {
    pub fn fill_bytes(&mut self, val: u8) {
        if self.len == 0 { return; }
        unsafe { simd::fill_bytes(self.data.as_mut_ptr() as *mut u8, val, self.len) };
    }
 
    pub fn fill_range(&mut self, start: usize, len: usize, val: u8) {
        assert!(start + len <= self.len, "ExBoundedArray::fill_range: out of bounds");
        unsafe {
            simd::fill_bytes(self.data.as_mut_ptr().add(start) as *mut u8, val, len)
        };
    }
 
    pub fn find_byte(&self, val: u8) -> Option<usize> {
        if self.len == 0 { return None; }
        unsafe { simd::find_byte(self.data.as_ptr() as *const u8, val, self.len) }
    }
 
    pub fn count_byte(&self, val: u8) -> usize {
        let ptr = self.data.as_ptr() as *const u8;
        let mut count = 0usize;
        let mut i = 0usize;
        while i < self.len {
            match unsafe { simd::find_byte(ptr.add(i), val, self.len - i) } {
                Some(off) => { count += 1; i += off + 1; }
                None      => break,
            }
        }
        count
    }
 
    pub fn extend_bytes(&mut self, src: &[u8]) -> bool {
        self.push_slice(src).is_ok()
    }
}
 
impl<T, const N: usize> Default for ExBoundedArray<T, N> {
    fn default() -> Self { Self::new() }
}
 
impl<T, const N: usize> Drop for ExBoundedArray<T, N> {
    fn drop(&mut self) {
        for i in 0..self.len {
            unsafe { ptr::drop_in_place(self.data[i].as_mut_ptr()) };
        }
    }
}
 
impl<T, const N: usize> Deref for ExBoundedArray<T, N> {
    type Target = [T];
    fn deref(&self) -> &[T] { self.as_slice() }
}

impl<T, const N: usize> DerefMut for ExBoundedArray<T, N> {
    fn deref_mut(&mut self) -> &mut [T] { self.as_mut_slice() }
}

impl<T, const N: usize> Index<usize> for ExBoundedArray<T, N> {
    type Output = T;
    fn index(&self, i: usize) -> &T { &self.as_slice()[i] }
}

impl<T, const N: usize> IndexMut<usize> for ExBoundedArray<T, N> {
    fn index_mut(&mut self, i: usize) -> &mut T { &mut self.as_mut_slice()[i] }
}