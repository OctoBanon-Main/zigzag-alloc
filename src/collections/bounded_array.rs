//! Fixed-capacity, stack-allocated array.
//!
//! [`ExBoundedArray<T, N>`] stores up to `N` elements of type `T` directly on
//! the stack without any heap allocation.  The capacity is a compile-time
//! constant; attempting to exceed it returns an error or panics, depending on
//! the method called.
//!
//! ## When to Use
//!
//! * Small, short-lived buffers whose maximum size is known at compile time.
//! * `no_std` environments without a heap.
//! * Hot paths where heap-allocation overhead is unacceptable.
//!
//! ## SIMD Extensions (`ExBoundedArray<u8, N>`)
//!
//! When `T = u8`, additional byte-level SIMD methods are available:
//! [`fill_bytes`](ExBoundedArray::fill_bytes),
//! [`fill_range`](ExBoundedArray::fill_range),
//! [`find_byte`](ExBoundedArray::find_byte),
//! [`count_byte`](ExBoundedArray::count_byte),
//! [`extend_bytes`](ExBoundedArray::extend_bytes).

use core::{
    mem::MaybeUninit,
    ops::{Deref, DerefMut, Index, IndexMut},
    ptr,
    slice
};

use crate::simd;

/// A fixed-capacity, stack-allocated array.
///
/// # Invariants
///
/// * Elements at indices `0..len` are fully initialised.
/// * Elements at indices `len..N` are uninitialised (`MaybeUninit`) and must
///   not be read through safe interfaces.
/// * `len <= N` at all times.
pub struct ExBoundedArray<T, const N: usize> {
    /// Storage for up to `N` elements; only `data[0..len]` are initialised.
    data: [MaybeUninit<T>; N],
    /// Number of initialised elements.
    len: usize,
}

impl<T, const N: usize> ExBoundedArray<T, N>  {
    /// Creates an empty `ExBoundedArray` with no initialised elements.
    ///
    /// This is a `const fn` and can be used in `static` initialisers.
    #[inline]
    pub const fn new() -> Self {
        Self {
            // SAFETY: An array of `MaybeUninit<T>` does not require
            // initialisation — interpreting uninitialised bytes as
            // `MaybeUninit<T>` is always valid.
            data: unsafe {
                MaybeUninit::<[MaybeUninit<T>; N]>::uninit().assume_init()
            },
            len: 0,
        }
    }

    /// Returns the number of initialised elements.
    #[inline] pub fn len(&self)       -> usize { self.len }
    /// Returns the compile-time maximum capacity `N`.
    #[inline] pub fn capacity(&self)  -> usize { N }
    /// Returns the number of additional elements that can be pushed.
    #[inline] pub fn remaining(&self) -> usize { N - self.len }
    /// Returns `true` if no elements are stored.
    #[inline] pub fn is_empty(&self)  -> bool  { self.len == 0 }
    /// Returns `true` if the array has reached its maximum capacity.
    #[inline] pub fn is_full(&self)   -> bool  { self.len == N }

    /// Returns a shared slice of the initialised elements.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: `data[0..len]` are all initialised; casting `*const MaybeUninit<T>`
        // to `*const T` is valid when the slots are initialised.
        unsafe { slice::from_raw_parts(self.data.as_ptr() as *const T, self.len) }
    }

    /// Returns a mutable slice of the initialised elements.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: Same as `as_slice`, plus unique access via `&mut self`.
        unsafe { slice::from_raw_parts_mut(self.data.as_mut_ptr() as *mut T, self.len) }
    }

    /// Appends `value` to the end of the array.
    ///
    /// Returns `Err(value)` if the array is full.
    #[inline]
    pub fn push(&mut self, value: T) -> Result<(), T> {
        if self.len == N { return Err(value); }
        self.data[self.len] = MaybeUninit::new(value);
        self.len += 1;
        Ok(())
    }

    /// Appends `value` without checking the capacity.
    ///
    /// # Safety
    ///
    /// `self.len < N` must hold; calling this when the array is full causes a
    /// write past the end of `data`, which is undefined behaviour.
    #[inline]
    pub unsafe fn push_unchecked(&mut self, value: T) {
        debug_assert!(self.len < N);
        self.data[self.len] = MaybeUninit::new(value);
        self.len += 1;
    }

    /// Removes and returns the last element, or `None` if empty.
    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 { return None; }
        self.len -= 1;
        // SAFETY: `data[len]` (old `len - 1`) was initialised; we read it
        // before decrementing `len`, effectively transferring ownership.
        Some(unsafe { self.data[self.len].assume_init_read() })
    }

    /// Inserts `value` at position `idx`, shifting all elements after it to
    /// the right.
    ///
    /// Returns `Err(value)` if the array is already full.
    ///
    /// # Panics
    ///
    /// Panics if `idx > self.len`.
    pub fn insert(&mut self, idx: usize, value: T) -> Result<(), T> {
        if self.len == N { return Err(value); }
        assert!(idx <= self.len, "ExBoundedArray::insert: index out of bounds");
        unsafe {
            // SAFETY: Shifting `len - idx` initialised elements one position
            // to the right.  Source and destination may overlap, so we use
            // `copy` (not `copy_nonoverlapping`).  The destination slot at
            // `idx + 1..=len` is within bounds because `len < N`.
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

    /// Removes the element at `idx` and returns it, shifting subsequent
    /// elements one position to the left.
    ///
    /// # Panics
    ///
    /// Panics if `idx >= self.len`.
    pub fn remove(&mut self, idx: usize) -> T {
        assert!(idx < self.len, "ExBoundedArray::remove: index out of bounds");
        unsafe {
            // SAFETY: `data[idx]` is initialised; we read it before overwriting
            // its slot via the `copy` below.
            let val = self.data[idx].assume_init_read();
            // SAFETY: Shifting `len - idx - 1` elements one position to the
            // left.  Source and destination overlap; `copy` is correct.
            ptr::copy(
                self.data.as_ptr().add(idx + 1),
                self.data.as_mut_ptr().add(idx),
                self.len - idx - 1,
            );
            self.len -= 1;
            val
        }
    }

    /// Removes the element at `idx` and fills the gap by moving the *last*
    /// element into it.
    ///
    /// This is O(1) but does **not** preserve the order of elements.
    ///
    /// # Panics
    ///
    /// Panics if `idx >= self.len`.
    pub fn swap_remove(&mut self, idx: usize) -> T {
        assert!(idx < self.len, "ExBoundedArray::swap_remove: index out of bounds");
        self.len -= 1;
        unsafe {
            if idx != self.len {
                // SAFETY: Both `data[len]` and `data[idx]` are initialised.
                // We read both and write the last element back into `idx`.
                let last = self.data[self.len].assume_init_read();
                let val  = self.data[idx].assume_init_read();
                self.data[idx] = MaybeUninit::new(last);
                val
            } else {
                // Removing the last element — just read it.
                // SAFETY: `data[len]` was initialised and `len` was decremented
                // above, so the slot is no longer considered live.
                self.data[self.len].assume_init_read()
            }
        }
    }

    /// Shortens the array to `new_len`, dropping excess elements.
    ///
    /// Does nothing if `new_len >= self.len`.
    pub fn truncate(&mut self, new_len: usize) {
        if new_len >= self.len { return; }
        let old_len = self.len;
        self.len = new_len;
        for i in new_len..old_len {
            // SAFETY: Indices `new_len..old_len` are initialised.
            unsafe { ptr::drop_in_place(self.data[i].as_mut_ptr()); }
        }
    }

    /// Removes all elements (drops each one).
    pub fn clear(&mut self) { self.truncate(0); }

    /// Appends copies of all items in `items` to the array.
    ///
    /// Returns `Err(n)` where `n` is the number of elements that did not fit.
    /// On error, as many items as possible have been pushed.
    ///
    /// Requires `T: Copy`; uses SIMD bulk-copy for efficiency.
    pub fn push_slice(&mut self, items: &[T]) -> Result<(), usize> where T: Copy {
        let n = items.len();
        if n == 0 { return Ok(()); }
        if self.len + n > N {
            return Err(n - (N - self.len));
        }
        let dst = unsafe { self.data.as_mut_ptr().add(self.len) } as *mut u8;
        let src = items.as_ptr() as *const u8;
        // SAFETY: `dst` points to uninitialised slots for exactly `n` elements;
        // `src` is a valid slice of `n` initialised elements of the same type.
        // `T: Copy` means no destructor is skipped.
        unsafe { simd::copy_bytes(dst, src, n * core::mem::size_of::<T>()) };
        self.len += n;
        Ok(())
    }

    /// Replaces all current elements with copies from `items`.
    ///
    /// Drops all existing elements first, then bulk-copies the new ones.
    ///
    /// Returns `Err(())` if `items.len() > N`.
    pub fn copy_from_slice(&mut self, items: &[T]) -> Result<(), ()> where T: Copy {
        if items.len() > N { return Err(()); }
        // Drop existing elements.
        for i in 0..self.len {
            // SAFETY: Indices `0..len` are initialised.
            unsafe { ptr::drop_in_place(self.data[i].as_mut_ptr()) };
        }
        let n   = items.len();
        let dst = self.data.as_mut_ptr() as *mut u8;
        let src = items.as_ptr() as *const u8;
        // SAFETY: `dst` can hold at least `N >= n` elements; `src` is valid
        // for `n` elements of `T`.
        unsafe { simd::copy_bytes(dst, src, n * core::mem::size_of::<T>()) };
        self.len = n;
        Ok(())
    }
}

impl<const N: usize> ExBoundedArray<u8, N> {
    /// Fills all `len` initialised bytes with `val` using SIMD.
    pub fn fill_bytes(&mut self, val: u8) {
        if self.len == 0 { return; }
        // SAFETY: `data.as_mut_ptr()` is valid for `len` initialised bytes.
        unsafe { simd::fill_bytes(self.data.as_mut_ptr() as *mut u8, val, self.len) };
    }

    /// Fills the subrange `[start, start + len)` with `val` using SIMD.
    ///
    /// # Panics
    ///
    /// Panics if `start + len > self.len`.
    pub fn fill_range(&mut self, start: usize, len: usize, val: u8) {
        assert!(start + len <= self.len, "ExBoundedArray::fill_range: out of bounds");
        // SAFETY: The bounds check above guarantees the target range is within
        // the initialised region.
        unsafe {
            simd::fill_bytes(self.data.as_mut_ptr().add(start) as *mut u8, val, len)
        };
    }

    /// Returns the offset of the first byte equal to `val`, or `None`.
    pub fn find_byte(&self, val: u8) -> Option<usize> {
        if self.len == 0 { return None; }
        // SAFETY: `data.as_ptr()` is valid for `len` initialised bytes.
        unsafe { simd::find_byte(self.data.as_ptr() as *const u8, val, self.len) }
    }

    /// Returns the number of bytes equal to `val` in the initialised region.
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

    /// Appends all bytes from `src` to the array.
    ///
    /// Returns `true` on success, `false` if there is not enough remaining
    /// capacity.
    pub fn extend_bytes(&mut self, src: &[u8]) -> bool {
        self.push_slice(src).is_ok()
    }
}

impl<T, const N: usize> Default for ExBoundedArray<T, N> {
    fn default() -> Self { Self::new() }
}

impl<T, const N: usize> Drop for ExBoundedArray<T, N> {
    /// Drops all initialised elements.  No heap memory is freed because the
    /// array is stack-allocated.
    fn drop(&mut self) {
        for i in 0..self.len {
            // SAFETY: Indices `0..len` are initialised.
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