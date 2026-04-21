//! Growable contiguous array with an explicit allocator.
//!
//! [`ExVec<T>`] is a dynamically-sized list that stores its elements in a
//! heap-allocated buffer obtained from a user-supplied [`Allocator`].  It is
//! functionally analogous to [`std::vec::Vec`] but never touches a global
//! allocator.
//!
//! ## Growth Policy
//!
//! The initial capacity is **4**.  Every time the buffer is exhausted, the
//! capacity is **doubled** (`new_cap = old_cap * 2`).  Data is migrated to the
//! new buffer using SIMD-accelerated `copy_bytes`.
//!
//! ## SIMD Extensions (`ExVec<u8>`)
//!
//! When `T = u8`, additional methods are available:
//! [`simd_fill`](ExVec::simd_fill), [`find_byte`](ExVec::find_byte),
//! [`for_each_byte_match`](ExVec::for_each_byte_match),
//! [`extend_filled`](ExVec::extend_filled).

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

/// A contiguous growable array backed by an explicit allocator.
///
/// # Invariants
///
/// * `ptr` is a valid, aligned allocation of `cap * size_of::<T>()` bytes
///   whenever `cap > 0`.  When `cap == 0`, `ptr` is a dangling non-null
///   pointer (never dereferenced).
/// * Elements at indices `0..len` are fully initialised.
/// * Elements at indices `len..cap` are uninitialised and must not be read.
///
/// # Lifetime
///
/// The allocator reference `'a` must outlive the `ExVec`.  The vec's buffer
/// is freed in `Drop` using the same allocator reference.
pub struct ExVec<'a, T> {
    /// Pointer to the start of the allocated buffer.
    ptr:     NonNull<T>,
    /// Number of initialised elements.
    len:     usize,
    /// Total buffer capacity in elements.
    cap:     usize,
    /// Allocator used for growth and deallocation.
    alloc:   &'a dyn Allocator,
    _marker: PhantomData<T>,
}

impl<'a, T> ExVec<'a, T> {
    /// Creates a new, empty `ExVec` that will use `alloc` for memory.
    ///
    /// No allocation is performed until the first element is pushed.
    pub fn new(alloc: &'a dyn Allocator) -> Self {
        Self { ptr: NonNull::dangling(), len: 0, cap: 0, alloc, _marker: PhantomData }
    }

    /// Returns the number of initialised elements.
    #[inline] pub fn len(&self)      -> usize    { self.len }
    /// Returns the total buffer capacity in elements.
    #[inline] pub fn capacity(&self) -> usize    { self.cap }
    /// Returns `true` if the vec contains no elements.
    #[inline] pub fn is_empty(&self) -> bool     { self.len == 0 }
    /// Returns a raw pointer to the start of the buffer.
    ///
    /// The pointer is only valid to dereference for indices `0..len`.
    #[inline] pub fn as_ptr(&self)   -> *const T { self.ptr.as_ptr() }

    /// Returns a shared slice of the initialised elements.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: `ptr` is valid for `len` initialised elements of type `T`.
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Returns a mutable slice of the initialised elements.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: Same as `as_slice`, plus unique access via `&mut self`.
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Appends `value` to the end of the vec.
    ///
    /// # Panics
    ///
    /// Panics if the allocator cannot satisfy the growth request.
    pub fn push(&mut self, value: T) {
        if self.len == self.cap { self.grow(); }
        // SAFETY: After `grow`, `cap > len`, so `ptr + len` is within the
        // allocated buffer and currently uninitialised — safe to write.
        unsafe { self.ptr.as_ptr().add(self.len).write(value) };
        self.len += 1;
    }

    /// Attempts to append `value`, returning `Err(value)` if OOM.
    pub fn try_push(&mut self, value: T) -> Result<(), T> {
        if self.len == self.cap && !self.try_grow() { return Err(value); }
        // SAFETY: Same as `push`.
        unsafe { self.ptr.as_ptr().add(self.len).write(value) };
        self.len += 1;
        Ok(())
    }

    /// Removes and returns the last element, or `None` if the vec is empty.
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 { return None; }
        self.len -= 1;
        // SAFETY: Element at `len` (old `len - 1`) was initialised; we
        // decrement `len` first so the slot is no longer considered initialised.
        Some(unsafe { self.ptr.as_ptr().add(self.len).read() })
    }

    /// Shortens the vec, keeping the first `new_len` elements and dropping
    /// the rest.  Does nothing if `new_len >= self.len`.
    pub fn truncate(&mut self, new_len: usize) {
        if new_len >= self.len { return; }
        let old_len = self.len;
        self.len = new_len;
        for i in new_len..old_len {
            // SAFETY: Indices in `new_len..old_len` are initialised; we drop
            // them in-place as we reduce `len`.
            unsafe { ptr::drop_in_place(self.ptr.as_ptr().add(i)) };
        }
    }

    /// Removes all elements, dropping each one.
    pub fn clear(&mut self) { self.truncate(0); }

    /// Forcibly sets the length to `new_len`.
    ///
    /// # Safety
    ///
    /// * `new_len` must be ≤ `cap`.
    /// * Elements in `old_len..new_len` (if extending) must have been
    ///   initialised by the caller before or immediately after this call.
    /// * Elements in `new_len..old_len` (if shrinking) must have already been
    ///   dropped by the caller, or be types that do not need dropping.
    pub unsafe fn set_len(&mut self, new_len: usize) {
        debug_assert!(new_len <= self.cap);
        self.len = new_len;
    }

    /// Appends a copy of every element in `items` to the vec.
    ///
    /// Grows the buffer (possibly multiple times) to fit all items.
    ///
    /// # Panics
    ///
    /// Panics if the allocator cannot satisfy any required growth.
    pub fn push_slice(&mut self, items: &[T]) where T: Copy {
        let n = items.len();
        if n == 0 { return; }

        while self.cap < self.len + n {
            self.grow();
        }

        let dst   = unsafe { self.ptr.as_ptr().add(self.len) } as *mut u8;
        let src   = items.as_ptr() as *const u8;
        let bytes = n * core::mem::size_of::<T>();

        // SAFETY: `dst` points to uninitialised space inside the buffer for
        // exactly `n` elements; `src` is a valid slice of `n` elements.
        // `T: Copy` means no destructor will be skipped.
        unsafe { simd::copy_bytes(dst, src, bytes) };
        self.len += n;
    }

    /// Panicking growth helper — calls `try_grow` and panics on OOM.
    #[cold]
    fn grow(&mut self) { assert!(self.try_grow(), "Vec: out of memory"); }

    /// Attempts to double the buffer capacity.
    ///
    /// Returns `true` on success, `false` if the allocator rejected the
    /// request.
    fn try_grow(&mut self) -> bool {
        let new_cap    = if self.cap == 0 { 4 } else { self.cap * 2 };
        let new_layout = match Layout::array::<T>(new_cap) {
            Ok(l) => l,
            Err(_) => return false,
        };
        let new_ptr = match unsafe { self.alloc.alloc(new_layout) } {
            Some(p) => p.cast::<T>(),
            None    => return false,
        };

        if self.cap > 0 {
            // SAFETY: `new_ptr` is valid for `new_cap` elements; `self.ptr` is
            // valid for `self.cap` elements.  We copy `self.len` initialised
            // elements; the rest of the new buffer remains uninitialised.
            // After copying, we release the old buffer with the exact layout
            // that was used to allocate it.
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
    /// Fills all initialised bytes with `val` using SIMD.
    pub fn simd_fill(&mut self, val: u8) {
        if self.len == 0 { return; }
        // SAFETY: `ptr` is valid for `len` bytes.
        unsafe { simd::fill_bytes(self.ptr.as_ptr(), val, self.len) };
    }

    /// Searches for the first occurrence of `val` in the initialised region.
    ///
    /// Returns the byte offset, or `None` if not found.
    pub fn find_byte(&self, val: u8) -> Option<usize> {
        if self.len == 0 { return None; }
        // SAFETY: `ptr` is valid for `len` bytes.
        unsafe { simd::find_byte(self.ptr.as_ptr(), val, self.len) }
    }

    /// Calls `f` with the offset of every byte in the initialised region that
    /// equals `val`.
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

    /// Appends `additional` bytes all set to `val`, growing the buffer as
    /// needed.
    ///
    /// # Panics
    ///
    /// Panics if the allocator fails to grow the buffer.
    pub fn extend_filled(&mut self, val: u8, additional: usize) {
        if additional == 0 { return; }
        while self.cap < self.len + additional {
            self.grow();
        }
        // SAFETY: `ptr + len` is valid for `additional` uninitialised bytes;
        // writing `val` initialises them.
        unsafe {
            simd::fill_bytes(self.ptr.as_ptr().add(self.len), val, additional);
            self.len += additional;
        }
    }
}

impl<'a> ExVec<'a, MaybeUninit<u8>> {
    /// Allocates a buffer of `cap` zero-filled `MaybeUninit<u8>` slots and
    /// returns a fully-populated `ExVec` (`len == cap`).
    ///
    /// Returns `None` if allocation fails or `cap` is zero.
    pub fn with_capacity_zeroed(alloc: &'a dyn Allocator, cap: usize) -> Option<Self> {
        if cap == 0 { return Some(Self::new(alloc)); }
        let layout = Layout::array::<MaybeUninit<u8>>(cap).ok()?;
        // SAFETY: `alloc` returns a pointer valid for `cap` bytes.
        let ptr    = unsafe { alloc.alloc(layout)?.cast::<MaybeUninit<u8>>() };
        // SAFETY: `ptr` is valid for `cap` bytes; zero-filling `MaybeUninit<u8>`
        // is always safe because `MaybeUninit` does not have a validity
        // requirement.
        unsafe { simd::fill_bytes(ptr.as_ptr() as *mut u8, 0, cap) };
        Some(Self { ptr, len: cap, cap, alloc, _marker: PhantomData })
    }

    /// Fills `len` bytes starting at `start` with `val`.
    ///
    /// # Panics
    ///
    /// Panics if `start + len > self.len`.
    pub fn fill_range(&mut self, start: usize, len: usize, val: u8) {
        assert!(start + len <= self.len, "fill_range: out of bounds");
        // SAFETY: The bounds check above guarantees the target range is within
        // the initialised region of the buffer.
        unsafe { simd::fill_bytes(self.ptr.as_ptr().add(start) as *mut u8, val, len) };
    }

    /// Searches for the first byte equal to `val` in the buffer.
    ///
    /// Returns the byte offset, or `None` if not found.
    pub fn find_byte(&self, val: u8) -> Option<usize> {
        // SAFETY: `ptr` is valid for `len` bytes.
        unsafe { simd::find_byte(self.ptr.as_ptr() as *const u8, val, self.len) }
    }
}

impl<T> Drop for ExVec<'_, T> {
    /// Drops all initialised elements and releases the buffer to the allocator.
    fn drop(&mut self) {
        if self.cap == 0 { return; }
        for i in 0..self.len {
            // SAFETY: Indices `0..len` are initialised.
            unsafe { ptr::drop_in_place(self.ptr.as_ptr().add(i)) };
        }
        let layout = Layout::array::<T>(self.cap).unwrap();
        // SAFETY: `ptr` was obtained from `alloc` with this exact `layout`.
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