//! Heap-allocated single value with an explicit allocator.
//!
//! [`ExBox<T>`] represents *unique ownership* of a heap-allocated `T`.  When
//! dropped it calls `T`'s destructor and returns the backing memory to the
//! allocator it was created with.
//!
//! This is the explicit-allocator analogue of [`std::boxed::Box`].

use core::{
    alloc::Layout,
    fmt,
    mem,
    ops::{Deref, DerefMut},
    ptr::{self, NonNull},
};

use crate::alloc::allocator::Allocator;
use crate::simd;

/// A pointer type for a single heap-allocated value with an explicit allocator.
///
/// # Ownership and Dropping
///
/// `ExBox` uniquely owns the value it points to.  On drop:
/// 1. `T`'s destructor is called via [`drop_in_place`](ptr::drop_in_place).
/// 2. The backing memory is returned to the allocator.
///
/// # Zero-Sized Types
///
/// For ZSTs (`size_of::<T>() == 0`) no allocation is performed; `ptr` is set
/// to `NonNull::dangling()` and `dealloc` is not called on drop.
///
/// # Lifetime
///
/// The allocator reference `'a` must outlive the `ExBox`.
pub struct ExBox<'a, T> {
    /// Non-null pointer to the heap-allocated value.
    ptr:   NonNull<T>,
    /// Allocator used to allocate and later free the backing memory.
    alloc: &'a dyn Allocator,
}

impl<'a, T> ExBox<'a, T> {
    /// Allocates memory for `T` and moves `value` into it.
    ///
    /// Returns `None` if the allocator fails.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let b = ExBox::new(42u64, &sys).unwrap();
    /// assert_eq!(*b, 42);
    /// ```
    pub fn new(value: T, alloc: &'a dyn Allocator) -> Option<Self> {
        let layout = Layout::new::<T>();
        let ptr: NonNull<T> = if layout.size() == 0 {
            // ZST — dangling pointer, never dereferenced.
            NonNull::dangling()
        } else {
            // SAFETY: `layout.size() > 0` as checked above.
            unsafe { alloc.alloc(layout)?.cast() }
        };
        // SAFETY: `ptr` is valid for a write of `T` (or dangling for ZSTs, but
        // ZST writes are no-ops in Rust).
        unsafe { ptr.as_ptr().write(value) };
        Some(Self { ptr, alloc })
    }

    /// Allocates zero-filled memory for `T`, then moves `value` into it.
    ///
    /// The allocation is zeroed *before* `value` is written, which may be
    /// useful for types with padding bytes that should be deterministic.
    ///
    /// Returns `None` if the allocator fails.
    pub fn new_zeroed(value: T, alloc: &'a dyn Allocator) -> Option<Self> {
        let layout = Layout::new::<T>();
        let ptr: NonNull<T> = if layout.size() == 0 {
            NonNull::dangling()
        } else {
            // SAFETY: `layout.size() > 0`.
            let p = unsafe { alloc.alloc(layout)? };
            // SAFETY: `p` is valid for `layout.size()` bytes.
            unsafe { simd::fill_bytes(p.as_ptr(), 0, layout.size()) };
            p.cast()
        };
        // SAFETY: `ptr` is valid for a write of `T`.
        unsafe { ptr.as_ptr().write(value) };
        Some(Self { ptr, alloc })
    }

    /// Consumes the box, returns the inner value, and deallocates the backing
    /// memory.
    ///
    /// # Implementation Note
    ///
    /// We read the value out of the pointer and then call
    /// [`mem::forget`] on the `ExBox` to prevent the `Drop` implementation
    /// from running a second destructor or freeing the (already freed) memory.
    pub fn unbox(b: Self) -> T {
        // SAFETY: `b.ptr` points to a valid, initialised `T`.  Reading it
        // transfers ownership.  We call `mem::forget(b)` immediately after so
        // `Drop` does not run — preventing double-drop and double-free.
        let value = unsafe { b.ptr.as_ptr().read() };
        let layout = Layout::new::<T>();
        if layout.size() > 0 {
            // SAFETY: `b.ptr` was obtained from `b.alloc` with this `layout`.
            unsafe { b.alloc.dealloc(b.ptr.cast(), layout) };
        }
        mem::forget(b);
        value
    }

    /// Zeroes all bytes of the allocation in-place (without moving or dropping
    /// `T`).
    ///
    /// # Safety
    ///
    /// After this call, the memory backing `*b` contains all zeros.  If `T`
    /// has any validity invariants (e.g. non-null pointers, non-zero
    /// discriminants) those invariants will be violated.  Accessing `*b` after
    /// calling `wipe` is undefined behaviour unless `T` is a type for which
    /// all-zeros is a valid representation.
    pub unsafe fn wipe(b: &mut Self) {
        let layout = Layout::new::<T>();
        if layout.size() > 0 {
            // SAFETY: Caller accepts responsibility for the consequences of
            // zeroing the bytes.  The pointer is valid for `layout.size()` bytes.
            unsafe { simd::fill_bytes(b.ptr.as_ptr() as *mut u8, 0, layout.size()) };
        }
    }

    /// Returns a raw const pointer to the contained value.
    #[inline]
    pub fn as_ptr(b: &Self) -> *const T { b.ptr.as_ptr() }

    /// Returns a raw mutable pointer to the contained value.
    #[inline]
    pub fn as_mut_ptr(b: &mut Self) -> *mut T { b.ptr.as_ptr() }
}

impl<T> Drop for ExBox<'_, T> {
    /// Runs `T`'s destructor and returns the backing memory to the allocator.
    fn drop(&mut self) {
        unsafe {
            // SAFETY: `self.ptr` is valid and initialised — we have unique
            // ownership of the value, so running the destructor is correct.
            ptr::drop_in_place(self.ptr.as_ptr());
            let layout = Layout::new::<T>();
            if layout.size() > 0 {
                // SAFETY: `self.ptr` was allocated from `self.alloc` with this
                // exact `layout`, and `drop_in_place` above did not free it.
                self.alloc.dealloc(self.ptr.cast(), layout);
            }
        }
    }
}

impl<T> Deref for ExBox<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        // SAFETY: `ptr` is valid and initialised; we hold shared access via `&self`.
        unsafe { self.ptr.as_ref() }
    }
}

impl<T> DerefMut for ExBox<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: `ptr` is valid and initialised; `&mut self` guarantees unique access.
        unsafe { self.ptr.as_mut() }
    }
}

impl<T: fmt::Debug> fmt::Debug for ExBox<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

impl<T: fmt::Display> fmt::Display for ExBox<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&**self, f)
    }
}

impl<T: PartialEq> PartialEq for ExBox<'_, T> {
    fn eq(&self, other: &Self) -> bool { **self == **other }
}