use core::{
    alloc::Layout,
    fmt,
    mem,
    ops::{Deref, DerefMut},
    ptr::{self, NonNull},
};

use crate::alloc::allocator::Allocator;
use crate::simd;

/// A pointer type for heap allocation with an explicit allocator.
///
/// `ExBox` represents unique ownership of a value. When it goes out of scope,
/// it will call `drop` on the inner value and deallocate the memory using 
/// the stored allocator reference.
pub struct ExBox<'a, T> {
    ptr:   NonNull<T>,
    alloc: &'a dyn Allocator,
}

impl<'a, T> ExBox<'a, T> {
    /// Allocates memory on the heap and moves `value` into it.
    /// Returns `None` if the allocator fails.
    pub fn new(value: T, alloc: &'a dyn Allocator) -> Option<Self> {
        let layout = Layout::new::<T>();
        let ptr: NonNull<T> = if layout.size() == 0 {
            NonNull::dangling()
        } else {
            unsafe { alloc.alloc(layout)?.cast() }
        };
        unsafe { ptr.as_ptr().write(value) };
        Some(Self { ptr, alloc })
    }

    pub fn new_zeroed(value: T, alloc: &'a dyn Allocator) -> Option<Self> {
        let layout = Layout::new::<T>();
        let ptr: NonNull<T> = if layout.size() == 0 {
            NonNull::dangling()
        } else {
            let p = unsafe { alloc.alloc(layout)? };
            unsafe { simd::fill_bytes(p.as_ptr(), 0, layout.size()) };
            p.cast()
        };
        unsafe { ptr.as_ptr().write(value) };
        Some(Self { ptr, alloc })
    }

    /// Consumes the box and returns the inner value, deallocating the heap memory.
    pub fn unbox(b: Self) -> T {
        // SAFETY: We read the value out of the pointer. Since we call `mem::forget(b)`
        // immediately after, the Drop implementation of ExBox won't run, preventing
        // a double-free or use-after-free of the inner value.
        let value = unsafe { b.ptr.as_ptr().read() };
        let layout = Layout::new::<T>();
        if layout.size() > 0 {
            unsafe { b.alloc.dealloc(b.ptr.cast(), layout) };
        }
        mem::forget(b);
        value
    }

    pub unsafe fn wipe(b: &mut Self) {
        let layout = Layout::new::<T>();
        if layout.size() > 0 {
            unsafe { simd::fill_bytes(b.ptr.as_ptr() as *mut u8, 0, layout.size()) };
        }
    }

    #[inline] pub fn as_ptr(b: &Self)     -> *const T { b.ptr.as_ptr() }
    #[inline] pub fn as_mut_ptr(b: &mut Self) -> *mut T { b.ptr.as_ptr() }
}

impl<T> Drop for ExBox<'_, T> {
    fn drop(&mut self) {
        unsafe {
            ptr::drop_in_place(self.ptr.as_ptr());
            let layout = Layout::new::<T>();
            if layout.size() > 0 {
                self.alloc.dealloc(self.ptr.cast(), layout);
            }
        }
    }
}

impl<T> Deref for ExBox<'_, T> {
    type Target = T;
    fn deref(&self) -> &T { unsafe { self.ptr.as_ref() } }
}

impl<T> DerefMut for ExBox<'_, T> {
    fn deref_mut(&mut self) -> &mut T { unsafe { self.ptr.as_mut() } }
}

impl<T: fmt::Debug> fmt::Debug for ExBox<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { fmt::Debug::fmt(&**self, f) }
}

impl<T: fmt::Display> fmt::Display for ExBox<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { fmt::Display::fmt(&**self, f) }
}

impl<T: PartialEq> PartialEq for ExBox<'_, T> {
    fn eq(&self, other: &Self) -> bool { **self == **other }
}