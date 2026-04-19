use core::{
    alloc::Layout, fmt, mem, ops::{Deref, DerefMut}, ptr::{self, NonNull}
};

use crate::alloc::allocator::Allocator;

pub struct ZigBox<'a, T> {
    ptr: NonNull<T>,
    alloc: &'a dyn Allocator,
}

impl<'a, T> ZigBox<'a, T> {
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

    pub fn unbox(b: Self) -> T {
        let value = unsafe { b.ptr.as_ptr().read() };

        let layout = Layout::new::<T>();
        if layout.size() > 0 {
            unsafe { b.alloc.dealloc(b.ptr.cast(), layout) };
        }

        mem::forget(b);
        value
    }

    #[inline]
    pub fn as_ptr(b: &Self) -> *const T {
        b.ptr.as_ptr()
    }

    #[inline]
    pub fn as_mut_ptr(b: &mut Self) -> *mut T {
        b.ptr.as_ptr()
    }
}

impl<T> Drop for ZigBox<'_, T> {
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

impl<T> Deref for ZigBox<'_, T> {
    type Target = T;
    fn deref(&self) -> &T { unsafe { self.ptr.as_ref() } }
}

impl<T> DerefMut for ZigBox<'_, T> {
    fn deref_mut(&mut self) -> &mut T { unsafe { self.ptr.as_mut() } }
}

impl<T: fmt::Debug> fmt::Debug for ZigBox<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

impl<T: fmt::Display> fmt::Display for ZigBox<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&**self, f)
    }
}

impl<T: PartialEq> PartialEq for ZigBox<'_, T> {
    fn eq(&self, other: &Self) -> bool { **self == **other }
}