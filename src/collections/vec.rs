use core::{
    alloc::Layout,
    marker::PhantomData,
    ops::{Deref, DerefMut, Index, IndexMut},
    ptr::{self, NonNull},
    slice,
};

use crate::alloc::allocator::Allocator;

pub struct ExVec<'a, T> {
    ptr: NonNull<T>,
    len: usize,
    cap: usize,
    alloc: &'a dyn Allocator,
    _marker: PhantomData<T>,
}

impl<'a, T> ExVec<'a, T> {
    pub fn new(alloc: &'a dyn Allocator) -> Self {
        Self {
            ptr: NonNull::dangling(),
            len: 0,
            cap: 0,
            alloc,
            _marker: PhantomData,
        }
    }

    #[inline] pub fn len(&self) -> usize { self.len }
    #[inline] pub fn capacity(&self) -> usize { self.cap }
    #[inline] pub fn is_empty(&self) -> bool { self.len == 0 }
    #[inline] pub fn as_ptr(&self) -> *const T { self.ptr.as_ptr() }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    pub fn push(&mut self, value: T) {
        if self.len == self.cap { self.grow(); }
        unsafe { self.ptr.as_ptr().add(self.len).write(value) };
        self.len += 1;
    }

    pub fn try_push(&mut self, value: T) -> Result<(), T> {
        if self.len == self.cap && !self.try_grow() {
            return Err(value);
        }

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

    #[cold]
    fn grow(&mut self) {
        assert!(self.try_grow(), "ExVec: out of memory");
    }

    fn try_grow(&mut self) -> bool {
        let new_cap = if self.cap == 0 { 4 } else { self.cap * 2 };
        let new_layout = match Layout::array::<T>(new_cap) {
            Ok(l) => l, Err(_) => return false,
        };
        let new_ptr = match unsafe { self.alloc.alloc(new_layout) } {
            Some(p) => p.cast::<T>(), None => return false,
        };
        if self.cap > 0 {
            unsafe {
                ptr::copy_nonoverlapping(self.ptr.as_ptr(), new_ptr.as_ptr(), self.len);
                let old_layout = Layout::array::<T>(self.cap).unwrap();
                self.alloc.dealloc(self.ptr.cast(), old_layout);
            }
        }
        self.ptr = new_ptr;
        self.cap = new_cap;
        true
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