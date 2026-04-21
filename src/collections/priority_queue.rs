use core::{alloc::Layout, marker::PhantomData, ptr::NonNull};

use crate::alloc::allocator::Allocator;
use crate::simd;
use super::OrdContext;

pub struct ExPriorityQueue<'a, T, C: OrdContext<T>> {
    ptr: NonNull<T>,
    len: usize,
    cap: usize,
    alloc: &'a dyn Allocator,
    ctx: C,
    _marker: PhantomData<T>,
}

impl<'a, T, C: OrdContext<T>> ExPriorityQueue<'a, T, C> {
    pub fn new(alloc: &'a dyn Allocator, ctx: C) -> Self {
        Self { ptr: NonNull::dangling(), len: 0, cap: 0, alloc, ctx, _marker: PhantomData }
    }

    #[inline] pub fn len(&self) -> usize { self.len }
    #[inline] pub fn is_empty(&self) -> bool { self.len == 0 }
    #[inline] pub fn capacity(&self) -> usize { self.cap }

    pub fn peek(&self) -> Option<&T> {
        if self.len == 0 { return None; }
        Some(unsafe { &*self.ptr.as_ptr() })
    }

    pub fn push(&mut self, value: T) {
        assert!(self.try_push(value).is_ok(), "ExPriorityQueue: OOM")
    }

    pub fn try_push(&mut self, value: T) -> Result<(), T> {
        if self.len == self.cap && !self.try_grow() { return Err(value); }
        unsafe { self.ptr.as_ptr().add(self.len).write(value) };
        self.len += 1;
        self.sift_up(self.len - 1);
        Ok(())
    }

    pub fn push_slice(&mut self, items: &[T]) where T: Copy {
        let n = items.len();
        if n == 0 { return; }

        while self.cap < self.len + n {
            assert!(self.try_grow(), "ExPriorityQueue: OOM")
        }

        let dst = unsafe { self.ptr.as_ptr().add(self.len) } as *mut u8;
        let src = items.as_ptr() as *const u8;
        unsafe { simd::copy_bytes(dst, src, n * core::mem::size_of::<T>()); }
        self.len += n;

        self.rebuild();
    }

    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 { return None; }
        self.swap(0, self.len - 1);
        self.len -= 1;
        let val = unsafe { self.ptr.as_ptr().add(self.len).read() };
        if self.len > 0 { self.sift_down(0); }
        Some(val)
    }

    pub fn remove_at(&mut self, idx: usize) -> T {
        assert!(idx < self.len, "PriorityQueue::remove_at: out of bounds");
        self.swap(idx, self.len - 1);
        self.len -= 1;
        let val = unsafe { self.ptr.as_ptr().add(self.len).read() };
        if idx < self.len { self.sift_down(idx); self.sift_up(idx); }
        val
    }
 
    pub fn rebuild(&mut self) {
        if self.len <= 1 { return; }
        let mut i = (self.len / 2).wrapping_sub(1);
        loop {
            self.sift_down(i);
            if i == 0 { break; }
            i -= 1;
        }
    }

    pub fn drain_sorted(&mut self, out: &mut [T]) where T: Copy {
        assert!(out.len() >= self.len, "drain_sorted: out slice too short");
        let mut i = 0;
        while let Some(val) = self.pop() {
            let dst = unsafe { out.as_mut_ptr().add(i) } as *mut u8;
            let src = &val as *const T as *const u8;
            unsafe { simd::copy_bytes(dst, src, core::mem::size_of::<T>()) };
            i += 1;
        }
    }
 
    pub fn for_each<F: FnMut(&T)>(&self, mut f: F) {
        for i in 0..self.len { f(unsafe { &*self.ptr.as_ptr().add(i) }); }
    }

    fn sift_up(&mut self, mut idx: usize) {
        while idx > 0 {
            let parent = (idx - 1) / 2;
            if self.ctx.less(
                unsafe { &*self.ptr.as_ptr().add(idx) },
                unsafe { &*self.ptr.as_ptr().add(parent) },
            ) {
                self.swap(idx, parent);
                idx = parent;
            } else { break; }
        }
    }
 
    fn sift_down(&mut self, mut idx: usize) {
        loop {
            let left  = 2 * idx + 1;
            let right = 2 * idx + 2;
            let mut best = idx;
 
            if left < self.len && self.ctx.less(
                unsafe { &*self.ptr.as_ptr().add(left) },
                unsafe { &*self.ptr.as_ptr().add(best) },
            ) { best = left; }
            if right < self.len && self.ctx.less(
                unsafe { &*self.ptr.as_ptr().add(right) },
                unsafe { &*self.ptr.as_ptr().add(best) },
            ) { best = right; }
 
            if best == idx { break; }
            self.swap(idx, best);
            idx = best;
        }
    }

    #[inline]
    fn swap(&mut self, a: usize, b: usize) {
        unsafe { core::ptr::swap(self.ptr.as_ptr().add(a), self.ptr.as_ptr().add(b)); }
    }

    #[cold]
    fn try_grow(&mut self) -> bool {
        let new_cap = if self.cap == 0 { 4 } else { self.cap * 2 };
        let layout  = match Layout::array::<T>(new_cap) { Ok(l) => l, Err(_) => return false };
        let new_ptr = match unsafe { self.alloc.alloc(layout) } {
            Some(p) => p.cast::<T>(), None => return false,
        };
        if self.cap > 0 {
            unsafe {
                simd::copy_bytes(
                    new_ptr.as_ptr() as *mut u8,
                    self.ptr.as_ptr() as *const u8,
                    self.len * core::mem::size_of::<T>(),
                );
                self.alloc.dealloc(self.ptr.cast(), Layout::array::<T>(self.cap).unwrap());
            }
        }
        self.ptr = new_ptr;
        self.cap = new_cap;
        true
    }
}

impl<T, C: OrdContext<T>> Drop for ExPriorityQueue<'_, T, C> {
    fn drop(&mut self) {
        if self.cap == 0 { return; }
        for i in 0..self.len {
            unsafe { core::ptr::drop_in_place(self.ptr.as_ptr().add(i)) };
        }
        unsafe { self.alloc.dealloc(self.ptr.cast(), Layout::array::<T>(self.cap).unwrap()) };
    }
}