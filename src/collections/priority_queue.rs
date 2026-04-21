use core::{alloc::Layout, marker::PhantomData, ptr::NonNull};

use crate::alloc::allocator::Allocator;
use crate::simd;
use super::OrdContext;

/// A Min-Heap or Max-Heap priority queue based on an explicit allocator.
///
/// The priority and heap property (Min vs Max) are determined by the 
/// provided [`OrdContext`].
pub struct ExPriorityQueue<'a, T, C: OrdContext<T>> {
    ptr: NonNull<T>,
    len: usize,
    cap: usize,
    alloc: &'a dyn Allocator,
    ctx: C,
    _marker: PhantomData<T>,
}

impl<'a, T, C: OrdContext<T>> ExPriorityQueue<'a, T, C> {
    /// Creates a new, empty priority queue using the provided allocator and context.
    /// Does not allocate until the first element is pushed.
    pub fn new(alloc: &'a dyn Allocator, ctx: C) -> Self {
        Self { ptr: NonNull::dangling(), len: 0, cap: 0, alloc, ctx, _marker: PhantomData }
    }

    /// Returns the number of elements in the queue.
    #[inline] pub fn len(&self) -> usize { self.len }

    /// Returns true if the queue contains no elements.
    #[inline] pub fn is_empty(&self) -> bool { self.len == 0 }

    /// Returns the total number of elements the queue can hold without reallocating.
    #[inline] pub fn capacity(&self) -> usize { self.cap }

    /// Returns a reference to the greatest element in the queue (according to context),
    /// or `None` if empty.
    pub fn peek(&self) -> Option<&T> {
        if self.len == 0 { return None; }
        // SAFETY: ptr is guaranteed to be valid and initialized for 0..len range.
        Some(unsafe { &*self.ptr.as_ptr() })
    }

    /// Pushes an element onto the priority queue.
    /// 
    /// # Panics
    /// Panics if the allocator fails to allocate memory.
    pub fn push(&mut self, value: T) {
        assert!(self.try_push(value).is_ok(), "ExPriorityQueue: OOM")
    }

    /// Attempts to push an element, returning `Err(value)` if memory allocation fails.
    pub fn try_push(&mut self, value: T) -> Result<(), T> {
        if self.len == self.cap && !self.try_grow() { return Err(value); }
        
        // SAFETY: ptr is valid for the current len due to try_grow check.
        unsafe { self.ptr.as_ptr().add(self.len).write(value) };
        self.len += 1;
        self.sift_up(self.len - 1);
        Ok(())
    }

    /// Pushes a slice of elements onto the queue and rebuilds the heap.
    /// More efficient than pushing elements individually.
    /// 
    /// # Panics
    /// Panics if the allocator fails to allocate memory.
    pub fn push_slice(&mut self, items: &[T]) where T: Copy {
        let n = items.len();
        if n == 0 { return; }

        while self.cap < self.len + n {
            assert!(self.try_grow(), "ExPriorityQueue: OOM")
        }

        // SAFETY: Pointer arithmetic is safe as we ensured capacity.
        // Copying via SIMD/Memcpy is safe for T: Copy.
        let dst = unsafe { self.ptr.as_ptr().add(self.len) } as *mut u8;
        let src = items.as_ptr() as *const u8;
        unsafe { simd::copy_bytes(dst, src, n * core::mem::size_of::<T>()); }
        self.len += n;

        self.rebuild();
    }

    /// Removes the greatest element from the priority queue and returns it, or `None` if empty.
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 { return None; }
        self.swap(0, self.len - 1);
        self.len -= 1;
        
        // SAFETY: Element was initialized, we decrement len to "transfer" ownership to caller.
        let val = unsafe { self.ptr.as_ptr().add(self.len).read() };
        if self.len > 0 { self.sift_down(0); }
        Some(val)
    }

    /// Removes the element at the given index.
    /// 
    /// # Panics
    /// Panics if `idx` is out of bounds.
    pub fn remove_at(&mut self, idx: usize) -> T {
        assert!(idx < self.len, "PriorityQueue::remove_at: out of bounds");
        self.swap(idx, self.len - 1);
        self.len -= 1;
        
        // SAFETY: We read the value from the last initialized slot.
        let val = unsafe { self.ptr.as_ptr().add(self.len).read() };
        if idx < self.len { 
            self.sift_down(idx); 
            self.sift_up(idx); 
        }
        val
    }
 
    /// Rebuilds the entire heap from scratch. O(n) complexity.
    pub fn rebuild(&mut self) {
        if self.len <= 1 { return; }
        let mut i = (self.len / 2).wrapping_sub(1);
        loop {
            self.sift_down(i);
            if i == 0 { break; }
            i -= 1;
        }
    }

    /// Clears the queue and moves all elements into the provided slice in sorted order.
    /// 
    /// # Panics
    /// Panics if `out.len()` is smaller than `self.len()`.
    pub fn drain_sorted(&mut self, out: &mut [T]) where T: Copy {
        assert!(out.len() >= self.len, "drain_sorted: out slice too short");
        let mut i = 0;
        while let Some(val) = self.pop() {
            // SAFETY: Manual copy is safe for T: Copy.
            let dst = unsafe { out.as_mut_ptr().add(i) } as *mut u8;
            let src = &val as *const T as *const u8;
            unsafe { simd::copy_bytes(dst, src, core::mem::size_of::<T>()) };
            i += 1;
        }
    }
 
    /// Calls the provided closure for every element in the underlying buffer.
    /// Note: Elements are NOT visited in sorted order.
    pub fn for_each<F: FnMut(&T)>(&self, mut f: F) {
        for i in 0..self.len { 
            // SAFETY: Iterating within initialized bounds.
            f(unsafe { &*self.ptr.as_ptr().add(i) }); 
        }
    }

    /// Re-establishes the heap invariant by "sifting up" an element at `idx`.
    fn sift_up(&mut self, mut idx: usize) {
        while idx > 0 {
            let parent = (idx - 1) / 2;
            // SAFETY: Indices are calculated within valid heap bounds.
            unsafe {
                if self.ctx.less(
                    &*self.ptr.as_ptr().add(idx),
                    &*self.ptr.as_ptr().add(parent),
                ) {
                    self.swap(idx, parent);
                    idx = parent;
                } else { break; }
            }
        }
    }
 
    /// Re-establishes the heap invariant by "sifting down" an element at `idx`.
    fn sift_down(&mut self, mut idx: usize) {
        loop {
            let left  = 2 * idx + 1;
            let right = 2 * idx + 2;
            let mut best = idx;
 
            // SAFETY: Addr arithmetic is safe as we check against self.len.
            unsafe {
                if left < self.len && self.ctx.less(
                    &*self.ptr.as_ptr().add(left),
                    &*self.ptr.as_ptr().add(best),
                ) { best = left; }
                if right < self.len && self.ctx.less(
                    &*self.ptr.as_ptr().add(right),
                    &*self.ptr.as_ptr().add(best),
                ) { best = right; }
            }
 
            if best == idx { break; }
            self.swap(idx, best);
            idx = best;
        }
    }

    /// Swaps elements at indices `a` and `b`.
    #[inline]
    fn swap(&mut self, a: usize, b: usize) {
        // SAFETY: Caller must ensure a and b are within 0..len.
        unsafe { core::ptr::swap(self.ptr.as_ptr().add(a), self.ptr.as_ptr().add(b)); }
    }

    /// Attempts to double the capacity of the queue.
    #[cold]
    fn try_grow(&mut self) -> bool {
        let new_cap = if self.cap == 0 { 4 } else { self.cap * 2 };
        let layout  = match Layout::array::<T>(new_cap) { Ok(l) => l, Err(_) => return false };
        let new_ptr = match unsafe { self.alloc.alloc(layout) } {
            Some(p) => p.cast::<T>(), None => return false,
        };
        
        if self.cap > 0 {
            // SAFETY: We copy existing initialized elements to the new allocation.
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
    /// Drops the queue and all its elements, returning memory to the allocator.
    fn drop(&mut self) {
        if self.cap == 0 { return; }
        for i in 0..self.len {
            // SAFETY: Dropping only initialized elements.
            unsafe { core::ptr::drop_in_place(self.ptr.as_ptr().add(i)) };
        }
        // SAFETY: Deallocating the buffer using the correct layout.
        unsafe { self.alloc.dealloc(self.ptr.cast(), Layout::array::<T>(self.cap).unwrap()) };
    }
}