//! Heap-based priority queue with an explicit allocator.
//!
//! [`ExPriorityQueue`] implements a binary heap whose ordering is determined
//! by a caller-supplied [`OrdContext<T>`].  Depending on whether the context
//! implements a min-ordering or a max-ordering, the queue acts as a min-heap
//! or a max-heap respectively.
//!
//! ## Complexity
//!
//! | Operation | Complexity |
//! |-----------|-----------|
//! | [`push`](ExPriorityQueue::push) | O(log n) amortised |
//! | [`pop`](ExPriorityQueue::pop) | O(log n) |
//! | [`peek`](ExPriorityQueue::peek) | O(1) |
//! | [`push_slice`](ExPriorityQueue::push_slice) | O(n) via `rebuild` |
//! | [`rebuild`](ExPriorityQueue::rebuild) | O(n) |
//!
//! ## Growth Policy
//!
//! The initial capacity is **4**; subsequent growth doubles the capacity.

use core::{alloc::Layout, marker::PhantomData, ptr::NonNull};

use crate::alloc::allocator::Allocator;
use crate::simd;
use super::OrdContext;

/// A binary-heap priority queue backed by an explicit allocator.
///
/// The "top" element is always the one for which `ctx.less(top, other)` is
/// `true` for every other element — i.e. the element that the context
/// considers *smallest*.  Use a min-context for a min-heap, a max-context for
/// a max-heap.
///
/// # Invariants
///
/// * `ptr` is a valid allocation of `cap * size_of::<T>()` bytes when `cap > 0`.
/// * Elements at indices `0..len` are fully initialised and satisfy the heap
///   property: for every `i`, `ctx.less(data[i], data[2*i+1])` and
///   `ctx.less(data[i], data[2*i+2])` are *false* (parent ≤ children).
///
/// # Lifetime
///
/// The allocator reference `'a` must outlive the `ExPriorityQueue`.
pub struct ExPriorityQueue<'a, T, C: OrdContext<T>> {
    /// Pointer to the heap array.
    ptr: NonNull<T>,
    /// Number of elements currently in the heap.
    len: usize,
    /// Buffer capacity in elements.
    cap: usize,
    /// Allocator used for buffer management.
    alloc: &'a dyn Allocator,
    /// Ordering context.
    ctx: C,
    _marker: PhantomData<T>,
}

impl<'a, T, C: OrdContext<T>> ExPriorityQueue<'a, T, C> {
    /// Creates a new, empty priority queue.
    ///
    /// No allocation is performed until the first element is pushed.
    pub fn new(alloc: &'a dyn Allocator, ctx: C) -> Self {
        Self { ptr: NonNull::dangling(), len: 0, cap: 0, alloc, ctx, _marker: PhantomData }
    }

    /// Returns the number of elements in the queue.
    #[inline]
    pub fn len(&self) -> usize { self.len }

    /// Returns `true` if the queue is empty.
    #[inline]
    pub fn is_empty(&self) -> bool { self.len == 0 }

    /// Returns the current buffer capacity in elements.
    #[inline]
    pub fn capacity(&self) -> usize { self.cap }

    /// Returns a reference to the top (smallest/largest) element, or `None`
    /// if the queue is empty.
    ///
    /// The top element is always at index 0 in the heap array.
    pub fn peek(&self) -> Option<&T> {
        if self.len == 0 { return None; }
        // SAFETY: `len > 0` implies at least one initialised element at index 0.
        Some(unsafe { &*self.ptr.as_ptr() })
    }

    /// Pushes `value` onto the queue.
    ///
    /// # Panics
    ///
    /// Panics if the backing allocator cannot grow the buffer.
    pub fn push(&mut self, value: T) {
        assert!(self.try_push(value).is_ok(), "ExPriorityQueue: OOM")
    }

    /// Attempts to push `value`, returning `Err(value)` if OOM.
    pub fn try_push(&mut self, value: T) -> Result<(), T> {
        if self.len == self.cap && !self.try_grow() { return Err(value); }

        // SAFETY: After `try_grow`, `cap > len`, so `ptr + len` is within the
        // buffer and currently uninitialised — safe to write.
        unsafe { self.ptr.as_ptr().add(self.len).write(value) };
        self.len += 1;
        self.sift_up(self.len - 1);
        Ok(())
    }

    /// Pushes all elements in `items` and rebuilds the heap in O(n) time.
    ///
    /// More efficient than calling [`push`](Self::push) in a loop when many
    /// elements are added at once.
    ///
    /// # Panics
    ///
    /// Panics if the backing allocator fails during growth.
    pub fn push_slice(&mut self, items: &[T]) where T: Copy {
        let n = items.len();
        if n == 0 { return; }

        while self.cap < self.len + n {
            assert!(self.try_grow(), "ExPriorityQueue: OOM")
        }

        let dst = unsafe { self.ptr.as_ptr().add(self.len) } as *mut u8;
        let src = items.as_ptr() as *const u8;
        // SAFETY: `dst` points to `n` uninitialised slots; `src` is a valid
        // slice of `n` elements.  `T: Copy` ensures no destructor is skipped.
        unsafe { simd::copy_bytes(dst, src, n * core::mem::size_of::<T>()); }
        self.len += n;

        // Re-establish the heap property over all elements in O(n).
        self.rebuild();
    }

    /// Removes and returns the top element, or `None` if empty.
    ///
    /// Swaps the root with the last element, decrements `len`, and sifts the
    /// new root down to restore the heap property.
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 { return None; }
        self.swap(0, self.len - 1);
        self.len -= 1;

        // SAFETY: The element we swapped to the end is now outside `0..len`,
        // so it is "logically removed".  Reading it transfers ownership.
        let val = unsafe { self.ptr.as_ptr().add(self.len).read() };
        if self.len > 0 { self.sift_down(0); }
        Some(val)
    }

    /// Removes the element at `idx` and returns it.
    ///
    /// Sifts up *and* down to restore the heap property regardless of whether
    /// the replacement element is smaller or larger than its neighbours.
    ///
    /// # Panics
    ///
    /// Panics if `idx >= self.len`.
    pub fn remove_at(&mut self, idx: usize) -> T {
        assert!(idx < self.len, "PriorityQueue::remove_at: out of bounds");
        self.swap(idx, self.len - 1);
        self.len -= 1;

        // SAFETY: Same as `pop`; the removed element is now past the live range.
        let val = unsafe { self.ptr.as_ptr().add(self.len).read() };
        if idx < self.len {
            self.sift_down(idx);
            self.sift_up(idx);
        }
        val
    }

    /// Rebuilds the heap from scratch in O(n) via bottom-up heapification.
    ///
    /// Called automatically by [`push_slice`](Self::push_slice); can also be
    /// called manually after bulk insertion via unsafe direct writes.
    pub fn rebuild(&mut self) {
        if self.len <= 1 { return; }
        // Start at the last internal node and sift each one down.
        let mut i = (self.len / 2).wrapping_sub(1);
        loop {
            self.sift_down(i);
            if i == 0 { break; }
            i -= 1;
        }
    }

    /// Drains the queue in sorted order into `out`.
    ///
    /// Repeatedly pops the top element and copies it into successive positions
    /// of `out`.  After this call the queue is empty.
    ///
    /// # Panics
    ///
    /// Panics if `out.len() < self.len()`.
    pub fn drain_sorted(&mut self, out: &mut [T]) where T: Copy {
        assert!(out.len() >= self.len, "drain_sorted: out slice too short");
        let mut i = 0;
        while let Some(val) = self.pop() {
            let dst = unsafe { out.as_mut_ptr().add(i) } as *mut u8;
            let src = &val as *const T as *const u8;
            // SAFETY: `dst` points to an initialised or uninitialised slot of
            // type `T`; `src` is the address of a local `T` value.  `T: Copy`
            // means the copy is bitwise-valid.
            unsafe { simd::copy_bytes(dst, src, core::mem::size_of::<T>()) };
            i += 1;
        }
    }

    /// Calls `f` with a reference to each element in **unspecified** order.
    ///
    /// Elements are visited in heap-array order (not sorted order).
    pub fn for_each<F: FnMut(&T)>(&self, mut f: F) {
        for i in 0..self.len {
            // SAFETY: Indices `0..len` are fully initialised.
            f(unsafe { &*self.ptr.as_ptr().add(i) });
        }
    }

    /// Moves the element at `idx` upward until the heap property is restored.
    ///
    /// Called after inserting a new element at the end of the array.
    fn sift_up(&mut self, mut idx: usize) {
        while idx > 0 {
            let parent = (idx - 1) / 2;
            // SAFETY: `idx` and `parent` are both within `0..len`.
            unsafe {
                if self.ctx.less(
                    &*self.ptr.as_ptr().add(idx),
                    &*self.ptr.as_ptr().add(parent),
                ) {
                    self.swap(idx, parent);
                    idx = parent;
                } else {
                    break;
                }
            }
        }
    }

    /// Moves the element at `idx` downward until the heap property is restored.
    ///
    /// Called after removing the root or after bulk insertion via `rebuild`.
    fn sift_down(&mut self, mut idx: usize) {
        loop {
            let left  = 2 * idx + 1;
            let right = 2 * idx + 2;
            let mut best = idx;

            // SAFETY: All index arithmetic is bounds-checked against `self.len`.
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

    /// Swaps the elements at indices `a` and `b`.
    ///
    /// # Safety
    ///
    /// Both `a` and `b` must be within `0..len`; this is guaranteed by all
    /// internal callers.
    #[inline]
    fn swap(&mut self, a: usize, b: usize) {
        // SAFETY: `ptr::swap` requires both pointers to be valid and non-overlapping
        // (since `a != b` in all call sites) within the initialised region.
        unsafe { core::ptr::swap(self.ptr.as_ptr().add(a), self.ptr.as_ptr().add(b)); }
    }

    /// Doubles the buffer capacity, migrating existing elements.
    ///
    /// Returns `false` if the allocator fails.
    #[cold]
    fn try_grow(&mut self) -> bool {
        let new_cap = if self.cap == 0 { 4 } else { self.cap * 2 };
        let layout  = match Layout::array::<T>(new_cap) {
            Ok(l) => l, Err(_) => return false,
        };
        let new_ptr = match unsafe { self.alloc.alloc(layout) } {
            Some(p) => p.cast::<T>(),
            None    => return false,
        };

        if self.cap > 0 {
            // SAFETY: `new_ptr` is valid for `new_cap` elements; we copy only
            // `len` initialised elements.  The old buffer is then released.
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
    /// Drops all elements and releases the backing buffer.
    fn drop(&mut self) {
        if self.cap == 0 { return; }
        for i in 0..self.len {
            // SAFETY: Indices `0..len` are fully initialised.
            unsafe { core::ptr::drop_in_place(self.ptr.as_ptr().add(i)) };
        }
        // SAFETY: `ptr` was obtained from `self.alloc` with this exact layout.
        unsafe { self.alloc.dealloc(self.ptr.cast(), Layout::array::<T>(self.cap).unwrap()) };
    }
}