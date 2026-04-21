//! Instrumented allocator wrapper for diagnostics and testing.
//!
//! [`CountingAllocator`] wraps any [`Allocator`] and records per-call
//! statistics: number of allocations / deallocations and total bytes involved.
//! It is transparent — every call is forwarded unchanged to the inner
//! allocator — and adds negligible overhead (four `Cell<usize>` increments).
//!
//! ## Typical Use Cases
//!
//! * **Unit tests** — assert that a data structure performs exactly the
//!   expected number of allocations.
//! * **Benchmarks** — measure live byte footprint.
//! * **Debug builds** — detect memory leaks (non-zero `bytes_live` at the end
//!   of a scope).

use core::{
    alloc::Layout,
    cell::Cell,
    ptr::NonNull,
};

use super::allocator::Allocator;

/// An [`Allocator`] wrapper that records allocation and deallocation statistics.
///
/// Wrap any existing allocator with `CountingAllocator::new(inner)` to
/// transparently instrument it.  Statistics can be queried at any time via
/// [`stats`](Self::stats) and reset via [`reset_stats`](Self::reset_stats).
///
/// # Thread Safety
///
/// `CountingAllocator` uses [`Cell`] internally, which is **not** `Sync`.
/// It is suitable for single-threaded use only.  For multi-threaded scenarios,
/// wrap an atomic-based allocator instead.
pub struct CountingAllocator<A: Allocator> {
    /// The wrapped allocator that performs actual memory management.
    pub(crate) inner: A,
    /// Total number of successful `alloc` calls since last reset.
    alloc_count:      Cell<usize>,
    /// Total number of `dealloc` calls since last reset.
    dealloc_count:    Cell<usize>,
    /// Total bytes requested via `alloc` since last reset.
    bytes_allocated:  Cell<usize>,
    /// Total bytes released via `dealloc` since last reset.
    bytes_freed:      Cell<usize>,
}

impl<A: Allocator> CountingAllocator<A> {
    /// Creates a new `CountingAllocator` wrapping `inner`.
    ///
    /// All counters start at zero.
    pub fn new(inner: A) -> Self {
        Self {
            inner,
            alloc_count:     Cell::new(0),
            dealloc_count:   Cell::new(0),
            bytes_allocated: Cell::new(0),
            bytes_freed:     Cell::new(0),
        }
    }

    /// Returns a snapshot of all statistics.
    ///
    /// Counters are cumulative since the last call to
    /// [`reset_stats`](Self::reset_stats).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let counting = CountingAllocator::new(SystemAllocator);
    /// let _ = unsafe { counting.alloc(Layout::new::<u64>()) };
    /// let stats = counting.stats();
    /// assert_eq!(stats.allocs, 1);
    /// assert_eq!(stats.bytes_allocated, 8);
    /// ```
    pub fn stats(&self) -> AllocStats {
        AllocStats {
            allocs:           self.alloc_count.get(),
            deallocs:         self.dealloc_count.get(),
            bytes_allocated:  self.bytes_allocated.get(),
            bytes_freed:      self.bytes_freed.get(),
            // `saturating_sub` prevents underflow in the pathological case where
            // a caller frees memory that was not tracked (e.g. from a different
            // allocator instance before wrapping).
            bytes_live:       self.bytes_allocated.get()
                                  .saturating_sub(self.bytes_freed.get()),
        }
    }

    /// Resets all counters to zero.
    ///
    /// Does **not** affect the underlying allocator or any live allocations.
    pub fn reset_stats(&self) {
        self.alloc_count.set(0);
        self.dealloc_count.set(0);
        self.bytes_allocated.set(0);
        self.bytes_freed.set(0);
    }
}

impl<A: Allocator> Allocator for CountingAllocator<A> {
    /// Forwards the allocation to the inner allocator and records statistics.
    ///
    /// Only successful allocations are counted; if the inner allocator returns
    /// `None`, the counters are not updated.
    ///
    /// # Safety
    ///
    /// Inherits all safety requirements from [`A::alloc`](Allocator::alloc).
    unsafe fn alloc(&self, layout: Layout) -> Option<NonNull<u8>> {
        // SAFETY: Forwarding unchanged — the caller satisfies the preconditions.
        let ptr = unsafe { self.inner.alloc(layout)? };
        self.alloc_count.set(self.alloc_count.get() + 1);
        self.bytes_allocated.set(self.bytes_allocated.get() + layout.size());
        Some(ptr)
    }

    /// Forwards the deallocation to the inner allocator and records statistics.
    ///
    /// # Safety
    ///
    /// Inherits all safety requirements from [`A::dealloc`](Allocator::dealloc).
    /// In particular, `ptr` must have been obtained from this allocator (and
    /// therefore from its inner allocator).
    unsafe fn dealloc(&self, ptr: NonNull<u8>, layout: Layout) {
        self.dealloc_count.set(self.dealloc_count.get() + 1);
        self.bytes_freed.set(self.bytes_freed.get() + layout.size());
        // SAFETY: Forwarding unchanged — the caller guarantees `ptr` and
        // `layout` match the original allocation.
        unsafe { self.inner.dealloc(ptr, layout) };
    }
}

/// A snapshot of allocation statistics produced by [`CountingAllocator::stats`].
pub struct AllocStats {
    /// Total number of successful allocations recorded.
    pub allocs:          usize,
    /// Total number of deallocations recorded.
    pub deallocs:        usize,
    /// Cumulative bytes requested across all successful allocations.
    pub bytes_allocated: usize,
    /// Cumulative bytes released across all deallocations.
    pub bytes_freed:     usize,
    /// Current live byte footprint: `bytes_allocated - bytes_freed` (saturating).
    pub bytes_live:      usize,
}