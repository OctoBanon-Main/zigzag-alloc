//! Lock-free bump (linear) allocator.
//!
//! A bump allocator maintains a single monotonically-increasing offset into a
//! fixed backing buffer.  Each allocation simply advances the offset by the
//! required (aligned) size.
//!
//! ## Trade-offs
//!
//! | Pro | Con |
//! |-----|-----|
//! | O(1) allocation with zero fragmentation | No individual deallocation |
//! | Thread-safe without a mutex (CAS loop) | Must call [`reset`] to reclaim memory |
//! | Minimal memory overhead per allocation | Buffer size fixed at construction |
//!
//! ## Usage
//!
//! ```rust,ignore
//! static mut BUF: [u8; 4096] = [0u8; 4096];
//! let bump = BumpAllocator::new(unsafe { &mut BUF });
//! let vec: ExVec<u32> = ExVec::new(&bump);
//! ```

use core::{
    alloc::Layout,
    ptr::NonNull,
    sync::atomic::{AtomicUsize, Ordering},
};

use super::allocator::Allocator;
use crate::simd;

/// A thread-safe, lock-free bump allocator backed by a static byte buffer.
///
/// Allocations are fulfilled by atomically advancing an internal offset;
/// no individual deallocation is possible.  Call [`reset`](Self::reset) (or
/// [`reset_zeroed`](Self::reset_zeroed)) to reclaim the entire buffer at once.
///
/// # Thread Safety
///
/// `BumpAllocator` is `Sync` + `Send`.  Concurrent `alloc` calls use a
/// compare-and-swap loop to guarantee each thread receives a non-overlapping
/// slice of the backing buffer.
///
/// # Invariants
///
/// * `start` always points to the first byte of the backing buffer.
/// * `0 <= offset <= size` at all times.
/// * Memory in `[start, start + offset)` has been "handed out" and must not be
///   written by the allocator itself until `reset` is called.
pub struct BumpAllocator {
    /// Pointer to the first byte of the backing buffer.
    start:  *mut u8,
    /// Total size of the backing buffer in bytes.
    size:   usize,
    /// Monotonically-increasing byte offset; updated atomically.
    offset: AtomicUsize,
}

// SAFETY: The bump allocator never aliases its own backing buffer after handing
// out a pointer, and the CAS loop ensures no two threads receive the same range.
// Therefore it is safe to share (`Sync`) and transfer (`Send`) across threads.
unsafe impl Sync for BumpAllocator {}
unsafe impl Send for BumpAllocator {}

impl BumpAllocator {
    /// Creates a new bump allocator that owns the given static buffer.
    ///
    /// The buffer must outlive the allocator (enforced by the `'static` bound).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// static mut BUF: [u8; 65536] = [0u8; 65536];
    /// let bump = BumpAllocator::new(unsafe { &mut BUF });
    /// ```
    pub fn new(buf: &'static mut [u8]) -> Self {
        Self {
            start:  buf.as_mut_ptr(),
            size:   buf.len(),
            offset: AtomicUsize::new(0),
        }
    }

    /// Returns the number of bytes that have been allocated from the buffer.
    #[inline]
    pub fn used(&self) -> usize { self.offset.load(Ordering::Relaxed) }

    /// Returns the number of bytes still available for allocation.
    #[inline]
    pub fn remaining(&self) -> usize { self.size.saturating_sub(self.used()) }

    /// Returns the total capacity of the backing buffer in bytes.
    #[inline]
    pub fn capacity(&self) -> usize { self.size }

    /// Resets the allocator, making the entire buffer available again.
    ///
    /// # Safety
    ///
    /// All pointers previously returned by [`alloc`](Allocator::alloc) or
    /// [`alloc_slice`](Self::alloc_slice) become **invalid** after this call.
    /// Any subsequent access to those pointers is undefined behaviour.
    #[inline]
    pub fn reset(&mut self) {
        // Relaxed ordering is sufficient here because `reset` takes `&mut self`,
        // ensuring exclusive access — no concurrent `alloc` can be in progress.
        self.offset.store(0, Ordering::Relaxed);
    }

    /// Resets the allocator and zero-fills the entire backing buffer with SIMD.
    ///
    /// Useful when the buffer may contain sensitive data that should not persist.
    ///
    /// # Safety
    ///
    /// Same as [`reset`](Self::reset): all previously returned pointers are
    /// invalidated.
    pub fn reset_zeroed(&mut self) {
        self.offset.store(0, Ordering::Relaxed);
        // SAFETY: `start` is valid for `size` bytes because it was obtained
        // from a `&'static mut [u8]` of exactly that length.
        unsafe { simd::fill_bytes(self.start, 0, self.size) };
    }

    /// Allocates a mutable byte slice of `size` bytes with the given `align`.
    ///
    /// Returns `None` if the remaining capacity is insufficient or if `align`
    /// is not a power of two.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let slice: &mut [u8] = bump.alloc_slice(64, 16).unwrap();
    /// ```
    pub fn alloc_slice(&self, size: usize, align: usize) -> Option<&mut [u8]> {
        let layout = Layout::from_size_align(size, align).ok()?;
        let ptr = self.alloc_raw(layout)?;
        // SAFETY: `alloc_raw` returns a pointer valid for exactly `size` bytes
        // with the requested alignment.  No aliasing can occur because `alloc_raw`
        // atomically claimed this range.
        unsafe {
            Some(core::slice::from_raw_parts_mut(ptr.as_ptr(), size))
        }
    }

    /// Core lock-free allocation primitive.
    ///
    /// Attempts to claim `layout.size()` bytes (properly aligned) from the
    /// backing buffer by atomically incrementing the offset.
    ///
    /// Returns `None` if there is not enough space.
    ///
    /// # Implementation Notes
    ///
    /// The CAS loop (`compare_exchange_weak`) is the only synchronisation
    /// primitive used.  On failure it retries with the latest observed value,
    /// which is correct because `offset` only ever increases.
    fn alloc_raw(&self, layout: Layout) -> Option<NonNull<u8>> {
        // SAFETY: Alignment is computed with standard bit-masking arithmetic.
        // `compare_exchange_weak` guarantees that the range `[aligned, end)` is
        // claimed by exactly one thread at a time.
        let size  = layout.size();
        let align = layout.align();
        let mut current = self.offset.load(Ordering::Relaxed);
        loop {
            // Round `current` up to the required alignment.
            let aligned = current.checked_add(align - 1)? & !(align - 1);
            let end     = aligned.checked_add(size)?;
            if end > self.size { return None; }

            // Atomically claim the range [aligned, end).
            // `AcqRel` on success ensures the written data is visible to other
            // threads that subsequently observe `offset >= end`.
            match self.offset.compare_exchange_weak(
                current, end,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    // SAFETY: `aligned` is within `[0, size)` of the backing
                    // buffer, so `start + aligned` is a valid, aligned pointer.
                    return NonNull::new(unsafe { self.start.add(aligned) });
                }
                Err(actual) => current = actual,
            }
        }
    }
}

impl Allocator for BumpAllocator {
    /// Allocates a block satisfying `layout` from the backing buffer.
    ///
    /// # Safety
    ///
    /// * `layout.size()` must be greater than zero.
    /// * The returned pointer is valid until [`reset`](Self::reset) is called
    ///   or the allocator is dropped (though dropping does not invalidate the
    ///   underlying `'static` buffer).
    unsafe fn alloc(&self, layout: Layout) -> Option<NonNull<u8>> {
        self.alloc_raw(layout)
    }

    /// No-op.  Individual deallocation is not supported by a bump allocator.
    ///
    /// Use [`reset`](Self::reset) to reclaim all memory at once.
    ///
    /// # Safety
    ///
    /// Even though this method is a no-op, callers must not use `ptr` after
    /// calling `dealloc` — doing so would be use-after-free once `reset` is
    /// eventually called.
    unsafe fn dealloc(&self, _: NonNull<u8>, _: Layout) {}
}