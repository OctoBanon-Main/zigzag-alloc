use core::{
    alloc::Layout,
    ptr::NonNull,
    sync::atomic::{AtomicUsize, Ordering},
};

use super::allocator::Allocator;
use crate::simd;

/// A thread-safe, lock-free bump allocator for fast linear allocations.
///
/// It uses a fixed-size buffer and an [`AtomicUsize`] offset to allocate memory.
/// Memory is never individually deallocated; use [`reset`](Self::reset) to
/// reclaim all memory at once.
pub struct BumpAllocator {
    start:  *mut u8,
    size:   usize,
    offset: AtomicUsize,
}

unsafe impl Sync for BumpAllocator {}
unsafe impl Send for BumpAllocator {}

impl BumpAllocator {
    /// Creates a new bump allocator using a static buffer.
    pub fn new(buf: &'static mut [u8]) -> Self {
        Self {
            start:  buf.as_mut_ptr(),
            size:   buf.len(),
            offset: AtomicUsize::new(0),
        }
    }

    /// Returns the number of bytes currently allocated from the buffer.
    #[inline] pub fn used(&self)      -> usize { self.offset.load(Ordering::Relaxed) }
    /// Returns the number of bytes available before the allocator is exhausted.
    #[inline] pub fn remaining(&self) -> usize { self.size.saturating_sub(self.used()) }
    /// Returns the total size of the backing buffer.
    #[inline] pub fn capacity(&self)  -> usize { self.size }

    /// Resets the allocator offset to zero.
    /// 
    /// # Safety
    /// All previously allocated references from this allocator are invalidated.
    #[inline]
    pub fn reset(&mut self) {
        self.offset.store(0, Ordering::Relaxed);
    }

    /// Resets the allocator and zeroes out the entire backing buffer using SIMD.
    pub fn reset_zeroed(&mut self) {
        self.offset.store(0, Ordering::Relaxed);
        unsafe { simd::fill_bytes(self.start, 0, self.size) };
    }

    /// Allocates a mutable byte slice of a specific size and alignment.
    pub fn alloc_slice(&self, size: usize, align: usize) -> Option<&mut [u8]> {
        let layout = Layout::from_size_align(size, align).ok()?;
        let ptr = self.alloc_raw(layout)?;
        // SAFETY: alloc_raw guarantees the pointer is valid for `size` and correctly aligned.
        unsafe {
            Some(core::slice::from_raw_parts_mut(ptr.as_ptr(), size))
        }
    }

    /// Attempts a lock-free allocation by incrementing the internal offset.
    ///
    /// This implementation uses a `compare_exchange` loop to ensure thread safety
    /// without using traditional mutexes.
    fn alloc_raw(&self, layout: Layout) -> Option<NonNull<u8>> {
        // SAFETY: Correct alignment is calculated using bitwise operations.
        // The AtomicUsize ensures that concurrent threads do not receive 
        // overlapping memory regions.
        let size  = layout.size();
        let align = layout.align();
        let mut current = self.offset.load(Ordering::Relaxed);
        loop {
            let aligned = current.checked_add(align - 1)? & !(align - 1);
            let end     = aligned.checked_add(size)?;
            if end > self.size { return None; }
            // Atomic Compare-And-Swap to ensure thread safety
            match self.offset.compare_exchange_weak(
                current, end,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_)    => return NonNull::new(unsafe { self.start.add(aligned) }),
                Err(act) => current = act,
            }
        }
    }
}

impl Allocator for BumpAllocator {
    unsafe fn alloc(&self, layout: Layout) -> Option<NonNull<u8>> {
        self.alloc_raw(layout)
    }
    unsafe fn dealloc(&self, _: NonNull<u8>, _: Layout) {}
}