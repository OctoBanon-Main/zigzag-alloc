//! Fixed-size object pool allocator.
//!
//! A pool allocator pre-allocates a contiguous *slab* of equal-sized *blocks*
//! and maintains a lock-free free-list over them.  Allocation pops one block
//! from the free-list; deallocation pushes it back.
//!
//! ## Characteristics
//!
//! | Property | Value |
//! |----------|-------|
//! | Allocation / deallocation cost | O(1), typically a single CAS |
//! | Fragmentation | Zero (fixed-size blocks) |
//! | Thread safety | Lock-free (`AtomicPtr` CAS loop) |
//! | Max allocation size | `block_layout.size()` |
//! | Individual deallocation | ✅ Supported |
//!
//! ## Example
//!
//! ```rust,ignore
//! let pool = PoolAllocator::typed::<[u8; 64]>(&sys, 128).unwrap();
//! let ptr  = unsafe { pool.alloc(Layout::new::<[u8; 64]>()) }.unwrap();
//! // ... use memory ...
//! unsafe { pool.dealloc(ptr, Layout::new::<[u8; 64]>()) };
//! ```

use core::{
    alloc::Layout,
    mem,
    ptr::{self, NonNull},
    sync::atomic::{AtomicPtr, Ordering},
};

use super::allocator::Allocator;
use crate::simd;

// ── FreeNode ─────────────────────────────────────────────────────────────────

/// An intrusive free-list node overlaid on top of unused pool blocks.
///
/// When a block is free its first bytes are reinterpreted as a `FreeNode`
/// holding a pointer to the next free block.  When a block is in use the
/// `FreeNode` overlay is no longer valid and must not be accessed.
struct FreeNode {
    /// Raw pointer to the next free block, or null if this is the last node.
    next: *mut FreeNode,
}

/// A lock-free, fixed-size block allocator backed by a single pre-allocated
/// slab.
///
/// At construction time the slab is divided into equal-sized *blocks* whose
/// size is at least `max(item_layout.size(), size_of::<FreeNode>())` and
/// whose alignment is at least `max(item_layout.align(), align_of::<FreeNode>())`.
/// All blocks are linked into a free-list; allocation atomically pops the
/// head; deallocation atomically pushes to the head.
///
/// # Constraints
///
/// * Allocations larger than `block_layout.size()` or with stricter alignment
///   than `block_layout.align()` are rejected (return `None`).
/// * The pool has a fixed `capacity`; once all blocks are in use, allocation
///   returns `None`.
///
/// # Thread Safety
///
/// `PoolAllocator` is `Sync` + `Send` when the backing allocator is `Sync` +
/// `Send`.  Both `alloc` and `dealloc` use lock-free CAS loops.
pub struct PoolAllocator<A: Allocator> {
    /// Backing allocator that owns the slab.
    backing:      A,
    /// Padded layout of a single block (>= item_layout, >= FreeNode layout).
    block_layout: Layout,
    /// Layout of the entire slab (`block_layout.size() * capacity`).
    slab_layout:  Layout,
    /// Pointer to the start of the slab.
    slab:         NonNull<u8>,
    /// Lock-free head of the free-list; null when the pool is exhausted.
    free_head:    AtomicPtr<FreeNode>,
    /// Total number of blocks in the pool.
    capacity:     usize,
}

// SAFETY: The free-list is maintained via lock-free CAS operations, ensuring
// that concurrent `alloc` and `dealloc` calls never corrupt the list.
// The slab pointer itself is not mutated after construction.
unsafe impl<A: Allocator + Sync> Sync for PoolAllocator<A> {}
unsafe impl<A: Allocator + Send> Send for PoolAllocator<A> {}

impl<A: Allocator> PoolAllocator<A> {
    /// Creates a new pool with `capacity` blocks sized for `item_layout`.
    ///
    /// The effective block size is `max(item_layout.size(), size_of::<FreeNode>())`
    /// padded to the effective alignment.  Returns `None` if the backing
    /// allocator fails to allocate the slab.
    ///
    /// # Arguments
    ///
    /// * `backing`     — Allocator used to allocate and later free the slab.
    /// * `item_layout` — Layout of the items that will be stored in the pool.
    /// * `capacity`    — Maximum number of live allocations the pool can hold.
    pub fn new(backing: A, item_layout: Layout, capacity: usize) -> Option<Self> {
        // The block must be large enough to hold either a user item or the
        // FreeNode overlay (whichever is bigger), and aligned sufficiently for
        // both.
        let block_size  = item_layout.size() .max(mem::size_of ::<FreeNode>());
        let block_align = item_layout.align().max(mem::align_of::<FreeNode>());

        let block_layout = Layout::from_size_align(block_size, block_align)
            .ok()?
            .pad_to_align();   // ensure stride is a multiple of alignment

        let total_size  = block_layout.size().checked_mul(capacity)?;
        let slab_layout = Layout::from_size_align(total_size, block_layout.align()).ok()?;

        // SAFETY: `slab_layout` has non-zero size (capacity > 0 is assumed by
        // the caller; zero capacity results in a zero-sized slab which the
        // backing allocator may or may not accept).
        let slab = unsafe { backing.alloc(slab_layout)? };

        // Zero-fill so that all FreeNode overlays start in a clean state.
        // SAFETY: `slab` is valid for `total_size` bytes as guaranteed by the
        // backing allocator.
        unsafe { simd::fill_bytes(slab.as_ptr(), 0, total_size) };

        // Build the free-list in reverse order so that the first block ends up
        // at the head (i.e. allocations proceed forward through the slab).
        let mut head: *mut FreeNode = ptr::null_mut();
        for i in (0..capacity).rev() {
            // SAFETY: `i * block_layout.size()` is within the slab bounds
            // because `i < capacity` and `total_size = block_layout.size() * capacity`.
            let block = unsafe {
                slab.as_ptr().add(i * block_layout.size()) as *mut FreeNode
            };
            // SAFETY: `block` points to a zero-initialised block that is at
            // least `size_of::<FreeNode>()` bytes — writing a FreeNode is safe.
            unsafe { (*block).next = head };
            head = block;
        }

        Some(Self {
            backing,
            block_layout,
            slab_layout,
            slab,
            free_head: AtomicPtr::new(head),
            capacity,
        })
    }

    /// Convenience constructor that derives `item_layout` from the type `T`.
    ///
    /// Equivalent to `PoolAllocator::new(backing, Layout::new::<T>(), capacity)`.
    pub fn typed<T>(backing: A, capacity: usize) -> Option<Self> {
        Self::new(backing, Layout::new::<T>(), capacity)
    }

    /// Returns the maximum number of blocks this pool can provide simultaneously.
    #[inline]
    pub fn capacity(&self) -> usize { self.capacity }

    /// Returns the padded layout of a single pool block.
    #[inline]
    pub fn block_layout(&self) -> Layout { self.block_layout }

    /// Returns the number of blocks currently on the free-list.
    ///
    /// This traverses the entire free-list and is O(free_count), so use only
    /// for diagnostics / debugging.
    pub fn free_count(&self) -> usize {
        let mut n    = 0usize;
        let mut node = self.free_head.load(Ordering::Relaxed);
        while !node.is_null() {
            n += 1;
            // SAFETY: Every non-null node on the free-list was written by
            // `dealloc` or during construction and points to a valid `FreeNode`.
            node = unsafe { (*node).next };
        }
        n
    }

    /// Zeroes the block at `ptr` and returns it to the pool.
    ///
    /// Useful when blocks may contain sensitive data.
    ///
    /// # Safety
    ///
    /// * `ptr` must have been obtained from this pool via [`alloc`](Allocator::alloc).
    /// * `ptr` must not be used after this call.
    pub unsafe fn dealloc_zeroed(&self, ptr: NonNull<u8>) {
        // SAFETY: `ptr` is valid for `block_layout.size()` bytes — guaranteed
        // by the caller.
        unsafe { simd::fill_bytes(ptr.as_ptr(), 0, self.block_layout.size()) };
        // SAFETY: Forwarding to `dealloc` which handles the free-list push.
        unsafe { self.dealloc(ptr, self.block_layout) };
    }

    /// Zeroes the entire slab without modifying the free-list.
    ///
    /// Intended for secure teardown.  **Do not call while any block is in use.**
    ///
    /// # Safety
    ///
    /// All blocks — including live ones — will have their bytes set to zero.
    /// Any live references into the pool will read garbage (all zeros).
    pub unsafe fn wipe_slab(&self) {
        // SAFETY: `slab` is valid for `slab_layout.size()` bytes as established
        // in the constructor.
        unsafe { simd::fill_bytes(self.slab.as_ptr(), 0, self.slab_layout.size()) };
    }
}

impl<A: Allocator> Allocator for PoolAllocator<A> {
    /// Pops one block from the free-list and returns a pointer to it.
    ///
    /// Returns `None` if the pool is exhausted or if `layout` is too large /
    /// too strictly aligned for the pool's block size.
    ///
    /// # Safety
    ///
    /// * `layout.size()` must be ≤ `block_layout.size()`.
    /// * `layout.align()` must be ≤ `block_layout.align()`.
    /// * The caller must eventually call [`dealloc`](Allocator::dealloc) with
    ///   the same pointer (and any layout whose size and align fit the pool).
    unsafe fn alloc(&self, layout: Layout) -> Option<NonNull<u8>> {
        if layout.size()  > self.block_layout.size()
        || layout.align() > self.block_layout.align()
        {
            return None;
        }

        // Atomically pop the head of the free-list.
        let mut head = self.free_head.load(Ordering::Acquire);
        loop {
            // If head is null the pool is exhausted.
            let node = NonNull::new(head)?;
            // SAFETY: `head` is a valid `FreeNode` pointer — maintained by the
            // constructor and the `dealloc` push path.
            let next = unsafe { (*head).next };
            match self.free_head.compare_exchange_weak(
                head, next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_)    => return Some(node.cast()),
                Err(act) => head = act,
            }
        }
    }

    /// Pushes a block back onto the free-list.
    ///
    /// The `_layout` parameter is ignored; what matters is that `ptr` belongs
    /// to this pool's slab.
    ///
    /// # Safety
    ///
    /// * `ptr` must have been obtained from **this** pool via
    ///   [`alloc`](Allocator::alloc).
    /// * `ptr` must not be accessed after this call.
    /// * Calling `dealloc` twice with the same pointer is undefined behaviour
    ///   (double-free corrupts the free-list).
    unsafe fn dealloc(&self, ptr: NonNull<u8>, _layout: Layout) {
        let node = ptr.as_ptr() as *mut FreeNode;

        // Atomically push the block onto the head of the free-list.
        let mut head = self.free_head.load(Ordering::Relaxed);
        loop {
            // SAFETY: `node` is within the slab so it is large enough for a
            // `FreeNode`.  Writing before the CAS succeeds is safe because if
            // the CAS fails we simply overwrite `next` again on the next attempt.
            unsafe { ptr::write(node, FreeNode { next: head }) };
            match self.free_head.compare_exchange_weak(
                head, node,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_)    => return,
                Err(act) => head = act,
            }
        }
    }
}

impl<A: Allocator> Drop for PoolAllocator<A> {
    /// Returns the entire slab to the backing allocator.
    fn drop(&mut self) {
        // SAFETY: `slab` was obtained from `backing` with `slab_layout` in the
        // constructor, and `backing` is still alive (it is owned by `self`).
        unsafe { self.backing.dealloc(self.slab, self.slab_layout) };
    }
}