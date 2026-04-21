//! Block-based arena allocator with bulk deallocation.
//!
//! An [`ArenaAllocator`] tracks every individual allocation in a singly-linked
//! list of [`Header`]–prefixed blocks.  Each call to
//! [`alloc`](Allocator::alloc) allocates a fresh block from the *backing*
//! allocator; [`dealloc`](Allocator::dealloc) is a deliberate no-op.  All
//! memory is reclaimed at once via [`reset`](ArenaAllocator::reset) or when
//! the arena is dropped.
//!
//! ## Trade-offs
//!
//! | Pro | Con |
//! |-----|-----|
//! | Individual allocations are very cheap | Each allocation has a `Header` overhead |
//! | Bulk free is O(n) in number of allocations | `dealloc` is a no-op — no reuse within a cycle |
//! | Works with any backing [`Allocator`] | Not thread-safe (`Cell` internals) |
//!
//! ## Example
//!
//! ```rust,ignore
//! let sys = SystemAllocator;
//! let arena = ArenaAllocator::new(&sys);
//! let vec: ExVec<u32> = ExVec::new(&arena);
//! // ... use vec ...
//! arena.reset(); // frees everything at once
//! ```

use core::{
    alloc::Layout,
    cell::Cell,
    ptr::{self, NonNull},
};

use super::allocator::Allocator;
use crate::simd;

// ── Header ───────────────────────────────────────────────────────────────────

/// Internal per-block metadata written at the *start* of every arena
/// allocation.
///
/// The linked-list formed by `next` pointers allows [`ArenaAllocator::reset`]
/// to walk all live allocations and return them to the backing allocator.
///
/// # Memory Layout
///
/// ```text
/// ┌──────────────────────────┬───────────────────────────────┐
/// │   Header  (user_offset)  │   User data  (user_size)      │
/// └──────────────────────────┴───────────────────────────────┘
/// ^                          ^
/// full_ptr (backing alloc)   full_ptr + user_offset (returned to caller)
/// ```
struct Header {
    /// Pointer to the *full block* of the previous allocation, i.e. the
    /// previous `full_ptr` (not the user pointer).  `None` marks the list end.
    next:        Option<NonNull<u8>>,
    /// Layout that was passed to the backing allocator for this block
    /// (`Header` + padding + user data).  Used to reconstruct the exact layout
    /// needed for deallocation.
    full_layout: Layout,
    /// Size of the user-requested portion of this block (bytes).
    user_size:   usize,
    /// Byte offset from `full_ptr` to the first byte of user data.
    /// Computed via `Layout::extend` to satisfy alignment requirements.
    user_offset: usize,
}

// ── ArenaAllocator ───────────────────────────────────────────────────────────

/// A block-based allocator that frees all memory at once.
///
/// `ArenaAllocator<A>` is generic over its *backing* allocator `A` which is
/// called for each individual block allocation and for every block
/// deallocation during [`reset`](Self::reset).
///
/// The internal state uses [`Cell`] to allow shared (`&self`) allocation,
/// which is necessary so that multiple collections can borrow the same arena
/// simultaneously.
///
/// # Dropping
///
/// Dropping an `ArenaAllocator` automatically calls [`reset`](Self::reset),
/// returning all memory to the backing allocator.
pub struct ArenaAllocator<A: Allocator> {
    /// Backing allocator used to obtain and release raw memory blocks.
    backing:     A,
    /// Head of the singly-linked list of live allocations (stores `full_ptr`).
    last_alloc:  Cell<Option<NonNull<u8>>>,
    /// Running count of allocations since last reset; useful for diagnostics.
    alloc_count: Cell<usize>,
}

impl<A: Allocator> ArenaAllocator<A> {
    /// Creates a new, empty arena backed by `backing`.
    ///
    /// No memory is allocated from `backing` until the first call to
    /// [`alloc`](Allocator::alloc).
    pub fn new(backing: A) -> Self {
        Self {
            backing,
            last_alloc:  Cell::new(None),
            alloc_count: Cell::new(0),
        }
    }

    /// Returns the number of live allocations currently managed by this arena.
    ///
    /// The counter is reset to zero by [`reset`](Self::reset) or
    /// [`reset_zeroed`](Self::reset_zeroed).
    #[inline]
    pub fn alloc_count(&self) -> usize { self.alloc_count.get() }

    /// Releases all memory blocks managed by this arena back to the backing
    /// allocator.
    ///
    /// After this call, all pointers previously returned by
    /// [`alloc`](Allocator::alloc) are invalid.
    ///
    /// # Safety
    ///
    /// The caller must ensure that **no** pointers previously obtained from
    /// this arena are used after `reset` returns (use-after-free).
    pub fn reset(&self) {
        let mut current = self.last_alloc.get();
        while let Some(full_ptr) = current {
            // SAFETY: `full_ptr` was written by `alloc` and points to a valid
            // `Header`.  We read it before deallocating the block it lives in.
            unsafe {
                let header: Header = ptr::read(full_ptr.as_ptr() as *const Header);
                current = header.next;
                // SAFETY: `full_ptr` + `header.full_layout` match the original
                // backing allocation exactly — this is the contract of `alloc`.
                self.backing.dealloc(full_ptr, header.full_layout);
            }
        }
        self.last_alloc.set(None);
        self.alloc_count.set(0);
    }

    /// Zeroes the user-data portion of every block before releasing them.
    ///
    /// Equivalent to [`reset`](Self::reset) but erases sensitive data first
    /// using SIMD-accelerated zero-fill.
    ///
    /// # Safety
    ///
    /// Same as [`reset`](Self::reset): all previously returned pointers are
    /// invalidated.
    pub fn reset_zeroed(&self) {
        let mut current = self.last_alloc.get();
        while let Some(full_ptr) = current {
            // SAFETY: `full_ptr` points to a valid `Header` whose `user_offset`
            // and `user_size` were computed by `alloc` and remain correct until
            // the block is deallocated here.
            unsafe {
                let header: Header = ptr::read(full_ptr.as_ptr() as *const Header);
                current = header.next;

                if header.user_size > 0 {
                    // SAFETY: `full_ptr + user_offset` is the start of the user
                    // data region which is `user_size` bytes long — both values
                    // come from `Layout::extend` so pointer arithmetic is safe.
                    let user_ptr = full_ptr.as_ptr().add(header.user_offset);
                    simd::fill_bytes(user_ptr, 0, header.user_size);
                }

                // SAFETY: See comment in `reset`.
                self.backing.dealloc(full_ptr, header.full_layout);
            }
        }
        self.last_alloc.set(None);
        self.alloc_count.set(0);
    }
}

impl<A: Allocator> Allocator for ArenaAllocator<A> {
    /// Allocates a new block, prepends a [`Header`], and returns a pointer to
    /// the user-data region.
    ///
    /// Internally calls the backing allocator for a single block large enough
    /// to hold both the `Header` and the user data (with proper alignment
    /// padding in between), then links the new block at the head of the
    /// internal list.
    ///
    /// # Safety
    ///
    /// * `layout.size()` must be greater than zero.
    /// * The returned pointer is valid until the next call to
    ///   [`reset`](ArenaAllocator::reset) or until the arena is dropped.
    unsafe fn alloc(&self, layout: Layout) -> Option<NonNull<u8>> {
        // Compute the combined layout: Header immediately followed by user data
        // with appropriate padding to satisfy `layout.align()`.
        // `Layout::extend` returns `(combined_layout, offset_of_user_data)`.
        let (full_layout, user_offset) = Layout::new::<Header>()
            .extend(layout)
            .ok()?;

        // SAFETY: Delegating to the backing allocator which must obey its own
        // `alloc` contract — returning a valid pointer or None.
        let full_ptr = unsafe { self.backing.alloc(full_layout)? };

        let header = Header {
            next:        self.last_alloc.get(),
            full_layout,
            user_size:   layout.size(),
            user_offset,
        };

        // SAFETY: `full_ptr` is valid for at least `size_of::<Header>()` bytes
        // because `full_layout` was computed with `Header` as its first member.
        unsafe { ptr::write(full_ptr.as_ptr() as *mut Header, header) };

        self.last_alloc.set(Some(full_ptr));
        self.alloc_count.set(self.alloc_count.get() + 1);

        // SAFETY: `user_offset` was produced by `Layout::extend`, which
        // guarantees it is within the bounds of the allocated block.
        NonNull::new(unsafe { full_ptr.as_ptr().add(user_offset) })
    }

    /// No-op.  Arena memory cannot be freed individually.
    ///
    /// To release all memory, call [`reset`](ArenaAllocator::reset) or drop
    /// the arena.
    ///
    /// # Safety
    ///
    /// Even though this method is a no-op, callers must not use `ptr` after
    /// calling `dealloc` — the pointer will become dangling once `reset` is
    /// eventually called.
    unsafe fn dealloc(&self, _: NonNull<u8>, _: Layout) {}
}

impl<A: Allocator> Drop for ArenaAllocator<A> {
    /// Releases all managed blocks by calling [`reset`](ArenaAllocator::reset).
    fn drop(&mut self) { self.reset(); }
}

// ── ArenaExt ─────────────────────────────────────────────────────────────────

/// Extension trait that adds zeroed allocation to arena-like allocators.
pub trait ArenaExt: Allocator {
    /// Allocates a block of memory and fills it with zeroes before returning.
    ///
    /// # Safety
    ///
    /// * `layout.size()` must be greater than zero.
    /// * Same lifetime and invalidation rules as the underlying
    ///   [`alloc`](Allocator::alloc) apply.
    unsafe fn alloc_zeroed(&self, layout: Layout) -> Option<NonNull<u8>>;
}

impl<A: Allocator> ArenaExt for ArenaAllocator<A> {
    /// Allocates a block and zero-fills the user-data region with SIMD.
    ///
    /// Delegates to [`alloc`](Allocator::alloc) and then applies a SIMD
    /// zero-fill, which is faster than a scalar loop for large allocations.
    ///
    /// # Safety
    ///
    /// * `layout.size()` must be greater than zero.
    /// * The returned pointer becomes invalid after [`reset`](ArenaAllocator::reset)
    ///   or when the arena is dropped.
    unsafe fn alloc_zeroed(&self, layout: Layout) -> Option<NonNull<u8>> {
        // SAFETY: Forwarding to `alloc` which upholds its own safety contract.
        let ptr = unsafe { self.alloc(layout)? };
        if layout.size() > 0 {
            // SAFETY: `ptr` is valid for exactly `layout.size()` bytes as
            // guaranteed by the successful `alloc` call above.
            unsafe { simd::fill_bytes(ptr.as_ptr(), 0, layout.size()) };
        }
        Some(ptr)
    }
}