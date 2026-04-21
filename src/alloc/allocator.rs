//! Core allocator abstraction.
//!
//! This module defines the [`Allocator`] trait — the single interface that
//! every allocator in the crate implements and that every collection depends on.
//!
//! The design is deliberately minimal: two methods (`alloc` / `dealloc`) that
//! mirror the semantics of `posix_memalign` / `free` while being safe to
//! compose behind higher-level wrappers.

use core::alloc::Layout;
use core::ptr::NonNull;

/// Low-level memory allocator interface.
///
/// Inspired by the explicit allocation model of Zig, this trait decouples
/// collections from any concrete or global allocator.  Callers retain full
/// control over *which* memory region is used and *when* it is released.
///
/// # Implementing `Allocator`
///
/// An implementation must uphold the following invariants:
///
/// * A successful call to [`alloc`](Allocator::alloc) returns a pointer that is
///   valid for `layout.size()` bytes and aligned to at least `layout.align()`.
/// * The returned memory may contain arbitrary (uninitialized) bytes.
/// * The allocated block remains valid until an explicit call to
///   [`dealloc`](Allocator::dealloc) with the *exact same pointer and layout*.
/// * Calling `dealloc` with a pointer that was not obtained from the same
///   allocator instance, or with a mismatched layout, is undefined behaviour.
pub trait Allocator {
    /// Allocates a block of memory described by `layout`.
    ///
    /// Returns `Some(ptr)` on success, where `ptr` is non-null and correctly
    /// aligned.  Returns `None` when the allocator cannot satisfy the request
    /// (e.g. out of memory, capacity exhausted).
    ///
    /// # Safety
    ///
    /// * `layout.size()` **must** be greater than zero.  Passing a zero-sized
    ///   layout is undefined behaviour; implementations may panic or return a
    ///   dangling pointer, but callers must never rely on that behaviour.
    /// * The caller must eventually call [`dealloc`](Allocator::dealloc) with
    ///   the returned pointer and the *same* `layout`, unless the allocator
    ///   uses bulk reclamation (e.g. arena reset).
    unsafe fn alloc(&self, layout: Layout) -> Option<NonNull<u8>>;

    /// Releases the memory block previously obtained from this allocator.
    ///
    /// For allocators that use bulk reclamation (e.g. [`ArenaAllocator`],
    /// [`BumpAllocator`]) this method is intentionally a no-op; memory is
    /// reclaimed collectively via `reset()` or on drop.
    ///
    /// # Safety
    ///
    /// * `ptr` **must** have been returned by a prior call to
    ///   [`alloc`](Allocator::alloc) on the **same** allocator instance.
    /// * `layout` **must** exactly match the layout passed to that `alloc` call.
    /// * After `dealloc` returns, `ptr` is invalid and must never be
    ///   dereferenced (prevents use-after-free).
    /// * Calling `dealloc` twice with the same pointer is undefined behaviour
    ///   (double-free).
    ///
    /// [`ArenaAllocator`]: crate::alloc::arena::ArenaAllocator
    /// [`BumpAllocator`]: crate::alloc::bump::BumpAllocator
    unsafe fn dealloc(&self, ptr: NonNull<u8>, layout: Layout);
}

/// Blanket implementation of [`Allocator`] for shared references.
///
/// This allows passing `&A` (or `&dyn Allocator`) into collections that are
/// generic over `A: Allocator`, avoiding the need to move ownership of the
/// allocator into every container.
///
/// # Example
///
/// ```rust,ignore
/// let sys = SystemAllocator;
/// let vec: ExVec<u32> = ExVec::new(&sys); // &SystemAllocator implements Allocator
/// ```
impl<A: Allocator + ?Sized> Allocator for &A {
    /// Forwards the allocation request to the underlying allocator.
    ///
    /// # Safety
    ///
    /// Inherits all safety requirements from [`A::alloc`](Allocator::alloc).
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> Option<NonNull<u8>> {
        // SAFETY: The caller is responsible for satisfying the layout preconditions
        // (non-zero size, valid alignment).  We simply forward the call unchanged.
        unsafe { (**self).alloc(layout) }
    }

    /// Forwards the deallocation request to the underlying allocator.
    ///
    /// # Safety
    ///
    /// Inherits all safety requirements from [`A::dealloc`](Allocator::dealloc).
    /// In particular, `ptr` must have been allocated by the same underlying
    /// allocator instance and `layout` must match the original allocation.
    #[inline]
    unsafe fn dealloc(&self, ptr: NonNull<u8>, layout: Layout) {
        // SAFETY: The caller guarantees `ptr` and `layout` are valid for the
        // underlying allocator.  We forward without modification.
        unsafe { (**self).dealloc(ptr, layout) }
    }
}