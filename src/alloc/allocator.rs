use core::alloc::Layout;
use core::ptr::NonNull;

/// The `Allocator` trait defines the interface for low-level memory management.
/// 
/// Inspired by the explicit memory management philosophy of Zig, this trait
/// allows containers to be decoupled from a global allocator, enabling fine-grained
/// control over memory placement and lifetime.
pub trait Allocator {
    /// Attempts to allocate a block of memory according to the specified [`Layout`].
    /// 
    /// # Errors
    /// Returns `None` if the memory cannot be allocated.
    ///
    /// # Safety
    /// * The `layout` must have a non-zero size.
    /// * Implementations must return a pointer that is aligned to at least `layout.align()`.
    /// * The allocated block must remain valid until it is explicitly deallocated.
    unsafe fn alloc(&self, layout: Layout) -> Option<NonNull<u8>>;

    /// Deallocates the memory block referenced by `ptr`.
    ///
    /// # Safety
    /// * `ptr` must have been previously allocated via the same allocator instance.
    /// * `layout` must exactly match the layout used during the allocation of `ptr`.
    /// * After calling `dealloc`, the memory block must not be accessed again (preventing use-after-free).
    unsafe fn dealloc(&self, ptr: NonNull<u8>, layout: Layout);
}

/// Blanket implementation of `Allocator` for references to types that implement `Allocator`.
/// 
/// This allows passing `&Allocator` into collections that expect an `Allocator` 
/// without needing to move ownership, facilitating the use of long-lived allocators.
impl<A: Allocator + ?Sized> Allocator for &A {
    /// Forwards the allocation request to the underlying allocator.
    /// 
    /// # Safety
    /// Inherits safety requirements from the underlying [`Allocator::alloc`] implementation.
    unsafe fn alloc(&self, layout: Layout) -> Option<NonNull<u8>> {
        // SAFETY: The safety invariants are maintained by the caller of the trait method.
        unsafe { (**self).alloc(layout) }
    }

    /// Forwards the deallocation request to the underlying allocator.
    /// 
    /// # Safety
    /// Inherits safety requirements from the underlying [`Allocator::dealloc`] implementation.
    unsafe fn dealloc(&self, ptr: NonNull<u8>, layout: Layout) {
        // SAFETY: The safety invariants are maintained by the caller of the trait method.
        unsafe { (**self).dealloc(ptr, layout) }
    }
}