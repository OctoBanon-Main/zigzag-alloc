use core::{
    alloc::Layout,
    cell::Cell,
    ptr::{self, NonNull},
};

use super::allocator::Allocator;
use crate::simd;

/// Internal metadata stored at the start of every allocated block in the arena.
/// 
/// This header allows the arena to form a linked list of allocations for 
/// batch deallocation.
struct Header {
    /// Pointer to the previous allocation in the chain.
    next:        Option<NonNull<u8>>,
    /// The full layout (Header + User data) used to allocate this block.
    full_layout: Layout,
    /// Size of the user-requested memory block.
    user_size:   usize,
    /// Offset from the start of the block to the user data.
    user_offset: usize,
}

/// A block-based allocator that deallocates all memory at once.
/// 
/// `ArenaAllocator` tracks every individual allocation in a linked list.
/// While `dealloc` is a no-op, calling `reset` or dropping the arena 
/// releases all backing memory.
pub struct ArenaAllocator<A: Allocator> {
    /// The underlying allocator used to fetch raw memory blocks.
    backing:     A,
    /// The head of the linked list of allocations.
    last_alloc:  Cell<Option<NonNull<u8>>>,
    /// Counter for the total number of active allocations.
    alloc_count: Cell<usize>,
}

impl<A: Allocator> ArenaAllocator<A> {
    /// Creates a new arena using the provided backing allocator.
    pub fn new(backing: A) -> Self {
        Self {
            backing,
            last_alloc:  Cell::new(None),
            alloc_count: Cell::new(0),
        }
    }

    /// Returns the number of allocations currently managed by the arena.
    #[inline] pub fn alloc_count(&self) -> usize { self.alloc_count.get() }

    /// Deallocates all memory blocks managed by this arena.
    /// 
    /// # Safety
    /// After calling `reset`, all pointers previously allocated from this 
    /// arena become invalid.
    pub fn reset(&self) {
        let mut current = self.last_alloc.get();
        while let Some(full_ptr) = current {
            // SAFETY: The header was written during `alloc` and is guaranteed 
            // to be valid until the block is deallocated here.
            unsafe {
                let header: Header = ptr::read(full_ptr.as_ptr() as *const Header);
                current = header.next;
                self.backing.dealloc(full_ptr, header.full_layout);
            }
        }
        self.last_alloc.set(None);
        self.alloc_count.set(0);
    }

    /// Zeroes out user memory before deallocating all blocks.
    /// 
    /// Useful for sensitive data that should not remain in memory.
    pub fn reset_zeroed(&self) {
        let mut current = self.last_alloc.get();
        while let Some(full_ptr) = current {
            // SAFETY: The header and user data offsets are verified during allocation.
            unsafe {
                let header: Header = ptr::read(full_ptr.as_ptr() as *const Header);
                current = header.next;

                if header.user_size > 0 {
                    let user_ptr = full_ptr.as_ptr().add(header.user_offset);
                    // Zero memory using SIMD-accelerated fill.
                    simd::fill_bytes(user_ptr, 0, header.user_size);
                }

                self.backing.dealloc(full_ptr, header.full_layout);
            }
        }
        self.last_alloc.set(None);
        self.alloc_count.set(0);
    }
}

impl<A: Allocator> Allocator for ArenaAllocator<A> {
    /// Allocates a new block of memory within the arena.
    /// 
    /// # Safety
    /// Inherits safety requirements from the underlying [`Allocator`].
    unsafe fn alloc(&self, layout: Layout) -> Option<NonNull<u8>> {
        // Compute layout for Header + User Data.
        let (full_layout, user_offset) = Layout::new::<Header>()
            .extend(layout)
            .ok()?;

        // SAFETY: The backing allocator must return a valid pointer or None.
        let full_ptr = unsafe { self.backing.alloc(full_layout)? };

        let header = Header {
            next:        self.last_alloc.get(),
            full_layout,
            user_size:   layout.size(),
            user_offset,
        };

        // SAFETY: The allocated block is guaranteed to be large enough for Header.
        unsafe { ptr::write(full_ptr.as_ptr() as *mut Header, header) };

        self.last_alloc.set(Some(full_ptr));
        self.alloc_count.set(self.alloc_count.get() + 1);

        // SAFETY: offset was calculated via `Layout::extend` to be within bounds.
        NonNull::new(unsafe { full_ptr.as_ptr().add(user_offset) })
    }

    /// No-op: Arena memory cannot be freed individually.
    /// Use [`reset`](Self::reset) to clear the entire arena.
    unsafe fn dealloc(&self, _: NonNull<u8>, _: Layout) {}
}

impl<A: Allocator> Drop for ArenaAllocator<A> {
    /// Automatically clears all allocated blocks when the arena is dropped.
    fn drop(&mut self) { self.reset(); }
}

/// Extension trait for zero-initializing arena memory.
pub trait ArenaExt: Allocator {
    /// Allocates a block of memory and fills it with zeroes.
    unsafe fn alloc_zeroed(&self, layout: Layout) -> Option<NonNull<u8>>;
}

impl<A: Allocator> ArenaExt for ArenaAllocator<A> {
    /// Implementation of zeroed allocation using SIMD acceleration.
    unsafe fn alloc_zeroed(&self, layout: Layout) -> Option<NonNull<u8>> {
        let ptr = unsafe { self.alloc(layout)? };
        if layout.size() > 0 {
            // SAFETY: ptr is valid for at least layout.size().
            unsafe { simd::fill_bytes(ptr.as_ptr(), 0, layout.size()) };
        }
        Some(ptr)
    }
}