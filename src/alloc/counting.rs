use core::{
    alloc::Layout,
    cell::Cell,
    ptr::NonNull,
};

use super::allocator::Allocator;

pub struct CountingAllocator<A: Allocator> {
    pub(crate) inner: A,
    alloc_count:      Cell<usize>,
    dealloc_count:    Cell<usize>,
    bytes_allocated:  Cell<usize>,
    bytes_freed:      Cell<usize>,
}

impl<A: Allocator> CountingAllocator<A> {
    pub fn new(inner: A) -> Self {
        Self {
            inner,
            alloc_count:     Cell::new(0),
            dealloc_count:   Cell::new(0),
            bytes_allocated: Cell::new(0),
            bytes_freed:     Cell::new(0),
        }
    }

    pub fn stats(&self) -> AllocStats {
        AllocStats {
            allocs:           self.alloc_count.get(),
            deallocs:         self.dealloc_count.get(),
            bytes_allocated:  self.bytes_allocated.get(),
            bytes_freed:      self.bytes_freed.get(),
            bytes_live:       self.bytes_allocated.get()
                                  .saturating_sub(self.bytes_freed.get()),
        }
    }

    pub fn reset_stats(&self) {
        self.alloc_count.set(0);
        self.dealloc_count.set(0);
        self.bytes_allocated.set(0);
        self.bytes_freed.set(0);
    }
}

impl<A: Allocator> Allocator for CountingAllocator<A> {
    unsafe fn alloc(&self, layout: Layout) -> Option<NonNull<u8>> {
        let ptr = unsafe { self.inner.alloc(layout)? };
        self.alloc_count.set(self.alloc_count.get() + 1);
        self.bytes_allocated.set(self.bytes_allocated.get() + layout.size());
        Some(ptr)
    }

    unsafe fn dealloc(&self, ptr: NonNull<u8>, layout: Layout) {
        self.dealloc_count.set(self.dealloc_count.get() + 1);
        self.bytes_freed.set(self.bytes_freed.get() + layout.size());
        unsafe { self.inner.dealloc(ptr, layout) };
    }
}

pub struct AllocStats {
    pub allocs:          usize,
    pub deallocs:        usize,
    pub bytes_allocated: usize,
    pub bytes_freed:     usize,
    pub bytes_live:      usize,
}