use core::{
    alloc::Layout,
    mem,
    ptr::{self, NonNull},
    sync::atomic::{AtomicPtr, Ordering}
};

use super::allocator::Allocator;

struct FreeNode {
    next: *mut FreeNode,
}

pub struct PoolAllocator<A: Allocator> {
    backing: A,
    block_layout: Layout,
    slab_layout: Layout,
    slab: NonNull<u8>,
    free_head: AtomicPtr<FreeNode>,
    capacity: usize,
}

unsafe impl<A: Allocator + Sync> Sync for PoolAllocator<A> {}
unsafe impl<A: Allocator + Send> Send for PoolAllocator<A> {}

impl<A: Allocator> PoolAllocator<A> {
    pub fn new(backing: A, item_layout: Layout, capacity: usize) -> Option<Self> {
        let block_size = item_layout.size().max(mem::size_of::<FreeNode>());
        let block_align = item_layout.align().max(mem::align_of::<FreeNode>());

        let block_layout = Layout::from_size_align(block_size, block_align)
            .ok()?
            .pad_to_align();

        let total_size = block_layout.size().checked_mul(capacity)?;
        let slab_layout = Layout::from_size_align(total_size, block_layout.align()).ok()?;

        let slab = unsafe { backing.alloc(slab_layout)? };

        let mut head: *mut FreeNode = ptr::null_mut();
        for i in (0..capacity).rev() {
            let block = unsafe {
                slab.as_ptr().add(i * block_layout.size()) as *mut FreeNode
            };

            unsafe { ptr::write(block, FreeNode { next: head }) };
            head = block;
        }

        Some(Self {
            backing,
            block_layout,
            slab_layout,
            slab,
            free_head: AtomicPtr::new(head),
            capacity
        })
    }

    pub fn typed<T>(backing: A, capacity: usize) -> Option<Self> {
        Self::new(backing, Layout::new::<T>(), capacity)
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

        pub fn free_count(&self) -> usize {
        let mut n    = 0usize;
        let mut node = self.free_head.load(Ordering::Relaxed);
        while !node.is_null() {
            n += 1;
            node = unsafe { (*node).next };
        }
        n
    }

    #[inline]
    pub fn block_layout(&self) -> Layout {
        self.block_layout
    }
}

impl<A: Allocator> Allocator for PoolAllocator<A> {
    unsafe fn alloc(&self, layout: Layout) -> Option<NonNull<u8>> {
        if layout.size() > self.block_layout.size()
        || layout.align() > self.block_layout.align()
        {
            return None;
        }

        let mut head = self.free_head.load(Ordering::Acquire);
        loop {
            let node = NonNull::new(head)?;

            let next = unsafe { (*head).next };

            match self.free_head.compare_exchange_weak(
                head,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Some(node.cast()),
                Err(actual) => head = actual,
            }
        }
    }

    unsafe fn dealloc(&self, ptr: NonNull<u8>, _layout: Layout) {
        let node = ptr.as_ptr() as *mut FreeNode;

        let mut head = self.free_head.load(Ordering::Relaxed);
        loop {
            unsafe { ptr::write(node, FreeNode { next: head }) };

            match self.free_head.compare_exchange_weak(
                head,
                node,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => return,
                Err(actual) => head = actual,
            }
        }
    }
}

impl<A: Allocator> Drop for PoolAllocator<A> {
    fn drop(&mut self) {
        unsafe { self.backing.dealloc(self.slab, self.slab_layout) };
    }
}