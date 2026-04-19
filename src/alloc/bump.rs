use core::{
    alloc::Layout,
    ptr::NonNull,
    sync::atomic::{AtomicUsize, Ordering},
};

use super::allocator::Allocator;

pub struct BumpAllocator {
    start: *mut u8,
    size: usize,
    offset: AtomicUsize,
}

unsafe impl Sync for BumpAllocator {}
unsafe impl Send for BumpAllocator {}

impl BumpAllocator {
    pub fn new(buf: &'static mut [u8]) -> Self {
        Self {
            start: buf.as_mut_ptr(),
            size: buf.len(),
            offset: AtomicUsize::new(0),
        }
    }

    #[inline]
    pub fn used(&self) -> usize {
        self.offset.load(Ordering::Relaxed)
    }

    #[inline]
    pub fn remaining(&self) -> usize {
        self.size.saturating_sub(self.used())
    }

    pub unsafe fn reset(&self) {
        self.offset.store(0, Ordering::Release);
    }
}

impl Allocator for BumpAllocator {
    unsafe fn alloc(&self, layout: Layout) -> Option<NonNull<u8>> {
        let size = layout.size();
        let align = layout.align();

        let mut current = self.offset.load(Ordering::Relaxed);
        loop {
            let aligned = current.checked_add(align - 1)? & !(align - 1);
            let end = aligned.checked_add(size)?;

            if end > self.size {
                return None;
            }

            match self.offset.compare_exchange_weak(
                current,
                end,
                Ordering::AcqRel,
                Ordering::Relaxed
            ) {
                Ok(_) => {
                    return NonNull::new(unsafe { self.start.add(aligned) });
                }
                Err(actual) => current = actual,
            }
        }
    }

    unsafe fn dealloc(&self, _: NonNull<u8>, _: Layout) {}
}