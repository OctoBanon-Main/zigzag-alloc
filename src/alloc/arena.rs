use core::{
    alloc::Layout,
    cell::Cell,
    ptr::{self, NonNull},
};

use super::allocator::Allocator;
use crate::simd;

struct Header {
    next:        Option<NonNull<u8>>,
    full_layout: Layout,
    user_size:   usize,
    user_offset: usize,
}

pub struct ArenaAllocator<A: Allocator> {
    backing:     A,
    last_alloc:  Cell<Option<NonNull<u8>>>,
    alloc_count: Cell<usize>,
}

impl<A: Allocator> ArenaAllocator<A> {
    pub fn new(backing: A) -> Self {
        Self {
            backing,
            last_alloc:  Cell::new(None),
            alloc_count: Cell::new(0),
        }
    }

    #[inline] pub fn alloc_count(&self) -> usize { self.alloc_count.get() }

    pub fn reset(&self) {
        let mut current = self.last_alloc.get();
        while let Some(full_ptr) = current {
            unsafe {
                let header: Header = ptr::read(full_ptr.as_ptr() as *const Header);
                current = header.next;
                self.backing.dealloc(full_ptr, header.full_layout);
            }
        }
        self.last_alloc.set(None);
        self.alloc_count.set(0);
    }

    pub fn reset_zeroed(&self) {
        let mut current = self.last_alloc.get();
        while let Some(full_ptr) = current {
            unsafe {
                let header: Header = ptr::read(full_ptr.as_ptr() as *const Header);
                current = header.next;

                if header.user_size > 0 {
                    let user_ptr = full_ptr.as_ptr().add(header.user_offset);
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
    unsafe fn alloc(&self, layout: Layout) -> Option<NonNull<u8>> {
        let (full_layout, user_offset) = Layout::new::<Header>()
            .extend(layout)
            .ok()?;

        let full_ptr = unsafe { self.backing.alloc(full_layout)? };

        let header = Header {
            next:        self.last_alloc.get(),
            full_layout,
            user_size:   layout.size(),
            user_offset,
        };
        unsafe { ptr::write(full_ptr.as_ptr() as *mut Header, header) };

        self.last_alloc.set(Some(full_ptr));
        self.alloc_count.set(self.alloc_count.get() + 1);

        NonNull::new(unsafe { full_ptr.as_ptr().add(user_offset) })
    }

    unsafe fn dealloc(&self, _: NonNull<u8>, _: Layout) {}
}

impl<A: Allocator> Drop for ArenaAllocator<A> {
    fn drop(&mut self) { self.reset(); }
}

pub trait ArenaExt: Allocator {
    unsafe fn alloc_zeroed(&self, layout: Layout) -> Option<NonNull<u8>>;
}

impl<A: Allocator> ArenaExt for ArenaAllocator<A> {
    unsafe fn alloc_zeroed(&self, layout: Layout) -> Option<NonNull<u8>> {
        let ptr = unsafe { self.alloc(layout)? };
        if layout.size() > 0 {
            unsafe { simd::fill_bytes(ptr.as_ptr(), 0, layout.size()) };
        }
        Some(ptr)
    }
}