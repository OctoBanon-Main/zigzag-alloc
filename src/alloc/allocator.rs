use core::alloc::Layout;
use core::ptr::NonNull;

pub trait Allocator {
    unsafe fn alloc(&self, layout: Layout) -> Option<NonNull<u8>>;
    unsafe fn dealloc(&self, ptr: NonNull<u8>, layout: Layout);
}

impl<A: Allocator + ?Sized> Allocator for &A {
    unsafe fn alloc(&self, layout: Layout) -> Option<NonNull<u8>> {
        unsafe { (**self).alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: NonNull<u8>, layout: Layout) {
        unsafe { (**self).dealloc(ptr, layout) }
    }
}