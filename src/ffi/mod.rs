use core::{alloc::Layout, ptr::NonNull};

use crate::alloc::{allocator::Allocator, system::SystemAllocator};

pub mod allocators;
pub mod collections;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct RawAllocHandle {
    pub(crate) data: *const (),
    pub(crate) vtable: *const (),
}

unsafe impl Sync for RawAllocHandle {}
unsafe impl Send for RawAllocHandle {}

impl RawAllocHandle {
    #[inline]
    pub(crate) fn from_ref(r: &dyn Allocator) -> Self {
        unsafe { core::mem::transmute(r) }
    }

    #[inline]
    pub(crate) unsafe fn as_ref<'a>(self) -> &'a dyn Allocator {
        unsafe { core::mem::transmute(self) }
    }
}

#[inline]
pub(crate) fn sys_box_new<T>(val: T) -> *mut T {
    let layout = Layout::new::<T>();
    let ptr: *mut T = if layout.size() == 0 {
        NonNull::dangling().as_ptr()
    } else {
        unsafe {
            SystemAllocator
                .alloc(layout)
                .expect("zigzag FFI: SystemAllocator OOM")
                .cast::<T>()
                .as_ptr()
        }
    };

    unsafe { ptr.write(val) };
    ptr
}

#[inline]
pub(crate) unsafe fn sys_box_drop<T>(ptr: *mut T) {
    if ptr.is_null() {
        return;
    }

    unsafe {
        core::ptr::drop_in_place(ptr);
        let layout = Layout::new::<T>();
        if layout.size() > 0 {
            SystemAllocator.dealloc(NonNull::new_unchecked(ptr as *mut u8), layout);
        }
    }
}