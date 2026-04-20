use core::{alloc::Layout, ffi::c_void, mem, ptr::NonNull};

use super::allocator::Allocator;

#[cfg(target_family = "unix")]
unsafe extern "C" {
    fn posix_memalign(memptr: *mut *mut c_void, alignment: usize, size: usize) -> i32;
 
    fn free(ptr: *mut c_void);
}

#[cfg(target_family = "windows")]
unsafe extern "C" {
    fn _aligned_malloc(size: usize, alignment: usize) -> *mut c_void;
 
    fn _aligned_free(ptr: *mut c_void);
}

pub struct SystemAllocator;

impl Allocator for SystemAllocator {
    unsafe fn alloc(&self, layout: Layout) -> Option<NonNull<u8>> {
        let size = layout.size();

        if size == 0 {
            return Some(NonNull::dangling());
        }

        let align = layout.align().max(mem::size_of::<*const ()>());

        #[cfg(target_family = "unix")]
        {
            let mut ptr: *mut c_void = core::ptr::null_mut();
            
            let min_align = core::mem::size_of::<*mut c_void>();
            let align = layout.align().max(min_align); 
            
            let size = layout.size();

            let rc = unsafe { libc::posix_memalign(&mut ptr, align, size) };
            
            if rc != 0 {
                return None;
            }
            NonNull::new(ptr as *mut u8)
        }

        #[cfg(target_family = "windows")]
        {
            let ptr = unsafe { _aligned_malloc(size, align) as *mut u8 };
            NonNull::new(ptr)
        }

        #[cfg(not(any(target_family = "unix", target_family = "windows")))]
        {
            let _ = align;
            None
        }
    }

    unsafe fn dealloc(&self, ptr: NonNull<u8>, layout: Layout) {
        if layout.size() == 0 {
            return;
        }

        #[cfg(target_family = "unix")]
        {
            free(ptr.as_ptr() as *mut c_void)
        }

        #[cfg(target_family = "windows")]
        {
            unsafe { _aligned_free(ptr.as_ptr() as *mut c_void) };
        }
    }
}