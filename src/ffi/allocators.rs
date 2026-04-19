use core::{alloc::Layout, ffi::c_void, ptr::NonNull};

use crate::alloc::{
    allocator::Allocator,
    arena::ArenaAllocator,
    bump::BumpAllocator,
    counting::CountingAllocator,
    pool::PoolAllocator,
    system::SystemAllocator
};

use super::{sys_box_drop, sys_box_new, RawAllocHandle};

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_alloc(
    alloc: RawAllocHandle,
    size: usize,
    align: usize,
) -> *mut c_void {
    let layout = match Layout::from_size_align(size, align) {
        Ok(l) => l,
        Err(_) => return core::ptr::null_mut(),
    };
    match unsafe { alloc.as_ref().alloc(layout) } {
        Some(p) => p.as_ptr() as *mut c_void,
        None => core::ptr::null_mut(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_dealloc(
    alloc: RawAllocHandle,
    ptr: *mut c_void,
    size: usize,
    align: usize,
) {
    if ptr.is_null() {
        return;
    }
    if let Ok(layout) = Layout::from_size_align(size, align) {
        unsafe {
            alloc
                .as_ref()
                .dealloc(NonNull::new_unchecked(ptr as *mut u8), layout);
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn zigzag_system_create() -> *mut SystemAllocator {
    sys_box_new(SystemAllocator)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_system_destroy(ptr: *mut SystemAllocator) {
    unsafe { sys_box_drop(ptr) };
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_system_as_alloc(ptr: *mut SystemAllocator) -> RawAllocHandle {
    RawAllocHandle::from_ref(unsafe { &*ptr })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_bump_create(buf: *mut u8, len: usize) -> *mut BumpAllocator {
    let slice: &'static mut [u8] = unsafe { core::slice::from_raw_parts_mut(buf, len) };
    sys_box_new(BumpAllocator::new(slice))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_bump_destroy(ptr: *mut BumpAllocator) {
    unsafe { sys_box_drop(ptr) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_bump_as_alloc(ptr: *mut BumpAllocator) -> RawAllocHandle {
    RawAllocHandle::from_ref(unsafe { &*ptr })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_bump_alloc(
    ptr: *mut BumpAllocator,
    size: usize,
    align: usize,
) -> *mut c_void {
    let layout = match Layout::from_size_align(size, align) {
        Ok(l) => l,
        Err(_) => return core::ptr::null_mut(),
    };
    match unsafe { (*ptr).alloc(layout) } {
        Some(p) => p.as_ptr() as *mut c_void,
        None => core::ptr::null_mut(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_bump_reset(ptr: *mut BumpAllocator) {
    unsafe { (*ptr).reset() }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_bump_used(ptr: *const BumpAllocator) -> usize {
    unsafe { (*ptr).used() }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_bump_remaining(ptr: *const BumpAllocator) -> usize {
    unsafe { (*ptr).remaining() }
}

#[unsafe(no_mangle)]
pub extern "C" fn zigzag_arena_create() -> *mut ArenaAllocator<SystemAllocator> {
    sys_box_new(ArenaAllocator::new(SystemAllocator))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_arena_destroy(ptr: *mut ArenaAllocator<SystemAllocator>) {
    unsafe { sys_box_drop(ptr) };
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_arena_as_alloc(ptr: *mut ArenaAllocator<SystemAllocator>) -> RawAllocHandle {
    RawAllocHandle::from_ref(unsafe { &*ptr })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_arena_alloc(
    ptr: *mut ArenaAllocator<SystemAllocator>,
    size: usize,
    align: usize
) -> *mut c_void {
    let layout = match Layout::from_size_align(size, align) {
        Ok(l) => l,
        Err(_) => return core::ptr::null_mut(),
    };

    match unsafe { (*ptr).alloc(layout) } {
        Some(p) => p.as_ptr() as *mut c_void,
        None => core::ptr::null_mut(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_arena_reset(ptr: *mut ArenaAllocator<SystemAllocator>) {
    unsafe { (*ptr).reset() };
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_arena_alloc_count(ptr: *const ArenaAllocator<SystemAllocator>) -> usize {
    unsafe { (*ptr).alloc_count() }
}

#[unsafe(no_mangle)]
pub extern "C" fn zigzag_pool_create(
    block_size: usize,
    block_align: usize,
    capacity: usize,
) -> *mut PoolAllocator<SystemAllocator> {
    let layout = match Layout::from_size_align(block_size, block_align) {
        Ok(l) => l,
        Err(_) => return core::ptr::null_mut(),
    };
    match PoolAllocator::new(SystemAllocator, layout, capacity) {
        Some(pool) => sys_box_new(pool),
        None => core::ptr::null_mut(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_pool_destroy(ptr: *mut PoolAllocator<SystemAllocator>) {
    unsafe { sys_box_drop(ptr) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_pool_as_alloc(
    ptr: *mut PoolAllocator<SystemAllocator>,
) -> RawAllocHandle {
    RawAllocHandle::from_ref(unsafe { &*ptr })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_pool_alloc(
    ptr: *mut PoolAllocator<SystemAllocator>,
) -> *mut c_void {
    let layout = unsafe { (*ptr).block_layout() };
    match unsafe { (*ptr).alloc(layout)} {
        Some(p) => p.as_ptr() as *mut c_void,
        None => core::ptr::null_mut(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_pool_dealloc(
    ptr: *mut PoolAllocator<SystemAllocator>,
    mem: *mut c_void,
) {
    if mem.is_null() {
        return;
    }
    let layout = unsafe { (*ptr).block_layout() };
    unsafe {
        (*ptr).dealloc(NonNull::new_unchecked(mem as *mut u8), layout);
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_pool_capacity(
    ptr: *const PoolAllocator<SystemAllocator>,
) -> usize {
    unsafe { (*ptr).capacity() }
}
 
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_pool_free_count(
    ptr: *const PoolAllocator<SystemAllocator>,
) -> usize {
    unsafe { (*ptr).free_count() }
}

#[repr(C)]
pub struct ZigzagAllocStats {
    pub allocs:   usize,
    pub deallocs: usize,
    pub bytes:    usize,
}
 
#[unsafe(no_mangle)]
pub extern "C" fn zigzag_counting_create() -> *mut CountingAllocator<SystemAllocator> {
    sys_box_new(CountingAllocator::new(SystemAllocator))
}
 
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_counting_destroy(
    ptr: *mut CountingAllocator<SystemAllocator>,
) {
    unsafe { sys_box_drop(ptr) }
}
 
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_counting_as_alloc(
    ptr: *mut CountingAllocator<SystemAllocator>,
) -> RawAllocHandle {
    RawAllocHandle::from_ref(unsafe { &*ptr })
}
 
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_counting_stats(
    ptr: *const CountingAllocator<SystemAllocator>,
) -> ZigzagAllocStats {
    let s = unsafe { (*ptr).stats() };
    ZigzagAllocStats {
        allocs:   s.allocs,
        deallocs: s.deallocs,
        bytes:    s.bytes,
    }
}
 
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_counting_alloc(
    ptr:   *mut CountingAllocator<SystemAllocator>,
    size:  usize,
    align: usize,
) -> *mut c_void {
    let layout = match Layout::from_size_align(size, align) {
        Ok(l)  => l,
        Err(_) => return core::ptr::null_mut(),
    };
    match unsafe { (*ptr).alloc(layout) } {
        Some(p) => p.as_ptr() as *mut c_void,
        None    => core::ptr::null_mut(),
    }
}
 
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_counting_dealloc(
    ptr:   *mut CountingAllocator<SystemAllocator>,
    mem:   *mut c_void,
    size:  usize,
    align: usize,
) {
    if mem.is_null() {
        return;
    }
    if let Ok(layout) = Layout::from_size_align(size, align) {
        unsafe {
            (*ptr).dealloc(NonNull::new_unchecked(mem as *mut u8), layout);
        }
    }
}
