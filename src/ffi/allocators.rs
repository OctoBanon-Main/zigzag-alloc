use core::{alloc::Layout, ffi::c_void, ptr::NonNull};

use crate::alloc::{
    allocator::Allocator, arena::ArenaAllocator, bump::BumpAllocator, counting::CountingAllocator,
    pool::PoolAllocator, system::SystemAllocator,
};

use super::{RawAllocHandle, sys_box_drop, sys_box_new};

/// Allocates memory using the provided allocator handle.
///
/// # Safety
/// - `alloc` must contain valid pointers for a live `Allocator`.
/// - `size` and `align` must represent a valid Rust `Layout`.
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
    // SAFETY: The caller guarantees `alloc` handle is valid.
    match unsafe { alloc.as_ref().alloc(layout) } {
        Some(p) => p.as_ptr() as *mut c_void,
        None => core::ptr::null_mut(),
    }
}

/// Deallocates memory using the provided allocator handle.
///
/// # Safety
/// - `alloc` must contain valid pointers for a live `Allocator`.
/// - `ptr` must point to a memory block currently allocated by this allocator.
/// - `size` and `align` must exactly match the parameters used to allocate the block.
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
        // SAFETY: Caller guarantees `ptr` was allocated via `alloc` with `layout`.
        unsafe {
            alloc
                .as_ref()
                .dealloc(NonNull::new_unchecked(ptr as *mut u8), layout);
        }
    }
}

/// Creates a new `SystemAllocator` and returns a raw pointer to it.
#[unsafe(no_mangle)]
pub extern "C" fn zigzag_system_create() -> *mut SystemAllocator {
    sys_box_new(SystemAllocator)
}

/// Destroys a `SystemAllocator` previously created by `zigzag_system_create`.
///
/// # Safety
/// - `ptr` must be a valid pointer returned by `zigzag_system_create`.
/// - Double-freeing is undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_system_destroy(ptr: *mut SystemAllocator) {
    // SAFETY: The caller guarantees the validity of `ptr`.
    unsafe { sys_box_drop(ptr) };
}

/// Converts a `SystemAllocator` pointer into a generic `RawAllocHandle`.
///
/// # Safety
/// - `ptr` must be a valid, live pointer to a `SystemAllocator`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_system_as_alloc(ptr: *mut SystemAllocator) -> RawAllocHandle {
    // SAFETY: The caller ensures `ptr` is valid and dereferenceable.
    RawAllocHandle::from_ref(unsafe { &*ptr })
}

/// Creates a new `BumpAllocator` using the provided memory buffer.
///
/// # Safety
/// - `buf` must point to a valid, mutable memory region of at least `len` bytes.
/// - The memory region must live for the entire lifetime of the `BumpAllocator`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_bump_create(buf: *mut u8, len: usize) -> *mut BumpAllocator {
    // SAFETY: The caller guarantees `buf` points to a valid sequence of `len` bytes.
    let slice: &'static mut [u8] = unsafe { core::slice::from_raw_parts_mut(buf, len) };
    sys_box_new(BumpAllocator::new(slice))
}

/// Destroys a `BumpAllocator`.
///
/// # Safety
/// - `ptr` must be a valid pointer returned by `zigzag_bump_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_bump_destroy(ptr: *mut BumpAllocator) {
    // SAFETY: The caller guarantees the validity of `ptr`.
    unsafe { sys_box_drop(ptr) }
}

/// Converts a `BumpAllocator` pointer into a generic `RawAllocHandle`.
///
/// # Safety
/// - `ptr` must be a valid, live pointer to a `BumpAllocator`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_bump_as_alloc(ptr: *mut BumpAllocator) -> RawAllocHandle {
    // SAFETY: The caller ensures `ptr` is valid and dereferenceable.
    RawAllocHandle::from_ref(unsafe { &*ptr })
}

/// Allocates memory directly from a `BumpAllocator`.
///
/// # Safety
/// - `ptr` must be a valid pointer to a `BumpAllocator`.
/// - `size` and `align` must form a valid `Layout`.
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
    // SAFETY: The caller guarantees `ptr` points to a valid `BumpAllocator`.
    match unsafe { (*ptr).alloc(layout) } {
        Some(p) => p.as_ptr() as *mut c_void,
        None => core::ptr::null_mut(),
    }
}

/// Resets the `BumpAllocator`, invalidating all previous allocations.
///
/// # Safety
/// - `ptr` must be a valid pointer to a `BumpAllocator`.
/// - All pointers previously returned by this allocator become invalid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_bump_reset(ptr: *mut BumpAllocator) {
    // SAFETY: The caller ensures `ptr` is valid.
    unsafe { (*ptr).reset() }
}

/// Returns the number of bytes currently used in the `BumpAllocator`.
///
/// # Safety
/// - `ptr` must be a valid pointer to a `BumpAllocator`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_bump_used(ptr: *const BumpAllocator) -> usize {
    // SAFETY: The caller ensures `ptr` is valid.
    unsafe { (*ptr).used() }
}

/// Returns the number of bytes remaining in the `BumpAllocator`.
///
/// # Safety
/// - `ptr` must be a valid pointer to a `BumpAllocator`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_bump_remaining(ptr: *const BumpAllocator) -> usize {
    // SAFETY: The caller ensures `ptr` is valid.
    unsafe { (*ptr).remaining() }
}

/// Creates a new `ArenaAllocator` backed by a `SystemAllocator`.
#[unsafe(no_mangle)]
pub extern "C" fn zigzag_arena_create() -> *mut ArenaAllocator<SystemAllocator> {
    sys_box_new(ArenaAllocator::new(SystemAllocator))
}

/// Destroys an `ArenaAllocator`.
///
/// # Safety
/// - `ptr` must be a valid pointer returned by `zigzag_arena_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_arena_destroy(ptr: *mut ArenaAllocator<SystemAllocator>) {
    // SAFETY: The caller guarantees the validity of `ptr`.
    unsafe { sys_box_drop(ptr) };
}

/// Converts an `ArenaAllocator` pointer into a generic `RawAllocHandle`.
///
/// # Safety
/// - `ptr` must be a valid, live pointer to an `ArenaAllocator`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_arena_as_alloc(
    ptr: *mut ArenaAllocator<SystemAllocator>,
) -> RawAllocHandle {
    // SAFETY: The caller ensures `ptr` is valid and dereferenceable.
    RawAllocHandle::from_ref(unsafe { &*ptr })
}

/// Allocates memory directly from an `ArenaAllocator`.
///
/// # Safety
/// - `ptr` must be a valid pointer to an `ArenaAllocator`.
/// - `size` and `align` must form a valid `Layout`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_arena_alloc(
    ptr: *mut ArenaAllocator<SystemAllocator>,
    size: usize,
    align: usize,
) -> *mut c_void {
    let layout = match Layout::from_size_align(size, align) {
        Ok(l) => l,
        Err(_) => return core::ptr::null_mut(),
    };

    // SAFETY: The caller guarantees `ptr` points to a valid `ArenaAllocator`.
    match unsafe { (*ptr).alloc(layout) } {
        Some(p) => p.as_ptr() as *mut c_void,
        None => core::ptr::null_mut(),
    }
}

/// Resets the `ArenaAllocator`, freeing all stored blocks.
///
/// # Safety
/// - `ptr` must be a valid pointer to an `ArenaAllocator`.
/// - All pointers previously returned by this arena become invalid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_arena_reset(ptr: *mut ArenaAllocator<SystemAllocator>) {
    // SAFETY: The caller ensures `ptr` is valid.
    unsafe { (*ptr).reset() };
}

/// Returns the total number of allocations made in the arena.
///
/// # Safety
/// - `ptr` must be a valid pointer to an `ArenaAllocator`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_arena_alloc_count(
    ptr: *const ArenaAllocator<SystemAllocator>,
) -> usize {
    // SAFETY: The caller ensures `ptr` is valid.
    unsafe { (*ptr).alloc_count() }
}

/// Creates a new `PoolAllocator` with a specific block size, alignment, and capacity.
/// Backed by a `SystemAllocator`.
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

/// Destroys a `PoolAllocator`.
///
/// # Safety
/// - `ptr` must be a valid pointer returned by `zigzag_pool_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_pool_destroy(ptr: *mut PoolAllocator<SystemAllocator>) {
    // SAFETY: The caller guarantees the validity of `ptr`.
    unsafe { sys_box_drop(ptr) }
}

/// Converts a `PoolAllocator` pointer into a generic `RawAllocHandle`.
///
/// # Safety
/// - `ptr` must be a valid, live pointer to a `PoolAllocator`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_pool_as_alloc(
    ptr: *mut PoolAllocator<SystemAllocator>,
) -> RawAllocHandle {
    // SAFETY: The caller ensures `ptr` is valid and dereferenceable.
    RawAllocHandle::from_ref(unsafe { &*ptr })
}

/// Allocates a single block directly from the `PoolAllocator`.
///
/// # Safety
/// - `ptr` must be a valid pointer to a `PoolAllocator`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_pool_alloc(
    ptr: *mut PoolAllocator<SystemAllocator>,
) -> *mut c_void {
    // SAFETY: The caller guarantees `ptr` points to a valid `PoolAllocator`.
    let layout = unsafe { (*ptr).block_layout() };
    match unsafe { (*ptr).alloc(layout) } {
        Some(p) => p.as_ptr() as *mut c_void,
        None => core::ptr::null_mut(),
    }
}

/// Deallocates a single block back into the `PoolAllocator`.
///
/// # Safety
/// - `ptr` must be a valid pointer to a `PoolAllocator`.
/// - `mem` must be a pointer previously obtained from this pool.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_pool_dealloc(
    ptr: *mut PoolAllocator<SystemAllocator>,
    mem: *mut c_void,
) {
    if mem.is_null() {
        return;
    }
    // SAFETY: The caller guarantees `ptr` points to a valid `PoolAllocator`.
    let layout = unsafe { (*ptr).block_layout() };
    // SAFETY: The caller guarantees `mem` is an active allocation from this pool.
    unsafe {
        (*ptr).dealloc(NonNull::new_unchecked(mem as *mut u8), layout);
    }
}

/// Returns the total block capacity of the `PoolAllocator`.
///
/// # Safety
/// - `ptr` must be a valid pointer to a `PoolAllocator`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_pool_capacity(ptr: *const PoolAllocator<SystemAllocator>) -> usize {
    // SAFETY: The caller ensures `ptr` is valid.
    unsafe { (*ptr).capacity() }
}

/// Returns the number of currently free blocks in the `PoolAllocator`.
///
/// # Safety
/// - `ptr` must be a valid pointer to a `PoolAllocator`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_pool_free_count(
    ptr: *const PoolAllocator<SystemAllocator>,
) -> usize {
    // SAFETY: The caller ensures `ptr` is valid.
    unsafe { (*ptr).free_count() }
}

/// Statistics returned by the `CountingAllocator`.
#[repr(C)]
pub struct ZigzagAllocStats {
    pub allocs: usize,
    pub deallocs: usize,
    pub bytes: usize,
}

/// Creates a new `CountingAllocator` backed by a `SystemAllocator`.
#[unsafe(no_mangle)]
pub extern "C" fn zigzag_counting_create() -> *mut CountingAllocator<SystemAllocator> {
    sys_box_new(CountingAllocator::new(SystemAllocator))
}

/// Destroys a `CountingAllocator`.
///
/// # Safety
/// - `ptr` must be a valid pointer returned by `zigzag_counting_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_counting_destroy(ptr: *mut CountingAllocator<SystemAllocator>) {
    // SAFETY: The caller guarantees the validity of `ptr`.
    unsafe { sys_box_drop(ptr) }
}

/// Converts a `CountingAllocator` pointer into a generic `RawAllocHandle`.
///
/// # Safety
/// - `ptr` must be a valid, live pointer to a `CountingAllocator`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_counting_as_alloc(
    ptr: *mut CountingAllocator<SystemAllocator>,
) -> RawAllocHandle {
    // SAFETY: The caller ensures `ptr` is valid and dereferenceable.
    RawAllocHandle::from_ref(unsafe { &*ptr })
}

/// Retrieves the current allocation statistics from the `CountingAllocator`.
///
/// # Safety
/// - `ptr` must be a valid pointer to a `CountingAllocator`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_counting_stats(
    ptr: *const CountingAllocator<SystemAllocator>,
) -> ZigzagAllocStats {
    // SAFETY: The caller ensures `ptr` is valid.
    let s = unsafe { (*ptr).stats() };
    ZigzagAllocStats {
        allocs: s.allocs,
        deallocs: s.deallocs,
        bytes: s.bytes_live,
    }
}

/// Allocates memory directly from a `CountingAllocator`.
///
/// # Safety
/// - `ptr` must be a valid pointer to a `CountingAllocator`.
/// - `size` and `align` must form a valid `Layout`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_counting_alloc(
    ptr: *mut CountingAllocator<SystemAllocator>,
    size: usize,
    align: usize,
) -> *mut c_void {
    let layout = match Layout::from_size_align(size, align) {
        Ok(l) => l,
        Err(_) => return core::ptr::null_mut(),
    };
    // SAFETY: The caller guarantees `ptr` points to a valid `CountingAllocator`.
    match unsafe { (*ptr).alloc(layout) } {
        Some(p) => p.as_ptr() as *mut c_void,
        None => core::ptr::null_mut(),
    }
}

/// Deallocates memory directly back into a `CountingAllocator`.
///
/// # Safety
/// - `ptr` must be a valid pointer to a `CountingAllocator`.
/// - `mem` must be a pointer currently allocated by this allocator.
/// - `size` and `align` must exactly match the parameters used to allocate the block.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_counting_dealloc(
    ptr: *mut CountingAllocator<SystemAllocator>,
    mem: *mut c_void,
    size: usize,
    align: usize,
) {
    if mem.is_null() {
        return;
    }
    if let Ok(layout) = Layout::from_size_align(size, align) {
        // SAFETY: Caller guarantees `mem` was allocated by `ptr` with `layout`.
        unsafe {
            (*ptr).dealloc(NonNull::new_unchecked(mem as *mut u8), layout);
        }
    }
}
