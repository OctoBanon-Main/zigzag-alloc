use core::{alloc::Layout, ffi::c_void, ptr::NonNull};

use crate::alloc::allocator::Allocator;
use crate::collections::{HashContext, OrdContext};

pub use crate::collections::bounded_array::ExBoundedArray;
pub use crate::collections::hash_map::ExHashMap;
pub use crate::collections::priority_queue::ExPriorityQueue;

use super::{RawAllocHandle, sys_box_drop, sys_box_new};

/// A dynamically sized, C-compatible vector that manages raw bytes.
pub struct FfiVec {
    ptr: *mut u8,
    len: usize,
    cap: usize,
    elem_size: usize,
    elem_align: usize,
    alloc: RawAllocHandle,
}

impl FfiVec {
    /// Creates a new, empty `FfiVec` bound to a specific allocator and element layout.
    fn new(alloc: RawAllocHandle, elem_size: usize, elem_align: usize) -> Self {
        Self {
            ptr: core::ptr::null_mut(),
            len: 0,
            cap: 0,
            elem_size,
            elem_align,
            alloc,
        }
    }

    /// Enlarges the vector's capacity. Returns `true` on success, `false` on OOM.
    fn grow(&mut self) -> bool {
        let new_cap = if self.cap == 0 { 4 } else { self.cap * 2 };
        let new_size = match new_cap.checked_mul(self.elem_size) {
            Some(s) if s > 0 => s,
            _ => return false,
        };
        let new_layout = match Layout::from_size_align(new_size, self.elem_align) {
            Ok(l) => l,
            Err(_) => return false,
        };

        // SAFETY: The allocator handle is guaranteed to be valid by FfiVec's constructor.
        let alloc_ref = unsafe { self.alloc.as_ref() };
        let new_ptr = match unsafe { alloc_ref.alloc(new_layout) } {
            Some(p) => p.as_ptr(),
            None => return false,
        };

        if self.cap > 0 {
            // SAFETY: Both `self.ptr` and `new_ptr` are valid, disjoint regions.
            unsafe {
                core::ptr::copy_nonoverlapping(self.ptr, new_ptr, self.len * self.elem_size);
                let old_size = self.cap * self.elem_size;
                let old_layout = Layout::from_size_align(old_size, self.elem_align).unwrap();
                alloc_ref.dealloc(NonNull::new_unchecked(self.ptr), old_layout);
            }
        }

        self.ptr = new_ptr;
        self.cap = new_cap;
        true
    }

    /// Pushes an element (copied by bytes) onto the end of the vector.
    fn push(&mut self, elem: *const u8) -> bool {
        if self.len == self.cap && !self.grow() {
            return false;
        }
        // SAFETY: `dst` is within bounds due to the `grow` check. `elem` must point
        // to a valid memory region of at least `self.elem_size` bytes.
        unsafe {
            let dst = self.ptr.add(self.len * self.elem_size);
            core::ptr::copy_nonoverlapping(elem, dst, self.elem_size);
        }
        self.len += 1;
        true
    }

    /// Pops an element from the end of the vector, writing it to `out` (if not null).
    fn pop(&mut self, out: *mut u8) -> bool {
        if self.len == 0 {
            return false;
        }
        self.len -= 1;
        if !out.is_null() {
            // SAFETY: `src` is within bounds. `out` must be valid for `elem_size` bytes.
            unsafe {
                let src = self.ptr.add(self.len * self.elem_size);
                core::ptr::copy_nonoverlapping(src, out, self.elem_size);
            }
        }
        true
    }

    /// Returns a pointer to the element at `idx`, or null if out of bounds.
    fn get(&self, idx: usize) -> *mut u8 {
        if idx >= self.len {
            return core::ptr::null_mut();
        }
        // SAFETY: `idx < self.len`, so the pointer arithmetic is within bounds.
        unsafe { self.ptr.add(idx * self.elem_size) }
    }

    /// Clears the vector without dropping its capacity.
    fn clear(&mut self) {
        self.len = 0;
    }
}

impl Drop for FfiVec {
    fn drop(&mut self) {
        if self.cap == 0 {
            return;
        }
        let size = self.cap * self.elem_size;
        let layout = Layout::from_size_align(size, self.elem_align)
            .expect("FfiVec::drop: invalid layout (internal error)");
        // SAFETY: `self.ptr` is valid and was allocated with the same layout.
        unsafe {
            let alloc = self.alloc.as_ref();
            alloc.dealloc(NonNull::new_unchecked(self.ptr), layout);
        }
    }
}

/// Creates a new generic FFI vector.
///
/// # Safety
/// - `alloc` must be a valid allocator handle.
/// - `elem_size` and `elem_align` must form a valid `Layout`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_vec_create(
    alloc: RawAllocHandle,
    elem_size: usize,
    elem_align: usize,
) -> *mut FfiVec {
    sys_box_new(FfiVec::new(alloc, elem_size, elem_align))
}

/// Destroys the FFI vector, freeing its contents and the structure itself.
///
/// # Safety
/// - `ptr` must be a valid pointer returned by `zigzag_vec_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_vec_destroy(ptr: *mut FfiVec) {
    // SAFETY: Caller ensures `ptr` is valid.
    unsafe { sys_box_drop(ptr) }
}

/// Pushes an element onto the vector. Returns 1 on success, 0 on OOM.
///
/// # Safety
/// - `ptr` must point to a valid `FfiVec`.
/// - `elem` must point to valid memory sized appropriately (`elem_size`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_vec_push(ptr: *mut FfiVec, elem: *const c_void) -> i32 {
    // SAFETY: Caller ensures `ptr` and `elem` are valid.
    if unsafe { (*ptr).push(elem as *const u8) } {
        1
    } else {
        0
    }
}

/// Pops an element from the vector into `out`. Returns 1 on success, 0 if empty.
///
/// # Safety
/// - `ptr` must point to a valid `FfiVec`.
/// - If `out` is not null, it must be valid for writes of `elem_size` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_vec_pop(ptr: *mut FfiVec, out: *mut c_void) -> i32 {
    // SAFETY: Caller ensures `ptr` and `out` are valid.
    if unsafe { (*ptr).pop(out as *mut u8) } {
        1
    } else {
        0
    }
}

/// Retrieves a pointer to the element at the specified index.
///
/// # Safety
/// - `ptr` must point to a valid `FfiVec`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_vec_get(ptr: *const FfiVec, idx: usize) -> *mut c_void {
    // SAFETY: Caller ensures `ptr` is valid.
    unsafe { (*ptr).get(idx) as *mut c_void }
}

/// Returns the current number of elements in the vector.
///
/// # Safety
/// - `ptr` must point to a valid `FfiVec`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_vec_len(ptr: *const FfiVec) -> usize {
    // SAFETY: Caller ensures `ptr` is valid.
    unsafe { (*ptr).len }
}

/// Returns the current capacity (in elements) of the vector.
///
/// # Safety
/// - `ptr` must point to a valid `FfiVec`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_vec_capacity(ptr: *const FfiVec) -> usize {
    // SAFETY: Caller ensures `ptr` is valid.
    unsafe { (*ptr).cap }
}

/// Clears the elements from the vector without deallocating the backing store.
///
/// # Safety
/// - `ptr` must point to a valid `FfiVec`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_vec_clear(ptr: *mut FfiVec) {
    // SAFETY: Caller ensures `ptr` is valid.
    unsafe { (*ptr).clear() }
}

/// Returns the raw pointer to the underlying data buffer.
///
/// # Safety
/// - `ptr` must point to a valid `FfiVec`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_vec_data(ptr: *const FfiVec) -> *mut c_void {
    // SAFETY: Caller ensures `ptr` is valid.
    unsafe { (*ptr).ptr as *mut c_void }
}

/// A dynamically sized, null-terminated C-compatible string.
pub struct FfiString {
    ptr: *mut u8,
    len: usize,
    cap: usize,
    alloc: RawAllocHandle,
}

impl FfiString {
    /// Creates a new, empty string.
    fn new(alloc: RawAllocHandle) -> Self {
        Self {
            ptr: core::ptr::null_mut(),
            len: 0,
            cap: 0,
            alloc,
        }
    }

    /// Computes the actual allocation size needed (capacity + 1 for null terminator).
    #[inline]
    fn alloc_size(cap: usize) -> usize {
        cap + 1
    }

    /// Grows the string to at least the `needed` capacity.
    fn grow_to(&mut self, needed: usize) -> bool {
        if needed <= self.cap {
            return true;
        }
        let new_cap = {
            let mut c = if self.cap == 0 { 16 } else { self.cap * 2 };
            while c < needed {
                c *= 2;
            }
            c
        };

        // SAFETY: `self.alloc` points to a valid allocator.
        let alloc_ref = unsafe { self.alloc.as_ref() };
        let new_layout = match Layout::from_size_align(Self::alloc_size(new_cap), 1) {
            Ok(l) => l,
            Err(_) => return false,
        };
        let new_ptr = match unsafe { alloc_ref.alloc(new_layout) } {
            Some(p) => p.as_ptr(),
            None => return false,
        };

        if self.cap > 0 {
            // SAFETY: Both buffers are valid and disjoint.
            unsafe {
                core::ptr::copy_nonoverlapping(self.ptr, new_ptr, self.len + 1);
                let old_layout = Layout::from_size_align(Self::alloc_size(self.cap), 1).unwrap();
                alloc_ref.dealloc(NonNull::new_unchecked(self.ptr), old_layout);
            }
        } else {
            // SAFETY: `new_ptr` is newly allocated and valid.
            unsafe { *new_ptr = 0 };
        }

        self.ptr = new_ptr;
        self.cap = new_cap;
        true
    }

    /// Appends raw bytes to the string and null-terminates it.
    fn push_bytes(&mut self, s: *const u8, len: usize) -> bool {
        if len == 0 {
            return true;
        }
        let needed = match self.len.checked_add(len) {
            Some(n) => n,
            None => return false,
        };
        if !self.grow_to(needed) {
            return false;
        }
        // SAFETY: Pointer arithmetic is within capacity bounds. `s` is valid for `len` bytes.
        unsafe {
            core::ptr::copy_nonoverlapping(s, self.ptr.add(self.len), len);
        }
        self.len += len;
        // SAFETY: `self.len` is strictly < `self.alloc_size()`.
        unsafe { *self.ptr.add(self.len) = 0 };
        true
    }

    /// Clears the string, resetting its length to 0 but retaining capacity.
    fn clear(&mut self) {
        self.len = 0;
        if !self.ptr.is_null() {
            // SAFETY: If `ptr` is not null, capacity > 0, so writing at offset 0 is safe.
            unsafe { *self.ptr = 0 };
        }
    }

    /// Returns a null-terminated C-compatible pointer.
    fn as_c_str(&self) -> *const u8 {
        if self.ptr.is_null() {
            static EMPTY: u8 = 0;
            return &EMPTY as *const u8;
        }
        self.ptr
    }
}

impl Drop for FfiString {
    fn drop(&mut self) {
        if self.cap == 0 {
            return;
        }
        let layout = Layout::from_size_align(Self::alloc_size(self.cap), 1)
            .expect("FfiString::drop: internal layout error");
        // SAFETY: `ptr` was allocated by the same allocator with the equivalent layout.
        unsafe {
            let alloc = self.alloc.as_ref();
            alloc.dealloc(NonNull::new_unchecked(self.ptr), layout);
        }
    }
}

/// Creates a new FFI string using the specified allocator.
///
/// # Safety
/// - `alloc` must be a valid allocator handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_string_create(alloc: RawAllocHandle) -> *mut FfiString {
    sys_box_new(FfiString::new(alloc))
}

/// Destroys an FFI string.
///
/// # Safety
/// - `ptr` must be a valid pointer returned by `zigzag_string_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_string_destroy(ptr: *mut FfiString) {
    // SAFETY: Caller ensures `ptr` is valid.
    unsafe { sys_box_drop(ptr) }
}

/// Pushes `len` bytes from `s` onto the string. Returns 1 on success, 0 on OOM.
///
/// # Safety
/// - `ptr` must point to a valid `FfiString`.
/// - `s` must be valid to read for at least `len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_string_push_str(
    ptr: *mut FfiString,
    s: *const u8,
    len: usize,
) -> i32 {
    // SAFETY: Caller ensures `ptr` and `s` are valid.
    if unsafe { (*ptr).push_bytes(s, len) } {
        1
    } else {
        0
    }
}

/// Pushes a null-terminated C string `s` onto the string. Returns 1 on success, 0 on OOM.
///
/// # Safety
/// - `ptr` must point to a valid `FfiString`.
/// - `s` must be a valid null-terminated string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_string_push_cstr(ptr: *mut FfiString, s: *const u8) -> i32 {
    let mut len = 0usize;
    // SAFETY: Caller ensures `s` is null-terminated.
    while unsafe { *s.add(len) } != 0 {
        len += 1;
    }
    // SAFETY: Caller ensures `ptr` is valid and `s` provides `len` bytes.
    if unsafe { (*ptr).push_bytes(s, len) } {
        1
    } else {
        0
    }
}

/// Returns a pointer to the null-terminated data of the string.
///
/// # Safety
/// - `ptr` must point to a valid `FfiString`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_string_as_ptr(ptr: *const FfiString) -> *const u8 {
    // SAFETY: Caller ensures `ptr` is valid.
    unsafe { (*ptr).as_c_str() }
}

/// Returns the current length of the string, excluding the null terminator.
///
/// # Safety
/// - `ptr` must point to a valid `FfiString`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_string_len(ptr: *const FfiString) -> usize {
    // SAFETY: Caller ensures `ptr` is valid.
    unsafe { (*ptr).len }
}

/// Clears the string, resetting length to 0 but maintaining capacity.
///
/// # Safety
/// - `ptr` must point to a valid `FfiString`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_string_clear(ptr: *mut FfiString) {
    // SAFETY: Caller ensures `ptr` is valid.
    unsafe { (*ptr).clear() }
}

/// A C-compatible struct representing hashing and equality functions.
#[repr(C)]
pub struct FfiHashContext {
    pub hash_fn: extern "C" fn(usize) -> u64,
    pub eq_fn: extern "C" fn(usize, usize) -> bool,
}

impl HashContext<usize> for FfiHashContext {
    fn hash(&self, k: &usize) -> u64 {
        (self.hash_fn)(*k)
    }
    fn eq(&self, a: &usize, b: &usize) -> bool {
        (self.eq_fn)(*a, *b)
    }
}

/// A generic hash map mapping a `usize` key to a raw void pointer.
pub struct FfiHashMap {
    map: ExHashMap<'static, usize, *mut c_void, FfiHashContext>,
}

/// Creates a new HashMap with custom hashing and equality functions.
///
/// # Safety
/// - `alloc` must be a valid allocator handle.
/// - The allocator backing the handle must live at least as long as the map.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_hashmap_create(
    alloc: RawAllocHandle,
    hash_fn: extern "C" fn(usize) -> u64,
    eq_fn: extern "C" fn(usize, usize) -> bool,
) -> *mut FfiHashMap {
    let ctx = FfiHashContext { hash_fn, eq_fn };
    // SAFETY: Caller ensures `alloc` is valid.
    let alloc_ref = unsafe { alloc.as_ref() };
    // SAFETY: Transmuting the lifetime to 'static is inherently unsafe but required
    // here because the FFI boundary cannot express lifetimes. The caller must guarantee
    // the allocator outlives the collection.
    let static_alloc: &'static dyn Allocator = unsafe { core::mem::transmute(alloc_ref) };

    sys_box_new(FfiHashMap {
        map: ExHashMap::new(static_alloc, ctx),
    })
}

/// Destroys the HashMap.
///
/// # Safety
/// - `ptr` must be a valid pointer returned by `zigzag_hashmap_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_hashmap_destroy(ptr: *mut FfiHashMap) {
    // SAFETY: Caller ensures `ptr` is valid.
    unsafe { sys_box_drop(ptr) }
}

/// Inserts a key-value pair into the map. Returns 1 on success, 0 on failure (e.g., OOM).
/// If a value already existed for the key, it is placed into `out_old_val` (if not null).
///
/// # Safety
/// - `ptr` must point to a valid `FfiHashMap`.
/// - `out_old_val` must be null or point to a valid location to write the old pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_hashmap_insert(
    ptr: *mut FfiHashMap,
    key: usize,
    val: *mut c_void,
    out_old_val: *mut *mut c_void,
) -> i32 {
    // SAFETY: Caller ensures `ptr` is valid.
    match unsafe { (*ptr).map.try_insert(key, val) } {
        Ok(Some(old)) => {
            if !out_old_val.is_null() {
                // SAFETY: Caller ensures `out_old_val` is writable.
                unsafe {
                    *out_old_val = old;
                }
            }
            1
        }
        Ok(None) => 1,
        Err(_) => 0,
    }
}

/// Retrieves the value associated with `key` from the map, returning null if not found.
///
/// # Safety
/// - `ptr` must point to a valid `FfiHashMap`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_hashmap_get(ptr: *const FfiHashMap, key: usize) -> *mut c_void {
    // SAFETY: Caller ensures `ptr` is valid.
    match unsafe { (*ptr).map.get(&key) } {
        Some(&val) => val,
        None => core::ptr::null_mut(),
    }
}

/// Removes a key from the map, returning its associated value (or null if absent).
///
/// # Safety
/// - `ptr` must point to a valid `FfiHashMap`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_hashmap_remove(ptr: *mut FfiHashMap, key: usize) -> *mut c_void {
    // SAFETY: Caller ensures `ptr` is valid.
    match unsafe { (*ptr).map.remove(&key) } {
        Some(val) => val,
        None => core::ptr::null_mut(),
    }
}

/// Returns the number of elements in the map.
///
/// # Safety
/// - `ptr` must point to a valid `FfiHashMap`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_hashmap_len(ptr: *const FfiHashMap) -> usize {
    // SAFETY: Caller ensures `ptr` is valid.
    unsafe { (*ptr).map.len() }
}

/// A C-compatible struct representing an ordering function for priority queues.
#[repr(C)]
pub struct FfiOrdContext {
    pub less_fn: extern "C" fn(*mut c_void, *mut c_void) -> bool,
}

impl OrdContext<*mut c_void> for FfiOrdContext {
    fn less(&self, a: &*mut c_void, b: &*mut c_void) -> bool {
        (self.less_fn)(*a, *b)
    }
}

/// A generic priority queue mapping elements as raw void pointers.
pub struct FfiPriorityQueue {
    pq: ExPriorityQueue<'static, *mut c_void, FfiOrdContext>,
}

/// Creates a new priority queue with a custom comparator.
///
/// # Safety
/// - `alloc` must be a valid allocator handle.
/// - The allocator backing the handle must live at least as long as the priority queue.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_pq_create(
    alloc: RawAllocHandle,
    less_fn: extern "C" fn(*mut c_void, *mut c_void) -> bool,
) -> *mut FfiPriorityQueue {
    let ctx = FfiOrdContext { less_fn };
    // SAFETY: Caller ensures `alloc` is valid.
    let alloc_ref = unsafe { alloc.as_ref() };
    // SAFETY: Required to satisfy bounded lifetimes over FFI. Caller must ensure
    // the allocator strictly outlives the priority queue.
    let static_alloc: &'static dyn Allocator = unsafe { core::mem::transmute(alloc_ref) };

    sys_box_new(FfiPriorityQueue {
        pq: ExPriorityQueue::new(static_alloc, ctx),
    })
}

/// Destroys the priority queue.
///
/// # Safety
/// - `ptr` must be a valid pointer returned by `zigzag_pq_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_pq_destroy(ptr: *mut FfiPriorityQueue) {
    // SAFETY: Caller ensures `ptr` is valid.
    unsafe { sys_box_drop(ptr) }
}

/// Pushes an element into the priority queue. Returns 1 on success, 0 on failure (OOM).
///
/// # Safety
/// - `ptr` must point to a valid `FfiPriorityQueue`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_pq_push(ptr: *mut FfiPriorityQueue, val: *mut c_void) -> i32 {
    // SAFETY: Caller ensures `ptr` is valid.
    match unsafe { (*ptr).pq.try_push(val) } {
        Ok(_) => 1,
        Err(_) => 0,
    }
}

/// Pops the highest-priority element from the queue. Returns null if empty.
///
/// # Safety
/// - `ptr` must point to a valid `FfiPriorityQueue`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_pq_pop(ptr: *mut FfiPriorityQueue) -> *mut c_void {
    // SAFETY: Caller ensures `ptr` is valid.
    match unsafe { (*ptr).pq.pop() } {
        Some(val) => val,
        None => core::ptr::null_mut(),
    }
}

/// Peeks at the highest-priority element without removing it. Returns null if empty.
///
/// # Safety
/// - `ptr` must point to a valid `FfiPriorityQueue`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_pq_peek(ptr: *const FfiPriorityQueue) -> *mut c_void {
    // SAFETY: Caller ensures `ptr` is valid.
    match unsafe { (*ptr).pq.peek() } {
        Some(&val) => val,
        None => core::ptr::null_mut(),
    }
}

/// Returns the number of elements in the priority queue.
///
/// # Safety
/// - `ptr` must point to a valid `FfiPriorityQueue`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_pq_len(ptr: *const FfiPriorityQueue) -> usize {
    // SAFETY: Caller ensures `ptr` is valid.
    unsafe { (*ptr).pq.len() }
}

/// A bounded array that holds exactly up to 256 `u8` elements.
pub struct FfiBoundedArray256 {
    arr: ExBoundedArray<u8, 256>,
}

/// Creates a new, empty `FfiBoundedArray256`.
#[unsafe(no_mangle)]
pub extern "C" fn zigzag_ba256_create() -> *mut FfiBoundedArray256 {
    sys_box_new(FfiBoundedArray256 {
        arr: ExBoundedArray::new(),
    })
}

/// Destroys the bounded array.
///
/// # Safety
/// - `ptr` must be a valid pointer returned by `zigzag_ba256_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_ba256_destroy(ptr: *mut FfiBoundedArray256) {
    // SAFETY: Caller ensures `ptr` is valid.
    unsafe { sys_box_drop(ptr) };
}

/// Pushes a byte into the bounded array. Returns 1 on success, 0 if it is full.
///
/// # Safety
/// - `ptr` must point to a valid `FfiBoundedArray256`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_ba256_push(ptr: *mut FfiBoundedArray256, val: u8) -> i32 {
    // SAFETY: Caller ensures `ptr` is valid.
    match unsafe { (*ptr).arr.push(val) } {
        Ok(_) => 1,
        Err(_) => 0,
    }
}

/// Pops a byte from the bounded array. Returns 1 on success, 0 if empty.
///
/// # Safety
/// - `ptr` must point to a valid `FfiBoundedArray256`.
/// - `out` must be null, or a valid pointer to receive the popped byte.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_ba256_pop(ptr: *mut FfiBoundedArray256, out: *mut u8) -> i32 {
    // SAFETY: Caller ensures `ptr` is valid.
    match unsafe { (*ptr).arr.pop() } {
        Some(val) => {
            if !out.is_null() {
                // SAFETY: Caller ensures `out` is writable.
                unsafe {
                    *out = val;
                }
            }
            1
        }
        None => 0,
    }
}

/// Returns the number of bytes currently stored in the bounded array.
///
/// # Safety
/// - `ptr` must point to a valid `FfiBoundedArray256`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_ba256_len(ptr: *const FfiBoundedArray256) -> usize {
    // SAFETY: Caller ensures `ptr` is valid.
    unsafe { (*ptr).arr.len() }
}

/// Fills the entire bounded array (up to capacity) with `val`.
///
/// # Safety
/// - `ptr` must point to a valid `FfiBoundedArray256`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_ba256_fill_bytes(ptr: *mut FfiBoundedArray256, val: u8) {
    // SAFETY: Caller ensures `ptr` is valid.
    unsafe { (*ptr).arr.fill_bytes(val) };
}

/// Counts the occurrences of `val` in the bounded array.
///
/// # Safety
/// - `ptr` must point to a valid `FfiBoundedArray256`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_ba256_count_byte(ptr: *const FfiBoundedArray256, val: u8) -> usize {
    // SAFETY: Caller ensures `ptr` is valid.
    unsafe { (*ptr).arr.count_byte(val) }
}

/// Returns a mutable raw pointer to the underlying slice data.
///
/// # Safety
/// - `ptr` must point to a valid `FfiBoundedArray256`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_ba256_data(ptr: *mut FfiBoundedArray256) -> *mut u8 {
    // SAFETY: Caller ensures `ptr` is valid.
    unsafe { (*ptr).arr.as_mut_slice().as_mut_ptr() }
}
