use core::{alloc::Layout, ffi::c_void, ptr::NonNull};

use super::{sys_box_drop, sys_box_new, RawAllocHandle};

pub struct FfiVec {
    ptr:        *mut u8,
    len:        usize,
    cap:        usize,
    elem_size:  usize,
    elem_align: usize,
    alloc:      RawAllocHandle,
}

impl FfiVec {
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

    fn grow(&mut self) -> bool {
        let new_cap  = if self.cap == 0 { 4 } else { self.cap * 2 };
        let new_size = match new_cap.checked_mul(self.elem_size) {
            Some(s) if s > 0 => s,
            _                => return false,
        };
        let new_layout = match Layout::from_size_align(new_size, self.elem_align) {
            Ok(l)  => l,
            Err(_) => return false,
        };

        let alloc_ref = unsafe { self.alloc.as_ref() };
        let new_ptr   = match unsafe { alloc_ref.alloc(new_layout) } {
            Some(p) => p.as_ptr(),
            None    => return false,
        };

        if self.cap > 0 {
            unsafe {
                core::ptr::copy_nonoverlapping(self.ptr, new_ptr, self.len * self.elem_size);
                let old_size   = self.cap * self.elem_size;
                let old_layout = Layout::from_size_align(old_size, self.elem_align).unwrap();
                alloc_ref.dealloc(NonNull::new_unchecked(self.ptr), old_layout);
            }
        }

        self.ptr = new_ptr;
        self.cap = new_cap;
        true
    }

    fn push(&mut self, elem: *const u8) -> bool {
        if self.len == self.cap && !self.grow() {
            return false;
        }
        unsafe {
            let dst = self.ptr.add(self.len * self.elem_size);
            core::ptr::copy_nonoverlapping(elem, dst, self.elem_size);
        }
        self.len += 1;
        true
    }

    fn pop(&mut self, out: *mut u8) -> bool {
        if self.len == 0 {
            return false;
        }
        self.len -= 1;
        if !out.is_null() {
            unsafe {
                let src = self.ptr.add(self.len * self.elem_size);
                core::ptr::copy_nonoverlapping(src, out, self.elem_size);
            }
        }
        true
    }

    fn get(&self, idx: usize) -> *mut u8 {
        if idx >= self.len {
            return core::ptr::null_mut();
        }
        unsafe { self.ptr.add(idx * self.elem_size) }
    }

    fn clear(&mut self) {
        self.len = 0;
    }
}

impl Drop for FfiVec {
    fn drop(&mut self) {
        if self.cap == 0 {
            return;
        }
        let size   = self.cap * self.elem_size;
        let layout = Layout::from_size_align(size, self.elem_align)
            .expect("FfiVec::drop: invalid layout (internal error)");
        unsafe {
            let alloc = self.alloc.as_ref();
            alloc.dealloc(NonNull::new_unchecked(self.ptr), layout);
        }
    }
}


#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_vec_create(
    alloc:      RawAllocHandle,
    elem_size:  usize,
    elem_align: usize,
) -> *mut FfiVec {
    sys_box_new(FfiVec::new(alloc, elem_size, elem_align))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_vec_destroy(ptr: *mut FfiVec) {
    unsafe { sys_box_drop(ptr) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_vec_push(ptr: *mut FfiVec, elem: *const c_void) -> i32 {
    if unsafe { (*ptr).push(elem as *const u8) } { 1 } else { 0 }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_vec_pop(ptr: *mut FfiVec, out: *mut c_void) -> i32 {
    if unsafe { (*ptr).pop(out as *mut u8) } { 1 } else { 0 }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_vec_get(ptr: *const FfiVec, idx: usize) -> *mut c_void {
    unsafe { (*ptr).get(idx) as *mut c_void }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_vec_len(ptr: *const FfiVec) -> usize {
    unsafe { (*ptr).len }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_vec_capacity(ptr: *const FfiVec) -> usize {
    unsafe { (*ptr).cap }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_vec_clear(ptr: *mut FfiVec) {
    unsafe { (*ptr).clear() }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_vec_data(ptr: *const FfiVec) -> *mut c_void {
    unsafe { (*ptr).ptr as *mut c_void }
}

pub struct FfiString {
    ptr:   *mut u8,
    len:   usize,
    cap:   usize,
    alloc: RawAllocHandle,
}

impl FfiString {
    fn new(alloc: RawAllocHandle) -> Self {
        Self {
            ptr: core::ptr::null_mut(),
            len: 0,
            cap: 0,
            alloc,
        }
    }

    #[inline]
    fn alloc_size(cap: usize) -> usize {
        cap + 1
    }

    fn grow_to(&mut self, needed: usize) -> bool {
        if needed <= self.cap {
            return true;
        }
        let new_cap = {
            let mut c = if self.cap == 0 { 16 } else { self.cap * 2 };
            while c < needed { c *= 2; }
            c
        };

        let alloc_ref = unsafe { self.alloc.as_ref() };
        let new_layout = match Layout::from_size_align(Self::alloc_size(new_cap), 1) {
            Ok(l)  => l,
            Err(_) => return false,
        };
        let new_ptr = match unsafe { alloc_ref.alloc(new_layout) } {
            Some(p) => p.as_ptr(),
            None    => return false,
        };

        if self.cap > 0 {
            unsafe {
                core::ptr::copy_nonoverlapping(self.ptr, new_ptr, self.len + 1);
                let old_layout = Layout::from_size_align(Self::alloc_size(self.cap), 1).unwrap();
                alloc_ref.dealloc(NonNull::new_unchecked(self.ptr), old_layout);
            }
        } else {
            unsafe { *new_ptr = 0 };
        }

        self.ptr = new_ptr;
        self.cap = new_cap;
        true
    }

    fn push_bytes(&mut self, s: *const u8, len: usize) -> bool {
        if len == 0 {
            return true;
        }
        let needed = match self.len.checked_add(len) {
            Some(n) => n,
            None    => return false,
        };
        if !self.grow_to(needed) {
            return false;
        }
        unsafe {
            core::ptr::copy_nonoverlapping(s, self.ptr.add(self.len), len);
        }
        self.len += len;
        unsafe { *self.ptr.add(self.len) = 0 };
        true
    }

    fn clear(&mut self) {
        self.len = 0;
        if !self.ptr.is_null() {
            unsafe { *self.ptr = 0 };
        }
    }

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
        unsafe {
            let alloc = self.alloc.as_ref();
            alloc.dealloc(NonNull::new_unchecked(self.ptr), layout);
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_string_create(alloc: RawAllocHandle) -> *mut FfiString {
    sys_box_new(FfiString::new(alloc))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_string_destroy(ptr: *mut FfiString) {
    unsafe { sys_box_drop(ptr) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_string_push_str(
    ptr: *mut FfiString,
    s:   *const u8,
    len: usize,
) -> i32 {
    if unsafe { (*ptr).push_bytes(s, len) } { 1 } else { 0 }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_string_push_cstr(ptr: *mut FfiString, s: *const u8) -> i32 {
    let mut len = 0usize;
    while unsafe { *s.add(len) } != 0 {
        len += 1;
    }
    if unsafe { (*ptr).push_bytes(s, len) } { 1 } else { 0 }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_string_as_ptr(ptr: *const FfiString) -> *const u8 {
    unsafe { (*ptr).as_c_str() }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_string_len(ptr: *const FfiString) -> usize {
    unsafe { (*ptr).len }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zigzag_string_clear(ptr: *mut FfiString) {
    unsafe { (*ptr).clear() }
}