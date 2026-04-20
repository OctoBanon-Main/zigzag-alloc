#![allow(dead_code)]

#[cfg(target_arch = "x86_64")]
mod x86;
#[cfg(target_arch = "aarch64")]
mod aarch64;
mod fallback;

pub const CTRL_EMPTY:   u8 = 0b1000_0000; // 0x80
pub const CTRL_DELETED: u8 = 0b1111_1110; // 0xFE

pub const GROUP_WIDTH: usize = 16;

#[derive(Copy, Clone)]
pub struct BitMask(pub u32);

impl BitMask {
    #[inline] pub fn is_empty(self)   -> bool        { self.0 == 0 }
    #[inline] pub fn any(self)        -> bool        { self.0 != 0 }

    #[inline] pub fn lowest(self)     -> Option<usize> {
        if self.0 == 0 { None } else { Some(self.0.trailing_zeros() as usize) }
    }

    #[inline] fn remove_lowest(self) -> Self { BitMask(self.0 & self.0.wrapping_sub(1)) }
}

impl Iterator for BitMask {
    type Item = usize;
    fn next(&mut self) -> Option<usize> {
        let pos = self.lowest()?;
        *self = self.remove_lowest();
        Some(pos)
    }
}

#[cfg(target_arch = "x86_64")]
pub use x86::Group;

#[cfg(target_arch = "aarch64")]
pub use aarch64::Group;

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub use fallback::Group;

#[inline]
pub unsafe fn and_words(dst: *mut usize, src: *const usize, n: usize) {
    #[cfg(target_arch = "x86_64")]
    unsafe { x86::and_words(dst, src, n); return; }
    #[cfg(target_arch = "aarch64")]
    unsafe { aarch64::and_words(dst, src, n); return; }
    #[allow(unreachable_code)]
    unsafe { fallback::and_words(dst, src, n); }
}

#[inline]
pub unsafe fn or_words(dst: *mut usize, src: *const usize, n: usize) {
    #[cfg(target_arch = "x86_64")]
    unsafe { x86::or_words(dst, src, n); return; }
    #[cfg(target_arch = "aarch64")]
    unsafe { aarch64::or_words(dst, src, n); return; }
    #[allow(unreachable_code)]
    unsafe { fallback::or_words(dst, src, n); }
}

#[inline]
pub unsafe fn xor_words(dst: *mut usize, src: *const usize, n: usize) {
    #[cfg(target_arch = "x86_64")]
    unsafe { x86::xor_words(dst, src, n); return; }
    #[cfg(target_arch = "aarch64")]
    unsafe { aarch64::xor_words(dst, src, n); return; }
    #[allow(unreachable_code)]
    unsafe { fallback::xor_words(dst, src, n); }
}

#[inline]
pub unsafe fn not_words(dst: *mut usize, n: usize) {
    #[cfg(target_arch = "x86_64")]
    unsafe { x86::not_words(dst, n); return; }
    #[cfg(target_arch = "aarch64")]
    unsafe { aarch64::not_words(dst, n); return; }
    #[allow(unreachable_code)]
    unsafe { fallback::not_words(dst, n); }
}

#[inline]
pub unsafe fn popcount_words(ptr: *const usize, n: usize) -> usize {
    #[cfg(target_arch = "x86_64")]
    unsafe { return x86::popcount_words(ptr, n); }
    #[cfg(target_arch = "aarch64")]
    unsafe { return aarch64::popcount_words(ptr, n); }
    #[allow(unreachable_code)]
    unsafe { fallback::popcount_words(ptr, n) }
}

#[inline]
pub unsafe fn fill_bytes(ptr: *mut u8, val: u8, n: usize) {
    #[cfg(target_arch = "x86_64")]
    unsafe { x86::fill_bytes(ptr, val, n); return; }
    #[cfg(target_arch = "aarch64")]
    unsafe { aarch64::fill_bytes(ptr, val, n); return; }
    #[allow(unreachable_code)]
    unsafe { fallback::fill_bytes(ptr, val, n); }
}

#[inline]
pub unsafe fn find_byte(ptr: *const u8, val: u8, n: usize) -> Option<usize> {
    #[cfg(target_arch = "x86_64")]
    unsafe { return x86::find_byte(ptr, val, n); }
    #[cfg(target_arch = "aarch64")]
    unsafe { return aarch64::find_byte(ptr, val, n); }
    #[allow(unreachable_code)]
    unsafe { fallback::find_byte(ptr, val, n) }
}

#[inline]
pub unsafe fn copy_bytes(dst: *mut u8, src: *const u8, n: usize) {
    #[cfg(target_arch = "x86_64")]
    unsafe { x86::copy_bytes(dst, src, n); return; }
    #[cfg(target_arch = "aarch64")]
    unsafe { aarch64::copy_bytes(dst, src, n); return; }
    #[allow(unreachable_code)]
    unsafe { fallback::copy_bytes(dst, src, n); }
}