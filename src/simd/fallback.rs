use super::{BitMask, CTRL_EMPTY};


pub struct Group(u64, u64);

impl Group {
    #[inline]
    pub unsafe fn load(ptr: *const u8) -> Self {
        unsafe {
            let lo = (ptr as *const u64).read_unaligned();
            let hi = (ptr.add(8) as *const u64).read_unaligned();
            Group(lo, hi)
        }
    }

    #[inline]
    pub fn match_byte(&self, byte: u8) -> BitMask {
        BitMask(swar_eq64(self.0, byte) | (swar_eq64(self.1, byte) << 8))
    }

    #[inline]
    pub fn match_empty(&self) -> BitMask {
        self.match_byte(CTRL_EMPTY)
    }

    #[inline]
    pub fn match_empty_or_deleted(&self) -> BitMask {
        let lo = msb_mask64(self.0);
        let hi = msb_mask64(self.1);
        BitMask(lo | (hi << 8))
    }

    #[inline]
    pub fn all_full(&self) -> bool {
        self.match_empty_or_deleted().is_empty()
    }
}

#[inline]
fn swar_eq64(word: u64, byte: u8) -> u32 {
    let v  = word ^ (u64::from_le_bytes([byte; 8]));
    let lo = v & 0x7F7F_7F7F_7F7F_7F7F;
    let hi = (lo.wrapping_add(0x7F7F_7F7F_7F7F_7F7F) | lo) & 0x8080_8080_8080_8080;
    let matched = !hi & 0x8080_8080_8080_8080;
    compress_msb8(matched)
}

#[inline]
fn compress_msb8(v: u64) -> u32 {
    let mut r = 0u32;
    for i in 0..8 {
        if v & (0x80u64 << (i * 8)) != 0 {
            r |= 1 << i;
        }
    }
    r
}

#[inline]
fn msb_mask64(word: u64) -> u32 {
    compress_msb8(word & 0x8080_8080_8080_8080)
}

#[inline]
pub unsafe fn and_words(dst: *mut usize, src: *const usize, n: usize) {
    unsafe {
        for i in 0..n {
            *dst.add(i) &= *src.add(i);
        }
    }
}

#[inline]
pub unsafe fn or_words(dst: *mut usize, src: *const usize, n: usize) {
    unsafe {
        for i in 0..n {
            *dst.add(i) |= *src.add(i);
        }
    }
}

#[inline]
pub unsafe fn xor_words(dst: *mut usize, src: *const usize, n: usize) {
    unsafe {
        for i in 0..n {
            *dst.add(i) ^= *src.add(i);
        }
    }
}

#[inline]
pub unsafe fn not_words(dst: *mut usize, n: usize) {
    unsafe {
        for i in 0..n {
            *dst.add(i) = !*dst.add(i);
        }
    }
}

#[inline]
pub unsafe fn popcount_words(ptr: *const usize, n: usize) -> usize {
    let mut acc = 0usize;
    unsafe {
        for i in 0..n {
            acc += (*ptr.add(i)).count_ones() as usize;
        }
    }
    acc
}

#[inline]
pub unsafe fn fill_bytes(ptr: *mut u8, val: u8, n: usize) {
    unsafe {
        for i in 0..n {
            *ptr.add(i) = val;
        }
    }
}

#[inline]
pub unsafe fn find_byte(ptr: *const u8, val: u8, n: usize) -> Option<usize> {
    unsafe {
        for i in 0..n {
            if *ptr.add(i) == val {
                return Some(i);
            }
        }
    }
    None
}

#[inline]
pub unsafe fn copy_bytes(dst: *mut u8, src: *const u8, n: usize) {
    unsafe { core::ptr::copy_nonoverlapping(src, dst, n) };
}