use core::arch::aarch64::*;
use super::{BitMask, CTRL_EMPTY, GROUP_WIDTH};

pub struct Group(uint8x16_t);

impl Group {
    #[inline]
    pub unsafe fn load(ptr: *const u8) -> Self {
        Group(unsafe { vld1q_u8(ptr) })
    }

    #[inline]
    pub fn match_byte(&self, byte: u8) -> BitMask {
        unsafe {
            let needle  = vdupq_n_u8(byte);
            let eq      = vceqq_u8(self.0, needle);
            BitMask(neon_movemask(eq))
        }
    }

    #[inline]
    pub fn match_empty(&self) -> BitMask {
        self.match_byte(CTRL_EMPTY)
    }

    #[inline]
    pub fn match_empty_or_deleted(&self) -> BitMask {
        unsafe {
            let msb = vshrq_n_u8(self.0, 7);
            let msb_expanded = vshlq_n_u8(msb, 7);
            BitMask(neon_movemask(vceqq_u8(msb_expanded, vdupq_n_u8(0x80))))
        }
    }

    #[inline]
    pub fn all_full(&self) -> bool {
        self.match_empty_or_deleted().is_empty()
    }
}

#[inline]
unsafe fn neon_movemask(eq: uint8x16_t) -> u32 {
    let shrunk = vshrn_n_u16::<4>(vreinterpretq_u16_u8(eq));
    let as_u64 = vget_lane_u64::<0>(vreinterpret_u64_u8(shrunk));
    let mut r = 0u32;
    for i in 0..16u32 {
        if (as_u64 >> (i * 4)) & 1 != 0 {
            r |= 1 << i;
        }
    }
    r
}

pub unsafe fn and_words(dst: *mut usize, src: *const usize, n: usize) {
    let d = dst as *mut u64;
    let s = src as *const u64;
    let chunks = n / 2;
    for i in 0..chunks {
        let a = vld1q_u64(d.add(i * 2));
        let b = vld1q_u64(s.add(i * 2));
        vst1q_u64(d.add(i * 2), vandq_u64(a, b));
    }
    for i in (chunks * 2)..n { *(dst.add(i)) &= *src.add(i); }
}

pub unsafe fn or_words(dst: *mut usize, src: *const usize, n: usize) {
    let d = dst as *mut u64;
    let s = src as *const u64;
    let chunks = n / 2;
    for i in 0..chunks {
        let a = vld1q_u64(d.add(i * 2));
        let b = vld1q_u64(s.add(i * 2));
        vst1q_u64(d.add(i * 2), vorrq_u64(a, b));
    }
    for i in (chunks * 2)..n { *dst.add(i) |= *src.add(i); }
}

pub unsafe fn xor_words(dst: *mut usize, src: *const usize, n: usize) {
    let d = dst as *mut u64;
    let s = src as *const u64;
    let chunks = n / 2;
    for i in 0..chunks {
        let a = vld1q_u64(d.add(i * 2));
        let b = vld1q_u64(s.add(i * 2));
        vst1q_u64(d.add(i * 2), veorq_u64(a, b));
    }
    for i in (chunks * 2)..n { *dst.add(i) ^= *src.add(i); }
}

pub unsafe fn not_words(dst: *mut usize, n: usize) {
    let d = dst as *mut u64;
    let chunks = n / 2;
    for i in 0..chunks {
        let a = vld1q_u64(d.add(i * 2));
        let inverted = vmvnq_u32(vreinterpretq_u32_u64(a));
        vst1q_u64(d.add(i * 2), vreinterpretq_u64_u32(inverted));
    }
    for i in (chunks * 2)..n { *dst.add(i) = !*dst.add(i); }
}

pub unsafe fn popcount_words(ptr: *const usize, n: usize) -> usize {
    let p = ptr as *const u64;
    let chunks = n / 2;
    let mut acc = 0u32;
    for i in 0..chunks {
        let v = vld1q_u8(p.add(i * 2) as *const u8);
        let cnt = vcntq_u8(v);
        acc += vaddvq_u8(cnt) as u32;
    }
    let mut scalar = acc as usize;
    for i in (chunks * 2)..n { scalar += (*ptr.add(i)).count_ones() as usize; }
    scalar
}

pub unsafe fn fill_bytes(ptr: *mut u8, val: u8, n: usize) {
    let v = vdupq_n_u8(val);
    let chunks = n / 16;
    for i in 0..chunks { vst1q_u8(ptr.add(i * 16), v); }
    for i in (chunks * 16)..n { *ptr.add(i) = val; }
}

pub unsafe fn find_byte(ptr: *const u8, val: u8, n: usize) -> Option<usize> {
    let needle = vdupq_n_u8(val);
    let chunks = n / 16;
    for i in 0..chunks {
        let block = vld1q_u8(ptr.add(i * 16));
        let eq    = vceqq_u8(block, needle);
        let mask  = neon_movemask(eq);
        if mask != 0 {
            return Some(i * 16 + mask.trailing_zeros() as usize);
        }
    }
    for i in (chunks * 16)..n {
        if *ptr.add(i) == val { return Some(i); }
    }
    None
}

pub unsafe fn copy_bytes(dst: *mut u8, src: *const u8, n: usize) {
    core::ptr::copy_nonoverlapping(src, dst, n);
}