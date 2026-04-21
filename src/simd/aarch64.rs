#![allow(dead_code)]

use core::arch::aarch64::*;
use super::{BitMask, CTRL_EMPTY};

pub struct Group(uint8x16_t);

impl Group {
    #[inline]
    pub unsafe fn load(ptr: *const u8) -> Self {
        Group(unsafe { vld1q_u8(ptr) })
    }

    #[inline]
    pub fn match_byte(&self, byte: u8) -> BitMask {
        BitMask(unsafe { neon_movemask(vceqq_u8(self.0, vdupq_n_u8(byte))) })
    }

    #[inline]
    pub fn match_empty(&self) -> BitMask {
        self.match_byte(CTRL_EMPTY)
    }

    #[inline]
    pub fn match_empty_or_deleted(&self) -> BitMask {
        unsafe {
            let msb  = vshrq_n_u8::<7>(self.0);
            let cmp  = vceqq_u8(msb, vdupq_n_u8(1));
            BitMask(neon_movemask(cmp))
        }
    }

    #[inline]
    pub fn all_full(&self) -> bool {
        self.match_empty_or_deleted().is_empty()
    }
}

#[inline]
unsafe fn neon_movemask(eq: uint8x16_t) -> u32 {
    unsafe {
        let shrunk  = vshrn_n_u16::<4>(vreinterpretq_u16_u8(eq));
        let as_u64  = vget_lane_u64::<0>(vreinterpret_u64_u8(shrunk));
        let mut r   = 0u32;
        for i in 0..16u32 {
            if (as_u64 >> (i * 4)) & 1 != 0 { r |= 1 << i; }
        }
        r
    }
}

pub unsafe fn and_words(dst: *mut usize, src: *const usize, n: usize) {
    unsafe {
        let (d, s) = (dst as *mut u64, src as *const u64);
        let c = n / 2;
        for i in 0..c {
            let v = vandq_u64(vld1q_u64(d.add(i * 2)), vld1q_u64(s.add(i * 2)));
            vst1q_u64(d.add(i * 2), v);
        }
        for i in (c * 2)..n { *dst.add(i) &= *src.add(i); }
    }
}

pub unsafe fn or_words(dst: *mut usize, src: *const usize, n: usize) {
    unsafe {
        let (d, s) = (dst as *mut u64, src as *const u64);
        let c = n / 2;
        for i in 0..c {
            let v = vorrq_u64(vld1q_u64(d.add(i * 2)), vld1q_u64(s.add(i * 2)));
            vst1q_u64(d.add(i * 2), v);
        }
        for i in (c * 2)..n { *dst.add(i) |= *src.add(i); }
    }
}

pub unsafe fn xor_words(dst: *mut usize, src: *const usize, n: usize) {
    unsafe {
        let (d, s) = (dst as *mut u64, src as *const u64);
        let c = n / 2;
        for i in 0..c {
            let v = veorq_u64(vld1q_u64(d.add(i * 2)), vld1q_u64(s.add(i * 2)));
            vst1q_u64(d.add(i * 2), v);
        }
        for i in (c * 2)..n { *dst.add(i) ^= *src.add(i); }
    }
}

pub unsafe fn not_words(dst: *mut usize, n: usize) {
    unsafe {
        let d = dst as *mut u64;
        let c = n / 2;
        for i in 0..c {
            let v = vld1q_u64(d.add(i * 2));
            let notv = vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(v)));
            vst1q_u64(d.add(i * 2), notv);
        }
        for i in (c * 2)..n { *dst.add(i) = !*dst.add(i); }
    }
}

pub unsafe fn popcount_words(ptr: *const usize, n: usize) -> usize {
    unsafe {
        let p = ptr as *const u64;
        let c = n / 2;
        let mut acc = 0u32;
        for i in 0..c {
            let bytes = vld1q_u8(p.add(i * 2) as *const u8);
            acc += vaddvq_u8(vcntq_u8(bytes)) as u32;
        }
        let mut total = acc as usize;
        for i in (c * 2)..n { total += (*ptr.add(i)).count_ones() as usize; }
        total
    }
}

pub unsafe fn fill_bytes(ptr: *mut u8, val: u8, n: usize) {
    unsafe {
        let v = vdupq_n_u8(val);
        let c = n / 16;
        for i in 0..c { vst1q_u8(ptr.add(i * 16), v); }
        for i in (c * 16)..n { *ptr.add(i) = val; }
    }
}

pub unsafe fn find_byte(ptr: *const u8, val: u8, n: usize) -> Option<usize> {
    unsafe {
        let needle = vdupq_n_u8(val);
        let c = n / 16;
        for i in 0..c {
            let block = vld1q_u8(ptr.add(i * 16));
            let mask  = neon_movemask(vceqq_u8(block, needle));
            if mask != 0 { return Some(i * 16 + mask.trailing_zeros() as usize); }
        }
        for i in (c * 16)..n { if *ptr.add(i) == val { return Some(i); } }
        None
    }
}

pub unsafe fn copy_bytes(dst: *mut u8, src: *const u8, n: usize) {
    unsafe { core::ptr::copy_nonoverlapping(src, dst, n) };
}