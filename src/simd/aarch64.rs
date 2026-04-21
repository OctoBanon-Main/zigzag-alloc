//! AArch64 SIMD backend using ARM NEON intrinsics.
//!
//! Provides 128-bit (16-byte) vector implementations of all SIMD primitives
//! required by the crate, matching the x86_64 SSE2 baseline in capability.
//!
//! # Safety
//!
//! Every public function is `unsafe` because raw pointer arithmetic is
//! performed without bounds checks.  Callers must ensure all pointer
//! arguments are valid for the stated number of bytes.

#![allow(dead_code)]

use core::arch::aarch64::*;
use super::{BitMask, CTRL_EMPTY};

/// A 16-byte SIMD group of hash-map control bytes, backed by `uint8x16_t`.
///
/// Used by [`ExHashMap`](crate::collections::ExHashMap) to compare 16 control
/// bytes in parallel against a target tag or the empty sentinel.
pub struct Group(uint8x16_t);

impl Group {
    /// Loads 16 bytes from `ptr` into a `Group`.
    ///
    /// # Safety
    ///
    /// `ptr` must be valid for 16 bytes of reads (alignment not required).
    #[inline]
    pub unsafe fn load(ptr: *const u8) -> Self {
        // SAFETY: `vld1q_u8` performs an unaligned 16-byte load; caller guarantees
        // that `ptr` is valid for at least 16 bytes.
        Group(unsafe { vld1q_u8(ptr) })
    }

    /// Returns a [`BitMask`] with bit `i` set wherever byte `i` equals `byte`.
    #[inline]
    pub fn match_byte(&self, byte: u8) -> BitMask {
        // SAFETY: NEON comparison operates on the already-loaded register value;
        // no memory access takes place here.
        BitMask(unsafe { neon_movemask(vceqq_u8(self.0, vdupq_n_u8(byte))) })
    }

    /// Returns a [`BitMask`] for positions equal to [`CTRL_EMPTY`].
    #[inline]
    pub fn match_empty(&self) -> BitMask {
        self.match_byte(CTRL_EMPTY)
    }

    /// Returns a [`BitMask`] for positions where the high bit (MSB) is set,
    /// i.e. `CTRL_EMPTY` or `CTRL_DELETED` slots.
    #[inline]
    pub fn match_empty_or_deleted(&self) -> BitMask {
        unsafe {
            // Shift each byte right by 7 to isolate the MSB into bit 0.
            let msb  = vshrq_n_u8::<7>(self.0);
            let cmp  = vceqq_u8(msb, vdupq_n_u8(1));
            BitMask(neon_movemask(cmp))
        }
    }

    /// Returns `true` if every byte in the group represents an occupied slot
    /// (MSB == 0).
    #[inline]
    pub fn all_full(&self) -> bool {
        self.match_empty_or_deleted().is_empty()
    }
}

/// Converts a 16-lane byte comparison mask (`uint8x16_t`) into a 16-bit
/// positional [`u32`] suitable for [`BitMask`].
///
/// ARM NEON does not have a direct equivalent of `_mm_movemask_epi8`, so we
/// emulate it:
/// 1. Narrow each 16-bit lane to 8 bits by shifting right 4 (packs two bytes
///    into one nibble, collapsing 16 bytes → 8 bytes in a `uint8x8_t`).
/// 2. Reinterpret as `uint64_t` and extract lane 0.
/// 3. Walk the 64 bits 4 at a time and collect the LSB of each nibble.
///
/// # Safety
///
/// `eq` must be the result of a byte-wise comparison (values are 0x00 or 0xFF).
#[inline]
unsafe fn neon_movemask(eq: uint8x16_t) -> u32 {
    unsafe {
        // Narrow to 8 bytes: each original byte's 0xFF → 0xF0 nibble, 0x00 → 0x00.
        let shrunk  = vshrn_n_u16::<4>(vreinterpretq_u16_u8(eq));
        let as_u64  = vget_lane_u64::<0>(vreinterpret_u64_u8(shrunk));
        let mut r   = 0u32;
        for i in 0..16u32 {
            // The LSB of each nibble corresponds to the original byte's match.
            if (as_u64 >> (i * 4)) & 1 != 0 { r |= 1 << i; }
        }
        r
    }
}

/// Computes `dst[i] &= src[i]` for `n` `usize` words using NEON 128-bit AND.
///
/// # Safety
///
/// * `dst` must be valid for `n * size_of::<usize>()` bytes of reads and writes.
/// * `src` must be valid for `n * size_of::<usize>()` bytes of reads.
/// * Regions must not overlap.
pub unsafe fn and_words(dst: *mut usize, src: *const usize, n: usize) {
    unsafe {
        let (d, s) = (dst as *mut u64, src as *const u64);
        let c = n / 2;
        for i in 0..c {
            // SAFETY: `d.add(i * 2)` and `s.add(i * 2)` are within their
            // respective valid ranges.
            let v = vandq_u64(vld1q_u64(d.add(i * 2)), vld1q_u64(s.add(i * 2)));
            vst1q_u64(d.add(i * 2), v);
        }
        // Scalar tail for any leftover odd word.
        for i in (c * 2)..n { *dst.add(i) &= *src.add(i); }
    }
}

/// Computes `dst[i] |= src[i]` for `n` `usize` words using NEON 128-bit OR.
///
/// # Safety
///
/// Same as [`and_words`].
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

/// Computes `dst[i] ^= src[i]` for `n` `usize` words using NEON 128-bit XOR.
///
/// # Safety
///
/// Same as [`and_words`].
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

/// Computes `dst[i] = !dst[i]` for `n` `usize` words using NEON NOT.
///
/// NEON lacks a direct integer NOT, so we use `vmvnq_u32` on the
/// reinterpreted lanes.
///
/// # Safety
///
/// * `dst` must be valid for `n * size_of::<usize>()` bytes of reads and writes.
pub unsafe fn not_words(dst: *mut usize, n: usize) {
    unsafe {
        let d = dst as *mut u64;
        let c = n / 2;
        for i in 0..c {
            let v    = vld1q_u64(d.add(i * 2));
            // Reinterpret as u32 for `vmvnq_u32` (bitwise NOT), then back to u64.
            let notv = vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(v)));
            vst1q_u64(d.add(i * 2), notv);
        }
        for i in (c * 2)..n { *dst.add(i) = !*dst.add(i); }
    }
}

/// Returns the total popcount across `n` `usize` words using NEON `vcntq_u8`.
///
/// `vcntq_u8` counts set bits in each byte; `vaddvq_u8` horizontally sums
/// the 16 byte-counts into a single `u8` (then widened to `u32`).
///
/// # Safety
///
/// * `ptr` must be valid for `n * size_of::<usize>()` bytes of reads.
pub unsafe fn popcount_words(ptr: *const usize, n: usize) -> usize {
    unsafe {
        let p = ptr as *const u64;
        let c = n / 2;
        let mut acc = 0u32;
        for i in 0..c {
            // Load 16 bytes (2 × u64), count bits per byte, then sum.
            let bytes = vld1q_u8(p.add(i * 2) as *const u8);
            acc += vaddvq_u8(vcntq_u8(bytes)) as u32;
        }
        let mut total = acc as usize;
        // Scalar tail for any leftover odd word.
        for i in (c * 2)..n { total += (*ptr.add(i)).count_ones() as usize; }
        total
    }
}

/// Sets `n` bytes starting at `ptr` to `val` using NEON 128-bit stores.
///
/// # Safety
///
/// * `ptr` must be valid for `n` bytes of writes.
pub unsafe fn fill_bytes(ptr: *mut u8, val: u8, n: usize) {
    unsafe {
        let v = vdupq_n_u8(val);
        let c = n / 16;
        for i in 0..c { vst1q_u8(ptr.add(i * 16), v); }
        // Scalar tail.
        for i in (c * 16)..n { *ptr.add(i) = val; }
    }
}

/// Returns the offset of the first byte equal to `val` in `[ptr, ptr+n)`.
///
/// Scans 16 bytes at a time using NEON comparison and movemask; falls back
/// to scalar for the tail.
///
/// # Safety
///
/// * `ptr` must be valid for `n` bytes of reads.
pub unsafe fn find_byte(ptr: *const u8, val: u8, n: usize) -> Option<usize> {
    unsafe {
        let needle = vdupq_n_u8(val);
        let c = n / 16;
        for i in 0..c {
            let block = vld1q_u8(ptr.add(i * 16));
            let mask  = neon_movemask(vceqq_u8(block, needle));
            if mask != 0 {
                return Some(i * 16 + mask.trailing_zeros() as usize);
            }
        }
        // Scalar tail.
        for i in (c * 16)..n {
            if *ptr.add(i) == val { return Some(i); }
        }
        None
    }
}

/// Copies `n` bytes from `src` to `dst` using `copy_nonoverlapping`.
///
/// # Safety
///
/// * `dst` must be valid for `n` bytes of writes.
/// * `src` must be valid for `n` bytes of reads.
/// * The source and destination regions must **not** overlap.
pub unsafe fn copy_bytes(dst: *mut u8, src: *const u8, n: usize) {
    // SAFETY: Caller guarantees non-overlapping regions of sufficient size.
    unsafe { core::ptr::copy_nonoverlapping(src, dst, n) };
}