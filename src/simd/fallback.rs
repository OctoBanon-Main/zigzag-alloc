//! Scalar SWAR (SIMD Within A Register) fallback backend.
//!
//! Used on platforms that are neither `x86_64` nor `aarch64`.
//!
//! The [`Group`] type emulates 16-byte SIMD operations using two `u64` words
//! and standard bitwise arithmetic (SWAR technique).  Byte-level operations
//! use simple scalar loops.
//!
//! This backend provides correct behaviour on all Rust targets at the cost
//! of reduced throughput compared to the hardware-accelerated backends.

// This module is private but its items are conditionally re-exported by
// `simd/mod.rs`.  `dead_code` is suppressed because the items are used
// through the cfg-gated re-export rather than by name from within this file.
#![allow(dead_code)]

use super::{BitMask, CTRL_EMPTY};

/// A 16-byte control-byte group emulated using two `u64` words.
///
/// The first word (`0`) covers bytes 0–7; the second word (`1`) covers bytes
/// 8–15.  All comparisons are performed with SWAR bitwise tricks rather than
/// hardware SIMD instructions.
pub struct Group(u64, u64);

impl Group {
    /// Loads 16 bytes from `ptr` as two unaligned `u64` words.
    ///
    /// # Safety
    ///
    /// `ptr` must be valid for 16 bytes of reads (alignment not required).
    #[inline]
    pub unsafe fn load(ptr: *const u8) -> Self {
        Group(
            // SAFETY: Each `read_unaligned` reads 8 bytes; caller guarantees
            // that `ptr` is valid for at least 16 bytes.
            unsafe { (ptr         as *const u64).read_unaligned() },
            unsafe { (ptr.add(8)  as *const u64).read_unaligned() },
        )
    }

    /// Returns a [`BitMask`] with bit `i` set wherever byte `i` equals `byte`.
    #[inline]
    pub fn match_byte(&self, byte: u8) -> BitMask {
        // Each `swar_eq8` call produces an 8-bit mask for its word.
        // The second word's bits are shifted left by 8 to occupy positions 8–15.
        BitMask(swar_eq8(self.0, byte) | (swar_eq8(self.1, byte) << 8))
    }

    /// Returns a [`BitMask`] for positions equal to [`CTRL_EMPTY`].
    #[inline]
    pub fn match_empty(&self) -> BitMask {
        self.match_byte(CTRL_EMPTY)
    }

    /// Returns a [`BitMask`] for positions where the MSB of the byte is set
    /// (i.e. `CTRL_EMPTY` or `CTRL_DELETED` slots).
    #[inline]
    pub fn match_empty_or_deleted(&self) -> BitMask {
        BitMask(msb_mask8(self.0) | (msb_mask8(self.1) << 8))
    }

    /// Returns `true` if every byte represents an occupied slot (MSB == 0).
    #[inline]
    pub fn all_full(&self) -> bool {
        self.match_empty_or_deleted().is_empty()
    }
}

/// SWAR zero-byte detection: returns an 8-bit mask where bit `i` is set if
/// byte `i` of `word` equals `byte`.
///
/// The algorithm XORs `word` with a broadcast of `byte`, then applies the
/// standard "has-zero-byte" trick: `(v - 0x01) & ~v & 0x80` detects zero bytes
/// in each lane.
#[inline]
fn swar_eq8(word: u64, byte: u8) -> u32 {
    let broadcast = u64::from_le_bytes([byte; 8]);
    let xored = word ^ broadcast;
    let lo    = xored.wrapping_sub(0x0101_0101_0101_0101);
    let hi    = !xored;
    // Bits set in `zeros` at position `8*i + 7` indicate that byte `i` matched.
    let zeros = (lo & hi) & 0x8080_8080_8080_8080;
    compress_msb(zeros)
}

/// Collects the MSB of every byte in `word` into a contiguous 8-bit value.
///
/// Equivalent to `_mm_movemask_epi8` but for a single 64-bit word.
#[inline]
fn msb_mask8(word: u64) -> u32 {
    compress_msb(word & 0x8080_8080_8080_8080)
}

/// Collects bit 7 of each byte in `v` into the low 8 bits of a `u32`.
///
/// `v` is assumed to have only the MSB of each byte potentially set (i.e. it
/// is already masked with `0x8080_8080_8080_8080`).
#[inline]
fn compress_msb(v: u64) -> u32 {
    let mut r = 0u32;
    for i in 0..8 {
        if v & (0x80u64 << (i * 8)) != 0 { r |= 1 << i; }
    }
    r
}

/// Computes `dst[i] &= src[i]` for `n` `usize` words (scalar).
///
/// # Safety
///
/// * `dst` must be valid for `n * size_of::<usize>()` bytes of reads and writes.
/// * `src` must be valid for `n * size_of::<usize>()` bytes of reads.
/// * Regions must not overlap.
pub unsafe fn and_words(dst: *mut usize, src: *const usize, n: usize) {
    for i in 0..n { unsafe { *dst.add(i) &= *src.add(i) }; }
}

/// Computes `dst[i] |= src[i]` for `n` `usize` words (scalar).
///
/// # Safety
///
/// Same as [`and_words`].
pub unsafe fn or_words(dst: *mut usize, src: *const usize, n: usize) {
    for i in 0..n { unsafe { *dst.add(i) |= *src.add(i) }; }
}

/// Computes `dst[i] ^= src[i]` for `n` `usize` words (scalar).
///
/// # Safety
///
/// Same as [`and_words`].
pub unsafe fn xor_words(dst: *mut usize, src: *const usize, n: usize) {
    for i in 0..n { unsafe { *dst.add(i) ^= *src.add(i) }; }
}

/// Computes `dst[i] = !dst[i]` for `n` `usize` words (scalar).
///
/// # Safety
///
/// * `dst` must be valid for `n * size_of::<usize>()` bytes of reads and writes.
pub unsafe fn not_words(dst: *mut usize, n: usize) {
    for i in 0..n { unsafe { *dst.add(i) = !*dst.add(i) }; }
}

/// Returns the total popcount across `n` `usize` words (scalar).
///
/// # Safety
///
/// * `ptr` must be valid for `n * size_of::<usize>()` bytes of reads.
pub unsafe fn popcount_words(ptr: *const usize, n: usize) -> usize {
    let mut acc = 0usize;
    for i in 0..n { acc += unsafe { (*ptr.add(i)).count_ones() as usize }; }
    acc
}

/// Sets `n` bytes starting at `ptr` to `val` (scalar loop).
///
/// # Safety
///
/// * `ptr` must be valid for `n` bytes of writes.
pub unsafe fn fill_bytes(ptr: *mut u8, val: u8, n: usize) {
    for i in 0..n { unsafe { *ptr.add(i) = val }; }
}

/// Returns the offset of the first byte equal to `val` in `[ptr, ptr+n)`.
///
/// Linear scalar scan.
///
/// # Safety
///
/// * `ptr` must be valid for `n` bytes of reads.
pub unsafe fn find_byte(ptr: *const u8, val: u8, n: usize) -> Option<usize> {
    for i in 0..n {
        if unsafe { *ptr.add(i) } == val { return Some(i); }
    }
    None
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