//! Platform-specific SIMD primitives.
//!
//! This module selects the best available implementation at compile time:
//!
//! | Target | Backend | Width |
//! |--------|---------|-------|
//! | `x86_64` | SSE2 (AVX2 if available) | 16 / 32 bytes |
//! | `aarch64` | NEON | 16 bytes |
//! | Everything else | Scalar SWAR fallback | 8 bytes (2 × u64) |
//!
//! All functions are re-exported at the module root so callers don't need
//! to name the concrete backend.
//!
//! ## Control byte constants
//!
//! [`CTRL_EMPTY`] and [`CTRL_DELETED`] are used by [`ExHashMap`] to mark
//! vacant / tombstoned slots in the control-byte array.
//!
//! [`ExHashMap`]: crate::collections::ExHashMap

#![allow(dead_code, unused_imports)]

#[cfg(target_arch = "x86_64")]
mod x86;
#[cfg(target_arch = "aarch64")]
mod aarch64;
mod fallback;

/// Control byte value for an **empty** hash-map slot.
///
/// The high bit is set (`0x80`) so that SIMD movemask operations can detect
/// empty-or-deleted slots with a single instruction.
pub const CTRL_EMPTY: u8 = 0x80;

/// Control byte value for a **deleted** (tombstone) hash-map slot.
pub const CTRL_DELETED: u8 = 0xFE;

/// Number of control bytes processed in a single SIMD group load.
///
/// This is always 16 regardless of whether the backend is SSE2 or NEON.
/// The fallback scalar backend emulates 16-byte groups via two `u64` words.
pub const GROUP_WIDTH: usize = 16;

/// A 16-bit positional mask returned by SIMD comparison operations.
///
/// Bit `i` is set if position `i` within a [`Group`] matched the comparison.
/// Implements [`Iterator`] to yield matching positions one at a time.
#[derive(Copy, Clone)]
pub struct BitMask(pub u32);

impl BitMask {
    /// Returns `true` if no positions matched.
    #[inline]
    pub fn is_empty(self) -> bool { self.0 == 0 }

    /// Returns `true` if at least one position matched.
    #[inline]
    pub fn any(self) -> bool { self.0 != 0 }

    /// Returns the position of the lowest set bit, or `None` if the mask is zero.
    #[inline]
    pub fn lowest(self) -> Option<usize> {
        if self.0 == 0 { None } else { Some(self.0.trailing_zeros() as usize) }
    }

    /// Returns a new `BitMask` with the lowest set bit cleared.
    #[inline]
    fn remove_lowest(self) -> Self { BitMask(self.0 & self.0.wrapping_sub(1)) }
}

impl Iterator for BitMask {
    type Item = usize;
    /// Yields the next matching position and removes it from the mask.
    fn next(&mut self) -> Option<usize> {
        let pos = self.lowest()?;
        *self = self.remove_lowest();
        Some(pos)
    }
}

/// A 16-byte group of control bytes loaded for parallel SIMD comparison.
///
/// The concrete type is platform-specific:
/// * `x86_64` — wraps `__m128i`
/// * `aarch64` — wraps `uint8x16_t`
/// * fallback  — wraps two `u64` words (SWAR)
#[cfg(target_arch = "x86_64")]
pub use x86::Group;

#[cfg(target_arch = "aarch64")]
pub use aarch64::Group;

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub use fallback::Group;

/// Computes `dst[i] &= src[i]` for `n` `usize`-sized words using SIMD.
///
/// # Safety
///
/// * `dst` must be valid for `n * size_of::<usize>()` bytes of both reads and writes.
/// * `src` must be valid for `n * size_of::<usize>()` bytes of reads.
/// * The regions must not overlap.
#[inline]
pub unsafe fn and_words(dst: *mut usize, src: *const usize, n: usize) {
    #[cfg(target_arch = "x86_64")]   { unsafe { x86::and_words(dst, src, n);     return; } }
    #[cfg(target_arch = "aarch64")]  { unsafe { aarch64::and_words(dst, src, n); return; } }
    #[allow(unreachable_code)]        unsafe { fallback::and_words(dst, src, n); }
}

/// Computes `dst[i] |= src[i]` for `n` `usize`-sized words using SIMD.
///
/// # Safety
///
/// * `dst` must be valid for `n * size_of::<usize>()` bytes of both reads and writes.
/// * `src` must be valid for `n * size_of::<usize>()` bytes of reads.
/// * The regions must not overlap.
#[inline]
pub unsafe fn or_words(dst: *mut usize, src: *const usize, n: usize) {
    #[cfg(target_arch = "x86_64")]   { unsafe { x86::or_words(dst, src, n);     return; } }
    #[cfg(target_arch = "aarch64")]  { unsafe { aarch64::or_words(dst, src, n); return; } }
    #[allow(unreachable_code)]        unsafe { fallback::or_words(dst, src, n); }
}

/// Computes `dst[i] ^= src[i]` for `n` `usize`-sized words using SIMD.
///
/// # Safety
///
/// * `dst` must be valid for `n * size_of::<usize>()` bytes of both reads and writes.
/// * `src` must be valid for `n * size_of::<usize>()` bytes of reads.
/// * The regions must not overlap.
#[inline]
pub unsafe fn xor_words(dst: *mut usize, src: *const usize, n: usize) {
    #[cfg(target_arch = "x86_64")]   { unsafe { x86::xor_words(dst, src, n);     return; } }
    #[cfg(target_arch = "aarch64")]  { unsafe { aarch64::xor_words(dst, src, n); return; } }
    #[allow(unreachable_code)]        unsafe { fallback::xor_words(dst, src, n); }
}

/// Computes `dst[i] = !dst[i]` for `n` `usize`-sized words using SIMD.
///
/// # Safety
///
/// * `dst` must be valid for `n * size_of::<usize>()` bytes of both reads and writes.
#[inline]
pub unsafe fn not_words(dst: *mut usize, n: usize) {
    #[cfg(target_arch = "x86_64")]   { unsafe { x86::not_words(dst, n);     return; } }
    #[cfg(target_arch = "aarch64")]  { unsafe { aarch64::not_words(dst, n); return; } }
    #[allow(unreachable_code)]        unsafe { fallback::not_words(dst, n); }
}

/// Returns the total popcount (number of set bits) across `n` `usize` words.
///
/// # Safety
///
/// * `ptr` must be valid for `n * size_of::<usize>()` bytes of reads.
#[inline]
pub unsafe fn popcount_words(ptr: *const usize, n: usize) -> usize {
    #[cfg(target_arch = "x86_64")]   { unsafe { return x86::popcount_words(ptr, n);     } }
    #[cfg(target_arch = "aarch64")]  { unsafe { return aarch64::popcount_words(ptr, n); } }
    #[allow(unreachable_code)]        unsafe { fallback::popcount_words(ptr, n) }
}

/// Sets all `n` bytes starting at `ptr` to `val` using SIMD.
///
/// Equivalent to `memset(ptr, val, n)` but uses vectorised stores.
///
/// # Safety
///
/// * `ptr` must be valid for `n` bytes of writes.
/// * `n` must not cause the write to go out of bounds.
#[inline]
pub unsafe fn fill_bytes(ptr: *mut u8, val: u8, n: usize) {
    #[cfg(target_arch = "x86_64")]   { unsafe { x86::fill_bytes(ptr, val, n);     return; } }
    #[cfg(target_arch = "aarch64")]  { unsafe { aarch64::fill_bytes(ptr, val, n); return; } }
    #[allow(unreachable_code)]        unsafe { fallback::fill_bytes(ptr, val, n); }
}

/// Returns the byte offset of the first occurrence of `val` in `[ptr, ptr+n)`,
/// or `None` if `val` does not appear.
///
/// Uses SIMD to scan 16 bytes at a time; falls back to scalar for the tail.
///
/// # Safety
///
/// * `ptr` must be valid for `n` bytes of reads.
#[inline]
pub unsafe fn find_byte(ptr: *const u8, val: u8, n: usize) -> Option<usize> {
    #[cfg(target_arch = "x86_64")]   { unsafe { return x86::find_byte(ptr, val, n);     } }
    #[cfg(target_arch = "aarch64")]  { unsafe { return aarch64::find_byte(ptr, val, n); } }
    #[allow(unreachable_code)]        unsafe { fallback::find_byte(ptr, val, n) }
}

/// Copies `n` bytes from `src` to `dst` using `copy_nonoverlapping`.
///
/// Equivalent to `memcpy(dst, src, n)`.
///
/// # Safety
///
/// * `dst` must be valid for `n` bytes of writes.
/// * `src` must be valid for `n` bytes of reads.
/// * The source and destination regions must **not** overlap; use
///   `core::ptr::copy` if overlap is possible.
#[inline]
pub unsafe fn copy_bytes(dst: *mut u8, src: *const u8, n: usize) {
    #[cfg(target_arch = "x86_64")]   { unsafe { x86::copy_bytes(dst, src, n);     return; } }
    #[cfg(target_arch = "aarch64")]  { unsafe { aarch64::copy_bytes(dst, src, n); return; } }
    #[allow(unreachable_code)]        unsafe { fallback::copy_bytes(dst, src, n); }
}