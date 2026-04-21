//! x86_64 SIMD backend using SSE2 (and optionally AVX2).
//!
//! All `Group` methods and byte-level operations use SSE2 as the baseline.
//! When the crate is compiled with `-C target-feature=+avx2`, the 256-bit
//! AVX2 paths are also compiled in and selected at runtime via `cfg` guards.
//!
//! # Safety
//!
//! Every public function is `unsafe` because:
//! * Raw pointer arithmetic is performed without bounds checks.
//! * SSE2 / AVX2 intrinsics may execute illegal instructions on CPUs that do
//!   not support them (though Rust's `target_feature` attributes prevent this
//!   when used correctly).

#![allow(dead_code, unused_imports)]

use core::arch::x86_64::*;
use super::{BitMask, CTRL_EMPTY, GROUP_WIDTH};

/// A 16-byte SIMD group of hash-map control bytes, backed by `__m128i`.
///
/// Used by [`ExHashMap`](crate::collections::ExHashMap) to compare 16 control
/// bytes in parallel against a target tag or the empty sentinel.
pub struct Group(__m128i);

impl Group {
    /// Loads 16 unaligned bytes from `ptr` into a `Group`.
    ///
    /// # Safety
    ///
    /// `ptr` must be valid for 16 bytes of reads (need not be aligned).
    #[inline]
    #[target_feature(enable = "sse2")]
    pub unsafe fn load(ptr: *const u8) -> Self {
        // SAFETY: Caller guarantees `ptr` is valid for 16 bytes.
        Group(unsafe { _mm_loadu_si128(ptr as *const __m128i) })
    }

    /// Returns a [`BitMask`] with bit `i` set wherever byte `i` equals `byte`.
    #[inline]
    #[target_feature(enable = "sse2")]
    pub fn match_byte(&self, byte: u8) -> BitMask {
        // SAFETY: SSE2 intrinsics operate on the already-loaded register value;
        // no memory access.
        let mask = _mm_movemask_epi8(_mm_cmpeq_epi8(self.0, _mm_set1_epi8(byte as i8)));
        BitMask(mask as u32)
    }

    /// Returns a [`BitMask`] with bit `i` set wherever byte `i` is [`CTRL_EMPTY`].
    #[inline]
    #[target_feature(enable = "sse2")]
    pub fn match_empty(&self) -> BitMask {
        self.match_byte(CTRL_EMPTY)
    }

    /// Returns a [`BitMask`] with bit `i` set wherever byte `i` has its high
    /// bit set (i.e. is `CTRL_EMPTY` or `CTRL_DELETED`).
    #[inline]
    #[target_feature(enable = "sse2")]
    pub fn match_empty_or_deleted(&self) -> BitMask {
        // `_mm_movemask_epi8` collects the MSB of each byte into a 16-bit mask.
        BitMask(_mm_movemask_epi8(self.0) as u32)
    }

    /// Returns `true` if every byte in the group is an occupied slot (MSB == 0).
    #[inline]
    pub fn all_full(&self) -> bool {
        // SAFETY: calls SSE2 intrinsic on an already-loaded register.
        unsafe { self.match_empty_or_deleted().is_empty() }
    }
}

/// Computes `dst[i] &= src[i]` for `n` `usize` words.
///
/// Dispatches to AVX2 (256-bit) if available, otherwise SSE2 (128-bit).
///
/// # Safety
///
/// * `dst` must be valid for `n * size_of::<usize>()` bytes of reads and writes.
/// * `src` must be valid for `n * size_of::<usize>()` bytes of reads.
/// * Regions must not overlap.
pub unsafe fn and_words(dst: *mut usize, src: *const usize, n: usize) {
    #[cfg(target_feature = "avx2")] { unsafe { and_words_avx2(dst, src, n); return; } }
    #[allow(unreachable_code)]        unsafe { and_words_sse2(dst, src, n); }
}

/// SSE2 implementation of `and_words` (128-bit vectors).
///
/// # Safety
///
/// Same as [`and_words`].  Requires SSE2 support (guaranteed on all x86_64 CPUs).
#[target_feature(enable = "sse2")]
unsafe fn and_words_sse2(dst: *mut usize, src: *const usize, n: usize) {
    let (d, s) = (dst as *mut __m128i, src as *const __m128i);
    let c = n / 2;
    unsafe {
        for i in 0..c {
            _mm_storeu_si128(d.add(i),
                _mm_and_si128(_mm_loadu_si128(d.add(i)), _mm_loadu_si128(s.add(i))));
        }
        for i in (c * 2)..n { *dst.add(i) &= *src.add(i) }
    };
}

/// AVX2 implementation of `and_words` (256-bit vectors).
///
/// # Safety
///
/// Same as [`and_words`].  Only compiled and called when `target_feature = "avx2"`.
#[cfg(target_feature = "avx2")]
#[target_feature(enable = "avx2")]
unsafe fn and_words_avx2(dst: *mut usize, src: *const usize, n: usize) {
    let (d, s) = (dst as *mut __m256i, src as *const __m256i);
    let c = n / 4;
    unsafe {
        for i in 0..c {
            _mm256_storeu_si256(d.add(i),
                _mm256_and_si256(_mm256_loadu_si256(d.add(i)), _mm256_loadu_si256(s.add(i))))
        }
        // Handle the remaining < 4 words with the SSE2 path.
        and_words_sse2(dst.add(c * 4), src.add(c * 4), n - c * 4)
    };
}

/// Computes `dst[i] |= src[i]` for `n` `usize` words.
///
/// # Safety
///
/// Same as [`and_words`].
pub unsafe fn or_words(dst: *mut usize, src: *const usize, n: usize) {
    #[cfg(target_feature = "avx2")] { unsafe { or_words_avx2(dst, src, n); return; } }
    #[allow(unreachable_code)]        unsafe { or_words_sse2(dst, src, n); }
}
#[target_feature(enable = "sse2")]
unsafe fn or_words_sse2(dst: *mut usize, src: *const usize, n: usize) {
    let (d, s) = (dst as *mut __m128i, src as *const __m128i);
    let c = n / 2;
    unsafe {
        for i in 0..c {
            _mm_storeu_si128(d.add(i),
                _mm_or_si128(_mm_loadu_si128(d.add(i)), _mm_loadu_si128(s.add(i))));
        }
        for i in (c * 2)..n { *dst.add(i) |= *src.add(i) }
    };
}
#[cfg(target_feature = "avx2")]
#[target_feature(enable = "avx2")]
unsafe fn or_words_avx2(dst: *mut usize, src: *const usize, n: usize) {
    let (d, s) = (dst as *mut __m256i, src as *const __m256i);
    let c = n / 4;
    unsafe {
        for i in 0..c {
            _mm256_storeu_si256(d.add(i),
                _mm256_or_si256(_mm256_loadu_si256(d.add(i)), _mm256_loadu_si256(s.add(i))));
        }
        or_words_sse2(dst.add(c * 4), src.add(c * 4), n - c * 4)
    };
}

/// Computes `dst[i] ^= src[i]` for `n` `usize` words.
///
/// # Safety
///
/// Same as [`and_words`].
pub unsafe fn xor_words(dst: *mut usize, src: *const usize, n: usize) {
    #[cfg(target_feature = "avx2")] { unsafe { xor_words_avx2(dst, src, n); return; } }
    #[allow(unreachable_code)]        unsafe { xor_words_sse2(dst, src, n); }
}
#[target_feature(enable = "sse2")]
unsafe fn xor_words_sse2(dst: *mut usize, src: *const usize, n: usize) {
    let (d, s) = (dst as *mut __m128i, src as *const __m128i);
    let c = n / 2;
    unsafe {
        for i in 0..c {
            _mm_storeu_si128(d.add(i),
                _mm_xor_si128(_mm_loadu_si128(d.add(i)), _mm_loadu_si128(s.add(i))));
        }
        for i in (c * 2)..n { *dst.add(i) ^= *src.add(i) }
    };
}
#[cfg(target_feature = "avx2")]
#[target_feature(enable = "avx2")]
unsafe fn xor_words_avx2(dst: *mut usize, src: *const usize, n: usize) {
    let (d, s) = (dst as *mut __m256i, src as *const __m256i);
    let c = n / 4;
    unsafe {
        for i in 0..c {
            _mm256_storeu_si256(d.add(i),
                _mm256_xor_si256(_mm256_loadu_si256(d.add(i)), _mm256_loadu_si256(s.add(i))));
        }
        xor_words_sse2(dst.add(c * 4), src.add(c * 4), n - c * 4)
    };
}

/// Computes `dst[i] = !dst[i]` for `n` `usize` words.
///
/// # Safety
///
/// * `dst` must be valid for `n * size_of::<usize>()` bytes of reads and writes.
pub unsafe fn not_words(dst: *mut usize, n: usize) {
    #[cfg(target_feature = "avx2")] { unsafe { not_words_avx2(dst, n); return; } }
    #[allow(unreachable_code)]        unsafe { not_words_sse2(dst, n); }
}
#[target_feature(enable = "sse2")]
unsafe fn not_words_sse2(dst: *mut usize, n: usize) {
    let d    = dst as *mut __m128i;
    // XOR with all-ones is equivalent to bitwise NOT.
    let ones = _mm_set1_epi8(-1i8);
    let c    = n / 2;
    unsafe {
        for i in 0..c {
            _mm_storeu_si128(d.add(i), _mm_xor_si128(_mm_loadu_si128(d.add(i)), ones));
        }
        for i in (c * 2)..n { *dst.add(i) = !*dst.add(i) }
    }
}
#[cfg(target_feature = "avx2")]
#[target_feature(enable = "avx2")]
unsafe fn not_words_avx2(dst: *mut usize, n: usize) {
    let d    = dst as *mut __m256i;
    let ones = _mm256_set1_epi8(-1i8);
    let c    = n / 4;
    unsafe {
        for i in 0..c {
            _mm256_storeu_si256(d.add(i), _mm256_xor_si256(_mm256_loadu_si256(d.add(i)), ones));
        }
        not_words_sse2(dst.add(c * 4), n - c * 4)
    };
}

/// Returns the total number of set bits across `n` `usize` words.
///
/// Uses the scalar `count_ones` because SSE2 does not have a fast popcount;
/// AVX-512 VPOPCNTDQ is not targeted here.
///
/// # Safety
///
/// * `ptr` must be valid for `n * size_of::<usize>()` bytes of reads.
pub unsafe fn popcount_words(ptr: *const usize, n: usize) -> usize {
    let mut acc = 0usize;
    for i in 0..n { acc += unsafe { (*ptr.add(i)).count_ones() as usize }; }
    acc
}

/// Sets `n` bytes starting at `ptr` to `val` using SSE2 / AVX2 stores.
///
/// # Safety
///
/// * `ptr` must be valid for `n` bytes of writes.
pub unsafe fn fill_bytes(ptr: *mut u8, val: u8, n: usize) {
    #[cfg(target_feature = "avx2")] { unsafe { fill_bytes_avx2(ptr, val, n); return; } }
    #[allow(unreachable_code)]        unsafe { fill_bytes_sse2(ptr, val, n); }
}
#[target_feature(enable = "sse2")]
unsafe fn fill_bytes_sse2(ptr: *mut u8, val: u8, n: usize) {
    let v = _mm_set1_epi8(val as i8);
    let d = ptr as *mut __m128i;
    let c = n / 16;
    for i in 0..c { unsafe { _mm_storeu_si128(d.add(i), v) }; }
    for i in (c * 16)..n { unsafe { *ptr.add(i) = val }; }
}
#[cfg(target_feature = "avx2")]
#[target_feature(enable = "avx2")]
unsafe fn fill_bytes_avx2(ptr: *mut u8, val: u8, n: usize) {
    let v = _mm256_set1_epi8(val as i8);
    let d = ptr as *mut __m256i;
    let c = n / 32;
    for i in 0..c { unsafe { _mm256_storeu_si256(d.add(i), v) }; }
    // Handle the remaining < 32 bytes with the SSE2 path.
    unsafe { fill_bytes_sse2(ptr.add(c * 32), val, n - c * 32); }
}

/// Returns the offset of the first byte equal to `val` in `[ptr, ptr+n)`.
///
/// Uses AVX2 (32-byte scan) when available and `n >= 32`, otherwise SSE2
/// (16-byte scan).  Scalar tail handles any leftover bytes.
///
/// # Safety
///
/// * `ptr` must be valid for `n` bytes of reads.
pub unsafe fn find_byte(ptr: *const u8, val: u8, n: usize) -> Option<usize> {
    #[cfg(target_feature = "avx2")]
    if n >= 32 { unsafe { return find_byte_avx2(ptr, val, n); } }
    unsafe { find_byte_sse2(ptr, val, n) }
}
#[target_feature(enable = "sse2")]
unsafe fn find_byte_sse2(ptr: *const u8, val: u8, n: usize) -> Option<usize> {
    let needle = _mm_set1_epi8(val as i8);
    let s = ptr as *const __m128i;
    let c = n / 16;
    for i in 0..c {
        // `_mm_movemask_epi8` packs the MSB of each byte; `_mm_cmpeq_epi8`
        // sets all bits of a byte when it matches the needle.
        let mask = unsafe { _mm_movemask_epi8(_mm_cmpeq_epi8(_mm_loadu_si128(s.add(i)), needle)) as u32 };
        if mask != 0 { return Some(i * 16 + mask.trailing_zeros() as usize); }
    }
    for i in (c * 16)..n { if unsafe { *ptr.add(i) } == val { return Some(i); } }
    None
}
#[cfg(target_feature = "avx2")]
#[target_feature(enable = "avx2")]
unsafe fn find_byte_avx2(ptr: *const u8, val: u8, n: usize) -> Option<usize> {
    let needle = _mm256_set1_epi8(val as i8);
    let s = ptr as *const __m256i;
    let c = n / 32;
    for i in 0..c {
        let mask = unsafe {
            _mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_loadu_si256(s.add(i)), needle)) as u32
        };
        if mask != 0 { return Some(i * 32 + mask.trailing_zeros() as usize); }
    }
    // Delegate the remaining < 32 bytes to the SSE2 path.
    unsafe { find_byte_sse2(ptr.add(c * 32), val, n - c * 32) }
}

/// Copies `n` bytes from `src` to `dst` using `copy_nonoverlapping`.
///
/// # Safety
///
/// * `dst` must be valid for `n` bytes of writes.
/// * `src` must be valid for `n` bytes of reads.
/// * The source and destination regions must **not** overlap.
pub unsafe fn copy_bytes(dst: *mut u8, src: *const u8, n: usize) {
    // SAFETY: Caller guarantees non-overlapping regions with sufficient size.
    unsafe { core::ptr::copy_nonoverlapping(src, dst, n) };
}