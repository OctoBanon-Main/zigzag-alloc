#![allow(dead_code, unused_imports)]

use core::arch::x86_64::*;
use super::{BitMask, CTRL_EMPTY, GROUP_WIDTH};

pub struct Group(__m128i);

impl Group {
    #[inline]
    #[target_feature(enable = "sse2")]
    pub unsafe fn load(ptr: *const u8) -> Self {
        Group(unsafe { _mm_loadu_si128(ptr as *const __m128i) })
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    pub fn match_byte(&self, byte: u8) -> BitMask {
        let mask = _mm_movemask_epi8(_mm_cmpeq_epi8(self.0, _mm_set1_epi8(byte as i8)));
        BitMask(mask as u32)
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    pub fn match_empty(&self) -> BitMask {
        self.match_byte(CTRL_EMPTY)
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    pub fn match_empty_or_deleted(&self) -> BitMask {
        BitMask(_mm_movemask_epi8(self.0) as u32)
    }

    #[inline]
    pub fn all_full(&self) -> bool {
        unsafe { self.match_empty_or_deleted().is_empty() }
    }
}

pub unsafe fn and_words(dst: *mut usize, src: *const usize, n: usize) {
    #[cfg(target_feature = "avx2")] { unsafe { and_words_avx2(dst, src, n); return; } }
    #[allow(unreachable_code)]        unsafe { and_words_sse2(dst, src, n); }
}
#[target_feature(enable = "sse2")]
unsafe fn and_words_sse2(dst: *mut usize, src: *const usize, n: usize) {
    let (d, s) = (dst as *mut __m128i, src as *const __m128i);
    let c = n / 2;
    unsafe {
        for i in 0..c {
            _mm_storeu_si128(d.add(i), _mm_and_si128(_mm_loadu_si128(d.add(i)), _mm_loadu_si128(s.add(i))));
        }
        for i in (c * 2)..n { *dst.add(i) &= *src.add(i) }
    };
}
#[cfg(target_feature = "avx2")]
#[target_feature(enable = "avx2")]
unsafe fn and_words_avx2(dst: *mut usize, src: *const usize, n: usize) {
    let (d, s) = (dst as *mut __m256i, src as *const __m256i);
    let c = n / 4;
    unsafe {
        for i in 0..c {
            _mm256_storeu_si256(d.add(i), _mm256_and_si256(_mm256_loadu_si256(d.add(i)), _mm256_loadu_si256(s.add(i))))
        }
        and_words_sse2(dst.add(c * 4), src.add(c * 4), n - c * 4)
    };
}

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
            _mm_storeu_si128(d.add(i), _mm_or_si128(_mm_loadu_si128(d.add(i)), _mm_loadu_si128(s.add(i))));
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
            _mm256_storeu_si256(d.add(i), _mm256_or_si256(_mm256_loadu_si256(d.add(i)), _mm256_loadu_si256(s.add(i))));
        }
        or_words_sse2(dst.add(c * 4), src.add(c * 4), n - c * 4)
    };
}

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
            _mm_storeu_si128(d.add(i), _mm_xor_si128(_mm_loadu_si128(d.add(i)), _mm_loadu_si128(s.add(i))));
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
            _mm256_storeu_si256(d.add(i), _mm256_xor_si256(_mm256_loadu_si256(d.add(i)), _mm256_loadu_si256(s.add(i))));
        }
        xor_words_sse2(dst.add(c * 4), src.add(c * 4), n - c * 4)
    };
}

pub unsafe fn not_words(dst: *mut usize, n: usize) {
    #[cfg(target_feature = "avx2")] { unsafe { not_words_avx2(dst, n); return; } }
    #[allow(unreachable_code)]        unsafe { not_words_sse2(dst, n); }
}
#[target_feature(enable = "sse2")]
unsafe fn not_words_sse2(dst: *mut usize, n: usize) {
    let d    = dst as *mut __m128i;
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

pub unsafe fn popcount_words(ptr: *const usize, n: usize) -> usize {
    let mut acc = 0usize;
    for i in 0..n { acc += unsafe { (*ptr.add(i)).count_ones() as usize }; }
    acc
}

pub unsafe fn fill_bytes(ptr: *mut u8, val: u8, n: usize) {
    #[cfg(target_feature = "avx2")] { unsafe { fill_bytes_avx2(ptr, val, n); return; } }
    #[allow(unreachable_code)]        unsafe { fill_bytes_sse2(ptr, val, n); }
}
#[target_feature(enable = "sse2")]
unsafe fn fill_bytes_sse2(ptr: *mut u8, val: u8, n: usize) {
    let v =  _mm_set1_epi8(val as i8);
    let d = ptr as *mut __m128i;
    let c = n / 16;
    for i in 0..c { unsafe { _mm_storeu_si128(d.add(i), v) }; }
    for i in (c * 16)..n { unsafe { *ptr.add(i) = val }; }
}
#[cfg(target_feature = "avx2")]
#[target_feature(enable = "avx2")]
unsafe fn fill_bytes_avx2(ptr: *mut u8, val: u8, n: usize) {
    let v =  _mm256_set1_epi8(val as i8);
    let d = ptr as *mut __m256i;
    let c = n / 32;
    for i in 0..c { unsafe { _mm256_storeu_si256(d.add(i), v) }; }
    unsafe { fill_bytes_sse2(ptr.add(c * 32), val, n - c * 32); }
}

pub unsafe fn find_byte(ptr: *const u8, val: u8, n: usize) -> Option<usize> {
    #[cfg(target_feature = "avx2")] if n >= 32 { unsafe { return find_byte_avx2(ptr, val, n); } }
    unsafe { find_byte_sse2(ptr, val, n) }
}
#[target_feature(enable = "sse2")]
unsafe fn find_byte_sse2(ptr: *const u8, val: u8, n: usize) -> Option<usize> {
    let needle = _mm_set1_epi8(val as i8);
    let s = ptr as *const __m128i;
    let c = n / 16;
    for i in 0..c {
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
        let mask = unsafe { _mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_loadu_si256(s.add(i)), needle)) as u32 };
        if mask != 0 { return Some(i * 32 + mask.trailing_zeros() as usize); }
    }
    unsafe { find_byte_sse2(ptr.add(c * 32), val, n - c * 32) }
}

pub unsafe fn copy_bytes(dst: *mut u8, src: *const u8, n: usize) {
    unsafe { core::ptr::copy_nonoverlapping(src, dst, n) };
}