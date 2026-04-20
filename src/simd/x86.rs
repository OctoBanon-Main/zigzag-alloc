use core::arch::x86_64::*;
use super::{BitMask, CTRL_EMPTY, CTRL_DELETED};

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
    pub fn match_deleted(&self) -> BitMask {
        self.match_byte(CTRL_DELETED)
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    pub fn match_empty_or_deleted(&self) -> BitMask {
        let mask = _mm_movemask_epi8(self.0);
        BitMask(mask as u32)
    }

    #[inline]
    pub fn all_full(&self) -> bool {
        unsafe { self.match_empty_or_deleted().is_empty() }
    }
}

pub unsafe fn and_words(dst: *mut usize, src: *const usize, n: usize) {
    #[cfg(target_feature = "avx2")]
    { unsafe { and_words_avx2(dst, src, n) }; return; }
    #[allow(unreachable_code)]
    { unsafe { and_words_sse2(dst, src, n) }; }
}

#[target_feature(enable = "sse2")]
unsafe fn and_words_sse2(dst: *mut usize, src: *const usize, n: usize) {
    let d = dst as *mut __m128i;
    let s = src as *const __m128i;
    let chunks = n / 2;
    for i in 0..chunks {
        unsafe {
            _mm_storeu_si128(d.add(i),
                _mm_and_si128(_mm_loadu_si128(d.add(i)), _mm_loadu_si128(s.add(i))))
        }
    }
    for i in (chunks * 2)..n {
        unsafe { *dst.add(i) &= *src.add(i); }
    }
}

#[cfg(target_feature = "avx2")]
#[target_feature(enable = "avx2")]
unsafe fn and_words_avx2(dst: *mut usize, src: *const usize, n: usize) {
    let d = dst as *mut __m256i;
    let s = src as *const __m256i;
    let chunks = n / 4;
    for i in 0..chunks {
        unsafe {
            _mm256_storeu_si256(d.add(i),
                _mm256_and_si256(_mm256_loadu_si256(d.add(i)), _mm256_loadu_si256(s.add(i))))
        };
    }
    let rem_start = chunks * 4;
    unsafe { and_words_sse2(dst.add(rem_start), src.add(rem_start), n - rem_start) };
}

pub unsafe fn or_words(dst: *mut usize, src: *const usize, n: usize) {
    #[cfg(target_feature = "avx2")]
    { unsafe { or_words_avx2(dst, src, n) }; return; }
    #[allow(unreachable_code)]
    { unsafe { or_words_sse2(dst, src, n) }; }
}

#[target_feature(enable = "sse2")]
unsafe fn or_words_sse2(dst: *mut usize, src: *const usize, n: usize) {
    let d = dst as *mut __m128i;
    let s = src as *const __m128i;
    let chunks = n / 2;
    for i in 0..chunks {
        unsafe {
            _mm_storeu_si128(d.add(i),
                _mm_or_si128(_mm_loadu_si128(d.add(i)), _mm_loadu_si128(s.add(i))))
        }
    }
    for i in (chunks * 2)..n { 
        unsafe { *dst.add(i) |= *src.add(i); } 
    }
}

#[cfg(target_feature = "avx2")]
#[target_feature(enable = "avx2")]
unsafe fn or_words_avx2(dst: *mut usize, src: *const usize, n: usize) {
    let d = dst as *mut __m256i;
    let s = src as *const __m256i;
    let chunks = n / 4;
    for i in 0..chunks {
        unsafe {
            _mm256_storeu_si256(d.add(i),
                _mm256_or_si256(_mm256_loadu_si256(d.add(i)), _mm256_loadu_si256(s.add(i))))
        };
    }
    let r = chunks * 4;
    unsafe { or_words_sse2(dst.add(r), src.add(r), n - r) };
}

pub unsafe fn xor_words(dst: *mut usize, src: *const usize, n: usize) {
    #[cfg(target_feature = "avx2")]
    { unsafe { xor_words_avx2(dst, src, n) }; return; }
    #[allow(unreachable_code)]
    { unsafe { xor_words_sse2(dst, src, n) }; }
}

#[target_feature(enable = "sse2")]
unsafe fn xor_words_sse2(dst: *mut usize, src: *const usize, n: usize) {
    let d = dst as *mut __m128i;
    let s = src as *const __m128i;
    let chunks = n / 2;
    for i in 0..chunks {
        unsafe {
            _mm_storeu_si128(d.add(i),
                _mm_xor_si128(_mm_loadu_si128(d.add(i)), _mm_loadu_si128(s.add(i))));
        }
    }
    for i in (chunks * 2)..n { 
        unsafe { *dst.add(i) ^= *src.add(i); } 
    }
}

#[cfg(target_feature = "avx2")]
#[target_feature(enable = "avx2")]
unsafe fn xor_words_avx2(dst: *mut usize, src: *const usize, n: usize) {
    let d = dst as *mut __m256i;
    let s = src as *const __m256i;
    let chunks = n / 4;
    for i in 0..chunks {
        unsafe {
            _mm256_storeu_si256(d.add(i),
                _mm256_xor_si256(_mm256_loadu_si256(d.add(i)), _mm256_loadu_si256(s.add(i))))
        };
    }
    let r = chunks * 4;
    unsafe { xor_words_sse2(dst.add(r), src.add(r), n - r) };
}

pub unsafe fn not_words(dst: *mut usize, n: usize) {
    #[cfg(target_feature = "avx2")]
    { unsafe { not_words_avx2(dst, n) }; return; }
    #[allow(unreachable_code)]
    { unsafe { not_words_sse2(dst, n) }; }
}

#[target_feature(enable = "sse2")]
unsafe fn not_words_sse2(dst: *mut usize, n: usize) {
    let d = dst as *mut __m128i;
    let chunks = n / 2;
    unsafe {
        let ones = _mm_set1_epi8(-1i8);
        for i in 0..chunks {
            _mm_storeu_si128(d.add(i), _mm_xor_si128(_mm_loadu_si128(d.add(i)), ones));
        }
        for i in (chunks * 2)..n { *dst.add(i) = !*dst.add(i); }
    }
}

#[cfg(target_feature = "avx2")]
#[target_feature(enable = "avx2")]
unsafe fn not_words_avx2(dst: *mut usize, n: usize) {
    let d = dst as *mut __m256i;
    let chunks = n / 4;
    unsafe {
        let ones = _mm256_set1_epi8(-1i8);
        for i in 0..chunks {
            _mm256_storeu_si256(d.add(i), _mm256_xor_si256(_mm256_loadu_si256(d.add(i)), ones));
        }
        let r = chunks * 4;
        not_words_sse2(dst.add(r), n - r);
    }
}

pub unsafe fn popcount_words(ptr: *const usize, n: usize) -> usize {
    let mut acc = 0usize;
    unsafe { for i in 0..n { acc += (*ptr.add(i)).count_ones() as usize; } };
    acc
}

pub unsafe fn fill_bytes(ptr: *mut u8, val: u8, n: usize) {
    #[cfg(target_feature = "avx2")]
    { unsafe { fill_bytes_avx2(ptr, val, n) }; return; }
    #[allow(unreachable_code)]
    { unsafe { fill_bytes_sse2(ptr, val, n) }; }
}

#[target_feature(enable = "sse2")]
unsafe fn fill_bytes_sse2(ptr: *mut u8, val: u8, n: usize) {
    let d = ptr as *mut __m128i;
    let chunks = n / 16;
    unsafe {
        let v = _mm_set1_epi8(val as i8);
        for i in 0..chunks { _mm_storeu_si128(d.add(i), v); }
        for i in (chunks * 16)..n { *ptr.add(i) = val; }
    };
}

#[cfg(target_feature = "avx2")]
#[target_feature(enable = "avx2")]
unsafe fn fill_bytes_avx2(ptr: *mut u8, val: u8, n: usize) {
    let d = ptr as *mut __m256i;
    let chunks = n / 32;
    unsafe {
        let v = _mm256_set1_epi8(val as i8);
        for i in 0..chunks { _mm256_storeu_si256(d.add(i), v); }
        let r = chunks * 32;
        fill_bytes_sse2(ptr.add(r), val, n - r);
    }
}

pub unsafe fn find_byte(ptr: *const u8, val: u8, n: usize) -> Option<usize> {
    #[cfg(target_feature = "avx2")]
    if n >= 32 { return unsafe { find_byte_avx2(ptr, val, n) }; }
    unsafe { find_byte_sse2(ptr, val, n) }
}

#[target_feature(enable = "sse2")]
unsafe fn find_byte_sse2(ptr: *const u8, val: u8, n: usize) -> Option<usize> {
    let s = ptr as *const __m128i;
    let chunks = n / 16;
    unsafe {
        let needle = _mm_set1_epi8(val as i8);
        for i in 0..chunks {
            let block = _mm_loadu_si128(s.add(i));
            let mask  = _mm_movemask_epi8(_mm_cmpeq_epi8(block, needle)) as u32;
            if mask != 0 {
                return Some(i * 16 + mask.trailing_zeros() as usize);
            }
        }
        for i in (chunks * 16)..n {
            if *ptr.add(i) == val { return Some(i); }
        }
    }
    None
}

#[cfg(target_feature = "avx2")]
#[target_feature(enable = "avx2")]
unsafe fn find_byte_avx2(ptr: *const u8, val: u8, n: usize) -> Option<usize> {
    let s = ptr as *const __m256i;
    let chunks = n / 32;
    unsafe {
        let needle = _mm256_set1_epi8(val as i8);
        for i in 0..chunks {
            let block = _mm256_loadu_si256(s.add(i));
            let mask  = _mm256_movemask_epi8(_mm256_cmpeq_epi8(block, needle)) as u32;
            if mask != 0 {
                return Some(i * 32 + mask.trailing_zeros() as usize);
            }
        }
        let r = chunks * 32;
        find_byte_sse2(ptr.add(r), val, n - r)
    }
}

pub unsafe fn copy_bytes(dst: *mut u8, src: *const u8, n: usize) {
    unsafe { core::ptr::copy_nonoverlapping(src, dst, n) };
}