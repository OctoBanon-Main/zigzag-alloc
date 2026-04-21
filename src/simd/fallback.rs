// Приватный модуль платформенной реализации.
// dead_code отключён: элементы используются через cfg-гейтированный re-export.
#![allow(dead_code)]

//! Scalar fallback: Group эмулируется через два `u64` (SWAR),
//! slice-операции — пословно.

use super::{BitMask, CTRL_EMPTY};

// ── Group (2 × u64 = 16 байт, SWAR) ──────────────────────────────────────────

pub struct Group(u64, u64);

impl Group {
    #[inline]
    pub unsafe fn load(ptr: *const u8) -> Self {
        Group(
            unsafe { (ptr         as *const u64).read_unaligned() },
            unsafe { (ptr.add(8)  as *const u64).read_unaligned() },
        )
    }

    #[inline]
    pub fn match_byte(&self, byte: u8) -> BitMask {
        BitMask(swar_eq8(self.0, byte) | (swar_eq8(self.1, byte) << 8))
    }

    #[inline]
    pub fn match_empty(&self) -> BitMask {
        self.match_byte(CTRL_EMPTY)
    }

    /// MSB каждого байта равен 1 для EMPTY/DELETED.
    #[inline]
    pub fn match_empty_or_deleted(&self) -> BitMask {
        BitMask(msb_mask8(self.0) | (msb_mask8(self.1) << 8))
    }

    #[inline]
    pub fn all_full(&self) -> bool {
        self.match_empty_or_deleted().is_empty()
    }
}

/// SWAR: маска байт в `word`, равных `byte`. Возвращает 8-битную маску позиций.
#[inline]
fn swar_eq8(word: u64, byte: u8) -> u32 {
    let broadcast = u64::from_le_bytes([byte; 8]);
    let xored = word ^ broadcast;
    // Нулевой байт в xored = совпадение.
    // Хак для определения нулевых байт: (v - 0x01) & ~v & 0x80
    let lo    = xored.wrapping_sub(0x0101_0101_0101_0101);
    let hi    = !xored;
    let zeros = (lo & hi) & 0x8080_8080_8080_8080;
    compress_msb(zeros)
}

/// Маска байт с MSB=1 в `word` → 8-битная маска позиций.
#[inline]
fn msb_mask8(word: u64) -> u32 {
    compress_msb(word & 0x8080_8080_8080_8080)
}

/// Собрать по 1 биту из MSB каждого байта u64 → u8 в u32.
#[inline]
fn compress_msb(v: u64) -> u32 {
    let mut r = 0u32;
    for i in 0..8 {
        if v & (0x80u64 << (i * 8)) != 0 { r |= 1 << i; }
    }
    r
}

// ── Bulk slice operations (scalar) ────────────────────────────────────────────

pub unsafe fn and_words(dst: *mut usize, src: *const usize, n: usize) {
    for i in 0..n { unsafe { *dst.add(i) &= *src.add(i) }; }
}

pub unsafe fn or_words(dst: *mut usize, src: *const usize, n: usize) {
    for i in 0..n { unsafe { *dst.add(i) |= *src.add(i) }; }
}

pub unsafe fn xor_words(dst: *mut usize, src: *const usize, n: usize) {
    for i in 0..n { unsafe { *dst.add(i) ^= *src.add(i) }; }
}

pub unsafe fn not_words(dst: *mut usize, n: usize) {
    for i in 0..n { unsafe { *dst.add(i) = !*dst.add(i) }; }
}

pub unsafe fn popcount_words(ptr: *const usize, n: usize) -> usize {
    let mut acc = 0usize;
    for i in 0..n { acc += unsafe { (*ptr.add(i)).count_ones() as usize }; }
    acc
}

pub unsafe fn fill_bytes(ptr: *mut u8, val: u8, n: usize) {
    for i in 0..n { unsafe { *ptr.add(i) = val }; }
}

pub unsafe fn find_byte(ptr: *const u8, val: u8, n: usize) -> Option<usize> {
    for i in 0..n { if unsafe { *ptr.add(i) } == val { return Some(i); } }
    None
}

pub unsafe fn copy_bytes(dst: *mut u8, src: *const u8, n: usize) {
    unsafe { core::ptr::copy_nonoverlapping(src, dst, n) };
}