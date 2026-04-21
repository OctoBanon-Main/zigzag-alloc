//! High-performance collections backed by explicit allocators.
//!
//! Unlike [`std::collections`], every type in this module:
//!
//! 1. **Never uses a global allocator.**  Each collection accepts an
//!    [`Allocator`] reference at construction time and routes all allocations
//!    through it.
//! 2. **Provides SIMD-accelerated operations** (fill, search, copy) where
//!    applicable via the internal `simd` module.
//! 3. **Decouples data from logic** via *context traits*: hashing and ordering
//!    are provided through [`HashContext`] and [`OrdContext`] rather than via
//!    `Hash` / `Ord` bounds.  This allows domain-specific algorithms without
//!    wrapper types.
//!
//! ## Collection Overview
//!
//! | Type | Description |
//! |------|-------------|
//! | [`ExVec<T>`] | Growable contiguous array |
//! | [`ExBox<T>`] | Single heap-allocated value |
//! | [`ExString`] | Growable UTF-8 string |
//! | [`ExHashMap<K, V, C>`] | Swiss-table open-addressing hash map |
//! | [`ExPriorityQueue<T, C>`] | Binary-heap priority queue |
//! | [`ExBoundedArray<T, N>`] | Fixed-capacity stack-allocated array |
//!
//! [`Allocator`]: crate::alloc::allocator::Allocator

pub mod vec;
pub mod boxed;
pub mod string;
pub mod hash_map;
pub mod priority_queue;
pub mod bounded_array;

pub use vec::ExVec;
pub use boxed::ExBox;
pub use string::ExString;
pub use hash_map::ExHashMap;
pub use priority_queue::ExPriorityQueue;
pub use bounded_array::ExBoundedArray;

/// Provides hashing and equality for keys of type `K`.
///
/// Implementors replace the standard `Hash` + `Eq` trait combination, allowing
/// custom hash functions without needing newtype wrappers.
///
/// # Contract
///
/// * If `eq(a, b)` returns `true`, then `hash(a) == hash(b)` must also hold.
/// * `eq` must be an equivalence relation (reflexive, symmetric, transitive).
pub trait HashContext<K> {
    /// Returns the 64-bit hash of `key`.
    fn hash(&self, key: &K) -> u64;
    /// Returns `true` if `a` and `b` should be considered equal keys.
    fn eq(&self, a: &K, b: &K) -> bool;
}

/// Provides a total ordering for elements of type `T`.
///
/// Implementors replace the standard `Ord` / `PartialOrd` traits, enabling
/// context-dependent orderings (e.g. reverse order, key-extracted order)
/// without newtype wrappers.
///
/// # Contract
///
/// `less` must define a strict weak ordering:
/// * Irreflexivity: `less(a, a)` is `false`.
/// * Asymmetry: if `less(a, b)` then `!less(b, a)`.
/// * Transitivity: if `less(a, b)` and `less(b, c)` then `less(a, c)`.
pub trait OrdContext<T> {
    /// Returns `true` if `a` should be ordered *before* `b`.
    fn less(&self, a: &T, b: &T) -> bool;
}

/// [`HashContext<u64>`] using FNV-1a hashing.
pub struct U64HashCtx;

impl HashContext<u64> for U64HashCtx {
    /// Hashes `k` using the FNV-1a algorithm over its little-endian bytes.
    #[inline]
    fn hash(&self, k: &u64) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325; // FNV offset basis
        for b in k.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0100_0000_01b3); // FNV prime
        }
        h
    }
    #[inline]
    fn eq(&self, a: &u64, b: &u64) -> bool { a == b }
}

/// [`HashContext<usize>`] that delegates to [`U64HashCtx`].
pub struct UsizeHashCtx;

impl HashContext<usize> for UsizeHashCtx {
    /// Hashes `k` by widening it to `u64` and delegating to [`U64HashCtx`].
    #[inline]
    fn hash(&self, k: &usize) -> u64 {
        U64HashCtx.hash(&(*k as u64))
    }
    #[inline]
    fn eq(&self, a: &usize, b: &usize) -> bool { a == b }
}

/// [`OrdContext<usize>`] that orders elements from smallest to largest
/// (min-heap).
pub struct UsizeMinCtx;

impl OrdContext<usize> for UsizeMinCtx {
    /// Returns `true` if `a < b`.
    #[inline]
    fn less(&self, a: &usize, b: &usize) -> bool { a < b }
}