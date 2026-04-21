//! Swiss-table inspired open-addressing hash map.
//!
//! [`ExHashMap`] is a high-performance hash map that uses a separate
//! *control byte* array to enable parallel SIMD key probing — the same
//! technique used by Google's Abseil `flat_hash_map` and Rust's
//! `hashbrown`.
//!
//! ## Algorithm Overview
//!
//! Each slot has a one-byte *control byte* stored in a contiguous `ctrl`
//! array.  The control byte is either:
//!
//! * [`CTRL_EMPTY`] (`0x80`) — the slot is vacant.
//! * A 7-bit hash tag `h2(hash)` — the slot is occupied by a key whose
//!   upper-57-bit hash matches `h2`.
//!
//! Lookups load a 16-byte [`Group`] from `ctrl` and use SIMD to find all
//! slots whose tag matches the query in a single instruction, drastically
//! reducing branch mispredictions and cache misses compared to traditional
//! chaining or linear probing.
//!
//! ## Load Factor
//!
//! The table grows when `len * 8 >= cap * 7` (87.5 % load factor).
//!
//! ## Context Trait
//!
//! Hashing and equality are provided by the [`HashContext<K>`] trait rather
//! than being hard-coded via `Hash` / `Eq`.  This allows callers to choose
//! domain-specific hash functions without wrapper types.

use core::{alloc::Layout, marker::PhantomData, mem::MaybeUninit, ptr::NonNull};

use crate::alloc::allocator::Allocator;
use crate::simd::{self, Group, GROUP_WIDTH, CTRL_EMPTY};
use super::HashContext;

/// Returns the low bits of `hash` used to select the initial probe position.
#[inline]
fn h1(hash: u64) -> usize { hash as usize }

/// Returns the 7-bit control tag stored in the `ctrl` array for an occupied slot.
///
/// Uses bits `[63:57]` of the hash so that `h1` and `h2` cover different
/// parts of the hash value, reducing correlation.
#[inline]
fn h2(hash: u64) -> u8 { ((hash >> 57) as u8) & 0x7F }

/// A single key-value slot in the hash map.
///
/// Slots are `MaybeUninit` because the map controls initialisation via the
/// parallel `ctrl` array; a slot is initialised if and only if its
/// corresponding control byte is not [`CTRL_EMPTY`].
struct Slot<K, V> {
    key: MaybeUninit<K>,
    val: MaybeUninit<V>,
}

/// A high-performance open-addressing hash map with SIMD probing.
///
/// # Type Parameters
///
/// * `'a` — Lifetime of the allocator reference.
/// * `K`  — Key type.
/// * `V`  — Value type.
/// * `C`  — [`HashContext<K>`] providing hash and equality functions.
///
/// # Memory Layout
///
/// Two separate heap allocations:
/// 1. **`ctrl`** — `cap + GROUP_WIDTH` control bytes.  The extra `GROUP_WIDTH`
///    bytes at the end are mirror copies of the first `GROUP_WIDTH` control
///    bytes, enabling SIMD group loads at the table boundary without
///    out-of-bounds access.
/// 2. **`data`** — `cap` [`Slot<K, V>`] entries.
///
/// Both allocations are freed on drop.
pub struct ExHashMap<'a, K, V, C: HashContext<K>> {
    /// Pointer to the control-byte array (`cap + GROUP_WIDTH` bytes).
    ctrl:    NonNull<u8>,
    /// Pointer to the slot array (`cap` entries).
    data:    NonNull<Slot<K, V>>,
    /// Current table capacity (always a power of two when non-zero).
    cap:     usize,
    /// Number of occupied slots.
    len:     usize,
    /// Allocator reference used for all internal allocations.
    alloc:   &'a dyn Allocator,
    /// Hashing and equality context.
    ctx:     C,
    _marker: PhantomData<(K, V)>,
}

impl<'a, K, V, C: HashContext<K>> ExHashMap<'a, K, V, C> {
    /// Creates a new, empty map that will allocate through `alloc`.
    ///
    /// No memory is allocated until the first insertion.
    pub fn new(alloc: &'a dyn Allocator, ctx: C) -> Self {
        Self {
            ctrl:    NonNull::dangling(),
            data:    NonNull::dangling(),
            cap:     0,
            len:     0,
            alloc,
            ctx,
            _marker: PhantomData,
        }
    }

    /// Returns the number of key-value pairs in the map.
    #[inline] pub fn len(&self)      -> usize { self.len }
    /// Returns the current table capacity.
    #[inline] pub fn capacity(&self) -> usize { self.cap }
    /// Returns `true` if the map contains no entries.
    #[inline] pub fn is_empty(&self) -> bool  { self.len == 0 }

    /// Returns a reference to the value associated with `key`, or `None`.
    pub fn get(&self, key: &K) -> Option<&V> {
        let (idx, _) = self.find(key)?;
        // SAFETY: `find` returns an index whose control byte is not
        // `CTRL_EMPTY`, so the slot at `idx` is fully initialised.
        Some(unsafe { (*self.data.as_ptr().add(idx)).val.assume_init_ref() })
    }

    /// Returns a mutable reference to the value associated with `key`, or `None`.
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        let (idx, _) = self.find(key)?;
        // SAFETY: Same as `get`; unique access via `&mut self`.
        Some(unsafe { (*self.data.as_ptr().add(idx)).val.assume_init_mut() })
    }

    /// Returns `true` if the map contains an entry for `key`.
    pub fn contains_key(&self, key: &K) -> bool { self.find(key).is_some() }

    /// Inserts `key`→`val`, returning the previous value for `key` if it existed.
    ///
    /// # Panics
    ///
    /// Panics if the allocator fails during table growth.
    pub fn insert(&mut self, key: K, val: V) -> Option<V> {
        self.try_insert(key, val).unwrap_or_else(|_| panic!("HashMap: OOM"))
    }

    /// Attempts to insert `key`→`val`.
    ///
    /// Returns:
    /// * `Ok(None)` — new entry inserted.
    /// * `Ok(Some(old))` — key already existed; old value returned.
    /// * `Err((key, val))` — allocation failed; inputs returned to caller.
    pub fn try_insert(&mut self, key: K, val: V) -> Result<Option<V>, (K, V)> {
        // Grow before reaching the 87.5 % load threshold.
        if self.cap == 0 || self.len * 8 >= self.cap * 7 {
            if !self.try_grow() { return Err((key, val)); }
        }
        let hash = self.ctx.hash(&key);

        // If the key already exists, overwrite its value.
        if let Some((idx, _)) = self.find(&key) {
            // SAFETY: Slot at `idx` is fully initialised (found by `find`).
            let old = unsafe { (*self.data.as_ptr().add(idx)).val.assume_init_read() };
            unsafe { (*self.data.as_ptr().add(idx)).val = MaybeUninit::new(val) };
            drop(key);
            return Ok(Some(old));
        }

        let slot = self.find_empty_slot(hash);
        unsafe {
            // SAFETY: `slot` is an empty slot within the allocated `data` array.
            (*self.data.as_ptr().add(slot)).key = MaybeUninit::new(key);
            (*self.data.as_ptr().add(slot)).val = MaybeUninit::new(val);
            // SAFETY: `slot < cap`; `set_ctrl` maintains the mirror invariant.
            self.set_ctrl(slot, h2(hash));
        }
        self.len += 1;
        Ok(None)
    }

    /// Removes the entry for `key` and returns its value, or `None`.
    ///
    /// Uses backward-shift deletion to preserve the probing invariant without
    /// a tombstone mechanism.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        let (idx, _) = self.find(key)?;

        // SAFETY: `find` guarantees the slot at `idx` is fully initialised.
        let val = unsafe {
            let s = self.data.as_ptr().add(idx);
            let v = (*s).val.assume_init_read();
            (*s).key.assume_init_drop();
            v
        };
        self.len -= 1;

        // Backward-shift deletion: slide subsequent elements back until we hit
        // an empty slot or an element that is already at its ideal position.
        let mask = self.cap - 1;
        let mut cur = idx;
        loop {
            let nxt      = (cur + 1) & mask;
            let nxt_ctrl = unsafe { *self.ctrl.as_ptr().add(nxt) };
            if nxt_ctrl == CTRL_EMPTY { break; }

            let nxt_ideal = {
                // SAFETY: The slot at `nxt` is occupied (ctrl != CTRL_EMPTY).
                let k = unsafe { (*self.data.as_ptr().add(nxt)).key.assume_init_ref() };
                h1(self.ctx.hash(k)) & mask
            };
            if is_between(cur, nxt_ideal, nxt) { break; }

            unsafe {
                // SAFETY: Both slots are within `0..cap` and `nxt` is occupied.
                let src  = self.data.as_ptr().add(nxt);
                let dst  = self.data.as_ptr().add(cur);
                let k    = (*src).key.assume_init_read();
                let v    = (*src).val.assume_init_read();
                (*dst).key = MaybeUninit::new(k);
                (*dst).val = MaybeUninit::new(v);
                self.set_ctrl(cur, nxt_ctrl);
                self.set_ctrl(nxt, CTRL_EMPTY);
            }
            cur = nxt;
        }
        // SAFETY: `cur` is within `0..cap`.
        unsafe { self.set_ctrl(cur, CTRL_EMPTY) };
        Some(val)
    }

    /// Iterates over all key-value pairs, calling `f` for each.
    ///
    /// The iteration order is unspecified.
    pub fn for_each<F: FnMut(&K, &V)>(&self, mut f: F) {
        for i in 0..self.cap {
            // SAFETY: Control byte `!= CTRL_EMPTY` means the slot is occupied
            // and fully initialised.
            if unsafe { *self.ctrl.as_ptr().add(i) } != CTRL_EMPTY {
                let s = unsafe { &*self.data.as_ptr().add(i) };
                f(unsafe { s.key.assume_init_ref() }, unsafe { s.val.assume_init_ref() });
            }
        }
    }

    /// Iterates over all key-value pairs with mutable access to values.
    pub fn for_each_mut<F: FnMut(&K, &mut V)>(&mut self, mut f: F) {
        for i in 0..self.cap {
            if unsafe { *self.ctrl.as_ptr().add(i) } != CTRL_EMPTY {
                let s = unsafe { &mut *self.data.as_ptr().add(i) };
                f(unsafe { s.key.assume_init_ref() }, unsafe { s.val.assume_init_mut() });
            }
        }
    }

    /// Searches for `key` using SIMD group probing.
    ///
    /// Returns `Some((slot_index, hash))` on a hit, `None` on a miss.
    ///
    /// # Algorithm
    ///
    /// Starting at `h1(hash) & mask`, loads 16 control bytes at a time and
    /// uses SIMD to find slots whose tag matches `h2(hash)`.  Stops when an
    /// empty slot is found (the key cannot appear beyond an empty slot due to
    /// the robin-hood invariant maintained by `remove`).
    fn find(&self, key: &K) -> Option<(usize, u64)> {
        if self.cap == 0 { return None; }
        let hash = self.ctx.hash(key);
        let tag  = h2(hash);
        let mask = self.cap - 1;
        let mut pos = h1(hash) & mask;
        loop {
            // SAFETY: `ctrl` is allocated with `cap + GROUP_WIDTH` bytes;
            // `pos < cap`, so loading a 16-byte group is always within bounds.
            let group = unsafe { Group::load(self.ctrl.as_ptr().add(pos)) };

            for bit in unsafe { group.match_byte(tag) } {
                let idx = (pos + bit) & mask;
                // SAFETY: `idx < cap`; control byte is non-empty so slot is initialised.
                if self.ctx.eq(
                    unsafe { (*self.data.as_ptr().add(idx)).key.assume_init_ref() },
                    key,
                ) {
                    return Some((idx, hash));
                }
            }

            if unsafe { group.match_empty().any() } { return None; }
            pos = (pos + GROUP_WIDTH) & mask;
        }
    }

    /// Finds the first empty slot for a key with the given `hash`.
    ///
    /// Assumes the table is not full (guaranteed by the load-factor check in
    /// `try_insert`).
    fn find_empty_slot(&self, hash: u64) -> usize {
        let mask = self.cap - 1;
        let mut pos = h1(hash) & mask;
        loop {
            // SAFETY: Same group-load safety as `find`.
            let group = unsafe { Group::load(self.ctrl.as_ptr().add(pos)) };
            if let Some(bit) = unsafe { group.match_empty().lowest() } {
                return (pos + bit) & mask;
            }
            pos = (pos + GROUP_WIDTH) & mask;
        }
    }

    /// Sets the control byte at `idx` to `val` and updates the mirror copy.
    ///
    /// The first `GROUP_WIDTH` control bytes are mirrored at offsets
    /// `[cap, cap + GROUP_WIDTH)` so that SIMD group loads at positions near
    /// the end of the table correctly wrap around to the beginning.
    ///
    /// # Safety
    ///
    /// * `idx` must be in `0..cap`.
    /// * `ctrl` must be allocated for at least `cap + GROUP_WIDTH` bytes.
    #[inline]
    unsafe fn set_ctrl(&mut self, idx: usize, val: u8) {
        unsafe {
            // SAFETY: `idx < cap < cap + GROUP_WIDTH` — always within the allocation.
            *self.ctrl.as_ptr().add(idx) = val;
            if idx < GROUP_WIDTH {
                // SAFETY: `cap + idx < cap + GROUP_WIDTH` — within the mirror region.
                *self.ctrl.as_ptr().add(self.cap + idx) = val;
            }
        }
    }

    /// Doubles the table capacity, rehashing all existing entries.
    ///
    /// Returns `true` on success, `false` if any allocation failed (in which
    /// case the map state is unchanged).
    #[cold]
    fn try_grow(&mut self) -> bool {
        let new_cap = if self.cap == 0 { GROUP_WIDTH } else { self.cap * 2 };

        let ctrl_layout = match Layout::array::<u8>(new_cap + GROUP_WIDTH) {
            Ok(l) => l, Err(_) => return false,
        };
        let data_layout = match Layout::array::<Slot<K, V>>(new_cap) {
            Ok(l) => l, Err(_) => return false,
        };

        // SAFETY: Both layouts have non-zero sizes (new_cap >= GROUP_WIDTH > 0).
        let new_ctrl = match unsafe { self.alloc.alloc(ctrl_layout) } {
            Some(p) => p, None => return false,
        };
        let new_data = match unsafe { self.alloc.alloc(data_layout) } {
            Some(p) => p.cast::<Slot<K, V>>(),
            None => {
                // SAFETY: `new_ctrl` was just allocated from `self.alloc` with
                // `ctrl_layout`; releasing it before returning is correct.
                unsafe { self.alloc.dealloc(new_ctrl, ctrl_layout) };
                return false;
            }
        };

        // SAFETY: `new_ctrl` is valid for `new_cap + GROUP_WIDTH` bytes.
        unsafe { simd::fill_bytes(new_ctrl.as_ptr(), CTRL_EMPTY, new_cap + GROUP_WIDTH) };

        let old_ctrl = self.ctrl;
        let old_data = self.data;
        let old_cap  = self.cap;

        self.ctrl = new_ctrl;
        self.data = new_data;
        self.cap  = new_cap;
        self.len  = 0;

        // Rehash all occupied entries from the old table.
        for i in 0..old_cap {
            // SAFETY: `old_ctrl` is valid for `old_cap + GROUP_WIDTH` bytes.
            let c = unsafe { *old_ctrl.as_ptr().add(i) };
            if c != CTRL_EMPTY {
                // SAFETY: Control byte is non-empty, so the slot is initialised.
                let k    = unsafe { (*old_data.as_ptr().add(i)).key.assume_init_read() };
                let v    = unsafe { (*old_data.as_ptr().add(i)).val.assume_init_read() };
                let hash = self.ctx.hash(&k);
                let slot = self.find_empty_slot(hash);
                unsafe {
                    (*self.data.as_ptr().add(slot)).key = MaybeUninit::new(k);
                    (*self.data.as_ptr().add(slot)).val = MaybeUninit::new(v);
                    self.set_ctrl(slot, h2(hash));
                }
                self.len += 1;
            }
        }

        if old_cap > 0 {
            // SAFETY: `old_ctrl` and `old_data` were allocated from `self.alloc`
            // with the corresponding layouts; releasing them now is correct.
            unsafe {
                self.alloc.dealloc(old_ctrl, Layout::array::<u8>(old_cap + GROUP_WIDTH).unwrap());
                self.alloc.dealloc(old_data.cast(), Layout::array::<Slot<K, V>>(old_cap).unwrap());
            }
        }
        true
    }
}

impl<K, V, C: HashContext<K>> Drop for ExHashMap<'_, K, V, C> {
    /// Drops all live key-value pairs and releases both backing allocations.
    fn drop(&mut self) {
        if self.cap == 0 { return; }
        for i in 0..self.cap {
            // SAFETY: Control byte `!= CTRL_EMPTY` means the slot is occupied.
            if unsafe { *self.ctrl.as_ptr().add(i) } != CTRL_EMPTY {
                unsafe {
                    (*self.data.as_ptr().add(i)).key.assume_init_drop();
                    (*self.data.as_ptr().add(i)).val.assume_init_drop();
                }
            }
        }
        // SAFETY: Both pointers were obtained from `self.alloc` with these
        // exact layouts during `try_grow`.
        unsafe {
            self.alloc.dealloc(self.ctrl, Layout::array::<u8>(self.cap + GROUP_WIDTH).unwrap());
            self.alloc.dealloc(self.data.cast(), Layout::array::<Slot<K, V>>(self.cap).unwrap());
        }
    }
}

/// Returns `true` if `ideal` is "between" `cur` and `nxt` in a circular table
/// of size `cap` (where positions are compared modulo `cap`).
///
/// Used by the backward-shift deletion algorithm to determine whether moving
/// an element would violate the probing invariant.
#[inline]
fn is_between(cur: usize, ideal: usize, nxt: usize) -> bool {
    (ideal <= cur && cur < nxt)
        || (cur < nxt && nxt < ideal)
        || (nxt < ideal && ideal <= cur)
}