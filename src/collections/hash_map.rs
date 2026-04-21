use core::{alloc::Layout, marker::PhantomData, mem::MaybeUninit, ptr::NonNull};

use crate::alloc::allocator::Allocator;
use crate::simd::{self, Group, GROUP_WIDTH, CTRL_EMPTY};
use super::HashContext;


#[inline] fn h1(hash: u64) -> usize { hash as usize }
#[inline] fn h2(hash: u64) -> u8 { ((hash >> 57) as u8) & 0x7F }


struct Slot<K, V> {
    key: MaybeUninit<K>,
    val: MaybeUninit<V>,
}

pub struct ExHashMap<'a, K, V, C: HashContext<K>> {
    ctrl:    NonNull<u8>,
    data:    NonNull<Slot<K, V>>,
    cap:     usize,
    len:     usize,
    alloc:   &'a dyn Allocator,
    ctx:     C,
    _marker: PhantomData<(K, V)>,
}

impl<'a, K, V, C: HashContext<K>> ExHashMap<'a, K, V, C> {
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

    #[inline] pub fn len(&self)      -> usize { self.len }
    #[inline] pub fn capacity(&self) -> usize { self.cap }
    #[inline] pub fn is_empty(&self) -> bool  { self.len == 0 }

    pub fn get(&self, key: &K) -> Option<&V> {
        let (idx, _) = self.find(key)?;
        Some(unsafe { (*self.data.as_ptr().add(idx)).val.assume_init_ref() })
    }

    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        let (idx, _) = self.find(key)?;
        Some(unsafe { (*self.data.as_ptr().add(idx)).val.assume_init_mut() })
    }

    pub fn contains_key(&self, key: &K) -> bool { self.find(key).is_some() }

    pub fn insert(&mut self, key: K, val: V) -> Option<V> {
        self.try_insert(key, val).unwrap_or_else(|_| panic!("HashMap: OOM"))
    }

    pub fn try_insert(&mut self, key: K, val: V) -> Result<Option<V>, (K, V)> {
        if self.cap == 0 || self.len * 8 >= self.cap * 7 {
            if !self.try_grow() { return Err((key, val)); }
        }
        let hash = self.ctx.hash(&key);

        if let Some((idx, _)) = self.find(&key) {
            let old = unsafe { (*self.data.as_ptr().add(idx)).val.assume_init_read() };
            unsafe { (*self.data.as_ptr().add(idx)).val = MaybeUninit::new(val) };
            drop(key);
            return Ok(Some(old));
        }

        let slot = self.find_empty_slot(hash);
        unsafe {
            (*self.data.as_ptr().add(slot)).key = MaybeUninit::new(key);
            (*self.data.as_ptr().add(slot)).val = MaybeUninit::new(val);
            self.set_ctrl(slot, h2(hash));
        }
        self.len += 1;
        Ok(None)
    }

    pub fn remove(&mut self, key: &K) -> Option<V> {
        let (idx, _) = self.find(key)?;

        let val = unsafe {
            let s = self.data.as_ptr().add(idx);
            let v = (*s).val.assume_init_read();
            (*s).key.assume_init_drop();
            v
        };
        self.len -= 1;

        let mask = self.cap - 1;
        let mut cur = idx;
        loop {
            let nxt      = (cur + 1) & mask;
            let nxt_ctrl = unsafe { *self.ctrl.as_ptr().add(nxt) };
            if nxt_ctrl == CTRL_EMPTY { break; }

            let nxt_ideal = {
                let k = unsafe { (*self.data.as_ptr().add(nxt)).key.assume_init_ref() };
                h1(self.ctx.hash(k)) & mask
            };
            if is_between(cur, nxt_ideal, nxt) { break; }

            unsafe {
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
        unsafe { self.set_ctrl(cur, CTRL_EMPTY) };
        Some(val)
    }

    pub fn for_each<F: FnMut(&K, &V)>(&self, mut f: F) {
        for i in 0..self.cap {
            if unsafe { *self.ctrl.as_ptr().add(i) } != CTRL_EMPTY {
                let s = unsafe { &*self.data.as_ptr().add(i) };
                f(unsafe { s.key.assume_init_ref() }, unsafe { s.val.assume_init_ref() });
            }
        }
    }

    pub fn for_each_mut<F: FnMut(&K, &mut V)>(&mut self, mut f: F) {
        for i in 0..self.cap {
            if unsafe { *self.ctrl.as_ptr().add(i) } != CTRL_EMPTY {
                let s = unsafe { &mut *self.data.as_ptr().add(i) };
                f(unsafe { s.key.assume_init_ref() }, unsafe { s.val.assume_init_mut() });
            }
        }
    }

    fn find(&self, key: &K) -> Option<(usize, u64)> {
        if self.cap == 0 { return None; }
        let hash = self.ctx.hash(key);
        let tag  = h2(hash);
        let mask = self.cap - 1;
        let mut pos = h1(hash) & mask;
        loop {
            let group = unsafe { Group::load(self.ctrl.as_ptr().add(pos)) };

            for bit in unsafe { group.match_byte(tag) } {
                let idx = (pos + bit) & mask;
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

    fn find_empty_slot(&self, hash: u64) -> usize {
        let mask = self.cap - 1;
        let mut pos = h1(hash) & mask;
        loop {
            let group = unsafe { Group::load(self.ctrl.as_ptr().add(pos)) };
            if let Some(bit) = unsafe { group.match_empty().lowest() } {
                return (pos + bit) & mask;
            }
            pos = (pos + GROUP_WIDTH) & mask;
        }
    }

    #[inline]
    unsafe fn set_ctrl(&mut self, idx: usize, val: u8) {
        unsafe {
            *self.ctrl.as_ptr().add(idx) = val;
            if idx < GROUP_WIDTH {
                *self.ctrl.as_ptr().add(self.cap + idx) = val;
            }
        }
    }

    #[cold]
    fn try_grow(&mut self) -> bool {
        let new_cap = if self.cap == 0 { GROUP_WIDTH } else { self.cap * 2 };
        let ctrl_layout = match Layout::array::<u8>(new_cap + GROUP_WIDTH) {
            Ok(l) => l, Err(_) => return false,
        };
        let data_layout = match Layout::array::<Slot<K, V>>(new_cap) {
            Ok(l) => l, Err(_) => return false,
        };

        let new_ctrl = match unsafe { self.alloc.alloc(ctrl_layout) } {
            Some(p) => p, None => return false,
        };
        let new_data = match unsafe { self.alloc.alloc(data_layout) } {
            Some(p) => p.cast::<Slot<K, V>>(),
            None => { unsafe { self.alloc.dealloc(new_ctrl, ctrl_layout) }; return false; }
        };

        unsafe { simd::fill_bytes(new_ctrl.as_ptr(), CTRL_EMPTY, new_cap + GROUP_WIDTH) };

        let old_ctrl = self.ctrl;
        let old_data = self.data;
        let old_cap  = self.cap;

        self.ctrl = new_ctrl;
        self.data = new_data;
        self.cap  = new_cap;
        self.len  = 0;

        for i in 0..old_cap {
            let c = unsafe { *old_ctrl.as_ptr().add(i) };
            if c != CTRL_EMPTY {
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
            unsafe {
                self.alloc.dealloc(old_ctrl, Layout::array::<u8>(old_cap + GROUP_WIDTH).unwrap());
                self.alloc.dealloc(old_data.cast(), Layout::array::<Slot<K, V>>(old_cap).unwrap());
            }
        }
        true
    }
}

impl<K, V, C: HashContext<K>> Drop for ExHashMap<'_, K, V, C> {
    fn drop(&mut self) {
        if self.cap == 0 { return; }
        for i in 0..self.cap {
            if unsafe { *self.ctrl.as_ptr().add(i) } != CTRL_EMPTY {
                unsafe {
                    (*self.data.as_ptr().add(i)).key.assume_init_drop();
                    (*self.data.as_ptr().add(i)).val.assume_init_drop();
                }
            }
        }
        unsafe {
            self.alloc.dealloc(self.ctrl, Layout::array::<u8>(self.cap + GROUP_WIDTH).unwrap());
            self.alloc.dealloc(self.data.cast(), Layout::array::<Slot<K, V>>(self.cap).unwrap());
        }
    }
}

#[inline]
fn is_between(cur: usize, ideal: usize, nxt: usize) -> bool {
    (ideal <= cur && cur < nxt)
        || (cur < nxt && nxt < ideal)
        || (nxt < ideal && ideal <= cur)
}