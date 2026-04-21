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

pub trait HashContext<K> {
    fn hash(&self, key: &K) -> u64;
    fn eq(&self, a: &K, b: &K) -> bool;
}
 
pub trait OrdContext<T> {
    fn less(&self, a: &T, b: &T) -> bool;
}

pub struct U64HashCtx;
impl HashContext<u64> for U64HashCtx {
    #[inline]
    fn hash(&self, k: &u64) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in k.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
        h
    }
    #[inline]
    fn eq(&self, a: &u64, b: &u64) -> bool { a == b }
}
pub struct UsizeHashCtx;

impl HashContext<usize> for UsizeHashCtx {
    #[inline]
    fn hash(&self, k: &usize) -> u64 {
        U64HashCtx.hash(&(*k as u64))
    }
    #[inline]
    fn eq(&self, a: &usize, b: &usize) -> bool { a == b }
}
 
pub struct UsizeMinCtx;
impl OrdContext<usize> for UsizeMinCtx {
    #[inline] fn less(&self, a: &usize, b: &usize) -> bool { a < b }
}