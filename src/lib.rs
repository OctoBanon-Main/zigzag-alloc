#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
pub mod panic;

pub mod alloc;
pub mod collections;

#[cfg(feature = "ffi")]
pub mod ffi;

mod simd;

pub use collections::{
    ExBox,
    ExHashMap,
    ExString,
    ExVec,
    ExPriorityQueue,
    ExBoundedArray,
};
