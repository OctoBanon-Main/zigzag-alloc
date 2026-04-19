#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
pub mod panic;

pub mod alloc;
pub mod collections;