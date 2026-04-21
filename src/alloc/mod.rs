//! Memory allocator trait and concrete implementations.
//!
//! This module forms the allocation layer of the **zigzag** crate.  All
//! collections in [`crate::collections`] are parameterised over an
//! [`allocator::Allocator`] reference, so the concrete allocator type can be
//! chosen at construction time without runtime polymorphism overhead.
//!
//! ## Available Allocators
//!
//! | Type | Module | Use case |
//! |------|--------|----------|
//! | [`system::SystemAllocator`] | `system` | General-purpose, delegates to `posix_memalign` / `_aligned_malloc` |
//! | [`bump::BumpAllocator`] | `bump` | Fast linear allocation from a static buffer; no individual free |
//! | [`arena::ArenaAllocator`] | `arena` | Per-block allocation with bulk free; useful for request-scoped data |
//! | [`counting::CountingAllocator`] | `counting` | Transparent wrapper that records allocation statistics |
//! | [`pool::PoolAllocator`] | `pool` | Fixed-size object pool with lock-free alloc / dealloc |

pub mod allocator;
pub mod system;
pub mod bump;
pub mod arena;
pub mod counting;
pub mod pool;