//! C-compatible Foreign Function Interface (FFI) bindings for Zigzag.
//!
//! Unlike standard Rust modules, every type and function in this module:
//!
//! 1. **Uses a strict C ABI.** Functions are exposed via `#[unsafe(no_mangle)]`
//!    and `extern "C"`, relying exclusively on raw pointers and `c_void` across
//!    the boundary.
//! 2. **Manually manages fat pointers.** Trait objects (like `dyn Allocator`)
//!    are destructured into a data pointer and a vtable via [`RawAllocHandle`]
//!    to safely pass them to and from C.
//! 3. **Never uses implicit global allocation.** Internal helper functions like
//!    [`sys_box_new`] explicitly route internal FFI allocations through the
//!    native [`SystemAllocator`].
//!
//! ## FFI Overview
//!
//! | Component | Description |
//! |-----------|-------------|
//! | [`allocators`] | C bindings for the allocator hierarchy (System, Arena, Pool, etc.) |
//! | [`collections`] | C bindings for high-performance collections (Vec, String, HashMap, etc.) |
//! | [`RawAllocHandle`] | A C-safe raw representation of a trait object pointer |
//! | `sys_box_new` | Internal helper to allocate FFI wrapper types |
//! | `sys_box_drop` | Internal helper to deallocate FFI wrapper types |
//!
//! [`SystemAllocator`]: crate::alloc::system::SystemAllocator

use core::{alloc::Layout, ptr::NonNull};

use crate::alloc::{allocator::Allocator, system::SystemAllocator};

pub mod allocators;
pub mod collections;

/// A C-compatible, raw representation of a trait object pointer to `dyn Allocator`.
///
/// This struct manually destructures a fat pointer into its data and vtable
/// components so it can be safely passed across FFI boundaries without relying
/// on Rust's unstable trait object ABI.
///
/// # Memory Layout
///
/// Matches the internal memory layout of `*mut dyn Allocator` (two thin pointers).
#[repr(C)]
#[derive(Copy, Clone)]
pub struct RawAllocHandle {
    pub(crate) data: *const (),
    pub(crate) vtable: *const (),
}

// SAFETY: The handle is essentially a raw pointer pair. Send/Sync semantics
// depend entirely on the underlying allocator, but we assume thread-safe
// allocators for the FFI boundary.
unsafe impl Sync for RawAllocHandle {}
unsafe impl Send for RawAllocHandle {}

impl RawAllocHandle {
    /// Creates a `RawAllocHandle` from a reference to an [`Allocator`].
    #[inline]
    pub(crate) fn from_ref(r: &dyn Allocator) -> Self {
        // SAFETY: Transmuting a fat pointer (`&dyn Allocator`) to two thin pointers
        // (`data` and `vtable`) is valid in this specific ABI context. It explicitly
        // relies on Rust's internal trait object memory layout.
        unsafe { core::mem::transmute(r) }
    }

    /// Reconstructs the `&dyn Allocator` trait object from the raw handle.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// * The original `data` and `vtable` pointers are still valid.
    /// * The pointers correspond to a live `Allocator` instance.
    #[inline]
    pub(crate) unsafe fn as_ref<'a>(self) -> &'a dyn Allocator {
        // SAFETY: Reversing the transmute done in `from_ref`. The caller
        // guarantees pointer validity and appropriate lifetimes.
        unsafe { core::mem::transmute(self) }
    }
}

/// Allocates memory for a generic type `T` using the `SystemAllocator` and places `val` into it.
///
/// Returns a raw pointer to the allocated `T`.
#[inline]
pub(crate) fn sys_box_new<T>(val: T) -> *mut T {
    let layout = Layout::new::<T>();
    let ptr: *mut T = if layout.size() == 0 {
        NonNull::dangling().as_ptr()
    } else {
        // SAFETY: We checked that layout size > 0. SystemAllocator guarantees
        // valid allocations for the requested layout.
        unsafe {
            SystemAllocator
                .alloc(layout)
                .expect("zigzag FFI: SystemAllocator OOM")
                .cast::<T>()
                .as_ptr()
        }
    };

    // SAFETY: `ptr` is either dynamically allocated and valid for writes of `T`,
    // or dangling but `T` is a Zero-Sized Type (ZST).
    unsafe { ptr.write(val) };
    ptr
}

/// Drops the value pointed to by `ptr` and deallocates its memory using `SystemAllocator`.
///
/// # Safety
///
/// * `ptr` must be derived from a previous call to `sys_box_new`.
/// * `ptr` must not be used after this function returns (use-after-free).
#[inline]
pub(crate) unsafe fn sys_box_drop<T>(ptr: *mut T) {
    if ptr.is_null() {
        return;
    }

    // SAFETY: Caller guarantees `ptr` is valid, properly aligned, and ownership
    // is fully transferred here for destruction.
    unsafe {
        core::ptr::drop_in_place(ptr);
        let layout = Layout::new::<T>();
        if layout.size() > 0 {
            // SAFETY: `ptr` was allocated with `SystemAllocator` and `Layout::new::<T>()`.
            SystemAllocator.dealloc(NonNull::new_unchecked(ptr as *mut u8), layout);
        }
    }
}
