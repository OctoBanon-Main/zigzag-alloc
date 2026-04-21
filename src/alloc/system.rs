//! System allocator backed by the OS memory-management API.
//!
//! [`SystemAllocator`] wraps platform alignment-aware allocation primitives:
//!
//! | Platform | Alloc              | Free              |
//! |----------|--------------------|-------------------|
//! | Unix     | `posix_memalign`   | `free`            |
//! | Windows  | `_aligned_malloc`  | `_aligned_free`   |
//! | Other    | always returns `None` | no-op          |
//!
//! This is the lowest-level allocator in the crate.  Higher-level allocators
//! (arena, bump, pool) typically use a `SystemAllocator` as their *backing*
//! allocator.

#[warn(unused_doc_comments)]
#[allow(unused_imports)]

use core::{alloc::Layout, ffi::c_void, mem, ptr::NonNull};

use super::allocator::Allocator;

/// Allocates `size` bytes with a minimum alignment of `alignment`.
///
/// On success writes the pointer to `*memptr` and returns 0.
/// On failure returns a non-zero error code and leaves `*memptr` unchanged.
///
/// # Safety
/// `alignment` must be a power of two and a multiple of `sizeof(void*)`.
/// `size` must be greater than zero.
#[cfg(target_family = "unix")]
unsafe extern "C" {
    fn posix_memalign(memptr: *mut *mut c_void, alignment: usize, size: usize) -> i32;

    /// Releases memory previously obtained from `posix_memalign` (or `malloc`).
    ///
    /// # Safety
    /// `ptr` must have been returned by a prior successful allocation call.
    /// Passing `null` is defined (no-op).  Double-free is undefined behaviour.
    fn free(ptr: *mut c_void);
}

/// Allocates `size` bytes with the given `alignment`.
///
/// Returns a non-null pointer on success, or null on failure.
///
/// # Safety
/// `alignment` must be a power of two.  `size` must be greater than zero.
#[cfg(target_family = "windows")]
unsafe extern "C" {
    fn _aligned_malloc(size: usize, alignment: usize) -> *mut c_void;

    /// Releases memory previously obtained from `_aligned_malloc`.
    ///
    /// # Safety
    /// `ptr` must have been returned by `_aligned_malloc`.
    /// Passing `null` is defined (no-op).  Double-free is undefined behaviour.
    fn _aligned_free(ptr: *mut c_void);
}

// ── SystemAllocator ──────────────────────────────────────────────────────────

/// An [`Allocator`] that delegates directly to the operating system.
///
/// `SystemAllocator` is a zero-sized type; it holds no state and is safe to
/// share across threads.  It is the typical *backing allocator* for arena,
/// pool, and bump allocators when running on a hosted OS.
///
/// # Platform Notes
///
/// * **Unix** — Uses `posix_memalign` to satisfy arbitrary alignment requests.
///   The effective alignment is always at least `sizeof(void*)` (usually 8 or
///   16 bytes) because `posix_memalign` requires it.
/// * **Windows** — Uses `_aligned_malloc` / `_aligned_free`.  The effective
///   alignment is at least `sizeof(void*)`.
/// * **Other** — [`alloc`](Allocator::alloc) always returns `None`.
pub struct SystemAllocator;

impl Allocator for SystemAllocator {
    /// Allocates a block of memory according to `layout`.
    ///
    /// Returns `Some(ptr)` where `ptr` is aligned to at least `layout.align()`
    /// and valid for `layout.size()` bytes.  Returns `None` if the OS call
    /// fails or if the target platform is unsupported.
    ///
    /// # Safety
    ///
    /// * `layout.size()` must be greater than zero; zero-sized allocations are
    ///   rejected by `posix_memalign` and `_aligned_malloc`.
    /// * The returned pointer must eventually be passed to
    ///   [`dealloc`](Allocator::dealloc) with the same `layout`.
    unsafe fn alloc(&self, layout: Layout) -> Option<NonNull<u8>> {
        let size = layout.size();

        if size == 0 {
            // Callers must not pass zero-sized layouts, but we guard defensively
            // by returning a dangling non-null pointer rather than invoking
            // undefined behaviour in the OS allocation function.
            return Some(NonNull::dangling());
        }

        #[cfg(target_family = "unix")]
        {
            let mut ptr: *mut c_void = core::ptr::null_mut();

            // SAFETY: posix_memalign requires alignment to be a power of two and
            // a multiple of sizeof(void*).  We enforce the latter by taking the
            // maximum of the requested alignment and the pointer size.
            let min_align = core::mem::size_of::<*mut c_void>();
            let align = layout.align().max(min_align);

            let rc = unsafe { posix_memalign(&mut ptr, align, size) };

            if rc != 0 {
                return None;
            }
            NonNull::new(ptr as *mut u8)
        }

        #[cfg(target_family = "windows")]
        {
            // SAFETY: _aligned_malloc requires alignment to be a power of two.
            // Layout::align() already guarantees that invariant.
            let align = layout.align().max(mem::size_of::<*const ()>());
            let ptr = unsafe { _aligned_malloc(size, align) as *mut u8 };
            NonNull::new(ptr)
        }

        #[cfg(not(any(target_family = "unix", target_family = "windows")))]
        {
            None
        }
    }

    /// Releases a block previously allocated by this `SystemAllocator`.
    ///
    /// # Safety
    ///
    /// * `ptr` must have been returned by a prior successful call to
    ///   [`alloc`](Allocator::alloc) on **this same** `SystemAllocator`.
    /// * `layout` must exactly match the layout passed to that `alloc` call.
    /// * After `dealloc` returns, `ptr` is invalid and must not be accessed.
    /// * Calling `dealloc` twice for the same pointer is undefined behaviour.
    unsafe fn dealloc(&self, ptr: NonNull<u8>, layout: Layout) {
        if layout.size() == 0 {
            // Matches the dangling-pointer fast-path in `alloc`.
            return;
        }

        #[cfg(target_family = "unix")]
        {
            // SAFETY: `ptr` was obtained from `posix_memalign` and has not been
            // freed before — both guaranteed by the caller of this method.
            unsafe { free(ptr.as_ptr() as *mut c_void) }
        }

        #[cfg(target_family = "windows")]
        {
            // SAFETY: `ptr` was obtained from `_aligned_malloc` and has not been
            // freed before — both guaranteed by the caller of this method.
            unsafe { _aligned_free(ptr.as_ptr() as *mut c_void) };
        }
    }
}