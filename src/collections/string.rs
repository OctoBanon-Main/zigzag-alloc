//! Growable UTF-8 string with an explicit allocator.
//!
//! [`ExString`] is a thin wrapper around [`ExVec<u8>`] that maintains the
//! invariant that the byte buffer is valid UTF-8.  It implements
//! [`fmt::Write`] so it can be used as the target of `write!` / `writeln!`.
//!
//! Like all collections in this crate, `ExString` never touches the global
//! allocator — every allocation is routed through the provided
//! [`Allocator`] reference.

use core::fmt::{self, Write as FmtWrite};

use crate::alloc::allocator::Allocator;
use super::ExVec;

/// A growable, allocator-backed UTF-8 string.
///
/// # Invariant
///
/// The byte buffer `buf` always contains valid UTF-8.  Methods that accept
/// raw `&str` or `char` values uphold this invariant automatically.  The
/// `as_str` / `as_bytes` accessors expose the buffer read-only so callers
/// cannot introduce invalid UTF-8.
///
/// # Lifetime
///
/// The allocator reference `'a` must outlive the `ExString`.
pub struct ExString<'a> {
    /// Underlying byte buffer; always valid UTF-8.
    buf: ExVec<'a, u8>,
}

impl<'a> ExString<'a> {
    /// Creates a new, empty `ExString` that will allocate through `alloc`.
    pub fn new(alloc: &'a dyn Allocator) -> Self {
        Self { buf: ExVec::new(alloc) }
    }

    /// Creates an `ExString` pre-populated with a copy of `s`.
    pub fn from_str(s: &str, alloc: &'a dyn Allocator) -> Self {
        let mut this = Self::new(alloc);
        this.push_str(s);
        this
    }

    /// Appends the string slice `s` to the end of this string.
    ///
    /// Grows the underlying buffer if necessary.
    ///
    /// # Panics
    ///
    /// Panics if the backing allocator cannot satisfy the growth request.
    pub fn push_str(&mut self, s: &str) {
        self.buf.push_slice(s.as_bytes());
    }

    /// Appends the Unicode scalar `ch` (encoded as UTF-8) to this string.
    ///
    /// # Panics
    ///
    /// Panics if the backing allocator cannot satisfy the growth request.
    pub fn push(&mut self, ch: char) {
        let mut tmp = [0u8; 4];
        self.push_str(ch.encode_utf8(&mut tmp));
    }

    /// Returns the string content as a `&str`.
    ///
    /// # Safety Justification
    ///
    /// The buffer is always valid UTF-8 because only `push_str` and `push`
    /// can append bytes, and both sources (`&str` / `char`) are guaranteed
    /// to be valid UTF-8 by Rust's type system.  `clear` simply sets `len = 0`
    /// without writing invalid bytes.
    #[inline]
    pub fn as_str(&self) -> &str {
        // SAFETY: `buf` contains valid UTF-8 at all times — see module invariant.
        unsafe { core::str::from_utf8_unchecked(self.buf.as_slice()) }
    }

    /// Returns the number of bytes (not Unicode code points) in the string.
    #[inline] pub fn len(&self)      -> usize { self.buf.len() }
    /// Returns `true` if the string contains no bytes.
    #[inline] pub fn is_empty(&self) -> bool  { self.buf.is_empty() }
    /// Returns the number of bytes the buffer can hold without reallocating.
    #[inline] pub fn capacity(&self) -> usize { self.buf.capacity() }
    /// Returns the raw byte slice of the string contents.
    #[inline] pub fn as_bytes(&self) -> &[u8] { self.buf.as_slice() }

    /// Clears the string, setting its length to zero.
    ///
    /// Does **not** release the backing allocation.
    pub fn clear(&mut self) {
        // SAFETY: Setting `len = 0` is safe for `u8` which has no destructor.
        unsafe { self.buf.set_len(0) };
    }

    /// Returns the byte offset of the first occurrence of `byte`, or `None`.
    #[inline]
    pub fn find_byte(&self, byte: u8) -> Option<usize> {
        self.buf.find_byte(byte)
    }

    /// Returns `true` if the string contains the given byte.
    #[inline]
    pub fn contains_byte(&self, byte: u8) -> bool {
        self.find_byte(byte).is_some()
    }

    /// Returns the number of times `byte` appears in the string.
    pub fn count_byte(&self, byte: u8) -> usize {
        let ptr = self.buf.as_ptr();
        let n   = self.buf.len();
        let mut count = 0usize;
        let mut i     = 0usize;
        while i < n {
            match unsafe { crate::simd::find_byte(ptr.add(i), byte, n - i) } {
                Some(off) => { count += 1; i += off + 1; }
                None      => break,
            }
        }
        count
    }

    /// Calls `f` with the byte offset of every occurrence of `byte`.
    pub fn for_each_byte_match<F: FnMut(usize)>(&self, byte: u8, mut f: F) {
        self.buf.for_each_byte_match(byte, &mut f);
    }

    /// Replaces every occurrence of `from` with `to` in-place.
    ///
    /// # Panics
    ///
    /// If `from` and `to` have different UTF-8 lengths, the resulting string
    /// may no longer be valid UTF-8.  This method is designed for single-byte
    /// replacements only (e.g. replacing `b'\n'` with `b' '`).
    pub fn replace_byte(&mut self, from: u8, to: u8) {
        let n   = self.buf.len();
        let ptr = self.buf.as_mut_slice().as_mut_ptr();
        let mut i = 0usize;
        while i < n {
            match unsafe { crate::simd::find_byte(ptr.add(i), from, n - i) } {
                Some(off) => {
                    // SAFETY: `i + off < n`, so `ptr + i + off` is within the
                    // initialised buffer.
                    unsafe { *ptr.add(i + off) = to };
                    i += off + 1;
                }
                None => break,
            }
        }
    }
}

impl FmtWrite for ExString<'_> {
    /// Appends `s` to the string.  Called by `write!` / `writeln!`.
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.push_str(s);
        Ok(())
    }
}

impl fmt::Display for ExString<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl fmt::Debug for ExString<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.as_str())
    }
}

impl PartialEq<str> for ExString<'_> {
    fn eq(&self, other: &str) -> bool { self.as_str() == other }
}

impl PartialEq for ExString<'_> {
    fn eq(&self, other: &Self) -> bool { self.as_str() == other.as_str() }
}