use core::fmt::{self, Write as FmtWrite};

use crate::alloc::allocator::Allocator;
use super::ExVec;

pub struct ExString<'a> {
    buf: ExVec<'a, u8>,
}

impl<'a> ExString<'a> {
    pub fn new(alloc: &'a dyn Allocator) -> Self {
        Self { buf: ExVec::new(alloc) }
    }

    pub fn from_str(s: &str, alloc: &'a dyn Allocator) -> Self {
        let mut this = Self::new(alloc);
        this.push_str(s);
        this
    }

    pub fn push_str(&mut self, s: &str) {
        self.buf.push_slice(s.as_bytes());
    }

    pub fn push(&mut self, ch: char) {
        let mut tmp = [0u8; 4];
        self.push_str(ch.encode_utf8(&mut tmp));
    }

    #[inline]
    pub fn as_str(&self) -> &str {
        unsafe { core::str::from_utf8_unchecked(self.buf.as_slice()) }
    }

    #[inline] pub fn len(&self)      -> usize { self.buf.len() }
    #[inline] pub fn is_empty(&self) -> bool  { self.buf.is_empty() }
    #[inline] pub fn capacity(&self) -> usize { self.buf.capacity() }
    #[inline] pub fn as_bytes(&self) -> &[u8] { self.buf.as_slice() }

    pub fn clear(&mut self) {
        unsafe { self.buf.set_len(0) };
    }

    #[inline]
    pub fn find_byte(&self, byte: u8) -> Option<usize> {
        self.buf.find_byte(byte)
    }

    #[inline]
    pub fn contains_byte(&self, byte: u8) -> bool {
        self.find_byte(byte).is_some()
    }

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

    pub fn for_each_byte_match<F: FnMut(usize)>(&self, byte: u8, mut f: F) {
        self.buf.for_each_byte_match(byte, &mut f);
    }

    pub fn replace_byte(&mut self, from: u8, to: u8) {
        let n   = self.buf.len();
        let ptr = self.buf.as_mut_slice().as_mut_ptr();
        let mut i = 0usize;
        while i < n {
            match unsafe { crate::simd::find_byte(ptr.add(i), from, n - i) } {
                Some(off) => {
                    unsafe { *ptr.add(i + off) = to };
                    i += off + 1;
                }
                None => break,
            }
        }
    }
}

impl FmtWrite for ExString<'_> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.push_str(s);
        Ok(())
    }
}

impl fmt::Display for ExString<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str(self.as_str()) }
}

impl fmt::Debug for ExString<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{:?}", self.as_str()) }
}

impl PartialEq<str> for ExString<'_> {
    fn eq(&self, other: &str) -> bool { self.as_str() == other }
}

impl PartialEq for ExString<'_> {
    fn eq(&self, other: &Self) -> bool { self.as_str() == other.as_str() }
}