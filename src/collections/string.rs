use core::fmt::{self, Write as FmtWrite};

use crate::alloc::allocator::Allocator;
use super::vec::ZigVec;

pub struct ZigString<'a> {
    buf: ZigVec<'a, u8>,
}

impl<'a> ZigString<'a> {
    pub fn new(alloc: &'a dyn Allocator) -> Self {
        Self { buf: ZigVec::new(alloc) }
    }

    pub fn from_str(s: &str, alloc: &'a dyn Allocator) -> Self {
        let mut this = Self::new(alloc);
        this.push_str(s);
        this
    }

    pub fn push_str(&mut self, s: &str) {
        for &b in s.as_bytes() {
            self.buf.push(b);
        }
    }

    pub fn push(&mut self, ch: char) {
        let mut tmp = [0u8; 4];
        self.push_str(ch.encode_utf8(&mut tmp));
    }

    #[inline]
    pub fn as_str(&self) -> &str {
        unsafe { core::str::from_utf8_unchecked(self.buf.as_slice()) }
    }

    #[inline] pub fn len(&self) -> usize { self.buf.len() }
    #[inline] pub fn is_empty(&self) -> bool { self.buf.is_empty() }
    #[inline] pub fn capacity(&self) -> usize { self.buf.capacity() }

    pub fn clear(&mut self) {
        unsafe { self.buf.set_len(0) };
    }

    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        self.buf.as_slice()
    }
}

impl FmtWrite for ZigString<'_> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.push_str(s);
        Ok(())
    }
}

impl fmt::Display for ZigString<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl fmt::Debug for ZigString<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.as_str())
    }
}

impl PartialEq<str> for ZigString<'_> {
    fn eq(&self, other: &str) -> bool { self.as_str() == other }
}

impl PartialEq for ZigString<'_> {
    fn eq(&self, other: &Self) -> bool { self.as_str() == other.as_str() }
}