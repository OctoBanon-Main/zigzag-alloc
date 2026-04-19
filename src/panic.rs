use core::fmt::{Write, self};
use core::panic::PanicInfo;

#[cfg(target_family = "unix")]
unsafe extern "C" {
    fn write(fd: i32, buf: *const u8, count: usize) -> isize;
}

#[cfg(target_family = "windows")]
unsafe extern "system" {
    fn GetStdHandle(nStdHandle: u32) -> *mut u8;
    fn WriteFile(
        hFile: *mut u8,
        lpBuffer: *const u8,
        nNumberOfBytesToWrite: u32,
        lpNumberOfBytesWritten: *mut u32,
        lpOverlapped: *mut u8,
    ) -> i32;
}

struct PanicWriter {
    buf: [u8; 512],
    pos: usize,
}

impl PanicWriter {
    const fn new() -> Self {
        Self {
            buf: [0u8; 512],
            pos: 0,
        }
    }

    fn flush(&self) {
        if self.pos == 0 {
            return;
        }

        #[cfg(target_family = "unix")]
        unsafe {
            write(2, self.buf.as_ptr(), self.pos);
        }

        #[cfg(target_family = "windows")]
        unsafe {
            const STD_ERROR_HANDLE: u32 = 0xFFFFFFF4u32;
            let handle = GetStdHandle(STD_ERROR_HANDLE);
            if !handle.is_null() {
                let mut written: u32 = 0;
                WriteFile(
                    handle,
                    self.buf.as_ptr(),
                    self.pos as u32,
                    &mut written,
                    core::ptr::null_mut()
                );
            }
        }

        #[cfg(not(any(target_family = "unix", target_family = "windows")))]
        let _ = self;
    }
}

impl fmt::Write for PanicWriter {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        let bytes = s.as_bytes();
        let available = self.buf.len() - self.pos;
        let to_copy = bytes.len().min(available);
        self.buf[self.pos..self.pos + to_copy].copy_from_slice(&bytes[..to_copy]);
        self.pos += to_copy;
        Ok(())
    }
}

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    let mut w = PanicWriter::new();
    let _ = writeln!(w, "\npanic: {info}");
    w.flush();

    loop {
        core::hint::spin_loop();
    }
}