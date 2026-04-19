use std::cell::UnsafeCell;

use zigzag::alloc::bump::BumpAllocator;
use zigzag::collections::ZigVec;

struct SyncCell<T>(UnsafeCell<T>);

unsafe impl<T> Sync for SyncCell<T> {}

static MEMORY: SyncCell<[u8; 1024]> = SyncCell(UnsafeCell::new([0; 1024]));
fn main() {
    let memory_ptr = MEMORY.0.get();
    let bump = unsafe { 
        BumpAllocator::new(&mut *memory_ptr) 
    };

    let mut stack_vec = ZigVec::new(&bump);
    
    stack_vec.push(10);
    stack_vec.push(20);
    stack_vec.push(30);

    println!("Bump usage: {}/{}", bump.used(), 1024);
    
    unsafe { bump.reset(); }
}