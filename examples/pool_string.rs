use zigzag::alloc::system::SystemAllocator;
use zigzag::alloc::pool::PoolAllocator;
use zigzag::collections::ExString;
use core::fmt::Write;

fn main() {
    let sys = SystemAllocator;

    let pool = PoolAllocator::typed::<[u8; 128]>(&sys, 5).unwrap();

    let mut log = ExString::new(&pool);
    write!(log, "Status: {}, Code: {}", "OK", 200).unwrap();

    println!("Log message: {}", log.as_str());
    println!("Slots left in pool: {}", pool.free_count());
}