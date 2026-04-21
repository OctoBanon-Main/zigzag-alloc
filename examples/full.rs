use zigzag_alloc::alloc::{
    system::SystemAllocator,
    arena::ArenaAllocator,
    pool::PoolAllocator,
    counting::CountingAllocator,
};
use zigzag_alloc::collections::{ExBox, ExVec, ExString};

fn main() {
    let sys = CountingAllocator::new(SystemAllocator);
    
    println!("=== ExBox & SystemAllocator ===");
    {
        let b = ExBox::new(42, &sys).expect("Failed to alloc Box");
        println!("Box value: {}", *b);
    }
    
    let stats = sys.stats();
    println!("Allocated: {} bytes", stats.bytes_allocated);
    println!("Deallocated: {} bytes", stats.bytes_freed);
    println!("Live bytes: {}\n", stats.bytes_live);

    println!("=== ExVec & Arena (Linear Allocation) ===");
    {
        let arena = ArenaAllocator::new(&sys);
        
        let mut v = ExVec::new(&arena);
        for i in 0..5 {
            v.push(i * 10);
        }
        println!("Vec: {:?}", v.as_slice());
        
        arena.reset();
        println!("Arena reset performed");
    }

    println!("\n=== ExString & Pool (Fixed Size Blocks) ===");
    {
        let pool = PoolAllocator::typed::<[u8; 64]>(&sys, 10)
            .expect("Failed to create Pool");
        
        let mut s = ExString::new(&pool);
        s.push_str("Hello from ZigZag!");
        
        println!("String in Pool: {}", s.as_str());
        println!("Pool free slots: {}", pool.free_count());
    }

    println!("\n=== Final Global Stats ===");
    let final_stats = sys.stats();
    println!("Total count of alloc() calls: {}", final_stats.allocs);
    println!("Total count of dealloc() calls: {}", final_stats.deallocs);
    println!("Total bytes allocated: {}", final_stats.bytes_allocated);
    println!("Total bytes freed: {}", final_stats.bytes_freed);
    println!("Current leak/live size: {} bytes", final_stats.bytes_live);
}