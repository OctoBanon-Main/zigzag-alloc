use zigzag::alloc::{
    system::SystemAllocator,
    arena::Arena,
    pool::PoolAllocator,
    counting::CountingAllocator,
};
use zigzag::collections::{ZigBox, ZigVec, ZigString};
use core::fmt::Write;

fn main() {
    let sys = CountingAllocator::new(SystemAllocator);
    
    println!("=== ZigBox & SystemAllocator ===");
    {
        let b = ZigBox::new(42, &sys).expect("Failed to alloc Box");
        println!("Box value: {}", *b);
    }
    let stats = sys.stats();
    println!("System stats after Box: {} bytes allocated\n", stats.bytes);

    println!("=== ZigVec & Arena (Linear Allocation) ===");
    {
        let arena = Arena::new(&sys);
        
        let mut v = ZigVec::new(&arena);
        for i in 0..5 {
            v.push(i * 10);
        }
        println!("Vec: {:?}", v.as_slice());
        
        arena.reset();
        println!("Arena reset performed");
    }

    println!("\n=== ZigString & Pool (Fixed Size Blocks) ===");
    {
        let pool = PoolAllocator::typed::<[u8; 64]>(&sys, 10)
            .expect("Failed to create Pool");
        
        let mut s = ZigString::new(&pool);
        write!(s, "Hello from {}!", "ZigZag").unwrap();
        
        println!("String in Pool: {}", s.as_str());
        println!("Pool free slots: {}", pool.free_count());
    }

    println!("\n=== Final Global Stats ===");
    let final_stats = sys.stats();
    println!("Total allocations: {}", final_stats.allocs);
    println!("Total deallocations: {}", final_stats.deallocs);
    println!("Total bytes handled: {}", final_stats.bytes);
}