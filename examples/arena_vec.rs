use zigzag::alloc::system::SystemAllocator;
use zigzag::alloc::arena::Arena;
use zigzag::collections::ZigVec;

fn main() {
    let sys = SystemAllocator;
    let arena = Arena::new(&sys);

    let mut numbers = ZigVec::new(&arena);

    for i in 1..=100 {
        numbers.push(i);
    }

    println!("Sum: {}", numbers.iter().sum::<i32>());
    
    arena.reset();
    println!("Arena memory reclaimed. Allocations count was: {}", arena.alloc_count());
}