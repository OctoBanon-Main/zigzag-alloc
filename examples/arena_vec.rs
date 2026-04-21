use zigzag_alloc::alloc::system::SystemAllocator;
use zigzag_alloc::alloc::arena::ArenaAllocator;
use zigzag_alloc::collections::ExVec;

fn main() {
    let sys = SystemAllocator;
    let arena = ArenaAllocator::new(&sys);

    let mut numbers = ExVec::new(&arena);

    for i in 1..=100 {
        numbers.push(i);
    }

    println!("Sum: {}", numbers.iter().sum::<i32>());
    
    arena.reset();
    println!("Arena memory reclaimed. Allocations count was: {}", arena.alloc_count());
}