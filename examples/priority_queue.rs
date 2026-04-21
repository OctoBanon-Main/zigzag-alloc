use zigzag_alloc::alloc::system::SystemAllocator;
use zigzag_alloc::collections::{ExPriorityQueue, OrdContext};

struct MinIntContext;
impl OrdContext<i32> for MinIntContext {
    fn less(&self, a: &i32, b: &i32) -> bool {
        a > b
    }
}

fn main() {
    let alloc = SystemAllocator;
    let ctx = MinIntContext;

    let mut pq = ExPriorityQueue::new(&alloc, ctx);

    pq.push(50);
    pq.push(10);
    pq.push(30);

    let data = [5, 1, 100];
    pq.push_slice(&data);

    while let Some(top) = pq.pop() {
        println!("Pop: {}", top);
    }
}