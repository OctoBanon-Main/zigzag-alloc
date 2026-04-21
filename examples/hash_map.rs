use zigzag::alloc::system::SystemAllocator;
use zigzag::collections::{ExHashMap, U64HashCtx};

fn main() {
    let alloc = SystemAllocator;
    let ctx = U64HashCtx;
    
    let mut map = ExHashMap::new(&alloc, ctx);

    map.insert(42, 100);
    map.insert(1337, 9000);

    if let Some(val) = map.get(&42) {
        println!("Found: {}", val);
    }

    map.remove(&1337);

    println!("Map len: {}, capacity: {}", map.len(), map.capacity());
}