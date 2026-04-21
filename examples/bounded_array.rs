use zigzag_alloc::collections::bounded_array::ExBoundedArray;

fn main() {
    let mut tasks = ExBoundedArray::<u32, 16>::new();

    tasks.push(10).expect("Buffer overflow");
    tasks.push(20).expect("Buffer overflow");
    
    let extra = [30, 40, 50];
    if tasks.remaining() >= extra.len() {
        tasks.push_slice(&extra).unwrap();
    }

    for task in tasks.iter() {
        println!("Task ID: {}", task);
    }

    println!("Total len: {}/{}", tasks.len(), tasks.capacity());
}