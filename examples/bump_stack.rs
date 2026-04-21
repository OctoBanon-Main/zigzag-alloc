use std::sync::OnceLock;
use std::sync::Mutex;
use zigzag_alloc::alloc::bump::BumpAllocator;
use zigzag_alloc::collections::ExVec;

static BUMP: OnceLock<Mutex<BumpAllocator>> = OnceLock::new();

fn get_bump() -> &'static Mutex<BumpAllocator> {
    BUMP.get_or_init(|| {
        let buffer = Box::leak(vec![0u8; 1024].into_boxed_slice());
        Mutex::new(BumpAllocator::new(buffer))
    })
}

fn main() {
    let bump_mutex = get_bump();
    let mut bump = bump_mutex.lock().unwrap();

    {
        let mut stack_vec = ExVec::new(&*bump);
        
        stack_vec.push(10);
        stack_vec.push(20);
        stack_vec.push(30);

        println!("Bump usage: {}/1024", bump.used());
    } 

    bump.reset();
    
    println!("Bump reset. Usage: {}", bump.used());
}