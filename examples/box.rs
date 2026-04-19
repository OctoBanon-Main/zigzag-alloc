use zigzag::alloc::system::SystemAllocator;
use zigzag::collections::ExBox;

fn main() {
    let alloc = SystemAllocator;

    let boxed_int = ExBox::new(1337, &alloc).expect("Out of memory");

    println!("Value: {}", *boxed_int);
    
    let val = ExBox::unbox(boxed_int);
    println!("Unboxed: {}", val);
}