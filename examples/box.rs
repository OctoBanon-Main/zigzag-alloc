use zigzag::alloc::system::SystemAllocator;
use zigzag::collections::ZigBox;

fn main() {
    let alloc = SystemAllocator;

    let boxed_int = ZigBox::new(1337, &alloc).expect("Out of memory");

    println!("Value: {}", *boxed_int);
    
    let val = ZigBox::unbox(boxed_int);
    println!("Unboxed: {}", val);
}