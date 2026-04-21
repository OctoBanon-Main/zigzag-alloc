# ZigZag Alloc

**A collection of explicit memory allocators and collections inspired by Zig.**

## Prerequisites

Before building the project, ensure you have the following installed:

- [Rust](https://www.rust-lang.org/tools/install) (latest stable version)
- A C compiler (like `gcc`, `clang`, or `msvc`) if you intend to use the C-ABI

## Steps to Build

**Clone the repository:**

First, clone the project repository to your local machine:

```bash
git clone https://github.com/OctoBanon-Main/zigzag-alloc.git
cd zigzag-alloc
```

**Build the project:**

To compile the library (including the static/dynamic libraries for C integration), run:

```bash
cargo build --release
```

The compiled artifacts will be available in the `target/release/` directory:

- Linux/macOS: `libzigzag_alloc.a` or `libzigzag_alloc.so`
- Windows: `zigzag_alloc.dll` and `zigzag_alloc.dll.lib`

## Memory Management Strategies

| Allocator  | Thread-Safe | Strategy      | Best Use Case                                           |
|------------|-------------|---------------|---------------------------------------------------------|
| `System`   | Yes         | OS native     | Base allocator for long-lived structures.               |
| `Bump`     | Yes         | Linear offset | Extremely fast, short-lived temp buffers.               |
| `Arena`    | No          | Linked blocks | Batch-processing tasks (request/frame lifetime).        |
| `Pool`     | Yes         | Free-list     | Uniform objects (nodes, packets) with no fragmentation. |
| `Counting` | No          | Wrapper       | Profiling, leak detection, and memory metrics.          |

## Usage

### In Rust

1. Add the dependency

Add this to your Cargo.toml:

```toml
[dependencies]
zigzag-alloc = "^1.0.0"
```

Or use the cargo CLI:

```bash
cargo add zigzag-alloc
```

2. Basic Usage

The core philosophy is explicit allocation. Unlike standard collections, you must provide an allocator reference when creating a container.

```rust
use zigzag_alloc::alloc::{system::SystemAllocator, arena::ArenaAllocator};
use zigzag_alloc::ExVec;

fn main() {
    // 1. Initialize a backing allocator (e.g., System)
    let sys = SystemAllocator;

    // 2. Create a fast Arena for batch allocations
    // The arena is tied to the system allocator's lifetime.
    let arena = ArenaAllocator::new(&sys);

    // 3. Bind the collection to the arena
    // All pushes will now use the arena's memory.
    let mut v = ExVec::new(&arena);
    
    v.push(42);
    v.push(1337);

    println!("Vector: {:?}", v.as_slice());

    // When 'arena' goes out of scope, all memory is reclaimed at once.
}
```

### In C

Include the header file and link against the compiled library:

1. Copy include/zigzag.h to your project.
2. Compile and link with the library:

```bash
gcc main.c -I./include -L./target/release -lzigzag_alloc -o zigzag_demo
```

Example usage in C:

```C
#include "zigzag.h"

zigzag_system_t* sys = zigzag_system_create();
zigzag_alloc_t alloc = zigzag_system_as_alloc(sys);
// Now you can pass 'alloc' to zigzag collections
```

## Features

- Manual Memory Control: Explicitly pass allocators to containers.
- High-Performance SIMD: Optimized backends for x86_64 (AVX2/SSE2) and AArch64 (NEON) with SWAR fallback.
- Arena & Bump: Blazing fast linear allocation strategies.
- Pool Allocator: Fixed-size block allocation to prevent fragmentation.
- Zero-cost FFI: Seamless integration with C/C++ via stable ABI.

## License

This project is licensed under the MIT license.
