# ZigZag

**A collection of explicit memory allocators and collections inspired by Zig.**

## Prerequisites

Before building the project, ensure you have the following installed:

- [Rust](https://www.rust-lang.org/tools/install) (latest stable version)
- A C compiler (like `gcc`, `clang`, or `msvc`) if you intend to use the C-ABI

## Steps to Build

**Clone the repository:**

First, clone the project repository to your local machine:

```bash
git clone https://github.com/OctoBanon-Main/zigzag.git
cd zigzag
```

**Build the project:**

To compile the library (including the static/dynamic libraries for C integration), run:

```bash
cargo build --release
```

The compiled artifacts will be available in the `target/release/` directory:

- Linux/macOS: `libzigzag.a` or `libzigzag.so`
- Windows: `zigzag.dll` and `zigzag.dll.lib`

## Usage

### In Rust

Add the library to your logic and use explicit allocators:

```rust
use zigzag::alloc::{system::SystemAllocator, arena::Arena};
use zigzag::collections::ZigVec;

let arena = Arena::new(SystemAllocator);
let mut v = ZigVec::new(&arena);
v.push(42);
```

### In C

Include the header file and link against the compiled library:

1. Copy include/zigzag.h to your project.
2. Compile and link with the library:

```bash
gcc main.c -I./include -L./target/release -lzigzag -o zigzag_demo
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
- Arena & Bump: Blazing fast linear allocation strategies.
- Pool Allocator: Fixed-size block allocation to prevent fragmentation.
- Zero-cost FFI: Seamless integration with C/C++ via stable ABI.

## License

This project is licensed under the MIT License.
