#include "zigzag_alloc.h"
#include <stdio.h>

int main() {
    // Create a stack-style array wrapper (fixed 256 bytes capacity)
    zigzag_ba256_t* tasks = zigzag_ba256_create();

    // Push bytes
    zigzag_ba256_push(tasks, 10);
    zigzag_ba256_push(tasks, 20);
    zigzag_ba256_push(tasks, 10); // Duplicate for counting

    // SIMD-optimized byte counting
    size_t tens = zigzag_ba256_count_byte(tasks, 10);
    printf("Occurrences of byte 10: %zu\n", tens);

    // SIMD-optimized fill
    zigzag_ba256_fill_bytes(tasks, 0xFF);
    printf("Array filled. First byte: 0x%02X\n", zigzag_ba256_data(tasks)[0]);

    zigzag_ba256_destroy(tasks);
    return 0;
}
