#include "zigzag_alloc.h"
#include <stdio.h>

int main() {
    // 1. Create a System Allocator and wrap it in an Arena
    zigzag_system_t* sys = zigzag_system_create();
    zigzag_alloc_t sys_alloc = zigzag_system_as_alloc(sys);

    zigzag_arena_t* arena = zigzag_arena_create(); // Backed by SystemAllocator internally
    zigzag_alloc_t arena_alloc = zigzag_arena_as_alloc(arena);

    // 2. Create a Vector of integers (int32_t) using the Arena
    zigzag_vec_t* numbers = zigzag_vec_create(arena_alloc, sizeof(int32_t), _Alignof(int32_t));

    // 3. Push 100 elements
    for (int32_t i = 1; i <= 100; i++) {
        zigzag_vec_push(numbers, &i);
    }

    // 4. Calculate Sum
    int32_t sum = 0;
    for (size_t i = 0; i < zigzag_vec_len(numbers); i++) {
        sum += *(int32_t*)zigzag_vec_get(numbers, i);
    }

    printf("Sum: %d\n", sum);
    printf("Allocations in arena: %zu\n", zigzag_arena_alloc_count(arena));

    // 5. Cleanup: Resetting the arena reclaims all vector memory
    zigzag_arena_reset(arena);
    printf("Arena reset. Remaining items in vec: %zu\n", zigzag_vec_len(numbers)); // Logically cleared

    zigzag_vec_destroy(numbers);
    zigzag_arena_destroy(arena);
    zigzag_system_destroy(sys);

    return 0;
}
