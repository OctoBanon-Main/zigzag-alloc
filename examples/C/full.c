#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "zigzag_alloc.h"

int main() {
    zigzag_counting_t *sys_counter = zigzag_counting_create();
    zigzag_alloc_t sys_alloc = zigzag_counting_as_alloc(sys_counter);

    printf("=== Manual Allocation & CountingAllocator ===\n");
    {
        int *val = (int*)zigzag_alloc(sys_alloc, sizeof(int), _Alignof(int));
        if (val) {
            *val = 42;
            printf("Value: %d\n", *val);
            zigzag_dealloc(sys_alloc, val, sizeof(int), _Alignof(int));
        }
    }

    zigzag_alloc_stats_t stats = zigzag_counting_stats(sys_counter);
    printf("Allocated: %zu bytes\n", stats.bytes);
    printf("Allocs count: %zu\n\n", stats.allocs);

    printf("=== ExVec & Arena (Linear Allocation) ===\n");
    {
        zigzag_arena_t *arena = zigzag_arena_create();
        zigzag_alloc_t arena_alloc = zigzag_arena_as_alloc(arena);

        zigzag_vec_t *v = zigzag_vec_create(arena_alloc, sizeof(int), _Alignof(int));
        
        for (int i = 0; i < 5; i++) {
            int val = i * 10;
            zigzag_vec_push(v, &val);
        }

        printf("Vec: [");
        for (size_t i = 0; i < zigzag_vec_len(v); i++) {
            printf("%d%s", *(int*)zigzag_vec_get(v, i), (i < 4 ? ", " : ""));
        }
        printf("]\n");

        zigzag_arena_reset(arena);
        printf("Arena reset performed\n");

        zigzag_vec_destroy(v);
        zigzag_arena_destroy(arena);
    }

    printf("\n=== ExString & Pool (Fixed Size Blocks) ===\n");
    {
        size_t block_size = 64;
        zigzag_pool_t *pool = zigzag_pool_create(block_size, 8, 10);
        zigzag_alloc_t pool_alloc = zigzag_pool_as_alloc(pool);

        zigzag_string_t *s = zigzag_string_create(pool_alloc);
        const char *msg = "Hello from ZigZag!";
        zigzag_string_push_str(s, (const uint8_t*)msg, strlen(msg));

        printf("String in Pool: %s\n", (const char*)zigzag_string_as_ptr(s));
        printf("Pool free slots: %zu\n", zigzag_pool_free_count(pool));

        zigzag_string_destroy(s);
        zigzag_pool_destroy(pool);
    }

    printf("\n=== Final Global Stats ===\n");
    zigzag_alloc_stats_t final_stats = zigzag_counting_stats(sys_counter);
    printf("Total count of alloc() calls: %zu\n", final_stats.allocs);
    printf("Total count of dealloc() calls: %zu\n", final_stats.deallocs);
    printf("Total bytes processed: %zu\n", final_stats.bytes);

    zigzag_counting_destroy(sys_counter);

    return 0;
}