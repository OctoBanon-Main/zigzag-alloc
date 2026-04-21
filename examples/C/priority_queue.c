#include "zigzag.h"
#include <stdio.h>
#include <stdint.h>

// Comparison for Min-Heap: returns true if a > b
bool min_heap_less(void* a, void* b) {
    return (intptr_t)a > (intptr_t)b;
}

int main() {
    // Initialization
    zigzag_system_t* sys = zigzag_system_create();
    zigzag_alloc_t alloc = zigzag_system_as_alloc(sys);

    // Create PQ for pointers/integers
    zigzag_pq_t* pq = zigzag_pq_create(alloc, min_heap_less);

    // Push some "values" (casting ints to pointers for demo)
    zigzag_pq_push(pq, (void*)50);
    zigzag_pq_push(pq, (void*)10);
    zigzag_pq_push(pq, (void*)30);

    printf("Priority Queue length: %zu\n", zigzag_pq_len(pq));

    // Pop and print (should be in ascending order: 10, 30, 50)
    while (zigzag_pq_len(pq) > 0) {
        void* val = zigzag_pq_pop(pq);
        printf("Popped: %lld\n", (long long)(intptr_t)val);
    }

    // Cleanup
    zigzag_pq_destroy(pq);
    zigzag_system_destroy(sys);

    return 0;
}
