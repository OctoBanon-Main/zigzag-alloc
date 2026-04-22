#include <stdio.h>
#include <stdint.h>
#include "zigzag_alloc.h"

static uint8_t MEMORY[1024];

int main() {
    zigzag_bump_t* bump = zigzag_bump_create(MEMORY, sizeof(MEMORY));
    
    zigzag_alloc_t alloc = zigzag_bump_as_alloc(bump);

    zigzag_vec_t* stack_vec = zigzag_vec_create(alloc, sizeof(int), _Alignof(int));

    for (int i = 1; i <= 3; i++) {
        int value = i * 10;
        zigzag_vec_push(stack_vec, &value);
    }

    printf("Bump usage: %zu/%zu\n", 
           zigzag_bump_used(bump), 
           sizeof(MEMORY));

    printf("Vector len: %zu\n", zigzag_vec_len(stack_vec));

    zigzag_vec_destroy(stack_vec);

    zigzag_bump_reset(bump);
    printf("Bump reset. Usage: %zu\n", zigzag_bump_used(bump));

    zigzag_bump_destroy(bump);

    return 0;
}