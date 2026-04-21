#include "zigzag.h"
#include <stdio.h>

// simple hash function for size_t keys
uint64_t my_hash(size_t key) {
    return (uint64_t)key ^ 0x9e3779b9;
}

bool my_eq(size_t a, size_t b) {
    return a == b;
}

int main() {
    zigzag_system_t* sys = zigzag_system_create();
    zigzag_alloc_t alloc = zigzag_system_as_alloc(sys);

    // Create a HashMap: Key=size_t, Value=void*
    zigzag_hashmap_t* map = zigzag_hashmap_create(alloc, my_hash, my_eq);

    // Insert data (keys are treated as size_t, values are pointers)
    size_t key = 42;
    char* value = "ZigZag is fast";
    zigzag_hashmap_insert(map, key, value, NULL);

    // Retrieve data
    char* found = (char*)zigzag_hashmap_get(map, 42);
    if (found) {
        printf("Found in Map: %s\n", found);
    }

    printf("Map len: %zu\n", zigzag_hashmap_len(map));

    zigzag_hashmap_destroy(map);
    zigzag_system_destroy(sys);
    return 0;
}
