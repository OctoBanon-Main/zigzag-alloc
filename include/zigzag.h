#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// --- Basic Types ---

/**
 * A raw representation of a Rust trait object for an Allocator.
 * Contains a pointer to the data and the vtable.
 */
typedef struct {
    void *data;
    void *vtable;
} zigzag_alloc_t;

/**
 * Statistics provided by the CountingAllocator.
 */
typedef struct {
    size_t allocs;   // Total number of allocations performed
    size_t deallocs; // Total number of deallocations performed
    size_t bytes;    // Current number of live bytes
} zigzag_alloc_stats_t;

// --- Opaque Handles ---

typedef struct ZigzagSystemAllocator   zigzag_system_t;
typedef struct ZigzagBumpAllocator     zigzag_bump_t;
typedef struct ZigzagArena             zigzag_arena_t;
typedef struct ZigzagPool              zigzag_pool_t;
typedef struct ZigzagCountingAllocator zigzag_counting_t;

typedef struct ZigzagVec               zigzag_vec_t;
typedef struct ZigzagString            zigzag_string_t;
typedef struct ZigzagHashMap           zigzag_hashmap_t;
typedef struct ZigzagPriorityQueue     zigzag_pq_t;
typedef struct ZigzagBoundedArray256   zigzag_ba256_t;

// --- Global Allocation Functions ---

/**
 * Allocate memory using a generic RawAllocHandle.
 */
void *zigzag_alloc(zigzag_alloc_t alloc, size_t size, size_t align);

/**
 * Deallocate memory using a generic RawAllocHandle.
 */
void  zigzag_dealloc(zigzag_alloc_t alloc, void *ptr, size_t size, size_t align);

// --- System Allocator ---

zigzag_system_t *zigzag_system_create(void);
void             zigzag_system_destroy(zigzag_system_t *ptr);
zigzag_alloc_t   zigzag_system_as_alloc(zigzag_system_t *ptr);

// --- Bump Allocator ---

/**
 * Create a bump allocator using a pre-allocated buffer.
 */
zigzag_bump_t *zigzag_bump_create(uint8_t *buf, size_t len);
void           zigzag_bump_destroy(zigzag_bump_t *ptr);
zigzag_alloc_t zigzag_bump_as_alloc(zigzag_bump_t *ptr);
void  *zigzag_bump_alloc(zigzag_bump_t *ptr, size_t size, size_t align);
void   zigzag_bump_reset(zigzag_bump_t *ptr);
size_t zigzag_bump_used(const zigzag_bump_t *ptr);
size_t zigzag_bump_remaining(const zigzag_bump_t *ptr);

// --- Arena Allocator ---

/**
 * An arena allocator that manages multiple chunks, backed by the System Allocator.
 */
zigzag_arena_t *zigzag_arena_create(void);
void           zigzag_arena_destroy(zigzag_arena_t *ptr);
zigzag_alloc_t zigzag_arena_as_alloc(zigzag_arena_t *ptr);
void  *zigzag_arena_alloc(zigzag_arena_t *ptr, size_t size, size_t align);
void   zigzag_arena_reset(zigzag_arena_t *ptr);
size_t zigzag_arena_alloc_count(const zigzag_arena_t *ptr);

// --- Pool Allocator ---

/**
 * Fixed-size block allocator to prevent fragmentation.
 */
zigzag_pool_t *zigzag_pool_create(size_t block_size, size_t block_align, size_t capacity);
void           zigzag_pool_destroy(zigzag_pool_t *ptr);
zigzag_alloc_t zigzag_pool_as_alloc(zigzag_pool_t *ptr);
void          *zigzag_pool_alloc(zigzag_pool_t *ptr);
void           zigzag_pool_dealloc(zigzag_pool_t *ptr, void *mem);
size_t         zigzag_pool_capacity(const zigzag_pool_t *ptr);
size_t         zigzag_pool_free_count(const zigzag_pool_t *ptr);

// --- Counting Allocator ---

/**
 * Wrapper allocator that tracks allocation statistics.
 */
zigzag_counting_t    *zigzag_counting_create(void);
void                  zigzag_counting_destroy(zigzag_counting_t *ptr);
zigzag_alloc_t        zigzag_counting_as_alloc(zigzag_counting_t *ptr);
zigzag_alloc_stats_t  zigzag_counting_stats(const zigzag_counting_t *ptr);
void                 *zigzag_counting_alloc(zigzag_counting_t *ptr, size_t size, size_t align);
void                  zigzag_counting_dealloc(zigzag_counting_t *ptr, void *mem, size_t size, size_t align);

// --- Collections: Vector and String ---

zigzag_vec_t *zigzag_vec_create(zigzag_alloc_t alloc, size_t elem_size, size_t elem_align);
void          zigzag_vec_destroy(zigzag_vec_t *ptr);
int           zigzag_vec_push(zigzag_vec_t *ptr, const void *elem); // Returns 1 on success, 0 on failure
int           zigzag_vec_pop(zigzag_vec_t *ptr, void *out);         // Returns 1 on success, 0 if empty
void         *zigzag_vec_get(const zigzag_vec_t *ptr, size_t idx);
size_t        zigzag_vec_len(const zigzag_vec_t *ptr);
size_t        zigzag_vec_capacity(const zigzag_vec_t *ptr);
void          zigzag_vec_clear(zigzag_vec_t *ptr);
void         *zigzag_vec_data(const zigzag_vec_t *ptr);

zigzag_string_t *zigzag_string_create(zigzag_alloc_t alloc);
void             zigzag_string_destroy(zigzag_string_t *ptr);
int              zigzag_string_push_str(zigzag_string_t *ptr, const uint8_t *s, size_t len);
int              zigzag_string_push_cstr(zigzag_string_t *ptr, const uint8_t *s);
const uint8_t   *zigzag_string_as_ptr(const zigzag_string_t *ptr); // Returns null-terminated C-string
size_t           zigzag_string_len(const zigzag_string_t *ptr);
void             zigzag_string_clear(zigzag_string_t *ptr);

// --- Collections: HashMap (Swiss-table inspired) ---

/**
 * Creates a HashMap where keys are size_t and values are void pointers.
 * @param hash_fn Function to generate a 64-bit hash from the key.
 * @param eq_fn Function to check equality between two keys.
 */
zigzag_hashmap_t *zigzag_hashmap_create(zigzag_alloc_t alloc,
                                        uint64_t (*hash_fn)(size_t),
                                        bool (*eq_fn)(size_t, size_t));
void              zigzag_hashmap_destroy(zigzag_hashmap_t *ptr);
int               zigzag_hashmap_insert(zigzag_hashmap_t *ptr, size_t key, void *val, void **out_old_val);
void             *zigzag_hashmap_get(const zigzag_hashmap_t *ptr, size_t key);
void             *zigzag_hashmap_remove(zigzag_hashmap_t *ptr, size_t key);
size_t            zigzag_hashmap_len(const zigzag_hashmap_t *ptr);

// --- Collections: PriorityQueue (Binary Heap) ---

/**
 * Creates a PriorityQueue for void pointers.
 * @param less_fn Comparison function (return true if a < b for a min-heap).
 */
zigzag_pq_t *zigzag_pq_create(zigzag_alloc_t alloc, bool (*less_fn)(void*, void*));
void         zigzag_pq_destroy(zigzag_pq_t *ptr);
int          zigzag_pq_push(zigzag_pq_t *ptr, void *val);
void        *zigzag_pq_pop(zigzag_pq_t *ptr);
void        *zigzag_pq_peek(const zigzag_pq_t *ptr);
size_t       zigzag_pq_len(const zigzag_pq_t *ptr);

// --- Collections: BoundedArray (Fixed Capacity Stack-allocated) ---

/**
 * Creates a heap-allocated wrapper for a 256-byte stack-style array.
 * Includes SIMD optimizations for byte-level operations.
 */
zigzag_ba256_t *zigzag_ba256_create(void);
void            zigzag_ba256_destroy(zigzag_ba256_t *ptr);
int             zigzag_ba256_push(zigzag_ba256_t *ptr, uint8_t val);
int             zigzag_ba256_pop(zigzag_ba256_t *ptr, uint8_t *out);
size_t          zigzag_ba256_len(const zigzag_ba256_t *ptr);
void            zigzag_ba256_fill_bytes(zigzag_ba256_t *ptr, uint8_t val); // Optimized with SIMD
size_t          zigzag_ba256_count_byte(const zigzag_ba256_t *ptr, uint8_t val);
uint8_t        *zigzag_ba256_data(zigzag_ba256_t *ptr);

#ifdef __cplusplus
}
#endif
