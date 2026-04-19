#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    void *data;
    void *vtable;
} zigzag_alloc_t;

typedef struct {
    size_t allocs;
    size_t deallocs;
    size_t bytes;
} zigzag_alloc_stats_t;

typedef struct ZigzagSystemAllocator   zigzag_system_t;
typedef struct ZigzagBumpAllocator     zigzag_bump_t;
typedef struct ZigzagArena             zigzag_arena_t;
typedef struct ZigzagPool              zigzag_pool_t;
typedef struct ZigzagCountingAllocator zigzag_counting_t;
typedef struct ZigzagVec               zigzag_vec_t;
typedef struct ZigzagString            zigzag_string_t;

void *zigzag_alloc(zigzag_alloc_t alloc, size_t size, size_t align);

void  zigzag_dealloc(zigzag_alloc_t alloc, void *ptr, size_t size, size_t align);

zigzag_system_t *zigzag_system_create(void);
void             zigzag_system_destroy(zigzag_system_t *ptr);
zigzag_alloc_t   zigzag_system_as_alloc(zigzag_system_t *ptr);

zigzag_bump_t *zigzag_bump_create(uint8_t *buf, size_t len);
void           zigzag_bump_destroy(zigzag_bump_t *ptr);
zigzag_alloc_t zigzag_bump_as_alloc(zigzag_bump_t *ptr);

void  *zigzag_bump_alloc(zigzag_bump_t *ptr, size_t size, size_t align);

void   zigzag_bump_reset(zigzag_bump_t *ptr);

size_t zigzag_bump_used(const zigzag_bump_t *ptr);
size_t zigzag_bump_remaining(const zigzag_bump_t *ptr);

zigzag_arena_t *zigzag_arena_create(void);

void           zigzag_arena_destroy(zigzag_arena_t *ptr);
zigzag_alloc_t zigzag_arena_as_alloc(zigzag_arena_t *ptr);

void  *zigzag_arena_alloc(zigzag_arena_t *ptr, size_t size, size_t align);

void   zigzag_arena_reset(zigzag_arena_t *ptr);
size_t zigzag_arena_alloc_count(const zigzag_arena_t *ptr);

zigzag_pool_t *zigzag_pool_create(size_t block_size, size_t block_align, size_t capacity);

void zigzag_pool_destroy(zigzag_pool_t *ptr);

zigzag_alloc_t zigzag_pool_as_alloc(zigzag_pool_t *ptr);

void *zigzag_pool_alloc(zigzag_pool_t *ptr);

void   zigzag_pool_dealloc(zigzag_pool_t *ptr, void *mem);
size_t zigzag_pool_capacity(const zigzag_pool_t *ptr);

size_t zigzag_pool_free_count(const zigzag_pool_t *ptr);

zigzag_counting_t    *zigzag_counting_create(void);
void                  zigzag_counting_destroy(zigzag_counting_t *ptr);
zigzag_alloc_t        zigzag_counting_as_alloc(zigzag_counting_t *ptr);
zigzag_alloc_stats_t  zigzag_counting_stats(const zigzag_counting_t *ptr);
void                 *zigzag_counting_alloc(zigzag_counting_t *ptr, size_t size, size_t align);
void                  zigzag_counting_dealloc(zigzag_counting_t *ptr, void *mem,
                                              size_t size, size_t align);

zigzag_vec_t *zigzag_vec_create(zigzag_alloc_t alloc, size_t elem_size, size_t elem_align);

void zigzag_vec_destroy(zigzag_vec_t *ptr);

int zigzag_vec_push(zigzag_vec_t *ptr, const void *elem);

int zigzag_vec_pop(zigzag_vec_t *ptr, void *out);

void  *zigzag_vec_get(const zigzag_vec_t *ptr, size_t idx);
size_t zigzag_vec_len(const zigzag_vec_t *ptr);
size_t zigzag_vec_capacity(const zigzag_vec_t *ptr);

void zigzag_vec_clear(zigzag_vec_t *ptr);

void *zigzag_vec_data(const zigzag_vec_t *ptr);

zigzag_string_t *zigzag_string_create(zigzag_alloc_t alloc);
void             zigzag_string_destroy(zigzag_string_t *ptr);

int zigzag_string_push_str(zigzag_string_t *ptr, const uint8_t *s, size_t len);

int zigzag_string_push_cstr(zigzag_string_t *ptr, const uint8_t *s);

const uint8_t *zigzag_string_as_ptr(const zigzag_string_t *ptr);

size_t zigzag_string_len(const zigzag_string_t *ptr);

void zigzag_string_clear(zigzag_string_t *ptr);


#ifdef __cplusplus
}
#endif