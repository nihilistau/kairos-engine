/* gguf.h — GGUF v3 model-file loader (shared across all engine backends).
 *
 * Memory-maps a .gguf file and parses the header, metadata key/values, and
 * tensor table into a flat, backend-agnostic descriptor. Tensor weight bytes
 * stay paged in the mapping; gguf_tensor_data() returns a pointer into it.
 *
 * Format reference: the GGUF v3 layout —
 *   magic "GGUF" (u32 LE) | version (u32) | tensor_count (u64) | kv_count (u64)
 *   kv_count   metadata entries: key(gguf_str) value_type(u32) value(typed)
 *   tensor_cnt tensor infos:     name(gguf_str) n_dims(u32) dims[u64] type(u32) offset(u64)
 *   padding to general.alignment (default 32)
 *   tensor data (each tensor at data_base + tensor.offset)
 * gguf_str = len(u64) + bytes (no NUL).
 */
#ifndef SP_ENGINE_GGUF_H
#define SP_ENGINE_GGUF_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* GGUF metadata value types. */
typedef enum {
    GGUF_T_UINT8 = 0, GGUF_T_INT8 = 1, GGUF_T_UINT16 = 2, GGUF_T_INT16 = 3,
    GGUF_T_UINT32 = 4, GGUF_T_INT32 = 5, GGUF_T_FLOAT32 = 6, GGUF_T_BOOL = 7,
    GGUF_T_STRING = 8, GGUF_T_ARRAY = 9, GGUF_T_UINT64 = 10, GGUF_T_INT64 = 11,
    GGUF_T_FLOAT64 = 12
} gguf_value_type;

/* ggml tensor dtypes (subset the engine handles; values are the on-disk ids). */
typedef enum {
    GGML_T_F32 = 0, GGML_T_F16 = 1, GGML_T_Q4_0 = 2, GGML_T_Q4_1 = 3,
    GGML_T_Q5_0 = 6, GGML_T_Q5_1 = 7, GGML_T_Q8_0 = 8, GGML_T_Q8_1 = 9,
    GGML_T_Q4_K = 12, GGML_T_Q5_K = 13, GGML_T_Q6_K = 14, GGML_T_Q8_K = 15,
    GGML_T_BF16 = 30, GGML_T_COUNT = 40
} ggml_type;

#define GGUF_MAX_DIMS 4
#define GGUF_NAME_MAX 256

typedef struct {
    char     name[GGUF_NAME_MAX];
    uint32_t n_dims;
    uint64_t dims[GGUF_MAX_DIMS];
    uint32_t type;            /* ggml_type */
    uint64_t offset;          /* relative to the tensor-data base */
    uint64_t n_elements;      /* product of dims */
    uint64_t nbytes;          /* byte size on disk (incl. quant block overhead) */
} gguf_tensor;

typedef struct {
    char            key[GGUF_NAME_MAX];
    gguf_value_type type;
    /* scalar value (valid for the matching scalar type) */
    union { uint64_t u; int64_t i; double f; } scalar;
    /* string value (type==STRING): NUL-terminated copy, else NULL */
    char           *str;
    /* array value (type==ARRAY): element type/count + raw element bytes */
    gguf_value_type arr_type;
    uint64_t        arr_len;
    const void     *arr_data;   /* points into the mapping */
} gguf_kv;

typedef struct gguf_ctx gguf_ctx;

/* Open + parse. Returns NULL on any error (bad magic/version, truncation,
 * out-of-bounds tensor, alignment violation). */
gguf_ctx *gguf_open(const char *path);
void      gguf_close(gguf_ctx *ctx);

/* Release just the file mapping (the large weight data) while keeping the parsed
 * tensor table and metadata (names, dims, scalar/string KVs) so the descriptor
 * stays usable for lookups. After this, gguf_tensor_data() returns NULL and a
 * STRING-array KV's element bytes (arr_data) are invalid — every consumer of the
 * mapping (weights, embedding, norms, tokenizer vocab) must have copied what it
 * needs first. Used by the packed-weight arena to drop the F16 source (§4.8).
 * gguf_close() remains safe to call afterwards. */
void      gguf_release_data(gguf_ctx *ctx);

uint32_t gguf_version(const gguf_ctx *ctx);
uint64_t gguf_n_tensors(const gguf_ctx *ctx);
uint64_t gguf_n_kv(const gguf_ctx *ctx);
uint64_t gguf_alignment(const gguf_ctx *ctx);
uint64_t gguf_data_offset(const gguf_ctx *ctx);   /* file offset of tensor data */
uint64_t gguf_file_size(const gguf_ctx *ctx);

const gguf_tensor *gguf_tensor_at(const gguf_ctx *ctx, uint64_t i);
const gguf_tensor *gguf_find_tensor(const gguf_ctx *ctx, const char *name);
/* pointer to a tensor's bytes inside the mapping (NULL if out of range). */
const void        *gguf_tensor_data(const gguf_ctx *ctx, const gguf_tensor *t);

const gguf_kv *gguf_kv_at(const gguf_ctx *ctx, uint64_t i);
const gguf_kv *gguf_find_kv(const gguf_ctx *ctx, const char *key);
/* typed scalar getters: return 1 and write *out if present with a compatible
 * type, else 0. Integer getters accept any integer width. */
int          gguf_get_u64(const gguf_ctx *ctx, const char *key, uint64_t *out);
int          gguf_get_f32(const gguf_ctx *ctx, const char *key, float *out);
const char  *gguf_get_str(const gguf_ctx *ctx, const char *key); /* NULL if absent */
/* One-pass safe walk of a STRING-array KV (e.g. tokenizer.ggml.tokens/merges):
 * fills `ptrs`/`lens` (capacity `cap`) with pointers into the mapping and lengths
 * (strings are NOT NUL-terminated). Returns the number written (min(arr_len,cap),
 * fewer if a bound is hit). 0 if the KV is missing or not a string array. */
uint64_t gguf_kv_str_array(const gguf_ctx *ctx, const gguf_kv *kv,
                           const char **ptrs, uint64_t *lens, uint64_t cap);

/* bytes-per-element-block helpers for a ggml_type (for size validation). */
const char *ggml_type_name(uint32_t type);

#ifdef __cplusplus
}
#endif
#endif /* SP_ENGINE_GGUF_H */
