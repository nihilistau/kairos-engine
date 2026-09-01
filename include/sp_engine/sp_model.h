/* sp_model.h — the .sp-model / .sp-tokenizer on-disk format + loader ABI.
 * Implements PPT-LAT-SP-MODEL-v0 Appendix B. CPU-only Phase 2-FMT.
 *
 * Two layers:
 *   1. The frozen byte structs (header, tensor entry, tokenizer header) + dtype
 *      enum, used by both the loader (src/io/sp_model_load.c) and the transcoder
 *      (tools/sp_transcode). These are #pragma pack(1) so sizeof matches the spec
 *      offsets exactly and a single memcpy maps file bytes onto the struct.
 *   2. sp_model_load / sp_model_unload — the L1 ABI handle: pure mmap + header
 *      parse + tensor-table pointer setup. ZERO malloc proportional to tensor
 *      data; the file IS the in-memory layout (§1).
 *
 * A separate adapter (sp_model_to_qwen3) reconstructs a qwen3_model the existing
 * gemma3_forward / qwen3_forward consume unchanged — that is NOT part of the ABI
 * handle and lives in src/io/sp_model_adapter.c.
 */
#ifndef SP_ENGINE_SP_MODEL_H
#define SP_ENGINE_SP_MODEL_H

#include <stdint.h>
#include <stddef.h>
#include "sp_engine/sp_status.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ── magic / version ── */
#define SP_MODEL_MAGIC   0x444D5053u   /* "SPMD" little-endian */
#define SP_TOK_MAGIC     0x4B545053u   /* "SPTK" little-endian */
#define SP_MODEL_VER_MAJOR 0u
#define SP_MODEL_VER_MINOR 1u
#define SP_HEADER_SIZE     512u
#define SP_TOK_HEADER_SIZE 128u
#define SP_TENSOR_ENTRY_SIZE 256u
#define SP_DATA_REGION_ALIGN 65536u    /* §2: one Win32 allocation-granularity unit */
#define SP_TENSOR_ALIGN      64u       /* §2: each tensor 64-aligned within the region */
#define SP_HEADER_CRC_COVER  360u      /* §3: CRC over bytes [0,360) */
#define SP_TOK_CRC_COVER     52u       /* §7: CRC over bytes [0,52) */
#define SP_SPINOR_SENTINEL   0xA5u     /* §6: byte 63 of each on-disk Spinor block */

/* §5 dtype_id enum. */
typedef enum {
    SP_DT_F32                  = 1,
    SP_DT_F16                  = 2,
    SP_DT_BF16                 = 3,
    SP_DT_OK_Q8                = 10,   /* O_K-lifted int8 + per-row Frobenius scale */
    SP_DT_OK_Q4                = 11,
    SP_DT_FROBENIUS_SCALE_FP32 = 12,   /* "<weight>.scale" companion to a Q8/Q4 tensor */
    SP_DT_OK_Q4B               = 13,   /* block-scaled Q4: int4 codes [-7,7] nibble-packed,
                                          per-32-block f16 scales in ".bscale" sibling
                                          (SPEC OK_Q4B, CONTRACT-SPEED 2026-06-07) */
    SP_DT_BLOCK_SCALE_FP16     = 14,   /* "<weight>.bscale": f16[rows * ceil(cols/32)] */
    SP_DT_SPINOR63             = 20,
    SP_DT_RING_RESIDUE_CRT_30_30 = 30,
    SP_DT_OK_INTEGER           = 31
} sp_dtype_id;

/* §3 arch_id enum (mirrors the engine's sp_arch_t but on the wire). */
typedef enum {
    SP_ARCH_ID_LLAMA3 = 1, SP_ARCH_ID_QWEN3 = 2, SP_ARCH_ID_GEMMA3 = 3,
    SP_ARCH_ID_DEEPSEEK_V4 = 4,
    /* 5 = SP_ARCH_ID_QWEN35 reserved (Phase 3-SSM) */
    SP_ARCH_ID_QWEN25 = 6,
    SP_ARCH_ID_GEMMA4 = 7,
    SP_ARCH_ID_QWEN36 = 8,  /* qwen35moe: Gated DeltaNet + MoE hybrid (Qwen3.6-35B-A3B) */
    SP_ARCH_ID_DIFFUSION_GEMMA = 9, /* diffusion-gemma: block masked-diffusion on the
                                       Gemma-4 MoE backbone (DiffusionGemma-26B-A4B; PR 24423).
                                       Backbone == gemma4 MoE; diffusion surface in dg_* fields. */
    SP_ARCH_ID_GEMMA4_ASSISTANT = 10 /* gemma4-assistant: EAGLE/MTP draft head. 4 tiny layers
                                        (hidden 1024) hung off dense model's residual: nextn.pre_projection
                                        [2*3840->1024] consumes concat(target_hidden, token_embed),
                                        nextn.post_projection [1024->3840] returns to the residual,
                                        shares dense model's embd+head. A spec-decode DRAFT, not a stand-
                                        alone LM (must be fed the target's last hidden each step). */,
    SP_ARCH_ID_GEMMA4_MOE = 11 /* gemma4 backbone + routed MoE FFN, AUTOREGRESSIVE.
                                  Gemma-4-26B-A4B: 128 experts / top-8, alongside a per-layer DENSE
                                  shared FFN, fused ffn_gate_up_exps [n_embd, 2*n_ff_exp, n_expert]
                                  + ffn_down_exps. Distinct from DIFFUSION_GEMMA (9), which is the
                                  SAME backbone under a block masked-diffusion decoder; this one
                                  keeps the ordinary gemma4 AR decode (KV ring / shear / persist).
                                  MoE geometry reuses the q36_n_expert* slots (the documented
                                  contract: backbone MoE geometry lives there). Global-attention
                                  layers may omit attn_v -> V aliases K (llama.cpp
                                  "use_alternative_attention"). */
} sp_arch_id;

/* §7 tokenizer type_id. Doubles as the tokenizer FAMILY tag (#115): legacy
 * blobs keep their old values/meaning; readers must HARD-ERROR on values they
 * do not implement (never silently fall back to another family). */
typedef enum {
    SP_TOK_SENTENCEPIECE = 0, SP_TOK_BPE_LLAMA3 = 1, SP_TOK_BPE_GPT2 = 2,
    SP_TOK_TIKTOKEN_O200K = 3,
    SP_TOK_GEMMA4_BPE = 4   /* gemma4 514k-merge U+2581-piece BPE (issue #115) */
} sp_tok_type_id;

#pragma pack(push, 1)

/* §3 — fixed 512-byte file header. */
typedef struct {
    uint32_t magic;                 /* 0   "SPMD" */
    uint16_t version_major;         /* 4 */
    uint16_t version_minor;         /* 6 */
    uint32_t header_size;           /* 8   == 512 */
    uint32_t arch_id;               /* 12 */
    uint32_t arch_struct_size;      /* 16  bytes of arch payload actually used */
    uint32_t arch_struct_capacity;  /* 20  == 256 */
    uint8_t  arch_struct[256];      /* 24  memcpy-direct payload: sp_arch_info (sp_l1.h; E_PARITY_3) */
    uint8_t  tokenizer_hash[32];    /* 280 SHA-256 of paired .sp-tokenizer file */
    uint32_t vocab_size;            /* 312 */
    uint32_t tensor_count;          /* 316 */
    uint64_t tensor_table_offset;   /* 320 == 512 */
    uint64_t tensor_data_offset;    /* 328 multiple of 65536 */
    uint64_t file_size;             /* 336 */
    uint64_t created_unix_seconds;  /* 344 */
    uint64_t transcoded_from;       /* 352 hash of upstream path; 0 if native */
    uint32_t header_crc32;          /* 360 CRC over [0,360) */
    uint8_t  reserved[148];         /* 364 zero-filled */
} sp_model_header;                  /* sizeof == 512 */

/* §4 — 256-byte tensor table entry. */
typedef struct {
    char     name[80];              /* 0   null-terminated */
    uint32_t dtype_id;              /* 80 */
    uint32_t n_dims;                /* 84  rank 1..8 */
    uint64_t dims[8];               /* 88  shape in elements (unused = 0) */
    uint64_t offset_in_data;        /* 152 byte offset from tensor_data_offset; %64==0 */
    uint64_t size_bytes;            /* 160 on-disk byte length */
    uint32_t block_size;            /* 168 on-disk bytes per block */
    uint32_t block_count;           /* 172 size_bytes / block_size */
    uint8_t  blake3[32];            /* 176 per-tensor digest */
    uint64_t name_hash;             /* 208 xxh64(name); table sorted ascending */
    uint8_t  reserved[40];          /* 216 zero-filled */
} sp_tensor_entry;                  /* sizeof == 256 */

/* §7 — 128-byte .sp-tokenizer header. */
typedef struct {
    uint32_t magic;                 /* 0   "SPTK" */
    uint16_t version_major;         /* 4 */
    uint16_t version_minor;         /* 6 */
    uint32_t header_size;           /* 8   == 128 */
    uint32_t type_id;               /* 12  sp_tok_type_id */
    uint32_t vocab_size;            /* 16 */
    uint32_t bos_token;             /* 20  or 0xFFFFFFFF */
    uint32_t eos_token;             /* 24 */
    uint32_t pad_token;             /* 28 */
    uint32_t unk_token;             /* 32 */
    uint64_t blob_offset;           /* 36 */
    uint64_t blob_size;             /* 44 */
    uint32_t header_crc32;          /* 52  CRC over [0,52) */
    uint8_t  reserved[72];          /* 56  zero-filled */
} sp_tok_header;                    /* sizeof == 128 */

#pragma pack(pop)

/* ── L1 ABI handle (E_FMT_1) ── */
typedef struct sp_model sp_model;

/* Load a .sp-model + paired .sp-tokenizer. Pure mmap + parse: verifies magic,
 * version_major==0, header CRC-32 over [0,360), tensor-table/data alignment,
 * file_size, and tokenizer-file SHA-256 == header.tokenizer_hash. Returns
 *   SP_OK on success, SP_EBADFORMAT (magic/version/struct), SP_EIO (open/mmap),
 *   SP_ETOKENIZER_HASH (sha mismatch), SP_EVOCAB (vocab mismatch).
 * No allocation proportional to tensor data; *out is a small heap handle holding
 * pointers into the mmap regions. */
sp_status sp_model_load(const char *sp_model_path, const char *sp_tokenizer_path,
                        sp_model **out);
void      sp_model_unload(sp_model *m);

/* Accessors over the loaded handle (pointers into the mmap; valid until unload). */
const sp_model_header *sp_model_get_header(const sp_model *m);
uint32_t               sp_model_tensor_count(const sp_model *m);
const sp_tensor_entry *sp_model_tensor_at(const sp_model *m, uint32_t i);
/* O(log N) binary search by xxh64(name) + name verify; NULL if absent. */
const sp_tensor_entry *sp_model_find_tensor(const sp_model *m, const char *name);
/* Pointer to a tensor's first byte inside the data mmap. */
const void            *sp_model_tensor_data(const sp_model *m, const sp_tensor_entry *e);
/* Tokenizer blob pointer + size inside the tokenizer mmap. */
const void            *sp_model_tokenizer_blob(const sp_model *m, uint64_t *size_out);

/* ── adapter (src/io/sp_model_adapter.c) ──
 * Reconstruct a qwen3_model the existing gemma3_forward / qwen3_forward consume
 * unchanged: matmul weights + embedding from the .sp-model OK_Q8 codes/scales
 * (rebuilt into a packed arena; row codes/scales memcpy'd out of the mmap so the
 * arena owns its buffers), norms as owned f32. The returned model has gguf==NULL
 * and released==1; free it with qwen3_free. NOT part of the L1 ABI handle.
 * Returns NULL on error (sp_last_error has detail). The sp_model handle must
 * outlive the returned qwen3_model (norms/embedding are copied; nothing else
 * borrows the mmap, so in practice it is self-contained). */
struct qwen3_model;
struct qwen3_model *sp_model_to_qwen3(const sp_model *m);
struct qwen3_model *sp_model_to_qwen25(const sp_model *m);
struct qwen3_model *sp_model_to_diffusion_gemma(const sp_model *m);  /* diffusion-gemma: gemma4 backbone + MoE FFN */
struct qwen3_model *sp_model_to_gemma4_moe(const sp_model *m);       /* gemma4 backbone + MoE FFN, AR decode (26B-A4B) */

#ifdef __cplusplus
}
#endif
#endif /* SP_ENGINE_SP_MODEL_H */
