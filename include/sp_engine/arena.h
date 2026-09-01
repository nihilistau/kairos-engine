/* arena.h — packed-weight arena: the load-bearing inline-compression memory
 * layout (roadmap §4.8). At load the matmul weight tensors are quantized once
 * into per-row Frobenius Q8 (or Q4 mixed-precision) codes; the forward-pass
 * matmul reads the packed codes and lifts inline (no per-matmul re-quantization,
 * and — once the source is released in Phase 1b — no decoded fp32 arena).
 *
 * The packed per-tensor FORMAT (per-row Q8/Q4, the byte layout the GPU backends
 * read) lives in the math core (sp/frobenius_lift.h: sp_frob_packed_tensor +
 * SP_FROB_ARENA_LAYOUT_VERSION). This file adds only the engine-side collection:
 * a named set of packed tensors built by iterating the model's weight tensors.
 *
 * Phase 1a: arena covers the matmul weights only (attn q/k/v/o, ffn gate/up/down,
 * LM-head `output`); the embedding and norms stay f32 from the GGUF mapping. Phase
 * 1b folds the embedding in and releases the mapping.
 */
#ifndef SP_ENGINE_ARENA_H
#define SP_ENGINE_ARENA_H

#include <stdint.h>
#include <stddef.h>
#include "sp/frobenius_lift.h"   /* sp_frob_packed_tensor + SP_FROB_ARENA_LAYOUT_VERSION */

#ifdef __cplusplus
extern "C" {
#endif

/* One arena entry: a GGUF tensor name plus its math-core mixed-precision packed
 * form (the SP_FROB_ARENA_LAYOUT_VERSION byte layout; per-row Q8/Q4). */
typedef struct {
    char                  name[256];
    sp_frob_packed_tensor pt;
} sp_arena_tensor;

typedef struct sp_arena sp_arena;

struct qwen3_model;   /* forward decl (sp_engine/model.h) */

/* Build a packed arena over the model's matmul weight tensors. `precision` is 8
 * (Q8) or 4 (Q4 mixed: rows whose Q4 round-trip rel-error exceeds `q4_promote`
 * are stored Q8). If `include_embed` is nonzero the token-embedding tensor is
 * packed too (Phase 1b — required before releasing the GGUF source); otherwise
 * the embedding stays f32 from the mapping (Phase 1a, which keeps the E_CPU_9
 * byte-identity-to-FROB gate). Norms are never packed. Returns NULL on error.
 * The model's GGUF mapping must still be open during the build. */
sp_arena *sp_arena_build(const struct qwen3_model *m, int precision, float q4_promote,
                         int include_embed);
void      sp_arena_free(sp_arena *a);

/* Lookup a packed tensor by its GGUF tensor name; NULL if not arena-ized. */
const sp_arena_tensor *sp_arena_find(const sp_arena *a, const char *name);

/* Adopt an array of already-packed named tensors into a fresh arena (the .sp-model
 * load path, Phase 2-FMT). `ts` is copied by value into the arena, which then OWNS
 * the inner pt.{codes,row_scale,row_off,row_prec} buffers (freed by sp_arena_free).
 * The caller must NOT free those buffers afterwards. `precision` records 8 or 4 for
 * sp_arena_precision. Returns NULL on a bad arg or alloc failure (in which case the
 * caller still owns `ts`). */
sp_arena *sp_arena_from_packed(const sp_arena_tensor *ts, int n, int precision);

/* Reconstruct row `r` of a packed tensor to `cols` f32 values (inline lift:
 * code * row_scale / qmax). Used for the embedding lookup when the embedding is
 * in the arena. Returns 0 on success. */
int sp_arena_dequant_row(const sp_arena_tensor *at, int r, float *dst);

size_t sp_arena_bytes(const sp_arena *a);       /* total packed bytes (codes+scales+meta) */
int    sp_arena_precision(const sp_arena *a);   /* 8 or 4 */
long   sp_arena_promoted(const sp_arena *a);    /* Q4 rows promoted to Q8 (0 for Q8 arena) */
long   sp_arena_total_rows(const sp_arena *a);

#ifdef __cplusplus
}
#endif
#endif /* SP_ENGINE_ARENA_H */
