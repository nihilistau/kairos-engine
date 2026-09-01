/* kernels.h — backend-agnostic CPU forward-pass kernels shared by every model
 * architecture (Qwen3 in forward.c, Gemma3 in gemma3.c, …). These were originally
 * static in forward.c; extracted verbatim so a second architecture reuses them
 * without duplicating the matmul/quant/norm/RoPE machinery. Behavior is unchanged
 * — the Qwen3 regression (E_CPU_2/3/4/…/COMPOSE) guards the extraction.
 *
 * The runtime gate knobs these kernels honor (SP_CPU_SCALAR, SP_ENGINE_F16_ACT,
 * SP_ENGINE_FROB, SP_Q4_PROMOTE) live here and are refreshed via
 * sp_kernels_read_env(); the forward-specific knobs (NTT-attn, KSTE, Spinor-KV)
 * stay with their forward pass. All default OFF = the pure-f32 reference path.
 */
#ifndef SP_ENGINE_KERNELS_H
#define SP_ENGINE_KERNELS_H

#include "sp_engine/model.h"

#ifdef __cplusplus
extern "C" {
#endif

/* f32 dot product. AVX2 (8-wide FMA) unless SP_CPU_SCALAR forces the scalar
 * reduction; the scalar tail matches the scalar path (E_CPU_4). */
float dot_f32(const float *a, const float *b, int n);

/* RMSNorm over n elements: out_i = x_i / sqrt(mean(x^2)+eps) * w_i. */
void rmsnorm(const float *x, const float *w, int n, float eps, float *out);

/* in-place RMSNorm of a single head vector (length d). */
void rmsnorm_head(float *v, const float *w, int d, float eps);

/* NEOX RoPE on a head vector (length d) at position p, given the rope base. */
void rope_neox(float *v, int d, int p, float base);

/* Causal GQA softmax attention for one query head over the cached K/V at
 * positions [0, pos]. K/V are laid out [s*KVD + kvh*HD] (s = key position,
 * kvh = the kv-head this query maps to). Computes scores = ascale * <qh, k_s>,
 * a max-shifted softmax, and the weighted V-sum into out[HD].
 *   win < 0  -> full causal (attend to all s in [0, pos]);
 *   win >= 0 -> sliding window, attend to s in [max(0, pos-win+1), pos].
 * `sc` is caller scratch of length >= pos+1. This is the plain f32 path; the
 * Qwen3 NTT-attention overlay (E_CPU_5) stays inline in forward.c. */
void kernels_attn_head(const float *qh, const float *KC, const float *VC,
                       int pos, int KVD, int kvh, int HD, float ascale, int win,
                       float *sc, float *out);

/* Y[t,j] = sum_i W[i,j] * X[t,i]; W is the gguf tensor [in,out] (ne0=in). Honors
 * the packed-weight arena (if built) and the SP_ENGINE_FROB/F16_ACT knobs. */
int matmul(const qwen3_model *m, const gguf_tensor *W,
           const float *X, int n_tok, int in, int out, float *Y);

/* Read a norm/scale weight tensor as f32 (owned copy after source release, else
 * directly from the GGUF mapping). NULL if not found post-release. */
const float *as_f32(const qwen3_model *m, const gguf_tensor *t);

/* Embedding lookup for token `tok` -> dst[E] (from the arena if packed, else the
 * GGUF mapping). Returns 0 on success. */
int embed_row(const qwen3_model *m, int32_t tok, int E, float *dst);

/* Refresh the kernel gate knobs from the environment (call once per forward entry,
 * before any matmul). */
void sp_kernels_read_env(void);

#ifdef __cplusplus
}
#endif
#endif /* SP_ENGINE_KERNELS_H */
