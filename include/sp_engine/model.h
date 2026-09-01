/* model.h — Qwen3 model: config + weight binding over a loaded GGUF.
 *
 * 2-CPU.B step 1: read the architecture hyperparameters and bind every weight
 * tensor to its gguf_tensor (data stays paged in the mapping). The forward pass
 * (next) dequantizes on demand via sp_dequant_row. Qwen3-0.6B specifics:
 * GQA (head_count 16 / head_count_kv 8), head_dim 128 (so Q proj is
 * n_head*head_dim = 2048 wide while K/V are n_head_kv*head_dim = 1024),
 * per-head Q/K RMSNorm, SwiGLU FFN, untied output, RoPE base 1e6.
 */
#ifndef SP_ENGINE_MODEL_H
#define SP_ENGINE_MODEL_H

#include "sp_engine/gguf.h"
#include "sp/kste.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Model architecture. Selects the forward pass + which optional config fields and
 * per-layer tensors are populated. Default 0 = Qwen3 (calloc-zeroed configs and
 * existing callers get the right value with no churn). */
/* NOTE 2026-06-02: these three structs MUST stay byte-identical to the core's
 * lib/shannon-prime-system/include/sp/model.h — qwen3_load (core) fills the core
 * layout, the backends read THIS one. A stale copy here (missing the gemma4/qwen36
 * fields the core added) made sizeof(qwen3_config) differ, shifting token_embd's
 * offset -> the backend read it as NULL -> segfault in embed_row. Synced to core. */
typedef enum { SP_ARCH_QWEN3 = 0, SP_ARCH_GEMMA3 = 1, SP_ARCH_QWEN25 = 2, SP_ARCH_GEMMA4 = 3,
               SP_ARCH_QWEN36 = 4, /* qwen35moe: Gated DeltaNet + MoE hybrid */
               SP_ARCH_DIFFUSION_GEMMA = 5, /* diffusion-gemma: gemma4 backbone + MoE FFN + block masked-diffusion */
               SP_ARCH_GEMMA4_MOE = 6 /* gemma4 backbone + MoE FFN, ORDINARY AR decode (26B-A4B) */ } sp_arch_t;

typedef struct {
    sp_arch_t arch;           /* SP_ARCH_QWEN3 (default) | SP_ARCH_GEMMA3 */
    uint32_t n_layers;        /* {arch}.block_count                      */
    uint32_t n_embd;          /* {arch}.embedding_length                 */
    uint32_t n_ff;            /* {arch}.feed_forward_length              */
    uint32_t n_head;          /* {arch}.attention.head_count             */
    uint32_t n_head_kv;       /* {arch}.attention.head_count_kv (GQA)    */
    uint32_t head_dim;        /* {arch}.attention.key_length             */
    uint32_t n_vocab;         /* from token_embd / output rows           */
    uint32_t context_length;
    uint32_t sliding_window;  /* gemma3 local-attn window; 0 = none      */
    float    rope_freq_base;  /* {arch}.rope.freq_base (gemma3: global)  */
    float    rms_eps;         /* {arch}.attention.layer_norm_rms_epsilon */
    int      has_qk_norm;     /* per-head Q/K RMSNorm present            */
    int      tied_embedding;  /* output.weight absent -> reuse token_embd*/
    /* ── Gemma4 (SP_ARCH_GEMMA4) extras; zero on all other archs. ── */
    uint32_t g4_hd_swa;          /* SWA head_dim (256)            */
    uint32_t g4_nh_swa;          /* SWA n_head (8)                */
    uint32_t g4_nkv_swa;         /* SWA n_head_kv (2)             */
    float    g4_rope_base_swa;   /* SWA RoPE base (1e4)           */
    uint32_t g4_n_embd_per_layer;/* per-layer-input width (256); 0 = no AltUp path */
    uint32_t g4_n_kv_from_start; /* layers [0,this) own KV; the rest reuse (shared-KV) */
    float    g4_logit_softcap;   /* final-logit softcap (30); 0 = none */
    uint32_t g4_swa_period;      /* SWA/global period (6); global when L%period==period-1 */
    /* ── Qwen3.6 / qwen35moe (SP_ARCH_QWEN36) extras; zero on other archs. ── */
    uint32_t q36_full_attn_interval; /* full-attn iff (L+1)%this==0 (4) */
    uint32_t q36_n_expert;           /* routed experts (256)            */
    uint32_t q36_n_expert_used;      /* top-k routing (8)               */
    uint32_t q36_n_ff_exp;           /* per-expert FFN dim (512)        */
    uint32_t q36_n_ff_shexp;         /* shared-expert FFN dim (512)     */
    float    q36_expert_weights_scale;/* scale on renormed top-k weights (1.0) */
    uint32_t q36_gdn_conv_k;     /* causal conv kernel (4)              */
    uint32_t q36_gdn_state;      /* GDN head_dim S (128)               */
    uint32_t q36_gdn_n_k_heads;  /* k/q heads H_k (16)                 */
    uint32_t q36_gdn_n_v_heads;  /* v heads H_v / dt_rank (32)         */
    uint32_t q36_gdn_inner;      /* d_inner = H_v*head_v_dim (4096)    */
    int32_t  q36_rope_sections[4];/* [11,11,10,0]                       */
    uint32_t q36_rope_dim;        /* rope.dimension_count (64)          */
    float    q36_rope_base;       /* rope.freq_base (1e7)               */
    uint32_t q36_nextn_predict_layers; /* trailing NextN/MTP blocks loaded-not-run (1) */
    /* ── DiffusionGemma (SP_ARCH_DIFFUSION_GEMMA) extras; zero on other archs. The
     * gemma4 backbone geometry (n_head/n_head_kv/head_dim/g4_*) + the MoE expert
     * counts (q36_n_expert/q36_n_expert_used/q36_n_ff_exp, REUSED) carry the model;
     * dg_canvas_length is the [prompt|canvas] split point (the diffusion surface). ── */
    uint32_t dg_canvas_length;       /* diffusion.canvas_length (256) */
} qwen3_config;

typedef struct {
    const gguf_tensor *attn_norm;     /* [n_embd]                        */
    const gguf_tensor *attn_q;        /* [n_embd, n_head*head_dim]       */
    const gguf_tensor *attn_k;        /* [n_embd, n_head_kv*head_dim]    */
    const gguf_tensor *attn_v;        /* [n_embd, n_head_kv*head_dim]    */
    const gguf_tensor *attn_output;   /* [n_head*head_dim, n_embd]       */
    const gguf_tensor *attn_q_norm;   /* [head_dim] or NULL              */
    const gguf_tensor *attn_k_norm;   /* [head_dim] or NULL              */
    const gguf_tensor *post_attn_norm;/* gemma3 post_attention_norm; NULL on qwen3 */
    const gguf_tensor *ffn_norm;      /* [n_embd]                        */
    const gguf_tensor *ffn_gate;      /* [n_embd, n_ff]                  */
    const gguf_tensor *ffn_up;        /* [n_embd, n_ff]                  */
    const gguf_tensor *ffn_down;      /* [n_ff, n_embd]                  */
    const gguf_tensor *post_ffw_norm; /* gemma3 post_ffw_norm; NULL on qwen3 */
    const gguf_tensor *attn_q_bias;  /* [n_head*head_dim] qwen25; NULL otherwise */
    const gguf_tensor *attn_k_bias;  /* [n_head_kv*head_dim] qwen25; NULL otherwise */
    const gguf_tensor *attn_v_bias;  /* [n_head_kv*head_dim] qwen25; NULL otherwise */
    /* ── Gemma4 per-layer-input (AltUp) block; NULL on other archs ── */
    const gguf_tensor *per_layer_inp_gate;  /* [n_embd, n_embd_per_layer] (GGUF inp_gate) */
    const gguf_tensor *per_layer_proj;       /* [n_embd_per_layer, n_embd] (GGUF proj)    */
    const gguf_tensor *per_layer_post_norm;  /* [n_embd] (GGUF post_norm)                 */
    const gguf_tensor *out_scale;            /* [1] (GGUF layer_output_scale)             */
    /* ── Qwen3.6 / qwen35moe (SP_ARCH_QWEN36); NULL/0 on other archs ── */
    int q36_is_recurrent;
    const gguf_tensor *gdn_qkv;       /* attn_qkv  [n_embd, key_dim*2+value_dim]   */
    const gguf_tensor *gdn_gate;      /* attn_gate [n_embd, value_dim] (z proj)    */
    const gguf_tensor *gdn_conv1d;    /* ssm_conv1d [conv_k, conv_channels]        */
    const gguf_tensor *gdn_dt_bias;   /* ssm_dt bias [dt_rank]                     */
    const gguf_tensor *gdn_a;         /* ssm_a [dt_rank] (A_log; gate = a*softplus)*/
    const gguf_tensor *gdn_alpha;     /* ssm_alpha [n_embd, dt_rank]               */
    const gguf_tensor *gdn_beta;      /* ssm_beta  [n_embd, dt_rank]               */
    const gguf_tensor *gdn_norm;      /* ssm_norm  [head_v_dim] gated output norm  */
    const gguf_tensor *gdn_out;       /* ssm_out   [value_dim, n_embd]             */
    const gguf_tensor *ffn_gate_inp;  /* router [n_embd, n_expert]                 */
    const gguf_tensor *ffn_gate_exps; /* [n_embd, n_ff_exp, n_expert] rank-3       */
    const gguf_tensor *ffn_up_exps;   /* [n_embd, n_ff_exp, n_expert] rank-3       */
    const gguf_tensor *ffn_down_exps; /* [n_ff_exp, n_embd, n_expert] rank-3       */
    const gguf_tensor *ffn_gate_inp_shexp; /* shared-expert gate [n_embd]          */
    const gguf_tensor *ffn_gate_shexp;/* [n_embd, n_ff_shexp]                      */
    const gguf_tensor *ffn_up_shexp;  /* [n_embd, n_ff_shexp]                      */
    const gguf_tensor *ffn_down_shexp;/* [n_ff_shexp, n_embd]                      */
    /* ── DiffusionGemma (SP_ARCH_DIFFUSION_GEMMA); NULL/0 on other archs. The MoE
     * FFN reuses ffn_gate_inp (router) + ffn_down_exps (down) above; the gate+up are
     * FUSED into ONE rank-3 tensor (n_ff_exp*2 = 1408 second dim). The dense shared
     * MLP reuses ffn_gate/ffn_up/ffn_down. Extra F32 scale sidecars + the MoE
     * sandwich norms + the per-layer encoder output scalar live here. ── */
    const gguf_tensor *ffn_gate_up_exps;  /* [n_embd, n_ff_exp*2, n_expert] rank-3 (FUSED gate|up) */
    const gguf_tensor *ffn_gate_inp_scale;/* router per-input scale [n_embd]                       */
    const gguf_tensor *ffn_down_exps_scale;/* per-expert down scale [n_expert]                     */
    const gguf_tensor *enc_out_scale;     /* per-layer encoder output scalar [1]                   */
    const gguf_tensor *pre_ffw_norm_2;    /* MoE-branch pre-norm  [n_embd]                         */
    const gguf_tensor *post_ffw_norm_1;   /* MoE-branch post-norm [n_embd]                         */
    const gguf_tensor *post_ffw_norm_2;   /* MoE-branch post-norm [n_embd]                         */
} qwen3_layer;

struct sp_arena;   /* sp_engine/arena.h — packed-weight arena (Phase 1a) */

typedef struct qwen3_model {
    gguf_ctx          *gguf;          /* owned */
    qwen3_config       cfg;
    const gguf_tensor *token_embd;    /* [n_embd, n_vocab]               */
    const gguf_tensor *output_norm;   /* [n_embd]                        */
    const gguf_tensor *output;        /* [n_embd, n_vocab] (==token_embd if tied) */
    qwen3_layer       *layers;        /* n_layers */
    struct sp_arena   *arena;         /* packed Q8/Q4 matmul weights when SP_ARENA set; else NULL */
    int                 released;
    const gguf_tensor **norm_src;     /* [n_norm] keys (norm tensors) */
    float             **norm_buf;     /* [n_norm] owned f32 copies */
    int                 n_norm;
    gguf_tensor        *synth_tensors;
    /* ── Gemma4 model-global tensors; NULL on other archs ── */
    const gguf_tensor  *per_layer_token_embd; /* [n_embd_per_layer*n_layers, n_vocab] */
    const gguf_tensor  *per_layer_model_proj; /* [n_embd, n_embd_per_layer*n_layers]  */
    const gguf_tensor  *per_layer_proj_norm;  /* [n_embd_per_layer]                   */
    const gguf_tensor  *rope_freqs;           /* [head_dim/2] global-layer freq factors */
    /* ── DiffusionGemma model-global self-conditioning block; NULL on other archs ── */
    const gguf_tensor  *self_cond_pre_norm;   /* [n_embd]                             */
    const gguf_tensor  *self_cond_gate;       /* [n_embd, n_ff]                       */
    const gguf_tensor  *self_cond_up;         /* [n_embd, n_ff]                       */
    const gguf_tensor  *self_cond_down;       /* [n_ff, n_embd]                       */
} qwen3_model;

/* Open the GGUF, read the qwen3 config, and bind every weight. Returns NULL on
 * a missing/inconsistent tensor or a non-qwen3 architecture. */
qwen3_model *qwen3_load(const char *path);
void         qwen3_free(qwen3_model *m);

/* Release the F16 GGUF source (Phase 1b, §4.8). Requires a packed arena that
 * includes the embedding (SP_ARENA + SP_ARENA_EMBED). Copies the norm weights to
 * owned f32, then unmaps the GGUF data — after this the forward reads only the
 * arena (matmul weights + embedding) and the owned norms, so peak memory drops
 * to the packed footprint. A tokenizer, if used afterwards, must have been loaded
 * owning (sp_tokenizer_load_ex(g,1)) before this call. Returns 0 on success.
 * qwen3_load performs this automatically when SP_ARENA_RELEASE=1. */
int          qwen3_release_source(qwen3_model *m);

/* f32 reference forward pass over a token-ID sequence (prefill, causal).
 * Writes logits for every position into `logits` (caller-allocated,
 * n_tokens * n_vocab, position-major). Returns 0 on success.
 * SP_ENGINE_BACKEND=cpu scalar reference path; correctness gate E_CPU_2. */
int qwen3_forward(const qwen3_model *m, const int32_t *tokens, int n_tokens,
                  float *logits);

/* Gemma3 f32 reference forward pass (prefill, causal). Same logits layout/return
 * as qwen3_forward. Requires a model loaded with arch == SP_ARCH_GEMMA3. The
 * Gemma deltas vs Qwen3: embedding scale (×√n_embd), sandwich norms (post-attn /
 * post-ffw on the residual branch), GeGLU FFN, local/global sliding-window attn
 * with dual RoPE base, tied LM head. Correctness gate M_GEMMA3_CPU. */
int gemma3_forward(const qwen3_model *m, const int32_t *tokens, int n_tokens,
                   float *logits);

/* Qwen2.5 f32 reference forward pass (prefill, causal). Same logits layout/return
 * as qwen3_forward. Requires a model loaded with arch == SP_ARCH_QWEN25. Deltas
 * vs Qwen3: no embedding scale, no QK norms, QKV biases added after projection,
 * SwiGLU FFN, no sandwich norms, no sliding-window attention. */
int qwen25_forward(const qwen3_model *m, const int32_t *tokens, int n_tokens,
                   float *logits);

/* As qwen3_forward, but if `kv_trees` is non-NULL it additionally KSTE-encodes
 * every cached K head-vector (the KSTE KV-cache overlay, gated in production by
 * SP_KSTE_KV=1; E_CPU_6). Each post-norm/post-RoPE K head-vector is quantized to
 * int32 and encoded to its 64-byte signature. `kv_trees` must hold
 * n_layers * n_tokens * n_head_kv entries, indexed
 * ((L*n_tokens + t)*n_head_kv + h). Pass NULL to skip (== qwen3_forward). */
int qwen3_forward_ex(const qwen3_model *m, const int32_t *tokens, int n_tokens,
                     float *logits, sp_kste_tree_t *kv_trees);

/* Greedy (argmax) autoregressive generation. `seq` holds `n_prompt` prompt token
 * IDs and must have capacity for at least n_prompt + n_gen entries; generated
 * tokens are appended in place. Stops after n_gen tokens, or earlier if `eos_id`
 * (>= 0) is produced. Returns the new total sequence length, or < 0 on error.
 *
 * Reference implementation: each step re-runs qwen3_forward over the whole prefix
 * and takes the argmax of the last position's logits. Correct by construction
 * (it reuses the E_CPU_2-validated forward) at O(n^2) cost; a persistent-KV-cache
 * decode is a later optimization to be validated for token-identity against this. */
int qwen3_generate(const qwen3_model *m, int32_t *seq, int n_prompt, int n_gen,
                   int eos_id);

/* Persistent-KV O(n) greedy decode. Same signature/semantics as qwen3_generate,
 * but maintains a position-indexed K/V cache so each token's weight matmuls run
 * once (O(n) total) instead of re-prefilling the prefix every step (O(n^2)).
 * Stores K/V post-RoPE; honors SP_ENGINE_FROB / SP_CPU_SCALAR / SP_KV_SPINOR.
 * Greedy output matches qwen3_generate up to the float-reassociation floor, so
 * GEN_KV gates on sequence (argmax) identity, not bit-equal logits. */
int qwen3_generate_kv(const qwen3_model *m, int32_t *seq, int n_prompt, int n_gen,
                      int eos_id);

/* ── MTP (Theorem T8): persistent-KV speculative decode ──────────────────────
 * qwen3_mtp_forward: batched APPEND-forward. Computes `nb` tokens at absolute
 * positions [basePos .. basePos+nb-1] over the caller's persistent f32 K/V cache
 * (kc/vc, layout [layer][slot][KVD], `cap` slots/layer). The prefix [0..basePos-1]
 * must already hold the confirmed post-RoPE K/V; this writes the new K/V into
 * slots [basePos..basePos+nb-1] and runs full causal attention over [0..basePos+t]
 * for each batch token t. logits[nb*n_vocab] receives every token's LM-head row.
 * Each position's math (RoPE pos, QK-norm, full causal attention, the batched
 * matmuls) is identical to the one-token path, so argmax is bit-identical to plain
 * greedy. Plain f32 path only (no Ring-2/recall/fuse/Spinor/KSTE overlays).
 * Returns 0 on success, non-zero on error. */
int qwen3_mtp_forward(const qwen3_model *m, const int32_t *batch, int nb,
                      int basePos, float *kc, float *vc, int cap, float *logits);

/* qwen3_mtp_decode: greedy decode realized via prompt-lookup speculation on top
 * of qwen3_mtp_forward. `seq` holds n_prompt prompt tokens, capacity >= n_prompt
 * + n_gen. K = draft depth (K=0 ⇒ plain incremental KV greedy on the same cache —
 * the apples-to-apples baseline); NG = prompt-lookup n-gram order. The emitted
 * sequence is byte-identical to greedy by construction (acceptance is argmax
 * equality). On success returns the new total length; *out_forwards (if non-NULL)
 * gets the number of append-forwards, *out_accept_sum / *out_accept_steps the mean
 * accept stats. The wall-clock win = (forwards saved) × the batched weight-read
 * amortization, because each verify reuses the cached prefix K/V. */
int qwen3_mtp_decode(const qwen3_model *m, int32_t *seq, int n_prompt, int n_gen,
                     int eos_id, int K, int NG,
                     long *out_forwards, long *out_accept_sum, long *out_accept_steps);

/* G2 (C2.1 Step 3): teacher-forced autoregressive perplexity over the DECODE path,
 * so the recall router (SP_RECALL_*) + two-ring (SP_RING2) are exercised exactly as
 * production generates — unlike sp_perplexity, which runs the dense prefill
 * qwen3_forward (no recall knobs). `toks[0,n_toks)` is the full corpus slice; the
 * positions [n_warm, n_toks-1) are scored (predict toks[pos+1] from logits at pos).
 * On success returns 0, sets *ppl = exp(mean NLL) and *n_scored. Shares the exact
 * generate_kv decode body (one forward, no path divergence). */
int qwen3_ppl_decode(const qwen3_model *m, int32_t *toks, int n_toks, int n_warm,
                     double *ppl, long *n_scored);

/* Q4 calibration stats from the most recent forward on the Q4 weight path
 * (SP_ENGINE_FROB=3 or 4): how many weight rows the mixed-precision calibration
 * promoted to Q8, out of the total rows seen. Both 0 otherwise. (E_CPU_7.) */
void qwen3_q4_stats(long *promoted, long *rows);

/* ── dequantization (forward pass reads weights through these) ── */
float    sp_f16_to_f32(uint16_t h);
/* Round f32 to IEEE half (round-to-nearest-even). Used by the ggml-faithful
 * validation path: ggml downcasts matmul activations (src1) to F16 when the
 * weight (src0) is F16, so SP_ENGINE_F16_ACT=1 mimics that to match the oracle
 * bit-for-bit under matched precision. */
uint16_t sp_f32_to_f16(float f);
/* Dequantize `n` elements of a tensor row starting at `src` (a pointer into the
 * mapping) of the given ggml_type into `dst` (f32). Supports F32, F16, Q8_0.
 * Returns 0 on success, nonzero for an unsupported type. */
int   sp_dequant_row(const void *src, uint32_t ggml_type, int n, float *dst);

#ifdef __cplusplus
}
#endif
#endif /* SP_ENGINE_MODEL_H */
