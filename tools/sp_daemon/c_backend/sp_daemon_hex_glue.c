/* sp_daemon_hex_glue.c — C glue between sp_l1.h's sp_forward_dispatch_fn ABI
 * and the engine's gemma3_forward_hexagon entry point (sp_hex_host.c:112).
 *
 * Also provides the "engine kernels.h" → math-core forward_dispatch SHIM.
 * sp_hex_host.c calls four engine-symbol entry points (matmul, embed_row,
 * as_f32, sp_kernels_read_env, plus qwen3_q4_stats indirectly via cpu_overlay).
 * If we link cpu_overlay.c into the daemon, those symbols collide with the
 * SAME-named symbols already linked from math-core's forward_dispatch.c
 * (which itself was relocated from cpu_overlay; see forward_dispatch.c:1-13).
 *
 * Resolution: omit cpu_overlay.c from the daemon-link build (CMakeLists
 * dropped it 2026-05-31 WIRE-HEX Stage 3) and define the four shim symbols
 * here as thin aliases over math-core's sp_matmul / sp_embed_row / sp_as_f32
 * / sp_kernels_read_env. The math-core variants have IDENTICAL signatures
 * (verified: lib/shannon-prime-system/include/sp/forward_dispatch.h:29-44).
 *
 * Sprint WIRE-HEX: the daemon's android binary, when SP_DAEMON_BACKEND=hex is
 * set, calls sp_session_register_forward_backend(s, NULL, sp_daemon_hex_forward).
 * On each subsequent sp_prefill_chunk, the math-core session routes through
 * this glue (sp_l1.h:§6) which forwards to the engine's V69 HVX backend.
 *
 * Type-safety: the L1 ABI passes the qwen3_model pointer as `const void *`
 * to keep sp_l1.h independent of sp/model.h (which lives in the engine bridge
 * tree). Cast back here. Both sp/model.h and sp_engine/model.h define the
 * same qwen3_model struct under the same SP_ENGINE_MODEL_H guard (verified
 * 2026-05-31 — only the math-core variant adds NTT.5c _ex2 declarations),
 * so the cast is bit-stable across both include paths.
 *
 * Error handling: gemma3_forward_hexagon returns 0 on success, 1 on failure.
 * sp_prefill_chunk maps non-zero to SP_EBADSTATE + sets sp_last_error via the
 * sp_set_error calls already inside sp_hex_host.c.
 */
#include "sp_engine/hexagon_backend.h"   /* gemma3_forward_hexagon */
#include "sp/forward_dispatch.h"          /* sp_matmul / sp_embed_row / sp_as_f32 / sp_kernels_read_env */
#include "sp/model.h"                     /* qwen3_model + gguf_tensor for shim signatures */
#include <stdint.h>

/* The L1 forward-dispatch ABI from sp/sp_l1.h §6. Forward-declared here so the
 * glue stays free of the math-core include path (the daemon Rust side bindgens
 * sp_l1.h directly; this C TU just needs the signature to match). */
typedef int (*sp_forward_dispatch_fn_t)(
    void *handle, const void *qm_opaque,
    const int32_t *tokens, int n_tok, float *logits);

/* Forward-dispatch entry point exposed to Rust via #[link]. The L1 contract
 * (sp_l1.h:§6) guarantees:
 *   - handle is the opaque pointer the daemon passed at register time. Ignored
 *     here — gemma3_forward_hexagon is a singleton (it caches the FastRPC
 *     handle + DSP weight blob keyed on the model pointer; see sp_hex_host.c:
 *     28-35,118).
 *   - qm_opaque is the session-owned qwen3_model* (math-core's sp_session.c
 *     reconstructs it via sp_model_to_gemma3; pointer remains valid for the
 *     session's lifetime).
 *   - tokens/n_tok/logits per the standard engine forward signature
 *     (src/forward/ppl.c:19).
 * Returns 0 on success. */
int sp_daemon_hex_forward(void *handle,
                          const void *qm_opaque,
                          const int32_t *tokens,
                          int n_tok,
                          float *logits) {
    (void)handle;  /* opaque; sp_hex_host owns its own statics */
    const qwen3_model *qm = (const qwen3_model *)qm_opaque;
    return gemma3_forward_hexagon(qm, tokens, n_tok, logits);
}

/* Lifecycle hook: when the daemon tears down its hex backend (or the model
 * unloads), release the cached FastRPC handle + DSP weight blob. Mirrors
 * the engine's qwen3_free path which already calls sp_hexagon_model_release
 * (include/sp_engine/hexagon_backend.h:26-27). */
void sp_daemon_hex_release(const void *qm_opaque) {
    const qwen3_model *qm = (const qwen3_model *)qm_opaque;
    sp_hexagon_model_release(qm);
}

/* ── Engine-kernel shim over math-core forward_dispatch ────────────────────
 *
 * sp_hex_host.c reaches for four `sp_engine/kernels.h` entry points by
 * unprefixed names (matmul, embed_row, as_f32, sp_kernels_read_env). In the
 * daemon-link build we DO NOT link cpu_overlay.c (it would duplicate
 * sp_kernels_read_env + qwen3_q4_stats already present in math-core's
 * libsp_forward_dispatch.a — see the file header comment). Instead, expose
 * the engine names as thin aliases over math-core's sp_* variants.
 *
 * Same TU as the L1 dispatcher so the link graph is one extra .o, not a
 * separate library, and so the lifetime of these symbols matches the
 * sp_daemon_hex_forward symbol exactly. */

int matmul(const qwen3_model *m, const gguf_tensor *W,
           const float *X, int n_tok, int in, int out, float *Y) {
    return sp_matmul(m, W, X, n_tok, in, out, Y);
}

int embed_row(const qwen3_model *m, int32_t tok, int E, float *dst) {
    return sp_embed_row(m, tok, E, dst);
}

const float *as_f32(const qwen3_model *m, const gguf_tensor *t) {
    return sp_as_f32(m, t);
}

/* sp_kernels_read_env: math-core's forward_dispatch.c:30 already exports
 * this name; the math-core variant fully replaces the engine's cpu_overlay
 * version (same env vars, same behavior — math-core was relocated FROM
 * cpu_overlay per forward_dispatch.c:1-13). sp_hex_host.c's call site
 * resolves to the math-core symbol; no shim entry needed here. */
