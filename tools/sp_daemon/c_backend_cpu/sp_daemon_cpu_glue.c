/* sp_daemon_cpu_glue.c — C glue between sp_l1.h's sp_forward_dispatch_fn ABI
 * and the engine's CPU forward entry points.
 *
 * Sprint WIRE-CPU (2026-06-01): symmetric to sp_daemon_hex_glue.c. The hex
 * variant aliased four engine kernel names (matmul/embed_row/as_f32/etc.) to
 * math-core's sp_* variants because hex compute lives on the cDSP. CPU is
 * different: we WANT the engine's vectorized kernel surface (the entire
 * AVX-512 + cpu_overlay.c value-add). That means the daemon-link static lib
 * links cpu_overlay.c (which exports `matmul`, `embed_row`, `dot_f32`,
 * `rmsnorm`, etc.) PLUS cpu_forward.c + cpu_gemma3.c (which call into them).
 *
 * Symbol collisions math-core <-> engine cpu surface:
 *   - sp_kernels_read_env   -> math-core/forward_dispatch.c:30  &&  cpu_overlay.c:30
 *   - qwen3_q4_stats        -> math-core/forward_dispatch.c:39  &&  cpu_overlay.c:39
 *   - gemma3_forward        -> math-core/gemma3.c:39            &&  cpu_gemma3.c:41
 *   - qwen3_forward         -> math-core/forward.c:300          &&  cpu_forward.c:249
 *   - qwen25_forward        -> math-core/qwen25.c               &&  cpu_forward.c
 *
 * Resolution: the CMakeLists for libsp_cpu_daemon_backend.a/.lib compiles the
 * engine cpu sources with -D macros renaming the colliding symbols to
 * *_cpu_impl variants. The non-colliding engine names (`matmul`, `embed_row`,
 * `dot_f32`, `rmsnorm`, `rmsnorm_head`, `rope_neox`, `kernels_attn_head`)
 * are UNIQUE to the engine and stay un-renamed -- they ARE what the engine
 * cpu_forward.c / cpu_gemma3.c call. Math-core's matmul lives under
 * sp_matmul (different name) so no collision there.
 *
 * This file exposes the renamed entries under their final "_cpu" suffix
 * (gemma3_forward_cpu / qwen3_forward_cpu) and provides sp_daemon_cpu_forward
 * as the L1 §6 dispatcher, arch-routing on m->cfg.arch.
 */
#include "sp_engine/cpu_backend.h"        /* declared symbols this TU exports   */
#include "sp_engine/model.h"               /* qwen3_model + sp_arch_t            */
#include <stdint.h>

/* Forward declarations of the renamed engine entry points. The CMake build
 * compiles cpu_gemma3.c / cpu_forward.c with these -D defines so their
 * definitions land under these names.
 *
 * NOTE: engine cpu sources only export qwen3_forward (handles Qwen3) +
 * gemma3_forward (handles Gemma3). There is no qwen25_forward in the engine
 * CPU backend (cpu_forward.c grep confirmed empty for "qwen25"). Stage 2
 * verification 2026-06-01: removed the qwen25 branch from the arch switch
 * below; SP_ARCH_QWEN25 routes through the engine's qwen3_forward path
 * (which handles the Qwen2.5 family via runtime config differences, NOT a
 * separate forward TU). Matches WIRE-HEX glue's two-arch routing. */
extern int gemma3_forward_cpu_impl(const qwen3_model *m, const int32_t *tokens,
                                    int n_tok, float *logits);
extern int qwen3_forward_cpu_impl(const qwen3_model *m, const int32_t *tokens,
                                   int n_tok, float *logits);

/* Public "_cpu"-suffixed entries: simple delegations. Keep them as wrappers
 * (not pure aliases) so future per-arch instrumentation can land here without
 * touching the engine TUs. */
int gemma3_forward_cpu(const qwen3_model *m, const int32_t *tokens,
                       int n_tok, float *logits) {
    return gemma3_forward_cpu_impl(m, tokens, n_tok, logits);
}

int qwen3_forward_cpu(const qwen3_model *m, const int32_t *tokens,
                      int n_tok, float *logits) {
    return qwen3_forward_cpu_impl(m, tokens, n_tok, logits);
}

/* The L1 forward-dispatch ABI from sp/sp_l1.h §6. Forward-declared here so
 * the glue TU stays free of the math-core include path (the daemon Rust side
 * bindgens sp_l1.h directly; this C TU just needs the signature to match). */
typedef int (*sp_forward_dispatch_fn_t)(
    void *handle, const void *qm_opaque,
    const int32_t *tokens, int n_tok, float *logits);

/* L1 §6 dispatcher. Arch-routes on m->cfg.arch.
 *   - SP_ARCH_GEMMA3              -> gemma3_forward_cpu_impl
 *   - SP_ARCH_QWEN3 / SP_ARCH_QWEN25 -> qwen3_forward_cpu_impl (engine cpu_forward.c
 *                                       handles both Qwen3 and Qwen2.5 via runtime
 *                                       config; there is no separate qwen25_forward
 *                                       TU in src/backends/cpu/)
 *   - other -> 1 (error; sp_prefill_chunk maps non-zero to SP_EBADSTATE) */
int sp_daemon_cpu_forward(void *handle,
                          const void *qm_opaque,
                          const int32_t *tokens,
                          int n_tok,
                          float *logits) {
    (void)handle;  /* opaque; CPU backend has no per-session statics */
    const qwen3_model *qm = (const qwen3_model *)qm_opaque;
    if (!qm) return 1;
    switch (qm->cfg.arch) {
        case SP_ARCH_GEMMA3:  return gemma3_forward_cpu_impl(qm, tokens, n_tok, logits);
        case SP_ARCH_QWEN3:   /* fall-through */
        case SP_ARCH_QWEN25:  return qwen3_forward_cpu_impl(qm, tokens, n_tok, logits);
        default:              return 1;
    }
}

/* Lifecycle hook: no-op for CPU backend (no FastRPC handle / no device weight
 * blob to release). Kept for ABI symmetry with sp_daemon_hex_release. */
void sp_daemon_cpu_release(const void *qm_opaque) {
    (void)qm_opaque;
}
