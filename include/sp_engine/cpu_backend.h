/* cpu_backend.h — C-ABI surface of the CPU AVX-512 backend (Phase 2-CPU.AVX,
 * daemon-link variant).
 *
 * Sprint WIRE-CPU (2026-06-01): symmetric to Phase 2-HX's hexagon_backend.h.
 * The engine's `gemma3_forward` / `qwen3_forward` symbols (in
 * src/backends/cpu/cpu_gemma3.c:41 and src/backends/cpu/cpu_forward.c:249)
 * collide with math-core's same-named reference symbols
 * (lib/shannon-prime-system/core/forward/gemma3.c:39 and forward.c:300) when
 * both archives are linked into the same binary (the sp_daemon situation).
 *
 * Resolution: the daemon-link standalone static lib (libsp_cpu_daemon_backend)
 * compiles the engine cpu sources with -Dgemma3_forward=gemma3_forward_cpu_impl
 * etc., then sp_daemon_cpu_glue.c exposes the renamed entry points under their
 * "_cpu" suffix. Math-core's reference forwards keep their canonical names.
 *
 * The daemon's sp_session_register_forward_backend hook calls
 * sp_daemon_cpu_forward (in sp_daemon_cpu_glue.c) which dispatches to
 * gemma3_forward_cpu based on the model's arch_id. */
#ifndef SP_ENGINE_CPU_BACKEND_H
#define SP_ENGINE_CPU_BACKEND_H

#include "sp_engine/sp_status.h"
#include "sp_engine/model.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Gemma3 f32 forward on CPU (AVX2-vectorized dot_f32 on f32/f16 GGUF tensors;
 * scalar inner loop on Q8 arena tensors per cpu_overlay.c:81-97). Same contract
 * as the math-core reference gemma3_forward: prefill, causal; writes
 * logits[n_tokens * n_vocab] position-major. Requires arch == SP_ARCH_GEMMA3.
 * Returns 0 on success. */
int gemma3_forward_cpu(const qwen3_model *m, const int32_t *tokens, int n_tokens,
                       float *logits);

/* Qwen3 f32/Q8/Q4 forward on CPU. Same contract as the math-core reference
 * qwen3_forward; prefill, causal; logits[n_tokens * n_vocab] position-major.
 * Requires arch == SP_ARCH_QWEN3. */
int qwen3_forward_cpu(const qwen3_model *m, const int32_t *tokens, int n_tokens,
                      float *logits);

#ifdef __cplusplus
}
#endif
#endif /* SP_ENGINE_CPU_BACKEND_H */
