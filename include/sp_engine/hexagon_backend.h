/* hexagon_backend.h — C-ABI surface of the Hexagon V69 HTP backend (Phase 2-HX).
 *
 * Built only for the aarch64-android host (SP_ENGINE_WITH_HEXAGON +
 * SP_ENGINE_TARGET_ANDROID); FastRPC is intra-device, so this runs ON the phone.
 * The host does the embedding lookup + tied LM head; the cDSP runs the 26
 * transformer layers + final RMSNorm on the per-row Q8 arena weights (sp_hex
 * FastRPC interface). SP_BACKEND=hexagon selects gemma3_forward_hexagon via ppl.c.
 */
#ifndef SP_ENGINE_HEXAGON_BACKEND_H
#define SP_ENGINE_HEXAGON_BACKEND_H

#include "sp_engine/sp_status.h"
#include "sp_engine/model.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Gemma3 forward on Hexagon: same contract as the CPU gemma3_forward (prefill,
 * causal; logits[n_tokens*n_vocab] position-major). Requires arch SP_ARCH_GEMMA3
 * and a Q8 packed arena (m->arena, SP_ARENA=q8). Opens the FastRPC handle + builds
 * the DSP weight blob once, cached by model pointer. Returns 0 on success. */
int gemma3_forward_hexagon(const qwen3_model *m, const int32_t *tokens, int n_tokens,
                           float *logits);

/* Release the cached FastRPC handle + DSP weight blob for model m (from qwen3_free). */
void sp_hexagon_model_release(const qwen3_model *m);

#ifdef __cplusplus
}
#endif
#endif /* SP_ENGINE_HEXAGON_BACKEND_H */
