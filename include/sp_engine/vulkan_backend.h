/* vulkan_backend.h — C-ABI surface of the Vulkan compute backend (Phase 2-VK).
 *
 * Built only when SP_ENGINE_WITH_VULKAN. Mirrors the CUDA backend (Phase 2-CU)
 * op-for-op: the forward entry points match the CPU gemma3_forward / qwen3_forward
 * exactly (same signature/logits layout); a Vulkan forward_fn is selected when
 * SP_BACKEND=vulkan. Every VkResult that escapes a call here is wrapped to
 * SP_EVULKAN with the detail in sp_last_error() (the frozen L1 ABI error contract,
 * sp_status.h). There is no cuBLAS analog — the GEMM is a hand-written tiled f32
 * compute shader (SPIR-V, compiled at build time via glslc).
 */
#ifndef SP_ENGINE_VULKAN_BACKEND_H
#define SP_ENGINE_VULKAN_BACKEND_H

#include "sp_engine/sp_status.h"
#include "sp_engine/model.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Number of Vulkan-capable physical devices visible (0 on none or loader error;
 * detail in sp_last_error on error). Cheap toolchain/link smoke. */
int sp_vulkan_device_count(void);

/* Device `dev` info: copies the name into `name` (capacity name_cap, always
 * NUL-terminated when name_cap>0) and writes the Vulkan apiVersion major/minor to
 * *sm_major/*sm_minor (when non-NULL — the CUDA-compute-capability analog). Returns
 * SP_OK or SP_EVULKAN. */
sp_status sp_vulkan_device_info(int dev, char *name, int name_cap,
                                int *sm_major, int *sm_minor);

/* Gemma3 f32 forward on Vulkan. Same contract as the CPU gemma3_forward: prefill,
 * causal; writes logits[n_tokens * n_vocab] position-major. Requires a model
 * loaded with arch == SP_ARCH_GEMMA3. Honors the packed-weight arena (Q8/Q4)
 * when m->arena is set, else the GGUF f32/f16 path. Returns 0 on success,
 * nonzero on error (sp_last_error has Vulkan detail). [VK.1/VK.2/VK.3] */
int gemma3_forward_vulkan(const qwen3_model *m, const int32_t *tokens, int n_tokens,
                          float *logits);

/* Qwen3 f32/Q8/Q4 forward on Vulkan. Same contract as the CPU qwen3_forward:
 * prefill, causal; logits[n_tokens * n_vocab] position-major. Requires arch ==
 * SP_ARCH_QWEN3. Honors the packed-weight arena when m->arena is set. [VK.4] */
int qwen3_forward_vulkan(const qwen3_model *m, const int32_t *tokens, int n_tokens,
                         float *logits);

/* As qwen3_forward_vulkan, but if kv_trees is non-NULL each post-norm/post-RoPE K
 * head-vector is KSTE-encoded to its 64-byte signature (E_VK_6) via the host
 * sp_kste_encode. kv_trees holds n_layers * n_tokens * n_kv_heads entries,
 * indexed ((L*n_tokens + t)*n_kv_heads + h). Pass NULL to skip (== forward). */
int qwen3_forward_vulkan_ex(const qwen3_model *m, const int32_t *tokens, int n_tokens,
                            float *logits, sp_kste_tree_t *kv_trees);

/* Release any cached device-resident weights for model `m` (called from
 * qwen3_free when the Vulkan backend is built). No-op if nothing cached. */
void sp_vulkan_model_release(const qwen3_model *m);

#ifdef __cplusplus
}
#endif
#endif /* SP_ENGINE_VULKAN_BACKEND_H */
