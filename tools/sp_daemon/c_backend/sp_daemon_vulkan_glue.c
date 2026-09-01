/* sp_daemon_vulkan_glue.c -- C glue between sp_l1.h's sp_forward_dispatch_fn
 * ABI and the engine's Vulkan compute-backend whole-forward entry points
 * (vulkan_forward.cpp:658 gemma3_forward_vulkan; :933 qwen3_forward_vulkan).
 *
 * Sprint WIRE-VULKAN: symmetric to sp_daemon_hex_glue.c. The daemon's host
 * binary, when SP_DAEMON_BACKEND=vulkan is set, calls
 *   sp_session_register_forward_backend(s, NULL, sp_daemon_vulkan_forward).
 * On each subsequent sp_prefill_chunk, the math-core session routes through
 * this glue (sp_l1.h:#6) which forwards to the engine's Vulkan backend.
 *
 * Arch routing: the Vulkan backend exports BOTH gemma3_forward_vulkan
 * (VK.1-3) and qwen3_forward_vulkan (VK.4, wraps qwen3_forward_vulkan_ex
 * with NULL kv_trees). The glue inspects m->cfg.arch and routes accordingly.
 * This is wider than the hex glue (gemma3-only) because the Vulkan backend
 * already supports both arches in ctest. Wider than the CUDA glue at the
 * same point in time -- same shape, different backend.
 *
 * Type-safety: the L1 ABI passes the qwen3_model pointer as `const void *`
 * to keep sp_l1.h independent of sp/model.h (which lives in the engine
 * include tree). Cast back here. Same arch enum (SP_ARCH_GEMMA3 /
 * SP_ARCH_QWEN3) on both math-core and engine sides.
 *
 * Error handling: gemma3_forward_vulkan / qwen3_forward_vulkan return 0 on
 * success, non-zero on error (per vulkan_backend.h:36-44). sp_prefill_chunk
 * maps non-zero to SP_EBADSTATE + preserves sp_last_error via the
 * sp_set_error calls already inside the Vulkan backend (vk_fail in
 * vulkan_backend.cpp:20 wraps every VkResult; the forward path also calls
 * sp_set_error directly for arch-check/n_tok-limit failures at
 * vulkan_forward.cpp:660-662, 785-787, 802).
 *
 * No kernel-name shim: the Vulkan backend reads math-core's packed-weight
 * arena directly via sp_arena + sp_frobenius (vulkan_forward.cpp's host
 * scratch path), and the daemon already links those archives from
 * libsp_forward_dispatch.a etc. No matmul / embed_row / as_f32 shim is
 * needed (Vulkan does its own GEMM in SPIR-V compute shaders, not via
 * the engine's CPU kernel surface). cpu_overlay.c is irrelevant on Vulkan.
 *
 * Lifecycle hook: sp_vulkan_model_release frees device-resident weights
 * cached by the engine's static (per the vulkan_backend.h:54-55 contract).
 * Called at daemon shutdown / model unload via release_for_model
 * (vulkan_forward_dispatch.rs).
 *
 * Build target: this TU is built into libsp_vulkan_daemon_backend.{a,lib}
 * by tools/sp_daemon/c_backend/CMakeLists.txt's
 * SP_DAEMON_BUILD_VULKAN_BACKEND branch. The daemon binary link step pulls
 * Vulkan::Vulkan (the loader -- vulkan-1.lib on Windows, libvulkan.so on
 * Linux) at sp-daemon link time via build.rs's wire_vulkan_backend block.
 */
#include "sp_engine/vulkan_backend.h"   /* gemma3_forward_vulkan + qwen3_forward_vulkan + sp_vulkan_model_release */
#include "sp/model.h"                    /* qwen3_model + sp_arch_t + SP_ARCH_GEMMA3 / SP_ARCH_QWEN3 */
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

extern void sp_set_error(const char *msg);

/* The L1 forward-dispatch ABI from sp/sp_l1.h #6. Forward-declared here so
 * the glue stays free of the math-core include path (the daemon Rust side
 * bindgens sp_l1.h directly; this C TU just needs the signature to match). */
typedef int (*sp_forward_dispatch_fn_t)(
    void *handle, const void *qm_opaque,
    const int32_t *tokens, int n_tok, float *logits);

/* Forward-dispatch entry point exposed to Rust via #[link]. The L1 contract
 * (sp_l1.h:#6) guarantees:
 *   - handle is the opaque pointer the daemon passed at register time.
 *     Ignored here -- the Vulkan backend caches its device-resident weight
 *     blob via statics in vulkan_forward.cpp keyed on the model pointer.
 *   - qm_opaque is the session-owned qwen3_model* (math-core's sp_session.c
 *     reconstructs it via sp_model_to_gemma3 / sp_model_to_qwen3; pointer
 *     remains valid for the session's lifetime).
 *   - tokens/n_tok/logits per the standard engine forward signature
 *     (src/forward/ppl.c:19).
 * Returns 0 on success, non-zero on error.
 *
 * Arch routing:
 *   SP_ARCH_GEMMA3 -> gemma3_forward_vulkan  (VK.1-3)
 *   SP_ARCH_QWEN3  -> qwen3_forward_vulkan   (VK.4; wraps qwen3_forward_vulkan_ex
 *                                              with kv_trees=NULL)
 *   else           -> sp_set_error + return -1
 *
 * SP_ARCH_QWEN25 is not yet covered by the Vulkan backend (qwen25_forward_vulkan
 * does not exist in vulkan_forward.cpp); routing to qwen3_forward_vulkan would
 * mismatch the arch check inside the kernel at vulkan_forward.cpp:785 and
 * fail; surface as error here. */
int sp_daemon_vulkan_forward(void *handle,
                             const void *qm_opaque,
                             const int32_t *tokens,
                             int n_tok,
                             float *logits) {
    (void)handle;  /* opaque; the engine owns its own statics */
    const qwen3_model *qm = (const qwen3_model *)qm_opaque;
    if (!qm) { sp_set_error("vulkan: NULL qwen3_model"); return -1; }
    switch (qm->cfg.arch) {
        case SP_ARCH_GEMMA3:
            return gemma3_forward_vulkan(qm, tokens, n_tok, logits);
        case SP_ARCH_QWEN3:
            return qwen3_forward_vulkan(qm, tokens, n_tok, logits);
        default:
            sp_set_error("vulkan: unsupported arch (only SP_ARCH_GEMMA3 / SP_ARCH_QWEN3)");
            return -1;
    }
}

/* Lifecycle hook: when the daemon tears down its vulkan backend (or the
 * model unloads), release the cached device-resident weight blob. Mirrors
 * sp_hexagon_model_release / sp_cuda_model_release. */
void sp_daemon_vulkan_release(const void *qm_opaque) {
    const qwen3_model *qm = (const qwen3_model *)qm_opaque;
    if (qm) sp_vulkan_model_release(qm);
}

#ifdef __cplusplus
}
#endif
