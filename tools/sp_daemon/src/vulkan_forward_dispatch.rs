//! Sprint WIRE-VULKAN — Vulkan compute backend full-forward dispatch.
//!
//! Symmetric to `hex_forward_dispatch.rs` (Sprint WIRE-HEX) but for the
//! HOST Vulkan compute backend (Windows / Linux / macOS desktop GPUs). The
//! 6-month-gap fix on this axis: the daemon's `sp_prefill_chunk` previously
//! routed to math-core's REFERENCE forward (`lib/shannon-prime-system/core/
//! forward/forward.c`, identified as "pure-f32 scalar" in the file header).
//! The engine's per-backend full-forward dispatchers (`src/forward/ppl.c:
//! 27-47`, `gemma3_forward_{cuda,vulkan,hexagon}`) were only ever wired
//! into `ppl.c` for the perplexity-eval harness — never into sp-daemon's
//! chat path. This module ships the wiring for the Vulkan backend.
//!
//! Architecture (Shape B per PLAN-WIRE-VULKAN.md, symmetric to WIRE-HEX):
//!  1. §6 in `sp/sp_l1.h` provides `sp_session_register_forward_backend`
//!     (opt-in full-forward hook; falls back to reference path when unset).
//!     Already shipped by WIRE-HEX — no math-core changes needed here.
//!  2. This module's `sp_daemon_vulkan_forward` C function (in
//!     `tools/sp_daemon/c_backend/sp_daemon_vulkan_glue.c`) is the
//!     `sp_forward_dispatch_fn` ABI target — it casts `qm_opaque` back to
//!     `qwen3_model *`, switches on `m->cfg.arch`, and calls either the
//!     engine's `gemma3_forward_vulkan` or `qwen3_forward_vulkan` (the
//!     Vulkan backend supports both arches per `vulkan_backend.h:36-44`).
//!  3. AppState owns the backend lifetime; daemon.rs registers it at
//!     startup when `SP_DAEMON_BACKEND=vulkan` is set.
//!
//! Scope discipline: prefill ONLY. `sp_decode_step` (persistent KV path)
//! is unhooked because `gemma3_forward_vulkan` re-runs the full forward
//! over the accumulated history per call (the engine's ppl-style usage);
//! hooking decode would be devastatingly slow without a per-backend
//! persistent-KV variant — different sprint. See sp_l1.h:§6 comment.
//!
//! Counter discipline: process-static atomic counter bumped per dispatch.
//! T_WIRE_VULKAN_RUNTIME_ACTIVE gate reads this via [`dispatch_count`].
//!
//! Vulkan device lifetime: the engine's Vulkan backend manages its own
//! VkInstance / VkDevice / VkQueue lifecycle via the lazy
//! `vk_ensure_instance` / `vk_ensure_device` path
//! (`vulkan_backend.cpp:42-125`), keyed off a process-global singleton
//! `g_ctx`. The daemon does NOT need to manage that handle. Device-resident
//! weights are cached by the engine per-model and released via
//! `sp_vulkan_model_release` at session/model teardown.
//!
//! Compile target: host platforms with a Vulkan loader. This module is
//! NOT gated by `target_os`; the feature flag itself gates compilation.

#![cfg(feature = "wire_vulkan_backend")]

use std::ffi::c_void;
use std::os::raw::c_int;
use std::sync::atomic::{AtomicU64, Ordering};

/// Dispatch counter bumped per forward call. Read by the WIRE-VULKAN smoke
/// harness to validate `T_WIRE_VULKAN_RUNTIME_ACTIVE`.
static WIRE_VULKAN_DISPATCH_COUNT: AtomicU64 = AtomicU64::new(0);

/// Read the dispatch count since process start.
pub fn dispatch_count() -> u64 {
    WIRE_VULKAN_DISPATCH_COUNT.load(Ordering::Relaxed)
}

/// Reset the dispatch count (smoke-harness gate window setup).
#[allow(dead_code)]
pub fn reset_dispatch_count() {
    WIRE_VULKAN_DISPATCH_COUNT.store(0, Ordering::Relaxed);
}

// ── C glue link surface ─────────────────────────────────────────────────────
// These symbols live in `tools/sp_daemon/c_backend/sp_daemon_vulkan_glue.c`,
// compiled into `libsp_vulkan_daemon_backend.{a,lib}` by
// `tools/sp_daemon/build-host-vulkan-backend.bat`. build.rs links the
// archive + the Vulkan loader (vulkan-1 on Windows, vulkan on Linux) when
// the `wire_vulkan_backend` feature is on.

unsafe extern "C" {
    /// Calls `gemma3_forward_vulkan(qm, tokens, n_tok, logits)` or
    /// `qwen3_forward_vulkan(...)` on the GPU via the Vulkan loader,
    /// arch-routed by `m->cfg.arch`. Returns 0 on success, non-zero on
    /// error. See `sp_daemon_vulkan_glue.c` for arch-switch logic.
    fn sp_daemon_vulkan_forward(
        handle: *mut c_void,
        qm_opaque: *const c_void,
        tokens: *const i32,
        n_tok: c_int,
        logits: *mut f32,
    ) -> c_int;

    /// Tear down the engine's cached device-resident weights for the given
    /// model. Optional — also called by qwen3_free in the engine when
    /// SP_ENGINE_WITH_VULKAN is on.
    fn sp_daemon_vulkan_release(qm_opaque: *const c_void);
}

/// Rust-side trampoline matching `sp_forward_dispatch_fn` from sp_l1.h:§6.
///
/// L1 invokes this via the function pointer registered by
/// [`register_with_session`]. Signature MUST match:
///
/// ```c
/// typedef int (*sp_forward_dispatch_fn)(
///     void *handle, const void *qm_opaque,
///     const int32_t *tokens, int n_tok, float *logits);
/// ```
///
/// We forward to the C glue (which arch-routes to gemma3_forward_vulkan
/// or qwen3_forward_vulkan) and bump the dispatch counter.
///
/// # Safety
/// - `qm_opaque` must point to a live `qwen3_model` borrowed from a session
///   (sp_session.c reconstructs it via sp_model_to_gemma3 /
///   sp_model_to_qwen3 at session create; pointer remains valid for
///   session lifetime).
/// - `tokens` must point to `n_tok` i32 elements.
/// - `logits` must point to `n_tok * n_vocab` f32 elements (n_vocab from
///   the model's arch_info).
/// - L1 holds the session's exclusive mutex; no concurrent forward calls
///   on the same session.
#[no_mangle]
pub unsafe extern "C" fn sp_wire_vulkan_forward_dispatch(
    handle: *mut c_void,
    qm_opaque: *const c_void,
    tokens: *const i32,
    n_tok: c_int,
    logits: *mut f32,
) -> c_int {
    if qm_opaque.is_null() || tokens.is_null() || logits.is_null() || n_tok <= 0 {
        return -1;
    }
    // Bump BEFORE the call so the counter reflects attempted dispatches —
    // the gate criterion is "trampoline was reached", not "the Vulkan
    // forward succeeded". Matches WIRE-HEX semantics: if vkAllocateMemory
    // OOMs partway through the dispatch, the counter still increments and
    // the daemon log surfaces the VkResult error via sp_last_error.
    WIRE_VULKAN_DISPATCH_COUNT.fetch_add(1, Ordering::Relaxed);
    unsafe { sp_daemon_vulkan_forward(handle, qm_opaque, tokens, n_tok, logits) }
}

/// Register the vulkan forward backend with an L1 session.
///
/// Returns SP_OK (== 0) on success. On failure, the session keeps its
/// existing forward dispatch (math-core reference) — the fallback is silent
/// at the L1 level; daemon.rs logs the registration outcome.
///
/// # Safety
/// `session_raw` must be a valid `*mut sp_session` pointer from
/// `SpSession::raw_ptr()` with the L2-side Mutex held (or the session
/// exclusively owned, e.g. during daemon startup before the AppState
/// Mutex<SpSession> wrap).
///
/// Uses `crate::ffi_l1` (the lib-crate L1 bindings); the binary-crate
/// `crate::ffi` (main.rs) is the SAME bindgen output, byte-identical layout.
/// The cast at the call site in daemon.rs reconciles the two type aliases.
pub unsafe fn register_with_session(
    session_raw: *mut crate::ffi_l1::sp_session,
) -> Result<(), String> {
    // SAFETY: caller holds the SpSession's Mutex (or owns exclusively); no
    // concurrent forward.
    let rc = unsafe {
        crate::ffi_l1::sp_session_register_forward_backend(
            session_raw,
            std::ptr::null_mut(),  // handle: vulkan backend is singleton (statics in vulkan_forward.cpp)
            Some(sp_wire_vulkan_forward_dispatch),
        )
    };
    if rc == crate::ffi_l1::sp_status_SP_OK {
        Ok(())
    } else {
        let detail = unsafe { std::ffi::CStr::from_ptr(crate::ffi_l1::sp_last_error()) }
            .to_string_lossy()
            .into_owned();
        Err(format!("sp_session_register_forward_backend → status={rc}: {detail}"))
    }
}

/// Tear down the engine's cached device-resident weights for the given
/// model. Called at AppState shutdown. Idempotent — safe to call after the
/// model has already been unloaded (no-op when the engine's per-model
/// statics don't match).
///
/// # Safety
/// `qm_opaque` must be a `qwen3_model *` from a session that was registered
/// with this backend. May be NULL (no-op).
#[allow(dead_code)]
pub unsafe fn release_for_model(qm_opaque: *const c_void) {
    if !qm_opaque.is_null() {
        unsafe { sp_daemon_vulkan_release(qm_opaque) };
    }
}
