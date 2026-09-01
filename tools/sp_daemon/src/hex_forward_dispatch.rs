//! Sprint WIRE-HEX — Hexagon V69 forward-backend dispatch.
//!
//! The 6-month-gap fix: the daemon's `sp_prefill_chunk` previously routed
//! to math-core's REFERENCE forward (`lib/shannon-prime-system/core/forward/
//! forward.c`, identified as "pure-f32 scalar" in the file header). The
//! engine's per-backend full-forward dispatchers
//! (`src/forward/ppl.c:27-47`, gemma3_forward_{cuda,vulkan,hexagon}) were
//! only ever wired into `ppl.c` for the perplexity-eval harness — never
//! into sp-daemon's chat path. This module ships the wiring for the
//! Hexagon V69 HVX backend.
//!
//! Architecture (Shape B per PLAN-WIRE-HEX.md):
//!  1. New §6 in `sp/sp_l1.h` adds `sp_session_register_forward_backend`
//!     (opt-in full-forward hook; falls back to reference path when unset).
//!  2. This module's `sp_daemon_hex_forward` C function (in
//!     `tools/sp_daemon/c_backend/sp_daemon_hex_glue.c`) is the
//!     `sp_forward_dispatch_fn` ABI target — it casts qm_opaque back to
//!     `qwen3_model *` and calls the engine's `gemma3_forward_hexagon`.
//!  3. AppState owns the backend lifetime; daemon.rs registers it at
//!     startup when `SP_DAEMON_BACKEND=hex` is set.
//!
//! Scope discipline: prefill ONLY. `sp_decode_step` (persistent KV path)
//! is unhooked because `gemma3_forward_hexagon` re-runs the full forward
//! over the accumulated history per call (the engine's ppl-style usage);
//! hooking decode would be devastatingly slow without a per-backend
//! persistent-KV variant — different sprint. See sp_l1.h:§6 comment.
//!
//! Counter discipline: process-static atomic counter bumped per dispatch.
//! T_WIRE_HEX_BACKEND_DISPATCHES gate reads this via [`dispatch_count`].
//!
//! Per `reference-mode-d-bridge-architecture`: sp_hex_host.c itself opens
//! its own FastRPC session under Unsigned PD (`DSPRPC_CONTROL_UNSIGNED_MODULE`
//! at sp_hex_host.c:73-77); the daemon does NOT need to manage that
//! handle — the engine's hex backend caches it keyed on the model pointer.

#![cfg(target_os = "android")]

use std::ffi::c_void;
use std::os::raw::c_int;
use std::sync::atomic::{AtomicU64, Ordering};

/// Dispatch counter bumped per forward call. Read by the WIRE-HEX smoke
/// harness to validate `T_WIRE_HEX_BACKEND_DISPATCHES`.
static WIRE_HEX_DISPATCH_COUNT: AtomicU64 = AtomicU64::new(0);

/// Read the dispatch count since process start.
pub fn dispatch_count() -> u64 {
    WIRE_HEX_DISPATCH_COUNT.load(Ordering::Relaxed)
}

/// Reset the dispatch count (smoke-harness gate window setup).
pub fn reset_dispatch_count() {
    WIRE_HEX_DISPATCH_COUNT.store(0, Ordering::Relaxed);
}

// ── C glue link surface ─────────────────────────────────────────────────────
// These symbols live in `tools/sp_daemon/c_backend/sp_daemon_hex_glue.c`,
// compiled into `libsp_hex_daemon_backend.a` by
// `tools/sp_daemon/build-android-hex-backend.bat`. build.rs links the
// archive when the `SP_DAEMON_LINK_HEX=1` env var is set during cargo build.

unsafe extern "C" {
    /// Calls `gemma3_forward_hexagon(qm, tokens, n_tok, logits)` on the
    /// cDSP via FastRPC. Returns 0 on success, non-zero on error.
    fn sp_daemon_hex_forward(
        handle: *mut c_void,
        qm_opaque: *const c_void,
        tokens: *const i32,
        n_tok: c_int,
        logits: *mut f32,
    ) -> c_int;

    /// Tear down the engine's cached FastRPC handle + DSP weight blob for
    /// the given model. Optional — also called by qwen3_free in the engine.
    fn sp_daemon_hex_release(qm_opaque: *const c_void);
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
/// We forward to the C glue (which forwards to gemma3_forward_hexagon)
/// and bump the dispatch counter.
///
/// # Safety
/// - `qm_opaque` must point to a live `qwen3_model` borrowed from a session
///   (sp_session.c reconstructs it via `sp_model_to_gemma3` at session
///   create; pointer remains valid for session lifetime).
/// - `tokens` must point to `n_tok` i32 elements.
/// - `logits` must point to `n_tok * n_vocab` f32 elements (n_vocab from
///   the model's arch_info).
/// - L1 holds the session's exclusive mutex; no concurrent forward calls
///   on the same session.
#[no_mangle]
pub unsafe extern "C" fn sp_wire_hex_forward_dispatch(
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
    // the gate criterion is "trampoline was reached", not "FastRPC succeeded".
    WIRE_HEX_DISPATCH_COUNT.fetch_add(1, Ordering::Relaxed);
    unsafe { sp_daemon_hex_forward(handle, qm_opaque, tokens, n_tok, logits) }
}

/// Register the hex forward backend with an L1 session.
///
/// Returns SP_OK (== 0) on success. On failure, the session keeps its
/// existing forward dispatch (math-core reference) — the fallback is silent
/// at the L1 level; daemon.rs logs the registration outcome.
///
/// # Safety
/// `session_raw` must be a valid `*mut sp_session` pointer from
/// `SpSession::raw_ptr()` with the L2-side Mutex held.
///
/// Uses `crate::ffi_l1` (the lib-crate L1 bindings); the binary-crate
/// `crate::ffi` (main.rs) is the SAME bindgen output, byte-identical layout.
/// The cast at the call site in daemon.rs reconciles the two type aliases.
pub unsafe fn register_with_session(
    session_raw: *mut crate::ffi_l1::sp_session,
) -> Result<(), String> {
    // SAFETY: caller holds the SpSession's Mutex; no concurrent forward.
    let rc = unsafe {
        crate::ffi_l1::sp_session_register_forward_backend(
            session_raw,
            std::ptr::null_mut(),  // handle: hex backend is singleton (statics in sp_hex_host.c)
            Some(sp_wire_hex_forward_dispatch),
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

/// Tear down the engine's cached FastRPC handle + DSP weight blob.
/// Called at AppState shutdown. Idempotent — safe to call after the
/// model has already been unloaded (no-op when the engine's static
/// `g_hx.key` doesn't match).
///
/// # Safety
/// `qm_opaque` must be a `qwen3_model *` from a session that was registered
/// with this backend. May be NULL (no-op).
pub unsafe fn release_for_model(qm_opaque: *const c_void) {
    if !qm_opaque.is_null() {
        unsafe { sp_daemon_hex_release(qm_opaque) };
    }
}
