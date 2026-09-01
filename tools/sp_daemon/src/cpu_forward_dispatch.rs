//! Sprint WIRE-CPU — CPU AVX-512 forward-backend dispatch.
//!
//! Symmetric port of `hex_forward_dispatch.rs`. The 6-month-gap fix: the
//! daemon's `sp_prefill_chunk` routes by default through math-core's
//! REFERENCE forward (`lib/shannon-prime-system/core/forward/forward.c`,
//! pure scalar f32). The engine's `cpu_forward.c` + `cpu_gemma3.c` +
//! `cpu_overlay.c` (with AVX2 `dot_f32` and optional AVX-512 primitives,
//! Phase 2-CPU.AVX) is the production CPU backend that ships in
//! `libsp_engine.lib` for the `sp_perplexity` / `sp_engine` test harness
//! BUT has never been reachable from the daemon's chat path.
//!
//! This module + `tools/sp_daemon/c_backend_cpu/sp_daemon_cpu_glue.c` +
//! `libsp_cpu_daemon_backend.lib` close that gap for the CPU backend,
//! exactly mirroring WIRE-HEX.
//!
//! Architecture (Shape B per PLAN-WIRE-HEX.md, now also PLAN-WIRE-CPU.md):
//!  1. §6 in `sp/sp_l1.h` exposes `sp_session_register_forward_backend`
//!     (opt-in full-forward hook; falls back to reference when unset).
//!     Same hook as WIRE-HEX.
//!  2. This module's `sp_daemon_cpu_forward` C function (in
//!     `tools/sp_daemon/c_backend_cpu/sp_daemon_cpu_glue.c`) is the
//!     `sp_forward_dispatch_fn` ABI target — it casts qm_opaque back to
//!     `qwen3_model *` and arch-routes into `gemma3_forward_cpu` /
//!     `qwen3_forward_cpu` / `qwen25_forward_cpu_impl`.
//!  3. AppState owns the backend lifetime (no per-session statics on CPU
//!     side; the backend is stateless apart from `cpu_overlay.c`'s
//!     runtime gate-knob globals which read env vars at every prefill).
//!     daemon.rs registers it at startup when `SP_DAEMON_BACKEND=cpu`
//!     is set.
//!
//! Counter discipline: process-static atomic counter bumped per dispatch.
//! T_WIRE_CPU_RUNTIME_ACTIVE gate reads this via [`dispatch_count`].

#![cfg(feature = "wire_cpu_backend")]

use std::ffi::c_void;
use std::os::raw::c_int;
use std::sync::atomic::{AtomicU64, Ordering};

/// Dispatch counter bumped per forward call. Read by the WIRE-CPU smoke
/// harness to validate `T_WIRE_CPU_RUNTIME_ACTIVE`.
static WIRE_CPU_DISPATCH_COUNT: AtomicU64 = AtomicU64::new(0);

/// Read the dispatch count since process start.
pub fn dispatch_count() -> u64 {
    WIRE_CPU_DISPATCH_COUNT.load(Ordering::Relaxed)
}

/// Reset the dispatch count (smoke-harness gate window setup).
pub fn reset_dispatch_count() {
    WIRE_CPU_DISPATCH_COUNT.store(0, Ordering::Relaxed);
}

// ── C glue link surface ─────────────────────────────────────────────────────
// These symbols live in `tools/sp_daemon/c_backend_cpu/sp_daemon_cpu_glue.c`,
// compiled into `sp_cpu_daemon_backend.lib` by
// `tools/sp_daemon/build-host-cpu-backend.bat`. build.rs links the archive
// when CARGO_FEATURE_WIRE_CPU_BACKEND is set during cargo build.

unsafe extern "C" {
    /// Calls `gemma3_forward_cpu` / `qwen3_forward_cpu` / `qwen25_forward_cpu_impl`
    /// based on `qm_opaque`'s arch_id. Returns 0 on success, non-zero on error.
    fn sp_daemon_cpu_forward(
        handle: *mut c_void,
        qm_opaque: *const c_void,
        tokens: *const i32,
        n_tok: c_int,
        logits: *mut f32,
    ) -> c_int;

    /// No-op for CPU (no FastRPC handle to release); kept for ABI symmetry.
    fn sp_daemon_cpu_release(qm_opaque: *const c_void);
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
/// We forward to the C glue (which arch-routes into the engine's renamed
/// per-arch CPU forwards) and bump the dispatch counter.
///
/// # Safety
/// - `qm_opaque` must point to a live `qwen3_model` borrowed from a session.
/// - `tokens` must point to `n_tok` i32 elements.
/// - `logits` must point to `n_tok * n_vocab` f32 elements.
/// - L1 holds the session's exclusive mutex; no concurrent forward calls.
#[no_mangle]
pub unsafe extern "C" fn sp_wire_cpu_forward_dispatch(
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
    // the gate criterion is "trampoline was reached", not "forward succeeded".
    WIRE_CPU_DISPATCH_COUNT.fetch_add(1, Ordering::Relaxed);
    unsafe { sp_daemon_cpu_forward(handle, qm_opaque, tokens, n_tok, logits) }
}

/// Register the CPU forward backend with an L1 session.
///
/// Returns Ok(()) on success. On failure, the session keeps its existing
/// forward dispatch (math-core reference) — the fallback is silent at the
/// L1 level; daemon.rs logs the registration outcome.
///
/// Unlike `hex_forward_dispatch::register_with_session` (which used the
/// lib-crate's `ffi_l1` and a pointer cast to bridge the binary-crate /
/// lib-crate Rust type alias), WIRE-CPU lives entirely in the binary crate
/// (it's host-targeted and the binary already has the L1 bindgen output
/// via `crate::ffi`). The caller passes a raw `*mut crate::ffi::sp_session`
/// pointer directly.
///
/// # Safety
/// `session_raw` must be a valid `*mut sp_session` pointer with the L2-side
/// Mutex held (the daemon owns the session exclusively at startup, BEFORE
/// it wraps the SpSession in `Mutex::new`).
pub unsafe fn register_with_session(
    session_raw: *mut crate::ffi::sp_session,
) -> Result<(), String> {
    let rc = unsafe {
        crate::ffi::sp_session_register_forward_backend(
            session_raw,
            std::ptr::null_mut(),
            Some(sp_wire_cpu_forward_dispatch),
        )
    };
    if rc == crate::ffi::sp_status_SP_OK {
        Ok(())
    } else {
        let detail = unsafe { std::ffi::CStr::from_ptr(crate::ffi::sp_last_error()) }
            .to_string_lossy()
            .into_owned();
        Err(format!("sp_session_register_forward_backend → status={rc}: {detail}"))
    }
}

/// Tear down any per-model cached resources. No-op for CPU (no FastRPC
/// session, no device weight blob); kept for ABI symmetry with
/// `hex_forward_dispatch::release_for_model`.
///
/// # Safety
/// `qm_opaque` must be a `qwen3_model *` from a session that was registered
/// with this backend. May be NULL (no-op).
pub unsafe fn release_for_model(qm_opaque: *const c_void) {
    if !qm_opaque.is_null() {
        unsafe { sp_daemon_cpu_release(qm_opaque) };
    }
}
