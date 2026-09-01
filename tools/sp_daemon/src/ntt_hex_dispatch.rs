//! §4-NTT Sprint NTT.5b — Hexagon backend dispatch trampolines.
//!
//! Bridges math-core's `sp_compute_ntt_dispatch_fn` C ABI to FastRPC method
//! 17 (`ntt_hvx_vtcm_oracle` — NTT.3 VTCM-aware HVX forward NTT) and
//! method 18 (`intt_hvx_oracle` — NTT.4 HVX INTT). Each trampoline:
//!
//!   1. Casts the opaque void* handle back to &ComputeBackend (which holds
//!      an Arc<FastRpcSession>).
//!   2. Marshals the per-prime u32[N] input + u32[N] output via the standard
//!      4-word primIn header `[q_idx, N, in_bytes, out_bytes]`.
//!   3. Invokes the corresponding IDL method via Arc<FastRpcSession> (no
//!      Mutex — see reference-fastrpc-concurrent-dispatch; the session is
//!      auto-Send+Sync and supports concurrent &self.invoke()).
//!   4. Copies the device output back into the caller's u32[N] buffer.
//!
//! Lifetime contract per sp_l1.h: the caller-supplied `*mut c_void` handle
//! must remain valid past the last invocation. AppState owns the
//! `Arc<ComputeBackend>` (cloned into a leaked Box<ComputeBackend> whose
//! raw pointer the daemon passes to L1 at memo session create time;
//! Drop of AppState frees it).
//!
//! NTT.5b ships these trampolines + AppState plumbing. The actual env-gated
//! activation in forward.c (so SP_ENGINE_NTT_ATTN_HEX=1 flips the routing
//! on) is OUT OF SCOPE per the spec.

#![cfg(target_os = "android")]

use std::ffi::c_void;
use std::os::raw::c_int;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::dsp_rpc::{make_scalars, FastRpcSession, RemoteArg, RemoteBuf};

/// Sprint NTT.5c dispatch counters. Bumped per trampoline call from any
/// thread. Read at smoke-harness end via `dispatch_counts()` to validate the
/// T_NTT5C_HEX_BACKEND_ROUTES_WHEN_REGISTERED gate (both must be > 0 after a
/// non-trivial prefill_chunk iteration with SP_ENGINE_NTT_ATTN=1 +
/// SP_ENGINE_NTT_ATTN_HEX=1 on a HD ∈ {2..256}\{512} Memory model).
///
/// Process-static (one set of counters per process). The L1 ABI guarantees
/// the trampolines are called only from threads that hold a valid Arc to the
/// ComputeBackend (see lifetime contract above), so the relaxed ordering is
/// fine — these counters are observability, not synchronization.
static NTT5C_FORWARD_DISPATCH_COUNT: AtomicU64 = AtomicU64::new(0);
static NTT5C_INVERSE_DISPATCH_COUNT: AtomicU64 = AtomicU64::new(0);

/// Read the current dispatch counts (forward, inverse). Both are u64
/// monotonic counters since process start.
pub fn dispatch_counts() -> (u64, u64) {
    (NTT5C_FORWARD_DISPATCH_COUNT.load(Ordering::Relaxed),
     NTT5C_INVERSE_DISPATCH_COUNT.load(Ordering::Relaxed))
}

/// Reset the dispatch counts (e.g. between gate runs). Use sparingly — most
/// callers should just read the delta over a window.
pub fn reset_dispatch_counts() {
    NTT5C_FORWARD_DISPATCH_COUNT.store(0, Ordering::Relaxed);
    NTT5C_INVERSE_DISPATCH_COUNT.store(0, Ordering::Relaxed);
}

/// IDL method indices in the NTT.5b worktree (post-NTT.4 merge).
/// Same as the values used by the existing NTT smoke binaries
/// (sp_ntt_3_dual_smoke.rs uses 17; sp_ntt_4_polymul_smoke.rs uses 17 + 18).
const METHOD_NTT_HVX_VTCM_FORWARD: u32 = 17;
const METHOD_INTT_HVX:             u32 = 18;

/// Opaque payload passed as the L1 `void *handle` arg.
///
/// Held in AppState as `Option<Arc<ComputeBackend>>`. When backend is
/// registered with a session, we leak an `Arc::clone` into a raw pointer
/// (drop happens at AppState teardown when the Arc count returns to 1).
pub struct ComputeBackend {
    /// Shared FastRPC session for cDSP dispatch. Wrapping in Arc (NOT
    /// Mutex) per `reference-fastrpc-concurrent-dispatch` — FastRpcSession
    /// is auto-Send+Sync and supports concurrent &self.invoke().
    pub session: Arc<FastRpcSession>,
}

impl ComputeBackend {
    /// Build a backend wrapping the given FastRpcSession.
    pub fn new(session: Arc<FastRpcSession>) -> Self {
        Self { session }
    }

    /// Returns the function pointer pair (forward, inverse) suitable for
    /// passing to `sp_session_register_compute_backend`. The associated
    /// handle is `self as *const _ as *mut c_void`.
    pub fn dispatch_fns() -> (
        unsafe extern "C" fn(*mut c_void, c_int, c_int, *const u32, *mut u32) -> c_int,
        unsafe extern "C" fn(*mut c_void, c_int, c_int, *const u32, *mut u32) -> c_int,
    ) {
        (sp_compute_ntt_forward_via_fastrpc,
         sp_compute_ntt_inverse_via_fastrpc)
    }
}

/// Common marshalling helper. Both forward (m17) and inverse (m18) take the
/// same IDL shape: `[q_idx, N, in_bytes, out_bytes]` primIn + in_buf + out_buf,
/// scalars `(method, n_in=2, n_out=1)`.
fn invoke_per_prime_ntt(
    backend: &ComputeBackend,
    method: u32,
    q_idx: i32,
    n: i32,
    in_buf: &[u32],
    out_buf: &mut [u32],
) -> Result<(), ()> {
    let n_bytes = (n as usize) * 4;
    if in_buf.len() != n as usize || out_buf.len() != n as usize {
        return Err(());
    }
    let mut prim_in: [u32; 4] = [
        q_idx as u32,
        n as u32,
        n_bytes as u32,
        n_bytes as u32,
    ];
    // The IDL marshals octet sequences. Build byte mirrors of the u32 buffers.
    let mut in_bytes: Vec<u8> = Vec::with_capacity(n_bytes);
    for v in in_buf { in_bytes.extend_from_slice(&v.to_le_bytes()); }
    let mut out_bytes: Vec<u8> = vec![0u8; n_bytes];

    let mut args = [
        RemoteArg { buf: RemoteBuf {
            pv: prim_in.as_mut_ptr() as *mut c_void,
            nlen: 16,
        }},
        RemoteArg { buf: RemoteBuf {
            pv: in_bytes.as_mut_ptr() as *mut c_void,
            nlen: n_bytes,
        }},
        RemoteArg { buf: RemoteBuf {
            pv: out_bytes.as_mut_ptr() as *mut c_void,
            nlen: n_bytes,
        }},
    ];

    if backend.session.invoke(make_scalars(method, 2, 1), &mut args).is_err() {
        return Err(());
    }

    // Decode bytes -> u32 output.
    for (i, c) in out_bytes.chunks_exact(4).enumerate() {
        out_buf[i] = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
    }
    Ok(())
}

/// C trampoline for FORWARD NTT dispatch.
///
/// Signature exactly matches `sp_compute_ntt_dispatch_fn` from sp_l1.h:
///   int (*)(void *handle, int q_idx, int N, const uint32_t *in, uint32_t *out)
///
/// Safety: handle must be a valid pointer to a `ComputeBackend` (typically
/// from `Box::into_raw(Box::new(ComputeBackend::new(...)))`). in_buf and
/// out_buf must point to N u32 elements each.
///
/// Returns 0 on success, -1 on null arg / marshalling error / FastRPC failure.
#[no_mangle]
pub unsafe extern "C" fn sp_compute_ntt_forward_via_fastrpc(
    handle: *mut c_void,
    q_idx: c_int,
    n: c_int,
    in_buf: *const u32,
    out_buf: *mut u32,
) -> c_int {
    if handle.is_null() || in_buf.is_null() || out_buf.is_null() || n <= 0 {
        return -1;
    }
    /* NTT.5c: bump the forward-dispatch counter for
     * T_NTT5C_HEX_BACKEND_ROUTES_WHEN_REGISTERED. Done BEFORE the FastRPC
     * call so we count attempted dispatches even if the device call fails;
     * the smoke harness's pass criterion is "counter > 0", i.e. the
     * trampoline reached. */
    NTT5C_FORWARD_DISPATCH_COUNT.fetch_add(1, Ordering::Relaxed);
    let backend: &ComputeBackend = &*(handle as *const ComputeBackend);
    let in_slice = std::slice::from_raw_parts(in_buf, n as usize);
    let out_slice = std::slice::from_raw_parts_mut(out_buf, n as usize);
    match invoke_per_prime_ntt(
        backend,
        METHOD_NTT_HVX_VTCM_FORWARD,
        q_idx as i32,
        n as i32,
        in_slice,
        out_slice,
    ) {
        Ok(()) => 0,
        Err(()) => -1,
    }
}

/// C trampoline for INVERSE NTT dispatch.
///
/// Same signature + safety contract as the forward variant; routes to IDL
/// method 18 (`intt_hvx_oracle`). The inverse outputs per-prime u32[N] in
/// [0, q) — math-core's wrapper recombines the two primes via
/// ntt_crt_recombine on host.
#[no_mangle]
pub unsafe extern "C" fn sp_compute_ntt_inverse_via_fastrpc(
    handle: *mut c_void,
    q_idx: c_int,
    n: c_int,
    in_buf: *const u32,
    out_buf: *mut u32,
) -> c_int {
    if handle.is_null() || in_buf.is_null() || out_buf.is_null() || n <= 0 {
        return -1;
    }
    /* NTT.5c: bump the inverse-dispatch counter. */
    NTT5C_INVERSE_DISPATCH_COUNT.fetch_add(1, Ordering::Relaxed);
    let backend: &ComputeBackend = &*(handle as *const ComputeBackend);
    let in_slice = std::slice::from_raw_parts(in_buf, n as usize);
    let out_slice = std::slice::from_raw_parts_mut(out_buf, n as usize);
    match invoke_per_prime_ntt(
        backend,
        METHOD_INTT_HVX,
        q_idx as i32,
        n as i32,
        in_slice,
        out_slice,
    ) {
        Ok(()) => 0,
        Err(()) => -1,
    }
}
