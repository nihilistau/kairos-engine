//! Sprint WIRE-CUDA-DECODE-GEMMA4 — persistent-KV decode backend dispatch.
//!
//! Symmetric to `cuda_forward_dispatch.rs` (WIRE-CUDA, PREFILL), but for the
//! **token-by-token DECODE** path the prefill hook cannot serve. See
//! `tools/sp_daemon/WIRE-CUDA-DECODE-GEMMA4.md` for the full design.
//!
//! WHY a second module: `sp_session_register_forward_backend` /
//! `sp_forward_dispatch_fn` (sp_l1.h §6) is PREFILL-ONLY — it re-runs the full
//! forward over the accumulated history per call, AND for a dense reference model OK_Q4B model the
//! tied full-vocab LM head is materialized only inside the DECODE path, so
//! driving decode through the prefill entry trips the guard at
//! `cuda_forward.cu:1627` (`-4: g4 probe: FULL head needs the f32 embd`).
//! The fix is a stateful, session-resident KV-decode verb mirroring the
//! already-frozen `gemma4_kv_*` C ABI (cuda_forward.cu, declared in
//! tests/test_gemma4_cuda.c:65-79).
//!
//! Architecture (mirrors WIRE-CUDA forward, Shape B):
//!  1. The future L1 verb `sp_session_register_kvdecode_backend` (sp_l1.h §6b,
//!     designed in the addendum, NOT yet in the frozen header) takes a
//!     dispatch TABLE (open/prefill/decode_step/rewind/position/close) over a
//!     session-resident handle, not a single stateless forward fn.
//!  2. This module's C glue (`sp_daemon_cuda_kvdecode_*` in
//!     `c_backend_cuda/sp_daemon_cuda_glue.c`) adapts that table onto the
//!     `gemma4_kv_*` symbols already compiled into
//!     `libsp_cuda_daemon_backend`.
//!  3. AppState owns the `sp_g4_kv*` handle lifetime (state.rs
//!     `cuda_kvdecode_handle`); daemon.rs opens it at startup when
//!     `SP_DAEMON_BACKEND=cuda` + `SP_DAEMON_KVDECODE=1` (INTEGRATION step).
//!
//! Null floor: this module compiles ONLY under `--features wire_cuda_backend`
//! (the same feature that links the CUDA lib carrying `gemma4_kv_*` — no new
//! feature). Without it the daemon binary is byte-identical to pre-WIRE-CUDA.
//!
//! SCAFFOLD: the device-wiring bodies are stubbed with `TODO(WIRE-CUDA-DECODE)`.
//! The real `gemma4_kv_*` calls + the `sp_session_register_kvdecode_backend`
//! header verb land at INTEGRATION (addendum §7). This file is the reviewable
//! skeleton + the link surface, and it COMPILES.

#![cfg(feature = "wire_cuda_backend")]

use std::ffi::c_void;
use std::os::raw::c_int;
use std::sync::atomic::{AtomicU64, Ordering};

/// Decode-step counter bumped per `decode_step` call. Read by the
/// `G-WIRE-CUDA-DECODE-GEMMA4` smoke harness to validate the verb was reached.
static KVDECODE_STEP_COUNT: AtomicU64 = AtomicU64::new(0);

/// Read the decode-step count since process start.
pub fn step_count() -> u64 {
    KVDECODE_STEP_COUNT.load(Ordering::Relaxed)
}

/// Reset the decode-step count (smoke-harness gate window setup).
pub fn reset_step_count() {
    KVDECODE_STEP_COUNT.store(0, Ordering::Relaxed);
}

// ── C glue link surface ─────────────────────────────────────────────────────
// These symbols live in `c_backend_cuda/sp_daemon_cuda_glue.c`, compiled into
// `libsp_cuda_daemon_backend` by `build-host-cuda-backend.bat`. build.rs links
// the static lib when the `wire_cuda_backend` feature is on. Each adapts one
// row of the `sp_kvdecode_dispatch_fn` table (addendum §2) onto a `gemma4_kv_*`
// symbol. The opaque `*mut c_void` handle is an `sp_g4_kv*` on the engine side.
unsafe extern "C" {
    /// `gemma4_kv_open(m, pmax)` -> `sp_g4_kv*` (as opaque handle).
    /// Returns NULL on failure (sp_last_error carries detail).
    fn sp_daemon_cuda_kvdecode_open(qm_opaque: *const c_void, pmax: c_int) -> *mut c_void;

    /// `gemma4_kv_prefill(s, toks, n)`. 0 on success.
    fn sp_daemon_cuda_kvdecode_prefill(
        handle: *mut c_void,
        tokens: *const i32,
        n_tok: c_int,
    ) -> c_int;

    /// #41 batch prefill — one n-wide batched forward that sinks K/V into the
    /// resident cache (CONTRACT-BATCH-PREFILL). Cold + ring-off + full-cache only;
    /// FLOAT (chat speed mode, not byte-exact). 0 ok, -1 on precondition fail
    /// (caller falls back to per-token `prefill`).
    fn sp_daemon_cuda_kvdecode_prefill_batched(
        handle: *mut c_void,
        tokens: *const i32,
        n_tok: c_int,
    ) -> c_int;

    /// #41b suffix batch — batched forward of ONLY the suffix at absolute positions
    /// `[from, from+n)`, extending a live cache standing exactly at `from` (the engine
    /// enforces dpos==from and errors otherwise). The CALLER owns the continuation-safe
    /// snapshot round-trip; see the routes.rs dispatch. 0 ok, -1 declined.
    fn sp_daemon_cuda_kvdecode_prefill_batched_from(
        handle: *mut c_void,
        tokens: *const i32,
        n_tok: c_int,
        from: c_int,
    ) -> c_int;

    /// One persistent-KV decode step at the live dpos. Writes the full-vocab
    /// logits row `[n_vocab]` for the NEXT position and advances dpos.
    /// TODO(WIRE-CUDA-DECODE): backed by the additive `gemma4_kv_decode_logits`
    /// symbol (addendum §3.1 option A) — NOT the argmax-only `gemma4_kv_decode`.
    fn sp_daemon_cuda_kvdecode_step(
        handle: *mut c_void,
        token: i32,
        logits: *mut f32,
    ) -> c_int;

    /// `gemma4_kv_rewind(s, n)`. O(1) cold-evict (`dpos -= n`). 0 on success.
    fn sp_daemon_cuda_kvdecode_rewind(handle: *mut c_void, n: c_int) -> c_int;

    /// CONTRACT-CHAT-FULLSTACK B2 RING-FIX — `gemma4_kv_reset(s)`. Clean reset to
    /// dpos=0 WITHOUT replaying the SWA-owner undo-journal (which `rewind(pos)`
    /// does and reads OOB past `Jmax` on the ring path once `pos>Jmax`). 0 on success.
    fn sp_daemon_cuda_kvdecode_reset(handle: *mut c_void) -> c_int;

    /// G-INT-2-FIX — `gemma4_kv_reset_cold(s)`. Like reset() but ALSO zeroes every
    /// owner K/V cache + the SWA journal, so a reconstruction starts cold (no judge
    /// residue can be attended after a B3-JUDGE nested pass). 0 on success.
    fn sp_daemon_cuda_kvdecode_reset_cold(handle: *mut c_void) -> c_int;

    /// KAIROS P1c-2 prefix shear — `gemma4_kv_shear(s, P)`: O(1) position
    /// restore to P when the SWA ring has never wrapped this residency.
    /// 0 ok; nonzero = wrapped/bad args (caller falls back to full prefill).
    fn sp_daemon_cuda_kvdecode_shear(handle: *mut c_void, p: c_int) -> c_int;

    /// PREFIX SNAPSHOT / RESTORE (2026-07-13). The shear cannot fire when the ring has
    /// wrapped, and the operator's 2517-token preamble is longer than ring_W=2048, so it
    /// wraps inside the preamble and the shear is dead code at his config. Copying the
    /// bytes works regardless of wrapping. Blob is host memory OWNED BY RUST.
    /// ONE-SHOT: this cache is decoded once and released, never continued. It unlocks the
    /// batched prefill for callers that never needed persistence in the first place.
    /// SCRATCH: ring-off, one-shot, sized to its own prompt. See `open_scratch`.
    fn sp_daemon_cuda_kvdecode_open_scratch(qm: *const c_void, pmax: c_int) -> *mut c_void;
    fn sp_daemon_cuda_kvdecode_set_oneshot(handle: *mut c_void, on: c_int) -> c_int;
    fn sp_daemon_cuda_kvdecode_prefix_bytes(handle: *mut c_void, p: c_int) -> i64;
    fn sp_daemon_cuda_kvdecode_snapshot_prefix(
        handle: *mut c_void, host: *mut c_void, cap: i64, p: c_int) -> i64;
    fn sp_daemon_cuda_kvdecode_restore_prefix(
        handle: *mut c_void, host: *const c_void, cap: i64, p: c_int) -> c_int;

    /// `gemma4_kv_pos(s)`. Current dpos, or -1 on NULL.
    fn sp_daemon_cuda_kvdecode_position(handle: *const c_void) -> c_int;

    /// `gemma4_kv_close(s)`. Frees the resident cache. NULL-safe.
    fn sp_daemon_cuda_kvdecode_close(handle: *mut c_void);

    /// CONTRACT-CHAT-FULLSTACK B1 — `gemma4_kv_byteexact_set(s, on)`. Toggles
    /// per-session byte-exact "auditable mode" on the resident cache. 0 on success.
    fn sp_daemon_cuda_kvdecode_byteexact(handle: *mut c_void, on: c_int) -> c_int;

    /// CONTRACT-CUDA-KV-FOUNDATION — `gemma4_kv_set_kv_flags(s, flags)`. Sets the KV
    /// codec flags (bit0 = SP_KV_SPINOR) on the resident cache. 0 on success.
    fn sp_daemon_cuda_kvdecode_kv_flags(handle: *mut c_void, flags: u32) -> c_int;

    /// CONTRACT-CHAT-FULLSTACK B2 (§6d-b) — `gemma4_kv_replay(s, epdir, npos, zero)`.
    /// Recall a stored episode's owner K/V into the resident cache at
    /// `[dpos, dpos+npos)` and advance dpos (SP_REPLAY into the live turn). `epdir`
    /// is a NUL-terminated path holding ep.mf/ep.k/ep.v; `zero!=0` = zeroed reject
    /// control. 0 on success.
    fn sp_daemon_cuda_kvdecode_replay(
        handle: *mut c_void,
        epdir: *const std::os::raw::c_char,
        npos: c_int,
        zero: c_int,
    ) -> c_int;

    /// B3-v10 ablation gate — memset-zero `k` episode positions (base+pos[i]) K/V.
    fn sp_daemon_cuda_kvdecode_ablate(
        handle: *mut c_void,
        base: c_int,
        pos: *const c_int,
        k: c_int,
    ) -> c_int;

    /// CONTRACT-CHAT-FULLSTACK B5 (§6e) — `gemma4_kv_inject_tokens(s, toks, n)`.
    /// TEXT through the single latent entry seam: per token, stage embd[id]*sqrt(E)
    /// into the inject buffer and step the real id, so the residual is bit-identical
    /// to prefill (the B5 parity proof). 0 on success.
    fn sp_daemon_cuda_kvdecode_inject_tokens(
        handle: *mut c_void,
        toks: *const i32,
        n: c_int,
    ) -> c_int;

    /// G-INT-2-FIX — `gemma4_kv_inject_tokens_atten(s, toks, n)`. The LIVE recall inject
    /// seam: same residual entry as inject_tokens, but the natively-minted memory K is
    /// scaled by the constant-budget alpha (SP_REPLAY_MTARGET, default 42) so a recalled
    /// episode BINDS instead of HIJACKING. 0 on success.
    fn sp_daemon_cuda_kvdecode_inject_tokens_atten(
        handle: *mut c_void,
        toks: *const i32,
        n: c_int,
    ) -> c_int;

    /// CONTRACT-CHAT-FULLSTACK B5 (§6e) — `gemma4_kv_inject_seq(s, embs, n_frames, ph)`.
    /// The GENERIC residual-frame channel: inject `n_frames` raw E-float residual
    /// vectors at consecutive positions, each minted at `ph_token`. The seam AUDIO
    /// (EAR/KAI-3) and MEMORY (decoded episode residuals) feed through. 0 on success.
    fn sp_daemon_cuda_kvdecode_inject_frames(
        handle: *mut c_void,
        embs: *const f32,
        n_frames: c_int,
        ph_token: c_int,
    ) -> c_int;

    /// CONTRACT-CHAT-FULLSTACK B3 (AUTONOMOUS RECALL) —
    /// `gemma4_kv_read_global_k(s, out, npos)`. Reads the GLOBAL-owner K rows
    /// `[0,npos)` out of the resident cache into `out` (packed
    /// `[n_global][npos][g_kvd]` row-major, global layers ascending). Returns the
    /// number of global layers written (>0) on success, -1 on error. The cache is
    /// byte-untouched (read-only D2H).
    /// The row stride `read_global_k` writes at: `g_nkv * g_hd`. Published by the
    /// engine because reconstructing it from `recall`'s accessors gets it wrong the
    /// moment g_nkv != 1 — which it is on gemma4-26b-a4b.
    fn sp_daemon_cuda_kvdecode_global_kvd(handle: *const c_void) -> c_int;
    fn sp_daemon_cuda_kvdecode_geometry(handle: *const c_void, out: *mut c_int, n: c_int) -> c_int;
    fn sp_daemon_cuda_kvdecode_read_global_k(
        handle: *const c_void,
        out: *mut f32,
        npos: c_int,
    ) -> c_int;

    /// CONTRACT-CHAT-FULLSTACK B3-v2 (q·K AUTONOMOUS RECALL) —
    /// `gemma4_kv_read_global_q(s, token, out)`. Runs one non-committing forward of
    /// `token` at the live dpos and reads the last-token GLOBAL-layer query (post-RoPE)
    /// into `out` (packed `[n_global][g_nh*g_hd]` row-major). dpos is rolled back; the
    /// cache is unchanged for the caller's subsequent replay + decode. Returns the
    /// number of global layers written (>0) on success, -1 on error.
    fn sp_daemon_cuda_kvdecode_read_global_q(
        handle: *mut c_void,
        token: i32,
        out: *mut f32,
    ) -> c_int;

    /// B4 NIGHTSHIFT Option-2 PROVENANCE FIX — capture a live episode through the
    /// SAME batched `gemma4_decode_cuda` forward the curator used, writing
    /// ep.k/ep.v/ep.mf into `out_dir` (via SP_XBAR_RECALL_WRITE set+unset around
    /// the call). Makes the live episode's on-disk K byte-compatible with the
    /// curated registry so the deployed W_c head works with ZERO retraining.
    /// `qm_opaque` = the session's borrowed `qwen3_model*`; 0 on success.
    fn sp_daemon_cuda_kvcapture_batched(
        qm_opaque: *const c_void,
        tokens: *const i32,
        n: c_int,
        out_dir: *const std::os::raw::c_char,
    ) -> c_int;
}

/// Open a session-resident KV-decode cache on the CUDA backend.
///
/// Returns the opaque `sp_g4_kv*` handle (as `*mut c_void`) on success, or an
/// error string. `pmax` is the max resident position count (context budget);
/// `qm_opaque` is the session-borrowed `qwen3_model*` (must be `SP_ARCH_GEMMA4`;
/// the tied head needs `SP_CUDA_DECODE_INT8=1` in the environment — see
/// `gemma4_kv_open` at cuda_forward.cu:3670).
///
/// # Safety
/// `qm_opaque` must point to a live `qwen3_model` borrowed from a session
/// (valid for the session lifetime).
pub unsafe fn open(qm_opaque: *const c_void, pmax: i32) -> Result<*mut c_void, String> {
    if qm_opaque.is_null() || pmax <= 0 {
        return Err("kvdecode open: NULL model or non-positive pmax".to_string());
    }
    // SAFETY: caller guarantees qm_opaque validity; pmax checked positive.
    let h = unsafe { sp_daemon_cuda_kvdecode_open(qm_opaque, pmax) };
    if h.is_null() {
        Err(last_error())
    } else {
        Ok(h)
    }
}

/// Ingest prompt history into the resident cache (stores K/V at `[dpos,dpos+n)`).
///
/// # Safety
/// `handle` must be a live `sp_g4_kv*` from [`open`]; `tokens` valid for `n_tok`.
pub unsafe fn prefill(handle: *mut c_void, tokens: &[i32]) -> Result<(), String> {
    if handle.is_null() || tokens.is_empty() {
        return Err("kvdecode prefill: NULL handle or empty tokens".to_string());
    }
    // SAFETY: handle live per caller; tokens slice gives ptr+len.
    let rc = unsafe {
        sp_daemon_cuda_kvdecode_prefill(handle, tokens.as_ptr(), tokens.len() as c_int)
    };
    if rc == 0 { Ok(()) } else { Err(last_error()) }
}

/// #41 batch prefill — one n-wide batched forward that sinks K/V into the resident
/// cache. Cold + ring-off + full-cache only (the C side enforces + errors otherwise);
/// FLOAT, a chat speed mode. Returns Err on precondition fail so the caller can fall
/// back to per-token `prefill` (byte-identical null floor).
///
/// # Safety
/// `handle` live per caller; `tokens` valid for its length.
pub unsafe fn prefill_batched(handle: *mut c_void, tokens: &[i32]) -> Result<(), String> {
    if handle.is_null() || tokens.is_empty() {
        return Err("kvdecode prefill_batched: NULL handle or empty tokens".to_string());
    }
    let rc = unsafe {
        sp_daemon_cuda_kvdecode_prefill_batched(handle, tokens.as_ptr(), tokens.len() as c_int)
    };
    if rc == 0 { Ok(()) } else { Err(last_error()) }
}

/// #41b suffix batch — batched forward of ONLY the suffix at absolute positions
/// `[from, from+len)`, extending a live cache standing exactly at `from`. The engine
/// enforces `dpos == from` (strict extension) and errors otherwise, so any
/// precondition miss falls back cleanly. The CALLER owns the continuation-safe
/// snapshot→reset→restore round-trip — this call alone leaves the same
/// journal-less state the cold batch does (G-PK2-BATCHRING).
///
/// # Safety
/// `handle` live per caller; `tokens` valid for its length.
pub unsafe fn prefill_batched_from(
    handle: *mut c_void,
    tokens: &[i32],
    from: i32,
) -> Result<(), String> {
    if handle.is_null() || tokens.is_empty() || from < 0 {
        return Err("kvdecode prefill_batched_from: NULL handle, empty tokens, or bad from".to_string());
    }
    let rc = unsafe {
        sp_daemon_cuda_kvdecode_prefill_batched_from(
            handle, tokens.as_ptr(), tokens.len() as c_int, from)
    };
    if rc == 0 { Ok(()) } else { Err(last_error()) }
}

/// B4 NIGHTSHIFT Option-2 PROVENANCE FIX — capture a live episode through the
/// curator's BATCHED `gemma4_decode_cuda` forward, writing `ep.k`/`ep.v`/`ep.mf`
/// into `out_dir`. The batched path evolves the gemma4 AltUp/PLE residual the same
/// way the curator did, so the resulting `ep.k` is byte-compatible with the curated
/// registry and the deployed W_c head selects the live episode with no retraining.
/// This allocates a scratch cache inside the engine and reuses the model's cached
/// device weights (no 9GB reload). `out_dir` must already exist. The caller MUST
/// hold the resident-cache Mutex (the capture is serialized so no concurrent reader
/// of SP_XBAR_RECALL_WRITE races the set+unset inside the glue).
///
/// # Safety
/// `qm` must be a live `qwen3_model*` borrowed from a session (valid for the call);
/// `tokens` valid for its length.
pub unsafe fn capture_batched(qm: *const c_void, tokens: &[i32], out_dir: &str) -> Result<(), String> {
    if qm.is_null() || tokens.is_empty() {
        return Err("kvcapture_batched: NULL model or empty tokens".to_string());
    }
    let c_dir = match std::ffi::CString::new(out_dir) {
        Ok(s) => s,
        Err(_) => return Err("kvcapture_batched: out_dir has interior NUL".to_string()),
    };
    // SAFETY: qm valid per caller; tokens slice gives ptr+len; c_dir owns the
    // NUL-terminated buffer for the duration of the call; glue forwards to
    // gemma4_decode_cuda with SP_XBAR_RECALL_WRITE set+unset tightly around it.
    let rc = unsafe {
        sp_daemon_cuda_kvcapture_batched(qm, tokens.as_ptr(), tokens.len() as c_int, c_dir.as_ptr())
    };
    if rc == 0 { Ok(()) } else { Err(last_error()) }
}

/// Decode one token, returning the full-vocab logits row for the next position.
///
/// `logits` must be a caller-allocated slice of `n_vocab` f32. L2 owns sampling
/// (greedy / temperature / top-p / spec-decode verify) over the returned row.
///
/// # Safety
/// `handle` must be a live `sp_g4_kv*` from [`open`]; `logits.len()` must equal
/// the model's `n_vocab` (the glue writes exactly that many floats).
pub unsafe fn decode_step(
    handle: *mut c_void,
    token: i32,
    logits: &mut [f32],
) -> Result<(), String> {
    if handle.is_null() || logits.is_empty() {
        return Err("kvdecode decode_step: NULL handle or empty logits".to_string());
    }
    // Bump BEFORE the call so the counter reflects attempted steps (the gate
    // criterion is "the verb was reached", not "CUDA succeeded").
    KVDECODE_STEP_COUNT.fetch_add(1, Ordering::Relaxed);
    // SAFETY: handle live per caller; logits slice gives the n_vocab buffer.
    let rc = unsafe { sp_daemon_cuda_kvdecode_step(handle, token, logits.as_mut_ptr()) };
    if rc == 0 { Ok(()) } else { Err(last_error()) }
}

/// O(1) cold-evict: shear the logical decode position back by `n`.
///
/// # Safety
/// `handle` must be a live `sp_g4_kv*` from [`open`].
pub unsafe fn rewind(handle: *mut c_void, n: i32) -> Result<(), String> {
    if handle.is_null() || n < 0 {
        return Err("kvdecode rewind: NULL handle or negative n".to_string());
    }
    // SAFETY: handle live per caller.
    let rc = unsafe { sp_daemon_cuda_kvdecode_rewind(handle, n) };
    if rc == 0 { Ok(()) } else { Err(last_error()) }
}

/// CONTRACT-CHAT-FULLSTACK B2 RING-FIX — clean per-request reset to dpos=0 without
/// journal replay. Use INSTEAD of `rewind(pos)` at chat-request start: `rewind`
/// replays the SWA-owner undo-journal and reads it OOB past `Jmax` once `pos>Jmax`
/// on the ring path; `reset` just zeroes the counters (stale ring slots are never
/// read — the next turn overwrites them in position order). The caller MUST hold
/// the cache Mutex.
///
/// # Safety
/// `handle` must be a live `sp_g4_kv*` from [`open`].
pub unsafe fn reset(handle: *mut c_void) -> Result<(), String> {
    if handle.is_null() {
        return Err("kvdecode reset: NULL handle".to_string());
    }
    // SAFETY: handle live per caller.
    let rc = unsafe { sp_daemon_cuda_kvdecode_reset(handle) };
    if rc == 0 { Ok(()) } else { Err(last_error()) }
}

/// G-INT-2-FIX — COLD reset: like [`reset`] but also zeroes every owner K/V cache
/// (and the SWA undo-journal) so a reconstruction truly starts cold. The B3-JUDGE
/// branch uses this after the nested judge forward so no stale judge K/V can be
/// attended during synthesis (the prompt-echo degeneration root cause). Byte-identical
/// to a plain reset for the normal null-floor path (the zeroed slots are never read
/// after a fresh prefill). The caller MUST hold the cache Mutex.
///
/// # Safety
/// `handle` must be a live `sp_g4_kv*` from [`open`].
pub unsafe fn reset_cold(handle: *mut c_void) -> Result<(), String> {
    if handle.is_null() {
        return Err("kvdecode reset_cold: NULL handle".to_string());
    }
    // SAFETY: handle live per caller.
    let rc = unsafe { sp_daemon_cuda_kvdecode_reset_cold(handle) };
    if rc == 0 { Ok(()) } else { Err(last_error()) }
}

/// KAIROS P1c-2 — PREFIX SHEAR: O(1) restore of the committed shared prefix.
/// When the SWA ring has never wrapped this residency (dpos <= ring_W), slots
/// [0..P) still hold positions [0..P) untouched, so a NEW chat over the same
/// constant preamble shears dpos back to P instead of the full cold re-prefill.
/// Byte-exact by construction — the rows ARE the rows prefill minted. Errors
/// when the ring has wrapped; the caller falls back to the full-prefill floor.
/// The caller MUST hold the cache Mutex.
///
/// # Safety
/// `handle` must be a live `sp_g4_kv*` from [`open`].
pub unsafe fn shear(handle: *mut c_void, p: i32) -> Result<(), String> {
    if handle.is_null() || p < 0 {
        return Err("kvdecode shear: NULL handle or negative P".to_string());
    }
    // SAFETY: handle live per caller.
    let rc = unsafe { sp_daemon_cuda_kvdecode_shear(handle, p) };
    if rc == 0 { Ok(()) } else { Err(last_error()) }
}

// ── PREFIX SNAPSHOT / RESTORE (2026-07-13) — THE 164-SECOND HELLO ────────────────
//
// The shear above is O(1) and byte-exact, and CANNOT FIRE AT THE OPERATOR'S CONFIG: his
// preamble is 2517 tokens and ring_W is 2048, so the SWA ring wraps INSIDE the preamble and
// the shear correctly refuses. The fallback was a full re-prefill from token 0 — 2679 tokens
// at 60 ms/tok = 164 SECONDS to say "Hello! How are you today?", when 2517 of those tokens
// were already sitting correct in VRAM.
//
// Copying the bytes works regardless of wrapping, because it does not reason about wrapping:
// it restores the cache to the exact state it was in. ~540 MiB over PCIe ≈ 90 ms.
//
// THE BLOB IS OWNED BY RUST. The engine never allocates it, so there is no hidden lifetime,
// nothing to leak when a session dies, and the size is checked on BOTH sides against
// prefix_bytes() — a blob from a different Pmax/ring_W cannot be restored into this session.

/// Open a SCRATCH cache: RING-OFF, one-shot, sized to its own prompt.
///
/// MEASURED, and it was my own bug. `/v1/oneshot` opened its scratch session with plain
/// `open()`, which reads the PROCESS's `SP_G4_KV_RING_W` — so a ~620-token judge call was
/// handed a 2048-slot SWA ring costing ~490 MB IT WOULD NEVER READ. On a 12 GB card with
/// ~100 MB free, the daemon spilled 860 MiB into host memory and everything crawled at
/// 218 ms/tok. I FIXED THE EVICTION AND PAID FOR IT WITH A SPILL — a fix that makes the
/// system slower in a different place is not a fix, it is a trade nobody agreed to.
///
/// A short-lived cache does not want a sliding window. It wants a FULL cache at its own small
/// Pmax: cheaper (620 slots, not 2048), simpler (slot == pos), and it is exactly the
/// "cold, ring-off, full cache" precondition the BATCHED prefill was built for. One decision,
/// two wins.
///
/// # Safety
/// `qm` must be the live `qwen3_model*` from the session.
pub unsafe fn open_scratch(qm: *const c_void, pmax: i32) -> Result<*mut c_void, String> {
    if qm.is_null() || pmax <= 0 {
        return Err("kvdecode open_scratch: NULL model or Pmax<=0".to_string());
    }
    let h = unsafe { sp_daemon_cuda_kvdecode_open_scratch(qm, pmax) };
    if h.is_null() { Err(last_error()) } else { Ok(h) }
}

/// Declare this cache ONE-SHOT: decoded once, released, never continued.
///
/// This is the precondition `gemma4_kv_prefill_batched` actually needs. The guard used to test
/// a PROCESS-WIDE `SP_PERSIST_KV`, so the chat path's need for persistence disqualified the
/// batched accelerant for the judge / reflect / classify calls — which have nothing whatsoever
/// to continue, and were paying ~78 s of per-token prefill to produce one YES/NO token.
///
/// NEVER set this on a cache anyone intends to reuse. A batched-ring prefill does not build the
/// incremental undo-journal the per-token path builds tick by tick, so a NEXT turn that tried to
/// reuse it would come back EMPTY (measured: G-PK2-BATCHRING). That is why the guard exists —
/// it just asked the wrong question.
///
/// # Safety
/// `handle` must be a live `sp_g4_kv*` from [`open`].
pub unsafe fn set_oneshot(handle: *mut c_void, on: bool) -> Result<(), String> {
    if handle.is_null() {
        return Err("kvdecode set_oneshot: NULL handle".to_string());
    }
    let rc = unsafe { sp_daemon_cuda_kvdecode_set_oneshot(handle, on as c_int) };
    if rc == 0 { Ok(()) } else { Err(last_error()) }
}

/// Bytes a prefix snapshot at `p` needs for THIS session's geometry.
///
/// # Safety
/// `handle` must be a live `sp_g4_kv*` from [`open`].
pub unsafe fn prefix_bytes(handle: *mut c_void, p: i32) -> Result<usize, String> {
    if handle.is_null() || p <= 0 {
        return Err("kvdecode prefix_bytes: NULL handle or P<=0".to_string());
    }
    let n = unsafe { sp_daemon_cuda_kvdecode_prefix_bytes(handle, p) };
    if n > 0 { Ok(n as usize) } else { Err(last_error()) }
}

/// Snapshot the cache AS IT STANDS at `dpos == p` into `buf`.
/// The engine REFUSES if dpos != p: we snapshot a STATE, never a guess.
///
/// # Safety
/// `handle` must be a live `sp_g4_kv*` from [`open`]. Caller holds the cache Mutex.
pub unsafe fn snapshot_prefix(handle: *mut c_void, buf: &mut [u8], p: i32) -> Result<usize, String> {
    if handle.is_null() || p <= 0 {
        return Err("kvdecode snapshot_prefix: NULL handle or P<=0".to_string());
    }
    let n = unsafe {
        sp_daemon_cuda_kvdecode_snapshot_prefix(
            handle, buf.as_mut_ptr() as *mut c_void, buf.len() as i64, p)
    };
    if n > 0 { Ok(n as usize) } else { Err(last_error()) }
}

/// Restore the cache to a snapshotted prefix and set dpos/commit_pos to `p`.
/// Works WITH a wrapped ring — that is the entire point of it.
///
/// # Safety
/// `handle` must be a live `sp_g4_kv*` from [`open`]. Caller holds the cache Mutex.
pub unsafe fn restore_prefix(handle: *mut c_void, buf: &[u8], p: i32) -> Result<(), String> {
    if handle.is_null() || p <= 0 {
        return Err("kvdecode restore_prefix: NULL handle or P<=0".to_string());
    }
    let rc = unsafe {
        sp_daemon_cuda_kvdecode_restore_prefix(
            handle, buf.as_ptr() as *const c_void, buf.len() as i64, p)
    };
    if rc == 0 { Ok(()) } else { Err(last_error()) }
}

/// CONTRACT-CHAT-FULLSTACK B1 — toggle byte-exact ("auditable") mode on the
/// resident cache. `on=true` routes the islands+attention through the
/// exact-integer dual-prime CRT-NTT substrate (run-to-run bit-identical);
/// `on=false` restores the float Stage-A path (byte-identical null floor).
/// The caller MUST hold the cache Mutex (the chat path sets it on at request
/// start, off at request end).
///
/// # Safety
/// `handle` must be a live `sp_g4_kv*` from [`open`].
pub unsafe fn set_byteexact(handle: *mut c_void, on: bool) -> Result<(), String> {
    if handle.is_null() {
        return Err("kvdecode set_byteexact: NULL handle".to_string());
    }
    // SAFETY: handle live per caller; glue forwards to gemma4_kv_byteexact_set.
    let rc = unsafe { sp_daemon_cuda_kvdecode_byteexact(handle, if on { 1 } else { 0 }) };
    if rc == 0 { Ok(()) } else { Err(last_error()) }
}

/// Set the KV codec flags (bit0 = SP_KV_SPINOR) on the resident cache.
/// CONTRACT-CUDA-KV-FOUNDATION. flags==0 = float null floor (default).
///
/// # Safety
/// `handle` must be a live `sp_g4_kv*` from [`open`].
pub unsafe fn set_kv_flags(handle: *mut c_void, flags: u32) -> Result<(), String> {
    if handle.is_null() {
        return Err("kvdecode set_kv_flags: NULL handle".to_string());
    }
    let rc = unsafe { sp_daemon_cuda_kvdecode_kv_flags(handle, flags) };
    if rc == 0 { Ok(()) } else { Err(last_error()) }
}

/// CONTRACT-CHAT-FULLSTACK B2 (§6d-b) — replay a stored episode into the resident
/// cache at `[dpos, dpos+npos)` (SP_REPLAY recall into the live turn). `epdir`
/// holds ep.mf/ep.k/ep.v; `zero=true` injects the zeroed reject control. On reject
/// the caller undoes it with `rewind(handle, npos)`. The caller MUST hold the
/// cache Mutex (the chat path replays before decode under the Mutex).
///
/// # Safety
/// `handle` must be a live `sp_g4_kv*` from [`open`].
pub unsafe fn replay(handle: *mut c_void, epdir: &str, npos: i32, zero: bool) -> Result<(), String> {
    if handle.is_null() || npos <= 0 {
        return Err("kvdecode replay: NULL handle or non-positive npos".to_string());
    }
    let c_epdir = match std::ffi::CString::new(epdir) {
        Ok(s) => s,
        Err(_) => return Err("kvdecode replay: epdir has interior NUL".to_string()),
    };
    // SAFETY: handle live per caller; c_epdir owns the NUL-terminated buffer for
    // the duration of the call; glue forwards to gemma4_kv_replay.
    let rc = unsafe {
        sp_daemon_cuda_kvdecode_replay(handle, c_epdir.as_ptr(), npos, if zero { 1 } else { 0 })
    };
    if rc == 0 { Ok(()) } else { Err(last_error()) }
}

/// B3-v10 ablation gate — content-blind `positions` (relative to the episode anchor `base`)
/// by memset-zeroing their K/V rows. Restored by the subsequent `rewind` (transient).
///
/// # Safety
/// `handle` must be a live `sp_g4_kv*` from [`open`].
pub unsafe fn ablate(handle: *mut c_void, base: i32, positions: &[i32]) -> Result<(), String> {
    if handle.is_null() { return Err("kvdecode ablate: NULL handle".to_string()); }
    if positions.is_empty() { return Ok(()); } // empty mask = no-op (no payload-token match)
    let pos: Vec<c_int> = positions.iter().map(|&p| p as c_int).collect();
    // SAFETY: handle live per caller; pos slice valid for the call.
    let rc = unsafe { sp_daemon_cuda_kvdecode_ablate(handle, base as c_int, pos.as_ptr(), pos.len() as c_int) };
    if rc == 0 { Ok(()) } else { Err(last_error()) }
}

/// CONTRACT-CHAT-FULLSTACK B5 (§6e) — TEXT through the single latent entry seam.
/// Per token id, the engine stages `embd[id]*sqrt(E)` into the inject buffer and steps
/// the real id, so the residual entering layer 0 is bit-identical to `prefill(&id,1)`.
/// This is the text SOURCE of the one residual seam (audio/memory enter the same way).
/// The caller MUST hold the cache Mutex.
///
/// # Safety
/// `handle` must be a live `sp_g4_kv*` from [`open`]; `tokens` valid for its length.
pub unsafe fn inject_tokens(handle: *mut c_void, tokens: &[i32]) -> Result<(), String> {
    if handle.is_null() || tokens.is_empty() {
        return Err("kvdecode inject_tokens: NULL handle or empty tokens".to_string());
    }
    // SAFETY: handle live per caller; tokens slice gives ptr+len.
    let rc = unsafe {
        sp_daemon_cuda_kvdecode_inject_tokens(handle, tokens.as_ptr(), tokens.len() as c_int)
    };
    if rc == 0 { Ok(()) } else { Err(last_error()) }
}

/// G-INT-2-FIX — the LIVE recall inject seam. Same residual entry as [`inject_tokens`]
/// but the natively-minted memory K is attenuated by the constant-budget alpha
/// (SP_REPLAY_MTARGET, default 42) so a recalled episode BINDS instead of HIJACKING the
/// synthesis. Use ONLY for the B3-JUDGE/B3-WC live-recall PICK injection — NOT for the
/// prompt-head / B5 frame ingest (those stay full-strength via [`inject_tokens`]).
///
/// # Safety
/// `handle` must be a live `sp_g4_kv*` from [`open`]; `tokens` valid for its length.
pub unsafe fn inject_tokens_atten(handle: *mut c_void, tokens: &[i32]) -> Result<(), String> {
    if handle.is_null() || tokens.is_empty() {
        return Err("kvdecode inject_tokens_atten: NULL handle or empty tokens".to_string());
    }
    // SAFETY: handle live per caller; tokens slice gives ptr+len.
    let rc = unsafe {
        sp_daemon_cuda_kvdecode_inject_tokens_atten(handle, tokens.as_ptr(), tokens.len() as c_int)
    };
    if rc == 0 { Ok(()) } else { Err(last_error()) }
}

/// CONTRACT-CHAT-FULLSTACK B5 (§6e) — the GENERIC residual-frame channel. Inject
/// `frames.len()/E` raw E-float residual vectors at consecutive positions, each minted
/// at `ph_token`. The seam AUDIO (EAR/KAI-3 projector) and MEMORY (decoded episode
/// residuals) feed a turn through. `frames` is row-major `[n_frames][E]`. The caller
/// MUST hold the cache Mutex and ensure `frames.len()` is a multiple of E (n_frames*E).
///
/// # Safety
/// `handle` must be a live `sp_g4_kv*` from [`open`]; `frames` valid for `n_frames*E`.
pub unsafe fn inject_frames(
    handle: *mut c_void,
    frames: &[f32],
    n_frames: i32,
    ph_token: i32,
) -> Result<(), String> {
    if handle.is_null() || frames.is_empty() || n_frames <= 0 {
        return Err("kvdecode inject_frames: NULL handle or empty frames".to_string());
    }
    // SAFETY: handle live per caller; frames slice gives the n_frames*E buffer.
    let rc = unsafe {
        sp_daemon_cuda_kvdecode_inject_frames(handle, frames.as_ptr(), n_frames, ph_token)
    };
    if rc == 0 { Ok(()) } else { Err(last_error()) }
}

/// CONTRACT-CHAT-FULLSTACK B3 (AUTONOMOUS RECALL) — read global-owner K `[0,npos)`
/// out of the resident cache for the daemon's C2 query-signature. `out` must hold
/// `n_global * npos * g_kvd` f32 (the caller sizes it from the known geometry:
/// the dense reference model has 8 global layers, g_kvd=512). Returns the number of global layers
/// written (>0) on success. The cache is byte-untouched.
///
/// # Safety
/// `handle` must be a live `sp_g4_kv*` from [`open`]; `out` valid for the buffer
/// size above; the caller MUST hold the cache Mutex (no concurrent decode).
/// The session's layout: NL, period, kvfs, g_nkv, g_hd, s_nkv, s_hd, ring_W, Pmax,
/// kv_fp16, kv_fp16_g. Published so forensics does not have to guess it.
///
/// # Safety
/// `handle` must be a live `sp_g4_kv*` from [`open`].
pub unsafe fn geometry(handle: *const c_void) -> Result<Vec<i32>, String> {
    if handle.is_null() { return Err("kvdecode geometry: NULL handle".to_string()); }
    let mut out = vec![0i32; 13];
    let n = unsafe { sp_daemon_cuda_kvdecode_geometry(handle, out.as_mut_ptr(), 13) };
    if n > 0 { out.truncate(n as usize); Ok(out) } else { Err(last_error()) }
}

/// The row stride `read_global_k` writes at, straight from the session.
///
/// # Safety
/// `handle` must be a live `sp_g4_kv*` from [`open`].
pub unsafe fn global_kvd(handle: *const c_void) -> Result<usize, String> {
    if handle.is_null() { return Err("kvdecode global_kvd: NULL handle".to_string()); }
    let n = unsafe { sp_daemon_cuda_kvdecode_global_kvd(handle) };
    if n > 0 { Ok(n as usize) } else { Err(last_error()) }
}

pub unsafe fn read_global_k(handle: *const c_void, out: &mut [f32], npos: i32) -> Result<i32, String> {
    if handle.is_null() || out.is_empty() || npos <= 0 {
        return Err("kvdecode read_global_k: NULL handle / empty out / non-positive npos".to_string());
    }
    // SAFETY: handle live per caller; out slice gives the packed buffer ptr.
    let n = unsafe { sp_daemon_cuda_kvdecode_read_global_k(handle, out.as_mut_ptr(), npos) };
    if n > 0 { Ok(n) } else { Err(last_error()) }
}

/// CONTRACT-CHAT-FULLSTACK B3-v2 (q·K AUTONOMOUS RECALL) — read the live query's
/// last-token GLOBAL-layer query for the attention-relevance selector. Runs one
/// non-committing forward of `token` at the live dpos; `out` must hold
/// `n_global * g_nh * g_hd` f32 (the dense reference model: 8 globals, g_nh*g_hd = 16*512 = 8192).
/// Returns the number of global layers written (>0). The cache is unchanged.
///
/// # Safety
/// `handle` must be a live `sp_g4_kv*` from [`open`]; `out` valid for the buffer size
/// above; the caller MUST hold the cache Mutex (no concurrent decode).
pub unsafe fn read_global_q(handle: *mut c_void, token: i32, out: &mut [f32]) -> Result<i32, String> {
    if handle.is_null() || out.is_empty() {
        return Err("kvdecode read_global_q: NULL handle / empty out".to_string());
    }
    // SAFETY: handle live per caller; out slice gives the packed buffer ptr.
    let n = unsafe { sp_daemon_cuda_kvdecode_read_global_q(handle, token, out.as_mut_ptr()) };
    if n > 0 { Ok(n) } else { Err(last_error()) }
}

/// Current decode position (`dpos`), or -1 on NULL.
///
/// # Safety
/// `handle` must be a live `sp_g4_kv*` from [`open`] (or NULL).
pub unsafe fn position(handle: *const c_void) -> i32 {
    // SAFETY: glue handles NULL.
    unsafe { sp_daemon_cuda_kvdecode_position(handle) }
}

/// Free the resident cache. Idempotent / NULL-safe.
///
/// # Safety
/// `handle` must be a live `sp_g4_kv*` from [`open`] (or NULL); not used after.
pub unsafe fn close(handle: *mut c_void) {
    if !handle.is_null() {
        // SAFETY: handle live per caller; not used after close.
        unsafe { sp_daemon_cuda_kvdecode_close(handle) };
    }
}

// ── §6b dispatch-table trampolines ──────────────────────────────────────────
// The bindgen `sp_kvdecode_dispatch_fn` fn-ptr fields are typed over the opaque
// `sp_kvdecode_handle` + `qwen3_model*` (as `*const c_void`); the C glue uses
// plain `void*`. These thin `extern "C"` trampolines bridge the two (an
// `sp_kvdecode_handle*` IS the glue's `void*` handle — an `sp_g4_kv*`). They
// forward straight to the already-filled glue symbols, with `decode_step`
// bumping the gate's step counter so the harness can confirm the verb was hit.

type KvHandle = crate::ffi_l1::sp_kvdecode_handle;

unsafe extern "C" fn tramp_open(
    qm_opaque: *const c_void,
    pmax: c_int,
    out: *mut *mut KvHandle,
) -> c_int {
    // SAFETY: glue checks NULL qm; `out` written with the opaque handle.
    let h = unsafe { sp_daemon_cuda_kvdecode_open(qm_opaque, pmax) };
    if h.is_null() {
        return -1;
    }
    if !out.is_null() {
        unsafe { *out = h as *mut KvHandle };
    }
    0
}

unsafe extern "C" fn tramp_prefill(h: *mut KvHandle, tokens: *const i32, n_tok: c_int) -> c_int {
    // SAFETY: glue validates args; handle is the opaque void* cast.
    unsafe { sp_daemon_cuda_kvdecode_prefill(h as *mut c_void, tokens, n_tok) }
}

unsafe extern "C" fn tramp_decode_step(h: *mut KvHandle, token: i32, logits: *mut f32) -> c_int {
    KVDECODE_STEP_COUNT.fetch_add(1, Ordering::Relaxed);
    // SAFETY: glue forwards to gemma4_kv_decode_logits on the resident cache.
    unsafe { sp_daemon_cuda_kvdecode_step(h as *mut c_void, token, logits) }
}

unsafe extern "C" fn tramp_rewind(h: *mut KvHandle, n: c_int) -> c_int {
    // SAFETY: glue validates args.
    unsafe { sp_daemon_cuda_kvdecode_rewind(h as *mut c_void, n) }
}

unsafe extern "C" fn tramp_position(h: *const KvHandle) -> c_int {
    // SAFETY: glue is NULL-safe.
    unsafe { sp_daemon_cuda_kvdecode_position(h as *const c_void) }
}

unsafe extern "C" fn tramp_close(h: *mut KvHandle) {
    // SAFETY: glue is NULL-safe.
    unsafe { sp_daemon_cuda_kvdecode_close(h as *mut c_void) };
}

/// The dispatch table handed to L1. `'static` so the pointer stays valid for
/// the whole process — L1 stores `&DT` and re-emits the fn pointers per decode.
static DT: crate::ffi_l1::sp_kvdecode_dispatch_fn = crate::ffi_l1::sp_kvdecode_dispatch_fn {
    open: Some(tramp_open),
    prefill: Some(tramp_prefill),
    decode_step: Some(tramp_decode_step),
    rewind: Some(tramp_rewind),
    position: Some(tramp_position),
    close: Some(tramp_close),
};

/// Register the CUDA KV-decode backend with an L1 session.
///
/// Opens the resident `sp_g4_kv` cache (via the glue `open`) and registers the
/// §6b dispatch table with the session through
/// `sp_session_register_kvdecode_backend`. After this returns, the session's
/// `sp_decode_step` routes the single-token forward through `tramp_decode_step`
/// → `gemma4_kv_decode_logits` on the resident handle.
///
/// Returns the opaque KV handle on success so the caller (AppState) can own its
/// lifetime and pass it back at `close` time (the resident cache is freed by
/// `release_for_model`). The caller MUST drive `prefill` (history ingest) on
/// the returned handle before the first `sp_decode_step`.
///
/// # Safety
/// `session_raw` must be a valid `*mut sp_session` with the L2-side Mutex held;
/// `qm_opaque` the session's borrowed `qwen3_model*` (valid for the session
/// lifetime).
pub unsafe fn register_with_session(
    session_raw: *mut crate::ffi_l1::sp_session,
    qm_opaque: *const c_void,
    pmax: i32,
) -> Result<*mut c_void, String> {
    // Step 1: open the resident KV cache.
    // SAFETY: caller guarantees qm_opaque + session validity.
    let handle = unsafe { open(qm_opaque, pmax) }?;

    // Step 2: point sp_decode_step at the glue dispatch table on this session.
    // SAFETY: caller holds the SpSession's Mutex; no concurrent decode.
    let rc = unsafe {
        crate::ffi_l1::sp_session_register_kvdecode_backend(
            session_raw,
            handle as *mut KvHandle,
            &DT as *const crate::ffi_l1::sp_kvdecode_dispatch_fn,
        )
    };
    if rc != crate::ffi_l1::sp_status_SP_OK {
        // Roll back the resident cache so we don't leak it on a failed register.
        unsafe { close(handle) };
        return Err(format!(
            "sp_session_register_kvdecode_backend → status={rc}: {}",
            last_error()
        ));
    }

    Ok(handle)
}

/// Tear down a resident KV cache opened via [`register_with_session`].
/// Called at AppState shutdown. Idempotent / NULL-safe.
///
/// # Safety
/// `handle` must be an `sp_g4_kv*` from [`register_with_session`] (or NULL).
pub unsafe fn release_for_model(handle: *mut c_void) {
    // SAFETY: close is NULL-safe; handle not used after.
    unsafe { close(handle) };
}

/// Fetch the last engine error string via the L1 ABI (`sp_last_error`).
fn last_error() -> String {
    // SAFETY: sp_last_error returns a process-static NUL-terminated C string.
    unsafe { std::ffi::CStr::from_ptr(crate::ffi_l1::sp_last_error()) }
        .to_string_lossy()
        .into_owned()
}

// ─── GEODESIC F3 (ADR-003 §5) — one-shot post-output_norm feature tap ───────
// Direct `gemma4_kv_*` symbol, same link pattern eagle_accept.rs already proves
// under `wire_cuda_backend` (NOT a glue-table row — additive, no C change).
// Semantics (cuda_forward.cu:4857): ARMS a one-shot capture; the NEXT decode
// step on this session D2H-copies its post-output_norm hidden (E floats — the
// exact feature the LM head consumes) into `feat`, then disarms. Never called
// unless `SP_F3_CAPTURE` is set ⇒ default-off = byte-identical null floor.
unsafe extern "C" {
    fn gemma4_kv_capture_feat(handle: *mut c_void, feat: *mut f32) -> c_int;
}

/// Arm the one-shot feature capture (GEODESIC F3, ADR-003 §5).
///
/// The capture fires on the NEXT [`decode_step`] on the same handle; callers
/// MUST place the arm immediately before an unconditional `decode_step` so an
/// armed pointer can never outlive its buffer (see the F3 site in routes.rs,
/// which `mem::forget`-leaks the buffer on the step's error path as a dangling-
/// write guard).
///
/// # Safety
/// `handle` must be a live `sp_g4_kv*` from [`open`]; `feat` must hold at least
/// the model's hidden dim (the dense reference model: 3840) and stay alive (unmoved, unresized)
/// until the next `decode_step` on this handle returns.
pub unsafe fn capture_feat_arm(handle: *mut c_void, feat: &mut [f32]) -> Result<(), String> {
    if handle.is_null() || feat.is_empty() {
        return Err("kvdecode capture_feat_arm: NULL handle or empty buffer".to_string());
    }
    // SAFETY: handle live per caller; feat outlives the next step per caller.
    let rc = unsafe { gemma4_kv_capture_feat(handle, feat.as_mut_ptr()) };
    if rc == 0 { Ok(()) } else { Err(last_error()) }
}

// ── GEODESIC G-FM-STEER (ADR-003 §4.2) — persistent pre-head steering ────────
// The engine copies `vec` to a device buffer on arm, so the slice need only live
// for the duration of this call (unlike the capture tap).
unsafe extern "C" {
    fn gemma4_kv_steer(handle: *mut c_void, vec: *const f32, alpha: f32) -> c_int;
}

/// Arm (non-empty `vec`, `alpha != 0`) or disarm (empty `vec` or `alpha == 0`)
/// persistent pre-head steering: every subsequent decode step adds `alpha*vec`
/// to the post-output_norm hidden before the LM head (token-distribution bias
/// only — cannot touch K/V or the residual stream). Never called unless
/// `SP_STEER_VEC` is set ⇒ default-off = byte-identical null floor.
///
/// # Safety
/// `handle` must be a live `sp_g4_kv*` from [`open`].
pub unsafe fn steer_set(handle: *mut c_void, vec: &[f32], alpha: f32) -> Result<(), String> {
    if handle.is_null() {
        return Err("kvdecode steer_set: NULL handle".to_string());
    }
    let disarm = vec.is_empty() || alpha == 0.0;
    // SAFETY: handle live per caller; engine copies vec before returning.
    let rc = unsafe {
        gemma4_kv_steer(handle,
            if disarm { std::ptr::null() } else { vec.as_ptr() },
            if disarm { 0.0 } else { alpha })
    };
    if rc == 0 { Ok(()) } else { Err(last_error()) }
}
