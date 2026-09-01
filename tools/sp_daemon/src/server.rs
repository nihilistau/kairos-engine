use std::sync::Arc;

use axum::{routing::{get, post}, Router, extract::DefaultBodyLimit};
use tower_http::{cors::CorsLayer, services::ServeDir};

use crate::routes::{
    v1_abort, v1_capture, v1_chat, v1_chat_stream_stub, v1_debug_backend_counts, v1_dialogue,
    v1_dsp_echo, v1_dsp_model_info, v1_events, v1_mesh_peers, v1_metrics, v1_node_telemetry,
    v1_pouw_ledger, v1_receipts,
};
// ONE-SHOT lives on the CUDA kvdecode path (it needs a scratch sp_g4_kv session), so it only
// exists with the wire_cuda_backend feature — same as the resident cache it exists to protect.
#[cfg(feature = "wire_cuda_backend")]
use crate::routes::v1_oneshot;
use crate::state::AppState;

/// Locate the console static dir independently of the launch CWD:
/// 1. SP_CONSOLE_DIR env override;
/// 2. `frontend_mockups` beside the running exe's crate root (exe lives at
///    tools/sp_daemon/target-*/release/sp-daemon.exe → up 3 = tools/sp_daemon);
/// 3. the historical CWD-relative fallback ("frontend_mockups").
fn frontend_dir() -> std::path::PathBuf {
    if let Ok(d) = std::env::var("SP_CONSOLE_DIR") {
        if !d.trim().is_empty() { return std::path::PathBuf::from(d); }
    }
    if let Ok(exe) = std::env::current_exe() {
        let mut p = exe.as_path();
        for _ in 0..4 {
            match p.parent() { Some(par) => p = par, None => break }
            let cand = p.join("frontend_mockups");
            if cand.join("console.html").is_file() { return cand; }
        }
    }
    std::path::PathBuf::from("frontend_mockups")
}

/// Unified router (Phase 2-L3.FG): the L1-backed inference + PoUW + mesh surface
/// serves both host and android (the C ABI links on android now). The two
/// `/v1/dsp/*` handlers each have host (501) and android (real) cfg variants, so
/// G-VERBATIM K-diff forensics route (CUDA-only handler).
#[cfg(feature = "wire_cuda_backend")]
fn kdiff_route() -> Router<Arc<AppState>> {
    Router::new()
        .route("/v1/debug/kdiff", post(crate::routes::v1_debug_kdiff))
        // Is the batched forward the same forward? Read-only, scratch session.
        .route("/v1/debug/kvdiff_batch", post(crate::routes::v1_debug_kvdiff_batch))
}
#[cfg(not(feature = "wire_cuda_backend"))]
fn kdiff_route() -> Router<Arc<AppState>> {
    Router::new()
}

/// ONE-SHOT: a call that is never continued must not cost a conversation.
/// CUDA-only — it needs a scratch `sp_g4_kv` session, which is the same machinery as the
/// resident cache it exists to protect.
#[cfg(feature = "wire_cuda_backend")]
fn oneshot_route() -> Router<Arc<AppState>> {
    Router::new().route("/v1/oneshot", post(v1_oneshot))
}
#[cfg(not(feature = "wire_cuda_backend"))]
fn oneshot_route() -> Router<Arc<AppState>> {
    Router::new()
}

/// SEM S1 (docs/SEMANTICS.md): the query-side L5 embed. CUDA-only — it needs the same
/// scratch `sp_g4_kv` machinery as /v1/oneshot, and honors the same doctrine: the
/// resident cache is not read, not written, not evicted.
#[cfg(feature = "wire_cuda_backend")]
fn embed_route() -> Router<Arc<AppState>> {
    Router::new()
        .route("/v1/embed", post(crate::routes::v1_embed))
        // the learned W_c + NULL selector as a read-only, stateless oracle (Phase C2
        // successor; the scoreboard is harness_tests/sem_wc_score.py)
        .route("/v1/recall_rank", post(crate::routes::v1_recall_rank))
}
#[cfg(not(feature = "wire_cuda_backend"))]
fn embed_route() -> Router<Arc<AppState>> {
    Router::new()
}

/// the routes are wired unconditionally and resolve per target.
pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/metrics",        get(v1_metrics))
        .route("/v1/chat",           post(v1_chat))
        // KAI-5 read-only hidden tap (default-off SP_HIDDEN_TAP=1; offline Voice-Head distill)
        .route("/v1/hidden",         post(crate::routes::v1_hidden))
        .route("/v1/chat/stream",    get(v1_chat_stream_stub))
        // Chat-integration: MeMo (Grounding → Entity ID → Synthesis) dialogue.
        .route("/v1/dialogue",       post(v1_dialogue))
        .route("/v1/abort/:id",      post(v1_abort))
        .route("/v1/capture",        post(v1_capture))
        .route("/v1/receipts",       get(v1_receipts))
        .route("/v1/events",         get(v1_events))
        .route("/v1/node/telemetry", get(v1_node_telemetry))
        .route("/v1/mesh/peers",     get(v1_mesh_peers))
        .route("/v1/pouw/ledger",    get(v1_pouw_ledger))
        .route("/v1/dsp/echo",       post(v1_dsp_echo))
        .route("/v1/dsp/model_info", get(v1_dsp_model_info))
        // Sprint WIRE-HEX: hex forward + NTT.5b/c dispatch counters; reads
        // process-static atomics, no L1 calls. Exposes wire_hex_active so
        // the smoke harness validates startup registration as well as
        // first-prefill dispatch.
        .route("/v1/debug/backend_counts", get(v1_debug_backend_counts))
        // G-VERBATIM forensics (2026-07-12): are two IDENTICAL tokens at different
        // positions distinguishable in the STORED KEYS? If not, attention cannot
        // tell the two "4"s of "4471" apart — the observed copy failure. CUDA only;
        // on other backends the handler is absent and this route is not wired.
        .merge(kdiff_route())
        // ONE-SHOT: a call that is never continued must not cost a conversation.
        // The watch judge / reflection / classifier were sending ~1450-token prompts down the
        // ONE RESIDENT KV SLOT — the same one holding his conversation. That cost 78 s of
        // per-token prefill to produce a single YES/NO token, AND evicted his chat on the way
        // out. They get their own scratch cache now, batched (legal: nothing will continue it),
        // released at the end. The resident KV is not read, not written, and NOT EVICTED.
        .merge(oneshot_route())
        // SEM S1: /v1/embed — l5_query_embed of a text, ep.l5 provenance, read-only.
        .merge(embed_route())
        // Console static files. ServeDir with a RELATIVE path resolves against the
        // process CWD, which depends on where the launcher ran (G-DENSE-SERVE aftermath:
        // engine-root launches 404'd the console). Resolve robustly: prefer the dir
        // beside the crate (exe at target-*/release/ => ../../../frontend_mockups),
        // fall back to CWD-relative for the historical tools/sp_daemon launch pattern.
        .fallback_service(ServeDir::new(frontend_dir()))
        .layer(CorsLayer::permissive())
        // KAI-4 native audio: inject_frames carries ~750 x 3840 f32 audio embeddings
        // (one per 40ms). As JSON that's tens of MB — Axum's 2MB default rejected it
        // (HTTP 413). Raise to 256 MB (audio + memory-frame injection headroom).
        .layer(DefaultBodyLimit::max(256 * 1024 * 1024))
        .with_state(state)
}
