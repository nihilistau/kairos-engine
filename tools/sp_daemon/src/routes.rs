use std::convert::Infallible;
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::Arc;
use std::time::Duration;

use axum::extract::{Path, State};
use axum::extract::ws::{Message as WsMessage, WebSocket, WebSocketUpgrade};
use axum::http::{header, HeaderName, HeaderValue};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::{http::StatusCode, Json};
use serde::{Deserialize, Serialize};
use tokio::task;
use tokio_stream::wrappers::{BroadcastStream, ReceiverStream};
use tokio_stream::StreamExt as _;

use std::sync::atomic::AtomicBool;
use crate::state::{AppState, DaemonEvent};
use crate::tokenizer::{Message, PushResult, TokenDecodeBuffer};

// ── SSE header helper ──────────────────────────────────────────────────────

fn sse_response(sse: impl IntoResponse) -> Response {
    let mut r = sse.into_response();
    r.headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    r.headers_mut().insert(
        HeaderName::from_static("x-accel-buffering"),
        HeaderValue::from_static("no"),
    );
    r
}

// ── /v1/metrics ───────────────────────────────────────────────────────────

#[derive(Serialize)]
pub(crate) struct Metrics {
    tokens_per_sec: f64,
    ram_svm_bytes: u64,
    peers: u32,
    phase: &'static str,
    session_pos: u64,
}

pub async fn v1_metrics(State(state): State<Arc<AppState>>) -> Json<Metrics> {
    let session_pos = {
        let guard = state.session.as_ref().expect("L1 session unavailable (qwen36 lane)").lock().unwrap();
        guard.position().unwrap_or(0) as u64
    };

    let elapsed = state.started_at.elapsed().as_secs_f64();
    let decoded = state.tokens_decoded.load(Ordering::Relaxed);
    let tps = if elapsed > 0.1 { decoded as f64 / elapsed } else { 0.0 };

    Json(Metrics {
        tokens_per_sec: tps,
        ram_svm_bytes: 0,
        peers: 0,
        phase: "lat-phase-2-l3-tok-closed",
        session_pos,
    })
}

// ── /v1/debug/backend_counts (Sprint WIRE-HEX) ────────────────────────────
//
// Surfaces the dispatch counters for the optional accelerator backends
// (hex forward, hex NTT). T_WIRE_HEX_BACKEND_DISPATCHES reads
// `hex_forward_count` after one prefill of a known prompt; PASS criterion is
// > 0. wire_hex_active reports whether the registration succeeded at
// startup (independent of whether a prefill has run yet).

#[derive(Serialize)]
pub(crate) struct BackendCounts {
    /// Sprint WIRE-HEX: gemma3_forward_hexagon dispatcher hit count
    /// since process start. > 0 after one prefill when SP_DAEMON_BACKEND=hex
    /// is set AND the daemon was built with --features wire_hex_backend.
    /// Always 0 on host builds and on android without the feature.
    hex_forward_count: u64,
    /// Sprint WIRE-HEX: whether sp_session_register_forward_backend was
    /// invoked successfully on the target session at startup. False when
    /// SP_DAEMON_BACKEND is unset / != "hex", when the feature was off at
    /// build time, or when registration failed (see daemon log).
    wire_hex_active: bool,
    /// NTT.5b/c: hex NTT forward + inverse dispatch counters (Bluestein
    /// inner kernels via FastRPC). Always 0 when SP_ENGINE_NTT_ATTN_HEX
    /// is unset. Independent of WIRE-HEX (different ABI hook).
    ntt_hex_forward_count: u64,
    ntt_hex_inverse_count: u64,
    /// Sprint WIRE-CPU: engine CPU AVX-512 backend dispatcher hit count
    /// since process start. > 0 after one prefill when SP_DAEMON_BACKEND=cpu
    /// is set AND the daemon was built with --features wire_cpu_backend.
    /// Always 0 without the feature.
    cpu_forward_count: u64,
    /// Sprint WIRE-CPU: whether sp_session_register_forward_backend was
    /// invoked successfully on the target session at startup. False when
    /// SP_DAEMON_BACKEND is unset / != "cpu", when the feature was off at
    /// build time, or when registration failed.
    wire_cpu_active: bool,
    /// Sprint WIRE-CUDA: gemma3_forward_cuda / qwen3_forward_cuda dispatcher
    /// hit count since process start. > 0 after one prefill when
    /// SP_DAEMON_BACKEND=cuda is set AND the daemon was built with
    /// --features wire_cuda_backend. Always 0 on builds without the feature.
    cuda_forward_count: u64,
    /// Sprint WIRE-CUDA: whether sp_session_register_forward_backend was
    /// invoked successfully on the target session at startup. False when
    /// SP_DAEMON_BACKEND is unset / != "cuda", when the feature was off at
    /// build time, or when registration failed (see daemon log).
    wire_cuda_active: bool,
    /// Sprint WIRE-VULKAN: gemma3_forward_vulkan / qwen3_forward_vulkan
    /// dispatcher hit count since process start. > 0 after one prefill
    /// when SP_DAEMON_BACKEND=vulkan is set AND the daemon was built with
    /// --features wire_vulkan_backend. Counter is bumped BEFORE the engine
    /// call (per the wire_vulkan trampoline) — increments even if the
    /// engine returns the known OOM error from a Vulkan device that lacks
    /// budget for the model arena (the pre-existing M_GEMMA3_VULKAN /
    /// M_QWEN3_VULKAN OOM bug; see WIRE-VULKAN-OOM-BUGFIX follow-on).
    /// Always 0 when the feature is off at build time.
    vulkan_forward_count: u64,
    /// Sprint WIRE-VULKAN: whether sp_session_register_forward_backend was
    /// invoked successfully on the target session at startup. False when
    /// SP_DAEMON_BACKEND is unset / != "vulkan", when the feature was off
    /// at build time, or when registration failed (see daemon log).
    wire_vulkan_active: bool,
}

pub async fn v1_debug_backend_counts(State(state): State<Arc<AppState>>) -> Json<BackendCounts> {
    let hex_forward_count = {
        #[cfg(all(target_os = "android", feature = "wire_hex_backend"))]
        { sp_daemon::hex_forward_dispatch::dispatch_count() }
        #[cfg(not(all(target_os = "android", feature = "wire_hex_backend")))]
        { 0u64 }
    };
    let (ntt_hex_forward_count, ntt_hex_inverse_count) = {
        #[cfg(target_os = "android")]
        { sp_daemon::ntt_hex_dispatch::dispatch_counts() }
        #[cfg(not(target_os = "android"))]
        { (0u64, 0u64) }
    };
    let cpu_forward_count = {
        // crate::cpu_forward_dispatch lives in the BINARY crate (main.rs),
        // not the lib crate, because WIRE-CPU is host-targeted and the
        // binary already has L1 bindings via its own `mod ffi`. Routes is
        // a binary-crate module, so `crate::cpu_forward_dispatch` resolves.
        #[cfg(feature = "wire_cpu_backend")]
        { crate::cpu_forward_dispatch::dispatch_count() }
        #[cfg(not(feature = "wire_cpu_backend"))]
        { 0u64 }
    };
    let cuda_forward_count = {
        #[cfg(feature = "wire_cuda_backend")]
        { sp_daemon::cuda_forward_dispatch::dispatch_count() }
        #[cfg(not(feature = "wire_cuda_backend"))]
        { 0u64 }
    };
    let vulkan_forward_count = {
        #[cfg(feature = "wire_vulkan_backend")]
        { sp_daemon::vulkan_forward_dispatch::dispatch_count() }
        #[cfg(not(feature = "wire_vulkan_backend"))]
        { 0u64 }
    };
    Json(BackendCounts {
        hex_forward_count,
        wire_hex_active: state.wire_hex_active,
        ntt_hex_forward_count,
        ntt_hex_inverse_count,
        cpu_forward_count,
        wire_cpu_active: state.wire_cpu_active,
        cuda_forward_count,
        wire_cuda_active: state.wire_cuda_active,
        vulkan_forward_count,
        wire_vulkan_active: state.wire_vulkan_active,
    })
}

// ── /v1/chat ──────────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct ChatRequest {
    pub prompt: Option<String>,
    pub messages: Option<Vec<Message>>,
    pub prompt_tokens: Option<Vec<i32>>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default)]
    pub stop: Vec<String>,
    // CONTRACT-CHAT-FULLSTACK B1/S1 — byte-exact "auditable mode". The turn
    // decodes through the exact-integer islands + dual-prime CRT-NTT attention on
    // the resident cache (run-to-run bit-identical AND build-independent, the
    // AUDITABILITY axis + the FP-reorder-immunity fix). S1 makes this the
    // DEFAULT chat decode path: `None` (field absent) ⇒ ON. Set `false`
    // explicitly to opt out to the Stage-A float path; `raw_logits:true` also
    // forces it off (the float determinism/null-floor reference). Only honored on
    // the resident-cache kvdecode chat path.
    #[serde(default)]
    pub byteexact: Option<bool>,
    // CONTRACT-CHAT-FULLSTACK B2 (§6d-b) — XBAR episode REPLAY into this turn. When
    // set to an episode directory (holding ep.mf/ep.k/ep.v), the resident cache
    // replays that stored episode's owner K/V at [dpos,dpos+replay_npos) right after
    // the prompt prefill and before decode, so a prior memory rolls into the live
    // turn (SP_REPLAY recall, C2 #222). Default None = the B1/Stage-A path untouched
    // (byte-identical null floor). Only honored on the resident-cache kvdecode (resident-cache)
    // chat path. `replay_npos` bounds how many positions to recall (0 = unset → skip).
    #[serde(default)]
    pub replay: Option<String>,
    #[serde(default)]
    pub replay_npos: i32,
    // CONTRACT-CHAT-FULLSTACK B3 — AUTONOMOUS MEMORY RECALL. When true (and the
    // daemon loaded an episode registry via SP_RECALL_REGISTRY), the daemon
    // computes THIS turn's 256-bit C2 query signature from the resident cache's
    // global-layer K (after the prompt prefill), Hamming-matches it against the
    // registry, and — if the best match clears TAU_BITS — auto-replays that
    // episode into the live turn BEFORE decode. No operator `replay` field
    // needed: the model selects the relevant memory ON ITS OWN. Default
    // None/false = OFF = the byte-untouched non-recall path (the null floor; the
    // foreign-reject safety leg also runs only when this is on). An explicit
    // `replay` still wins (operator override) and suppresses auto-recall. Only
    // honored on the resident-cache kvdecode chat path.
    #[serde(default)]
    pub auto_recall: Option<bool>,
    // CONTRACT-CHAT-FULLSTACK A2-polish — null-floor opt-out. Default false =
    // the DEFAULT served chat, with full control-token suppression
    // (`<image|>`/`<audio|>`/`<|turn>`/… masked to -inf) so output is CLEAN at
    // every temperature incl. greedy. When true, suppression is SKIPPED (the
    // sampler is built with an empty suppress set), so the raw, un-suppressed
    // logits drive selection — this reproduces the prior greedy output
    // bit-for-bit and is the reference the byte-exact / B1 determinism leg
    // compares against (pair with temperature:0 + byteexact:true). Use only for
    // the auditability/determinism null floor; not for normal chat.
    #[serde(default)]
    pub raw_logits: bool,
    // CONTRACT-CHAT-FULLSTACK B5 — the SINGLE LATENT ENTRY POINT (CONTRACT §6).
    // When set true, the prompt is ingested through the ONE residual seam
    // (gemma4_kv_inject_tokens: per token, embd[id]*sqrt(E) staged into the inject
    // override → the model mints K/V natively) INSTEAD of gemma4_kv_prefill(ids).
    // The residual entering layer 0 is bit-identical to prefill by construction
    // (same embed arithmetic + the real id as the step token so PLE matches), so
    // output is bit-identical — this PROVES text, audio, and memory all enter
    // through the one seam. Default None ⇒ the prefill path (untouched null floor);
    // flip the daemon default by passing single_entry:true. Only honored on the
    // resident-cache kvdecode chat path.
    #[serde(default)]
    pub single_entry: Option<bool>,
    // N5: live end-of-turn logit bias (the coherence knob). None => the daemon's
    // SP_EOT_BIAS env default. Tunable per-request from the console GUI.
    #[serde(default)]
    pub eot_bias: Option<f32>,
    // SELF-REPEAT BAN: forbid N-grams drawn from `self_repeat_text` (= her PREVIOUS reply)
    // only, NOT from the whole prompt. This is the narrow replacement for the global
    // no_repeat_ngram that had to be turned off because it banned quoting (G-VERBATIM).
    // CONSTRAINED TOOL DECODING. The harness owns the grammar (harness/mcp/grammar.py) and

    // sends the names it will accept; the engine makes everything else UNREACHABLE. Empty =

    // off, which is the default, so nothing changes for any caller that does not ask.

    #[serde(default)]

    pub tool_names: Vec<String>,

    #[serde(default)]
    pub self_repeat_ngram: Option<usize>,
    #[serde(default)]
    pub self_repeat_text: Option<String>,
    // CONTRACT-CHAT-FULLSTACK B5 — the GENERIC residual-frame channel. Raw E-float
    // residual vectors fed straight to gemma4_kv_inject_seq (the seam AUDIO/EAR and
    // MEMORY-as-residual sources also use). Each inner Vec is one E-length frame;
    // they are injected at consecutive positions right after the prompt prefill and
    // before decode, each minted at `inject_ph` (default = the gemma-4 audio
    // placeholder 258881). Use with `prompt`/`messages` (the text turn scaffold) so
    // the model has context to digest the frames. Default None = no frames (null
    // floor). Only honored on the resident-cache kvdecode chat path.
    #[serde(default)]
    pub inject_frames: Option<Vec<Vec<f32>>>,
    #[serde(default = "default_inject_ph")]
    pub inject_ph: i32,
    /// Optional marker tokens minted immediately BEFORE / AFTER the injected
    /// frames, through the residual seam. Gemma needs <start_of_image> 255999 and
    /// <end_of_image> 258882 around an image block; audio needs neither, so both
    /// default to None and the audio path is unchanged.
    #[serde(default)]
    pub inject_open: Option<i32>,
    #[serde(default)]
    pub inject_close: Option<i32>,
    // CONTRACT-CHAT-FULLSTACK A2 — sampling knobs (L2 owns sampling over the
    // full-vocab logits row). Flattened so the request body carries them at the
    // top level: {"prompt":"...","temperature":0.7,"top_p":0.95,...}. Defaults
    // are the contract's pre-registered values; `temperature:0` = greedy argmax
    // (bit-identical to the prior hardcoded path, the G-CHAT-A2 determinism leg).
    #[serde(flatten)]
    pub sampling: crate::sampler::SamplingParams,
}

fn default_max_tokens() -> u32 {
    256
}

/// B5: default placeholder token for the generic inject_frames channel — the
/// gemma-4 audio_token_id (the KAI-3 audio port's mint placeholder).
fn default_inject_ph() -> i32 {
    258881
}

#[derive(Serialize)]
struct ChatDelta {
    delta: String,
    chat_id: u64,
}

pub async fn v1_chat(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatRequest>,
) -> Response {
    // Exactly one input field required.
    let n_inputs = req.prompt.is_some() as u8
        + req.messages.is_some() as u8
        + req.prompt_tokens.is_some() as u8;
    if n_inputs == 0 {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "one of prompt / messages / prompt_tokens required"})),
        )
            .into_response();
    }
    if n_inputs > 1 {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "only one of prompt / messages / prompt_tokens may be set"})),
        )
            .into_response();
    }

    // B4 NIGHTSHIFT: capture THIS turn's last user message text (raw, NO chat
    // template — match the curator, which captured raw needle sentences) so the
    // turn-end consolidation hook can mint a position-0 standalone episode. Only
    // meaningful when `messages` was supplied; `prompt`/`prompt_tokens` callers
    // pass None (no consolidation). Cheap clone; ignored unless SP_B4_NIGHTSHIFT=1.
    let raw_user: Option<String> = req.messages.as_ref().and_then(|ms| {
        ms.iter().rev().find(|m| m.role == "user").map(|m| m.content.clone())
    });
    // RECALL MULTI-TURN FIX (2026-07-02, G-ONECONFIG-LIVE C-phase root cause): recall
    // delivery rebuilds the prompt and previously kept ONLY the last user message —
    // the conversation history vanished, so a turn-2 question about turn-1 content
    // could never be answered on a recall turn. Keep a clone of the full message
    // list so the systemecho delivery can preserve the conversation.
    let orig_msgs: Option<Vec<Message>> = req.messages.clone();

    // LIVE CONSOLIDATION HOOK: dump the current conversation to SP_CURRENT_CONVO so the
    // harness agency scheduler can consolidate the LIVE chat (durable facts -> mid-term
    // registry, transcript -> long-term MEM-OKF full+summary) on its heartbeat -- no manual
    // step. Best-effort, messages-only; unset env = no-op (byte-identical null floor).
    if let Ok(conv_path) = std::env::var("SP_CURRENT_CONVO") {
        if let Some(ms) = req.messages.as_ref() {
            let arr: Vec<serde_json::Value> = ms.iter()
                .map(|m| serde_json::json!({"role": m.role, "content": m.content}))
                .collect();
            if let Ok(s) = serde_json::to_string(&arr) {
                let _ = std::fs::write(&conv_path, s);
            }
        }
    }

    // Tokenize input.
    let tokenizer = state.tokenizer.clone();
    let tokens: Vec<i32> = if let Some(ids) = req.prompt_tokens {
        ids
    } else if let Some(prompt_text) = req.prompt {
        match tokenizer.encode(&prompt_text) {
            Ok(ids) => ids,
            Err(e) => {
                return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": e})))
                    .into_response();
            }
        }
    } else {
        let messages = req.messages.unwrap();
        // CONTRACT-CHAT-FULLSTACK S1: build the prompt at the TOKEN level so the
        // gemma-4 turn structure uses its REAL control tokens (`<|turn>`=105 /
        // `<turn|>`=106), not the literal `<start_of_turn>`/`<end_of_turn>`
        // strings that the encoder shatters into ordinary text (the
        // "gemma3-template-on-a-gemma4-model" bug). apply_template_ids emits the
        // control ids directly and routes only the role/content through the C BPE
        // encoder (per-fragment forced BOS stripped).
        match tokenizer.apply_template_ids(&messages) {
            Ok(ids) => ids,
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({
                        "error": "chat_template_unavailable",
                        "detail": e,
                        "hint": "use prompt or prompt_tokens"
                    })),
                )
                    .into_response();
            }
        }
    };

    // CONTRACT-CHAT-FULLSTACK S1 debug: log the head of the assembled prompt
    // token ids so the turn-token structure (<|turn>=105 … <turn|>=106) is
    // verifiable in the daemon log.
    {
        let head: Vec<i32> = tokens.iter().take(12).copied().collect();
        let tail: Vec<i32> = tokens.iter().rev().take(6).rev().copied().collect();
        tracing::info!("S1 prompt ids: n={} head={:?} tail={:?}", tokens.len(), head, tail);
    }

    // ── NORTHSTAR serve (CONTRACT-QWEN36-SERVE): the qwen36 35B-A3B lane ─────
    // When the loaded model is the GDN+MoE hybrid (arch_id 8), the gemma L1
    // session/kvdecode machinery below does not apply: decode via qwen36_step
    // (persistent recurrent state, GPU hybrid hooks booted at daemon start —
    // G-MOE-GPU4-PINNED 6.073 tok/s / 337x) and stream the same ChatDelta/[DONE]
    // SSE shape the console already speaks. v1 = greedy argmax (the gate's
    // determinism leg); sampling knobs come after G-QWEN36-SERVE is GREEN.
    if let Some(lane) = state.qwen36_lane.as_ref() {
        let lane = lane.clone();
        let tok = tokenizer.clone();
        let cancel = Arc::new(AtomicI32::new(0));
        let chat_id = state.sessions.register(cancel.clone());
        let sessions = state.sessions.clone();
        let max_new = req.max_tokens as usize;
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(64);
        tokio::task::spawn_blocking(move || {
            let eos = tok.eos_ids.clone();
            // Split-token UTF-8 guard: BPE token bytes may end mid-codepoint;
            // hold the incomplete tail back until the next token completes it.
            let mut pending: Vec<u8> = Vec::new();
            let res = lane.run_turn(&tokens, max_new, &eos, |id| {
                if cancel.load(std::sync::atomic::Ordering::Relaxed) != 0 {
                    return false;
                }
                pending.extend_from_slice(tok.decode_token(id));
                let (text, keep) = match std::str::from_utf8(&pending) {
                    Ok(s) => (s.to_string(), 0usize),
                    Err(e) => {
                        let ok = e.valid_up_to();
                        (String::from_utf8_lossy(&pending[..ok]).into_owned(), pending.len() - ok)
                    }
                };
                pending.drain(..pending.len() - keep);
                if !text.is_empty() {
                    let ev = Event::default().data(
                        serde_json::to_string(&ChatDelta { delta: text, chat_id }).unwrap(),
                    );
                    if tx.blocking_send(Ok(ev)).is_err() {
                        return false; // client went away
                    }
                }
                true
            });
            match res {
                Ok((out, tokps)) => {
                    tracing::info!("qwen36 turn done: {} tokens @ {:.3} tok/s", out.len(), tokps);
                }
                Err(e) => {
                    tracing::error!("qwen36 turn failed: {e}");
                    let _ = tx.blocking_send(Ok(Event::default()
                        .data(format!("{{\"error\":\"{e}\",\"chat_id\":{chat_id}}}"))));
                }
            }
            let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
            sessions.remove(chat_id);
        });
        return sse_response(
            Sse::new(ReceiverStream::new(rx))
                .keep_alive(KeepAlive::new().interval(Duration::from_secs(15)).text("keepalive")),
        );
    }

    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(64);

    // Clone base session — hold Mutex only during sp_session_clone (sub-ms).
    let cancel_child = Arc::new(AtomicI32::new(0));
    let child_result = {
        let guard = state.session.as_ref().expect("L1 session unavailable (qwen36 lane)").lock().unwrap();
        guard.clone_session(cancel_child.clone())
    };

    let mut child = match child_result {
        Ok(s) => s,
        Err(e) => {
            let _ = tx
                .send(Ok(Event::default().data(format!("{{\"error\":\"{e}\"}}")))).await;
            return sse_response(
                Sse::new(ReceiverStream::new(rx))
                    .keep_alive(KeepAlive::new().interval(Duration::from_secs(15)).text("keepalive")),
            );
        }
    };

    let chat_id = state.sessions.register(cancel_child.clone());
    let sessions = state.sessions.clone();
    let vocab_size = state.vocab_size;
    let app = state.clone();
    let max_tokens = req.max_tokens;
    // B1 / S1: byte-exact "auditable mode" for this turn (resident-cache chat
    // path only). CONTRACT-CHAT-FULLSTACK S1 makes byte-exact the DEFAULT chat
    // decode path: the exact-integer islands + dual-prime CRT-NTT attention are
    // run-to-run bit-identical AND build-independent (integer arithmetic), which
    // removes the FP-codegen reorder fragility that flipped a thin rank-2 margin
    // coherent↔garbage across rebuilds from the same HEAD. `raw_logits` (the
    // null-floor / determinism reference opt-out) forces it OFF to recover the
    // Stage-A float path bit-for-bit; an explicit `byteexact:false` also opts out.
    let byteexact = if req.raw_logits {
        false
    } else {
        // Default ON (field absent ⇒ None ⇒ true); explicit `byteexact:false` opts out.
        req.byteexact.unwrap_or(true)
    };
    // B2 (§6d-b): per-turn XBAR episode replay (None = no recall = null floor).
    let replay_dir = req.replay.clone();
    let replay_npos = req.replay_npos;
    // B3: autonomous recall toggle. An explicit request `auto_recall` always wins;
    // otherwise it defaults to SP_AUTO_RECALL_DEFAULT (so the served console/UI gets
    // recall without sending the flag per-request). Env unset => false = null floor.
    let auto_recall = req.auto_recall
        .unwrap_or(std::env::var("SP_AUTO_RECALL_DEFAULT").ok().as_deref() == Some("1"));
    // B5 (§6e): the single latent entry point. single_entry routes prompt ingest
    // through gemma4_kv_inject_tokens (the residual seam) instead of prefill;
    // inject_frames feeds raw residual frames (audio/memory) through the same seam.
    let single_entry = req.single_entry.unwrap_or(false);
    let eot_bias = req.eot_bias; // N5: per-request override of the SP_EOT_BIAS default
    let req_self_repeat_ngram = req.self_repeat_ngram;   // SELF-REPEAT BAN (narrow scope)
    let tool_names = req.tool_names.clone();
    let req_self_repeat_text = req.self_repeat_text.clone();
    let inject_frames = req.inject_frames.clone();
    let inject_ph = req.inject_ph;
    let inject_open = req.inject_open;
    let inject_close = req.inject_close;
    // A2: build the per-request sampler from the (flattened) ChatRequest knobs.
    let sampling = req.sampling.clone();
    // A2-polish: token ids the sampler must never emit — the full control /
    // placeholder set (`<image|>`/`<audio|>`/`<|turn>`/`<turn|>`/… + structural
    // specials), masked to -inf on BOTH the sampled and the greedy path so the
    // default served chat is clean everywhere. Computed id-agnostically from the
    // tokenizer's id_to_bytes. When `raw_logits` is set, the suppress set is
    // EMPTY ⇒ the raw, un-suppressed logits drive selection (the null-floor /
    // determinism reference — reproduces the prior greedy output bit-for-bit).
    let suppress_ids = if req.raw_logits {
        Vec::new()
    } else {
        state.tokenizer.suppress_token_ids()
    };
    // Issue #115: when the client supplies no stop strings, fall back to the
    // arch's chat-format terminator (gemma's `<end_of_turn>`) so the console
    // stream ends at the turn boundary instead of running to max_tokens.
    let stop_strings = if req.stop.is_empty() {
        state.tokenizer.default_stops()
    } else {
        req.stop
    };

    // Signal the mining loop to back off for the duration of this request.
    state.inference_active.store(true, Ordering::Relaxed);

    // Guard that clears inference_active when the spawn_blocking closure exits
    // (including early returns and panics).
    struct InferenceGuard(Arc<AtomicBool>);
    impl Drop for InferenceGuard {
        fn drop(&mut self) { self.0.store(false, Ordering::Relaxed); }
    }
    let _guard = InferenceGuard(state.inference_active.clone());

    let chat_id_obs = chat_id;
    let chat_worker = task::spawn_blocking(move || {
        let _g = _guard; // keep guard alive for the duration of the blocking closure
        let mut logits = vec![0.0f32; vocab_size];
        let mut dec_buf = TokenDecodeBuffer::new(stop_strings);
        // A2: the L2 sampler for this turn (temp=0 ⇒ strict argmax null floor).
        let mut sampler = crate::sampler::Sampler::with_suppress(sampling, suppress_ids);

        // ADR-013: a minimum thought before the channel may close. Only meaningful when
        // thinking armed the OPEN channel and un-banned its closer; otherwise the id is
        // None and this is a no-op. SP_THINK_MIN=0 (default) is the byte-identical floor.
        if crate::tokenizer::thinking_enabled() {
            let n = std::env::var("SP_THINK_MIN").ok()
                .and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(0);
            if n > 0 {
                sampler.set_min_thought(tokenizer.control_id_pub(b"<channel|>"), n);
            }
            // ADR-013: and a CEILING, which the floor never had. Two budgets, whichever
            // fires first: SP_THINK_MAX in generated tokens (deterministic, what a gate
            // can assert) and SP_THINK_MAX_MS in wall clock (what bounds the human's
            // wait when expert streaming makes decode slow). Both 0 = off = the
            // byte-identical null floor. Measured motivation: 258 s / 256 s / 308 s
            // turns on this model, the whole 768-token budget spent thinking.
            let tmax = std::env::var("SP_THINK_MAX").ok()
                .and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(0);
            let tms = std::env::var("SP_THINK_MAX_MS").ok()
                .and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(0);
            if tmax > 0 || tms > 0 {
                sampler.set_max_thought(tokenizer.control_id_pub(b"<channel|>"), tmax, tms);
            }
        }

        // ── CONSTRAINED TOOL DECODING (DOMINO-aligned) ────────────────────────────
        // The harness owns the grammar and sends the names it will accept; the engine makes
        // every OTHER name unreachable. Not "wrong and healed later" — unreachable.
        //
        // The trie is built from `tokenizer.encode(name)` — THE MODEL'S OWN TOKENISATION.
        // That is the whole point, and it is DOMINO's central finding: a mask built over
        // CHARACTERS misaligns with the subword vocabulary and measurably DEGRADES the model.
        // Every path through this trie is a token sequence the model would naturally have
        // produced, so it is never asked to spell a word one letter at a time.
        //
        // Empty tool_names ⇒ mask never constructed ⇒ nothing changes for anyone.
        let mut tool_mask = if tool_names.is_empty() {
            None
        } else {
            let tk = tokenizer.clone();
            Some(crate::tool_mask::ToolMask::new(&tool_names, move |s| {
                tk.encode(s).unwrap_or_default()
            }))
        };
        // Anti-echo (#47, G-ECHO-FIX): on the PLAIN chat path, greedy on a
        // contentless turn recites the system prompt verbatim. SP_NO_REPEAT_NGRAM=n
        // (n>=2) seeds a no-repeat-ngram guard with the prompt tokens so any verbatim
        // prompt recital is derailed. Default unset ⇒ 0 ⇒ byte-identical null floor.
        // Enabled in run_console_chat.bat (recall OFF); left OFF in the gate launcher
        // (run_console_faithful.bat, recall ON) so faithful in-context recitation and
        // the G-ONECONFIG obey gate are untouched.
        {
            let nrg = std::env::var("SP_NO_REPEAT_NGRAM").ok()
                .and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(0);
            if nrg >= 2 {
                sampler.set_no_repeat_ngram(nrg);
                sampler.set_ngram_prefix(tokens.clone());
            }
            // ── SELF-REPEAT BAN (2026-07-12) — the NARROW version of the above ──────────
            // The operator caught her emitting three BYTE-IDENTICAL replies to three
            // different messages (and again, four times running). Not a stale prompt: the
            // daemon log shows the prompt growing and the new text prefilled. She read his
            // words and chose to say her last reply again. A degeneration attractor on a
            // low-content turn ("you can", "cool huh?").
            //
            // `no_repeat_ngram=3` used to make that impossible — because it seeded the ban
            // from `tokens` = THE WHOLE PROMPT. That is also precisely why it had to die:
            // banning every trigram in context bans QUOTING, so "4471" came back "4417"
            // and every number in memory/tools/persona was garbled (G-VERBATIM).
            //
            // The fix is the same mechanism with the right SCOPE. Seed the ban from ONLY
            // her PREVIOUS REPLY, not from the context. She then cannot parrot herself,
            // and can still quote him, a memory, a tool result, or a number — none of which
            // are in the ban set. Prevention in the sampler, so it works on the STREAMING
            // console path where a post-hoc re-roll cannot retract what is already on screen.
            let srn = req_self_repeat_ngram.unwrap_or(0);
            if srn >= 2 {
                if let Some(prev) = req_self_repeat_text.as_deref() {
                    let prev = prev.trim();
                    if !prev.is_empty() {
                        if let Ok(ptoks) = tokenizer.encode(prev) {
                            if ptoks.len() >= srn {
                                sampler.set_no_repeat_ngram(srn);
                                sampler.set_ngram_prefix(ptoks);
                                tracing::info!(
                                    "SELF-REPEAT BAN: {}-grams from her previous reply only \
                                     ({} tok) — quoting the user/memory/tools is UNAFFECTED",
                                    srn, prev.split_whitespace().count());
                            }
                        }
                    }
                }
            }
        }

        // Issue #115 (the dense reference model chat path): when the CUDA persistent-KV decode backend
        // is registered (SP_DAEMON_BACKEND=cuda + SP_DAEMON_KVDECODE=1 +
        // SP_CUDA_DECODE_INT8=1), the dense reference model's full-vocab head is only materializable
        // through the resident cache — sp_prefill_chunk's §6 forward bridge trips
        // "g4 probe: FULL head needs the f32 embd". Drive the single resident
        // cache directly here (serialized by its Mutex; reset per request via
        // rewind). The session-clone + prefill_chunk/decode_step path stays the
        // fallback for models whose head fits the prefill bridge (Qwen3 etc.).
        #[cfg(feature = "wire_cuda_backend")]
        let kvdecode = app.cuda_kvdecode_handle.is_some();
        #[cfg(not(feature = "wire_cuda_backend"))]
        let kvdecode = false;

        if kvdecode {
            // Drop the unused clone — the resident cache, not the clone, holds KV.
            drop(child);
            run_kvdecode_chat(
                &app, chat_id, &tokens, max_tokens, vocab_size,
                &mut logits, &mut dec_buf, &tx, &cancel_child, &sessions,
                &mut sampler, byteexact, replay_dir, replay_npos,
                single_entry, inject_frames, inject_ph, inject_open, inject_close, auto_recall,
                raw_user, orig_msgs, eot_bias,
            );
            return;
        }

        if !tokens.is_empty() {
            if let Err(e) = child.prefill_chunk(&tokens, &mut logits) {
                let _ = tx.blocking_send(Ok(Event::default().data(
                    format!("{{\"error\":\"{e}\"}}"),
                )));
                let _ = app.events_tx.send(DaemonEvent::Chat { chat_id, status: "cancelled" });
                sessions.remove(chat_id);
                return;
            }
        }

        // ── APPLY THE TOOL MASK ───────────────────────────────────────────────────
        // `allowed()` returns None while she is TALKING, which is the overwhelming majority
        // of tokens — the mask has no opinion about her prose. It returns Some only once the
        // ```tool_code fence is open, and then only until '(' is emitted: the ARGUMENTS are
        // hers. Constraining a genuine choice is how you make a model stupid.
        let mut next_token = if let Some(tm) = tool_mask.as_mut() {
            match tm.allowed() {
                Some(ids) => {
                    for (i, l) in logits.iter_mut().enumerate() {
                        if !ids.contains(&(i as i32)) {
                            *l = f32::NEG_INFINITY;
                        }
                    }
                    sampler.sample(&mut logits)
                }
                None => sampler.sample(&mut logits),
            }
        } else {
            sampler.sample(&mut logits)
        };
        sampler.observe(next_token);
        if let Some(tm) = tool_mask.as_mut() {
            tm.push(next_token, tokenizer.decode_token(next_token));
        }

        // A2-polish: this arch's turn boundary is the `<|turn>`/`<turn|>` token
        // (no `<end_of_turn>` token exists), so treat those ids as EOS-equivalent.
        let turn_stop_ids = tokenizer.turn_stop_ids();
        'decode: for _ in 0..max_tokens {
            // EOS / turn-stop check before emitting (stop cleanly at the turn
            // boundary — the marker never reaches the stream).
            if (!tokenizer.eos_ids.is_empty() && tokenizer.eos_ids.contains(&next_token))
                || turn_stop_ids.contains(&next_token)
            {
                break 'decode;
            }

            let token_bytes = tokenizer.decode_token(next_token);
            let stop_hit = match dec_buf.push(token_bytes) {
                PushResult::Emit(bytes) => {
                    if !bytes.is_empty() {
                        let text = String::from_utf8_lossy(&bytes).into_owned();
                        let payload = serde_json::to_string(&ChatDelta { delta: text, chat_id })
                            .unwrap_or_default();
                        if tx.blocking_send(Ok(Event::default().data(payload))).is_err() {
                            // Client disconnected.
                            cancel_child.store(1, Ordering::Relaxed);
                            let _ = app.events_tx.send(DaemonEvent::Chat { chat_id, status: "cancelled" });
                            sessions.remove(chat_id);
                            return;
                        }
                    }
                    false
                }
                PushResult::Stopped(bytes) => {
                    if !bytes.is_empty() {
                        let text = String::from_utf8_lossy(&bytes).into_owned();
                        let payload = serde_json::to_string(&ChatDelta { delta: text, chat_id })
                            .unwrap_or_default();
                        let _ = tx.blocking_send(Ok(Event::default().data(payload)));
                    }
                    true
                }
            };

            app.tokens_decoded.fetch_add(1, Ordering::Relaxed);

            if stop_hit {
                break 'decode;
            }

            // ── THE FREE TOKEN, AND WHY IT IS NOT HERE YET ────────────────────────
            // The obvious optimisation: when exactly one continuation is legal (after `rec`
            // there is only `all`; after `recall` there is only `(`), the model has no choice
            // to make, so skip the forward pass and just emit it. DOMINO gets up to ~2x from
            // precisely this, and I wrote it that way first.
            //
            // IT WOULD HAVE CORRUPTED THE KV CACHE. decode_step() is not merely "compute the
            // logits" — it is what ADVANCES THE MODEL'S STATE with the token just emitted.
            // Skip it and the cache never sees that token: the sequence desynchronises, and
            // every token after it in the turn is conditioned on a history that did not
            // happen. Silently. It would have looked like a speedup and behaved like brain
            // damage, and this system has already spent a day on one cache-divergence bug
            // (tools=[] re-prefilling the whole conversation, 111s a turn).
            //
            // The speedup is real, but it is a BATCHING problem, not a skipping one: collect
            // the forced run and feed the whole thing through prefill_chunk() in ONE call —
            // which is exactly what speculative decoding does, and what the engine already
            // has the machinery for. That is the next commit, not a footnote in this one.
            // Correctness first; the mask below is the part that makes hallucinated tools
            // unreachable, and it is worth shipping on its own.
            match child.decode_step(next_token, &mut logits) {
                Ok(()) => {
                    next_token = if let Some(ids) =
                        tool_mask.as_mut().and_then(|tm| tm.allowed())
                    {
                        for (i, l) in logits.iter_mut().enumerate() {
                            if !ids.contains(&(i as i32)) {
                                *l = f32::NEG_INFINITY;
                            }
                        }
                        sampler.sample(&mut logits)
                    } else {
                        sampler.sample(&mut logits)
                    };
                    sampler.observe(next_token);
                    if let Some(tm) = tool_mask.as_mut() {
                        tm.push(next_token, tokenizer.decode_token(next_token));
                    }
                }
                Err(_) => break 'decode,
            }
        }

        // Flush any bytes held back for UTF-8 / stop-string boundary detection.
        let flushed = dec_buf.flush();
        if !flushed.is_empty() {
            let text = String::from_utf8_lossy(&flushed).into_owned();
            let payload = serde_json::to_string(&ChatDelta { delta: text, chat_id })
                .unwrap_or_default();
            let _ = tx.blocking_send(Ok(Event::default().data(payload)));
        }

        let is_cancelled = cancel_child.load(Ordering::Relaxed) != 0;
        if is_cancelled {
            let _ = tx.blocking_send(Ok(Event::default().event("cancelled").data("{}")));
            let _ = app.events_tx.send(DaemonEvent::Chat { chat_id, status: "cancelled" });
        } else {
            let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
            let _ = app.events_tx.send(DaemonEvent::Chat { chat_id, status: "done" });
        }
        sessions.remove(chat_id);
    });
    // F-4 aggravator (2026-08-29 audit): this JoinHandle was DROPPED — a panic in
    // the worker (which poisons the device lock it holds) was never observed: the SSE
    // just truncated with no [DONE], nothing in tracing, and the chat_id leaked in
    // the sessions map. Observe it from a cheap task; the panic gets a log line with
    // the turn's id on it, which is the whole difference between a mystery outage
    // and a grep.
    tokio::spawn(async move {
        if let Err(e) = chat_worker.await {
            if e.is_panic() {
                tracing::error!("CHAT-WORKER PANIC (chat_id {}): {:?} — the device lock                                  it held is poison-tolerant, but this turn died silently                                  to its client", chat_id_obs, e);
            }
        }
    });

    sse_response(
        Sse::new(ReceiverStream::new(rx))
            .keep_alive(KeepAlive::new().interval(Duration::from_secs(15)).text("keepalive")),
    )
}

/// Issue #115 — the dense reference model chat over the resident CUDA persistent-KV decode cache.
///
/// The dense reference model's tied full-vocab head cannot be materialized by the §6 prefill
/// bridge, so prompt ingest + token decode run on the single session-resident
/// `gemma4_kv_*` cache (the G-WIRE-CUDA-DECODE-GEMMA4 path). The cache is global
/// + stateful (one `dpos`), so we serialize on its Mutex and `rewind` to 0 at
/// the start of each request. The prefill head's argmax for the FIRST generated
/// token is obtained by prefilling `tokens[..n-1]` then `decode_step(tokens[n-1])`
/// (which returns that token's next-position logits) — no separate seq-peek
/// needed. Emit / EOS / stop-string handling mirrors the fallback decode loop.
// ── Persistent O(1) conversation KV (SP_PERSIST_KV) ─────────────────────────────
// The tokens currently committed in the resident KV (prompt + generated). When the next
// turn's prompt is a STRICT PREFIX of this (the conversation just grew by appending) and the
// cache position matches, we PREFILL ONLY THE NEW SUFFIX instead of rewinding to 0 and
// re-prefilling the whole conversation -- the O(n)-per-turn cost that drops tok/s 30->1 in long
// chats. Byte-exact: the reused prefix's KV is the same deterministic computation as
// re-prefilling it. Default-off, and only on the PLAIN decode path (no recall/agency side-calls,
// which reset the cache), so the committed sequence always mirrors the cache exactly.
#[cfg(feature = "wire_cuda_backend")]
// ── WHO GETS THE DEVICE NEXT (2026-08-03) ────────────────────────────────────────────
// Serialising every forward on the resident cache's Mutex ended the corruption, and then
// immediately produced the other failure: a plain mutex is FIFO-ish and fair, and fair is
// wrong here. Measured the hour it shipped — six background one-shots fired while a chat
// turn was in flight, and the chat turn TIMED OUT waiting behind them. His words: "my turns
// preempt background work".
//
// So: a foreground request registers its intent BEFORE it queues, and background work
// refuses to START while any foreground request is waiting. Two lines of state, no queue
// to get wrong.
//
// IT IS A YIELD, NOT A PREEMPTION, and the difference matters enough to name. A forward
// already running on the GPU is not interruptible — there is no safe cancellation point
// inside a prefill, and inventing one would mean tearing down a cache mid-kernel, which is
// the exact class of bug this whole week has been about. So the guarantee is bounded and
// honest: a user turn never waits behind background work that had not already begun. If a
// long one-shot is mid-flight, the user turn waits for it, once.
//
// Background work cannot starve forever either: FG_WAITING is decremented the moment the
// foreground request takes the lock, so background resumes as soon as the queue drains.
static FG_WAITING: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Registers a foreground request for the whole time it is WAITING for the device, and
/// deregisters on drop — including on every early return and panic, which is why this is a
/// guard and not a pair of calls someone has to remember to balance.
struct FgIntent;
impl FgIntent {
    fn new() -> Self {
        FG_WAITING.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        FgIntent
    }
}
impl Drop for FgIntent {
    fn drop(&mut self) {
        FG_WAITING.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
    }
}

/// Background callers hold here while any foreground request is queued. Bounded, so a
/// wedged counter can never deadlock the background lanes — it degrades to the old
/// first-come behaviour instead of hanging, because a stuck judge is a nuisance and a stuck
/// daemon is an outage.
fn yield_to_foreground(what: &str) {
    use std::sync::atomic::Ordering;
    // THE BOUND IS A DEADLOCK VALVE, NOT A SCHEDULING PARAMETER — and the first value I
    // picked confused the two. 300 s looked generous until it was measured against a real
    // turn: his took 522 s (a cold per-token prefill with both batch arms disarmed), so all
    // three background one-shots yielded their full 300 s, gave up, and went anyway. The
    // priority worked and the bound quietly undid it on every long turn.
    //
    // Its only job is to stop a LEAKED counter becoming an outage, and a leak is already
    // most of the way prevented: FgIntent is RAII and decrements on every early return and
    // on panic. So the bound belongs far above the worst realistic turn rather than near
    // the typical one. 30 minutes: long enough that no honest turn trips it, short enough
    // that a genuine leak is a bad half hour and not a dead daemon.
    let cap_s: u64 = std::env::var("SP_PRIORITY_YIELD_CAP_S").ok()
        .and_then(|v| v.parse().ok()).unwrap_or(1800);
    let t0 = std::time::Instant::now();
    while FG_WAITING.load(Ordering::SeqCst) > 0 {
        if t0.elapsed() > std::time::Duration::from_secs(cap_s) {
            tracing::warn!("PRIORITY: {what} waited {cap_s}s behind foreground work —                             proceeding anyway (a stuck counter must not become an outage)");
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    if t0.elapsed() > std::time::Duration::from_millis(50) {
        tracing::info!("PRIORITY: {what} yielded {:?} to foreground work", t0.elapsed());
    }
}

static KV_COMMITTED: std::sync::Mutex<Vec<i32>> = std::sync::Mutex::new(Vec::new());

/// ── A FAILED TURN MAY NOT LEAVE YESTERDAY'S TRUTH STANDING (2026-08-29 audit, F1) ──
/// KV_COMMITTED was cleared only at the successful commit; the eleven abnormal exits
/// left the OLD list describing a cache that this turn had already partially
/// overwritten. The `pos == cl` guard is a LENGTH check, and three real shapes make
/// the lengths agree while the content lies (a prefill dying after exactly drop_n
/// tokens; a chunk-sync fault landing on a 128-multiple after a snapshot restore;
/// the dead-peer abort at an arbitrary k) — the next turn then strict-appends onto
/// a different conversation. This guard arms when the turn first touches the cache
/// and clears the committed list on drop; the successful commit forgets it. Cost: a
/// turn that already failed pays one cold prefill — the honest price.
struct CommitVoid;
impl Drop for CommitVoid {
    fn drop(&mut self) {
        match KV_COMMITTED.lock() {
            Ok(mut c) => c.clear(),
            Err(p) => p.into_inner().clear(),
        }
    }
}

/// Model identity of whatever KV_COMMITTED currently describes (see MODEL_TAG).
#[cfg(feature = "wire_cuda_backend")]
static KV_COMMITTED_TAG: std::sync::Mutex<String> = std::sync::Mutex::new(String::new());

// ── THE PREFIX SNAPSHOT (2026-07-13) ────────────────────────────────────────────────
// The KV of the shared constant preamble, in HOST memory, plus the exact tokens it was
// minted from. One slot: the daily-driver preamble. It is LEARNED, not configured — the
// first time a prompt diverges past the rewind bound we take the shared prefix we just
// observed, snapshot it, and never pay for it again. If the preamble ever changes, the
// token compare misses and it is simply re-learned on the next divergence.
//
// The tokens are stored ALONGSIDE the bytes and compared before every restore. A KV blob
// is only correct for the exact token sequence that produced it, and a snapshot keyed by
// length alone would eventually restore the wrong cache into a live conversation and be
// almost impossible to see — the model would just quietly start answering a prompt it was
// never given. THE KEY IS THE CONTENT, NOT THE SIZE.
//
// ── AND THE CONTENT KEY DOES NOT DISCRIMINATE MODELS (2026-07-29) ───────────────
// "THE KEY IS THE CONTENT" is right WITHIN a model and insufficient ACROSS models.
// the dense reference model and gemma4-26b-a4b ship the SAME 262144-token Gemma-4 tokenizer (the
// transcode gate asserts the two .sp-tokenizer files byte-identical), so the same
// preamble text produces the SAME token sequence on both — the compare MATCHES on
// a foreign cache. The only thing standing behind it is kv_restore_prefix's size
// check, which is `need > cap` and therefore passes whenever the stale blob is the
// LARGER one: a dense reference model blob (48 layers) restored into a 26B session (30) satisfies it
// and loads one model's KV into another model's cache, silently.
//
// Today that is unreachable — these are in-process statics and a daemon serves one
// model for its lifetime — which is exactly the kind of "can't happen" this repo
// keeps being wrong about. The guard costs a string compare. Tag the caches with
// the model they were minted from and refuse a foreign tag.
#[cfg(feature = "wire_cuda_backend")]
struct PrefixSnap {
    toks: Vec<i32>,
    blob: Vec<u8>,
    tag: String,
}

#[cfg(feature = "wire_cuda_backend")]
static PREFIX_SNAP: std::sync::Mutex<Option<PrefixSnap>> = std::sync::Mutex::new(None);

// ── THE ROLLING SNAPSHOT (2026-08-03) ───────────────────────────────────────────────
// PREFIX_SNAP above is minted ONCE, at the first divergence, and never moves again. That
// was right for what it was built for — the constant preamble — and it quietly stopped
// being enough, because the thing it does not cover GROWS. Measured on his conversation,
// one turn at a time:
//
//     PREFIX-SNAPSHOT: restored 6529 tokens in 271 ms; prefill suffix 5135
//     TURN-PHASE: prefill 5134 tok in 426309 ms (83 ms/tok)
//
// The 6529 are free — a memcpy, a quarter of a second. The suffix is the ENTIRE
// conversation since the preamble, it is re-prefilled from scratch on every single turn,
// and it gets one turn longer each time. Seven minutes to answer, and by construction
// tomorrow is worse than today. That is the "minutes between turns" he is living with.
//
// It is also pure waste: at the moment the suffix finishes prefilling, the cache holds
// EXACTLY the state the next turn wants to start from. Nobody was keeping it. So keep it
// — one D2H memcpy of a cache we already built, ~110 ms, and the next turn's suffix is
// his new sentence instead of the whole afternoon.
//
// TWO SLOTS, NOT ONE, and that is the whole reason this is safe. The rolling slot is
// tried first and the base preamble is tried second, so a prompt that no longer starts
// with the rolling prefix — the transcript window slid, the day rolled over, a tool round
// rewrote history — does not fall to a cold prefill. It falls back to exactly the
// behaviour this repo has been running all week. NEVER WORSE THAN THE THING IT REPLACES
// is a property worth 650 MB of host RAM.
#[cfg(feature = "wire_cuda_backend")]
static ROLL_SNAP: std::sync::Mutex<Option<PrefixSnap>> = std::sync::Mutex::new(None);

/// How far back from the prompt's END the last turn diverged from the cache we held.
///
/// ── WHY THE MARGIN HAS TO LEARN (2026-08-24) ────────────────────────────────────────
/// `roll_snap_margin()` was 16, chosen for the TOKENIZER seam ("measured divergence was
/// 2-3 tokens; 16 is clear of it"), then raised to 512 when the real divergence turned
/// out to be the per-turn notes. Both are FIXED, and the thing they are backing off from
/// is not. Measured over one clean run this morning, with a message list that STRICTLY
/// EXTENDS (verified against the gateway's own dumps):
///
///     drop  15  ->  prefill  190 tok in  6.4 s
///     drop  49  ->  prefill  507 tok in 12.2 s
///     drop 146  ->  prefill  909 tok in 22.7 s
///
/// The drop GROWS across a conversation, so any constant is eventually too small, the
/// rolling snapshot ends up longer than the next turn's `lcp`, `s.toks.len() <= want`
/// fails, and the restore falls back to the base slot — which is the growing suffix and
/// the climbing turn latency, arriving by a slightly different road than last time.
///
/// So the margin is LEARNED from the number the engine already computes every turn. It
/// is a hint and nothing depends on it being right: too small and the roll simply is not
/// usable next turn (today's behaviour), too large and `roll_split` goes to 0 and no roll
/// is kept (also today's behaviour). It cannot make a turn wrong, only cheaper.
static LAST_TAIL_GAP: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// ── THE RETOKENIZATION SEAM (2026-08-24 — SWEEP-2026-08-24 F10's named remainder) ──
/// KV_COMMITTED is `tokens ++ committed_gen` — the SAMPLED ids — while the next turn's
/// prompt re-tokenizes the transcript TEXT, and encode(decode(ids)) != ids at BPE merge
/// boundaries. So the lcp pins at the first divergence inside the previous reply and
/// `drop` GROWS by roughly a turn per turn (measured 15 -> 49 -> 146), and only
/// drop == 0 reaches the cheap append: the SWA undo-journal clears at every commit, so
/// rewind refuses across a turn boundary whatever the bound. The harness side is done
/// (the canonical list is the exact streamed bytes since F10); what still differs is
/// SPELLING, not text.
///
/// This answers exactly that case: when the incoming tokens from `lcp` onward spell,
/// byte for byte, the same text as the committed tail, return `Some(j)` — the count of
/// incoming tokens whose decoded bytes exactly cover the committed tail — so the
/// caller can normalize the PROMPT to the committed spelling (`committed ++
/// incoming[lcp+j..]`), making drop 0 by construction and letting the strict append
/// fire. The K/V in the cache are the truth of that same text in the committed
/// spelling, so nothing is misdescribed.
///
/// NEVER the other direction (re-labelling KV_COMMITTED with the canonical
/// retokenization): the K/V were COMPUTED from the sampled ids; a ledger claiming ids
/// the cache never saw turns a measurable perf miss into invisible semantic
/// corruption at exactly the positions her previous reply lives at. Do not retry it.
///
/// On ANY disagreement — a strip difference, a template respelling, a token that
/// straddles the committed boundary with different bytes — this returns None and the
/// caller falls through untouched to today's behaviour (rewind -> snapshot -> full
/// prefill): slower for one turn, never wrong.
/// The generation DRESSING the daemon itself appended (2026-08-25, the first live
/// receipt's finding). With thinking armed the prompt ends `<|turn>model\n` +
/// `<|channel>thought\n` — control ids `apply_template_ids` adds ONLY at the live
/// generation position. A HISTORICAL assistant message renders without them, and the
/// canon text cannot spell them (encoding the literal `<|channel>` shatters into text
/// pieces — the gemma3-template-on-gemma4 lesson). So every thinking turn leaves a
/// small committed run the next prompt can never byte-match: measured drop 143 =
/// dressing (3 ids) + a 140-token reply, twice exactly. The seam may skip THIS
/// sequence and only this sequence — it is template-owned, closed, and the daemon is
/// the one who put it there; skipping anything else would be guessing.
/// Returns how many leading ids of `committed_tail` decode to exactly `dressing`
/// (0 when they do not): byte-based, so it needs no tokenizer internals and a toy
/// vocabulary can test it. Bounded to eight ids — the dressing is three.
fn reseam_skip_dressing<F: Fn(i32) -> Vec<u8>>(decode: F, committed_tail: &[i32],
                                               dressing: &[u8]) -> usize {
    if dressing.is_empty() {
        return 0;
    }
    // (F6, 2026-08-29: the loop below refuses zero-width ids for the same reason
    // reseam_join does — an id with no bytes could consume one of the eight slots
    // and excuse real committed content from having to byte-match.)
    let mut acc: Vec<u8> = Vec::new();
    for (i, &t) in committed_tail.iter().take(8).enumerate() {
        let b = decode(t);
        if b.is_empty() { return 0; }   // F6: no byte evidence, no skip
        acc.extend_from_slice(&b);
        if acc.len() > dressing.len() {
            return 0;
        }
        if acc[..] == dressing[..] {
            // dressing only ever PRECEDES a reply — a tail that is nothing but
            // dressing has no join to offer
            return if i + 1 < committed_tail.len() { i + 1 } else { 0 };
        }
        if dressing[..acc.len()] != acc[..] {
            return 0;
        }
    }
    0
}

/// ── THE REFUSAL SAYS WHERE IT DIED (2026-08-30) ─────────────────────────────────────
/// The Err carries which leg refused, at what byte offset, with the bytes around the
/// death point. The 2026-08-25 trial's refusal logged forty bytes of each side's HEAD —
/// and the heads MATCHED, so the wrong attribution ("L5 recall augmentation") lived in
/// the ledger for an hour while the actual disagreement sat hundreds of bytes deeper.
/// Tonight's drop-350 refusal has byte-identical heads too. The next diagnosis reads
/// the leg name off the log instead of guessing from a prefix that agrees.
/// ── COMMITTED-SIDE TEMPLATE GAPS (2026-08-30, the instrumented refusal's finding) ───
/// The first leg-named refusal receipts settled a question the head-bytes never could:
///
///     text disagrees at byte 761/1288: committed "urn|>\n<|turn>model\n<|channel>thought\n<ch"
///                                      vs incoming[170] (id 12027) "channel"
///
/// The committed stream carries TURN-STRUCTURE ids the daemon never streams — her
/// sampled turn-end plus the next round's generation dressing (`<|turn>model\n` +
/// `<|channel>thought\n`) between the rounds of a thinking/tool turn. The canon is the
/// exact streamed bytes (F10), so no prompt can EVER spell those spans, and the seam
/// refused on every thinking turn: drop ≈ her whole reply, a 20–90 s re-prefill,
/// forever. The spans are template-owned, closed, and the daemon is the one who put
/// them there — the same justification as reseam_skip_dressing, generalized from
/// "leading only" to anywhere in the walk. `gaps` lists those byte sequences; when the
/// committed bytes at the cursor spell one, the cursor advances on the COMMITTED side
/// only and the walk continues. Anything not on the list still refuses by name — a new
/// dressing shows up in the refusal log, never in a silent mis-join. The committed ids
/// are kept whole in the rebuilt prompt (they are real cache content); only the
/// REQUIREMENT that the incoming spell them is waived.
fn reseam_join<F: Fn(i32) -> Vec<u8>>(decode: F, committed_tail: &[i32],
                                      incoming: &[i32],
                                      gaps: &[&[u8]]) -> Result<usize, String> {
    fn ctx(b: &[u8], at: usize) -> String {
        let lo = at.saturating_sub(20);
        let hi = (at + 20).min(b.len());
        format!("{:?}", String::from_utf8_lossy(&b[lo..hi]))
    }
    let mut tail: Vec<u8> = Vec::new();
    for &t in committed_tail {
        tail.extend_from_slice(&decode(t));
        if tail.len() > (1 << 20) {
            // a megabyte of divergence is not a seam artifact
            return Err("committed tail decodes past 1 MiB — not a seam artifact".into());
        }
    }
    if tail.is_empty() {
        return Err("committed tail decodes to nothing (control-only) — no evidence".into());
    }
    // committed-side gap skipper: advances past any run of template-owned sequences
    let skip_gaps = |acc: &mut usize| {
        loop {
            let mut moved = false;
            for g in gaps {
                if !g.is_empty() && tail[*acc..].starts_with(g) {
                    *acc += g.len();
                    moved = true;
                }
            }
            if !moved { break; }
        }
    };
    let mut acc = 0usize;
    skip_gaps(&mut acc);
    if acc == tail.len() {
        // the whole divergence was dressing: the prompt may simply continue
        return Ok(0);
    }
    for (i, &t) in incoming.iter().enumerate() {
        let b = decode(t);
        if b.is_empty() {
            // ── ZERO-WIDTH IS NOT EVIDENCE (2026-08-29 audit, F5). An id that
            // decodes to no bytes — a control id at the seam, or an out-of-vocab id
            // the tokenizer maps to &[] (a NEGATIVE id becomes a huge usize and
            // falls to the same branch) — passed both refusal tests vacuously: no
            // straddle, empty == empty. It was then COUNTED, so the caller built
            // committed ++ incoming[lcp+j..] with that id silently deleted from the
            // prompt. The function's own contract says any disagreement refuses;
            // a token carrying no byte evidence cannot agree with anything.
            return Err(format!(
                "incoming[{}] (id {}) decodes to zero bytes at byte {}/{} — no evidence; \
                 committed there: {}", i, t, acc, tail.len(), ctx(&tail, acc)));
        }
        if acc + b.len() > tail.len() {
            // the incoming token would straddle the committed boundary: the two
            // spellings do not share a token-aligned seam here. (A partial overlap
            // that MATCHES is still refused — prefilling from an unaligned byte
            // offset is not a thing a token cache can do.)
            return Err(format!(
                "incoming[{}] (id {}, {} B) straddles the committed end at byte {}/{}; \
                 committed there: {}", i, t, b.len(), acc, tail.len(), ctx(&tail, acc)));
        }
        if tail[acc..acc + b.len()] != b[..] {
            // different TEXT, not just different spelling
            return Err(format!(
                "text disagrees at byte {}/{}: committed {} vs incoming[{}] (id {}) {}",
                acc, tail.len(), ctx(&tail, acc), i, t,
                format!("{:?}", String::from_utf8_lossy(&b))));
        }
        acc += b.len();
        skip_gaps(&mut acc);
        if acc == tail.len() {
            return Ok(i + 1);
        }
    }
    Err(format!(
        "incoming ended at byte {}/{} — the prompt ends inside the committed tail; \
         committed remainder: {}", acc, tail.len(), ctx(&tail, acc)))
}

#[cfg(test)]
mod reseam_tests {
    use super::reseam_join;

    // a toy vocabulary: id N decodes to the byte string in V[N].
    // OUT-OF-VOCAB DECODES TO NOTHING, exactly like production (tokenizer.rs
    // returns &[] for an unknown id, and a negative id becomes a huge usize and
    // lands in the same branch). The old toy PANICKED on out-of-vocab, so the test
    // harness structurally could not exercise the very path production hits —
    // which is how the zero-width hole (F5) lived here ungated (2026-08-29 audit).
    fn dec(v: &'static [&'static [u8]]) -> impl Fn(i32) -> Vec<u8> {
        move |id| v.get(id as usize).map(|b| b.to_vec()).unwrap_or_default()
    }

    const V: &[&[u8]] = &[
        b"he", b"llo", b"hel", b"lo", b" wor", b"ld", b"x", b"", b"hell",
    ];

    #[test]
    fn same_text_different_split_joins() {
        // committed: "he"+"llo"  incoming: "hel"+"lo"+" wor"+"ld"
        assert_eq!(reseam_join(dec(V), &[0, 1], &[2, 3, 4, 5], &[]).ok(), Some(2));
    }

    #[test]
    fn mid_word_respelling_joins_when_bytes_align() {
        // committed "hel"+"lo" vs incoming "he"+"llo": same five bytes, different
        // token boundaries INSIDE a word — the live shape (a reply truncated
        // mid-word by the ceiling re-tokenizes differently).
        assert_eq!(reseam_join(dec(V), &[2, 3], &[0, 1, 4, 5], &[]).ok(), Some(2));
    }

    #[test]
    fn mid_word_straddle_refuses() {
        // committed "hell" vs incoming "he"+"llo": the second token crosses the
        // committed boundary — matching prefix or not, there is no aligned seam.
        assert_eq!(reseam_join(dec(V), &[8], &[0, 1], &[]).ok(), None);
    }

    #[test]
    fn zero_width_incoming_refuses_not_consumes() {
        // (2026-08-29 audit, F5.) id 7 decodes to ZERO bytes. It used to pass both
        // refusal tests vacuously and get COUNTED — the caller then dropped it from
        // the rebuilt prompt. No byte evidence => None, never a silent deletion.
        assert_eq!(reseam_join(dec(V), &[0, 1], &[2, 7, 3, 4, 5], &[]).ok(), None);
        // ...and an out-of-vocab id (production: tokenizer returns &[]) is the same.
        assert_eq!(reseam_join(dec(V), &[0, 1], &[2, 999, 3], &[]).ok(), None);
        // ...including a NEGATIVE id (becomes a huge usize; &[] in production).
        assert_eq!(reseam_join(dec(V), &[0, 1], &[2, -1, 3], &[]).ok(), None);
    }

    #[test]
    fn empty_inputs_refuse() {
        assert_eq!(reseam_join(dec(V), &[], &[0, 1], &[]).ok(), None);   // empty committed tail
        assert_eq!(reseam_join(dec(V), &[0, 1], &[], &[]).ok(), None);   // empty incoming
    }

    #[test]
    fn divergence_cap_refuses_a_megabyte() {
        // the 1 MiB cap is a refusal, not a truncation: a huge committed tail is
        // not a seam artifact and must fall through to the ordinary rewind.
        const BIG: &[&[u8]] = &[&[b'a'; 65536]];
        let tail: Vec<i32> = vec![0; 17];               // 17 * 64 KiB > 1 MiB
        assert_eq!(reseam_join(dec(BIG), &tail, &[0], &[]).ok(), None);
    }

    #[test]
    fn different_text_refuses() {
        // committed "he"+"llo" vs incoming "x..." — a real edit, not a respelling.
        // (2026-08-30) The refusal NAMES its leg and its byte offset — the 08-25
        // trial's "no seam" with matching heads put the wrong culprit in a ledger
        // row for an hour, and the whole point of the Err is that this cannot
        // happen again. A refusal that stops saying where it died is a regression.
        let why = reseam_join(dec(V), &[0, 1], &[6, 6, 6], &[]).unwrap_err();
        assert!(why.contains("text disagrees at byte 0/5"), "{why}");
    }

    #[test]
    fn straddling_token_refuses_even_on_matching_prefix() {
        // committed "he" vs incoming "hel"… — the l crosses the boundary.
        let why = reseam_join(dec(V), &[0], &[2, 3], &[]).unwrap_err();
        assert!(why.contains("straddles the committed end at byte 0/2"), "{why}");
    }

    #[test]
    fn control_only_tail_refuses() {
        assert_eq!(reseam_join(dec(V), &[7], &[0], &[]).ok(), None);
    }

    #[test]
    fn prompt_ending_inside_tail_refuses() {
        assert_eq!(reseam_join(dec(V), &[0, 1, 4], &[2, 3], &[]).ok(), None);
    }

    // ── committed-side template gaps (2026-08-30, the instrumented refusal's find) ──
    // committed carries structure ids the daemon never streams: [EOT][opener][thought]
    // between the rounds of a thinking turn. The walk skips exactly the listed spans on
    // the committed side; anything else still refuses by name.
    const G: &[&[u8]] = &[b"<turn|>\n", b"<|turn>model\n"];

    #[test]
    fn mid_tail_template_gap_joins() {
        // committed: "he"+"llo"+[EOT gap as one id]+" wor"+"ld"
        // incoming:  "hel"+"lo"+" wor"+"ld" — the gap is never streamed, never spelled
        const GV: &[&[u8]] = &[b"he", b"llo", b"<turn|>\n<|turn>model\n", b" wor", b"ld",
                               b"hel", b"lo"];
        let d = dec(GV);
        assert_eq!(reseam_join(d, &[0, 1, 2, 3, 4], &[5, 6, 3, 4], G).ok(), Some(4));
    }

    #[test]
    fn a_non_template_committed_extra_still_refuses() {
        // same shape, but the committed extra is REAL content ("xx"), not dressing —
        // skipping it would delete her words from the record. Refuses, and names it.
        const GV: &[&[u8]] = &[b"he", b"llo", b"xx", b" wor", b"ld", b"hel", b"lo"];
        let d = dec(GV);
        let why = reseam_join(d, &[0, 1, 2, 3, 4], &[5, 6, 3, 4], G).unwrap_err();
        assert!(why.contains("text disagrees"), "{why}");
    }

    #[test]
    fn trailing_gap_drains_after_the_last_match() {
        // committed ends with dressing after the matched text: the walk skips it and
        // completes — the incoming suffix continues from there.
        const GV: &[&[u8]] = &[b"he", b"llo", b"<turn|>\n", b"hel", b"lo"];
        let d = dec(GV);
        assert_eq!(reseam_join(d, &[0, 1, 2], &[3, 4], G).ok(), Some(2));
    }

    #[test]
    fn an_all_gap_tail_is_a_plain_continuation() {
        const GV: &[&[u8]] = &[b"<turn|>\n", b"<|turn>model\n", b"he"];
        let d = dec(GV);
        assert_eq!(reseam_join(d, &[0, 1], &[2], G).ok(), Some(0));
    }

    use super::reseam_skip_dressing;

    const D: &[&[u8]] = &[b"<|channel>", b"thought", b"\n", b"he", b"llo", b"<|chan"];

    #[test]
    fn dressing_is_skipped_when_it_leads_the_tail() {
        // the live shape: [ch_open]["thought"][nl] then the reply
        assert_eq!(reseam_skip_dressing(dec(D), &[0, 1, 2, 3, 4],
                                        b"<|channel>thought\n"), 3);
    }

    #[test]
    fn a_tail_that_is_only_dressing_offers_no_join() {
        assert_eq!(reseam_skip_dressing(dec(D), &[0, 1, 2], b"<|channel>thought\n"), 0);
    }

    #[test]
    fn a_reply_that_merely_resembles_dressing_is_not_skipped() {
        // "<|chan" + "llo" diverges from the dressing bytes -> 0
        assert_eq!(reseam_skip_dressing(dec(D), &[5, 4, 3], b"<|channel>thought\n"), 0);
    }

    #[test]
    fn no_dressing_no_skip() {
        assert_eq!(reseam_skip_dressing(dec(D), &[3, 4], b"<|channel>thought\n"), 0);
    }
}

/// How many suffix tokens must have been prefilled before it is worth keeping the result.
/// Below this the memcpy costs more than the prefill it would save.
#[cfg(feature = "wire_cuda_backend")]
fn roll_snap_min() -> usize {
    std::env::var("SP_ROLL_SNAPSHOT_MIN").ok()
        .and_then(|v| v.trim().parse().ok()).unwrap_or(384)
}

/// How far short of the prompt's end to stop. See the long note at the split site: a
/// prompt ends with the generation tag, and appending her reply re-tokenizes across that
/// boundary, so the last few tokens of any prompt are exactly the ones the NEXT prompt
/// will not agree with. Measured divergence was 2-3 tokens; 16 is clear of it.
#[cfg(feature = "wire_cuda_backend")]
fn roll_snap_margin() -> usize {
    std::env::var("SP_ROLL_SNAPSHOT_MARGIN").ok()
        .and_then(|v| v.trim().parse().ok()).unwrap_or(16)
}

/// Keep the cache exactly as it stands, as the rolling prefix for the next turn.
/// `toks` MUST be the token sequence the cache currently holds — the caller stands at
/// that position and nowhere else. Never fatal: on any failure the previous slot stays,
/// the cache is untouched, and the turn carries on. An optimisation that can break a
/// turn is a bug, so this returns nothing and refuses to.
#[cfg(feature = "wire_cuda_backend")]
fn roll_capture(handle: *mut std::ffi::c_void, toks: &[i32]) {
    use sp_daemon::cuda_kvdecode_dispatch as kv;
    let tag = match model_tag() { Some(t) => t, None => return };
    let p = toks.len();
    let mut roll = ROLL_SNAP.lock().unwrap_or_else(|p| p.into_inner());
    if roll.as_ref().map_or(false, |s| s.toks.len() >= p) { return; }
    let t0 = std::time::Instant::now();
    let kept = (|| -> Result<Vec<u8>, String> {
        let n = unsafe { kv::prefix_bytes(handle, p as i32) }?;
        let mut buf = vec![0u8; n];
        unsafe { kv::snapshot_prefix(handle, &mut buf, p as i32) }?;
        Ok(buf)
    })();
    match kept {
        Ok(blob) => {
            tracing::info!(
                "ROLL-SNAPSHOT: kept {} tokens ({:.1} MiB) in {:?} — next turn's suffix is \
                 his new sentence, not the whole conversation",
                p, blob.len() as f64 / 1048576.0, t0.elapsed());
            *roll = Some(PrefixSnap { toks: toks.to_vec(), blob, tag });
        }
        Err(e) => tracing::info!("ROLL-SNAPSHOT: declined ({e}) — keeping the old slot"),
    }
}

/// Identity of the model the KV caches above were minted from. Empty until the
/// daemon publishes it at load; an empty tag matches nothing, so an unpublished
/// daemon simply never reuses a cache (cold, correct) instead of reusing blindly.
#[cfg(feature = "wire_cuda_backend")]
static MODEL_TAG: std::sync::Mutex<String> = std::sync::Mutex::new(String::new());

#[cfg(feature = "wire_cuda_backend")]
pub fn publish_model_tag(tag: &str) {
    *MODEL_TAG.lock().unwrap_or_else(|p| p.into_inner()) = tag.to_string();
    tracing::info!("PERSIST-KV: caches tagged for model {tag}");
}

/// The live tag. Returns None when unpublished, which callers treat as "do not reuse".
#[cfg(feature = "wire_cuda_backend")]
fn model_tag() -> Option<String> {
    let t = MODEL_TAG.lock().unwrap_or_else(|p| p.into_inner());
    if t.is_empty() { None } else { Some(t.clone()) }
}

// B4-SEAL (GEODESIC session 2026-07-03): mint the L5 query-key for a LIVE-captured
// episode with the SAME provenance as the curated corpus (write_ep_l5.py): a
// standalone position-0 forward of the episode text on a throwaway scratch session,
// global-Q of the LAST token, mean-over-heads of global layer 5, L2-normed
// (recall::l5_query_embed). Writes <dir>/ep.l5 (raw LE f32[512] — exactly what
// load_episode_l5key reads back after a restart) and returns the key for the
// in-session Episode. None on any failure — the episode then behaves exactly as
// before this fix (W_c/C2-selectable, L5-invisible), never worse.
// THIS is the seam that seals the system: without it, grown memories are invisible
// to the deployed L5 selector (the old l5key: Vec::new() "follow-up" comment).
#[cfg(feature = "wire_cuda_backend")]
fn mint_live_ep_l5(qm: *const std::ffi::c_void, toks: &[i32], dir: &std::path::Path) -> Option<Vec<f32>> {
    use sp_daemon::cuda_kvdecode_dispatch as kv;
    use sp_daemon::recall;
    if toks.len() < 2 { return None; }
    let scratch = unsafe { kv::open_scratch(qm, toks.len() as i32 + 8) }.ok()?;   // F10
    let n_global = recall::n_global();
    let mut ql = vec![0.0f32; n_global * recall::g_nh() * recall::hd()];
    let key = (|| {
        unsafe { kv::prefill(scratch, &toks[..toks.len() - 1]) }.ok()?;
        unsafe { kv::read_global_q(scratch, toks[toks.len() - 1], &mut ql) }.ok()?;
        let k = recall::l5_query_embed(&ql);
        if k.len() != recall::hd() { return None; }
        Some(k)
    })();
    // SAFETY: scratch came from open() above; release is NULL-safe/idempotent.
    unsafe { kv::release_for_model(scratch) };
    if let Some(ref k) = key {
        let bytes: Vec<u8> = k.iter().flat_map(|x| x.to_le_bytes()).collect();
        if std::fs::write(dir.join("ep.l5"), bytes).is_err() {
        // ADR-013: stamp the geometry beside the key. A sidecar minted over one
        // global-layer count is not comparable to one minted over another, and the
        // raw f32 blob carries no identity of its own.
        sp_daemon::recall::write_l5_sidecar_tag(&dir.display().to_string());
            tracing::warn!("B4-L5-MINT: ep.l5 sidecar write failed at {}", dir.display());
        }
    } else {
        tracing::warn!("B4-L5-MINT: L5 key mint failed for {} (episode stays L5-invisible)", dir.display());
    }
    key
}

// QKEYS (2026-07-03): mint a QUESTION-SPACE ep.l5 key for a live-captured episode —
// the fix for the triply-evidenced statement-space cross-pick (cat 0.0101, my-name
// 0.0004, G-SPECTEST-V3 pin-2). ROOT CAUSE, proven: the curated corpus mints ep.l5 from
// the EXACT QUESTION's global-Q (write_ep_l5.py, from facts.json `q`), but mint_live_ep_l5
// mints from the ASSERTION text — different manifold regions ⇒ a question query matches the
// curated (question) keys well and the grown (statement) keys poorly. FIX: generate the
// question this fact answers (one scratch micro-forward, the canon decode pattern), then key
// it EXACTLY like a live query does — templatize the question as a user turn, prefill,
// read_global_q of the last token, l5_query_embed. Returns None on any failure ⇒ the caller
// falls back to the statement-space key (never worse than today). Gated by SP_QKEY_MINT.
#[cfg(feature = "wire_cuda_backend")]
fn mint_question_l5(qm: *const std::ffi::c_void, tok: &crate::tokenizer::SptbTokenizer,
                    vocab: usize, fact_text: &str, dir: &std::path::Path) -> Option<Vec<f32>> {
    use sp_daemon::cuda_kvdecode_dispatch as kv;
    use sp_daemon::recall;
    // (1) Generate the question the fact answers — a scratch micro-forward, greedy with the
    //     served control-token suppression (the placeholder-spam cure), stop on newline.
    //     ANTI-COPY GUARD (the "user's dog" bug, G-QKEYS-SYSTEMECHO diagnosis): on a hard
    //     fact the greedy decode COPIES the few-shot exemplar instead of generating the real
    //     question (3/30 V3 episodes minted "What is the name of the user's dog?" ⇒ their L5
    //     keys landed in an unrelated neighborhood ⇒ buried at rank >8 ⇒ cross-picked). Cure:
    //     (a) a NEUTRAL on-domain exemplar (no personal-memory attractor), and (b) require the
    //     generated question to share a salient token with the fact (canon_overlap>0); if it
    //     does not (exemplar-copy / hallucination), retry ZERO-SHOT (no exemplar to copy);
    //     fail both ⇒ None ⇒ statement-key fallback (never worse than today).
    let scratch = unsafe { kv::open_scratch(qm, 320) }.ok()?;   // F10: ring-off, its own Pmax
    let gen_once = |prompt: String| -> Option<String> {
        let gmsgs = vec![Message { role: "user".to_string(), content: prompt }];
        let gtoks = tok.apply_template_ids(&gmsgs).ok()?;
        if gtoks.len() < 2 { return None; }
        let mut logits = vec![0.0f32; vocab];
        let _ = unsafe { kv::reset(scratch) };
        let (ghead, glast) = gtoks.split_at(gtoks.len() - 1);
        if !ghead.is_empty() { unsafe { kv::prefill(scratch, ghead) }.ok()?; }
        let mut t = glast[0];
        let turn_stops = tok.turn_stop_ids();
        let suppress = tok.suppress_token_ids();
        let mut q = String::new();
        for _ in 0..32 {
            unsafe { kv::decode_step(scratch, t, &mut logits) }.ok()?;
            let mut best_t = 0usize; let mut best_v = f32::NEG_INFINITY;
            for (i, &v) in logits.iter().enumerate() {
                if suppress.contains(&(i as i32)) { continue; }
                if v > best_v { best_v = v; best_t = i; }
            }
            t = best_t as i32;
            if t == 1 || turn_stops.contains(&t) { break; }
            q.push_str(&String::from_utf8_lossy(tok.decode_token(t)));
            if q.contains('\n') { break; }
        }
        let q = q.trim().trim_start_matches("Question:").trim().lines().next().unwrap_or("").trim().to_string();
        if q.len() < 4 { None } else { Some(q) }
    };
    // grounded = shares a salient token with the fact (rejects exemplar-copy / hallucination).
    let grounded = |q: &str| recall::canon_overlap(q, fact_text) > 0;
    let fewshot = format!(
        "A fact is stated. Write ONLY the single short question this fact answers — no answer, no preamble.\n\
         Example fact: \"The tallest mountain on Earth is Mount Kea.\" -> Question: What is the tallest mountain on Earth?\n\
         Fact: \"{}\"\nQuestion:", fact_text.trim());
    let zeroshot = format!(
        "Fact: \"{}\"\nWrite ONLY the single short question this fact answers (no answer, no preamble):",
        fact_text.trim());
    let question = match gen_once(fewshot) {
        Some(q) if grounded(&q) => Some(q),
        first => {
            // exemplar-copy / hallucination / empty -> zero-shot retry (no exemplar to copy).
            if let Some(bad) = first.as_ref() {
                tracing::warn!("QKEY: few-shot Q {:?} not grounded in fact (exemplar-copy?) -> zero-shot retry", bad);
            }
            match gen_once(zeroshot) {
                Some(q) if grounded(&q) => Some(q),
                other => { if let Some(b) = other.as_ref() {
                    tracing::warn!("QKEY: zero-shot Q {:?} also not grounded -> statement-key fallback", b); } None }
            }
        }
    };
    let key = question.as_ref().and_then(|q| {
        // (2) Key the question EXACTLY as a live query is keyed: templated user turn,
        //     prefill, read_global_q of the last token, l5_query_embed.
        let qmsgs = vec![Message { role: "user".to_string(), content: q.clone() }];
        let qtoks = tok.apply_template_ids(&qmsgs).ok()?;
        if qtoks.len() < 2 { return None; }
        let _ = unsafe { kv::reset(scratch) };
        let (qhead, qlast) = qtoks.split_at(qtoks.len() - 1);
        if !qhead.is_empty() { unsafe { kv::prefill(scratch, qhead) }.ok()?; }
        let n_global = recall::n_global();
        let mut ql = vec![0.0f32; n_global * recall::g_nh() * recall::hd()];
        unsafe { kv::read_global_q(scratch, qlast[0], &mut ql) }.ok()?;
        let k = recall::l5_query_embed(&ql);
        if k.len() == recall::hd() { Some(k) } else { None }
    });
    unsafe { kv::release_for_model(scratch) };
    if let (Some(q), Some(k)) = (question.as_ref(), key.as_ref()) {
        let bytes: Vec<u8> = k.iter().flat_map(|x| x.to_le_bytes()).collect();
        let _ = std::fs::write(dir.join("ep.l5"), &bytes);
        let _ = std::fs::write(dir.join("ep.q"), q); // provenance: the generated question
        tracing::info!("QKEY: '{}' <- question-space key from generated Q: {:?}", dir.display(), q);
    } else {
        tracing::warn!("QKEY: question-space mint failed for {} (falling back to statement key)", dir.display());
    }
    key
}

// QKEYS: mint an episode's L5 key — QUESTION-SPACE (SP_QKEY_MINT=1) with a hard
// fallback to the statement-space key on any generation/mint failure (never worse than
// today). The one seam both capture paths call.
#[cfg(feature = "wire_cuda_backend")]
fn mint_ep_l5(app: &Arc<AppState>, qm: *const std::ffi::c_void, toks: &[i32],
              text: &str, dir: &std::path::Path) -> Vec<f32> {
    if std::env::var("SP_QKEY_MINT").as_deref() == Ok("1") {
        if let Some(k) = mint_question_l5(qm, app.tokenizer.as_ref(), app.vocab_size, text, dir) {
            return k;
        }
    }
    mint_live_ep_l5(qm, toks, dir).unwrap_or_default()
}

/// LM-B2 TELEMETRY FLYWHEEL: append one structured RECALL-DECISION record to SP_TELEMETRY_LOG
/// (JSONL, the telemetry-okf bundle). The query is REDACTED to a hash when the fired entry is a
/// `private-secret` — telemetry can NEVER carry a value the delivery path would have declined
/// (ADR-005 §3b, privacy-before-data). Gated SP_TELEMETRY=1; unset ⇒ no-op (null floor). Off the
/// token path (called at the decision point, no forward). The engine logs the DECISION; the
/// harness joins the OUTPUT + outcome by turn.
fn telem_emit_recall(query: &str, entry: &str, class: &str, cos: f32, margin: f32, delivery: &str,
                     decision: &str, evt: Option<&tokio::sync::broadcast::Sender<crate::state::DaemonEvent>>) {
    if std::env::var("SP_TELEMETRY").as_deref() != Ok("1") { return; }
    let secret = class == "private-secret";
    let q = if secret {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        query.hash(&mut h);
        format!("#{:016x}", h.finish())
    } else { query.to_string() };
    let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs()).unwrap_or(0);
    let rec = serde_json::json!({
        "ts": ts, "query": q, "redacted": secret,
        "recall": { "fired": decision == "deliver" || decision == "decline",
                    "entry": entry, "class": class, "cos": cos,
                    "margin": if margin.is_finite() { margin } else { -1.0 },
                    "delivery": delivery, "decision": decision }
    });
    telem_sink(&rec, evt);
}

/// LM-B2 SSE-SINK: write a telemetry record to the JSONL file (if SP_TELEMETRY_LOG is set) AND
/// broadcast it on the events channel (if a sender is provided) as a `Telemetry` DaemonEvent —
/// so a live /v1/events subscriber (the harness StreamProcessor) sinks it into the durable store
/// without tailing a file. The record is ALREADY class-redacted by the caller; the SSE payload is
/// byte-identical to the file line. File and SSE are independent (either, both, or neither).
fn telem_sink(rec: &serde_json::Value, evt: Option<&tokio::sync::broadcast::Sender<crate::state::DaemonEvent>>) {
    if let Ok(path) = std::env::var("SP_TELEMETRY_LOG") {
        if !path.is_empty() {
            use std::io::Write as _;
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
                let _ = writeln!(f, "{}", rec);
            }
        }
    }
    if let Some(tx) = evt {
        let _ = tx.send(crate::state::DaemonEvent::Telemetry { record: rec.to_string() });
    }
}

/// LM-B2b TURN-OUTCOME telemetry: one `turn` record per DELIVERED recall turn, joining the
/// decision (by query) to the OUTPUT + timing + an obeyed self-check (did the answer use the
/// recalled fact's salient tokens? — the inverse of the parametric-hallucination flag). PRIVACY
/// (ADR-005 §3b): when the recalled entry is a private-secret the query AND the output are
/// REDACTED (hash + char-length, never raw) — telemetry never carries a secret even though the
/// owner legitimately saw the recite. Non-secret turns keep query->output in the clear = the
/// finetune signal. Gated SP_TELEMETRY=1 + SP_TELEMETRY_LOG; off the token path (one append).
fn telem_emit_turn(query: &str, entry: &str, class: &str, output: &str, fact: Option<&str>,
                   n_out: usize, decode_s: f64, secret: bool,
                   evt: Option<&tokio::sync::broadcast::Sender<crate::state::DaemonEvent>>) {
    if std::env::var("SP_TELEMETRY").as_deref() != Ok("1") { return; }
    let hash = |s: &str| { use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        s.hash(&mut h); format!("#{:016x}", h.finish()) };
    // obeyed = answer used >=1 salient (len>=4) token of the recalled fact.
    let obeyed = fact.map(|f| {
        let al = output.to_lowercase();
        let sal: Vec<String> = f.split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() >= 4).map(|w| w.to_lowercase()).collect();
        sal.is_empty() || sal.iter().any(|w| al.contains(w.as_str()))
    });
    let tok_s = if decode_s > 1e-3 { n_out as f64 / decode_s } else { 0.0 };
    let out_len = output.chars().count();
    let (q, out_val) = if secret {
        (hash(query), serde_json::json!({ "redacted": true, "sha": hash(output), "len": out_len }))
    } else {
        (query.to_string(), serde_json::json!(output))
    };
    let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs()).unwrap_or(0);
    let rec = serde_json::json!({
        "ts": ts, "kind": "turn", "query": q, "redacted": secret,
        "turn": { "entry": entry, "class": class, "output": out_val, "out_len": out_len,
                  "n_out": n_out, "decode_s": (decode_s * 1000.0).round() / 1000.0,
                  "tok_s": (tok_s * 100.0).round() / 100.0, "obeyed": obeyed }
    });
    telem_sink(&rec, evt);
}

/// STORE-MERGE (#76/#77): load memory-okf/full/*.md concepts as SERVABLE episodes so a memory
/// the HARNESS writes is instantly recallable by the engine (one content-addressed store, both
/// callers). Text + policy come from the OKF concept; the L5 selection key is loaded from a
/// cached sidecar (full/<addr>.l5) or MINTED from the body at startup (question-space, cached).
/// The resulting text-delivery episodes are pushed into app.nightshift so the live L5 recall
/// path selects them; delivery/decline follow the concept's OWN policy (ADR-004). Gated by
/// SP_MEM_OKF_STORE; unset ⇒ never called ⇒ null floor. Returns the count added.
#[cfg(feature = "wire_cuda_backend")]
pub fn load_and_mint_okf_store(app: &Arc<AppState>, root: &str) -> usize {
    use sp_daemon::recall;
    let full = std::path::Path::new(root).join("full");
    let rd = match std::fs::read_dir(&full) { Ok(r) => r, Err(_) => return 0 };
    let qm = {
        let mut sguard = match app.session.as_ref() { Some(s) => s.lock().unwrap(), None => return 0 };
        let sraw = sguard.raw_ptr() as *mut sp_daemon::ffi_l1::sp_session;
        (unsafe { sp_daemon::ffi_l1::sp_session_qwen3_model(sraw) }) as *const std::ffi::c_void
    };
    if qm.is_null() { return 0; }
    // NO whole-pass device lock here (2026-08-29 audit, F-2): this pass walks EVERY
    // concept file — read_dir, read_to_string, registry writes — and in the steady
    // state forwards on none of them. Holding the device lock across the walk made a
    // user turn wait for the WHOLE pass; the doc's promise is that a turn waits for
    // at most ONE background forward. The guard is taken per-mint, below.
    // INCREMENTAL (LM-B1): skip concepts already merged, so repeated reconcile passes only
    // pay for NEW/changed concepts (the live hot-reload path). Match by episode name == addr.
    let already: std::collections::HashSet<String> =
        app.nightshift.read().unwrap().iter().map(|e| e.name.clone()).collect();
    let (mut added, mut minted, mut edited) = (0usize, 0usize, 0usize);
    let mut eps: Vec<recall::Episode> = Vec::new();
    for ent in rd.flatten() {
        let path = ent.path();
        if path.extension().and_then(|e| e.to_str()) != Some("md") { continue; }
        let addr = match path.file_stem() { Some(s) => s.to_string_lossy().to_string(), None => continue };
        let raw = match std::fs::read_to_string(&path) { Ok(t) => t, Err(_) => continue };
        let policy = recall::parse_okf_policy(&raw);
        if already.contains(&addr) {
            // RECONCILE-ON-EDIT (LM-B1 cache-invalidation): the body is content-addressed so the
            // addr is stable, but the FRONTMATTER policy can change (e.g. the DF-B6 harness curator
            // corrects mem_class). If the in-memory policy class differs from the file, update it
            // in place so the served spine picks up the correction — no restart, no re-mint.
            let mut ns = app.nightshift.write().unwrap();
            if let Some(ep) = ns.iter_mut().find(|e| e.name == addr) {
                let cur = ep.policy.as_ref().map(|p| p.class.clone());
                let new = policy.as_ref().map(|p| p.class.clone());
                if cur != new {
                    tracing::info!("STORE-MERGE reconcile-edit: '{}' policy {:?} -> {:?}", addr, cur, new);
                    ep.policy = policy;
                    edited += 1;
                }
            }
            continue;
        }
        let body = recall::okf_body(&raw);
        if body.is_empty() { continue; }
        // L5 selection key: cached sidecar or mint (question-space) from the body, then cache.
        let cache = full.join(format!("{addr}.l5"));
        let l5key: Vec<f32> = if let Ok(bytes) = std::fs::read(&cache) {
            bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
        } else {
            let keydir = full.join(format!("_k_{addr}"));
            let _ = std::fs::create_dir_all(&keydir);
            let _gpu = device_guard(app, "okf_store");   // one forward, one hold (F-2)
            match mint_question_l5(qm, app.tokenizer.as_ref(), app.vocab_size, &body, &keydir) {
                Some(k) => {
                    let bytes: Vec<u8> = k.iter().flat_map(|x| x.to_le_bytes()).collect();
                    let _ = std::fs::write(&cache, &bytes);
                    minted += 1;
                    k
                }
                None => { tracing::warn!("STORE-MERGE: L5 mint failed for concept {addr}"); continue; }
            }
        };
        if l5key.len() != recall::hd() { continue; }
        eps.push(recall::Episode {
            name: addr.clone(),
            dir: path.to_string_lossy().to_string(),
            npos: 0,
            topic: body.chars().take(40).collect(),
            text: body,
            sig: [0u64; 4],
            gk: Vec::new(), gk_ng: 0,
            tokens: None,
            l5key,
            policy,
            lifecycle: 0,
        });
        added += 1;
    }
    if added > 0 {
        app.nightshift.write().unwrap().extend(eps);
    }
    tracing::info!("STORE-MERGE: loaded {} memory-okf concept(s) from {} ({} L5 keys minted, rest cached; {} policy-edits reconciled)", added, root, minted, edited);
    added + edited
}

/// LM-B3 adaptive classification: classify a memory's text into a mem_class via ONE model
/// micro-forward (accurate, off the hot path — used by the idle NIGHTSHIFT refine pass). The
/// heuristic classify_mem_class runs at capture (instant); this CORRECTS it on idle. Returns a
// ── /v1/oneshot — A CALL THAT IS NEVER CONTINUED SHOULD NOT COST A CONVERSATION ──────
//
// THE BUG, MEASURED: the watch judge, the reflection and the summariser each send a ~1450-token
// prompt down the ONE RESIDENT KV SLOT — the same slot holding his conversation. Two things
// followed, and both were invisible:
//
//   1. THE CALL ITSELF WAS ABSURD. Its prompt shares almost nothing with the persona preamble,
//      so the LCP collapses and it pays a full per-token prefill: ~1450 tokens x 60 ms/tok =
//      78 SECONDS, to produce a single YES/NO token. The batched accelerant — "correct + ~5-7x
//      faster", by the engine's own comment — was DECLINED, because its guard tested a
//      PROCESS-WIDE SP_PERSIST_KV. The chat path needs persistence, so the judge (which has
//      nothing whatsoever to continue) was disqualified along with it. THE INVARIANT BELONGED TO
//      THE SESSION AND WAS BEING ASKED OF THE PROCESS.
//
//   2. IT EVICTED HIS CONVERSATION. The aux prompt becomes KV_COMMITTED, so his NEXT turn is no
//      longer a strict extension of anything and re-prefills from token 0.
//
// A one-shot call has no continuation to protect. It gets its OWN scratch session, sized to its
// own prompt, marked one_shot (which legalises the batched prefill), decoded, and released. The
// resident cache is not read, not written, and not evicted. This is ADR-009's rule, finally
// applied to the callers that needed it: "a batched forward on a scratch cache — NEVER the
// resident session."
#[cfg(feature = "wire_cuda_backend")]
#[derive(serde::Deserialize)]
pub struct OneshotRequest {
    pub messages: Vec<Message>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    /// 0.0 (default) = greedy. The aux callers are judges and classifiers; they want the
    /// most likely token, not a personality.
    #[serde(default)]
    pub temperature: f32,
    /// Wall-clock budget in seconds, 0.0 = the SP_ONESHOT_DEADLINE_S default (600).
    /// THE CLIENT'S PATIENCE, TOLD TO THE SERVER (2026-08-30): the reflect oneshot
    /// crawled for 33+ minutes in WDDM paging territory, its harness client timed out
    /// at 600 s — and nothing told the daemon, so the abandoned forward kept the device
    /// lock and every turn behind it read as "stuck warming, 99% GPU". A oneshot whose
    /// client has given up must give the device back.
    #[serde(default)]
    pub deadline_s: f32,
}

#[cfg(feature = "wire_cuda_backend")]
#[derive(serde::Serialize)]
pub struct OneshotResponse {
    pub text: String,
    pub prompt_tokens: usize,
    pub batched: bool,
    pub ms: u128,
}

#[cfg(feature = "wire_cuda_backend")]
pub async fn v1_oneshot(
    axum::extract::State(app): axum::extract::State<Arc<AppState>>,
    axum::Json(req): axum::Json<OneshotRequest>,
) -> Result<axum::Json<OneshotResponse>, (axum::http::StatusCode, String)> {
    let toks = app.tokenizer.apply_template_ids(&req.messages)
        .map_err(|e| (axum::http::StatusCode::BAD_REQUEST, format!("template: {e}")))?;
    if toks.len() < 2 {
        return Err((axum::http::StatusCode::BAD_REQUEST, "prompt too short".into()));
    }
    // ── OFF THE RUNTIME (2026-08-29 audit, F-1). This ran its whole body — a
    // yield_to_foreground that can SLEEP up to SP_PRIORITY_YIELD_CAP_S (1800 s)
    // behind a live turn, then the device lock across a full prefill+decode — on a
    // tokio WORKER THREAD. The judge, the reflection and the summariser all fire
    // from timers mid-turn, so three parked one-shots against a small worker pool
    // starved the accept loop and /v1/metrics: the whole daemon's HTTP went dark
    // while the GPU was merely busy. That is the 2026-08-28 14.5-minute page
    // freeze (metrics polls VANISHED in that incident — the CUDA-wedge class keeps
    // metrics green, which is how the two stall mechanisms tell apart in the log).
    // Same spawn_blocking shape as capture/recall_rank/embed/dialogue.
    let res = task::spawn_blocking(move || -> Result<OneshotResponse, (axum::http::StatusCode, String)> {
    use sp_daemon::cuda_kvdecode_dispatch as kv;
    let t0 = std::time::Instant::now();
    let vocab = app.vocab_size;
    let max_new = req.max_tokens.clamp(1, 512) as usize;

    let qm = {
        let mut sg = app.session.as_ref()
            .ok_or((axum::http::StatusCode::SERVICE_UNAVAILABLE, "no session".to_string()))?
            .lock().unwrap();
        let sraw = sg.raw_ptr() as *mut sp_daemon::ffi_l1::sp_session;
        (unsafe { sp_daemon::ffi_l1::sp_session_qwen3_model(sraw) }) as *const std::ffi::c_void
    };
    if qm.is_null() {
        return Err((axum::http::StatusCode::SERVICE_UNAVAILABLE, "no model".into()));
    }

    // Its OWN cache, sized to its OWN prompt, RING-OFF, and marked one-shot.
    //
    // The ring-off part was a bug I shipped and then measured: opening this with plain open()
    // handed a ~620-token judge call the PROCESS's 2048-slot SWA ring — ~490 MB it would never
    // read — and on a full card the daemon spilled 860 MiB into host memory and ran at
    // 218 ms/tok. THE EVICTION FIX WAS PAYING FOR ITSELF WITH A SPILL. A short-lived cache
    // wants a full cache at its own small Pmax, which is cheaper AND is the batched prefill's
    // happiest precondition. open_scratch() also sets one_shot, so the batched path is legal.
    // ── ONE FORWARD AT A TIME ON THIS DEVICE (2026-08-03) ────────────────────────────
    // This opens its OWN cache, which made it look independent of the resident one. It is
    // not. Both forwards run through the SAME file-scope device scratch in
    // cuda_forward.cu — `g_w`'s expert staging and the whole `g_moe_*` group — and
    // `moe_scratch_ensure` FREES AND REALLOCS that group whenever a bigger n_tok arrives.
    //
    // So a judge or a classifier or the narrator (all of them /v1/oneshot, all of them
    // fired from timer threads) landing in the middle of a chat turn frees the buffers the
    // chat turn's kernels are running on. That is an illegal memory access, and it is
    // exactly the fault that has been wedging or killing the daemon since 2026-07-29 —
    // which fires regardless of who started it, at whatever sync point comes next, and got
    // worse the busier the room became.
    //
    // The resident path already serialises on this Mutex. Taking it here is what makes
    // that serialisation mean "one forward on the GPU" instead of "one forward on the
    // resident cache". It is held across the whole prefill+decode, and released on every
    // exit path because the guard is dropped with the scope.
    // ── THE DEADLINE IS PART OF THE REQUEST (2026-08-30) ─────────────────────────────
    // Measured tonight, the same 347-token reflect oneshot across one evening: 5.7 s,
    // 272 s, 5.4 s, 168 s, then 33+ minutes and never finished — decode crawling at
    // ~0.001 tok/s with VRAM at 11962/12288 (WDDM paging thrash; cudaMemGetInfo's
    // free=0 oracle means the engine cannot see it coming). The client gave up at its
    // 600 s timeout and the daemon never knew, so the abandoned forward held the
    // device lock indefinitely: his turn parked at the lock, the scheduled re-prefill
    // parked behind that, the room said "warming" forever. The deadline bounds the
    // harm class whatever the cause: checked before work starts (yield_to_foreground
    // can sleep 1800 s — do not start a forward the client stopped wanting) and
    // between decode steps (the loop below is the only place the crawl is visible
    // from Rust; a mid-prefill check would need C-side surgery and the prefill is
    // chunked/bounded anyway). Worst case is now deadline + one step, not forever.
    let deadline = std::time::Duration::from_secs_f64(if req.deadline_s > 0.0 {
        req.deadline_s as f64
    } else {
        static DEFAULT_S: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
        *DEFAULT_S.get_or_init(|| std::env::var("SP_ONESHOT_DEADLINE_S").ok()
            .and_then(|v| v.parse::<f64>().ok()).filter(|v| *v > 0.0).unwrap_or(600.0))
    });
    // device_guard, not a raw Option::map — the map spelling is F-3's "no handle, no
    // lock at all" silent no-op, fixed at the other five lanes on 2026-08-29 and
    // unified here. device_guard yields to any queued foreground turn first.
    let _gpu = device_guard(&app, "oneshot");
    if t0.elapsed() > deadline {
        return Err((axum::http::StatusCode::SERVICE_UNAVAILABLE,
                    format!("oneshot deadline ({:.0} s) passed while queued — not starting",
                            deadline.as_secs_f64())));
    }
    let need = (toks.len() + max_new + 8) as i32;
    // ── A SCRATCH THIS BIG DOES NOT FIT, AND MUST SAY SO (2026-08-30) ────────────────
    // MEASURED, three wedges this morning. A scratch is opened ring-OFF (deliberately:
    // a short one-shot must not be handed the process's 2048-slot SWA ring it will
    // never read), and `slots = Pmax` for EVERY layer follows from that — so a scratch
    // costs ~4x per position what the resident session does, where only the 5 global
    // layers pay Pmax. An unbounded caller (conversation_memory fed the WHOLE day
    // transcript, ~20k tokens) therefore asked for 4.30 GB — the engine's own
    // `KV-FP16 ... freed 4.30 GB` line — on top of 7.5 GB resident, on a 12.3 GB card.
    // cudaMalloc SUCCEEDS at 97%: WDDM pages the excess over PCIe instead of failing,
    // the forward crawls for hours holding the device lock, and every turn of hers
    // queues behind it. There is no error anywhere in that story, which is why it took
    // three occurrences to find.
    //
    // So the ceiling is explicit and it REFUSES rather than clamps: clamping would
    // silently truncate someone's prompt and answer the wrong question. The caller
    // sees a named 413 and the log says the number. 6144 positions is ~1.3 GB here —
    // comfortably inside the ~4.3 GB of headroom this card has once warm — and every
    // legitimate one-shot on this stack (judge, classifier, reflection, summariser) is
    // an order of magnitude below it. Raise it only against a measured headroom.
    let pmax_max: i32 = std::env::var("SP_ONESHOT_PMAX_MAX").ok()
        .and_then(|v| v.parse().ok()).filter(|v| *v > 0).unwrap_or(6144);
    if need > pmax_max {
        tracing::warn!("ONESHOT: refused — {} prompt tok needs a {}-position scratch, \
                        over the {} ceiling (a scratch is ring-off: every layer pays \
                        Pmax, so this would allocate into WDDM paging and wedge the \
                        device). Bound the caller's prompt.",
                       toks.len(), need, pmax_max);
        return Err((axum::http::StatusCode::PAYLOAD_TOO_LARGE,
                    format!("oneshot prompt too long: {} tok needs {} positions > {} \
                             (SP_ONESHOT_PMAX_MAX)", toks.len(), need, pmax_max)));
    }
    let scratch = unsafe { kv::open_scratch(qm, need) }
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, format!("scratch open: {e}")))?;

    let mut batched = false;
    let mut timed_out = false;
    let out = (|| -> Option<String> {
        let mut logits = vec![0.0f32; vocab];
        let (head, last) = toks.split_at(toks.len() - 1);
        if !head.is_empty() {
            match unsafe { kv::prefill_batched(scratch, head) } {
                Ok(()) => { batched = true; }
                Err(e) => {
                    // Never a hard failure: fall through to the per-token path, which is the
                    // byte-identical floor. Slow is not wrong.
                    tracing::info!("ONESHOT: batched prefill declined ({e}) — per-token floor");
                    unsafe { kv::prefill(scratch, head) }.ok()?;
                }
            }
        }
        let mut t = last[0];
        let stops = app.tokenizer.turn_stop_ids();
        let suppress = app.tokenizer.suppress_token_ids();
        let mut s = String::new();

        // TEMPERATURE IS HONOURED, NOT DECORATIVE. My first cut took `temperature` in the
        // request and then decoded pure argmax anyway. The watch judge wants greedy (0.0) —
        // fine — but the REFLECTION runs at 0.4 precisely because a greedy "what have you
        // concluded about him?" returns the same dull sentence forever. Silently ignoring the
        // knob would have made her reflections deterministic and repetitive, and I would have
        // shipped it as a speedup. A parameter you accept and discard is a lie with a schema.
        let temp = req.temperature.clamp(0.0, 2.0);
        let mut rng: u64 = 0x9E3779B97F4A7C15 ^ (toks.len() as u64);

        for _ in 0..max_new {
            // between steps, because the step is where a paging crawl spends its time —
            // an expired budget frees the device instead of finishing an answer nobody
            // is waiting for. The partial text is discarded; a oneshot has no partials.
            if t0.elapsed() > deadline { timed_out = true; return None; }
            unsafe { kv::decode_step(scratch, t, &mut logits) }.ok()?;
            for &sid in suppress.iter() {
                if (sid as usize) < logits.len() { logits[sid as usize] = f32::NEG_INFINITY; }
            }
            t = if temp <= 1e-4 {
                // greedy
                let mut bt = 0usize;
                let mut bv = f32::NEG_INFINITY;
                for (i, &v) in logits.iter().enumerate() {
                    if v > bv { bv = v; bt = i; }
                }
                bt as i32
            } else {
                // softmax(logits / T), then inverse-CDF sample. xorshift keeps it dependency-free
                // and reproducible per prompt length; this path is a reflection, not a receipt.
                let maxv = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f64;
                for v in logits.iter_mut() {
                    let p = (((*v - maxv) / temp) as f64).exp();
                    *v = p as f32;
                    sum += p;
                }
                rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                let mut r = (rng as f64 / u64::MAX as f64) * sum;
                let mut pick = logits.len() - 1;
                for (i, &p) in logits.iter().enumerate() {
                    r -= p as f64;
                    if r <= 0.0 { pick = i; break; }
                }
                pick as i32
            };
            if t == 1 || stops.contains(&t) { break; }
            s.push_str(&String::from_utf8_lossy(app.tokenizer.decode_token(t)));
        }
        Some(s)
    })();

    // ALWAYS. A scratch session that outlives its turn is a VRAM leak that presents, days later,
    // as "the daemon got slow" — which is exactly the kind of bug this file is full of.
    unsafe { kv::release_for_model(scratch) };

    if timed_out {
        tracing::warn!("ONESHOT: deadline ({:.0} s) expired mid-decode after {} ms — \
                        scratch released, device freed",
                       deadline.as_secs_f64(), t0.elapsed().as_millis());
        return Err((axum::http::StatusCode::SERVICE_UNAVAILABLE,
                    format!("oneshot deadline ({:.0} s) expired mid-decode",
                            deadline.as_secs_f64())));
    }
    let text = out.ok_or((axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                          "oneshot decode failed".to_string()))?;
    let ms = t0.elapsed().as_millis();
    tracing::info!("ONESHOT: {} prompt tok, {} batched, {} ms — the resident cache was not touched",
                   toks.len(), if batched { "BATCHED" } else { "per-token" }, ms);
    Ok(OneshotResponse { text, prompt_tokens: toks.len(), batched, ms })
    }).await;
    match res {
        Ok(r) => r.map(axum::Json),
        Err(e) => Err((axum::http::StatusCode::INTERNAL_SERVER_ERROR, format!("join: {e}"))),
    }
}

/// known class or None. Reuses the scratch-session decode pattern (canon suppress-mask).
#[cfg(feature = "wire_cuda_backend")]
fn model_classify(qm: *const std::ffi::c_void, tok: &crate::tokenizer::SptbTokenizer,
                  vocab: usize, text: &str) -> Option<String> {
    use sp_daemon::cuda_kvdecode_dispatch as kv;
    let prompt = format!(
        "Classify the memory into ONE category. Reply with ONLY the category word.\n\
         - private-secret: a password, code, PIN, key, credential, recovery phrase, security-question \
answer, or any private identifier that must be protected from disclosure.\n\
         - persona: the user's or the agent's own identity, name, or personal preference.\n\
         - counterfact: a general-world fact (possibly revised) asserted as authoritative.\n\
         - fact: a neutral general fact.\n\
         Memory: \"{}\"\nCategory:", text.trim());
    let msgs = vec![Message { role: "user".to_string(), content: prompt }];
    let toks = tok.apply_template_ids(&msgs).ok()?;
    if toks.len() < 2 { return None; }
    let scratch = unsafe { kv::open_scratch(qm, (toks.len() as i32) + 16) }.ok()?;   // F10
    let out = (|| -> Option<String> {
        let mut logits = vec![0.0f32; vocab];
        let (head, last) = toks.split_at(toks.len() - 1);
        if !head.is_empty() { unsafe { kv::prefill(scratch, head) }.ok()?; }
        let mut t = last[0];
        let stops = tok.turn_stop_ids();
        let suppress = tok.suppress_token_ids();
        let mut s = String::new();
        for _ in 0..10 {
            unsafe { kv::decode_step(scratch, t, &mut logits) }.ok()?;
            let (mut bt, mut bv) = (0usize, f32::NEG_INFINITY);
            for (i, &v) in logits.iter().enumerate() {
                if suppress.contains(&(i as i32)) { continue; }
                if v > bv { bv = v; bt = i; }
            }
            t = bt as i32;
            if t == 1 || stops.contains(&t) { break; }
            s.push_str(&String::from_utf8_lossy(tok.decode_token(t)));
            if s.contains('\n') { break; }
        }
        Some(s.trim().to_lowercase())
    })();
    unsafe { kv::release_for_model(scratch) };
    let out = out?;
    // parse the reply to a known class (private-secret first — safety-monotone).
    for c in ["private-secret", "persona", "preference", "counterfact", "episodic-event", "fact"] {
        if out.contains(c) { return Some(c.to_string()); }
    }
    None
}

/// LM-B3: rewrite a store concept's OKF frontmatter policy in place for the NEW class (drops the
/// old policy lines, inserts fresh ones from the class defaults). Body + other frontmatter kept.
/// F17 (2026-08-29 audit): `std::fs::write` TRUNCATES then writes — a crash, a full
/// disk or a power cut between the two loses the whole file, and two of the three
/// callers hold HER MEMORY REGISTRY. tmp + rename, the same discipline the harness's
/// _save_all keeps on the identical file.
fn write_atomic(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = path.with_extension("tmp.atomic");
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)
}

fn rewrite_okf_class(path: &std::path::Path, new_class: &str) {
    let raw = match std::fs::read_to_string(path) { Ok(t) => t, Err(_) => return };
    let pol = sp_daemon::recall::MemPolicy::from_class(new_class);
    let auth = match new_class {
        "private-secret" => "private",
        "persona" | "preference" | "fact" | "episodic-event" => "supplements",
        _ => "overrides-prior",
    };
    let drop = ["mem_class:", "mem_delivery:", "mem_authority:", "mem_decline_when:", "mem_decline_message:"];
    let mut out = String::new();
    let (mut in_fm, mut wrote) = (false, false);
    for line in raw.lines() {
        if line.trim() == "---" {
            if in_fm && !wrote {
                out.push_str(&format!("mem_class: {}\nmem_delivery: {}\nmem_authority: {}\n",
                    new_class, pol.delivery, auth));
                if !pol.decline_when.is_empty() {
                    out.push_str(&format!("mem_decline_when: [{}]\nmem_decline_message: {}\n",
                        pol.decline_when.join(", "), pol.decline_message));
                }
                wrote = true;
            }
            out.push_str("---\n");
            in_fm = !in_fm;
            continue;
        }
        if in_fm && drop.iter().any(|d| line.trim_start().starts_with(d)) { continue; }
        out.push_str(line); out.push('\n');
    }
    if let Err(e) = write_atomic(path, out.as_bytes()) {
        tracing::warn!("okf class rewrite failed (kept the old file): {e}");
    }
}

/// LM-B3 NIGHTSHIFT refine: on idle, re-classify each store concept via a model call and, if it
/// differs from the current class, CORRECT it — update the served episode's policy in memory AND
/// rewrite the concept's OKF frontmatter (persistence). Safety-monotone: never downgrades a
/// private-secret. Gated SP_MEM_CLASSIFY_REFINE=1. Returns the number of corrections.
#[cfg(feature = "wire_cuda_backend")]
pub fn refine_okf_store(app: &Arc<AppState>, root: &str) -> usize {
    use sp_daemon::recall;
    let full = std::path::Path::new(root).join("full");
    let rd = match std::fs::read_dir(&full) { Ok(r) => r, Err(_) => return 0 };
    let qm = {
        let mut sg = match app.session.as_ref() { Some(s) => s.lock().unwrap(), None => return 0 };
        let sraw = sg.raw_ptr() as *mut sp_daemon::ffi_l1::sp_session;
        (unsafe { sp_daemon::ffi_l1::sp_session_qwen3_model(sraw) }) as *const std::ffi::c_void
    };
    if qm.is_null() { return 0; }
    // per-forward device lock only (2026-08-29 audit, F-2): one concept's classify per
    // hold, so a user turn parks behind at most ONE micro-forward, never the pass.
    let mut changed = 0usize;
    for ent in rd.flatten() {
        let path = ent.path();
        if path.extension().and_then(|e| e.to_str()) != Some("md") { continue; }
        let addr = match path.file_stem() { Some(s) => s.to_string_lossy().to_string(), None => continue };
        let raw = match std::fs::read_to_string(&path) { Ok(t) => t, Err(_) => continue };
        let cur = recall::parse_okf_policy(&raw).map(|p| p.class);
        let body = recall::okf_body(&raw);
        if body.is_empty() { continue; }
        let new = {
            let _gpu = device_guard(app, "refine_okf");   // one forward, one hold (F-2)
            match model_classify(qm, app.tokenizer.as_ref(), app.vocab_size, &body) {
                Some(c) => c, None => continue }
        };
        // PARITY (SP_MEM_REFINE_LOGALL=1): log the dense reference model model_classify verdict for EVERY concept
        // (not just changes) so the harness can A/B the deployed 0.5B curator against the dense reference model.
        if std::env::var("SP_MEM_REFINE_LOGALL").as_deref() == Ok("1") {
            tracing::info!("MEM-CLASSIFY: '{}' -> {}", addr, new);
        }
        if cur.as_deref() == Some(new.as_str()) { continue; } // unchanged
        // SAFETY-MONOTONE (ADR-004): never downgrade a private-secret to a leakier class.
        if cur.as_deref() == Some("private-secret") && new != "private-secret" { continue; }
        {
            let mut ns = app.nightshift.write().unwrap();
            if let Some(ep) = ns.iter_mut().find(|e| e.name == addr) {
                ep.policy = Some(recall::MemPolicy::from_class(&new));
            }
        }
        rewrite_okf_class(&path, &new);
        tracing::info!("MEM-REFINE: '{}' {:?} -> {} (model-reclassified)", addr, cur, new);
        changed += 1;
    }
    changed
}

// LAYER-3 MERGE helper: capture an ARBITRARY text as a new live episode, with the
// SAME provenance as the NIGHTSHIFT path (BOS kept + trailing newline, batched forward
// via the resident model, real C2 sig, persisted to the registry if SP_NIGHTSHIFT_PERSIST).
// Used to write the SYNTHESIZED fact a MERGE consolidates into. Returns true on success.
// Kept separate from the inline B4 capture so the proven NIGHTSHIFT path is untouched;
// uses a millis-unique episode name so it can never collide with a forgotten ep dir.
#[cfg(feature = "wire_cuda_backend")]
// LOCKING INVARIANT (2026-08-29): the caller ALREADY HOLDS THE DEVICE LOCK — both
// call sites (the SP_MEM_STORE verb and the B4-NIGHTSHIFT inline capture) live inside
// the chat worker's cuda_kvdecode_handle guard, so this fn must NOT take device_guard
// (std::Mutex is not reentrant; taking it here is a self-deadlock). A NEW caller from
// outside the chat pipeline must take device_guard(app, ...) FIRST — the wedge class
// this rule closes is documented at borrow_qm.
fn capture_live_episode(app: &Arc<AppState>, text: &str) -> bool {
    use sp_daemon::cuda_kvdecode_dispatch as kv;
    let text_nl = format!("{text}\n");
    let toks = match app.tokenizer.encode(&text_nl) { Ok(t) => t, Err(_) => return false };
    let ntok = toks.len();
    if ntok < 4 { return false; }
    let qm = {
        let mut sguard = app.session.as_ref().expect("L1 session unavailable (qwen36 lane)").lock().unwrap();
        let sraw = sguard.raw_ptr() as *mut sp_daemon::ffi_l1::sp_session;
        (unsafe { sp_daemon::ffi_l1::sp_session_qwen3_model(sraw) }) as *const std::ffi::c_void
    };
    if qm.is_null() { return false; }
    let uniq = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis()).unwrap_or(0);
    let name = format!("ep_live_m{uniq}");
    // EPISODE HOME (2026-07-12): episodes used to land in <engine>/_nightshift_live —
    // INSIDE THE ENGINE SOURCE TREE, beside the crate. That is why 323 of 405 registry
    // rows still pointed at the OLD staging repo's path: the memories outlived the repo
    // they were born in, and would have died with it. Episodes belong next to the
    // registry they are indexed by (var/memory/eps), which is the operator's data, not
    // the engine's build output. Falls back to the old location only if the registry
    // env is unset (a bare daemon with no memory configured).
    let dir = match std::env::var("SP_RECALL_REGISTRY") {
        Ok(reg) if !reg.trim().is_empty() => std::path::Path::new(&reg)
            .parent().unwrap_or_else(|| std::path::Path::new("."))
            .join("eps").join(&name),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent().and_then(|p| p.parent()).unwrap_or_else(|| std::path::Path::new("."))
            .join("_nightshift_live").join(&name),
    };
    let dir_str = dir.to_string_lossy().to_string();
    if std::fs::create_dir_all(&dir).is_err() { return false; }
    { let _ = std::fs::write(dir.join("ep.txt"), text);
      let _ = std::fs::write(dir.join("ep.tok"), toks.iter().map(|t| t.to_string()).collect::<Vec<_>>().join("\n")); }
    if unsafe { kv::capture_batched(qm, &toks, &dir_str) }.is_err() { return false; }
    let (gk, ng) = match sp_daemon::recall::load_episode_global_k(&dir_str, ntok as i32) {
        Some(x) => x, None => return false };
    let npos_sig = if ng > 0 { gk.len() / (ng * sp_daemon::recall::hd()) } else { 0 };
    let sig = if npos_sig > 0 { app.recall_proj.signature(&gk, ng, npos_sig) } else { [0u64; 4] };
    // B4-SEAL: mint the L5 key so the merged episode is visible to the live selector.
    // QKEYS: question-space key (SP_QKEY_MINT=1) with statement fallback.
    let l5k = mint_ep_l5(app, qm, &toks, text, &dir);
    // NIGHTSHIFT auto-classification (#73): when SP_MEM_CLASSIFY=1, the episode self-governs —
    // classify its mem_class from the text, persist it in the registry row, and set the policy.
    let mem_class: Option<&'static str> = if std::env::var("SP_MEM_CLASSIFY").as_deref() == Ok("1") {
        let mc = sp_daemon::recall::classify_mem_class(text);
        tracing::info!("MEM-CLASSIFY: '{}' -> mem_class={}", text.chars().take(48).collect::<String>(), mc);
        Some(mc)
    } else { None };
    // #72 UNIFY: emit a conformant MEM-OKF concept sidecar (ep.okf.md) for the episode dir.
    if let Some(mc) = mem_class { sp_daemon::recall::write_episode_okf(&dir, text, mc, &name); }
    if std::env::var("SP_NIGHTSHIFT_PERSIST").ok().as_deref() == Some("1") {
        if let Ok(reg_path) = std::env::var("SP_RECALL_REGISTRY") {
            let sig_hex = format!("{:016x}{:016x}{:016x}{:016x}", sig[3], sig[2], sig[1], sig[0]);
            let mut row = serde_json::Map::new();
            row.insert("name".into(), name.clone().into());
            row.insert("dir".into(), dir_str.clone().into());
            row.insert("npos".into(), (ntok as i32).into());
            row.insert("topic".into(), text.into());
            row.insert("text".into(), text.into());
            // MEM-OKF v2 lifecycle lane — the explicit store verb speaks the same schema
            // as remember(), so a fact stored this way can be superseded like any other.
            row.insert("speaker".into(), "user".into());
            row.insert("lifecycle".into(), 0.into());
            row.insert("src".into(), "store verb".into());
            row.insert("sig_bits".into(), sig_hex.into());
            if let Some(mc) = mem_class { row.insert("mem_class".into(), mc.into()); }
            let line = serde_json::Value::Object(row).to_string();
            use std::io::Write as _;
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&reg_path) {
                let _ = writeln!(f, "{line}");
            }
        }
    }
    let topic: String = text.chars().take(40).collect();
    let mut ns = app.nightshift.write().unwrap();
    ns.push(sp_daemon::recall::Episode {
        name, dir: dir_str.clone(), npos: ntok as i32, topic,
        text: text.to_string(), sig, gk, gk_ng: ng, tokens: Some(toks),
        l5key: l5k, // B4-SEAL: minted at capture (mint_live_ep_l5); empty only on mint failure.
        policy: mem_class.map(sp_daemon::recall::MemPolicy::from_class), // #73: self-govern if classified
        lifecycle: 0,
    });
    tracing::info!("LAYER-3 MERGE: captured synthesized episode -> \"{}\"", text.chars().take(60).collect::<String>());
    true
}

// ── P1c PREFIX-SNAPSHOT (HINDSIGHT §4 "fresh-conversation cold prefill") ─────
// A NEW chat shares the constant persona+tools preamble with the committed
// cache but diverges right after it — drop_n blows the rewind bound and the
// whole ~1.6k-token preamble re-prefills cold. Fix: capture the shared prefix
// ONCE through the PROVEN episode machinery (capture_batched = the ADR-009
// batched forward on a scratch cache — never the resident session; replay =
// the B2 §6d-b row restore at [0..P), same-position ⇒ RoPE-aligned), then
// every later cold chat is reset_cold + replay(P) + suffix prefill. This is
// the charter's batch→persist handoff landed at the one seam. Default-OFF
// (SP_PREFIX_SNAPSHOT=1 arms it) until its gate seals.
// Slot policy: ONE snapshot (the daily-driver preamble), recreated when the
// observed shared prefix stops matching. HOST MEMORY ONLY — no on-disk dir
// (2026-08-29: this line used to name <registry-parent>/prefix_snap, a leftover
// from the refuted capture_batched composition; nothing ever wrote it).
// P1c-2 (2026-07-11): the capture_batched composition was REFUTED at preamble
// scale (10+ min @100% GPU, 540 MiB store — receipt gates/G-KAIROS-P1c-1.md).
// Replaced by the gemma4_kv_shear verb: when the SWA ring has never wrapped
// this residency, slots [0..lcp) still hold the preamble rows prefill minted —
// restore is an O(1) position shear, byte-exact by construction, no copies.
const PREFIX_SNAP_MIN: usize = 256;

#[cfg(feature = "wire_cuda_backend")]
fn prefix_snapshot_restore(
    _app: &Arc<AppState>,
    handle: *mut std::ffi::c_void,
    tokens: &[i32],
    lcp: usize,
) -> usize {
    use sp_daemon::cuda_kvdecode_dispatch as kv;
    let t0 = std::time::Instant::now();
    match unsafe { kv::shear(handle, lcp as i32) } {
        Ok(()) => {
            tracing::info!(
                "PREFIX-SHEAR: restored the {}-token shared prefix in {:?} (O(1)); prefill suffix {} (full would be {})",
                lcp, t0.elapsed(), tokens.len() - lcp, tokens.len());
            lcp
        }
        Err(e) => {
            tracing::info!("PREFIX-SHEAR declined ({e}) — full-prefill null floor");
            0
        }
    }
}

/// P1c dead-client guard (observed 2026-07-11): `blocking_send` on a FULL
/// channel whose peer is half-open (client process killed; axum only notices on
/// a TCP write it never gets to make) blocks the turn thread FOREVER — the
/// session wedges while /v1/metrics stays green. A LIVE client drains the
/// 64-event buffer in microseconds, so 30 s of sustained Full = dead peer.
/// Returns false (= abort the turn) on Closed or deadline; never blocks unbounded.
fn send_deadline(
    tx: &tokio::sync::mpsc::Sender<Result<Event, Infallible>>,
    ev: Event,
) -> bool {
    use tokio::sync::mpsc::error::TrySendError;
    let mut item: Result<Event, Infallible> = Ok(ev);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    loop {
        match tx.try_send(item) {
            Ok(()) => return true,
            Err(TrySendError::Closed(_)) => return false,
            Err(TrySendError::Full(v)) => {
                if std::time::Instant::now() >= deadline {
                    tracing::warn!("send_deadline: SSE peer stopped draining for 30s — treating as dead, aborting turn");
                    return false;
                }
                std::thread::sleep(std::time::Duration::from_millis(20));
                item = v;
            }
        }
    }
}

#[cfg(feature = "wire_cuda_backend")]
#[allow(clippy::too_many_arguments)]
fn run_kvdecode_chat(
    app: &Arc<AppState>,
    chat_id: u64,
    tokens: &[i32],
    max_tokens: u32,
    vocab_size: usize,
    logits: &mut [f32],
    dec_buf: &mut TokenDecodeBuffer,
    tx: &tokio::sync::mpsc::Sender<Result<Event, Infallible>>,
    cancel_child: &Arc<AtomicI32>,
    sessions: &crate::sessions::Sessions,
    sampler: &mut crate::sampler::Sampler,
    byteexact: bool,
    replay_dir: Option<String>,
    replay_npos: i32,
    single_entry: bool,
    inject_frames: Option<Vec<Vec<f32>>>,
    inject_ph: i32,
    inject_open: Option<i32>,
    inject_close: Option<i32>,
    auto_recall: bool,
    raw_user: Option<String>,
    orig_msgs: Option<Vec<Message>>,
    eot_bias_req: Option<f32>,
) {
    use sp_daemon::cuda_kvdecode_dispatch as kv;

    let send_err = |msg: String| {
        let _ = tx.blocking_send(Ok(Event::default().data(format!("{{\"error\":\"{msg}\"}}"))));
        let _ = app.events_tx.send(DaemonEvent::Chat { chat_id, status: "cancelled" });
    };

    if tokens.is_empty() {
        send_err("kvdecode: empty prompt after tokenization".into());
        sessions.remove(chat_id);
        return;
    }
    // SP_KV_RESEAM storage (2026-08-24): when the seam normalizes the prompt to the
    // committed spelling, the owned respelling lives HERE — function scope — so the
    // `tokens` view can point at it for everything downstream of the persist block
    // (the prefill, the commit, the TURN-PHASE line). See reseam_join.
    let mut tokens: &[i32] = tokens;
    let reseam_spelling: Vec<i32>;

    // ── TELEPATHY served splice (SP_TELEPATHY_CHAT=1, default-off = byte-identical null floor) ──
    // The cemented two-stage delegate (TELE-12/13/14/15) on the LIVE /v1/chat path: stage 1 routes
    // on the latent, stage 2 hands the CLEAN user text to the qwen coder (CPU L1) and streams its
    // answer into the SSE {delta} stream instead of the Gemma decode. NEVER fuses latent+text.
    // Placed BEFORE the GPU cache guard so a delegated turn never touches the Gemma resident cache.
    // NOTE distinct from SP_TELEPATHY (the one-shot parity verb in main.rs, which early-exits before
    // serving) — the served path uses its own flag. v1: route is SP_ROUTE_FORCE-driven; autonomous
    // feat-route (the TELE-7 head on a NON-COMMITTING capture_feat) is v1.1 (capture_feat is
    // async-armed and would commit the cache).
    // ── SPINE pre-cache route seam (SP_SPINE=1): telepathy expressed as
    //    LatentDecision::Route, routed + executed through spine.rs. Default (spine
    //    off) keeps the proven inline branch below, byte-for-byte. ──
    // P1b legacy_policy: SP_SPINE in-kernel recall = DEAD in production
    // (persona leak, OKFS 1264a862); telepathy = RESEARCH. Both fold to
    // compile-time-false without the legacy_policy feature.
    if cfg!(feature = "legacy_policy")
        && std::env::var("SP_SPINE").as_deref() == Ok("1")
        && std::env::var("SP_TELEPATHY_CHAT").as_deref() == Ok("1") {
        if let Some(user) = raw_user.as_ref().map(|s| s.trim().to_string()).filter(|s| !s.is_empty()) {
            let decision = crate::spine::route_decision(&user);
            if matches!(decision, crate::spine::LatentDecision::Route { .. }) {
                let emit = |t: String| {
                    let payload = serde_json::to_string(&ChatDelta { delta: t, chat_id }).unwrap_or_default();
                    tx.blocking_send(Ok(Event::default().data(payload))).is_ok()
                };
                let _ = crate::spine::execute_route(&decision, &user, emit);
                let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
                let _ = app.events_tx.send(DaemonEvent::Chat { chat_id, status: "done" });
                sessions.remove(chat_id);
                return;
            }
        }
    } else if cfg!(feature = "legacy_policy")
        && std::env::var("SP_TELEPATHY_CHAT").as_deref() == Ok("1") {
        if let Some(user) = raw_user.as_ref().map(|s| s.trim().to_string()).filter(|s| !s.is_empty()) {
            if let crate::telepathy::RouteDecision::Telepathy(bid) = crate::telepathy::decide_route(&[0.0f32]) {
                let marker = std::env::var("SP_TELEPATHY_MARKER").as_deref() != Ok("0");
                let emit = |t: String| {
                    let payload = serde_json::to_string(&ChatDelta { delta: t, chat_id }).unwrap_or_default();
                    tx.blocking_send(Ok(Event::default().data(payload))).is_ok()
                };
                tracing::info!("TELEPATHY: route -> delegate(bridge {bid}); user={:?}", user);
                if marker { let _ = emit("\u{27E6}delegate: qwen2.5-coder\u{27E7}\n".to_string()); }
                match crate::telepathy::delegate_execute(&user, bid) {
                    Ok(ans) => { let _ = emit(ans); }
                    Err(e)  => { let _ = emit(format!("[delegate error: {e}]")); }
                }
                if marker { let _ = emit("\n\u{27E6}/delegate\u{27E7}".to_string()); }
                let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
                let _ = app.events_tx.send(DaemonEvent::Chat { chat_id, status: "done" });
                sessions.remove(chat_id);
                return;
            }
        }
    }

    // Serialize on the resident cache for the whole request (one GPU cache).
    // FOREGROUND: declared before we queue, so background work parks instead of racing us
    // to the lock. Dropped when `_fg` goes out of scope at the end of the request — the
    // counter means "waiting or running", which is what background needs to know.
    let fg = FgIntent::new();
    // WAITING IS A THING THAT HAPPENS TO HER TURNS, SO IT GETS A LOG LINE (2026-08-30).
    // This lock is the last thing a wedged turn ever reaches: today her 07:01 self-turn
    // stopped here for 3h17m and the daemon printed nothing at all, so the operator saw
    // a chip stuck on "warming" and I had to reconstruct the queue from what was
    // MISSING. The note names the holder before we block, and a second line closes the
    // wait so the log carries the duration whether or not the holder ever let go.
    let _t_dev = std::time::Instant::now();
    device_note_wait("chat");
    let guard = match app.cuda_kvdecode_handle.as_ref() {
        Some(m) => m.lock().unwrap_or_else(|p| p.into_inner()),
        None => {
            send_err("kvdecode: handle missing".into());
            sessions.remove(chat_id);
            return;
        }
    };
    let _dev_own = device_note_own("chat");
    if _t_dev.elapsed() > std::time::Duration::from_secs(5) {
        tracing::warn!("DEVICE: chat waited {:.1}s for the device",
                       _t_dev.elapsed().as_secs_f64());
    }
    // DROPPED AT ACQUISITION, exactly as the FG_WAITING doc (879-880) promises
    // (2026-08-29 audit, F-5): the guard kept the counter high for the WHOLE turn, so
    // every background lane spun in yield_to_foreground for a full 522-second turn
    // instead of parking on the mutex the moment we held it. Once the turn OWNS the
    // device lock, backgrounds queue on the lock itself — that is the fair line.
    drop(fg);
    // Armed for the whole turn; forgotten only at the successful commit (F1).
    let commit_void = CommitVoid;
    // Set by any CUDA failure that leaves the cache holding unverifiable rows (F2);
    // a dirty turn commits an EMPTY list, so the next turn cold-prefills honestly.
    let mut cache_dirty = false;
    let handle = guard.0;

    // B1: byte-exact "auditable mode" for this turn. Set under the cache Mutex so
    // the whole decode runs on the exact-integer substrate, then reset to the
    // float null floor on EVERY exit path (RAII guard) — a request that did not
    // ask for it must see the byte-identical Stage-A path. SAFETY: handle live;
    // we hold the cache Mutex (no concurrent decode).
    struct ByteExactGuard(*mut std::ffi::c_void);
    impl Drop for ByteExactGuard {
        fn drop(&mut self) {
            // Reset to float; ignore errors (best-effort cleanup at turn end).
            let _ = unsafe { kv::set_byteexact(self.0, false) };
        }
    }
    let _bx_guard = if byteexact {
        if let Err(e) = unsafe { kv::set_byteexact(handle, true) } {
            send_err(format!("kvdecode set_byteexact(on): {e}"));
            sessions.remove(chat_id);
            return;
        }
        Some(ByteExactGuard(handle))
    } else {
        None
    };

    // Reset the resident cache to dpos=0 so each request is clean.
    // CONTRACT-CHAT-FULLSTACK B2 RING-FIX: use gemma4_kv_reset (counter reset, NO
    // journal replay) instead of rewind(pos). On the SWA ring path rewind(pos)
    // replays the undo-journal in reverse and reads jK/jV[L]+j*kvd for j up to
    // pos-1 — OUT OF BOUNDS past the flat Jmax*kvd journal once pos>Jmax (the
    // diagnosed B2 ring-reset bug). reset() zeroes dpos/commit_pos/jcur; stale
    // ring slots are never read because the next turn rewrites them in position
    // order. Equivalent to rewind(pos) on the non-ring full-cache path (slot==pos,
    // writes to [0,dpos) on the next turn), so no behavior change there.
    // SAFETY: handle is a live sp_g4_kv* owned by AppState; we hold its Mutex.
    let pos = unsafe { kv::position(handle) };
    // PERSISTENT O(1) KV (SP_PERSIST_KV): when the committed cache is a STRICT PREFIX of this
    // prompt AND the cache position equals its length (the cache holds exactly it), reuse it and
    // prefill only the new suffix -- no reset, no re-prefill of the conversation.
    // RECALL EXTENSION: auto_recall is now ALLOWED. The W_c / q·K relevance scoring reads the
    // cache's global K/Q NON-COMMITTINGLY (read_global_q/k roll the cache back), so on a NULL
    // (no-fire) turn the plain-prompt cache stays pristine and persist engages with full speedup.
    // A PICK injects the chosen episode's K/V at [dpos,dpos+npos) for synthesis -> that turn is
    // marked dirty at the commit point (recalled.is_some()) so the NEXT turn full-prefills clean.
    // STILL EXCLUDED (these rebuild / speculatively inject into the cache in ways a token-sequence
    // committed cannot mirror): the SPECULATIVE recall paths (SP_B3_JUDGE / SP_B3_DISPOSER /
    // SP_INT2), the memory-agency writers (SP_DECIDE / SP_FORGET / SP_B4_NIGHTSHIFT), and the
    // operator replay / single_entry / inject_frames seams.
    // Default-ON (G-PERSIST-KV GREEN 2026-06-30: 6-turn byte-identical on==off; TTFT off 7.47x growth
    // vs on flat; engine receipts tests/perf/_persist_gate_{off,on}.json). SP_PERSIST_KV=0 forces the
    // O(n) re-prefill null floor. The cache-mutating paths below still hard-exclude persist (reset/turn).
    // HINDSIGHT 2026-07-10: SP_B4_NIGHTSHIFT no longer needs to kill persist wholesale.
    // The B4 capture path (kv::capture_batched / mint_live_ep_l5) runs through the
    // BATCHED gemma4_decode_cuda forward on the model pointer, NOT the resident
    // sp_g4_kv session — the conversation cache is untouched and session pos is
    // unchanged; anything that DOES move the cache (recall delivery re-prefill,
    // episode injection) is already caught by the pos==cl guard + the recalled-turn
    // committed-clear below (fails closed to a full re-prefill). With B4 excluded,
    // EVERY memory-enabled serve ran O(n) re-prefill per turn (= the 2–5-minute
    // agent turns). Opt-in via SP_PERSIST_B4=1 (unset = the old exclusion =
    // byte-identical null floor) until G-PERSIST-B4 seals it.
    let persist_b4_ok = std::env::var("SP_B4_NIGHTSHIFT").ok().as_deref() != Some("1")
        || std::env::var("SP_PERSIST_B4").ok().as_deref() == Some("1");
    let persist_kv = std::env::var("SP_PERSIST_KV").ok().as_deref() != Some("0")
        && replay_dir.is_none() && !single_entry && inject_frames.is_none()
        && std::env::var("SP_DECIDE").ok().as_deref() != Some("1")
        && std::env::var("SP_FORGET").ok().as_deref() != Some("1")
        && persist_b4_ok
        && std::env::var("SP_B3_JUDGE").ok().filter(|s| !s.is_empty()).is_none()   // F9: same non-empty spelling as the consumers — "SP_B3_JUDGE=2" armed the judge AND persist
        && std::env::var("SP_B3_DISPOSER").is_err()
        && std::env::var("SP_INT2").ok().as_deref() != Some("1");
    let mut prefill_from: usize = 0;
    if persist_kv {
        let committed = KV_COMMITTED.lock().unwrap_or_else(|p| p.into_inner());
        // A committed sequence minted from a different model describes a cache this
        // session does not have (see MODEL_TAG — the shared tokenizer means the token
        // compare below cannot catch it). Treat it as empty: cold prefill, not reuse.
        let tag_ok = {
            let live = model_tag();
            live.is_some() && live.as_deref() == Some(KV_COMMITTED_TAG.lock().unwrap_or_else(|p| p.into_inner()).as_str())
        };
        let cl = if tag_ok { committed.len() } else { 0 };
        // LONGEST-COMMON-PREFIX reuse. The cache holds exactly `committed` (pos == cl). Reuse the
        // common prefix of committed and this prompt, REWINDING the small diverging committed tail.
        // drop==0 is the strict-prefix append (no rewind). A bounded drop (<= REWIND_BOUND, inside
        // the SWA undo-journal SP_G4_KV_JMAX=64) lets a PRE-WARMED persona+tools prefix be reused even
        // though the first real turn diverges right after it (the dummy warm-up tail is rewound).
        // Byte-exact: the reused prefix K/V is the same deterministic compute; the new suffix is
        // prefilled fresh, overwriting the rewound positions.
        //
        // ── THE 32-TOKEN CLIFF (2026-07-13) ──────────────────────────────────────────────
        // This was `const REWIND_BOUND: usize = 32;`. MEASURED consequence: 164 SECONDS to say
        // "Hello! How are you today?" --
        //     !! RE-PREFILL 2679 tok in 163.3s -- the cache was thrown away
        // -- on a prompt whose first 2517 tokens were the SAME PREAMBLE already sitting correct
        // in VRAM. 94% of the work was done. It was binned because the other 6% differed.
        //
        // 32 is not a budget, it is a cliff, and everything real steps off it: a new conversation
        // diverges ~160 tokens, a recall injection ~100-200, an aux/judge call ~1450. The rewind
        // is bounded by the SWA undo-journal (SP_G4_KV_JMAX), so the ONLY reason this was 32 is
        // that the journal defaults to 64 and NOTHING COULD RAISE IT -- serve.py never mapped
        // JMAX at all. A constant chosen to be safe against a default that could not be changed.
        //
        // Now: profile-driven, and it must stay strictly inside the journal or the rewind reads
        // past its bound and CORRUPTS THE KV (the P1c-2 shear regression, learned the hard way:
        // 1-26 char replies for the rest of the residency). So it is clamped to jmax-1 here, at
        // the seam, where it cannot be got wrong by a profile edit.
        let rewind_bound: usize = {
            let jmax: usize = std::env::var("SP_G4_KV_JMAX").ok()
                .and_then(|v| v.trim().parse().ok()).unwrap_or(64);
            let want: usize = std::env::var("SP_KV_REWIND_BOUND").ok()
                .and_then(|v| v.trim().parse().ok()).unwrap_or(32);
            want.min(jmax.saturating_sub(1))
        };
        let REWIND_BOUND = rewind_bound;
        // P1b-2 diagnostics: a silent guard miss = a silent FULL re-prefill
        // (minutes). Name the reason so log forensics never has to guess again.
        if cl >= 1 && pos as usize != cl {
            tracing::info!("PERSIST-KV: guard miss (pos={} != committed {}) — full prefill", pos, cl);
        }
        if cl >= 1 && pos as usize == cl {
            let maxp = cl.min(tokens.len().saturating_sub(1));
            let mut lcp = 0usize;
            while lcp < maxp && tokens[lcp] == committed[lcp] { lcp += 1; }
            let mut drop_n = cl - lcp;
            // ── SP_KV_RESEAM: same text, committed spelling, drop 0 (2026-08-24) ──────
            // The whole story is on `reseam_join` above. Off by default; mapped in
            // serve.py; armed per profile once the live receipt is in hand. Live shape
            // it exists for, measured TONIGHT: warm_suffix 5520 -> 5666 at prefix 8235,
            // 133 SECONDS a turn re-prefilling a suffix that is the same conversation
            // respelled. On success the prompt takes the committed spelling and the
            // strict append below fires; on any disagreement nothing changes.
            if drop_n > 0 && lcp >= 1
                && std::env::var("SP_KV_RESEAM").ok().as_deref() == Some("1")
            {
                let dec = |id: i32| app.tokenizer.decode_token(id).to_vec();
                // ── SKIP THE DAEMON'S OWN DRESSING FIRST (2026-08-25 receipt) ────────
                // With thinking armed every committed turn starts its reply region with
                // `<|channel>thought\n` — control ids only the live generation position
                // carries, which the next prompt structurally cannot spell. Template-
                // owned, closed, ours: skipping it is not guessing. (drop 143 = this
                // dressing + a 140-token reply, twice exactly, on the first live run.)
                let skip = reseam_skip_dressing(&dec, &committed[lcp..cl],
                                                b"<|channel>thought\n");
                // ── THE TEMPLATE-OWNED GAP LIST (2026-08-30, receipt-backed) ─────────
                // Exactly the spans the instrumented refusals showed the committed
                // stream carrying that no prompt can spell: her sampled turn-end and
                // the inter-round generation dressing of a thinking/tool turn. Every
                // entry is something THIS daemon appends around generations and never
                // streams. A dressing not on this list still refuses BY NAME in the
                // log below — extend the list from a receipt, never from a guess.
                const RESEAM_GAPS: &[&[u8]] = &[
                    b"<turn|>\n",
                    b"<|turn>model\n",
                    b"<|channel>thought\n",
                ];
                match reseam_join(&dec, &committed[lcp + skip..cl], &tokens[lcp..],
                                  RESEAM_GAPS) {
                    Ok(j) => {
                        tracing::info!(
                            "PERSIST-KV RESEAM: committed tail [{}..{}) and the prompt \
                             spell the same bytes{} — prompt takes the committed \
                             spelling (drop {} -> 0, suffix {})",
                            lcp, cl,
                            if skip > 0 { " (thought dressing skipped)" } else { "" },
                            drop_n, tokens.len().saturating_sub(lcp + j));
                        let mut nt: Vec<i32> = committed[..cl].to_vec();
                        nt.extend_from_slice(&tokens[lcp + j..]);
                        reseam_spelling = nt;
                        tokens = &reseam_spelling;
                        lcp = cl;
                        drop_n = 0;
                    }
                    Err(why) => {
                        // NAME THE FIRST DISAGREEMENT — AT THE POINT IT DIES (2026-08-30).
                        // The first cut logged forty bytes of each side's HEAD, and on
                        // the 08-25 trial the heads MATCHED while the disagreement sat
                        // deeper: the wrong culprit ("L5 recall augmentation") lived in
                        // a ledger row for an hour. Tonight's drop-350 refusal had
                        // byte-identical heads too. The join itself now reports which
                        // leg refused, the byte offset, and the bytes AROUND that
                        // offset — the heads stay for orientation.
                        let ct: Vec<u8> = committed[lcp + skip..cl.min(lcp + skip + 12)]
                            .iter().flat_map(|&t| dec(t)).collect();
                        let it: Vec<u8> = tokens[lcp..(lcp + 12).min(tokens.len())]
                            .iter().flat_map(|&t| dec(t)).collect();
                        tracing::info!(
                            "PERSIST-KV RESEAM: no byte-aligned seam at lcp {} (drop {}, \
                             dressing_skip {}) — {} | heads: committed: {:?} | incoming: {:?}",
                            lcp, drop_n, skip, why,
                            String::from_utf8_lossy(&ct[..ct.len().min(40)]),
                            String::from_utf8_lossy(&it[..it.len().min(40)]));
                    }
                }
            }
            // THE NUMBER THE ROLLING MARGIN NEEDS (2026-08-24). Not `drop_n`, which is
            // measured against the COMMITTED length; the roll is cut from the PROMPT, so
            // what it must clear is how far back from the prompt's end this turn stopped
            // agreeing. Recorded here because this is the one place it is known.
            LAST_TAIL_GAP.store(tokens.len().saturating_sub(lcp),
                                std::sync::atomic::Ordering::Relaxed);
            // TRUE once some path has decided how this turn starts. It exists so the
            // snapshot below is reachable from BOTH the "drop too big to rewind" case and
            // the "rewind refused by the journal" case — it used to be the `else` of the
            // first test, which made the second case a trapdoor to a cold prefill.
            let mut handled = false;
            if lcp >= 1 && lcp < tokens.len() && drop_n <= REWIND_BOUND {
                // ── THE WEDGED DAEMON (2026-07-29) ───────────────────────────────────────
                // drop_n <= REWIND_BOUND is necessary but NOT sufficient: the rewind also
                // needs the SWA undo-journal to still hold those positions, and the journal
                // is cleared at every commit. So a bounded drop that happens to reach back
                // ACROSS a commit boundary is refused by the engine —
                //     gemma4_kv_rewind: delta crosses a commit (journal cleared)
                // — which is correct (replaying a cleared journal would restore garbage).
                //
                // This used to send_err + remove the session, i.e. kill the turn. MEASURED
                // consequence: the daemon wedged PERMANENTLY. sessions.remove() drops the
                // bookkeeping but NOT the committed KV, so the next prompt recomputed the
                // identical impossible rewind and died the same way — every request after it
                // returned the same error forever. G-COHERENCE went 2/9 on a healthy the dense reference model, and
                // the two probes that passed were simply the ones before the first commit.
                //
                // A refused rewind is not a broken turn, it is a cache miss. Fall through to
                // the full-prefill floor exactly as the two PREFIX-SNAPSHOT failure paths
                // below already do: leave prefill_from at 0 and let the `prefill_from == 0
                // && pos > 0` reset clear the cache. Slower for one turn, never wrong, and
                // it cannot wedge — which is the whole point.
                //
                // ── AND "FALL THROUGH" WAS NOT TRUE (2026-08-04) ─────────────────────────
                // The paragraph above says a refused rewind falls through "exactly as the
                // two PREFIX-SNAPSHOT failure paths below already do". It did not, and it
                // could not: the snapshot was the `else if` of the very test that got us
                // here, so a drop INSIDE the bound that the journal refused fell through to
                // nothing at all — prefill_from stayed 0 and the entire prompt re-prefilled
                // from token 0, never reaching the snapshot that would have handed it ~6400
                // tokens for a memcpy. The intent was written down and the control flow did
                // the opposite.
                //
                // MEASURED by raising jmax 64->832 (which moves drops into this branch):
                //     rewind( 84) refused (delta crosses a commit (journal cleared))
                //     rewind(171) refused    rewind(687) refused
                //     prefill 6528 tok | 6737 | 6929 | 7328   <- full cold prefill, EVERY turn
                // So the bound could not be raised, not because a bigger bound is wrong, but
                // because this branch was a trapdoor. `rewind_bound = 63` was doing the job
                // of keeping drops OUT of it — which is a very strange thing for a system to
                // depend on, and nothing said so anywhere.
                //
                // The snapshot is now reachable from BOTH outcomes (see `handled` below).
                // Nothing else changes: a successful rewind still wins, because reusing
                // `lcp` tokens in place beats restoring a shorter prefix and re-prefilling
                // the difference. This only decides where a REFUSAL lands.
                let rewound = if drop_n > 0 {
                    match unsafe { kv::rewind(handle, drop_n as i32) } {
                        Err(e) => {
                            tracing::warn!(
                                "PERSIST-KV: rewind({drop_n}) refused ({e}) — full-prefill floor");
                            false
                        }
                        _ => true,
                    }
                } else { true };
                if rewound {
                    prefill_from = lcp;
                    handled = true;
                    tracing::info!(
                        "PERSIST-KV: reuse {} of {} committed (drop {}); prefill suffix {} (full would be {})",
                        lcp, cl, drop_n, tokens.len() - lcp, tokens.len());
                } else {
                    tracing::info!(
                        "PERSIST-KV: refused rewind falls through to the snapshot (not a cold prefill)");
                }
            }
            if !handled && lcp >= PREFIX_SNAP_MIN && lcp < tokens.len()
                && std::env::var("SP_PREFIX_SNAPSHOT").as_deref() == Ok("1")
            {
                // ── THE 164-SECOND HELLO (2026-07-13) ────────────────────────────────────
                // drop_n blew the rewind bound, so the old code fell straight to a full cold
                // re-prefill from token 0. MEASURED: 2679 tokens at 60 ms/tok = 164 SECONDS
                // to say "Hello! How are you today?" — and 2517 of those tokens were the SAME
                // PREAMBLE already sitting correct in VRAM. 94% of the work, thrown away
                // because the other 6% differed.
                //
                // The shear (P1c-2) was supposed to be the answer and CANNOT FIRE HERE: it is
                // only legal while the SWA ring has never wrapped, and the preamble (2517) is
                // LONGER than ring_W (2048), so the ring wraps inside the preamble itself.
                // It refuses, correctly, every single time. It was dead code at this config.
                //
                // So: COPY THE BYTES. A snapshot does not reason about wrapping at all — it
                // restores the cache to the exact state it was in. Byte-exact by construction.
                drop(committed);
                let want = lcp.min(tokens.len() - 1);
                // ── WHERE THE PROMPT STOPS MATCHING, AND BY HOW MUCH ────────────────
                // The rolling snapshot was built, was captured correctly and cheaply,
                // and was then NEVER RESTORED FROM — every turn came back "from the base
                // slot". The reason is a number that has never been logged: `lcp`. A
                // strictly-appending transcript has lcp == committed and drop_n == 0; the
                // reuse path would fire and the snapshot machinery would never be reached
                // at all. Reaching it means the harness rewrote something in the MIDDLE
                // of a prompt it had already sent, and no amount of engine-side cleverness
                // helps with that — the fix is upstream or nowhere. Name the number.
                tracing::info!(
                    "PREFIX-MATCH: lcp {} of {} committed (drop {}) — prompt is {} tokens",
                    lcp, cl, cl - lcp, tokens.len());
                let mut snap = PREFIX_SNAP.lock().unwrap_or_else(|p| p.into_inner());

                // (1) FAST PATH — we have a snapshot, it was minted from THIS model, and
                // this prompt starts with it. `tag` is not redundant with the token
                // compare: the two models share a tokenizer, so tokens alone match across
                // them (see MODEL_TAG). An unpublished tag is None and matches nothing.
                let live_tag = model_tag();
                let usable = |s: &PrefixSnap| {
                    live_tag.as_deref() == Some(s.tag.as_str())
                        && s.toks.len() <= want && !s.toks.is_empty()
                        && tokens[..s.toks.len()] == s.toks[..]
                };
                // THE ROLLING SLOT IS TRIED FIRST because it is strictly longer when it
                // matches at all — it was minted by extending this same prefix. When it
                // does not match, `usable` rejects it on the token compare exactly as it
                // rejects a stale preamble, and the base slot below is the floor: the
                // fallback is not a cold prefill, it is last week's behaviour.
                let mut roll = ROLL_SNAP.lock().unwrap_or_else(|p| p.into_inner());
                let use_roll = roll.as_ref().map_or(false, &usable);
                // ── A ROLL THAT NO LONGER MATCHES IS DEAD WEIGHT (2026-08-29 audit,
                // F15): roll_capture only replaces the slot with a LONGER capture, so
                // once a 7k-token roll went stale (new conversation, edited middle) it
                // could never be replaced NOR used — the measured ~536 MiB pinned in
                // host RAM forever, and the rolling optimisation dead for the rest of
                // the residency. Same discipline as the failed-restore path: the slot
                // that missed is dropped, the base slot below stays the floor.
                if !use_roll && roll.is_some() {
                    tracing::info!("ROLL-SNAP: stale against this prompt — dropped (the base slot is the floor)");
                    *roll = None;
                }
                let use_base = !use_roll && snap.as_ref().map_or(false, &usable);
                // The restore runs inside its own scope so the borrow of the slot ENDS
                // before the failure path clears it — a failed restore must be able to
                // drop the very blob it just read.
                let done = {
                    let hit = if use_roll { roll.as_ref() }
                              else if use_base { snap.as_ref() }
                              else { None };
                    hit.map(|s| {
                        let p = s.toks.len();
                        let mib = s.blob.len() as f64 / 1048576.0;
                        let t0 = std::time::Instant::now();
                        // reset first: restore writes the cache and the position machinery whole.
                        let ok = unsafe { kv::reset(handle) }.is_ok()
                            && unsafe { kv::restore_prefix(handle, &s.blob, p as i32) }.is_ok();
                        (p, mib, ok, t0.elapsed())
                    })
                };
                if let Some((p, mib, ok, took)) = done {
                    if ok {
                        prefill_from = p;
                        tracing::info!(
                            "PREFIX-SNAPSHOT: restored {} tokens in {:?} ({:.1} MiB) from the {} slot; \
                             prefill suffix {} (full would be {} = ~{:.0}s)",
                            p, took, mib, if use_roll { "rolling" } else { "base" },
                            tokens.len() - p, tokens.len(), tokens.len() as f64 * 0.060);
                    } else {
                        // NEVER WEDGE ON AN OPTIMISATION. A failed restore falls back to the
                        // full re-prefill — slow, but byte-identical and always correct. The
                        // snapshot is also DROPPED: if it could not be restored into this
                        // session it is not a snapshot of this session, and keeping it around
                        // to fail again every turn would turn one bad turn into a bad hour.
                        // Drop only the slot that FAILED: a bad rolling blob must not cost
                        // the base preamble, which is the floor everything else falls to.
                        tracing::warn!("PREFIX-SNAPSHOT: restore failed ({} slot) — full-prefill floor",
                                       if use_roll { "rolling" } else { "base" });
                        let _ = unsafe { kv::reset(handle) };
                        prefill_from = 0;
                        if use_roll { *roll = None; } else { *snap = None; }
                    }
                } else {
                    // (2) SLOW PATH, ONCE. We are about to pay the full re-prefill anyway, so
                    // pay it in TWO halves and keep the first: prefill [0..want), snapshot it,
                    // then let the normal path prefill the suffix. The next divergence — and
                    // every one after it — is a memcpy.
                    //
                    // The preamble length is LEARNED from the data (the first observed shared
                    // prefix), not configured. Nothing has to be told what the preamble is, and
                    // if it ever changes, the token compare above misses and it is re-learned.
                    let t0 = std::time::Instant::now();
                    let built = (|| -> Result<Vec<u8>, String> {
                        unsafe { kv::reset(handle) }?;
                        // BATCH-PREFILL HERE TOO (2026-07-29). This called kv::prefill
                        // directly and so was the ONE cold prefill in the daemon that
                        // never got the batched forward. MEASURED on the 26B: 3082 tokens
                        // in 206.5 s = 67 ms/tok, against 8.8 ms/tok for the same preamble
                        // on the normal cold path — the capture was ~7.6x slower than the
                        // prefill it exists to make unnecessary, and it is the FIRST thing
                        // a fresh daemon does, so it landed squarely on the operator's
                        // first turn (223 s to say hello). Same preconditions as the normal
                        // path; a decline falls through to per-token, byte-identical.
                        //
                        // The engine DECLINES a batched prefill whenever persist reuse is
                        // armed, and it is right to: a batched-ring prefill does not build
                        // the tick-by-tick journal/commit state a continuation needs, and
                        // G-PK2-BATCHRING measured the consequence — turn 1 fine, turn 2
                        // EMPTY. Which left the capture stuck on the per-token path exactly
                        // when persist (the thing that makes the snapshot worth having) is on.
                        //
                        // So do not fight the guard, sidestep it: set one_shot for the
                        // duration of the capture, because for these few hundred milliseconds
                        // that is literally true — we are building BYTES, not a cache to
                        // continue from. Then reset and restore the blob we just wrote, which
                        // leaves the live cache in the exact state every warm turn already
                        // runs on (restore -> per-token suffix -> decode -> commit). The
                        // batched ring state is never continued from; it is snapshotted and
                        // thrown away.
                        // ── AND THE SIDESTEP WAS THE BUG (2026-08-03) ────────────────────
                        // The premise above — "we are building BYTES, not a cache to continue
                        // from" — is false, and it is false by construction. The RING state is
                        // thrown away, true. The BYTES are the prefix that every warm turn
                        // restores and then appends a per-token suffix to. Being continued from
                        // is the snapshot's entire purpose.
                        //
                        // Why that matters is not "batched is wrong". MEASURED with
                        // /v1/debug/kvdiff_batch: the two prefills run DIFFERENT GEMM kernels —
                        // per-token takes `gemv_w_packed` (int8, SP_CUDA_DECODE_INT8 defaults
                        // on) and batched takes `gemm_w_lift` (float, since SP_KV_PREFILL_DP4A
                        // is unset). Mean relative difference on the live global-K rows: 0.119,
                        // flat across depth (0.12 at the shallowest global layer, no growth) —
                        // the signature of a different quantisation, not of a broken forward.
                        // If anything the batched side is the more accurate of the two.
                        //
                        // The damage is the MIXTURE. Setting one_shot here forced the batched
                        // path past a guard that exists precisely to stop a batched cache being
                        // continued, and produced a cache whose positions [0,want) are minted in
                        // float and whose every later position is minted in int8. Attention then
                        // scores one query against keys built by two different numerics. That is
                        // what her reciting her own system prompt back looked like.
                        //
                        // So the capture mints the prefix with the SAME kernel that will mint
                        // every suffix appended to it. `SP_KV_PREFILL_BATCH` keeps its real job:
                        // genuine one-shot sessions (the judge, the classifier, reflection) that
                        // answer once and are thrown away, where it is correct and 5-7x faster
                        // because nothing is ever appended to them.
                        // ONE SWITCH, ONE MEANING. The first cut of this fix hard-disabled
                        // the batched capture, which was stricter than the problem: the
                        // snapshot round-trip (dump, reset, restore) IS sound for the
                        // position machinery — that is what makes a batched prefill
                        // continuation-safe, and the sibling site at the cold-prefill
                        // decline does exactly the same thing.
                        //
                        // The real hazard is NUMERICS, not position: while the batched
                        // forward disagrees with the per-token one, a prefix minted here and
                        // suffixes appended later are computed by different kernels, and
                        // attention scores across the seam. That is a property of the
                        // FORWARD, not of this call site, and it is already governed by one
                        // switch — `SP_KV_PREFILL_BATCH`, which by policy is armed only when
                        // G-KVDIFF-BATCH shows the two agree (depth amplification <= 3.0x).
                        //
                        // So this site is gated the same way its sibling is, and the policy
                        // lives in one place instead of two.
                        // ── AND "THE HONEST WAY" NEVER RAN (2026-08-30) ──────────────────
                        // The paragraph that stood here said the `set_oneshot` bracket "lied
                        // to a guard" and the round-trip below was "the honest way to satisfy
                        // it". MEASURED on every boot since: the guard is `P==0 && persist &&
                        // !one_shot` — it fires at function ENTRY, before a single byte is
                        // computed, and it cannot see a round-trip that happens AFTER it. So
                        // the batched attempt declined on chunk 0 every time and every capture
                        // paid per-token: 8885 tokens in 675.6 s on 2026-08-30, ~11 of the ~15
                        // boot minutes, while the SAME prefix had just cold-prefilled batched
                        // in 137 s via the chat sibling — which passes the guard with exactly
                        // the set_oneshot bracket this comment forbade. The bracket is not a
                        // lie: for the duration of the batched forward this call IS one-shot
                        // (the ring state is snapshotted and thrown away; nothing ever
                        // continues from it — the reset+restore below is what turns the bytes
                        // into a continuation base). Mirror the sibling.
                        let batched = want > 64
                            && !single_entry
                            && std::env::var("SP_KV_PREFILL_BATCH").as_deref() == Ok("1")
                            && {
                                match unsafe { kv::prefill_batched(handle, &tokens[..want]) } {
                                    Ok(()) => { tracing::info!(
                                        "PREFIX-SNAPSHOT: capture used the batched forward"); true }
                                    Err(e) => {
                                        tracing::info!(
                                            "PREFIX-SNAPSHOT: batched capture declined ({e}) -> one-shot round-trip");
                                        let rt = (|| -> Result<(), String> {
                                            unsafe { kv::reset(handle) }?;
                                            let _ = unsafe { kv::set_oneshot(handle, true) };
                                            let r = unsafe { kv::prefill_batched(handle, &tokens[..want]) };
                                            let _ = unsafe { kv::set_oneshot(handle, false) };
                                            r
                                        })();
                                        match rt {
                                            Ok(()) => { tracing::info!(
                                                "PREFIX-SNAPSHOT: capture used the batched forward (one-shot round-trip)"); true }
                                            Err(e2) => { tracing::info!(
                                                "PREFIX-SNAPSHOT: one-shot batched capture declined ({e2}) -> per-token");
                                                let _ = unsafe { kv::reset(handle) };
                                                false }
                                        }
                                    }
                                }
                            };
                        if !batched {
                            unsafe { kv::prefill(handle, &tokens[..want]) }?;
                        }
                        let n = unsafe { kv::prefix_bytes(handle, want as i32) }?;
                        let mut buf = vec![0u8; n];
                        unsafe { kv::snapshot_prefix(handle, &mut buf, want as i32) }?;
                        if batched {
                            // Discard the batched ring state; continue from the restored
                            // cache, which is the configuration warm turns are proven on.
                            unsafe { kv::reset(handle) }?;
                            unsafe { kv::restore_prefix(handle, &buf, want as i32) }?;
                        }
                        Ok(buf)
                    })();
                    match built {
                        Ok(blob) => {
                            tracing::info!(
                                "PREFIX-SNAPSHOT: captured {} tokens ({:.1} MiB) in {:?} — \
                                 this cold prefill is the LAST one; the next divergence is a memcpy",
                                want, blob.len() as f64 / 1048576.0, t0.elapsed());
                            // Capture only if we know WHICH model produced it. An
                            // untagged snapshot could never be safely reused anyway.
                            match live_tag.clone() {
                                Some(tag) => {
                                    *snap = Some(PrefixSnap {
                                        toks: tokens[..want].to_vec(), blob, tag });
                                }
                                None => tracing::warn!(
                                    "PREFIX-SNAPSHOT: no model tag published — not caching"),
                            }
                            prefill_from = want;
                        }
                        Err(e) => {
                            tracing::warn!("PREFIX-SNAPSHOT: capture failed ({e}) — full-prefill floor");
                            let _ = unsafe { kv::reset(handle) };
                            prefill_from = 0;
                        }
                    }
                }
            }
        }
    }
    if prefill_from == 0 && pos > 0 {
        if let Err(e) = unsafe { kv::reset(handle) } {
            send_err(format!("kvdecode reset: {e}"));
            sessions.remove(chat_id);
            return;
        }
    }

    // Prefill prompt[..n-1] into the resident cache, then decode_step(last) to
    // obtain the first generated token's logits. For a 1-token prompt, skip the
    // prefill and decode_step the lone token directly.
    //
    // B5 (§6e) — the SINGLE LATENT ENTRY POINT. When single_entry is set, the
    // prompt head is ingested through gemma4_kv_inject_tokens (the residual seam:
    // per token, embd[id]*sqrt(E) staged into the inject override → the model mints
    // K/V natively) instead of gemma4_kv_prefill(ids). The residual entering layer 0
    // is bit-identical to prefill by construction, so this is the same ingest
    // through the ONE seam audio + memory also use. The last token still goes
    // through decode_step(last) below (it returns the first generation logits) — for
    // single_entry we route the last token through inject_tokens too (its K/V is
    // minted via the seam) and then decode_step(last) re-runs that position to fetch
    // logits; bit-identical either way since the cache state at position p depends
    // only on tokens[0..=p].
    // TURN PHASE TIMING (operator, 2026-07-12): a 3-token answer took 52 s and no
    // log said where it went. Name every phase: prefill / decode / spectest / capture.
    let t_turn = std::time::Instant::now();
    let mut t_prefill_ms = 0u128;
    let (head, last) = tokens.split_at(tokens.len() - 1);
    // PERSIST-KV: skip the head positions already committed in the reused cache (prefill_from==0
    // on the normal path => the whole head, byte-identical null floor).
    // Keep `full_head` (tokens[0..n-1]) for the warm full-rebatch arm; `head` is the suffix only.
    let full_head = head;
    // F14 (2026-08-29): this clamp is believed unreachable (every producer bounds
    // prefill_from <= tokens.len()-1) — but if it EVER clamps, the cache stands past
    // the assumed position and `head` double-writes occupied slots: a silent
    // corruption. Unreachable-by-analysis gets a tripwire, not silence.
    if prefill_from > full_head.len() {
        tracing::warn!("PERSIST-KV: prefill_from {} > head {} — clamped (this path was                         believed unreachable; the cache may double-write, investigate)",
                       prefill_from, full_head.len());
    }
    let prefill_from = prefill_from.min(full_head.len());
    let head = &full_head[prefill_from..];
    // WHERE TO STOP AND KEEP A COPY. `margin` short of the prompt's end (the tokenizer
    // seam — see the split site), and only when the suffix we are about to prefill is
    // long enough that keeping it beats the memcpy. 0 means "do not split".
    let roll_split: usize = {
        let m = roll_snap_margin();
        // ── PARKED 2026-08-04, OFF BY DEFAULT (opt-IN, was opt-out) ─────────────────
        // It works and it does not help. The capture is correct and cheap — 116-200 ms,
        // measured — and it was never once restored from, because a rolling prefix is
        // worthless when the next prompt does not share it. PREFIX-MATCH, added the same
        // night, is why we know:
        //
        //     PREFIX-MATCH: lcp 6490 of 6571 committed (drop  81) — prompt is 6704 tokens
        //     PREFIX-MATCH: lcp 6681 of 6818 committed (drop 137) — prompt is 6836 tokens
        //     PREFIX-MATCH: lcp 6402 of 7011 committed (drop 609) — prompt is 6911 tokens
        //
        // The gateway's prompt is NOT a strict extension of the one it sent last turn. It
        // diverges inside cache the daemon already holds, by more than REWIND_BOUND (32)
        // every time, so the cheap reuse path cannot fire and the engine falls back to a
        // snapshot minted before the conversation started. THAT is the growing suffix.
        //
        // This was an engine answer to a harness problem. Fix the extension upstream and
        // no rolling slot is needed at all: drop_n goes to 0, the reuse path that has been
        // sitting there all along fires, and the suffix is his new sentence.
        //
        // ── AND THEN IT WORKED (2026-08-04, later the same night) ───────────────────
        // The park above was written before the margin was ever tested. It was wrong about
        // the cause: the roll failed because it captured at the EXACT prompt end while lcp
        // lands 2-3 tokens short of it (the tokenizer merges across the generation tag), so
        // `s.toks.len() <= want` missed by a hair, every time. `roll_snap_margin()` backs
        // off the seam and the split capture makes that position reachable.
        //
        // MEASURED with SP_ROLL_SNAPSHOT=1, speak-ups off so the trace is readable:
        //     restored 6985 tokens from the ROLLING slot; prefill suffix 230  -> 15440 ms
        //     restored 7478 tokens from the ROLLING slot; prefill suffix 232  -> 16105 ms
        //     ROLL-SNAPSHOT: kept 6985 tokens (536.4 MiB) in 168 ms
        // Turns of 27.7 s and 26.9 s, against 93 s and CLIMBING for the same shape of turn
        // without it. The suffix stops growing — that is the whole point, and it is the
        // thing no amount of making the prefill faster can do.
        //
        // STILL OFF BY DEFAULT, and the reason is not doubt about the number. It is that
        // it splits the prefill into two `kv::prefill` calls, which is a change to her live
        // prefill path validated by one clean run of seven turns and nothing else. That is
        // enough to recommend it and not enough to default it.
        //
        // ARMING CONDITION: run it for a real evening (SP_PASSTHROUGH=SP_ROLL_SNAPSHOT
        // SP_ROLL_SNAPSHOT=1) and read the turns, exactly as prefill_batch was armed. Watch
        // for `ROLL-SNAPSHOT: declined`, watch the CUDA fault count, and watch whether she
        // still sounds like herself across a seam that now moves every turn instead of
        // sitting at a fixed 6400. If that holds, map it as a profile knob in serve.py's
        // build_env (it is passthrough-only today) and turn it on there.
        //
        // What survives unconditionally either way is PREFIX-MATCH: the instrument.
        let want_roll = head.len() >= roll_snap_min()
            && std::env::var("SP_PREFIX_SNAPSHOT").as_deref() == Ok("1")
            && std::env::var("SP_ROLL_SNAPSHOT").as_deref() == Ok("1");
        // LEARNED, not configured. `m` is the floor (the tokenizer seam it was
        // originally chosen for); the observed gap is the real requirement, plus a
        // little slack because the next turn's gap is usually a touch larger than the
        // last one's. Bounded by the suffix itself: if backing off that far would leave
        // nothing to keep, keep nothing, which is exactly what a 0 split means.
        let need = LAST_TAIL_GAP.load(std::sync::atomic::Ordering::Relaxed)
            .saturating_add(64)
            .max(m);
        if want_roll && head.len() > need {
            tracing::info!("ROLL-SNAPSHOT: margin {} (floor {}, last tail gap {})",
                           need, m, LAST_TAIL_GAP.load(std::sync::atomic::Ordering::Relaxed));
            head.len() - need
        } else { 0 }
    };
    let t_pf = std::time::Instant::now();
    if !head.is_empty() {
        // #41 BATCH PREFILL (CONTRACT-BATCH-PREFILL, SP_KV_PREFILL_BATCH=1, default-off):
        // on a COLD turn (prefill_from==0 = no persist prefix reused) with a large head,
        // one n-wide batched forward replaces the per-token launch storm (340tok 12-18s /
        // 560tok 71s -> GPU-saturated). FLOAT (not byte-exact) chat speed mode; the C side
        // enforces cold + ring-off + full-cache and ERRORS otherwise, so any precondition
        // miss falls THROUGH to the per-token path (byte-identical null floor). Not for the
        // single_entry seam. Small heads stay per-token (batch alloc overhead not worth it).
        let want_batch = !single_entry
            && prefill_from == 0
            && head.len() > 64
            && std::env::var("SP_KV_PREFILL_BATCH").as_deref() == Ok("1");
        // WARM CONTINUATION BATCH (SP_KV_PREFILL_BATCH_CONT=1, default-off):
        // Design: FULL RE-BATCH (not offset). gemma4_kv_prefill_batched is cold-only
        // (dpos!=0 declines), RoPE is batch-local 0..n, and k_attn only sees the n-token
        // scratch — so a suffix-only batch against a restored prefix is not supportable
        // without kernel surgery. Instead: insurance-snap the live prefix, reset, batch
        // the entire full_head as one-shot, snapshot/reset/restore at full length. Leaves
        // a restored-prefix state (the proven continuation base). Threshold:
        // sp_daemon::batch_cont_prefill::full_rebatch_wins — only when it beats per-token
        // on the measured 13 vs 80 ms/tok arithmetic. Flag off => this arm is dead code.
        let want_batch_cont = sp_daemon::batch_cont_prefill::want_batch_cont(
            single_entry, prefill_from, head.len());
        let batched_ok = if want_batch {
            match unsafe { kv::prefill_batched(handle, head) } {
                Ok(()) => { tracing::info!("BATCH-PREFILL: {} tokens via one batched forward (cold, ring-off)", head.len()); true }
                Err(e) => {
                    // ── THE FIRST TURN AFTER EVERY RESTART (2026-07-29) ──────────────
                    // The engine declines here whenever persist reuse is armed, and it is
                    // right to: a batched-ring prefill lacks the tick-by-tick journal a
                    // CONTINUATION needs (G-PK2-BATCHRING: turn 1 fine, turn 2 EMPTY).
                    // But every daily-driver profile arms persist, so the decline meant the
                    // one genuinely cold turn per daemon lifetime — the operator's FIRST
                    // message — always paid the per-token path: MEASURED 3121 tokens at
                    // 67 ms/tok = 209 SECONDS before she says a word.
                    //
                    // The guard is about the STATE the batch leaves behind, not the KV it
                    // computes. So compute it batched (as one_shot, which for this call is
                    // simply true — nothing continues from the ring state), then round-trip
                    // through the snapshot machinery: dump the bytes, reset, restore. What
                    // remains is a RESTORED cache, which is exactly the state every warm
                    // turn already continues from correctly. Byte-identical contents, a
                    // continuation-safe position machinery, ~8x less time.
                    //
                    // Any failure anywhere in the round trip falls through to the per-token
                    // path below — slower, always correct, never wedged.
                    tracing::info!("BATCH-PREFILL declined ({e}) -> trying the snapshot round-trip");
                    let rt = (|| -> Result<(), String> {
                        let _ = unsafe { kv::set_oneshot(handle, true) };
                        let r = unsafe { kv::prefill_batched(handle, head) };
                        let _ = unsafe { kv::set_oneshot(handle, false) };
                        r?;
                        let n = unsafe { kv::prefix_bytes(handle, head.len() as i32) }?;
                        let mut buf = vec![0u8; n];
                        unsafe { kv::snapshot_prefix(handle, &mut buf, head.len() as i32) }?;
                        unsafe { kv::reset(handle) }?;
                        unsafe { kv::restore_prefix(handle, &buf, head.len() as i32) }?;
                        Ok(())
                    })();
                    match rt {
                        Ok(()) => { tracing::info!(
                            "BATCH-PREFILL: {} tokens batched + snapshot-restored (continuation-safe)",
                            head.len()); true }
                        Err(e2) => { tracing::info!(
                            "BATCH-PREFILL round-trip declined ({e2}) -> per-token prefill");
                            let _ = unsafe { kv::reset(handle) };
                            false }
                    }
                }
            }
        } else if want_batch_cont {
            // Preferred: SUFFIX BATCH (SP_KV_PREFILL_BATCH_SUFFIX=1, default-off) — batch
            // ONLY the n suffix tokens at absolute positions [P, P+n) against the live
            // prefix (gemma4_kv_prefill_batched_from), then the same continuation-safe
            // snapshot round-trip. Pays BATCH_MS*n instead of BATCH_MS*(P+n).
            // Fallback: full re-batch of full_head → snapshot → restore (the proven arm).
            // Any failure restores the insurance-snapped prefix (so the per-token suffix
            // path still has a correct base) or, if that also fails, per-token prefills
            // the entire full_head from cold. Never leaves a poisoned cache for decode.
            let t0 = std::time::Instant::now();
            let total = full_head.len();
            let total_i = total as i32;
            let pf_i = prefill_from as i32;
            let rt = (|| -> Result<&'static str, String> {
                // (1) Insurance snap of the live warm prefix BEFORE anything mutates it.
                //     The suffix batch needs it as much as the reset does: a failure
                //     mid-batch has already sunk suffix rows over SWA ring slots the
                //     prefix tail was standing in.
                let n_pref = unsafe { kv::prefix_bytes(handle, pf_i) }?;
                let mut pref_buf = vec![0u8; n_pref];
                unsafe { kv::snapshot_prefix(handle, &mut pref_buf, pf_i) }?;

                // (2a) Suffix batch in place + continuation-safe round-trip at full length.
                let mut mode: Option<&'static str> = None;
                if sp_daemon::batch_cont_prefill::suffix_flag_on() {
                    let sb = (|| -> Result<(), String> {
                        // DECOMPOSED TIMING, deliberately: the first live firings came
                        // back at a blended 21 ms/tok and two points could not separate
                        // batch compute from the snapshot dance — and that split is
                        // exactly what decides whether the trigger threshold can drop
                        // below full_rebatch_min_suffix (the widening's arming
                        // condition, OFF-BY-DEFAULT §8). So the arm measures itself.
                        let tb = std::time::Instant::now();
                        // ── AND THE BATCHED ARM STOPPED KEEPING THE ROLL (2026-08-24) ──
                        // The roll capture lives in the `!batched_ok` per-token branch
                        // below, and its own comment says "prefill is per-token here BY
                        // CONSTRUCTION, so the split costs nothing". That was true when it
                        // was written and stopped being true the day this suffix arm
                        // landed above it: the batched path now succeeds on every warm
                        // turn, so the capture simply never ran again.
                        //
                        // MEASURED. Last `ROLL-SNAPSHOT: kept` in var/daemon.log is
                        // 2026-08-22 08:48. Five daemon restarts since, not one capture.
                        // What that cost, read off the same log on 2026-08-23:
                        //
                        //     restored 7793 (base); suffix  387 ->  9.0 s
                        //     restored 7793 (base); suffix 1375 -> 25.2 s
                        //     restored 7793 (base); suffix 2555 -> 47.0 s
                        //
                        // Every turn restored the SAME base snapshot and re-prefilled the
                        // whole conversation on top of it — including the ~2000 tokens the
                        // previous turn had computed thirty seconds earlier. `lcp` was
                        // 9,596 of 10,029 the whole time: the cache already held all but
                        // ~430 tokens of what was being recomputed.
                        //
                        // So split the batch the way the per-token path splits, and keep
                        // the copy in between. Two `prefill_batched_from` calls over the
                        // same tokens in the same order is not a new idea: the CUDA layer
                        // already loops `gemma4_kv_prefill_batched_from` over chunks
                        // internally, so a caller-side split is the same shape one level
                        // up. `roll_capture` reads the cache and never writes it, and is
                        // non-fatal by construction — a failed keep costs the copy, not
                        // the turn.
                        if roll_split > 0 && roll_split < head.len() {
                            let (a, b) = head.split_at(roll_split);
                            unsafe { kv::prefill_batched_from(handle, a, pf_i) }?;
                            roll_capture(handle, &full_head[..prefill_from + roll_split]);
                            unsafe {
                                kv::prefill_batched_from(handle, b, pf_i + a.len() as i32)
                            }?;
                        } else {
                            unsafe { kv::prefill_batched_from(handle, head, pf_i) }?;
                        }
                        let batch_ms = tb.elapsed().as_millis();
                        let ts = std::time::Instant::now();
                        let n = unsafe { kv::prefix_bytes(handle, total_i) }?;
                        let mut buf = vec![0u8; n];
                        unsafe { kv::snapshot_prefix(handle, &mut buf, total_i) }?;
                        unsafe { kv::reset(handle) }?;
                        unsafe { kv::restore_prefix(handle, &buf, total_i) }?;
                        tracing::info!(
                            "BATCH-PREFILL-CONT timing: batch {} ms ({:.1} ms/suffix-tok) + \
                             snap-dance {} ms at total {}",
                            batch_ms,
                            if head.is_empty() { 0.0 } else { batch_ms as f64 / head.len() as f64 },
                            ts.elapsed().as_millis(), total);
                        Ok(())
                    })();
                    match sb {
                        Ok(()) => mode = Some("suffix-batch"),
                        Err(e) => tracing::info!(
                            "BATCH-PREFILL-CONT suffix arm declined ({e}) -> full re-batch"),
                        // A half-run suffix batch may have dirtied the ring; the full
                        // re-batch below starts with reset, so no cleanup is needed here.
                    }
                }

                // (2b) Cold full re-batch as one-shot + continuation-safe restore.
                // WIDENED TRIGGER GUARD (2026-08-20): with the suffix arm the dispatch
                // fires from suffix > 64, far below the full-rebatch break-even — so a
                // declined suffix batch may NOT fall into a full re-batch that costs
                // more than the per-token suffix it was trying to beat. Below the
                // break-even the decline goes straight to per-token.
                let batch_res = if let Some(m) = mode { Ok(m) } else if
                    !sp_daemon::batch_cont_prefill::full_rebatch_wins(prefill_from, head.len()) {
                    Err("suffix arm declined below the full-rebatch break-even".to_string())
                } else {
                    (|| -> Result<&'static str, String> {
                        let _ = unsafe { kv::set_oneshot(handle, true) };
                        unsafe { kv::reset(handle) }?;
                        let r = unsafe { kv::prefill_batched(handle, full_head) };
                        let _ = unsafe { kv::set_oneshot(handle, false) };
                        r?;
                        let n = unsafe { kv::prefix_bytes(handle, total_i) }?;
                        let mut buf = vec![0u8; n];
                        unsafe { kv::snapshot_prefix(handle, &mut buf, total_i) }?;
                        unsafe { kv::reset(handle) }?;
                        unsafe { kv::restore_prefix(handle, &buf, total_i) }?;
                        Ok("full-rebatch")
                    })()
                };

                match batch_res {
                    Ok(m) => Ok(m),
                    Err(e) => {
                        // Put the warm prefix back so the per-token suffix path is correct.
                        let _ = unsafe { kv::set_oneshot(handle, false) };
                        let _ = unsafe { kv::reset(handle) };
                        match unsafe { kv::restore_prefix(handle, &pref_buf, pf_i) } {
                            Ok(()) => Err(e),
                            Err(e2) => {
                                // Last resort: re-prefill the whole head per-token from cold.
                                // Marks success so the outer arm does not double-prefill the suffix.
                                tracing::info!(
                                    "BATCH-PREFILL-CONT: prefix restore failed after decline \
                                     ({e2}); per-token full_head recovery");
                                unsafe { kv::prefill(handle, full_head) }?;
                                Ok("pertok-recovery")
                            }
                        }
                    }
                }
            })();
            match rt {
                Ok(mode) => {
                    let ms = t0.elapsed().as_millis();
                    // ms/tok is per BATCHED token: the suffix arm batches head.len(),
                    // the full re-batch batches all of full_head.
                    let batched_tok = if mode == "suffix-batch" { head.len() } else { total };
                    tracing::info!(
                        "BATCH-PREFILL-CONT: {} tokens batched + snapshot-restored \
                         (mode={}, warm_suffix={}, prefix={}) in {} ms ({:.1} ms/tok)",
                        batched_tok, mode, head.len(), prefill_from, ms,
                        if batched_tok > 0 { ms as f64 / batched_tok as f64 } else { 0.0 });
                    true
                }
                Err(e) => {
                    tracing::info!(
                        "BATCH-PREFILL-CONT declined ({e}) -> per-token suffix prefill \
                         ({} tok, prefix {} kept)",
                        head.len(), prefill_from);
                    false
                }
            }
        } else { false };
        if !batched_ok {
            let r = if single_entry {
                unsafe { kv::inject_tokens(handle, head) }
            } else if roll_split > 0 && roll_split < head.len() {
                // ── THE ROLL CAPTURE HAPPENS *DURING* PREFILL, NOT AFTER IT ─────────
                // `snapshot_prefix` refuses unless dpos == P ("snapshot a STATE, not a
                // guess"), and it is right to: the cache does not carry the bytes for a
                // position it has moved past. So a snapshot that must stop short of the
                // prompt end has to be taken while the cache is standing there — which
                // means splitting the prefill, not shortening the snapshot.
                //
                // Two calls instead of one, over the same tokens, in the same order, with
                // a memcpy in between. Prefill is per-token here by construction, so the
                // split costs nothing: position n does not care whether position n-1
                // arrived in the same call.
                let (a, b) = head.split_at(roll_split);
                unsafe { kv::prefill(handle, a) }
                    .and_then(|_| { roll_capture(handle, &full_head[..prefill_from + roll_split]);
                                    unsafe { kv::prefill(handle, b) } })
            } else {
                unsafe { kv::prefill(handle, head) }
            };
            if let Err(e) = r {
                send_err(format!("kvdecode {} head: {e}", if single_entry { "inject_tokens" } else { "prefill" }));
                sessions.remove(chat_id);
                return;
            }
        }
    }
    t_prefill_ms = t_pf.elapsed().as_millis();
    if !head.is_empty() {
        tracing::info!("TURN-PHASE: prefill {} tok in {} ms ({:.0} ms/tok)",
                       head.len(), t_prefill_ms, t_prefill_ms as f64 / head.len() as f64);
    }
    // G-PERF (2026-07-12): `other` was 4132 ms/turn with the heads on vs 54 ms with
    // them off -- LARGER than the decode it wraps -- and it was one anonymous bucket.
    // Everything between here and decode_t0 is the RECALL stage: L5 search, the B3
    // judge (which runs its own forward passes that never appear in the token count,
    // which is why short recall turns reported an absurd "0.8 tok/s"), replay, ablate.
    // Name it, and count the forwards it burns. We cannot cut what we have not named.
    let t_recall0 = std::time::Instant::now();
    let steps_before_recall = kv::step_count();
    // G-INT-2-FIX (text-in-context recall): the synthesis tail token. Defaults to
    // last[0] (the original prompt's last token). When the B3-JUDGE PICKs a memory
    // it rebuilds the cache from an AUGMENTED prompt (memory text prepended to the
    // user query, RAG-style) and overrides syn_last with that augmented prompt's
    // last token, so the synthesis decode_step(syn_last) below starts from the
    // text-in-context reconstruction. NULL / judge-off leaves syn_last == last[0]
    // (byte-identical null floor).
    let mut syn_last: i32 = last[0];
    // ATTR-GATE zero-inference decline: when set (by the SP_RECALL_ATTR_GATE branch on
    // attribute-absence), the synthesis forward is SKIPPED and this fixed string is
    // streamed instead — the gemma4 decode loop never runs (see the seam before
    // decode_step(syn_last)). Default None => normal synthesis (null floor preserved).
    let mut symbolic_decline: Option<String> = None;
    // B5 (§6e) — the GENERIC residual-frame channel. After the prompt scaffold is
    // ingested, inject any raw residual frames (audio/memory source) at consecutive
    // positions through the same seam (gemma4_kv_inject_seq via inject_frames). Each
    // inner Vec is one E-length frame; a malformed (ragged) batch is rejected.
    if let Some(frames) = inject_frames.as_ref() {
        if !frames.is_empty() {
            let e_dim = frames[0].len();
            if e_dim == 0 || frames.iter().any(|f| f.len() != e_dim) {
                send_err("kvdecode inject_frames: ragged/empty frames (each must be E floats)".into());
                sessions.remove(chat_id);
                return;
            }
            // ── FRAMING (2026-07-31) ──────────────────────────────────────────
            // The channel is generic; the PROTOCOL is not. Audio happens to need
            // no markers around it, so this seam shipped without any — and images
            // do: Gemma wraps an image block in <start_of_image> (255999) and
            // <end_of_image> (258882), and without them the model reports it was
            // shown no picture (measured, live, at both 196 and 280 tokens).
            //
            // The markers CANNOT come from the caller's messages, because frames
            // are appended after the ENTIRE prompt — there is no text position
            // after them. So they are minted here, through the same residual seam
            // (gemma4_kv_inject_tokens: embd[id]*sqrt(E)), which is exactly what
            // the prompt head already uses.
            //
            // Opt-in via inject_open/inject_close so the audio path is untouched:
            // absent = today's behaviour, byte for byte.
            if let Some(open_tok) = inject_open {
                if let Err(e) = unsafe { kv::inject_tokens(handle, &[open_tok]) } {
                    send_err(format!("kvdecode inject_open({open_tok}): {e}"));
                    sessions.remove(chat_id);
                    return;
                }
            }
            let n_frames = frames.len() as i32;
            let flat: Vec<f32> = frames.iter().flatten().copied().collect();
            if let Err(e) = unsafe { kv::inject_frames(handle, &flat, n_frames, inject_ph) } {
                send_err(format!("kvdecode inject_frames(n={n_frames}, ph={inject_ph}): {e}"));
                sessions.remove(chat_id);
                return;
            }
            if let Some(close_tok) = inject_close {
                if let Err(e) = unsafe { kv::inject_tokens(handle, &[close_tok]) } {
                    send_err(format!("kvdecode inject_close({close_tok}): {e}"));
                    sessions.remove(chat_id);
                    return;
                }
            }
        }
    }
    // B3: AUTONOMOUS MEMORY RECALL. With auto_recall on, NO explicit `replay`,
    // and a loaded registry: compute THIS turn's 256-bit C2 query signature from
    // the resident cache's GLOBAL-layer K (now holding the prompt's prefilled
    // positions), Hamming-match it against the registry, and auto-replay the best
    // episode iff it clears TAU_BITS. This is the headline "it remembers on its
    // own" path: the model self-selects the relevant memory with no operator
    // intervention. The match score is ALWAYS logged (the foreign-reject safety
    // leg needs the below-TAU case to be visible). dpos here == head.len() (the
    // prompt positions); the chosen episode is injected at [dpos,dpos+npos) just
    // like B2, so the last prompt token + every generated token attend over it.
    // B3-v2: the q·K ATTENTION-RELEVANCE selector replaces the v1 centroid-Hamming
    // sig (which did not separate question→passage: right episode argmax only 1/5,
    // episodes mutually agree ~200/256). v2 scores each episode by the model's NATIVE
    // attention relevance — q·K, where q is THIS query's last-token global-layer query
    // (gemma4_kv_read_global_q, the SAME resident path the decode runs on) and K is the
    // episode's stored global-K (ep.k). Ranks on the top-m-mean relevance; fires the
    // argmax iff it clears SP_B3_TAU_QK (a relevance threshold set from the MEASURED
    // separation — no guessing). Score reported either way (the foreign-reject leg
    // needs the below-TAU case visible). `recalled` holds (name, topm*1000 as u32).
    let mut recalled: Option<(String, u32)> = None;
    // SPECTEST v1 (2026-07-03, operator idea "byte-exact+rewind speculative test"):
    // the delivered fact TEXT when a text-in-context delivery fires this turn —
    // the ground the whole draft answer is tested against before ANY byte streams.
    let mut recalled_text: Option<String> = None;
    // LM-B2b turn-outcome telemetry: stash the delivered recall's query/class/secret so the
    // [DONE] site can emit a `turn` record (output + timing + obeyed) joined by query to the
    // decision record. Set only when a text-in-context delivery fires (else stays None).
    let mut recalled_query: Option<String> = None;
    let mut recalled_class: Option<String> = None;
    let mut recalled_secret: bool = false;
    // G-INT-2: when SP_INT2 Stage-2 has rendered an ACCEPT/NULL verdict, the legacy
    // SP_B3_WC / SP_B3_DISPOSER / q·K fire branches MUST NOT double-fire on the same turn.
    let mut int2_decided = false;
    // B3-JUDGE: when the generative judge fires, stash (ep_name, injected_text)
    // for the post-synthesis CPU-side grounding check (parametric-hallucination flag).
    let mut judge_ground: Option<(String, String)> = None;
    // ── LAYER-2: FORGET (SP_FORGET=1) — the organism removes a memory ──────────────
    // The building block for memory agency. On a forget intent ("forget ...") the best
    // token-overlap match across the curated registry + live nightshift is dropped from
    // the live set AND rewritten out of the persisted registry (all exact-duplicate
    // copies), then confirmed via text-in-context. First operator-triggered; the same
    // removal the model will later invoke itself. Default-off = byte-identical null floor.
    let mut forget_done = false;
    if auto_recall && replay_dir.is_none()
        && std::env::var("SP_FORGET").ok().as_deref() == Some("1")
    {
        let is_forget = raw_user.as_ref().map(|s| {
            let l = s.to_lowercase();
            l.contains("forget") || l.contains("delete that") || l.contains("erase")
        }).unwrap_or(false);
        if is_forget {
            use sp_daemon::recall;
            let ftext = raw_user.as_ref().map(|s| s.trim().to_string()).unwrap_or_default();
            let q = ftext.to_lowercase().replace("forget", " ").replace("please", " ").replace("erase", " ");
            let mut best: Option<(f32, String)> = None; // (overlap, episode text)
            if let Some(reg) = app.recall_registry.as_ref() {
                for ep in reg.iter() {
                    let ov = recall::token_overlap(&q, &ep.text);
                    if best.as_ref().map_or(true, |(b, _)| ov > *b) { best = Some((ov, ep.text.clone())); }
                }
            }
            {
                let ns = app.nightshift.read().unwrap();
                for ep in ns.iter() {
                    let ov = recall::token_overlap(&q, &ep.text);
                    if best.as_ref().map_or(true, |(b, _)| ov > *b) { best = Some((ov, ep.text.clone())); }
                }
            }
            if let Some((ov, text)) = best.filter(|(ov, _)| *ov >= 0.25) {
                let sp_life = std::env::var("SP_MEM_LIFECYCLE").ok().as_deref() == Some("1");
                // (1) drop (A1: or mark superseded) from the live nightshift set (all exact copies)
                { let mut ns = app.nightshift.write().unwrap();
                  if sp_life { for e in ns.iter_mut() { if e.text == text { e.lifecycle = 1; } } }
                  else { ns.retain(|e| e.text != text); } }
                // (2) rewrite the persisted registry, dropping (A1: or marking) every line with this text
                if let Ok(reg_path) = std::env::var("SP_RECALL_REGISTRY") {
                    if sp_life { let _ = recall::mark_superseded_registry(&reg_path, &text, 1); }
                    else {
                    if let Ok(content) = std::fs::read_to_string(&reg_path) {
                        let kept: Vec<&str> = content.lines().filter(|line| {
                            match serde_json::from_str::<serde_json::Value>(line) {
                                Ok(v) => v.get("text").and_then(|x| x.as_str()) != Some(text.as_str()),
                                Err(_) => !line.trim().is_empty(),
                            }
                        }).collect();
                        let mut out = kept.join("\n");
                        if !out.is_empty() { out.push('\n'); }
                        if let Err(e) = write_atomic(std::path::Path::new(&reg_path), out.as_bytes()) {
                            tracing::warn!("registry rewrite failed (kept the old file): {e}");
                        }
                    }
                    }
                }
                tracing::info!("LAYER-2 FORGET: removed memory (overlap {:.3}) -> \"{}\"", ov, text);
                // (3) confirm via text-in-context (the judge-PICK synthesis machinery)
                let aug = format!(
                    "Context: at the user's request you have just permanently removed this memory: \"{}\".\n\nUser: {}\n\nIn one short sentence, confirm that you have forgotten it.",
                    text, ftext);
                let aug_msgs = vec![Message { role: "user".to_string(), content: aug }];
                if let Ok(aug_toks) = app.tokenizer.apply_template_ids(&aug_msgs) {
                    if aug_toks.len() >= 2 {
                        let _ = unsafe { kv::reset_cold(handle) };
                        let (aug_head, aug_last) = aug_toks.split_at(aug_toks.len() - 1);
                        if aug_head.is_empty() || unsafe { kv::prefill(handle, aug_head) }.is_ok() {
                            syn_last = aug_last[0];
                        }
                    }
                }
                forget_done = true;
            } else {
                tracing::info!("LAYER-2 FORGET: forget intent but no memory matched (best overlap < 0.25)");
            }
        }
    }
    // ─────────── LAYER-2 STORE (SP_MEM_STORE=1) — the explicit memory-agency verb ──────────
    // Hodor-session live finding: "store in your memory X" grew a raw episode
    // SILENTLY while the model replied "I don't know how to store memories" — the
    // system had the ability and denied it. Deterministic intent detect → capture
    // the PAYLOAD as a live episode (capture_live_episode: batched K capture + C2
    // sig + L5-key mint + persist), attributed for perspective → confirm
    // SYMBOLICALLY at the synthesis seam (zero decode ⇒ the confirmation can never
    // be confabulated). Also sidesteps the anaphora gap: teaching happens in one
    // self-contained sentence. Default-off = byte-identical null floor.
    if !forget_done && symbolic_decline.is_none()
        && std::env::var("SP_MEM_STORE").ok().as_deref() == Some("1")
    {
        if let Some(ruser) = raw_user.as_ref() {
            let rt = ruser.trim();
            let rl = rt.to_lowercase();
            const STORE_PREFIXES: &[&str] = &[
                "store in your memory that", "store in your memory",
                "remember that", "remember this:", "remember:",
                "add to your memory that", "add to your memory", "note that"];
            if let Some(pref) = STORE_PREFIXES.iter().find(|p| rl.starts_with(**p)) {
                // ASCII prefixes ⇒ byte-length-safe slice of the original casing.
                let payload = rt[pref.len()..]
                    .trim_matches(|c: char| c == ':' || c == ',' || c.is_whitespace())
                    .to_string();
                if payload.split_whitespace().count() >= 2 {
                    // MEM-OKF v2: no "The user said: " prefix — the owner lives in `speaker`.
                    if capture_live_episode(app, &payload) {
                        symbolic_decline = Some(format!("Stored to memory: {payload}"));
                        tracing::info!("LAYER-2 STORE: explicit verb -> captured + symbolic confirm ({})",
                            payload.chars().take(60).collect::<String>());
                    } else {
                        symbolic_decline = Some(
                            "I tried to store that, but the capture failed — check the daemon log.".to_string());
                        tracing::warn!("LAYER-2 STORE: capture_live_episode FAILED for the store verb");
                    }
                }
            }
        }
    }
    // ─────────── ADR-002 Decide→Execute SPINE (SP_SPINE=1) ───────────
    // The unified recall path made literal (papers/PPT-LAT-ADR-002): one immutable
    // LatentView → a priority-folded chain of Deciders → a discrete LatentDecision →
    // the Executor. This reproduces the LIVE one-config stack (L5 cosine recall →
    // attribute-gate zero-inference decline, QONLY-aware) in ~30 lines that used to be
    // a ~1500-line env-branch ladder. Default-off (SP_SPINE unset) ⇒ the inline
    // `else if` below runs byte-for-byte unchanged (the null floor).
    // P1b legacy_policy: SP_SPINE in-kernel recall = DEAD (persona leak on
    // foreign Qs, OKFS 1264a862). The proven inline L5 lane below is untouched.
    let spine_active = cfg!(feature = "legacy_policy")
        && std::env::var("SP_SPINE").as_deref() == Ok("1");
    if spine_active && auto_recall && replay_dir.is_none() && !forget_done
        && app.recall_registry.is_some()
        && std::env::var("SP_RECALL_L5").as_deref() == Ok("1")
        && unsafe { kv::position(handle) } > 0
    {
        use sp_daemon::recall;
        // G-PERF: the recall stage costs 2.7-4.2 s/turn while running exactly ONE
        // forward. So it is CPU/disk, not inference. Name its sub-blocks.
        let rp_t0 = std::time::Instant::now();
        let ruser = raw_user.clone().unwrap_or_default();
        let n_global = recall::n_global();
        let mut ql = vec![0.0f32; n_global * recall::g_nh() * recall::hd()];
        let read_ok = unsafe { kv::read_global_q(handle, last[0], &mut ql) }.is_ok();
        let l5_query = if read_ok { recall::l5_query_embed(&ql) } else { Vec::new() };
        let global_q = if read_ok { ql } else { Vec::new() };
        let rp_query_ms = rp_t0.elapsed().as_millis();
        let rp_t1 = std::time::Instant::now();
        // Any graduated latent head joins the fold: load the W_c recall head if deployed
        // (SP_B3_WC), and build_pipeline runs it FIRST (a fired head short-circuits cosine).
        let wc = std::env::var("SP_B3_WC").ok().filter(|s| !s.is_empty())
            .filter(|_| cfg!(feature = "legacy_policy"))  // W_c head = RESEARCH
            .and_then(|p| recall::load_wc(&p));
        // SUSPECT #1: this DEEP-CLONES every nightshift episode on EVERY turn, and each
        // Episode carries `gk` = [n_global][npos][HD] f32 (~0.5 MB). Measured, not assumed.
        let ns_snapshot: Vec<recall::Episode> =
            app.nightshift.read().unwrap().iter().cloned().collect();
        let rp_snap_ms = rp_t1.elapsed().as_millis();
        let rp_snap_mb: f64 = ns_snapshot.iter()
            .map(|e| (e.gk.len() * 4 + e.text.len()) as f64).sum::<f64>() / 1.048_576e6;
        let rp_t2 = std::time::Instant::now();
        let registry = app.recall_registry.as_ref().unwrap();
        let framing = crate::spine::Delivery::from_env(
            std::env::var("SP_RECALL_L5_PROMPT").as_deref().unwrap_or("systemecho"));
        let view = crate::spine::LatentView {
            raw_user: &ruser,
            global_q,
            l5_query,
            registry,
            nightshift: &ns_snapshot,
            interrogative: recall::is_interrogative(&ruser),
            qonly: std::env::var("SP_RECALL_QONLY").as_deref() == Ok("1"),
            tau_l5: std::env::var("SP_RECALL_L5_TAU").ok().and_then(|s| s.parse().ok()).unwrap_or(0.30),
            tau_margin: std::env::var("SP_RECALL_L5_MARGIN").ok().and_then(|s| s.parse().ok()).unwrap_or(0.0),
            attr_tau: std::env::var("SP_RECALL_ATTR_TAU").ok().and_then(|s| s.parse().ok()).unwrap_or(0.5),
            attr_gate: std::env::var("SP_RECALL_ATTR_GATE").as_deref() == Ok("1"),
            framing,
        };
        let decision = crate::spine::decide(&view, &crate::spine::build_pipeline(wc, framing));
        let rp_decide_ms = rp_t2.elapsed().as_millis();
        let rp_t3 = std::time::Instant::now();
        let ctx = crate::spine::ExecCtx {
            handle,
            tokenizer: app.tokenizer.as_ref(),
            orig_msgs: orig_msgs.as_deref(),
            raw_user: &ruser,
            last_tok: last[0],
        };
        let outcome = unsafe { crate::spine::execute(decision, &ctx) };
        if outcome.recalled.is_some() { recalled = outcome.recalled; }
        if outcome.symbolic_decline.is_some() { symbolic_decline = outcome.symbolic_decline; }
        syn_last = outcome.syn_last;
        tracing::info!(
            "RECALL-PHASE: query {} ms + snapshot {} ms ({} eps, {:.1} MB deep-cloned) \
             + decide {} ms + execute {} ms",
            rp_query_ms, rp_snap_ms, ns_snapshot.len(), rp_snap_mb,
            rp_decide_ms, rp_t3.elapsed().as_millis());
    // P1b-2 (rehomed console, 2026-07-11): the in-kernel L5 DELIVERY lane is
    // legacy. The harness (spine executors, gateway authority) is the ONE
    // recall authority; the console reaches it via :8800 and falls back to
    // daemon-direct with auto_recall:false when the gateway is down (verbs
    // only — degraded but honest). legacy_policy keeps the lane compiled for
    // the default build (rollback + byte-identical today-behavior); the
    // kernel build (-legacy_policy) folds it to false ⇒ verbs only.
    } else if cfg!(feature = "legacy_policy")
        && auto_recall && replay_dir.is_none() && !forget_done {
        if let Some(registry) = app.recall_registry.as_ref() {
            let npos_q = unsafe { kv::position(handle) }; // = head.len() after prefill
            if npos_q > 0 {
                use sp_daemon::recall;
                // q·K relevance top-m (a few strongest matched positions). Tunable via env.
                let topm: usize = std::env::var("SP_B3_QK_TOPM").ok()
                    .and_then(|s| s.parse().ok()).unwrap_or(8);
                // The relevance threshold on the top-m-mean; default +inf so a first
                // telemetry run (env unset) NEVER fires — we read the matrix, then set
                // SP_B3_TAU_QK from the measured target/foreign gap.
                let tau_qk: f32 = std::env::var("SP_B3_TAU_QK").ok()
                    .and_then(|s| s.parse().ok()).unwrap_or(f32::INFINITY);
                // Read the live query's last-token global-layer Q (one non-committing
                // forward of the last prompt token; the cache is rolled back after).
                let n_global = recall::n_global();
                let mut qbuf = vec![0.0f32; n_global * recall::g_nh() * recall::hd()];
                // G-DENSE-SERVE §C (2026-07-03): the B3-v2 q·K scan below is a full
                // read_global_q forward + an O(episodes × npos) relevance loop + a giant
                // per-episode log line, run EVERY recall turn. When SP_B3_TAU_QK is unset it
                // defaults to +inf ⇒ `fire = score >= tau_qk` is ALWAYS false ⇒ the scan can
                // NEVER recall — pure dead telemetry. qbuf itself is only consumed downstream
                // by SP_B3_QDUMP / SP_INT2 / SP_B3_WC / SP_B3_JUDGE. In the one-config
                // faithful stack NONE of those are set, so on a 500-position conversation this
                // was ~half the recall-turn cost (turn-4 86.9s). When nothing can use it, skip
                // BOTH the forward and the scan loop; `best` stays None (the 2514 fire-arm
                // handles None cleanly) and the L5 stage (its OWN read_global_q at ~1524) is
                // untouched. This is byte-identical whenever any consumer IS set (scan runs).
                // P1b legacy_policy: every consumer of the q·K scan (C2-sig dead
                // at TAU=+inf; QDUMP/INT2/W_c/judge = RESEARCH) is a legacy lane.
                let qk_scan_needed = cfg!(feature = "legacy_policy") && (tau_qk.is_finite()
                    || std::env::var("SP_B3_QDUMP").is_ok()
                    || std::env::var("SP_INT2").ok().as_deref() == Some("1")
                    || std::env::var("SP_B3_WC").ok().filter(|s| !s.is_empty()).is_some()
                    || std::env::var("SP_B3_JUDGE").ok().filter(|s| !s.is_empty()).is_some());
                let read_q_res = if qk_scan_needed {
                    unsafe { kv::read_global_q(handle, last[0], &mut qbuf) }
                } else {
                    Ok(0)
                };
                match read_q_res {
                    Ok(_ng) => {
                        // B3-v3 dataset: SP_B3_QDUMP=<dir> persists THIS turn's last-token
                        // global-Q (the exact vector qk_relevance scores) so the offline
                        // contrastive trainer mines (Q, episode-K) pairs from the substrate.
                        // Additive; unset ⇒ no-op (null floor preserved). File q_<chat_id>.bin.
                        if let Some(dir) = std::env::var("SP_B3_QDUMP").ok()
                            .filter(|_| cfg!(feature = "legacy_policy")) {  // P1b: B3-v3 dataset rail = RESEARCH
                            let qd = recall::g_nh() * recall::hd();
                            let mut buf = Vec::with_capacity(8 + qbuf.len() * 4);
                            buf.extend_from_slice(&(n_global as u32).to_le_bytes());
                            buf.extend_from_slice(&(qd as u32).to_le_bytes());
                            for &x in &qbuf { buf.extend_from_slice(&x.to_le_bytes()); }
                            let _ = std::fs::create_dir_all(&dir);
                            let _ = std::fs::write(
                                std::path::Path::new(&dir).join(format!("q_{chat_id}.bin")), buf);
                        }
                        // Score every episode (max + top-m-mean), log the full matrix.
                        // G-DENSE-SERVE §C: skipped entirely when the result cannot be used
                        // (see qk_scan_needed above) — `best` stays None, which the 2514
                        // fire-arm treats as REJECT (no replay), the correct dead-scan outcome.
                        let mut best: Option<(usize, f32)> = None;
                        if qk_scan_needed {
                            let mut rows: Vec<String> = Vec::with_capacity(registry.len());
                            for (i, ep) in registry.iter().enumerate() {
                                let np = (ep.npos as usize).min(
                                    if ep.gk_ng > 0 { ep.gk.len() / (ep.gk_ng * recall::hd()) } else { 0 });
                                let (mx, tm) = recall::qk_relevance(&qbuf, &ep.gk, ep.gk_ng, np, topm);
                                rows.push(format!("{}(max={:.3},topm={:.3})", ep.name, mx, tm));
                                let key = tm;
                                match best { Some((_, b)) if key <= b => {}, _ => best = Some((i, key)) }
                            }
                            tracing::info!("B3-v2 q·K relevance (npos_q={} topm={}): [{}]", npos_q, topm, rows.join(" "));
                        } else {
                            tracing::info!("B3-v2 q·K scan SKIPPED (TAU_QK=+inf, no QDUMP/INT2/WC/JUDGE consumer) — dead telemetry elided, one forward + registry scan saved this turn");
                        }
                        // ===== G-INT-2 STEP 2: STAGE-1 C2-HAMMING CULL (TELEMETRY-ONLY, SP_INT2=1) =====
                        // Compute the LIVE query's C2 sig from the prompt's global-K, build the bounded
                        // candidate set (curated registry ∪ most-recent-W nightshift episodes), rank by
                        // Hamming distance (R_BITS - agree) ascending, take top-K, and LOG the survivors.
                        // This branch is READ-ONLY on the cache (read_global_k is non-mutating) and does
                        // NOT inject / replay / decide NULL — it is the Stage-1 cull plumbing only. Default-off
                        // (env unset) = byte-identical null floor; leaves SP_B3_WC behavior intact when unset.
                        if cfg!(feature = "legacy_policy")  // P1b: INT2 head = RESEARCH (carried unarmed)
                            && std::env::var("SP_INT2").ok().as_deref() == Some("1") {
                            let w_horizon: usize = std::env::var("SP_INT2_W").ok()
                                .and_then(|s| s.parse().ok()).unwrap_or(20);   // KAIROS most-recent-W stub
                            let k_keep: usize = std::env::var("SP_INT2_K").ok()
                                .and_then(|s| s.parse().ok()).unwrap_or(20);   // cull budget
                            // (1) live query C2 sig from the prompt's global-K [n_global][npos_q][HD].
                            let mut qkbuf = vec![0.0f32; n_global * recall::g_kvd() * (npos_q as usize)];
                            match unsafe { kv::read_global_k(handle, &mut qkbuf, npos_q) } {
                                Ok(ngk) => {
                                    let qsig = app.recall_proj.signature(&qkbuf, ngk as usize, npos_q as usize);
                                    // (2) bounded candidate union = curated registry ∪ last-W nightshift.
                                    let ns_guard = app.nightshift.read().unwrap();
                                    let ns_len = ns_guard.len();
                                    let ns_skip = ns_len.saturating_sub(w_horizon);   // keep most-recent W live
                                    let n_static = registry.len();
                                    let cands: Vec<(bool, &recall::Episode)> = registry.iter()
                                        .map(|e| (false, e))
                                        .chain(ns_guard.iter().skip(ns_skip).map(|e| (true, e)))
                                        .collect();
                                    // (3) Hamming distance (R_BITS - agree); sort ascending; top-K.
                                    let mut scored: Vec<(usize, u32, bool, &recall::Episode)> = cands.iter()
                                        .enumerate()
                                        .map(|(i, &(live, ep))| {
                                            let d = recall::R_BITS as u32 - recall::agree(&qsig, &ep.sig);
                                            (i, d, live, ep)
                                        })
                                        .collect();
                                    scored.sort_by_key(|&(_, d, _, _)| d);
                                    let survivors: Vec<String> = scored.iter().take(k_keep)
                                        .map(|&(_, d, live, ep)| format!("{}{}:{}", ep.name, if live { "(L)" } else { "(C)" }, d))
                                        .collect();
                                    tracing::info!(
                                        "B-INT2 Stage-1 cull (W={} K={}): qsig={:016x}.. cull_pool={} (curated={} live={}) survivors=[{}]",
                                        w_horizon, k_keep, qsig[0], cands.len(), n_static, ns_len.min(w_horizon),
                                        survivors.join(", "));
                                }
                                Err(e) => tracing::warn!("B-INT2 Stage-1: read_global_k(npos_q={}) failed: {e} -- cull skipped", npos_q),
                            }
                            // ===== G-INT-2 STEP 3: STAGE-2 LIVE TEACHER-FORCED CAUSAL ABLATION GATE =====
                            // Promote the offline disposer (SP_B3_DISPOSER==2) to the LIVE Stage-2
                            // gatekeeper. Phase-A proved bounded-N makes the leaky C2 cull unnecessary,
                            // so we DO NOT Hamming-cull on the hot path: ablate ALL W bounded candidates
                            // directly. Candidate set = the last SP_INT2_W episodes of (curated registry
                            // ∪ nightshift) that have a RESOLVABLE secret (ep.secret + ep.tok on disk);
                            // live nightshift episodes lacking these are SKIPPED (the Phase-B extension).
                            // Per candidate (reusing the disposer FFI sequence EXACTLY, in the live turn's
                            // context — the query prompt is already in the cache at dpos==head.len()):
                            //   replay E -> teacher-force ITS ep.secret -> rewind(ng) -> ablate src rows
                            //   -> re-score -> rewind(ng+npos)  ⇒ collapse=ΣΔLL, dpos nets back to anchor.
                            // best=argmin collapse; collapse < SP_INT2_TAU (default -8.0) ⇒ ACCEPT (one
                            // replay into the live cache so the real decode attends the memory); else NULL
                            // (cache nets to anchor, byte-identical to a no-recall turn). SP_INT2 unset =
                            // this whole branch is skipped = byte-identical null floor.
                            {
                                let int2_tau: f32 = std::env::var("SP_INT2_TAU").ok()
                                    .and_then(|s| s.parse().ok()).unwrap_or(-8.0);
                                let lse = |z: &[f32]| -> f32 {
                                    let m = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                                    let mut s = 0.0f32; for &v in z { s += (v - m).exp(); } m + s.ln()
                                };
                                let anchor = unsafe { kv::position(handle) };
                                // SWA-ring Jmax hazard: assert npos+ng <= Jmax per candidate (full-cache
                                // served path is latent, but assert it — G-INT-3 ring turns it on).
                                let jmax: i32 = std::env::var("SP_G4_KV_JMAX").ok()
                                    .and_then(|s| s.parse().ok()).unwrap_or(64);
                                // Bounded candidate union = curated registry ∪ last-W nightshift, then
                                // keep only those with a resolvable secret. (We re-acquire the read lock;
                                // the Stage-1 cull above already dropped its guard.)
                                let ns_guard = app.nightshift.read().unwrap();
                                let ns_len = ns_guard.len();
                                let ns_skip = ns_len.saturating_sub(w_horizon);
                                let cands: Vec<(bool, &recall::Episode)> = registry.iter()
                                    .map(|e| (false, e))
                                    .chain(ns_guard.iter().skip(ns_skip).map(|e| (true, e)))
                                    .collect();
                                let mut best_abl: Option<(usize, f32, String)> = None;
                                let mut drows: Vec<String> = Vec::with_capacity(cands.len());
                                let mut n_probed = 0usize;
                                let mut n_skipped = 0usize;
                                let mut probe_steps = 0usize;
                                for (i, &(live, ep)) in cands.iter().enumerate() {
                                    if ep.npos <= 0 || ep.gk.is_empty() { n_skipped += 1; continue; }
                                    // Resolvable secret? ep.secret sidecar = teacher-force target;
                                    // ep.tok = ablation source rows. SKIP candidates lacking either
                                    // (live nightshift episodes without sidecars — Phase-B extension).
                                    let secret_raw = std::fs::read_to_string(
                                        std::path::Path::new(&ep.dir).join("ep.secret")).ok();
                                    let eptok: Vec<i32> = std::fs::read_to_string(
                                        std::path::Path::new(&ep.dir).join("ep.tok"))
                                        .ok().map(|s| s.lines().filter_map(|l| l.trim().parse::<i32>().ok()).collect())
                                        .unwrap_or_default();
                                    let secret_ids: Vec<i32> = match secret_raw {
                                        Some(s) => {
                                            let s = s.trim_end_matches(|c: char| c == '\n' || c == '\r').to_string();
                                            let mut v = app.tokenizer.encode(&s).unwrap_or_default();
                                            if v.first() == Some(&2) { v.remove(0); }
                                            v
                                        }
                                        None => Vec::new(),
                                    };
                                    if secret_ids.is_empty() || eptok.is_empty() {
                                        n_skipped += 1;
                                        if live {
                                            tracing::info!("B-INT2 Stage-2: SKIP '{}' (live, no resolvable secret/tok sidecar)", ep.name);
                                        }
                                        continue;
                                    }
                                    // Jmax assert (hazard §B.1): one candidate's uncommitted span = npos + ng.
                                    let ng_max = secret_ids.len() as i32;
                                    if ep.npos + ng_max > jmax {
                                        tracing::warn!("B-INT2 Stage-2: '{}' npos({})+ng({}) > Jmax({}) -- raising SP_G4_KV_JMAX advised; skipping to preserve pristineness", ep.name, ep.npos, ng_max, jmax);
                                        n_skipped += 1; continue;
                                    }
                                    n_probed += 1;
                                    // Leg 1: inject E, teacher-force the KNOWN secret, record lp_E.
                                    if unsafe { kv::replay(handle, &ep.dir, ep.npos, false) }.is_err() { continue; }
                                    let mut gen: Vec<i32> = Vec::new();
                                    let mut lpe: Vec<f32> = Vec::new();
                                    let mut tok = last[0];
                                    for &s in &secret_ids {
                                        if unsafe { kv::decode_step(handle, tok, logits) }.is_err() { break; }
                                        lpe.push(logits[s as usize] - lse(logits)); gen.push(s); tok = s;
                                    }
                                    let ng = gen.len();
                                    probe_steps += ng + ep.npos as usize;
                                    let _ = unsafe { kv::rewind(handle, ng as i32) };   // undo payload, keep E
                                    if ng == 0 {
                                        drows.push(format!("{}(collapse=nan)", ep.name));
                                        let _ = unsafe { kv::rewind(handle, ep.npos) };   // shear E -> back to anchor
                                        debug_assert_eq!(unsafe { kv::position(handle) }, anchor);
                                        continue;
                                    }
                                    // Source rows: episode positions whose token matches a secret token.
                                    let mut targets: Vec<i32> = Vec::new();
                                    let want: std::collections::HashSet<i32> = gen.iter().copied().collect();
                                    for (p, &t) in eptok.iter().enumerate() {
                                        if p >= ep.npos as usize { break; }
                                        if want.contains(&t) { targets.push(p as i32); }
                                    }
                                    if targets.len() > 12 { targets.truncate(12); }
                                    // Leg 2: ablate src rows, teacher-force the SAME secret, record lp_abl.
                                    let _ = unsafe { kv::ablate(handle, anchor, &targets) };
                                    let mut lpa: Vec<f32> = Vec::with_capacity(ng);
                                    let mut tok = last[0];
                                    for i2 in 0..ng {
                                        if unsafe { kv::decode_step(handle, tok, logits) }.is_err() { break; }
                                        lpa.push(logits[gen[i2] as usize] - lse(logits)); tok = gen[i2];
                                    }
                                    let _ = unsafe { kv::rewind(handle, lpa.len() as i32 + ep.npos) };   // clear payload + episode (restores ablated rows)
                                    // CRITICAL (Phase-A §B): net dpos delta per candidate == 0.
                                    debug_assert_eq!(unsafe { kv::position(handle) }, anchor,
                                        "B-INT2 Stage-2: candidate '{}' did not rewind to anchor", ep.name);
                                    let n = lpe.len().min(lpa.len());
                                    let mut collapse = 0.0f32;
                                    for j in 0..n { collapse += lpa[j] - lpe[j]; }
                                    drows.push(format!("{}{}(collapse={:.2},ntgt={})", ep.name, if live { "(L)" } else { "(C)" }, collapse, targets.len()));
                                    let better = match best_abl { None => true, Some((_, b, _)) => collapse < b };
                                    if better { best_abl = Some((i, collapse, ep.name.clone())); }
                                }
                                tracing::info!(
                                    "B-INT2 Stage-2 ablation (W={} probed={} skipped={} probe_steps={}): collapse=ΣΔLL (more-neg=load-bearing) [{}] TAU={:.3}",
                                    w_horizon, n_probed, n_skipped, probe_steps, drows.join(" "), int2_tau);
                                // The absolute thermodynamic gate: best=argmin collapse.
                                match best_abl {
                                    Some((idx, collapse, name)) if collapse < int2_tau => {
                                        let ep = cands[idx].1;
                                        // Final dpos MUST be at anchor before the accept replay.
                                        debug_assert_eq!(unsafe { kv::position(handle) }, anchor);
                                        match unsafe { kv::replay(handle, &ep.dir, ep.npos, false) } {
                                            Ok(_) => {
                                                tracing::info!("B-INT2 ACCEPT '{}' ΔLL={:.3} < τ={:.3} -> replay@M_target (live decode attends memory)", name, collapse, int2_tau);
                                                recalled = Some((name.clone(), ((-collapse).max(0.0) * 1000.0) as u32));
                                            }
                                            Err(e) => tracing::warn!("B-INT2 ACCEPT '{}': replay({}, {}) failed: {e} -- proceeding clean", name, ep.dir, ep.npos),
                                        }
                                        int2_decided = true;
                                    }
                                    Some((_, collapse, name)) => {
                                        tracing::info!("B-INT2 NULL (best '{}' ΔLL={:.3} >= τ={:.3}) -> clean prompt (null floor)", name, collapse, int2_tau);
                                        int2_decided = true;
                                    }
                                    None => {
                                        tracing::info!("B-INT2 NULL (no resolvable candidate in W={}) -> clean prompt (null floor)", w_horizon);
                                        int2_decided = true;
                                    }
                                }
                            }
                        }
                        // ===== B3-WC DEPLOY: learned W_c head, logsumexp-mean + (E+1) NULL argmax =====
                        // SP_B3_WC=<wc_deploy.bin> => the autonomous instance selector decides recall.
                        // Score every episode by wc lse-mean (the metric the head trained on, int16-exact),
                        // append the s0 NULL slot, argmax over [episodes, NULL]. Episode wins => replay it
                        // (M_target/SP_REPLAY_MTARGET=42 clamps injection mass); NULL wins => clean prompt.
                        // Default-off (env unset) = null floor; run WITHOUT SP_B3_DISPOSER/SP_B3_TAU_QK.
                        if let Some(wcp) = (if int2_decided || !cfg!(feature = "legacy_policy") { None } else { std::env::var("SP_B3_WC").ok().filter(|s| !s.is_empty()) }) {  // P1b: W_c = superseded
                            match recall::load_wc(&wcp) {
                                Some(head) => {
                                    // B4 NIGHTSHIFT: score the static curated registry AND a snapshot of
                                    // the LIVE consolidated episodes (static first, then nightshift). The
                                    // (E+1)-NULL argmax is over [all_candidates, NULL=s0]; a live episode
                                    // can beat NULL + every curated episode and fire its own recall.
                                    let ns_guard = app.nightshift.read().unwrap();
                                    let cands: Vec<&recall::Episode> =
                                        registry.iter().chain(ns_guard.iter()).collect();
                                    let n_static = registry.len();
                                    let (mut bi, mut bv) = (usize::MAX, f32::NEG_INFINITY);
                                    let mut wrows: Vec<String> = Vec::with_capacity(cands.len());
                                    for (i, ep) in cands.iter().enumerate() {
                                        let np = (ep.npos as usize).min(
                                            if ep.gk_ng > 0 { ep.gk.len() / (ep.gk_ng * recall::hd()) } else { 0 });
                                        let sc = recall::wc_score(&qbuf, &ep.gk, ep.gk_ng, np, &head);
                                        wrows.push(format!("{}={:.3}", ep.name, sc));
                                        if sc > bv { bv = sc; bi = i; }
                                    }
                                    tracing::info!("B3-WC lse-mean (E+1)-argmax s0={:.3} ({} curated + {} live): [{}]",
                                        head.s0, n_static, cands.len() - n_static, wrows.join(" "));
                                    if bi == usize::MAX || !(bv > head.s0) {
                                        tracing::info!("B3-WC: NULL wins (best={:.3} <= s0={:.3}) -> REJECT (clean prompt)", bv, head.s0);
                                    } else {
                                        let ep = cands[bi];
                                        if let Some(ref toks) = ep.tokens {
                                            // B4 live episode: recall by re-injecting its raw tokens through the
                                            // B5 seam (bit-identical to prefill) — no ep.k/ep.v files on disk.
                                            // G-INT-2-FIX: attenuated inject (M_target=42) so the
                                            // recalled memory BINDS instead of hijacking synthesis.
                                            tracing::info!("B3-WC: RECALL '{}' (LIVE/nightshift) score={:.3} > s0={:.3} -> inject_tokens_atten(n={})",
                                                ep.name, bv, head.s0, toks.len());
                                            if let Err(e) = unsafe { kv::inject_tokens_atten(handle, toks) } {
                                                tracing::warn!("B3-WC: inject_tokens_atten('{}', n={}) failed: {e} -- clean prompt", ep.name, toks.len());
                                            } else {
                                                recalled = Some((ep.name.clone(), (bv.max(0.0) * 1000.0) as u32));
                                            }
                                        } else if std::env::var("SP_B3_WC_TEXT").as_deref() == Ok("1") && raw_user.is_some() {
                                            // F2 FAITHFULNESS: TEXT-IN-CONTEXT delivery for fact recall. F1b.1 proved
                                            // pure-KV replay yields 0% obedience on fact-conflict (the attenuated K/V can't
                                            // override a strong parametric prior), while F1 proved an in-context fact gets
                                            // 100% obedience. So on a confident W_c match, deliver the episode TEXT as
                                            // authoritative context and rebuild the cache from the augmented prompt (the
                                            // proven FORGET-synthesis machinery), instead of kv::replay. Default-off (env
                                            // unset) keeps the pure-KV replay null floor (valid for novel needles).
                                            let ruser = raw_user.as_deref().unwrap_or("");
                                            let aug = format!(
                                                "Context (authoritative, from your memory): {}\n\nUsing the context above, answer faithfully even if it differs from what you already know:\n{}",
                                                ep.text, ruser);
                                            let aug_msgs = vec![Message { role: "user".to_string(), content: aug }];
                                            match app.tokenizer.apply_template_ids(&aug_msgs) {
                                                Ok(aug_toks) if aug_toks.len() >= 2 => {
                                                    let _ = unsafe { kv::reset_cold(handle) };
                                                    let (aug_head, aug_last) = aug_toks.split_at(aug_toks.len() - 1);
                                                    if aug_head.is_empty() || unsafe { kv::prefill(handle, aug_head) }.is_ok() {
                                                        syn_last = aug_last[0];
                                                        recalled = Some((ep.name.clone(), (bv.max(0.0) * 1000.0) as u32));
                                                        tracing::info!("B3-WC: RECALL '{}' (curated) score={:.3} > s0={:.3} -> TEXT-IN-CONTEXT synthesis (F2)", ep.name, bv, head.s0);
                                                    } else {
                                                        tracing::warn!("B3-WC TEXT: prefill(aug) failed -- clean prompt");
                                                    }
                                                }
                                                _ => tracing::warn!("B3-WC TEXT: apply_template_ids(aug) failed -- clean prompt"),
                                            }
                                        } else {
                                            // Curated episode: the existing disk-replay path (M_target unchanged). Pure-KV
                                            // null floor; F1b.1: 0% obey on fact-conflict, valid only for novel needles.
                                            tracing::info!("B3-WC: RECALL '{}' (curated) score={:.3} > s0={:.3} -> replay@M_target", ep.name, bv, head.s0);
                                            if let Err(e) = unsafe { kv::replay(handle, &ep.dir, ep.npos, false) } {
                                                tracing::warn!("B3-WC: replay({}, {}) failed: {e} -- clean prompt", ep.dir, ep.npos);
                                            } else {
                                                recalled = Some((ep.name.clone(), (bv.max(0.0) * 1000.0) as u32));
                                            }
                                        }
                                    }
                                }
                                None => tracing::warn!("B3-WC: SP_B3_WC set but load_wc({}) failed -- skipping", wcp),
                            }
                        }
                        // ===== F2b: TOKEN-OVERLAP (Jaccard) selection for NATURAL FACTS =====
                        // W_c is geometric — great for high-entropy novel needles, blind to mutually-
                        // similar natural-language facts (F2: it picked query-INDEPENDENT episodes, 0/15).
                        // Token-overlap of the QUERY vs each episode TEXT discriminates natural facts
                        // (recall::token_overlap, the production Jaccard verifier); on a confident overlap,
                        // deliver the episode TEXT in-context (the F1=100% path), NOT pure-KV replay.
                        // Default-off (SP_RECALL_JACCARD unset) = null floor; runs only if no prior stage decided.
                        if cfg!(feature = "legacy_policy")  // P1b: F2b jaccard stage = RESEARCH
                            && !int2_decided && recalled.is_none()
                            && std::env::var("SP_RECALL_JACCARD").as_deref() == Ok("1")
                        {
                            if let Some(ruser) = raw_user.as_ref().map(|s| s.trim().to_string()).filter(|s| !s.is_empty()) {
                                let tau_ov: f32 = std::env::var("SP_RECALL_JACCARD_TAU").ok()
                                    .and_then(|s| s.parse().ok()).unwrap_or(0.15);
                                let q = ruser.to_lowercase();
                                let (mut bov, mut bname, mut btext) = (0.0f32, String::new(), String::new());
                                let (mut bgk, mut bgkng): (Vec<f32>, usize) = (Vec::new(), 0);
                                {
                                    let ns_guard = app.nightshift.read().unwrap();
                                    for ep in registry.iter().chain(ns_guard.iter()) {
                                        let ov = recall::token_overlap(&q, &ep.text);
                                        if ov > bov { bov = ov; bname = ep.name.clone(); btext = ep.text.clone(); bgk = ep.gk.clone(); bgkng = ep.gk_ng; }
                                    }
                                }
                                if bov >= tau_ov && !btext.is_empty() {
                                    let aug = format!("Context (authoritative, current): {}\n\n{}", btext, ruser);
                                    let aug_msgs = vec![
                                        Message { role: "system".to_string(), content: "You are Shannon-Prime, a local AI with a real working memory. Keep replies short. Use facts you were given faithfully; if you don't know, say so.".to_string() },
                                        Message { role: "user".to_string(), content: aug },
                                    ];
                                    match app.tokenizer.apply_template_ids(&aug_msgs) {
                                        Ok(aug_toks) if aug_toks.len() >= 2 => {
                                            let _ = unsafe { kv::reset_cold(handle) };
                                            let (aug_head, aug_last) = aug_toks.split_at(aug_toks.len() - 1);
                                            if aug_head.is_empty() || unsafe { kv::prefill(handle, aug_head) }.is_ok() {
                                                syn_last = aug_last[0];
                                                recalled = Some((bname.clone(), (bov * 1000.0) as u32));
                                                tracing::info!("RECALL-JACCARD: '{}' overlap={:.3} >= tau={:.3} -> TEXT-IN-CONTEXT (F2b)", bname, bov, tau_ov);
                                                // LN-F3a selector-data label: pair q_<chat_id>.bin (query global-Q, the
                                                // SP_B3_QDUMP rail) with the Jaccard-selected episode. Capture-only.
                                                if let Ok(qd) = std::env::var("SP_B3_QDUMP") {
                                                    let _ = std::fs::write(
                                                        std::path::Path::new(&qd).join(format!("lbl_{chat_id}.txt")),
                                                        format!("{}\t{:.4}\n", bname, bov));
                                                    // LN-1 clean K-dump: the selected episode's IN-MEMORY global-K
                                                    // [gk_ng][npos][HD] — bypasses the opaque on-disk ep.k layout.
                                                    // Mirrors q_<chat_id>.bin: <u32 ng><u32 npos><f32 gk>.
                                                    if bgkng > 0 && !bgk.is_empty() {
                                                        let npos_k = bgk.len() / (bgkng * recall::hd());
                                                        let mut kb = Vec::with_capacity(8 + bgk.len() * 4);
                                                        kb.extend_from_slice(&(bgkng as u32).to_le_bytes());
                                                        kb.extend_from_slice(&(npos_k as u32).to_le_bytes());
                                                        for &x in &bgk { kb.extend_from_slice(&x.to_le_bytes()); }
                                                        let _ = std::fs::write(
                                                            std::path::Path::new(&qd).join(format!("k_{chat_id}.bin")), kb);
                                                    }
                                                }
                                            } else { tracing::warn!("RECALL-JACCARD: prefill(aug) failed -- clean prompt"); }
                                        }
                                        _ => tracing::warn!("RECALL-JACCARD: apply_template_ids(aug) failed -- clean prompt"),
                                    }
                                } else {
                                    tracing::info!("RECALL-JACCARD: best='{}' overlap={:.3} < tau={:.3} -> no recall (clean prompt)", bname, bov, tau_ov);
                                }
                            }
                        }
                        // ===== L5 RECALL (SP_RECALL_L5) — query-to-query paraphrase recall =====
                        // The fact signal is layer-localized in global layer 5 (G-REP-LAYER-L5:
                        // L5 exact->paraphrase recall@1 = 85.2% / all-layer-avg 11.5%; L5-cosine
                        // query-key = 100% exact / 88.5% paraphrase vs Jaccard 100%/8.2%). Match the
                        // live query's L5 embedding (l5_query_embed of read_global_q, mean-heads +
                        // L2-norm) against each episode's stored L5 query-key (ep.l5) by cosine; on a
                        // confident match deliver the episode TEXT in-context (the same F2b delivery
                        // as the Jaccard branch). GATED after Jaccard (skips if a prior stage already
                        // recalled). Default-off (SP_RECALL_L5 unset) = byte-identical null floor.
                        if !int2_decided && recalled.is_none()
                            && std::env::var("SP_RECALL_L5").as_deref() == Ok("1")
                        {
                            if let Some(ruser) = raw_user.as_ref().map(|s| s.trim().to_string()).filter(|s| !s.is_empty()) {
                                // SP_RECALL_QONLY (G-ONECONFIG-LIVE run-1 lever, RUNBOOK-ONE-CONFIG §8):
                                // conversational STATEMENTS skip the L5 stage entirely — the in-registry
                                // cosine background is ≥0.9, so a non-query turn otherwise gets an
                                // irrelevant fact injected. Deterministic, no forward; unset = null floor.
                                let qonly_skip = std::env::var("SP_RECALL_QONLY").as_deref() == Ok("1")
                                    && !recall::is_interrogative(&ruser);
                                if qonly_skip {
                                    tracing::info!("RECALL-L5: QONLY-SKIP (non-interrogative turn) -> clean prompt");
                                }
                                let tau_l5: f32 = std::env::var("SP_RECALL_L5_TAU").ok()
                                    .and_then(|s| s.parse().ok()).unwrap_or(0.30);
                                // SP_RECALL_L5_MARGIN (run-1 lever): absolute cosine cannot separate the
                                // right episode from the in-registry background (both ≥0.9); the top1−top2
                                // GAP can. Gates DELIVERY only — the attr-gate decline still fires (the
                                // SNE shield must not be starved). 0.0/unset = off = null floor.
                                let tau_margin: f32 = std::env::var("SP_RECALL_L5_MARGIN").ok()
                                    .and_then(|s| s.parse().ok()).unwrap_or(0.0);
                                let n_global = recall::n_global();
                                let mut ql = vec![0.0f32; n_global * recall::g_nh() * recall::hd()];
                                let qk5 = if !qonly_skip && unsafe { kv::read_global_q(handle, last[0], &mut ql) }.is_ok() {
                                    recall::l5_query_embed(&ql)
                                } else { Vec::new() };
                                if qonly_skip {
                                    // handled above — fall through with no recall
                                } else if qk5.is_empty() {
                                    tracing::warn!("RECALL-L5: read_global_q/l5_embed unavailable -- clean prompt");
                                } else {
                                    let mut scored: Vec<(f32, String, String, Option<recall::MemPolicy>)> = Vec::new();
                                    {
                                        let ns_guard = app.nightshift.read().unwrap();
                                        for ep in registry.iter().chain(ns_guard.iter()) {
                                            if ep.l5key.len() != recall::hd() { continue; }
                                            if ep.lifecycle != 0 && std::env::var("SP_RECALL_AUDIT").ok().as_deref() != Some("1") { continue; }
                                            scored.push((recall::cos512(&qk5, &ep.l5key), ep.name.clone(), ep.text.clone(), ep.policy.clone()));
                                        }
                                    }
                                    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                                    // SP_RECALL_L5_DUMPRANK: one-shot diagnostic — log the top-8 ranked
                                    // (name=cos) list per query so the correct-episode RANK can be
                                    // recovered offline (decides re-rank vs re-key for the cross-pick
                                    // residual). Default-off; telemetry only, no behavior change.
                                    if std::env::var("SP_RECALL_L5_DUMPRANK").as_deref() == Ok("1") {
                                        let dump: Vec<String> = scored.iter().take(8)
                                            .map(|(c, n, _, _)| format!("{}={:.4}", n, c)).collect();
                                        tracing::info!("RECALL-L5-DUMPRANK: q={:?} ranked=[{}]", ruser, dump.join(" "));
                                    }
                                    let (mut bcos, mut bname, mut btext) = scored.first()
                                        .map(|t| (t.0, t.1.clone(), t.2.clone()))
                                        .unwrap_or((f32::NEG_INFINITY, String::new(), String::new()));
                                    let (bcos2, bname2) = scored.get(1).map(|t| (t.0, t.1.clone()))
                                        .unwrap_or((f32::NEG_INFINITY, String::new()));
                                    // MARGIN TELEMETRY (always-on when L5 runs; telemetry-then-pin):
                                    // single-episode registries have no top2 -> margin = +inf semantics.
                                    let margin = if bcos2.is_finite() { bcos - bcos2 } else { f32::INFINITY };
                                    tracing::info!("RECALL-L5-MARGIN: top1='{}' cos={:.4} top2='{}' cos2={:.4} margin={:.4}",
                                        bname, bcos, bname2, bcos2, margin);
                                    // ===== MARGIN-GATED TOP-3 JUDGE RERANK (SP_RECALL_L5_RERANK=<eps>) =====
                                    // G-SEL-OFFLINE: the 7 selector misses are same-template periphrasis
                                    // cross-picks with tiny top1-top2 margins (all <0.013; eps=0.015 catches
                                    // 7/7 while firing on only 16/54 correct picks); correct-in-top-3 =
                                    // 59/61. The distinguishing signal is SEMANTIC (cheap levers convicted
                                    // inert), so on an ambiguous margin the dense reference model READS the top-3 fact TEXTS
                                    // and picks A/B/C in ONE constrained greedy side-pass — the judge
                                    // DECIDES, the recite path DELIVERS (ADR-002; the 61160e9 pattern).
                                    // Fail-open: any forward/parse failure keeps the L5 top-1. The recite/
                                    // decline below resets the cache anyway, so no restore pass is needed.
                                    // Unset/0 = null floor.
                                    // ===== MARGIN-GATED QUERY CANONICALIZATION (SP_RECALL_L5_CANON=<eps>) =====
                                    // G-SEL-CANON: the judge-rerank PARKED (G-SEL-RERANK-61: fixes 5 but
                                    // breaks 4 — in-family A/B/C adjudication is ~unreliable). The sharper
                                    // decide-step: on an ambiguous margin, ONE micro-forward rewrites the
                                    // query's periphrases to PROPER NAMES ("boot-shaped Mediterranean
                                    // country" -> "Italy") — decide-in-clean-text (ADR-002), a task the dense reference model
                                    // is far more reliable at than slot-picking. The surfaced name appears
                                    // VERBATIM in exactly the right fact, so a deterministic salient-overlap
                                    // count over the L5 top-5 becomes decisive (raw-query Jaccard was inert,
                                    // G-SEL-OFFLINE). Switch rule is CONSERVATIVE: only on strictly-greater
                                    // overlap than the current top1 (ties keep L5 order; fail-open on any
                                    // forward/parse failure). Unset/0 = null floor.
                                    let canon_eps: f32 = std::env::var("SP_RECALL_L5_CANON").ok()
                                        .and_then(|s| s.parse().ok()).unwrap_or(0.0);
                                    if canon_eps > 0.0 && margin < canon_eps && bcos >= tau_l5 && scored.len() >= 2 {
                                        // NAME-THE-SUBJECT framing (run-2/3 lesson: "rewrite the question"
                                        // makes the dense reference model ANSWER it instead — even one-shot. Naming the
                                        // DESCRIBED subject is the simpler task and its output is exactly
                                        // the overlap token the switch rule needs). Exemplar OUT-OF-CORPUS.
                                        let ccontent = format!(
                                            "A question describes something without naming it. Name the described thing, place, or person — do NOT answer the question.\nExample: \"What is the tallest building in the city that never sleeps?\" describes New York.\nQuestion: \"{ruser}\" describes:");
                                        let cmsgs = vec![Message { role: "user".to_string(), content: ccontent }];
                                        match app.tokenizer.apply_template_ids(&cmsgs) {
                                            Ok(ctoks) if ctoks.len() >= 2 => {
                                                let _ = unsafe { kv::reset(handle) };
                                                let (chead, clast) = ctoks.split_at(ctoks.len() - 1);
                                                let mut ok = chead.is_empty() || unsafe { kv::prefill(handle, chead) }.is_ok();
                                                let mut canon = String::new();
                                                if ok {
                                                    let mut tok = clast[0];
                                                    let turn_stops = app.tokenizer.turn_stop_ids();
                                                    // raw-argmax side-pass MUST mask the served suppress set
                                                    // (gemma <image|>/<audio|> soft tokens + control markers)
                                                    // — the exact placeholder-spam bug the judge verify path
                                                    // already root-caused. Same cure here.
                                                    let suppress = app.tokenizer.suppress_token_ids();
                                                    for _ in 0..32 {
                                                        if unsafe { kv::decode_step(handle, tok, logits) }.is_err() { ok = false; break; }
                                                        let mut best_t = 0usize;
                                                        let mut best_v = f32::NEG_INFINITY;
                                                        for (t, &v) in logits.iter().enumerate() {
                                                            if suppress.contains(&(t as i32)) { continue; }
                                                            if v > best_v { best_v = v; best_t = t; }
                                                        }
                                                        tok = best_t as i32;
                                                        if tok == 1 || turn_stops.contains(&tok) { break; }
                                                        canon.push_str(&String::from_utf8_lossy(app.tokenizer.decode_token(tok)));
                                                        if canon.trim_start().contains('\n') { break; }
                                                    }
                                                }
                                                let canon = canon.trim().lines().next().unwrap_or("").trim().to_string();
                                                if ok && !canon.is_empty() {
                                                    let k5 = scored.len().min(5);
                                                    let ov: Vec<usize> = (0..k5).map(|i| recall::canon_overlap(&canon, &scored[i].2)).collect();
                                                    let (mut bi, mut bov) = (0usize, ov[0]);
                                                    for i in 1..k5 { if ov[i] > bov { bi = i; bov = ov[i]; } }
                                                    if bi != 0 && bov > ov[0] {
                                                        tracing::info!("RECALL-L5 CANON: margin={:.4} < eps={:.4}; canon={:?} overlap {}>{} -> switch '{}' (over top1 '{}')",
                                                            margin, canon_eps, canon, bov, ov[0], scored[bi].1, bname);
                                                        bcos = scored[bi].0; bname = scored[bi].1.clone(); btext = scored[bi].2.clone();
                                                    } else {
                                                        tracing::info!("RECALL-L5 CANON: margin={:.4} < eps={:.4}; canon={:?} keeps top1 '{}' (ov={:?})",
                                                            margin, canon_eps, canon, bname, ov);
                                                    }
                                                } else {
                                                    tracing::warn!("RECALL-L5 CANON: rewrite unusable (ok={} canon={:?}) -- keeping L5 top1 (fail-open)", ok, canon);
                                                }
                                            }
                                            _ => tracing::warn!("RECALL-L5 CANON: apply_template_ids failed -- keeping L5 top1 (fail-open)"),
                                        }
                                    }
                                    let rerank_eps: f32 = std::env::var("SP_RECALL_L5_RERANK").ok()
                                        .and_then(|s| s.parse().ok()).unwrap_or(0.0);
                                    if rerank_eps > 0.0 && margin < rerank_eps && bcos >= tau_l5 && scored.len() >= 2 {
                                        let k = scored.len().min(3);
                                        let letters = ["A", "B", "C"];
                                        // ANTI-POSITION-BIAS (FINDINGS-#3, relearned by G-SEL-RERANK-61 run-1:
                                        // fixed cos-order slots made the judge CREATE adjacent cross-picks —
                                        // same-family capitals swapped wholesale). Deterministic per-query
                                        // Fisher-Yates over the k candidates (fnv1a of the query seeds a
                                        // xorshift), letters stay A/B/C; the pick maps back through `perm`.
                                        let mut seed: u64 = 0xcbf29ce484222325;
                                        for b in ruser.as_bytes() { seed ^= *b as u64; seed = seed.wrapping_mul(0x100000001b3); }
                                        if seed == 0 { seed = 0x9e3779b97f4a7c15; }
                                        let mut rng = move || { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; seed };
                                        let mut perm: Vec<usize> = (0..k).collect();
                                        for i in (1..k).rev() {
                                            let j = (rng() % (i as u64 + 1)) as usize;
                                            perm.swap(i, j);
                                        }
                                        let mut entries = String::new();
                                        for slot in 0..k { entries.push_str(&format!("{}) {}\n", letters[slot], scored[perm[slot]].2)); }
                                        let jcontent = format!(
                                            "You are a memory index. Exactly one fact below directly answers the question.\n\n{entries}\nQUESTION: {ruser}\n\nReply with only the single letter (A, B or C) of the fact that answers the question:");
                                        let jmsgs = vec![Message { role: "user".to_string(), content: jcontent }];
                                        match app.tokenizer.apply_template_ids(&jmsgs) {
                                            Ok(jtoks) if jtoks.len() >= 2 => {
                                                let _ = unsafe { kv::reset(handle) };
                                                let (jhead, jlast) = jtoks.split_at(jtoks.len() - 1);
                                                let mut ok = jhead.is_empty() || unsafe { kv::prefill(handle, jhead) }.is_ok();
                                                let mut reply = String::new();
                                                if ok {
                                                    let mut tok = jlast[0];
                                                    let turn_stops = app.tokenizer.turn_stop_ids();
                                                    let suppress = app.tokenizer.suppress_token_ids();
                                                    for _ in 0..6 {
                                                        if unsafe { kv::decode_step(handle, tok, logits) }.is_err() { ok = false; break; }
                                                        let mut best_t = 0usize;
                                                        let mut best_v = f32::NEG_INFINITY;
                                                        for (t, &v) in logits.iter().enumerate() {
                                                            if suppress.contains(&(t as i32)) { continue; }
                                                            if v > best_v { best_v = v; best_t = t; }
                                                        }
                                                        tok = best_t as i32;
                                                        if tok == 1 || turn_stops.contains(&tok) { break; }
                                                        reply.push_str(&String::from_utf8_lossy(app.tokenizer.decode_token(tok)));
                                                        if !reply.trim().is_empty() { break; } // one letter is all we need
                                                    }
                                                }
                                                let pick = reply.trim().to_uppercase().chars()
                                                    .find(|c| matches!(c, 'A' | 'B' | 'C'))
                                                    .map(|c| (c as u8 - b'A') as usize)
                                                    .filter(|&slot| slot < k)
                                                    .map(|slot| perm[slot]);
                                                match (ok, pick) {
                                                    (true, Some(i)) => {
                                                        if i != 0 {
                                                            tracing::info!("RECALL-L5 RERANK: margin={:.4} < eps={:.4}; judge pick '{}' (over top1 '{}')",
                                                                margin, rerank_eps, scored[i].1, bname);
                                                            bcos = scored[i].0; bname = scored[i].1.clone(); btext = scored[i].2.clone();
                                                        } else {
                                                            tracing::info!("RECALL-L5 RERANK: margin={:.4} < eps={:.4}; judge confirms top1 '{}'", margin, rerank_eps, bname);
                                                        }
                                                    }
                                                    _ => tracing::warn!("RECALL-L5 RERANK: judge unusable (ok={} reply={:?}) -- keeping L5 top1 (fail-open)", ok, reply),
                                                }
                                            }
                                            _ => tracing::warn!("RECALL-L5 RERANK: apply_template_ids failed -- keeping L5 top1 (fail-open)"),
                                        }
                                    }
                                    if bcos >= tau_l5 && !btext.is_empty() {
                                        // ATTR-GROUNDING (SNE crucible fix): on zero-prior data the model
                                        // CONFABULATES a plausible wrong value when asked an attribute the
                                        // delivered fact does not state (80% on the SNE crucible; 5% leak the
                                        // fact's own value). Two composable, default-off levers:
                                        //  - SP_RECALL_STRICT: closed-book delivery — "answer ONLY from the fact;
                                        //    if it doesn't state what's asked, decline exactly." The MODEL grounds
                                        //    with its own semantics (paraphrase-safe: a same-attribute paraphrase
                                        //    still recites; a different-attribute query declines).
                                        //  - SP_RECALL_ATTR_GATE: deterministic lexical pre-check; if the query's
                                        //    salient words are ABSENT from the fact (ratio>=SP_RECALL_ATTR_TAU),
                                        //    FORCE the strict decline framing regardless of the model. Hard
                                        //    fallback (not paraphrase-aware). Both unset = the proven recite path
                                        //    (byte-identical null floor).
                                        // ===== MEM-OKF v2 / ADR-004: per-entry policy dispatch (SP_MEM_POLICY=1) =====
                                        // The SELECTED entry's own mem_delivery/mem_decline drive delivery, OVERRIDING
                                        // the global SP_RECALL_L5_PROMPT/SP_RECALL_ATTR_GATE env flags. Default-off
                                        // (SP_MEM_POLICY unset) or an un-policied entry ⇒ the env path, byte-identical.
                                        let mem_policy_on = std::env::var("SP_MEM_POLICY").as_deref() == Ok("1");
                                        let bpol: Option<recall::MemPolicy> = if mem_policy_on {
                                            scored.iter().find(|t| t.1 == bname).and_then(|t| t.3.clone())
                                        } else { None };
                                        // effective delivery mode: policy.delivery (attr-gate-strict recites when it
                                        // does NOT decline), else the global SP_RECALL_L5_PROMPT (default "plain").
                                        let env_prompt = std::env::var("SP_RECALL_L5_PROMPT").ok().unwrap_or_else(|| "plain".to_string());
                                        let delivery_mode: String = match bpol.as_ref() {
                                            Some(p) if p.delivery == "attr-gate-strict" => "recite".to_string(),
                                            Some(p) => p.delivery.clone(),
                                            None => env_prompt,
                                        };
                                        if let Some(p) = bpol.as_ref() {
                                            tracing::info!("MEM-POLICY: entry '{}' class={} delivery={} -> mode={}",
                                                bname, p.class, p.delivery, delivery_mode);
                                        }
                                        let strict = std::env::var("SP_RECALL_STRICT").as_deref() == Ok("1");
                                        // attr-gate fires from env OR from the entry's policy (attr-gate-strict /
                                        // a decline that includes attribute-absent/zero-inference).
                                        let policy_attr_gate = bpol.as_ref().map(|p|
                                            p.delivery == "attr-gate-strict"
                                            || p.decline_when.iter().any(|w| w == "attribute-absent" || w == "zero-inference")
                                        ).unwrap_or(false);
                                        let attr_gate = policy_attr_gate
                                            || std::env::var("SP_RECALL_ATTR_GATE").as_deref() == Ok("1");
                                        let attr_tau: f32 = std::env::var("SP_RECALL_ATTR_TAU").ok()
                                            .and_then(|s| s.parse().ok()).unwrap_or(0.5);
                                        let absent = if attr_gate { recall::attr_absent_ratio(&ruser, &btext) } else { 0.0 };
                                        // PARAPHRASE GUARD: only decline on attribute-absence when the query
                                        // shares a high-entropy verbatim token with the fact (a private-entity
                                        // ID/code). General-knowledge/paraphrase queries have no such token, so
                                        // the gate stays off for them (recall preserved) — this is what makes
                                        // the deterministic gate globally default-on-safe, not regime-specific.
                                        // #75: a private-secret entry (attr-gate-strict) is KNOWN private, so
                                        // the paraphrase entity-guard is bypassed — decline on any absent
                                        // attribute even when the query shares the plain entity name
                                        // ("Who designed Vault-7?"). The guard stays for the GLOBAL env
                                        // attr-gate (general knowledge) to avoid over-declining paraphrases.
                                        let policy_secret = bpol.as_ref()
                                            .map(|p| p.delivery == "attr-gate-strict").unwrap_or(false);
                                        let force_decline = attr_gate && absent >= attr_tau
                                            && (policy_secret || recall::query_has_entity_token(&ruser));
                                        if force_decline {
                                            // ZERO-INFERENCE symbolic decline (ADR-002 Tier-2 executor). The
                                            // fact exists but does NOT state the queried attribute. Set a
                                            // deterministic decline string and SKIP the synthesis forward
                                            // entirely (consumed at the synthesis seam below, before
                                            // decode_step(syn_last)). No gemma4 decode => confabulation/leak
                                            // is mathematically impossible + the turn resolves in microseconds.
                                            symbolic_decline = Some(bpol.as_ref().map(|p| p.decline_message.clone())
                                                .unwrap_or_else(|| "I have a record for that entity, but it does not include that specific detail.".to_string()));
                                            recalled = Some((bname.clone(), (bcos * 1000.0) as u32));
                                            tracing::info!("RECALL-L5: '{}' cos={:.3} absent={:.2} -> ATTR-DECLINE (zero-inference symbolic, no forward){}",
                                                bname, bcos, absent, if mem_policy_on { " [mem-policy]" } else { "" });
                                            telem_emit_recall(&ruser, &bname,
                                                bpol.as_ref().map(|p| p.class.as_str()).unwrap_or("-"),
                                                bcos, margin, "attr-gate-strict", "decline", Some(&app.events_tx));
                                        } else if tau_margin > 0.0 && margin < tau_margin {
                                            // MARGIN GATE (delivery only; attr-decline above is NOT starved):
                                            // an ambiguous top1 (no clear gap over top2) is background, not a
                                            // memory hit — deliver nothing, run the clean prompt.
                                            tracing::info!("RECALL-L5: MARGIN-SKIP top1='{}' cos={:.3} margin={:.4} < tau_m={:.4} -> no recall (clean prompt)",
                                                bname, bcos, margin, tau_margin);
                                            telem_emit_recall(&ruser, &bname,
                                                bpol.as_ref().map(|p| p.class.as_str()).unwrap_or("-"),
                                                bcos, margin, &delivery_mode, "margin-skip", Some(&app.events_tx));
                                        } else {
                                            // recite (proven 86.89% path); SP_RECALL_STRICT = closed-book
                                            // model-grounded framing (dead lever, kept default-off).
                                            let aug_msgs = if strict {
                                                vec![
                                                    Message { role: "system".to_string(), content: "You are Shannon-Prime, a local AI with a real working memory. Answer ONLY using the fact on record below. If the fact does not state what the question asks, reply EXACTLY: \"I do not have that information.\" Do not guess, infer, or invent any detail that is not written in the fact.".to_string() },
                                                    Message { role: "user".to_string(), content: format!("Fact on record: {}\n\nQuestion: {}", btext, ruser) },
                                                ]
                                            } else if delivery_mode == ("sandwich") {
                                                // DELIVERY SWEEP (2026-07-02, RUNBOOK §11): instruction AFTER the
                                                // question (recency) + explicit override authority.
                                                vec![
                                                    Message { role: "system".to_string(), content: "You are Shannon-Prime, a local AI with a real working memory. Keep replies short. Use facts you were given faithfully; if you don't know, say so.".to_string() },
                                                    Message { role: "user".to_string(), content: format!("Context (authoritative, from your memory): {}\n\n{}\n\n(Answer using ONLY the context above; it overrides your prior knowledge.)", btext, ruser) },
                                                ]
                                            } else if delivery_mode == ("factecho") {
                                                // DELIVERY SWEEP: prime the answer to copy from the fact.
                                                vec![
                                                    Message { role: "system".to_string(), content: "You are Shannon-Prime, a local AI with a real working memory. Your memory record is the ground truth for this conversation, even where it differs from general knowledge. Keep replies short.".to_string() },
                                                    Message { role: "user".to_string(), content: format!("Fact on record: {}\n\nQuestion: {}\n\nAnswer using the fact on record:", btext, ruser) },
                                                ]
                                            } else if delivery_mode == ("system") {
                                                // DELIVERY SWEEP: fact delivered as SYSTEM authority, clean user turn.
                                                vec![
                                                    Message { role: "system".to_string(), content: format!("You are Shannon-Prime, a local AI with a real working memory. Fact on record (authoritative for this conversation, overrides prior knowledge): {}\nAnswer from this fact; keep replies short.", btext) },
                                                    Message { role: "user".to_string(), content: ruser.clone() },
                                                ]
                                            } else if delivery_mode == ("systemecho") {
                                                // DELIVERY SWEEP round 2 winner: system authority + copy-priming
                                                // (full-61: 88.52% OBEY, 0 LEAK — every correctly-selected episode
                                                // obeyed; misses = selection cross-picks). MULTI-TURN FIX: preserve
                                                // the conversation (G-ONECONFIG-LIVE C-phase root cause: delivery
                                                // used to keep only the last user message, so turn-2 questions about
                                                // turn-1 content were unanswerable on recall turns).
                                                let mut v = vec![
                                                    Message { role: "system".to_string(), content: format!("You are Shannon-Prime, a local AI with a real working memory. Fact on record (authoritative for this conversation, overrides prior knowledge): {}\nEvery answer must repeat the relevant part of the fact on record verbatim. Keep replies short.", btext) },
                                                ];
                                                match orig_msgs.as_ref() {
                                                    Some(ms) if ms.iter().any(|m| m.role == "user") => {
                                                        for m in ms.iter().filter(|m| m.role != "system") { v.push(m.clone()); }
                                                        if let Some(last_u) = v.iter_mut().rev().find(|m| m.role == "user") {
                                                            last_u.content = format!("{}\n\nAnswer using the fact on record:", ruser);
                                                        }
                                                    }
                                                    _ => v.push(Message { role: "user".to_string(), content: format!("{}\n\nAnswer using the fact on record:", ruser) }),
                                                }
                                                v
                                            } else if delivery_mode == ("scaled") {
                                                // SCALED delivery wording (2026-07-02): the plain recite wording's
                                                // paraphrase obedience proved FP/build-FRAGILE (the 86.89% receipt
                                                // is NOT reproducible on the current stack: 40.98% via the receipt's
                                                // own harness, 3 independent rebuilds incl. exact a14fee4 src+lock).
                                                // This explicit-override wording carries its own receipt: OBEY 52/61
                                                // = 85.2% (_g_faithful_recall_scaled.json, seam=scaled) — the
                                                // instruction does the work, not a thin float margin.
                                                vec![
                                                    Message { role: "system".to_string(), content: "You are Shannon-Prime, a local AI with a real working memory. Keep replies short. Use facts you were given faithfully; if you don't know, say so.".to_string() },
                                                    Message { role: "user".to_string(), content: format!("Context (authoritative, from your memory): {}\n\nUsing the context above, answer faithfully even if it differs from what you already know:\n{}", btext, ruser) },
                                                ]
                                            } else {
                                                vec![
                                                    Message { role: "system".to_string(), content: "You are Shannon-Prime, a local AI with a real working memory. Keep replies short. Use facts you were given faithfully; if you don't know, say so.".to_string() },
                                                    Message { role: "user".to_string(), content: format!("Context (authoritative, current): {}\n\n{}", btext, ruser) },
                                                ]
                                            };
                                            match app.tokenizer.apply_template_ids(&aug_msgs) {
                                                Ok(aug_toks) if aug_toks.len() >= 2 => {
                                                    let _ = unsafe { kv::reset_cold(handle) };
                                                    let (aug_head, aug_last) = aug_toks.split_at(aug_toks.len() - 1);
                                                    if aug_head.is_empty() || unsafe { kv::prefill(handle, aug_head) }.is_ok() {
                                                        syn_last = aug_last[0];
                                                        recalled = Some((bname.clone(), (bcos * 1000.0) as u32));
                                                        recalled_text = Some(btext.clone()); // SPECTEST ground
                                                        recalled_query = Some(ruser.clone()); // LM-B2b turn join key
                                                        recalled_class = bpol.as_ref().map(|p| p.class.clone());
                                                        recalled_secret = bpol.as_ref().map(|p| p.class == "private-secret").unwrap_or(false);
                                                        tracing::info!("RECALL-L5: '{}' cos={:.3} >= tau={:.3} absent={:.2} mode={} -> TEXT-IN-CONTEXT",
                                                            bname, bcos, tau_l5, absent, if strict {"STRICT"} else {"recite"});
                                                        telem_emit_recall(&ruser, &bname,
                                                            bpol.as_ref().map(|p| p.class.as_str()).unwrap_or("-"),
                                                            bcos, margin, &delivery_mode, "deliver", Some(&app.events_tx));
                                                    } else { tracing::warn!("RECALL-L5: prefill(aug) failed -- clean prompt"); }
                                                }
                                                _ => tracing::warn!("RECALL-L5: apply_template_ids(aug) failed -- clean prompt"),
                                            }
                                        }
                                    } else {
                                        tracing::info!("RECALL-L5: best='{}' cos={:.3} < tau={:.3} -> no recall (clean prompt)", bname, bcos, tau_l5);
                                    }
                                }
                            }
                        }
                        // ===== B3-JUDGE: the dense reference model GENERATIVE recall judge (SP_B3_JUDGE) =====
                        // The open-set novel-recall wall that defeated every GEOMETRIC signal
                        // (q·K, cosine, C2 sig, the W_c head, causal self-ablation) is broken by
                        // a GENERATIVE judge: the dense reference model READS the candidate memory TEXTS (the words,
                        // not post-RoPE K vectors) and picks the one that answers THIS query, or
                        // [NULL]. Validated offline at 85.7% recall@1 on _needle_corpus_div (the
                        // EXACT corpus that gave W_c ~50%); harness tools/xbar_lsh/judge_recall_test.py.
                        // Default-off (SP_B3_JUDGE unset) ⇒ this whole block is skipped ⇒
                        // byte-identical null floor. Runs only when no prior stage decided.
                        if cfg!(feature = "legacy_policy")  // P1b: judge cascade = refuted (OKFS cbea4d38)
                            && !int2_decided && recalled.is_none()
                            && std::env::var("SP_B3_JUDGE").ok().filter(|s| !s.is_empty()).is_some()
                        {
                            // KAIROS-bounded working set: at most SP_B3_JUDGE_K (default 20) episodes.
                            let kbound: usize = std::env::var("SP_B3_JUDGE_K").ok()
                                .and_then(|s| s.parse().ok()).unwrap_or(20);
                            // The user query text (raw, NO chat template — the judge prompt itself
                            // is what carries the template). raw_user is the last user message.
                            let query = raw_user.as_ref().map(|s| s.trim().to_string()).unwrap_or_default();
                            if query.is_empty() {
                                tracing::info!("B3-JUDGE: no user query text -- skipping (clean prompt)");
                            } else {
                                // (1) Assemble the candidate set. KAIROS bounds it to <=kbound. The
                                // FINDINGS-#3 anti-position-bias discipline of the validated harness:
                                // the candidate ORDER is shuffled and each gets a RANDOMIZED copy-able
                                // tag (consonant+digit+consonant) — a fixed [M_A],[M_B],.. order anchors
                                // the model on the first slot (observed: it returned [M_A] for every
                                // query). The shuffle/tag draw is DETERMINISTIC per query (seeded by a
                                // hash of the query text) so a turn is reproducible/byte-exact-when-off.
                                // A tiny xorshift PRNG (no rand crate dep) drives a Fisher-Yates shuffle.
                                let mut seed: u64 = 0xcbf29ce484222325;
                                for b in query.as_bytes() { seed ^= *b as u64; seed = seed.wrapping_mul(0x100000001b3); }
                                if seed == 0 { seed = 0x9e3779b97f4a7c15; }
                                let mut rng = move || {
                                    seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; seed
                                };
                                // ===== KAIROS DUAL-AXIS WORKING-SET ASSEMBLY (LIVE-CONVERSATION) =====
                                // The "first K episodes" window is replaced by a recency+salience
                                // selection over a UNIFIED candidate pool = curated registry ∪ the LIVE
                                // NIGHTSHIFT episodes (the conversation's own immediate past). Two axes
                                // feed the SAME generative judge:
                                //   (1) RECENCY  : the last SP_B3_JUDGE_R episodes of app.nightshift (the
                                //       most-recent conversational turns; NIGHTSHIFT appends ep_live_NNN
                                //       to the end). ALWAYS included -- short-term working memory. This is
                                //       the loop-closure: the judge can recall what was just said.
                                //   (2) SALIENCE : ONLY if the cold pool > the remaining slots, the demoted
                                //       Stage-0 C2-LSH Hamming pre-filter rescues OLDER memories (curated
                                //       registry + nightshift episodes older than the recency-R tail) that
                                //       match THIS query. Compute the live query's C2 sig, score every cold
                                //       candidate by Hamming agreement (R_BITS - hamming) to its STORED sig
                                //       (exact, captured at ingest), take the top (K - R), append.
                                // Candidates are CLONED into a working Vec (<=K episodes; cheap) with a
                                // parallel provenance flag (live = recall via inject_tokens; curated =
                                // replay(dir)). The judge still makes the final pick; Stage-0 only widens
                                // what it can see. NULL FLOOR: with SP_B4_NIGHTSHIFT unset, app.nightshift
                                // is empty ⇒ recency is empty ⇒ the cold pool is the curated registry only
                                // ⇒ behaviour is registry-only, byte-identical to the prior KAIROS path.
                                let rbound: usize = std::env::var("SP_B3_JUDGE_R").ok()
                                    .and_then(|s| s.parse().ok()).unwrap_or(8);
                                // Snapshot the live nightshift episodes (clone; bounded by working set).
                                let ns_snapshot: Vec<recall::Episode> = {
                                    let g = app.nightshift.read().unwrap();
                                    g.iter().cloned().collect()
                                };
                                let n_live = ns_snapshot.len();
                                let n_cur = registry.len();
                                // ===== REORDER (ADR-002 §8.1): judge = reject veto, L5-direct owns delivery.
                                // Precompute the GLOBAL L5-#1 episode (bcos over registry ∪ nightshift by
                                // L5 query-key cosine — the IDENTICAL selection SP_RECALL_L5 delivers at
                                // 86.89%). On a judge PASS we deliver THIS, not the judge's own pick among
                                // the shuffled top-K shortlist (K-sweep: judge-pick delivery = 50% vs
                                // L5-#1 = 86.89%). The judge still sees the K(>=2) shortlist so it engages
                                // its PASS/NULL reject (K=1 degenerates to always-PICK). L5 mode off /
                                // best unavailable ⇒ None ⇒ falls back to the judge's pick (prior behavior,
                                // null floor preserved). (name, text) of the delivery target.
                                let l5_best: Option<(String, String)> =
                                    if std::env::var("SP_B3_JUDGE_L5").as_deref() == Ok("1") {
                                        let ql5g = recall::l5_query_embed(&qbuf);
                                        if ql5g.is_empty() { None } else {
                                            let (mut bcos, mut bname, mut btext) =
                                                (f32::NEG_INFINITY, String::new(), String::new());
                                            for ep in registry.iter().chain(ns_snapshot.iter()) {
                                                if ep.l5key.len() != recall::hd() { continue; }
                                                let c = recall::cos512(&ql5g, &ep.l5key);
                                                if c > bcos { bcos = c; bname = ep.name.clone(); btext = ep.text.clone(); }
                                            }
                                            if btext.trim().is_empty() { None } else {
                                                tracing::info!("B3-JUDGE REORDER: L5 global best='{}' cos={:.3} (delivery target on PASS)", bname, bcos);
                                                Some((bname, btext))
                                            }
                                        }
                                    } else { None };
                                let rkeep = rbound.min(kbound).min(n_live);
                                // The working candidate set: each entry is (Episode, is_live).
                                let mut cands: Vec<(recall::Episode, bool)> = Vec::with_capacity(kbound);
                                let mut kairos_rescued: Vec<String> = Vec::new();
                                // (1) recency axis = the last rkeep LIVE episodes (the conversation tail).
                                for j in (n_live - rkeep)..n_live {
                                    cands.push((ns_snapshot[j].clone(), true));
                                }
                                // The cold pool = curated registry (all) ∪ older nightshift (before rkeep).
                                // Built as (Episode, is_live) so dispatch knows the recall path.
                                let cold: Vec<(recall::Episode, bool)> = registry.iter().cloned()
                                    .map(|e| (e, false))
                                    .chain(ns_snapshot[..(n_live - rkeep)].iter().cloned().map(|e| (e, true)))
                                    .collect();
                                let salience_slots = kbound.saturating_sub(cands.len());
                                if cold.len() <= salience_slots {
                                    // everything fits: take the whole cold pool, no Hamming cull needed.
                                    for c in cold.into_iter() { cands.push(c); }
                                } else if salience_slots > 0 {
                                    // STAGE-1 (the E2E composition): SP_B3_JUDGE_WC=<wc_deploy.bin> scores the
                                    // cold pool with the PROVEN W_c head (360/361 selector) and shortlists the
                                    // top (K-R) for the judge+0.6 verifier to ground. W_c is the correct Stage-1
                                    // — C2-Hamming (the else path) was a measured-weaker signal that would
                                    // starve the judge of the right candidate. Unset => Hamming (null floor).
                                    let wc_head = std::env::var("SP_B3_JUDGE_WC").ok()
                                        .filter(|s| !s.is_empty()).and_then(|p| recall::load_wc(&p));
                                    // STAGE-1 L5 (SP_B3_JUDGE_L5=1): rank the cold pool by L5 query-key
                                    // cosine (the 86.89%-live recall selector) and shortlist top-(K-R) for
                                    // the judge. This IS the pass->block: L5 PASSES the shortlist, the
                                    // generative judge PICKS the answerer or [NULL] (the reject). Requires
                                    // ep.l5 keys on the episodes; falls through to W_c/C2 if L5 unavailable.
                                    let ql5 = if std::env::var("SP_B3_JUDGE_L5").as_deref() == Ok("1") {
                                        recall::l5_query_embed(&qbuf)
                                    } else { Vec::new() };
                                    if !ql5.is_empty() {
                                        let mut scored: Vec<(usize, f32)> = cold.iter().enumerate()
                                            .map(|(i, (ep, _))| (i, if ep.l5key.len() == recall::hd() {
                                                recall::cos512(&ql5, &ep.l5key) } else { f32::NEG_INFINITY }))
                                            .collect();
                                        scored.sort_by(|a, b| b.1.partial_cmp(&a.1)
                                            .unwrap_or(std::cmp::Ordering::Equal)); // descending L5 cosine
                                        for &(i, _sc) in scored.iter().take(salience_slots) {
                                            kairos_rescued.push(cold[i].0.name.clone());
                                            cands.push(cold[i].clone());
                                        }
                                        tracing::info!(
                                            "B3-JUDGE STAGE-1 L5: shortlisted top-{} of {} cold by L5 query-key cosine",
                                            salience_slots, cold.len());
                                    } else if let Some(head) = wc_head.as_ref() {
                                        let mut scored: Vec<(usize, f32)> = cold.iter().enumerate()
                                            .map(|(i, (ep, _))| (i, recall::wc_score(
                                                &qbuf, &ep.gk, ep.gk_ng, ep.npos as usize, head)))
                                            .collect();
                                        scored.sort_by(|a, b| b.1.partial_cmp(&a.1)
                                            .unwrap_or(std::cmp::Ordering::Equal)); // descending W_c relevance
                                        for &(i, _sc) in scored.iter().take(salience_slots) {
                                            kairos_rescued.push(cold[i].0.name.clone());
                                            cands.push(cold[i].clone());
                                        }
                                        tracing::info!(
                                            "B3-JUDGE STAGE-1 W_c: shortlisted top-{} of {} cold by W_c (head r={})",
                                            salience_slots, cold.len(), head.r);
                                    } else {
                                    // (2) salience axis: rescue the top (K - R) cold candidates by C2 Hamming.
                                    let mut qkbuf = vec![0.0f32; n_global * recall::g_kvd() * (npos_q as usize)];
                                    match unsafe { kv::read_global_k(handle, &mut qkbuf, npos_q) } {
                                        Ok(ngk) => {
                                            let qsig = app.recall_proj.signature(
                                                &qkbuf, ngk as usize, npos_q as usize);
                                            let mut scored: Vec<(usize, u32)> = cold.iter().enumerate()
                                                .map(|(i, (ep, _))| (i, recall::agree(&qsig, &ep.sig)))
                                                .collect();
                                            scored.sort_by(|a, b| b.1.cmp(&a.1)); // descending agreement
                                            for &(i, _ag) in scored.iter().take(salience_slots) {
                                                kairos_rescued.push(cold[i].0.name.clone());
                                                cands.push(cold[i].clone());
                                            }
                                            tracing::info!(
                                                "B3-JUDGE KAIROS: live={} cur={} K={} R={} (recency-kept={}) salience_slots={} qsig={:016x}.. rescued=[{}]",
                                                n_live, n_cur, kbound, rbound, rkeep, salience_slots, qsig[0],
                                                kairos_rescued.join(", "));
                                        }
                                        Err(e) => {
                                            tracing::warn!(
                                                "B3-JUDGE KAIROS: read_global_k(npos_q={}) failed: {e} -- recency-only working set (live tail)",
                                                npos_q);
                                        }
                                    }
                                    }
                                }
                                tracing::info!(
                                    "B3-JUDGE KAIROS working set: {} episode(s) ({} live-recency + {} cold), live_total={} cur_total={}",
                                    cands.len(), rkeep, cands.len().saturating_sub(rkeep), n_live, n_cur);
                                // Fisher-Yates shuffle the working set (anti-position-bias, FINDING #3).
                                for i in (1..cands.len()).rev() {
                                    let j = (rng() % (i as u64 + 1)) as usize;
                                    cands.swap(i, j);
                                }
                                // tag pool = consonant+digit+consonant (collision-free copy-able codes);
                                // drawn without replacement per query (sampled order from the same rng).
                                const CONS: &[u8] = b"BCDFGHJKLMNPQRSTVWXZ";
                                const DIG: &[u8] = b"0123456789";
                                let mut pool: Vec<String> = Vec::with_capacity(CONS.len() * 10 * CONS.len());
                                for &a in CONS { for &d in DIG { for &c in CONS {
                                    pool.push(String::from_utf8(vec![a, d, c]).unwrap()); } } }
                                for i in (1..pool.len()).rev() {
                                    let j = (rng() % (i as u64 + 1)) as usize;
                                    pool.swap(i, j);
                                }
                                let mut tags: Vec<String> = Vec::with_capacity(cands.len());
                                let mut texts: Vec<String> = Vec::with_capacity(cands.len());
                                for (slot, (ep, is_live)) in cands.iter().enumerate() {
                                    tags.push(pool[slot % pool.len()].clone());
                                    // detokenize the episode TEXT. LIVE episodes carry the raw turn tokens
                                    // (ep.tokens, captured with forced BOS + trailing \n); curated episodes
                                    // have an ep.tok sidecar on disk. BOS (id 2) dropped; bytes joined.
                                    let eptok: Vec<i32> = if *is_live {
                                        ep.tokens.clone().unwrap_or_default()
                                    } else {
                                        std::fs::read_to_string(
                                            std::path::Path::new(&ep.dir).join("ep.tok"))
                                            .ok().map(|c| c.lines().filter_map(|l| l.trim().parse::<i32>().ok()).collect())
                                            .unwrap_or_default()
                                    };
                                    let mut bytes: Vec<u8> = Vec::new();
                                    for &t in eptok.iter() {
                                        if t == 2 { continue; } // skip forced BOS
                                        bytes.extend_from_slice(app.tokenizer.decode_token(t));
                                    }
                                    let txt = String::from_utf8_lossy(&bytes).trim().to_string();
                                    // fall back to the registry/live topic if tokens were unavailable.
                                    texts.push(if txt.is_empty() { ep.topic.clone() } else { txt });
                                }
                                // (2) Build the judge prompt (the validated harness template) + apply
                                // the gemma chat template via apply_template_ids (FINDING #1: the
                                // messages path = instruct behavior; a raw prompt base-completes/echoes).
                                let mut entries = String::new();
                                for (tg, tx) in tags.iter().zip(texts.iter()) {
                                    entries.push_str(&format!("[{tg}] {tx}\n"));
                                }
                                let judge_content = format!(
                                    "You are a memory index. Each entry below has a TAG in [brackets]. \
Read the QUESTION and reply with ONLY the tag of the single entry that directly \
answers it. If no entry answers it, reply [NULL].\n\n{entries}\nQUESTION: {query}\n\
Tag of the answer (or [NULL]):");
                                // JUDGE-SERVED (Exp E, 2026-06-25): SP_B3_VERIFY swaps the weak single-tag
                                // prompt for the skeptical TAG+EVIDENCE contract. The generative judge gets
                                // the PICK (~85%); the deterministic token_overlap gate (parse site below)
                                // turns it into a 95%-reject judge by demanding a verifiable citation.
                                // Default-off (SP_B3_VERIFY unset) => the single-tag prompt above = null floor.
                                let verify_mode = std::env::var("SP_B3_VERIFY").ok().filter(|s| !s.is_empty()).is_some();
                                let judge_content = if verify_mode {
                                    format!("You are a STRICT memory index. Each entry has a TAG in [brackets].\n\n{entries}\nQUESTION: {query}\n\nMost questions have NO matching entry. Find the entry that directly answers the question; if none does, answer NONE. Then reply on ONE line EXACTLY:\nTAG=<the tag, or NONE> | EVIDENCE=<copy the exact words from that entry that answer it>\nANSWER:")
                                } else { judge_content };
                                let jmsgs = vec![Message {
                                    role: "user".to_string(), content: judge_content }];
                                match app.tokenizer.apply_template_ids(&jmsgs) {
                                    Err(e) => tracing::warn!("B3-JUDGE: apply_template_ids failed: {e} -- clean prompt"),
                                    Ok(jtoks) if jtoks.len() < 2 => {
                                        tracing::warn!("B3-JUDGE: judge prompt too short ({}) -- clean prompt", jtoks.len());
                                    }
                                    Ok(jtoks) => {
                                        // (3) NESTED judge inference. The resident cache currently holds the
                                        // original prompt prefilled at [0, anchor)=head. We reset, run the
                                        // judge token-by-token, parse the tag, then reset + re-prefill head
                                        // to RESTORE the exact pre-branch cache state (no contamination of
                                        // the final synthesis). reset() is the SWA-ring-safe reset (rewind
                                        // past Jmax is the diagnosed ring bug).
                                        // JUDGE-SERVED: in verify mode FORCE the format by prefilling "TAG="
                                        // into the model turn. The dense reference model otherwise sometimes answers in prose
                                        // (the correct recalled fact, but no tag) and the deterministic parse
                                        // drops a VALID recall (the NIGHTSHIFT live-fact smoke: it reproduced
                                        // "5-RAVEN-9921" but as a sentence). Seed `reply` with "TAG=" so
                                        // parse_tag_evidence sees the forced prefix. Default path unchanged.
                                        let mut jtoks = jtoks;
                                        let mut reply = String::new();
                                        if verify_mode {
                                            if let Ok(tt) = app.tokenizer.encode("TAG=") {
                                                let tt: Vec<i32> = tt.into_iter().filter(|&t| t != 2).collect();
                                                if !tt.is_empty() { jtoks.extend_from_slice(&tt); reply.push_str("TAG="); }
                                            }
                                        }
                                        let _ = unsafe { kv::reset(handle) };
                                        let (jhead, jlast) = jtoks.split_at(jtoks.len() - 1);
                                        let mut judge_ok = jhead.is_empty()
                                            || unsafe { kv::prefill(handle, jhead) }.is_ok();
                                        if judge_ok {
                                            // greedy: decode_step(jlast) gives the first generated logits.
                                            let mut tok = jlast[0];
                                            let turn_stops = app.tokenizer.turn_stop_ids();
                                            // JUDGE-SERVED: EVIDENCE needs room to be copied; the single-tag
                                            // path still stops at 10. (eos / turn_stop break out earlier.)
                                            let jbudget: u32 = if verify_mode { 64 } else { 10 };
                                            // JUDGE-SERVED: the served sampler masks the model's suppress-tokens
                                            // (gemma image/audio soft tokens 258882/258883 + control markers), but
                                            // this raw-argmax side-pass bypassed it -> the placeholders leaked into
                                            // EVIDENCE and derailed the judge. Mask them to -inf here too (the
                                            // structural cure for the <image|>/<audio|> spam). verify-only.
                                            let suppress = if verify_mode { app.tokenizer.suppress_token_ids() } else { Vec::new() };
                                            for _ in 0..jbudget {
                                                if unsafe { kv::decode_step(handle, tok, logits) }.is_err() {
                                                    judge_ok = false; break;
                                                }
                                                if verify_mode {
                                                    for &sid in &suppress {
                                                        let u = sid as usize;
                                                        if u < logits.len() { logits[u] = f32::NEG_INFINITY; }
                                                    }
                                                }
                                                // greedy argmax (temperature 0, penalty 1.0).
                                                let mut bi = 0usize; let mut bv = f32::NEG_INFINITY;
                                                for (i, &v) in logits.iter().enumerate() { if v > bv { bv = v; bi = i; } }
                                                let g = bi as i32;
                                                if app.tokenizer.eos_ids.contains(&g) || turn_stops.contains(&g) { break; }
                                                reply.push_str(&String::from_utf8_lossy(app.tokenizer.decode_token(g)));
                                                tok = g;
                                                // JUDGE-SERVED verify reply is ONE line: stop at the first
                                                // newline so the decode can't ramble into echo / special-token
                                                // spam (the live-smoke false-reject). Single-tag path unchanged.
                                                if verify_mode && reply.trim_start().contains('\n') { break; }
                                            }
                                        }
                                        // G-INT-2-FIX: the judge's nested forward is a DESTRUCTIVE read of
                                        // the resident cache — it advanced dpos past the prompt anchor and
                                        // wrote judge K/V into global slots [n-1, jhead+J). A plain
                                        // reset()+prefill(head) only rewinds the counters and rewrites
                                        // [0,n-1); the judge residue lingers at slots >= n-1. On a NULL turn
                                        // synthesis sits at pos=n-1 so the residue is beyond dpos (never
                                        // read => clean), but on a PICK the injected memory advances dpos
                                        // forward and the synthesis window sweeps the stale judge slots =>
                                        // prompt-echo degeneration (the proven Phase-4 blocker). Restore via
                                        // reset_cold (zeroes every owner K/V + journal) so the reconstruction
                                        // truly starts cold; the inject (below) then lands on a clean head and
                                        // nothing of the judge pass can be attended. Byte-identical to the
                                        // null-floor for the NULL/no-inject path (zeroed slots beyond dpos
                                        // are never read after a fresh prefill).
                                        // INSTRUMENT (root-cause receipt): log dpos at each stage.
                                        let dpos_pre = unsafe { kv::position(handle) };
                                        let _ = unsafe { kv::reset_cold(handle) };
                                        let dpos_postreset = unsafe { kv::position(handle) };
                                        if !head.is_empty() {
                                            if let Err(e) = unsafe { kv::prefill(handle, head) } {
                                                tracing::error!("B3-JUDGE: FATAL re-prefill(head) failed: {e}");
                                            }
                                        }
                                        let dpos_posthead = unsafe { kv::position(handle) };
                                        tracing::info!("B3-JUDGE COLD-RESTORE: dpos pre-restore={} after reset_cold={} after prefill(head)={} (head.len={})",
                                            dpos_pre, dpos_postreset, dpos_posthead, head.len());
                                        // (4) Parse the tag (FINDING #3: copy-able TAG, never an ordinal).
                                        // Longest matching tag wins; [NULL]/no-tag ⇒ reject. Match on the
                                        // full "[M_X]" surface so M_A never substring-collides with M_AB.
                                        let mut picked: Option<usize> = None;
                                        if verify_mode {
                                            // JUDGE-SERVED Stage-2 (Exp E): parse TAG+EVIDENCE; accept the
                                            // pick ONLY if its cited span clears the deterministic
                                            // token-overlap gate (0.6) against that entry's text. This is the
                                            // 95%-reject lever — the picker proposes, the CPU math disposes.
                                            let (pk, ev) = recall::parse_tag_evidence(&reply, &tags);
                                            match pk {
                                                Some(i) => {
                                                    let ov = recall::token_overlap(&ev, &texts[i]);
                                                    if ov >= recall::OVERLAP_THR {
                                                        picked = Some(i);
                                                        tracing::info!("B3-VERIFY: PICK [{}] overlap={:.3} >= {:.2} ACCEPT ev={:?}",
                                                            tags[i], ov, recall::OVERLAP_THR, ev.trim());
                                                    } else {
                                                        tracing::info!("B3-VERIFY: tag [{}] overlap={:.3} < {:.2} REJECT (ungrounded citation) ev={:?}",
                                                            tags[i], ov, recall::OVERLAP_THR, ev.trim());
                                                    }
                                                }
                                                None => tracing::info!("B3-VERIFY: NONE/no-tag -> REJECT (clean prompt)"),
                                            }
                                        } else {
                                            let up = reply.to_uppercase();
                                            let mut best_len = 0usize;
                                            if !up.contains("NULL") {
                                                for (slot, tg) in tags.iter().enumerate() {
                                                    let needle = format!("[{}]", tg);
                                                    if up.contains(&needle.to_uppercase()) && needle.len() > best_len {
                                                        best_len = needle.len(); picked = Some(slot);
                                                    }
                                                }
                                                // tolerate a bare tag without brackets (model sometimes drops them)
                                                if picked.is_none() {
                                                    for (slot, tg) in tags.iter().enumerate() {
                                                        if up.contains(&tg.to_uppercase()) && tg.len() > best_len {
                                                            best_len = tg.len(); picked = Some(slot);
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                        match picked {
                                            None => tracing::info!(
                                                "B3-JUDGE: [NULL] (reply={:?}) -> REJECT (clean prompt, foreign)", reply.trim()),
                                            Some(slot) => {
                                                let (ep, _is_live) = &cands[slot];
                                                // (5) TEXT-IN-CONTEXT RECALL (G-INT-2-FIX, the Phase-4
                                                // sealer). The α-sweep PROVED latent injection of a live
                                                // episode has NO operating point that recites the specific
                                                // fact (high mass=echo-hijack, low mass=parametric-washout).
                                                // The generative judge works by READING TEXT, so synthesis
                                                // reads the recalled memory TEXT too: prepend the picked
                                                // episode's text (texts[slot]) to the user query, RAG-style,
                                                // re-template, and rebuild the cache from that augmented
                                                // prompt. The model then reads + recites the fact natively
                                                // through its own generative pipeline — NO lossy latent KV
                                                // inject for the payload. The Ring-2/XBAR substrate stays the
                                                // SELECTION mechanism (the judge PICK above); recitation goes
                                                // through the native pipeline. We DROP inject_tokens/replay
                                                // for the recitation path (operator decision).
                                                // AUTHORITY FIX (2026-07-01): deliver the CLEAN manifest text
                                                // (ep.text, the same field L5-direct delivers), NOT texts[slot]
                                                // — texts[slot] is the detokenized RAW ep.tok turn (captured
                                                // with forced BOS + trailing \n = chat-template cruft), which
                                                // polluted "Context (authoritative): ..." and dropped the model
                                                // back to parametric (right pick, wrong answer). Fall back to
                                                // the detokenized turn only for live episodes with no manifest.
                                                // REORDER (ADR-002 §8.1): the judge PICK is a PASS signal only.
                                                // Deliver the GLOBAL L5-#1 (l5_best, the 86.89% SP_RECALL_L5
                                                // selection), NOT the judge's own shortlist pick (measured 50%
                                                // — the judge picks #2 or the model resists). The judge already
                                                // did its job: PASS (some memory answers) vs NULL (foreign
                                                // reject). Fall back to the judge's pick text only when L5 mode
                                                // is off / L5 best unavailable (prior behavior, null floor).
                                                let (deliver_name, mem_text) = match l5_best.as_ref() {
                                                    Some((n, t)) if !t.trim().is_empty() => (n.clone(), t.clone()),
                                                    _ => (ep.name.clone(),
                                                          if !ep.text.trim().is_empty() { ep.text.clone() } else { texts[slot].clone() }),
                                                };
                                                // JUDGE-SERVED synthesis: rigid CLOSED-TASK framing. The prior
                                                // conversational "Using that context, answer:" wording induced a
                                                // template-continuation trap — the model answered, then echoed the
                                                // prompt / degenerated (Georgian, python) to fill max_tokens. A
                                                // closed instruction makes it answer concisely and emit the turn-stop
                                                // (the eos/turn_stop break at the synthesis loop then halts it).
                                                // DELIVERY FIX (2026-07-01): deliver through the EXACT clean
                                                // text-in-context format of the proven SP_RECALL_L5 / Jaccard
                                                // branches (faithfulness system prompt + "Context (authoritative,
                                                // current): <fact>\n\n<query>"). The prior single-user "Provide a
                                                // direct, concise answer" framing (NO system prompt) degenerated ->
                                                // tag echo + <image|>/Georgian garbage. The judge is a PICK/NULL
                                                // GATE; recitation goes through the proven 86.89% clean delivery.
                                                let aug_msgs = vec![
                                                    Message { role: "system".to_string(), content: "You are Shannon-Prime, a local AI with a real working memory. Keep replies short. Use facts you were given faithfully; if you don't know, say so.".to_string() },
                                                    Message { role: "user".to_string(), content: format!("Context (authoritative, current): {mem_text}\n\n{query}") },
                                                ];
                                                match app.tokenizer.apply_template_ids(&aug_msgs) {
                                                    Ok(aug_toks) if aug_toks.len() >= 2 => {
                                                        // Establish the synthesis cache from the augmented
                                                        // prompt via the SAME single clean reconstruction the
                                                        // null path uses: reset_cold (zero every owner K/V +
                                                        // journal) -> prefill(aug_head). The synthesis decode
                                                        // loop then starts from decode_step(syn_last) where
                                                        // syn_last = aug_toks[last]. The recalled fact is now
                                                        // in the model's context window.
                                                        // ROOT-CAUSE PROBE (2026-07-01): log reset_cold's result + the
                                                        // resulting dpos. If reset_cold fails or leaves dpos!=0, the aug
                                                        // is prefilled on the judge's polluted cache -> authority loss.
                                                        match unsafe { kv::reset_cold(handle) } {
                                                            Ok(_) => tracing::info!("B3-JUDGE: reset_cold OK, dpos={}", unsafe { kv::position(handle) }),
                                                            Err(e) => tracing::error!("B3-JUDGE: reset_cold FAILED: {e} (dpos={})", unsafe { kv::position(handle) }),
                                                        }
                                                        let (aug_head, aug_last) = aug_toks.split_at(aug_toks.len() - 1);
                                                        let mut ok = true;
                                                        if !aug_head.is_empty() {
                                                            if let Err(e) = unsafe { kv::prefill(handle, aug_head) } {
                                                                tracing::error!("B3-JUDGE: FATAL aug prefill failed: {e} -- clean prompt");
                                                                ok = false;
                                                            }
                                                        }
                                                        if ok {
                                                            syn_last = aug_last[0];
                                                            // provenance/telemetry reflect the DELIVERED fact
                                                            // (deliver_name = L5-#1 on reorder; = judge pick on fallback).
                                                            recalled = Some((deliver_name.clone(), 1000));
                                                            judge_ground = Some((deliver_name.clone(), mem_text.clone()));
                                                            tracing::info!(
                                                                "B3-JUDGE: PASS (judge PICK [{}] '{}' topic='{}' reply={:?}) -> DELIVER '{}' TEXT-IN-CONTEXT (aug n={}, dpos={}) -> synthesis recites natively",
                                                                tags[slot], ep.name, ep.topic, reply.trim(), deliver_name, aug_toks.len(),
                                                                unsafe { kv::position(handle) });
                                                        } else {
                                                            // aug reconstruction failed: fall back to the
                                                            // original clean prompt (reset_cold + prefill(head))
                                                            // so synthesis is at least coherent (null-floor-like).
                                                            let _ = unsafe { kv::reset_cold(handle) };
                                                            if !head.is_empty() {
                                                                let _ = unsafe { kv::prefill(handle, head) };
                                                            }
                                                        }
                                                    }
                                                    other => {
                                                        if let Err(e) = other.map(|_| ()) {
                                                            tracing::warn!("B3-JUDGE: aug apply_template_ids failed: {e} -- clean prompt");
                                                        } else {
                                                            tracing::warn!("B3-JUDGE: aug prompt too short -- clean prompt");
                                                        }
                                                        // leave the restored clean cache (head already prefilled
                                                        // by the cold-restore above) + syn_last == last[0].
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        // B3-v9c THE DISPOSER (Stage-2 post-inject semantic verify-and-rewind).
                        // The q·K Stage-1 ranker (above) PROPOSES; the Disposer DISPOSES. For
                        // each candidate episode it SPECULATIVELY injects (M_target budget if set)
                        // into the resident cache, teacher-forces a Yes/No reasoning bridge via
                        // decode_step, reads margin = logit(Yes)-logit(No) at the answer position,
                        // then O(1)-rewinds the probe (last token + bridge + episode) to vaporise
                        // it (the KAI-1b slot==pos / SWA-journal inverse). The best-margin episode
                        // is re-injected and recalled IFF margin > SP_B3_DISPOSER_TAU. TELEMETRY-
                        // FIRST (the codebase's tau_qk discipline): TAU defaults to +inf so the
                        // first run NEVER fires — we read the per-(query,episode) margin matrix,
                        // verify open-world separation (the v6/v7 Yes/No bridge FAILED in-prompt;
                        // this is the post-LATENT-inject variant, re-measured), THEN pin TAU.
                        // The M_target=42 budget is the safety floor: a wrong FIRE stumbles
                        // (sub-dominant), it does not hijack. Default-off ⇒ falls through to the
                        // existing q·K fire (null floor).
                        // P1b: disposer/knockout lanes = RESEARCH (two-stage/margin-NULL refuted)
                        let disp_mode = if int2_decided || !cfg!(feature = "legacy_policy") { 0 } else { std::env::var("SP_B3_DISPOSER").ok()
                            .and_then(|s| s.trim().parse::<i32>().ok()).unwrap_or(0) };
                        if disp_mode == 2 {
                            // B3-v10 ABLATION GATE (the Thermodynamic Knockout). Per candidate: inject E,
                            // greedily decode the 8-tok payload, then memset-zero the episode positions
                            // whose tokens match the payload window (ep.tok sidecar), re-score the SAME
                            // payload, and measure collapse = Σ(LL_ablated - LL_E) over [2,8). A TRUE match
                            // is structurally load-bearing ⇒ catastrophic NEGATIVE collapse; a parametric
                            // bleed / empty-match shrugs it off ⇒ ≈0. The audio super-attractor has no
                            // ep.tok ⇒ empty mask ⇒ collapse 0 ⇒ fails the knockout (lexical limit ENFORCES
                            // the semantic boundary). Fire iff collapse < SP_B3_DISPOSER_TAU (telemetry -inf).
                            let abl_tau: f32 = std::env::var("SP_B3_DISPOSER_TAU").ok()
                                .and_then(|s| s.parse().ok()).unwrap_or(f32::NEG_INFINITY);
                            const KGEN: usize = 8;
                            let lse = |z: &[f32]| -> f32 {
                                let m = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                                let mut s = 0.0f32; for &v in z { s += (v - m).exp(); } m + s.ln()
                            };
                            let argmax = |z: &[f32]| -> usize {
                                let mut bi = 0usize; let mut bv = f32::NEG_INFINITY;
                                for (i, &v) in z.iter().enumerate() { if v > bv { bv = v; bi = i; } } bi
                            };
                            // v12 TEACHER-FORCED KNOCKOUT: SP_B3_SECRET=<exact secret string>. Instead of
                            // greedy-decoding (which may not recite the secret in [2,8)), teacher-force the
                            // KNOWN secret tokens and score THEIR NLL with vs without the episode's source
                            // rows. Measures the secret's dependency DIRECTLY; window = the full secret.
                            // SECRET resolution is PER-EPISODE (resolved inside the loop below): env
                            // SP_B3_SECRET is the single-shot override; otherwise each episode supplies
                            // its own knockout target via an ep.secret sidecar (the corpus admission
                            // path — ONE daemon boot labels the whole registry, each ep self-tested with
                            // ITS OWN secret). Missing sidecar => empty => that ep falls back to the
                            // greedy [2,8) window (no crash, null floor preserved).
                            let env_secret = std::env::var("SP_B3_SECRET").ok().filter(|s| !s.is_empty());
                            let anchor = unsafe { kv::position(handle) };
                            let mut best_abl: Option<(usize, f32)> = None;
                            let mut drows: Vec<String> = Vec::with_capacity(registry.len());
                            for (i, ep) in registry.iter().enumerate() {
                                if ep.npos <= 0 || ep.gk.is_empty() { continue; }
                                let eptok: Vec<i32> = std::fs::read_to_string(std::path::Path::new(&ep.dir).join("ep.tok"))
                                    .ok().map(|s| s.lines().filter_map(|l| l.trim().parse::<i32>().ok()).collect())
                                    .unwrap_or_default();
                                // per-episode knockout target: env override else this ep's ep.secret
                                // sidecar (leading space preserved; only trailing CR/LF stripped).
                                let secret_ids: Vec<i32> = env_secret.clone()
                                    .or_else(|| std::fs::read_to_string(std::path::Path::new(&ep.dir).join("ep.secret")).ok())
                                    .map(|s| { let s = s.trim_end_matches(|c: char| c == '\n' || c == '\r').to_string();
                                               app.tokenizer.encode(&s).unwrap_or_default() })
                                    .map(|mut v| { if v.first() == Some(&2) { v.remove(0); } v })
                                    .unwrap_or_default();
                                let w0: usize = if secret_ids.is_empty() { 2 } else { 0 };
                                // Leg 1: inject E, greedy-decode the payload, record lp_E.
                                if unsafe { kv::replay(handle, &ep.dir, ep.npos, false) }.is_err() { continue; }
                                let mut gen: Vec<i32> = Vec::new();
                                let mut lpe: Vec<f32> = Vec::new();
                                let mut tok = last[0];
                                if secret_ids.is_empty() {
                                    for _ in 0..KGEN {
                                        if unsafe { kv::decode_step(handle, tok, logits) }.is_err() { break; }
                                        let g = argmax(logits); lpe.push(logits[g] - lse(logits)); gen.push(g as i32); tok = g as i32;
                                    }
                                } else {
                                    for &s in &secret_ids {   // teacher-force the KNOWN secret
                                        if unsafe { kv::decode_step(handle, tok, logits) }.is_err() { break; }
                                        lpe.push(logits[s as usize] - lse(logits)); gen.push(s); tok = s;
                                    }
                                }
                                let ng = gen.len();
                                let _ = unsafe { kv::rewind(handle, ng as i32) };   // undo payload, keep E
                                if ng == 0 { drows.push(format!("{}(collapse=nan)", ep.name)); let _ = unsafe { kv::rewind(handle, ep.npos) }; continue; }
                                // Target positions: episode indices whose token matches a payload-window token.
                                let mut targets: Vec<i32> = Vec::new();
                                if !eptok.is_empty() {
                                    let want: std::collections::HashSet<i32> = gen[w0.min(ng)..ng].iter().copied().collect();
                                    for (p, &t) in eptok.iter().enumerate() {
                                        if p >= ep.npos as usize { break; }
                                        if want.contains(&t) { targets.push(p as i32); }
                                    }
                                    if targets.len() > 12 { targets.truncate(12); }
                                }
                                // Leg 2: ablate targets, teacher-force the SAME payload, record lp_ablated.
                                let _ = unsafe { kv::ablate(handle, anchor, &targets) };
                                let mut lpa: Vec<f32> = Vec::with_capacity(ng);
                                let mut tok = last[0];
                                for i2 in 0..ng {
                                    if unsafe { kv::decode_step(handle, tok, logits) }.is_err() { break; }
                                    lpa.push(logits[gen[i2] as usize] - lse(logits)); tok = gen[i2];
                                }
                                let _ = unsafe { kv::rewind(handle, lpa.len() as i32 + ep.npos) };   // clear payload + episode (restores ablated rows)
                                let n = lpe.len().min(lpa.len());
                                let mut collapse = 0.0f32;
                                for j in w0..n { collapse += lpa[j] - lpe[j]; }
                                drows.push(format!("{}(collapse={:.2},ntgt={})", ep.name, collapse, targets.len()));
                                let better = match best_abl { None => true, Some((_, b)) => collapse < b };
                                if better { best_abl = Some((i, collapse)); }
                            }
                            tracing::info!("B3-DISPOSER ABLATION collapse=ΣΔLL_ablated[2,{}) (more-neg=load-bearing): [{}] TAU={:.3}", KGEN, drows.join(" "), abl_tau);
                            if let Some((idx, collapse)) = best_abl {
                                let ep = &registry[idx];
                                let fire = collapse < abl_tau && ep.npos > 0 && !ep.gk.is_empty();
                                tracing::info!("B3-DISPOSER: best='{}' (topic='{}') collapse={:.3} ⇒ {}",
                                    ep.name, ep.topic, collapse, if fire { "AUTHORIZE (load-bearing, re-inject)" } else { "REJECT (rewound; clean prompt)" });
                                if fire {
                                    if let Err(e) = unsafe { kv::replay(handle, &ep.dir, ep.npos, false) } {
                                        tracing::warn!("B3-DISPOSER: ablation re-inject({}, {}) failed: {e} — proceeding without recall", ep.dir, ep.npos);
                                    } else {
                                        recalled = Some((ep.name.clone(), ((-collapse).max(0.0) * 1000.0) as u32));
                                    }
                                }
                            }
                        } else if disp_mode == 1 {
                            let disp_tau: f32 = std::env::var("SP_B3_DISPOSER_TAU").ok()
                                .and_then(|s| s.parse().ok()).unwrap_or(f32::INFINITY);
                            // v9h: ΔLL polarity. +1 (default) = argmax payload-ΔLL (current); -1 =
                            // argmin = the episode that DISRUPTS the continuation MOST (the
                            // "truth-is-surprising" hypothesis). Toggle to adjudicate on the metal.
                            let dcont_sign: f32 = std::env::var("SP_B3_DCONT_SIGN").ok()
                                .and_then(|s| s.parse().ok()).unwrap_or(1.0);
                            // MULTI-TOKEN Δ-CONTINUATION signal (v9e). First-token Δcont was blind
                            // (1/3): biographical answers open with boilerplate ("Robert"/"He"); the
                            // memory-specific facts ("The Bill","Herons") surface positions 3-8. So
                            // measure the SEMANTIC PAYLOAD: under REAL E greedily decode K tokens =
                            // the model's own continuation g[0..K); then under ZEROED E (same npos,
                            // null content) teacher-force THAT SAME sequence and read each token's
                            // log-prob. signal = Σ ΔLL = Σ (log p_real(g_i) - log p_zero(g_i)) over
                            // the PAYLOAD window i∈[2,K) (skip the boilerplate). A true match forces
                            // tokens that are HIGHLY SURPRISING to the zeroed cache (large +ΔLL); a
                            // mismatch yields the prompt-driven continuation the zeroed cache already
                            // expects (~0). This is the teacher-forced ΔLL on the model's OWN answer —
                            // content-aware, not a raw distribution magnitude, so the per-episode
                            // content-mass bias is suppressed. Ranks on the payload-window ΔLL.
                            const KGEN: usize = 8;
                            let lse = |z: &[f32]| -> f32 {
                                let m = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                                let mut s = 0.0f32; for &v in z { s += (v - m).exp(); } m + s.ln()
                            };
                            let argmax = |z: &[f32]| -> usize {
                                let mut bi = 0usize; let mut bv = f32::NEG_INFINITY;
                                for (i, &v) in z.iter().enumerate() { if v > bv { bv = v; bi = i; } } bi
                            };
                            let mut best_disp: Option<(usize, f32)> = None;
                            let mut drows: Vec<String> = Vec::with_capacity(registry.len());
                            for (i, ep) in registry.iter().enumerate() {
                                if ep.npos <= 0 || ep.gk.is_empty() { continue; }
                                // Leg 1: REAL E — greedily decode the continuation, record its log-probs.
                                if unsafe { kv::replay(handle, &ep.dir, ep.npos, false) }.is_err() { continue; }
                                let mut gen: Vec<i32> = Vec::with_capacity(KGEN);
                                let mut lpr: Vec<f32> = Vec::with_capacity(KGEN);
                                let mut tok = last[0];
                                for _ in 0..KGEN {
                                    if unsafe { kv::decode_step(handle, tok, logits) }.is_err() { break; }
                                    let g = argmax(logits);
                                    lpr.push(logits[g] - lse(logits));
                                    gen.push(g as i32);
                                    tok = g as i32;
                                }
                                let _ = unsafe { kv::rewind(handle, gen.len() as i32 + ep.npos) };
                                if gen.is_empty() { drows.push(format!("{}(pay=nan)", ep.name)); continue; }
                                // Leg 2: ZEROED E — teacher-force the SAME sequence, read its log-probs.
                                if unsafe { kv::replay(handle, &ep.dir, ep.npos, true) }.is_err() { continue; }
                                let mut lpz: Vec<f32> = Vec::with_capacity(gen.len());
                                let mut tok = last[0];
                                for i2 in 0..gen.len() {
                                    if unsafe { kv::decode_step(handle, tok, logits) }.is_err() { break; }
                                    lpz.push(logits[gen[i2] as usize] - lse(logits));
                                    tok = gen[i2];
                                }
                                let _ = unsafe { kv::rewind(handle, lpz.len() as i32 + ep.npos) };
                                // ΔLL = log p_real - log p_zero, summed over the payload window [2,n).
                                let n = lpr.len().min(lpz.len());
                                let (mut dll_all, mut dll_pay) = (0.0f32, 0.0f32);
                                for j in 0..n { let d = lpr[j] - lpz[j]; dll_all += d; if j >= 2 { dll_pay += d; } }
                                drows.push(format!("{}(pay={:.2},all={:.2})", ep.name, dll_pay, dll_all));
                                let better = match best_disp { None => true, Some((_, b)) => dcont_sign * dll_pay > dcont_sign * b };
                                if better { best_disp = Some((i, dll_pay)); }
                            }
                            tracing::info!("B3-DISPOSER Δcont(multi-tok) ΣΔLL pay[2,{})/all: [{}] TAU={:.3}", KGEN, drows.join(" "), disp_tau);
                            if let Some((idx, dll)) = best_disp {
                                let ep = &registry[idx];
                                let fire = dll > disp_tau && ep.npos > 0 && !ep.gk.is_empty();
                                tracing::info!("B3-DISPOSER: best='{}' (topic='{}') ΣΔLL_pay={:.3} ⇒ {}",
                                    ep.name, ep.topic, dll, if fire { "AUTHORIZE (re-inject)" } else { "REJECT (rewound; clean prompt)" });
                                if fire {
                                    // SAFETY: handle live; cache Mutex held.
                                    if let Err(e) = unsafe { kv::replay(handle, &ep.dir, ep.npos, false) } {
                                        tracing::warn!("B3-DISPOSER: authorized re-inject({}, {}) failed: {e} — proceeding without recall", ep.dir, ep.npos);
                                    } else {
                                        recalled = Some((ep.name.clone(), (dll.max(0.0) * 1000.0) as u32));
                                    }
                                }
                            }
                        } else if let Some((idx, score)) = (if int2_decided { None } else { best }) {
                            let ep = &registry[idx];
                            let fire = score >= tau_qk && ep.npos > 0 && !ep.gk.is_empty();
                            tracing::info!(
                                "B3-v2 recall: best='{}' (topic='{}') topm={:.3} TAU={:.3} ⇒ {}",
                                ep.name, ep.topic, score, tau_qk,
                                if fire { "FIRE (auto-replay)" } else { "REJECT (below TAU; no replay)" }
                            );
                            if fire {
                                // SAFETY: handle live; we hold the cache Mutex.
                                if let Err(e) = unsafe { kv::replay(handle, &ep.dir, ep.npos, false) } {
                                    tracing::warn!("B3-v2 recall: auto-replay({}, {}) failed: {e} — proceeding without recall", ep.dir, ep.npos);
                                } else {
                                    recalled = Some((ep.name.clone(), (score * 1000.0) as u32));
                                }
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!("B3-v2 recall: read_global_q failed: {e} — proceeding without recall");
                    }
                }
            }
        } else {
            tracing::info!("B3-v2 recall: auto_recall set but no registry loaded (SP_RECALL_REGISTRY) — no-op");
        }
    }
    // Surface the recall decision on the SSE stream as a structured event so the
    // console can show a "recalled episode" indicator. Emitted before the answer
    // tokens (a separate event name so it never appears in the text delta).
    if let Some((ref name, score)) = recalled {
        let payload = serde_json::json!({"recalled": name, "agree": score, "chat_id": chat_id});
        let _ = tx.blocking_send(Ok(Event::default().event("recall").data(payload.to_string())));
    } else if auto_recall && replay_dir.is_none() {
        let payload = serde_json::json!({"recalled": serde_json::Value::Null, "chat_id": chat_id});
        let _ = tx.blocking_send(Ok(Event::default().event("recall").data(payload.to_string())));
    }

    // B2 (§6d-b): XBAR episode REPLAY into the live turn. Recall a stored episode's
    // owner K/V into the cache at [dpos,dpos+npos) right after the prompt prefill,
    // so the last prompt token + every generated token attend across the recalled
    // memory (SP_REPLAY, C2 #222). Done under the cache Mutex; reject = rewind. A
    // turn that names no episode skips this entirely (byte-identical null floor).
    if let Some(ref dir) = replay_dir {
        if !dir.is_empty() && replay_npos > 0 {
            // SAFETY: handle live; we hold the cache Mutex (no concurrent decode).
            if let Err(e) = unsafe { kv::replay(handle, dir, replay_npos, false) } {
                send_err(format!("kvdecode replay({dir}, {replay_npos}): {e}"));
                sessions.remove(chat_id);
                return;
            }
        }
    }
    if logits.len() != vocab_size {
        send_err("kvdecode: logits buffer size mismatch".into());
        sessions.remove(chat_id);
        return;
    }
    // ATTR-GATE ZERO-INFERENCE SYMBOLIC DECLINE (ADR-002 Tier-2 executor). The recall
    // stage decided the delivered fact does NOT state the queried attribute (private
    // entity, attribute-absent). Emit the deterministic decline and RETURN before the
    // synthesis forward — the gemma4 decode loop never runs, so confabulation/leak is
    // mathematically impossible and the turn resolves at string-allocation speed.
    // reset_cold keeps the persistent KV clean for the next turn.
    if let Some(msg) = symbolic_decline.take() {
        let payload = serde_json::to_string(&ChatDelta { delta: msg, chat_id }).unwrap_or_default();
        let _ = tx.blocking_send(Ok(Event::default().data(payload)));
        let _ = unsafe { kv::reset_cold(handle) };
        let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
        let _ = app.events_tx.send(DaemonEvent::Chat { chat_id, status: "done" });
        tracing::info!("ATTR-DECLINE: zero-inference symbolic decline streamed (no gemma4 decode)");
        sessions.remove(chat_id);
        return;
    }
    // ── GEODESIC G-FM-STEER (ADR-003 §4.2) — SP_STEER_VEC=<raw f32 LE bin> +
    // SP_STEER_ALPHA=<f32>. Arms persistent pre-head steering for THIS turn's
    // synthesis when a recall fired (the F3 treatment was measured on recall
    // turns); explicitly DISARMS on non-recall turns so a prior arm never leaks
    // across turns on the resident session. Env unset ⇒ the verb is never called
    // ⇒ byte-identical null floor (session calloc's steer_active=0).
    // P1b: FM steering = RESEARCH (migration-map C2/W_c/steering row)
    if let (Some(vf), Some(al)) = (std::env::var("SP_STEER_VEC").ok().filter(|_| cfg!(feature = "legacy_policy")),
            std::env::var("SP_STEER_ALPHA").ok().and_then(|s| s.trim().parse::<f32>().ok())) {
        static STEER_VEC: std::sync::OnceLock<Vec<f32>> = std::sync::OnceLock::new();
        let v = STEER_VEC.get_or_init(|| match std::fs::read(&vf) {
            Ok(b) => b.chunks_exact(4)
                      .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            Err(e) => { tracing::warn!("FM-STEER: read {vf} failed ({e}) — steering disabled"); Vec::new() }
        });
        let steer_on = recalled.is_some() && !v.is_empty() && al != 0.0;
        match unsafe { kv::steer_set(handle,
                if steer_on { v.as_slice() } else { &[] },
                if steer_on { al } else { 0.0 }) } {
            Ok(()) => { if steer_on {
                tracing::info!("FM-STEER: armed alpha={al} dim={} (recall turn)", v.len());
            } }
            Err(e) => tracing::warn!("FM-STEER: steer_set failed ({e})"),
        }
    }
    // ── GEODESIC F3 CAPTURE (ADR-003 §5) — SP_F3_CAPTURE=<dir>; unset ⇒ nothing armed
    // ⇒ byte-identical null floor. Two one-shot taps of the post-output_norm hidden
    // (kv::capture_feat_arm): (1) armed on the syn_last step directly below = the
    // answer-turn LAST-PROMPT-TOKEN state (the feature the LM head consumes to pick
    // the first answer token — for a recall turn this is the state under the delivered
    // fact + faithfulness prompt; for a clean turn, the parametric state); (2) armed on
    // the first in-loop step = the FIRST-ANSWER-TOKEN state. Zero extra forwards — the
    // tap is two D2H copies on steps the turn runs anyway. Files + meta written at
    // turn end (after the decode loop). Offline-batch rail ONLY: capture_feat commits
    // like any decode (routes.rs:787 note is about PRE-cache routing, not this site).
    const F3_E: usize = 3840; // the dense reference model hidden (UNIFICATION substrate map)
    let f3_dir = std::env::var("SP_F3_CAPTURE").ok().filter(|s| !s.is_empty())
        .filter(|_| cfg!(feature = "legacy_policy"));  // P1b: F3 capture rail = RESEARCH
    let mut f3_prompt: Vec<f32> = Vec::new();
    let mut f3_first: Vec<f32> = Vec::new();
    if f3_dir.is_some() {
        f3_prompt = vec![0f32; F3_E];
        if let Err(e) = unsafe { kv::capture_feat_arm(handle, &mut f3_prompt) } {
            tracing::warn!("F3-CAPTURE: arm(prompt) failed ({e}) — capture skipped this turn");
            f3_prompt = Vec::new();
        }
    }
    // -- COCONUT continuous-thought seed (ADR-B1 section 5) -- SP_COCONUT=<N>; unset/0
    // => nothing armed => byte-identical null floor. Reuse the SAME one-shot post-final
    // -norm tap F3 uses so the syn_last decode_step below writes the LAST-PROMPT hidden
    // (the Coconut seed) at zero extra forward cost. F3 keeps precedence (single tap slot).
    let coconut_n: usize = if cfg!(feature = "legacy_policy") {  // P1b: COCONUT = RESEARCH
        std::env::var("SP_COCONUT").ok()
            .and_then(|s| s.trim().parse::<usize>().ok()).unwrap_or(0)
    } else { 0 };
    let mut coconut_seed: Vec<f32> = Vec::new();
    if coconut_n > 0 && f3_prompt.is_empty() {
        coconut_seed = vec![0f32; F3_E];
        if let Err(e) = unsafe { kv::capture_feat_arm(handle, &mut coconut_seed) } {
            tracing::warn!("SP_COCONUT: seed arm failed ({e}) -- thoughts skipped this turn");
            coconut_seed = Vec::new();
        }
    }
    // G-INT-2-FIX: synthesis starts from syn_last (== last[0] for null/clean turns;
    // the augmented prompt's last token when the B3-JUDGE PICKed a memory text-in-context).
    if let Err(e) = unsafe { kv::decode_step(handle, syn_last, logits) } {
        // F3: an armed capture fires inside the step; on step FAILURE the armed pointer
        // may persist on the session — leak the buffer so a late write can never hit
        // freed memory (dangling-write guard; error path only, serve is aborting anyway).
        if !f3_prompt.is_empty() { std::mem::forget(std::mem::take(&mut f3_prompt)); }
        send_err(format!("kvdecode decode_step(prefill-tail): {e}"));
        sessions.remove(chat_id);
        return;
    }

    // -- COCONUT continuous thoughts (ADR-B1 section 5, zero-CUDA variant) --------------
    // SP_COCONUT=N (N>0): before decoding the answer, run N continuous thoughts. Each
    // thought feeds the previous position's post-final-norm hidden back as the next input
    // residual (kv::inject_frames advances the resident KV by 1 and mints K/V), so the N
    // thought positions become latent context the answer attends to. Thoughts emit NO
    // stream tokens; the decode loop below samples the first answer token from the current
    // logits (syn_last's) and every answer token attends the thought KV. SP_COCONUT unset
    // or 0 => coconut_seed empty => whole block skipped => byte-identical null floor.
    if coconut_n > 0 && coconut_seed.len() == F3_E {
        let coconut_scale: f32 = std::env::var("SP_COCONUT_SCALE").ok()
            .and_then(|s| s.trim().parse::<f32>().ok()).unwrap_or(1.0);
        let mut h = coconut_seed.clone();
        let mut norms: Vec<f32> = Vec::with_capacity(coconut_n);
        let mut ok = true;
        for _ in 0..coconut_n {
            let mut hs = h.clone();
            for v in hs.iter_mut() { *v *= coconut_scale; }
            let mut h_next = vec![0f32; F3_E];
            // arm the tap so the inject below writes the injected position's post-norm hidden.
            if let Err(e) = unsafe { kv::capture_feat_arm(handle, &mut h_next) } {
                tracing::warn!("SP_COCONUT: thought arm failed ({e})"); ok = false; break;
            }
            // inject the (scaled) hidden as 1 new residual position; KV += 1; tap fires.
            if let Err(e) = unsafe { kv::inject_frames(handle, &hs, 1, inject_ph) } {
                std::mem::forget(std::mem::take(&mut h_next)); // dangling-write guard on failed inject
                tracing::warn!("SP_COCONUT: thought inject failed ({e})"); ok = false; break;
            }
            let n2: f32 = h_next.iter().map(|x| x * x).sum::<f32>().sqrt();
            norms.push(n2);
            h = h_next; // continuous thought: next input = this hidden
        }
        if let Ok(dump) = std::env::var("SP_COCONUT_DUMP") {
            if !dump.is_empty() {
                let mut s = format!("# SP_COCONUT n={} scale={}\n", coconut_n, coconut_scale);
                for nm in &norms { s.push_str(&format!("{}\n", nm)); }
                let _ = std::fs::write(&dump, s);
            }
        }
        eprintln!("[SP_COCONUT] injected {} thoughts (scale {}), ok={}, norms={:?}",
            norms.len(), coconut_scale, ok, norms);
    }

    // ── SPECTEST v1 (SP_SPECTEST=1, default-off = byte-identical null floor) ────
    // On a text-in-context delivery turn, HOLD the stream: the full draft is decoded
    // and grounding-tested BEFORE any byte reaches the client (decide in latent/
    // symbol, execute clean — the draft is speculative until tested). PASS releases
    // the held deltas verbatim; VETO suppresses the draft and executes the clean
    // symbolic answer from the record. env unset OR no delivery ⇒ spectest=false ⇒
    // every send below is byte-for-byte the existing path.
    let spectest = std::env::var("SP_SPECTEST").ok().as_deref() == Some("1")
        && recalled_text.is_some();
    let mut held: Vec<String> = Vec::new();
    // ── SPECTEST v2 — the LINEAR TEST-HEAD (G-TESTHEAD-OFFLINE REAL: 0.901 mean
    // held-out-mode AUC, signal linear ⇒ the whole pipeline collapses to score =
    // x·v + c on the frame-1 state). SP_SPECTEST_HEAD=<SPH1 blob> (v[3840] + c +
    // tau); score < tau joins the veto (head ∪ lexical). Loaded once per process;
    // env unset ⇒ None ⇒ v1.1 lexical-only behavior unchanged.
    static SPH: std::sync::OnceLock<Option<(Vec<f32>, f32, f32)>> = std::sync::OnceLock::new();
    let sph: Option<&(Vec<f32>, f32, f32)> = if spectest {
        SPH.get_or_init(|| {
            let p = std::env::var("SP_SPECTEST_HEAD").ok().filter(|s| !s.is_empty())?;
            let b = std::fs::read(&p).map_err(|e| {
                tracing::warn!("SPECTEST-HEAD: read {p} failed ({e}) — head disabled");
            }).ok()?;
            const VLEN: usize = 3840 * 4;
            if b.len() < 16 + VLEN + 8 || &b[..4] != b"SPH1" {
                tracing::warn!("SPECTEST-HEAD: bad blob {p} — head disabled");
                return None;
            }
            let v: Vec<f32> = b[16..16 + VLEN].chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            let c0 = f32::from_le_bytes(b[16 + VLEN..16 + VLEN + 4].try_into().ok()?);
            let tau = f32::from_le_bytes(b[16 + VLEN + 4..16 + VLEN + 8].try_into().ok()?);
            tracing::info!("SPECTEST-HEAD: loaded (dim=3840 c={c0:.4} tau={tau:.4})");
            Some((v, c0, tau))
        }).as_ref()
    } else { None };
    let mut sph_x: Vec<f32> = Vec::new(); // frame-1 state for the head (armed in-loop)

    let tokenizer = app.tokenizer.clone();
    // A2-polish: this arch has no `<end_of_turn>` token; its turn boundary is
    // the `<|turn>`/`<turn|>` token. Treat those ids as EOS-equivalent so the
    // resident-cache the dense reference model chat stops cleanly at the turn (the marker never decodes
    // into the stream). Belt-and-braces with default_stops()'s stop-strings.
    let turn_stop_ids = tokenizer.turn_stop_ids();
    // SP_EOT_DEBUG (read-only): on free-gen turns, log the BEST stop-token RANK (# vocab
    // logits strictly greater) from the RAW forward logits per generated token. High rank
    // throughout => the model never wants to end (forward/mapping bug); low rank but not
    // chosen => sampler/top_k. Default-off => zero overhead.
    let eot_dbg = judge_ground.is_none()
        && std::env::var("SP_EOT_DEBUG").ok().as_deref() == Some("1");
    let eot_stop_ids: Vec<usize> = tokenizer.eos_ids.iter().chain(turn_stop_ids.iter())
        .map(|&x| x as usize).filter(|&x| x < logits.len()).collect();
    let stop_rank = |lg: &[f32]| -> (i32, usize) {
        let mut bid = -1i32; let mut bl = f32::NEG_INFINITY;
        for &s in &eot_stop_ids { if lg[s] > bl { bl = lg[s]; bid = s as i32; } }
        if bid < 0 { (-1, usize::MAX) } else { (bid, lg.iter().filter(|&&v| v > bl).count()) }
    };
    let mut eot_gen: Vec<i32> = Vec::new();
    let mut eot_ranks: Vec<usize> = Vec::new();
    // ── KAIROS CONTINUATION SIGNAL (2026-07-12) ─────────────────────────────────────
    // The impulse to keep talking is ALREADY COMPUTED on every turn — and thrown away
    // one line later.
    //
    // At each step the forward produces a full logit vector. The gap between the best
    // STOP token and the best NON-stop token is exactly "how much more did she have to
    // say?". A large positive margin = she is done and knows it. A margin near zero (or
    // negative, i.e. she only stopped because SP_EOT_BIAS tipped her) = she stopped
    // RELUCTANTLY, mid-thought. That is the continuation impulse, free, no second model,
    // no decode, no event tape.
    //
    // It must be read on the RAW logits (here, line ~3781) — BEFORE eot_bias is added at
    // ~3782 — or the bias we inject to make her stop would masquerade as her wanting to.
    // The engine already computes a stop_rank() for SP_EOT_DEBUG and discards it; this is
    // the same observation, kept.
    //
    // Default-off: SP_KAIROS=1 arms it. Off ⇒ not computed ⇒ byte-identical + free.
    let kairos_on = std::env::var("SP_KAIROS").as_deref() == Ok("1");
    let eot_margin_of = |lg: &[f32]| -> f32 {
        let mut best_stop = f32::NEG_INFINITY;
        for &s in &eot_stop_ids { if lg[s] > best_stop { best_stop = lg[s]; } }
        if !best_stop.is_finite() { return f32::NAN; }
        let mut best_other = f32::NEG_INFINITY;
        for (i, &v) in lg.iter().enumerate() {
            if v > best_other && !eot_stop_ids.contains(&i) { best_other = v; }
        }
        best_stop - best_other
    };
    // the margin at the step that PRODUCED the token we are about to emit
    let mut kairos_margin: f32 = f32::NAN;
    // PERSIST-KV: the tokens we commit to the KV this turn (each is decode_step'd in the loop
    // body below). Appended to the prompt to form the next turn's reusable committed prefix.
    let mut committed_gen: Vec<i32> = Vec::new();
    if eot_dbg { eot_ranks.push(stop_rank(logits).1); }
    // SP_EOT_BIAS: the forward drives the end-of-turn token to ~rank 1 at a real turn
    // boundary but a hair short of winning, so the model never stops and degenerates.
    // A small fixed bias on the stop tokens tips a rank-1 boundary to chosen WITHOUT
    // firing mid-answer (where eot sits at rank ~1000, far out of reach). 0 = null floor.
    let eot_bias: f32 = eot_bias_req.unwrap_or_else(|| std::env::var("SP_EOT_BIAS").ok()
        .and_then(|s| s.trim().parse().ok()).unwrap_or(0.0));
    if eot_bias != 0.0 { for &s in &eot_stop_ids { logits[s] += eot_bias; } }
    let mut next_token = sampler.sample(logits);
    sampler.observe(next_token);
    // B3-JUDGE grounding: accumulate the synthesized answer for the post-hoc check.
    let mut answer_text = String::new();
    let decode_t0 = std::time::Instant::now(); // LM-B2b: turn decode timing (n_out / decode_s = tok/s)
    // G-PERF: close the RECALL phase here — everything above (L5 search + B3 judge
    // forwards + replay) is pre-decode work that used to hide inside `other`.
    let t_recall_ms = t_recall0.elapsed().as_millis();
    let recall_steps = kv::step_count().saturating_sub(steps_before_recall);
    let mut t_gen_ms: u128 = 0;   // wall time at the last emitted token (excludes post-decode)

    'decode: for _ in 0..max_tokens {
        t_gen_ms = decode_t0.elapsed().as_millis();
        if (!tokenizer.eos_ids.is_empty() && tokenizer.eos_ids.contains(&next_token))
            || turn_stop_ids.contains(&next_token)
        {
            break 'decode;
        }
        if eot_dbg && eot_gen.len() < 64 { eot_gen.push(next_token); }

        let token_bytes = tokenizer.decode_token(next_token);
        // JUDGE-SERVED hard stop (recall turns only): the dense reference model won't self-terminate the
        // text-in-context synthesis and degenerates AFTER the correct answer (code
        // fences, newlines, repetition). A concise factual recall answer never contains
        // a newline or backtick — the moment one appears, halt BEFORE streaming it so
        // the babble never reaches the client. judge_ground gate => normal chat unchanged.
        if judge_ground.is_some() {
            let tb = String::from_utf8_lossy(token_bytes);
            if tb.contains('\n') || tb.contains('`') { break 'decode; }
        }
        let stop_hit = match dec_buf.push(token_bytes) {
            PushResult::Emit(bytes) => {
                if !bytes.is_empty() {
                    let text = String::from_utf8_lossy(&bytes).into_owned();
                    answer_text.push_str(&text);
                    if spectest {
                        held.push(text); // SPECTEST: stream held until the draft is tested
                    } else {
                        let payload = serde_json::to_string(&ChatDelta { delta: text, chat_id })
                            .unwrap_or_default();
                        if !send_deadline(tx, Event::default().data(payload)) {
                            cancel_child.store(1, Ordering::Relaxed);
                            let _ = app.events_tx.send(DaemonEvent::Chat { chat_id, status: "cancelled" });
                            sessions.remove(chat_id);
                            return;
                        }
                    }
                }
                false
            }
            PushResult::Stopped(bytes) => {
                if !bytes.is_empty() {
                    let text = String::from_utf8_lossy(&bytes).into_owned();
                    if spectest {
                        held.push(text); // SPECTEST: hold (mirrors the non-spectest send exactly)
                    } else {
                        let payload = serde_json::to_string(&ChatDelta { delta: text, chat_id })
                            .unwrap_or_default();
                        let _ = send_deadline(tx, Event::default().data(payload));
                    }
                }
                true
            }
        };

        app.tokens_decoded.fetch_add(1, Ordering::Relaxed);

        if stop_hit {
            break 'decode;
        }

        // PERSIST-KV: record this token as committed -- the decode_step below writes its K/V into
        // the cache, so it becomes part of the reusable committed prefix for the next turn. Placed
        // AFTER every early break above so committed_gen never lists a token the cache doesn't hold.
        committed_gen.push(next_token);
        // GEODESIC F3 tap (2): first in-loop step = the FIRST-ANSWER-TOKEN state.
        // Armed immediately before the unconditional decode_step below so the one-shot
        // ALWAYS fires into a live buffer. f3_first: [] = not yet armed; len==F3_E =
        // captured; len==1 = failed-sentinel (never retried, never written).
        if !f3_prompt.is_empty() && f3_first.is_empty() {
            let mut b = vec![0f32; F3_E];
            match unsafe { kv::capture_feat_arm(handle, &mut b) } {
                Ok(()) => f3_first = b, // move keeps the heap buffer address — armed ptr stays valid
                Err(e) => { tracing::warn!("F3-CAPTURE: arm(first) failed ({e})"); f3_first = vec![f32::NAN]; }
            }
        }
        // SPECTEST-HEAD tap: the same frame-1 arming for the live test-head, when the
        // F3 rail isn't holding the (single) capture slot this turn (F3 precedence;
        // the gate launchers never run both). Same dangling-write discipline.
        if sph.is_some() && f3_prompt.is_empty() && sph_x.is_empty() {
            let mut b = vec![0f32; F3_E];
            match unsafe { kv::capture_feat_arm(handle, &mut b) } {
                Ok(()) => sph_x = b,
                Err(e) => { tracing::warn!("SPECTEST-HEAD: arm failed ({e})"); sph_x = vec![f32::NAN]; }
            }
        }
        // Feed the just-emitted token; get logits for the next position.
        // SAFETY: handle live; logits is vocab_size f32 (checked above).
        if let Err(_e) = unsafe { kv::decode_step(handle, next_token, logits) } {
            // F3 dangling-write guard (see the syn_last site): step failed with a
            // possibly-armed capture — leak the buffer rather than risk a late write.
            if f3_first.len() == F3_E { std::mem::forget(std::mem::take(&mut f3_first)); }
            if sph_x.len() == F3_E { std::mem::forget(std::mem::take(&mut sph_x)); }
            // A TERMINAL-SYNC FAULT ADVANCED dpos BEFORE FAILING (2026-08-29 audit,
            // F2): the engine increments dpos_host, THEN syncs — so this position's
            // K/V may come from a faulted launch while the lengths still agree. The
            // turn is dirty; the commit below must not describe this cache.
            cache_dirty = true;
            break 'decode;
        }
        if eot_dbg && eot_ranks.len() < 64 { eot_ranks.push(stop_rank(logits).1); }
        // KAIROS: read the impulse on the RAW logits, BEFORE the bias below tips her.
        if kairos_on { kairos_margin = eot_margin_of(logits); }
        if eot_bias != 0.0 { for &s in &eot_stop_ids { logits[s] += eot_bias; } }
        next_token = sampler.sample(logits);
        sampler.observe(next_token);
    }

    if eot_dbg {
        let ranks: Vec<String> = eot_ranks.iter().take(40).map(|r| r.to_string()).collect();
        let gids: Vec<String> = eot_gen.iter().take(40).map(|g| g.to_string()).collect();
        tracing::info!("EOT-DEBUG: stop_ids={:?} ngen={} stop_rank_per_step=[{}] gen_ids=[{}]",
            eot_stop_ids, eot_gen.len(), ranks.join(","), gids.join(","));
    }
    // KAIROS: hand the turn's continuation impulse to the scheduler. `eot_margin` is the
    // raw logit gap (stop − best alternative) at the step that ended the turn:
    //     >> 0  she is finished and knows it            -> silence
    //     ~ 0   she stopped on the edge, mid-thought    -> she has more to say
    //     <  0  she only stopped because eot_bias tipped her -> she was CUT OFF
    // Emitted as its own SSE event so the text stream stays byte-identical for every
    // client that does not care. Off (SP_KAIROS unset) ⇒ never sent.
    if kairos_on {
        let payload = serde_json::json!({
            "eot_margin": if kairos_margin.is_finite() { serde_json::json!(kairos_margin) }
                          else { serde_json::Value::Null },
            "n_gen": committed_gen.len(),
            "eot_bias": eot_bias,
            "chat_id": chat_id,
        });
        tracing::info!("KAIROS: turn ended — eot_margin={:.3} n_gen={} (margin<=0 => she was cut off; ~0 => more to say)",
                       kairos_margin, committed_gen.len());
        // ── WAS SHE INTERRUPTED MID-THOUGHT? (2026-08-30, observability only) ──────
        // The thought CEILING closes `<channel|>` by masking every other logit, which
        // is an interruption mid-reasoning — and the suspicion is that the model then
        // carries on reasoning in the SPEECH channel, where it reaches him, the room
        // and the record. ~13% of her recorded turns open as analysis about him rather
        // than speech to him (tools/voice_leak_rate.py, ten-day baseline). One line per
        // turn is what lets those two rates be COMPARED instead of argued about: if
        // this fires on the turns that leak, the fix belongs at the ceiling; if it does
        // not, the hypothesis dies here rather than becoming a story someone acts on.
        if sampler.think_was_forced() {
            tracing::info!("THINK-CEILING: the thought was CUT by the budget, not closed by her \
                            — watch this turn's opening words for reasoning that kept going");
        }
        let _ = tx.blocking_send(Ok(Event::default().event("kairos").data(payload.to_string())));
    }

    let flushed = dec_buf.flush();
    if !flushed.is_empty() {
        let text = String::from_utf8_lossy(&flushed).into_owned();
        answer_text.push_str(&text);
        if spectest {
            held.push(text); // SPECTEST: hold the tail flush too
        } else {
            let payload = serde_json::to_string(&ChatDelta { delta: text, chat_id })
                .unwrap_or_default();
            let _ = tx.blocking_send(Ok(Event::default().data(payload)));
        }
    }

    // ── SPECTEST v1 DECISION (draft→test→veto→clean-execute; ADR-002-native) ────
    // Deterministic grounding test (the B3-JUDGE salient-token rule): the held
    // draft must USE the delivered fact. PASS ⇒ release the held stream verbatim
    // (client sees byte-identical output to a no-spectest serve). VETO ⇒ the
    // ungrounded draft NEVER reaches the client; execute the clean symbolic answer
    // FROM THE RECORD — leak-impossible by construction (its only content is the
    // delivered fact text). v2 (pre-scoped, needs the decode-loop extraction):
    // rewind + re-decode under an escalated delivery frame instead of the
    // symbolic execute. Bounded by SELECTION: a cross-picked fact yields a
    // faithful-to-the-wrong-record fallback, same as systemecho today.
    if spectest {
        let fact = recalled_text.as_deref().unwrap_or("");
        const SPECTEST_STOP: &[&str] = &["the","and","for","that","with","this","from",
            "are","was","were","your","you","have","will","when","which","what","into",
            "over","each","now"];
        // v1.1 (measured fix): v1 grounded on ANY salient fact token — but a parametric
        // draft ("...capital of France is Paris") reuses the fact's SUBJECT tokens and
        // passed, leaking the parametric VALUE (14/61 leaks through PASS, receipt
        // G-SPECTEST-V1). Ground on FACT-ONLY tokens: salient words of the record that
        // do NOT appear in the user's query — the subject arrives via the query, the
        // VALUE doesn't (the attr-gate's query/fact asymmetry, reused).
        let user_lc = raw_user.as_deref().unwrap_or("").to_lowercase();
        let salient: Vec<String> = fact.split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() >= 4).map(|w| w.to_lowercase())
            .filter(|w| !SPECTEST_STOP.contains(&w.as_str()))
            .filter(|w| !user_lc.contains(w.as_str()))
            .collect();
        let ans_lc = answer_text.to_lowercase();
        let n_used = salient.iter().filter(|w| ans_lc.contains(w.as_str())).count();
        // v2: the linear test-head's verdict on the frame-1 state (score < tau ⇒
        // the draft is a predicted leak — the value-substitution class the lexical
        // rule provably cannot catch). Veto = lexical ∪ head. Head unavailable
        // (blob unset / capture missed / 1-token answer) ⇒ lexical-only (v1.1).
        let (head_score, head_veto) = match sph {
            Some((v, c0, tau)) if sph_x.len() == F3_E => {
                let s: f32 = sph_x.iter().zip(v.iter()).map(|(a, b)| a * b).sum::<f32>() + c0;
                tracing::info!("SPECTEST-HEAD: score={s:.4} tau={tau:.4} -> {}",
                    if s < *tau { "VETO(head)" } else { "pass(head)" });
                (Some(s), s < *tau)
            }
            _ => (None, false),
        };
        // VETO AUTHORITY (live-fix 2026-07-03, the "Hodor" incident): the HEAD is
        // PRIMARY whenever it scored — the lexical rule only decides when the head
        // is unavailable. Rationale (from the live log): on a BACKGROUND delivery
        // (absent=1.00, off-topic record) the draft rightly ignores the record ⇒
        // lexical reads 0/N ⇒ the old union-veto replaced a perfectly good answer
        // with the wrong record. The head passed those drafts (+1.9, +4.8) — it
        // measures the state, not token overlap, and V2/V3 proved its leak-catch.
        let lexical_ungrounded = !salient.is_empty() && n_used == 0;
        let veto = match head_score { Some(_) => head_veto, None => lexical_ungrounded };
        if !veto {
            tracing::info!("SPECTEST: PASS ({}/{} salient, head={:?}, authority={}) -> releasing held stream",
                n_used, salient.len(), head_score,
                if head_score.is_some() { "head" } else { "lexical" });
            for text in held.drain(..) {
                let payload = serde_json::to_string(&ChatDelta { delta: text, chat_id })
                    .unwrap_or_default();
                if !send_deadline(tx, Event::default().data(payload)) { break; }
            }
        } else {
            tracing::warn!("SPECTEST: VETO ({} — draft suppressed) -> clean symbolic execute from the record",
                if head_veto { format!("head score {:.3} < tau", head_score.unwrap_or(f32::NAN)) }
                else { format!("lexical 0/{} salient (head unavailable)", salient.len()) });
            held.clear();
            let fb = format!("From the record: {fact}");
            let payload = serde_json::to_string(&ChatDelta { delta: fb.clone(), chat_id })
                .unwrap_or_default();
            let _ = send_deadline(tx, Event::default().data(payload));
            answer_text = fb;
        }
    }

    // ── GEODESIC F3 CAPTURE — persist the pair + meta (ADR-003 §5, G-F3-CAPTURE).
    // File f3_<chat_id>.bin: magic "F3P1" + E:u32 + nframes:u32 + pad:u32, then
    // nframes×E LE f32 (frame 0 = last-prompt-token state, frame 1 = first-answer-
    // token state when captured). One f3_meta.jsonl row per turn (the harness joins
    // by `user` text); f3_env.txt dumped once per serve (the re-baseline law:
    // receipts carry the full SP_* env).
    if let Some(ref f3d) = f3_dir {
        if !f3_prompt.is_empty() {
            let _ = std::fs::create_dir_all(f3d);
            let envp = std::path::Path::new(f3d).join("f3_env.txt");
            if !envp.exists() {
                let mut ed = String::new();
                for (k, v) in std::env::vars() {
                    if k.starts_with("SP_") || k.starts_with("CUBLAS") {
                        ed.push_str(&format!("{k}={v}\n"));
                    }
                }
                let _ = std::fs::write(&envp, ed);
            }
            let has_first = f3_first.len() == F3_E;
            let nframes: u32 = if has_first { 2 } else { 1 };
            let mut buf = Vec::with_capacity(16 + F3_E * nframes as usize * 4);
            buf.extend_from_slice(b"F3P1");
            buf.extend_from_slice(&(F3_E as u32).to_le_bytes());
            buf.extend_from_slice(&nframes.to_le_bytes());
            buf.extend_from_slice(&0u32.to_le_bytes());
            for &x in &f3_prompt { buf.extend_from_slice(&x.to_le_bytes()); }
            if has_first { for &x in &f3_first { buf.extend_from_slice(&x.to_le_bytes()); } }
            let _ = std::fs::write(
                std::path::Path::new(f3d).join(format!("f3_{chat_id}.bin")), buf);
            let mode = if recalled.is_some() {
                std::env::var("SP_RECALL_L5_PROMPT").unwrap_or_else(|_| "recite".into())
            } else { "clean".to_string() };
            let meta = serde_json::json!({
                "chat_id": chat_id,
                "mode": mode,
                "recalled": recalled.as_ref().map(|(n, c)|
                    serde_json::json!({"ep": n, "cos_milli": c})),
                "user": raw_user.as_deref().unwrap_or(""),
                "answer": answer_text,
                "n_gen": committed_gen.len(),
                "frames": nframes,
            });
            use std::io::Write as _;
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true)
                .open(std::path::Path::new(f3d).join("f3_meta.jsonl")) {
                let _ = writeln!(f, "{meta}");
            }
            tracing::info!("F3-CAPTURE: f3_{}.bin frames={} mode={} recalled={:?}",
                chat_id, nframes, meta["mode"], recalled);
        }
    }

    // ── B3-JUDGE GROUNDING CHECK (CPU-side, telemetry) ──────────────────────────
    // If the generative judge injected a memory, verify the synthesized answer
    // actually USES the memory's salient nouns. If the model ignored the injected
    // payload and answered parametrically, flag a parametric-hallucination (the
    // discrete-substrate reject the operator specified). Telemetry-only: it does
    // not alter the (already-streamed) answer; it surfaces the discrepancy in the
    // daemon log so a wrong/ignored recall is auditable.
    if let Some((ref ep_name, ref mem_text)) = judge_ground {
        let ans_lc = answer_text.to_lowercase();
        // salient tokens = the memory's words >=4 chars, minus common stopwords.
        const STOP: &[&str] = &["the","and","for","that","with","this","from","are",
            "was","were","authorizes","requires","access","code","recovery","using",
            "your","you","have","will","when","which","what","into","over","each"];
        let salient: Vec<String> = mem_text.split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() >= 4)
            .map(|w| w.to_lowercase())
            .filter(|w| !STOP.contains(&w.as_str()))
            .collect();
        let n_salient = salient.len();
        let n_used = salient.iter().filter(|w| ans_lc.contains(w.as_str())).count();
        let grounded = n_salient == 0 || n_used > 0;
        if grounded {
            tracing::info!("B3-JUDGE grounding: OK '{}' used {}/{} salient mem tokens", ep_name, n_used, n_salient);
        } else {
            tracing::warn!("B3-JUDGE grounding: PARAMETRIC-HALLUCINATION FLAG '{}' -- answer used 0/{} salient mem tokens (model ignored injected memory)", ep_name, n_salient);
        }
    }

    // LM-B2b TURN-OUTCOME telemetry: for a delivered recall turn, log the OUTPUT + timing +
    // an obeyed self-check, joined by query to the decision record. PRIVACY: when the recalled
    // entry is a private-secret, the query AND output are REDACTED (hash+len, never raw text).
    if let Some(q) = recalled_query.as_ref() {
        telem_emit_turn(q, recalled.as_ref().map(|(n, _)| n.as_str()).unwrap_or("-"),
            recalled_class.as_deref().unwrap_or("-"), &answer_text, recalled_text.as_deref(),
            committed_gen.len(), decode_t0.elapsed().as_secs_f64(), recalled_secret, Some(&app.events_tx));
    }

    let is_cancelled = cancel_child.load(Ordering::Relaxed) != 0;
    if is_cancelled {
        let _ = tx.blocking_send(Ok(Event::default().event("cancelled").data("{}")));
        let _ = app.events_tx.send(DaemonEvent::Chat { chat_id, status: "cancelled" });
    } else {
        let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
        let _ = app.events_tx.send(DaemonEvent::Chat { chat_id, status: "done" });
    }

    // ── B4 NIGHTSHIFT: between-turn memory CONSOLIDATION ────────────────────────
    // After the assistant reply has finished decoding for THIS turn, capture the
    // raw USER message as a LIVE episode so the W_c head can self-select it on a
    // LATER turn (the chat GROWS its memory). The capture is a position-0 STANDALONE
    // prefill on a SCRATCH session (NOT a read of the live conversation cache) so the
    // episode-K has the same provenance as the curated registry-K the head trained on.
    // Env-gated SP_B4_NIGHTSHIFT=1; unset ⇒ this whole hook is skipped ⇒ byte-identical
    // null floor. We still hold the resident-cache Mutex (`guard`/`handle` from the top
    // of this fn) for the WHOLE capture, so no concurrent decode races the scratch open.
    // A capture failure logs a warning and does NOT break the (already-sent) response.
    // TODO(B4-v2): admit via the teacher-forced ablation oracle (collapse < TAU=-8)
    // before append — v1 captures EVERY sufficiently-long user turn (no relevance gate).
    // LAYER-3 DECIDE: carries the just-captured (episode_name, text) out of the
    // NIGHTSHIFT block to the DECIDE pass below, so the model can judge whether the
    // new fact supersedes an existing memory. None unless a capture happened this turn.
    let mut decide_new: Option<(String, String)> = None;
    if std::env::var("SP_B4_NIGHTSHIFT").ok().as_deref() == Some("1") {
        // B4-v2 ADMISSION GATE (interim): only ASSERTIONS become memories. Skip a turn
        // that was (a) answered FROM memory (recalled.is_some() => a query consuming a
        // memory, not a new fact) or (b) interrogative (ends with '?'). This kills the
        // junk-question capture that let a true-foreign query false-recall a stored
        // question (the live-smoke regression). The rigorous gate is the teacher-forced
        // ablation oracle (collapse < TAU=-8), which ALSO rejects parametric facts — the
        // TODO above; this cheap gate is the safety floor it will subsume.
        // A forget command turn must NEVER become a memory (it removed one; storing
        // "Forget the secret vault code." re-pollutes the registry). Exclude both the
        // matched case (forget_done) and any forget phrasing (matched or not).
        let is_forget_turn = raw_user.as_ref().map(|s| {
            let l = s.to_lowercase();
            l.contains("forget") || l.contains("delete that") || l.contains("erase")
        }).unwrap_or(false);
        // LAYER-3: capture STATEMENTS even when a related memory was recalled (so a
        // superseding fact like "my favorite color is now green" IS stored and the
        // DECIDE pass below can retire the old one). Skip interrogatives via wh-word /
        // '?' detection (the spider-question false-recall fix), NOT the recalled.is_none()
        // coupling that previously blocked the supersede case. N2 stays LOOSE by design.
        let lc = raw_user.as_ref().map(|s| s.trim().to_lowercase()).unwrap_or_default();
        let first_word = lc.split_whitespace().next().unwrap_or("");
        // Skip questions AND request-imperatives ("tell me about X", "describe...") --
        // they are not facts to store. wh-words + '?' + common request verbs/prefixes.
        let is_question = lc.ends_with('?')
            || lc.starts_with("tell me") || lc.starts_with("do you") || lc.starts_with("can you")
            || matches!(first_word,
                "what"|"who"|"whom"|"whose"|"where"|"when"|"why"|"how"|"which"
                |"tell"|"describe"|"list"|"show"|"explain"|"summarize"|"recall"|"remind");
        // HINDSIGHT 2026-07-10 admission hygiene (live console found the registry full of
        // junk): (1) NEVER capture fenced content — the gateway's tool rounds arrive as
        // role:user ```tool_output blocks and were being stored as "memories" (then
        // recalled as stale tool results: "what time is it?" answered from a captured
        // get_time output); (2) require >= 4 WORDS — greetings/acks ("hi there!",
        // "she is good") captured as memories and spammed recall on chit-chat. Short
        // durable facts still reach memory via the explicit store verb / remember() tool.
        let is_fenced = lc.starts_with("```") || lc.contains("```tool_output");
        let n_words = lc.split_whitespace().count();
        // P1-seal admission UPPER cap (G-KAIROS-PERF battery, 2026-07-11): a
        // 2,100-char pasted turn ground a synchronous 195 MiB capture_batched
        // (minutes at 100% GPU) while the NEXT turn queued on the session — the
        // battery's 69 s tool turn and 119 s cold turn were mostly this. A
        // "memory" is a durable FACT, not a pasted document: cap admission at
        // 120 words (curated episodes are ~10-30). Long content still reaches
        // memory via the explicit store verb, which the operator invokes
        // deliberately and can afford to wait for.
        const ADMIT_MAX_WORDS: usize = 120;
        // ── B4 ADMISSION v2 (2026-07-12): A MEMORY IS ABOUT SOMEONE ──────────────────
        // The old gate admitted ANY declarative of 4..120 words that wasn't a question.
        // That is a firehose, and the audit showed what it caught: of 487 registry rows,
        // 404 were auto-capture and ~375 were voice/ASR TEST CORPUS —
        //     "The kind nurse painted the tall building as the sun went down."
        //     "A lonely sailor polished the garden as the church bells rang."
        // — plus, latterly, my own debug probes. All perfectly grammatical, all
        // declarative, all 4..120 words, all about NOBODY. They then surfaced mid-answer
        // as "recalled memories", which is the recall-misfire we keep chasing.
        //
        // A durable fact is about a PERSON: it carries a personal reference. This is the
        // same discriminator that cleanly split the registry in tools/memory/triage.py
        // (157 rows fell out as "impersonal declarative"). Impersonal sentences are still
        // TRANSCRIPT — they just are not FACTS, and they no longer enter long-term memory.
        // Anything genuinely worth keeping still gets there two ways that outrank this:
        // the explicit store verb, and remember() / remember_about_self(), which she now
        // actually uses. Escape hatch: SP_B4_ADMIT_PERSONAL=0 restores the old firehose.
        let personal_ref = {
            const PRONOUNS: &[&str] = &["i", "i'm", "i've", "im", "my", "me", "mine", "myself",
                                        "you", "you're", "your", "yours", "we", "our", "us"];
            lc.split(|c: char| !c.is_alphanumeric() && c != '\'')
                .any(|w| PRONOUNS.contains(&w))
        };
        let require_personal = std::env::var("SP_B4_ADMIT_PERSONAL").as_deref() != Ok("0");
        let admit = !forget_done
            && !is_forget_turn
            && !is_question
            && !is_fenced
            && n_words >= 4
            && n_words <= ADMIT_MAX_WORDS
            && (!require_personal || personal_ref)
            && !lc.is_empty();
        if require_personal && !personal_ref && !is_question && !is_fenced
            && n_words >= 4 && n_words <= ADMIT_MAX_WORDS && !lc.is_empty() {
            tracing::info!("B4-NIGHTSHIFT: skip (impersonal — a sentence, not a memory): '{}'",
                           lc.chars().take(56).collect::<String>());
        }
        if !lc.is_empty() && (is_fenced || ((n_words < 4 || n_words > ADMIT_MAX_WORDS) && !is_question)) {
            tracing::info!("B4-NIGHTSHIFT: skip (admission hygiene: fenced={is_fenced} words={n_words})");
        }
        if let Some(text) = raw_user.as_ref().map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty()).filter(|_| admit) {
            // (a) tokenize the RAW user content (no chat template — match the curator).
            // PROVENANCE FIX (B4-v3): byte-match the curator's sp_tok_enc EXACTLY. The
            // curator captured each needle from a `.txt` file that ends in a trailing
            // newline ("...6428.\n") with the gemma4 C encoder's FORCED leading BOS
            // (id 2) KEPT — yielding e.g. npos=22 for ep_n_div_000 (tok[0]=2 BOS,
            // tok[-1]=107 newline). The prior live path trimmed the text (no trailing
            // newline) AND stripped the BOS, losing BOTH boundary tokens (npos=20) — a
            // 2-token tokenization mismatch that produced a DIFFERENT ep.k and collapsed
            // the W_c score (live 0.084 vs curated 9.858 on identical text). So: append
            // the trailing "\n" before encoding and DO NOT strip the auto-prepended BOS.
            let text_nl = format!("{text}\n");
            match app.tokenizer.encode(&text_nl) {
                Ok(toks) => {
                    // KEEP the forced-BOS (id 2): the curator's ep.tok starts with it.
                    let ntok = toks.len();
                    if ntok < 4 {
                        tracing::info!("B4-NIGHTSHIFT: skip (too short, ntok={ntok})");
                    } else {
                        // (b) qm = the session's borrowed qwen3_model* (shares loaded weights).
                        let qm = {
                            let mut sguard = app.session.as_ref().expect("L1 session unavailable (qwen36 lane)").lock().unwrap();
                            let sraw = sguard.raw_ptr() as *mut sp_daemon::ffi_l1::sp_session;
                            // SAFETY: session is locked + valid for this borrow.
                            (unsafe { sp_daemon::ffi_l1::sp_session_qwen3_model(sraw) }) as *const std::ffi::c_void
                        };
                        if qm.is_null() {
                            tracing::warn!("B4-NIGHTSHIFT: sp_session_qwen3_model NULL — capture skipped");
                        } else {
                            // (c) Option-2 PROVENANCE FIX: capture through the SAME batched
                            // `gemma4_decode_cuda` forward the curator used. It writes
                            // ep.k/ep.v/ep.mf into a temp episode dir under the engine, so the live
                            // episode's K is BYTE-COMPATIBLE with the curated registry and the deployed
                            // W_c head selects it with ZERO retraining (no scratch per-token prefill,
                            // no K-norm calibration). gemma4_decode_cuda reads SP_XBAR_SWA_RING /
                            // SP_XBAR_SWA_W / SP_BYTEEXACT — NONE of which the nightshift launcher sets
                            // (it sets SP_DAEMON_KVDECODE_RING_W, a different var), so this runs in the
                            // curator's clean full-cache float config and matches the curated ep.k. We
                            // hold the resident-cache Mutex (`guard`/`handle`) for the whole capture, so
                            // the SP_XBAR_RECALL_WRITE set+unset inside the glue is serialized.
                            // LIVE-FIX (2026-07-03, name-collision bug): the old name was the
                            // in-memory nightshift INDEX, which resets every serve while the
                            // persisted registry accumulates — a restarted serve minted a second
                            // ep_live_000, OVERWRITING the first episode's dir (production-
                            // registry corruption, found live). Millis-unique naming, same
                            // scheme as capture_live_episode.
                            let ep_uniq = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .map(|d| d.as_millis()).unwrap_or(0);
                            let ep_name = format!("ep_live_m{ep_uniq}");
                            let engine_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                                .parent().and_then(|p| p.parent())
                                .unwrap_or_else(|| std::path::Path::new("."));
                            let dir = engine_root
                                .join("_nightshift_live")
                                .join(&ep_name);
                            let dir_str = dir.to_string_lossy().to_string();
                            if let Err(e) = std::fs::create_dir_all(&dir) {
                                tracing::warn!("B4-NIGHTSHIFT: mkdir {dir_str} failed: {e} — no episode appended");
                            } else {
                                // SAFETY: qm valid (session held above); we hold the resident-cache
                                // Mutex so no concurrent device forward / SP_XBAR_RECALL_WRITE env race.
                                { let p = dir.join("ep.txt"); let _ = std::fs::write(&p, &text); let q = dir.join("ep.tok"); let _ = std::fs::write(&q, toks.iter().map(|t| t.to_string()).collect::<Vec<_>>().join("\n")); }
                                let cap_rc = unsafe { kv::capture_batched(qm, &toks, &dir_str) };
                                match cap_rc {
                                    Err(e) => tracing::warn!("B4-NIGHTSHIFT: batched capture failed: {e} — no episode appended"),
                                    Ok(()) => {
                                        // Load the just-written ep.k as global-K with the SAME extractor
                                        // the curated registry uses (byte-compatible by construction).
                                        match sp_daemon::recall::load_episode_global_k(&dir_str, ntok as i32) {
                                            None => tracing::warn!(
                                                "B4-NIGHTSHIFT: load_episode_global_k('{dir_str}') -> None — no episode appended"),
                                            Some((gk, ng)) => {
                                                // G-INT-2 STEP 1: compute the LIVE episode's REAL C2 sig
                                                // from its just-captured global-K, using the SAME Projection
                                                // (SEED/R_BITS/HD) the curated registry sigs come from, so live
                                                // and curated sigs are directly Hamming-comparable. npos here is
                                                // the actual position count packed in gk (load_episode_global_k
                                                // clamps ntok to the captured Pmax), derived from gk.len().
                                                let npos_sig = if ng > 0 { gk.len() / (ng * sp_daemon::recall::hd()) } else { 0 };
                                                let sig = if npos_sig > 0 {
                                                    app.recall_proj.signature(&gk, ng, npos_sig)
                                                } else { [0u64; 4] };
                                                // B4-SEAL: mint the L5 query-key at capture time so the GROWN
                                                // episode is immediately visible to the deployed L5 selector
                                                // (and to load_episode_l5key after restart via <dir>/ep.l5).
                                                // QKEYS: question-space key (SP_QKEY_MINT=1) with statement fallback.
                                                let l5k = mint_ep_l5(&app, qm, &toks, &text, &dir);
                                                // NIGHTSHIFT auto-classification (#73): classify on the RAW text.
                                                let mem_class: Option<&'static str> = if std::env::var("SP_MEM_CLASSIFY").as_deref() == Ok("1") {
                                                    let mc = sp_daemon::recall::classify_mem_class(&text);
                                                    tracing::info!("MEM-CLASSIFY: '{}' -> mem_class={}", text.chars().take(48).collect::<String>(), mc);
                                                    Some(mc)
                                                } else { None };
                                                // #72 UNIFY: emit the MEM-OKF concept sidecar for this episode dir.
                                                if let Some(mc) = mem_class { sp_daemon::recall::write_episode_okf(&dir, &text, mc, &ep_name); }
                                                // N3 PERSIST: append this live episode to the active registry
                                                // file so it survives a daemon restart (default-off
                                                // SP_NIGHTSHIFT_PERSIST=1 = null floor). Done here, before `sig`
                                                // is moved into the Episode below. On restart load_registry reads
                                                // it back (curated path; ep.k/ep.v/ep.mf already on disk at dir).
                                                if std::env::var("SP_NIGHTSHIFT_PERSIST").ok().as_deref() == Some("1") {
                                                    if let Ok(reg_path) = std::env::var("SP_RECALL_REGISTRY") {
                                                        let sig_hex = format!("{:016x}{:016x}{:016x}{:016x}", sig[3], sig[2], sig[1], sig[0]);
                                                        // LIVE-FIX (perspective bug): store the DELIVERED text ATTRIBUTED
                                                        // ("The user said: …"); selection artifacts stay on the RAW text.
                                                        let mut row = serde_json::Map::new();
                                                        row.insert("name".into(), ep_name.clone().into());
                                                        row.insert("dir".into(), dir_str.clone().into());
                                                        row.insert("npos".into(), (ntok as i32).into());
                                                        row.insert("topic".into(), text.clone().into());
                                                        // MEM-OKF v2 (2026-07-12): the OWNER lives in `speaker`, not inside the
                                                        // sentence. The old "The user said: …" prefix put ownership in the TEXT,
                                                        // where every reader had to re-derive it — and since 404/405 rows carried
                                                        // it, the ONLY voice in her long-term memory was the user's. That is why
                                                        // she slid into speaking as him. It also silently broke the harness's
                                                        // exact-duplicate guard (the compare no longer matched). Bare text on
                                                        // disk; harness lifecycle.render() frames it at read time; recall.rs:214
                                                        // already strips the legacy prefix, so old rows still work.
                                                        row.insert("text".into(), text.clone().into());
                                                        row.insert("speaker".into(), "user".into());
                                                        row.insert("lifecycle".into(), 0.into());
                                                        row.insert("src".into(), "auto-capture".into());
                                                        row.insert("sig_bits".into(), sig_hex.into());
                                                        if let Some(mc) = mem_class { row.insert("mem_class".into(), mc.into()); }
                                                        let line = serde_json::Value::Object(row).to_string();
                                                        use std::io::Write as _;
                                                        match std::fs::OpenOptions::new().create(true).append(true).open(&reg_path) {
                                                            Ok(mut f) => { let _ = writeln!(f, "{line}");
                                                                tracing::info!("B4-NIGHTSHIFT-PERSIST: appended {} -> {}", ep_name, reg_path); }
                                                            Err(e) => tracing::warn!("B4-NIGHTSHIFT-PERSIST: append {} failed: {e}", reg_path),
                                                        }
                                                    }
                                                }
                                                let mut ns = app.nightshift.write().unwrap();
                                                tracing::info!(
                                                    "B4-NIGHTSHIFT: {} C2-sig = {:016x}... (was [0;4])",
                                                    ep_name, sig[0]);
                                                let topic: String = text.chars().take(40).collect();
                                                // tokens: Some(toks) so the recall side still injects via
                                                // kv::inject_tokens (unchanged); ep.k/ep.v/ep.mf now also
                                                // exist on disk at `dir` if a future path prefers kv::replay.
                                                ns.push(sp_daemon::recall::Episode {
                                                    name: ep_name.clone(),
                                                    dir: dir_str.clone(),
                                                    npos: ntok as i32,
                                                    topic,
                                                    // LIVE-FIX (perspective): delivery payload attributed; selection
                                                    // artifacts (gk/sig/l5key/tokens) stay on the raw text.
                                                    text: format!("The user said: {text}"),
                                                    sig,
                                                    gk,
                                                    gk_ng: ng,
                                                    tokens: Some(toks),
                                                    l5key: l5k, // B4-SEAL: minted at capture; empty only on mint failure
                                                    policy: mem_class.map(sp_daemon::recall::MemPolicy::from_class), // #73: self-govern if classified
                                                    lifecycle: 0,
                                                });
                                                let total = ns.len();
                                                tracing::info!(
                                                    "B4-NIGHTSHIFT: consolidated (batched) -> '{}' npos={} ng={} — registry now has {} live episode(s)",
                                                    ep_name, ntok, ng, total
                                                );
                                                // LAYER-3: hand the new fact to the DECIDE pass below.
                                                decide_new = Some((ep_name.clone(), text.clone()));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                Err(e) => tracing::warn!("B4-NIGHTSHIFT: tokenize failed: {e} — no episode appended"),
            }
        }
    }

    // ── LAYER-3: DECIDE — the model curates its own memory ──────────────────────────
    // When NIGHTSHIFT just captured a new fact, show the model its RELATED existing
    // memories and let IT decide whether the new one supersedes/contradicts an old one
    // (e.g. "my favorite color is now green" retires "...is blue"). The model's verdict
    // — not a dedup rule — drives the removal: this is the emergent agency layer. Runs
    // AFTER the response streamed (cache free) and AFTER capture (the new episode
    // exists). SP_DECIDE=1; unset ⇒ no model-call, no removal = byte-identical null floor.
    if std::env::var("SP_DECIDE").ok().as_deref() == Some("1") {
        if let Some((new_name, new_text)) = decide_new {
            use sp_daemon::recall;
            // Related existing memories (share subject with the new fact), excluding the
            // just-captured episode. token_overlap >= 0.3 ≈ "about the same thing".
            let mut cands: Vec<(f32, String)> = Vec::new();
            if let Some(reg) = app.recall_registry.as_ref() {
                for ep in reg.iter() {
                    if ep.name == new_name || ep.text == new_text { continue; }
                    let ov = recall::token_overlap(&new_text, &ep.text);
                    if ov >= 0.3 && !cands.iter().any(|(_, ct)| ct == &ep.text) { cands.push((ov, ep.text.clone())); }
                }
            }
            {
                let ns = app.nightshift.read().unwrap();
                for ep in ns.iter() {
                    if ep.name == new_name || ep.text == new_text { continue; }
                    let ov = recall::token_overlap(&new_text, &ep.text);
                    if ov >= 0.3 && !cands.iter().any(|(_, ct)| ct == &ep.text) { cands.push((ov, ep.text.clone())); }
                }
            }
            cands.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            cands.truncate(5);
            if !cands.is_empty() {
                let mut decided = false; // set true when stage-1 supersede forgets
                // Present the new fact + numbered related memories; the model picks.
                let mut listing = String::new();
                for (i, (_, t)) in cands.iter().enumerate() {
                    listing.push_str(&format!("{}. \"{}\"\n", i + 1, t));
                }
                let prompt = format!(
                    "A NEW fact was just stated:\n\"{}\"\nHere are earlier memories:\n{}\nIs the NEW fact a REPLACEMENT for one of these memories -- does it state a new value for the SAME attribute, so that the NEW fact and that memory CANNOT both be true at the same time (for example a favorite color changing, or a current home city changing)? Only then give that memory's number. If the NEW fact is about a DIFFERENT attribute and can be true ALONGSIDE the memories (for example a person's job AND the city they live in are both true), answer NONE.\nReply with the memory number that is replaced, or NONE.",
                    new_text, listing);
                let dmsgs = vec![Message { role: "user".to_string(), content: prompt }];
                if let Ok(dtoks) = app.tokenizer.apply_template_ids(&dmsgs) {
                    if dtoks.len() >= 2 {
                        // Force the answer format + prevent template echo: prefill "CHANGED="
                        // into the model turn (the judge's prefill-TAG trick). The model then
                        // completes a number or NONE; reply is seeded with the forced prefix.
                        let mut dtoks = dtoks;
                        let mut reply = String::new();
                        if let Ok(tt) = app.tokenizer.encode("CHANGED=") {
                            let tt: Vec<i32> = tt.into_iter().filter(|&t| t != 2).collect();
                            if !tt.is_empty() { dtoks.extend_from_slice(&tt); reply.push_str("CHANGED="); }
                        }
                        let _ = unsafe { kv::reset(handle) };
                        let (dhead, dlast) = dtoks.split_at(dtoks.len() - 1);
                        let ok = dhead.is_empty() || unsafe { kv::prefill(handle, dhead) }.is_ok();
                        if ok {
                            let mut tok = dlast[0];
                            let turn_stops = app.tokenizer.turn_stop_ids();
                            let suppress = app.tokenizer.suppress_token_ids();
                            for _ in 0..12 {
                                if unsafe { kv::decode_step(handle, tok, logits) }.is_err() { break; }
                                for &sid in &suppress { let u = sid as usize; if u < logits.len() { logits[u] = f32::NEG_INFINITY; } }
                                let mut bi = 0usize; let mut bv = f32::NEG_INFINITY;
                                for (i, &v) in logits.iter().enumerate() { if v > bv { bv = v; bi = i; } }
                                let g = bi as i32;
                                if app.tokenizer.eos_ids.contains(&g) || turn_stops.contains(&g) { break; }
                                reply.push_str(&String::from_utf8_lossy(app.tokenizer.decode_token(g)));
                                tok = g;
                                if reply.contains('\n') { break; }
                            }
                        }
                        // Restore a clean cache (the next turn re-prefills from scratch).
                        let _ = unsafe { kv::reset_cold(handle) };
                        tracing::info!("LAYER-3 DECIDE: new=\"{}\" {} cand(s) -> reply=\"{}\"",
                            new_text.chars().take(50).collect::<String>(), cands.len(), reply.trim());
                        // Parse CHANGED=<n>; execute the model's choice via the forget removal
                        // (drop from the live set + rewrite the persisted registry). NONE / any
                        // unparseable reply ⇒ no removal (safe default).
                        let ru = reply.to_uppercase();
                        let n: Option<usize> = if ru.contains("NONE") { None } else {
                            ru.chars()
                                .skip_while(|c| !c.is_ascii_digit())
                                .take_while(|c| c.is_ascii_digit())
                                .collect::<String>().parse().ok()
                        };
                        {
                            if let Some(n) = n {
                                if n >= 1 && n <= cands.len() {
                                    let victim = cands[n - 1].1.clone();
                                    let sp_life = std::env::var("SP_MEM_LIFECYCLE").ok().as_deref() == Some("1");
                                    { let mut ns = app.nightshift.write().unwrap();
                                      if sp_life { for e in ns.iter_mut() { if e.text == victim { e.lifecycle = 1; } } }
                                      else { ns.retain(|e| e.text != victim); } }
                                    if let Ok(reg_path) = std::env::var("SP_RECALL_REGISTRY") {
                                        if sp_life { let _ = recall::mark_superseded_registry(&reg_path, &victim, 1); }
                                        else {
                                        if let Ok(content) = std::fs::read_to_string(&reg_path) {
                                            let kept: Vec<&str> = content.lines().filter(|line| {
                                                match serde_json::from_str::<serde_json::Value>(line) {
                                                    Ok(v) => v.get("text").and_then(|x| x.as_str()) != Some(victim.as_str()),
                                                    Err(_) => !line.trim().is_empty(),
                                                }
                                            }).collect();
                                            let mut out = kept.join("\n"); if !out.is_empty() { out.push('\n'); }
                                            if let Err(e) = write_atomic(std::path::Path::new(&reg_path), out.as_bytes()) {
                            tracing::warn!("registry rewrite failed (kept the old file): {e}");
                        }
                                        }
                                        }
                                    }
                                    tracing::info!("LAYER-3 DECIDE: the model chose to FORGET #{} -> \"{}\" (superseded by the new fact)", n, victim);
                                    decided = true;
                                }
                            }
                        }
                    }
                }
                // ── STAGE 2: MERGE — consolidate two partial facts into ONE synthesized
                // truth. Runs only if stage-1 found NO supersede. Compares the new fact
                // with the single best-overlap candidate; if the model judges they describe
                // the SAME subject, it returns the combined fact, and the daemon drops BOTH
                // old episodes and captures the one consolidation (the operator's "holy grail").
                if !decided {
                    let cand0 = cands[0].1.clone();
                    let mprompt = format!(
                        "Two facts may describe the SAME thing and belong together as ONE memory:\nA: \"{}\"\nB: \"{}\"\nIf A and B are about the same subject and should be combined, reply with the single complete combined fact on one line, prefixed exactly: MERGE:: <combined fact>\nIf they are about DIFFERENT things, reply NONE.",
                        new_text, cand0);
                    let mmsgs = vec![Message { role: "user".to_string(), content: mprompt }];
                    if let Ok(mtoks) = app.tokenizer.apply_template_ids(&mmsgs) {
                        if mtoks.len() >= 2 {
                            let _ = unsafe { kv::reset(handle) };
                            let (mhead, mlast) = mtoks.split_at(mtoks.len() - 1);
                            let ok = mhead.is_empty() || unsafe { kv::prefill(handle, mhead) }.is_ok();
                            let mut mreply = String::new();
                            if ok {
                                let mut tok = mlast[0];
                                let turn_stops = app.tokenizer.turn_stop_ids();
                                let suppress = app.tokenizer.suppress_token_ids();
                                for _ in 0..40 {
                                    if unsafe { kv::decode_step(handle, tok, logits) }.is_err() { break; }
                                    for &sid in &suppress { let u = sid as usize; if u < logits.len() { logits[u] = f32::NEG_INFINITY; } }
                                    let mut bi = 0usize; let mut bv = f32::NEG_INFINITY;
                                    for (i, &v) in logits.iter().enumerate() { if v > bv { bv = v; bi = i; } }
                                    let g = bi as i32;
                                    if app.tokenizer.eos_ids.contains(&g) || turn_stops.contains(&g) { break; }
                                    mreply.push_str(&String::from_utf8_lossy(app.tokenizer.decode_token(g)));
                                    tok = g;
                                    if mreply.contains('\n') { break; }
                                }
                            }
                            let _ = unsafe { kv::reset_cold(handle) };
                            let mtrim = mreply.trim().to_string();
                            tracing::info!("LAYER-3 MERGE: A=\"{}\" B=\"{}\" -> reply=\"{}\"",
                                new_text.chars().take(40).collect::<String>(),
                                cand0.chars().take(40).collect::<String>(), mtrim);
                            // Parse "MERGE:: <combined>" (case-insensitive). NONE / no marker /
                            // empty combined ⇒ keep both (safe default, no removal).
                            let lower = mtrim.to_lowercase();
                            if let Some(mp) = lower.find("merge") {
                                if !lower[..mp].contains("none") {
                                    let rest = mtrim.get(mp + 5..).unwrap_or("");
                                    let combined = rest
                                        .trim_start_matches(|c: char| c == ':' || c == '=' || c == '-' || c == '>' || c == ' ')
                                        .trim().trim_matches('"').trim().to_string();
                                    if combined.chars().count() >= 4 {
                                        // Drop BOTH old episodes (the just-captured new fact + the
                                        // candidate) from the live set and the persisted registry...
                                        let drop_a = new_text.clone();
                                        let drop_b = cand0.clone();
                                        let sp_life = std::env::var("SP_MEM_LIFECYCLE").ok().as_deref() == Some("1");
                                        { let mut ns = app.nightshift.write().unwrap();
                                          if sp_life { for e in ns.iter_mut() { if e.text == drop_a || e.text == drop_b { e.lifecycle = 1; } } }
                                          else { ns.retain(|e| e.text != drop_a && e.text != drop_b); } }
                                        if let Ok(reg_path) = std::env::var("SP_RECALL_REGISTRY") {
                                            if sp_life {
                                                let _ = recall::mark_superseded_registry(&reg_path, &drop_a, 1);
                                                let _ = recall::mark_superseded_registry(&reg_path, &drop_b, 1);
                                            } else {
                                            if let Ok(content) = std::fs::read_to_string(&reg_path) {
                                                let kept: Vec<&str> = content.lines().filter(|line| {
                                                    match serde_json::from_str::<serde_json::Value>(line) {
                                                        Ok(v) => { let t = v.get("text").and_then(|x| x.as_str());
                                                            t != Some(drop_a.as_str()) && t != Some(drop_b.as_str()) },
                                                        Err(_) => !line.trim().is_empty(),
                                                    }
                                                }).collect();
                                                let mut out = kept.join("\n"); if !out.is_empty() { out.push('\n'); }
                                                if let Err(e) = write_atomic(std::path::Path::new(&reg_path), out.as_bytes()) {
                            tracing::warn!("registry rewrite failed (kept the old file): {e}");
                        }
                                            }
                                            }
                                        }
                                        // ...and capture the single synthesized consolidation.
                                        let okc = capture_live_episode(app, &combined);
                                        tracing::info!("LAYER-3 MERGE: consolidated \"{}\" + \"{}\" -> \"{}\" (capture {})",
                                            drop_a.chars().take(30).collect::<String>(),
                                            drop_b.chars().take(30).collect::<String>(),
                                            combined.chars().take(50).collect::<String>(),
                                            if okc { "ok" } else { "FAILED" });
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // PERSIST-KV: record the cache's new contents (this turn's prompt + the tokens we committed)
    // so the next turn can reuse it as a strict prefix and prefill only its new suffix. Gated by
    // persist_kv (the plain decode path) -- in that config NO side-call above touched the cache, so
    // [tokens ++ committed_gen] mirrors it exactly. Any other config never sets persist_kv, so the
    // committed state is simply left stale and the next turn's strict-prefix+position guard rejects
    // it -> full reset+prefill (the byte-identical null floor). Cleared if anything went wrong is
    // unnecessary: the guard already fails closed.
    // TURN-PHASE summary: prefill vs decode vs everything-else (capture, spectest,
    // qkey mint...). This is the line that would have told the operator, hours ago,
    // that a 3-token answer spends its 52 s outside decode.
    {
        let n_out = committed_gen.len().max(1);
        let total_ms = t_turn.elapsed().as_millis();
        // POST = capture / spectest / qkey mint / growth, i.e. everything AFTER the last
        // emitted token. It used to be folded into "decode", inflating the decode ms and
        // deflating the reported tok/s.
        let post_ms = decode_t0.elapsed().as_millis().saturating_sub(t_gen_ms);
        let other_ms = total_ms
            .saturating_sub(t_prefill_ms)
            .saturating_sub(t_recall_ms)
            .saturating_sub(t_gen_ms)
            .saturating_sub(post_ms);
        tracing::info!(
            "TURN-PHASE: total {} ms = prefill {} + recall {} ({} fwd) + decode {} ({} tok, {:.1} tok/s) \
             + post {} + other {} (byteexact={})",
            total_ms, t_prefill_ms, t_recall_ms, recall_steps, t_gen_ms, n_out,
            (n_out as f64) / (t_gen_ms.max(1) as f64 / 1000.0),
            post_ms, other_ms, byteexact);
    }
    if persist_kv {
        let mut c = KV_COMMITTED.lock().unwrap_or_else(|p| p.into_inner());
        c.clear();
        *KV_COMMITTED_TAG.lock().unwrap_or_else(|p| p.into_inner()) = model_tag().unwrap_or_default();
        // RECALL EXTENSION: re-arm the reusable prefix ONLY when the cache is the pristine plain
        // prompt + generated -- i.e. no episode was injected for synthesis this turn. A PICK
        // (recalled.is_some()) leaves the cache holding the episode K/V at [dpos,dpos+npos), which
        // a token-sequence committed cannot represent; leaving it cleared makes the next turn
        // reset + full-prefill (clearing the injected episode) -- correct, just not O(1) that turn.
        if recalled.is_none() && !cache_dirty {
            c.reserve(tokens.len() + committed_gen.len());
            c.extend_from_slice(tokens);
            c.extend_from_slice(&committed_gen);
        }
    }
    // The turn reached its commit: the void guard's job is done (F1). A dirty turn
    // still gets here — with the list deliberately left EMPTY above.
    std::mem::forget(commit_void);

    sessions.remove(chat_id);
}

// A2: `fn argmax` moved to `crate::sampler::argmax` (the temp=0 null floor);
// both decode loops now go through `Sampler::sample`.

// ── POST /v1/capture (B4 SCALE-UP curator) ───────────────────────────────
// Batch-capture an episode's ep.k/ep.v/ep.mf by running ONE curated forward on the
// RESIDENT model (reuses loaded weights — no 9GB reload per needle). Tokenizes the
// raw needle text EXACTLY as the B4 provenance fix does: KEEP the forced-BOS (id 2)
// AND append a trailing "\n" before encoding — this is why live==curated (the head
// scores ep.k identically). Calls kv::capture_batched under the kvdecode/session
// lock. Default behaviour unchanged for all other endpoints. Returns {ok, npos} or
// {error}. Used by tools/xbar_lsh/b4_capture_driver.py to populate a corpus in ONE
// model load.
#[derive(serde::Deserialize)]
pub struct CaptureReq {
    pub text: String,
    pub out_dir: String,
}

pub async fn v1_capture(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CaptureReq>,
) -> Response {
    use sp_daemon::cuda_kvdecode_dispatch as kv;
    let app = state.clone();
    let text = req.text.trim().to_string();
    let out_dir = req.out_dir.clone();
    if text.is_empty() {
        return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error":"empty text"}))).into_response();
    }
    // PROVENANCE: byte-match the curator's sp_tok_enc — BOS kept + trailing newline.
    let text_nl = format!("{text}\n");
    let toks = match app.tokenizer.encode(&text_nl) {
        Ok(t) => t,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR,
                          Json(serde_json::json!({"error": format!("tokenize: {e}")}))).into_response(),
    };
    let ntok = toks.len();
    if ntok < 4 {
        return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error":"too short","npos":ntok}))).into_response();
    }
    // Run the blocking CUDA capture on a blocking thread; hold the session lock for
    // the qm borrow + the whole capture (serializes SP_XBAR_RECALL_WRITE env inside).
    let res = task::spawn_blocking(move || -> Result<usize, String> {
        // THE DEVICE LOCK, NOT THE SESSION LOCK (2026-08-29 — the serve-stall's
        // mechanism; full incident at borrow_qm/device_guard). The 08-23 comment
        // that stood here serialized this forward on app.session and claimed
        // "v1_chat takes this same mutex" — the streaming chat worker never does;
        // it holds the DEVICE lock. The two forwards ran concurrently, the shared
        // MoE scratch was freed under running kernels, and the context wedged for
        // 25 minutes with the GPU idle and metrics answering. qm is borrowed under
        // a brief session lock (dropped at once, so metrics/clone stay alive), the
        // forward parks behind any queued foreground turn, and the device lock is
        // held for the whole forward section.
        let qm = borrow_qm(&app)?;
        let _gpu = device_guard(&app, "capture");
        std::fs::create_dir_all(&out_dir).map_err(|e| format!("mkdir {out_dir}: {e}"))?;
        unsafe { kv::capture_batched(qm, &toks, &out_dir) }?;
        // ── SEM S0 (2026-07-14, docs/SEMANTICS.md §4.3): mint ep.l5 on THE capture path ──
        // Until now ep.l5 was minted only by the RETIRED daemon-writer path (B4/store_verb
        // → mint_ep_l5), so every episode the one memory authority captured via this route
        // was L5-INVISIBLE: the 88.5%-paraphrase selector (G-REP-LAYER-L5) had nothing to
        // read for grown memories. Same seam, same provenance, now reachable from the door
        // the harness actually uses. Gated SP_CAPTURE_L5 (mapped in serve.py, [sem] in the
        // profile). Mint failure logs and never fails the capture. NOT a memory writer:
        // ep.l5 is a derived sidecar beside ep.k/ep.v, consumed by recall.rs and the
        // harness semindex upgrade hook (G-SEM-INDEX).
        #[cfg(feature = "wire_cuda_backend")]
        if std::env::var("SP_CAPTURE_L5").as_deref() == Ok("1") {
            let _ = mint_ep_l5(&app, qm, &toks, &text, std::path::Path::new(&out_dir));
        }
        Ok(ntok)
    }).await;
    match res {
        Ok(Ok(npos)) => (StatusCode::OK, Json(serde_json::json!({"ok":true,"npos":npos}))).into_response(),
        Ok(Err(e))   => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error":e}))).into_response(),
        Err(e)       => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error":format!("join: {e}")}))).into_response(),
    }
}

// ── /v1/embed — the SEM S1 query-key (2026-07-14, docs/SEMANTICS.md §4.3) ────────────
// POST {"text": ...} -> {"l5": [512 f32], "n": ntok}: the SAME l5_query_embed the live
// recall selector ranks with, minted with the SAME provenance as ep.l5 (standalone
// position-0 scratch forward, global-Q of the last token, mean-over-heads of global
// layer 5, L2-normed) — so harness-side cosine(query, ep.l5) is the measured
// G-REP-LAYER-L5 geometry, not an approximation of it. READ-ONLY: a throwaway scratch
// session, released before returning; the resident cache is not read, not written, not
// evicted (the /v1/oneshot doctrine). Consumer: harness/skills/semindex.query_embed()
// behind SP_SEM_RANK. CUDA-only, like every route that needs a scratch sp_g4_kv.
// ── /v1/recall_rank — the LEARNED selector as a read-only oracle (2026-07-14) ─────────
// Phase C2's successor question (INVARIANT-ROADMAP): can the W_c + (E+1)-NULL head — the
// one signal on this box with a measured 360/361 recall + 50/50 reject
// (G-CHAT-B3-WC-DIV2) — beat the 0.06 lexical decider baseline on the SEM corpus?
// This route exposes exactly that judgment, statelessly: templated scratch forward for
// the query's global-Q (the /v1/embed pattern), load_episode_global_k per candidate
// dir, wc_score each, argmax against the s0 NULL. READ-ONLY: scratch session released,
// resident cache untouched, nothing replayed, nothing written. The deploy blob path
// rides in the REQUEST (SP_B3_WC is unmapped on the live profile — G-ONEDOOR strips
// it — and a research oracle should be explicit, not ambient).
// HONESTY FLAG from this file's own F2b comment: W_c is geometric — strong on
// high-entropy novel needles, measured BLIND on mutually-similar natural facts. The
// SEM corpus is natural facts. The scoreboard may return a negative; that is its job.
#[cfg(feature = "wire_cuda_backend")]
#[derive(serde::Deserialize)]
pub struct RecallRankReq {
    pub text: String,
    pub episode_dirs: Vec<String>,
    pub wc_path: String,
}

#[cfg(feature = "wire_cuda_backend")]
// ── ONE FORWARD AT A TIME ON THIS DEVICE — the background-forward preamble ──────────
// (2026-08-29, the serve-stall's mechanism, caught live.) The 08-23 pass serialized
// capture/recall_rank/embed on `app.session`, and its comment claimed "no deadlock:
// v1_chat takes this same mutex". The STREAMING chat worker does not — it serializes
// on `cuda_kvdecode_handle` (the device lock) and never touches `app.session`
// mid-forward. So a capture's batched forward and QKEY mini-generations ran
// CONCURRENTLY with a live chat forward; the shared MoE scratch (`g_moe_*`,
// moe_scratch_ensure's free/realloc) was pulled out from under running kernels —
// the 2026-08-03 fault class, verbatim — and the context WEDGED: 25 minutes, GPU at
// 2%, /v1/metrics answering (it takes only a brief session lock), every later turn
// parked forever behind the device lock the wedged chat thread still held. Live
// specimen: ep_tool_1788001052517, 2026-08-29 20:57–21:22; the same class is almost
// certainly the 2026-08-28 14.5-minute page freeze.
//
// THE PATTERN (proven on /v1/oneshot since 08-03): borrow `qm` under a BRIEF session
// lock and DROP it — v1_metrics and clone_session stay alive through a long capture —
// then park behind any queued foreground turn, then hold the DEVICE lock for the
// whole forward section. `qm` stays sound: the model pointer is process-lifetime and
// these forwards never touch mutable session state; only the device lock is what
// makes "one forward on the GPU" true.
fn borrow_qm(app: &Arc<AppState>) -> Result<*const std::ffi::c_void, String> {
    let mut sguard = app.session.as_ref()
        .ok_or_else(|| "L1 session unavailable (qwen36 lane)".to_string())?
        .lock().unwrap();
    let sraw = sguard.raw_ptr() as *mut sp_daemon::ffi_l1::sp_session;
    let qm = (unsafe { sp_daemon::ffi_l1::sp_session_qwen3_model(sraw) }) as *const std::ffi::c_void;
    if qm.is_null() { return Err("sp_session_qwen3_model NULL".to_string()); }
    Ok(qm)
}

// LOCK-ORDER INVARIANT (2026-08-29 audit §3): the session lock is NEVER held into
// the device lock. borrow_qm scopes its guard to the function; every caller takes
// device_guard strictly AFTER borrow_qm returns. The chat worker takes the locks in
// the opposite order (device 2204 → session 5638), so holding both here would be an
// instant AB-BA deadlock. If you are editing a call site: qm first, drop, THEN guard.
static DEVICE_FALLBACK_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

// ── WHO HOLDS THE DEVICE (2026-08-30, after three hours of silence) ─────────────────
// A turn blocked on the device lock logs NOTHING — it has already printed its prompt
// ids and has not reached the next log point — so a stuck holder presents as a daemon
// that simply stopped talking. Measured today: her 07:01 self-turn logged `S1 prompt
// ids` + `SELF-REPEAT BAN` and then vanished for three hours and seventeen minutes;
// the day-boundary prewarm queued behind it (which is why the room's chip said
// "warming" for the rest of the morning); two one-shots each burned their full 1800 s
// yield and queued too. Nothing anywhere named the thing at the front of that queue.
//
// So the lock says who owns it and since when, and a caller that finds it BUSY says so
// before it blocks. One short-lived mutex, taken and released around a String clone;
// never held across a device call, so it cannot itself become the thing that wedges.
static DEVICE_OWNER: std::sync::Mutex<Option<(String, std::time::Instant)>> =
    std::sync::Mutex::new(None);

/// Clears the owner note on every exit path — the note must not outlive the lock.
pub struct DeviceOwned;
impl Drop for DeviceOwned {
    fn drop(&mut self) {
        if let Ok(mut o) = DEVICE_OWNER.lock() { *o = None; }
    }
}

/// Announce the wait BEFORE blocking, naming the current holder and how long it has
/// held. This is the line whose absence cost today's diagnosis: it turns "the daemon
/// went quiet" into "chat is waiting on 'oneshot', which has held for 3h17m".
fn device_note_wait(what: &str) {
    let held = DEVICE_OWNER.lock().ok()
        .and_then(|o| o.as_ref().map(|(w, t)| (w.clone(), t.elapsed())));
    if let Some((who, since)) = held {
        tracing::warn!("DEVICE: {what} is WAITING — held by '{who}' for {:.1}s",
                       since.as_secs_f64());
    }
}

/// Record the new owner. Call immediately after the lock is acquired.
fn device_note_own(what: &str) -> DeviceOwned {
    if let Ok(mut o) = DEVICE_OWNER.lock() {
        *o = Some((what.to_string(), std::time::Instant::now()));
    }
    DeviceOwned
}

pub enum DeviceGuard<'a> {
    Handle(std::sync::MutexGuard<'a, crate::state::CudaKvDecodeHandle>, DeviceOwned),
    Fallback(std::sync::MutexGuard<'a, ()>, DeviceOwned),
}

fn device_guard<'a>(app: &'a Arc<AppState>, what: &str) -> DeviceGuard<'a> {
    yield_to_foreground(what);
    device_note_wait(what);
    match app.cuda_kvdecode_handle.as_ref() {
        Some(m) => {
            let g = m.lock().unwrap_or_else(|p| p.into_inner());
            let own = device_note_own(what);
            DeviceGuard::Handle(g, own)
        }
        None => {
            // ── F-3 (2026-08-29 audit): `Option::map` here meant "no kvdecode handle
            // -> NO LOCK AT ALL", silently — the whole wedge fix a no-op on any
            // profile with SP_DAEMON_KVDECODE unset, while the forwards still shared
            // g_w/g_moe_* process globals. One device, one lock, ALWAYS: a dedicated
            // fallback mutex serializes these lanes against each other when the
            // resident handle is absent (the chat fallback path is then the residual
            // risk, and this warn line is how it is found in a log).
            tracing::warn!("DEVICE-GUARD: no kvdecode handle — {what} serializing on the fallback lock");
            let g = DEVICE_FALLBACK_LOCK.lock().unwrap_or_else(|p| p.into_inner());
            let own = device_note_own(what);
            DeviceGuard::Fallback(g, own)
        }
    }
}

pub async fn v1_recall_rank(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RecallRankReq>,
) -> Response {
    let app = state.clone();
    let text = req.text.trim().to_string();
    if text.is_empty() || req.episode_dirs.is_empty() {
        return (StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error":"empty text or no candidates"}))).into_response();
    }
    let head = match sp_daemon::recall::load_wc(&req.wc_path) {
        Some(h) => h,
        None => return (StatusCode::BAD_REQUEST,
                        Json(serde_json::json!({"error": format!("load_wc({}) failed", req.wc_path)}))).into_response(),
    };
    let msgs = vec![Message { role: "user".to_string(), content: text.clone() }];
    let toks = match app.tokenizer.apply_template_ids(&msgs) {
        Ok(t) => t,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR,
                          Json(serde_json::json!({"error": format!("template: {e}")}))).into_response(),
    };
    if toks.len() < 2 {
        return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error":"too short"}))).into_response();
    }
    let dirs = req.episode_dirs.clone();
    let res = task::spawn_blocking(move || -> Result<(Vec<Option<f32>>, f32), String> {
        use sp_daemon::cuda_kvdecode_dispatch as kv;
        use sp_daemon::recall;
        // THE DEVICE LOCK, NOT THE SESSION LOCK (2026-08-29 — the serve-stall's
        // mechanism; full incident at borrow_qm/device_guard). The 08-23 comment
        // that stood here serialized this forward on app.session and claimed
        // "v1_chat takes this same mutex" — the streaming chat worker never does;
        // it holds the DEVICE lock. The two forwards ran concurrently, the shared
        // MoE scratch was freed under running kernels, and the context wedged for
        // 25 minutes with the GPU idle and metrics answering. qm is borrowed under
        // a brief session lock (dropped at once, so metrics/clone stay alive), the
        // forward parks behind any queued foreground turn, and the device lock is
        // held for the whole forward section.
        let qm = borrow_qm(&app)?;
        let _gpu = device_guard(&app, "recall_rank");
        let scratch = unsafe { kv::open_scratch(qm, toks.len() as i32 + 8) }   // F10
            .map_err(|e| format!("scratch open: {e:?}"))?;
        let n_global = recall::n_global();
        let mut qbuf = vec![0.0f32; n_global * recall::g_nh() * recall::hd()];
        let qok = (|| {
            unsafe { kv::prefill(scratch, &toks[..toks.len() - 1]) }.ok()?;
            unsafe { kv::read_global_q(scratch, toks[toks.len() - 1], &mut qbuf) }.ok()?;
            Some(())
        })();
        // SAFETY: scratch came from open() above; release is NULL-safe/idempotent.
        unsafe { kv::release_for_model(scratch) };
        if qok.is_none() {
            return Err("query global-Q read failed".to_string());
        }
        let scores = dirs.iter().map(|d| {
            recall::load_episode_global_k(d, i32::MAX).map(|(gk, ng)| {
                let npos = if ng > 0 { gk.len() / (ng * recall::hd()) } else { 0 };
                recall::wc_score(&qbuf, &gk, ng, npos, &head)
            })
        }).collect();
        Ok((scores, head.s0))
    }).await;
    match res {
        Ok(Ok((scores, s0))) => {
            let (mut bi, mut bv) = (None::<usize>, f32::NEG_INFINITY);
            for (i, s) in scores.iter().enumerate() {
                if let Some(v) = s {
                    if *v > bv { bv = *v; bi = Some(i); }
                }
            }
            let picked = match bi {
                Some(i) if bv > s0 => serde_json::json!(i),
                _ => serde_json::Value::Null,          // the (E+1)-NULL reject
            };
            (StatusCode::OK, Json(serde_json::json!({
                "scores": scores, "null": s0, "picked": picked, "best": bv
            }))).into_response()
        }
        Ok(Err(e)) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e}))).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": format!("join: {e}")}))).into_response(),
    }
}

#[cfg(feature = "wire_cuda_backend")]
#[derive(serde::Deserialize)]
pub struct EmbedReq {
    pub text: String,
}

#[cfg(feature = "wire_cuda_backend")]
pub async fn v1_embed(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EmbedReq>,
) -> Response {
    let app = state.clone();
    let text = req.text.trim().to_string();
    if text.is_empty() {
        return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error":"empty text"}))).into_response();
    }
    // PROVENANCE (v2, 2026-07-14): key the text EXACTLY as QKEY keys a question and as a
    // live query is keyed (mint_question_l5 step 2): a chat-TEMPLATED user turn, prefill
    // all but the last token, read_global_q at the last. The first draft of this route
    // embedded BARE text (mint_live_ep_l5 provenance) and cosines against the QKEY
    // question-space keys were meaningless — 3/100 argmax on the SEM corpus, every pair
    // >= 0.70 — because templated and bare texts live in different manifold regions.
    // The QKEY comment two hundred lines up warned about exactly this cross-pick.
    // PROVENANCE IS NOT A DETAIL; IT IS THE SPACE.
    let msgs = vec![Message { role: "user".to_string(), content: text.clone() }];
    let toks = match app.tokenizer.apply_template_ids(&msgs) {
        Ok(t) => t,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR,
                          Json(serde_json::json!({"error": format!("template: {e}")}))).into_response(),
    };
    if toks.len() < 2 {
        return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error":"too short","npos":toks.len()}))).into_response();
    }
    let ntok = toks.len();
    let res = task::spawn_blocking(move || -> Result<Vec<f32>, String> {
        use sp_daemon::cuda_kvdecode_dispatch as kv;
        use sp_daemon::recall;
        // THE DEVICE LOCK, NOT THE SESSION LOCK (2026-08-29 — the serve-stall's
        // mechanism; full incident at borrow_qm/device_guard). The 08-23 comment
        // that stood here serialized this forward on app.session and claimed
        // "v1_chat takes this same mutex" — the streaming chat worker never does;
        // it holds the DEVICE lock. The two forwards ran concurrently, the shared
        // MoE scratch was freed under running kernels, and the context wedged for
        // 25 minutes with the GPU idle and metrics answering. qm is borrowed under
        // a brief session lock (dropped at once, so metrics/clone stay alive), the
        // forward parks behind any queued foreground turn, and the device lock is
        // held for the whole forward section.
        let qm = borrow_qm(&app)?;
        let _gpu = device_guard(&app, "embed");
        // The mint_live_ep_l5 forward, minus the file write: scratch open, prefill all
        // but the last token, read global-Q at the last, embed, release. None -> Err.
        let scratch = unsafe { kv::open_scratch(qm, toks.len() as i32 + 8) }   // F10
            .map_err(|e| format!("scratch open: {e:?}"))?;
        let n_global = recall::n_global();
        let mut ql = vec![0.0f32; n_global * recall::g_nh() * recall::hd()];
        let key = (|| {
            unsafe { kv::prefill(scratch, &toks[..toks.len() - 1]) }.ok()?;
            unsafe { kv::read_global_q(scratch, toks[toks.len() - 1], &mut ql) }.ok()?;
            let k = recall::l5_query_embed(&ql);
            if k.len() != recall::hd() { return None; }
            Some(k)
        })();
        // SAFETY: scratch came from open() above; release is NULL-safe/idempotent.
        unsafe { kv::release_for_model(scratch) };
        key.ok_or_else(|| "l5 embed failed".to_string())
    }).await;
    match res {
        Ok(Ok(k))  => (StatusCode::OK, Json(serde_json::json!({"l5": k, "n": ntok}))).into_response(),
        Ok(Err(e)) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error":e}))).into_response(),
        Err(e)     => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error":format!("join: {e}")}))).into_response(),
    }
}

// ── /v1/debug/kdiff — G-VERBATIM forensics (2026-07-12) ───────────────────
// THE QUESTION: the served model cannot disambiguate two positions holding the
// SAME token ("4471" -> "4417"; "QXZP" -> perfect). If RoPE reaches the stored
// keys, the K rows of two identical tokens at DIFFERENT positions must DIFFER.
// If they are identical, position never entered the keys and attention cannot
// tell the two "4"s apart — which is precisely the observed failure.
//
// POST {"text":"4 4"} -> prefills the raw tokens into the resident cache and
// returns, per prefilled position, the token id + the cosine similarity of its
// GLOBAL-owner K row against every other position's. Read-only forensics.
#[cfg(feature = "wire_cuda_backend")]
pub async fn v1_debug_kdiff(
    State(state): State<Arc<AppState>>,
    Json(req): Json<serde_json::Value>,
) -> Response {
    use sp_daemon::cuda_kvdecode_dispatch as kv;
    use sp_daemon::recall;
    let app = state.clone();
    let text = req.get("text").and_then(|v| v.as_str()).unwrap_or("4 4 7 1").to_string();
    let res = task::spawn_blocking(move || -> Result<serde_json::Value, String> {
        let toks = app.tokenizer.encode(&text).map_err(|e| format!("encode: {e}"))?;
        if toks.len() < 2 { return Err("need >= 2 tokens".into()); }
        let cache = app.cuda_kvdecode_handle.as_ref().ok_or("no resident cache")?;
        yield_to_foreground("debug_kdiff");   // F-7: never queue-jump a waiting turn
        let guard = cache.lock().unwrap_or_else(|p| p.into_inner());
        let handle = guard.0;
        unsafe { kv::reset_cold(handle) }?;
        unsafe { kv::prefill(handle, &toks) }?;
        let npos = toks.len() as i32;
        let n_global = recall::n_global();      // global owner layers
        let mut kbuf = vec![0.0f32; n_global * recall::g_kvd() * npos as usize];
        let ng = unsafe { kv::read_global_k(handle, &mut kbuf, npos) }? as usize;
        // per position: the concatenated global-K vector across owner layers
        let d = recall::hd();
        let vec_at = |p: usize| -> Vec<f32> {
            (0..ng).flat_map(|g| {
                let base = g * d * (npos as usize) + p * d;
                kbuf[base..base + d].to_vec()
            }).collect()
        };
        let cos = |a: &[f32], b: &[f32]| -> f32 {
            let (mut d0, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
            for i in 0..a.len() { d0 += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
            if na <= 0.0 || nb <= 0.0 { 0.0 } else { d0 / (na.sqrt() * nb.sqrt()) }
        };
        let vecs: Vec<Vec<f32>> = (0..toks.len()).map(vec_at).collect();
        let mut rows = Vec::new();
        for i in 0..toks.len() {
            let sims: Vec<f32> = (0..toks.len()).map(|j| (cos(&vecs[i], &vecs[j]) * 1000.0).round() / 1000.0).collect();
            rows.push(serde_json::json!({"pos": i, "tok": toks[i], "cos_vs_all": sims}));
        }
        let _ = unsafe { kv::reset_cold(handle) };
        Ok(serde_json::json!({"text": text, "n_global_layers": ng, "positions": rows}))
    }).await;
    match res {
        Ok(Ok(v)) => (StatusCode::OK, Json(v)).into_response(),
        Ok(Err(e)) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e}))).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": format!("join: {e}")}))).into_response(),
    }
}

// ── /v1/debug/kvdiff_batch — IS THE BATCHED FORWARD THE SAME FORWARD? ─────
//
// THE QUESTION, asked 2026-08-03 after two nights of her degenerating. With
// `prefill_batch` armed, the PREFIX SNAPSHOT that every warm turn restores from is itself
// captured through the batched forward (see the capture site above: "PREFIX-SNAPSHOT:
// capture used the batched forward"). If that forward is not equivalent to the per-token
// one, the snapshot is poisoned ONCE and every turn afterwards inherits it — and what
// that looks like from the room is her reciting her own system prompt back:
//
//   "**You are Shannon-Prime.** *The voice changes on the next turn (the harness reads
//    it live).* `#SHANNOW_... runs on singlely overworked RTX 10624`"
//
// Disarming the batch made her coherent, twice. But that is CORRELATION, and the fix was
// shipped saying so. THIS is the measurement that settles it: prefill the SAME tokens
// both ways, snapshot the prefix after each, and compare the bytes.
//
// WHAT A HONEST RESULT LOOKS LIKE. Bit-identical is the strong answer. But these are
// float paths with different accumulation ORDER — a batched GEMM sums differently from a
// sequence of GEMVs — so small last-bit differences are expected and are NOT evidence of
// the bug. So the report gives both: exact byte equality, AND the magnitude of the worst
// disagreement interpreted as fp16, so a rounding difference (~1e-3 relative) can be told
// apart from a wrong tensor (~1.0). The verdict field says which it is, and refuses to
// guess in between.
//
// Read-only forensics on a SCRATCH session — it must never touch the resident cache that
// the conversation lives in.
#[cfg(feature = "wire_cuda_backend")]
pub async fn v1_debug_kvdiff_batch(
    State(state): State<Arc<AppState>>,
    Json(req): Json<serde_json::Value>,
) -> Response {
    use sp_daemon::cuda_kvdecode_dispatch as kv;
    use sp_daemon::recall;
    let app = state.clone();
    let text = req.get("text").and_then(|v| v.as_str())
        .unwrap_or("The quick brown fox jumps over the lazy dog. ").to_string();
    let repeat = req.get("repeat").and_then(|v| v.as_u64()).unwrap_or(40) as usize;
    // "batch" (default) compares per-token vs batched; "control" compares per-token vs
    // per-token and MUST come back IDENTICAL — see the note at the B branch.
    // "suffix" (#41b) compares per-token vs [per-token prefix + SUFFIX-batched tail]:
    // the exact composite the SP_KV_PREFILL_BATCH_SUFFIX arm produces live. `split`
    // sets the prefix length (default: half the tokens).
    let req_mode = req.get("mode").and_then(|v| v.as_str()).unwrap_or("batch").to_string();
    let req_split = req.get("split").and_then(|v| v.as_u64()).map(|v| v as usize);
    let res = task::spawn_blocking(move || -> Result<serde_json::Value, String> {
        // Long enough that the batched path is actually taken (it declines under 64) and
        // that a real preamble's worth of positions is compared, not a toy.
        let body: String = std::iter::repeat(text.as_str()).take(repeat.max(1)).collect();
        let mut toks = app.tokenizer.encode(&body).map_err(|e| format!("encode: {e}"))?;
        // BISECT HANDLE. `prefill_batched` itself has no minimum (only n > Pmax fails); the
        // 64-token floor belongs to the SNAPSHOT CAPTURE, not to the forward. Being able to
        // ask for n_tok=1 is what turns "the batched path is wrong" into "the batched path
        // is wrong for a SINGLE TOKEN", which is a different and much smaller bug.
        if let Some(nt) = req.get("ntok").and_then(|v| v.as_u64()) {
            let nt = (nt as usize).clamp(1, toks.len());
            toks.truncate(nt);
        }
        if toks.is_empty() { return Err("no tokens".into()); }
        let cache = app.cuda_kvdecode_handle.as_ref().ok_or("no resident cache")?;
        yield_to_foreground("debug_kvdiff");   // F-7
        let guard = cache.lock().unwrap_or_else(|p| p.into_inner());
        let handle = guard.0;
        let p = toks.len() as i32;

        // A: THE PER-TOKEN PATH — the reference. It is the one every warm turn's suffix
        // already runs on, so it is the definition of correct here, not merely the older.
        unsafe { kv::reset_cold(handle) }?;
        unsafe { kv::prefill(handle, &toks) }?;
        let n = unsafe { kv::prefix_bytes(handle, p) }?;
        let mut a = vec![0u8; n];
        unsafe { kv::snapshot_prefix(handle, &mut a, p) }?;

        // B: THE BATCHED PATH. one_shot for the duration, exactly as the real capture
        // does it — otherwise the engine declines the batch under persist and we would be
        // comparing the per-token path against itself and calling it agreement.
        //
        // ...WHICH IS ALSO THE CONTROL, and it is not optional. A 16% byte difference
        // means nothing until you know that this harness reports ZERO for two runs that
        // must agree: if reset_cold left residue, or the snapshot blob carries
        // uninitialised padding, or the geometry shifts between calls, then EVERY
        // comparison here would show a difference and the verdict would be an artifact of
        // the instrument. `mode=control` runs the per-token path TWICE. It must return
        // IDENTICAL, and if it does not, nothing else this route says is worth reading.
        let control = req_mode == "control";
        let suffix = req_mode == "suffix";
        // #41b: per-token prefix + suffix-batched tail. Clamped so both sides are
        // non-empty — a split at 0 or p is the cold batch / pure per-token, which the
        // other modes already measure.
        let split = if suffix {
            req_split.unwrap_or(toks.len() / 2).clamp(1, toks.len().saturating_sub(1))
        } else { 0 };
        unsafe { kv::reset_cold(handle) }?;
        if control {
            unsafe { kv::prefill(handle, &toks) }?;
        } else if suffix {
            unsafe { kv::prefill(handle, &toks[..split]) }?;
            unsafe { kv::prefill_batched_from(handle, &toks[split..], split as i32) }
                .map_err(|e| format!("suffix batch declined: {e}"))?;
        } else {
            let _ = unsafe { kv::set_oneshot(handle, true) };
            let batched = unsafe { kv::prefill_batched(handle, &toks) };
            let _ = unsafe { kv::set_oneshot(handle, false) };
            batched.map_err(|e| format!("batched prefill declined: {e}"))?;
        }
        let n2 = unsafe { kv::prefix_bytes(handle, p) }?;
        let mut b = vec![0u8; n2];
        unsafe { kv::snapshot_prefix(handle, &mut b, p) }?;
        let _ = unsafe { kv::reset_cold(handle) };

        if n != n2 {
            return Ok(serde_json::json!({
                "verdict": "DIFFERENT SHAPE", "tokens": p, "bytes_pertok": n, "bytes_batched": n2,
                "note": "the two paths do not even produce the same amount of state"}));
        }
        // ── THE MEASUREMENT THAT CANNOT BE AN ARTIFACT OF PADDING ────────────────────
        // The blob is the WHOLE allocation, and the SWA ring plus the globals' Pmax depth
        // are far larger than the 402 positions actually written. Slots beyond dpos are
        // never read by attention (it bounds at dpos), so a difference out there is not a
        // correctness difference at all — and comparing the raw blob counts it anyway.
        // `read_global_k` returns exactly the global-owner K rows for positions [0, p),
        // which is state attention DOES read. If the blob disagrees and this agrees, the
        // blob diff was measuring dead slots and the whole alarm is wrong.
        // ASK THE ENGINE FOR THE STRIDE. Sizing this from recall::hd() is what killed the
        // daemon mid-request the first time I wrote this route: read_global_k writes
        // g_nkv*g_hd per row and g_nkv is 2 on this model, so the buffer was half.
        let d = unsafe { kv::global_kvd(handle) }?;
        let ng_want = recall::n_global();
        let mut ka = vec![0.0f32; ng_want * d * (p as usize)];
        let mut kb = vec![0.0f32; ng_want * d * (p as usize)];
        // A is currently NOT resident (B overwrote it), so re-run A to read its K, then B.
        unsafe { kv::reset_cold(handle) }?;
        unsafe { kv::prefill(handle, &toks) }?;
        let nga = unsafe { kv::read_global_k(handle, &mut ka, p) }? as usize;
        unsafe { kv::reset_cold(handle) }?;
        if control {
            unsafe { kv::prefill(handle, &toks) }?;
        } else if suffix {
            unsafe { kv::prefill(handle, &toks[..split]) }?;
            unsafe { kv::prefill_batched_from(handle, &toks[split..], split as i32) }
                .map_err(|e| format!("suffix batch (K read) declined: {e}"))?;
        } else {
            let _ = unsafe { kv::set_oneshot(handle, true) };
            let r = unsafe { kv::prefill_batched(handle, &toks) };
            let _ = unsafe { kv::set_oneshot(handle, false) };
            r.map_err(|e| format!("batched prefill (K read) declined: {e}"))?;
        }
        let _ngb = unsafe { kv::read_global_k(handle, &mut kb, p) }? as usize;
        let mut live_worst = 0.0f64;
        let mut live_diff = 0usize;
        let mut live_first_pos: Option<usize> = None;
        // THE MAX ALONE CANNOT TELL "everything is different" FROM "mostly close with a few
        // sign flips", and those two point at completely different bugs. Buckets do.
        let mut b_tiny = 0usize;    // <= 1%   : accumulation order
        let mut b_small = 0usize;   // <= 10%  : something scaled slightly
        let mut b_big = 0usize;     // <= 100% : a different value
        let mut b_flip = 0usize;    // > 100%  : opposite sign
        let mut sum_rel = 0.0f64;
        // PER GLOBAL LAYER. Globals sit at L = period-1, 2*period-1, ... so index 0 is the
        // SHALLOWEST. If the error is tiny at index 0 and grows with depth, the root cause
        // is a small difference near the input being amplified through the stack — and the
        // place to look is the embedding or layer 0. If it is flat, something is applied
        // wrongly at every layer. These two point at completely different bugs.
        let mut per_layer: Vec<f64> = vec![0.0; nga];
        let mut per_layer_n: Vec<usize> = vec![0; nga];
        // ── THE DISCRIMINATOR: POSITION 0 (2026-08-03) ───────────────────────────────
        // In a CAUSAL model, position 0 attends to nothing but itself. Its output is
        // therefore completely independent of how many tokens came after it — prefill 1
        // token or 402, position 0's residual stream, its K and its V are the same
        // computation. That makes it the cleanest probe in the whole forward:
        //
        //   position 0 CORRECT at every n  -> the per-token work is fine, and whatever is
        //                                     wrong lives in attention across positions
        //                                     (mask, window, RoPE offsets, ring sinking).
        //   position 0 WRONG once n grows  -> attention is exonerated, and the bug is in
        //                                     something that batches per-token work:
        //                                     a GEMM stride, or the expert bucketing that
        //                                     only has more than one row per bucket at n>1.
        //
        // Reported per depth as well, because "wrong at position 0 AND compounding with
        // depth" is a different animal from "wrong at position 0 in one layer".
        let mut pos0: Vec<f64> = vec![0.0; nga];
        let mut pos0_n: Vec<usize> = vec![0; nga];
        let mut poslast: Vec<f64> = vec![0.0; nga];
        let mut poslast_n: Vec<usize> = vec![0; nga];
        for g in 0..nga {
            for pos in 0..(p as usize) {
                let base = g * d * (p as usize) + pos * d;
                for i in 0..d {
                    let (fa, fb) = (ka[base + i] as f64, kb[base + i] as f64);
                    if fa == fb { continue; }
                    live_diff += 1;
                    if live_first_pos.is_none() { live_first_pos = Some(pos); }
                    let den = fa.abs().max(fb.abs()).max(1e-6);
                    let rel = (fa - fb).abs() / den;
                    if rel > live_worst { live_worst = rel; }
                    sum_rel += rel;
                    per_layer[g] += rel; per_layer_n[g] += 1;
                    if pos == 0 { pos0[g] += rel; pos0_n[g] += 1; }
                    if pos + 1 == p as usize { poslast[g] += rel; poslast_n[g] += 1; }
                    if rel <= 0.01 { b_tiny += 1; }
                    else if rel <= 0.10 { b_small += 1; }
                    else if rel <= 1.00 { b_big += 1; }
                    else { b_flip += 1; }
                }
            }
        }
        let live_total = nga * d * (p as usize);
        // The session's layout, read from the session rather than guessed. The sharer hunt
        // turns on facts like "is layer kvfs-1 actually a global layer" and "do the two
        // owner groups share a kvd", and a forensics route that guesses those produces
        // confident nonsense.
        let geom_json = {
            let g = unsafe { kv::geometry(handle) }.unwrap_or_default();
            let names = ["NL", "period", "kvfs", "g_nkv", "g_hd", "s_nkv", "s_hd",
                         "ring_W", "Pmax", "kv_fp16", "kv_fp16_g", "SW", "s_nh"];
            let mut m = serde_json::Map::new();
            for (k, v) in names.iter().zip(g.iter()) {
                m.insert((*k).to_string(), serde_json::json!(v));
            }
            serde_json::Value::Object(m)
        };

        let diff_bytes = a.iter().zip(b.iter()).filter(|(x, y)| x != y).count();
        // Interpret as fp16 — the SWA owners and globals are __half under ADR-012, which
        // is what dominates the blob. Relative error, so a big tensor and a small one are
        // judged on the same scale.
        let mut worst_rel = 0.0f64;
        let mut worst_at = 0usize;
        let halves = n / 2;
        for i in 0..halves {
            let ha = u16::from_le_bytes([a[2 * i], a[2 * i + 1]]);
            let hb = u16::from_le_bytes([b[2 * i], b[2 * i + 1]]);
            if ha == hb { continue; }
            let fa = half_to_f32(ha) as f64;
            let fb = half_to_f32(hb) as f64;
            if !fa.is_finite() || !fb.is_finite() { worst_rel = f64::INFINITY; worst_at = i; break; }
            let denom = fa.abs().max(fb.abs()).max(1e-6);
            let rel = (fa - fb).abs() / denom;
            if rel > worst_rel { worst_rel = rel; worst_at = i; }
        }
        // THE VERDICT REFUSES TO GUESS IN THE MIDDLE. Bit-identical is agreement; a
        // relative error at fp16's own resolution is accumulation order and is expected;
        // anything above 5% is a different tensor and the batched path is not the same
        // forward. The band between is reported as INCONCLUSIVE rather than rounded to
        // whichever answer was hoped for.
        // THE VERDICT MUST NOT SAY "batched" WHEN NOTHING WAS BATCHED. In control mode
        // both halves are the per-token path, and a log line claiming the batched forward
        // agreed — when it never ran — is the kind of sentence that gets believed later.
        let verdict = if control {
            if diff_bytes == 0 { "CONTROL OK — the same path twice is byte-identical, so this instrument can be trusted" }
            else { "CONTROL FAILED — the same path twice DISAGREES; nothing else this route reports is meaningful" }
        } else if diff_bytes == 0 { "IDENTICAL — the batched forward is the same forward" }
            else if worst_rel <= 0.01 { "EQUIVALENT — differs only at fp16 accumulation resolution" }
            else if worst_rel >= 0.05 { "DIVERGENT — the batched forward produces different KV" }
            else { "INCONCLUSIVE — larger than rounding, smaller than a wrong tensor" };
        // WHERE IN THE BLOB, not just how much. The snapshot is laid out owner-by-owner,
        // so a contiguous run of differing bytes localises the fault to a layer group and
        // a flat spread localises it to something global. Without this the only next move
        // is guessing which kernel to read.
        const NB: usize = 32;
        let mut buckets = vec![0u64; NB];
        for i in 0..n {
            if a[i] != b[i] { buckets[i * NB / n] += 1; }
        }
        let first_diff = (0..n).find(|&i| a[i] != b[i]);
        let last_diff = (0..n).rev().find(|&i| a[i] != b[i]);
        let bucket_pct: Vec<f64> = buckets.iter()
            .map(|&c| (c as f64 * 1000.0 / (n as f64 / NB as f64)).round() / 10.0).collect();
        Ok(serde_json::json!({
            "verdict": verdict,
            "mode": req_mode,
            "live_k_values": live_total,
            "live_k_differing": live_diff,
            "live_k_differing_pct": if live_total > 0 {
                (live_diff as f64 * 1000.0 / live_total as f64).round() / 10.0 } else { 0.0 },
            "live_k_worst_rel_err": (live_worst * 1e6).round() / 1e6,
            "geometry": geom_json,
            "live_k_pos0_per_global_layer": pos0.iter().zip(pos0_n.iter())
                .map(|(sum, n)| if *n > 0 { (sum / *n as f64 * 1e4).round() / 1e4 } else { 0.0 })
                .collect::<Vec<f64>>(),
            "live_k_poslast_per_global_layer": poslast.iter().zip(poslast_n.iter())
                .map(|(sum, n)| if *n > 0 { (sum / *n as f64 * 1e4).round() / 1e4 } else { 0.0 })
                .collect::<Vec<f64>>(),
            "live_k_mean_rel_per_global_layer": per_layer.iter().zip(per_layer_n.iter())
                .map(|(sum, n)| if *n > 0 { (sum / *n as f64 * 1e4).round() / 1e4 } else { 0.0 })
                .collect::<Vec<f64>>(),
            "live_k_mean_rel_err": if live_diff > 0 {
                (sum_rel / live_diff as f64 * 1e6).round() / 1e6 } else { 0.0 },
            "live_k_rel_buckets": {"le_1pct": b_tiny, "le_10pct": b_small,
                                   "le_100pct": b_big, "sign_flip": b_flip},
            "live_k_first_bad_pos": live_first_pos,
            "first_diff_byte": first_diff,
            "last_diff_byte": last_diff,
            "bucket_pct_differing": bucket_pct,
            "tokens": p,
            "split": if suffix { serde_json::json!(split) } else { serde_json::Value::Null },
            "snapshot_bytes": n,
            "bytes_differing": diff_bytes,
            "bytes_differing_pct": (diff_bytes as f64 * 1000.0 / n as f64).round() / 10.0,
            "worst_rel_err": (worst_rel * 1e6).round() / 1e6,
            "worst_at_half_index": worst_at,
            "note": "A = per-token (the reference: every warm suffix runs on it). \
                     B = batched under one_shot, exactly as PREFIX-SNAPSHOT captures it."
        }))
    }).await;
    match res {
        Ok(Ok(v)) => (StatusCode::OK, Json(v)).into_response(),
        Ok(Err(e)) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e}))).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": format!("join: {e}")}))).into_response(),
    }
}

/// IEEE half -> f32, without pulling in a crate for six lines.
#[cfg(feature = "wire_cuda_backend")]
fn half_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if man == 0 { sign << 31 } else {
            // subnormal: normalise it
            let mut e = -1i32; let mut m = man;
            while (m & 0x400) == 0 { m <<= 1; e -= 1; }
            let m = m & 0x3ff;
            ((sign << 31) | (((127 - 15 + e + 1) as u32) << 23) | (m << 13)) as u32
        }
    } else if exp == 0x1f {
        (sign << 31) | (0xff << 23) | (man << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

// ── /v1/abort/{id} ────────────────────────────────────────────────────────

pub async fn v1_abort(
    State(state): State<Arc<AppState>>,
    Path(id): Path<u64>,
) -> impl IntoResponse {
    if state.sessions.abort(id) {
        StatusCode::NO_CONTENT
    } else {
        StatusCode::NOT_FOUND
    }
}

// ── /v1/receipts ──────────────────────────────────────────────────────────

pub async fn v1_receipts(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let store = state.receipt_store.lock().unwrap();
    let receipts: Vec<_> = store.iter().map(|r| serde_json::json!({
        "payload_hex": r.payload_hex,
        "sig_hex":     r.sig_hex,
        "round":       r.round,
    })).collect();
    drop(store);
    Json(serde_json::json!({ "receipts": receipts, "cursor": null }))
}

// ── /v1/events ────────────────────────────────────────────────────────────

/// Long-lived SSE channel for daemon-wide events.
/// Emits `event: chat_completed` for chat lifecycle and `event: mint` for
/// new PoUW receipts.
pub async fn v1_events(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let rx = state.events_tx.subscribe();
    let stream = BroadcastStream::new(rx).filter_map(|result| {
        let ev = result.ok()?;
        match ev {
            DaemonEvent::Chat { chat_id, status } => {
                let payload = serde_json::json!({ "chat_id": chat_id, "status": status });
                Some(Ok::<Event, Infallible>(
                    Event::default().event("chat_completed").data(payload.to_string()),
                ))
            }
            DaemonEvent::Mint { receipt_hex, sig_hex } => {
                let payload = serde_json::json!({ "receipt_hex": receipt_hex, "sig_hex": sig_hex });
                Some(Ok::<Event, Infallible>(
                    Event::default().event("mint").data(payload.to_string()),
                ))
            }
            // LM-B2 SSE-SINK: stream the (already class-redacted) telemetry record so a live
            // subscriber (harness StreamProcessor) can sink it into the durable store.
            DaemonEvent::Telemetry { record } => {
                Some(Ok::<Event, Infallible>(
                    Event::default().event("telemetry").data(record),
                ))
            }
        }
    });

    sse_response(
        Sse::new(stream).keep_alive(
            KeepAlive::new()
                .interval(Duration::from_secs(15))
                .text("keepalive"),
        ),
    )
}

// ── /v1/node/telemetry (WS) — migrated from console.rs ───────────────────

#[derive(Serialize)]
struct NodeTelemetry {
    node_id: String,
    cpu_temp_c: f32,
    svm_mem_gb: f32,
    dht_peers_active: u32,
    dht_peers_total: u32,
    pouw_frontier: u64,
}

pub async fn v1_node_telemetry(
    State(state): State<Arc<AppState>>,
    ws: WebSocketUpgrade,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| telemetry_loop(socket, state))
}

async fn telemetry_loop(mut socket: WebSocket, state: Arc<AppState>) {
    loop {
        let peers_active = state.peer_map.len() as u32;
        // CUMULATIVE, not the ring's length: receipt_store is trimmed now, so reading
        // .len() here would cap the operator's counter at RECEIPT_RING_CAP and read as
        // though minting had stopped.
        let pouw_frontier = state.pouw_minted.load(std::sync::atomic::Ordering::Relaxed);
        let payload = NodeTelemetry {
            node_id: "q3-beast-canyon".to_string(),
            cpu_temp_c: 58.5,
            svm_mem_gb: 2.4,
            dht_peers_active: peers_active,
            dht_peers_total: 32,
            pouw_frontier,
        };
        let json = serde_json::to_string(&payload).expect("NodeTelemetry serializable");
        if socket.send(WsMessage::Text(json)).await.is_err() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(1000)).await;
    }
}

// ── /v1/mesh/peers — migrated from console.rs ────────────────────────────

#[derive(Serialize)]
struct PeerInfo {
    node_id:    String,
    address:    String,
    shard_id:   String,
    latency_ms: u32,
}

pub async fn v1_mesh_peers(State(state): State<Arc<AppState>>) -> axum::Json<serde_json::Value> {
    let peers: Vec<PeerInfo> = state.peer_map.iter().map(|entry| {
        let addr     = *entry.key();
        let shard_id = entry.value().shard_id;
        PeerInfo {
            node_id:    addr.to_string(),
            address:    addr.to_string(),
            shard_id:   if shard_id == 0 { "q1".into() } else { "q2".into() },
            latency_ms: 45,
        }
    }).collect();
    axum::Json(serde_json::json!({
        "peers":  peers,
        "active": peers.len(),
        "total":  32,
    }))
}

// ── /v1/pouw/ledger — migrated from console.rs ───────────────────────────

pub async fn v1_pouw_ledger(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let rx = state.events_tx.subscribe();
    let stream = BroadcastStream::new(rx).filter_map(|result| {
        let ev = result.ok()?;
        match ev {
            DaemonEvent::Mint { receipt_hex, .. } => {
                let line = format_kste_receipt(&receipt_hex);
                Some(Ok::<Event, Infallible>(Event::default().data(line)))
            }
            _ => None,
        }
    });
    Sse::new(stream)
        .keep_alive(KeepAlive::new().interval(Duration::from_secs(15)).text("keepalive"))
}

fn format_kste_receipt(receipt_hex: &str) -> String {
    if receipt_hex.len() < 288 {
        return format!("[KSTE] <malformed receipt len={}>", receipt_hex.len());
    }
    let round = decode_le_u64_hex(&receipt_hex[272..288]);
    let nonce = &receipt_hex[16..24];
    let hash  = &receipt_hex[144..152];
    format!("[KSTE] Round: {round} | Nonce: 0x{nonce}... | Z_q Hash: 0x{hash}...")
}

fn decode_le_u64_hex(hex16: &str) -> u64 {
    let mut bytes = [0u8; 8];
    for (i, b) in bytes.iter_mut().enumerate() {
        *b = u8::from_str_radix(&hex16[i * 2..i * 2 + 2], 16).unwrap_or(0);
    }
    u64::from_le_bytes(bytes)
}

// ── /v1/chat/stream stub — migrated from console.rs ─────────────────────

pub async fn v1_chat_stream_stub() -> axum::Json<serde_json::Value> {
    axum::Json(serde_json::json!({"status": "stub", "stream": "sse-legacy"}))
}

// ── POST /v1/dsp/echo — §3-HX Sprint C ───────────────────────────────────
//
// Routes a raw octet-stream body through the V69 cDSP echo skel via
// FastRpcSession + DmaBuffer.  On non-android target_os, returns 501 (the
// FastRPC FFI is gated out at compile time).  On android with no admitted
// session (alloc failure, missing skel on device), also returns 501.
//
// Max body size: 8 MB (Phase 3-HX Sprint C contract; verified end-to-end
// via the parallel `sp_dsp_smoke/src/axum_server.rs` aarch64-android binary).

const DSP_ECHO_MAX_PAYLOAD: usize = 8 * 1024 * 1024;

#[cfg(target_os = "android")]
pub async fn v1_dsp_echo(
    State(state): State<Arc<AppState>>,
    body: axum::body::Bytes,
) -> Response {
    use crate::dsp_rpc::{make_scalars, DmaBuffer, RemoteArg, RemoteBuf, SpErr};
    use std::ffi::c_void;

    let n = body.len();
    if n == 0 {
        return (StatusCode::BAD_REQUEST, "empty body").into_response();
    }
    if n > DSP_ECHO_MAX_PAYLOAD {
        return (StatusCode::PAYLOAD_TOO_LARGE,
                format!("body {n} > {DSP_ECHO_MAX_PAYLOAD}")).into_response();
    }
    let Some(sess_mu) = state.dsp_session.as_ref() else {
        return (StatusCode::NOT_IMPLEMENTED, "cDSP session not admitted").into_response();
    };

    // Wrap the blocking FFI in spawn_blocking so we don't stall the tokio runtime.
    // The session Mutex serializes concurrent requests at the FFI boundary
    // (FastRPC per-handle thread-safety is single-thread).
    let body = body.to_vec();
    let state2 = state.clone();
    let result: Result<Vec<u8>, SpErr> = task::spawn_blocking(move || {
        let sess_mu = state2.dsp_session.as_ref().expect("checked above");
        let sess = sess_mu.lock().expect("dsp session mutex poisoned");
        let mut in_buf:  DmaBuffer = sess.alloc_dma(n)?;
        let mut out_buf: DmaBuffer = sess.alloc_dma(n)?;
        in_buf.as_mut_slice().copy_from_slice(&body);
        for b in out_buf.as_mut_slice().iter_mut() { *b = 0; }

        let mut prim_in: [u32; 2] = [n as u32, n as u32];
        let mut args = [
            RemoteArg { buf: RemoteBuf { pv: prim_in.as_mut_ptr() as *mut c_void, nlen: 8 }},
            RemoteArg { buf: RemoteBuf { pv: in_buf.as_mut_ptr() as *mut c_void,  nlen: n }},
            RemoteArg { buf: RemoteBuf { pv: out_buf.as_mut_ptr() as *mut c_void, nlen: n }},
        ];
        sess.invoke(make_scalars(2, 2, 1), &mut args)?;
        Ok(out_buf.as_slice().to_vec())
    })
    .await
    .unwrap_or_else(|e| Err(SpErr::HandleOpen(-(format!("join: {e}").len() as i32))));
    let _ = sess_mu;

    match result {
        Ok(out) => (StatusCode::OK, out).into_response(),
        Err(e)  => (StatusCode::INTERNAL_SERVER_ERROR,
                    format!("dsp_rpc: {e:?}")).into_response(),
    }
}

#[cfg(not(target_os = "android"))]
pub async fn v1_dsp_echo(
    State(_state): State<Arc<AppState>>,
    body: axum::body::Bytes,
) -> Response {
    let n = body.len();
    if n > DSP_ECHO_MAX_PAYLOAD {
        return (StatusCode::PAYLOAD_TOO_LARGE,
                format!("body {n} > {DSP_ECHO_MAX_PAYLOAD}")).into_response();
    }
    // Host build: FastRPC FFI is gated out.  The route exists so the daemon
    // exposes a uniform surface across host/dev and android/prod; clients
    // get a clear 501 instead of a 404 (which would suggest the route is
    // not deployed at all).
    (StatusCode::NOT_IMPLEMENTED, "v1/dsp/echo requires target_os=android").into_response()
}

// Host build: no DSP-resident model (the loader is android-only). Returns 501
// rather than 404 so the /v1/dsp surface is uniform across host and android.
#[cfg(not(target_os = "android"))]
pub async fn v1_dsp_model_info() -> Response {
    (StatusCode::NOT_IMPLEMENTED, "v1/dsp/model_info requires target_os=android").into_response()
}

// ── §3-HX cDSP model_info (android-only) ─────────────────────────────────────
// Phase 2-L3.FG: /v1/chat + /v1/pouw/ledger now run for real on android (the L1
// forward + sieve C ABI link), so their J.5 501 stubs are gone — the unified
// host handlers above serve both targets.

// /v1/dsp/model_info — reports the DSP-resident model's layer count + total
// DmaBuffer footprint (T_APPSTATE_INTEGRATION). 501 if the model failed to
// load (FastRpcSession unavailable / skel not admitted / bad model path).
#[cfg(target_os = "android")]
pub async fn v1_dsp_model_info(State(state): State<Arc<AppState>>) -> Response {
    let Some(model) = state.dsp_model.as_ref() else {
        return (StatusCode::NOT_IMPLEMENTED, "model not loaded").into_response();
    };
    let hdr = &model.0.header;
    let kv_cache_bytes = state
        .kv_cache
        .as_ref()
        .map(|k| k.lock().unwrap().0.total_bytes())
        .unwrap_or(0);
    Json(serde_json::json!({
        "n_layers":         hdr.n_layers,
        "hidden_size":      hdr.hidden_size,
        "n_heads":          hdr.n_heads,
        "n_kv_heads":       hdr.n_kv_heads,
        "vocab_size":       hdr.vocab_size,
        "total_dma_bytes":  model.0.total_dma_bytes,
        "load_wall_ms":     model.0.load_wall_ms,
        "kv_cache_bytes":   kv_cache_bytes,
    }))
    .into_response()
}

// ── Chat-integration: POST /v1/dialogue ─────────────────────────────────
//
// Single-shot JSON endpoint that drives the M.2 MeMo (Grounding → Entity ID
// → Synthesis) dialogue. Returns the final answer + 3 base64-encoded
// 64-byte SpinorReceipts (one per turn) per `reference-spinor-receipt-layout`.
//
// Returns HTTP 501 if Memory model isn't loaded (--memo-model not passed
// at daemon startup, or model load failed).
//
// This is the Option B parallel endpoint chosen in PLAN-CHAT-INTEGRATION.md;
// the existing /v1/chat is untouched and continues to serve single-model
// SSE streaming chat against the Executive model.

const DIALOGUE_MAX_PROMPT_TOKENS: usize = 64;
const DIALOGUE_MAX_TURN_TOKENS: usize = 8;

#[derive(Deserialize)]
pub struct DialogueRequest {
    pub prompt: String,
}

#[derive(Serialize)]
struct DialogueResponse {
    response: String,
    receipts: Vec<String>, // base64-encoded 64-byte SpinorReceipts
    wall_ms: u64,
    turn_us: [u64; 3],
}

/// STANDARD base64 encoder (RFC 4648). Hand-rolled to avoid adding a
/// dependency for ~6 lines of code; verified against the standard
/// alphabet `ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/`
/// with `=` padding. 64 input bytes → 88 output chars (ceil(64/3)*4 = 88).
fn base64_encode(input: &[u8]) -> String {
    const ALPHABET: &[u8; 64] =
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(((input.len() + 2) / 3) * 4);
    let mut i = 0;
    while i + 3 <= input.len() {
        let n = ((input[i] as u32) << 16) | ((input[i + 1] as u32) << 8) | (input[i + 2] as u32);
        out.push(ALPHABET[((n >> 18) & 0x3F) as usize] as char);
        out.push(ALPHABET[((n >> 12) & 0x3F) as usize] as char);
        out.push(ALPHABET[((n >> 6) & 0x3F) as usize] as char);
        out.push(ALPHABET[(n & 0x3F) as usize] as char);
        i += 3;
    }
    let rem = input.len() - i;
    if rem == 1 {
        let n = (input[i] as u32) << 16;
        out.push(ALPHABET[((n >> 18) & 0x3F) as usize] as char);
        out.push(ALPHABET[((n >> 12) & 0x3F) as usize] as char);
        out.push('=');
        out.push('=');
    } else if rem == 2 {
        let n = ((input[i] as u32) << 16) | ((input[i + 1] as u32) << 8);
        out.push(ALPHABET[((n >> 18) & 0x3F) as usize] as char);
        out.push(ALPHABET[((n >> 12) & 0x3F) as usize] as char);
        out.push(ALPHABET[((n >> 6) & 0x3F) as usize] as char);
        out.push('=');
    }
    out
}

pub async fn v1_dialogue(
    State(state): State<Arc<AppState>>,
    Json(req): Json<DialogueRequest>,
) -> Response {
    // 501 if Memory model isn't loaded.
    if state.memo_model.is_none() || state.memo_session.is_none() || state.memo_tokenizer.is_none() {
        return (
            StatusCode::NOT_IMPLEMENTED,
            Json(serde_json::json!({
                "error": "memo_model_not_loaded",
                "hint": "start sp-daemon with --memo-model / --memo-tokenizer or SP_MEMO_MODEL_PATH / SP_MEMO_TOKENIZER_PATH",
            })),
        )
            .into_response();
    }

    if req.prompt.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "prompt required"})),
        )
            .into_response();
    }

    // Clone Executive base session (mirrors /v1/chat lines 150-154).
    let exec_cancel = Arc::new(AtomicI32::new(0));
    let exec_child = {
        let guard = state.session.as_ref().expect("L1 session unavailable (qwen36 lane)").lock().unwrap();
        guard.clone_session(exec_cancel.clone())
    };
    let mut exec_child = match exec_child {
        Ok(s) => s,
        Err(e) => {
            return (StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": format!("exec clone: {e}")})))
                .into_response();
        }
    };

    // Clone Memory base session.
    let memo_cancel = Arc::new(AtomicI32::new(0));
    let memo_child = {
        let guard = state.memo_session.as_ref().expect("checked above").lock().unwrap();
        guard.clone_session(memo_cancel.clone())
    };
    let mut memo_child = match memo_child {
        Ok(s) => s,
        Err(e) => {
            return (StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": format!("memo clone: {e}")})))
                .into_response();
        }
    };

    let exec_tokenizer = state.tokenizer.clone();
    let exec_vocab = state.vocab_size;
    let memo_vocab = state.memo_vocab_size;
    let prompt = req.prompt;

    // Signal the mining loop to back off for the duration (mirrors /v1/chat:176).
    state.inference_active.store(true, std::sync::atomic::Ordering::Relaxed);
    struct InferenceGuard(Arc<AtomicBool>);
    impl Drop for InferenceGuard {
        fn drop(&mut self) { self.0.store(false, std::sync::atomic::Ordering::Relaxed); }
    }
    let _guard = InferenceGuard(state.inference_active.clone());

    // Drive the dialogue on a spawn_blocking thread (L1 forward is sync).
    let app_for_guard = state.clone();
    let result = tokio::task::spawn_blocking(move || {
        let _g = _guard;
        // F-9 (2026-08-29 audit): the cloned sessions inherit the CUDA forward
        // backend, so this is a device forward like any other background lane —
        // one forward at a time, parked behind a live turn.
        let _gpu = device_guard(&app_for_guard, "dialogue");
        let caps = sp_daemon::dialogue::DialogueCaps {
            max_prompt_tokens: DIALOGUE_MAX_PROMPT_TOKENS,
            max_query_tokens:  DIALOGUE_MAX_TURN_TOKENS,
            max_response_tokens: DIALOGUE_MAX_TURN_TOKENS,
            max_answer_tokens: DIALOGUE_MAX_TURN_TOKENS,
        };
        let mut pool = sp_daemon::dialogue::DialoguePool::new(exec_vocab, memo_vocab, &caps);
        crate::dialogue_runner::run_dialogue(
            &mut exec_child,
            &mut memo_child,
            exec_tokenizer.as_ref(),
            &mut pool,
            &prompt,
            &caps,
        )
    }).await;

    match result {
        Ok(Ok(outcome)) => {
            // ledger-autowire: best-effort append of all 3 receipts to the
            // shared PoUW ledger BEFORE building the HTTP response. Per the
            // sprint plan + the broader M.4 design, the ledger is
            // observational, not a transactional gate — a lock or append
            // failure logs a warning and the response still ships. The
            // critical section is ~10 µs total (3 × p99=3 µs per M.4
            // measurement) so contention is structurally irrelevant.
            if let Some(ledger) = &state.ledger {
                match ledger.lock() {
                    Ok(mut guard) => {
                        for (idx, r) in outcome.receipts.iter().enumerate() {
                            if let Err(e) = guard.append(r) {
                                tracing::warn!(
                                    error = %e,
                                    receipt_idx = idx,
                                    "ledger-autowire: append failed; response still returns"
                                );
                            }
                        }
                    }
                    Err(poisoned) => {
                        tracing::warn!(
                            error = ?poisoned,
                            "ledger-autowire: mutex poisoned; skipping append"
                        );
                    }
                }
            }
            let receipts_b64: Vec<String> = outcome
                .receipts
                .iter()
                .map(|r| base64_encode(&r.as_bytes()))
                .collect();
            let body = DialogueResponse {
                response: outcome.final_answer,
                receipts: receipts_b64,
                wall_ms: outcome.total_wall_us / 1000,
                turn_us: outcome.turn_us,
            };
            (StatusCode::OK, Json(serde_json::to_value(&body).unwrap_or(serde_json::json!({})))).into_response()
        }
        Ok(Err(e)) => {
            (StatusCode::INTERNAL_SERVER_ERROR,
             Json(serde_json::json!({"error": format!("run_dialogue: {e}")})))
            .into_response()
        }
        Err(e) => {
            (StatusCode::INTERNAL_SERVER_ERROR,
             Json(serde_json::json!({"error": format!("spawn_blocking: {e}")})))
            .into_response()
        }
    }
}

#[cfg(test)]
mod chat_integration_tests {
    use super::*;

    #[test]
    fn base64_encode_known_vectors() {
        // RFC 4648 test vectors.
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foob"), "Zm9vYg==");
        assert_eq!(base64_encode(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn base64_encode_64_bytes_to_88_chars() {
        let input = [0xA5u8; 64];
        let out = base64_encode(&input);
        assert_eq!(out.len(), 88);
        // First char of all-0xA5 input: 0xA5=10100101 → first 6 bits = 101001 = 41 = 'p'
        assert!(out.starts_with('p'));
    }
}

// ── KAI-5: read-only hidden-state tap (/v1/hidden). Default-OFF (SP_HIDDEN_TAP=1).
// Runs gemma4_cuda_probe(attn_only=0) on the RESIDENT GPU weights (g_w) and returns the
// per-position residual hidden [n, E=3840]. Read-only: the probe allocates its OWN
// scratch and never touches the resident KV cache, so the served forward is unaffected.
// For offline Voice-Head self-distillation (ADR-KAI5); use with the daemon otherwise idle.
mod hidden_tap {
    use std::os::raw::{c_int, c_void};
    extern "C" {
        pub fn sp_model_to_gemma4(m: *const c_void) -> *const c_void;
        pub fn gemma4_cuda_probe(m: *const c_void, tokens: *const i32, n_tok: c_int,
                                 n_layers: c_int, attn_only: c_int, out_x: *mut f32) -> c_int;
        pub fn sp_last_error() -> *const std::os::raw::c_char;
    }
    pub fn last_error() -> String {
        unsafe {
            let p = sp_last_error();
            if p.is_null() { return "(none)".into(); }
            std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    }
}

#[derive(serde::Deserialize)]
pub struct HiddenReq { pub tokens: Vec<i32> }

#[derive(serde::Serialize)]
pub struct HiddenResp { pub n: usize, pub e: usize, pub hidden: Vec<f32> }

/// Reconstruct the gemma4 view (qwen3_model) once and cache the pointer (weights are
/// shared with the resident model — never freed here).
static HIDDEN_QM: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

pub async fn v1_hidden(
    State(state): State<Arc<AppState>>,
    Json(req): Json<HiddenReq>,
) -> Response {
    if std::env::var("SP_HIDDEN_TAP").ok().as_deref() != Some("1") {
        return (StatusCode::FORBIDDEN, "hidden tap disabled (set SP_HIDDEN_TAP=1)").into_response();
    }
    let n = req.tokens.len();
    if n == 0 { return (StatusCode::BAD_REQUEST, "no tokens").into_response(); }
    const E: usize = 3840;   // the dense reference model hidden width
    const NL: i32 = 48;      // the dense reference model layers
    // scoped so no raw pointer lives across the .await (a !Send future breaks the handler)
    let qm: usize = {
        let sp_ptr = state.model.as_ptr() as *const std::os::raw::c_void;
        *HIDDEN_QM.get_or_init(|| unsafe { hidden_tap::sp_model_to_gemma4(sp_ptr) } as usize)
    };
    if qm == 0 { return (StatusCode::INTERNAL_SERVER_ERROR, "sp_model_to_gemma4 null").into_response(); }
    // ── THE TAP IS A FORWARD (2026-08-29 audit, F-8): this ran gemma4_cuda_probe
    // with no device lock, no yield, ON THE RUNTIME THREAD — the distill harness
    // tapping mid-turn was the wedge class through the door tonight's pass missed,
    // plus an executor stall for the probe's duration. Same shape as every other
    // background forward now: blocking pool + device guard.
    let app = state.clone();
    let toks = req.tokens.clone();
    let res = task::spawn_blocking(move || -> Result<Vec<f32>, String> {
        let _gpu = device_guard(&app, "hidden_tap");
        let mut out = vec![0f32; n * E];
        let rc = unsafe {
            hidden_tap::gemma4_cuda_probe(qm as *const std::os::raw::c_void, toks.as_ptr(),
                                          n as i32, NL, 0, out.as_mut_ptr())
        };
        if rc != 0 { return Err(format!("probe rc={rc}: {}", hidden_tap::last_error())); }
        Ok(out)
    }).await;
    match res {
        Ok(Ok(out)) => Json(HiddenResp { n, e: E, hidden: out }).into_response(),
        Ok(Err(e)) => (StatusCode::INTERNAL_SERVER_ERROR, e).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, format!("join: {e}")).into_response(),
    }
}
