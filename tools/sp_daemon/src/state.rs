use std::net::SocketAddr;
use std::sync::{atomic::{AtomicI32, AtomicU64, AtomicBool}, Arc, Mutex};
use std::time::Instant;
use dashmap::DashMap;
use tokio::sync::broadcast;

// ConnectedPeer lives in the lib (quic_shard) so run_garner_loop can write it
// without creating a lib→binary dependency.
pub use sp_daemon::network::quic_shard::ConnectedPeer;

use ed25519_dalek::SigningKey;
use serde::Serialize;

use crate::session::{SpModel, SpSession};
use crate::sessions::Sessions;
use crate::tokenizer::SptbTokenizer;
// ledger-autowire: shared PoUW ledger handle for /v1/dialogue auto-append.
use sp_daemon::pouw_ledger::Ledger;

/// Events broadcast on the /v1/events SSE channel.
#[derive(Clone, Debug)]
pub enum DaemonEvent {
    /// A chat request completed or was cancelled.
    Chat { chat_id: u64, status: &'static str },
    /// A sieve-fold receipt was minted.
    Mint { receipt_hex: String, sig_hex: String },
    /// LM-B2/B2b: a memory telemetry record (decision or turn), pre-serialized JSON, already
    /// class-redacted at the emit site. Broadcast so a live subscriber (e.g. the harness
    /// StreamProcessor) can sink it into the durable store without tailing the JSONL file.
    Telemetry { record: String },
}

/// A minted PoUW receipt stored in AppState::receipt_store.
#[derive(Clone, Serialize)]
pub struct ReceiptRecord {
    /// Hex-encoded 152-byte receipt wire format (frozen v1).
    pub payload_hex: String,
    /// Hex-encoded 64-byte ed25519 signature over payload_hex bytes.
    pub sig_hex:     String,
    /// Global sieve-fold counter at mint time.
    pub round:       u64,
}

/// Shared daemon state — threaded through axum via `State<Arc<AppState>>`.
///
/// Drop order (declaration order) keeps models alive past every session:
/// sessions drop before models, so mmap-backed pointers in sp_session remain valid.
///
/// Phase 2-L3.FG: unified across host + android (the L1 C ABI now links on
/// android). The cDSP DSP-bridge fields at the end are `cfg(android)` additions;
/// host builds are byte-identical to pre-L3.FG (the cfg(android) fields vanish).
pub struct AppState {
    /// Target/verifier model. Kept alive so session mmap-backed pointers remain valid.
    #[allow(dead_code)]
    pub model: SpModel,
    /// Base target session at position 0. Held only briefly during sp_session_clone.
    /// `None` ONLY on the qwen36 lane (arch_id 8): the L1 session layer does not
    /// dispatch the GDN+MoE hybrid, so those routes are unreachable there
    /// (CONTRACT-QWEN36-SERVE) — every user unwraps with an explicit expect.
    pub session: Option<Mutex<SpSession>>,
    #[allow(dead_code)]
    pub cancel_flag: Arc<AtomicI32>,
    /// Draft model for speculative decoding (Phase 4-SPEC). None in single-model mode.
    #[allow(dead_code)]
    pub draft_model: Option<SpModel>,
    /// Base draft session at position 0. Cloned per spec-decode request.
    pub draft_session: Option<Mutex<SpSession>>,
    /// Active chat registry — maps chat_id → per-chat cancel_flag.
    pub sessions: Arc<Sessions>,
    /// Logits buffer width for this model; set once at startup from sp_arch_info.
    pub vocab_size: usize,
    /// Sprint WIRE-HEX — true when the engine's `gemma3_forward_hexagon`
    /// dispatcher is registered on `session` for sp_prefill_chunk.
    /// Toggled by SP_DAEMON_BACKEND=hex at startup (only meaningful when the
    /// daemon was built with the `wire_hex_backend` Cargo feature; always
    /// false on host builds and on android builds without the feature).
    /// Surfaced via /v1/metrics so the headline tok/s diff is attributable.
    pub wire_hex_active: bool,
    /// Sprint WIRE-CPU — true when the engine's CPU AVX-512 backend
    /// (cpu_overlay.c + cpu_forward.c + cpu_gemma3.c) is registered on
    /// `session` for sp_prefill_chunk. Toggled by SP_DAEMON_BACKEND=cpu at
    /// startup (only meaningful when the daemon was built with the
    /// `wire_cpu_backend` Cargo feature; always false without the feature).
    /// HOST target — works on Windows MSVC + Linux GCC/Clang. Surfaced via
    /// /v1/debug/backend_counts alongside `wire_hex_active`.
    pub wire_cpu_active: bool,
    /// Sprint WIRE-CUDA — true when the engine's `gemma3_forward_cuda` /
    /// `qwen3_forward_cuda` dispatcher is registered on `session` for
    /// sp_prefill_chunk. Toggled by SP_DAEMON_BACKEND=cuda at startup (only
    /// meaningful when the daemon was built with the `wire_cuda_backend`
    /// Cargo feature; always false on builds without the feature). Surfaced
    /// via /v1/debug/backend_counts.
    pub wire_cuda_active: bool,
    /// Sprint WIRE-VULKAN — true when the engine's gemma3_forward_vulkan /
    /// qwen3_forward_vulkan dispatcher is registered on `session` for
    /// sp_prefill_chunk. Toggled by SP_DAEMON_BACKEND=vulkan at startup
    /// (only meaningful when the daemon was built with the
    /// `wire_vulkan_backend` Cargo feature; always false otherwise).
    /// Host-side (Windows / Linux / macOS). Surfaced via
    /// /v1/debug/backend_counts so the headline tok/s diff is attributable.
    pub wire_vulkan_active: bool,
    /// Lifetime token counter for rolling tps in /v1/metrics.
    pub tokens_decoded: AtomicU64,
    /// Daemon start time (for tps denominator).
    pub started_at: Instant,
    /// Broadcast channel for daemon-wide events (/v1/events subscribers).
    pub events_tx: broadcast::Sender<DaemonEvent>,
    /// Tokenizer built from the .sp-tokenizer SPTB blob at startup.
    pub tokenizer: Arc<SptbTokenizer>,
    /// True while a /v1/chat request is actively being processed.
    /// The mining loop backs off when this is set.
    pub inference_active: Arc<AtomicBool>,
    /// The most recent PoUW receipts. A RING, capped at mining::RECEIPT_RING_CAP — it was
    /// an unbounded Vec growing ~40 MB/day at the observed ~1 receipt/second.
    pub receipt_store: Arc<Mutex<Vec<ReceiptRecord>>>,
    /// CUMULATIVE receipts minted this process. Separate from receipt_store.len() on
    /// purpose: the store is now trimmed, and telemetry's pouw_frontier is a running
    /// TOTAL — reporting the ring's length instead would silently cap the operator's
    /// counter at the ring size and read as if minting had stopped.
    pub pouw_minted: Arc<AtomicU64>,
    /// ed25519 node keypair used to sign PoUW receipts.
    pub node_signing_key: SigningKey,
    /// Active QUIC peers indexed by remote SocketAddr.
    /// Populated by run_garner_loop on accept; cleared on connection close.
    /// Empty until the coordinator is wired into daemon startup.
    pub peer_map: Arc<DashMap<SocketAddr, ConnectedPeer>>,

    // Chat-integration: Memory model for the MeMo dialogue endpoint (/v1/dialogue).
    // All three Memory-* fields are Option<...>: None when --memo-model is not
    // passed at startup; the /v1/dialogue route returns HTTP 501 in that case.
    // Drop order (declaration order) puts memo_session before memo_model so the
    // L1 session destructor runs first, then sp_model_unload, like the
    // target-model pair above.
    /// Chat-integration: Memory base session (cloned per /v1/dialogue request).
    pub memo_session: Option<Mutex<SpSession>>,
    /// Chat-integration: Memory tokenizer (used by dialogue_runner's
    /// `final_answer` byte stream — Memory side decodes its own tokens).
    pub memo_tokenizer: Option<Arc<SptbTokenizer>>,
    /// Chat-integration: Memory model handle. Kept alive after memo_session so
    /// the mmap-backed pointers in the session remain valid.
    pub memo_model: Option<SpModel>,
    /// Chat-integration: vocab size of the Memory model — used to pre-allocate
    /// DialoguePool's memo_logits slot at request time.
    pub memo_vocab_size: usize,

    // ledger-autowire: optional PoUW receipt ledger. When set
    // (--pouw-ledger-path / SP_POUW_LEDGER_PATH at startup), the /v1/dialogue
    // handler appends each of the 3 SpinorReceipts per dialogue to this
    // shared, mutex-serialized handle. None disables ledger autowire (the
    // HTTP response still returns receipts; nothing is persisted daemon-side).
    pub ledger: Option<Arc<Mutex<Ledger>>>,

    /// Sprint WIRE-CUDA-DECODE-GEMMA4 — session-resident `sp_g4_kv*` KV-decode
    /// cache (the persistent-KV decode path the prefill hook cannot serve; see
    /// `WIRE-CUDA-DECODE-GEMMA4.md`). `Some` when the daemon was built with
    /// `--features wire_cuda_backend` AND opened the cache at startup
    /// (`SP_DAEMON_BACKEND=cuda` + `SP_DAEMON_KVDECODE=1`, the INTEGRATION step).
    /// The inner pointer is an opaque device handle owned by the CUDA backend;
    /// `CudaKvDecodeHandle` is the `Send + Sync` wrapper. `Mutex` serializes the
    /// step calls (the resident cache is single-writer). Dropped before `model`
    /// (declaration order) so the device cache frees while the model is alive.
    #[cfg(feature = "wire_cuda_backend")]
    pub cuda_kvdecode_handle: Option<Mutex<CudaKvDecodeHandle>>,

    /// CONTRACT-CHAT-FULLSTACK B3 — AUTONOMOUS MEMORY RECALL. The episode
    /// registry (loaded from `SP_RECALL_REGISTRY` at startup; one JSONL row per
    /// episode {dir, npos, topic, sig_bits}) + the frozen ±1 C2 projection R.
    /// `Some` only when the env var points at a readable registry AND >=1 row
    /// parsed; otherwise `None` and `auto_recall:true` is a no-op (the turn falls
    /// straight through to the byte-untouched non-recall path). Read-only after
    /// startup, so no Mutex.
    pub recall_registry: Option<Vec<sp_daemon::recall::Episode>>,
    pub recall_proj: Arc<sp_daemon::recall::Projection>,

    /// CONTRACT-CHAT-FULLSTACK B4 — NIGHTSHIFT. Live, between-turn consolidated
    /// episodes (`SP_B4_NIGHTSHIFT=1`): a user turn that states a fact is captured
    /// at turn-end as a position-0 standalone episode (W_c-head-compatible) and
    /// pushed here, so the head can self-select it on a LATER turn. Unlike the
    /// immutable curated `recall_registry`, this grows live ⇒ RwLock. Empty until a
    /// turn is consolidated; default-off (env unset) ⇒ never written ⇒ null floor.
    pub nightshift: Arc<std::sync::RwLock<Vec<sp_daemon::recall::Episode>>>,

    /// NORTHSTAR serve (CONTRACT-QWEN36-SERVE) — the qwen36 (35B-A3B GDN+MoE
    /// hybrid) chat lane. `Some` only when the loaded model's arch_id ==
    /// SP_ARCH_ID_QWEN36 (8); /v1/chat then decodes via qwen36_step (GPU hybrid
    /// hooks booted once at daemon start) instead of the gemma L1
    /// session/kvdecode path. `None` = every existing path byte-untouched.
    pub qwen36_lane: Option<Arc<crate::qwen36_lane::Qwen36Lane>>,

    // ── §3-HX cDSP bridge (android-only) ─────────────────────────────────────
    /// §3-HX Sprint C — FastRpcSession for the V69 cDSP echo skel. `None` if the
    /// skel could not be admitted; `/v1/dsp/echo` then returns 501. Per-request
    /// Mutex serializes the FFI invoke (FastRPC per-handle is single-thread).
    #[cfg(target_os = "android")]
    pub dsp_session: Option<Mutex<crate::dsp_rpc::FastRpcSession>>,
    /// §3-HX Sprint J.5 — DSP-resident Qwen3 model, loaded at startup into
    /// per-tensor rpcmem DmaBuffers, backed by a leaked `&'static` FastRpcSession
    /// separate from `dsp_session`. `None` → `/v1/dsp/model_info` 501s.
    #[cfg(target_os = "android")]
    pub dsp_model: Option<Arc<ModelHandle>>,
    /// Per-layer K/V DmaBuffers at ctx_max=4096. `Mutex` is for Sprint K's
    /// decode-time mutation; L3.FG only reads `total_bytes()`.
    #[cfg(target_os = "android")]
    pub kv_cache: Option<Arc<Mutex<KvCacheHandle>>>,

    /// §4-NTT Sprint NTT.5b — optional compute-backend for Memory model's
    /// NTT-attention routed through Hexagon HVX via FastRPC. Held in
    /// AppState so the Arc<FastRpcSession> + raw pointer registered with
    /// L1's `sp_session_register_compute_backend` stays valid for the
    /// session's lifetime. None when SP_ENGINE_NTT_ATTN_HEX is unset OR
    /// the Memory model is not loaded.
    ///
    /// NOTE: registering this backend with the Memory session is plumbing
    /// only — the actual env-gated activation in forward.c is OUT OF SCOPE
    /// for NTT.5b per the sprint spec. Until forward.c wire-up lands in a
    /// follow-on sprint, the registered backend is stored but never invoked.
    #[cfg(target_os = "android")]
    pub ntt_hex_backend: Option<Arc<sp_daemon::ntt_hex_dispatch::ComputeBackend>>,
}

/// Sprint WIRE-CUDA-DECODE-GEMMA4 — `Send + Sync` wrapper for the resident
/// `sp_g4_kv*` KV-decode cache handle (an opaque CUDA device pointer, hence
/// `!Send + !Sync`). axum's `State<Arc<AppState>>` requires `Send + Sync`.
///
/// Soundness: the handle is created on the startup thread by
/// `cuda_kvdecode_dispatch::register_with_session` and every subsequent
/// `decode_step` / `rewind` / `close` is serialized by the enclosing
/// `Mutex<CudaKvDecodeHandle>` in `AppState` — only one thread ever holds the
/// raw pointer at a time, and the CUDA backend's own stream (g_w.stream) is the
/// single device queue. The pointer is never dereferenced in Rust; it only
/// crosses the FFI boundary back into the glue, which owns the device state.
#[cfg(feature = "wire_cuda_backend")]
pub struct CudaKvDecodeHandle(pub *mut std::ffi::c_void);
#[cfg(feature = "wire_cuda_backend")]
unsafe impl Send for CudaKvDecodeHandle {}
#[cfg(feature = "wire_cuda_backend")]
unsafe impl Sync for CudaKvDecodeHandle {}
#[cfg(feature = "wire_cuda_backend")]
impl Drop for CudaKvDecodeHandle {
    fn drop(&mut self) {
        // SAFETY: the pointer is an sp_g4_kv* from register_with_session (or
        // NULL); close is NULL-safe; not used after drop.
        unsafe { sp_daemon::cuda_kvdecode_dispatch::release_for_model(self.0) };
    }
}

/// §3-HX Sprint J.5 — `Send + Sync` wrapper for the DSP-resident model.
///
/// `DspModel<'static>` holds `Vec<DmaBuffer>`, and `DmaBuffer` wraps a raw
/// rpcmem `*mut u8` → `!Send + !Sync`. axum's `State<Arc<AppState>>` requires
/// `Send + Sync`. The unsafe impls are sound for J.5 because the model is
/// **load-and-read-only**: after `DspModel::load` returns on the startup
/// thread, only plain metadata (`header`, `total_dma_bytes`) is read via
/// `/v1/dsp/model_info`; the rpcmem pointers are never dereferenced or invoked
/// across threads. Sprint K, which drives FastRPC invokes against these
/// buffers, must add per-invoke serialization (the deferred `Mutex`).
#[cfg(target_os = "android")]
pub struct ModelHandle(pub crate::dsp_model::DspModel<'static>);
#[cfg(target_os = "android")]
unsafe impl Send for ModelHandle {}
#[cfg(target_os = "android")]
unsafe impl Sync for ModelHandle {}

/// `Send + Sync` wrapper for the DSP-resident KV cache. Same soundness
/// argument as [`ModelHandle`]: J.5 reads only `total_bytes()`.
#[cfg(target_os = "android")]
pub struct KvCacheHandle(pub crate::kv_cache::KvCache<'static>);
#[cfg(target_os = "android")]
unsafe impl Send for KvCacheHandle {}
#[cfg(target_os = "android")]
unsafe impl Sync for KvCacheHandle {}
