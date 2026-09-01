mod daemon;
#[cfg(target_os = "android")]
mod dsp_rpc;
// §3-HX Sprint J.5 — import (not move) the Sprint J loader from sp_dsp_smoke.
// #[path] keeps the loader files in their crate; `use crate::dsp_rpc::…` inside
// them resolves to this crate's dsp_rpc (signatures verified identical), so no
// duplicate FastRpcSession type and no smoke [lib] target. Tech-debt: the
// cross-tree path marks the loader's eventual real home (a shared crate).
#[cfg(target_os = "android")]
#[path = "../../sp_dsp_smoke/src/dsp_model.rs"]
mod dsp_model;
#[cfg(target_os = "android")]
#[path = "../../sp_dsp_smoke/src/kv_cache.rs"]
mod kv_cache;
// Phase 2-L3.FG: the L1 C-ABI inference path + sieve-backed mining/tokenizer now
// cross-compile and link on aarch64-android (build.rs links build-android-libs),
// so these modules are unconditional again. The J.5 host-gating is removed; the
// android-only surface is just the cDSP bridge (dsp_rpc/dsp_model/kv_cache).
mod ffi;
mod mining;
mod routes;
// CONTRACT-CHAT-FULLSTACK A2: the L2 sampler (temperature / top-k / top-p /
// repetition penalty / seeded RNG) over the full-vocab logits row. temp=0 =
// strict argmax null floor (G-CHAT-A2 determinism leg).
mod sampler;
mod server;
mod session;
mod sessions;
// NORTHSTAR serve (CONTRACT-QWEN36-SERVE): the qwen36 35B-A3B GDN+MoE chat lane.
mod qwen36_lane;
// ADR-002 realized: the Decide→Execute spine (unifies the recall/decline/route logic).
mod spine;
mod sieve_ffi;
mod spec;
mod state;
mod tokenizer;
// CONSTRAINED TOOL DECODING (DOMINO-aligned): a hallucinated tool name is not a typo to be
// healed in the harness — it is a token sequence the sampler cannot produce. Off unless the
// request supplies tool_names, and it masks NOTHING while she is talking.
mod tool_mask;
// Chat-integration: daemon-callable MeMo dialogue runner — host + android
// safe; drives M.2's 3-turn Grounding → Entity ID → Synthesis protocol
// through the existing crate::session::SpSession wrapper.
mod dialogue_runner;

// KAI-1 Alpha — the inference-driven `decide_via_model` heartbeat decider
// (binary-crate half of KAIROS; the lib half is sp_daemon::kairos). Behind the
// off-by-default `kairos` feature: when unset this module does not exist and the
// daemon binary is byte-identical. Dispatched by the SP_KAIROS_ALPHA env gate at
// the top of main() — never touches the daemon startup path.
#[cfg(feature = "kairos")]
mod kairos_runner;
mod nightshift_curator;
// SP_EAGLE_ACCEPT one-shot: framework-faithful live MTP single-token acceptance probe.
// The file is #![cfg(feature="wire_cuda_backend")] so this module is empty without the
// CUDA backend. It must live in the binary (not a [[bin]]) to reach the binary-private
// tokenizer/sampler/session (the token-management contract).
mod eagle_accept;
// Telepathy — the LatentBridge (tokenizer-free latent->latent transport). Pure-Rust object + adapter
// + routing primitive + fail-closed license; SP_TELEPATHY default-off. See telepathy.rs / the spec.
mod telepathy;

// Sprint WIRE-CPU — host CPU AVX-512 full-forward backend dispatcher for
// sp_l1.h:§6. Symmetric to the lib-crate's hex_forward_dispatch (android).
// Activates when SP_DAEMON_BACKEND=cpu is set AND the daemon was built
// with --features wire_cpu_backend so libsp_cpu_daemon_backend is linked.
#[cfg(feature = "wire_cpu_backend")]
mod cpu_forward_dispatch;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "sp-daemon",
    about = "Shannon Prime L3 HTTP/SSE daemon (lat-phase-2-l3-core)",
    long_about = "Long-lived daemon that wraps the frozen L1 C ABI in an HTTP \
                  server on 127.0.0.1:8080. All four frontends (mobile, desktop, \
                  watch, CLI) attach to this process."
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Cmd>,

    // Internal flag: this process IS the inner daemon (spawned by `start`).
    #[arg(long, hide = true)]
    daemon_inner: bool,
    #[arg(long, default_value = "", hide = true)]
    model: String,
    #[arg(long, default_value = "", hide = true)]
    tokenizer: String,
    #[arg(long, default_value = "", hide = true)]
    draft_model: String,
    #[arg(long, default_value = "", hide = true)]
    draft_tokenizer: String,
    // Chat-integration: Memory model for /v1/dialogue (empty = disabled).
    #[arg(long, default_value = "", hide = true)]
    memo_model: String,
    #[arg(long, default_value = "", hide = true)]
    memo_tokenizer: String,
    // ledger-autowire: PoUW receipt ledger path (empty = disabled).
    #[arg(long, default_value = "", hide = true)]
    pouw_ledger_path: String,
    #[arg(long, default_value = "0", hide = true)]
    quic_port: u16,
    #[arg(long, default_value = "8080", hide = true)]
    port: u16,
    #[arg(long, default_value = "", hide = true)]
    peer: String,
    #[arg(long, default_value = "", hide = true)]
    peers: String,
}

#[derive(Subcommand)]
enum Cmd {
    /// Spawn the daemon process and exit.
    Start {
        /// Path to the .sp-model file (or set SP_MODEL_PATH).
        #[arg(long, env = "SP_MODEL_PATH")]
        model: String,
        /// Path to the .sp-tokenizer file (or set SP_TOKENIZER_PATH).
        #[arg(long, env = "SP_TOKENIZER_PATH")]
        tokenizer: String,
        /// Path to draft .sp-model for Phase 4-SPEC speculative decode (optional).
        #[arg(long, env = "SP_DRAFT_MODEL_PATH", default_value = "")]
        draft_model: String,
        /// Path to draft .sp-tokenizer (optional, required if --draft-model is set).
        #[arg(long, env = "SP_DRAFT_TOKENIZER_PATH", default_value = "")]
        draft_tokenizer: String,
        /// Chat-integration: path to Memory .sp-model for the /v1/dialogue MeMo
        /// endpoint (optional; if unset the endpoint returns HTTP 501).
        #[arg(long, env = "SP_MEMO_MODEL_PATH", default_value = "")]
        memo_model: String,
        /// Chat-integration: path to Memory .sp-tokenizer (required if
        /// --memo-model is set).
        #[arg(long, env = "SP_MEMO_TOKENIZER_PATH", default_value = "")]
        memo_tokenizer: String,
        /// ledger-autowire: path to the PoUW receipt ledger (append-only file).
        /// If set, every /v1/dialogue invocation auto-appends its 3
        /// SpinorReceipts here in addition to returning them in the HTTP
        /// response. If unset, the autowire is disabled (no ledger
        /// persistence; receipts still surface in the response).
        #[arg(long, env = "SP_POUW_LEDGER_PATH", default_value = "")]
        pouw_ledger_path: String,
        /// UDP port for the QUIC DHT mesh coordinator (set SP_QUIC_PORT or 0 to disable).
        #[arg(long, env = "SP_QUIC_PORT", default_value = "0")]
        quic_port: u16,
        /// TCP port for the main HTTP API server (set SP_HTTP_PORT).
        #[arg(long, env = "SP_HTTP_PORT", default_value = "8080")]
        port: u16,
        /// Dial this QUIC peer address on startup (e.g. 127.0.0.1:5000). Back-compat alias for --peers with one entry.
        #[arg(long, default_value = "")]
        peer: String,
        /// Comma-separated list of QUIC peer addresses to dial on startup (set SP_PEERS).
        #[arg(long, env = "SP_PEERS", default_value = "")]
        peers: String,
    },
    /// Stop the running daemon (sends SIGTERM / taskkill).
    Stop,
    /// Reload configuration (no-op for v0).
    Reload,
}

#[tokio::main]
async fn main() {
    // KAI-1 Alpha (G-KAIROS-1 telemetry) — env-gated, runs the inference-driven
    // heartbeat over a scripted §2b tape and exits BEFORE any clap parsing or
    // daemon startup. Feature-gated so default builds don't compile this at all
    // (null floor). Usage:
    //   SP_KAIROS_ALPHA=1 SP_KAIROS_MODEL=... SP_KAIROS_TOK=... \
    //   SP_KAIROS_TAPE=tools/sp_daemon/tests/fixtures/kairos/tape_smoke.txt \
    //   [SP_KAIROS_REPORT=results/kairos_alpha.json] sp-daemon
    #[cfg(feature = "kairos")]
    if std::env::var("SP_NIGHTSHIFT_OFFLINE").as_deref() == Ok("1") {
        let model = std::env::var("SP_KAIROS_MODEL").unwrap_or_default();
        let tok = std::env::var("SP_KAIROS_TOK").unwrap_or_default();
        let live = std::env::var("SP_NIGHTSHIFT_LIVE").unwrap_or_else(|_| "_nightshift_live".to_string());
        if model.is_empty() || tok.is_empty() {
            eprintln!("SP_NIGHTSHIFT_OFFLINE=1 requires SP_KAIROS_MODEL, SP_KAIROS_TOK");
            std::process::exit(2);
        }
        match nightshift_curator::run_kairos_curator(&model, &tok, &live) {
            Ok((a, r)) => { eprintln!("[curator] accepted={a} rejected={r}"); std::process::exit(0); }
            Err(e) => { eprintln!("[curator] FAILED: {e}"); std::process::exit(1); }
        }
    }
    #[cfg(feature = "kairos")]
    if std::env::var("SP_KAIROS_ALPHA").as_deref() == Ok("1") {
        let model = std::env::var("SP_KAIROS_MODEL").unwrap_or_default();
        let tok = std::env::var("SP_KAIROS_TOK").unwrap_or_default();
        let tape = std::env::var("SP_KAIROS_TAPE").unwrap_or_default();
        if model.is_empty() || tok.is_empty() || tape.is_empty() {
            eprintln!("SP_KAIROS_ALPHA=1 requires SP_KAIROS_MODEL, SP_KAIROS_TOK, SP_KAIROS_TAPE");
            std::process::exit(2);
        }
        match kairos_runner::run_kairos_alpha(&model, &tok, &tape) {
            Ok((log, counters)) => {
                let json = kairos_runner::report_json(&log, &counters);
                println!("{json}");
                if let Ok(path) = std::env::var("SP_KAIROS_REPORT") {
                    if let Err(e) = std::fs::write(&path, &json) {
                        eprintln!("[kairos-alpha] report write {path} failed: {e}");
                    } else {
                        eprintln!("[kairos-alpha] report written: {path}");
                    }
                }
                eprintln!(
                    "[kairos-alpha] DONE ticks={} noop_ok={} action_ok={} false_action={} missed={} malformed={}",
                    counters.ticks, counters.noop_correct, counters.action_correct,
                    counters.false_actions, counters.missed_events, counters.malformed
                );
                std::process::exit(0);
            }
            Err(e) => {
                eprintln!("[kairos-alpha] FAILED: {e}");
                std::process::exit(1);
            }
        }
    }

    // SP_EAGLE_ACCEPT — framework-faithful live MTP acceptance probe (one-shot, exits
    // before clap/daemon startup). Mirrors the curator/kairos one-shot gates. Needs the
    // CUDA backend (gemma4_kv_* + gemma4_draft_*). Usage:
    //   SP_EAGLE_ACCEPT=1 SP_MODEL_PATH=…b1.sp-model SP_TOKENIZER_PATH=…b1.sp-tokenizer \
    //   SP_DRAFT_GGUF=…<draft>-it-F16-MTP.gguf [SP_EAGLE_N=48] [SP_DRAFT_ASCALE=one] sp-daemon
    #[cfg(feature = "wire_cuda_backend")]
    if std::env::var("SP_EAGLE_ACCEPT").as_deref() == Ok("1") {
        let model = std::env::var("SP_MODEL_PATH").unwrap_or_default();
        let tok = std::env::var("SP_TOKENIZER_PATH").unwrap_or_default();
        let draft = std::env::var("SP_DRAFT_GGUF").unwrap_or_default();
        let n: usize = std::env::var("SP_EAGLE_N").ok().and_then(|s| s.parse().ok()).unwrap_or(48);
        if model.is_empty() || tok.is_empty() || draft.is_empty() {
            eprintln!("SP_EAGLE_ACCEPT=1 requires SP_MODEL_PATH, SP_TOKENIZER_PATH, SP_DRAFT_GGUF");
            std::process::exit(2);
        }
        match eagle_accept::run_eagle_accept(&model, &tok, &draft, n) {
            Ok((a, c)) => { eprintln!("[eagle] DONE accept={a}/{c}"); std::process::exit(0); }
            Err(e) => { eprintln!("[eagle] FAILED: {e}"); std::process::exit(1); }
        }
    }
    // SP_EAGLE_CAPTURE — flywheel data capture (greedy rollout over a corpus -> (feature, KV, label)).
    //   SP_EAGLE_CAPTURE=1 SP_MODEL_PATH=… SP_TOKENIZER_PATH=… SP_EAGLE_CORPUS=corpus.txt \
    //   SP_EAGLE_OUT=_eagle_data [SP_EAGLE_GEN=64] sp-daemon
    #[cfg(feature = "wire_cuda_backend")]
    if std::env::var("SP_EAGLE_CAPTURE").as_deref() == Ok("1") {
        let model = std::env::var("SP_MODEL_PATH").unwrap_or_default();
        let tok = std::env::var("SP_TOKENIZER_PATH").unwrap_or_default();
        let corpus = std::env::var("SP_EAGLE_CORPUS").unwrap_or_default();
        let out = std::env::var("SP_EAGLE_OUT").unwrap_or_else(|_| "_eagle_data".to_string());
        let gen: usize = std::env::var("SP_EAGLE_GEN").ok().and_then(|s| s.parse().ok()).unwrap_or(64);
        if model.is_empty() || tok.is_empty() || corpus.is_empty() {
            eprintln!("SP_EAGLE_CAPTURE=1 requires SP_MODEL_PATH, SP_TOKENIZER_PATH, SP_EAGLE_CORPUS");
            std::process::exit(2);
        }
        match eagle_accept::run_eagle_capture(&model, &tok, &corpus, &out, gen) {
            Ok(n) => { eprintln!("[capture] DONE {n} seqs"); std::process::exit(0); }
            Err(e) => { eprintln!("[capture] FAILED: {e}"); std::process::exit(1); }
        }
    }
    // SP_LI_CAPTURE — Latent Interceptor capture (KAIROS event tape -> frame-end feature + action label).
    //   SP_LI_CAPTURE=1 SP_MODEL_PATH=… SP_TOKENIZER_PATH=… SP_LI_TAPE=kairos_tape.txt SP_LI_OUT=_li_data
    #[cfg(feature = "wire_cuda_backend")]
    if std::env::var("SP_LI_CAPTURE").as_deref() == Ok("1") {
        let model = std::env::var("SP_MODEL_PATH").unwrap_or_default();
        let tok = std::env::var("SP_TOKENIZER_PATH").unwrap_or_default();
        let tape = std::env::var("SP_LI_TAPE").unwrap_or_default();
        let out = std::env::var("SP_LI_OUT").unwrap_or_else(|_| "_li_data".to_string());
        if model.is_empty() || tok.is_empty() || tape.is_empty() {
            eprintln!("SP_LI_CAPTURE=1 requires SP_MODEL_PATH, SP_TOKENIZER_PATH, SP_LI_TAPE");
            std::process::exit(2);
        }
        match eagle_accept::run_li_capture(&model, &tok, &tape, &out) {
            Ok(n) => { eprintln!("[li-capture] DONE {n} samples"); std::process::exit(0); }
            Err(e) => { eprintln!("[li-capture] FAILED: {e}"); std::process::exit(1); }
        }
    }
    // SP_LI_ORACLE — LIVE KAIROS action gate (the Latent Interceptor). Per tick: the dense reference model prefill -> tap
    // latent -> CPU probe -> NO_OP short-circuits the decode; action wakes the dense reference model.
    //   SP_LI_ORACLE=1 SP_MODEL_PATH=… SP_TOKENIZER_PATH=… SP_LI_TAPE=tape SP_LI_HEAD=_li_head.bin
    #[cfg(feature = "wire_cuda_backend")]
    if std::env::var("SP_LI_ORACLE").as_deref() == Ok("1") {
        let model = std::env::var("SP_MODEL_PATH").unwrap_or_default();
        let tok = std::env::var("SP_TOKENIZER_PATH").unwrap_or_default();
        let tape = std::env::var("SP_LI_TAPE").unwrap_or_default();
        let head = std::env::var("SP_LI_HEAD").unwrap_or_else(|_| "_li_head.bin".to_string());
        if model.is_empty() || tok.is_empty() || tape.is_empty() {
            eprintln!("SP_LI_ORACLE=1 requires SP_MODEL_PATH, SP_TOKENIZER_PATH, SP_LI_TAPE (+ SP_LI_HEAD)");
            std::process::exit(2);
        }
        match eagle_accept::run_li_oracle(&model, &tok, &tape, &head) {
            Ok(()) => { eprintln!("[li-oracle] DONE"); std::process::exit(0); }
            Err(e) => { eprintln!("[li-oracle] FAILED: {e}"); std::process::exit(1); }
        }
    }
    // SP_LI_HEARTBEAT — PERSISTENT-CONTRACT KAIROS heartbeat (true Latent Interceptor deploy).
    //   SP_LI_HEARTBEAT=1 SP_MODEL_PATH=… SP_TOKENIZER_PATH=… SP_LI_TAPE=tape SP_LI_HEAD=_li_head.bin
    #[cfg(feature = "wire_cuda_backend")]
    if std::env::var("SP_LI_HEARTBEAT").as_deref() == Ok("1") {
        let model = std::env::var("SP_MODEL_PATH").unwrap_or_default();
        let tok = std::env::var("SP_TOKENIZER_PATH").unwrap_or_default();
        let tape = std::env::var("SP_LI_TAPE").unwrap_or_default();
        let head = std::env::var("SP_LI_HEAD").unwrap_or_else(|_| "_li_head.bin".to_string());
        if model.is_empty() || tok.is_empty() || tape.is_empty() {
            eprintln!("SP_LI_HEARTBEAT=1 requires SP_MODEL_PATH, SP_TOKENIZER_PATH, SP_LI_TAPE (+ SP_LI_HEAD)");
            std::process::exit(2);
        }
        match eagle_accept::run_li_heartbeat(&model, &tok, &tape, &head) {
            Ok(()) => { eprintln!("[li-heartbeat] DONE"); std::process::exit(0); }
            Err(e) => { eprintln!("[li-heartbeat] FAILED: {e}"); std::process::exit(1); }
        }
    }
    // SP_MH_GATE — Memory Head gate: latent -> pooled_K -> curator R -> C2 sig -> MEM-OKF.
    //   SP_MH_GATE=1 SP_MODEL_PATH=… SP_TOKENIZER_PATH=… SP_LI_TAPE=tape SP_MH_HEAD=_mh_head.bin SP_DRAFT_GGUF=…
    #[cfg(feature = "wire_cuda_backend")]
    if std::env::var("SP_MH_GATE").as_deref() == Ok("1") {
        let model = std::env::var("SP_MODEL_PATH").unwrap_or_default();
        let tok = std::env::var("SP_TOKENIZER_PATH").unwrap_or_default();
        let tape = std::env::var("SP_LI_TAPE").unwrap_or_default();
        let head = std::env::var("SP_MH_HEAD").unwrap_or_else(|_| "_mh_head.bin".to_string());
        let draft = std::env::var("SP_DRAFT_GGUF").unwrap_or_default();
        if model.is_empty() || tok.is_empty() || tape.is_empty() || draft.is_empty() {
            eprintln!("SP_MH_GATE=1 requires SP_MODEL_PATH, SP_TOKENIZER_PATH, SP_LI_TAPE, SP_DRAFT_GGUF (+ SP_MH_HEAD)");
            std::process::exit(2);
        }
        match eagle_accept::run_mh_gate(&model, &tok, &tape, &head, &draft) {
            Ok(()) => { eprintln!("[mh-gate] DONE"); std::process::exit(0); }
            Err(e) => { eprintln!("[mh-gate] FAILED: {e}"); std::process::exit(1); }
        }
    }
    // SP_LI_RETURN — latent-injection return path demo (tool result -> KV ring, no prompt re-feed).
    #[cfg(feature = "wire_cuda_backend")]
    if std::env::var("SP_LI_RETURN").as_deref() == Ok("1") {
        let model = std::env::var("SP_MODEL_PATH").unwrap_or_default();
        let tok = std::env::var("SP_TOKENIZER_PATH").unwrap_or_default();
        if model.is_empty() || tok.is_empty() {
            eprintln!("SP_LI_RETURN=1 requires SP_MODEL_PATH, SP_TOKENIZER_PATH"); std::process::exit(2);
        }
        match eagle_accept::run_li_return(&model, &tok) {
            Ok(()) => { eprintln!("[li-return] DONE"); std::process::exit(0); }
            Err(e) => { eprintln!("[li-return] FAILED: {e}"); std::process::exit(1); }
        }
    }
    // SP_TH_LOOP — closed loop: latent -> tool head -> fire tool -> return-path inject -> continue.
    #[cfg(feature = "wire_cuda_backend")]
    if std::env::var("SP_TH_LOOP").as_deref() == Ok("1") {
        let model = std::env::var("SP_MODEL_PATH").unwrap_or_default();
        let tok = std::env::var("SP_TOKENIZER_PATH").unwrap_or_default();
        let head = std::env::var("SP_TH_HEAD").unwrap_or_else(|_| "_tool_head.bin".to_string());
        if model.is_empty() || tok.is_empty() {
            eprintln!("SP_TH_LOOP=1 requires SP_MODEL_PATH, SP_TOKENIZER_PATH (+ SP_TH_HEAD, SP_LI_LABELS)"); std::process::exit(2);
        }
        match eagle_accept::run_th_loop(&model, &tok, &head) {
            Ok(()) => { eprintln!("[th-loop] DONE"); std::process::exit(0); }
            Err(e) => { eprintln!("[th-loop] FAILED: {e}"); std::process::exit(1); }
        }
    }
    // SP_TELEPATHY — LatentBridge: load adapter, fail-closed license, in-engine transfer + parity vs
    // Python, routing primitive. Default-off (this branch only runs when SP_TELEPATHY=1 = null floor).
    if std::env::var("SP_TELEPATHY").as_deref() == Ok("1") {
        match telepathy::run_telepathy() {
            Ok(()) => { eprintln!("[telepathy] DONE"); std::process::exit(0); }
            Err(e) => { eprintln!("[telepathy] FAILED: {e}"); std::process::exit(1); }
        }
    }
    // SP_TELEPATHY_LIVE — the cemented two-stage LIVE delegate (TELE-12/13/14): stage 1 decide_route on
    // the latent, stage 2 run the qwen coder on the CLEAN-TEXT task via CPU L1 (never fuse). Default-off.
    if std::env::var("SP_TELEPATHY_LIVE").as_deref() == Ok("1") {
        match telepathy::run_telepathy_live() {
            Ok(()) => { eprintln!("[telepathy-live] DONE"); std::process::exit(0); }
            Err(e) => { eprintln!("[telepathy-live] FAILED: {e}"); std::process::exit(1); }
        }
    }
    // SP_TELEPATHY_NATIVE — TELE-14 standalone sovereign native cross-family delegate (no Python):
    // load the qwen coder + native qwen3_decode_cuda on the clean-text task. Default-off.
    #[cfg(feature = "wire_cuda_backend")]
    if std::env::var("SP_TELEPATHY_NATIVE").as_deref() == Ok("1") {
        let cm = std::env::var("SP_TELEPATHY_CODER_MODEL").unwrap_or_default();
        let ct = std::env::var("SP_TELEPATHY_CODER_TOK").unwrap_or_default();
        if cm.is_empty() || ct.is_empty() {
            eprintln!("SP_TELEPATHY_NATIVE=1 requires SP_TELEPATHY_CODER_MODEL + SP_TELEPATHY_CODER_TOK"); std::process::exit(2);
        }
        match eagle_accept::run_telepathy_native(&cm, &ct) {
            Ok(()) => { eprintln!("[telepathy-native] DONE"); std::process::exit(0); }
            Err(e) => { eprintln!("[telepathy-native] FAILED: {e}"); std::process::exit(1); }
        }
    }

    let cli = Cli::parse();

    if cli.daemon_inner {
        // Trick #8: arm a remote peer as THE ARM Ring-2 backend before serving.
        // SP_RING2_PEER=<host:port> -> the canonical decode's spilled K-residue
        // blocks travel the QUIC socket instead of the local store. Registered
        // from a plain thread: the client owns its own runtime (block_on would
        // panic inside this tokio context).
        // Trick #8 server switch: SP_RING2_SERVE=<host:port> turns this daemon
        // into a Ring-2 peer store — bind the QUIC listener and serve residue
        // blocks alongside everything else (we are inside the tokio runtime,
        // so the endpoint binds against the live reactor).
        if let Ok(serve_addr) = std::env::var("SP_RING2_SERVE") {
            if let Ok(sock) = serve_addr.parse::<std::net::SocketAddr>() {
                tokio::spawn(async move {
                    match sp_daemon::network::ring2_quic::bind_ring2_server(sock) {
                        Ok(ep) => {
                            tracing::info!("SP_INFO: Ring-2 peer store SERVING on {sock}");
                            let _ = sp_daemon::network::ring2_quic::run_ring2_server(ep).await;
                        }
                        Err(e) => tracing::warn!("SP_WARN: SP_RING2_SERVE bind failed: {e}"),
                    }
                });
            } else {
                tracing::warn!("SP_WARN: SP_RING2_SERVE unparsable: {serve_addr}");
            }
        }
        if let Ok(peer_addr) = std::env::var("SP_RING2_PEER") {
            if let Ok(sock) = peer_addr.parse::<std::net::SocketAddr>() {
                let _ = std::thread::spawn(move || {
                    match sp_daemon::network::ring2_quic::register_ring2_quic(sock) {
                        Ok(()) => tracing::info!("SP_INFO: SP_RING2_PEER registered: {sock}"),
                        Err(e) => tracing::warn!("SP_WARN: SP_RING2_PEER registration failed: {e}"),
                    }
                }).join();
            } else {
                tracing::warn!("SP_WARN: SP_RING2_PEER unparsable: {peer_addr}");
            }
        }
        daemon::run_inner(
            &cli.model, &cli.tokenizer,
            &cli.draft_model, &cli.draft_tokenizer,
            // Chat-integration: Memory model wiring.
            &cli.memo_model, &cli.memo_tokenizer,
            // ledger-autowire: PoUW ledger path (empty = disabled).
            &cli.pouw_ledger_path,
            cli.quic_port, cli.port, &cli.peer, &cli.peers,
        ).await;
        return;
    }

    match cli.command {
        Some(Cmd::Start { model, tokenizer, draft_model, draft_tokenizer, memo_model, memo_tokenizer, pouw_ledger_path, quic_port, port, peer, peers }) =>
            daemon::cmd_start(&model, &tokenizer, &draft_model, &draft_tokenizer, &memo_model, &memo_tokenizer, &pouw_ledger_path, quic_port, port, &peer, &peers),
        Some(Cmd::Stop) => daemon::cmd_stop(),
        Some(Cmd::Reload) => daemon::cmd_reload(),
        None => {
            eprintln!("Usage: sp-daemon <start|stop|reload>");
            std::process::exit(1);
        }
    }
}
