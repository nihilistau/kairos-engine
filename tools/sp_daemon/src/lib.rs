pub mod network;
// DESIGN-NO-EXACT-PROFILE §4b: the integer-ring NTT FFI is part of the `exact` surface (default-on).
// The standard-FP profile (--no-default-features) drops it in lockstep with the ring archives.
// NOTE: network/quic_shard.rs (the QUIC garner NTT mesh) still references this; completing the FP
// profile link requires cfg-gating those recombine sites too (pending — G-NOEXACT-BUILD).
#[cfg(feature = "exact")]
pub mod ntt_ffi;

// CONTRACT-CHAT-FULLSTACK B3 — AUTONOMOUS MEMORY RECALL. The C2 discrete
// bit-collision resolver (tools/curator/discrete_resolve.py) ported to Rust so
// /v1/chat can compute a turn's query signature on its own and Hamming-match the
// episode registry — the model "remembers" with no operator-specified `replay`.
// Pure host-safe Rust (splitmix64 ±1 projection + popcount); the registry + R
// matrix live in AppState, gated behind the per-request `auto_recall` flag
// (default off = byte-untouched null floor). The actual cache-K read is the
// wire_cuda-only gemma4_kv_read_global_k; this module is arithmetic only.
pub mod recall;

// §4-MeMo Sprint M.2 — Zero-copy dialogue loop (Grounding → Entity ID →
// Synthesis) + Spinor receipt envelope. Host-build-safe: the module itself
// compiles on host (struct + tests); the L1-driven `run_dialogue()` helper
// lives in the android binary (sp_memo_m2_dialogue_smoke.rs).
pub mod dialogue;

// Chat-integration: the daemon-callable MeMo dialogue runner lives in the
// binary crate (src/dialogue_runner.rs declared from main.rs) because it
// depends on session/tokenizer modules that are binary-crate-local. The
// host-safe primitives (SpinorReceipt, DialoguePool, argmax) stay in
// `dialogue` above.

#[cfg(target_os = "android")]
pub mod dsp_rpc;

// §4-NTT Sprint NTT.5b — Hexagon backend dispatch trampolines for math-core's
// sp_compute_ntt_dispatch_fn ABI. Routes the inner Bluestein NTT calls through
// FastRPC method 17 (forward, VTCM-aware HVX) and method 18 (HVX INTT). Held
// in AppState as Option<Arc<ComputeBackend>>; registered with the Memory L1
// session at daemon startup when SP_ENGINE_NTT_ATTN_HEX=1 is set.
#[cfg(target_os = "android")]
pub mod ntt_hex_dispatch;

// Sprint WIRE-HEX — full-forward backend dispatcher for sp_l1.h:§6. The
// 6-month-gap fix: routes sp_prefill_chunk through the engine's
// gemma3_forward_hexagon (cDSP V69 HVX layers + final norm) instead of
// math-core's reference forward. Active when SP_DAEMON_BACKEND=hex is set
// AND the daemon was built with SP_DAEMON_LINK_HEX=1 so the
// libsp_hex_daemon_backend.a static lib is linked.
#[cfg(all(target_os = "android", feature = "wire_hex_backend"))]
pub mod hex_forward_dispatch;

// Sprint WIRE-CUDA — full-forward backend dispatcher for sp_l1.h:§6,
// symmetric to hex_forward_dispatch. Routes sp_prefill_chunk through the
// engine's gemma3_forward_cuda / qwen3_forward_cuda (CUDA PTX backend)
// instead of math-core's reference forward. Active when
// SP_DAEMON_BACKEND=cuda is set AND the daemon was built with
// --features wire_cuda_backend so libsp_cuda_daemon_backend.lib (or .a
// on Linux) is linked. Host x86_64 (NVIDIA GPU); no target_os constraint
// beyond what CUDA itself requires.
#[cfg(feature = "wire_cuda_backend")]
pub mod cuda_forward_dispatch;

// Sprint WIRE-CUDA-DECODE-GEMMA4 — persistent-KV DECODE backend dispatch, the
// step-wise sibling of cuda_forward_dispatch (which is PREFILL-ONLY by the
// sp_l1.h §6 contract). Routes the daemon's token-by-token sp_decode_step
// through the engine's resident `gemma4_kv_*` cache (cuda_forward.cu) so a dense reference model
// OK_Q4B model can decode with the tied full-vocab head materialized (the
// prefill hook trips `g4 probe: FULL head needs the f32 embd`). Rides the SAME
// `wire_cuda_backend` feature (the CUDA lib it links already carries the
// gemma4_kv_* symbols). Design: tools/sp_daemon/WIRE-CUDA-DECODE-GEMMA4.md.
#[cfg(feature = "wire_cuda_backend")]
pub mod cuda_kvdecode_dispatch;

// Sprint WIRE-VULKAN — host-side analog of hex_forward_dispatch for the
// Vulkan compute backend. Routes sp_prefill_chunk through
// gemma3_forward_vulkan / qwen3_forward_vulkan (the engine's host SPIR-V
// dispatch path). Active when SP_DAEMON_BACKEND=vulkan is set AND the
// daemon was built with `--features wire_vulkan_backend` so the
// libsp_vulkan_daemon_backend.{a,lib} static lib + vulkan loader are
// linked. Host-only (Windows / Linux / macOS where vulkan-1.{dll,so}
// resolves); no target_os gate, just the feature flag.
#[cfg(feature = "wire_vulkan_backend")]
pub mod vulkan_forward_dispatch;

// §4-MeMo Sprint M.1 — re-export the L1 C ABI bindings from the lib crate so
// android binaries (e.g. sp_memo_m1_smoke) pick up the link dependency on
// the math-core static libs via the lib's own link graph. On host this is
// harmless; on android this is what propagates -lsp_session etc. through
// to per-binary link steps (cargo's `rustc-link-lib` from build.rs reaches
// binaries only via the lib crate's symbol graph).
//
// Sprint WIRE-CUDA: also exposed on host when `wire_cuda_backend` is
// enabled — `cuda_forward_dispatch::register_with_session` needs
// `sp_session_register_forward_backend` at host link time.
//
// Sprint WIRE-VULKAN: also un-gated for host when `wire_vulkan_backend` is
// on, so the host trampoline `vulkan_forward_dispatch::register_with_session`
// can call `sp_session_register_forward_backend` via the lib-crate L1
// bindings. The math-core static libs are linked by build.rs unconditionally
// on host (line 117-126), so re-exporting the bindings is type-safe.
// Trick #8 (ring2_quic): the ARM Ring-2 registry bindings are needed on every
// host build now, and build.rs links the math-core libs unconditionally on
// host — so the cfg gate is dropped (it predated the network tier).
#[allow(non_upper_case_globals, non_camel_case_types, non_snake_case, dead_code, clippy::all)]
pub mod ffi_l1 {
    include!(concat!(env!("OUT_DIR"), "/sp_bindings.rs"));
}

// Batched prefill for warm continuations (SP_KV_PREFILL_BATCH_CONT, default-off).
// Pure threshold + flag helpers; the CUDA round-trip lives in routes.rs.
// Design: full re-batch (not offset) — see batch_cont_prefill::full_rebatch_wins.
pub mod batch_cont_prefill;

// M.5 (routing): KSTE-routed sparse Memory activation primitive. Public from
// the lib crate so sp_memo_m5_routing_smoke binary can `use sp_daemon::memo_routing`.
// Host build also exposes this (build.rs wires KSTE encoder symbols into the lib's
// own link closure via the bindgen wrapper header).
pub mod memo_routing;

// M.4 (ledger): PoUW receipt ledger + mesh replay primitive. Host-buildable
// (file I/O + SpinorReceipt::as_bytes round-trip, no L1 ABI needed); the
// android smoke binary sp_memo_m4_ledger_smoke drives it on the S22U.
pub mod pouw_ledger;

// KAI-1 — the KAIROS heartbeat-null control plane (implements the ratified
// papers/CONTRACT-KAIROS-K0-K1.md §2.5/§2a/§2b handoff ABI). Behind the
// off-by-default `kairos` feature: null-floor = byte-identical binary when
// unset, exactly like the wire_* backends. Pure host-safe Rust — the §2.5
// type system (coordinates, never prose), the §2b deterministic event-tape
// reader, the per-tick receipt log, and the tokio heartbeat loop that mirrors
// mining.rs's yield-to-inference idiom. The model-decode decision seam
// (`decide_via_model`) is named inside; the stub decider proves the loop's
// nervous system only.
#[cfg(feature = "kairos")]
pub mod kairos;

#[cfg(feature = "swarm")]
pub mod swarm;
