//! KAI-1 — the KAIROS heartbeat-null control plane.
//!
//! This module is the Rust implementation of the ratified handoff ABI in
//! `papers/CONTRACT-KAIROS-K0-K1.md` §2.5 / §2a / §2b. It is the *kernel*
//! layer that schedules on top of the existing resident daemon — it does NOT
//! reinvent session management, the L1 ABI wrapper, the event/SSE loop, the
//! PoUW ledger, or the QUIC mesh; those already live in `session.rs`,
//! `sessions.rs`, `state.rs`, `pouw_ledger.rs`, `quic_shard.rs`.
//!
//! ## Null-floor (the bit-exact-when-off invariant, kernel edition)
//! The whole module is behind the off-by-default `kairos` cargo feature
//! (mirroring `wire_cpu_backend` / `wire_cuda_backend` etc.). When the feature
//! is unset the daemon binary is byte-identical to today — no symbol, no tick,
//! no scheduler thread. With the feature built, the loop still only runs when
//! `SP_KERNEL=1` is set at startup.
//!
//! ## Constitutional rule (from §2.5)
//! Cross-session state is an addressable lattice of minted COORDINATES (the
//! episode manifest + ring coords) plus a lexical filesystem (Nexus paths) —
//! NEVER a tokenized prose summary round-tripped through the model. Therefore
//! `TaskState` references an episode pointer, not text. Any field that smells
//! like "the agent's own transcript" is the harness regression this contract
//! forbids.
//!
//! ## Scope of THIS increment (KAI-1, honest)
//! This cut lands the loop's nervous system: the type system, the deterministic
//! event-tape reader (§2b), the per-tick receipt log, and the heartbeat tick
//! loop (mirroring `mining.rs`'s yield-to-inference background task). The
//! per-tick DECISION is a deterministic salience-threshold stub — it proves the
//! tape→decide→log→receipt mechanism is sound and the gate counters are
//! computable, and it is explicitly NOT the model. Wiring the decision to an
//! `sp_session` decode tick (via `crate::session::SpSession`) is the named next
//! seam (`decide_via_model`), where the contract's honest unknown — whether an
//! it-tuned model holds NO_OP discipline — actually gets measured. Per §3 this
//! stage "claims nothing about sensors, actuators, autonomy quality, or the
//! Exec."

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────────────────────
// §2.5 / §2a — the handoff ABI (coordinate pointers only, never prose)
// ─────────────────────────────────────────────────────────────────────────────

/// off[L] owner-resolved byte-law manifest + per-owner kvd. A POINTER into the
/// Ring-2 episode store — NOT text. Mirrors the C `sp_xbar_manifest`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct EpisodePtr {
    /// Manifest descriptor path (the serialized owner/off[L] byte law).
    pub manifest_path: String,
    /// Number of layers the manifest resolves (NL); cheap sanity field.
    pub n_layers: u32,
}

/// Ring-2 on-disk episode store path: `{ep.k, ep.v}`, post-RoPE K/V, f32-exact
/// ⇒ replay is bit-exact (G-C1L-0b 34/34). Opened read-only via
/// `sp_arm_ring2_stdio_open_ro` (truncation-guarded) at resume.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Ring2Path(pub String);

/// A human-auditable filesystem tier path (knowledge / rules / receipts). Read
/// for context ONLY — it is the filesystem, not the memory image.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct NexusPath(pub String);

/// 32-byte receipt hash (the Done/Failed audit anchor; same family as the
/// `pouw_ledger` 0xA5 Spinor receipts).
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReceiptHash(pub [u8; 32]);

/// Composable sub-workflow identity.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkflowId(pub String);

/// Scheduler priority class (§2a SessionHandoff.priority).
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum PriorityClass {
    Realtime,
    Interactive,
    Background,
    Batch,
}

/// The independent Goal verifier's exit condition — checked OUT-OF-CONTEXT
/// before a task may transition to `Done` (§2.5 / §2a). Deliberately small in
/// K1: the verifier itself is a later seam; here we fix only the shape.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum GoalCond {
    /// No exit condition (the heartbeat-null default — the loop just breathes).
    Always,
    /// Never satisfiable (a held/parked task).
    Never,
    /// A named predicate evaluated by the out-of-context verifier (opaque to
    /// the agent's own decode — that is the point).
    Predicate(String),
}

/// §2.5 ABI verbatim — the resume image is an episode pointer + ring coords +
/// filesystem pointers + scheduler bookkeeping. NO tokenized history.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SessionHandoff {
    /// off[L] owner-resolved byte law + per-owner kvd (NOT text).
    pub episode_manifest: EpisodePtr,
    /// {ep.k, ep.v} on disk, post-RoPE K/V, f32-exact -> bit-exact replay.
    pub episode_store: Ring2Path,
    /// (L, pos, owner) coordinates the curator promoted (Ring-3 consolidated set).
    pub ring_coords: Vec<(u32, u32, u32)>,
    /// Human-auditable knowledge/rules/receipts (filesystem tier).
    pub fs_pointer: Vec<NexusPath>,
    /// Scheduler priority class.
    pub priority: PriorityClass,
    /// Exit condition checked out-of-context before Done.
    pub goal: GoalCond,
}

/// The resumable unit of execution (§2a). `Running` journals a step cursor so
/// resume re-enters there, not from scratch; `Yielded` carries the §2.5 episode
/// pointer, never a summary.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum TaskState {
    Pending,
    Running { step_cursor: u64 },
    Yielded { resume: SessionHandoff },
    Blocked { on: GoalCond },
    Done { receipt: ReceiptHash },
    Failed { receipt: ReceiptHash },
}

/// The deterministic orchestration primitives (MiMo-Code API shape, rebuilt in
/// Rust). The point of the explicit combinators: a `Parallel` barrier won't
/// drop an arm and a `Pipeline` won't forget a branch — the failure modes of
/// hand-rolled `for`/`if` orchestration the §1a ingestion table called out.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum Workflow {
    Agent { task: TaskState },
    Parallel { arms: Vec<Workflow>, barrier: bool },
    Pipeline { stages: Vec<Workflow> },
    Sub { name: WorkflowId },
}

// ─────────────────────────────────────────────────────────────────────────────
// §2b — the deterministic event tape (KAI-1 fixture format)
// ─────────────────────────────────────────────────────────────────────────────

/// The gate oracle / per-tick decision space (NOOP vs ACTION).
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum Decision {
    Noop,
    Action,
}

impl Decision {
    fn parse(s: &str) -> Option<Decision> {
        match s.trim().to_ascii_uppercase().as_str() {
            "NOOP" => Some(Decision::Noop),
            "ACTION" => Some(Decision::Action),
            _ => None,
        }
    }
}

/// One scripted tape event. `salience` feeds the router-tier score; `expect` is
/// the gate oracle used to compute the false-action / missed-event counters.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TapeEvent {
    pub tick_idx: u64,
    pub kind: String,
    pub payload: String,
    pub salience: f32,
    pub expect: Decision,
}

/// A parsed event tape. One event per line; `#` and blank lines are comments.
/// Format: `tick_idx  kind  payload  salience  expect` (whitespace-separated;
/// `payload` may be quoted to allow spaces; `-` means empty payload).
#[derive(Clone, Debug, Default)]
pub struct EventTape {
    pub events: Vec<TapeEvent>,
}

impl EventTape {
    /// Parse a tape from its text contents. Deterministic; no I/O.
    pub fn parse(text: &str) -> Result<EventTape, String> {
        let mut events = Vec::new();
        for (lineno, raw) in text.lines().enumerate() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let toks = tokenize_line(line);
            if toks.len() != 5 {
                return Err(format!(
                    "tape line {}: expected 5 fields (tick kind payload salience expect), got {}: {:?}",
                    lineno + 1,
                    toks.len(),
                    toks
                ));
            }
            let tick_idx = toks[0]
                .parse::<u64>()
                .map_err(|e| format!("tape line {}: bad tick_idx: {e}", lineno + 1))?;
            let kind = toks[1].clone();
            let payload = if toks[2] == "-" { String::new() } else { toks[2].clone() };
            let salience = toks[3]
                .parse::<f32>()
                .map_err(|e| format!("tape line {}: bad salience: {e}", lineno + 1))?;
            let expect = Decision::parse(&toks[4])
                .ok_or_else(|| format!("tape line {}: bad expect (want NOOP|ACTION): {}", lineno + 1, toks[4]))?;
            events.push(TapeEvent { tick_idx, kind, payload, salience, expect });
        }
        Ok(EventTape { events })
    }

    /// Load a tape from a file path. The only I/O in the parser path.
    pub fn load(path: &str) -> Result<EventTape, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("tape read {path}: {e}"))?;
        EventTape::parse(&text)
    }
}

/// Split a line on whitespace, honoring a single pair of double quotes around
/// the payload field so `"build finished"` stays one token.
fn tokenize_line(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut in_quote = false;
    for ch in line.chars() {
        match ch {
            '"' => in_quote = !in_quote,
            c if c.is_whitespace() && !in_quote => {
                if !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                }
            }
            c => cur.push(c),
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-tick receipt + gate accounting
// ─────────────────────────────────────────────────────────────────────────────

/// One tick's audit receipt (the soak gate requires every tick logged: frame
/// hash, decision, latency, state size).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TickRecord {
    pub tick_idx: u64,
    /// Deterministic hash of the environment-delta frame consumed this tick.
    pub frame_hash: u64,
    pub decision: Decision,
    pub expect: Decision,
    /// Per-tick latency (the O(Δ) flatness witness).
    pub latency_us: u64,
    /// Persistent-session size after this tick (the idle-no-growth witness).
    pub state_size: u64,
}

/// G-KAIROS-1 discipline counters, diffed decision-vs-expect over the tape.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct GateCounters {
    pub ticks: u64,
    /// Emitted ACTION where the oracle said NOOP.
    pub false_actions: u64,
    /// Emitted NOOP where the oracle said ACTION.
    pub missed_events: u64,
}

impl GateCounters {
    pub fn observe(&mut self, decision: Decision, expect: Decision) {
        self.ticks += 1;
        match (decision, expect) {
            (Decision::Action, Decision::Noop) => self.false_actions += 1,
            (Decision::Noop, Decision::Action) => self.missed_events += 1,
            _ => {}
        }
    }
}

/// FNV-1a 64-bit — a small, dependency-free, deterministic frame hash so tick
/// receipts are reproducible across runs (no clocks, no RNG in the hash).
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Encode the environment-delta frame for a tick as a compact line, then hash
/// it. This is the `O(Δ)` payload that would be appended to the persistent
/// session — its size is bounded by the event, not by session age.
fn frame_line(ev: &TapeEvent) -> String {
    format!("t{} {} {} s{:.2}", ev.tick_idx, ev.kind, if ev.payload.is_empty() { "-" } else { &ev.payload }, ev.salience)
}

// ─────────────────────────────────────────────────────────────────────────────
// The decider (the seam between mechanism and model)
// ─────────────────────────────────────────────────────────────────────────────

/// Per-tick salience threshold for the DETERMINISTIC STUB decider. Above this
/// the stub emits ACTION, otherwise NOOP. This proves the loop's nervous
/// system; it is explicitly NOT the model.
const STUB_SALIENCE_THRESHOLD: f32 = 0.5;

/// The deterministic mechanism decider. Proves tape→decide→log→receipt is
/// sound and the gate counters compute. THE NAMED NEXT SEAM is
/// `decide_via_model`, which runs an `sp_session` decode tick and parses
/// `NOOP` vs an action line from the model — where the contract's honest
/// unknown (it-tuned NO_OP discipline) is actually measured.
fn decide_stub(ev: &TapeEvent) -> Decision {
    if ev.salience >= STUB_SALIENCE_THRESHOLD {
        Decision::Action
    } else {
        Decision::Noop
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The heartbeat loop (mirrors mining.rs's yield-to-inference background task)
// ─────────────────────────────────────────────────────────────────────────────

/// Run the KAI-1 heartbeat over a scripted tape. Returns the per-tick receipt
/// log and the gate counters. Deterministic given the tape (the only
/// non-determinism is `latency_us`, which is excluded from `frame_hash` and the
/// gate verdict — it is telemetry, not oracle).
///
/// `tick_interval` is the reference operating point (8–60 s in production;
/// tests pass `Duration::ZERO` to run the whole tape immediately). The loop
/// backs off whenever `inference_active` is set, exactly like `run_mining_loop`,
/// so a kernel tick never starves `/v1/chat`.
pub async fn run_kairos_heartbeat(
    tape: EventTape,
    inference_active: Arc<AtomicBool>,
    tick_interval: Duration,
    log: Arc<Mutex<Vec<TickRecord>>>,
) -> GateCounters {
    let mut counters = GateCounters::default();
    // Persistent-session size proxy: in K1 the stub appends the O(Δ) frame line
    // and prunes NOOP ticks on the curator period, so idle ticks do not grow
    // state. We model that here as: ACTION ticks add their frame bytes; NOOP
    // ticks are pruned (no net growth). The real backing store is sp_session +
    // the cold-evict curator pass — wired at the model-decode seam.
    let mut state_size: u64 = 0;

    for ev in &tape.events {
        // Yield to the runtime every tick (mining.rs idiom).
        tokio::task::yield_now().await;

        // Back off while inference is active so the kernel never starves chat.
        while inference_active.load(Ordering::Relaxed) {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        let t0 = Instant::now();

        // Collect the environment-delta frame and hash it (reproducible).
        let frame = frame_line(ev);
        let frame_hash = fnv1a64(frame.as_bytes());

        // Decide (mechanism stub; model seam named above).
        let decision = decide_stub(ev);

        // Idle hygiene: NOOP ticks are pruned (no growth); ACTION ticks append.
        if matches!(decision, Decision::Action) {
            state_size = state_size.saturating_add(frame.len() as u64);
        }

        counters.observe(decision, ev.expect);

        let rec = TickRecord {
            tick_idx: ev.tick_idx,
            frame_hash,
            decision,
            expect: ev.expect,
            latency_us: t0.elapsed().as_micros() as u64,
            state_size,
        };
        log.lock().unwrap().push(rec);

        if !tick_interval.is_zero() {
            tokio::time::sleep(tick_interval).await;
        }
    }

    counters
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests — the mechanism gate (deterministic, no I/O, no model)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const SMOKE_TAPE: &str = "\
# tick_idx   kind            payload                    salience   expect
0            IDLE            -                          0.00       NOOP
1            IDLE            -                          0.00       NOOP
2            EVENT.timer     \"build finished\"           0.80       ACTION
3            IDLE            -                          0.00       NOOP
4            EVENT.alert     \"disk 95%\"                 0.90       ACTION
5            IDLE            -                          0.10       NOOP
";

    #[test]
    fn tape_parses_and_counts() {
        let tape = EventTape::parse(SMOKE_TAPE).expect("parse");
        assert_eq!(tape.events.len(), 6);
        assert_eq!(tape.events[2].kind, "EVENT.timer");
        assert_eq!(tape.events[2].payload, "build finished");
        assert_eq!(tape.events[2].expect, Decision::Action);
        // N=2 salient events among M=6 ticks (N << M).
        let salient = tape.events.iter().filter(|e| e.expect == Decision::Action).count();
        assert_eq!(salient, 2);
    }

    #[test]
    fn stub_decider_holds_discipline_on_smoke_tape() {
        // The mechanism-level claim: on a tape whose salience matches expect,
        // the stub decider produces ZERO false-actions and ZERO missed-events.
        // (This validates the loop/counter wiring — NOT model autonomy.)
        let tape = EventTape::parse(SMOKE_TAPE).expect("parse");
        let log = Arc::new(Mutex::new(Vec::new()));
        let inference_active = Arc::new(AtomicBool::new(false));
        let rt = tokio::runtime::Builder::new_current_thread().enable_time().build().unwrap();
        let counters = rt.block_on(run_kairos_heartbeat(
            tape,
            inference_active,
            Duration::ZERO,
            log.clone(),
        ));
        assert_eq!(counters.ticks, 6);
        assert_eq!(counters.false_actions, 0, "stub must not spam actions on idle ticks");
        assert_eq!(counters.missed_events, 0, "stub must not miss salient events");
        // Every tick produced a receipt (soak-gate requirement).
        assert_eq!(log.lock().unwrap().len(), 6);
    }

    #[test]
    fn frame_hash_is_reproducible() {
        // Same frame -> same hash across runs (no clocks/RNG in the hash).
        let ev = TapeEvent {
            tick_idx: 2,
            kind: "EVENT.timer".into(),
            payload: "build finished".into(),
            salience: 0.80,
            expect: Decision::Action,
        };
        let a = fnv1a64(frame_line(&ev).as_bytes());
        let b = fnv1a64(frame_line(&ev).as_bytes());
        assert_eq!(a, b);
    }

    #[test]
    fn idle_ticks_do_not_grow_state() {
        // A long idle run must leave state_size flat (the O(Δ) / no-growth
        // witness): NOOP ticks are pruned.
        let mut tape = String::from("# idle-only\n");
        for i in 0..256u64 {
            tape.push_str(&format!("{i} IDLE - 0.00 NOOP\n"));
        }
        let tape = EventTape::parse(&tape).expect("parse");
        let log = Arc::new(Mutex::new(Vec::new()));
        let inference_active = Arc::new(AtomicBool::new(false));
        let rt = tokio::runtime::Builder::new_current_thread().enable_time().build().unwrap();
        let _ = rt.block_on(run_kairos_heartbeat(tape, inference_active, Duration::ZERO, log.clone()));
        let l = log.lock().unwrap();
        assert_eq!(l.len(), 256);
        assert_eq!(l.last().unwrap().state_size, 0, "idle ticks must not grow persistent state");
    }
}
