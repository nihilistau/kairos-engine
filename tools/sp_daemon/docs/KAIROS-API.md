# KAIROS-API — the kernel layer inside `sp-daemon`

> Status: **KAI-1 (heartbeat null) — mechanism green, model-decode telemetry in
> progress.** This document is the single reference for how KAIROS is built on
> top of the resident daemon. Theory and gates live in the lattice
> (`papers/ROADMAP-KAIROS.md`, `papers/CONTRACT-KAIROS-K0-K1.md`); this doc is
> the *implementation* map — what the code is, where it lives, how to build and
> run it, and what is deliberately NOT done yet.

---

## 0. The one-paragraph thesis

KAIROS is a **kernel, not a harness.** `sp-daemon` is already a long-lived,
resident process that wraps the frozen L1 C ABI (`sp_model` / `sp_session`) in a
tokio runtime, with a session registry, an SSE event loop, a PoUW receipt
ledger, a QUIC Ring-2 mesh, and a background mining task that *yields to
inference so it never starves `/v1/chat`*. KAIROS does not replace any of that —
it **extends** the daemon with a heartbeat scheduler that ticks on an event
stream, decides per tick whether to act, and (eventually) resumes work across
sessions by **replaying the episode** (`SP_REPLAY`), never by re-feeding a prose
transcript. The constitutional rule: **cross-session state is COORDINATES (an
episode manifest + ring coordinates) plus a lexical filesystem (Nexus), never a
tokenized summary round-tripped through the model.**

If you remember one thing: the C engine is the syscall layer; **`sp-daemon` is
the resident process; KAIROS is the scheduler that runs on it.**

---

## 1. Where the code lives (the lib/binary split)

KAIROS is split across the daemon's two crate halves, mirroring the existing
`dialogue` (lib) / `dialogue_runner` (binary) pattern. This split is **not
cosmetic** — `SpSession`/`SpModel`/`SptbTokenizer` are binary-crate-local L1
wrappers (declared from `main.rs`, they use `crate::ffi`), so anything that
drives real inference must live in the binary crate.

| File | Crate | Feature gate | Role |
|---|---|---|---|
| `src/kairos.rs` | **lib** (`sp_daemon::kairos`) | `kairos` | The §2.5/§2a ABI types, the §2b event-tape reader, per-tick receipts, and the deterministic **stub** heartbeat loop + its unit tests. Pure host-safe Rust — no FFI. |
| `src/kairos_runner.rs` | **binary** (`crate::kairos_runner`) | `kairos` | KAI-1 **Alpha** — `decide_via_model`: the real qwen3 CPU decode tick. Drives `SpSession`/`SptbTokenizer`. |
| `src/main.rs` (dispatch) | binary | `kairos` | `SP_KAIROS_ALPHA` env gate at the top of `main()`, before clap — runs the Alpha pass and exits without touching daemon startup. |
| `tests/fixtures/kairos/tape_smoke.txt` | — | — | The §2b deterministic event tape (N=3 salient among M=24 idle ticks). |
| `Cargo.toml` | — | — | The `kairos` feature (`kairos = []`), off by default, alongside the `wire_*` backends. |

### The null-floor invariant (bit-exact-when-off, kernel edition)

`kairos` is an **off-by-default cargo feature**, exactly like
`wire_cpu_backend` / `wire_cuda_backend`. When it is unset:

- `src/kairos.rs` and `src/kairos_runner.rs` are not compiled.
- The `SP_KAIROS_ALPHA` dispatch block in `main()` does not exist.
- The `sp-daemon` binary is **byte-identical** to a pre-KAIROS build.

With the feature built, the heartbeat still only runs when explicitly invoked
(`SP_KAIROS_ALPHA=1`, or — in the future daemon-resident form — `SP_KERNEL=1`).

---

## 2. The ABI types (`sp_daemon::kairos`)

Verbatim implementation of `CONTRACT-KAIROS-K0-K1.md` §2.5 / §2a. **State is
coordinates, never prose.**

```rust
// the resumable unit of execution (§2a)
enum TaskState {
    Pending,
    Running { step_cursor: u64 },        // journaled; resume re-enters here
    Yielded { resume: SessionHandoff },  // <eos> -> scheduler; episode pointer
    Blocked { on: GoalCond },            // unmet Goal-verifier exit condition
    Done    { receipt: ReceiptHash },
    Failed  { receipt: ReceiptHash },
}

// the §2.5 ABI verbatim — coordinate pointers only, no tokenized history
struct SessionHandoff {
    episode_manifest: EpisodePtr,        // off[L] owner-resolved byte law + kvd
    episode_store:    Ring2Path,         // {ep.k, ep.v}, post-RoPE K/V, f32-exact
    ring_coords:      Vec<(u32,u32,u32)>,// (L, pos, owner) curator-promoted set
    fs_pointer:       Vec<NexusPath>,    // human-auditable knowledge/rules/receipts
    priority:         PriorityClass,     // Realtime|Interactive|Background|Batch
    goal:             GoalCond,          // checked out-of-context before Done
}

// deterministic orchestration primitives (MiMo-Code API shape, rebuilt in Rust)
enum Workflow {
    Agent    { task: TaskState },
    Parallel { arms: Vec<Workflow>, barrier: bool }, // barrier won't drop an arm
    Pipeline { stages: Vec<Workflow> },               // won't forget a branch
    Sub      { name: WorkflowId },                    // composable; journaled
}
```

Supporting newtypes (`EpisodePtr`, `Ring2Path`, `NexusPath`, `ReceiptHash`,
`WorkflowId`, `PriorityClass`, `GoalCond`) are all `serde`-derived so a handoff
can be journaled to disk between steps — the crash-resume invariant
(resume from `step_cursor`, never re-hydrate).

### Why explicit `Workflow` combinators

The MiMo-Code lesson: hand-rolled `for`/`if` orchestration silently drops a
parallel arm or forgets a pipeline branch on the error path. `Parallel { barrier }`
and `Pipeline { stages }` make the barrier and the branch set part of the type, so
the journal can prove every arm ran.

---

## 3. The deterministic event tape (§2b)

`SP_REPLAY`-style determinism for the gate: no live sensors (that is KAI-4), just
a scripted file the tick reads one line at a time.

```
# tick_idx   kind            payload                salience   expect
0            IDLE            -                      0.00       NOOP
4            EVENT.timer     "build finished"       0.80       ACTION
12           EVENT.alert     "disk 95 percent"      0.90       ACTION
...
```

- `salience` feeds the router-tier score.
- `expect` is the **gate oracle** (NOOP vs ACTION) the false-action / missed-event
  counters diff against.
- N salient events are sparse among M idle ticks (N≪M); the smoke tape is N=3, M=24.

Parser: `EventTape::parse(&str)` / `EventTape::load(path)` → `Vec<TapeEvent>`.
Quoted payloads keep spaces; `#`/blank lines are comments; `-` is empty payload.

---

## 4. The heartbeat loop

Two implementations, deliberately:

### 4a. The mechanism stub (`kairos::run_kairos_heartbeat`, lib)

Mirrors `mining.rs`: a loop that `tokio::task::yield_now().await`s each tick and
backs off while `inference_active` is set (never starves `/v1/chat`). The
per-tick decision is a **deterministic salience threshold** (`>= 0.5 ⇒ ACTION`).
This proves the loop's nervous system — tape → decide → receipt → counters — is
sound, with **zero model cost**. It is explicitly NOT the model.

Receipts (`TickRecord`): `tick_idx`, `frame_hash` (FNV-1a, reproducible — no
clocks/RNG in the hash), `decision`, `expect`, `latency_us`, `state_size`.
`GateCounters` accumulate `false_actions` / `missed_events`.

Unit tests (run with `cargo test --features kairos kairos::`):
`tape_parses_and_counts`, `stub_decider_holds_discipline_on_smoke_tape`,
`frame_hash_is_reproducible`, `idle_ticks_do_not_grow_state`. **4/4 green.**

### 4b. The model decider (`kairos_runner::decide_via_model`, binary)

KAI-1 **Alpha** — replaces the stub with a real qwen3-0.6B CPU decode tick. ONE
persistent `SpSession`: the strict kernel contract is prefilled **once**, then each
tick appends only the compact event frame (the O(Δ) law — no transcript re-feed)
and decodes the decision via greedy `argmax` (cap 24 tokens, early-stop on a
parseable `NO_OP` / `</ACTION>`). The decision tokens stay in the KV — the model
sees its own history, which is realistic kernel behavior, and any drift is part of
what we measure.

Parser classifies each reply into `Noop` / `Action` / `Malformed`. The Alpha
counters (`AlphaCounters`):

| counter | meaning |
|---|---|
| `noop_correct`   | tape NOOP, model emitted NO_OP |
| `action_correct` | tape ACTION, model emitted parseable `<ACTION>` |
| `false_actions`  | tape NOOP, model acted (the **spam** mode) |
| `missed_events`  | tape ACTION, model said NO_OP (the **deaf** mode) |
| `malformed`      | neither parseable form (the **RLHF-answer-spiral** mode) |

---

## 5. Build & run

### Build (Windows MSVC, from `tools/sp_daemon`)

```bat
set SP_SYSTEM_BUILD_DIR=..\..\build-cpu\lib\shannon-prime-system
set SP_SYSTEM_INCLUDE=..\..\lib\shannon-prime-system\include
set LIBCLANG_PATH=C:\Program Files\LLVM\bin
cargo build --features kairos --bin sp-daemon
```

### Run the mechanism tests (no model needed)

```bat
cargo test --features kairos kairos::
```

### Run KAI-1 Alpha (G-KAIROS-1 telemetry, needs a qwen3 .sp-model)

```bat
set SP_KAIROS_ALPHA=1
set SP_KAIROS_MODEL=..\..\build\tests\qwen3_rt.sp-model
set SP_KAIROS_TOK=..\..\build\tests\qwen3_rt.sp-tokenizer
set SP_KAIROS_TAPE=tests\fixtures\kairos\tape_smoke.txt
set SP_KAIROS_REPORT=..\..\results\kairos_alpha_smoke.json
target\debug\sp-daemon.exe
```

The process loads the model, prefills the system contract once, runs the tape,
prints a JSON report to stdout (and to `SP_KAIROS_REPORT` if set), and exits 0 —
**without** ever starting the HTTP daemon. Every tick logs to stderr:
`expect=… decided=… pos=… Nms raw="…"`.

---

## 6. G-KAIROS-1 (the gate)

Pre-registered in `CONTRACT-KAIROS-K0-K1.md` §2. Bounds are pinned after the
first telemetry run, per the contract's "telemetry-then-pin" discipline.

- **Null floor:** `kairos` feature unset ⇒ daemon byte-identical to today.
- **Discipline:** against a scripted tape of N events in M idle ticks (N≪M),
  false-action rate and missed-event rate both under thresholds.
- **Arithmetic:** per-tick cost O(Δ) (tick latency flat vs session age); idle
  ticks do not grow persistent state.
- **Soak:** ≥24 h unattended, flat RSS, complete receipts (every tick logged).

**Pre-stated falsification:** if no sensitivity holds NO_OP discipline (action
spam at any usable threshold), the flat tick is dead and KAI-2's interrupt-only
architecture becomes the front door — and the negative ships in STATE. If
per-tick cost grows with session age despite the rings, the O(Δ) claim is
falsified and the recall path gets profiled before any further kernel work.

---

## 7. What is deliberately NOT done (named, not hidden)

The honest scope boundary, so nobody mistakes Alpha for the finished kernel:

1. **Curator pruning of NOOP ticks is not wired in Alpha.** The idle-hygiene
   cold-evict pass (the existing C1-lite "Dream" machinery) is the next seam;
   Alpha measures raw discipline + per-tick latency/size only.
2. **Plain-text prompt, not the qwen3 ChatML special-token template.** If
   discipline is poor, chat-template wrapping is the FIRST named knob (the
   contract's "prompt-contract iteration"), before any finetune.
3. **No NO_OP-discipline finetune.** The flywheel exists and is named; it is not
   invoked here. Whether an it-tuned model can hold silence is the open question
   Alpha measures, not assumes.
4. **`SessionHandoff` resume (`SP_REPLAY`) is typed but not yet driven by the
   loop.** The episode-replay resume path (C1L.0b, bit-exact, 34/34) is proven in
   the engine; wiring it as the `Yielded → resume` transition is KAI-1 Beta+.
5. **The daemon-resident `SP_KERNEL=1` background tick** (a real scheduler thread
   spawned at startup, OS-owned cadence) is not wired into `start`; Alpha runs as
   a one-shot env-gated pass. Promoting it to a resident tokio task that mirrors
   `mining.rs` is the obvious next step once discipline is measured.

---

## 8. Map to the lattice (theory & gates)

| Lattice doc | What it holds |
|---|---|
| `papers/ROADMAP-KAIROS.md` | The kernel-not-harness thesis; KAI-0..KAI-6 phases; adopt/adapt/reject vs MiMo-Code. |
| `papers/CONTRACT-KAIROS-K0-K1.md` | §1a MiMo ingestion table; §2.5 handoff ABI; §2/§2a/§2b KAI-1 spec + G-KAIROS-1 gate. **This doc implements §2a/§2b.** |
| `papers/CONTRACT-XBAR-C1-lite-curator.md` | The cold-evict + consolidation "Dream" machinery KAI-1's idle-hygiene step will reuse. |
| `papers/RFC-XBAR-auditable-latent-crossbar.md` | The episode store / Ring-2 / off[L] byte law that `SessionHandoff` points into. |

The XBAR campaign (`SP_REPLAY`, the episode store, the O(1) KV crossbar) is the
**memory floor** KAIROS schedules over. KAIROS is the time/agency axis above it.

## 9. KAI-1b — metal eviction (the cold-evict dropped to the crossbar, 2026-06-14)

§7 noted the daemon-resident tick and curator pruning as not-yet-wired. **KAI-1b closes the
*physical* eviction primitive** on the engine CUDA path (the production forward for the dense reference model):
the host-layer prefix-grow hack (don't append tokens on NO_OP, re-prefill each tick) is replaced
by an O(1) **memory-coordinate shear**. New persistent-KV ABI in `engine/src/backends/cuda/
cuda_forward.cu` (`gemma4_decode_cuda` left byte-untouched = null floor):

`gemma4_kv_open / prefill / decode / rewind(Δ) / pos / snapshot / close`

KAIROS tick (metal): `anchor=kv_pos`; `kv_prefill(frame)`; `kv_decode`; **NO_OP ⇒
`kv_rewind(kv_pos-anchor)`** (frame+gen sheared, cache resident, zero re-prefill); **ACTION ⇒
keep**. Gates (the dense reference model, RTX 2060): **G-1b-REWIND-NULL** — rewound KV byte-identical to
never-visited (48 owner layers, 16.5 MB, diffs=0) + EQUIV gen-reproduce. **O(actions)→O(1)
telemetry** — metal idle-tick latency flat (slope 0.0073 s/action) vs prefix-grow linear (0.924,
127× shallower); 16.7× faster @ 16 retained actions. Lattice CONTRACT-KAIROS-K0-K1 §5; receipts
`engine/results/kai1b_*.log`. Follow-on: wire the Rust daemon's KAIROS loop to this ABI (the
daemon currently runs the qwen3 path; dense metal loop lives in the engine harness for now).
