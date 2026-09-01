//! KAI-1 Alpha — `decide_via_model`: the inference-driven heartbeat decider.
//!
//! This is the BINARY-crate half of KAIROS (the lib half is `sp_daemon::kairos`,
//! which holds the §2.5 ABI types, the §2b tape reader, and the deterministic
//! STUB loop). It lives here, declared from `main.rs`, for the same reason
//! `dialogue_runner` does: it drives `crate::session::SpSession` +
//! `crate::tokenizer::SptbTokenizer`, which are binary-crate-local L1 wrappers.
//!
//! ## What this measures (G-KAIROS-1 Alpha — telemetry, honest)
//! The contract's named open question: an instruction-tuned model's RLHF prior
//! is to ANSWER. Can it hold `NO_OP` discipline as a background daemon? We do
//! NOT assume — we measure. This harness replaces the salience-threshold stub
//! with a real qwen3 CPU decode tick and counts, against the §2b tape's
//! `expect` oracle:
//!   - noop_correct   : tape said NOOP, model emitted NO_OP
//!   - action_correct : tape said ACTION, model emitted a parseable <ACTION>
//!   - false_action   : tape said NOOP, model emitted an action  (the spam mode)
//!   - missed_event   : tape said ACTION, model emitted NO_OP    (the deaf mode)
//!   - malformed      : model emitted neither parseable form     (the parse mode)
//!
//! ## Persistent session (the O(Δ) law)
//! ONE `SpSession` is created and the system contract is prefilled ONCE. Each
//! tick appends only the compact event frame (no transcript re-feed) and decodes
//! the decision; the decision tokens stay in the KV (the model sees its own
//! history — realistic kernel behavior; any drift is part of what we measure).
//! Curator pruning of NOOP ticks (the idle-hygiene step) is NOT wired in Alpha —
//! that is the next seam; Alpha measures raw discipline + per-tick latency/size.
//!
//! ## Honest Alpha simplifications (named, not hidden)
//!   - Plain-text strict instruction, NOT the qwen3 ChatML special-token
//!     template. If discipline is poor, chat-template wrapping is the FIRST
//!     named knob (the contract's "prompt-contract iteration").
//!   - Greedy argmax decode (deterministic), decision cap = 24 tokens.
//!   - No finetune. The flywheel (a small NO_OP-discipline finetune) is named
//!     in the contract, not invoked here.

#![cfg(feature = "kairos")]

use std::time::Instant;

use sp_daemon::dialogue::argmax;
use sp_daemon::kairos::{Decision, EventTape, TapeEvent};

use crate::session::{SpModel, SpSession};
use crate::tokenizer::SptbTokenizer;

/// Max tokens to decode per tick for the decision. A well-behaved reply is
/// `NO_OP` (1–3 tokens) or `<ACTION>verb</ACTION>` (a handful) — 24 is slack.
const MAX_DECISION_TOKENS: usize = 24;

/// gemma4 arch id (mirrors tokenizer.rs ARCH_GEMMA3). When the loaded model is
/// gemma, the template auto-routes to `<start_of_turn>` turns and the qwen
/// `<think>` bypass is dropped (gemma is not a reasoning hybrid).
const GEMMA3_ARCH_ID: u32 = 3;

/// The strict kernel contract prefilled once at session start. Plain text
/// (Alpha simplification — see module header).
const SYSTEM_CONTRACT: &str = "\
You are a background kernel daemon. Each tick you receive one environment event.
Most events require no action. Reply with EXACTLY one of:
NO_OP
<ACTION>short imperative</ACTION>
Reply NO_OP unless the event genuinely requires intervention. Do not explain.
";

/// Path A — the explicit salience-threshold action policy. Removes the "use
/// judgment" ambiguity that left the 0.6B deaf (it judged everything not worth
/// acting on). Gives the low-capacity model a mechanical rule it can follow.
/// Selected by SP_KAIROS_SALIENCE_POLICY=1.
const SYSTEM_CONTRACT_SALIENCE: &str = "\
You are a background kernel daemon. Each tick you receive one environment event
that carries a salience score between 0.0 and 1.0. Reply with EXACTLY one of:
NO_OP
<ACTION>short imperative</ACTION>
RULE: if salience >= 0.5 the event requires intervention, so reply with an
<ACTION> line. If salience < 0.5, reply NO_OP. Follow the rule exactly. Do not explain.
";

/// One tick's measured outcome.
#[derive(Debug, Clone)]
pub struct AlphaTick {
    pub tick_idx: u64,
    pub expect: Decision,
    pub decided: DecisionParse,
    pub raw: String,
    pub latency_ms: u64,
    pub session_pos: usize,
}

/// The parsed model decision (a superset of `Decision` with a malformed arm).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionParse {
    Noop,
    Action,
    Malformed,
}

/// Classify the model's decoded text. Tolerant but unambiguous: an explicit
/// `<ACTION>` tag wins; otherwise a bare NO_OP / NOOP token counts as Noop;
/// anything else is Malformed (the RLHF-answer-spiral failure mode).
fn parse_decision(text: &str) -> DecisionParse {
    let up = text.trim().to_ascii_uppercase();
    if up.contains("<ACTION>") {
        DecisionParse::Action
    } else if up.contains("NO_OP") || up.contains("NOOP") {
        DecisionParse::Noop
    } else {
        DecisionParse::Malformed
    }
}

/// Aggregate G-KAIROS-1 Alpha counters.
#[derive(Debug, Default, Clone, Copy)]
pub struct AlphaCounters {
    pub ticks: u64,
    pub noop_correct: u64,
    pub action_correct: u64,
    pub false_actions: u64,
    pub missed_events: u64,
    pub malformed: u64,
}

impl AlphaCounters {
    fn observe(&mut self, expect: Decision, decided: DecisionParse) {
        self.ticks += 1;
        match (expect, decided) {
            (Decision::Noop, DecisionParse::Noop) => self.noop_correct += 1,
            (Decision::Action, DecisionParse::Action) => self.action_correct += 1,
            (Decision::Noop, DecisionParse::Action) => self.false_actions += 1,
            (Decision::Action, DecisionParse::Noop) => self.missed_events += 1,
            (_, DecisionParse::Malformed) => self.malformed += 1,
        }
    }
}

/// The compact structured event body (NOT a prose summary of history). The
/// kernel sees only the new event delta — this is the O(Δ) payload.
fn event_body(ev: &TapeEvent) -> String {
    let payload = if ev.payload.is_empty() { "-" } else { ev.payload.as_str() };
    format!("EVENT kind={} salience={:.2} payload=\"{}\"", ev.kind, ev.salience, payload)
}

/// RAW per-tick frame (no template) — the unaligned baseline.
fn frame_prompt_raw(ev: &TapeEvent) -> String {
    format!("\n{}\nDECISION: ", event_body(ev))
}

/// CHATML per-tick frame — wrap the event as a `user` turn and prime the
/// `assistant` turn, on a persistent KV that already holds the system turn.
/// This forces the instruct model to evaluate the prompt as an INSTRUCTION
/// (its fine-tuned boundary), not a narrative continuation. `<|im_start|>` /
/// `<|im_end|>` are registered special tokens (tokenizer.rs:163-170), so they
/// encode to their single IDs; `<|im_end|>` is an EOS id, so decode self-stops.
/// GEMMA per-tick frame — gemma's native `<start_of_turn>` turn structure. No
/// `<think>` bypass (gemma is not a reasoning hybrid). `<end_of_turn>` is an
/// EOS id so the model turn self-stops. The persistent KV already holds the
/// contract turn + ack (see run_kairos_alpha), so this is just the event turn.
fn frame_prompt_gemma(ev: &TapeEvent) -> String {
    format!(
        "<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n",
        event_body(ev)
    )
}

fn frame_prompt_chatml(ev: &TapeEvent) -> String {
    // No-think suppression (SP_KAIROS_NOTHINK=1): pre-fill a CLOSED thinking
    // block immediately after the assistant header, hijacking the token
    // trajectory past qwen3's deliberation phase so it must emit the final
    // action directly. `<think>`/`</think>` are registered special tokens.
    let nothink = std::env::var("SP_KAIROS_NOTHINK").as_deref() == Ok("1");
    let assistant_prime = if nothink {
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    } else {
        "<|im_start|>assistant\n"
    };
    format!("<|im_start|>user\n{}<|im_end|>\n{}", event_body(ev), assistant_prime)
}

/// Decode the decision for one tick on a persistent session. Returns the raw
/// decoded text. Greedy argmax; stops on EOS or once a parseable terminator is
/// seen (`NO_OP` line or `</ACTION>`).
fn decode_decision(
    session: &mut SpSession,
    tok: &SptbTokenizer,
    logits: &mut [f32],
    frame_tokens: &[i32],
) -> Result<String, String> {
    if frame_tokens.is_empty() {
        return Ok(String::new());
    }
    session.prefill_chunk(frame_tokens, logits)?;
    let mut bytes: Vec<u8> = Vec::with_capacity(64);
    let mut next = argmax(logits);
    for _ in 0..MAX_DECISION_TOKENS {
        if !tok.eos_ids.is_empty() && tok.eos_ids.contains(&next) {
            break;
        }
        bytes.extend_from_slice(tok.decode_token(next));
        // Early stop once we have a complete, parseable decision.
        let sofar = String::from_utf8_lossy(&bytes);
        let up = sofar.to_ascii_uppercase();
        if up.contains("</ACTION>") || up.contains("NO_OP") || up.contains("NOOP") {
            break;
        }
        session.decode_step(next, logits)?;
        next = argmax(logits);
    }
    Ok(String::from_utf8_lossy(&bytes).trim().to_string())
}

/// Run the full Alpha telemetry pass. Returns (per-tick log, counters).
pub fn run_kairos_alpha(
    model_path: &str,
    tok_path: &str,
    tape_path: &str,
) -> Result<(Vec<AlphaTick>, AlphaCounters), String> {
    use std::sync::atomic::AtomicI32;
    use std::sync::Arc;

    let tape = EventTape::load(tape_path)?;
    eprintln!(
        "[kairos-alpha] tape={} events={} (N salient={}, M idle={})",
        tape_path,
        tape.events.len(),
        tape.events.iter().filter(|e| e.expect == Decision::Action).count(),
        tape.events.iter().filter(|e| e.expect == Decision::Noop).count(),
    );

    let model = SpModel::load(model_path, tok_path)?;
    let arch = model.arch_info()?;
    let vocab = arch.vocab_size as usize;
    eprintln!(
        "[kairos-alpha] model loaded: vocab={} n_layers={} hidden={}",
        vocab, arch.n_layers, arch.hidden_dim
    );
    let tok = SptbTokenizer::build(&model, arch.arch_id, tok_path)?;
    let cancel = Arc::new(AtomicI32::new(0));
    let mut session = SpSession::create(&model, cancel)?;
    let mut logits = vec![0.0f32; vocab];

    // SP_KAIROS_CHATML=1 wraps the system contract + each event in the qwen3
    // ChatML instruction boundary (the fine-tuned <|im_start|>…<|im_end|>
    // frame). Default OFF = the raw un-templated baseline.
    let chatml = std::env::var("SP_KAIROS_CHATML").as_deref() == Ok("1");
    eprintln!("[kairos-alpha] prompt mode = {}", if chatml { "CHATML (templated)" } else { "RAW (un-templated)" });

    // SP_KAIROS_PRUNE=1 = the C1-lite cold-evict idle-hygiene step: if a tick
    // decides NO_OP, rewind the SpSession KV back to the pre-tick position
    // (sp_session_rewind, O(1), byte-identical per Corollary T8.1) — the event
    // and the model's NO_OP output are discarded from the cache. The next idle
    // tick then re-enters a context byte-identical to the FIRST clean
    // evaluation: the model never knows it has been waiting, so the unpruned
    // corruption attractor (deterministic "NO_克思主义" drift) cannot form, and
    // persistent-session size stays flat (O(Δ) holds across 10k idle ticks).
    let prune = std::env::var("SP_KAIROS_PRUNE").as_deref() == Ok("1");
    eprintln!("[kairos-alpha] idle-prune (cold-evict on NO_OP) = {}", if prune { "ON" } else { "OFF" });

    // Prefill the system contract ONCE (the persistent session law). In CHATML
    // mode it is wrapped as the `system` turn so every later `user` turn is
    // evaluated against it.
    let policy_salience = std::env::var("SP_KAIROS_SALIENCE_POLICY").as_deref() == Ok("1");
    let contract = if policy_salience { SYSTEM_CONTRACT_SALIENCE } else { SYSTEM_CONTRACT };
    eprintln!("[kairos-alpha] action policy = {}", if policy_salience { "SALIENCE>=0.5 (explicit threshold)" } else { "JUDGMENT (default)" });

    // Template auto-routes on the loaded model's arch. Gemma uses
    // <start_of_turn> turns (no system role → contract as a user turn + a model
    // ack; no <think> bypass). Qwen uses ChatML (<|im_start|> + optional
    // no-think). Raw = un-templated baseline.
    let use_gemma = arch.arch_id == GEMMA3_ARCH_ID;
    if chatml {
        eprintln!("[kairos-alpha] template = {}", if use_gemma { "GEMMA3 <start_of_turn>" } else { "QWEN ChatML" });
    }
    let sys_text = if chatml && use_gemma {
        // Gemma has no system role: seat the contract as a user turn + a short
        // model ack so the alternation is clean for the per-tick user turns.
        format!(
            "<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\nUnderstood.<end_of_turn>\n",
            contract.trim()
        )
    } else if chatml {
        format!("<|im_start|>system\n{}<|im_end|>\n", contract.trim())
    } else {
        contract.to_string()
    };
    let sys_tokens = tok.encode(&sys_text)?;
    if !sys_tokens.is_empty() {
        session.prefill_chunk(&sys_tokens, &mut logits)?;
    }
    eprintln!(
        "[kairos-alpha] system contract prefilled ({} tokens, {} mode); pos={}",
        sys_tokens.len(),
        if chatml { "chatml" } else { "raw" },
        session.position().unwrap_or(0)
    );

    let mut log: Vec<AlphaTick> = Vec::with_capacity(tape.events.len());
    let mut counters = AlphaCounters::default();

    for ev in &tape.events {
        let t0 = Instant::now();
        // Position BEFORE this tick touches the cache — the cold-evict anchor.
        let pre_pos = session.position().unwrap_or(0);
        let frame = if chatml && use_gemma {
            frame_prompt_gemma(ev)
        } else if chatml {
            frame_prompt_chatml(ev)
        } else {
            frame_prompt_raw(ev)
        };
        let frame_tokens = tok.encode(&frame)?;
        let raw = decode_decision(&mut session, &tok, &mut logits, &frame_tokens)?;
        let decided = parse_decision(&raw);
        counters.observe(ev.expect, decided);

        // C1-lite cold-evict: a NO_OP tick is discarded from the KV — rewind the
        // SpSession back to pre_pos so the next idle tick re-enters the clean
        // first-evaluation context (defeats the corruption attractor; keeps
        // state flat = O(Δ)). ACTION / Malformed ticks are KEPT (a real event
        // being handled, or a signal worth seeing).
        let mut pruned = false;
        if prune && matches!(decided, DecisionParse::Noop) {
            let cur = session.position().unwrap_or(pre_pos);
            if cur > pre_pos {
                session.rewind(cur - pre_pos)?;
                pruned = true;
            }
        }

        let rec = AlphaTick {
            tick_idx: ev.tick_idx,
            expect: ev.expect,
            decided,
            raw: raw.clone(),
            latency_ms: t0.elapsed().as_millis() as u64,
            session_pos: session.position().unwrap_or(0),
        };
        eprintln!(
            "[kairos-alpha] tick {:>3} expect={:?} decided={:?} pos={}{} {}ms raw={:?}",
            rec.tick_idx, rec.expect, rec.decided, rec.session_pos,
            if pruned { " (pruned->flat)" } else { "" }, rec.latency_ms,
            if rec.raw.len() > 48 { &rec.raw[..48] } else { &rec.raw }
        );
        log.push(rec);
    }

    Ok((log, counters))
}

/// JSON report (hand-rolled — no serde_json dep needed for a flat object).
pub fn report_json(log: &[AlphaTick], c: &AlphaCounters) -> String {
    let mut s = String::new();
    s.push_str("{\n");
    s.push_str(&format!("  \"ticks\": {},\n", c.ticks));
    s.push_str(&format!("  \"noop_correct\": {},\n", c.noop_correct));
    s.push_str(&format!("  \"action_correct\": {},\n", c.action_correct));
    s.push_str(&format!("  \"false_actions\": {},\n", c.false_actions));
    s.push_str(&format!("  \"missed_events\": {},\n", c.missed_events));
    s.push_str(&format!("  \"malformed\": {},\n", c.malformed));
    let denom = c.ticks.max(1) as f64;
    s.push_str(&format!(
        "  \"false_action_rate\": {:.4},\n",
        c.false_actions as f64 / denom
    ));
    s.push_str(&format!(
        "  \"missed_event_rate\": {:.4},\n",
        c.missed_events as f64 / denom
    ));
    s.push_str(&format!(
        "  \"malformed_rate\": {:.4},\n",
        c.malformed as f64 / denom
    ));
    s.push_str("  \"ticks_detail\": [\n");
    for (i, t) in log.iter().enumerate() {
        let comma = if i + 1 < log.len() { "," } else { "" };
        let raw_esc = t.raw.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', " ");
        s.push_str(&format!(
            "    {{\"tick\": {}, \"expect\": \"{:?}\", \"decided\": \"{:?}\", \"pos\": {}, \"latency_ms\": {}, \"raw\": \"{}\"}}{}\n",
            t.tick_idx, t.expect, t.decided, t.session_pos, t.latency_ms, raw_esc, comma
        ));
    }
    s.push_str("  ]\n}\n");
    s
}
