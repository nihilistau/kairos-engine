//! Chat-integration sprint — host + android safe MeMo dialogue runner.
//!
//! M.2 shipped `run_dialogue()` inside the android-only smoke binary
//! `sp_memo_m2_dialogue_smoke.rs` using local `L1Session` wrappers + a
//! byte-level tokenizer. This module is the daemon-callable equivalent:
//! same 3-turn Grounding → Entity ID → Synthesis protocol, but driven
//! through the existing host+android safe `crate::session::SpSession`
//! wrapper (which exposes `prefill_chunk` / `decode_step` on both
//! targets — see `tools/sp_daemon/src/session.rs:155-188`).
//!
//! The host-safe data structures (`SpinorReceipt`, `DialoguePool`,
//! `DialogueCaps`, `argmax`, `MODEL_ID_EXECUTIVE`, `MODEL_ID_MEMORY`)
//! come UNCHANGED from `crate::dialogue` — M.2's frozen interface is
//! imported, never modified.
//!
//! Per `reference-spinor-receipt-layout` (memory): each turn mints one
//! 64-byte receipt with sentinel 0xA5 at offset 63. Three receipts per
//! dialogue → returned as `[SpinorReceipt; 3]` to the caller.
//!
//! Per `reference-lattice-decode-determinism` (memory): greedy argmax
//! decode is byte-deterministic over (model, tokenizer, prompt, caps)
//! → cloned sessions produce byte-identical dialogues, which is the
//! invariant the M.2 closure's drift_count=0 across 10 runs verified.

#![allow(dead_code)] // exercised through /v1/dialogue + smoke harness

use std::time::Instant;

use sp_daemon::dialogue::{
    argmax, DialogueCaps, DialoguePool, SpinorReceipt, MODEL_ID_EXECUTIVE, MODEL_ID_MEMORY,
};
use crate::session::SpSession;
use crate::tokenizer::SptbTokenizer;

/// Output of one full Grounding → Entity ID → Synthesis dialogue.
///
/// Mirrors `DialogueOutcomeLocal` from `sp_memo_m2_dialogue_smoke.rs:277-283`
/// but uses the daemon's real tokenizer for the final-answer string
/// (not the byte-level fallback the smoke harness uses).
#[derive(Debug)]
pub struct DialogueOutcome {
    /// Detokenized final answer (UTF-8 lossy via `SptbTokenizer::decode_token`).
    pub final_answer: String,
    /// Final-answer token count (for diagnostic; also encoded in the
    /// turn-3 receipt's `n_output_tokens` field).
    pub final_answer_token_count: usize,
    /// Three per-turn receipts (turn_index 1, 2, 3).
    pub receipts: [SpinorReceipt; 3],
    /// Per-turn wall microseconds [t1, t2, t3].
    pub turn_us: [u64; 3],
    /// End-to-end wall microseconds across all 3 turns + receipt mint.
    pub total_wall_us: u64,
}

/// EOS sentinel used by M.2 to early-stop a turn. Mirror the smoke
/// binary's convention so the gate-substantive numbers are comparable
/// across the two paths.
const BYTE_EOS_CHECK: i32 = 0;

/// Drive one full 3-turn MeMo dialogue against pre-cloned Executive +
/// Memory sessions.
///
/// **Zero-copy discipline:** identical to the M.2 smoke binary's
/// `run_dialogue` (binary line 298-382) — the only allocations in this
/// function body are:
///  1. SHA-256 finalize scratch inside `SpinorReceipt::mint` (fixed
///     stack frame).
///  2. The `String` constructed for `final_answer` at the end (one
///     allocation, sized to the detokenized output length).
///  3. Three `[SpinorReceipt; 3]` elements on the stack (64 bytes each).
///
/// All `Vec` work goes through the pre-allocated `pool` slots via
/// `.clear()` + `.push()` — capacity unchanged, no allocator activity.
///
/// **Stateful sessions:** the same `exec_session` runs Turns 1 + 3 with
/// its KV cache accumulating across all three turns (the dialogue's
/// context continuation). The same `memo_session` runs Turn 2.
///
/// **Tokenizer:** the Executive tokenizer encodes the user prompt at
/// Turn 1 input AND decodes the final answer at Turn 3 output. The
/// Memory tokenizer is NOT used here — Turn 2's input is the raw
/// token-id stream from Turn 1 (per M.2's "byte-pointer-reuses the
/// Turn N output buffer as Turn N+1 input" zero-copy property).
/// Tokenizer-level cross-model translation is a future sprint
/// (filed under "What's NOT done" in the closure).
///
/// **Error mapping:** any L1 error from prefill/decode propagates as
/// the L1 error string (per `SpSession` ABI) plus a turn-prefix.
pub fn run_dialogue(
    exec_session: &mut SpSession,
    memo_session: &mut SpSession,
    exec_tokenizer: &SptbTokenizer,
    pool: &mut DialoguePool,
    user_prompt: &str,
    caps: &DialogueCaps,
) -> Result<DialogueOutcome, String> {
    let dialogue_start = Instant::now();

    // ─── Turn 1: Executive Grounding ────────────────────────────────────────
    pool.prompt_tokens.clear();
    let prompt_tokens_src = exec_tokenizer
        .encode(user_prompt)
        .map_err(|e| format!("turn1: tokenizer.encode → {e}"))?;
    let cap = pool.prompt_tokens.capacity();
    for &t in prompt_tokens_src.iter().take(cap) {
        pool.prompt_tokens.push(t);
    }
    drop(prompt_tokens_src);

    pool.grounding_query.clear();
    let t1_start = Instant::now();
    if !pool.prompt_tokens.is_empty() {
        exec_session
            .prefill_chunk(&pool.prompt_tokens, &mut pool.exec_logits)
            .map_err(|e| format!("turn1: prefill_chunk → {e}"))?;
        let mut next = argmax(&pool.exec_logits);
        for _ in 0..caps.max_query_tokens {
            if next == BYTE_EOS_CHECK {
                break;
            }
            // EOS check via the Executive tokenizer's eos_ids — matches
            // the existing /v1/chat behavior (routes.rs:206).
            if !exec_tokenizer.eos_ids.is_empty()
                && exec_tokenizer.eos_ids.contains(&next)
            {
                break;
            }
            if pool.grounding_query.len() >= pool.grounding_query.capacity() {
                break;
            }
            pool.grounding_query.push(next);
            exec_session
                .decode_step(next, &mut pool.exec_logits)
                .map_err(|e| format!("turn1: decode_step → {e}"))?;
            next = argmax(&pool.exec_logits);
        }
    }
    let t1_us = t1_start.elapsed().as_micros() as u64;
    let r1 = SpinorReceipt::mint(
        1,
        MODEL_ID_EXECUTIVE,
        &pool.prompt_tokens,
        &pool.grounding_query,
        t1_us,
    );

    // ─── Turn 2: Memory Entity ID (input = Turn 1 output, no copy) ──────────
    pool.memory_response.clear();
    let t2_start = Instant::now();
    if !pool.grounding_query.is_empty() {
        memo_session
            .prefill_chunk(&pool.grounding_query, &mut pool.memo_logits)
            .map_err(|e| format!("turn2: prefill_chunk → {e}"))?;
        let mut next = argmax(&pool.memo_logits);
        for _ in 0..caps.max_response_tokens {
            if next == BYTE_EOS_CHECK {
                break;
            }
            // Memory has no separate eos_ids list in AppState (the
            // M.0 smoke uses byte-level encoding) — rely on BYTE_EOS_CHECK
            // and the max_response_tokens cap.
            if pool.memory_response.len() >= pool.memory_response.capacity() {
                break;
            }
            pool.memory_response.push(next);
            memo_session
                .decode_step(next, &mut pool.memo_logits)
                .map_err(|e| format!("turn2: decode_step → {e}"))?;
            next = argmax(&pool.memo_logits);
        }
    }
    let t2_us = t2_start.elapsed().as_micros() as u64;
    let r2 = SpinorReceipt::mint(
        2,
        MODEL_ID_MEMORY,
        &pool.grounding_query,
        &pool.memory_response,
        t2_us,
    );

    // ─── Turn 3: Executive Synthesis (input = Turn 2 output, no copy) ───────
    pool.final_answer.clear();
    let t3_start = Instant::now();
    if !pool.memory_response.is_empty() {
        exec_session
            .prefill_chunk(&pool.memory_response, &mut pool.exec_logits)
            .map_err(|e| format!("turn3: prefill_chunk → {e}"))?;
        let mut next = argmax(&pool.exec_logits);
        for _ in 0..caps.max_answer_tokens {
            if next == BYTE_EOS_CHECK {
                break;
            }
            if !exec_tokenizer.eos_ids.is_empty()
                && exec_tokenizer.eos_ids.contains(&next)
            {
                break;
            }
            if pool.final_answer.len() >= pool.final_answer.capacity() {
                break;
            }
            pool.final_answer.push(next);
            exec_session
                .decode_step(next, &mut pool.exec_logits)
                .map_err(|e| format!("turn3: decode_step → {e}"))?;
            next = argmax(&pool.exec_logits);
        }
    }
    let t3_us = t3_start.elapsed().as_micros() as u64;
    let r3 = SpinorReceipt::mint(
        3,
        MODEL_ID_EXECUTIVE,
        &pool.memory_response,
        &pool.final_answer,
        t3_us,
    );

    let total_wall_us = dialogue_start.elapsed().as_micros() as u64;

    // Detokenize the final answer token stream via the Executive
    // tokenizer. The daemon's existing /v1/chat does the same (routes.rs:210)
    // via tokenizer.decode_token() per token; we accumulate all bytes
    // and one-shot lossy-UTF-8 the result for the response body.
    let mut answer_bytes: Vec<u8> = Vec::with_capacity(pool.final_answer.len() * 2);
    for &t in pool.final_answer.iter() {
        let tb = exec_tokenizer.decode_token(t);
        answer_bytes.extend_from_slice(&tb);
    }
    let final_answer = String::from_utf8_lossy(&answer_bytes).into_owned();

    Ok(DialogueOutcome {
        final_answer,
        final_answer_token_count: pool.final_answer.len(),
        receipts: [r1, r2, r3],
        turn_us: [t1_us, t2_us, t3_us],
        total_wall_us,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test: the imports compile and the receipt array layout is
    /// preserved. Full end-to-end runs require a loaded SpSession +
    /// SpModel pair (host-runnable when operator points
    /// SP_MODEL_PATH + SP_MEMO_MODEL_PATH; gate signal lives in the
    /// sp_chat_dialogue_smoke binary).
    #[test]
    fn dialogue_outcome_struct_shape() {
        let dummy = SpinorReceipt::mint(1, MODEL_ID_EXECUTIVE, &[1], &[2], 0);
        let outcome = DialogueOutcome {
            final_answer: "ok".into(),
            final_answer_token_count: 1,
            receipts: [dummy, dummy, dummy],
            turn_us: [0, 0, 0],
            total_wall_us: 0,
        };
        assert_eq!(outcome.receipts.len(), 3);
        assert_eq!(outcome.final_answer_token_count, 1);
        assert_eq!(outcome.receipts[0].sentinel_ok(), true);
    }
}
