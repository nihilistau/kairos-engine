//! Discrete speculative decode loop for the Shannon-Prime Lattice (Phase 4-SPEC).
//!
//! Protocol: draft generates K tokens by greedy argmax. Target verifies each by
//! sequential argmax comparison. Accept/reject is binary integer equality — no
//! softmax, no probability ratios, no temperature. Corollary T8.1 guarantees
//! sp_session_rewind restores byte-identical KV state.
use crate::session::SpSession;

/// Result of one spec_step call.
pub struct SpecResult {
    /// Tokens accepted by the target (prefix of the K draft tokens).
    pub accepted: Vec<i32>,
    /// Target's logits AFTER the last accepted token (ready for next step's argmax).
    pub next_target_logits: Vec<f32>,
    /// Draft's logits ready for the next step.
    /// None on rejection — caller must decode the corrected token into draft to re-sync.
    pub next_draft_logits: Option<Vec<f32>>,
}

/// One speculative decode iteration.
///
/// - `target`: the per-chat cloned target session, at position P.
/// - `draft`:  the per-chat cloned draft session, at position P.
/// - `target_logits`: logits computed by the target at position P (current).
/// - `draft_logits`:  logits computed by the draft at position P (current).
/// - `k`: number of draft tokens to speculate.
/// - `vocab_size`: logits buffer width (allocated by caller).
///
/// After return: both sessions are at position P + accepted.len().
pub fn spec_step(
    target: &mut SpSession,
    draft: &mut SpSession,
    target_logits: &[f32],
    draft_logits: &[f32],
    k: usize,
    vocab_size: usize,
) -> Result<SpecResult, String> {
    // ── Phase 1: draft generates K tokens (K-1 decode_steps, draft ends at P+K-1) ──
    // d[i] = argmax of draft logits at position P+i.
    // After K-1 decode_steps, draft is at position P+(K-1).
    let mut draft_tokens: Vec<i32> = Vec::with_capacity(k);
    let mut d_logits = vec![0f32; vocab_size];

    draft_tokens.push(argmax(draft_logits));
    for _ in 1..k {
        let last = *draft_tokens.last().unwrap();
        draft.decode_step(last, &mut d_logits)?;
        draft_tokens.push(argmax(&d_logits));
    }
    // draft is now at P + K-1. draft_tokens = [d_0, ..., d_{K-1}].

    // ── Phase 2: target verifies sequentially ─────────────────────────────────────
    let mut t_logits = target_logits.to_vec();
    let mut accepted: Vec<i32> = Vec::with_capacity(k);

    for (k_idx, &draft_tok) in draft_tokens.iter().enumerate() {
        if argmax(&t_logits) == draft_tok {
            // Accept: advance target by one step.
            accepted.push(draft_tok);
            target.decode_step(draft_tok, &mut t_logits)?;
        } else {
            // Reject at position k_idx. Target is at P + k_idx (correct).
            // Draft is at P + K-1. Rewind draft by (K-1 - k_idx) to reach P + k_idx.
            let rewind_by = k - 1 - k_idx;
            if rewind_by > 0 {
                draft.rewind(rewind_by)?;
            }
            // Draft is at P+k_idx. Caller must decode the corrected token (argmax of
            // next_target_logits) into draft to re-sync before the next spec step.
            return Ok(SpecResult {
                accepted,
                next_target_logits: t_logits,
                next_draft_logits: None,
            });
        }
    }

    // All K accepted. Target is at P+K. Draft is at P+K-1 — advance one step.
    draft.decode_step(*draft_tokens.last().unwrap(), &mut d_logits)?;
    // Draft is now at P+K, aligned with target. d_logits = draft logits at P+K.

    Ok(SpecResult {
        accepted,
        next_target_logits: t_logits,
        next_draft_logits: Some(d_logits),
    })
}

pub fn argmax(logits: &[f32]) -> i32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as i32)
        .unwrap_or(0)
}
