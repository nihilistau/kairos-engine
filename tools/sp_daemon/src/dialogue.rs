//! §4-MeMo Sprint M.2 — Zero-copy dialogue loop (Grounding → Entity ID →
//! Synthesis) with per-turn Spinor receipt envelopes.
//!
//! Three-turn protocol:
//!   Turn 1 (Grounding): Executive consumes the user prompt; emits a
//!     "grounding query" (token stream that probes Memory).
//!   Turn 2 (Entity ID): Memory consumes Turn 1's output; emits the factual
//!     response. Input pointer is the SAME `Vec<i32>` slot Turn 1 wrote — no
//!     copy at the orchestrator layer (the L1 ABI itself takes the slice by
//!     raw pointer; pointer is reused across turns).
//!   Turn 3 (Synthesis): Executive consumes Turn 2's output; emits the final
//!     answer. Same orchestrator-side pointer reuse.
//!
//! Per `reference-zero-copy-invariant` and the M.2 PLAN's pragmatic
//! interpretation: the L1 `sp_prefill_chunk` / `sp_decode_step` API takes
//! caller-allocated buffers (`Vec<i32>` for tokens, `Vec<f32>` for logits).
//! Strict "no allocation in loop body" is achieved by pre-allocating ALL
//! per-turn buffers in [`DialoguePool::new`] and ONLY `.clear()`+`.push()`
//! inside the loop body (Vec capacity unchanged → no allocator activity).
//!
//! Per `reference-heterogeneous-soc-crt-tricks` Trick #9: the per-turn
//! receipt is a 64-byte cache-line-aligned audit envelope (63 bytes payload
//! + 0xA5 sentinel at offset 63). NOT a payload format; payload lives in
//! the token/logits buffers. The receipt is the INPUT to a future Sprint
//! M.4 PoUW receipt ledger.

#![allow(dead_code)] // some helpers exercised only by android binary

use sha2::{Digest, Sha256};

// ─── SpinorReceipt (Trick #9: 64-byte cache-line audit envelope) ───────────

/// Per-turn integrity envelope. Exactly 64 bytes — one ARM Cortex-X2 L1
/// D-cache line. Sentinel byte 0xA5 at offset 63 per the manifesto.
///
/// `#[repr(C, packed)]` so the byte layout is locked across compiler
/// revisions and target ABIs. Compile-time `size_of` assertion below
/// catches any accidental field/padding change.
#[repr(C, packed)]
#[derive(Clone, Copy, Debug)]
pub struct SpinorReceipt {
    /// 1, 2, or 3 — Grounding / Entity ID / Synthesis.
    pub turn_index: u8,
    /// 0xE = Executive, 0x4D = Memory.
    pub model_id: u8,
    /// Zero padding for 32-bit alignment of `wall_us`.
    pub _pad: [u8; 2],
    /// Per-turn wall-clock microseconds. u32::MAX ≈ 71 minutes — sufficient.
    pub wall_us: u32,
    /// BLAKE-style digest (SHA-256 truncated to 24 bytes) over input token
    /// buffer + turn metadata. See [`hash_buf`].
    pub input_hash: [u8; 24],
    /// SHA-256 truncated to 24 bytes over output token buffer + turn metadata.
    pub output_hash: [u8; 24],
    /// Count of input tokens (for cross-check with hashed length).
    pub n_input_tokens: u32,
    /// Count of output tokens (turn-cap < 256, so u8 fits).
    pub n_output_tokens: u8,
    /// Zero padding; reserved for future M.4 ledger metadata.
    pub _reserved: [u8; 2],
    /// 0xA5 — Trick #9 inter-island integrity sentinel.
    pub sentinel: u8,
}

// Compile-time guard: the 64-byte invariant is the load-bearing claim.
const _: [(); 64] = [(); core::mem::size_of::<SpinorReceipt>()];
const _: [(); 0]  = [(); core::mem::align_of::<SpinorReceipt>() - 1]; // align == 1 under packed

/// Model identity constants — also stored in the receipt at byte 1.
pub const MODEL_ID_EXECUTIVE: u8 = 0xE;
pub const MODEL_ID_MEMORY: u8 = 0x4D; // 'M' ASCII; visually 'M' in hexdump.

/// Manifesto Trick #9 sentinel.
pub const SPINOR_SENTINEL: u8 = 0xA5;

/// SHA-256 over (model_id || turn_index || token-bytes-as-LE), truncated
/// to 24 bytes. Domain-separates the hash so the same token stream hashed
/// for different (turn, model) doesn't collide.
fn hash_buf(turn_index: u8, model_id: u8, tokens: &[i32]) -> [u8; 24] {
    let mut hasher = Sha256::new();
    hasher.update([model_id, turn_index]);
    // SAFETY: bytewise view of i32 slice; little-endian device, packed inputs.
    let bytes: &[u8] = unsafe {
        core::slice::from_raw_parts(
            tokens.as_ptr() as *const u8,
            tokens.len() * core::mem::size_of::<i32>(),
        )
    };
    hasher.update(bytes);
    let full = hasher.finalize();
    let mut out = [0u8; 24];
    out.copy_from_slice(&full[..24]);
    out
}

impl SpinorReceipt {
    /// Mint a receipt for one completed dialogue turn. No allocation beyond
    /// the SHA-256 finalize scratch (small fixed stack frame).
    pub fn mint(
        turn_index: u8,
        model_id: u8,
        input_tokens: &[i32],
        output_tokens: &[i32],
        wall_us: u64,
    ) -> Self {
        let input_hash = hash_buf(turn_index, model_id, input_tokens);
        let output_hash = hash_buf(turn_index, model_id, output_tokens);
        SpinorReceipt {
            turn_index,
            model_id,
            _pad: [0, 0],
            wall_us: wall_us.min(u32::MAX as u64) as u32,
            input_hash,
            output_hash,
            n_input_tokens: input_tokens.len().min(u32::MAX as usize) as u32,
            n_output_tokens: output_tokens.len().min(u8::MAX as usize) as u8,
            _reserved: [0, 0],
            sentinel: SPINOR_SENTINEL,
        }
    }

    /// Reinterpret as a 64-byte byte array — for hex-dump / wire serialization.
    pub fn as_bytes(&self) -> [u8; 64] {
        let mut out = [0u8; 64];
        // SAFETY: SpinorReceipt is #[repr(C, packed)], size_of == 64
        // (compile-time-asserted above).
        unsafe {
            core::ptr::copy_nonoverlapping(
                self as *const SpinorReceipt as *const u8,
                out.as_mut_ptr(),
                64,
            );
        }
        out
    }

    /// True iff the sentinel byte is the Trick #9 0xA5 marker.
    pub fn sentinel_ok(&self) -> bool {
        self.sentinel == SPINOR_SENTINEL
    }

    /// True iff input_hash and output_hash are both non-zero
    /// (i.e. the receipt was actually computed over real buffers, not
    /// minted from empty slices).
    pub fn hashes_nonzero(&self) -> bool {
        self.input_hash.iter().any(|&b| b != 0)
            && self.output_hash.iter().any(|&b| b != 0)
    }

    // ─── mesh-canonical-order: u16 sequence rank in `_reserved[0..2]` ──────
    //
    // Sprint `mesh-canonical-order` (follow-on to M.4). The 2-byte
    // `_reserved` field carries a little-endian `u16` sequence rank used by
    // `Ledger::canonical_sort` + `Ledger::replay_canonical_into` to produce
    // a cross-device byte-identical ledger.
    //
    // Layout invariant: rank lives at on-wire offsets 61-62. Sentinel
    // (offset 63) and ALL other bytes (offsets 0-60) are unchanged. The
    // 64-byte `repr(C, packed)` size invariant is preserved exactly.
    //
    // Per `feedback-no-silent-gate-revisions`: the dispatch prompt
    // requested an additional `u32 device_id` at `_reserved[4..8]`. The
    // struct has only 2 bytes of `_reserved`; this constraint was surfaced
    // upstream in the plan-commit and Option A (rank-only on-wire;
    // tiebreak by input_hash at sort time) was adopted. Device_id remains
    // available to callers out-of-band — the canonical sort tiebreaks
    // determinstically on the existing 24-byte SHA-256-truncated
    // input_hash field, which is dense (≥192-bit entropy) and
    // domain-separated by (model_id, turn_index) per [`hash_buf`].

    /// Return a NEW receipt with `_reserved[0..2]` populated as the
    /// little-endian `u16` sequence rank. ALL other fields — turn_index,
    /// model_id, _pad, wall_us, input_hash, output_hash, n_input_tokens,
    /// n_output_tokens, sentinel — are preserved bit-exact.
    ///
    /// On-wire effect: bytes at offsets 61-62 carry the rank in
    /// little-endian; byte 63 keeps the 0xA5 sentinel.
    pub fn with_sequence_rank(mut self, rank: u16) -> SpinorReceipt {
        self._reserved = rank.to_le_bytes();
        self
    }

    /// Read the `u16` sequence rank from `_reserved[0..2]` (on-wire
    /// offsets 61-62). Returns 0 for a freshly minted receipt that has
    /// not been stamped via [`SpinorReceipt::with_sequence_rank`].
    pub fn sequence_rank(&self) -> u16 {
        // Field access through `#[repr(C, packed)]` requires a local copy
        // (taking a reference would be UB under packed layout).
        let r = self._reserved;
        u16::from_le_bytes(r)
    }
}

// ─── Argmax (greedy sampler) ───────────────────────────────────────────────

/// Index of the maximum-logit token. Greedy decode — matches the
/// `decode-determinism invariant` from `reference-lattice-decode-determinism`.
pub fn argmax(logits: &[f32]) -> i32 {
    let mut best_i: i32 = 0;
    let mut best_v: f32 = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i as i32;
        }
    }
    best_i
}

// ─── DialoguePool — pre-allocated buffers reused across turns ──────────────

/// Per-turn byte caps. All `Vec`s in [`DialoguePool`] are sized to these
/// limits at pool construction; the loop body NEVER reallocates.
#[derive(Clone, Copy, Debug)]
pub struct DialogueCaps {
    pub max_prompt_tokens: usize,
    pub max_query_tokens: usize,
    pub max_response_tokens: usize,
    pub max_answer_tokens: usize,
}

impl Default for DialogueCaps {
    fn default() -> Self {
        // Conservative defaults: a "What is X?"-shaped prompt fits in 32
        // tokens; the grounding query is typically short; Memory's response
        // and the synthesized answer are capped to keep dialogue turns
        // bounded for the smoke harness.
        DialogueCaps {
            max_prompt_tokens: 64,
            max_query_tokens: 64,
            max_response_tokens: 128,
            max_answer_tokens: 128,
        }
    }
}

/// Pre-allocated scratch space for one dialogue. Constructed ONCE per
/// dialogue session; reused across the three turns with `.clear()` (which
/// resets `len` to 0 but leaves `capacity` unchanged → zero allocator
/// activity in the loop body).
pub struct DialoguePool {
    /// Logits buffer for Executive (vocab_size_exec slots, f32).
    pub exec_logits: Vec<f32>,
    /// Logits buffer for Memory (vocab_size_memo slots, f32).
    pub memo_logits: Vec<f32>,
    /// Tokenized user prompt (input to Turn 1).
    pub prompt_tokens: Vec<i32>,
    /// Turn 1 output / Turn 2 input.
    pub grounding_query: Vec<i32>,
    /// Turn 2 output / Turn 3 input.
    pub memory_response: Vec<i32>,
    /// Turn 3 output (the final answer in token form).
    pub final_answer: Vec<i32>,
}

impl DialoguePool {
    pub fn new(vocab_exec: usize, vocab_memo: usize, caps: &DialogueCaps) -> Self {
        DialoguePool {
            exec_logits: vec![0f32; vocab_exec],
            memo_logits: vec![0f32; vocab_memo],
            prompt_tokens: Vec::with_capacity(caps.max_prompt_tokens),
            grounding_query: Vec::with_capacity(caps.max_query_tokens),
            memory_response: Vec::with_capacity(caps.max_response_tokens),
            final_answer: Vec::with_capacity(caps.max_answer_tokens),
        }
    }

    /// Reset all token buffers to zero length (capacity preserved → no
    /// allocator activity). Call at the start of each dialogue run.
    pub fn reset_tokens(&mut self) {
        self.prompt_tokens.clear();
        self.grounding_query.clear();
        self.memory_response.clear();
        self.final_answer.clear();
    }
}

// ─── DialogueOutcome ───────────────────────────────────────────────────────

#[derive(Debug)]
pub struct DialogueOutcome {
    /// Detokenized synthesis (UTF-8 lossy from the token decoder).
    pub final_answer: String,
    /// Three per-turn receipts.
    pub receipts: [SpinorReceipt; 3],
    /// End-to-end wall-clock microseconds for all three turns + receipts.
    pub total_wall_us: u64,
}

// ─── Tests (host build, no L1 required) ────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spinor_receipt_is_64_bytes() {
        // Belt-and-suspenders next to the compile-time const-eval guard.
        assert_eq!(core::mem::size_of::<SpinorReceipt>(), 64);
    }

    #[test]
    fn spinor_sentinel_at_offset_63() {
        let r = SpinorReceipt::mint(1, MODEL_ID_EXECUTIVE, &[1, 2, 3], &[4, 5], 12345);
        let bytes = r.as_bytes();
        assert_eq!(bytes[63], SPINOR_SENTINEL,
            "sentinel byte must be 0xA5 at offset 63 per Trick #9");
    }

    #[test]
    fn spinor_turn_index_at_offset_0() {
        let r = SpinorReceipt::mint(2, MODEL_ID_MEMORY, &[1, 2, 3], &[4, 5], 100);
        let bytes = r.as_bytes();
        assert_eq!(bytes[0], 2);
    }

    #[test]
    fn spinor_model_id_at_offset_1() {
        let r = SpinorReceipt::mint(1, MODEL_ID_EXECUTIVE, &[1], &[2], 0);
        let bytes = r.as_bytes();
        assert_eq!(bytes[1], MODEL_ID_EXECUTIVE);

        let r = SpinorReceipt::mint(2, MODEL_ID_MEMORY, &[1], &[2], 0);
        let bytes = r.as_bytes();
        assert_eq!(bytes[1], MODEL_ID_MEMORY);
    }

    #[test]
    fn spinor_hashes_are_nonzero_for_nonempty_inputs() {
        let r = SpinorReceipt::mint(1, MODEL_ID_EXECUTIVE, &[1, 2, 3], &[4, 5], 0);
        assert!(r.hashes_nonzero(), "non-empty token buffers must produce non-zero hashes");
        assert!(r.sentinel_ok());
    }

    #[test]
    fn spinor_hashes_domain_separated_by_turn() {
        let r1 = SpinorReceipt::mint(1, MODEL_ID_EXECUTIVE, &[1, 2, 3], &[], 0);
        let r2 = SpinorReceipt::mint(2, MODEL_ID_EXECUTIVE, &[1, 2, 3], &[], 0);
        // Same model + same input tokens, different turn — hashes must differ.
        assert_ne!(r1.input_hash, r2.input_hash,
            "input_hash must domain-separate by turn_index");
    }

    #[test]
    fn spinor_hashes_domain_separated_by_model() {
        let re = SpinorReceipt::mint(1, MODEL_ID_EXECUTIVE, &[1, 2, 3], &[], 0);
        let rm = SpinorReceipt::mint(1, MODEL_ID_MEMORY, &[1, 2, 3], &[], 0);
        assert_ne!(re.input_hash, rm.input_hash,
            "input_hash must domain-separate by model_id");
    }

    #[test]
    fn spinor_wall_us_at_offset_4() {
        let r = SpinorReceipt::mint(1, MODEL_ID_EXECUTIVE, &[1], &[2], 0x1234_5678);
        let bytes = r.as_bytes();
        let w = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        assert_eq!(w, 0x1234_5678);
    }

    #[test]
    fn spinor_wall_us_clamped_to_u32_max() {
        let r = SpinorReceipt::mint(1, MODEL_ID_EXECUTIVE, &[1], &[2], u64::MAX);
        // accessing through a packed struct field needs a copy
        let w = { r.wall_us };
        assert_eq!(w, u32::MAX);
    }

    #[test]
    fn argmax_picks_max() {
        let logits = vec![1.0, 5.0, 3.0, 2.0];
        assert_eq!(argmax(&logits), 1);
    }

    #[test]
    fn argmax_first_on_tie() {
        // strict-greater-than means first index wins on tie — matches
        // the routes.rs argmax semantics (max_by + Ordering::Equal -> a).
        let logits = vec![5.0, 5.0, 5.0];
        assert_eq!(argmax(&logits), 0);
    }

    // ─── mesh-canonical-order: sequence rank helpers ───────────────────

    #[test]
    fn with_sequence_rank_sets_reserved_at_offsets_61_62() {
        let r = SpinorReceipt::mint(1, MODEL_ID_EXECUTIVE, &[1, 2, 3], &[4, 5], 1234)
            .with_sequence_rank(42);
        let bytes = r.as_bytes();
        // 42 little-endian = 0x2A 0x00.
        assert_eq!(bytes[61], 0x2A, "rank low byte at offset 61");
        assert_eq!(bytes[62], 0x00, "rank high byte at offset 62");
    }

    #[test]
    fn with_sequence_rank_preserves_sentinel() {
        let r = SpinorReceipt::mint(2, MODEL_ID_MEMORY, &[1], &[2], 0)
            .with_sequence_rank(0xBEEF);
        let bytes = r.as_bytes();
        assert_eq!(bytes[63], SPINOR_SENTINEL,
            "sentinel 0xA5 at offset 63 must survive with_sequence_rank");
    }

    #[test]
    fn with_sequence_rank_preserves_all_other_bytes() {
        let base = SpinorReceipt::mint(3, MODEL_ID_EXECUTIVE, &[7, 11, 13], &[17, 19], 0xABCD);
        let stamped = base.with_sequence_rank(0x1234);
        let b0 = base.as_bytes();
        let b1 = stamped.as_bytes();
        // bytes 0..61 must be identical
        assert_eq!(&b0[0..61], &b1[0..61],
            "with_sequence_rank must only touch offsets 61-62");
        // byte 63 sentinel survives
        assert_eq!(b1[63], SPINOR_SENTINEL);
    }

    #[test]
    fn sequence_rank_round_trips() {
        for rank in [0u16, 1, 42, 255, 256, 32767, 32768, 65535] {
            let r = SpinorReceipt::mint(1, MODEL_ID_EXECUTIVE, &[1], &[2], 0)
                .with_sequence_rank(rank);
            assert_eq!(r.sequence_rank(), rank,
                "sequence_rank() must round-trip through with_sequence_rank({rank})");
        }
    }

    #[test]
    fn fresh_receipt_has_rank_zero() {
        let r = SpinorReceipt::mint(1, MODEL_ID_EXECUTIVE, &[1], &[2], 0);
        assert_eq!(r.sequence_rank(), 0,
            "freshly minted receipt must have rank=0 (unstamped sentinel)");
    }

    #[test]
    fn with_sequence_rank_preserves_64_byte_invariant() {
        let r = SpinorReceipt::mint(1, MODEL_ID_EXECUTIVE, &[1], &[2], 0)
            .with_sequence_rank(0xDEAD);
        assert_eq!(r.as_bytes().len(), 64,
            "SpinorReceipt remains 64 bytes post-stamping");
    }

    #[test]
    fn with_sequence_rank_le_byte_order() {
        // 0x1234 -> bytes [0x34, 0x12] (little-endian)
        let r = SpinorReceipt::mint(1, MODEL_ID_EXECUTIVE, &[1], &[2], 0)
            .with_sequence_rank(0x1234);
        let bytes = r.as_bytes();
        assert_eq!(bytes[61], 0x34, "low byte first (LE)");
        assert_eq!(bytes[62], 0x12, "high byte second (LE)");
    }

    #[test]
    fn dialogue_pool_caps_no_realloc() {
        let caps = DialogueCaps {
            max_prompt_tokens: 8,
            max_query_tokens: 4,
            max_response_tokens: 8,
            max_answer_tokens: 4,
        };
        let mut pool = DialoguePool::new(16, 16, &caps);
        let cap_before = (
            pool.prompt_tokens.capacity(),
            pool.grounding_query.capacity(),
            pool.memory_response.capacity(),
            pool.final_answer.capacity(),
        );
        // Fill to cap, clear, refill — capacity should be unchanged.
        for _ in 0..8 { pool.prompt_tokens.push(1); }
        for _ in 0..4 { pool.grounding_query.push(2); }
        for _ in 0..8 { pool.memory_response.push(3); }
        for _ in 0..4 { pool.final_answer.push(4); }
        pool.reset_tokens();
        for _ in 0..8 { pool.prompt_tokens.push(1); }
        let cap_after = (
            pool.prompt_tokens.capacity(),
            pool.grounding_query.capacity(),
            pool.memory_response.capacity(),
            pool.final_answer.capacity(),
        );
        assert_eq!(cap_before, cap_after,
            "DialoguePool token Vecs must not reallocate within their caps");
    }
}
