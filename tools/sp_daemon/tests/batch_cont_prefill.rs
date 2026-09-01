//! G-BATCH-CONT-PREFILL — host-safe gate for warm-continuation batched prefill.
//!
//! Why this is not a live CUDA state-contract test: the crate has no model-backed
//! integration harness that can open a `sp_g4_kv` without linking the full CUDA
//! backend + a GGUF on disk (same class of blocker as `test_quic_shard.rs` notes
//! for the probe linker). The pure threshold/flag contract is what we can prove
//! offline; the live state contract (restored-prefix length + subsequent decode)
//! is the manual script at `engine/tools/test_batch_cont.md`.
//!
//! Residual risk (honest): without a GPU model load we cannot assert byte-identical
//! logits against a pure per-token run. The code path ends in the same
//! snapshot/reset/restore machinery the cold BATCH-PREFILL arm already uses in
//! production, so position/length identity is the testable bar here.

use sp_daemon::batch_cont_prefill::{
    cont_flag_enabled, full_rebatch_min_suffix, full_rebatch_wins, suffix_batch_min_suffix,
    suffix_batch_wins, want_batch_cont_gated, MIN_SUFFIX, SNAP_OVERHEAD_MS,
};

// ROT NOTE (2026-08-29 audit): want_batch_cont_gated grew its `suffix_on` second
// argument when the SUFFIX-ONLY arm landed (2026-08-20) and this file was never
// updated — the crate's WHOLE test target stopped compiling, so every Rust test in
// the daemon was dark for nine days while the gates said the arms were proven.
// Calls updated to (flag_on, suffix_on, single_entry, prefill_from, suffix_len),
// and the new arm gets its own coverage below.

#[test]
fn flag_off_is_dead_code_predicate() {
    // Even with a suffix that would win the arithmetic, flag-off declines.
    assert!(!cont_flag_enabled(None));
    assert!(!cont_flag_enabled(Some("0")));
    assert!(!want_batch_cont_gated(false, false, false, 2500, 2000));
}

#[test]
fn flag_on_falls_through_when_threshold_misses() {
    // Broken precondition: suffix too short for the prefix length.
    let p = 3000usize;
    let n = 100usize; // well under min_suffix for P=3000
    assert!(n <= full_rebatch_min_suffix(p));
    assert!(!full_rebatch_wins(p, n));
    assert!(
        !want_batch_cont_gated(true, false, false, p, n),
        "threshold miss must leave want_batch_cont false so routes fall through to per-token"
    );
}

#[test]
fn flag_on_and_threshold_hit_arms_the_path() {
    assert!(want_batch_cont_gated(true, false, false, 2500, 800));
    // single_entry seam never takes the batch (matches cold want_batch).
    assert!(!want_batch_cont_gated(true, false, true, 2500, 800));
}

#[test]
fn min_suffix_never_below_floor() {
    assert!(full_rebatch_min_suffix(0) >= MIN_SUFFIX || full_rebatch_min_suffix(0) == usize::MAX);
    assert!(full_rebatch_min_suffix(10) >= MIN_SUFFIX);
    // Large prefix: floor is the speed break-even, not MIN_SUFFIX.
    let m = full_rebatch_min_suffix(5000);
    assert!(m > MIN_SUFFIX);
    // Sanity: overhead is in the formula (if overhead were 0, min would be lower).
    assert!(SNAP_OVERHEAD_MS > 0);
}

#[test]
fn cold_prefill_from_zero_never_takes_cont_arm() {
    // The cold arm is SP_KV_PREFILL_BATCH; CONT must not steal it.
    assert!(!full_rebatch_wins(0, 9999));
    assert!(!want_batch_cont_gated(true, false, false, 0, 9999));
}

#[test]
fn suffix_arm_has_its_own_floor_and_its_own_win() {
    // The 2026-08-20 SUFFIX-ONLY arm: a warm continuation well over the alloc
    // floor takes the suffix batch; one at/below the floor falls to per-token;
    // and the single_entry seam still never batches.
    let floor = suffix_batch_min_suffix();
    assert!(suffix_batch_wins(2500, floor + 1));
    assert!(!suffix_batch_wins(2500, floor));
    assert!(want_batch_cont_gated(true, true, false, 2500, floor + 1));
    assert!(!want_batch_cont_gated(true, true, false, 2500, floor));
    assert!(!want_batch_cont_gated(true, true, true, 2500, floor + 1));
    // ...and a cold start (prefill_from == 0) is never the suffix arm's business.
    assert!(!suffix_batch_wins(0, 9999));
}
