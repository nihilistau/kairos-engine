//! Batched prefill for warm continuations (`SP_KV_PREFILL_BATCH_CONT`).
//!
//! Design choice: **full re-batch**, not offset batch. See the module docs on
//! [`full_rebatch_wins`] and the routes.rs call site. Pure host-safe helpers so
//! the threshold arithmetic and flag contract are gated without a GPU.

/// Env flag that arms warm-continuation batched prefill. Default-off: unset or
/// any value other than `"1"` leaves the prefill dispatch byte-identical to
/// pre-feature code.
pub const FLAG_ENV: &str = "SP_KV_PREFILL_BATCH_CONT";

/// Minimum suffix length before any batch attempt is worth the setup cost.
/// Matches the cold `SP_KV_PREFILL_BATCH` floor (`head.len() > 64`).
pub const MIN_SUFFIX: usize = 64;

/// Measured batch throughput (ms/tok) on this hardware (brief: 13 ms/tok on a
/// 9,714-token cold prefill). Used only for the win-threshold arithmetic.
pub const BATCH_MS_PER_TOK: usize = 13;

/// Measured per-token warm-suffix prefill (ms/tok). Brief range 76–89; use 80
/// as a mid-band estimate so the threshold is not optimistic.
pub const PERTOK_MS_PER_TOK: usize = 80;

/// Snapshot/restore round-trip overhead budget (ms): one prefix insurance snap
/// + one full snap + one restore, plus the set_oneshot/reset dance. Conservative.
pub const SNAP_OVERHEAD_MS: usize = 200;

/// True when the env value is exactly `"1"`. Unset / `"0"` / anything else is off.
/// Pure over a borrowed string so tests never race on process-wide env.
pub fn cont_flag_enabled(val: Option<&str>) -> bool {
    val == Some("1")
}

/// True when `SP_KV_PREFILL_BATCH_CONT=1`. Any other value (or unset) is off.
pub fn cont_flag_on() -> bool {
    cont_flag_enabled(std::env::var(FLAG_ENV).ok().as_deref())
}

/// #41b — env flag that upgrades the warm-continuation arm from full re-batch to
/// SUFFIX-ONLY batch (`gemma4_kv_prefill_batched_from`): batch the `n` new tokens at
/// absolute positions `[P, P+n)` against the live prefix, then the same
/// continuation-safe snapshot round-trip. Cost drops from `BATCH_MS * (P+n)` to
/// `BATCH_MS * n` — measured 96 s vs a predicted ~20 s at warm_suffix=1396 /
/// prefix=6518. Default-off; only meaningful when [`FLAG_ENV`] is also armed
/// (the suffix arm lives INSIDE the cont dispatch, same trigger threshold — widening
/// the trigger window below `full_rebatch_min_suffix` is a separate, separately
/// measured change once the snapshot overhead is read from live logs).
pub const SUFFIX_FLAG_ENV: &str = "SP_KV_PREFILL_BATCH_SUFFIX";

/// True when the env value is exactly `"1"`. Pure twin of [`cont_flag_enabled`].
pub fn suffix_flag_enabled(val: Option<&str>) -> bool {
    val == Some("1")
}

/// True when `SP_KV_PREFILL_BATCH_SUFFIX=1`. Any other value (or unset) is off.
pub fn suffix_flag_on() -> bool {
    suffix_flag_enabled(std::env::var(SUFFIX_FLAG_ENV).ok().as_deref())
}

/// Measured suffix-batch forward rate (ms per SUFFIX token), 2026-08-20 live
/// decomposition: 20.3 / 20.7 / 21.3 across suffixes 1477–4184 at prefix 6687.
/// Higher than the cold batch's 13 because every suffix token's attention pays
/// for the full committed prefix — exactly the work per-token prefill also pays,
/// which is why per-token runs 80–99 ms/tok at the same contexts.
pub const SUFFIX_MS_PER_TOK: usize = 22;

/// Measured snapshot->reset->restore overhead for the suffix arm: 256 / 280 /
/// 313 ms at totals 8.2k–10.9k. Budgeted 500 so the threshold stays honest if
/// the cache grows toward Pmax.
pub const SUFFIX_SNAP_MS: usize = 500;

/// Minimum suffix at which the SUFFIX batch beats per-token:
/// `SUFFIX_MS*n + SNAP < PERTOK*n  ⇔  n > SNAP/(PERTOK-SUFFIX_MS)` — with the
/// measured constants that is n > 8, so the binding floor is [`MIN_SUFFIX`]
/// (the same batch-alloc-overhead floor the cold arm uses). Kept as arithmetic
/// rather than a bare 64 so a future re-measurement changes one constant.
pub fn suffix_batch_min_suffix() -> usize {
    let denom = PERTOK_MS_PER_TOK.saturating_sub(SUFFIX_MS_PER_TOK);
    if denom == 0 {
        return usize::MAX;
    }
    (SUFFIX_SNAP_MS / denom).max(MIN_SUFFIX)
}

/// Whether the suffix arm is worth attempting. Warm turns only — the cold path
/// belongs to `SP_KV_PREFILL_BATCH`.
pub fn suffix_batch_wins(prefill_from: usize, suffix_len: usize) -> bool {
    prefill_from > 0 && suffix_len > suffix_batch_min_suffix()
}

/// Minimum suffix length at which a **full re-batch** of `prefill_from + n`
/// tokens is expected to beat per-token prefill of the `n`-token suffix alone.
///
/// Arithmetic (measured constants above):
/// ```text
///   cost_batch  ≈ BATCH_MS * (P + n) + SNAP_OVERHEAD_MS
///   cost_pertok ≈ PERTOK_MS * n
///   win when cost_batch < cost_pertok
///        ⇔  n > (BATCH_MS * P + SNAP_OVERHEAD_MS) / (PERTOK_MS - BATCH_MS)
/// ```
/// Equality prefers the proven per-token path (strict `>`).
///
/// Why full re-batch (not offset batch): `gemma4_kv_prefill_batched` is cold-only
/// (`dpos_host != 0` declines), RoPE uses batch-local positions `0..n`, and
/// `k_attn` only attends inside the n-token scratch — it cannot see a restored
/// prefix. Suffix-only batching would need kernel surgery. Full re-batch from a
/// reset cache as one-shot, then snapshot/reset/restore, reuses the already-safe
/// cold round-trip pattern and leaves a continuation-safe restored-prefix state.
pub fn full_rebatch_min_suffix(prefill_from: usize) -> usize {
    let denom = PERTOK_MS_PER_TOK.saturating_sub(BATCH_MS_PER_TOK);
    if denom == 0 {
        return usize::MAX;
    }
    let by_speed = (BATCH_MS_PER_TOK * prefill_from + SNAP_OVERHEAD_MS) / denom;
    // Floor at MIN_SUFFIX so tiny suffixes never attempt the round-trip.
    by_speed.max(MIN_SUFFIX)
}

/// Whether a full re-batch of the entire head is expected to win over per-token
/// suffix prefill. Callers still gate on `!single_entry` and `cont_flag_on()`.
pub fn full_rebatch_wins(prefill_from: usize, suffix_len: usize) -> bool {
    if prefill_from == 0 {
        // Cold path is `SP_KV_PREFILL_BATCH`, not this flag.
        return false;
    }
    suffix_len > full_rebatch_min_suffix(prefill_from)
}

/// Combined dispatch predicate for the warm-continuation batch arm.
/// Pure of CUDA state — the call site still owns fall-through on engine errors.
/// `flag_on` is the already-resolved env check (see [`cont_flag_on`]).
///
/// WIDENED 2026-08-20: with the suffix arm armed, the trigger is the suffix
/// threshold (floor 64 — the snap dance measured ~300 ms, effectively free),
/// not the full-rebatch break-even. The call site must then gate the
/// full-rebatch FALLBACK on [`full_rebatch_wins`] itself: below that
/// break-even a declined suffix batch falls straight to per-token, because a
/// full re-batch of prefix+suffix would cost more than the per-token suffix.
pub fn want_batch_cont_gated(
    flag_on: bool,
    suffix_on: bool,
    single_entry: bool,
    prefill_from: usize,
    suffix_len: usize,
) -> bool {
    if !flag_on || single_entry {
        return false;
    }
    if suffix_on {
        suffix_batch_wins(prefill_from, suffix_len)
    } else {
        full_rebatch_wins(prefill_from, suffix_len)
    }
}

/// Live dispatch predicate: reads both flags once.
pub fn want_batch_cont(single_entry: bool, prefill_from: usize, suffix_len: usize) -> bool {
    want_batch_cont_gated(cont_flag_on(), suffix_flag_on(), single_entry,
                          prefill_from, suffix_len)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flag_default_off() {
        assert!(!cont_flag_enabled(None));
        assert!(!cont_flag_enabled(Some("")));
        assert!(!cont_flag_enabled(Some("0")));
        assert!(!cont_flag_enabled(Some("true")));
        assert!(cont_flag_enabled(Some("1")));
    }

    #[test]
    fn suffix_flag_default_off() {
        assert!(!suffix_flag_enabled(None));
        assert!(!suffix_flag_enabled(Some("")));
        assert!(!suffix_flag_enabled(Some("0")));
        assert!(!suffix_flag_enabled(Some("true")));
        assert!(suffix_flag_enabled(Some("1")));
    }

    #[test]
    fn cold_never_wins_cont_arm() {
        // Even a huge "suffix" with prefill_from==0 is not this path.
        assert!(!full_rebatch_wins(0, 10_000));
        assert!(!want_batch_cont_gated(true, false, false, 0, 10_000));
    }

    #[test]
    fn small_suffix_declines() {
        // P=2500 → min_n ≈ (13*2500 + 200)/67 = 488; also floored at 64.
        assert!(!full_rebatch_wins(2500, 64));
        assert!(!full_rebatch_wins(2500, 400));
        assert!(!full_rebatch_wins(2500, 488)); // equality → prefer per-token
        assert!(full_rebatch_wins(2500, 489));
    }

    #[test]
    fn short_prefix_uses_min_suffix_floor() {
        // P=100 → by_speed = (1300+200)/67 = 22 → floored to 64.
        assert_eq!(full_rebatch_min_suffix(100), MIN_SUFFIX);
        assert!(!full_rebatch_wins(100, 64));
        assert!(full_rebatch_wins(100, 65));
    }

    #[test]
    fn single_entry_blocks() {
        assert!(!want_batch_cont_gated(true, false, true, 2500, 800));
        assert!(want_batch_cont_gated(true, false, false, 2500, 800));
        // Flag off is dead code regardless of geometry.
        assert!(!want_batch_cont_gated(false, false, false, 2500, 800));
        assert!(!want_batch_cont_gated(false, true, false, 2500, 800));
    }

    #[test]
    fn suffix_arm_widens_the_trigger() {
        // Measured 2026-08-20: snap-dance ~300 ms, suffix forward ~21 ms/tok →
        // the binding floor is MIN_SUFFIX, not the full-rebatch break-even.
        assert_eq!(suffix_batch_min_suffix(), MIN_SUFFIX);
        // A 200-token suffix at a 6.7k prefix: per-token pays ~16 s today.
        // Full-rebatch loses (would batch 6.9k tokens); the suffix arm fires.
        assert!(!full_rebatch_wins(6700, 200));
        assert!(suffix_batch_wins(6700, 200));
        assert!(!want_batch_cont_gated(true, false, false, 6700, 200)); // cont-only: still per-token
        assert!(want_batch_cont_gated(true, true, false, 6700, 200));   // suffix armed: fires
        // Below the alloc floor nobody batches.
        assert!(!suffix_batch_wins(6700, 64));
        // Cold turns are never this arm's business.
        assert!(!suffix_batch_wins(0, 5000));
        assert!(!want_batch_cont_gated(true, true, false, 0, 5000));
    }

    #[test]
    fn threshold_arithmetic_stated() {
        // Documented in the brief: ~4,300 tok full re-batch vs long suffix.
        // P=3000, n=1300 → min = (13*3000+200)/67 = 569; 1300 wins.
        assert!(full_rebatch_wins(3000, 1300));
        // P=3000, n=200 → loses (pay for 3200 batch vs 200 per-token).
        assert!(!full_rebatch_wins(3000, 200));
    }
}
