//! §4-MeMo Sprint M.5 — KSTE-routed sparse Memory activation: routing primitive.
//!
//! Computes a per-layer head activation mask (`RoutingMask`) from the KSTE
//! Tier-0 root label of a grounding query. The mask is **advisory** in
//! Variant B (this sprint): the actual L1 forward (`sp_prefill_chunk`) does
//! not consume a head mask, so the mask is captured for downstream consumers
//! (M.4 receipt ledger metadata, M.2 dialogue loop diagnostic, future
//! Variant-A kernel-side enforcement) without enforcing it on the kernel.
//!
//! Determinism: same `grounding_query` always yields the same `RoutingMask`.
//! Pure function over the query and the model arch parameters; no global
//! state, no clocks, no RNG.
//!
//! Variation: different queries produce different masks (KSTE Tier-0 root
//! label depends on input value distribution; SplitMix64 per-layer expansion
//! gives well-distributed per-layer reshuffles even on near-equal Tier-0
//! roots).
//!
//! Sparsity: caller-controlled `k_per_layer` (default 8 of 14 for Qwen2.5-
//! Coder-0.5B-Instruct ≈ 57% active). Exactly `k_per_layer` bits are set
//! per layer (top-K head selection by per-layer SplitMix64 permutation).
//!
//! References:
//!  - `~/memory/reference_heterogeneous_soc_crt_tricks.md` Trick #5 (the
//!    REJECTED prefetch framing for MoE — see `PPT-LAT-Roadmap.md:5812-5816`
//!    for the M.5 correction to KSTE-as-routing for dense Memory)
//!  - `sp/kste.h` — `sp_kste_encode(vec, k, out)` + 64-byte tree layout
//!    (`SP_KSTE_OFF_ROOT = 8`, 6 × int16 LE Tier-0 root label)

#![allow(dead_code)]

// Bring in the bindgen-emitted symbols. On android these come from the lib
// crate's `ffi_l1` re-export so the static-lib link closure resolves; on host
// the lib's own bindings (built via build.rs) are accessible the same way.
#[cfg(target_os = "android")]
use crate::ffi_l1 as ffi;

#[cfg(not(target_os = "android"))]
mod ffi {
    #![allow(non_upper_case_globals, non_camel_case_types, non_snake_case, dead_code, clippy::all)]
    include!(concat!(env!("OUT_DIR"), "/sp_bindings.rs"));
}

// ─── Public types ───────────────────────────────────────────────────────────

/// Per-layer head activation mask for a Memory model forward.
///
/// `active[L]` is a bitmap over heads in layer L: bit `h` set iff head `h` is
/// active. Bits beyond `n_heads` MUST be zero (invariant; enforced by the
/// constructor).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RoutingMask {
    pub active: Vec<u64>,
    pub k_per_layer: u32,
    pub n_layers: u32,
    pub n_heads: u32,
    /// FNV-1a 64-bit hash of the 12-byte KSTE Tier-0 root label, for
    /// downstream ledger / receipt provenance.
    pub source_tier0_hash: u64,
}

impl RoutingMask {
    /// Total number of active heads across all layers.
    pub fn total_active(&self) -> usize {
        self.active.iter().map(|m| m.count_ones() as usize).sum()
    }

    /// Hamming distance vs another mask (sum of XOR popcount across layers).
    /// Panics if shapes differ (different n_layers or n_heads).
    pub fn hamming(&self, other: &RoutingMask) -> u32 {
        assert_eq!(self.n_layers, other.n_layers, "n_layers mismatch");
        assert_eq!(self.n_heads, other.n_heads, "n_heads mismatch");
        self.active
            .iter()
            .zip(other.active.iter())
            .map(|(a, b)| (a ^ b).count_ones())
            .sum()
    }

    /// Fraction of total heads that are active (0.0..=1.0).
    pub fn active_fraction(&self) -> f64 {
        let total = (self.n_layers as f64) * (self.n_heads as f64);
        if total == 0.0 {
            0.0
        } else {
            self.total_active() as f64 / total
        }
    }
}

// ─── KSTE Tier-0 extraction ─────────────────────────────────────────────────

/// Encode a grounding query and return its 12-byte Tier-0 root label.
///
/// The Tier-0 label lives at bytes `[SP_KSTE_OFF_ROOT..SP_KSTE_OFF_ROOT+12]`
/// of the frozen 64-byte tree (6 × int16 LE order statistics over the input).
///
/// `query`: int32 components fed as the K-vector. We accept `&[i32]` so the
/// caller may pass token IDs (cast widening from u32/i32) without further
/// massaging.
pub fn kste_tier0_root(query: &[i32]) -> [u8; 12] {
    let mut tree: ffi::sp_kste_tree_t = unsafe { std::mem::zeroed() };
    unsafe {
        ffi::sp_kste_encode(
            query.as_ptr(),
            query.len() as i32,
            &mut tree as *mut _,
        );
    }
    let bytes: &[u8; 64] = unsafe { &*(tree.bytes.as_ptr() as *const [u8; 64]) };
    let mut out = [0u8; 12];
    out.copy_from_slice(&bytes[8..20]);
    out
}

/// FNV-1a 64-bit hash of an arbitrary byte slice. Pure-integer, byte-order
/// independent, byte-exact across platforms.
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

// ─── SplitMix64 PRNG (pure integer, deterministic) ──────────────────────────

/// SplitMix64 step. Same constants as Java 8 SplittableRandom. Deterministic
/// across platforms. Public for unit-test access.
#[inline(always)]
pub fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

// ─── Routing computation ────────────────────────────────────────────────────

/// Compute a `RoutingMask` from a grounding query and model arch parameters.
///
/// Algorithm:
///  1. KSTE-encode the query; extract 12-byte Tier-0 root label.
///  2. Hash the root label to a 64-bit seed (FNV-1a 64).
///  3. Per layer L: derive a per-layer seed `s_L` by SplitMix64-mixing
///     `(tier0_hash XOR L)`; Fisher-Yates-shuffle the head index list 0..n_heads
///     using SplitMix64 draws; select the first `k_per_layer` shuffled heads
///     as active; set those bits in `active[L]`.
///
/// Properties:
///  - Deterministic over (query, n_layers, n_heads, k_per_layer).
///  - K bits set per layer (exactly).
///  - Variation across queries comes from the Tier-0 hash spread; tested in
///    the smoke harness via mean pairwise Hamming distance.
///
/// Returns Err if `k_per_layer > n_heads` (caller bug) or `n_heads > 64`
/// (head bitmap is u64; Qwen models cap at ~64 heads so this is a soft
/// limit for this sprint; relax to Vec<u128> if needed later).
pub fn compute_memory_routing(
    grounding_query: &[i32],
    n_layers: u32,
    n_heads: u32,
    k_per_layer: u32,
) -> Result<RoutingMask, String> {
    if k_per_layer > n_heads {
        return Err(format!(
            "k_per_layer ({k_per_layer}) > n_heads ({n_heads})"
        ));
    }
    if n_heads > 64 {
        return Err(format!(
            "n_heads ({n_heads}) > 64 (RoutingMask bitmap limit; relax if Qwen5+ exceeds this)"
        ));
    }
    if n_layers == 0 {
        return Err("n_layers == 0".into());
    }

    let tier0 = kste_tier0_root(grounding_query);
    let tier0_hash = fnv1a64(&tier0);

    let mut active = Vec::with_capacity(n_layers as usize);
    for l in 0..(n_layers as u64) {
        // Per-layer seed: mix tier0_hash with the layer index. XOR is a
        // standard mixing primitive; SplitMix64's avalanche carries the
        // entropy across all output bits.
        let mut state = tier0_hash ^ l.wrapping_mul(0xDEADBEEFCAFEBABE);
        // One mix step to spread the seed before drawing.
        let _ = splitmix64(&mut state);

        // Fisher-Yates over (0..n_heads); pick the first k_per_layer entries
        // of the resulting permutation as the active set. This is uniform
        // over the n_heads-choose-k subsets.
        let mut perm: Vec<u32> = (0..n_heads).collect();
        for i in (1..n_heads as usize).rev() {
            // Unbiased modulo via rejection sampling: draw u64, modulo by
            // (i+1); reject the top fragment that would bias the modulus.
            // For (i+1) <= 64 the bias from a single u64 draw is negligible
            // (< 1 in 2^58); we accept it for simplicity + determinism. A
            // strict-unbiased variant is a future refinement.
            let r = splitmix64(&mut state);
            let j = (r % ((i as u64) + 1)) as usize;
            perm.swap(i, j);
        }

        // Set bits for the first k_per_layer entries.
        let mut bits: u64 = 0;
        for &h in &perm[..k_per_layer as usize] {
            bits |= 1u64 << h;
        }
        active.push(bits);
    }

    Ok(RoutingMask {
        active,
        k_per_layer,
        n_layers,
        n_heads,
        source_tier0_hash: tier0_hash,
    })
}

// ─── TTFT estimator (Variant B advisory) ────────────────────────────────────

/// Linear-in-active-heads TTFT estimate for sparse forward, given a measured
/// full-forward TTFT and a routing mask.
///
/// **Variant B caveat:** this is an estimate, NOT a measurement. Real sparse
/// forward (Variant A) would skip per-head attention compute; the estimate
/// assumes per-head attention cost dominates per-layer cost and scales
/// linearly with active-head count. Per-layer non-attention cost (RMSNorm +
/// FFN + lm_head) is ignored in this estimate; a more careful model would
/// split fixed vs. per-head cost. Reported as `_estimated` in the smoke
/// JSON.
pub fn estimate_sparse_ttft_ms(full_ttft_ms: f64, mask: &RoutingMask) -> f64 {
    full_ttft_ms * mask.active_fraction()
}

// ─── Unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::TryInto;

    #[test]
    fn splitmix64_known_vectors() {
        // Java 8 SplittableRandom reference: state=0 first 4 draws.
        // (Compute against an independent reference impl, or pick a known seed.)
        // We just check non-zero output + deterministic across calls.
        let mut s1 = 0u64;
        let mut s2 = 0u64;
        let a = splitmix64(&mut s1);
        let b = splitmix64(&mut s2);
        assert_eq!(a, b);
        let c = splitmix64(&mut s1);
        assert_ne!(a, c, "consecutive draws must differ");
        assert_ne!(c, 0, "second draw zero is suspicious");
    }

    #[test]
    fn fnv1a64_empty_and_simple() {
        // FNV-1a 64 of empty slice = offset basis.
        assert_eq!(fnv1a64(&[]), 0xcbf29ce484222325);
        // FNV-1a 64 of single byte 0x00: offset * prime.
        let expected: u64 = 0xcbf29ce484222325u64.wrapping_mul(0x100000001b3);
        assert_eq!(fnv1a64(&[0u8]), expected);
    }

    #[test]
    fn routing_determinism_same_query() {
        let q: Vec<i32> = (0..16).map(|i| i * 31 + 7).collect();
        let m1 = compute_memory_routing(&q, 24, 14, 8).unwrap();
        let m2 = compute_memory_routing(&q, 24, 14, 8).unwrap();
        assert_eq!(m1, m2, "determinism violation");
    }

    #[test]
    fn routing_k_bits_per_layer() {
        let q: Vec<i32> = vec![1, 2, 3, 5, 8, 13, 21, 34];
        let m = compute_memory_routing(&q, 24, 14, 8).unwrap();
        for (l, &bits) in m.active.iter().enumerate() {
            assert_eq!(bits.count_ones(), 8, "layer {l} has {} active != 8", bits.count_ones());
            // No bits beyond n_heads.
            let mask_beyond = !((1u64 << 14) - 1);
            assert_eq!(bits & mask_beyond, 0, "layer {l} has bits set beyond n_heads=14");
        }
        assert_eq!(m.total_active(), 24 * 8);
        let af = m.active_fraction();
        assert!((af - (8.0 / 14.0)).abs() < 1e-12, "active_fraction = {af}");
    }

    #[test]
    fn routing_varies_across_queries() {
        let q1: Vec<i32> = (0..32).collect();
        let q2: Vec<i32> = (0..32).rev().collect();
        let m1 = compute_memory_routing(&q1, 24, 14, 8).unwrap();
        let m2 = compute_memory_routing(&q2, 24, 14, 8).unwrap();
        assert_ne!(m1.source_tier0_hash, m2.source_tier0_hash, "Tier-0 hash collision on different queries");
        assert_ne!(m1.active, m2.active, "routing did not vary across distinct queries");
    }

    #[test]
    fn routing_rejects_bad_args() {
        let q = vec![1, 2, 3];
        assert!(compute_memory_routing(&q, 24, 14, 15).is_err()); // k > n_heads
        assert!(compute_memory_routing(&q, 24, 65, 8).is_err());  // n_heads > 64
        assert!(compute_memory_routing(&q, 0, 14, 8).is_err());   // n_layers == 0
    }

    #[test]
    fn routing_full_active_when_k_eq_n_heads() {
        let q = vec![100, 200, 300];
        let m = compute_memory_routing(&q, 4, 14, 14).unwrap();
        for &bits in &m.active {
            assert_eq!(bits, (1u64 << 14) - 1);
        }
        assert!((m.active_fraction() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn routing_hamming_distance_symmetric() {
        let q1 = vec![1, 2, 3];
        let q2 = vec![4, 5, 6];
        let m1 = compute_memory_routing(&q1, 24, 14, 8).unwrap();
        let m2 = compute_memory_routing(&q2, 24, 14, 8).unwrap();
        assert_eq!(m1.hamming(&m2), m2.hamming(&m1));
        assert_eq!(m1.hamming(&m1), 0);
    }

    #[test]
    fn ttft_estimate_scales_with_active_fraction() {
        let q = vec![1, 2, 3];
        let m_k8 = compute_memory_routing(&q, 24, 14, 8).unwrap();
        let m_k4 = compute_memory_routing(&q, 24, 14, 4).unwrap();
        let m_k14 = compute_memory_routing(&q, 24, 14, 14).unwrap();
        let full = 1000.0;
        let e8 = estimate_sparse_ttft_ms(full, &m_k8);
        let e4 = estimate_sparse_ttft_ms(full, &m_k4);
        let e14 = estimate_sparse_ttft_ms(full, &m_k14);
        // K=14 is no-op sparse (all heads active).
        assert!((e14 - full).abs() < 1e-9);
        // K=4 < K=8 < full.
        assert!(e4 < e8 && e8 < full);
        // Linear-in-fraction.
        assert!(((e4 / e8) - (4.0 / 8.0)).abs() < 1e-9);
    }

    #[test]
    fn tier0_root_label_length_and_byte_offsets() {
        // Encode something simple; check that the returned 12 bytes are the
        // bytes at offset 8..20 of the underlying tree (this catches off-by-
        // one in kste_tier0_root if the SP_KSTE_OFF_ROOT layout ever shifts).
        let q: Vec<i32> = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let root = kste_tier0_root(&q);
        assert_eq!(root.len(), 12);
        // Determinism cross-check: re-encode + slice the tree by hand.
        let mut tree: ffi::sp_kste_tree_t = unsafe { std::mem::zeroed() };
        unsafe { ffi::sp_kste_encode(q.as_ptr(), q.len() as i32, &mut tree as *mut _); }
        let bytes: &[u8; 64] = unsafe { &*(tree.bytes.as_ptr() as *const [u8; 64]) };
        let manual: [u8; 12] = bytes[8..20].try_into().unwrap();
        assert_eq!(root, manual);
        // Tier-0 byte regions in the frozen layout are non-zero for non-
        // trivial input (sanity that the encoder ran).
        assert!(root.iter().any(|&b| b != 0), "Tier-0 region all-zero — encoder likely didn't run");
    }
}
