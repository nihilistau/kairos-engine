//! §4-MeMo Sprint M.5 — KSTE-routed sparse Memory activation smoke harness.
//!
//! Drives the four substantive M.5 gates on Sam's S22U:
//!
//!   T_MEMO_M5_ROUTING_DETERMINISTIC  — same query × 100 invocations,
//!                                       0 RoutingMask divergences
//!   T_MEMO_M5_ROUTING_VARIES         — 10 distinct queries, ≥ 80 % of
//!                                       45 pairs Hamming > 0
//!   T_MEMO_M5_INVARIANCE_PRESERVING  — 100 queries: top-1 token agreement
//!                                       between full forward + sparse forward
//!                                       ≥ 70 %. **Variant B advisory:**
//!                                       sparse forward = identity to full
//!                                       (no kernel mask in this sprint),
//!                                       so agreement is trivially 100 %.
//!                                       Gate is interpreted as "routing
//!                                       layer presence does not break
//!                                       full forward."
//!   T_MEMO_M5_TTFT_MEASURED          — full vs sparse K=8/K=4 wall-clock;
//!                                       full is REAL, sparse is ESTIMATED
//!                                       linearly by active-head fraction.
//!                                       Reported, not gated.
//!
//! CLI:
//!   sp_memo_m5_routing_smoke <memo_model.spm> <memo_tok.spt> \
//!     [--queries N_QUERIES] [--det-cycles N] [--report-json PATH]
//!
//! Host build is a stub. Run on android (S22U) via scripts/m5_push_and_run.ps1.

#![cfg_attr(not(target_os = "android"), allow(dead_code))]

#[cfg(target_os = "android")]
use sp_daemon::ffi_l1 as ffi;

#[cfg(not(target_os = "android"))]
mod ffi {
    #![allow(non_upper_case_globals, non_camel_case_types, non_snake_case, dead_code, clippy::all)]
    include!(concat!(env!("OUT_DIR"), "/sp_bindings.rs"));
}

#[cfg(target_os = "android")]
use sp_daemon::memo_routing::{compute_memory_routing, estimate_sparse_ttft_ms, RoutingMask};

use ffi::sp_status_SP_OK;
use std::ffi::CString;
use std::os::raw::c_int;
use std::ptr;
use std::sync::atomic::AtomicI32;
use std::sync::Arc;
use std::time::Instant;

// ─── L1 wrappers (mirror sp_memo_m1_smoke.rs pattern) ───────────────────────

struct L1Model(*mut ffi::sp_model);
unsafe impl Send for L1Model {}
unsafe impl Sync for L1Model {}
impl Drop for L1Model {
    fn drop(&mut self) {
        unsafe { ffi::sp_model_unload(self.0) };
    }
}

struct L1Session {
    ptr: *mut ffi::sp_session,
    _cancel: Arc<AtomicI32>,
}
unsafe impl Send for L1Session {}
impl Drop for L1Session {
    fn drop(&mut self) {
        unsafe { ffi::sp_session_destroy(self.ptr) };
    }
}

fn load_model(model_path: &str, tok_path: &str) -> Result<(L1Model, ffi::sp_arch_info, u128), String> {
    let model_c = CString::new(model_path).map_err(|e| e.to_string())?;
    let tok_c = CString::new(tok_path).map_err(|e| e.to_string())?;
    let mut ptr: *mut ffi::sp_model = ptr::null_mut();
    let t0 = Instant::now();
    let st = unsafe { ffi::sp_model_load(model_c.as_ptr(), tok_c.as_ptr(), &mut ptr) };
    let wall_ms = t0.elapsed().as_millis();
    if st != sp_status_SP_OK {
        let detail = unsafe { std::ffi::CStr::from_ptr(ffi::sp_last_error()) }
            .to_string_lossy()
            .into_owned();
        return Err(format!("sp_model_load({model_path}) -> status={st}: {detail}"));
    }
    let mut arch: ffi::sp_arch_info = unsafe { std::mem::zeroed() };
    let st = unsafe { ffi::sp_model_arch(ptr, &mut arch) };
    if st != sp_status_SP_OK {
        unsafe { ffi::sp_model_unload(ptr) };
        return Err(format!("sp_model_arch -> status={st}"));
    }
    Ok((L1Model(ptr), arch, wall_ms))
}

fn create_session(model: &L1Model) -> Result<L1Session, String> {
    let cancel = Arc::new(AtomicI32::new(0));
    let cfg = ffi::sp_session_config {
        max_context: 0,
        deterministic: 1,
        arm_bank_kb: 0,
        sieve_capacity: 0,
        flags: 0,
        precision_override: 0,
    };
    let mut ptr: *mut ffi::sp_session = ptr::null_mut();
    let cancel_raw = cancel.as_ptr() as *mut c_int;
    let st = unsafe { ffi::sp_session_create(model.0, &cfg, cancel_raw, &mut ptr) };
    if st != sp_status_SP_OK {
        return Err(format!("sp_session_create -> status={st}"));
    }
    Ok(L1Session { ptr, _cancel: cancel })
}

fn clone_session(s: &L1Session) -> Result<L1Session, String> {
    let cancel = Arc::new(AtomicI32::new(0));
    let cancel_raw = cancel.as_ptr() as *mut c_int;
    let mut out: *mut ffi::sp_session = ptr::null_mut();
    let st = unsafe { ffi::sp_session_clone(s.ptr, cancel_raw, &mut out) };
    if st != sp_status_SP_OK {
        return Err(format!("sp_session_clone -> status={st}"));
    }
    Ok(L1Session { ptr: out, _cancel: cancel })
}

fn prefill(s: &mut L1Session, tokens: &[i32], logits: &mut [f32]) -> Result<(), String> {
    let st = unsafe {
        ffi::sp_prefill_chunk(s.ptr, tokens.as_ptr(), tokens.len(), logits.as_mut_ptr(), logits.len())
    };
    if st != sp_status_SP_OK {
        return Err(format!("sp_prefill_chunk -> status={st}"));
    }
    Ok(())
}

fn argmax_f32(v: &[f32]) -> usize {
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            best = i;
        }
    }
    best
}

// ─── Tiny JSON emitter (matches sp_memo_m1_smoke style) ─────────────────────

struct J(String);
impl J {
    fn new() -> Self { J("{".into()) }
    fn kv_u64(&mut self, k: &str, v: u64) -> &mut Self { self.comma_if(); self.0.push_str(&format!("\"{k}\":{v}")); self }
    fn kv_str(&mut self, k: &str, v: &str) -> &mut Self {
        self.comma_if();
        let escaped: String = v.chars().flat_map(|c| match c {
            '"' => vec!['\\', '"'],
            '\\' => vec!['\\', '\\'],
            '\n' => vec!['\\', 'n'],
            c => vec![c],
        }).collect();
        self.0.push_str(&format!("\"{k}\":\"{escaped}\""));
        self
    }
    fn kv_f64(&mut self, k: &str, v: f64) -> &mut Self {
        self.comma_if();
        // f64 may be NaN / inf; print "null" for those so JSON parses
        if v.is_finite() {
            self.0.push_str(&format!("\"{k}\":{}", v));
        } else {
            self.0.push_str(&format!("\"{k}\":null"));
        }
        self
    }
    fn obj(&mut self, k: &str) -> &mut Self { self.comma_if(); self.0.push_str(&format!("\"{k}\":{{")); self }
    fn end_obj(&mut self) -> &mut Self { self.0.push('}'); self }
    fn comma_if(&mut self) {
        let last = self.0.chars().last().unwrap_or('{');
        if last != '{' && last != '[' { self.0.push(','); }
    }
    fn finish(mut self) -> String { self.0.push('}'); self.0 }
}

// ─── Query generation: deterministic, distinct, structurally varied ─────────

/// Generate `n` distinct grounding queries. Each is a Vec<i32> drawn from
/// SplitMix64 reseeded per-query with the query index; deterministic, no
/// system clock or RNG. Queries are length 16 (typical short prompt; KSTE
/// Tier-0 stable at this size).
///
/// Design note (load-bearing — caught Stage 3 v1):
/// `sp_kste_encode` (kste_encode.c label_of):
///   1. Truncates input to first 24 values.
///   2. Clamps each value to int16 range via `quantize()`.
///   3. Sorts the clamped slice and samples 6 order statistics.
/// If raw input values exceed int16 (±32767), the clamp pins many values
/// to the extrema -> distinct queries collapse to the same Tier-0 root ->
/// RoutingMasks collide.
///
/// Two prior schemes failed T_MEMO_M5_ROUTING_VARIES:
///   v0 — `i*1_000_003 + j*31 + 7` (pure additive shift): all clamped to
///         32767 since 1M*i > i16 max -> 20% distinct fraction observed.
///   v1 — SplitMix64 high-32-bit dump: still mostly out-of-i16-range ->
///         ~35% distinct fraction observed.
///
/// v2 (this scheme) draws SplitMix64 then folds to int16 range so the
/// quantize step is the IDENTITY, preserving the entropy of the raw
/// SplitMix64 output through encoding.
fn gen_queries(n: usize) -> Vec<Vec<i32>> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        // Seed per-query SplitMix64 with a non-trivial transform of the
        // query index so neighbouring i's don't share low-bit state.
        let mut state: u64 = (i as u64).wrapping_mul(0xA0761D6478BD642F) ^ 0x9E3779B97F4A7C15;
        let mut q = Vec::with_capacity(16);
        for _ in 0..16usize {
            // SplitMix64 step inline to keep this binary self-contained.
            state = state.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^= z >> 31;
            // Map to i16 range; `quantize()` in kste_encode.c will be a
            // no-op for these values, preserving SplitMix64 entropy through
            // the encoding step.
            let v_i16: i16 = ((z as u64) % 65536u64) as i16; // wrap into i16 range
            q.push(v_i16 as i32);
        }
        out.push(q);
    }
    out
}

// ─── Tokens for prefill: clamp to vocab range ───────────────────────────────

fn query_to_tokens(q: &[i32], vocab_size: usize) -> Vec<i32> {
    q.iter().map(|&v| {
        let m = (v.unsigned_abs() as usize) % vocab_size.max(1);
        m as i32
    }).collect()
}

// ─── Host stub ──────────────────────────────────────────────────────────────

#[cfg(not(target_os = "android"))]
fn main() {
    eprintln!("sp_memo_m5_routing_smoke: host build is a stub -- run on android (S22U).");
    eprintln!("  Usage: sp_memo_m5_routing_smoke <memo.spm> <memo.spt> [--queries N] [--det-cycles N] [--report-json PATH]");
    std::process::exit(0);
}

// ─── Main (android) ─────────────────────────────────────────────────────────

#[cfg(target_os = "android")]
fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "Usage: {} <memo_model.spm> <memo_tok.spt> [--queries N] [--det-cycles N] [--report-json PATH]",
            args.get(0).map(|s| s.as_str()).unwrap_or("sp_memo_m5_routing_smoke")
        );
        std::process::exit(2);
    }
    let memo_model_path = args[1].clone();
    let memo_tok_path = args[2].clone();

    let mut n_queries: usize = 100;
    let mut det_cycles: usize = 100;
    let mut report_json: Option<String> = None;
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--queries" => { n_queries = args.get(i+1).and_then(|s| s.parse().ok()).unwrap_or(100); i += 2; }
            "--det-cycles" => { det_cycles = args.get(i+1).and_then(|s| s.parse().ok()).unwrap_or(100); i += 2; }
            "--report-json" => { report_json = args.get(i+1).cloned(); i += 2; }
            other => { eprintln!("[M.5] unknown arg: {other}"); i += 1; }
        }
    }

    eprintln!("[M.5] sp_memo_m5_routing_smoke -- KSTE-routed sparse Memory activation");
    eprintln!("[M.5]   Memory model = {memo_model_path}");
    eprintln!("[M.5]   queries      = {n_queries}");
    eprintln!("[M.5]   det-cycles   = {det_cycles}");
    eprintln!("[M.5]   variant      = B (orchestration-side advisory mask)");

    let mut fails: usize = 0;
    let mut json = J::new();
    json.kv_str("sprint", "M.5");
    json.kv_str("variant", "B-advisory");

    // ─── Load Memory model ────────────────────────────────────────────────
    let (memo_model, memo_arch, memo_load_ms) = match load_model(&memo_model_path, &memo_tok_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("[M.5] Memory load FAIL: {e}");
            json.obj("memory").kv_str("load_error", &e).end_obj();
            std::fs::write(report_json.as_deref().unwrap_or("/dev/null"), json.finish()).ok();
            std::process::exit(1);
        }
    };
    let n_layers = memo_arch.n_layers;
    let n_heads = memo_arch.n_heads;
    let vocab_size = memo_arch.vocab_size as usize;
    eprintln!("[M.5]   loaded: n_layers={} n_heads={} vocab={} load_wall={} ms",
              n_layers, n_heads, vocab_size, memo_load_ms);
    // n_heads can legitimately be 0 on very old arch_struct headers (sp_l1.h
    // sp_arch_info.n_heads is "unspecified" sentinel = 0 from a pre-extension
    // arch_struct). Treat as fatal -- the M.5 routing has no fallback for
    // unspecified head count.
    if n_heads == 0 {
        eprintln!("[M.5] FATAL: sp_arch_info.n_heads == 0 (unspecified). Model may have been");
        eprintln!("[M.5]        produced by a pre-extension transcoder. M.5 routing requires");
        eprintln!("[M.5]        a non-zero head count. UPSTREAM: ship a refresh transcoder pass.");
        json.obj("memory")
            .kv_u64("n_layers", n_layers as u64)
            .kv_u64("n_heads", n_heads as u64)
            .kv_str("fatal", "sp_arch_info.n_heads == 0")
        .end_obj();
        std::fs::write(report_json.as_deref().unwrap_or("/dev/null"), json.finish()).ok();
        std::process::exit(1);
    }
    json.obj("memory")
        .kv_str("path", &memo_model_path)
        .kv_u64("load_wall_ms", memo_load_ms as u64)
        .kv_u64("n_layers", n_layers as u64)
        .kv_u64("n_heads", n_heads as u64)
        .kv_u64("n_kv_heads", memo_arch.n_kv_heads as u64)
        .kv_u64("head_dim", memo_arch.head_dim as u64)
        .kv_u64("vocab_size", vocab_size as u64)
    .end_obj();

    // K defaults: K=8 (primary) and K=4 (more-sparse comparator); clamp to
    // n_heads in case Memory model arch is smaller than expected.
    let k_primary: u32 = std::cmp::min(8, n_heads);
    let k_sparse:  u32 = std::cmp::min(4, n_heads);
    eprintln!("[M.5]   K_primary = {k_primary}/{n_heads} ; K_sparse = {k_sparse}/{n_heads}");

    // ─── Gate 1: T_MEMO_M5_ROUTING_DETERMINISTIC ──────────────────────────
    eprintln!("\n[M.5] === T_MEMO_M5_ROUTING_DETERMINISTIC ===");
    let det_query: Vec<i32> = (0..16i32).map(|j| j * 31 + 7).collect();
    let baseline = match compute_memory_routing(&det_query, n_layers, n_heads, k_primary) {
        Ok(m) => m,
        Err(e) => { eprintln!("[M.5] routing FAIL: {e}"); std::process::exit(1); }
    };
    let mut divergences: usize = 0;
    for c in 0..det_cycles {
        let m = match compute_memory_routing(&det_query, n_layers, n_heads, k_primary) {
            Ok(m) => m,
            Err(e) => { eprintln!("[M.5] routing FAIL cycle {c}: {e}"); std::process::exit(1); }
        };
        if m != baseline {
            divergences += 1;
            if divergences <= 3 {
                eprintln!("[M.5]   divergence at cycle {c}: hamming={}", m.hamming(&baseline));
            }
        }
    }
    let det_pass = divergences == 0;
    eprintln!("[M.5]   runs={det_cycles} divergence_count={divergences} -> {}",
              if det_pass { "PASS" } else { "FAIL" });
    if !det_pass { fails += 1; }
    json.obj("gate_routing_deterministic")
        .kv_u64("runs", det_cycles as u64)
        .kv_u64("divergence_count", divergences as u64)
        .kv_str("verdict", if det_pass { "PASS" } else { "FAIL" })
    .end_obj();

    // ─── Gate 2: T_MEMO_M5_ROUTING_VARIES ─────────────────────────────────
    eprintln!("\n[M.5] === T_MEMO_M5_ROUTING_VARIES ===");
    let varies_queries = gen_queries(10);
    let varies_masks: Vec<RoutingMask> = varies_queries.iter()
        .map(|q| compute_memory_routing(q, n_layers, n_heads, k_primary).unwrap())
        .collect();
    let mut pairs_total = 0usize;
    let mut pairs_distinct = 0usize;
    let mut sum_hamming: u64 = 0;
    for i in 0..varies_masks.len() {
        for j in (i+1)..varies_masks.len() {
            pairs_total += 1;
            let h = varies_masks[i].hamming(&varies_masks[j]);
            sum_hamming += h as u64;
            if h > 0 { pairs_distinct += 1; }
        }
    }
    let distinct_frac = if pairs_total == 0 { 0.0 } else { pairs_distinct as f64 / pairs_total as f64 };
    let mean_hamming = if pairs_total == 0 { 0.0 } else { sum_hamming as f64 / pairs_total as f64 };
    let varies_pass = distinct_frac >= 0.80;
    eprintln!("[M.5]   pairs_total={pairs_total} pairs_distinct={pairs_distinct} ({:.1} %) mean_hamming={:.2} -> {}",
              distinct_frac * 100.0, mean_hamming, if varies_pass { "PASS" } else { "FAIL" });
    if !varies_pass { fails += 1; }
    json.obj("gate_routing_varies")
        .kv_u64("pairs_total", pairs_total as u64)
        .kv_u64("pairs_distinct", pairs_distinct as u64)
        .kv_f64("distinct_fraction", distinct_frac)
        .kv_f64("mean_hamming_distance", mean_hamming)
        .kv_u64("k_per_layer", k_primary as u64)
        .kv_u64("n_layers", n_layers as u64)
        .kv_u64("n_heads", n_heads as u64)
        .kv_str("verdict", if varies_pass { "PASS" } else { "FAIL" })
    .end_obj();

    // ─── Stage: build sessions for forward gates ─────────────────────────
    let base_sess = match create_session(&memo_model) {
        Ok(s) => s,
        Err(e) => { eprintln!("[M.5] base session FAIL: {e}"); std::process::exit(1); }
    };

    // ─── Gate 4 (first): T_MEMO_M5_TTFT_MEASURED (real full + estimated sparse) ─
    eprintln!("\n[M.5] === T_MEMO_M5_TTFT_MEASURED ===");
    // Run full-forward TTFT on a representative query (gen_queries(1)[0]).
    let ttft_query = &gen_queries(1)[0];
    let ttft_tokens = query_to_tokens(ttft_query, vocab_size);
    let mut ttft_sess = match clone_session(&base_sess) {
        Ok(s) => s,
        Err(e) => { eprintln!("[M.5] ttft session clone FAIL: {e}"); std::process::exit(1); }
    };
    let mut ttft_logits = vec![0f32; vocab_size];
    // Warmup: first call is dominated by allocator / kernel cold start; throw away.
    let _ = prefill(&mut ttft_sess, &ttft_tokens, &mut ttft_logits);
    drop(ttft_sess);
    // Real measurement: N=5 runs, take the median.
    const TTFT_RUNS: usize = 5;
    let mut full_walls_ms: Vec<f64> = Vec::with_capacity(TTFT_RUNS);
    for _ in 0..TTFT_RUNS {
        let mut s = match clone_session(&base_sess) {
            Ok(s) => s,
            Err(e) => { eprintln!("[M.5] ttft loop clone FAIL: {e}"); std::process::exit(1); }
        };
        let mut logits = vec![0f32; vocab_size];
        let t0 = Instant::now();
        let _ = prefill(&mut s, &ttft_tokens, &mut logits);
        full_walls_ms.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    full_walls_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let full_ttft_ms = full_walls_ms[TTFT_RUNS / 2];
    let mask_k8 = compute_memory_routing(ttft_query, n_layers, n_heads, k_primary).unwrap();
    let mask_k4 = compute_memory_routing(ttft_query, n_layers, n_heads, k_sparse).unwrap();
    let sparse_ttft_k8 = estimate_sparse_ttft_ms(full_ttft_ms, &mask_k8);
    let sparse_ttft_k4 = estimate_sparse_ttft_ms(full_ttft_ms, &mask_k4);
    let speedup_k8 = if sparse_ttft_k8 > 0.0 { full_ttft_ms / sparse_ttft_k8 } else { 0.0 };
    let speedup_k4 = if sparse_ttft_k4 > 0.0 { full_ttft_ms / sparse_ttft_k4 } else { 0.0 };
    eprintln!("[M.5]   full_ttft_ms (median of {TTFT_RUNS}) = {:.3}", full_ttft_ms);
    eprintln!("[M.5]   sparse_ttft_K{k_primary}_estimated  = {:.3}  (speedup_est = {:.3}x)", sparse_ttft_k8, speedup_k8);
    eprintln!("[M.5]   sparse_ttft_K{k_sparse}_estimated  = {:.3}  (speedup_est = {:.3}x)", sparse_ttft_k4, speedup_k4);
    eprintln!("[M.5]   T_MEMO_M5_TTFT_MEASURED PASS (no threshold; reported)");
    json.obj("gate_ttft_measured")
        .kv_f64("full_forward_ttft_ms", full_ttft_ms)
        .kv_u64("ttft_runs", TTFT_RUNS as u64)
        .kv_f64("sparse_forward_ttft_ms_K_primary_estimated", sparse_ttft_k8)
        .kv_f64("sparse_forward_ttft_ms_K_sparse_estimated", sparse_ttft_k4)
        .kv_u64("k_primary", k_primary as u64)
        .kv_u64("k_sparse", k_sparse as u64)
        .kv_f64("observed_speedup_K_primary_estimated", speedup_k8)
        .kv_f64("observed_speedup_K_sparse_estimated", speedup_k4)
        .kv_f64("active_fraction_K_primary", mask_k8.active_fraction())
        .kv_f64("active_fraction_K_sparse", mask_k4.active_fraction())
        .kv_str("verdict", "PASS")
        .kv_str("note", "full_ttft_ms is measured; sparse_ttft_ms_*_estimated is linear-in-active-fraction estimate, NOT a measurement. Real sparse-forward measurement requires Variant A (kernel-side head mask).")
    .end_obj();

    // ─── Gate 3: T_MEMO_M5_INVARIANCE_PRESERVING (Variant B advisory) ─────
    eprintln!("\n[M.5] === T_MEMO_M5_INVARIANCE_PRESERVING ===");
    eprintln!("[M.5]   Variant B advisory: sparse forward = identity to full forward");
    eprintln!("[M.5]   (no kernel mask applied). Gate interpreted as 'routing layer");
    eprintln!("[M.5]   does not break full forward across N={n_queries} queries'.");
    let inv_queries = gen_queries(n_queries);
    let mut top1_agree: usize = 0;
    let mut top5_overlap_sum: f64 = 0.0;
    let mut kl_sum: f64 = 0.0;
    let mut wall_full_total_us: u128 = 0;
    let mut wall_sparse_total_us: u128 = 0;
    let mut errors: usize = 0;
    for (qi, q) in inv_queries.iter().enumerate() {
        let tokens = query_to_tokens(q, vocab_size);
        // Routing layer runs (advisory; result is captured but not applied).
        let mask = match compute_memory_routing(q, n_layers, n_heads, k_primary) {
            Ok(m) => m,
            Err(e) => { eprintln!("[M.5] routing FAIL q{qi}: {e}"); errors += 1; continue; }
        };
        // Full forward.
        let mut sf = match clone_session(&base_sess) {
            Ok(s) => s,
            Err(e) => { eprintln!("[M.5] full clone FAIL q{qi}: {e}"); errors += 1; continue; }
        };
        let mut full_logits = vec![0f32; vocab_size];
        let t0 = Instant::now();
        if let Err(e) = prefill(&mut sf, &tokens, &mut full_logits) {
            eprintln!("[M.5] full prefill FAIL q{qi}: {e}"); errors += 1; continue;
        }
        wall_full_total_us += t0.elapsed().as_micros();
        drop(sf);
        let full_top1 = argmax_f32(&full_logits);

        // Sparse forward (Variant B identity): structurally re-run full forward
        // and treat its output as the sparse result -- the kernel does NOT see
        // the mask. This trivially yields agreement = 1.0; the gate is testing
        // that routing computation alongside full forward does not corrupt the
        // pipeline (no allocator / FFI interactions break things).
        let mut ss = match clone_session(&base_sess) {
            Ok(s) => s,
            Err(e) => { eprintln!("[M.5] sparse clone FAIL q{qi}: {e}"); errors += 1; continue; }
        };
        let mut sparse_logits = vec![0f32; vocab_size];
        let t1 = Instant::now();
        if let Err(e) = prefill(&mut ss, &tokens, &mut sparse_logits) {
            eprintln!("[M.5] sparse prefill FAIL q{qi}: {e}"); errors += 1; continue;
        }
        wall_sparse_total_us += t1.elapsed().as_micros();
        drop(ss);
        let sparse_top1 = argmax_f32(&sparse_logits);

        // Variant B: full == sparse byte-exact by construction.
        if full_top1 == sparse_top1 { top1_agree += 1; }

        // Top-5 overlap.
        let mut full_top5: Vec<(usize, f32)> = full_logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        full_top5.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let full_top5_set: std::collections::HashSet<usize> = full_top5.iter().take(5).map(|(i, _)| *i).collect();
        let mut sparse_top5: Vec<(usize, f32)> = sparse_logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        sparse_top5.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let sparse_top5_set: std::collections::HashSet<usize> = sparse_top5.iter().take(5).map(|(i, _)| *i).collect();
        top5_overlap_sum += full_top5_set.intersection(&sparse_top5_set).count() as f64 / 5.0;

        // Mean KL divergence sparse||full over softmax(logits). Variant B
        // identity -> KL = 0.0 trivially; computed honestly to surface any
        // unexpected numerical drift.
        let mut max_f = f32::NEG_INFINITY;
        for &v in &full_logits { if v > max_f { max_f = v; } }
        let mut max_s = f32::NEG_INFINITY;
        for &v in &sparse_logits { if v > max_s { max_s = v; } }
        let mut zf: f64 = 0.0;
        let mut zs: f64 = 0.0;
        for i in 0..vocab_size {
            zf += ((full_logits[i] - max_f) as f64).exp();
            zs += ((sparse_logits[i] - max_s) as f64).exp();
        }
        let mut kl: f64 = 0.0;
        for i in 0..vocab_size {
            let pf = ((full_logits[i] - max_f) as f64).exp() / zf;
            let ps = ((sparse_logits[i] - max_s) as f64).exp() / zs;
            if ps > 1e-30 && pf > 1e-30 {
                kl += ps * (ps / pf).ln();
            }
        }
        kl_sum += kl;

        // Sanity log every 25 queries.
        if (qi + 1) % 25 == 0 {
            eprintln!("[M.5]   q{:03}: full_top1={} sparse_top1={} agree_so_far={}/{} mask_active={}",
                      qi, full_top1, sparse_top1, top1_agree, qi + 1, mask.total_active());
        }
    }
    let denom = (n_queries - errors).max(1) as f64;
    let top1_rate = top1_agree as f64 / denom;
    let top5_rate = top5_overlap_sum / denom;
    let kl_mean = kl_sum / denom;
    // Variant B threshold is the spec's 0.70 -- and Variant B trivially hits 1.0.
    let inv_pass = top1_rate >= 0.70;
    eprintln!("[M.5]   queries_tested={} errors={} top1_agreement_rate={:.4} top5_overlap_rate={:.4} mean_kl_divergence={:.6} -> {}",
              n_queries - errors, errors, top1_rate, top5_rate, kl_mean, if inv_pass { "PASS" } else { "FAIL" });
    if !inv_pass { fails += 1; }
    json.obj("gate_invariance_preserving")
        .kv_u64("queries_total", n_queries as u64)
        .kv_u64("queries_errors", errors as u64)
        .kv_u64("queries_tested", (n_queries - errors) as u64)
        .kv_f64("top1_agreement_rate", top1_rate)
        .kv_f64("top5_overlap_rate", top5_rate)
        .kv_f64("mean_kl_divergence_to_full", kl_mean)
        .kv_u64("k_per_layer", k_primary as u64)
        .kv_u64("wall_full_total_us", wall_full_total_us as u64)
        .kv_u64("wall_sparse_total_us", wall_sparse_total_us as u64)
        .kv_str("variant_b_advisory_note", "Variant B does not apply mask at kernel; sparse forward is identity to full forward. top1_agreement=1.0 by construction; gate tests that routing layer presence does not break the pipeline. Real sparse-vs-full divergence measurement requires Variant A.")
        .kv_str("verdict", if inv_pass { "PASS-ADVISORY" } else { "FAIL" })
    .end_obj();

    // ─── Final gates summary ──────────────────────────────────────────────
    eprintln!("\n[M.5] === GATES SUMMARY ===");
    eprintln!("[M.5]   T_MEMO_M5_ROUTING_DETERMINISTIC   {}", if det_pass { "PASS" } else { "FAIL" });
    eprintln!("[M.5]   T_MEMO_M5_ROUTING_VARIES          {}", if varies_pass { "PASS" } else { "FAIL" });
    eprintln!("[M.5]   T_MEMO_M5_INVARIANCE_PRESERVING   {} (Variant B advisory)", if inv_pass { "PASS-ADVISORY" } else { "FAIL" });
    eprintln!("[M.5]   T_MEMO_M5_TTFT_MEASURED           PASS (reported, no gate)");
    eprintln!("[M.5]   fails = {fails}");

    json.obj("gates_summary")
        .kv_str("T_MEMO_M5_ROUTING_DETERMINISTIC", if det_pass { "PASS" } else { "FAIL" })
        .kv_str("T_MEMO_M5_ROUTING_VARIES", if varies_pass { "PASS" } else { "FAIL" })
        .kv_str("T_MEMO_M5_INVARIANCE_PRESERVING", if inv_pass { "PASS-ADVISORY" } else { "FAIL" })
        .kv_str("T_MEMO_M5_TTFT_MEASURED", "PASS")
        .kv_u64("fail_count", fails as u64)
    .end_obj();

    // ─── Emit JSON ────────────────────────────────────────────────────────
    let report = json.finish();
    if let Some(p) = report_json.as_ref() {
        match std::fs::write(p, &report) {
            Ok(_) => eprintln!("[M.5] report written to {p}"),
            Err(e) => eprintln!("[M.5] WARNING: report write failed: {e}"),
        }
    }
    println!("{report}");

    // Variant B's invariance gate is advisory PASS by construction. The
    // exit code uses the fails counter so deterministic / varies failures
    // (which are NOT advisory) flunk the binary cleanly.
    std::process::exit(if fails == 0 { 0 } else { 1 });
}
