//! §4-NTT Sprint NTT-bench — tokens/sec measurement on Sam's S22U.
//!
//! Measures end-to-end prefill + decode wall-clock for one (model, config)
//! cell at a time. The driver script (ntt_bench_toks_run.ps1) invokes this
//! binary 6 times — once per cell — with the appropriate env vars set so
//! math-core's `g_ntt_attn` static init reads the correct value at process
//! start.
//!
//! Cell matrix (driver script):
//!     cell 1: Executive  fp32        (no env vars)
//!     cell 2: Executive  host NTT    SP_ENGINE_NTT_ATTN=1
//!     cell 3: Executive  hex NTT     SP_ENGINE_NTT_ATTN=1 + SP_ENGINE_NTT_ATTN_HEX=1
//!     cell 4: Memory     fp32        (no env vars)
//!     cell 5: Memory     host NTT    SP_ENGINE_NTT_ATTN=1
//!     cell 6: Memory     hex NTT     SP_ENGINE_NTT_ATTN=1 + SP_ENGINE_NTT_ATTN_HEX=1
//!
//! Within each invocation: load model, then run 3 reps of (clone session →
//! prefill 16 tokens → decode 32 steps), so the model+harness boot cost is
//! amortized across reps. Per-rep prefill + decode wall-clock + per-step
//! decode breakdown all captured.
//!
//! Output: emits a JSON object to stdout AND appends the same object as a
//! line in the `--report-jsonl` file. Driver script concatenates the lines
//! into the final report.
//!
//! Gates (per PLAN-NTT-bench.md §"Substantive gates"):
//!   T_NTT_BENCH_ALL_CELLS_COMPLETE        all 6 cells run non-NaN toks/sec
//!   T_NTT_BENCH_FP32_BASELINE_CAPTURED    cells 1 + 4 PASS
//!   T_NTT_BENCH_NTT_HOST_VS_HEX_BOTH_RUN  cells 2,3,5,6 PASS
//!   T_NTT_BENCH_REPORT_LANDS              JSON + closure markdown both written
//!
//! CLI:
//!   sp_ntt_bench_toks --cell N --model-path PATH --tok-path PATH \
//!                     --model-label LABEL --config-label LABEL \
//!                     [--report-jsonl PATH]
//!
//! On android:
//!   adb push sp_ntt_bench_toks /data/local/tmp/
//!   adb shell chmod +x /data/local/tmp/sp_ntt_bench_toks
//!   # driver script handles the 6 invocations + env var per cell
//!
//! Host build is a stub (prints "android-only", exits 0).

#![cfg_attr(not(target_os = "android"), allow(dead_code))]

#[cfg(target_os = "android")]
use sp_daemon::ffi_l1 as ffi;

#[cfg(target_os = "android")]
use sp_daemon::ntt_hex_dispatch;

#[cfg(target_os = "android")]
use ffi::sp_status_SP_OK;

use std::ffi::CString;
use std::os::raw::c_int;
use std::ptr;
use std::sync::atomic::AtomicI32;
use std::sync::Arc;
use std::time::{Duration, Instant};

// ─── L1 wrappers (mirror M.1 / NTT.5c-smoke verbatim) ──────────────────────

#[cfg(target_os = "android")]
struct L1Model(*mut ffi::sp_model);
#[cfg(target_os = "android")]
unsafe impl Send for L1Model {}
#[cfg(target_os = "android")]
unsafe impl Sync for L1Model {}
#[cfg(target_os = "android")]
impl Drop for L1Model {
    fn drop(&mut self) {
        unsafe { ffi::sp_model_unload(self.0) };
    }
}

#[cfg(target_os = "android")]
struct L1Session {
    ptr: *mut ffi::sp_session,
    _cancel: Arc<AtomicI32>,
}
#[cfg(target_os = "android")]
unsafe impl Send for L1Session {}
#[cfg(target_os = "android")]
impl Drop for L1Session {
    fn drop(&mut self) {
        unsafe { ffi::sp_session_destroy(self.ptr) };
    }
}

#[cfg(target_os = "android")]
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
        return Err(format!("sp_model_load({model_path}) → status={st}: {detail}"));
    }
    let mut arch: ffi::sp_arch_info = unsafe { std::mem::zeroed() };
    let st = unsafe { ffi::sp_model_arch(ptr, &mut arch) };
    if st != sp_status_SP_OK {
        unsafe { ffi::sp_model_unload(ptr) };
        return Err(format!("sp_model_arch → status={st}"));
    }
    Ok((L1Model(ptr), arch, wall_ms))
}

#[cfg(target_os = "android")]
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
        return Err(format!("sp_session_create → status={st}"));
    }
    Ok(L1Session { ptr, _cancel: cancel })
}

#[cfg(target_os = "android")]
fn clone_session(s: &L1Session) -> Result<L1Session, String> {
    let cancel = Arc::new(AtomicI32::new(0));
    let cancel_raw = cancel.as_ptr() as *mut c_int;
    let mut out: *mut ffi::sp_session = ptr::null_mut();
    let st = unsafe { ffi::sp_session_clone(s.ptr, cancel_raw, &mut out) };
    if st != sp_status_SP_OK {
        return Err(format!("sp_session_clone → status={st}"));
    }
    Ok(L1Session { ptr: out, _cancel: cancel })
}

#[cfg(target_os = "android")]
fn prefill(s: &mut L1Session, tokens: &[i32], logits: &mut [f32]) -> Result<(), String> {
    let st = unsafe {
        ffi::sp_prefill_chunk(s.ptr, tokens.as_ptr(), tokens.len(), logits.as_mut_ptr(), logits.len())
    };
    if st != sp_status_SP_OK {
        return Err(format!("sp_prefill_chunk → status={st}"));
    }
    Ok(())
}

#[cfg(target_os = "android")]
fn decode_one(s: &mut L1Session, tok: i32, logits: &mut [f32]) -> Result<(), String> {
    let st = unsafe {
        ffi::sp_decode_step(s.ptr, tok, logits.as_mut_ptr(), logits.len())
    };
    if st != sp_status_SP_OK {
        return Err(format!("sp_decode_step(tok={tok}) → status={st}"));
    }
    Ok(())
}

#[cfg(target_os = "android")]
fn argmax(v: &[f32]) -> i32 {
    let mut a = 0usize;
    for (i, &x) in v.iter().enumerate().skip(1) {
        if x > v[a] {
            a = i;
        }
    }
    a as i32
}

#[cfg(target_os = "android")]
fn all_finite(v: &[f32]) -> bool {
    v.iter().all(|x| x.is_finite())
}

// ─── Backend setup (config C cells) — mirrors sp_ntt_5c_forward_smoke ──────

#[cfg(target_os = "android")]
fn maybe_open_backend() -> Option<Arc<ntt_hex_dispatch::ComputeBackend>> {
    use sp_daemon::dsp_rpc::FastRpcSession;
    const COMPUTE_SKEL_URI: &str =
        "file:///libsp_compute_skel.so?sp_compute_skel_handle_invoke&_modver=1.0&_dom=cdsp";
    match FastRpcSession::new(COMPUTE_SKEL_URI) {
        Ok(s) => {
            eprintln!("[bench] FastRpcSession open OK");
            Some(Arc::new(ntt_hex_dispatch::ComputeBackend::new(Arc::new(s))))
        }
        Err(e) => {
            eprintln!("[bench] FastRpcSession open failed: {e:?} — backend disabled");
            None
        }
    }
}

#[cfg(target_os = "android")]
fn register_backend_on_session(
    s: &L1Session,
    backend: &Arc<ntt_hex_dispatch::ComputeBackend>,
) -> Result<(), String> {
    let backend_clone = Arc::clone(backend);
    let leaked: *mut ntt_hex_dispatch::ComputeBackend =
        Arc::into_raw(backend_clone) as *mut _;
    let (fwd, inv) = ntt_hex_dispatch::ComputeBackend::dispatch_fns();
    let rc = unsafe {
        ffi::sp_session_register_compute_backend(
            s.ptr,
            leaked as *mut std::os::raw::c_void,
            Some(fwd),
            Some(inv),
        )
    };
    if rc != sp_status_SP_OK {
        unsafe { Arc::from_raw(leaked); }
        return Err(format!("sp_session_register_compute_backend → status={rc}"));
    }
    Ok(())
}

// ─── JSON helpers (tiny — avoid serde dep) ─────────────────────────────────

fn esc_json(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 4);
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o
}

// ─── Rep result types ──────────────────────────────────────────────────────

#[derive(Default, Clone)]
struct RepResult {
    rep_index: u32,
    prefill_wall_us: u64,
    prefill_toks_per_sec: f64,
    decode_n_run: u32,
    decode_total_wall_us: u64,
    decode_toks_per_sec: f64,
    decode_first_step_us: u64,    // cold-ish per-step inside this rep
    decode_steady_mean_us: f64,   // mean of steps 2..N
    first_argmax_after_prefill: i32,
    last_decoded_token: i32,
    error: Option<String>,
    warning: Option<String>,
}

// ─── Host stub (non-android) ───────────────────────────────────────────────
#[cfg(not(target_os = "android"))]
fn main() {
    eprintln!("sp_ntt_bench_toks: host build is a stub — run on android (S22U)");
    eprintln!("  See header doc for adb push + invocation; use ntt_bench_toks_run.ps1 driver.");
    std::process::exit(0);
}

// ─── Main (android) ────────────────────────────────────────────────────────
#[cfg(target_os = "android")]
fn main() {
    // Argument parsing (manual; mirror M.1 style)
    let args: Vec<String> = std::env::args().collect();
    let mut cell: u32 = 0;
    let mut model_path: Option<String> = None;
    let mut tok_path: Option<String> = None;
    let mut model_label: Option<String> = None;
    let mut config_label: Option<String> = None;
    let mut report_jsonl: Option<String> = None;
    let mut prompt_len: usize = 16;
    let mut decode_n: usize = 32;
    let mut reps: u32 = 3;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--cell" => {
                cell = args.get(i+1).and_then(|s| s.parse().ok()).unwrap_or(0);
                i += 2;
            }
            "--model-path" => { model_path = args.get(i+1).cloned(); i += 2; }
            "--tok-path"   => { tok_path   = args.get(i+1).cloned(); i += 2; }
            "--model-label" => { model_label = args.get(i+1).cloned(); i += 2; }
            "--config-label" => { config_label = args.get(i+1).cloned(); i += 2; }
            "--report-jsonl" => { report_jsonl = args.get(i+1).cloned(); i += 2; }
            "--prompt-len" => { prompt_len = args.get(i+1).and_then(|s| s.parse().ok()).unwrap_or(16); i += 2; }
            "--decode-n"   => { decode_n   = args.get(i+1).and_then(|s| s.parse().ok()).unwrap_or(32); i += 2; }
            "--reps"       => { reps       = args.get(i+1).and_then(|s| s.parse().ok()).unwrap_or(3); i += 2; }
            other => { eprintln!("[bench] unknown arg: {other}"); i += 1; }
        }
    }

    if cell == 0 || model_path.is_none() || tok_path.is_none()
        || model_label.is_none() || config_label.is_none()
    {
        eprintln!("Usage: {} --cell N --model-path PATH --tok-path PATH \\",
                  args.get(0).map(|s| s.as_str()).unwrap_or("sp_ntt_bench_toks"));
        eprintln!("       --model-label LABEL --config-label LABEL \\");
        eprintln!("       [--report-jsonl PATH] [--prompt-len 16] [--decode-n 32] [--reps 3]");
        std::process::exit(2);
    }
    let model_path = model_path.unwrap();
    let tok_path = tok_path.unwrap();
    let model_label = model_label.unwrap();
    let config_label = config_label.unwrap();

    let ntt_attn = std::env::var("SP_ENGINE_NTT_ATTN")
        .map(|v| v.trim() == "1").unwrap_or(false);
    let ntt_attn_hex = std::env::var("SP_ENGINE_NTT_ATTN_HEX")
        .map(|v| v.trim() == "1").unwrap_or(false);

    eprintln!("[bench] cell={} model={} config={} prompt_len={} decode_n={} reps={}",
              cell, model_label, config_label, prompt_len, decode_n, reps);
    eprintln!("[bench]   env: SP_ENGINE_NTT_ATTN={} SP_ENGINE_NTT_ATTN_HEX={}",
              ntt_attn, ntt_attn_hex);

    // ─── Load model ────────────────────────────────────────────────────────
    let (model, arch, load_ms) = match load_model(&model_path, &tok_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("[bench] FATAL load: {e}");
            emit_cell_failure(&report_jsonl, cell, &model_label, &config_label,
                              &model_path, ntt_attn, ntt_attn_hex,
                              &format!("load_model: {e}"));
            std::process::exit(3);
        }
    };
    eprintln!(
        "[bench] model loaded in {load_ms} ms: arch_id={} HD={} hidden={} layers={} vocab={}",
        arch.arch_id, arch.head_dim, arch.hidden_dim, arch.n_layers, arch.vocab_size,
    );

    let hd = arch.head_dim;
    let pot_bluestein = hd >= 2 && hd <= 256 && (hd & (hd - 1)) == 0;
    let direct_pr = hd == 128 || hd == 256 || hd == 512;
    eprintln!("[bench]   HD={hd}: direct_pr_admissible={} bluestein_admissible={}",
              direct_pr, pot_bluestein);

    // ─── Optional backend setup (config C only) ────────────────────────────
    let backend = if ntt_attn_hex {
        maybe_open_backend()
    } else {
        None
    };

    // ─── Base session for cloning ──────────────────────────────────────────
    let base = match create_session(&model) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[bench] FATAL session create: {e}");
            emit_cell_failure(&report_jsonl, cell, &model_label, &config_label,
                              &model_path, ntt_attn, ntt_attn_hex,
                              &format!("create_session: {e}"));
            std::process::exit(3);
        }
    };

    // ─── Run reps ──────────────────────────────────────────────────────────
    let prompt: Vec<i32> = (1..=(prompt_len as i32)).collect();
    let vocab = arch.vocab_size as usize;

    let mut rep_results: Vec<RepResult> = Vec::with_capacity(reps as usize);
    let mut cell_error: Option<String> = None;

    for rep in 0..reps {
        let mut rr = RepResult::default();
        rr.rep_index = rep;

        // Fresh session per rep (M.1 clone pattern).
        let mut sess = match clone_session(&base) {
            Ok(s) => s,
            Err(e) => {
                rr.error = Some(format!("clone: {e}"));
                eprintln!("[bench]   rep {rep}: clone FAIL: {e}");
                rep_results.push(rr);
                cell_error.get_or_insert_with(|| format!("rep {rep} clone fail"));
                continue;
            }
        };

        // Register backend on the SESSION (config C only).
        if let Some(ref b) = backend {
            if let Err(e) = register_backend_on_session(&sess, b) {
                eprintln!("[bench]   rep {rep}: backend register WARN: {e}");
                // Continue; bench captures the fact that backend register failed.
            }
        }

        // Reset dispatch counters per rep so we can capture per-rep counts.
        ntt_hex_dispatch::reset_dispatch_counts();
        let (fwd0, inv0) = ntt_hex_dispatch::dispatch_counts();

        // ─── Prefill ───────────────────────────────────────────────────────
        let mut logits = vec![0f32; vocab];
        let t0 = Instant::now();
        let pre_rc = prefill(&mut sess, &prompt, &mut logits);
        let pre_us = t0.elapsed().as_micros();
        if let Err(e) = pre_rc {
            rr.error = Some(format!("prefill: {e}"));
            rr.prefill_wall_us = pre_us as u64;
            eprintln!("[bench]   rep {rep}: prefill FAIL: {e}");
            rep_results.push(rr);
            cell_error.get_or_insert_with(|| format!("rep {rep} prefill fail"));
            continue;
        }
        if !all_finite(&logits) {
            rr.error = Some("prefill_logits_nonfinite".into());
            rr.prefill_wall_us = pre_us as u64;
            eprintln!("[bench]   rep {rep}: prefill logits non-finite");
            rep_results.push(rr);
            cell_error.get_or_insert_with(|| format!("rep {rep} nonfinite"));
            continue;
        }
        rr.prefill_wall_us = pre_us as u64;
        rr.prefill_toks_per_sec = prompt_len as f64 / (pre_us as f64 / 1_000_000.0);
        rr.first_argmax_after_prefill = argmax(&logits);
        eprintln!("[bench]   rep {rep}: prefill {} toks in {} us = {:.3} tok/s (argmax={})",
                  prompt_len, pre_us, rr.prefill_toks_per_sec, rr.first_argmax_after_prefill);

        // ─── Decode loop ───────────────────────────────────────────────────
        let mut step_walls_us: Vec<u64> = Vec::with_capacity(decode_n);
        let mut next_tok = rr.first_argmax_after_prefill;
        let mut decoded: u32 = 0;
        for step in 0..decode_n {
            let mut step_logits = vec![0f32; vocab];
            let t0 = Instant::now();
            let dc_rc = decode_one(&mut sess, next_tok, &mut step_logits);
            let step_us = t0.elapsed().as_micros() as u64;
            if let Err(e) = dc_rc {
                eprintln!("[bench]   rep {rep}: decode step {step} FAIL: {e} \
                          (decoded so far: {decoded})");
                if decoded < 8 {
                    rr.error = Some(format!("decode_partial: {decoded} steps ({e})"));
                    cell_error.get_or_insert_with(|| format!("rep {rep} decode <8"));
                }
                break;
            }
            if !all_finite(&step_logits) {
                eprintln!("[bench]   rep {rep}: decode step {step} non-finite (decoded {decoded})");
                if decoded < 8 {
                    rr.error = Some(format!("decode_nonfinite: {decoded} steps"));
                    cell_error.get_or_insert_with(|| format!("rep {rep} decode nonfinite"));
                }
                break;
            }
            step_walls_us.push(step_us);
            next_tok = argmax(&step_logits);
            decoded += 1;
        }
        rr.decode_n_run = decoded;
        let total: u64 = step_walls_us.iter().copied().sum();
        rr.decode_total_wall_us = total;
        if total > 0 && decoded > 0 {
            rr.decode_toks_per_sec = decoded as f64 / (total as f64 / 1_000_000.0);
        }
        rr.decode_first_step_us = step_walls_us.first().copied().unwrap_or(0);
        if step_walls_us.len() > 1 {
            let tail: u64 = step_walls_us[1..].iter().copied().sum();
            rr.decode_steady_mean_us = tail as f64 / (step_walls_us.len() - 1) as f64;
        }
        rr.last_decoded_token = next_tok;

        let (fwd1, inv1) = ntt_hex_dispatch::dispatch_counts();
        eprintln!("[bench]   rep {rep}: decode {} steps in {} us = {:.3} tok/s \
                   (first_step={}us steady_mean={:.1}us) dispatch fwd+={} inv+={}",
                  decoded, total, rr.decode_toks_per_sec,
                  rr.decode_first_step_us, rr.decode_steady_mean_us,
                  fwd1 - fwd0, inv1 - inv0);

        // Pack per-rep dispatch counter delta into a WARNING field if hex was expected
        // but no dispatches happened (informational, not a hard fail; wall-clock
        // measurements are still valid — the forward just took the direct sp_pr
        // path which has no backend hook per NTT.5c CLOSURE §"What's NOT done").
        // Use a distinct field so the aggregator doesn't drop this rep from stats.
        if ntt_attn_hex && backend.is_some() && (fwd1 - fwd0) == 0 && rr.error.is_none() {
            rr.warning = Some(format!("hex_no_dispatch (HD={hd} likely direct_pr path)"));
            eprintln!("[bench]   rep {rep}: WARN hex expected but 0 dispatches (HD={hd})");
        }

        rep_results.push(rr);
        drop(sess);

        // Small pause between reps to settle the device.
        std::thread::sleep(Duration::from_millis(100));
    }

    // ─── Final dispatch counter snapshot ──────────────────────────────────
    let (fwd_final, inv_final) = ntt_hex_dispatch::dispatch_counts();

    // ─── Aggregate across reps (mean / min / max) ─────────────────────────
    let mut pre_tokps: Vec<f64> = Vec::new();
    let mut dec_tokps: Vec<f64> = Vec::new();
    let mut pre_walls: Vec<u64> = Vec::new();
    let mut dec_walls: Vec<u64> = Vec::new();
    for r in &rep_results {
        if r.error.is_none() {
            pre_tokps.push(r.prefill_toks_per_sec);
            dec_tokps.push(r.decode_toks_per_sec);
            pre_walls.push(r.prefill_wall_us);
            dec_walls.push(r.decode_total_wall_us);
        }
    }
    let prefill_mean = mean(&pre_tokps);
    let prefill_min = min_f(&pre_tokps);
    let prefill_max = max_f(&pre_tokps);
    let decode_mean  = mean(&dec_tokps);
    let decode_min   = min_f(&dec_tokps);
    let decode_max   = max_f(&dec_tokps);

    eprintln!("\n[bench] ═══ cell {cell} summary ({model_label} / {config_label}) ═══");
    eprintln!("[bench]   reps_ok = {}/{reps}", pre_tokps.len());
    eprintln!("[bench]   prefill toks/s: mean={:.3} min={:.3} max={:.3}",
              prefill_mean, prefill_min, prefill_max);
    eprintln!("[bench]   decode  toks/s: mean={:.3} min={:.3} max={:.3}",
              decode_mean, decode_min, decode_max);
    eprintln!("[bench]   total dispatch counts: fwd={} inv={}", fwd_final, inv_final);

    // ─── Emit JSON object ─────────────────────────────────────────────────
    let json = build_cell_json(
        cell, &model_label, &config_label, &model_path,
        ntt_attn, ntt_attn_hex,
        load_ms as u64,
        &arch,
        direct_pr, pot_bluestein,
        prompt_len, decode_n, reps,
        &rep_results,
        prefill_mean, prefill_min, prefill_max,
        decode_mean, decode_min, decode_max,
        fwd_final, inv_final,
        cell_error.as_deref(),
    );
    println!("{json}");
    if let Some(path) = report_jsonl.as_deref() {
        use std::io::Write;
        match std::fs::OpenOptions::new().create(true).append(true).open(path) {
            Ok(mut f) => {
                let line = format!("{json}\n");
                if let Err(e) = f.write_all(line.as_bytes()) {
                    eprintln!("[bench] WARN: write report-jsonl {path}: {e}");
                } else {
                    eprintln!("[bench] appended cell {cell} to {path}");
                }
            }
            Err(e) => {
                eprintln!("[bench] WARN: open report-jsonl {path}: {e}");
            }
        }
    }

    // Cleanup (explicit for clarity)
    drop(base);
    drop(model);
    drop(backend);

    if cell_error.is_some() {
        eprintln!("[bench] cell {cell} had errors; exit 1");
        std::process::exit(1);
    }
    std::process::exit(0);
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() { return f64::NAN; }
    v.iter().sum::<f64>() / v.len() as f64
}
fn min_f(v: &[f64]) -> f64 {
    v.iter().copied().fold(f64::INFINITY, f64::min)
}
fn max_f(v: &[f64]) -> f64 {
    v.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

#[cfg(target_os = "android")]
fn build_cell_json(
    cell: u32,
    model_label: &str,
    config_label: &str,
    model_path: &str,
    ntt_attn: bool,
    ntt_attn_hex: bool,
    load_ms: u64,
    arch: &ffi::sp_arch_info,
    direct_pr: bool,
    bluestein: bool,
    prompt_len: usize,
    decode_n: usize,
    reps: u32,
    rep_results: &[RepResult],
    prefill_mean: f64, prefill_min: f64, prefill_max: f64,
    decode_mean: f64, decode_min: f64, decode_max: f64,
    fwd_total: u64, inv_total: u64,
    cell_error: Option<&str>,
) -> String {
    let mut s = String::with_capacity(2048);
    s.push('{');
    s.push_str(&format!("\"cell\":{cell}"));
    s.push_str(&format!(",\"model_label\":\"{}\"", esc_json(model_label)));
    s.push_str(&format!(",\"config_label\":\"{}\"", esc_json(config_label)));
    s.push_str(&format!(",\"model_path\":\"{}\"", esc_json(model_path)));
    s.push_str(&format!(",\"env\":{{\"SP_ENGINE_NTT_ATTN\":{},\"SP_ENGINE_NTT_ATTN_HEX\":{}}}",
                        ntt_attn, ntt_attn_hex));
    s.push_str(&format!(",\"load_wall_ms\":{load_ms}"));
    s.push_str(&format!(",\"arch\":{{\"arch_id\":{},\"head_dim\":{},\"hidden_dim\":{},\"n_layers\":{},\"vocab_size\":{}}}",
                        arch.arch_id, arch.head_dim, arch.hidden_dim, arch.n_layers, arch.vocab_size));
    s.push_str(&format!(",\"dispatch\":{{\"direct_pr_admissible\":{},\"bluestein_admissible\":{}}}",
                        direct_pr, bluestein));
    s.push_str(&format!(",\"prompt_len\":{prompt_len},\"decode_n_requested\":{decode_n},\"reps\":{reps}"));

    s.push_str(",\"reps_detail\":[");
    for (i, r) in rep_results.iter().enumerate() {
        if i > 0 { s.push(','); }
        s.push('{');
        s.push_str(&format!("\"rep_index\":{},\"prefill_wall_us\":{},\"prefill_toks_per_sec\":{:.6},\"decode_n_run\":{},\"decode_total_wall_us\":{},\"decode_toks_per_sec\":{:.6},\"decode_first_step_us\":{},\"decode_steady_mean_us\":{:.3},\"first_argmax_after_prefill\":{},\"last_decoded_token\":{}",
            r.rep_index, r.prefill_wall_us, r.prefill_toks_per_sec, r.decode_n_run, r.decode_total_wall_us, r.decode_toks_per_sec, r.decode_first_step_us, r.decode_steady_mean_us, r.first_argmax_after_prefill, r.last_decoded_token));
        if let Some(e) = r.error.as_deref() {
            s.push_str(&format!(",\"error\":\"{}\"", esc_json(e)));
        }
        if let Some(w) = r.warning.as_deref() {
            s.push_str(&format!(",\"warning\":\"{}\"", esc_json(w)));
        }
        s.push('}');
    }
    s.push(']');

    s.push_str(&format!(",\"prefill_toks_per_sec\":{{\"mean\":{:.6},\"min\":{:.6},\"max\":{:.6}}}",
                        prefill_mean, prefill_min, prefill_max));
    s.push_str(&format!(",\"decode_toks_per_sec\":{{\"mean\":{:.6},\"min\":{:.6},\"max\":{:.6}}}",
                        decode_mean, decode_min, decode_max));
    s.push_str(&format!(",\"dispatch_counts\":{{\"forward_total\":{fwd_total},\"inverse_total\":{inv_total}}}"));

    if let Some(e) = cell_error {
        s.push_str(&format!(",\"cell_error\":\"{}\"", esc_json(e)));
    }
    s.push('}');
    s
}

#[cfg(target_os = "android")]
fn emit_cell_failure(
    report_jsonl: &Option<String>,
    cell: u32,
    model_label: &str,
    config_label: &str,
    model_path: &str,
    ntt_attn: bool,
    ntt_attn_hex: bool,
    error: &str,
) {
    let s = format!(
        "{{\"cell\":{cell},\"model_label\":\"{}\",\"config_label\":\"{}\",\"model_path\":\"{}\",\"env\":{{\"SP_ENGINE_NTT_ATTN\":{},\"SP_ENGINE_NTT_ATTN_HEX\":{}}},\"cell_error\":\"{}\"}}",
        esc_json(model_label), esc_json(config_label), esc_json(model_path),
        ntt_attn, ntt_attn_hex, esc_json(error)
    );
    println!("{s}");
    if let Some(path) = report_jsonl.as_deref() {
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
            let _ = f.write_all(format!("{s}\n").as_bytes());
        }
    }
}
