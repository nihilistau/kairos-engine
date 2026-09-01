//! §4-MeMo Sprint M.1 — dual-model budget audit + concurrent invoke + no-
//! interference smoke harness on Sam's S22U.
//!
//! Loads Executive (Qwen3-0.6B) AND Memory (Qwen2.5-Coder-0.5B-Instruct) on
//! the device concurrently via the L1 sp_model_load API. Drives the L1
//! sp_prefill_chunk forward path on BOTH models on TWO ARM threads
//! concurrently and asserts bit-identical output vs each model's solo
//! baseline. Per `reference-fastrpc-concurrent-dispatch`, host-side L1
//! forwards transparently dispatch HVX kernels to the shared cDSP scheduler
//! which engages V69 SSR:XA={4,5} dual vector contexts for cross-thread
//! parallelism. From this harness's PoV we only need to spawn two threads
//! and measure wall-clock + bit-equality.
//!
//! Per `feedback-leak-gate-allocator-warmup`, leak gate is second-half
//! VmRSS slope ≤ 256 KB, NOT total delta.
//!
//! Gates:
//!   T_MEMO_BUDGET_AUDIT      JSON+narrative report; MemAvailable_post ≥ 2048 MB
//!   T_MEMO_DUAL_LOAD         both models load_success=true; combined wall < 30 s
//!   T_MEMO_DUAL_INVOKE       concurrent (exec || memo) outputs == solo baselines
//!   T_MEMO_NO_INTERFERENCE   1000 cycles: drift==0, errors==0,
//!                            second-half VmRSS slope ≤ 256 KB
//!
//! CLI:
//!   sp_memo_m1_smoke <exec_model.spm> <exec_tok.spt> \
//!                    <memo_model.spm> <memo_tok.spt> \
//!                    [--cycles N] [--report-json PATH]
//!
//! On android (target):
//!   adb push sp_memo_m1_smoke /data/local/tmp/
//!   adb push qwen25-coder-0.5b-memory.sp-model /data/local/tmp/
//!   adb push qwen25-coder-0.5b-memory.sp-tokenizer /data/local/tmp/
//!   adb shell chmod +x /data/local/tmp/sp_memo_m1_smoke
//!   adb shell 'ADSP_LIBRARY_PATH="/data/local/tmp;" \
//!     /data/local/tmp/sp_memo_m1_smoke \
//!       /data/local/tmp/qwen3_rt.sp-model \
//!       /data/local/tmp/qwen3_rt.sp-tokenizer \
//!       /data/local/tmp/qwen25-coder-0.5b-memory.sp-model \
//!       /data/local/tmp/qwen25-coder-0.5b-memory.sp-tokenizer \
//!       --report-json /data/local/tmp/m1_report.json'
//!
//! Host build:
//!   cargo build --release --bin sp_memo_m1_smoke
//! (host stub just prints "android-only" and exits 0.)

// Host build stubs everything below to a single eprintln; only android target
// uses the L1 wrappers + meminfo readers + JSON emitter. Silence the unused-
// item warnings that follow on host builds.
#![cfg_attr(not(target_os = "android"), allow(dead_code))]

// §M.1 link discipline: use the lib crate's `ffi_l1` re-export so the math-core
// static libs flow into THIS binary's link closure via the lib crate's symbol
// graph. probe.rs / spec_validate.rs each had their own `mod ffi { include!(...) }`
// and that pattern does NOT propagate the build.rs `rustc-link-lib` directives
// to per-binary link steps on android (cargo treats each bin's link closure
// independently when the bin doesn't depend on the lib).
#[cfg(target_os = "android")]
use sp_daemon::ffi_l1 as ffi;

// Host-only local ffi (the lib re-export is android-cfg-gated; host needs the
// type symbols for the dead-code stubs below to compile).
#[cfg(not(target_os = "android"))]
mod ffi {
    #![allow(non_upper_case_globals, non_camel_case_types, non_snake_case, dead_code, clippy::all)]
    include!(concat!(env!("OUT_DIR"), "/sp_bindings.rs"));
}

use ffi::sp_status_SP_OK;
use std::ffi::CString;
use std::os::raw::c_int;
use std::ptr;
use std::sync::atomic::AtomicI32;
use std::sync::Arc;
use std::time::Instant;

// ─── L1 wrappers (Send+Sync where appropriate) ──────────────────────────────

struct L1Model(*mut ffi::sp_model);
// SAFETY: sp_model is immutable after sp_model_load per L1 ABI.
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
// SAFETY: sp_session is Send but NOT Sync per L1 ABI; one thread at a time.
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

fn prefill(s: &mut L1Session, tokens: &[i32], logits: &mut [f32]) -> Result<(), String> {
    let st = unsafe {
        ffi::sp_prefill_chunk(s.ptr, tokens.as_ptr(), tokens.len(), logits.as_mut_ptr(), logits.len())
    };
    if st != sp_status_SP_OK {
        return Err(format!("sp_prefill_chunk → status={st}"));
    }
    Ok(())
}

// ─── /proc/self/status + /proc/meminfo readers ──────────────────────────────

fn vmrss_kb() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmRSS:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|n| n.parse::<u64>().ok())
        })
        .unwrap_or(0)
}

#[derive(Clone, Default)]
struct MemInfo {
    memtotal_kb: u64,
    memfree_kb: u64,
    memavailable_kb: u64,
}

fn read_meminfo() -> MemInfo {
    let mut mi = MemInfo::default();
    if let Ok(s) = std::fs::read_to_string("/proc/meminfo") {
        for l in s.lines() {
            let mut it = l.split_whitespace();
            let key = it.next().unwrap_or("");
            let val = it.next().and_then(|v| v.parse::<u64>().ok()).unwrap_or(0);
            match key {
                "MemTotal:" => mi.memtotal_kb = val,
                "MemFree:" => mi.memfree_kb = val,
                "MemAvailable:" => mi.memavailable_kb = val,
                _ => {}
            }
        }
    }
    mi
}

// ─── Tiny JSON emitter (avoid serde dep just for this report) ───────────────

struct J(String);
impl J {
    fn new() -> Self {
        J("{".into())
    }
    fn kv_u64(&mut self, k: &str, v: u64) -> &mut Self {
        self.comma_if();
        self.0.push_str(&format!("\"{k}\":{v}"));
        self
    }
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
        self.0.push_str(&format!("\"{k}\":{v}"));
        self
    }
    fn obj(&mut self, k: &str) -> &mut Self {
        self.comma_if();
        self.0.push_str(&format!("\"{k}\":{{"));
        self
    }
    fn end_obj(&mut self) -> &mut Self {
        self.0.push('}');
        self
    }
    fn comma_if(&mut self) {
        let last = self.0.chars().last().unwrap_or('{');
        if last != '{' && last != '[' {
            self.0.push(',');
        }
    }
    fn finish(mut self) -> String {
        self.0.push('}');
        self.0
    }
}

// ─── Host stub (non-android) ────────────────────────────────────────────────
#[cfg(not(target_os = "android"))]
fn main() {
    eprintln!("sp_memo_m1_smoke: host build is a stub — run on android (S22U)");
    eprintln!("  See header doc for adb push + invocation.");
    std::process::exit(0);
}

// ─── Main (android) ─────────────────────────────────────────────────────────
#[cfg(target_os = "android")]
fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!(
            "Usage: {} <exec_model.spm> <exec_tok.spt> <memo_model.spm> <memo_tok.spt> [--cycles N] [--report-json PATH]",
            args.get(0).map(|s| s.as_str()).unwrap_or("sp_memo_m1_smoke")
        );
        std::process::exit(2);
    }
    let exec_model_path = args[1].clone();
    let exec_tok_path = args[2].clone();
    let memo_model_path = args[3].clone();
    let memo_tok_path = args[4].clone();

    let mut cycles: usize = 1000;
    let mut report_json: Option<String> = None;
    let mut i = 5;
    while i < args.len() {
        match args[i].as_str() {
            "--cycles" => {
                cycles = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(1000);
                i += 2;
            }
            "--report-json" => {
                report_json = args.get(i + 1).cloned();
                i += 2;
            }
            other => {
                eprintln!("[M.1] unknown arg: {other}");
                i += 1;
            }
        }
    }

    let mut fails: usize = 0;
    let mut json = J::new();
    json.kv_str("sprint", "M.1");

    // ─── Stage 1: pre-load budget snapshot ───────────────────────────────
    let vmrss_pre = vmrss_kb();
    let meminfo_pre = read_meminfo();
    eprintln!("[M.1] pre-load VmRSS={} KB; MemTotal={} KB MemAvailable={} KB",
              vmrss_pre, meminfo_pre.memtotal_kb, meminfo_pre.memavailable_kb);
    json.obj("device")
        .kv_u64("memtotal_kb", meminfo_pre.memtotal_kb)
        .end_obj();
    json.obj("pre_load")
        .kv_u64("vmrss_kb", vmrss_pre)
        .kv_u64("memtotal_kb", meminfo_pre.memtotal_kb)
        .kv_u64("memfree_kb", meminfo_pre.memfree_kb)
        .kv_u64("memavailable_kb", meminfo_pre.memavailable_kb)
        .end_obj();

    // ─── Stage 2: T_MEMO_DUAL_LOAD — load Executive, then Memory ─────────
    eprintln!("\n[M.1] ═══ T_MEMO_DUAL_LOAD ═══");
    let dual_load_t0 = Instant::now();

    let (exec_model, exec_arch, exec_load_ms) = match load_model(&exec_model_path, &exec_tok_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("[M.1]   Executive load FAIL: {e}");
            // Cannot continue without Executive; emit closure JSON and exit 1.
            json.obj("executive").kv_str("load_error", &e).end_obj();
            json.obj("gates").kv_str("T_MEMO_DUAL_LOAD", "FAIL").end_obj();
            std::fs::write(report_json.as_deref().unwrap_or("/dev/null"), json.finish()).ok();
            std::process::exit(1);
        }
    };
    let vmrss_after_exec = vmrss_kb();
    let meminfo_after_exec = read_meminfo();
    eprintln!("[M.1]   Executive loaded: vocab={} n_layers={} hidden={} load_wall={} ms VmRSS={} KB",
              exec_arch.vocab_size, exec_arch.n_layers, exec_arch.hidden_dim, exec_load_ms, vmrss_after_exec);
    json.obj("executive")
        .kv_str("path", &exec_model_path)
        .kv_u64("load_wall_ms", exec_load_ms as u64)
        .kv_u64("vmrss_after_kb", vmrss_after_exec)
        .kv_u64("vmrss_delta_kb", vmrss_after_exec.saturating_sub(vmrss_pre))
        .kv_u64("memavailable_after_kb", meminfo_after_exec.memavailable_kb)
        .obj("arch")
            .kv_u64("vocab_size", exec_arch.vocab_size as u64)
            .kv_u64("n_layers", exec_arch.n_layers as u64)
            .kv_u64("hidden_dim", exec_arch.hidden_dim as u64)
        .end_obj()
        .end_obj();

    let (memo_model, memo_arch, memo_load_ms) = match load_model(&memo_model_path, &memo_tok_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("[M.1]   Memory load FAIL: {e}");
            json.obj("memory").kv_str("load_error", &e).end_obj();
            json.obj("gates").kv_str("T_MEMO_DUAL_LOAD", "FAIL").end_obj();
            std::fs::write(report_json.as_deref().unwrap_or("/dev/null"), json.finish()).ok();
            std::process::exit(1);
        }
    };
    let vmrss_after_memo = vmrss_kb();
    let meminfo_after_memo = read_meminfo();
    let dual_load_wall_ms = dual_load_t0.elapsed().as_millis();
    eprintln!("[M.1]   Memory loaded: vocab={} n_layers={} hidden={} load_wall={} ms VmRSS={} KB",
              memo_arch.vocab_size, memo_arch.n_layers, memo_arch.hidden_dim, memo_load_ms, vmrss_after_memo);
    eprintln!("[M.1]   combined dual-load wall = {} ms (gate threshold: <30000)", dual_load_wall_ms);

    json.obj("memory")
        .kv_str("path", &memo_model_path)
        .kv_u64("load_wall_ms", memo_load_ms as u64)
        .kv_u64("vmrss_after_kb", vmrss_after_memo)
        .kv_u64("vmrss_delta_kb", vmrss_after_memo.saturating_sub(vmrss_after_exec))
        .kv_u64("memavailable_after_kb", meminfo_after_memo.memavailable_kb)
        .obj("arch")
            .kv_u64("vocab_size", memo_arch.vocab_size as u64)
            .kv_u64("n_layers", memo_arch.n_layers as u64)
            .kv_u64("hidden_dim", memo_arch.hidden_dim as u64)
        .end_obj()
        .end_obj();

    let dual_load_pass = dual_load_wall_ms < 30_000;
    eprintln!("[M.1]   T_MEMO_DUAL_LOAD {}", if dual_load_pass { "PASS" } else { "FAIL" });
    if !dual_load_pass { fails += 1; }

    // ─── Stage 3: T_MEMO_BUDGET_AUDIT ────────────────────────────────────
    eprintln!("\n[M.1] ═══ T_MEMO_BUDGET_AUDIT ═══");
    let total_vmrss_delta_kb = vmrss_after_memo.saturating_sub(vmrss_pre);
    let headroom_mb = meminfo_after_memo.memavailable_kb / 1024;
    let android_os_estimate_mb = (meminfo_pre.memtotal_kb.saturating_sub(meminfo_pre.memavailable_kb)) / 1024;
    eprintln!("[M.1]   total VmRSS delta = {} KB ({:.1} MB)", total_vmrss_delta_kb, total_vmrss_delta_kb as f64 / 1024.0);
    eprintln!("[M.1]   MemAvailable post-load = {} KB ({:.1} MB)", meminfo_after_memo.memavailable_kb, headroom_mb as f64);
    eprintln!("[M.1]   Android OS reservation (pre-load) ≈ {} MB", android_os_estimate_mb);
    eprintln!("[M.1]   Gate threshold: headroom_mb ≥ 2048");

    let budget_pass = headroom_mb >= 2048;
    eprintln!("[M.1]   T_MEMO_BUDGET_AUDIT {}", if budget_pass { "PASS" } else { "FAIL" });
    if !budget_pass { fails += 1; }

    json.obj("post_load")
        .kv_u64("vmrss_kb", vmrss_after_memo)
        .kv_u64("vmrss_total_delta_kb", total_vmrss_delta_kb)
        .kv_u64("memavailable_kb", meminfo_after_memo.memavailable_kb)
        .kv_u64("headroom_mb", headroom_mb)
        .kv_u64("android_os_estimate_mb", android_os_estimate_mb)
        .kv_u64("dual_load_wall_ms", dual_load_wall_ms as u64)
        .end_obj();

    // ─── Stage 4: solo baselines for T_MEMO_DUAL_INVOKE ──────────────────
    eprintln!("\n[M.1] ═══ Solo baselines ═══");
    let exec_vocab = exec_arch.vocab_size as usize;
    let memo_vocab = memo_arch.vocab_size as usize;
    let tokens: Vec<i32> = vec![1, 2, 3];

    let exec_base = match create_session(&exec_model) {
        Ok(s) => s,
        Err(e) => { eprintln!("[M.1] exec base session FAIL: {e}"); std::process::exit(1); }
    };
    let memo_base = match create_session(&memo_model) {
        Ok(s) => s,
        Err(e) => { eprintln!("[M.1] memo base session FAIL: {e}"); std::process::exit(1); }
    };

    // Solo Executive baseline
    let mut exec_solo = match clone_session(&exec_base) {
        Ok(s) => s,
        Err(e) => { eprintln!("[M.1] exec clone FAIL: {e}"); std::process::exit(1); }
    };
    let mut exec_logits = vec![0f32; exec_vocab];
    let t0 = Instant::now();
    if let Err(e) = prefill(&mut exec_solo, &tokens, &mut exec_logits) {
        eprintln!("[M.1] exec solo prefill FAIL: {e}");
        std::process::exit(1);
    }
    let exec_solo_us = t0.elapsed().as_micros();
    let exec_baseline = exec_logits.clone();
    eprintln!("[M.1]   exec solo prefill[1,2,3] = [{:.6}, {:.6}, {:.6}] wall={} μs",
              exec_baseline[0], exec_baseline[1], exec_baseline[2], exec_solo_us);
    drop(exec_solo);

    // Solo Memory baseline
    let mut memo_solo = match clone_session(&memo_base) {
        Ok(s) => s,
        Err(e) => { eprintln!("[M.1] memo clone FAIL: {e}"); std::process::exit(1); }
    };
    let mut memo_logits = vec![0f32; memo_vocab];
    let t0 = Instant::now();
    if let Err(e) = prefill(&mut memo_solo, &tokens, &mut memo_logits) {
        eprintln!("[M.1] memo solo prefill FAIL: {e}");
        std::process::exit(1);
    }
    let memo_solo_us = t0.elapsed().as_micros();
    let memo_baseline = memo_logits.clone();
    eprintln!("[M.1]   memo solo prefill[1,2,3] = [{:.6}, {:.6}, {:.6}] wall={} μs",
              memo_baseline[0], memo_baseline[1], memo_baseline[2], memo_solo_us);
    drop(memo_solo);

    let sequential_wall_us = exec_solo_us + memo_solo_us;
    eprintln!("[M.1]   sequential (exec+memo) wall = {} μs", sequential_wall_us);

    // ─── Stage 5: T_MEMO_DUAL_INVOKE — concurrent ────────────────────────
    eprintln!("\n[M.1] ═══ T_MEMO_DUAL_INVOKE ═══");
    let exec_c = match clone_session(&exec_base) {
        Ok(s) => s,
        Err(e) => { eprintln!("[M.1] exec clone (concurrent) FAIL: {e}"); std::process::exit(1); }
    };
    let memo_c = match clone_session(&memo_base) {
        Ok(s) => s,
        Err(e) => { eprintln!("[M.1] memo clone (concurrent) FAIL: {e}"); std::process::exit(1); }
    };

    let tokens_a = tokens.clone();
    let tokens_b = tokens.clone();
    let exec_v = exec_vocab;
    let memo_v = memo_vocab;

    let conc_start = Instant::now();
    let h_a = std::thread::spawn(move || -> Result<(Vec<f32>, u128), String> {
        let mut s = exec_c;
        let mut logits = vec![0f32; exec_v];
        let t0 = Instant::now();
        prefill(&mut s, &tokens_a, &mut logits)?;
        let us = t0.elapsed().as_micros();
        Ok((logits, us))
    });
    let h_b = std::thread::spawn(move || -> Result<(Vec<f32>, u128), String> {
        let mut s = memo_c;
        let mut logits = vec![0f32; memo_v];
        let t0 = Instant::now();
        prefill(&mut s, &tokens_b, &mut logits)?;
        let us = t0.elapsed().as_micros();
        Ok((logits, us))
    });
    let r_a = h_a.join().expect("thread A");
    let r_b = h_b.join().expect("thread B");
    let conc_wall_us = conc_start.elapsed().as_micros();

    let (exec_conc_logits, exec_conc_us) = match r_a {
        Ok(t) => t,
        Err(e) => { eprintln!("[M.1] exec concurrent FAIL: {e}"); std::process::exit(1); }
    };
    let (memo_conc_logits, memo_conc_us) = match r_b {
        Ok(t) => t,
        Err(e) => { eprintln!("[M.1] memo concurrent FAIL: {e}"); std::process::exit(1); }
    };

    eprintln!("[M.1]   exec concurrent prefill wall = {} μs", exec_conc_us);
    eprintln!("[M.1]   memo concurrent prefill wall = {} μs", memo_conc_us);
    eprintln!("[M.1]   concurrent total wall        = {} μs", conc_wall_us);

    let speedup = sequential_wall_us as f64 / conc_wall_us.max(1) as f64;
    eprintln!("[M.1]   speedup = {:.3}× (sequential / concurrent)", speedup);

    // Bit-identical check
    let exec_match = exec_conc_logits == exec_baseline;
    let memo_match = memo_conc_logits == memo_baseline;
    eprintln!("[M.1]   exec concurrent == solo baseline: {}", if exec_match { "PASS" } else { "FAIL (DRIFT)" });
    eprintln!("[M.1]   memo concurrent == solo baseline: {}", if memo_match { "PASS" } else { "FAIL (DRIFT)" });
    if !exec_match {
        let diff_pos: Vec<usize> = (0..exec_vocab).filter(|&i| exec_conc_logits[i] != exec_baseline[i]).take(5).collect();
        eprintln!("[M.1]   first diverging exec logit indices: {:?}", diff_pos);
        for &p in &diff_pos {
            eprintln!("[M.1]     [{}] solo={} concurrent={}", p, exec_baseline[p], exec_conc_logits[p]);
        }
    }
    if !memo_match {
        let diff_pos: Vec<usize> = (0..memo_vocab).filter(|&i| memo_conc_logits[i] != memo_baseline[i]).take(5).collect();
        eprintln!("[M.1]   first diverging memo logit indices: {:?}", diff_pos);
        for &p in &diff_pos {
            eprintln!("[M.1]     [{}] solo={} concurrent={}", p, memo_baseline[p], memo_conc_logits[p]);
        }
    }

    let dual_invoke_pass = exec_match && memo_match;
    eprintln!("[M.1]   T_MEMO_DUAL_INVOKE {}", if dual_invoke_pass { "PASS" } else { "FAIL" });
    if !dual_invoke_pass { fails += 1; }

    json.obj("dual_invoke")
        .kv_u64("exec_solo_us", exec_solo_us as u64)
        .kv_u64("memo_solo_us", memo_solo_us as u64)
        .kv_u64("sequential_wall_us", sequential_wall_us as u64)
        .kv_u64("exec_conc_us", exec_conc_us as u64)
        .kv_u64("memo_conc_us", memo_conc_us as u64)
        .kv_u64("concurrent_wall_us", conc_wall_us as u64)
        .kv_f64("speedup", speedup)
        .kv_str("exec_match", if exec_match { "true" } else { "false" })
        .kv_str("memo_match", if memo_match { "true" } else { "false" })
        .end_obj();

    // ─── Stage 6: T_MEMO_NO_INTERFERENCE — N-cycle loop ──────────────────
    eprintln!("\n[M.1] ═══ T_MEMO_NO_INTERFERENCE ({} cycles) ═══", cycles);
    eprintln!("[M.1] Per feedback-leak-gate-allocator-warmup: gate metric is");
    eprintln!("[M.1] second-half VmRSS slope ≤ 256 KB, NOT total delta.");

    let cycle_t0 = Instant::now();
    let vmrss_loop_start = vmrss_kb();
    let mut vmrss_mid = vmrss_loop_start;
    let mut executive_drift_count: usize = 0;
    let mut memory_drift_count: usize = 0;
    let mut fastrpc_errors: usize = 0;
    let mut cycles_run: usize = 0;

    let half = cycles / 2;
    for i in 0..cycles {
        let exec_c = match clone_session(&exec_base) {
            Ok(s) => s,
            Err(e) => { eprintln!("[M.1] iter {}: exec clone err: {}", i, e); fastrpc_errors += 1; continue; }
        };
        let memo_c = match clone_session(&memo_base) {
            Ok(s) => s,
            Err(e) => { eprintln!("[M.1] iter {}: memo clone err: {}", i, e); fastrpc_errors += 1; continue; }
        };
        let tokens_a = tokens.clone();
        let tokens_b = tokens.clone();
        let exec_v = exec_vocab;
        let memo_v = memo_vocab;
        let h_a = std::thread::spawn(move || -> Result<Vec<f32>, String> {
            let mut s = exec_c;
            let mut logits = vec![0f32; exec_v];
            prefill(&mut s, &tokens_a, &mut logits)?;
            Ok(logits)
        });
        let h_b = std::thread::spawn(move || -> Result<Vec<f32>, String> {
            let mut s = memo_c;
            let mut logits = vec![0f32; memo_v];
            prefill(&mut s, &tokens_b, &mut logits)?;
            Ok(logits)
        });
        let r_a = h_a.join().expect("iter A");
        let r_b = h_b.join().expect("iter B");
        match (r_a, r_b) {
            (Ok(la), Ok(lb)) => {
                if la != exec_baseline { executive_drift_count += 1; }
                if lb != memo_baseline { memory_drift_count += 1; }
            }
            _ => {
                fastrpc_errors += 1;
            }
        }
        cycles_run += 1;
        if i + 1 == half {
            vmrss_mid = vmrss_kb();
            eprintln!("[M.1]   iter {}: VmRSS = {} KB (mid checkpoint)", i + 1, vmrss_mid);
        }
        if (i + 1) % 100 == 0 {
            eprintln!("[M.1]   iter {}: drift_exec={} drift_memo={} errs={} VmRSS={} KB",
                      i + 1, executive_drift_count, memory_drift_count, fastrpc_errors, vmrss_kb());
        }
    }
    let cycle_wall = cycle_t0.elapsed();
    let vmrss_loop_end = vmrss_kb();
    let first_half_delta_kb = vmrss_mid as i64 - vmrss_loop_start as i64;
    let second_half_delta_kb = vmrss_loop_end as i64 - vmrss_mid as i64;
    let total_delta_kb = vmrss_loop_end as i64 - vmrss_loop_start as i64;

    eprintln!("[M.1]   cycles_run                  = {}", cycles_run);
    eprintln!("[M.1]   executive_drift_count       = {}", executive_drift_count);
    eprintln!("[M.1]   memory_drift_count          = {}", memory_drift_count);
    eprintln!("[M.1]   fastrpc_errors              = {}", fastrpc_errors);
    eprintln!("[M.1]   vmrss_loop_start_kb         = {}", vmrss_loop_start);
    eprintln!("[M.1]   vmrss_loop_mid_kb           = {}", vmrss_mid);
    eprintln!("[M.1]   vmrss_loop_end_kb           = {}", vmrss_loop_end);
    eprintln!("[M.1]   vmrss_first_half_delta_kb   = {}", first_half_delta_kb);
    eprintln!("[M.1]   vmrss_second_half_delta_kb  = {}  (load-bearing — gate ≤256 KB)", second_half_delta_kb);
    eprintln!("[M.1]   vmrss_total_delta_kb        = {}  (diagnostic; allocator warmup expected)", total_delta_kb);
    eprintln!("[M.1]   wall = {:.2} s ({:.2} ms/iter)",
              cycle_wall.as_secs_f64(),
              cycle_wall.as_secs_f64() * 1000.0 / cycles_run.max(1) as f64);

    let no_interference_pass = executive_drift_count == 0
        && memory_drift_count == 0
        && fastrpc_errors == 0
        && second_half_delta_kb.abs() <= 256;
    eprintln!("[M.1]   T_MEMO_NO_INTERFERENCE {}", if no_interference_pass { "PASS" } else { "FAIL" });
    if !no_interference_pass { fails += 1; }

    json.obj("no_interference")
        .kv_u64("cycles_requested", cycles as u64)
        .kv_u64("cycles_run", cycles_run as u64)
        .kv_u64("executive_drift_count", executive_drift_count as u64)
        .kv_u64("memory_drift_count", memory_drift_count as u64)
        .kv_u64("fastrpc_errors", fastrpc_errors as u64)
        .kv_u64("vmrss_loop_start_kb", vmrss_loop_start)
        .kv_u64("vmrss_loop_mid_kb", vmrss_mid)
        .kv_u64("vmrss_loop_end_kb", vmrss_loop_end)
        .kv_f64("vmrss_first_half_delta_kb", first_half_delta_kb as f64)
        .kv_f64("vmrss_second_half_delta_kb", second_half_delta_kb as f64)
        .kv_f64("vmrss_total_delta_kb", total_delta_kb as f64)
        .kv_f64("wall_s", cycle_wall.as_secs_f64())
        .end_obj();

    // ─── Gates summary ────────────────────────────────────────────────────
    json.obj("gates")
        .kv_str("T_MEMO_DUAL_LOAD", if dual_load_pass { "PASS" } else { "FAIL" })
        .kv_str("T_MEMO_BUDGET_AUDIT", if budget_pass { "PASS" } else { "FAIL" })
        .kv_str("T_MEMO_DUAL_INVOKE", if dual_invoke_pass { "PASS" } else { "FAIL" })
        .kv_str("T_MEMO_NO_INTERFERENCE", if no_interference_pass { "PASS" } else { "FAIL" })
        .end_obj();

    // ─── Cleanup ─────────────────────────────────────────────────────────
    drop(exec_base);
    drop(memo_base);
    drop(exec_model);
    drop(memo_model);

    let final_json = json.finish();
    if let Some(path) = report_json.as_deref() {
        if let Err(e) = std::fs::write(path, &final_json) {
            eprintln!("[M.1] WARN: failed to write report-json to {path}: {e}");
        } else {
            eprintln!("[M.1] report JSON written to {path}");
        }
    }

    eprintln!("\n[M.1] ═══ SUMMARY ═══");
    eprintln!("[M.1]   T_MEMO_DUAL_LOAD          {}", if dual_load_pass { "PASS" } else { "FAIL" });
    eprintln!("[M.1]   T_MEMO_BUDGET_AUDIT       {}", if budget_pass { "PASS" } else { "FAIL" });
    eprintln!("[M.1]   T_MEMO_DUAL_INVOKE        {}", if dual_invoke_pass { "PASS" } else { "FAIL" });
    eprintln!("[M.1]   T_MEMO_NO_INTERFERENCE    {}", if no_interference_pass { "PASS" } else { "FAIL" });

    if fails == 0 {
        eprintln!("[M.1] ALL GATES PASS");
        std::process::exit(0);
    } else {
        eprintln!("[M.1] {} gate(s) FAILED", fails);
        std::process::exit(1);
    }
}
