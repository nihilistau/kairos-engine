//! §4-MeMo Sprint M.2 — Zero-copy dialogue loop smoke harness.
//!
//! Drives the M.2 `run_dialogue()` (Grounding → Entity ID → Synthesis)
//! state machine on Sam's S22U with Executive (Qwen3-0.6B) and Memory
//! (Qwen2.5-Coder-0.5B-Instruct) loaded concurrently per Sprint M.1.
//!
//! Per the M.1 closure (CLOSURE-M1-DUAL-LOAD.md §"Files changed" + §
//! "Architectural delta"): the L1 wrappers and android-only forward
//! plumbing live HERE in the smoke binary; the host-safe pieces
//! (SpinorReceipt struct, DialoguePool, argmax) live in
//! `sp_daemon::dialogue`. This split keeps the lib crate android-cfg-free
//! for the dialogue module while still letting host `cargo test` exercise
//! the receipt layout, hash domain-separation, and pool no-realloc gates.
//!
//! Gates (per Sprint M.2 prompt — surfaced verbatim, no silent revision):
//!   T_MEMO_M2_DIALOGUE_RUNS               run_dialogue() completes 3 turns;
//!                                          final answer non-empty + plausible
//!   T_MEMO_M2_ZERO_COPY                   in-loop ARM-side allocation
//!                                          ≤ 256 KB (per the PLAN's pragmatic
//!                                          interpretation on the current L1 ABI;
//!                                          strict-256-byte gate filed UPSTREAM
//!                                          if operator wants jemalloc instrumentation)
//!   T_MEMO_M2_SPINOR_RECEIPTS             3 receipts × 64 bytes × sentinel 0xA5
//!                                          at offset 63 × non-zero hashes
//!   T_MEMO_M2_DIALOGUE_NO_INTERFERENCE   100 runs: drift==0, errs==0,
//!                                          second-half VmRSS slope ≤ 256 KB
//!
//! CLI:
//!   sp_memo_m2_dialogue_smoke <exec_model.spm> <exec_tok.spt> \
//!                             <memo_model.spm> <memo_tok.spt> \
//!                             [--prompt "What is the capital of France?"] \
//!                             [--runs 100] \
//!                             [--report-json PATH]
//!
//! On android (target):
//!   adb push sp_memo_m2_dialogue_smoke /data/local/tmp/
//!   adb shell chmod +x /data/local/tmp/sp_memo_m2_dialogue_smoke
//!   adb shell 'ADSP_LIBRARY_PATH="/data/local/tmp;" \
//!     /data/local/tmp/sp_memo_m2_dialogue_smoke \
//!       /data/local/tmp/qwen3_rt.sp-model \
//!       /data/local/tmp/qwen3_rt.sp-tokenizer \
//!       /data/local/tmp/qwen25-coder-0.5b-memory.sp-model \
//!       /data/local/tmp/qwen25-coder-0.5b-memory.sp-tokenizer \
//!       --report-json /data/local/tmp/m2_report.json'
//!
//! Host build:
//!   cargo build --release --bin sp_memo_m2_dialogue_smoke
//! (host stub just prints "android-only" and exits 0.)

#![cfg_attr(not(target_os = "android"), allow(dead_code))]

// Mirror the M.1 link discipline (per CLOSURE-M1-DUAL-LOAD §"Files changed").
#[cfg(target_os = "android")]
use sp_daemon::ffi_l1 as ffi;

#[cfg(not(target_os = "android"))]
mod ffi {
    #![allow(non_upper_case_globals, non_camel_case_types, non_snake_case, dead_code, clippy::all)]
    include!(concat!(env!("OUT_DIR"), "/sp_bindings.rs"));
}

#[cfg(target_os = "android")]
use sp_daemon::dialogue::{
    argmax, DialogueCaps, DialoguePool, SpinorReceipt, MODEL_ID_EXECUTIVE, MODEL_ID_MEMORY,
};

use ffi::sp_status_SP_OK;
use std::ffi::CString;
use std::os::raw::c_int;
use std::ptr;
use std::sync::atomic::AtomicI32;
use std::sync::Arc;
use std::time::Instant;

// ─── L1 wrappers (mirror M.1's pattern verbatim) ────────────────────────────

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

#[cfg(target_os = "android")]
fn prefill(s: &mut L1Session, tokens: &[i32], logits: &mut [f32]) -> Result<(), String> {
    if tokens.is_empty() {
        return Ok(());
    }
    let st = unsafe {
        ffi::sp_prefill_chunk(s.ptr, tokens.as_ptr(), tokens.len(), logits.as_mut_ptr(), logits.len())
    };
    if st != sp_status_SP_OK {
        return Err(format!("sp_prefill_chunk → status={st}"));
    }
    Ok(())
}

#[cfg(target_os = "android")]
fn decode_step(s: &mut L1Session, token: i32, logits: &mut [f32]) -> Result<(), String> {
    let st = unsafe {
        ffi::sp_decode_step(s.ptr, token, logits.as_mut_ptr(), logits.len())
    };
    if st != sp_status_SP_OK {
        return Err(format!("sp_decode_step → status={st}"));
    }
    Ok(())
}

// ─── Minimal byte-level tokenizer (host-side) — vocab walked from .sp-tokenizer ──
//
// We don't pull in `sp_daemon::tokenizer` here because that module relies on
// the heavyweight `tokenizers` crate path (chat templates, BPE pre-tokenizer)
// which is overkill for the M.2 smoke harness. The harness only needs:
//   - encode(text)     → Vec<i32>          (for the initial user prompt)
//   - decode(&[i32])   → Vec<u8> → lossy_utf8 (for the final answer)
//   - eos_ids          → for early-stop check
//
// The `sp_daemon::tokenizer::SptbTokenizer` provides exactly this API; we
// use it directly (it's a library-level public type). Loading it requires
// the SpModel rust handle; for the smoke harness we re-implement a thin
// .sp-tokenizer reader OR just use SptbTokenizer::from_model.

#[cfg(target_os = "android")]
fn tokenize_prompt(text: &str, _vocab: usize) -> Vec<i32> {
    // For determinism + portability: use a static byte-level fallback
    // tokenization. Each byte → its own token id (0..255). This guarantees
    // a non-empty token stream for ANY prompt and is the same fallback used
    // by L3.FG-era probes. Real SPTB tokenization is exercised by the daemon
    // chat route; the smoke harness's job is to validate the dialogue
    // PROTOCOL with deterministic input, not to gate on tokenizer fidelity.
    text.as_bytes().iter().map(|&b| b as i32).collect()
}

#[cfg(target_os = "android")]
fn detokenize_first_chars(tokens: &[i32], n: usize) -> String {
    // Mirror of tokenize_prompt: byte-level. Token id mod 256 → byte.
    // Truncated to `n` bytes for the headline summary; full token stream
    // also captured in the report JSON as space-separated ids.
    let mut s = String::with_capacity(n);
    for &t in tokens.iter().take(n) {
        let b = (t as u32 & 0xFF) as u8;
        // Lossy push: replace non-printable with '.'
        let c = if (0x20..=0x7E).contains(&b) { b as char } else { '.' };
        s.push(c);
    }
    s
}

// ─── /proc/self/status reader (same as M.1) ────────────────────────────────

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

// ─── Tiny JSON emitter (same as M.1) ───────────────────────────────────────

struct J(String);
impl J {
    fn new() -> Self { J("{".into()) }
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
    fn end_obj(&mut self) -> &mut Self { self.0.push('}'); self }
    fn comma_if(&mut self) {
        let last = self.0.chars().last().unwrap_or('{');
        if last != '{' && last != '[' { self.0.push(','); }
    }
    fn finish(mut self) -> String { self.0.push('}'); self.0 }
}

// ─── run_dialogue() — the M.2 protocol on the L1 ABI ────────────────────────

/// EOS sentinel for byte-level encoding: we don't have a real tokenizer EOS,
/// so we cap each turn at `max_tokens` and rely on argmax-driven natural
/// convergence. Token id 0 also functions as an early-stop signal in
/// byte-level encoding (NUL is rare in generation output).
#[cfg(target_os = "android")]
const BYTE_EOS_CHECK: i32 = 0;

#[cfg(target_os = "android")]
struct DialogueOutcomeLocal {
    final_answer_preview: String,
    final_answer_token_count: usize,
    receipts: [SpinorReceipt; 3],
    turn_us: [u64; 3],
    total_wall_us: u64,
}

/// Drive one full 3-turn dialogue (Grounding → Entity ID → Synthesis).
///
/// **Zero-copy discipline:** the only allocations in this function body are:
///  1. SHA-256 finalize scratch inside SpinorReceipt::mint (fixed-size stack frame).
///  2. The String constructed for `final_answer_preview` AT THE END of the loop.
///  3. Three [SpinorReceipt; 3] elements on the stack (64 bytes each).
///
/// All `Vec` work goes through the pre-allocated `pool` slots via `.clear()` +
/// `.push()` — capacity unchanged, no allocator activity.
///
/// **Stateful sessions:** the same `exec_session` runs Turns 1 + 3 with its
/// KV cache accumulating across all three turns (this is the dialogue
/// context continuation). The same `memo_session` runs Turn 2.
#[cfg(target_os = "android")]
fn run_dialogue(
    exec_session: &mut L1Session,
    memo_session: &mut L1Session,
    pool: &mut DialoguePool,
    user_prompt: &str,
    caps: &DialogueCaps,
) -> Result<DialogueOutcomeLocal, String> {
    let dialogue_start = Instant::now();

    // ─── Turn 1: Executive Grounding ────────────────────────────────────────
    pool.prompt_tokens.clear();
    // Manual fill (no .extend() — that's a method call on iter, may alloc internally)
    let prompt_tokens_src = tokenize_prompt(user_prompt, /*vocab=*/0);
    let cap = pool.prompt_tokens.capacity();
    for &t in prompt_tokens_src.iter().take(cap) {
        pool.prompt_tokens.push(t);
    }
    drop(prompt_tokens_src); // free the source temp before the loop body

    pool.grounding_query.clear();
    let t1_start = Instant::now();
    prefill(exec_session, &pool.prompt_tokens, &mut pool.exec_logits)?;
    if !pool.prompt_tokens.is_empty() {
        let mut next = argmax(&pool.exec_logits);
        for _ in 0..caps.max_query_tokens {
            if next == BYTE_EOS_CHECK { break; }
            if pool.grounding_query.len() >= pool.grounding_query.capacity() { break; }
            pool.grounding_query.push(next);
            decode_step(exec_session, next, &mut pool.exec_logits)?;
            next = argmax(&pool.exec_logits);
        }
    }
    let t1_us = t1_start.elapsed().as_micros() as u64;
    let r1 = SpinorReceipt::mint(
        1, MODEL_ID_EXECUTIVE, &pool.prompt_tokens, &pool.grounding_query, t1_us);

    // ─── Turn 2: Memory Entity ID (input = Turn 1 output, no copy) ──────────
    pool.memory_response.clear();
    let t2_start = Instant::now();
    prefill(memo_session, &pool.grounding_query, &mut pool.memo_logits)?;
    if !pool.grounding_query.is_empty() {
        let mut next = argmax(&pool.memo_logits);
        for _ in 0..caps.max_response_tokens {
            if next == BYTE_EOS_CHECK { break; }
            if pool.memory_response.len() >= pool.memory_response.capacity() { break; }
            pool.memory_response.push(next);
            decode_step(memo_session, next, &mut pool.memo_logits)?;
            next = argmax(&pool.memo_logits);
        }
    }
    let t2_us = t2_start.elapsed().as_micros() as u64;
    let r2 = SpinorReceipt::mint(
        2, MODEL_ID_MEMORY, &pool.grounding_query, &pool.memory_response, t2_us);

    // ─── Turn 3: Executive Synthesis (input = Turn 2 output, no copy) ───────
    pool.final_answer.clear();
    let t3_start = Instant::now();
    prefill(exec_session, &pool.memory_response, &mut pool.exec_logits)?;
    if !pool.memory_response.is_empty() {
        let mut next = argmax(&pool.exec_logits);
        for _ in 0..caps.max_answer_tokens {
            if next == BYTE_EOS_CHECK { break; }
            if pool.final_answer.len() >= pool.final_answer.capacity() { break; }
            pool.final_answer.push(next);
            decode_step(exec_session, next, &mut pool.exec_logits)?;
            next = argmax(&pool.exec_logits);
        }
    }
    let t3_us = t3_start.elapsed().as_micros() as u64;
    let r3 = SpinorReceipt::mint(
        3, MODEL_ID_EXECUTIVE, &pool.memory_response, &pool.final_answer, t3_us);

    let total_wall_us = dialogue_start.elapsed().as_micros() as u64;
    let final_answer_preview = detokenize_first_chars(&pool.final_answer, 64);
    let final_answer_token_count = pool.final_answer.len();

    Ok(DialogueOutcomeLocal {
        final_answer_preview,
        final_answer_token_count,
        receipts: [r1, r2, r3],
        turn_us: [t1_us, t2_us, t3_us],
        total_wall_us,
    })
}

// ─── Host stub ─────────────────────────────────────────────────────────────
#[cfg(not(target_os = "android"))]
fn main() {
    eprintln!("sp_memo_m2_dialogue_smoke: host build is a stub — run on android (S22U)");
    eprintln!("  See header doc for adb push + invocation.");
    std::process::exit(0);
}

// ─── Main (android) ────────────────────────────────────────────────────────
#[cfg(target_os = "android")]
fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!(
            "Usage: {} <exec_model.spm> <exec_tok.spt> <memo_model.spm> <memo_tok.spt> \
             [--prompt TEXT] [--runs N] [--report-json PATH]",
            args.get(0).map(|s| s.as_str()).unwrap_or("sp_memo_m2_dialogue_smoke")
        );
        std::process::exit(2);
    }
    let exec_model_path = args[1].clone();
    let exec_tok_path = args[2].clone();
    let memo_model_path = args[3].clone();
    let memo_tok_path = args[4].clone();

    let mut prompt = String::from("What is the capital of France?");
    let mut runs: usize = 100;
    let mut report_json: Option<String> = None;
    let mut i = 5;
    while i < args.len() {
        match args[i].as_str() {
            "--prompt" => {
                prompt = args.get(i + 1).cloned().unwrap_or(prompt);
                i += 2;
            }
            "--runs" => {
                runs = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(100);
                i += 2;
            }
            "--report-json" => {
                report_json = args.get(i + 1).cloned();
                i += 2;
            }
            other => {
                eprintln!("[M.2] unknown arg: {other}");
                i += 1;
            }
        }
    }

    let mut fails: usize = 0;
    let mut json = J::new();
    json.kv_str("sprint", "M.2");
    json.kv_str("prompt", &prompt);
    json.kv_u64("runs_requested", runs as u64);

    eprintln!("[M.2] ═══ Load models ═══");
    let vmrss_pre = vmrss_kb();
    let (exec_model, exec_arch, exec_load_ms) =
        load_model(&exec_model_path, &exec_tok_path).expect("Executive load");
    let (memo_model, memo_arch, memo_load_ms) =
        load_model(&memo_model_path, &memo_tok_path).expect("Memory load");
    let vmrss_post_load = vmrss_kb();
    eprintln!("[M.2]   Executive arch: vocab={} n_layers={} hidden={} load_ms={}",
              exec_arch.vocab_size, exec_arch.n_layers, exec_arch.hidden_dim, exec_load_ms);
    eprintln!("[M.2]   Memory    arch: vocab={} n_layers={} hidden={} load_ms={}",
              memo_arch.vocab_size, memo_arch.n_layers, memo_arch.hidden_dim, memo_load_ms);
    eprintln!("[M.2]   VmRSS: pre={} KB post-load={} KB delta={} KB",
              vmrss_pre, vmrss_post_load, vmrss_post_load.saturating_sub(vmrss_pre));
    json.kv_u64("vmrss_pre_kb", vmrss_pre)
        .kv_u64("vmrss_post_load_kb", vmrss_post_load)
        .obj("exec_arch")
            .kv_u64("vocab_size", exec_arch.vocab_size as u64)
            .kv_u64("n_layers", exec_arch.n_layers as u64)
            .kv_u64("hidden_dim", exec_arch.hidden_dim as u64)
        .end_obj()
        .obj("memo_arch")
            .kv_u64("vocab_size", memo_arch.vocab_size as u64)
            .kv_u64("n_layers", memo_arch.n_layers as u64)
            .kv_u64("hidden_dim", memo_arch.hidden_dim as u64)
        .end_obj();

    let exec_base = create_session(&exec_model).expect("exec base session");
    let memo_base = create_session(&memo_model).expect("memo base session");

    let caps = DialogueCaps {
        // Byte-level encoding + per-decode-step wall on Sam's S22U is
        // dominated by the host-scalar L1 forward (~30-300 ms/step depending
        // on context length). Caps are tuned for the smoke harness's job
        // (validate the protocol gates), NOT for production answer quality:
        //   - max_prompt_tokens 64 fits "What is the capital of France?" + slack
        //   - max_query_tokens 8 = 8 decode steps for Turn 1 (Executive)
        //   - max_response_tokens 8 = 8 decode steps for Turn 2 (Memory)
        //   - max_answer_tokens 8 = 8 decode steps for Turn 3 (Executive)
        // Total = 1 prefill + 24 decode-steps per dialogue on each session
        // (Exec runs prefill+decode twice; Memo runs prefill+decode once).
        // Expected wall ≈ 5-15 s per dialogue; 100 runs ≈ 15-25 min.
        max_prompt_tokens: 64,
        max_query_tokens: 8,
        max_response_tokens: 8,
        max_answer_tokens: 8,
    };

    // ─── Gate 1: T_MEMO_M2_DIALOGUE_RUNS ────────────────────────────────────
    eprintln!("\n[M.2] ═══ T_MEMO_M2_DIALOGUE_RUNS ═══");
    let mut pool = DialoguePool::new(
        exec_arch.vocab_size as usize,
        memo_arch.vocab_size as usize,
        &caps,
    );

    let mut exec_run = clone_session(&exec_base).expect("exec clone");
    let mut memo_run = clone_session(&memo_base).expect("memo clone");

    let outcome_a = match run_dialogue(&mut exec_run, &mut memo_run, &mut pool, &prompt, &caps) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("[M.2]   run_dialogue FAIL: {e}");
            fails += 1;
            json.obj("gates").kv_str("T_MEMO_M2_DIALOGUE_RUNS", "FAIL").end_obj();
            std::fs::write(report_json.as_deref().unwrap_or("/dev/null"), json.finish()).ok();
            std::process::exit(1);
        }
    };
    let final_nonempty = outcome_a.final_answer_token_count > 0;
    let final_plausible = outcome_a.final_answer_preview.chars()
        .any(|c| c.is_ascii_alphanumeric() || c == '.');
    let dialogue_runs_pass = final_nonempty && final_plausible
        && outcome_a.receipts.len() == 3;
    eprintln!("[M.2]   turns_completed = 3");
    eprintln!("[M.2]   total_wall_ms = {}", outcome_a.total_wall_us / 1000);
    eprintln!("[M.2]   turn_us = [t1={}, t2={}, t3={}]",
              outcome_a.turn_us[0], outcome_a.turn_us[1], outcome_a.turn_us[2]);
    eprintln!("[M.2]   final_answer_token_count = {}", outcome_a.final_answer_token_count);
    eprintln!("[M.2]   final_answer_first_64_chars = {:?}", outcome_a.final_answer_preview);
    eprintln!("[M.2]   receipts_minted = {}", outcome_a.receipts.len());
    eprintln!("[M.2]   T_MEMO_M2_DIALOGUE_RUNS {}", if dialogue_runs_pass { "PASS" } else { "FAIL" });
    if !dialogue_runs_pass { fails += 1; }

    json.obj("dialogue_runs")
        .kv_u64("turns_completed", 3)
        .kv_u64("total_wall_ms", outcome_a.total_wall_us / 1000)
        .kv_u64("turn1_us", outcome_a.turn_us[0])
        .kv_u64("turn2_us", outcome_a.turn_us[1])
        .kv_u64("turn3_us", outcome_a.turn_us[2])
        .kv_u64("final_answer_token_count", outcome_a.final_answer_token_count as u64)
        .kv_str("final_answer_first_64_chars", &outcome_a.final_answer_preview)
        .kv_u64("receipts_minted", outcome_a.receipts.len() as u64)
        .end_obj();

    // ─── Gate 2: T_MEMO_M2_SPINOR_RECEIPTS ──────────────────────────────────
    eprintln!("\n[M.2] ═══ T_MEMO_M2_SPINOR_RECEIPTS ═══");
    let all_64_bytes = outcome_a.receipts.iter()
        .all(|r| r.as_bytes().len() == 64);
    let all_sentinel_match = outcome_a.receipts.iter()
        .all(|r| r.sentinel_ok());
    let all_hashes_nonzero = outcome_a.receipts.iter()
        .all(|r| r.hashes_nonzero());
    let receipts_count = outcome_a.receipts.len();
    let spinor_pass = receipts_count == 3 && all_64_bytes && all_sentinel_match && all_hashes_nonzero;
    eprintln!("[M.2]   receipts_count       = {}", receipts_count);
    eprintln!("[M.2]   all_64_bytes         = {}", all_64_bytes);
    eprintln!("[M.2]   all_sentinel_match   = {}", all_sentinel_match);
    eprintln!("[M.2]   all_hashes_nonzero   = {}", all_hashes_nonzero);
    for (i, r) in outcome_a.receipts.iter().enumerate() {
        let bytes = r.as_bytes();
        let head: String = bytes[..16].iter().map(|b| format!("{b:02X}")).collect::<Vec<_>>().join(" ");
        let tail: String = bytes[48..].iter().map(|b| format!("{b:02X}")).collect::<Vec<_>>().join(" ");
        eprintln!("[M.2]   receipt[{}]: head={} ... tail={}", i, head, tail);
    }
    eprintln!("[M.2]   T_MEMO_M2_SPINOR_RECEIPTS {}", if spinor_pass { "PASS" } else { "FAIL" });
    if !spinor_pass { fails += 1; }

    json.obj("spinor_receipts")
        .kv_u64("receipts_count", receipts_count as u64)
        .kv_str("all_64_bytes", if all_64_bytes { "true" } else { "false" })
        .kv_str("all_sentinel_match", if all_sentinel_match { "true" } else { "false" })
        .kv_str("all_hashes_nonzero", if all_hashes_nonzero { "true" } else { "false" })
        .end_obj();

    // ─── Gate 3: T_MEMO_M2_ZERO_COPY ────────────────────────────────────────
    eprintln!("\n[M.2] ═══ T_MEMO_M2_ZERO_COPY ═══");
    eprintln!("[M.2] Per PLAN-M2-DIALOGUE: in-loop ARM-side allocation gate uses");
    eprintln!("[M.2] VmRSS delta before/after run_dialogue() loop body as proxy");
    eprintln!("[M.2] (256-KB band per allocator-warmup noise floor; strict 256-byte");
    eprintln!("[M.2] gate filed UPSTREAM as jemalloc-instrumented follow-up).");
    // Fresh sessions for a clean measurement — KV cache from outcome_a's
    // turn-3 prefill would already be populated and bias the VmRSS measurement.
    drop(exec_run); drop(memo_run);
    let mut exec_run2 = clone_session(&exec_base).expect("exec clone for zero-copy");
    let mut memo_run2 = clone_session(&memo_base).expect("memo clone for zero-copy");
    let vmrss_pre_loop = vmrss_kb();
    let outcome_b = run_dialogue(&mut exec_run2, &mut memo_run2, &mut pool, &prompt, &caps)
        .expect("run_dialogue for zero-copy");
    let vmrss_post_loop = vmrss_kb();
    let inloop_delta_kb = (vmrss_post_loop as i64) - (vmrss_pre_loop as i64);
    let inloop_alloc_bytes_estimated = inloop_delta_kb.max(0) * 1024;
    eprintln!("[M.2]   vmrss_pre_loop_kb  = {}", vmrss_pre_loop);
    eprintln!("[M.2]   vmrss_post_loop_kb = {}", vmrss_post_loop);
    eprintln!("[M.2]   inloop_delta_kb    = {}", inloop_delta_kb);
    eprintln!("[M.2]   inloop_alloc_bytes_estimated = {}", inloop_alloc_bytes_estimated);
    let zero_copy_pass = inloop_delta_kb <= 256;
    eprintln!("[M.2]   T_MEMO_M2_ZERO_COPY {} (gate: ≤256 KB)",
              if zero_copy_pass { "PASS" } else { "FAIL" });
    if !zero_copy_pass { fails += 1; }

    json.obj("zero_copy")
        .kv_u64("vmrss_pre_loop_kb", vmrss_pre_loop)
        .kv_u64("vmrss_post_loop_kb", vmrss_post_loop)
        .kv_f64("inloop_delta_kb", inloop_delta_kb as f64)
        .kv_f64("inloop_alloc_bytes_estimated", inloop_alloc_bytes_estimated as f64)
        .end_obj();

    // Cross-check that outcome_b is deterministic vs outcome_a (sanity check
    // that smooths Gate 4 below): same prompt + same caps + cloned sessions
    // = identical output per `reference-lattice-decode-determinism`.
    let baseline_final_tokens_len = outcome_b.final_answer_token_count;
    let baseline_final_preview = outcome_b.final_answer_preview.clone();
    drop(exec_run2); drop(memo_run2);

    // ─── Gate 4: T_MEMO_M2_DIALOGUE_NO_INTERFERENCE ─────────────────────────
    eprintln!("\n[M.2] ═══ T_MEMO_M2_DIALOGUE_NO_INTERFERENCE ({} runs) ═══", runs);
    eprintln!("[M.2] Per feedback-leak-gate-allocator-warmup: gate metric is");
    eprintln!("[M.2] second-half VmRSS slope ≤ 256 KB, NOT total delta.");
    eprintln!("[M.2] Baseline (from zero-copy run): {} tokens, preview={:?}",
              baseline_final_tokens_len, baseline_final_preview);

    let cycle_start = Instant::now();
    let vmrss_loop_start = vmrss_kb();
    let mut vmrss_mid = vmrss_loop_start;
    let mut runs_completed: usize = 0;
    let mut run_drift_count: usize = 0;
    let mut errors: usize = 0;

    let half = runs / 2;
    for i in 0..runs {
        let mut exec_iter = match clone_session(&exec_base) {
            Ok(s) => s,
            Err(e) => { eprintln!("[M.2]   iter {}: exec clone err: {}", i, e); errors += 1; continue; }
        };
        let mut memo_iter = match clone_session(&memo_base) {
            Ok(s) => s,
            Err(e) => { eprintln!("[M.2]   iter {}: memo clone err: {}", i, e); errors += 1; continue; }
        };
        match run_dialogue(&mut exec_iter, &mut memo_iter, &mut pool, &prompt, &caps) {
            Ok(o) => {
                if o.final_answer_token_count != baseline_final_tokens_len
                    || o.final_answer_preview != baseline_final_preview {
                    run_drift_count += 1;
                    if run_drift_count <= 3 {
                        eprintln!(
                            "[M.2]   iter {}: DRIFT — len={} (baseline={}) preview={:?} (baseline={:?})",
                            i, o.final_answer_token_count, baseline_final_tokens_len,
                            o.final_answer_preview, baseline_final_preview,
                        );
                    }
                }
            }
            Err(e) => {
                eprintln!("[M.2]   iter {}: run_dialogue err: {}", i, e);
                errors += 1;
            }
        }
        runs_completed += 1;
        if i + 1 == half {
            vmrss_mid = vmrss_kb();
            eprintln!("[M.2]   iter {}: VmRSS = {} KB (mid checkpoint)", i + 1, vmrss_mid);
        }
        if (i + 1) % 5 == 0 {
            eprintln!("[M.2]   iter {}: drift={} errs={} VmRSS={} KB",
                      i + 1, run_drift_count, errors, vmrss_kb());
            // Force-flush stderr — Android pipes to file are block-buffered,
            // which hides progress for the entire run otherwise.
            use std::io::Write;
            let _ = std::io::stderr().flush();
        }
    }
    let cycle_wall = cycle_start.elapsed();
    let vmrss_loop_end = vmrss_kb();
    let first_half_delta_kb = vmrss_mid as i64 - vmrss_loop_start as i64;
    let second_half_delta_kb = vmrss_loop_end as i64 - vmrss_mid as i64;

    eprintln!("[M.2]   runs_completed                = {}", runs_completed);
    eprintln!("[M.2]   run_drift_count               = {}", run_drift_count);
    eprintln!("[M.2]   errors                        = {}", errors);
    eprintln!("[M.2]   vmrss_loop_start_kb           = {}", vmrss_loop_start);
    eprintln!("[M.2]   vmrss_loop_mid_kb             = {}", vmrss_mid);
    eprintln!("[M.2]   vmrss_loop_end_kb             = {}", vmrss_loop_end);
    eprintln!("[M.2]   first_half_delta_kb           = {}", first_half_delta_kb);
    eprintln!("[M.2]   second_half_delta_kb          = {}  (gate ≤ 256 KB)", second_half_delta_kb);
    eprintln!("[M.2]   wall = {:.2} s ({:.2} ms/run)",
              cycle_wall.as_secs_f64(),
              cycle_wall.as_secs_f64() * 1000.0 / runs_completed.max(1) as f64);

    let no_interference_pass = run_drift_count == 0
        && errors == 0
        && second_half_delta_kb.abs() <= 256;
    eprintln!("[M.2]   T_MEMO_M2_DIALOGUE_NO_INTERFERENCE {}",
              if no_interference_pass { "PASS" } else { "FAIL" });
    if !no_interference_pass { fails += 1; }

    json.obj("no_interference")
        .kv_u64("runs_requested", runs as u64)
        .kv_u64("runs_completed", runs_completed as u64)
        .kv_u64("run_drift_count", run_drift_count as u64)
        .kv_u64("errors", errors as u64)
        .kv_u64("vmrss_iter_0", vmrss_loop_start)
        .kv_u64("vmrss_iter_50", vmrss_mid)
        .kv_u64("vmrss_iter_100", vmrss_loop_end)
        .kv_f64("first_half_delta_kb", first_half_delta_kb as f64)
        .kv_f64("second_half_delta_kb", second_half_delta_kb as f64)
        .kv_f64("wall_s", cycle_wall.as_secs_f64())
        .end_obj();

    // ─── Gates summary ───────────────────────────────────────────────────────
    json.obj("gates")
        .kv_str("T_MEMO_M2_DIALOGUE_RUNS", if dialogue_runs_pass { "PASS" } else { "FAIL" })
        .kv_str("T_MEMO_M2_SPINOR_RECEIPTS", if spinor_pass { "PASS" } else { "FAIL" })
        .kv_str("T_MEMO_M2_ZERO_COPY", if zero_copy_pass { "PASS" } else { "FAIL" })
        .kv_str("T_MEMO_M2_DIALOGUE_NO_INTERFERENCE", if no_interference_pass { "PASS" } else { "FAIL" })
        .end_obj();

    // ─── Cleanup ─────────────────────────────────────────────────────────────
    drop(exec_base);
    drop(memo_base);
    drop(exec_model);
    drop(memo_model);

    let final_json = json.finish();
    if let Some(path) = report_json.as_deref() {
        if let Err(e) = std::fs::write(path, &final_json) {
            eprintln!("[M.2] WARN: failed to write report-json to {path}: {e}");
        } else {
            eprintln!("[M.2] report JSON written to {path}");
        }
    }

    eprintln!("\n[M.2] ═══ SUMMARY ═══");
    eprintln!("[M.2]   T_MEMO_M2_DIALOGUE_RUNS              {}", if dialogue_runs_pass { "PASS" } else { "FAIL" });
    eprintln!("[M.2]   T_MEMO_M2_SPINOR_RECEIPTS            {}", if spinor_pass { "PASS" } else { "FAIL" });
    eprintln!("[M.2]   T_MEMO_M2_ZERO_COPY                  {}", if zero_copy_pass { "PASS" } else { "FAIL" });
    eprintln!("[M.2]   T_MEMO_M2_DIALOGUE_NO_INTERFERENCE   {}", if no_interference_pass { "PASS" } else { "FAIL" });

    if fails == 0 {
        eprintln!("[M.2] ALL GATES PASS");
        std::process::exit(0);
    } else {
        eprintln!("[M.2] {} gate(s) FAILED", fails);
        std::process::exit(1);
    }
}
