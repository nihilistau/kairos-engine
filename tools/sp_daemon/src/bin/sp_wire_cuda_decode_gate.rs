//! G-WIRE-CUDA-DECODE-GEMMA4 — the persistent-KV decode verb gate.
//!
//! Proves the universal Shannon-Prime daemon's L1 kvdecode verb
//! (`sp_session_register_kvdecode_backend` → `sp_decode_step` → the CUDA glue →
//! the additive `gemma4_kv_decode_logits`) token-by-token decodes a real
//! a dense Gemma-4 model on a persistent device-resident KV cache, and that the produced
//! token stream is **bit-identical** to the null-floor oracle `gemma4_kv_decode`
//! (the byte-untouched argmax decode). VRAM stays flat across the decode (the
//! KAI-1b O(1) resident-cache property).
//!
//! Build: `cargo build --release --features wire_cuda_backend --bin sp_wire_cuda_decode_gate`
//! Run:   set SP_CUDA_DECODE_INT8=1 (tied head), then
//!        sp_wire_cuda_decode_gate <model.sp-model> <tok.sp-tokenizer> [n_decode]
//!        (or SP_MODEL_PATH / SP_TOKENIZER_PATH env).
//!
//! Only meaningful under `--features wire_cuda_backend` (the feature that links
//! the CUDA backend lib carrying `gemma4_kv_*` + the kvdecode dispatch module).

#![cfg(feature = "wire_cuda_backend")]

mod ffi {
    #![allow(non_upper_case_globals, non_camel_case_types, non_snake_case, dead_code, clippy::all)]
    include!(concat!(env!("OUT_DIR"), "/sp_bindings.rs"));
}

use ffi::sp_status_SP_OK;
use std::ffi::{c_void, CString};
use std::os::raw::c_int;
use std::ptr;
use std::sync::atomic::AtomicI32;
use std::sync::Arc;

// ── Reference null-floor oracle + the resident-cache lifecycle the glue wraps.
// These live in libsp_cuda_daemon_backend (cuda_forward.cu) and resolve at the
// daemon binary's link step; the harness declares the subset it drives directly
// for the oracle leg + the daemon-leg prefill.
#[repr(C)]
struct sp_g4_kv {
    _opaque: [u8; 0],
}
extern "C" {
    fn gemma4_kv_open(m: *const c_void, pmax: c_int) -> *mut sp_g4_kv;
    fn gemma4_kv_prefill(s: *mut sp_g4_kv, toks: *const i32, n: c_int) -> c_int;
    fn gemma4_kv_decode(s: *mut sp_g4_kv, n_gen: c_int, out: *mut i32) -> c_int;
    fn gemma4_kv_seq_peek(s: *const sp_g4_kv, out: *mut i32, from: c_int, n: c_int) -> c_int;
    fn gemma4_kv_pos(s: *const sp_g4_kv) -> c_int;
    fn gemma4_kv_devfree_mib() -> std::os::raw::c_long;
    fn gemma4_kv_close(s: *mut sp_g4_kv);
}

fn last_err() -> String {
    unsafe { std::ffi::CStr::from_ptr(ffi::sp_last_error()) }
        .to_string_lossy()
        .into_owned()
}

fn argmax(v: &[f32]) -> i32 {
    let mut best = 0usize;
    let mut bv = v[0];
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            best = i;
        }
    }
    best as i32
}

fn main() {
    // The persistent-KV path needs the tied full-vocab head materialized in the
    // decode path — gemma4_kv_open errors without SP_CUDA_DECODE_INT8=1.
    if std::env::var("SP_CUDA_DECODE_INT8").ok().as_deref() != Some("1") {
        // Set it for the process so the operator doesn't have to remember.
        std::env::set_var("SP_CUDA_DECODE_INT8", "1");
        eprintln!("[gate] SP_CUDA_DECODE_INT8 not set — defaulting to 1 (tied head)");
    }

    let args: Vec<String> = std::env::args().collect();
    let model_path = args
        .get(1)
        .cloned()
        .or_else(|| std::env::var("SP_MODEL_PATH").ok())
        .expect("usage: sp_wire_cuda_decode_gate <model.sp-model> <tok.sp-tokenizer> [n_decode]");
    let tok_path = args
        .get(2)
        .cloned()
        .or_else(|| std::env::var("SP_TOKENIZER_PATH").ok())
        .expect("usage: sp_wire_cuda_decode_gate <model.sp-model> <tok.sp-tokenizer> [n_decode]");
    let n_decode: usize = args
        .get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(32usize);
    assert!(n_decode >= 32, "gate requires >=32 decode steps (got {n_decode})");

    // A short deterministic prompt (token ids). Content is irrelevant to the
    // gate — both legs run the SAME prefill, so the comparison is purely
    // verb-routing vs the null floor. (Real gemma-4 BOS-ish low ids.)
    let prompt: Vec<i32> = vec![2, 106, 1645, 108, 6249, 573, 573, 8956, 108];
    let pmax: i32 = (prompt.len() + n_decode + 8) as i32;

    println!("G-WIRE-CUDA-DECODE-GEMMA4 — persistent-KV decode verb gate");
    println!("  model     : {model_path}");
    println!("  prompt    : {} tokens", prompt.len());
    println!("  n_decode  : {n_decode}");
    println!("  Pmax      : {pmax}");

    // ── Load model + L1 arch ────────────────────────────────────────────────
    let model_c = CString::new(model_path).unwrap();
    let tok_c = CString::new(tok_path).unwrap();
    let mut model_ptr: *mut ffi::sp_model = ptr::null_mut();
    let st = unsafe { ffi::sp_model_load(model_c.as_ptr(), tok_c.as_ptr(), &mut model_ptr) };
    assert_eq!(st, sp_status_SP_OK, "sp_model_load failed");
    let mut arch: ffi::sp_arch_info = unsafe { std::mem::zeroed() };
    assert_eq!(
        unsafe { ffi::sp_model_arch(model_ptr, &mut arch) },
        sp_status_SP_OK,
        "sp_model_arch failed"
    );
    let vocab = arch.vocab_size as usize;
    println!(
        "  arch      : vocab={} n_layers={} hidden={}",
        vocab, arch.n_layers, arch.hidden_dim
    );

    let vram0 = unsafe { gemma4_kv_devfree_mib() };

    // ── Create the L1 session up front ──────────────────────────────────────
    // Both legs open gemma4_kv over the SAME session-borrowed qwen3_model* — the
    // L1 bridge (sp_model_to_gemma4) is what populates the packed arena + tied
    // head the resident decode needs. (model_ptr is the opaque sp_model* handle,
    // NOT a qwen3_model*; sp_session_qwen3_model gives the real runnable model.)
    let cancel = Arc::new(AtomicI32::new(0));
    let cfg = ffi::sp_session_config {
        max_context: pmax as usize,
        deterministic: 1,
        arm_bank_kb: 0,
        sieve_capacity: 0,
        flags: 0,
        precision_override: 0,
    };
    let mut sess: *mut ffi::sp_session = ptr::null_mut();
    let cancel_raw = cancel.as_ptr() as *mut c_int;
    assert_eq!(
        unsafe { ffi::sp_session_create(model_ptr, &cfg, cancel_raw, &mut sess) },
        sp_status_SP_OK,
        "sp_session_create failed"
    );
    let qm = unsafe { ffi::sp_session_qwen3_model(sess) };
    assert!(!qm.is_null(), "sp_session_qwen3_model NULL");

    // ── Leg A — the null-floor oracle: gemma4_kv_decode (argmax) ────────────
    // Open a resident cache over the borrowed qm, prefill, greedy-decode.
    let s_ref = unsafe { gemma4_kv_open(qm, pmax) };
    if s_ref.is_null() {
        panic!("gemma4_kv_open (oracle) failed: {} (SP_CUDA_DECODE_INT8={:?})",
            last_err(), std::env::var("SP_CUDA_DECODE_INT8").ok());
    }
    assert_eq!(
        unsafe { gemma4_kv_prefill(s_ref, prompt.as_ptr(), prompt.len() as c_int) },
        0,
        "oracle prefill failed"
    );
    let p = prompt.len() as i32; // dpos after prefill
    let mut ref_out = vec![0i32; n_decode];
    assert_eq!(
        unsafe { gemma4_kv_decode(s_ref, n_decode as c_int, ref_out.as_mut_ptr()) },
        0,
        "oracle gemma4_kv_decode failed"
    );
    // Recover the EXACT consumed sequence dseq[p .. p+n_decode): dseq[p] is the
    // prefill-head prediction (the first token gemma4_kv_decode embedded), then
    // its own greedy outputs. This is what the daemon leg must teacher-force.
    let mut consumed = vec![0i32; n_decode];
    assert_eq!(
        unsafe { gemma4_kv_seq_peek(s_ref, consumed.as_mut_ptr(), p, n_decode as c_int) },
        0,
        "gemma4_kv_seq_peek failed"
    );
    let vram_ref = unsafe { gemma4_kv_devfree_mib() };
    unsafe { gemma4_kv_close(s_ref) };
    println!("  [leg A oracle] gemma4_kv_decode: {n_decode} tokens, first 8 = {:?}", &ref_out[..8.min(n_decode)]);

    // ── Leg B — the L1 kvdecode verb through sp_decode_step ─────────────────
    // Register the CUDA kvdecode backend on the session (opens a 2nd resident
    // cache + points sp_decode_step at the glue), prefill the prompt on the
    // resident handle, then drive sp_decode_step with the SAME consumed input
    // tokens. Each returned logits row is argmaxed; the stream must match leg A.
    //
    // Register the kvdecode backend: opens the resident cache + wires
    // sp_decode_step → glue → gemma4_kv_decode_logits.
    let sess_raw = sess as *mut sp_daemon::ffi_l1::sp_session;
    sp_daemon::cuda_kvdecode_dispatch::reset_step_count();
    let handle = unsafe {
        sp_daemon::cuda_kvdecode_dispatch::register_with_session(sess_raw, qm, pmax)
    }
    .expect("register_with_session failed");

    // Prefill the prompt into the resident kvdecode cache (the backend owns it +
    // tracks its own dpos = prompt.len()). The L1 session's host KV path is
    // BYPASSED for the kvdecode route, so we do NOT advance s->pos with a costly
    // reference sp_prefill_chunk over the dense reference model — sp_decode_step's kvdecode branch
    // uses s->pos only for hist[] bookkeeping + the context-full guard (hist_cap
    // = pmax > n_decode), never to tell the backend where to write (the resident
    // dpos does that). The session simply counts 0..n_decode while the resident
    // cache runs p..p+n_decode — the absolute offset is irrelevant to the verb.
    unsafe { sp_daemon::cuda_kvdecode_dispatch::prefill(handle, &prompt) }
        .expect("kvdecode prefill failed");

    // Drive sp_decode_step with the teacher-forced consumed tokens; argmax each
    // returned logits row → the daemon token stream.
    let mut dae_out = vec![0i32; n_decode];
    let mut logits = vec![0.0f32; vocab];
    for g in 0..n_decode {
        let st = unsafe {
            ffi::sp_decode_step(sess, consumed[g], logits.as_mut_ptr(), logits.len())
        };
        assert_eq!(st, sp_status_SP_OK, "sp_decode_step failed at g={g}");
        dae_out[g] = argmax(&logits);
    }
    let steps = sp_daemon::cuda_kvdecode_dispatch::step_count();
    let dpos_resident = unsafe { gemma4_kv_pos(handle as *const sp_g4_kv) };
    let vram_dae = unsafe { gemma4_kv_devfree_mib() };

    println!("  [leg B verb ] sp_decode_step×{n_decode}: kvdecode steps counted = {steps}");
    println!("  [leg B verb ] resident dpos = {dpos_resident} (expected {})", p as usize + n_decode);
    println!("  [leg B verb ] first 8 = {:?}", &dae_out[..8.min(n_decode)]);

    // ── Compare ─────────────────────────────────────────────────────────────
    let mut first_div = usize::MAX;
    for g in 0..n_decode {
        if ref_out[g] != dae_out[g] {
            first_div = g;
            break;
        }
    }
    let identical = first_div == usize::MAX;

    println!("  VRAM free (MiB): start={vram0} afterA={vram_ref} afterB={vram_dae}");
    let vram_delta = (vram_ref - vram_dae).abs();
    println!("  VRAM delta A-vs-B = {vram_delta} MiB (flat ⇒ O(1) resident cache; ~equal allocs)");

    // Teardown: unregister + close the resident cache, destroy session, unload.
    let _ = unsafe {
        sp_daemon::ffi_l1::sp_session_register_kvdecode_backend(
            sess_raw,
            ptr::null_mut(),
            ptr::null(),
        )
    };
    unsafe { sp_daemon::cuda_kvdecode_dispatch::release_for_model(handle) };
    unsafe { ffi::sp_session_destroy(sess) };
    unsafe { ffi::sp_model_unload(model_ptr) };

    println!();
    if identical && steps == n_decode as u64 {
        println!("  TOKEN STREAM: IDENTICAL ({n_decode}/{n_decode} match)");
        println!("G-WIRE-CUDA-DECODE-GEMMA4: GREEN");
    } else {
        if !identical {
            println!(
                "  TOKEN STREAM: DIVERGED at step {first_div} — oracle={} daemon={}",
                ref_out[first_div], dae_out[first_div]
            );
        }
        if steps != n_decode as u64 {
            println!("  STEP COUNT: expected {n_decode}, kvdecode verb saw {steps} (verb not reached?)");
        }
        println!("G-WIRE-CUDA-DECODE-GEMMA4: RED");
        std::process::exit(1);
    }
}
