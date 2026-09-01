/// Throwaway FFI probe — verifies sp_session_clone + sp_prefill_chunk +
/// sp_decode_step are sound before wiring async SSE around them.
///
/// Run:
///   cargo run --bin probe -- <model.spm> <tok.spt>
/// or set SP_MODEL_PATH / SP_TOKENIZER_PATH env vars and omit the args.
///
/// Expected output:
///   arch: vocab=... n_layers=...
///   base session created
///   clone OK
///   prefill(3) OK — logits[0..3] = [...]
///   decode(1) OK — position=4, logits[0..3] = [...]
///   PROBE PASS

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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args
        .get(1)
        .cloned()
        .or_else(|| std::env::var("SP_MODEL_PATH").ok())
        .expect("Usage: probe <model.spm> <tok.spt>  (or SP_MODEL_PATH / SP_TOKENIZER_PATH)");
    let tok_path = args
        .get(2)
        .cloned()
        .or_else(|| std::env::var("SP_TOKENIZER_PATH").ok())
        .expect("Usage: probe <model.spm> <tok.spt>");

    // ── 1. Load model ─────────────────────────────────────────────────────────
    let model_c = CString::new(model_path).unwrap();
    let tok_c = CString::new(tok_path).unwrap();
    let mut model_ptr: *mut ffi::sp_model = ptr::null_mut();
    let st = unsafe { ffi::sp_model_load(model_c.as_ptr(), tok_c.as_ptr(), &mut model_ptr) };
    assert_eq!(st, sp_status_SP_OK, "sp_model_load failed");

    // ── 2. Arch info (vocab_size sizes the caller-allocated logits buffer) ────
    let mut arch: ffi::sp_arch_info = unsafe { std::mem::zeroed() };
    let st = unsafe { ffi::sp_model_arch(model_ptr, &mut arch) };
    assert_eq!(st, sp_status_SP_OK, "sp_model_arch failed");
    println!(
        "arch: vocab={} n_layers={} hidden={}",
        arch.vocab_size, arch.n_layers, arch.hidden_dim
    );

    // ── 3. Create base session ─────────────────────────────────────────────────
    let cancel_base = Arc::new(AtomicI32::new(0));
    let cfg = ffi::sp_session_config {
        max_context: 0,
        deterministic: 1,
        arm_bank_kb: 0,
        sieve_capacity: 0,
        flags: 0,
        precision_override: 0,
    };
    let mut base_ptr: *mut ffi::sp_session = ptr::null_mut();
    // SAFETY: Arc keeps cancel_base live for the session's lifetime.
    let base_cancel_raw = cancel_base.as_ptr() as *mut c_int;
    let st = unsafe { ffi::sp_session_create(model_ptr, &cfg, base_cancel_raw, &mut base_ptr) };
    assert_eq!(st, sp_status_SP_OK, "sp_session_create failed");
    println!("base session created");

    // ── 4. Clone → child session ───────────────────────────────────────────────
    let cancel_child = Arc::new(AtomicI32::new(0));
    let child_cancel_raw = cancel_child.as_ptr() as *mut c_int;
    let mut child_ptr: *mut ffi::sp_session = ptr::null_mut();
    let st = unsafe { ffi::sp_session_clone(base_ptr, child_cancel_raw, &mut child_ptr) };
    assert_eq!(st, sp_status_SP_OK, "sp_session_clone failed");
    println!("clone OK");

    // ── 5. Prefill 3 tokens on child ──────────────────────────────────────────
    let tokens: &[i32] = &[1, 2, 3];
    let vocab = arch.vocab_size as usize;
    let mut logits = vec![0.0f32; vocab];
    let st = unsafe {
        ffi::sp_prefill_chunk(
            child_ptr,
            tokens.as_ptr(),
            tokens.len(),
            logits.as_mut_ptr(),
            logits.len(),
        )
    };
    assert_eq!(st, sp_status_SP_OK, "sp_prefill_chunk failed");
    println!("prefill(3) OK — logits[0..3] = {:?}", &logits[..3]);

    // ── 6. Decode one token ────────────────────────────────────────────────────
    let st =
        unsafe { ffi::sp_decode_step(child_ptr, 4, logits.as_mut_ptr(), logits.len()) };
    assert_eq!(st, sp_status_SP_OK, "sp_decode_step failed");

    let mut pos: usize = 0;
    unsafe { ffi::sp_session_position(child_ptr, &mut pos) };
    assert_eq!(pos, 4, "expected position 4 after prefill(3)+decode(1), got {pos}");
    println!("decode(1) OK — position={pos}, logits[0..3] = {:?}", &logits[..3]);

    // ── 7. Cleanup ─────────────────────────────────────────────────────────────
    unsafe { ffi::sp_session_destroy(child_ptr) };
    unsafe { ffi::sp_session_destroy(base_ptr) };
    unsafe { ffi::sp_model_unload(model_ptr) };

    println!("PROBE PASS");
}
