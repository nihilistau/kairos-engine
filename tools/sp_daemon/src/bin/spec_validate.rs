/// spec_validate — validates Corollary T8.1 (no ghost contamination after rewind)
/// and the discrete accept/reject spec loop.
///
/// Run:
///   cargo run --bin spec_validate -- <target.spm> <target.spt> <draft.spm> <draft.spt>
///
/// Gates:
///   M_SPEC_1: bit-identity after rewind + planted acceptance rate output correctness
///   M_SPEC_2: sp_session_position tracking across accept/reject cycles

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

// ── FFI helpers ──────────────────────────────────────────────────────────────

fn load_model(model_path: &str, tok_path: &str) -> *mut ffi::sp_model {
    let model_c = CString::new(model_path).unwrap();
    let tok_c = CString::new(tok_path).unwrap();
    let mut ptr: *mut ffi::sp_model = ptr::null_mut();
    let st = unsafe { ffi::sp_model_load(model_c.as_ptr(), tok_c.as_ptr(), &mut ptr) };
    assert_eq!(st, sp_status_SP_OK, "sp_model_load failed for {model_path}");
    ptr
}

fn arch(model: *mut ffi::sp_model) -> ffi::sp_arch_info {
    let mut info: ffi::sp_arch_info = unsafe { std::mem::zeroed() };
    let st = unsafe { ffi::sp_model_arch(model, &mut info) };
    assert_eq!(st, sp_status_SP_OK, "sp_model_arch failed");
    info
}

fn create_session(model: *const ffi::sp_model, cancel: &Arc<AtomicI32>) -> *mut ffi::sp_session {
    let cfg = ffi::sp_session_config {
        max_context: 0,
        deterministic: 1,
        arm_bank_kb: 0,
        sieve_capacity: 0,
        flags: 0,
        precision_override: 0,
    };
    let mut ptr: *mut ffi::sp_session = ptr::null_mut();
    let raw = cancel.as_ptr() as *mut c_int;
    let st = unsafe { ffi::sp_session_create(model, &cfg, raw, &mut ptr) };
    assert_eq!(st, sp_status_SP_OK, "sp_session_create failed");
    ptr
}

fn clone_session(src: *const ffi::sp_session, cancel: &Arc<AtomicI32>) -> *mut ffi::sp_session {
    let mut out: *mut ffi::sp_session = ptr::null_mut();
    let raw = cancel.as_ptr() as *mut c_int;
    let st = unsafe { ffi::sp_session_clone(src, raw, &mut out) };
    assert_eq!(st, sp_status_SP_OK, "sp_session_clone failed");
    out
}

fn prefill(session: *mut ffi::sp_session, tokens: &[i32], logits: &mut [f32]) {
    let st = unsafe {
        ffi::sp_prefill_chunk(session, tokens.as_ptr(), tokens.len(), logits.as_mut_ptr(), logits.len())
    };
    assert_eq!(st, sp_status_SP_OK, "sp_prefill_chunk failed");
}

fn decode(session: *mut ffi::sp_session, token: i32, logits: &mut [f32]) {
    let st = unsafe { ffi::sp_decode_step(session, token, logits.as_mut_ptr(), logits.len()) };
    assert_eq!(st, sp_status_SP_OK, "sp_decode_step({token}) failed");
}

fn rewind(session: *mut ffi::sp_session, n: usize) {
    let st = unsafe { ffi::sp_session_rewind(session, n) };
    assert_eq!(st, sp_status_SP_OK, "sp_session_rewind({n}) failed");
}

fn position(session: *const ffi::sp_session) -> usize {
    let mut pos: usize = 0;
    unsafe { ffi::sp_session_position(session, &mut pos) };
    pos
}

fn destroy(session: *mut ffi::sp_session) {
    unsafe { ffi::sp_session_destroy(session) };
}

fn argmax(logits: &[f32]) -> i32 {
    logits.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as i32)
        .unwrap_or(0)
}

fn logits_bits_match(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| x.to_bits() == y.to_bits())
}

// ── Fixture ───────────────────────────────────────────────────────────────────

const K: usize = 4;
const PREFILL_TOKENS: &[i32] = &[1, 100, 200, 300, 400, 500, 600, 700];

// ── Protocol A+B: planted acceptance rates + T8.1 rewind identity ─────────────

fn protocol_ab(
    target_model: *mut ffi::sp_model,
    draft_model: *mut ffi::sp_model,
    target_vocab: usize,
    draft_vocab: usize,
) {
    println!("=== Protocol A+B: planted acceptance rates + T8.1 rewind identity ===");
    let acceptance_configs: &[(&str, Box<dyn Fn(usize, i32) -> i32>)] = &[
        ("100%", Box::new(|_, correct| correct)),
        ("0%",   Box::new(|_, correct| (correct + 1) % 150000)),
        ("50%",  Box::new(|step, correct| if step % 2 == 0 { correct } else { (correct + 1) % 150000 })),
        ("25%",  Box::new(|step, correct| if step % 4 == 0 { correct } else { (correct + 1) % 150000 })),
        ("75%",  Box::new(|step, correct| if step % 4 != 0 { correct } else { (correct + 1) % 150000 })),
    ];

    for (rate_name, draft_fn) in acceptance_configs {
        println!("  rate={rate_name}");
        let t_cancel = Arc::new(AtomicI32::new(0));
        let d_cancel = Arc::new(AtomicI32::new(0));
        let ref_cancel = Arc::new(AtomicI32::new(0));

        let target = create_session(target_model, &t_cancel);
        let draft  = create_session(draft_model,  &d_cancel);

        let mut t_logits = vec![0f32; target_vocab];
        let mut d_logits = vec![0f32; draft_vocab];
        prefill(target, PREFILL_TOKENS, &mut t_logits);
        prefill(draft,  PREFILL_TOKENS, &mut d_logits);

        let mut total_accepted = 0usize;

        for step in 0..100 {
            let p_before = position(target);
            assert_eq!(p_before, PREFILL_TOKENS.len() + total_accepted,
                "M_SPEC_2: target position mismatch at step {step}");

            let correct_tok = argmax(&t_logits);

            let mut draft_tokens: Vec<i32> = Vec::with_capacity(K);
            for ki in 0..K {
                draft_tokens.push(draft_fn(step * K + ki, correct_tok));
            }

            let draft_pos_before = position(draft);
            // Clone draft at P BEFORE overshoot — the never-overshot T8.1 reference.
            let ref_at_p = clone_session(draft, &ref_cancel);
            for ki in 0..(K - 1) {
                decode(draft, draft_tokens[ki], &mut d_logits);
            }
            assert_eq!(position(draft), draft_pos_before + K - 1,
                "M_SPEC_2: draft position after K-1 steps wrong at step {step}");

            // Target verifies sequentially.
            let mut first_reject: Option<usize> = None;
            for ki in 0..K {
                let target_pick = argmax(&t_logits);
                if target_pick == draft_tokens[ki] {
                    decode(target, draft_tokens[ki], &mut t_logits);
                    total_accepted += 1;
                } else {
                    first_reject = Some(ki);
                    break;
                }
            }

            if let Some(ki) = first_reject {
                let rewind_by = K - 1 - ki;

                assert_eq!(position(draft), draft_pos_before + K - 1,
                    "M_SPEC_2: draft should be at P+K-1 before rewind at step {step}");

                if rewind_by > 0 {
                    // True T8.1 test: rewind draft to P+ki; advance ref_at_p by ki steps.
                    rewind(draft, rewind_by);
                    assert_eq!(position(draft), draft_pos_before + ki,
                        "M_SPEC_2: draft position after rewind should be P+ki at step {step}");

                    let mut ref_logits = vec![0f32; draft_vocab];
                    for m in 0..ki {
                        decode(ref_at_p, draft_tokens[m], &mut ref_logits);
                    }

                    // Probe decode both at P+ki with same fixed token — must produce
                    // bit-identical logits at P+ki+1 (T8.1 claim).
                    let mut draft_probe_logits = vec![0f32; draft_vocab];
                    let probe_tok = 1i32;
                    decode(draft,    probe_tok, &mut draft_probe_logits);
                    decode(ref_at_p, probe_tok, &mut ref_logits);
                    assert!(logits_bits_match(&draft_probe_logits, &ref_logits),
                        "M_SPEC_1 FAIL (T8.1): logits after rewind differ from reference \
                         at step {step} ki={ki} rewind_by={rewind_by}");

                    // Restore both probe steps back to P+ki.
                    rewind(draft,    1);
                    rewind(ref_at_p, 1);
                    assert_eq!(position(draft), draft_pos_before + ki,
                        "M_SPEC_2: draft position after T8.1 probe restore at step {step}");
                }
            } else {
                // All K accepted: sync draft to P+K.
                decode(draft, draft_tokens[K - 1], &mut d_logits);
                assert_eq!(position(draft), draft_pos_before + K,
                    "M_SPEC_2: draft should be at P+K after full acceptance at step {step}");
            }

            destroy(ref_at_p);

            let final_target_pos = position(target);
            let final_draft_pos  = position(draft);
            assert_eq!(final_target_pos, final_draft_pos,
                "M_SPEC_2: target/draft position mismatch after step {step}: \
                 target={final_target_pos} draft={final_draft_pos}");
        }

        println!("    rate={rate_name} PASS (100 steps, {total_accepted} accepted)");
        destroy(target);
        destroy(draft);
    }
    println!("Protocol A+B: PASS");
}

// ── Protocol C: 500-token natural soak ────────────────────────────────────────

fn protocol_c(
    target_model: *mut ffi::sp_model,
    draft_model: *mut ffi::sp_model,
    target_vocab: usize,
    draft_vocab: usize,
) {
    println!("=== Protocol C: 500-token natural soak ===");
    let t_cancel = Arc::new(AtomicI32::new(0));
    let d_cancel = Arc::new(AtomicI32::new(0));
    let target = create_session(target_model, &t_cancel);
    let draft  = create_session(draft_model,  &d_cancel);

    let mut t_logits = vec![0f32; target_vocab];
    let mut d_logits = vec![0f32; draft_vocab];
    prefill(target, PREFILL_TOKENS, &mut t_logits);
    prefill(draft,  PREFILL_TOKENS, &mut d_logits);

    // Pre-pass: record what pure greedy target would emit — M_SPEC_1 reference.
    let ref_cancel = Arc::new(AtomicI32::new(0));
    let ref_target = create_session(target_model, &ref_cancel);
    let mut ref_logits_pre = vec![0f32; target_vocab];
    prefill(ref_target, PREFILL_TOKENS, &mut ref_logits_pre);
    let mut reference_tokens: Vec<i32> = Vec::with_capacity(500);
    for _ in 0..500 {
        let tok = argmax(&ref_logits_pre);
        reference_tokens.push(tok);
        decode(ref_target, tok, &mut ref_logits_pre);
    }
    destroy(ref_target);
    println!("  reference pre-pass: 500 greedy tokens recorded");

    let mut total_output = 0usize;
    let mut total_accepted = 0usize;

    while total_output < 500 {
        let mut draft_tokens: Vec<i32> = Vec::with_capacity(K);
        draft_tokens.push(argmax(&d_logits));
        for _ in 1..K {
            let last = *draft_tokens.last().unwrap();
            decode(draft, last, &mut d_logits);
            draft_tokens.push(argmax(&d_logits));
        }
        // draft is at P+K-1.

        let mut step_accepted = 0usize;
        for ki in 0..K {
            let target_pick = argmax(&t_logits);
            if target_pick == draft_tokens[ki] {
                assert_eq!(draft_tokens[ki], reference_tokens[total_output],
                    "M_SPEC_1: accepted token {} differs from greedy reference {} at output #{}",
                    draft_tokens[ki], reference_tokens[total_output], total_output);
                decode(target, draft_tokens[ki], &mut t_logits);
                step_accepted += 1;
                total_accepted += 1;
                total_output += 1;
            } else {
                let rewind_by = K - 1 - ki;
                if rewind_by > 0 { rewind(draft, rewind_by); }
                let corrected = target_pick;
                assert_eq!(corrected, reference_tokens[total_output],
                    "M_SPEC_1: corrected token {} differs from greedy reference {} at output #{}",
                    corrected, reference_tokens[total_output], total_output);
                decode(target, corrected, &mut t_logits);
                decode(draft,  corrected, &mut d_logits);
                total_output += 1;
                break;
            }
        }
        if step_accepted == K {
            decode(draft, draft_tokens[K - 1], &mut d_logits);
        }

        let tp = position(target);
        let dp = position(draft);
        assert_eq!(tp, dp,
            "M_SPEC_2: position mismatch after output #{total_output}: target={tp} draft={dp}");
    }

    let accept_rate = total_accepted as f64 / 500.0 * 100.0;
    println!("  500 tokens output, {total_accepted} accepted ({accept_rate:.1}%), PASS");
    destroy(target);
    destroy(draft);
    println!("Protocol C: PASS");
}

// ── Protocol C-Synth: forced-rejection soak (covers rewind with same-model fixture) ──

fn protocol_c_synth(
    target_model: *mut ffi::sp_model,
    draft_model: *mut ffi::sp_model,
    target_vocab: usize,
    draft_vocab: usize,
) {
    println!("=== Protocol C-Synth: 200-token soak, forced rejection every 3rd batch ===");
    let t_cancel = Arc::new(AtomicI32::new(0));
    let d_cancel = Arc::new(AtomicI32::new(0));
    let target = create_session(target_model, &t_cancel);
    let draft  = create_session(draft_model,  &d_cancel);

    let mut t_logits = vec![0f32; target_vocab];
    let mut d_logits = vec![0f32; draft_vocab];
    prefill(target, PREFILL_TOKENS, &mut t_logits);
    prefill(draft,  PREFILL_TOKENS, &mut d_logits);

    let ref_cancel = Arc::new(AtomicI32::new(0));
    let ref_target = create_session(target_model, &ref_cancel);
    let mut ref_logits = vec![0f32; target_vocab];
    prefill(ref_target, PREFILL_TOKENS, &mut ref_logits);
    let mut reference_tokens: Vec<i32> = Vec::with_capacity(200);
    for _ in 0..200 {
        let tok = argmax(&ref_logits);
        reference_tokens.push(tok);
        decode(ref_target, tok, &mut ref_logits);
    }
    destroy(ref_target);

    let mut total_output = 0usize;
    let mut total_accepted = 0usize;
    let mut batch = 0usize;

    while total_output < 200 {
        let mut draft_tokens: Vec<i32> = Vec::with_capacity(K);
        draft_tokens.push(argmax(&d_logits));
        for _ in 1..K {
            let last = *draft_tokens.last().unwrap();
            decode(draft, last, &mut d_logits);
            draft_tokens.push(argmax(&d_logits));
        }
        // Force rejection at first position every 3rd batch.
        if batch % 3 == 2 {
            draft_tokens[0] = (draft_tokens[0] + 1) % target_vocab as i32;
        }
        batch += 1;

        let mut step_accepted = 0usize;
        for ki in 0..K {
            let target_pick = argmax(&t_logits);
            if target_pick == draft_tokens[ki] {
                assert_eq!(draft_tokens[ki], reference_tokens[total_output],
                    "M_SPEC_1 C-synth: accepted differs from greedy reference at #{}", total_output);
                decode(target, draft_tokens[ki], &mut t_logits);
                step_accepted += 1;
                total_accepted += 1;
                total_output += 1;
                if total_output >= 200 { break; }  // done; avoid OOB on next ki
            } else {
                let rewind_by = K - 1 - ki;
                if rewind_by > 0 { rewind(draft, rewind_by); }
                let corrected = target_pick;
                assert_eq!(corrected, reference_tokens[total_output],
                    "M_SPEC_1 C-synth: corrected differs from greedy reference at #{}", total_output);
                decode(target, corrected, &mut t_logits);
                decode(draft,  corrected, &mut d_logits);
                total_output += 1;
                break;
            }
        }
        if step_accepted == K {
            decode(draft, draft_tokens[K - 1], &mut d_logits);
        }
        if total_output >= 200 { break; }  // skip position check on final partial batch
        let tp = position(target);
        let dp = position(draft);
        assert_eq!(tp, dp,
            "M_SPEC_2 C-synth: position mismatch at output #{total_output}: target={tp} draft={dp}");
    }

    let accept_rate = total_accepted as f64 / 200.0 * 100.0;
    println!("  200 tokens, {total_accepted} accepted ({accept_rate:.1}%), PASS");
    destroy(target);
    destroy(draft);
    println!("Protocol C-Synth: PASS");
}

// ── main ──────────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!("Usage: spec_validate <target.spm> <target.spt> <draft.spm> <draft.spt>");
        std::process::exit(1);
    }
    let (target_spm, target_spt) = (&args[1], &args[2]);
    let (draft_spm,  draft_spt)  = (&args[3], &args[4]);

    println!("Loading target: {target_spm}");
    let target_model = load_model(target_spm, target_spt);
    let target_arch = arch(target_model);
    println!("  vocab={} n_layers={}", target_arch.vocab_size, target_arch.n_layers);

    println!("Loading draft: {draft_spm}");
    let draft_model = load_model(draft_spm, draft_spt);
    let draft_arch = arch(draft_model);
    println!("  vocab={} n_layers={}", draft_arch.vocab_size, draft_arch.n_layers);

    let target_vocab = target_arch.vocab_size as usize;
    let draft_vocab  = draft_arch.vocab_size  as usize;

    assert_eq!(target_vocab, draft_vocab,
        "target/draft vocab mismatch: {} vs {} — both must be Qwen2.5 arch",
        target_vocab, draft_vocab);

    protocol_ab(    target_model, draft_model, target_vocab, draft_vocab);
    protocol_c(     target_model, draft_model, target_vocab, draft_vocab);
    protocol_c_synth(target_model, draft_model, target_vocab, draft_vocab);

    unsafe {
        ffi::sp_model_unload(target_model);
        ffi::sp_model_unload(draft_model);
    }

    println!("\nM_SPEC_1: PASS");
    println!("M_SPEC_2: PASS");
    println!("T8.1 VALIDATED: sp_session_rewind restores byte-identical KV state");
}
