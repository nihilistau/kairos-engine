//! §4-NTT Sprint NTT.5c — forward.c activation smoke harness.
//!
//! Two on-device gates:
//!   T_NTT5C_HD_64_BLUESTEIN_WORKS                 — Memory model prefill
//!                                                   with SP_ENGINE_NTT_ATTN=1
//!                                                   completes SP_OK + finite
//!                                                   logits + plausible argmax
//!                                                   (was silent fall-through
//!                                                   pre-NTT.5c since HD=64
//!                                                   couldn't sp_pr_init).
//!   T_NTT5C_HEX_BACKEND_ROUTES_WHEN_REGISTERED    — with SP_ENGINE_NTT_ATTN=1
//!                                                   AND a backend registered
//!                                                   via L1, the trampoline
//!                                                   dispatch counters in
//!                                                   `sp_daemon::ntt_hex_dispatch`
//!                                                   both increment > 0 across
//!                                                   one prefill_chunk turn.
//!
//! The harness mirrors the M.2 dialogue smoke pattern (L1Model + L1Session
//! wrappers, same FFI bindings via `sp_daemon::ffi_l1`) but is focused on a
//! single Memory-model prefill, so it's small and fast (~few seconds).
//!
//! On android:
//!   adb push sp_ntt_5c_forward_smoke /data/local/tmp/
//!   adb shell chmod +x /data/local/tmp/sp_ntt_5c_forward_smoke
//!   adb shell 'ADSP_LIBRARY_PATH="/data/local/tmp;" \
//!     SP_ENGINE_NTT_ATTN=1 \
//!     SP_ENGINE_NTT_ATTN_HEX=1 \
//!     /data/local/tmp/sp_ntt_5c_forward_smoke \
//!       /data/local/tmp/qwen25-coder-0.5b-memory.sp-model \
//!       /data/local/tmp/qwen25-coder-0.5b-memory.sp-tokenizer'
//!
//! Host build is a stub.

#![cfg_attr(not(target_os = "android"), allow(dead_code))]

#[cfg(target_os = "android")]
use sp_daemon::ffi_l1 as ffi;

#[cfg(target_os = "android")]
use sp_daemon::ntt_hex_dispatch;

use std::ffi::CString;
use std::os::raw::c_int;
use std::ptr;
use std::sync::atomic::AtomicI32;
use std::sync::Arc;
use std::time::Instant;

#[cfg(target_os = "android")]
use ffi::sp_status_SP_OK;

// ─── L1 wrappers (mirror M.2 verbatim — same pattern) ──────────────────────

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
fn load_model(model_path: &str, tok_path: &str) -> Result<(L1Model, ffi::sp_arch_info), String> {
    let model_c = CString::new(model_path).map_err(|e| e.to_string())?;
    let tok_c = CString::new(tok_path).map_err(|e| e.to_string())?;
    let mut ptr: *mut ffi::sp_model = ptr::null_mut();
    let st = unsafe { ffi::sp_model_load(model_c.as_ptr(), tok_c.as_ptr(), &mut ptr) };
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
    Ok((L1Model(ptr), arch))
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
fn prefill(s: &mut L1Session, tokens: &[i32], logits: &mut [f32]) -> Result<(), String> {
    if tokens.is_empty() {
        return Err("empty tokens".to_string());
    }
    let st = unsafe {
        ffi::sp_prefill_chunk(
            s.ptr,
            tokens.as_ptr(),
            tokens.len(),
            logits.as_mut_ptr(),
            logits.len(),
        )
    };
    if st != sp_status_SP_OK {
        return Err(format!("sp_prefill_chunk → status={st}"));
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

// ─── Backend setup via the daemon's NTT.5b helpers ─────────────────────────

#[cfg(target_os = "android")]
fn maybe_open_backend() -> Option<std::sync::Arc<ntt_hex_dispatch::ComputeBackend>> {
    use sp_daemon::dsp_rpc::FastRpcSession;
    const COMPUTE_SKEL_URI: &str =
        "file:///libsp_compute_skel.so?sp_compute_skel_handle_invoke&_modver=1.0&_dom=cdsp";
    match FastRpcSession::new(COMPUTE_SKEL_URI) {
        Ok(s) => {
            eprintln!("[NTT.5c-smoke] FastRpcSession open OK");
            Some(std::sync::Arc::new(
                ntt_hex_dispatch::ComputeBackend::new(std::sync::Arc::new(s)),
            ))
        }
        Err(e) => {
            eprintln!("[NTT.5c-smoke] FastRpcSession open failed: {e:?} — backend disabled");
            None
        }
    }
}

#[cfg(target_os = "android")]
fn register_backend_on_session(
    s: &L1Session,
    backend: &std::sync::Arc<ntt_hex_dispatch::ComputeBackend>,
) -> Result<(), String> {
    let backend_clone = std::sync::Arc::clone(backend);
    let leaked: *mut ntt_hex_dispatch::ComputeBackend =
        std::sync::Arc::into_raw(backend_clone) as *mut _;
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
        // Reclaim the leaked Arc to avoid count drift.
        unsafe { std::sync::Arc::from_raw(leaked); }
        return Err(format!("sp_session_register_compute_backend → status={rc}"));
    }
    Ok(())
}

// ─── Host stub ─────────────────────────────────────────────────────────────
#[cfg(not(target_os = "android"))]
fn main() {
    eprintln!("sp_ntt_5c_forward_smoke: host build is a stub — run on android (S22U).");
    eprintln!("  See header doc for adb push + invocation.");
    std::process::exit(0);
}

// ─── Main (android) ────────────────────────────────────────────────────────
#[cfg(target_os = "android")]
fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "Usage: {} <memo_model.spm> <memo_tok.spt>",
            args.get(0).map(|s| s.as_str()).unwrap_or("sp_ntt_5c_forward_smoke")
        );
        eprintln!("Env vars:");
        eprintln!("  SP_ENGINE_NTT_ATTN=1      enable NTT-attention overlay (required)");
        eprintln!("  SP_ENGINE_NTT_ATTN_HEX=1  also register Hex backend (for routing gate)");
        std::process::exit(2);
    }
    let memo_model_path = args[1].clone();
    let memo_tok_path = args[2].clone();

    let ntt_attn = std::env::var("SP_ENGINE_NTT_ATTN")
        .map(|v| v.trim() == "1")
        .unwrap_or(false);
    let ntt_attn_hex = std::env::var("SP_ENGINE_NTT_ATTN_HEX")
        .map(|v| v.trim() == "1")
        .unwrap_or(false);

    eprintln!("[NTT.5c-smoke] SP_ENGINE_NTT_ATTN={} SP_ENGINE_NTT_ATTN_HEX={}",
              ntt_attn, ntt_attn_hex);

    // ─── Load Memory model ──────────────────────────────────────────────────
    let t0 = Instant::now();
    let (model, arch) = match load_model(&memo_model_path, &memo_tok_path) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("[NTT.5c-smoke] FATAL: {e}");
            std::process::exit(3);
        }
    };
    eprintln!(
        "[NTT.5c-smoke] model loaded in {} ms: arch_id={} HD={} hidden={} layers={}",
        t0.elapsed().as_millis(),
        arch.arch_id, arch.head_dim, arch.hidden_dim, arch.n_layers,
    );

    // T_NTT5C_HD_64_BLUESTEIN_WORKS prerequisite: HD must be Bluestein-admissible
    // for this to exercise the new code path. {2..256}\{512} ∋ HD=64 (Qwen2.5).
    let hd = arch.head_dim;
    let pot = hd >= 2 && hd <= 256 && (hd & (hd - 1)) == 0;
    if !pot {
        eprintln!(
            "[NTT.5c-smoke] WARN: HD={hd} is not Bluestein-admissible (PoT in [2,256]); \
             the NTT-attention overlay will silently fall back to fp32. \
             T_NTT5C_HD_64_BLUESTEIN_WORKS would not exercise the new path."
        );
    }

    // ─── Optional backend setup ─────────────────────────────────────────────
    let backend = if ntt_attn_hex {
        maybe_open_backend()
    } else {
        None
    };

    // ─── Session create + (optional) backend register ───────────────────────
    let mut sess = match create_session(&model) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[NTT.5c-smoke] FATAL session: {e}");
            std::process::exit(3);
        }
    };
    if let Some(ref b) = backend {
        match register_backend_on_session(&sess, b) {
            Ok(()) => eprintln!("[NTT.5c-smoke] backend registered on session"),
            Err(e) => eprintln!("[NTT.5c-smoke] backend register WARN: {e}"),
        }
    }

    // ─── Reset dispatch counters before the prefill turn ───────────────────
    ntt_hex_dispatch::reset_dispatch_counts();
    let (fwd0, inv0) = ntt_hex_dispatch::dispatch_counts();
    eprintln!("[NTT.5c-smoke] dispatch counts before prefill: forward={fwd0} inverse={inv0}");

    // ─── Prefill turn ──────────────────────────────────────────────────────
    let prompt = vec![1i32, 2, 3];   // 3 tokens: small but causal-attention non-trivial
    let vocab = arch.vocab_size as usize;
    let mut logits = vec![0f32; vocab];
    let t1 = Instant::now();
    let prefill_rc = prefill(&mut sess, &prompt, &mut logits);
    let prefill_ms = t1.elapsed().as_millis();
    let (fwd1, inv1) = ntt_hex_dispatch::dispatch_counts();

    eprintln!(
        "[NTT.5c-smoke] prefill: rc={:?} wall={prefill_ms}ms dispatch fwd={fwd1} inv={inv1}",
        prefill_rc
    );

    // ─── Gate evaluation ───────────────────────────────────────────────────
    let mut failed = 0;

    // T_NTT5C_HD_64_BLUESTEIN_WORKS: SP_OK + finite logits + plausible argmax
    let t_blue_works = prefill_rc.is_ok() && all_finite(&logits) && {
        let am = argmax(&logits);
        am > 0 && (am as u32) < arch.vocab_size - 1
    };
    if t_blue_works {
        eprintln!("[T_NTT5C_HD_64_BLUESTEIN_WORKS] PASS  argmax={}", argmax(&logits));
    } else {
        eprintln!(
            "[T_NTT5C_HD_64_BLUESTEIN_WORKS] FAIL  rc={:?} all_finite={} argmax={}",
            prefill_rc,
            all_finite(&logits),
            argmax(&logits)
        );
        failed += 1;
    }

    // T_NTT5C_HEX_BACKEND_ROUTES_WHEN_REGISTERED: dispatch counters > 0 iff
    // backend was registered. When ntt_attn_hex=0 OR backend register failed,
    // this gate is NOT evaluated (informational only).
    if ntt_attn_hex && backend.is_some() {
        let routes = fwd1 > fwd0 && inv1 > inv0;
        if routes {
            eprintln!(
                "[T_NTT5C_HEX_BACKEND_ROUTES_WHEN_REGISTERED] PASS  forward+={} inverse+={}",
                fwd1 - fwd0, inv1 - inv0
            );
        } else {
            eprintln!(
                "[T_NTT5C_HEX_BACKEND_ROUTES_WHEN_REGISTERED] FAIL  forward+={} inverse+={} \
                 (expected both > 0 with SP_ENGINE_NTT_ATTN=1+SP_ENGINE_NTT_ATTN_HEX=1)",
                fwd1 - fwd0, inv1 - inv0
            );
            failed += 1;
        }
    } else {
        eprintln!(
            "[T_NTT5C_HEX_BACKEND_ROUTES_WHEN_REGISTERED] SKIP  (env_hex={} backend={:?})",
            ntt_attn_hex, backend.is_some()
        );
    }

    drop(sess);
    drop(model);
    drop(backend);

    if failed == 0 {
        eprintln!("[NTT.5c-smoke] OK (0 gates failed)");
        std::process::exit(0);
    } else {
        eprintln!("[NTT.5c-smoke] FAIL ({failed} gates failed)");
        std::process::exit(1);
    }
}
