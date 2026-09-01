//! NIGHTSHIFT offline curator — G-NIGHTSHIFT-CURATOR (lattice CONTRACT-NIGHTSHIFT-CURATOR.md).
//!
//! Iterates the live B4 episodes in `_nightshift_live/`, uses the dense reference model to EXTRACT
//! the invariant `ep.secret` from `ep.txt` (the §5 distillation lever), runs the
//! teacher-forced causal-ablation ADMIT oracle (reusing the disposer FFI
//! sequence verbatim), and EMITS conformant MEM-OKF episode records (addr = C2
//! signature) for accepted (novel) episodes. Parametric leakage collapses ~0.
//!
//! Null floor: only runs when main.rs sees SP_NIGHTSHIFT_OFFLINE=1. Opens its OWN
//! kvdecode handle and never touches the served cache.
#![cfg(feature = "kairos")]

use std::collections::HashSet;
use std::ffi::c_void;
use std::sync::atomic::AtomicI32;
use std::sync::Arc;

use sp_daemon::cuda_kvdecode_dispatch as kv;
use sp_daemon::dialogue::argmax;
use crate::session::{SpModel, SpSession};
use crate::tokenizer::SptbTokenizer;
use sp_daemon::recall::{self, Projection};

const TAU: f32 = -8.0;

fn lse(z: &[f32]) -> f32 {
    let m = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut s = 0.0f32;
    for &v in z { s += (v - m).exp(); }
    m + s.ln()
}

fn sig_hex(s: &[u64; 4]) -> String {
    format!("{:016x}{:016x}{:016x}{:016x}", s[3], s[2], s[1], s[0])
}

/// Fallback distillation (last sentence) if the model extractor fails.
fn distill_secret(text: &str) -> String {
    let t = text.trim();
    if let Some(idx) = t.trim_end_matches('.').rfind(". ") {
        t[idx + 2..].trim().to_string()
    } else {
        t.to_string()
    }
}

/// §5 model-call extractor: a strict few-shot prompt forces the dense reference model to emit ONLY
/// the invariant distinctive fact from `ep_txt`. Greedy-decoded via a fresh kv
/// handle (prefill-by-decode_step), stop on newline/EOS or 16 tokens.
unsafe fn extract_secret(qm: *const c_void, tok: &SptbTokenizer, ep_txt: &str, logits: &mut [f32]) -> Option<String> {
    let prompt = format!(
        "Extract the single distinctive fact from the text. Output ONLY the fact, nothing else.\n\
         Text: The capital of France is Paris.\nFact: Paris\n\
         Text: The vault access code is 8-FALCON-7729 and it expires at midnight.\nFact: 8-FALCON-7729\n\
         Text: {}\nFact:",
        ep_txt.trim());
    let ptoks = tok.encode(&prompt).ok()?;
    if ptoks.is_empty() { return None; }
    let h = kv::open(qm, ptoks.len() as i32 + 32).ok()?;
    for &t in &ptoks {
        if kv::decode_step(h, t, logits).is_err() { kv::close(h); return None; }
    }
    let mut out: Vec<i32> = Vec::new();
    let mut next = argmax(logits);
    for _ in 0..16 {
        if !tok.eos_ids.is_empty() && tok.eos_ids.contains(&next) { break; }
        let piece = tok.decode_token(next).to_vec();
        if String::from_utf8_lossy(&piece).contains('\n') { break; }
        out.push(next);
        if kv::decode_step(h, next, logits).is_err() { break; }
        next = argmax(logits);
    }
    kv::close(h);
    if out.is_empty() { return None; }
    let bytes: Vec<u8> = out.iter().flat_map(|&t| tok.decode_token(t).to_vec()).collect();
    let s = String::from_utf8_lossy(&bytes).trim().to_string();
    if s.is_empty() { None } else { Some(s) }
}

/// Teacher-forced causal-ablation collapse (ΣΔLL). Mirrors the disposer.
unsafe fn admit(
    handle: *mut c_void,
    tok: &SptbTokenizer,
    dir: &str,
    npos: i32,
    eptok: &[i32],
    secret: &str,
    logits: &mut [f32],
) -> Option<f32> {
    let mut secret_ids = tok.encode(secret).ok()?;
    if secret_ids.first() == Some(&2) { secret_ids.remove(0); }
    if secret_ids.is_empty() || eptok.is_empty() { return None; }
    let anchor = kv::position(handle);
    if kv::replay(handle, dir, npos, false).is_err() { return None; }
    let seed = *eptok.last().unwrap();
    let mut gen: Vec<i32> = Vec::new();
    let mut lpe: Vec<f32> = Vec::new();
    let mut t = seed;
    for &s in &secret_ids {
        if kv::decode_step(handle, t, logits).is_err() { break; }
        lpe.push(logits[s as usize] - lse(logits));
        gen.push(s);
        t = s;
    }
    let ng = gen.len();
    let _ = kv::rewind(handle, ng as i32);
    if ng == 0 { let _ = kv::rewind(handle, npos); return None; }
    let want: HashSet<i32> = gen.iter().copied().collect();
    let mut targets: Vec<i32> = Vec::new();
    for (p, &tk) in eptok.iter().enumerate() {
        if p >= npos as usize { break; }
        if want.contains(&tk) { targets.push(p as i32); }
    }
    if targets.len() > 12 { targets.truncate(12); }
    let _ = kv::ablate(handle, anchor, &targets);
    let mut lpa: Vec<f32> = Vec::with_capacity(ng);
    let mut t = seed;
    for i in 0..ng {
        if kv::decode_step(handle, t, logits).is_err() { break; }
        lpa.push(logits[gen[i] as usize] - lse(logits));
        t = gen[i];
    }
    let _ = kv::rewind(handle, lpa.len() as i32 + npos);
    debug_assert_eq!(kv::position(handle), anchor);
    let n = lpe.len().min(lpa.len());
    let mut collapse = 0.0f32;
    for j in 0..n { collapse += lpa[j] - lpe[j]; }
    Some(collapse)
}

pub fn run_kairos_curator(
    model_path: &str,
    tok_path: &str,
    live_dir: &str,
) -> Result<(usize, usize), String> {
    let model = SpModel::load(model_path, tok_path)?;
    let arch = model.arch_info()?;
    let vocab = arch.vocab_size as usize;
    let tok = SptbTokenizer::build(&model, arch.arch_id, tok_path)?;
    let cancel = Arc::new(AtomicI32::new(0));
    let mut session = SpSession::create(&model, cancel)?;
    let sraw = session.raw_ptr() as *mut sp_daemon::ffi_l1::sp_session;
    let qm = (unsafe { sp_daemon::ffi_l1::sp_session_qwen3_model(sraw) }) as *const c_void;
    if qm.is_null() { return Err("curator: sp_session_qwen3_model NULL".to_string()); }
    let proj = Projection::build();
    let mut logits = vec![0.0f32; vocab];
    let okf_mem = std::env::var("SP_OKF_MEM").unwrap_or_default();
    let okf_root = std::env::var("SP_OKF_ROOT").unwrap_or_else(|_| "memory-okf".to_string());
    let okf_py = std::env::var("SP_OKF_PY").unwrap_or_else(|_| "python".to_string());

    let mut accepted = 0usize;
    let mut rejected = 0usize;
    let entries = std::fs::read_dir(live_dir).map_err(|e| format!("read_dir {live_dir}: {e}"))?;
    for ent in entries {
        let dir = match ent { Ok(e) => e.path(), Err(_) => continue };
        if !dir.is_dir() { continue; }
        let dir_str = dir.to_string_lossy().to_string();
        let text = match std::fs::read_to_string(dir.join("ep.txt")) { Ok(t) => t, Err(_) => continue };
        let eptok: Vec<i32> = match std::fs::read_to_string(dir.join("ep.tok")) {
            Ok(s) => s.lines().filter_map(|l| l.trim().parse::<i32>().ok()).collect(),
            Err(_) => continue,
        };
        if eptok.len() < 4 { continue; }
        let npos = eptok.len() as i32;
        let secret = match unsafe { extract_secret(qm, &tok, &text, &mut logits) } {
            Some(s) => s,
            None => distill_secret(&text),
        };
        eprintln!("[curator] {dir_str} secret=\"{secret}\"");
        let _ = std::fs::write(dir.join("ep.secret"), &secret);
        let handle = match unsafe { kv::open(qm, npos + 64) } {
            Ok(h) => h,
            Err(e) => { eprintln!("[curator] kv::open failed for {dir_str}: {e}"); continue; }
        };
        let collapse = unsafe { admit(handle, &tok, &dir_str, npos, &eptok, &secret, &mut logits) };
        unsafe { kv::close(handle) };
        match collapse {
            Some(c) if c < TAU => {
                accepted += 1;
                let (gk, ng) = recall::load_episode_global_k(&dir_str, npos).unwrap_or((Vec::new(), 0));
                let np = if ng > 0 { gk.len() / (ng * recall::hd()) } else { 0 };
                let sig = if np > 0 { proj.signature(&gk, ng, np) } else { [0u64; 4] };
                let addr = format!("c2sig_{}", sig_hex(&sig));
                let topic: String = text.trim().chars().take(60).collect();
                eprintln!("[curator] ACCEPT {dir_str} collapse={c:.2} addr={addr}");
                if !okf_mem.is_empty() {
                    match std::process::Command::new(&okf_py)
                        .arg(&okf_mem)
                        .args(["add", "--root", &okf_root, "--kind", "episode",
                               "--addr", &addr, "--blob-ref", &dir_str,
                               "--keys", &topic, "--summary", &topic, "--title", &topic,
                               "--status", "ACTIVE", "--gate", "G-NIGHTSHIFT-CURATOR",
                               "--detail", &secret])
                        .output() {
                        Ok(o) => eprintln!("[curator] emit rc={} {}", o.status, String::from_utf8_lossy(&o.stdout).trim()),
                        Err(e) => eprintln!("[curator] emit FAILED: {e}"),
                    }
                }
            }
            Some(c) => { rejected += 1; eprintln!("[curator] REJECT {dir_str} collapse={c:.2} >= TAU={TAU}"); }
            None => { rejected += 1; eprintln!("[curator] SKIP {dir_str} (no admit)"); }
        }
    }
    eprintln!("[curator] DONE accepted={accepted} rejected={rejected}");
    Ok((accepted, rejected))
}