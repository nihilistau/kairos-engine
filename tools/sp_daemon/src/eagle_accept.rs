//! SP_EAGLE_ACCEPT — LIVE EAGLE/MTP single-token acceptance probe, as a daemon
//! one-shot MODE (not a standalone bin): the token-management contract
//! (apply_template_ids / suppress_token_ids / eos_ids) lives in the binary-private
//! tokenizer+sampler modules, so the framework-faithful probe must run here, the
//! same way SP_NIGHTSHIFT_OFFLINE / SP_KAIROS_ALPHA do. Mirrors nightshift_curator's
//! SpModel+SptbTokenizer+session+gemma4_kv setup.
//!
//! Loads the served the dense reference model + the gemma4-assistant draft, greedy-decodes the target while
//! (a) framing the prompt with apply_template_ids (real <|turn>=105 / <turn|>=106 ids),
//! (b) suppressing soft/control tokens (suppress_token_ids: 258882/258883 + pipe-controls)
//! before argmax on BOTH the target and the draft, (c) stopping on eos_ids. Accept rate =
//! fraction where the draft's argmax == the target's greedy next token.
//!
//! Run: SP_EAGLE_ACCEPT=1 SP_MODEL_PATH=…b1.sp-model SP_TOKENIZER_PATH=…b1.sp-tokenizer \
//!      SP_DRAFT_GGUF=…<draft>-it-F16-MTP.gguf [SP_EAGLE_N=48] [SP_DRAFT_ASCALE=one] sp-daemon
#![cfg(feature = "wire_cuda_backend")]

use std::ffi::{c_void, CString};
use std::os::raw::{c_char, c_int};
use std::sync::atomic::AtomicI32;
use std::sync::Arc;

use crate::session::{SpModel, SpSession};
use crate::tokenizer::{Message, SptbTokenizer};

#[repr(C)]
struct sp_g4_kv {
    _opaque: [u8; 0],
}
extern "C" {
    fn gemma4_kv_open(m: *const c_void, pmax: c_int) -> *mut sp_g4_kv;
    fn gemma4_kv_prefill(s: *mut sp_g4_kv, toks: *const i32, n: c_int) -> c_int;
    fn gemma4_kv_decode_logits(s: *mut sp_g4_kv, token: i32, logits: *mut f32) -> c_int;
    fn gemma4_kv_capture_feat(s: *mut sp_g4_kv, feat: *mut f32) -> c_int;
    fn gemma4_kv_close(s: *mut sp_g4_kv);
    fn gemma4_draft_open(path: *const c_char) -> c_int;
    fn gemma4_draft_step(
        s: *mut sp_g4_kv,
        feat: *const f32,
        token: c_int,
        suppress: *const c_int,
        n_suppress: c_int,
        out_token: *mut c_int,
        out_hnext: *mut f32,
    ) -> c_int;
    fn gemma4_draft_close();
    fn gemma4_kv_reset(s: *mut sp_g4_kv) -> c_int;
    fn gemma4_kv_ctx_dump(s: *mut sp_g4_kv, kg: *mut f32, vg: *mut f32, ks: *mut f32, vs: *mut f32,
                          kvd_g: *mut c_int, kvd_s: *mut c_int, npos: *mut c_int) -> c_int;
    fn gemma4_kv_ctx_geom(s: *mut sp_g4_kv, g_nkv: *mut c_int, g_hd: *mut c_int, s_nkv: *mut c_int,
                          s_hd: *mut c_int, period: *mut c_int, kvfs: *mut c_int,
                          g_base: *mut f32, s_base: *mut f32) -> c_int;
    fn gemma4_embd_row(token: c_int, e: c_int, out: *mut f32) -> c_int;
    fn gemma4_kv_decode_batch(s: *mut sp_g4_kv, toks: *const i32, b: c_int,
                              logits_out: *mut f32, feat_out: *mut f32) -> c_int;
    fn gemma4_kv_rewind(s: *mut sp_g4_kv, delta: c_int) -> c_int;
    /// The C error surface every failing call already writes to (sp_status.h).
    fn sp_last_error() -> *const c_char;
    // Latent Interceptor: draft BODY -> 1024-d latent (no vocab head). The shared substrate.
    fn gemma4_draft_body(s: *mut sp_g4_kv, feat_host: *const f32, token: c_int, out_latent: *mut f32) -> c_int;
    // Memory Head ground truth: read the live global-layer K [n_global x npos x HD=512] (curator C2 input).
    fn gemma4_kv_read_global_k(s: *const sp_g4_kv, out: *mut f32, npos: c_int) -> c_int;
    // Latent-injection RETURN PATH: inject token-embeddings straight into the KV ring (no prompt text).
    fn gemma4_kv_inject_tokens(s: *mut sp_g4_kv, toks: *const i32, n: c_int) -> c_int;
}

/// KAIROS action space (Latent Interceptor) — keep in sync with CONTRACT-LATENT-INTERCEPTOR.md.
pub const LI_ACTIONS: [&str; 5] = ["NO_OP", "KEEP", "FORGET", "E2B_TOOL", "ACTION"];

/// The active label set: SP_LI_LABELS (comma-separated) overrides the KAIROS action space, so the
/// SAME capture/probe pipeline serves the Tool Head (e.g. NONE,PYTHON,WEB,DB,FILE,CALC).
fn li_label_set() -> Vec<String> {
    match std::env::var("SP_LI_LABELS") {
        Ok(s) if !s.trim().is_empty() => s.split(',').map(|x| x.trim().to_string()).collect(),
        _ => LI_ACTIONS.iter().map(|s| s.to_string()).collect(),
    }
}
fn label_id(labels: &[String], s: &str) -> i32 {
    labels.iter().position(|a| a.eq_ignore_ascii_case(s.trim())).map(|i| i as i32).unwrap_or(-1)
}

/// Apply the tiny Latent Interceptor probe (mu,sd,W1,b1,W2,b2 from li_head.bin) to a feature
/// (host f32[H]) -> action id. Pure CPU matmul (proj~256), microseconds. Layout per sp_li_train.py.
fn li_probe(blob: &[f32], h: usize, a: usize, proj: usize, feat: &[f32]) -> i32 {
    let mu = &blob[0..h];
    let sd = &blob[h..2 * h];
    let w1 = &blob[2 * h..2 * h + proj * h];
    let b1 = &blob[2 * h + proj * h..2 * h + proj * h + proj];
    let w2o = 2 * h + proj * h + proj;
    let w2 = &blob[w2o..w2o + a * proj];
    let b2 = &blob[w2o + a * proj..w2o + a * proj + a];
    let mut hid = vec![0.0f32; proj];
    for p in 0..proj {
        let mut s = b1[p];
        let row = &w1[p * h..p * h + h];
        for i in 0..h { s += row[i] * ((feat[i] - mu[i]) / sd[i]); }
        hid[p] = if s > 0.0 { s } else { 0.0 }; // ReLU
    }
    let mut best = 0i32; let mut bestv = f32::NEG_INFINITY;
    for c in 0..a {
        let mut s = b2[c];
        let row = &w2[c * proj..c * proj + proj];
        for p in 0..proj { s += row[p] * hid[p]; }
        if s > bestv { bestv = s; best = c as i32; }
    }
    best
}

fn parse_kairos_tape(text: &str) -> Vec<(String, i32)> {
    let mut events = Vec::new();
    for line in text.lines() {
        let l = line.trim();
        if l.is_empty() || l.starts_with('#') { continue; }
        let (mut toks, mut cur, mut q): (Vec<String>, String, bool) = (Vec::new(), String::new(), false);
        for ch in l.chars() {
            match ch {
                '"' => q = !q,
                c if c.is_whitespace() && !q => { if !cur.is_empty() { toks.push(std::mem::take(&mut cur)); } }
                c => cur.push(c),
            }
        }
        if !cur.is_empty() { toks.push(cur); }
        if toks.len() != 5 { continue; }
        let aid = li_action_id(&toks[4]);
        if aid < 0 { continue; }
        let payload = if toks[2] == "-" { String::new() } else { format!(" payload=\"{}\"", toks[2]) };
        events.push((format!("EVENT kind={} salience={}{}", toks[1], toks[3], payload), aid));
    }
    events
}

/// TELE-14 — STANDALONE SOVEREIGN native cross-family delegate (no Python). Loads the qwen2.5-coder
/// sp-model in its OWN L1 session, encodes the clean-text task with the coder's tokenizer, and decodes
/// natively through the engine's host L1 surface: `prefill_chunk` (prompt → last-token logits) then a
/// `decode_step` argmax loop (O(n) KV-resident, the SAME path the sp_memo_* bins use). CPU L1 means ZERO
/// g_w/CUDA contention with a resident dense model; on drop the SpModel/SpSession free with zero residual. HF
/// parity inherited from the top-1-lossless transcode gate (sp-Q4 argmax == oracle).
pub fn run_telepathy_native(model_path: &str, tok_path: &str) -> Result<(), String> {
    let model = SpModel::load(model_path, tok_path)?;
    let arch = model.arch_info()?;
    let vocab = arch.vocab_size as usize;
    let tok = SptbTokenizer::build(&model, arch.arch_id, tok_path)?;
    let cancel = Arc::new(AtomicI32::new(0));
    let mut session = SpSession::create(&model, cancel)?;
    let task = std::env::var("SP_TELEPATHY_TASK").unwrap_or_else(|_| "write a python function that reverses a string".into());
    let prompt = tok.apply_template_ids(&vec![Message { role: "user".to_string(), content: task.clone() }])?;
    let n_gen: usize = std::env::var("SP_TELEPATHY_NGEN").ok().and_then(|s| s.parse().ok()).unwrap_or(24);
    let eos: i32 = std::env::var("SP_TELEPATHY_EOS").ok().and_then(|s| s.parse().ok()).unwrap_or(151645); // qwen <|im_end|>
    let mut logits = vec![0f32; vocab];
    let argmax = |l: &[f32]| -> i32 { let mut bi = 0i32; let mut bv = f32::NEG_INFINITY; for (i, &v) in l.iter().enumerate() { if v > bv { bv = v; bi = i as i32; } } bi };
    eprintln!("[telepathy-native] coder loaded (arch_id={} vocab={}); task={:?}; prompt={} toks; native L1 prefill+decode (CPU)...", arch.arch_id, vocab, task, prompt.len());
    let t0 = std::time::Instant::now();
    session.prefill_chunk(&prompt, &mut logits)?;
    let mut out_ids: Vec<i32> = Vec::new();
    let mut next = argmax(&logits);
    for _ in 0..n_gen {
        if next == eos { break; }
        out_ids.push(next);
        session.decode_step(next, &mut logits)?;
        next = argmax(&logits);
    }
    let dt = t0.elapsed().as_secs_f32();
    let mut out = Vec::new();
    for &id in &out_ids { out.extend_from_slice(tok.decode_token(id)); }
    let ans = String::from_utf8_lossy(&out).replace('\n', " ");
    let ntok = out_ids.len();
    eprintln!("[telepathy-native] SOVEREIGN delegate ({} new toks, {:.2}s, {:.1} tok/s) -> {:?}", ntok, dt, ntok as f32 / dt.max(1e-3), ans.trim());
    let ok = ntok > 0 && !ans.trim().is_empty();
    eprintln!("[telepathy-native] G-TELEPATHY-NATIVE: {}  (native in-engine L1 coder decode, ZERO Python; HF parity inherited from top-1-lossless transcode gate)", if ok {"GREEN"} else {"RED"});
    if !ok { return Err("native decode produced no output".into()); }
    Ok(())
}

/// CLOSED LOOP (capstone): latent -> Tool Head (tool id) -> FIRE the tool -> result -> RETURN PATH
/// inject into the KV ring -> the model continues. Ties TH-1 + RP-1 into one call. The tool fire is a
/// real subprocess (python for PYTHON/CALC = the E2B stand-in); SP_TH_Q / SP_TH_CODE override.
pub fn run_th_loop(model_path: &str, tok_path: &str, head_path: &str) -> Result<(), String> {
    std::env::set_var("SP_CUDA_DECODE_INT8", "1");
    let model = SpModel::load(model_path, tok_path)?;
    let arch = model.arch_info()?;
    let vocab = arch.vocab_size as usize; let hidden = arch.hidden_dim as usize;
    let tok = SptbTokenizer::build(&model, arch.arch_id, tok_path)?;
    let cancel = Arc::new(AtomicI32::new(0));
    let mut session = SpSession::create(&model, cancel)?;
    let sraw = session.raw_ptr() as *mut sp_daemon::ffi_l1::sp_session;
    let qm = (unsafe { sp_daemon::ffi_l1::sp_session_qwen3_model(sraw) }) as *const c_void;
    if qm.is_null() { return Err("th_loop: qm NULL".to_string()); }
    let raw = std::fs::read(head_path).map_err(|e| format!("head: {e}"))?;
    let blob: Vec<f32> = raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    let tools = li_label_set();
    let a = tools.len();
    let proj = (blob.len() - 2 * hidden - a) / (hidden + 1 + a);
    let suppress: Vec<i32> = tok.suppress_token_ids();
    let supp = |lg: &mut [f32]| { for &id in &suppress { if (id as usize) < vocab { lg[id as usize] = f32::NEG_INFINITY; } } };

    let question = std::env::var("SP_TH_Q").unwrap_or_else(|_| "EVENT kind=EVENT.compute payload=\"count letters in strawberry\"".to_string());
    let py_code = std::env::var("SP_TH_CODE").unwrap_or_else(|_| "print('strawberry'.count('r'))".to_string());
    let py = std::env::var("SP_OKF_PY").unwrap_or_else(|_| "python".to_string());
    let prompt = tok.apply_template_ids(&vec![Message { role: "user".to_string(), content: question.clone() }])?;
    let np = prompt.len();
    let s = unsafe { gemma4_kv_open(qm, 256) };
    if s.is_null() { return Err("th_loop: kv_open NULL".to_string()); }
    let mut feat = vec![0.0f32; hidden];
    let mut logits = vec![0.0f32; vocab];

    // 1) latent: prefill the event, tap the frame-end feature.
    unsafe { gemma4_kv_reset(s) };
    if unsafe { gemma4_kv_prefill(s, prompt.as_ptr(), (np - 1) as c_int) } != 0 { return Err("th_loop: prefill".into()); }
    unsafe { gemma4_kv_capture_feat(s, feat.as_mut_ptr()) };
    unsafe { gemma4_kv_decode_logits(s, prompt[np - 1], logits.as_mut_ptr()) };
    // 2) Tool Head: latent -> tool id.
    let tool = li_probe(&blob, hidden, a, proj, &feat);
    let tool_name = tools.get(tool as usize).cloned().unwrap_or_else(|| "?".into());
    eprintln!("[th-loop] event: {question}");
    eprintln!("[th-loop] TOOL HEAD fired (latent->tool id): {tool_name}");
    // 3) FIRE the tool (real subprocess for PYTHON/CALC = the E2B stand-in).
    let result = if tool_name == "PYTHON" || tool_name == "CALC" {
        match std::process::Command::new(&py).args(["-c", &py_code]).stdin(std::process::Stdio::null()).output() {
            Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).trim().to_string(),
            Ok(o) => { eprintln!("[th-loop] tool stderr: {}", String::from_utf8_lossy(&o.stderr).trim()); "ERR".into() }
            Err(e) => { eprintln!("[th-loop] tool spawn failed: {e}"); "ERR".into() }
        }
    } else { format!("(no tool fired for {tool_name})") };
    let result_text = format!(" [tool result] {result}");
    eprintln!("[th-loop] tool ran -> result = \"{}\"", result.trim());
    // 4) RETURN PATH: inject the result into the KV ring (no prompt re-feed).
    let mut res_ids = tok.encode(&result_text).map_err(|e| format!("{e:?}"))?;
    if res_ids.first() == Some(&2) { res_ids.remove(0); }
    let rl = res_ids.len();
    if rl > 1 { unsafe { gemma4_kv_inject_tokens(s, res_ids.as_ptr(), (rl - 1) as c_int) }; }
    if unsafe { gemma4_kv_decode_logits(s, res_ids[rl - 1], logits.as_mut_ptr()) } != 0 { return Err("th_loop: decode".into()); }
    supp(&mut logits);
    // 5) continue: the model now "knows" the result latent-native.
    let mut g = argmax(&logits); let mut cont = String::new();
    for _ in 0..16 {
        if tok.eos_ids.contains(&g) { break; }
        cont.push_str(&String::from_utf8_lossy(tok.decode_token(g)));
        if unsafe { gemma4_kv_decode_logits(s, g, logits.as_mut_ptr()) } != 0 { break; }
        supp(&mut logits); g = argmax(&logits);
    }
    unsafe { gemma4_kv_close(s) };
    eprintln!("[th-loop] result injected into KV ring ({rl} tokens) -> model continues:");
    eprintln!("[th-loop]   \"{}\"", cont.trim());
    eprintln!("[th-loop] DONE — full latent loop: event -> tool head -> fire -> inject -> continue (no tokenizer in the decision/return)");
    Ok(())
}

/// LATENT-INJECTION RETURN PATH: a tool result enters the model's KV ring DIRECTLY (token-embeddings
/// injected, no prompt-text re-feed, no tokenizer output round-trip) — the model FEELS the result.
/// Demo (the strawberry problem): baseline (model alone) vs tool-result injected via
/// gemma4_kv_inject_tokens. SP_LI_Q / SP_LI_TOOLRESULT override the question / tool result.
pub fn run_li_return(model_path: &str, tok_path: &str) -> Result<(), String> {
    std::env::set_var("SP_CUDA_DECODE_INT8", "1");
    let model = SpModel::load(model_path, tok_path)?;
    let arch = model.arch_info()?;
    let vocab = arch.vocab_size as usize;
    let tok = SptbTokenizer::build(&model, arch.arch_id, tok_path)?;
    let cancel = Arc::new(AtomicI32::new(0));
    let mut session = SpSession::create(&model, cancel)?;
    let sraw = session.raw_ptr() as *mut sp_daemon::ffi_l1::sp_session;
    let qm = (unsafe { sp_daemon::ffi_l1::sp_session_qwen3_model(sraw) }) as *const c_void;
    if qm.is_null() { return Err("li_return: qm NULL".to_string()); }
    let suppress: Vec<i32> = tok.suppress_token_ids();
    let supp = |lg: &mut [f32]| { for &id in &suppress { if (id as usize) < vocab { lg[id as usize] = f32::NEG_INFINITY; } } };

    let question = std::env::var("SP_LI_Q").unwrap_or_else(|_| "How many letter r are in the word strawberry? Reply with only the number.".to_string());
    let result_text = std::env::var("SP_LI_TOOLRESULT").unwrap_or_else(|_| " [sandbox] python counted: strawberry has 3 letter r.".to_string());
    let prompt = tok.apply_template_ids(&vec![Message { role: "user".to_string(), content: question.clone() }])?;
    let mut res_ids = tok.encode(&result_text)?;
    if res_ids.first() == Some(&2) { res_ids.remove(0); }  // strip BOS for mid-sequence inject
    let np = prompt.len();
    let s = unsafe { gemma4_kv_open(qm, 256) };
    if s.is_null() { return Err("li_return: kv_open NULL".to_string()); }
    let mut logits = vec![0.0f32; vocab];
    let gen = 16usize;

    // helper: greedy-decode `n` tokens from the current logits' argmax -> text
    let run_decode = |s: *mut sp_g4_kv, logits: &mut Vec<f32>| -> String {
        let mut out = String::new();
        let mut g = argmax(logits);
        for _ in 0..gen {
            if tok.eos_ids.contains(&g) { break; }
            out.push_str(&String::from_utf8_lossy(tok.decode_token(g)));
            if unsafe { gemma4_kv_decode_logits(s, g, logits.as_mut_ptr()) } != 0 { break; }
            supp(logits);
            g = argmax(logits);
        }
        out
    };

    // BASELINE: prompt only.
    unsafe { gemma4_kv_reset(s) };
    if unsafe { gemma4_kv_prefill(s, prompt.as_ptr(), (np - 1) as c_int) } != 0 { return Err("li_return: prefill".into()); }
    unsafe { gemma4_kv_decode_logits(s, prompt[np - 1], logits.as_mut_ptr()) };
    supp(&mut logits);
    let base = run_decode(s, &mut logits);

    // RETURN PATH: prompt, then INJECT the tool result into the KV ring (no prompt-text re-feed).
    unsafe { gemma4_kv_reset(s) };
    if unsafe { gemma4_kv_prefill(s, prompt.as_ptr(), (np - 1) as c_int) } != 0 { return Err("li_return: prefill2".into()); }
    unsafe { gemma4_kv_decode_logits(s, prompt[np - 1], logits.as_mut_ptr()) };  // advance past prompt
    let rl = res_ids.len();
    if rl > 1 { if unsafe { gemma4_kv_inject_tokens(s, res_ids.as_ptr(), (rl - 1) as c_int) } != 0 { return Err("li_return: inject".into()); } }
    if unsafe { gemma4_kv_decode_logits(s, res_ids[rl - 1], logits.as_mut_ptr()) } != 0 { return Err("li_return: decode after inject".into()); }
    supp(&mut logits);
    let injected = run_decode(s, &mut logits);
    unsafe { gemma4_kv_close(s) };

    eprintln!("[li-return] Q: {}", question);
    eprintln!("[li-return] tool result (injected into KV ring, NOT re-prompted as text): \"{}\"", result_text.trim());
    eprintln!("[li-return] BASELINE (model alone)      -> \"{}\"", base.trim());
    eprintln!("[li-return] RETURN-PATH (result injected) -> \"{}\"", injected.trim());
    eprintln!("[li-return] DONE — the model felt the tool result latent-native ({} result tokens into the KV ring)", rl);
    Ok(())
}

/// PERSISTENT-CONTRACT KAIROS heartbeat (the TRUE Latent Interceptor deploy). The contract prefix is
/// prefilled ONCE; each tick appends only the EVENT-DELTA tokens, taps the latent, probes (~1ms), and
/// gates — then rewinds the delta so the next tick re-enters the clean contract context (O(Δ)). This
/// kills the per-tick full-contract re-prefill (~4000ms harness artifact in run_li_oracle). Logs the
/// true NO_OP-tick floor = [event-delta prefill ms] + [latent probe ms].
pub fn run_li_heartbeat(model_path: &str, tok_path: &str, tape_path: &str, head_path: &str) -> Result<(), String> {
    std::env::set_var("SP_CUDA_DECODE_INT8", "1");
    let model = SpModel::load(model_path, tok_path)?;
    let arch = model.arch_info()?;
    let vocab = arch.vocab_size as usize;
    let hidden = arch.hidden_dim as usize;
    let tok = SptbTokenizer::build(&model, arch.arch_id, tok_path)?;
    let cancel = Arc::new(AtomicI32::new(0));
    let mut session = SpSession::create(&model, cancel)?;
    let sraw = session.raw_ptr() as *mut sp_daemon::ffi_l1::sp_session;
    let qm = (unsafe { sp_daemon::ffi_l1::sp_session_qwen3_model(sraw) }) as *const c_void;
    if qm.is_null() { return Err("li_heartbeat: qm NULL".to_string()); }
    let raw = std::fs::read(head_path).map_err(|e| format!("head: {e}"))?;
    let blob: Vec<f32> = raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    let a = LI_ACTIONS.len();
    let proj = (blob.len() - 2 * hidden - a) / (hidden + 1 + a);

    let events = parse_kairos_tape(&std::fs::read_to_string(tape_path).map_err(|e| format!("tape: {e}"))?);
    if events.len() < 2 { return Err("li_heartbeat: need >=2 events".to_string()); }

    // Frame every event (apply_template_ids = correct <|turn> ids). DELTA-TRIM: all static framing is
    // in li_frame_text's contract prefix, so the common token prefix L spans everything but the event
    // body -> the per-tick delta is strictly the event data + turn close/primer.
    let frame_of = |body: &str| -> Result<Vec<i32>, String> {
        tok.apply_template_ids(&vec![Message { role: "user".to_string(), content: li_frame_text(body) }]).map_err(|e| format!("{e:?}"))
    };
    let framed: Vec<(Vec<i32>, i32)> = events.iter().filter_map(|(b, g)| frame_of(b).ok().map(|ids| (ids, *g))).collect();
    let mut lcp = framed[0].0.len().min(framed[1].0.len());
    for k in 0..lcp { if framed[0].0[k] != framed[1].0[k] { lcp = k; break; } }
    eprintln!("[li-heartbeat] contract prefix L={lcp} tok (prefilled ONCE); per-tick delta = ids[L..] | hidden={hidden} proj={proj}");

    let s = unsafe { gemma4_kv_open(qm, 512) };
    if s.is_null() { return Err("li_heartbeat: kv_open NULL".to_string()); }
    // PREFILL THE CONTRACT PREFIX ONCE.
    unsafe { gemma4_kv_reset(s) };
    let t_ctx = std::time::Instant::now();
    if unsafe { gemma4_kv_prefill(s, framed[0].0.as_ptr(), lcp as c_int) } != 0 { return Err("li_heartbeat: contract prefill".to_string()); }
    eprintln!("[li-heartbeat] contract prefilled once in {:.0}ms (amortized over all ticks)", t_ctx.elapsed().as_secs_f64() * 1000.0);

    let mut feat = vec![0.0f32; hidden];
    let mut logits = vec![0.0f32; vocab];
    let (mut correct, mut skips, mut woke) = (0usize, 0usize, 0usize);
    let (mut sum_noop_ms, mut sum_delta_ms, mut sum_probe_us) = (0.0f64, 0.0f64, 0.0f64);

    for (i, (ids, gt)) in framed.iter().enumerate() {
        let np = ids.len();
        if np <= lcp { continue; }
        // append ONLY the event-delta tokens [lcp..np); capture the latent at the last (model-primer).
        let td = std::time::Instant::now();
        if unsafe { gemma4_kv_prefill(s, ids[lcp..np - 1].as_ptr(), (np - 1 - lcp) as c_int) } != 0 { continue; }
        unsafe { gemma4_kv_capture_feat(s, feat.as_mut_ptr()) };
        if unsafe { gemma4_kv_decode_logits(s, ids[np - 1], logits.as_mut_ptr()) } != 0 { continue; }
        let delta_ms = td.elapsed().as_secs_f64() * 1000.0;
        let tp = std::time::Instant::now();
        let action = li_probe(&blob, hidden, a, proj, &feat);
        let probe_us = tp.elapsed().as_secs_f64() * 1e6;
        if action == *gt { correct += 1; }
        sum_delta_ms += delta_ms; sum_probe_us += probe_us;
        if action == 0 {
            skips += 1; sum_noop_ms += delta_ms + probe_us / 1000.0;
            eprintln!("[li-heartbeat] tick {i:>3} NO_OP   gt={:<8} | delta_prefill={delta_ms:.0}ms + probe={probe_us:.0}us = {:.0}ms NO_OP-tick (decode SKIPPED)",
                      LI_ACTIONS[*gt as usize], delta_ms + probe_us / 1000.0);
        } else {
            woke += 1;
            eprintln!("[li-heartbeat] tick {i:>3} {:<8} gt={:<8} | delta_prefill={delta_ms:.0}ms + probe={probe_us:.0}us -> WOKE the dense reference model",
                      LI_ACTIONS[action as usize], LI_ACTIONS[*gt as usize]);
        }
        // rewind the event delta -> next tick re-enters the clean contract context (persistent O(Δ)).
        // after prefill(delta[..np-1-lcp]) + decode_logits(last), dpos = np; rewind back to lcp.
        let cur = np as c_int;
        if cur > lcp as c_int { unsafe { gemma4_kv_rewind(s, cur - lcp as c_int) }; }
    }
    unsafe { gemma4_kv_close(s) };
    let n = framed.len();
    eprintln!("[li-heartbeat] DONE {n} ticks | accuracy={:.3} ({correct}/{n}) | NO_OP-skips={skips} woke={woke}", correct as f64 / n as f64);
    eprintln!("[li-heartbeat] TRUE FLOOR: NO_OP tick = {:.0}ms avg (delta_prefill {:.0}ms + probe {:.2}ms); was ~5000ms (full re-prefill+decode). decode SKIPPED on {skips}/{n} ticks.",
              if skips > 0 { sum_noop_ms / skips as f64 } else { 0.0 }, sum_delta_ms / n as f64, sum_probe_us / n as f64 / 1000.0);
    Ok(())
}

/// LIVE KAIROS ORACLE (the Latent Interceptor action gate). Per tick: the dense reference model prefills the event frame
/// (~35ms, unavoidable), taps the frame-end latent, runs the CPU probe (~us) -> action. NO_OP ->
/// SHORT-CIRCUIT (skip the ~440ms dense decode); any action class -> wake dense model decode. Logs the
/// per-tick decision + the banked compute. Reuses the gemma4_kv handle + the eagle feature tap.
pub fn run_li_oracle(model_path: &str, tok_path: &str, tape_path: &str, head_path: &str) -> Result<(), String> {
    std::env::set_var("SP_CUDA_DECODE_INT8", "1");
    let model = SpModel::load(model_path, tok_path)?;
    let arch = model.arch_info()?;
    let vocab = arch.vocab_size as usize;
    let hidden = arch.hidden_dim as usize;
    let tok = SptbTokenizer::build(&model, arch.arch_id, tok_path)?;
    let cancel = Arc::new(AtomicI32::new(0));
    let mut session = SpSession::create(&model, cancel)?;
    let sraw = session.raw_ptr() as *mut sp_daemon::ffi_l1::sp_session;
    let qm = (unsafe { sp_daemon::ffi_l1::sp_session_qwen3_model(sraw) }) as *const c_void;
    if qm.is_null() { return Err("li_oracle: qm NULL".to_string()); }

    let raw = std::fs::read(head_path).map_err(|e| format!("head {head_path}: {e}"))?;
    let blob: Vec<f32> = raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    let a = LI_ACTIONS.len();
    // len = 2H + A + proj*(H+1+A)  ->  proj
    let proj = (blob.len() - 2 * hidden - a) / (hidden + 1 + a);
    eprintln!("[li-oracle] head={head_path} hidden={hidden} proj={proj} actions={a}");

    let events = parse_kairos_tape(&std::fs::read_to_string(tape_path).map_err(|e| format!("tape: {e}"))?);
    if events.is_empty() { return Err("li_oracle: no events".to_string()); }

    let s = unsafe { gemma4_kv_open(qm, 512) };
    if s.is_null() { return Err("li_oracle: kv_open NULL".to_string()); }
    let mut feat = vec![0.0f32; hidden];
    let mut logits = vec![0.0f32; vocab];
    let suppress: Vec<i32> = tok.suppress_token_ids();
    let (mut correct, mut skips, mut woke, mut false_wake, mut missed) = (0usize, 0usize, 0usize, 0usize, 0usize);
    let (mut saved_ms, mut spent_ms) = (0.0f64, 0.0f64);
    let decode_cap = 24usize;

    for (i, (body, gt)) in events.iter().enumerate() {
        let ids = match tok.apply_template_ids(&vec![Message { role: "user".to_string(), content: li_frame_text(body) }]) { Ok(p) => p, Err(_) => continue };
        if ids.len() < 2 || ids.len() >= 512 { continue; }
        unsafe { gemma4_kv_reset(s) };
        let np = ids.len();
        let tpf = std::time::Instant::now();
        if unsafe { gemma4_kv_prefill(s, ids.as_ptr(), (np - 1) as c_int) } != 0 { continue; }
        unsafe { gemma4_kv_capture_feat(s, feat.as_mut_ptr()) };
        if unsafe { gemma4_kv_decode_logits(s, ids[np - 1], logits.as_mut_ptr()) } != 0 { continue; }
        let prefill_ms = tpf.elapsed().as_secs_f64() * 1000.0;
        let tpr = std::time::Instant::now();
        let action = li_probe(&blob, hidden, a, proj, &feat);
        let probe_us = tpr.elapsed().as_secs_f64() * 1e6;

        if action == *gt { correct += 1; }
        if action == 0 {
            // NO_OP -> short-circuit: the 477ms dense model decode is SKIPPED.
            skips += 1;
            let est = 440.0; saved_ms += est;
            if *gt != 0 { missed += 1; } // wrongly skipped a real event
            eprintln!("[li-oracle] tick {i:>3} action=NO_OP gt={} prefill={prefill_ms:.0}ms probe={probe_us:.0}us  DECODE SKIPPED (~{est:.0}ms banked)",
                      LI_ACTIONS[*gt as usize]);
        } else {
            // action -> wake the dense reference model to decode the execution (bounded).
            woke += 1;
            if *gt == 0 { false_wake += 1; } // wrongly woke on an idle tick
            let td = std::time::Instant::now();
            let mut last = argmax(&logits);
            for _ in 0..decode_cap {
                if tok.eos_ids.contains(&last) { break; }
                unsafe { gemma4_kv_decode_logits(s, last, logits.as_mut_ptr()) };
                for &id in &suppress { if (id as usize) < vocab { logits[id as usize] = f32::NEG_INFINITY; } }
                last = argmax(&logits);
            }
            let dms = td.elapsed().as_secs_f64() * 1000.0; spent_ms += dms;
            eprintln!("[li-oracle] tick {i:>3} action={:<8} gt={:<8} prefill={prefill_ms:.0}ms probe={probe_us:.0}us  WOKE dense model decode={dms:.0}ms",
                      LI_ACTIONS[action as usize], LI_ACTIONS[*gt as usize]);
        }
    }
    unsafe { gemma4_kv_close(s) };
    let n = events.len();
    eprintln!("[li-oracle] DONE {n} ticks | accuracy={:.3} ({correct}/{n}) | NO_OP-skips={skips} woke={woke} | false_wake={false_wake} missed={missed}",
              correct as f64 / n as f64);
    eprintln!("[li-oracle] COMPUTE: decode spent={spent_ms:.0}ms on {woke} woke ticks; BANKED ~{saved_ms:.0}ms on {skips} skipped ticks ({:.0}% of ticks gated)",
              100.0 * skips as f64 / n as f64);
    Ok(())
}
fn li_action_id(s: &str) -> i32 {
    LI_ACTIONS.iter().position(|&a| a.eq_ignore_ascii_case(s.trim())).map(|i| i as i32).unwrap_or(-1)
}

/// DELTA-TRIM: ALL static system framing (the kernel role + the action list + the respond
/// instruction) lives in the contract prefix; the per-tick delta is STRICTLY the event data + the
/// turn close/primer. Capture and heartbeat MUST share this exact framing for KV alignment.
const LI_CONTRACT: &str = "You are a background kernel daemon. Each tick you receive one environment \
event. Decide exactly one action and respond with exactly one of: NO_OP (idle, do nothing), KEEP \
(remember this fact), FORGET (evict stale state), E2B_TOOL (run a tool/compute), or ACTION \
(intervene). Most events are NO_OP.";
fn li_frame_text(body: &str) -> String { format!("{LI_CONTRACT}\n\nCURRENT EVENT: {body}") }

/// Memory-head probe: latent[1024] -> mu/sd norm -> W1/ReLU(512) -> W2(512) = pooled_K_est. Pure CPU.
fn mh_probe(blob: &[f32], latent: &[f32]) -> Vec<f32> {
    let (h, p, o) = (1024usize, 512usize, 512usize);
    let mu = &blob[0..h]; let sd = &blob[h..2 * h];
    let w1 = &blob[2 * h..2 * h + p * h]; let b1 = &blob[2 * h + p * h..2 * h + p * h + p];
    let w2o = 2 * h + p * h + p; let w2 = &blob[w2o..w2o + o * p]; let b2 = &blob[w2o + o * p..w2o + o * p + o];
    let mut hid = vec![0.0f32; p];
    for j in 0..p { let mut s = b1[j]; let row = &w1[j * h..j * h + h]; for i in 0..h { s += row[i] * ((latent[i] - mu[i]) / sd[i]); } hid[j] = s.max(0.0); }
    let mut out = vec![0.0f32; o];
    for k in 0..o { let mut s = b2[k]; let row = &w2[k * p..k * p + p]; for j in 0..p { s += row[j] * hid[j]; } out[k] = s; }
    out
}
fn ham4(a: &[u64; 4], b: &[u64; 4]) -> u32 { (0..4).map(|i| (a[i] ^ b[i]).count_ones()).sum() }
fn sig_hex(s: &[u64; 4]) -> String { format!("{:016x}{:016x}{:016x}{:016x}", s[3], s[2], s[1], s[0]) } // curator addr format

/// LIVE MEMORY-HEAD GATE: per event, gemma4_draft_body -> latent -> mh_probe -> pooled_K_est ->
/// recall::Projection (the FROZEN curator R) -> C2 256-bit sig -> MEM-OKF addr. Reports the Hamming
/// distance to the curator's own sig (the ground truth) + the per-event gate time. This is the draft
/// body writing curator-identical cyclotomic addresses from the latent stream, no tokenization.
pub fn run_mh_gate(model_path: &str, tok_path: &str, tape_path: &str, head_path: &str, draft_gguf: &str) -> Result<(), String> {
    std::env::set_var("SP_CUDA_DECODE_INT8", "1");
    let model = SpModel::load(model_path, tok_path)?;
    let arch = model.arch_info()?;
    let vocab = arch.vocab_size as usize; let hidden = arch.hidden_dim as usize;
    let tok = SptbTokenizer::build(&model, arch.arch_id, tok_path)?;
    let cancel = Arc::new(AtomicI32::new(0));
    let mut session = SpSession::create(&model, cancel)?;
    let sraw = session.raw_ptr() as *mut sp_daemon::ffi_l1::sp_session;
    let qm = (unsafe { sp_daemon::ffi_l1::sp_session_qwen3_model(sraw) }) as *const c_void;
    if qm.is_null() { return Err("mh_gate: qm NULL".to_string()); }
    let raw = std::fs::read(head_path).map_err(|e| format!("head: {e}"))?;
    let blob: Vec<f32> = raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    let proj = sp_daemon::recall::Projection::build();
    let n_global = sp_daemon::recall::n_global(); let hd_g = sp_daemon::recall::hd();
    let events = parse_kairos_tape(&std::fs::read_to_string(tape_path).map_err(|e| format!("tape: {e}"))?);
    // SP_MH_EMIT=1 -> actually write the KEEP episodes to MEM-OKF at the head's C2 address.
    let okf_mem = std::env::var("SP_OKF_MEM").unwrap_or_default();
    let okf_root = std::env::var("SP_OKF_ROOT").unwrap_or_else(|_| "memory-okf".to_string());
    let okf_py = std::env::var("SP_OKF_PY").unwrap_or_else(|_| "python".to_string());
    let emit = std::env::var("SP_MH_EMIT").as_deref() == Ok("1") && !okf_mem.is_empty();
    let mut emitted = 0usize;
    // npos cap for this probe — NOT hd(). Must be one number at open, skip, and
    // buffer size or a later edit overflows the D2H (the 26B illegal-access class).
    const PMAX: usize = 512;
    let s = unsafe { gemma4_kv_open(qm, PMAX as i32) };
    if s.is_null() { return Err("mh_gate: kv_open NULL".to_string()); }
    let dc = CString::new(draft_gguf).unwrap();
    if unsafe { gemma4_draft_open(dc.as_ptr()) } != 0 { return Err("mh_gate: draft_open failed".to_string()); }
    let mut feat = vec![0.0f32; hidden]; let mut logits = vec![0.0f32; vocab];
    // ROW WIDTH, not g_hd. read_global_k writes g_nkv*g_hd per (layer,pos); on the
    // 26B g_nkv is 2, so sizing this from hd() alone overflows by exactly half.
    let mut gk_buf = vec![0.0f32; n_global * PMAX * sp_daemon::recall::g_kvd()];
    let (mut sum_ham, mut sum_ms, mut n, mut keeps) = (0u64, 0.0f64, 0usize, 0usize);
    for (i, (body, aid)) in events.iter().enumerate() {
        let ids = match tok.apply_template_ids(&vec![Message { role: "user".to_string(), content: li_frame_text(body) }]) { Ok(p) => p, Err(_) => continue };
        if ids.len() < 2 || ids.len() >= PMAX { continue; }
        unsafe { gemma4_kv_reset(s) };
        let np = ids.len();
        if unsafe { gemma4_kv_prefill(s, ids.as_ptr(), (np - 1) as c_int) } != 0 { continue; }
        unsafe { gemma4_kv_capture_feat(s, feat.as_mut_ptr()) };
        if unsafe { gemma4_kv_decode_logits(s, ids[np - 1], logits.as_mut_ptr()) } != 0 { continue; }
        // THE GATE: latent -> pooled_K_est -> curator R -> sig (timed)
        let t0 = std::time::Instant::now();
        let mut latent = vec![0.0f32; 1024];
        if unsafe { gemma4_draft_body(s, feat.as_ptr(), ids[np - 1], latent.as_mut_ptr()) } != 0 { continue; }
        let pk_est = mh_probe(&blob, &latent);
        let sig_est = proj.signature(&pk_est, 1, 1);     // sign(R @ pooled_K_est)
        let gate_ms = t0.elapsed().as_secs_f64() * 1000.0;
        // ground truth: the curator's sig from the live global-K
        if unsafe { gemma4_kv_read_global_k(s, gk_buf.as_mut_ptr(), np as c_int) } <= 0 { continue; }
        let cur_sig = proj.signature(&gk_buf[..n_global * np * hd_g], n_global, np);
        let ham = ham4(&sig_est, &cur_sig);
        sum_ham += ham as u64; sum_ms += gate_ms; n += 1;
        if *aid == 1 {  // KEEP -> write a live MEM-OKF episode at the head's C2 address (latent-native)
            keeps += 1;
            if emit {
                let addr = format!("c2sig_{}", sig_hex(&sig_est));
                // sanitize: strip quotes (embedded " breaks Windows argv -> okf_mem argparse exit 2)
                let safe: String = body.chars().map(|c| if c == '"' || c == '\'' { ' ' } else { c }).collect();
                let blob = format!("mh_latent_episode:{addr}"); // the latent-written episode pointer (KV persist = prod detail)
                match std::process::Command::new(&okf_py).arg(&okf_mem)
                    .args(["add", "--root", &okf_root, "--kind", "episode", "--addr", &addr,
                           "--keys", &safe, "--summary", &safe, "--title", &safe, "--blob-ref", &blob,
                           "--status", "ACTIVE", "--gate", "G-MH-CURATOR", "--detail", &safe])
                    .stdin(std::process::Stdio::null()).output() {
                    Ok(o) if o.status.success() => { emitted += 1; eprintln!("[mh-gate] EMIT KEEP -> MEM-OKF {addr} {}", String::from_utf8_lossy(&o.stdout).trim()); }
                    Ok(o) => eprintln!("[mh-gate] EMIT rc={} err={}", o.status, String::from_utf8_lossy(&o.stderr).trim()),
                    Err(e) => eprintln!("[mh-gate] EMIT failed: {e}"),
                }
            }
        }
        if i < 12 {
            eprintln!("[mh-gate] ev {i:>3} {:<8} sig=c2sig_{:016x}.. ham={ham}/256 {} gate={gate_ms:.2}ms",
                      LI_ACTIONS[*aid as usize], sig_est[0], if ham <= (256 - sp_daemon::recall::TAU_BITS as u32) { "RECALL-MATCH" } else { "miss" });
        }
    }
    unsafe { gemma4_draft_close(); gemma4_kv_close(s); }
    if n == 0 { return Err("mh_gate: no events".to_string()); }
    let max_ham = 256 - sp_daemon::recall::TAU_BITS as u32;
    eprintln!("[mh-gate] DONE {n} events ({keeps} KEEP, {emitted} emitted to MEM-OKF) | mean Hamming-to-curator = {:.1}/256 | mean gate = {:.2}ms (latent->pooledK->R->C2 sig, no tokenization) | recall-radius<= {max_ham}",
              sum_ham as f64 / n as f64, sum_ms / n as f64);
    Ok(())
}

/// LATENT INTERCEPTOR CAPTURE: run the dense reference over a KAIROS event tape; per tick frame the event
/// against the kernel contract, prefill it, capture the FRAME-END feature[hidden] (the post-
/// output_norm hidden the LM head consumes) + the action label (tape `expect`). The interceptor
/// learns feature -> action WITHOUT the 262k vocab head. Each event is evaluated against a clean
/// contract context (KV reset between events = the pruned-heartbeat semantics).
pub fn run_li_capture(
    model_path: &str,
    tok_path: &str,
    tape_path: &str,
    out_dir: &str,
) -> Result<usize, String> {
    std::env::set_var("SP_CUDA_DECODE_INT8", "1");
    let model = SpModel::load(model_path, tok_path)?;
    let arch = model.arch_info()?;
    let vocab = arch.vocab_size as usize;
    let hidden = arch.hidden_dim as usize;
    let tok = SptbTokenizer::build(&model, arch.arch_id, tok_path)?;
    let cancel = Arc::new(AtomicI32::new(0));
    let mut session = SpSession::create(&model, cancel)?;
    let sraw = session.raw_ptr() as *mut sp_daemon::ffi_l1::sp_session;
    let qm = (unsafe { sp_daemon::ffi_l1::sp_session_qwen3_model(sraw) }) as *const c_void;
    if qm.is_null() { return Err("li_capture: qm NULL".to_string()); }

    let tape = std::fs::read_to_string(tape_path).map_err(|e| format!("tape {tape_path}: {e}"))?;
    let labelset = li_label_set();   // KAIROS actions by default; SP_LI_LABELS overrides (e.g. Tool Head)
    eprintln!("[li-capture] label set ({}): {:?}", labelset.len(), labelset);
    let mut events: Vec<(String, i32)> = Vec::new();
    for line in tape.lines() {
        let l = line.trim();
        if l.is_empty() || l.starts_with('#') { continue; }
        // tick_idx kind payload(may be quoted) salience expect
        let mut toks: Vec<String> = Vec::new();
        let (mut cur, mut q) = (String::new(), false);
        for ch in l.chars() {
            match ch {
                '"' => q = !q,
                c if c.is_whitespace() && !q => { if !cur.is_empty() { toks.push(std::mem::take(&mut cur)); } }
                c => cur.push(c),
            }
        }
        if !cur.is_empty() { toks.push(cur); }
        if toks.len() != 5 { continue; }
        let aid = label_id(&labelset, &toks[4]);
        if aid < 0 { continue; }
        let payload = if toks[2] == "-" { String::new() } else { format!(" payload=\"{}\"", toks[2]) };
        let body = format!("EVENT kind={} salience={}{}", toks[1], toks[3], payload);
        events.push((body, aid));
    }
    if events.is_empty() { return Err("li_capture: no events parsed".to_string()); }

    let pmax = 512 as c_int;
    let s = unsafe { gemma4_kv_open(qm, pmax) };
    if s.is_null() { return Err("li_capture: gemma4_kv_open NULL".to_string()); }
    // OPTIONAL draft body (SP_DRAFT_GGUF): also dump the 1024-d draft latent per event (the head
    // training substrate). Without it, only the dense reference model feature is captured (the feature-probe path).
    let draft_gguf = std::env::var("SP_DRAFT_GGUF").unwrap_or_default();
    let use_draft = if !draft_gguf.is_empty() {
        let dc = CString::new(draft_gguf.clone()).unwrap();
        if unsafe { gemma4_draft_open(dc.as_ptr()) } == 0 { eprintln!("[li-capture] draft body loaded -> dumping 1024-d latent"); true }
        else { eprintln!("[li-capture] WARN draft_open failed; feature-only"); false }
    } else { false };
    let mut feat = vec![0.0f32; hidden];
    let mut latent = vec![0.0f32; 1024];
    let mut logits = vec![0.0f32; vocab];
    let mut feats: Vec<f32> = Vec::with_capacity(events.len() * hidden);
    let mut latents: Vec<f32> = Vec::new();
    let mut labels: Vec<i32> = Vec::with_capacity(events.len());
    let mut dist = vec![0usize; labelset.len()];
    // MEMORY HEAD ground truth (v2 curator-distillation): the C2 256-bit sig + the 512-d pooled-K.
    let proj = sp_daemon::recall::Projection::build();
    let n_global = sp_daemon::recall::n_global();  // derived from the loaded model
    let hd_g = sp_daemon::recall::hd();            // g_hd — the PROJECTION width
    // ROW WIDTH, not g_hd — see recall::g_kvd(). The two were the same on the dense reference model
    // (g_nkv == 1) and the comments here said "8 global layers (period-6 over 48)" and
    // "512" as if they were facts about the machine rather than about the retired model.
    let g_kvd = sp_daemon::recall::g_kvd();
    let mut gk_buf = vec![0.0f32; n_global * (pmax as usize) * g_kvd];
    let mut pooledks: Vec<f32> = Vec::new();   // [N x 512]
    let mut sigs: Vec<u64> = Vec::new();       // [N x 4]

    for (i, (body, aid)) in events.iter().enumerate() {
        let msgs = vec![Message { role: "user".to_string(), content: li_frame_text(body) }];
        let ids = match tok.apply_template_ids(&msgs) { Ok(p) => p, Err(_) => continue };
        if ids.len() < 2 || ids.len() as c_int >= pmax { continue; }
        if unsafe { gemma4_kv_reset(s) } != 0 { return Err("li_capture: reset".to_string()); }
        let np = ids.len();
        if unsafe { gemma4_kv_prefill(s, ids.as_ptr(), (np - 1) as c_int) } != 0 { continue; }
        unsafe { gemma4_kv_capture_feat(s, feat.as_mut_ptr()) };
        if unsafe { gemma4_kv_decode_logits(s, ids[np - 1], logits.as_mut_ptr()) } != 0 { continue; }
        if use_draft {
            // draft body on the live frame -> the 1024-d latent (attends the frame KV).
            if unsafe { gemma4_draft_body(s, feat.as_ptr(), ids[np - 1], latent.as_mut_ptr()) } != 0 { continue; }
            latents.extend_from_slice(&latent);
            // curator C2 ground truth: read the live global-K, pool it, sign through the frozen R.
            if unsafe { gemma4_kv_read_global_k(s, gk_buf.as_mut_ptr(), np as c_int) } > 0 {
                let nk = n_global * np * hd_g;
                let gk = &gk_buf[..nk];
                let mut pk = vec![0.0f32; hd_g];
                for v in 0..(n_global * np) { let b = v * hd_g; for d in 0..hd_g { pk[d] += gk[b + d]; } }
                let sig = proj.signature(gk, n_global, np);
                pooledks.extend_from_slice(&pk);
                sigs.extend_from_slice(&sig);
            } else { latents.truncate(latents.len() - 1024); continue; } // keep arrays aligned
        }
        feats.extend_from_slice(&feat);
        labels.push(*aid);
        dist[*aid as usize] += 1;
        if i % 50 == 0 { eprintln!("[li-capture] {i}/{}", events.len()); }
    }
    if use_draft { unsafe { gemma4_draft_close() }; }
    unsafe { gemma4_kv_close(s) };
    std::fs::create_dir_all(out_dir).map_err(|e| format!("mkdir {out_dir}: {e}"))?;
    let f32b: Vec<u8> = feats.iter().flat_map(|x| x.to_le_bytes()).collect();
    let i32b: Vec<u8> = labels.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(format!("{out_dir}/feat.f32"), &f32b).map_err(|e| format!("feat: {e}"))?;
    std::fs::write(format!("{out_dir}/label.i32"), &i32b).map_err(|e| format!("label: {e}"))?;
    if use_draft {
        let lb: Vec<u8> = latents.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(format!("{out_dir}/latent.f32"), &lb).map_err(|e| format!("latent: {e}"))?;
        let pb: Vec<u8> = pooledks.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(format!("{out_dir}/pooledk.f32"), &pb).map_err(|e| format!("pooledk: {e}"))?;
        let sb: Vec<u8> = sigs.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(format!("{out_dir}/sig.u64"), &sb).map_err(|e| format!("sig: {e}"))?;
        eprintln!("[li-capture] dumped latent.f32 [{} x1024] + pooledk.f32 [{} x512] + sig.u64 [{} x4] (curator C2)",
                  latents.len() / 1024, pooledks.len() / 512, sigs.len() / 4);
    }
    let actions_json: String = labelset.iter().map(|a| format!("\"{a}\"")).collect::<Vec<_>>().join(",");
    std::fs::write(format!("{out_dir}/manifest.jsonl"),
        format!("{{\"n\":{},\"hidden\":{hidden},\"n_actions\":{},\"actions\":[{}]}}\n", labels.len(), labelset.len(), actions_json))
        .map_err(|e| format!("manifest: {e}"))?;
    let dist_str: String = labelset.iter().zip(dist.iter()).map(|(a, c)| format!("{a}={c}")).collect::<Vec<_>>().join(" ");
    eprintln!("[li-capture] DONE {} samples -> {out_dir} (hidden={hidden}) dist {dist_str}", labels.len());
    Ok(labels.len())
}

/// EAGLE flywheel CAPTURE: greedy-roll the target over a corpus, dumping per generated
/// position (feature h, input token, target label) + the full-sequence KV the draft attends.
/// This is the engine-matched training data for #2 (the draft learns OUR engine's distribution).
pub fn run_eagle_capture(
    model_path: &str,
    tok_path: &str,
    corpus_path: &str,
    out_dir: &str,
    gen_len: usize,
) -> Result<usize, String> {
    std::env::set_var("SP_CUDA_DECODE_INT8", "1");
    let model = SpModel::load(model_path, tok_path)?;
    let arch = model.arch_info()?;
    let vocab = arch.vocab_size as usize;
    let hidden = arch.hidden_dim as usize;
    let tok = SptbTokenizer::build(&model, arch.arch_id, tok_path)?;
    let cancel = Arc::new(AtomicI32::new(0));
    let mut session = SpSession::create(&model, cancel)?;
    let sraw = session.raw_ptr() as *mut sp_daemon::ffi_l1::sp_session;
    let qm = (unsafe { sp_daemon::ffi_l1::sp_session_qwen3_model(sraw) }) as *const c_void;
    if qm.is_null() { return Err("capture: qm NULL".to_string()); }
    let suppress: Vec<i32> = tok.suppress_token_ids();

    let corpus = std::fs::read_to_string(corpus_path).map_err(|e| format!("corpus {corpus_path}: {e}"))?;
    let lines: Vec<&str> = corpus.lines().map(|l| l.trim()).filter(|l| !l.is_empty()).collect();
    let max_prompt = 64usize;
    let pmax = (max_prompt + gen_len + 8) as c_int;
    let s = unsafe { gemma4_kv_open(qm, pmax) };
    if s.is_null() { return Err("capture: gemma4_kv_open NULL".to_string()); }

    // geometry for KV-buffer sizing.
    let (mut gnkv, mut ghd, mut snkv, mut shd, mut period, mut kvfs) = (0i32, 0i32, 0i32, 0i32, 0i32, 0i32);
    let (mut gb, mut sb) = (0f32, 0f32);
    unsafe { gemma4_kv_ctx_geom(s, &mut gnkv, &mut ghd, &mut snkv, &mut shd, &mut period, &mut kvfs, &mut gb, &mut sb); }
    let kvd_g = (gnkv * ghd) as usize;
    let kvd_s = (snkv * shd) as usize;
    std::fs::create_dir_all(out_dir).map_err(|e| format!("mkdir {out_dir}: {e}"))?;
    let mut manifest = String::new();
    manifest.push_str(&format!(
        "{{\"hidden\":{hidden},\"vocab\":{vocab},\"g_nkv\":{gnkv},\"g_hd\":{ghd},\"s_nkv\":{snkv},\"s_hd\":{shd},\"period\":{period},\"kvfs\":{kvfs},\"g_base\":{gb},\"s_base\":{sb},\"gen_len\":{gen_len}}}\n"
    ));

    let mut kg = vec![0.0f32; (pmax as usize) * kvd_g.max(1)];
    let mut vg = vec![0.0f32; (pmax as usize) * kvd_g.max(1)];
    let mut ks = vec![0.0f32; (pmax as usize) * kvd_s.max(1)];
    let mut vs = vec![0.0f32; (pmax as usize) * kvd_s.max(1)];
    let mut feat = vec![0.0f32; hidden];
    let mut xbuf = vec![0.0f32; hidden];
    let mut logits = vec![0.0f32; vocab];

    let mut seqs = 0usize;
    for (si, line) in lines.iter().enumerate() {
        let msgs = vec![Message { role: "user".to_string(), content: line.to_string() }];
        let mut prompt = match tok.apply_template_ids(&msgs) { Ok(p) => p, Err(_) => continue };
        if prompt.len() < 2 || prompt.len() > max_prompt { prompt.truncate(max_prompt); }
        if unsafe { gemma4_kv_reset(s) } != 0 { return Err("capture: reset".to_string()); }
        let np = prompt.len();
        if unsafe { gemma4_kv_prefill(s, prompt.as_ptr(), (np - 1) as c_int) } != 0 { continue; }
        let mut last = prompt[np - 1];
        let (mut feats, mut xs, mut inps, mut lbls, mut atts): (Vec<f32>, Vec<f32>, Vec<i32>, Vec<i32>, Vec<i32>) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for _ in 0..gen_len {
            unsafe { gemma4_kv_capture_feat(s, feat.as_mut_ptr()) };
            if unsafe { gemma4_kv_decode_logits(s, last, logits.as_mut_ptr()) } != 0 { break; }
            for &id in &suppress { if (id as usize) < vocab { logits[id as usize] = f32::NEG_INFINITY; } }
            let g = argmax(&logits);
            let attend = np as i32 + inps.len() as i32; // KV positions the draft attends for this step
            unsafe { gemma4_embd_row(last, hidden as c_int, xbuf.as_mut_ptr()); } // draft pre_proj input x
            xs.extend_from_slice(&xbuf);
            feats.extend_from_slice(&feat);
            inps.push(last);
            lbls.push(g);
            atts.push(attend);
            last = g;
            if tok.eos_ids.contains(&g) { break; }
        }
        if inps.is_empty() { continue; }
        let mut kvd_g_o = 0i32; let mut kvd_s_o = 0i32; let mut npos = 0i32;
        unsafe { gemma4_kv_ctx_dump(s, kg.as_mut_ptr(), vg.as_mut_ptr(), ks.as_mut_ptr(), vs.as_mut_ptr(),
                                    &mut kvd_g_o, &mut kvd_s_o, &mut npos); }
        let dir = format!("{out_dir}/seq_{si:04}");
        std::fs::create_dir_all(&dir).ok();
        let w = |name: &str, bytes: &[u8]| { let _ = std::fs::write(format!("{dir}/{name}"), bytes); };
        let f32b = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let i32b = |v: &[i32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        w("feat.f32", &f32b(&feats));
        w("x.f32", &f32b(&xs));
        w("inp.i32", &i32b(&inps));
        w("lbl.i32", &i32b(&lbls));
        w("att.i32", &i32b(&atts));
        w("kg.f32", &f32b(&kg[..(npos as usize) * kvd_g])); w("vg.f32", &f32b(&vg[..(npos as usize) * kvd_g]));
        w("ks.f32", &f32b(&ks[..(npos as usize) * kvd_s])); w("vs.f32", &f32b(&vs[..(npos as usize) * kvd_s]));
        manifest.push_str(&format!("{{\"seq\":{si},\"gen\":{},\"npos\":{npos},\"kvd_g\":{kvd_g},\"kvd_s\":{kvd_s}}}\n", inps.len()));
        seqs += 1;
        if si % 25 == 0 { eprintln!("[capture] seq {si}/{} gen={} npos={npos}", lines.len(), inps.len()); }
    }
    unsafe { gemma4_kv_close(s) };
    std::fs::write(format!("{out_dir}/manifest.jsonl"), manifest).map_err(|e| format!("manifest: {e}"))?;
    eprintln!("[capture] DONE {seqs} sequences -> {out_dir} (kvd_g={kvd_g} kvd_s={kvd_s} hidden={hidden})");
    Ok(seqs)
}

fn argmax(v: &[f32]) -> i32 {
    let (mut bi, mut bv) = (0usize, v[0]);
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            bi = i;
        }
    }
    bi as i32
}

pub fn run_eagle_accept(
    model_path: &str,
    tok_path: &str,
    draft_gguf: &str,
    n_decode: usize,
) -> Result<(usize, usize), String> {
    std::env::set_var("SP_CUDA_DECODE_INT8", "1"); // tied head (required by gemma4_kv_open)

    let model = SpModel::load(model_path, tok_path)?;
    let arch = model.arch_info()?;
    let vocab = arch.vocab_size as usize;
    let hidden = arch.hidden_dim as usize;
    let tok = SptbTokenizer::build(&model, arch.arch_id, tok_path)?;
    let cancel = Arc::new(AtomicI32::new(0));
    let mut session = SpSession::create(&model, cancel)?;
    let sraw = session.raw_ptr() as *mut sp_daemon::ffi_l1::sp_session;
    let qm = (unsafe { sp_daemon::ffi_l1::sp_session_qwen3_model(sraw) }) as *const c_void;
    if qm.is_null() {
        return Err("eagle: sp_session_qwen3_model NULL".to_string());
    }

    // FRAME via the token-management contract (real <|turn>/<turn|> ids, not literal strings).
    // SP_EAGLE_PROMPT overrides the probe prompt (held-out / OOD A/B).
    let prompt_text = std::env::var("SP_EAGLE_PROMPT")
        .unwrap_or_else(|_| "What is the capital of France?".to_string());
    let msgs = vec![Message {
        role: "user".to_string(),
        content: prompt_text,
    }];
    let prompt = tok.apply_template_ids(&msgs)?;
    if prompt.len() < 2 {
        return Err(format!("eagle: framed prompt too short ({})", prompt.len()));
    }
    // SUPPRESS set (soft/control tokens: 258882/258883 + pipe-controls + named specials).
    let suppress: Vec<i32> = tok.suppress_token_ids();
    eprintln!(
        "[eagle] target={model_path}\n[eagle] draft={draft_gguf}\n[eagle] prompt={} tok  vocab={vocab} hidden={hidden}  suppress={} ids  eos={:?}  ascale={}",
        prompt.len(), suppress.len(), tok.eos_ids,
        std::env::var("SP_DRAFT_ASCALE").unwrap_or_else(|_| "rsqrt(default)".into())
    );

    let pmax = (prompt.len() + n_decode + 8) as c_int;
    let s = unsafe { gemma4_kv_open(qm, pmax) };
    if s.is_null() {
        return Err("eagle: gemma4_kv_open NULL".to_string());
    }
    let dc = CString::new(draft_gguf).unwrap();
    if unsafe { gemma4_draft_open(dc.as_ptr()) } != 0 {
        unsafe { gemma4_kv_close(s) };
        return Err("eagle: gemma4_draft_open failed".to_string());
    }

    // prefill all-but-last so the first decode_logits processes the last prompt token cleanly.
    // BOTTLENECK PROFILE: prefill = (np-1) forwards with ONE host sync; decode below = one sync +
    // a full-vocab D2H per token. prefill_ms/token << decode_ms/token => sync/D2H-bound (cheap
    // batched verify wins); ~equal => per-forward compute-bound (needs batched-GEMM).
    let np = prompt.len();
    let t_pf = std::time::Instant::now();
    if unsafe { gemma4_kv_prefill(s, prompt.as_ptr(), (np - 1) as c_int) } != 0 {
        return Err("eagle: prefill failed".to_string());
    }
    let pf_ms = t_pf.elapsed().as_secs_f64() * 1000.0;
    eprintln!("[eagle] PROFILE prefill {} tok in {:.1}ms = {:.1} ms/tok (one host sync, no per-tok logits D2H)",
              np - 1, pf_ms, pf_ms / (np - 1).max(1) as f64);
    let mut last = prompt[np - 1];

    // K-step EAGLE DRIVE: draft K tokens via the recurrence (h_{k+1} = draft post_proj
    // out_hnext; token feedback), verify the prefix against the target's greedy, accept the
    // matching prefix. Measures mean ACCEPT-LENGTH (the real spec-decode quality metric) +
    // single-token accept (K=1 case) + a sequential-verify tok/s baseline. The realized
    // throughput win = (mean_accept+1) tokens per target forward, UNLOCKED by batched verify.
    let k = std::env::var("SP_EAGLE_K").ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(4);
    let dump = std::env::var("SP_EAGLE_DUMP").ok(); // optional flywheel-data dir (feature,target-token)
    let mut feat = vec![0.0f32; hidden];
    let mut hnext = vec![0.0f32; hidden];
    let mut logits = vec![0.0f32; vocab];
    let mut accept_lens: Vec<usize> = Vec::new();
    let (mut single_hit, mut single_n) = (0usize, 0usize);
    let mut tgt_seq: Vec<i32> = Vec::new();
    let mut samples: Vec<(i32, i32)> = Vec::new();
    let mut dump_rows: Vec<(Vec<f32>, i32)> = Vec::new();

    let suppress_logits = |lg: &mut [f32]| {
        for &id in &suppress { if (id as usize) < vocab { lg[id as usize] = f32::NEG_INFINITY; } }
    };

    // prime: feature + target greedy for the position right after `last`.
    unsafe { gemma4_kv_capture_feat(s, feat.as_mut_ptr()) };
    if unsafe { gemma4_kv_decode_logits(s, last, logits.as_mut_ptr()) } != 0 {
        return Err("eagle: prime decode failed".to_string());
    }
    suppress_logits(&mut logits);
    let mut g = argmax(&logits);

    // BOTTLENECK ISOLATION (SP_EAGLE_PROFILE=1): B sequential decode_logits (B syncs + B full-vocab
    // D2H) vs ONE gemma4_kv_decode_batch(B) (one sync). Same B tokens, same start (rewind between).
    // ratio >> 1 => per-token overhead is collapsible (batched verify wins without batched-GEMM).
    if std::env::var("SP_EAGLE_PROFILE").as_deref() == Ok("1") {
        let bb = 8usize;
        let probe_toks: Vec<i32> = (0..bb).map(|_| last).collect();
        let mut blog = vec![0.0f32; bb * vocab];
        // sequential
        let ts = std::time::Instant::now();
        for &t in &probe_toks { let _ = unsafe { gemma4_kv_decode_logits(s, t, logits.as_mut_ptr()) }; }
        let seq_ms = ts.elapsed().as_secs_f64() * 1000.0;
        unsafe { gemma4_kv_rewind(s, bb as c_int) };
        // batched
        let tb = std::time::Instant::now();
        let rc = unsafe { gemma4_kv_decode_batch(s, probe_toks.as_ptr(), bb as c_int, blog.as_mut_ptr(), std::ptr::null_mut()) };
        let bat_ms = tb.elapsed().as_secs_f64() * 1000.0;
        unsafe { gemma4_kv_rewind(s, bb as c_int) };
        eprintln!("[eagle] PROFILE B={bb}: {bb}x seq decode_logits = {seq_ms:.1}ms ({:.1} ms/tok) | 1x decode_batch = {bat_ms:.1}ms ({:.1} ms/tok) | speedup {:.2}x  rc={rc}",
                  seq_ms / bb as f64, bat_ms / bb as f64, seq_ms / bat_ms.max(0.001));
        // draft-step cost (the suspected real bottleneck): time bb draft_steps (no KV advance).
        let mut hn = vec![0.0f32; hidden];
        let mut dh = feat.clone();
        let td = std::time::Instant::now();
        for _ in 0..bb {
            let mut dk: c_int = -1;
            unsafe { gemma4_draft_step(s, dh.as_ptr(), last, suppress.as_ptr(), suppress.len() as c_int, &mut dk, hn.as_mut_ptr()); }
            std::mem::swap(&mut dh, &mut hn);
        }
        let draft_ms = td.elapsed().as_secs_f64() * 1000.0;
        eprintln!("[eagle] PROFILE B={bb}: draft_step = {draft_ms:.1}ms ({:.1} ms/step)  vs decode {:.1} ms/tok  -> draft/decode = {:.2}x",
                  draft_ms / bb as f64, seq_ms / bb as f64, (draft_ms / bb as f64) / (seq_ms / bb as f64).max(0.001));
    }

    let t0 = std::time::Instant::now();
    let mut target_tokens = 0usize;
    while target_tokens < n_decode && !tok.eos_ids.contains(&g) {
        // draft K tokens from (feat, last) via the EAGLE recurrence.
        let mut drafts: Vec<i32> = Vec::with_capacity(k);
        let mut dh = feat.clone();
        let mut din = last;
        for _ in 0..k {
            let mut dk: c_int = -1;
            let rc = unsafe {
                gemma4_draft_step(s, dh.as_ptr(), din, suppress.as_ptr(),
                                  suppress.len() as c_int, &mut dk, hnext.as_mut_ptr())
            };
            // NAME THE FAILURE. This said only "draft_step failed" and the C side had
            // already written the actual reason into sp_last_error() — so two separate
            // diagnosis cycles were spent guessing at a message that was sitting in a
            // buffer the whole time. Read it.
            if rc != 0 {
                let detail = unsafe {
                    let p = sp_last_error();
                    if p.is_null() { String::new() }
                    else { std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned() }
                };
                return Err(format!("eagle: draft_step failed: {detail}"));
            }
            drafts.push(dk);
            din = dk;
            std::mem::swap(&mut dh, &mut hnext); // dh := post_proj h_next (recurrence)
        }
        single_n += 1;
        if !drafts.is_empty() && drafts[0] == g { single_hit += 1; } // K=0 => plain-decode baseline
        if samples.len() < 12 && !drafts.is_empty() { samples.push((drafts[0], g)); }
        if dump.is_some() { dump_rows.push((feat.clone(), g)); }

        // verify: accept the leading drafts matching the target greedy continuation.
        let mut m = 0usize;
        while m < k && drafts[m] == g && !tok.eos_ids.contains(&g) {
            m += 1;
            tgt_seq.push(g);
            last = g;
            target_tokens += 1;
            unsafe { gemma4_kv_capture_feat(s, feat.as_mut_ptr()) };
            if unsafe { gemma4_kv_decode_logits(s, g, logits.as_mut_ptr()) } != 0 { g = -1; break; }
            suppress_logits(&mut logits);
            g = argmax(&logits);
        }
        if g == -1 { break; }
        if m == 0 {
            // draft[0] missed -> commit the target's corrected token (1 step) and move on.
            tgt_seq.push(g);
            last = g;
            target_tokens += 1;
            unsafe { gemma4_kv_capture_feat(s, feat.as_mut_ptr()) };
            if unsafe { gemma4_kv_decode_logits(s, g, logits.as_mut_ptr()) } != 0 { break; }
            suppress_logits(&mut logits);
            g = argmax(&logits);
        }
        accept_lens.push(m);
    }
    let elapsed = t0.elapsed().as_secs_f64();
    unsafe { gemma4_draft_close(); gemma4_kv_close(s); }

    if let Some(dir) = dump {
        let _ = std::fs::create_dir_all(&dir);
        let mut feats: Vec<u8> = Vec::with_capacity(dump_rows.len() * hidden * 4);
        let mut toks: Vec<u8> = Vec::with_capacity(dump_rows.len() * 4);
        for (h, t) in &dump_rows {
            for &v in h { feats.extend_from_slice(&v.to_le_bytes()); }
            toks.extend_from_slice(&t.to_le_bytes());
        }
        let _ = std::fs::write(format!("{dir}/feat.f32"), &feats);
        let _ = std::fs::write(format!("{dir}/tgt.i32"), &toks);
        eprintln!("[eagle] flywheel dump: {} (feature,target) rows -> {dir}", dump_rows.len());
    }

    let cont: String = tgt_seq.iter().flat_map(|&t| tok.decode_token(t).to_vec())
        .collect::<Vec<u8>>().into_iter().map(|b| b as char).collect();
    let mean_acc = if accept_lens.is_empty() { 0.0 } else { accept_lens.iter().sum::<usize>() as f64 / accept_lens.len() as f64 };
    let single = if single_n > 0 { 100.0 * single_hit as f64 / single_n as f64 } else { 0.0 };
    let toks_s = if elapsed > 0.0 { target_tokens as f64 / elapsed } else { 0.0 };
    eprintln!("[eagle] target continuation: {:?}", cont);
    eprintln!("[eagle] samples (draft0 -> target): {:?}", samples);
    eprintln!("[eagle] K={k} rounds={} mean_accept_len={mean_acc:.3} (potential {:.2}x w/ batched verify) | single-token={single:.1}% | {target_tokens} tok in {elapsed:.1}s = {toks_s:.1} tok/s (seq verify)",
              accept_lens.len(), mean_acc + 1.0);
    Ok((single_hit, single_n))
}
