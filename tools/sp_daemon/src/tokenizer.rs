//! SPTB tokenizer adapter — parses the .sp-tokenizer blob and wraps it in the
//! `tokenizers` crate for prompt encoding / token decoding.

use std::ffi::{c_char, c_int, CString};
use std::sync::Arc;

use ahash::AHashMap;
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::{AddedToken, Tokenizer};

use crate::session::SpModel;

const TYPE_BPE_LLAMA3: u32 = 1;
const TYPE_BPE_GPT2: u32 = 2;
// Issue #115: GEMMA4_BPE on-disk family tag (sp_tok_type_id::SP_TOK_GEMMA4_BPE).
// The Rust `tokenizers` crate cannot drive the 514k-merge U+2581-piece gemma4
// BPE, so this lane routes encode through the engine's PROVEN C encoder
// (T_G4_TOK_PARITY: byte-for-byte vs the llama oracle).
const TYPE_GEMMA4_BPE: u32 = 4;

const ARCH_QWEN3: u32 = 2;
const ARCH_GEMMA3: u32 = 3;
/// qwen35moe (Qwen3.6-35B-A3B GDN+MoE hybrid) — SP_ARCH_ID_QWEN36 = 8 in
/// sp_model.h (the L1 wire arch_id; NOT the internal sp_arch_t SP_ARCH_QWEN36=4).
/// Same ChatML surface as ARCH_QWEN3 (im_start/im_end, <|endoftext|>).
const ARCH_QWEN36: u32 = 8;
const ARCH_QWEN25: u32 = 6;
const ARCH_GEMMA4: u32 = 7;
/// gemma4 backbone + routed MoE FFN, ordinary AR decode (Gemma-4-26B-A4B) —
/// SP_ARCH_ID_GEMMA4_MOE = 11 in sp_model.h. Its VOCAB is byte-identical to the
/// the dense reference model's (same 262144 tokens, `<|turn>`=105 / `<turn|>`=106 / `<eos>`=1), so it
/// speaks exactly the gemma-3/4 turn grammar.
const ARCH_GEMMA4_MOE: u32 = 11;

/// THE gemma-turn-grammar predicate. Six separate `match` arms in this file used
/// to spell out `ARCH_GEMMA3 | ARCH_GEMMA4` — stop tokens, turn stops, default
/// stops, the text template, and the id-level template. A new gemma-4 variant
/// that is added to five of the six is broken in exactly one way, silently, and
/// the comment at `find_eos_ids` records what that costs: "the live daemon logged
/// eos_ids=[] for arch_id=7, so decode never stopped on EOS". The list lives HERE,
/// once; the match arms are guards over it.
#[inline]
fn is_gemma_turn_arch(arch_id: u32) -> bool {
    matches!(arch_id, ARCH_GEMMA3 | ARCH_GEMMA4 | ARCH_GEMMA4_MOE)
}

// ── Engine C tokenizer FFI (issue #115) ──────────────────────────────────────
// The proven gemma4 BPE encoder, compiled from src/tokenizer/{tokenizer,
// gemma4_bpe}.c by build.rs (cc) and linked into the daemon. We FFI the small
// load/encode/free surface (bindgen'd from sp_engine/tokenizer.h into
// crate::ffi_l1) rather than reimplementing the encoder in Rust.
use crate::ffi::{
    sp_tokenizer, sp_tokenizer_encode, sp_tokenizer_free, sp_tokenizer_load_tokfile,
    sp_tokenizer_vocab_size,
};

/// RAII wrapper around the engine C `sp_tokenizer` (gemma4 lane).
///
/// SAFETY / Send+Sync: after `sp_tokenizer_load_tokfile` returns, the handle is
/// READ-ONLY — `sp_tokenizer_encode` only reads the loaded vocab/merge tables
/// (no internal mutation, no shared global state) and the engine builds no
/// thread-local caches. The daemon shares one handle across rayon/tokio
/// blocking threads via Arc<AppState>, so concurrent `encode` calls are sound.
/// The handle is freed exactly once on Drop.
struct CTokenizer {
    handle: *mut sp_tokenizer,
}

impl CTokenizer {
    /// Load a .sp-tokenizer through the engine's blob loader. The loader
    /// dispatches on the on-disk family tag; for gemma4 it builds the
    /// 514k-merge BPE tables.
    fn load(path: &str) -> Result<Self, String> {
        let c_path = CString::new(path)
            .map_err(|_| format!("tokenizer path has interior NUL: {path}"))?;
        // SAFETY: c_path outlives the call; the loader copies what it needs.
        let handle = unsafe { sp_tokenizer_load_tokfile(c_path.as_ptr()) };
        if handle.is_null() {
            return Err(format!(
                "sp_tokenizer_load_tokfile failed for {path} (see daemon stderr for the engine diagnostic)"
            ));
        }
        Ok(CTokenizer { handle })
    }

    fn vocab_size(&self) -> u32 {
        // SAFETY: handle is a valid, non-null sp_tokenizer for self's lifetime.
        unsafe { sp_tokenizer_vocab_size(self.handle) }
    }

    /// Encode UTF-8 text to token IDs via the proven gemma4 BPE (parse_special=1
    /// so chat-template control surfaces like <start_of_turn> emit as single
    /// IDs). BOS is auto-prepended by the engine (gemma4 forces add_bos=1).
    fn encode(&self, text: &str) -> Result<Vec<i32>, String> {
        // First call sizes the output; the engine returns the full count even
        // when it exceeds the buffer, so we grow-and-retry on truncation.
        let mut cap: usize = text.len() + 16;
        loop {
            let mut out = vec![0i32; cap];
            // SAFETY: text ptr/len describe a valid UTF-8 slice; out has `cap`
            // i32 slots; handle is a valid read-only sp_tokenizer.
            let n = unsafe {
                sp_tokenizer_encode(
                    self.handle,
                    text.as_ptr() as *const c_char,
                    text.len(),
                    /*parse_special=*/ 1,
                    out.as_mut_ptr(),
                    cap as c_int,
                )
            };
            if n < 0 {
                return Err("sp_tokenizer_encode returned -1".to_string());
            }
            let n = n as usize;
            if n <= cap {
                out.truncate(n);
                return Ok(out);
            }
            // Truncated — the return value is the true count; resize and retry.
            cap = n;
        }
    }
}

impl Drop for CTokenizer {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: handle was produced by sp_tokenizer_load_tokfile and is
            // freed exactly once (Drop runs once; handle nulled implicitly).
            unsafe { sp_tokenizer_free(self.handle) };
        }
    }
}

// SAFETY: see CTokenizer doc — the handle is read-only after load.
unsafe impl Send for CTokenizer {}
unsafe impl Sync for CTokenizer {}

// ── Message / TemplateError ────────────────────────────────────────────────

#[derive(serde::Deserialize, Debug, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug)]
pub struct TemplateError {
    pub arch_id: u32,
}

impl std::fmt::Display for TemplateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "no chat template for arch_id={}", self.arch_id)
    }
}

// ── SPTB parser ────────────────────────────────────────────────────────────

struct SptbData {
    type_id: u32,
    vocab: Vec<String>,
    merges: Vec<(String, String)>,
}

fn read_u32(buf: &[u8], pos: &mut usize) -> Option<u32> {
    let bytes: [u8; 4] = buf.get(*pos..*pos + 4)?.try_into().ok()?;
    *pos += 4;
    Some(u32::from_le_bytes(bytes))
}

fn read_str(buf: &[u8], pos: &mut usize) -> Option<String> {
    let len = read_u32(buf, pos)? as usize;
    let s = std::str::from_utf8(buf.get(*pos..*pos + len)?).ok()?.to_string();
    *pos += len;
    Some(s)
}

fn parse_sptb(blob: &[u8]) -> Result<SptbData, String> {
    let mut pos = 0usize;

    let magic = read_u32(blob, &mut pos).ok_or("SPTB: truncated magic")?;
    if magic != 0x42545053 {
        return Err(format!("SPTB: bad magic 0x{magic:08X}"));
    }
    let type_id    = read_u32(blob, &mut pos).ok_or("SPTB: truncated type_id")?;
    let vocab_size = read_u32(blob, &mut pos).ok_or("SPTB: truncated vocab_size")? as usize;
    let n_merges   = read_u32(blob, &mut pos).ok_or("SPTB: truncated n_merges")? as usize;

    let mut vocab = Vec::with_capacity(vocab_size);
    for i in 0..vocab_size {
        vocab.push(read_str(blob, &mut pos)
            .ok_or_else(|| format!("SPTB: truncated vocab[{i}]"))?);
    }

    // SentencePiece: skip f32 scores that follow vocab
    if type_id == 0 {
        let skip = vocab_size * 4;
        if pos + skip > blob.len() {
            return Err("SPTB: truncated scores section".to_string());
        }
        pos += skip;
    }

    let mut merges = Vec::with_capacity(n_merges);
    for i in 0..n_merges {
        let s = read_str(blob, &mut pos)
            .ok_or_else(|| format!("SPTB: truncated merge[{i}]"))?;
        let sep = s.find(' ')
            .ok_or_else(|| format!("SPTB: merge[{i}] has no space separator"))?;
        merges.push((s[..sep].to_string(), s[sep + 1..].to_string()));
    }

    Ok(SptbData { type_id, vocab, merges })
}

// ── GPT2 byte-level inverse mapping ───────────────────────────────────────

fn gpt2_char_to_byte(c: char) -> Option<u8> {
    let cp = c as u32;
    match cp {
        0x0021..=0x007E => Some(cp as u8),
        0x00A1..=0x00AC => Some(cp as u8),
        0x00AE..=0x00FF => Some(cp as u8),
        0x0100..=0x0120 => Some((cp - 0x0100) as u8),
        0x0121           => Some(0x7F),
        0x0122..=0x0142 => Some((cp - 0x0122 + 0x80) as u8),
        0x0143           => Some(0xAD),
        _ => None,
    }
}

fn gpt2_decode(token_str: &str) -> Vec<u8> {
    token_str.chars().filter_map(gpt2_char_to_byte).collect()
}

/// ADR-013: is the gemma4-MoE thought channel armed? (profile `[decode] thinking`,
/// env SP_THINKING). Read once — it is a boot-time property of the served config, and
/// the prompt shape and the suppression set MUST agree on it: an open channel with a
/// banned closer reasons forever, a closed channel with an emittable closer leaks a
/// stray token. One answer, both callers.
pub fn thinking_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("SP_THINKING").as_deref() == Ok("1"))
}

// ── generation_config.json (model-intended suppress/eos) ────────────────────

/// CONTRACT-CHAT-FULLSTACK S1 — load `suppress_tokens` + `eos_token_id` from the
/// model's `generation_config.json` if it sits beside the tokenizer/model.
/// Returns `(suppress_ids, eos_ids)`; both empty when the file is absent or
/// unparsable (we fall back to the id-agnostic rules). This is the authoritative
/// model-intended generation contract (the contract requires honoring it); the
/// gemma-4 file carries `suppress_tokens:[258883,258882]` (the audio/image soft
/// tokens) and `eos_token_id:1` (`<eos>`).
fn load_generation_config(tok_path: &str) -> (Vec<i32>, Vec<i32>) {
    let dir = std::path::Path::new(tok_path)
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_default();
    let cfg = dir.join("generation_config.json");
    let text = match std::fs::read_to_string(&cfg) {
        Ok(t) => t,
        Err(_) => return (Vec::new(), Vec::new()),
    };
    let v: serde_json::Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(_) => return (Vec::new(), Vec::new()),
    };
    let mut suppress = Vec::new();
    if let Some(arr) = v.get("suppress_tokens").and_then(|x| x.as_array()) {
        for x in arr {
            if let Some(i) = x.as_i64() {
                suppress.push(i as i32);
            }
        }
    }
    let mut eos = Vec::new();
    match v.get("eos_token_id") {
        Some(serde_json::Value::Number(n)) => {
            if let Some(i) = n.as_i64() {
                eos.push(i as i32);
            }
        }
        Some(serde_json::Value::Array(arr)) => {
            for x in arr {
                if let Some(i) = x.as_i64() {
                    eos.push(i as i32);
                }
            }
        }
        _ => {}
    }
    (suppress, eos)
}

// ── EOS detection ──────────────────────────────────────────────────────────

fn find_eos_ids(arch_id: u32, vocab: &[String]) -> Vec<i32> {
    let names: &[&str] = match arch_id {
        ARCH_QWEN3 | ARCH_QWEN25 | ARCH_QWEN36 => &["<|im_end|>", "<|endoftext|>"],
        // gemma4 (arch_id=7) shares gemma3's turn markers. The live daemon
        // logged eos_ids=[] for arch_id=7, so decode never stopped on EOS —
        // issue #115 fixes that here.
        a if is_gemma_turn_arch(a) => &["<eos>", "<end_of_turn>"],
        _ => &[],
    };
    vocab.iter().enumerate()
        .filter_map(|(id, tok)| {
            if names.contains(&tok.as_str()) { Some(id as i32) } else { None }
        })
        .collect()
}

// ── SptbTokenizer ──────────────────────────────────────────────────────────

pub struct SptbTokenizer {
    inner: Option<Tokenizer>,
    /// Issue #115: the engine C gemma4 BPE encoder, loaded from the same
    /// .sp-tokenizer path. `Some` only for the GEMMA4_BPE lane (type_id=4);
    /// `encode` prefers it when present. Decode stays on the Rust id_to_bytes
    /// path (the ▁→space mapping already decodes gemma4 correctly).
    c_tokenizer: Option<CTokenizer>,
    id_to_bytes: Vec<Vec<u8>>,
    pub eos_ids: Vec<i32>,
    pub arch_id: u32,
    type_id: u32,
    /// id-agnostic `<bos>` id (for stripping the per-fragment forced BOS when
    /// assembling a multi-fragment chat prompt). `None` if absent.
    bos_id: Option<i32>,
    /// CONTRACT-CHAT-FULLSTACK S1 — token ids from the model's own
    /// `generation_config.json` `suppress_tokens` (the authoritative
    /// model-intended suppression, e.g. the gemma-4 image/audio soft tokens
    /// 258882/258883). Empty when no config file sits beside the model. These are
    /// UNIONed into `suppress_token_ids()` on top of the id-agnostic control
    /// rule (which already covers them, but the config is the authoritative
    /// source the contract requires us to honor).
    config_suppress: Vec<i32>,
}

impl SptbTokenizer {
    /// `tok_path` is the .sp-tokenizer file the daemon was started with
    /// (--tokenizer / SP_TOKENIZER_PATH). For the gemma4 lane (type_id=4) it is
    /// re-opened through the engine's proven C encoder; for the BPE/SPM lanes it
    /// is unused (the Rust `tokenizers`/id_to_bytes paths handle them).
    pub fn build(model: &SpModel, arch_id: u32, tok_path: &str) -> Result<Arc<Self>, String> {
        let blob = model.tokenizer_blob()?;
        let data = parse_sptb(blob)?;

        let is_bpe = data.type_id == TYPE_BPE_GPT2 || data.type_id == TYPE_BPE_LLAMA3;
        let is_gemma4 = data.type_id == TYPE_GEMMA4_BPE;

        let inner = if is_bpe {
            let vocab_map: AHashMap<String, u32> = data.vocab.iter().enumerate()
                .map(|(i, s)| (s.clone(), i as u32))
                .collect();

            let bpe = BPE::builder()
                .vocab_and_merges(vocab_map, data.merges.clone())
                .build()
                .map_err(|e| format!("BPE::build: {e}"))?;

            let mut tokenizer = Tokenizer::new(bpe);
            tokenizer.with_pre_tokenizer(Some(ByteLevel::new(false, true, true)));

            // Add tokens that look like special markers so they aren't split
            let specials: Vec<AddedToken> = data.vocab.iter()
                .filter(|tok| tok.starts_with('<') && tok.ends_with('>'))
                .map(|tok| AddedToken::from(tok.as_str(), true))
                .collect();
            if !specials.is_empty() {
                tokenizer.add_special_tokens(&specials);
            }

            Some(tokenizer)
        } else {
            None
        };

        let id_to_bytes: Vec<Vec<u8>> = if is_bpe {
            data.vocab.iter().map(|s| gpt2_decode(s)).collect()
        } else {
            // SentencePiece: ▁ (U+2581) represents a word-boundary space
            data.vocab.iter().map(|s| s.replace('\u{2581}', " ").into_bytes()).collect()
        };

        // CONTRACT-CHAT-FULLSTACK S1 — read the model's own generation_config.json
        // (`suppress_tokens` + `eos_token_id`) if it sits beside the model. The
        // authoritative, config-driven source the contract requires; the
        // id-agnostic control rule below remains the backstop (it already covers
        // these). Absent file ⇒ empty, relies on the backstop.
        let (config_suppress, mut config_eos) = load_generation_config(tok_path);

        let mut eos_ids = find_eos_ids(arch_id, &data.vocab);
        // Union the config's eos_token_id (e.g. <eos>=1) if present.
        config_eos.retain(|e| (*e as usize) < data.vocab.len());
        for e in &config_eos {
            if !eos_ids.contains(e) {
                eos_ids.push(*e);
            }
        }
        // id-agnostic <bos> (gemma4 forces add_bos; strip the per-fragment copy
        // when assembling a multi-fragment chat prompt).
        let bos_id = data.vocab.iter().position(|t| t == "<bos>").map(|i| i as i32);
        // CONTRACT-CHAT-FULLSTACK S1: gemma-4's REAL turn terminator on this
        // artifact is the single token `<turn|>` (the SPTK header's eos_token),
        // NOT a `<end_of_turn>` string (which does not exist in this vocab). Add
        // it to eos_ids so decode stops cleanly at the turn boundary by ID, the
        // same robust path as `<eos>`. (turn_stop_ids covers it too — belt &
        // braces.) Derived id-agnostically; harmless if already present.
        if is_gemma_turn_arch(arch_id) {
            if let Some(end) = data.vocab.iter().position(|t| t == "<turn|>") {
                let end = end as i32;
                if !eos_ids.contains(&end) {
                    eos_ids.push(end);
                }
            }
        }

        // Issue #115: the gemma4 lane routes encode through the engine's proven
        // C BPE encoder. Load it from the same .sp-tokenizer path and sanity-
        // check the vocab size against the parsed blob (catches a path mismatch).
        let c_tokenizer = if is_gemma4 {
            let c = CTokenizer::load(tok_path)?;
            let cvs = c.vocab_size() as usize;
            if cvs != data.vocab.len() {
                return Err(format!(
                    "gemma4 C tokenizer vocab {cvs} != blob vocab {} (path mismatch?: {tok_path})",
                    data.vocab.len()
                ));
            }
            Some(c)
        } else {
            None
        };

        Ok(Arc::new(SptbTokenizer {
            inner,
            c_tokenizer,
            id_to_bytes,
            eos_ids,
            arch_id,
            type_id: data.type_id,
            bos_id,
            config_suppress,
        }))
    }

    pub fn encode(&self, text: &str) -> Result<Vec<i32>, String> {
        // Issue #115: gemma4 uses the engine C encoder (the Rust `tokenizers`
        // crate cannot drive the 514k-merge U+2581-piece BPE).
        if let Some(c) = self.c_tokenizer.as_ref() {
            return c.encode(text);
        }
        let tok = self.inner.as_ref()
            .ok_or_else(|| format!("encode not supported for tokenizer type_id={}", self.type_id))?;
        let enc = tok.encode(text, false)
            .map_err(|e| format!("tokenizer encode: {e}"))?;
        Ok(enc.get_ids().iter().map(|&id| id as i32).collect())
    }

    /// CONTRACT-CHAT-FULLSTACK S1 — encode `text` WITHOUT the auto-prepended BOS.
    /// The engine C gemma4 encoder FORCES add_bos=1 (llama PR #21500), so when we
    /// assemble a chat prompt from multiple fragments we must strip the spurious
    /// per-fragment BOS (id `bos_id`); only the single leading BOS at the head of
    /// the assembled prompt is correct. For the non-gemma4 lanes the Rust path
    /// already encodes without BOS, so this is a pass-through there.
    fn encode_no_bos(&self, text: &str) -> Result<Vec<i32>, String> {
        let ids = self.encode(text)?;
        if self.c_tokenizer.is_some() {
            // gemma4 C encoder prepends BOS — drop exactly one leading BOS.
            if let Some(bos) = self.bos_id {
                if ids.first() == Some(&bos) {
                    return Ok(ids[1..].to_vec());
                }
            }
        }
        Ok(ids)
    }

    /// CONTRACT-CHAT-FULLSTACK S1 — the id of a single-token control surface,
    /// looked up id-agnostically from the decoded vocab bytes (robust to id
    /// shifts across artifacts). Returns `None` if this artifact has no such
    /// token (e.g. a gemma family that genuinely lacks turn markers).
    fn control_id(&self, surface: &[u8]) -> Option<i32> {
        self.id_to_bytes.iter().position(|b| b == surface).map(|i| i as i32)
    }

    /// ADR-013: the same lookup, for callers outside this module (the sampler needs
    /// `<channel|>` to enforce a minimum thought). Same id-agnostic rule, one impl.
    pub fn control_id_pub(&self, surface: &[u8]) -> Option<i32> {
        self.control_id(surface)
    }

    pub fn decode_token(&self, id: i32) -> &[u8] {
        match self.id_to_bytes.get(id as usize) {
            Some(b) => b,
            None => &[],
        }
    }

    /// Default stop strings for this arch's chat format, used when the request
    /// supplies none. Issue #115: this gemma4 artifact's vocab has no
    /// `<end_of_turn>` *token* (the turn terminator is the literal text
    /// `<end_of_turn>`, which BPE-splits into several pieces), so EOS-by-id
    /// cannot catch it — the stop-string path on `TokenDecodeBuffer` does.
    /// `<eos>` (id 1) stays covered by `eos_ids`.
    ///
    /// CONTRACT-CHAT-FULLSTACK A2-polish: this artifact ALSO has no
    /// `<end_of_turn>` token at all (its turn markers are `<|turn>`/`<turn|>`,
    /// caught id-side by `turn_stop_ids`). Belt-and-braces, we add structural
    /// stop-strings that catch the model running INTO the next turn — a fresh
    /// role prefix after a blank line (the chat template emits
    /// `<start_of_turn>model\n…`; an over-running model tends to start a new
    /// `Question:`/`User:`/`<start_of_turn>` block). Kept CONSERVATIVE (only
    /// patterns that begin a *new* turn, never mid-answer punctuation) so a
    /// genuine answer is not truncated.
    pub fn default_stops(&self) -> Vec<String> {
        match self.arch_id {
            a if is_gemma_turn_arch(a) => vec![
                "<end_of_turn>".to_string(),
                "<start_of_turn>".to_string(),
                "<|turn>".to_string(),
                "<turn|>".to_string(),
                "\n\nQuestion:".to_string(),
                "\nUser:".to_string(),
                "\n\nUser".to_string(),
                "\nQuestion:".to_string(),
                // CONTRACT-CHAT-FULLSTACK S1 — this OK_Q4B QAT artifact has a weak
                // turn terminator: it often answers correctly then, instead of
                // emitting the `<turn|>` STOP token, runs on by opening a NEW turn
                // in PLAIN TEXT — most commonly the bare role line `\nuser\n` /
                // `\nmodel\n` (the gemma turn-prefix written as text), observed
                // empirically in the S1 transcripts. Catch those new-turn openers
                // as stop-strings (a fresh role line begins the next turn, so the
                // answer is complete). Kept to the role-line pattern (never
                // mid-answer punctuation) so a genuine answer is not truncated.
                "\nuser\n".to_string(),
                "\nmodel\n".to_string(),
                "\nUser\n".to_string(),
                "\nModel\n".to_string(),
            ],
            _ => Vec::new(),
        }
    }

    /// CONTRACT-CHAT-FULLSTACK A2-polish — token ids that terminate the turn,
    /// treated as EOS-equivalent for this arch. This dense reference model-b1 artifact has
    /// NO `<end_of_turn>` token; its turn markers are the single tokens
    /// `<|turn>` (id 105) and `<turn|>` (id 106). When the model emits one we
    /// stop the turn cleanly (before the marker ever decodes into the stream),
    /// the same way `eos_ids` stops on `<eos>`. Computed id-agnostically from
    /// `id_to_bytes` so it survives id shifts across artifacts.
    pub fn turn_stop_ids(&self) -> Vec<i32> {
        match self.arch_id {
            a if is_gemma_turn_arch(a) => {
                let mut out = Vec::new();
                for (id, bytes) in self.id_to_bytes.iter().enumerate() {
                    if bytes == b"<|turn>" || bytes == b"<turn|>" {
                        out.push(id as i32);
                    }
                }
                out
            }
            _ => Vec::new(),
        }
    }

    /// CONTRACT-CHAT-FULLSTACK A2-polish — every control / placeholder token id
    /// the text-chat sampler must never emit, derived ID-AGNOSTICALLY from the
    /// decoded vocab bytes so it is robust to id shifts across artifacts.
    ///
    /// This dense reference model-b1 artifact's control markers fall in two clusters
    /// (`<|tool>`/`<|turn>`/`<turn|>`/`<|think|>`/`<|channel>`… at ids 46–106,
    /// and `<|image>`/`<image|>`/`<audio|>`/`<|video|>`… at ids 255 999–258 884).
    /// They all share the discriminating pattern: the decoded bytes **start with
    /// `<`, end with `>`, AND contain a pipe `|`**. That pattern catches every
    /// pipe-wrapped control marker (19 ids on this artifact) while excluding
    /// legitimate HTML-style text tokens (`<div>`, `<html>` — no pipe) and code
    /// fragments (`|<`, `|</` — don't start with `<`). It is verified to have
    /// ZERO false positives on the b1 vocab.
    ///
    /// In addition we suppress the named structural specials that are never
    /// valid TEXT output — `<pad>`, `<bos>`, `<unk>`, `<mask>`, and the
    /// `<unusedN>` reserve — matched by EXACT decoded bytes (not the broad
    /// `<…>` rule, which would wrongly catch `<div>`-style text). `<eos>` is
    /// deliberately left OUT here (it is the legitimate end-of-sequence, handled
    /// by `eos_ids`; suppressing it would prevent clean stopping).
    ///
    /// Masking is to `-inf` in the sampler (standard suppress-tokens / bad-words
    /// practice). Applied on BOTH the sampled and the greedy (temp=0) served
    /// path so greedy is usable too; a `raw_logits` request opts OUT to recover
    /// the un-suppressed null-floor reference (the B1 determinism leg).
    pub fn suppress_token_ids(&self) -> Vec<i32> {
        // ── THE TOKEN THAT ENDS THE THOUGHT (2026-07-29) ─────────────────────────
        // `<channel|>` is pipe-wrapped, so the id-agnostic control rule below bans it
        // — correctly, while thinking is off. With the channel left OPEN for thinking,
        // banning it means the model can never CLOSE its thought: it would reason
        // until max_tokens, every turn, and never reach an answer. So when (and only
        // when) thinking is armed on this arch, `<channel|>` is emittable. Everything
        // else stays banned, including `<|channel>` — she may finish a thought, she
        // may not open a second one.
        let allow_close = self.arch_id == ARCH_GEMMA4_MOE && thinking_enabled();
        let mut out = Vec::new();
        for (id, bytes) in self.id_to_bytes.iter().enumerate() {
            if is_suppressed_control(bytes) {
                if allow_close && bytes.as_slice() == b"<channel|>" {
                    continue;
                }
                out.push(id as i32);
            }
        }
        // CONTRACT-CHAT-FULLSTACK S1: UNION the model's own generation_config
        // `suppress_tokens` (authoritative source). The id-agnostic rule above
        // already covers the gemma-4 soft tokens, so this is usually a no-op, but
        // it is the config-driven path the contract requires and survives any
        // artifact whose soft tokens don't match the pipe-control surface rule.
        for &id in &self.config_suppress {
            if (id as usize) < self.id_to_bytes.len() && !out.contains(&id) {
                out.push(id);
            }
        }
        out
    }

    pub fn apply_template(&self, messages: &[Message]) -> Result<String, TemplateError> {
        match self.arch_id {
            ARCH_QWEN3 | ARCH_QWEN25 | ARCH_QWEN36 => Ok(chatml_template(messages)),
            // gemma4 (arch_id=7) uses the identical <start_of_turn>…<end_of_turn>
            // format as gemma3 (issue #115 — the `messages` path now works).
            a if is_gemma_turn_arch(a) => Ok(gemma3_template(messages)),
            _ => Err(TemplateError { arch_id: self.arch_id }),
        }
    }

    /// CONTRACT-CHAT-FULLSTACK S1 — build the chat prompt directly as token IDs.
    ///
    /// THE BUG this fixes: the gemma-4 turn structure on this artifact uses the
    /// SINGLE control tokens `<|turn>` (id 105, start-of-turn) and `<turn|>`
    /// (id 106, end-of-turn) — there is NO `<start_of_turn>`/`<end_of_turn>`
    /// token in the vocab (a full vocab scan finds neither; the only turn tokens
    /// are `<|turn>`/`<turn|>`). The old text template emitted the literal
    /// strings `<start_of_turn>…<end_of_turn>`, which the gemma4 BPE encoder
    /// (whose blob lane has `n_spec==0`, so it special-splits NOTHING) shattered
    /// into ordinary text pieces (`<`,`start`,`_of`,`_turn`,`>`,…). The model
    /// therefore NEVER saw a real turn token — it was prompted in a format it was
    /// never trained on, so rank-1 rode a suppressed soft token and the answer
    /// hung on a thin, FP-reorderable rank-2 margin (the coherent↔garbage flip).
    ///
    /// The fix assembles the prompt at the TOKEN level: the real control ids are
    /// emitted directly (looked up id-agnostically from the vocab — `<|turn>`,
    /// `<turn|>`, `\n`, leading `<bos>`), and only the role label + message
    /// CONTENT go through the proven C BPE encoder (with the spurious per-fragment
    /// forced BOS stripped). This reproduces gemma-4's trained turn format
    /// exactly:  <bos> (<|turn>{role}\n{content}<turn|>\n)*  <|turn>model\n
    ///
    /// If this artifact genuinely lacks the turn tokens, falls back to the text
    /// template + plain encode (minimal correct format + `<eos>` stop).
    pub fn apply_template_ids(&self, messages: &[Message]) -> Result<Vec<i32>, String> {
        match self.arch_id {
            a if is_gemma_turn_arch(a) => {
                // Real turn tokens, id-agnostic. (start_of_turn ≡ <|turn>, 105;
                // end_of_turn ≡ <turn|>, 106 on the b1 artifact.)
                let (start_turn, end_turn, nl) = match (
                    self.control_id(b"<|turn>"),
                    self.control_id(b"<turn|>"),
                    self.control_id(b"\n"),
                ) {
                    (Some(s), Some(e), Some(n)) => (s, e, n),
                    _ => {
                        // No turn tokens — fall back to the text template (minimal
                        // correct format); decode still stops on <eos>.
                        let text = gemma3_template(messages);
                        return self.encode(&text);
                    }
                };
                let mut out: Vec<i32> = Vec::new();
                if let Some(bos) = self.bos_id {
                    out.push(bos);
                }
                for msg in messages {
                    let role = if msg.role == "assistant" { "model" } else { msg.role.as_str() };
                    out.push(start_turn);
                    // role label + newline as content (no special tokens here)
                    out.extend(self.encode_no_bos(role)?);
                    out.push(nl);
                    out.extend(self.encode_no_bos(&msg.content)?);
                    out.push(end_turn);
                    out.push(nl);
                }
                // generation prompt: <|turn>model\n
                out.push(start_turn);
                out.extend(self.encode_no_bos("model")?);
                out.push(nl);
                // ADR-013: CLOSE THE THOUGHT CHANNEL on gemma4-MoE.
                //
                // Gemma-4-26B-A4B is a THINKING model. Its chat template renders
                // reasoning into a channel:
                //     '<|channel>thought\n' + thinking_text + '\n<channel|>'
                // and the generation prompt decides whether that channel is INVITED
                // (left open) or SUPPRESSED (opened and closed immediately):
                //     '<|turn>model\n' + '<|channel>thought\n<channel|>'
                //
                // We emitted only '<|turn>model\n', so the model opened its own thought
                // channel and reasoned in the open — the console showed
                // "<thought\nThinking Process:\n\n1. **Analyze the Request:** ..." and
                // never reached the answer inside the token budget. At ~6 tok/s decode
                // that reasoning is also the most expensive thing on the turn.
                //
                // So emit the CLOSED form. Strictly arch-11 gated: the dense reference model shares this
                // vocab (the tokens exist there too) but does not use channels, and
                // adding them to its prompt would change the daily driver's behaviour
                // and its G-VERBATIM baseline for no reason.
                //
                // ── AND THINKING IS NOW A KNOB, NOT A VERDICT (2026-07-29) ────────
                // Closing the channel unconditionally threw away the one thing this
                // model has that the dense reference model does not. SP_THINKING=1 (profile
                // [decode] thinking) leaves the channel OPEN instead: the model
                // reasons inside it and closes it itself with `<channel|>`, which the
                // gateway splits on to separate thought from reply. Default OFF, so
                // the null floor is exactly the closed form above.
                if self.arch_id == ARCH_GEMMA4_MOE {
                    match (self.control_id(b"<|channel>"), self.control_id(b"<channel|>")) {
                        (Some(ch_open), Some(ch_close)) => {
                            out.push(ch_open);
                            out.extend(self.encode_no_bos("thought")?);
                            out.push(nl);
                            if !thinking_enabled() {
                                out.push(ch_close);
                            }
                        }
                        // Absent channel tokens: leave the prompt as-is rather than
                        // guess. The model reasons aloud, which is visible and
                        // debuggable — unlike a half-formed channel, which is not.
                        _ => {}
                    }
                }
                Ok(out)
            }
            // Non-gemma archs: assemble text then encode (BOS handled by the lane).
            ARCH_QWEN3 | ARCH_QWEN25 | ARCH_QWEN36 => self.encode(&chatml_template(messages)),
            _ => Err(format!("no chat template for arch_id={}", self.arch_id)),
        }
    }
}

/// A2 helper: true if `hay` contains `needle` as a contiguous subslice.
#[allow(dead_code)]
fn contains_subslice(hay: &[u8], needle: &[u8]) -> bool {
    if needle.is_empty() || needle.len() > hay.len() {
        return needle.is_empty();
    }
    hay.windows(needle.len()).any(|w| w == needle)
}

/// A2-polish helper: true if `bytes` is a `<unusedN>` reserve token (the
/// gemma reserved-id block), matched structurally (`<unused` … `>`) so the
/// whole reserve range is suppressed regardless of count. These are never
/// valid text output.
fn is_unused_reserve(bytes: &[u8]) -> bool {
    bytes.starts_with(b"<unused") && bytes.last() == Some(&b'>')
}

/// A2-polish — the single per-token classifier for `suppress_token_ids`,
/// extracted so the rule is unit-testable without a loaded vocab. A token's
/// decoded bytes are a suppressed control/placeholder marker iff:
///   (1) PIPE-WRAPPED control marker: starts `<`, ends `>`, contains `|`
///       (catches `<|turn>`/`<turn|>`/`<image|>`/`<audio|>`/`<|video|>`/
///        `<|tool>`/`<|think|>`/`<|channel>`… — every pipe-marked control id),
///       while EXCLUDING legit HTML-ish text (`<div>`/`<html>` — no pipe) and
///       code fragments (`|<`, `|</` — don't start with `<`); OR
///   (2) a NAMED structural special never valid as text (`<pad>`/`<bos>`/
///       `<unk>`/`<mask>` or the `<unusedN>` reserve), matched exactly.
/// `<eos>` is intentionally NOT here (legit end-of-sequence; handled by eos_ids).
fn is_suppressed_control(bytes: &[u8]) -> bool {
    // CONTRACT-CHAT-FULLSTACK S1 — the TURN tokens `<|turn>`/`<turn|>` (gemma-4's
    // start/end-of-turn, ids 105/106 on the b1 artifact) are NOT suppressed: the
    // end token `<turn|>` is THE turn terminator (the SPTK header's eos_token), so
    // suppressing it (it otherwise matches the pipe-control rule) would mask -inf
    // the only token that lets the model STOP — the model then answers correctly
    // but runs on forever (the no-self-stop bug). They are handled by the turn-
    // stop / eos_ids path in the decode loop (caught before they reach the
    // stream), exactly like `<eos>`. Excluded here the same way `<eos>` is.
    if matches!(bytes, b"<|turn>" | b"<turn|>") {
        return false;
    }
    let pipe_control = bytes.first() == Some(&b'<')
        && bytes.last() == Some(&b'>')
        && bytes.contains(&b'|');
    let named_special =
        matches!(bytes, b"<pad>" | b"<bos>" | b"<unk>" | b"<mask>") || is_unused_reserve(bytes);
    pipe_control || named_special
}

#[cfg(test)]
mod a2_polish_tests {
    use super::*;

    #[test]
    fn suppresses_pipe_control_markers() {
        // The actual the dense reference model control markers (verified by vocab scan).
        // NB: <|turn>/<turn|> are NOT here — S1 excludes the turn terminators
        // from suppression (they are the stop tokens; see does_not_suppress_turn).
        for t in [
            b"<image|>".as_slice(),
            b"<audio|>",
            b"<|image>",
            b"<|audio>",
            b"<|image|>",
            b"<|video|>",
            b"<|tool>",
            b"<tool|>",
            b"<|think|>",
            b"<|channel>",
            b"<channel|>",
        ] {
            assert!(is_suppressed_control(t), "should suppress {:?}", String::from_utf8_lossy(t));
        }
    }

    #[test]
    fn suppresses_named_specials_and_reserve() {
        for t in [b"<pad>".as_slice(), b"<bos>", b"<unk>", b"<mask>", b"<unused0>", b"<unused255>"] {
            assert!(is_suppressed_control(t), "should suppress {:?}", String::from_utf8_lossy(t));
        }
    }

    #[test]
    fn does_not_suppress_turn() {
        // S1: the turn terminators are the STOP tokens; suppressing them is the
        // no-self-stop bug. They must NOT be in the suppress set.
        for t in [b"<|turn>".as_slice(), b"<turn|>"] {
            assert!(!is_suppressed_control(t), "must NOT suppress turn token {:?}", String::from_utf8_lossy(t));
        }
    }

    #[test]
    fn does_not_suppress_text_or_eos() {
        // <eos> is legit end-of-sequence (eos_ids owns it); never suppress it.
        // HTML-ish text tokens (no pipe) and code fragments (don't start '<')
        // must survive — zero false positives.
        for t in [
            b"<eos>".as_slice(),
            b"<div>",
            b"<html>",
            b"<br>",
            b"|<",
            b"|</",
            b">|</",
            b"turn",
            b" image",
            b"hello",
            b"",
        ] {
            assert!(!is_suppressed_control(t), "must NOT suppress {:?}", String::from_utf8_lossy(t));
        }
    }
}

// ── Chat templates ─────────────────────────────────────────────────────────

fn chatml_template(messages: &[Message]) -> String {
    let mut out = String::new();
    if messages.first().map(|m| m.role != "system").unwrap_or(true) {
        out.push_str("<|im_start|>system\n<|im_end|>\n");
    }
    for msg in messages {
        out.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", msg.role, msg.content));
    }
    out.push_str("<|im_start|>assistant\n");
    out
}

fn gemma3_template(messages: &[Message]) -> String {
    let mut out = String::new();
    for msg in messages {
        let role = if msg.role == "assistant" { "model" } else { msg.role.as_str() };
        out.push_str(&format!("<start_of_turn>{}\n{}<end_of_turn>\n", role, msg.content));
    }
    out.push_str("<start_of_turn>model\n");
    out
}

// ── TokenDecodeBuffer ──────────────────────────────────────────────────────

pub enum PushResult {
    Emit(Vec<u8>),
    Stopped(Vec<u8>),
}

pub struct TokenDecodeBuffer {
    buf: Vec<u8>,
    stop_bytes: Vec<Vec<u8>>,
    hold: usize,
}

impl TokenDecodeBuffer {
    pub fn new(stop_strings: Vec<String>) -> Self {
        let stop_bytes: Vec<Vec<u8>> = stop_strings.iter()
            .map(|s| s.as_bytes().to_vec())
            .collect();
        let max_len = stop_bytes.iter().map(|s| s.len()).max().unwrap_or(0);
        let hold = max_len.saturating_sub(1);
        TokenDecodeBuffer { buf: Vec::new(), stop_bytes, hold }
    }

    pub fn push(&mut self, bytes: &[u8]) -> PushResult {
        self.buf.extend_from_slice(bytes);

        for stop in &self.stop_bytes {
            if let Some(pos) = memmem(&self.buf, stop) {
                let before = self.buf[..pos].to_vec();
                self.buf.clear();
                return PushResult::Stopped(before);
            }
        }

        if self.buf.len() <= self.hold {
            return PushResult::Emit(vec![]);
        }

        let candidate_end = self.buf.len() - self.hold;
        let emit_end = valid_utf8_up_to(&self.buf[..candidate_end]);
        if emit_end == 0 {
            return PushResult::Emit(vec![]);
        }

        let emit = self.buf[..emit_end].to_vec();
        self.buf.drain(..emit_end);
        PushResult::Emit(emit)
    }

    /// Flush remaining buffer bytes (call on stream end or cancel).
    pub fn flush(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.buf)
    }
}

fn memmem(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

fn valid_utf8_up_to(buf: &[u8]) -> usize {
    match std::str::from_utf8(buf) {
        Ok(_) => buf.len(),
        Err(e) => e.valid_up_to(),
    }
}
