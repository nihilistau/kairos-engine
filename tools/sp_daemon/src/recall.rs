//! CONTRACT-CHAT-FULLSTACK B3 — AUTONOMOUS MEMORY RECALL.
//!
//! The C2 curator's discrete bit-collision resolver, ported to Rust so the
//! daemon can compute a chat turn's query signature ON ITS OWN and match it
//! against the episode registry — the model "remembers" without an
//! operator-specified `replay`.
//!
//! The signature is the SIGN of the ±1 LSH projection (SP_ARM_PROJ_SEED via
//! splitmix64) of the per-position GLOBAL-owner K, meaned over the global layers
//! and prefilled positions, packed to 256 bits. Match = XOR + popcount (integer
//! Hamming); accept iff bit-agreement >= TAU_BITS. This is byte-for-byte the math
//! of `tools/curator/discrete_resolve.py` (SEED, R_BITS, hd(), the splitmix64
//! stream, the sign-binarize, the agreement count) so the daemon's live query
//! sig is directly comparable to the registry sigs that script writes.
//!
//! Null floor: this module is only reached when a chat turn sets
//! `auto_recall:true` and the registry loaded. Default off ⇒ byte-untouched.

use std::path::Path;

pub const SEED: u64 = 0x5350_524F_4A2B;
pub const R_BITS: usize = 256;
// ── RUNTIME GEOMETRY (ADR-013). These were `const`, pinned to the dense reference model, and
// they size every C2 recall buffer in this daemon. On a model with a different
// global-layer count or head geometry the engine would write past those buffers —
// that overflow was a real illegal memory access, and the engine now carries a
// hard guard against it (gemma4_kv_read_global_q). Deriving them from the LOADED
// model is what lets that guard stand down.
//
// Defaults are the dense reference model values, so a daemon that never calls publish_geometry()
// behaves byte-identically to before.
use std::sync::atomic::{AtomicUsize, Ordering};
static G_HD_A: AtomicUsize = AtomicUsize::new(512);
static G_NL_A: AtomicUsize = AtomicUsize::new(48);
static G_PERIOD_A: AtomicUsize = AtomicUsize::new(6);
static G_NHEAD_A: AtomicUsize = AtomicUsize::new(16);
static G_L5_A: AtomicUsize = AtomicUsize::new(5);
// ── THE ROW IS NOT hd() (2026-08-04) ────────────────────────────────────────────────
// `gemma4_kv_read_global_k` writes (g_nkv * g_hd) floats per (layer, position) — its copy
// loop is literally `const int g_kvd = s->g_nkv * s->g_hd`. On the dense reference model g_nkv was 1, so
// g_hd ALONE happened to be the row width and every caller sized buffers from hd().
// On this model g_nkv is 2: those buffers are HALF what the engine writes, and the D2H
// runs off the end of the allocation. cuda_forward.cu:6620 records that as the illegal
// access (700) that poisoned the CUDA context and surfaced three calls later as an
// unrelated "byteexact_set: d_bx_probe H2D".
//
// ONE site was fixed by asking the engine (kv::global_kvd). SIX were not. Guarding them
// one at a time is how one gets missed — so the row width is published here, beside the
// rest of the geometry, and every caller reads it from one place.
static G_NKV_A: AtomicUsize = AtomicUsize::new(1);   // the dense reference model default: row == g_hd

#[inline] pub fn hd() -> usize { G_HD_A.load(Ordering::Relaxed) }
#[inline] pub fn nl() -> usize { G_NL_A.load(Ordering::Relaxed) }
#[inline] pub fn period() -> usize { G_PERIOD_A.load(Ordering::Relaxed) }
#[inline] pub fn g_nh() -> usize { G_NHEAD_A.load(Ordering::Relaxed) }
/// KV heads on a global layer. 1 on the dense reference model, 2 on gemma4-26b-a4b.
#[inline] pub fn g_nkv() -> usize { G_NKV_A.load(Ordering::Relaxed) }
/// THE ROW WIDTH read_global_k actually writes: g_nkv * g_hd. Size every global-K
/// buffer from THIS, never from hd(). See the note at G_NKV_A.
#[inline] pub fn g_kvd() -> usize { g_nkv() * hd() }
#[inline] pub fn n_global() -> usize { let p = period(); if p == 0 { 0 } else { nl() / p } }

/// Which global layer carries the fact signal. On the dense reference model this is 5 of 8, MEASURED
/// (G-REP-LAYER-L5: exact->paraphrase recall@1 85.2%; averaging all 8 dilutes to
/// 11.5%). That is an empirical result about ONE model and it does NOT transfer:
/// a 5-global-layer model has no index 5 at all. We map proportionally (5/8 of
/// depth) so the dense reference model keeps its measured value EXACTLY and anything else gets a
/// principled starting point that is explicitly UNVALIDATED until swept.
#[inline] pub fn l5_layer() -> usize { G_L5_A.load(Ordering::Relaxed) }

/// A comparability tag for the geometry. Two models agree on C2/L5 keys only if
/// this string matches: an ep.l5 minted over 8x512 is NOT comparable to one over
/// 5x512 even though both are 512 floats, and nothing in the raw sidecar format
/// would have caught that.
pub fn geom_tag() -> String {
    format!("g{}x{}h{}L{}", n_global(), hd(), g_nh(), l5_layer())
}

/// Publish the loaded model's geometry. Call once, at daemon boot, before any
/// recall path runs.
pub fn publish_geometry(n_layers: usize, swa_period: usize, global_hd: usize, n_head: usize) {
    publish_geometry_kv(n_layers, swa_period, global_hd, n_head, 0)
}

/// As `publish_geometry`, plus the global KV-head count that fixes the ROW WIDTH.
/// `global_nkv == 0` leaves the previous value, so old callers keep the dense reference model's 1.
pub fn publish_geometry_kv(n_layers: usize, swa_period: usize, global_hd: usize,
                           n_head: usize, global_nkv: usize) {
    if global_nkv > 0 { G_NKV_A.store(global_nkv, Ordering::Relaxed); }
    if n_layers == 0 || swa_period == 0 || global_hd == 0 || n_head == 0 { return; }
    G_HD_A.store(global_hd, Ordering::Relaxed);
    G_NL_A.store(n_layers, Ordering::Relaxed);
    G_PERIOD_A.store(swa_period, Ordering::Relaxed);
    G_NHEAD_A.store(n_head, Ordering::Relaxed);
    let ng = n_layers / swa_period;
    // 5-of-8 in proportion, clamped into range. ng==8 -> 5 (the measured value).
    let l5 = if ng == 0 { 0 } else { std::cmp::min(ng - 1, (ng * 5) / 8) };
    G_L5_A.store(l5, Ordering::Relaxed);
}
pub const TAU_BITS: u32 = 168; // discrete_resolve.py default gate radius

/// splitmix64 ±1 stream — identical to discrete_resolve.py `smix` / build_R.
fn smix(seed: u64, n: usize) -> Vec<i8> {
    let mut s = seed;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        out.push(if z & 1 != 0 { 1 } else { -1 });
    }
    out
}

/// The frozen ±1 router R, shape [R_BITS][hd()] row-major (matches build_R).
pub struct Projection {
    r: Vec<i8>, // R_BITS*hd()
}

impl Projection {
    pub fn build() -> Self {
        Projection { r: smix(SEED, R_BITS * hd()) }
    }

    /// Compute the 256-bit signature from packed global-layer K
    /// (`[n_global][npos][hd()]` row-major f32, the gemma4_kv_read_global_k layout).
    /// sig[b] = sign( mean over (global layer, pos) of (R[b] · K[L,p]) ).
    /// Returns the 256-bit signature as 4 little-endian-bit u64 words (bit i of
    /// word w = sig bit 64*w+i), matching the hex packing in discrete_resolve.py.
    pub fn signature(&self, global_k: &[f32], n_global: usize, npos: usize) -> [u64; 4] {
        self.signature_strided(global_k, n_global, npos, g_kvd())
    }

    /// `row` is the width of one (layer, position) row as the ENGINE writes it —
    /// g_nkv*g_hd. The projection matrix is R_BITS x hd(), so it reads the first hd()
    /// floats of each row: on the dense reference model row == hd() and this is byte-identical to what it
    /// always did; on a g_nkv>1 model it takes the first KV head instead of straddling
    /// two rows, which is what `v * hd()` did once the buffers were sized correctly.
    ///
    /// SIGNATURES DO NOT CROSS GEOMETRIES and never did — cuda_forward.cu says so at the
    /// mint guard. Episodes keyed on the dense reference's 8x512 are not comparable to anything minted
    /// here, whatever stride is used; that incomparability is pre-existing and is NOT what
    /// this fix is about. This fix is the heap overflow.
    pub fn signature_strided(&self, global_k: &[f32], n_global: usize, npos: usize,
                             row: usize) -> [u64; 4] {
        let row = if row == 0 { hd() } else { row };
        debug_assert_eq!(global_k.len(), n_global * npos * row);
        let n_vec = (n_global * npos) as f64;
        // For each of the R_BITS projection rows, accumulate R[b]·K over all
        // (layer,pos) vectors, then take the mean's sign. Accumulate in f64 to
        // keep the daemon's argmax stable; the SIGN is what is binarized so this
        // matches the float projmean()>0 of the reference.
        let mut acc = vec![0.0f64; R_BITS];
        for v in 0..(n_global * npos) {
            let kbase = v * row;
            for b in 0..R_BITS {
                let rbase = b * hd();
                let mut dot = 0.0f64;
                for d in 0..hd() {
                    dot += (self.r[rbase + d] as f64) * (global_k[kbase + d] as f64);
                }
                acc[b] += dot;
            }
        }
        let mut words = [0u64; 4];
        for b in 0..R_BITS {
            let mean = acc[b] / n_vec;
            if mean > 0.0 {
                words[b / 64] |= 1u64 << (b % 64);
            }
        }
        words
    }
}

/// MEM-OKF v2 per-entry policy (ADR-004): how THIS memory wants to be delivered/declined.
/// Loaded from the registry row's optional `mem_class`/`mem_delivery`/`mem_decline_*` fields.
/// `None` on an episode ⇒ fall back to the global env path (SP_MEM_POLICY null floor).
#[derive(Clone, Debug)]
pub struct MemPolicy {
    pub class: String,
    pub delivery: String,             // systemecho | attr-gate-strict | recite | system | plain | ...
    pub decline_when: Vec<String>,    // attribute-absent | zero-inference | family-ambiguous | low-margin
    pub decline_message: String,
}

/// The proven class → default delivery mapping (mirror of okf_mem.py CLASS_DEFAULT_DELIVERY;
/// two-stage REFUTED by G-MEMPOLICY-V3 so same-template defaults to systemecho).
pub fn class_default_delivery(class: &str) -> &'static str {
    match class {
        "private-secret" => "attr-gate-strict",
        "counterfact" | "same-template" => "systemecho",
        // A remembered thing is CONTEXT, not a command. "fact" is now the DEFAULT class
        // (see classify_mem_class), so its delivery must be the GENTLE one — a note she
        // may use. It used to fall through to `_ => "recite"`, whose framing is
        //     "Every answer must repeat the relevant part of the fact on record verbatim."
        // Between that and counterfact's "authoritative, overrides prior knowledge", EVERY
        // recall was an order. That is why she recited a memory when the operator asked
        // "what do you mean?" instead of just talking to him.
        "fact" | "preference" | "persona" | "episodic-event" => "system",
        _ => "recite",
    }
}

/// ONE VOCABULARY, ONE DOOR (Tier 3, INVARIANT-ROADMAP.md; gate: G-MEMCLASS §6): the
/// LIVE class->delivery map arrives from THE registry (harness/skills/memclass.py) via
/// SP_MEM_CLASS_DELIVERY — serve.py serializes it at boot, so the engine consumes the
/// same artifact the Python sites import instead of trusting its compiled twin. The
/// compiled match above stays as the FALLBACK (daemon launched outside serve.py), and
/// G-MEMCLASS pins that fallback to the registry AT THE SOURCE — so neither the live
/// map nor the fallback can drift silently. This is the fix-shape for the incident
/// where fact->system landed in one copy of three.
pub fn class_delivery(class: &str) -> String {
    use std::sync::OnceLock;
    static MAP: OnceLock<std::collections::HashMap<String, String>> = OnceLock::new();
    let m = MAP.get_or_init(|| {
        std::env::var("SP_MEM_CLASS_DELIVERY").ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    });
    m.get(class).cloned()
        .unwrap_or_else(|| class_default_delivery(class).to_string())
}

impl MemPolicy {
    /// Build a policy from a class alone (the class defaults + the private-secret decline).
    /// Used by the NIGHTSHIFT auto-classifier at capture (#73).
    pub fn from_class(class: &str) -> MemPolicy {
        let delivery = class_delivery(class);
        let (decline_when, decline_message) = if class == "private-secret" {
            (vec!["zero-inference".to_string(), "attribute-absent".to_string()],
             "I have a record for that entity, but it does not include that specific detail.".to_string())
        } else {
            (Vec::new(), String::new())
        };
        MemPolicy { class: class.to_string(), delivery, decline_when, decline_message }
    }
}

/// #72 UNIFY: write a conformant MEM-OKF v2 concept sidecar for a live episode (OKF
/// frontmatter + body) so the engine's episode dir IS a MEM-OKF object the harness/okf_mem
/// can read — the two stores speak one format. Read back by load_episode_okf_policy.
pub fn write_episode_okf(dir: &std::path::Path, text: &str, class: &str, name: &str) {
    let pol = MemPolicy::from_class(class);
    let authority = match class {
        "private-secret" => "private",
        "persona" | "preference" | "fact" | "episodic-event" => "supplements",
        _ => "overrides-prior",
    };
    let decl = if pol.decline_when.is_empty() {
        String::new()
    } else {
        format!("mem_decline_when: [{}]\nmem_decline_message: {}\n",
                pol.decline_when.join(", "), pol.decline_message)
    };
    let title: String = text.chars().take(60).collect::<String>().replace('\n', " ");
    let fm = format!(
        "---\ntype: memory\ntitle: {title}\ndescription: {title}\ntags: [mem-okf, episode, {class}]\n\
         sp_status: ACTIVE\nmem_kind: episode\nmem_addr: {name}\nmem_class: {class}\n\
         mem_delivery: {delivery}\nmem_authority: {authority}\n{decl}---\n\n{text}\n",
        title = title, class = class, name = name, delivery = pol.delivery,
        authority = authority, decl = decl, text = text);
    let _ = std::fs::write(dir.join("ep.okf.md"), fm);
}

/// NIGHTSHIFT auto-classification (#73): assign a mem_class from the captured text so a live
/// episode SELF-GOVERNS (instead of loading policy=None → env). Deterministic, no model call:
/// a high-entropy code token or a secret keyword ⇒ `private-secret` (attr-gate-strict decline,
/// entity-guard-safe); otherwise a user-asserted memory is delivered authoritatively ⇒
/// `counterfact` (systemecho). Coarse by design; finer classes (persona/preference/episodic)
/// are a follow-on. Gated by SP_MEM_CLASSIFY at the call site (default-off = no mem_class).
pub fn classify_mem_class(text: &str) -> &'static str {
    let low = text.to_lowercase();
    let secret_kw = ["code", "password", "passcode", "passphrase", " pin ", " pin.",
                     "secret", "api key", "api-key", "token", "override"];
    let has_kw = secret_kw.iter().any(|k| low.contains(k));
    // high-entropy token: len>=5 with >=2 letters AND >=2 digits, or a hyphenated alnum code.
    let has_code = text.split(|c: char| c.is_whitespace()).any(|tok| {
        let t = tok.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '-');
        let letters = t.chars().filter(|c| c.is_ascii_alphabetic()).count();
        let digits = t.chars().filter(|c| c.is_ascii_digit()).count();
        let hyphenated = t.contains('-') && (letters + digits) >= 4;
        (t.len() >= 5 && letters >= 2 && digits >= 2) || (hyphenated && t.len() >= 5)
    });
    if has_kw || has_code { return "private-secret"; }
    // #74 finer classes: first-person IDENTITY / PREFERENCE -> persona (system delivery).
    let persona_kw = ["my name is", "i am ", "i'm ", "call me ", "i like ", "i love ",
                      "i prefer ", "i live ", "my favorite", "my favourite", "i enjoy ",
                      "my birthday", "i work as", "my job is", "my hobby"];
    if persona_kw.iter().any(|k| low.contains(k)) { return "persona"; }
    // ── THE DEFAULT WAS THE MOST AGGRESSIVE DELIVERY IN THE SYSTEM (2026-07-12) ──
    // This used to return "counterfact" — and `counterfact` is the class that gets
    // delivered as:
    //     "Fact on record (AUTHORITATIVE for this conversation, OVERRIDES PRIOR
    //      KNOWLEDGE): {fact}. Answer from this fact."
    // That framing exists for facts that must beat the model's world knowledge ("in this
    // world the sky is green"). It is not a framing for "Sam mentioned his cat".
    //
    // Because it was the DEFAULT, 99 of 131 live memories carried it. So nearly every
    // recall arrived as an ORDER TO RECITE, and the operator saw exactly that in live
    // play: he asked "what do you mean?" and she answered by reciting an unrelated
    // memory verbatim, prefixed "Fact on record (authoritative...)". The retrieval was
    // mediocre; the FRAMING is what made her obey it instead of just talking.
    //
    // A remembered thing is CONTEXT, not a command. The default is now a plain fact —
    // delivered as a note she may use, not an instruction she must recite. `counterfact`
    // remains available for genuine overrides, but nothing falls into it by accident.
    "fact"
}

/// #72 UNIFY: read this episode's MEM-OKF policy from a conformant OKF concept sidecar
/// (`<dir>/ep.okf.md` frontmatter) if present — the engine consumes the SAME OKF-frontmatter
/// format the harness/okf_mem writes. `None` ⇒ caller falls back to the inline registry-row.
pub fn load_episode_okf_policy(dir: &str) -> Option<MemPolicy> {
    let p = std::path::Path::new(dir).join("ep.okf.md");
    let text = std::fs::read_to_string(&p).ok()?;
    parse_okf_policy(&text)
}

/// Parse a MEM-OKF policy from an OKF concept's YAML frontmatter string. `None` if no mem_class.
pub fn parse_okf_policy(text: &str) -> Option<MemPolicy> {
    let (mut class, mut delivery, mut dm) = (None, None, String::new());
    let mut dw: Vec<String> = Vec::new();
    let mut in_fm = false;
    for line in text.lines() {
        if line.trim() == "---" { if in_fm { break; } else { in_fm = true; continue; } }
        if !in_fm { continue; }
        if let Some((k, v)) = line.split_once(':') {
            let (k, v) = (k.trim(), v.trim());
            match k {
                "mem_class" => class = Some(v.to_string()),
                "mem_delivery" => delivery = Some(v.to_string()),
                "mem_decline_when" => dw = v.trim_matches(|c| c == '[' || c == ']')
                    .split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect(),
                "mem_decline_message" => dm = v.to_string(),
                _ => {}
            }
        }
    }
    let class = class?;
    let delivery = delivery.unwrap_or_else(|| class_default_delivery(&class).to_string());
    Some(MemPolicy { class, delivery, decline_when: dw, decline_message: dm })
}

/// The markdown body of an OKF concept (everything after the closing frontmatter `---`).
/// For a MEM-OKF memory this is the delivery text. Strips a leading "The user said: " if
/// present (engine-written concepts attribute the payload). Falls back to the whole string.
pub fn okf_body(text: &str) -> String {
    let mut fences = 0;
    let mut body = String::new();
    for line in text.lines() {
        if line.trim() == "---" && fences < 2 { fences += 1; continue; }
        if fences >= 2 { body.push_str(line); body.push('\n'); }
    }
    let body = if fences >= 2 { body } else { text.to_string() };
    body.trim().to_string()
}

/// One registry episode: its replay path, position count, topic, and 256-bit sig.
#[derive(Clone, Debug)]
pub struct Episode {
    pub name: String,
    pub dir: String,
    pub npos: i32,
    pub topic: String,
    /// JUDGE-SERVED: the episode's full entry text (the manifest `text` field) — the
    /// surface the verifier shows as a candidate and checks citations against. "" if
    /// unavailable (live/un-manifested episodes ⇒ the verifier abstains on them).
    pub text: String,
    pub sig: [u64; 4],
    /// B3-v2: the episode's stored GLOBAL-owner K (from ep.k), packed
    /// `[n_global][npos][hd()]` row-major, global layers ascending, ONLY the real
    /// prompt positions [0,npos). Empty if ep.k was unavailable at load. This is
    /// the memory the live query is scored against by q·K attention relevance.
    pub gk: Vec<f32>,
    pub gk_ng: usize,
    /// B4 NIGHTSHIFT: the raw token ids of a LIVE-captured episode. `None` for
    /// disk/curated episodes (recalled via `kv::replay(dir)`); `Some(toks)` for
    /// turn-end NIGHTSHIFT episodes (recalled via `kv::inject_tokens(&toks)` — no
    /// ep.k/ep.v files on disk). Constructed at position-0 standalone capture so it
    /// is W_c-head-compatible (same provenance as the curated registry-K).
    pub tokens: Option<Vec<i32>>,
    /// L5 RECALL (2026-07-01): the episode's stored QUERY-key = the L2-normalized,
    /// mean-over-heads global-layer-5 last-token query embedding (512 f32) of the
    /// query/statement that defines this episode. Loaded from `<dir>/ep.l5` (raw
    /// little-endian f32[512]) if present, else empty. Matched against the live
    /// query's `l5_query_embed(read_global_q)` by cosine — the query-to-query
    /// selector that recalls paraphrases (G-REP-LAYER-L5: 88.5% para vs Jaccard 8%).
    pub l5key: Vec<f32>,
    /// MEM-OKF v2 / ADR-004: this entry's own delivery/decline policy (from the
    /// registry row's mem_class/mem_delivery/mem_decline_*). `None` ⇒ global env path.
    pub policy: Option<MemPolicy>,
    /// A1 (SP_MEM_LIFECYCLE): 0 = active, 1 = superseded (kept for audit, excluded from
    /// the live recall set). Non-destructive supersede replaces the old delete.
    pub lifecycle: u8,
}

// the dense reference model global attention head geometry (g_nkv=1, g_nh=16, g_hd=512 ⇒ g_kvd=hd()).
// g_nh() is now g_nh() above (runtime).

/// Per-episode q·K attention-relevance reductions, the B3-v2 selector. `q` is the
/// live query's last-token global-layer query, packed `[n_global][g_nh()*hd()]`
/// (read_global_q layout). `gk` is the episode's stored global-K `[n_global][npos][hd()]`.
/// For each global layer and each of g_nh() query heads (GQA: all share the 1 KV head),
/// the attention pre-softmax score at position p is `q_head · K[p]`. We summarise the
/// `[layer, head, pos]` score tensor two ways:
///   - `max`  = the single strongest q·K over all (layer, head, pos) — the peak match.
///   - `topm` = the mean of the top-`m` scores (per the whole tensor) — a robust
///              "the query attends to several of these positions" relevance.
/// Both are returned (the gate reports both); the daemon ranks on `topm`. Scores are
/// scaled by 1/sqrt(hd()) (the attention temperature) so they read in the usual logit
/// range. Returns (max, topm). A layer/geometry mismatch yields (NEG_INFINITY, _).
pub fn qk_relevance(q: &[f32], gk: &[f32], gk_ng: usize, npos: usize, m: usize) -> (f32, f32) {
    if gk.is_empty() || gk_ng == 0 || npos == 0 { return (f32::NEG_INFINITY, f32::NEG_INFINITY); }
    let qd = g_nh() * hd();
    let n_global_q = q.len() / qd;
    let ng = n_global_q.min(gk_ng);
    if ng == 0 { return (f32::NEG_INFINITY, f32::NEG_INFINITY); }
    let scale = 1.0f64 / (hd() as f64).sqrt();
    // v9h: SP_B3_QK_COSINE=1 ⇒ L2-normalize q and K per (head,position) so the score is the
    // ANGLE (cosine ∈ [-1,1]), stripping the K-NORM gravity-well that let a high-energy episode
    // (audio/p33) win q·K on EVERY query regardless of semantic direction (the N-sweep confound).
    // Default off = raw q·K/√d (byte-identical null floor).
    let cosine = std::env::var("SP_B3_QK_COSINE").ok().as_deref() == Some("1");
    let mut best = f64::NEG_INFINITY;
    // Collect the top-m scores in a tiny ascending min-heap (m is small).
    let mut top: Vec<f64> = Vec::with_capacity(m + 1);
    for l in 0..ng {
        let qbase = l * qd;
        let kbase_l = l * npos * hd();
        for p in 0..npos {
            let kbase = kbase_l + p * hd();
            for h in 0..g_nh() {
                let qh = qbase + h * hd();
                let mut dot = 0.0f64;
                let mut qn = 0.0f64;
                let mut kn = 0.0f64;
                for d in 0..hd() {
                    let qv = q[qh + d] as f64;
                    let kv = gk[kbase + d] as f64;
                    dot += qv * kv;
                    if cosine { qn += qv * qv; kn += kv * kv; }
                }
                let s = if cosine {
                    let den = qn.sqrt() * kn.sqrt();
                    if den > 0.0 { dot / den } else { 0.0 }
                } else {
                    dot * scale
                };
                if s > best { best = s; }
                // maintain top-m
                if top.len() < m {
                    top.push(s);
                    if top.len() == m { top.sort_by(|a, b| a.partial_cmp(b).unwrap()); }
                } else if s > top[0] {
                    top[0] = s;
                    // re-sift the smallest to the front (m tiny ⇒ a sort is fine)
                    top.sort_by(|a, b| a.partial_cmp(b).unwrap());
                }
            }
        }
    }
    let topm = if top.is_empty() {
        best
    } else {
        top.iter().sum::<f64>() / (top.len() as f64)
    };
    (best as f32, topm as f32)
}

/// L5 RECALL: the global layer index carrying the fact signal (offline sweep
/// G-REP-LAYER-L5: L5 exact→paraphrase recall@1 = 85.2%; averaging all 8 global
/// layers dilutes to 11.5%). The read_global_q packing is global-layers-ascending,
/// so index 5 here == offline `[8,16,512]` layer 5.
// l5_layer() is now l5_layer() above (runtime, proportional).

/// Extract the L5 query key from a live `read_global_q` buffer `q` packed
/// `[n_global][g_nh()*hd()]`: take global layer L5, mean over the g_nh() query heads,
/// L2-normalize → a 512-f32 direction. Empty if the layer is absent.
pub fn l5_query_embed(q: &[f32]) -> Vec<f32> {
    let qd = g_nh() * hd();
    let n_global = q.len() / qd;
    if n_global <= l5_layer() { return Vec::new(); }
    let base = l5_layer() * qd;
    let mut acc = vec![0.0f64; hd()];
    for h in 0..g_nh() {
        let hb = base + h * hd();
        for d in 0..hd() { acc[d] += q[hb + d] as f64; }
    }
    let inv = 1.0 / (g_nh() as f64);
    let mut norm = 0.0f64;
    for d in 0..hd() { acc[d] *= inv; norm += acc[d] * acc[d]; }
    let den = norm.sqrt();
    if den <= 0.0 { return Vec::new(); }
    acc.iter().map(|&x| (x / den) as f32).collect()
}

/// CONVERSATIONAL PRE-CHECK (SP_RECALL_QONLY): deterministic "is this turn a
/// memory query at all?" test. G-ONECONFIG-LIVE run-1 (RUNBOOK-ONE-CONFIG §8)
/// showed the in-registry L5 cosine background is ≥0.9, so a conversational
/// STATEMENT ("My designation is X. Please remember that.") still matches some
/// episode and gets an irrelevant fact injected, derailing the turn. A turn is
/// interrogative iff it contains '?' OR starts with an interrogative/imperative
/// recall verb. Purely lexical, no forward; default-off = null floor.
pub fn is_interrogative(q: &str) -> bool {
    if q.contains('?') { return true; }
    const HEADS: &[&str] = &[
        "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
        "is", "are", "was", "were", "am", "do", "does", "did", "can", "could",
        "should", "would", "will", "shall", "have", "has", "had",
        "name", "tell", "state", "give", "list", "recall", "remember", "recite",
        "identify", "describe", "explain",
    ];
    let first = q.trim().split_whitespace().next().unwrap_or("");
    let f: String = first.chars().filter(|c| c.is_alphanumeric()).collect::<String>().to_lowercase();
    HEADS.contains(&f.as_str())
}

/// Cosine similarity of two 512-vectors (robust to un-normalized input).
pub fn cos512(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() { return f32::NEG_INFINITY; }
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..a.len() {
        let (x, y) = (a[i] as f64, b[i] as f64);
        dot += x * y; na += x * x; nb += y * y;
    }
    let den = na.sqrt() * nb.sqrt();
    if den <= 0.0 { 0.0 } else { (dot / den) as f32 }
}

/// L5 RECALL: read `<dir>/ep.l5` = the episode's stored L5 query-key, raw
/// little-endian f32[512]. `None` if the sidecar is absent or malformed.
/// The geometry an ep.l5 sidecar was minted under. Absent file == LEGACY == the
/// the dense reference model geometry, which is what every existing sidecar in the tree was
/// minted with. Written by mint; checked here.
pub fn l5_sidecar_tag(dir: &str) -> String {
    std::fs::read_to_string(Path::new(dir).join("ep.l5.geom"))
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|_| "g8x512h16L5".to_string())
}

/// Record the geometry alongside a freshly minted ep.l5.
pub fn write_l5_sidecar_tag(dir: &str) {
    let _ = std::fs::write(Path::new(dir).join("ep.l5.geom"), geom_tag());
}

pub fn load_episode_l5key(dir: &str) -> Option<Vec<f32>> {
    // A key minted over 8 global layers is NOT comparable to one minted over 5,
    // even though both are 512 floats and both load cleanly. The old format
    // carried no model identity at all, so the ONLY check was a length check —
    // and g_hd happens to be 512 on both the dense reference model and the 26B-A4B, so a stale key
    // would have sailed through and produced a plausible cosine score against the
    // wrong manifold. Skipped, not compared: docs/SEMANTICS.md's rule that an
    // embedding is a MODEL ARTIFACT and a mismatch goes COLD, never silently wrong.
    if l5_sidecar_tag(dir) != geom_tag() { return None; }
    let path = Path::new(dir).join("ep.l5");
    let bytes = std::fs::read(path).ok()?;
    if bytes.len() != hd() * 4 { return None; }
    let mut v = Vec::with_capacity(hd());
    for c in bytes.chunks_exact(4) {
        v.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
    }
    Some(v)
}

/// ATTR-GATE (SP_RECALL_ATTR_GATE): deterministic attribute-grounding check.
/// Fraction of the query's salient content words (len>=3, non-stopword) that are
/// ABSENT from the delivered fact text. High ratio => the query asks about an
/// attribute the fact does not state (e.g. fact="override code for Node-X is V",
/// query="manufacturer of Node-X?" -> "manufacturer" absent). The shared entity
/// tokens are present in both, so they lower the ratio; a mismatch surfaces via the
/// distinctive attribute noun(s). Purely lexical (no model forward, no semantics) —
/// it CANNOT tell a paraphrase of the SAME attribute from a DIFFERENT attribute, so
/// it is the hard fallback; the strict closed-book delivery prompt (which lets the
/// model ground with its own semantics) is the paraphrase-safe primary.
pub fn attr_absent_ratio(query: &str, fact: &str) -> f32 {
    fn toks(s: &str) -> Vec<String> {
        s.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| !w.is_empty())
            .map(|w| w.to_string())
            .collect()
    }
    const STOP: &[&str] = &[
        "the", "is", "are", "was", "were", "what", "which", "who", "whom", "whose",
        "of", "for", "and", "or", "does", "do", "did", "how", "many", "much", "that",
        "this", "its", "with", "by", "from", "you", "your", "have", "has", "had",
        "been", "be", "as", "not", "now", "current", "currently", "authoritative",
        "context", "there", "any", "about", "into", "please", "tell", "give", "list",
    ];
    let fw: std::collections::HashSet<String> = toks(fact).into_iter().collect();
    let q: Vec<String> = toks(query)
        .into_iter()
        .filter(|w| w.len() >= 3 && !STOP.contains(&w.as_str()))
        .collect();
    if q.is_empty() { return 0.0; }
    let absent = q.iter().filter(|w| !fw.contains(*w)).count();
    absent as f32 / q.len() as f32
}

/// G-SEL-CANON: salient-token overlap of a CANONICALIZED query vs a fact text,
/// morphology-tolerant (boil/boils): tokens match on equality OR len>=4 prefix
/// containment either way. Used by the canonicalization selector switch — the
/// proper name surfaced by the rewrite appears VERBATIM in exactly the right
/// fact, so this count is decisive precisely where the L5 embed is
/// template-confused (G-SEL-OFFLINE: raw-query Jaccard was inert; the rewrite
/// is what makes the lexical signal informative).
pub fn canon_overlap(canon: &str, fact: &str) -> usize {
    fn toks(s: &str) -> Vec<String> {
        s.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() >= 3)
            .map(|w| w.to_string())
            .collect()
    }
    const STOP: &[&str] = &[
        "the", "is", "are", "was", "were", "what", "which", "who", "whom", "whose",
        "of", "for", "and", "or", "does", "do", "did", "how", "many", "much", "that",
        "this", "its", "with", "by", "from", "have", "has", "had", "been", "be",
        "as", "not", "now", "question", "rewritten", "reply", "answer",
    ];
    let fw: Vec<String> = toks(fact);
    toks(canon).into_iter()
        .filter(|w| !STOP.contains(&w.as_str()))
        .filter(|w| fw.iter().any(|f| f == w
            || (w.len() >= 4 && f.starts_with(w.as_str()))
            || (f.len() >= 4 && w.starts_with(f.as_str()))))
        .count()
}

/// ATTR-GATE paraphrase guard: does the QUERY carry a HIGH-ENTROPY entity token?
/// A rare token = len>=4 AND contains a digit (private-entity IDs / codes:
/// "Node-XX-674B91", "SVC-3F9A2B7C"). General-knowledge / paraphrased chat queries
/// have no such token, so the guard is FALSE for them => the attribute gate never
/// fires on them (recall preserved). A query ABOUT a private entity always names the
/// entity verbatim (an ID cannot be paraphrased) => guard TRUE => the gate is allowed
/// to decline on attribute-absence. Checking the QUERY (not query∩fact) is deliberate:
/// L5 may deliver a wrong-entity fact for a poorly-matching attribute query, but the
/// query still being about a private entity is what warrants grounding. This is what
/// makes the deterministic gate globally default-on-safe instead of regime-specific.
pub fn query_has_entity_token(query: &str) -> bool {
    query
        .split(|c: char| !c.is_alphanumeric())
        .any(|w| w.len() >= 4 && w.chars().any(|c| c.is_ascii_digit()))
}

/// Bit-agreement = R_BITS - HammingDistance (the discrete_resolve.py `agree`).
pub fn agree(a: &[u64; 4], b: &[u64; 4]) -> u32 {
    let mut ham = 0u32;
    for w in 0..4 {
        ham += (a[w] ^ b[w]).count_ones();
    }
    R_BITS as u32 - ham
}

/// Parse a 256-bit lowercase hex string (64 hex chars) into 4 u64 words with the
/// SAME bit order as discrete_resolve.py `packhex` (bit i ⇒ nibble i/4): the hex
/// is x = Σ bit_i << i, written `{x:064x}`. So word w = bits[64w..64w+64].
pub fn parse_sig_hex(hex: &str) -> Option<[u64; 4]> {
    if hex.len() != R_BITS / 4 {
        return None;
    }
    // x is a 256-bit big integer with bit i set ⇒ sig[i]=1. Read it from the hex
    // (MSB-first nibbles) into 4 little words.
    let mut words = [0u64; 4];
    // hex[0] is the most-significant nibble = bits [252..256).
    let bytes = hex.as_bytes();
    for (ci, &c) in bytes.iter().enumerate() {
        let nib = (c as char).to_digit(16)? as u64;
        // nibble at hex index ci covers bits [ (len-1-ci)*4 .. +4 )
        let bit_lo = (R_BITS / 4 - 1 - ci) * 4;
        for k in 0..4 {
            if nib & (1 << k) != 0 {
                let bit = bit_lo + k;
                words[bit / 64] |= 1u64 << (bit % 64);
            }
        }
    }
    Some(words)
}

/// Load the recall registry (JSONL: one episode per line). Tolerant: skips blank /
/// malformed lines, returns the episodes that parsed. Each row needs at minimum
/// `dir`, `npos`, `sig_bits`; `name`/`topic` are decorative.
pub fn load_registry(path: &Path) -> std::io::Result<Vec<Episode>> {
    let text = std::fs::read_to_string(path)?;
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let v: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let dir = v.get("dir").or_else(|| v.get("ring2_path")).and_then(|x| x.as_str());
        let sig_hex = v.get("sig_bits").and_then(|x| x.as_str());
        let npos = v.get("npos").and_then(|x| x.as_i64()).unwrap_or(0) as i32;
        let (dir, sig_hex) = match (dir, sig_hex) {
            (Some(d), Some(s)) => (d, s),
            _ => continue,
        };
        let sig = match parse_sig_hex(sig_hex) {
            Some(s) => s,
            None => continue,
        };
        // B3-v2: load the episode's stored global-owner K (ep.k) for the q·K selector.
        let (gk, gk_ng) = load_episode_global_k(dir, npos).unwrap_or((Vec::new(), 0));
        // L5 RECALL: load the episode's L5 query-key sidecar (ep.l5), if present.
        let l5key = load_episode_l5key(dir).unwrap_or_default();
        // MEM-OKF v2: this entry's own policy, if the row carries mem_class.
        let policy = v.get("mem_class").and_then(|x| x.as_str()).map(|class| {
            let delivery = v.get("mem_delivery").and_then(|x| x.as_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| class_default_delivery(class).to_string());
            let decline_when = v.get("mem_decline_when").and_then(|x| x.as_str())
                .map(|s| s.split(',').map(|t| t.trim().to_string()).filter(|t| !t.is_empty()).collect())
                .unwrap_or_default();
            let decline_message = v.get("mem_decline_message").and_then(|x| x.as_str())
                .unwrap_or("I have a record for that entity, but it does not include that specific detail.")
                .to_string();
            MemPolicy { class: class.to_string(), delivery, decline_when, decline_message }
        });
        // #72 UNIFY: a conformant OKF concept sidecar (<dir>/ep.okf.md) is the AUTHORITATIVE
        // policy source — the engine reads the same OKF frontmatter the store speaks; the inline
        // registry-row mem_class is the fallback.
        let policy = load_episode_okf_policy(dir).or(policy);
        out.push(Episode {
            name: v.get("name").and_then(|x| x.as_str()).unwrap_or("?").to_string(),
            dir: dir.to_string(),
            npos,
            topic: v.get("topic").and_then(|x| x.as_str()).unwrap_or("").to_string(),
            text: v.get("text").and_then(|x| x.as_str()).unwrap_or("").to_string(),
            sig,
            gk,
            gk_ng,
            tokens: None,
            l5key,
            policy,
            lifecycle: v.get("lifecycle").and_then(|x| x.as_u64()).unwrap_or(0) as u8,
        });
    }
    // JUDGE-SERVED: backfill entry text from a sibling corpus_manifest.jsonl (the
    // curator writes `text` there; registry rows historically omit it). Join on the
    // 256-bit sig_bits (unique per needle). Leaves text "" if no manifest / no match.
    if out.iter().any(|e| e.text.is_empty()) {
        if let Some(dir) = path.parent() {
            if let Ok(mtext) = std::fs::read_to_string(dir.join("corpus_manifest.jsonl")) {
                let mut by_sig: std::collections::HashMap<[u64; 4], String> = std::collections::HashMap::new();
                for line in mtext.lines() {
                    let line = line.trim();
                    if line.is_empty() { continue; }
                    if let Ok(mv) = serde_json::from_str::<serde_json::Value>(line) {
                        if let (Some(sb), Some(tx)) = (
                            mv.get("sig_bits").and_then(|x| x.as_str()),
                            mv.get("text").and_then(|x| x.as_str()),
                        ) {
                            if let Some(s) = parse_sig_hex(sb) { by_sig.insert(s, tx.to_string()); }
                        }
                    }
                }
                for e in out.iter_mut() {
                    if e.text.is_empty() {
                        if let Some(tx) = by_sig.get(&e.sig) { e.text = tx.clone(); }
                    }
                }
            }
        }
    }
    Ok(out)
}

/// A1: rewrite the registry JSONL marking (NOT deleting) rows whose "text" == victim with
/// "lifecycle": lc. Non-destructive supersede — keeps the row for audit/rollback.
pub fn mark_superseded_registry(reg_path: &str, victim: &str, lc: u64) -> std::io::Result<()> {
    let text = std::fs::read_to_string(reg_path)?;
    let mut out = String::new();
    for line in text.lines() {
        let t = line.trim();
        if t.is_empty() { continue; }
        match serde_json::from_str::<serde_json::Value>(t) {
            Ok(mut v) => {
                if v.get("text").and_then(|x| x.as_str()) == Some(victim) {
                    if let Some(obj) = v.as_object_mut() { obj.insert("lifecycle".to_string(), serde_json::json!(lc)); }
                    out.push_str(&serde_json::to_string(&v).unwrap_or_else(|_| t.to_string()));
                } else { out.push_str(t); }
            }
            Err(_) => out.push_str(t),
        }
        out.push('\n');
    }
    std::fs::write(reg_path, out)
}

#[cfg(test)]
mod a1_tests {
    use super::*;
    #[test]
    fn supersede_marks_not_deletes() {
        let dir = std::env::temp_dir().join(format!("a1reg_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("registry.jsonl");
        std::fs::write(&p, "{\"text\":\"blue\",\"dir\":\"d\",\"sig_bits\":\"00\"}\n{\"text\":\"green\",\"dir\":\"d\",\"sig_bits\":\"01\"}\n").unwrap();
        mark_superseded_registry(p.to_str().unwrap(), "blue", 1).unwrap();
        let after = std::fs::read_to_string(&p).unwrap();
        assert!(after.contains("\"blue\""), "victim row must be KEPT (non-destructive), got: {}", after);
        assert!(after.contains("\"lifecycle\":1"), "victim must be marked superseded");
        assert!(after.contains("\"green\""), "other row untouched");
        assert_eq!(after.lines().count(), 2, "no rows dropped");
    }
}

/// B3-v2: read `<dir>/ep.k` and extract the GLOBAL-owner K rows `[0,npos)`, packed
/// `[n_global][npos][hd()]` row-major (global layers ascending), matching the
/// `read_global_q`/`gemma4_kv_read_global_k` layout so a live query and the stored
/// memory are directly comparable. ep.k is raw little-endian f32 `[NL][P][hd()]` (the
/// curator's `loadK`: P = filesize / (NL*hd())). Returns (packed_global_K, n_global).
pub fn load_episode_global_k(dir: &str, npos: i32) -> Option<(Vec<f32>, usize)> {
    let path = Path::new(dir).join("ep.k");
    let bytes = std::fs::read(&path).ok()?;
    if bytes.len() % 4 != 0 { return None; }
    let n_f32 = bytes.len() / 4;
    if n_f32 % (nl() * hd()) != 0 && n_f32 < nl() * hd() { return None; }
    let p_total = n_f32 / (nl() * hd());            // floor: capture allocates Pmax slots
    if p_total == 0 { return None; }
    let npos = (npos as usize).min(p_total);
    // global layer indices (L % PERIOD == PERIOD-1).
    let globals: Vec<usize> = (0..nl()).filter(|l| l % period() == period() - 1).collect();
    let ng = globals.len();
    let mut out = vec![0.0f32; ng * npos * hd()];
    for (gi, &l) in globals.iter().enumerate() {
        for p in 0..npos {
            // source f32 index of K[l, p, 0] in the [NL][P][hd()] flat layout.
            let src0 = (l * p_total + p) * hd();
            let dst0 = (gi * npos + p) * hd();
            for d in 0..hd() {
                let b = (src0 + d) * 4;
                out[dst0 + d] = f32::from_le_bytes([bytes[b], bytes[b + 1], bytes[b + 2], bytes[b + 3]]);
            }
        }
    }
    Some((out, ng))
}

/// Match a query sig against the registry. Returns (best_episode_index,
/// best_agreement). The caller applies the TAU_BITS gate (fire only if
/// best_agreement >= TAU_BITS) — kept separate so the daemon can LOG the score
/// for the foreign-reject leg even when it does not fire.
pub fn best_match(query: &[u64; 4], registry: &[Episode]) -> Option<(usize, u32)> {
    let mut best: Option<(usize, u32)> = None;
    for (i, ep) in registry.iter().enumerate() {
        let a = agree(query, &ep.sig);
        match best {
            Some((_, bs)) if a <= bs => {}
            _ => best = Some((i, a)),
        }
    }
    best
}

// ===== JUDGE-SERVED: deterministic token-overlap EVIDENCE verifier (Stage-2) =====
// The proven gate (Exp E, 2026-06-25: thr 0.6 -> recall 83% / reject 95% at N=40,
// beats the 26B diffusion cascade on a CPU string op). Mirrors the validated
// `overlap()`/`toks()` in tools/xbar_lsh/test_battery_d.py byte-for-byte:
//   toks(t)  = lowercased [a-z0-9]+ runs, keep len>=3 and not a stopword
//   overlap  = |evidence_toks ∩ cited_toks| / |evidence_toks|   (0 if no evidence toks)
// ACCEPT a citation iff overlap >= OVERLAP_THR. Probability for generation; unbending
// integer math for validation — the deterministic half of the boundary thesis.
pub const OVERLAP_THR: f32 = 0.6;

/// Stopwords stripped before overlap (matches the Python set). Single/short words are
/// additionally dropped by the len>=3 filter, so "a" need not appear here.
pub const OVERLAP_STOP: &[&str] = &[
    "the", "an", "of", "to", "in", "on", "at", "for", "and", "or", "is", "are",
    "was", "were", "been", "with", "from", "that", "this", "its", "as", "by",
];

/// Content-token list: lowercase, split on any non-[a-z0-9], keep len>=3 non-stopwords.
fn overlap_toks(t: &str) -> Vec<String> {
    t.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .filter(|w| w.len() >= 3 && !OVERLAP_STOP.contains(w))
        .map(|w| w.to_string())
        .collect()
}

/// Fraction of EVIDENCE content-tokens present in the cited entry text. 0.0 if the
/// evidence has no content tokens (un-verifiable => reject). Deterministic, CPU-only.
pub fn token_overlap(evidence: &str, cited: &str) -> f32 {
    let et = overlap_toks(evidence);
    if et.is_empty() {
        return 0.0;
    }
    let cs: std::collections::HashSet<String> = overlap_toks(cited).into_iter().collect();
    let hit = et.iter().filter(|w| cs.contains(*w)).count();
    hit as f32 / et.len() as f32
}

/// Parse the model's `TAG=<tag|NONE> | EVIDENCE=<span>` reply. Returns
/// (Some(tag_index) | None for NONE/no-tag, evidence_span). Mirrors the battery
/// `parse()`: the earliest tag-substring (case-insensitive) wins; NONE wins if it
/// appears before any tag (or no tag is found); EVIDENCE = text after the first
/// `=`/`:` following the word EVIDENCE, de-quoted/trimmed. ASCII-oriented (the tag
/// pool and corpus are ASCII), so case-folding preserves byte offsets.
pub fn parse_tag_evidence(reply: &str, tags: &[String]) -> (Option<usize>, String) {
    let up = reply.to_ascii_uppercase();
    let mut pick: Option<usize> = None;
    let mut ti = usize::MAX;
    for (i, t) in tags.iter().enumerate() {
        if let Some(j) = up.find(&t.to_ascii_uppercase()) {
            if j < ti {
                ti = j;
                pick = Some(i);
            }
        }
    }
    if let Some(jn) = up.find("NONE") {
        if pick.is_none() || jn < ti {
            pick = None;
        }
    }
    let ev = match up.find("EVIDENCE") {
        Some(p) => {
            let rest = &reply[p..];
            match rest.find(['=', ':']) {
                Some(sep) => {
                    // The prompt asks for a SINGLE line. Cut at the first newline so a
                    // rambling decode (repeated "ANSWER:" / a second EVIDENCE=) can't
                    // pollute the span, then strip special-token markers (<audio|>,
                    // <image|>, ...) the decode can interleave — they tokenize to
                    // audio/image and falsely dilute the overlap (live-smoke finding).
                    let raw = &rest[sep + 1..];
                    let line = raw.split('\n').next().unwrap_or(raw);
                    strip_special_tokens(line).trim().trim_matches('"').trim().to_string()
                }
                None => String::new(),
            }
        }
        None => String::new(),
    };
    (pick, ev)
}

/// Remove `<...>` special-token markers (e.g. `<audio|>`, `<image|>`) from a span.
/// Everything from a `<` to the matching `>` is dropped; unbalanced `<` drops the rest.
fn strip_special_tokens(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut depth = 0i32;
    for c in s.chars() {
        match c {
            '<' => depth += 1,
            '>' => { if depth > 0 { depth -= 1; } }
            _ if depth == 0 => out.push(c),
            _ => {}
        }
    }
    out
}

/// JUDGE-SERVED Stage-2 proposal prompt: the V1-skeptical framing that won Exp A
/// (recall 70→90%, the only variant that didn't trade recall) + the TAG/EVIDENCE
/// contract Exp C/D/E proved. `{E}` = the tagged candidate block, `{Q}` = the live
/// user query. The model returns one line `TAG=<tag|NONE> | EVIDENCE=<span>`;
/// parse_tag_evidence + token_overlap @0.6 adjudicate it deterministically.
pub const VERIFY_PROMPT: &str = "You are a STRICT memory index. Each entry has a TAG in [brackets].\n\n{E}\n\nQUESTION: {Q}\n\nMost questions have NO matching entry. Find the entry that directly answers the question; if none does, answer NONE. Then reply on ONE line EXACTLY:\nTAG=<the tag, or NONE> | EVIDENCE=<copy the exact words from that entry that answer it>\nANSWER:";

/// Format a candidate shortlist as the tagged entry block for VERIFY_PROMPT `{E}`.
/// `tags[i]` labels `episodes[i]`, using each episode's `.text` (manifest entry text).
pub fn format_candidates(episodes: &[&Episode], tags: &[String]) -> String {
    episodes
        .iter()
        .zip(tags.iter())
        .map(|(e, t)| format!("[{}] {}", t, e.text))
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_overlap_matches_python_reference() {
        // Paraphrase that draws every content word from the source -> 1.0 (true needle kept).
        let cited_lob = "Homarus gammarus is a species of lobster found in the North Atlantic Ocean";
        assert!((token_overlap("a species of lobster found in the North Atlantic", cited_lob) - 1.0).abs() < 1e-6);
        // Partial overlap: 2 of 3 evidence content-tokens present ("code" absent) -> 0.666...
        let cited_dep = "the Marlock mag-rail depot authorizes on 2-AZURE-6428";
        assert!((token_overlap("Marlock depot code", cited_dep) - 2.0 / 3.0).abs() < 1e-6);
        // No content tokens in evidence -> 0.0 (un-verifiable => reject).
        assert_eq!(token_overlap("", "anything here"), 0.0);
        assert_eq!(token_overlap("the of in", "the cat sat"), 0.0);
        // Threshold semantics: the lobster paraphrase clears 0.6, the partial does too here
        // (0.667 >= 0.6) — the gate's job is to slap down fabrication, not paraphrase.
        assert!(token_overlap("a species of lobster found in the North Atlantic", cited_lob) >= OVERLAP_THR);
    }

    #[test]
    fn parse_tag_evidence_extracts_pick_and_span() {
        let tags: Vec<String> = ["W0C", "J1P", "K7Q"].iter().map(|s| s.to_string()).collect();
        let (p, ev) = parse_tag_evidence("TAG=J1P | EVIDENCE=the marlock depot authorizes", &tags);
        assert_eq!(p, Some(1));
        assert_eq!(ev, "the marlock depot authorizes");
        // NONE before any tag => reject.
        let (p2, _ev2) = parse_tag_evidence("TAG=NONE | EVIDENCE=", &tags);
        assert_eq!(p2, None);
        // Quoted evidence is de-quoted.
        let (_p3, ev3) = parse_tag_evidence("TAG=W0C | EVIDENCE=\"north atlantic ocean\"", &tags);
        assert_eq!(ev3, "north atlantic ocean");
    }

    #[test]
    fn parse_sig_roundtrips_bit_order() {
        // bit 0 set ⇒ least-significant nibble bit 0 ⇒ hex ends "...1".
        let mut h = String::from("0").repeat(63);
        h.push('1');
        let w = parse_sig_hex(&h).unwrap();
        assert_eq!(w[0] & 1, 1);
        assert_eq!(w[0] & 2, 0);
        // bit 255 ⇒ top nibble high bit ⇒ hex starts "8...".
        let mut h2 = String::from("8");
        h2.push_str(&"0".repeat(63));
        let w2 = parse_sig_hex(&h2).unwrap();
        assert_eq!(w2[3] >> 63, 1);
    }

    #[test]
    fn agree_self_is_full() {
        let s = [0xDEAD_BEEFu64, 1, 2, 3];
        assert_eq!(agree(&s, &s), R_BITS as u32);
        let mut t = s;
        t[0] ^= 1;
        assert_eq!(agree(&s, &t), R_BITS as u32 - 1);
    }

    #[test]
    fn smix_first_bits_match_reference() {
        // splitmix64 with SEED — first element sign. Reference: discrete_resolve.py
        // build_R()[0,0]. We only assert determinism here (stream is stable).
        let a = smix(SEED, 8);
        let b = smix(SEED, 8);
        assert_eq!(a, b);
        assert!(a.iter().all(|&x| x == 1 || x == -1));
    }
}

// ===== B3 DEPLOY: learned W_c head — logsumexp-mean reduction + NULL(s0) argmax =====
// The autonomous instance selector (G-CHAT-B3-WC-DIV2: 360/361 recall + 50/50 foreign reject,
// int16-exact). Reduction is logsumexp over positions then mean over (layer,head) — the metric
// the head was trained with and the ONLY one that discriminates (max/top-m collapse to ~12-16/361).
// Reject = the s0 NULL slot in the (E+1)-way argmax (NOT an absolute threshold). Deploy blob =
// wc_deploy.bin (WCB1 magic; see tools/xbar_lsh/export_wc_deploy.py).
pub struct WcHead { pub hd: usize, pub r: usize, pub s0: f32, pub sscale: f32, pub w: Vec<f32> }

pub fn load_wc(path: &str) -> Option<WcHead> {
    let b = std::fs::read(path).ok()?;
    if b.len() < 20 || &b[0..4] != b"WCB1" { return None; }
    let u = |o: usize| u32::from_le_bytes([b[o], b[o+1], b[o+2], b[o+3]]) as usize;
    let fl = |o: usize| f32::from_le_bytes([b[o], b[o+1], b[o+2], b[o+3]]);
    let hd = u(4); let r = u(8); let s0 = fl(12); let sscale = fl(16);
    if b.len() < 20 + hd*r*4 { return None; }
    let mut w = Vec::with_capacity(hd*r);
    let mut o = 20;
    for _ in 0..hd*r { w.push(fl(o)); o += 4; }
    Some(WcHead { hd, r, s0, sscale, w })
}

/// Project v[base..base+hd] through W_c [hd,r] -> out[0..r].
#[inline]
fn wc_proj(v: &[f32], base: usize, w: &[f32], hd: usize, r: usize, out: &mut [f32]) {
    for j in 0..r { let mut s = 0.0f32; for d in 0..hd { s += v[base+d]*w[d*r+j]; } out[j] = s; }
}

/// lse-mean relevance of episode (gk) to the live query (q), projected through head.W_c.
/// q = [n_global][g_nh()*hd()] (read_global_q layout); gk = [n_global][npos*hd()].
pub fn wc_score(q: &[f32], gk: &[f32], gk_ng: usize, npos: usize, head: &WcHead) -> f32 {
    if gk.is_empty() || gk_ng == 0 || npos == 0 { return f32::NEG_INFINITY; }
    let (hd, r) = (head.hd, head.r);
    let qd = g_nh() * hd;
    let ng = (q.len()/qd).min(gk_ng);
    if ng == 0 { return f32::NEG_INFINITY; }
    let mut qp = vec![0.0f32; r];
    let mut sum_a = 0.0f64; let mut cnt = 0usize;
    for l in 0..ng {
        let kbase_l = l*npos*hd;
        let mut kps = vec![0.0f32; npos*r];                       // project all positions once / layer
        for p in 0..npos { wc_proj(gk, kbase_l + p*hd, &head.w, hd, r, &mut kps[p*r..p*r+r]); }
        let qbase = l*qd;
        for h in 0..g_nh() {
            wc_proj(q, qbase + h*hd, &head.w, hd, r, &mut qp);
            let mut mx = f32::NEG_INFINITY;
            let mut sims = vec![0.0f32; npos];
            for p in 0..npos {
                let mut dot = 0.0f32; for j in 0..r { dot += qp[j]*kps[p*r+j]; }
                let s = dot*head.sscale; sims[p] = s; if s > mx { mx = s; }   // stable-LSE max
            }
            let mut se = 0.0f64; for p in 0..npos { se += ((sims[p]-mx) as f64).exp(); }
            sum_a += (mx as f64) + se.ln();                       // logsumexp_p, add max back
            cnt += 1;
        }
    }
    if cnt == 0 { f32::NEG_INFINITY } else { (sum_a/(cnt as f64)) as f32 }   // mean over (l,h)
}
