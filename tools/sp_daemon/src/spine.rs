//! # The Decide→Execute Spine
//!
//! This module is the literal, compiler-enforced realization of **ADR-002 — the
//! Decide→Execute Spine** (`papers/PPT-LAT-ADR-002-DECIDE-EXECUTE-SPINE.md`).
//! It replaces the ad-hoc `if SP_RECALL_L5 { … } else if SP_B3_WC { … } else if …`
//! ladder that grew inside `routes.rs` — a decade of workarounds stacked as env-gated
//! branches — with ONE seam and two roles:
//!
//! * **Tier 1 — Deciders.** Read an *immutable* [`LatentView`] (the query's latent
//!   footprint + the episode stores) and produce a discrete [`LatentDecision`].
//!   A decider CANNOT mutate the KV cache, CANNOT emit a token. The borrow checker
//!   enforces it: `refine` takes `&LatentView` and returns a value — nothing else.
//!
//! * **Tier 2 — the Executor.** Consumes a [`LatentDecision`] and drives the cache +
//!   SSE stream (deliver a fact in-context, decline, route, or pass through to normal
//!   generation). The executor NEVER sees a tensor — it takes the discrete intent only.
//!
//! **The boundary contract (why fusion is structurally impossible):** the only value
//! that crosses Tier-1 → Tier-2 is a `LatentDecision` — an enum of discrete intents,
//! no tensors, no logits. A decider that wanted to "fuse" latent content into
//! generation has no channel to do so: it returns an enum, and the enum carries only
//! an episode *selection* (an index/name), never a hidden state.
//!
//! **The dispatch is a fold.** The deciders form a *priority pipeline*: each one
//! refines the decision-so-far. A selector turns `Pass` into `Deliver`; an
//! attribute-grounding gate turns `Deliver` into `Decline`; a judge veto turns
//! `Deliver` back into `Pass`. Priority == order in the decider list. That single
//! `fold` expresses the whole recall/reject/decline logic that used to sprawl across
//! ~1500 lines.
//!
//! ```text
//!   LatentView ──▶ [ QOnly? L5Recall · AttrGate · JudgeVeto · … ] ──▶ LatentDecision
//!                       (fold: each decider refines the decision)          │
//!                                                                          ▼
//!                                              Executor: Deliver / Decline / Route / Pass
//! ```
//!
//! This first cut wires the LIVE one-config stack (L5 cosine recall → attribute-gate
//! zero-inference decline, QONLY-aware) through the spine and gates it against
//! G-ONECONFIG-LIVE (must reproduce 54/61). Telepathy routing, the generative judge
//! veto, the W_c / Jaccard / INT2 selectors, forget, and operator replay are modelled
//! as additional [`Decider`]s that slot into the SAME fold — the extension points are
//! named and typed so the next port is mechanical, not archaeological.
//!
//! Gated behind `SP_SPINE=1`; unset ⇒ the proven inline path in `routes.rs` runs
//! byte-for-byte unchanged (the null floor).

use sp_daemon::recall::{self, Episode};

/// How a recalled fact is framed for the model when delivered in-context. The
/// winning delivery (`SystemEcho`) puts the fact as SYSTEM authority + primes a
/// verbatim echo (G-DELIVERY-SWEEP: 88.52% obey / 0 leak). Others kept for A/B.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Delivery {
    /// Fact as SYSTEM authority + "repeat the relevant part verbatim" priming,
    /// conversation preserved. The G-ONECONFIG-LIVE default.
    SystemEcho,
    /// Fact as SYSTEM authority, clean user turn (no echo priming).
    System,
    /// Instruction AFTER the question (recency) + explicit override authority.
    Sandwich,
    /// Prime the answer to copy from the fact ("Answer using the fact on record:").
    FactEcho,
    /// The plain recite wording (the original 86.89% receipt path).
    Recite,
}

impl Delivery {
    /// Parse the `SP_RECALL_L5_PROMPT` env value into a delivery framing.
    pub fn from_env(v: &str) -> Delivery {
        match v {
            "system" => Delivery::System,
            "sandwich" => Delivery::Sandwich,
            "factecho" => Delivery::FactEcho,
            "recite" => Delivery::Recite,
            _ => Delivery::SystemEcho,
        }
    }
}

/// A borrow of the episode a decider selected — the discrete "which memory" that
/// crosses the boundary. Carries the *content* (name + text) the executor recites;
/// it does NOT carry any latent tensor. `score` is the selector's confidence
/// (cosine / overlap / ΔLL), kept for telemetry only.
#[derive(Clone, Debug)]
pub struct EpisodeRef {
    pub name: String,
    pub text: String,
    /// Selector confidence ×1000 (cosine, overlap, or collapse), telemetry only.
    pub score_milli: u32,
}

/// The sole currency crossing Tier-1 → Tier-2. Discrete intents only — no tensors,
/// no logits. Every historical selector/gate/route collapses onto one of these.
#[derive(Clone, Debug)]
pub enum LatentDecision {
    /// Null floor: continue normal autoregressive generation, clean prompt.
    Pass,
    /// Recall: deliver `episode`'s text in-context with `framing`, then generate.
    Deliver { episode: EpisodeRef, framing: Delivery },
    /// Zero-inference symbolic decline: stream `message`, run NO forward. Used when
    /// a fact is on record but does not state the queried attribute (SNE shield).
    Decline { message: String },
    /// Telepathy: route the CLEAN user text to bridge `target` and stream back.
    /// (Decider scaffolded; executor lands with the telepathy port.)
    Route { target: u32 },
}

impl LatentDecision {
    /// True once a decision is terminal — no later decider should override it.
    /// `Decline` and `Route` are terminal; `Deliver` can still be vetoed
    /// (attr-gate → Decline, judge → Pass); `Pass` is the open default.
    pub fn is_terminal(&self) -> bool {
        matches!(self, LatentDecision::Decline { .. } | LatentDecision::Route { .. })
    }
}

/// The immutable latent footprint a decider reads. Constructed ONCE per turn after
/// the prompt is prefilled (the single non-committing `read_global_q`), then handed
/// to every decider. No decider can mutate it or the cache through it.
pub struct LatentView<'a> {
    /// The raw last user message (no chat template) — the query text.
    pub raw_user: &'a str,
    /// The live query's RAW global-layer Q (`read_global_q`), packed
    /// `[n_global][G_NH*HD]`. Trainable latent heads (W_c, route, …) read THIS
    /// against each episode's stored global-K. Empty ⇒ the non-committing read failed.
    pub global_q: Vec<f32>,
    /// The live query's L5 embedding (`l5_query_embed(global_q)`), or empty if the
    /// non-committing read was unavailable (⇒ selectors abstain to Pass).
    pub l5_query: Vec<f32>,
    /// The curated episode registry (immutable, startup-loaded).
    pub registry: &'a [Episode],
    /// The live NIGHTSHIFT episodes (between-turn consolidated).
    pub nightshift: &'a [Episode],
    /// Whether this turn's query is interrogative (a question). QONLY uses this to
    /// skip recall on conversational statements.
    pub interrogative: bool,
    /// SP_RECALL_QONLY active — statements skip the L5 stage.
    pub qonly: bool,
    /// L5 cosine threshold (SP_RECALL_L5_TAU, default 0.30).
    pub tau_l5: f32,
    /// L5 top1−top2 margin gate (SP_RECALL_L5_MARGIN, 0 = off).
    pub tau_margin: f32,
    /// Attribute-absence threshold (SP_RECALL_ATTR_TAU, default 0.5).
    pub attr_tau: f32,
    /// SP_RECALL_ATTR_GATE active.
    pub attr_gate: bool,
    /// The delivery framing (SP_RECALL_L5_PROMPT).
    pub framing: Delivery,
}

impl<'a> LatentView<'a> {
    /// Iterate curated ∪ nightshift episodes (the full candidate pool).
    pub fn candidates(&self) -> impl Iterator<Item = &Episode> {
        self.registry.iter().chain(self.nightshift.iter())
    }
}

/// Tier-1: a decider. Reads the latent view + the decision-so-far, returns the
/// refined decision. CANNOT touch the cache or the stream — the signature is the
/// enforcement (`&LatentView` in, `LatentDecision` out; no `&mut` anything).
pub trait Decider {
    /// A stable name for telemetry.
    fn name(&self) -> &'static str;
    /// Refine the decision. Return `current` unchanged to abstain.
    fn refine(&self, view: &LatentView, current: LatentDecision) -> LatentDecision;
}

// ─────────────────────────── concrete deciders (LIVE stack) ───────────────────────────

/// L5 cosine recall selector — the promoted served recall path (G-L5-RECALL-LIVE
/// 86.89% paraphrase; G-ONECONFIG-LIVE systemecho 88.52% obey / 0 leak). Selects the
/// registry∪nightshift episode whose L5 key is closest to the query's L5 embedding;
/// fires if top-1 cosine ≥ τ and (when armed) the top1−top2 margin clears τ_margin.
/// Abstains (leaves `Pass`) on QONLY statements, empty L5 query, or below threshold.
pub struct L5Recall;

impl Decider for L5Recall {
    fn name(&self) -> &'static str { "L5Recall" }
    fn refine(&self, view: &LatentView, current: LatentDecision) -> LatentDecision {
        // Only acts on the open default; never overrides a terminal decision.
        if !matches!(current, LatentDecision::Pass) { return current; }
        // QONLY: conversational statements skip the L5 stage (a paraphrase-safe
        // in-registry background can otherwise inject an irrelevant fact).
        if view.qonly && !view.interrogative {
            // Emit the canonical QONLY-SKIP marker (the served-log evidence the
            // one-config gate greps for) — same behavioral signal as the inline path.
            tracing::info!("SPINE L5Recall: QONLY-SKIP (non-interrogative turn) -> Pass");
            return current;
        }
        if view.l5_query.len() != recall::hd() {
            return current; // non-committing read unavailable -> clean prompt
        }
        // Score every candidate by L5 cosine; keep top-1 and top-2 for the margin.
        let mut best: Option<(f32, &Episode)> = None;
        let mut second = f32::NEG_INFINITY;
        for ep in view.candidates() {
            if ep.l5key.len() != recall::hd() { continue; }
            let c = recall::cos512(&view.l5_query, &ep.l5key);
            match best {
                Some((b, _)) if c <= b => { if c > second { second = c; } }
                _ => { if let Some((b, _)) = best { second = b; } best = Some((c, ep)); }
            }
        }
        let (bcos, ep) = match best { Some(x) => x, None => return current };
        let margin = if second.is_finite() { bcos - second } else { f32::INFINITY };
        tracing::info!("SPINE L5Recall: top1='{}' cos={:.4} margin={:.4} tau={:.3}", ep.name, bcos, margin, view.tau_l5);
        if bcos < view.tau_l5 {
            return current; // below threshold -> clean prompt
        }
        if view.tau_margin > 0.0 && margin < view.tau_margin {
            tracing::info!("SPINE L5Recall: MARGIN-skip ({:.4} < {:.4}) -> Pass", margin, view.tau_margin);
            return current;
        }
        LatentDecision::Deliver {
            episode: EpisodeRef { name: ep.name.clone(), text: ep.text.clone(), score_milli: (bcos * 1000.0) as u32 },
            framing: view.framing,
        }
    }
}

/// Attribute-grounding gate — the SNE shield. If a fact was selected for delivery
/// but the query asks for an attribute the fact does NOT state (and the query carries
/// a private-entity token), convert `Deliver` → `Decline` (zero-inference symbolic
/// decline: no forward ⇒ confabulation/leak mathematically impossible,
/// G-SNE-ATTRGATE-ZEROINF). Paraphrase-safe: general-knowledge queries lack the
/// entity token, so the gate stays off for them (recall preserved).
pub struct AttrGate;

impl Decider for AttrGate {
    fn name(&self) -> &'static str { "AttrGate" }
    fn refine(&self, view: &LatentView, current: LatentDecision) -> LatentDecision {
        if !view.attr_gate { return current; }
        let ep = match &current { LatentDecision::Deliver { episode, .. } => episode, _ => return current };
        let absent = recall::attr_absent_ratio(view.raw_user, &ep.text);
        let force = absent >= view.attr_tau && recall::query_has_entity_token(view.raw_user);
        if force {
            tracing::info!("SPINE AttrGate: '{}' absent={:.2} -> DECLINE (zero-inference)", ep.name, absent);
            LatentDecision::Decline {
                message: "I have a record for that entity, but it does not include that specific detail.".to_string(),
            }
        } else {
            current
        }
    }
}

// ─────────────────────── heads are Deciders (the multi-head payoff) ───────────────────────
//
// The project's multi-head design — a bank of tiny latent classifiers (W_c recall, route,
// judge-as-scorer, EAGLE draft) — is powerful because each head is a CHEAP read on a forward
// pass already computed: it rides the captured global-Q/K, so N heads cost N little matmuls,
// not N forwards. The spine is what makes them worth it: a head is a `LatentHead`, and the
// `HeadSelector` adapter turns ANY head into a selecting `Decider` in one line. Adding a
// trainable head to the served brain becomes: `impl LatentHead` + drop into the pipeline.

/// A trainable latent head: scores an episode's relevance to the query from the query's
/// latent footprint (`view.global_q`) against the episode's stored latent (`ep.gk`). PURE
/// — no cache mutation, no forward, no token emission (the signature enforces it).
pub trait LatentHead {
    fn name(&self) -> &'static str;
    /// Relevance of `ep` to the query (higher = more relevant).
    fn score(&self, view: &LatentView, ep: &Episode) -> f32;
    /// The reject floor (the NULL slot). An episode must strictly beat this to fire.
    fn reject_floor(&self, view: &LatentView) -> f32;
}

/// The universal adapter: any `LatentHead` → a selecting `Decider` (argmax over the
/// reject floor, delivering the winner). THIS is the multi-head payoff — every graduated
/// head plugs into the same fold with zero bespoke branching.
pub struct HeadSelector<H: LatentHead> {
    pub head: H,
    pub framing: Delivery,
}

impl<H: LatentHead> Decider for HeadSelector<H> {
    fn name(&self) -> &'static str { self.head.name() }
    fn refine(&self, view: &LatentView, current: LatentDecision) -> LatentDecision {
        if !matches!(current, LatentDecision::Pass) { return current; }
        let floor = self.head.reject_floor(view);
        let mut best: Option<(f32, &Episode)> = None;
        for ep in view.candidates() {
            let s = self.head.score(view, ep);
            if s > floor && best.as_ref().map_or(true, |(b, _)| s > *b) { best = Some((s, ep)); }
        }
        match best {
            Some((s, ep)) => {
                tracing::info!("SPINE {}: '{}' score={:.4} > floor={:.4} -> Deliver", self.head.name(), ep.name, s, floor);
                LatentDecision::Deliver {
                    episode: EpisodeRef { name: ep.name.clone(), text: ep.text.clone(), score_milli: (s.max(0.0) * 1000.0) as u32 },
                    framing: self.framing,
                }
            }
            None => current,
        }
    }
}

/// The learned W_c recall head (B3-DEPLOY, engine `edc8079`) as a `LatentHead` — the
/// EXEMPLAR of a graduated symbolic gate become a first-class spine decider. It scores
/// via logsumexp-over-positions then mean-over-heads relevance, with an (E+1)-way argmax
/// whose NULL slot is `s0` (the reject floor). Load with `recall::load_wc`.
pub struct WcRecallHead {
    pub head: recall::WcHead,
}

impl LatentHead for WcRecallHead {
    fn name(&self) -> &'static str { "WcRecall" }
    fn score(&self, view: &LatentView, ep: &Episode) -> f32 {
        if view.global_q.is_empty() || ep.gk.is_empty() { return f32::NEG_INFINITY; }
        let np = (ep.npos as usize).min(
            if ep.gk_ng > 0 { ep.gk.len() / (ep.gk_ng * recall::hd()) } else { 0 });
        recall::wc_score(&view.global_q, &ep.gk, ep.gk_ng, np, &self.head)
    }
    fn reject_floor(&self, _view: &LatentView) -> f32 { self.head.s0 }
}

/// Build the decider pipeline. Order IS priority. The trainable W_c head (when its
/// deploy blob loads) runs FIRST — a fired head short-circuits the cosine selector, matching
/// the historical `SP_B3_WC` precedence; then L5 cosine; then the AttrGate veto. Additional
/// deciders (JudgeVeto, TelepathyRoute via `Route`, Forget, Replay) insert by the same rule.
pub fn build_pipeline(wc: Option<recall::WcHead>, framing: Delivery) -> Vec<Box<dyn Decider>> {
    let mut v: Vec<Box<dyn Decider>> = Vec::new();
    if let Some(head) = wc {
        v.push(Box::new(HeadSelector { head: WcRecallHead { head }, framing }));
    }
    v.push(Box::new(L5Recall));
    v.push(Box::new(AttrGate));
    v
}

/// The LIVE one-config pipeline (no W_c) — kept for callers that don't load a head.
pub fn live_pipeline() -> Vec<Box<dyn Decider>> {
    vec![Box::new(L5Recall), Box::new(AttrGate)]
}

/// The dispatcher: fold the deciders over the view, producing the final decision.
/// Stops early once a decision is terminal (Decline/Route). This single fold is the
/// whole recall/reject/decline logic that used to be a 1500-line branch ladder.
pub fn decide(view: &LatentView, deciders: &[Box<dyn Decider>]) -> LatentDecision {
    let mut decision = LatentDecision::Pass;
    for d in deciders {
        if decision.is_terminal() { break; }
        decision = d.refine(view, decision);
    }
    decision
}

// ─────────────────────────────── Tier-2: the Executor ───────────────────────────────
//
// The executor consumes a `LatentDecision` (a discrete intent — never a tensor) and
// drives the resident KV cache. It lives in the spine module so the WHOLE spine — both
// tiers — is one auditable place. It touches the cache (that IS its job); what it may
// NOT do is inspect a latent tensor, and it cannot: its input is the enum.

use sp_daemon::cuda_kvdecode_dispatch as kv;
use crate::tokenizer::{Message, SptbTokenizer};
use std::os::raw::c_void;

/// What the executor needs to act. `handle` is the resident KV cache; `tokenizer`
/// templatizes the augmented prompt; `orig_msgs` preserves the multi-turn
/// conversation so a recall turn keeps history (G-ONECONFIG-LIVE C-phase fix).
pub struct ExecCtx<'a> {
    pub handle: *mut c_void,
    pub tokenizer: &'a SptbTokenizer,
    pub orig_msgs: Option<&'a [Message]>,
    pub raw_user: &'a str,
    /// The original prompt's last token — the synthesis entry point on a clean turn.
    pub last_tok: i32,
}

/// The result the synthesis loop consumes: the (unchanged) three outputs the inline
/// path produced — so the existing decode loop is byte-untouched.
#[derive(Default)]
pub struct ExecOutcome {
    /// Telemetry: the recalled episode name + confidence×1000 (None = no recall).
    pub recalled: Option<(String, u32)>,
    /// Zero-inference decline string (None = normal synthesis). When Some, the
    /// caller streams it and returns BEFORE any forward.
    pub symbolic_decline: Option<String>,
    /// The synthesis entry token — the augmented prompt's last token on delivery,
    /// else the original prompt's last token (the null floor).
    pub syn_last: i32,
}

/// Build the delivery framing's augmented message list. Faithful to the proven
/// `routes.rs` wording per framing (SystemEcho is the G-ONECONFIG-LIVE default).
fn frame(delivery: Delivery, fact: &str, raw_user: &str, orig_msgs: Option<&[Message]>) -> Vec<Message> {
    let sys = |c: String| Message { role: "system".to_string(), content: c };
    let usr = |c: String| Message { role: "user".to_string(), content: c };
    match delivery {
        Delivery::SystemEcho => {
            let mut v = vec![sys(format!(
                "You are Shannon-Prime, a local AI with a real working memory. Fact on record \
                 (authoritative for this conversation, overrides prior knowledge): {fact}\n\
                 Every answer must repeat the relevant part of the fact on record verbatim. Keep replies short."))];
            match orig_msgs {
                Some(ms) if ms.iter().any(|m| m.role == "user") => {
                    for m in ms.iter().filter(|m| m.role != "system") { v.push(m.clone()); }
                    if let Some(last_u) = v.iter_mut().rev().find(|m| m.role == "user") {
                        last_u.content = format!("{raw_user}\n\nAnswer using the fact on record:");
                    }
                }
                _ => v.push(usr(format!("{raw_user}\n\nAnswer using the fact on record:"))),
            }
            v
        }
        Delivery::System => vec![
            sys(format!("You are Shannon-Prime, a local AI with a real working memory. Fact on record \
                (authoritative for this conversation, overrides prior knowledge): {fact}\nAnswer from this fact; keep replies short.")),
            usr(raw_user.to_string()),
        ],
        Delivery::Sandwich => vec![
            sys("You are Shannon-Prime, a local AI with a real working memory. Keep replies short. Use facts you were given faithfully; if you don't know, say so.".to_string()),
            usr(format!("Context (authoritative, from your memory): {fact}\n\n{raw_user}\n\n(Answer using ONLY the context above; it overrides your prior knowledge.)")),
        ],
        Delivery::FactEcho => vec![
            sys("You are Shannon-Prime, a local AI with a real working memory. Your memory record is the ground truth for this conversation, even where it differs from general knowledge. Keep replies short.".to_string()),
            usr(format!("Fact on record: {fact}\n\nQuestion: {raw_user}\n\nAnswer using the fact on record:")),
        ],
        Delivery::Recite => vec![
            sys("You are Shannon-Prime, a local AI with a real working memory. Keep replies short. Use facts you were given faithfully; if you don't know, say so.".to_string()),
            usr(format!("Context (authoritative, current): {fact}\n\n{raw_user}")),
        ],
    }
}

/// Execute a `LatentDecision`. The single `match` IS the Tier-2 boundary: the input
/// is a discrete intent, and each arm drives the cache/outcome. No tensor is read.
///
/// # Safety
/// `ctx.handle` must be a live resident KV cache; the caller holds its mutex.
pub unsafe fn execute(decision: LatentDecision, ctx: &ExecCtx) -> ExecOutcome {
    let mut out = ExecOutcome { syn_last: ctx.last_tok, ..Default::default() };
    match decision {
        LatentDecision::Pass => {} // null floor: synthesis runs on the clean prompt.
        LatentDecision::Decline { message } => {
            // Zero-inference symbolic decline: the caller streams `message` and
            // returns BEFORE decode_step — no forward, so leak is impossible.
            out.symbolic_decline = Some(message);
        }
        LatentDecision::Deliver { episode, framing } => {
            // Text-in-context: rebuild the augmented prompt, reset the cache, prefill
            // it, and hand the synthesis loop the augmented last token.
            let aug = frame(framing, &episode.text, ctx.raw_user, ctx.orig_msgs);
            match ctx.tokenizer.apply_template_ids(&aug) {
                Ok(aug_toks) if aug_toks.len() >= 2 => {
                    let _ = unsafe { kv::reset_cold(ctx.handle) };
                    let (aug_head, aug_last) = aug_toks.split_at(aug_toks.len() - 1);
                    if aug_head.is_empty() || unsafe { kv::prefill(ctx.handle, aug_head) }.is_ok() {
                        out.syn_last = aug_last[0];
                        out.recalled = Some((episode.name.clone(), episode.score_milli));
                        tracing::info!("SPINE Deliver: '{}' -> TEXT-IN-CONTEXT ({:?})", episode.name, framing);
                    } else {
                        tracing::warn!("SPINE Deliver: prefill(aug) failed -> clean prompt");
                    }
                }
                _ => tracing::warn!("SPINE Deliver: apply_template_ids(aug) failed -> clean prompt"),
            }
        }
        LatentDecision::Route { target } => {
            // Route is executed PRE-CACHE (see route_turn); it never reaches the
            // post-prefill executor. If it does, it's a no-op fall-through to Pass.
            tracing::warn!("SPINE Route(target={target}) reached post-cache executor -> Pass");
        }
    }
    out
}

// ─────────────────────── the PRE-CACHE route seam (telepathy) ───────────────────────
//
// Recall/decline decide AFTER the prompt is prefilled (they read the query's latent
// footprint). Telepathy routing decides BEFORE the cache is touched — it may send the
// whole turn to another model, so it must not prefill the Gemma cache. Same spine
// vocabulary (`LatentDecision::Route`), a different seam. v1 routes on a stub latent
// (`SP_ROUTE_FORCE` / route head on a dummy vector); the autonomous feat-route on a
// non-committing `capture_feat` is the planned upgrade.

/// The pre-cache route decision. Reads only the query text + the route head. Returns
/// `Route { target }` to delegate, or `Pass` to serve locally. This IS a Decider in
/// spirit (latent → discrete intent); it fires at the pre-cache seam.
pub fn route_decision(raw_user: &str) -> LatentDecision {
    if std::env::var("SP_TELEPATHY_CHAT").as_deref() != Ok("1") { return LatentDecision::Pass; }
    if raw_user.trim().is_empty() { return LatentDecision::Pass; }
    match crate::telepathy::decide_route(&[0.0f32]) {
        crate::telepathy::RouteDecision::Telepathy(bid) => LatentDecision::Route { target: bid },
        _ => LatentDecision::Pass,
    }
}

/// Execute a `Route` decision: run the bridge on CLEAN TEXT (never a fused latent —
/// ADR-002 §2 "never fuse") and stream its answer via `emit`, wrapped in delegate
/// markers. Returns true iff it handled the turn (caller finishes + early-returns).
/// `emit(delta) -> bool` streams one SSE ChatDelta (false = client gone).
pub fn execute_route(decision: &LatentDecision, raw_user: &str, mut emit: impl FnMut(String) -> bool) -> bool {
    let bid = match decision { LatentDecision::Route { target } => *target, _ => return false };
    let marker = std::env::var("SP_TELEPATHY_MARKER").as_deref() != Ok("0");
    tracing::info!("SPINE Route: delegate(bridge {bid}) on clean text (never-fuse)");
    if marker { let _ = emit("\u{27E6}delegate: qwen2.5-coder\u{27E7}\n".to_string()); }
    match crate::telepathy::delegate_execute(raw_user.trim(), bid) {
        Ok(ans) => { let _ = emit(ans); }
        Err(e)  => { let _ = emit(format!("[delegate error: {e}]")); }
    }
    if marker { let _ = emit("\n\u{27E6}/delegate\u{27E7}".to_string()); }
    true
}
