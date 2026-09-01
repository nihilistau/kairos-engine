//! L2 sampler — CONTRACT-CHAT-FULLSTACK Stage A2.
//!
//! The kvdecode ABI (`gemma4_kv_decode_logits`, cuda_kvdecode_dispatch.rs:139)
//! deliberately returns the full-vocab logits row so that **L2 owns sampling**.
//! This module is that owner: temperature / top-k / top-p (nucleus) /
//! repetition+frequency penalty over the generated history, with a seedable,
//! deterministic RNG.
//!
//! NULL FLOOR / DETERMINISM CONTRACT (gate G-CHAT-A2 / A2-polish):
//!   `temperature == 0.0` ⇒ pure argmax (no penalty, no top-k/p, no RNG draw —
//!   the same `max_by partial_cmp` tie-break). A2-polish: control-token
//!   suppression is now applied on the greedy path too, so the DEFAULT served
//!   greedy chat is clean. The byte-for-byte-identical-to-old-`fn argmax` null
//!   floor is recovered by constructing the sampler with an EMPTY suppress set
//!   (ChatRequest.raw_logits=true) — with no ids to mask, temp=0 is exactly the
//!   prior hardcoded argmax. So: default greedy = clean; raw_logits greedy =
//!   the determinism reference both A1 legs / the B1 determinism leg compare to.
//!
//! Given a seed, sampling is fully reproducible: same logits + same history +
//! same seed ⇒ same token. The RNG is a small SplitMix64 (no external dep).

use serde::Deserialize;

/// Sampling knobs, plumbed from `ChatRequest`. Defaults are the contract's
/// pre-registered values (temperature 0.7, top_p 0.95, top_k 40, rep_pen 1.1).
#[derive(Debug, Clone, Deserialize)]
pub struct SamplingParams {
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default = "default_top_k")]
    pub top_k: u32,
    #[serde(default = "default_repetition_penalty")]
    pub repetition_penalty: f32,
    /// Optional frequency penalty (subtractive, scaled by occurrence count).
    /// 0.0 = off (default). Distinct from repetition_penalty (multiplicative).
    #[serde(default)]
    pub frequency_penalty: f32,
    /// Optional seed for the RNG. `None` ⇒ derive from a process clock so
    /// successive requests differ; `Some(s)` ⇒ fully reproducible.
    #[serde(default)]
    pub seed: Option<u64>,
    /// Anti-echo (#47): forbid emitting a token that would complete an n-gram
    /// already present in (prompt-prefix ++ generated history). 0 = off (default,
    /// null floor). Applied on the GREEDY path too (that is where #47 lives). Used
    /// on the PLAIN chat path only (recall OFF); the recall/systemecho path must
    /// recite in-context facts, so it leaves this at 0. See G-ECHO-FIX.
    #[serde(default)]
    pub no_repeat_ngram: usize,
}

fn default_temperature() -> f32 {
    0.7
}
fn default_top_p() -> f32 {
    0.95
}
fn default_top_k() -> u32 {
    40
}
fn default_repetition_penalty() -> f32 {
    1.1
}

impl Default for SamplingParams {
    fn default() -> Self {
        SamplingParams {
            temperature: default_temperature(),
            top_p: default_top_p(),
            top_k: default_top_k(),
            repetition_penalty: default_repetition_penalty(),
            frequency_penalty: 0.0,
            seed: None,
            no_repeat_ngram: 0,
        }
    }
}

/// SplitMix64 — tiny deterministic PRNG. Avoids an external `rand` dependency
/// for the ~10 lines of sampling RNG we need; well-distributed for f64 draws.
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        // Avoid the all-zero fixed point of SplitMix64.
        Rng {
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform f64 in [0, 1).
    fn next_f64(&mut self) -> f64 {
        // 53-bit mantissa.
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// The stateful sampler for one chat turn. Owns its RNG and tracks the
/// generated-token history (for repetition / frequency penalty).
pub struct Sampler {
    params: SamplingParams,
    rng: Rng,
    /// Generated token ids so far this turn, for the repetition penalty.
    history: Vec<i32>,
    /// Token ids whose logit is forced to -inf before sampling — the full
    /// control/placeholder set (`<image|>`/`<audio|>`/`<|turn>`/`<turn|>`/… and
    /// the named structural specials), derived id-agnostically by the tokenizer
    /// (`suppress_token_ids`). A2-polish: applied on BOTH the sampled and the
    /// greedy (temp==0) path so the default served chat is clean everywhere. The
    /// byte-exact/determinism null floor recovers the un-suppressed reference by
    /// constructing the sampler with this set EMPTY (ChatRequest.raw_logits=true).
    suppress: Vec<i32>,
    /// Anti-echo (#47): the prompt token ids that PRECEDE generation. The
    /// no-repeat-ngram guard scans (ngram_prefix ++ history) so a verbatim
    /// recital of the prompt (e.g. the system prompt on a contentless turn) is
    /// broken as soon as it re-traces an n-gram already in the prompt. Empty =
    /// off (the sample() guard also no-ops when params.no_repeat_ngram < 2).
    ngram_prefix: Vec<i32>,
    /// ADR-013 minimum thought: the `<channel|>` id, masked until `think_min`
    /// tokens have been generated. None/0 = off (byte-identical null floor).
    think_close: Option<i32>,
    think_min: usize,
    /// ADR-013 MAXIMUM thought — the missing twin of the floor above. `think_min`
    /// could only make her think MORE; nothing capped it, and on gemma4-26b-a4b that
    /// showed up as single turns of 258 s, 256 s and 308 s where she spent the entire
    /// 768-token budget reasoning and the reader waited five minutes for two lines.
    ///
    /// Two budgets, whichever fires first, because they fail differently: a token
    /// budget is deterministic and reproducible but says nothing about wall time when
    /// decode is slow (expert streaming on a 12 GB card is exactly that case), and a
    /// time budget bounds the human's wait but is not reproducible. Gates need the
    /// first; the person waiting needs the second.
    ///
    /// 0 = off for both, which is the byte-identical null floor.
    think_max: usize,
    think_max_ms: u64,
    /// Whether the close token has ALREADY been emitted this turn. Load-bearing: the
    /// floor can use `history.len()` as its proxy because it only ever fires early,
    /// but a ceiling keyed on the same counter would re-force the close token during
    /// the ANSWER — every token past the budget would be masked down to `<channel|>`
    /// and the reply would be nothing but channel closes. The ceiling must fire once.
    think_closed: bool,
    /// Set when the CEILING forced the close rather than the model choosing it.
    /// Read once per turn by the chat route; see the note at the forcing site.
    think_forced: bool,
    /// Turn start, for the wall-clock budget. The sampler is constructed per request,
    /// so construction time IS the turn start.
    turn_start: std::time::Instant,
}

impl Sampler {
    pub fn new(params: SamplingParams) -> Self {
        Self::with_suppress(params, Vec::new())
    }

    /// Anti-echo (#47): seed the no-repeat-ngram guard with the prompt tokens.
    pub fn set_ngram_prefix(&mut self, prefix: Vec<i32>) {
        self.ngram_prefix = prefix;
    }

    /// Anti-echo (#47): set the no-repeat-ngram order (n>=2 to enable, 0 = off).
    pub fn set_no_repeat_ngram(&mut self, n: usize) {
        self.params.no_repeat_ngram = n;
    }

    /// ADR-013: enforce a minimum thought before the channel may close.
    pub fn set_min_thought(&mut self, close_id: Option<i32>, min_tokens: usize) {
        self.think_close = close_id;
        self.think_min = min_tokens;
    }

    /// ADR-013: enforce a MAXIMUM thought — close the channel once either budget is
    /// spent. `max_tokens` counts generated tokens this turn; `max_ms` is wall clock
    /// from sampler construction. Either 0 = that budget is off; both 0 = no ceiling
    /// (the byte-identical null floor).
    pub fn set_max_thought(&mut self, close_id: Option<i32>, max_tokens: usize, max_ms: u64) {
        // The close id is shared with the floor; setting it here too means the ceiling
        // works whether or not SP_THINK_MIN was armed.
        if close_id.is_some() {
            self.think_close = close_id;
        }
        self.think_max = max_tokens;
        self.think_max_ms = max_ms;
    }

    /// Has a thought budget been spent? Separated from `sample()` so the reason is
    /// testable and so the two budgets read as the independent things they are.
    #[inline]
    fn thought_budget_spent(&self) -> bool {
        if self.think_max > 0 && self.history.len() >= self.think_max {
            return true;
        }
        if self.think_max_ms > 0
            && self.turn_start.elapsed().as_millis() as u64 >= self.think_max_ms
        {
            return true;
        }
        false
    }

    pub fn with_suppress(params: SamplingParams, suppress: Vec<i32>) -> Self {
        let seed = params.seed.unwrap_or_else(|| {
            // Non-deterministic default seed from a monotonic clock.
            use std::time::{SystemTime, UNIX_EPOCH};
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0x1234_5678_9ABC_DEF0)
        });
        Sampler {
            rng: Rng::new(seed),
            history: Vec::new(),
            params,
            suppress,
            ngram_prefix: Vec::new(),
            think_close: None,
            think_min: 0,
            think_max: 0,
            think_max_ms: 0,
            think_closed: false,
            think_forced: false,
            turn_start: std::time::Instant::now(),
        }
    }

    /// True when this sampler is the strict-argmax null floor (temp == 0).
    #[inline]
    pub fn is_greedy(&self) -> bool {
        self.params.temperature <= 0.0
    }

    /// TRUE when the thought ceiling forced `<channel|>` this turn — i.e. she was
    /// interrupted mid-reasoning rather than finishing. Observability only.
    pub fn think_was_forced(&self) -> bool { self.think_forced }

    /// Record a chosen token into the penalty history.
    pub fn observe(&mut self, token: i32) {
        // The ceiling fires ONCE. Without this the budget check would re-force the close
        // token on every token of the answer, and the reply would be nothing but closes.
        if let Some(id) = self.think_close {
            if token == id {
                self.think_closed = true;
            }
        }
        self.history.push(token);
    }

    /// Pick the next token from a full-vocab logits row.
    ///
    /// `temperature == 0` ⇒ strict argmax (byte-identical to the old `argmax`):
    /// no penalty, no truncation, no RNG. Otherwise: apply repetition/frequency
    /// penalties over `history`, scale by temperature, truncate to top-k then
    /// top-p (nucleus), softmax over the survivors, and draw.
    pub fn sample(&mut self, logits: &mut [f32]) -> i32 {
        if logits.is_empty() {
            return 0;
        }

        // ── suppress placeholder/control tokens (the <image|>/<audio|>/<|turn>
        // loop attractors) ──
        // CONTRACT-CHAT-FULLSTACK A2-polish: suppression is now applied on the
        // greedy (temp==0) path TOO, so the DEFAULT served chat is clean at
        // every temperature incl. greedy. The byte-exact / determinism null
        // floor recovers the un-suppressed reference by constructing the
        // sampler with an EMPTY suppress set (ChatRequest.raw_logits=true) — so
        // here, an empty `suppress` is the pass-through that reproduces the old
        // hardcoded argmax bit-for-bit.
        for &id in &self.suppress {
            let idx = id as usize;
            if idx < logits.len() {
                logits[idx] = f32::NEG_INFINITY;
            }
        }

        // ── A MINIMUM THOUGHT (ADR-013, 2026-07-29) ──────────────────────────────
        // Opening the thought channel is not enough to make a model think. MEASURED
        // on gemma4-26b-a4b: with the channel open and a persona section explicitly
        // inviting her to use it, she emits `<channel|>` as the VERY FIRST token,
        // every turn — an empty thought — then answers. The instruction sits ~3400
        // tokens back; the close token is the highest-probability continuation right
        // here. A system prompt does not win that argument.
        //
        // So the floor is enforced where the decision is made: the channel-close
        // token is masked until `think_min` tokens have been generated this turn.
        // This is the same shape as a reasoning budget — not a nudge, a floor.
        //
        // It is a FLOOR, never a ceiling: once past it she closes whenever she likes,
        // so a turn that needs no thinking costs think_min tokens and no more.
        // think_min = 0 disables it entirely (the null floor: she decides alone).
        if self.think_min > 0 && self.history.len() < self.think_min {
            if let Some(id) = self.think_close {
                let idx = id as usize;
                if idx < logits.len() {
                    logits[idx] = f32::NEG_INFINITY;
                }
            }
        }

        // ── ADR-013 MAXIMUM thought: the ceiling, mirroring the floor above ──────────
        // The floor could only ever make her think MORE. Nothing capped it, and on
        // gemma4-26b-a4b that produced single turns of 258 s / 256 s / 308 s in which
        // she spent the whole 768-token budget reasoning and the reader waited five
        // minutes for two lines of reply.
        //
        // Enforced the same way and in the same place, because that is where the
        // decision is made: once a budget is spent, FORCE the close token by masking
        // everything else. Masking rather than an early return means it holds on both
        // the sampled and the greedy path, exactly like the floor and the suppress set.
        //
        // `think_closed` is what makes this fire ONCE. Keyed on history.len() alone it
        // would re-force the close token on every token of the answer and the reply
        // would be nothing but channel closes — the floor gets away with the same proxy
        // only because it exclusively fires early.
        //
        // Both budgets 0 ⇒ this block cannot execute ⇒ byte-identical null floor.
        if !self.think_closed && self.thought_budget_spent() {
            if let Some(id) = self.think_close {
                let idx = id as usize;
                if idx < logits.len() {
                    for (i, l) in logits.iter_mut().enumerate() {
                        if i != idx {
                            *l = f32::NEG_INFINITY;
                        }
                    }
                    // ── WAS THE THOUGHT CUT OFF, OR DID SHE FINISH IT? (2026-08-30) ──
                    // Observability only; no behaviour changes here. A forced close is
                    // an interruption MID-REASONING, and the suspicion this exists to
                    // test is that the model then simply carries on reasoning — in the
                    // SPEECH channel, where it reaches him, the room and the record.
                    // Measured the same night: ~13% of her recorded turns open as
                    // analysis about him rather than speech to him (chronic, ten-day
                    // baseline in tools/voice_leak_rate.py). If this flag fires at
                    // about that rate and on those turns, the ceiling is the cause and
                    // the fix belongs at the ceiling — not in a stripper that would
                    // have to guess her voice. If it does not, the hypothesis dies here
                    // instead of becoming a story someone acts on.
                    self.think_forced = true;
                }
            }
        }

        // ── anti-echo (#47): no-repeat-ngram over (prompt-prefix ++ history) ──
        // Forbid the token that would complete an n-gram already present in the
        // sequence, so a verbatim recital of the prompt (the system prompt on a
        // contentless turn) is broken as soon as it re-traces a prompt n-gram.
        // Applied BEFORE the greedy return so it works at temp==0 (where #47 is).
        // Off when no_repeat_ngram < 2 ⇒ byte-identical null floor.
        let n = self.params.no_repeat_ngram;
        if n >= 2 {
            ban_repeat_ngram(logits, &self.ngram_prefix, &self.history, n);
        }

        // ── temp == 0: strict argmax null floor ──
        // With an empty suppress set this is byte-identical to the prior
        // hardcoded `fn argmax` (the raw_logits determinism leg). With the
        // default suppress set it is the clean greedy path.
        if self.is_greedy() {
            return argmax(logits);
        }

        // ── repetition / frequency penalty over generated history ──
        if (self.params.repetition_penalty - 1.0).abs() > f32::EPSILON
            || self.params.frequency_penalty.abs() > f32::EPSILON
        {
            // Count occurrences once (frequency penalty scales by count).
            // History is short (<= max_tokens), so a per-call scan is cheap.
            let rp = self.params.repetition_penalty;
            let fp = self.params.frequency_penalty;
            // Walk history; apply the multiplicative rep-penalty once per unique
            // token and the additive frequency penalty per occurrence.
            let mut seen_rep: std::collections::HashSet<i32> =
                std::collections::HashSet::new();
            for &t in &self.history {
                let idx = t as usize;
                if idx >= logits.len() {
                    continue;
                }
                if (rp - 1.0).abs() > f32::EPSILON && seen_rep.insert(t) {
                    // Standard HF-style rep penalty: divide if logit > 0, else
                    // multiply (push the logit further negative).
                    let l = logits[idx];
                    logits[idx] = if l > 0.0 { l / rp } else { l * rp };
                }
                if fp.abs() > f32::EPSILON {
                    logits[idx] -= fp;
                }
            }
        }

        // ── temperature scale ──
        let temp = self.params.temperature.max(1e-6);
        for l in logits.iter_mut() {
            *l /= temp;
        }

        // ── build a candidate list (index, logit), sorted desc by logit ──
        let mut cand: Vec<(usize, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &l)| (i, l))
            .collect();
        // Partial sort by logit descending (NaN treated as smallest).
        cand.sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });

        // ── top-k truncation ──
        let k = if self.params.top_k == 0 {
            cand.len()
        } else {
            (self.params.top_k as usize).min(cand.len())
        };
        cand.truncate(k);

        // ── softmax over the survivors (numerically stable) ──
        let max_logit = cand.first().map(|&(_, l)| l).unwrap_or(0.0);
        let mut probs: Vec<f64> = cand
            .iter()
            .map(|&(_, l)| ((l - max_logit) as f64).exp())
            .collect();
        let sum: f64 = probs.iter().sum();
        if sum <= 0.0 || !sum.is_finite() {
            // Degenerate (all -inf / NaN) — fall back to argmax of the survivors.
            return cand.first().map(|&(i, _)| i as i32).unwrap_or(0);
        }
        for p in probs.iter_mut() {
            *p /= sum;
        }

        // ── top-p (nucleus): keep the smallest prefix whose mass >= top_p ──
        let top_p = self.params.top_p.clamp(0.0, 1.0) as f64;
        let mut nucleus = cand.len();
        if top_p < 1.0 {
            let mut acc = 0.0f64;
            for (i, &p) in probs.iter().enumerate() {
                acc += p;
                if acc >= top_p {
                    nucleus = i + 1;
                    break;
                }
            }
        }
        let nucleus = nucleus.max(1);

        // Renormalize the nucleus.
        let nucleus_sum: f64 = probs[..nucleus].iter().sum();

        // ── draw ──
        let r = self.rng.next_f64() * nucleus_sum;
        let mut acc = 0.0f64;
        for i in 0..nucleus {
            acc += probs[i];
            if r < acc {
                return cand[i].0 as i32;
            }
        }
        // Floating-point fall-through: return the last nucleus member.
        cand[nucleus - 1].0 as i32
    }
}

/// Anti-echo (#47): forbid the token that would complete an `n`-gram already
/// present in `prefix ++ history`. Standard no-repeat-ngram, but seeded with the
/// prompt so a verbatim recital of the prompt (the system prompt on a contentless
/// turn) is derailed the instant the generated suffix re-traces a prompt n-gram.
/// Sets the banned logits to -inf. `n < 2` ⇒ no-op.
fn ban_repeat_ngram(logits: &mut [f32], prefix: &[i32], history: &[i32], n: usize) {
    if n < 2 {
        return;
    }
    // Effective sequence = prefix ++ history (bounded by prompt + max_tokens).
    let seq: Vec<i32> = prefix.iter().chain(history.iter()).copied().collect();
    if seq.len() < n {
        return;
    }
    let m = seq.len();
    let suffix = &seq[m - (n - 1)..]; // the current trailing (n-1)-gram
    // Every earlier occurrence of `suffix` that is FOLLOWED by a token bans that
    // follower. Sources i range so that i..i+(n-1) is the match and i+(n-1) is the
    // follower; i <= m-1-(n-1) = m-n excludes the trailing suffix itself.
    let last_src = m - n; // inclusive
    for i in 0..=last_src {
        if &seq[i..i + (n - 1)] == suffix {
            let banned = seq[i + (n - 1)];
            if (banned as usize) < logits.len() {
                logits[banned as usize] = f32::NEG_INFINITY;
            }
        }
    }
}

/// Strict argmax — first index wins ties (matches the prior `fn argmax` in
/// routes.rs exactly, the temp=0 null floor for G-CHAT-A2).
pub fn argmax(logits: &[f32]) -> i32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as i32)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temp_zero_is_argmax() {
        // The null-floor invariant (G-CHAT-A2): temp=0 ⇒ the sampler returns
        // EXACTLY what the old hardcoded `fn argmax` returned. std `max_by`
        // returns the LAST element among equal maxima, so with two 5.0 maxima
        // (indices 1 and 4) the result is index 4 — and crucially, identical
        // between the sampler's greedy path and the free `argmax` helper (which
        // is byte-for-byte the prior routes.rs argmax).
        let mut s = Sampler::new(SamplingParams {
            temperature: 0.0,
            ..Default::default()
        });
        let mut logits = vec![0.1, 5.0, -2.0, 4.9, 5.0];
        let via_sampler = s.sample(&mut logits);
        let logits2 = vec![0.1, 5.0, -2.0, 4.9, 5.0];
        let via_argmax = argmax(&logits2);
        assert_eq!(via_sampler, via_argmax, "temp=0 must equal the old argmax");
        assert_eq!(via_sampler, 4, "max_by returns the last of equal maxima");

        // Unique-max case: unambiguous.
        let mut s2 = Sampler::new(SamplingParams { temperature: 0.0, ..Default::default() });
        let mut l3 = vec![0.1, 9.0, -2.0, 4.9, 5.0];
        assert_eq!(s2.sample(&mut l3), 1);
    }

    #[test]
    fn seeded_is_deterministic() {
        let p = SamplingParams {
            temperature: 0.8,
            top_p: 0.95,
            top_k: 40,
            repetition_penalty: 1.1,
            frequency_penalty: 0.0,
            seed: Some(42),
            ..Default::default()
        };
        let logits_master = vec![1.0f32, 2.0, 3.0, 0.5, -1.0, 2.5, 1.5, 0.0];
        let mut a = Sampler::new(p.clone());
        let mut b = Sampler::new(p);
        for _ in 0..32 {
            let mut la = logits_master.clone();
            let mut lb = logits_master.clone();
            let ta = a.sample(&mut la);
            let tb = b.sample(&mut lb);
            assert_eq!(ta, tb, "seeded sampler must be reproducible");
            a.observe(ta);
            b.observe(tb);
        }
    }

    #[test]
    fn top_k_one_is_argmax_even_with_temp() {
        let mut s = Sampler::new(SamplingParams {
            temperature: 1.0,
            top_k: 1,
            top_p: 1.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            seed: Some(7),
            // ..Default::default() rather than another literal: the next field added to
            // SamplingParams should not break the test suite a second time.
            ..Default::default()
        });
        let mut logits = vec![0.1, 9.0, 3.0, 2.0];
        // top_k=1 collapses the candidate set to the single max ⇒ deterministic.
        assert_eq!(s.sample(&mut logits), 1);
    }

    #[test]
    fn suppress_applies_on_greedy_path() {
        // A2-polish: control-token suppression must fire even at temp=0 so the
        // default greedy chat is clean. Token 1 has the max logit but is in the
        // suppress set ⇒ greedy must pick the next-best UNsuppressed token (3).
        let mut s = Sampler::with_suppress(
            SamplingParams { temperature: 0.0, ..Default::default() },
            vec![1],
        );
        let mut logits = vec![0.1, 9.0, -2.0, 4.9, 5.0];
        // index 1 (9.0) suppressed ⇒ index 4 (5.0) wins.
        assert_eq!(s.sample(&mut logits), 4);
    }

    #[test]
    fn raw_logits_empty_suppress_is_old_argmax() {
        // The null-floor opt-out: with an EMPTY suppress set (raw_logits=true),
        // temp=0 reproduces the prior hardcoded argmax bit-for-bit even when a
        // control token is the max — no masking happens.
        let mut s = Sampler::with_suppress(
            SamplingParams { temperature: 0.0, ..Default::default() },
            Vec::new(),
        );
        let mut logits = vec![0.1, 9.0, -2.0, 4.9, 5.0];
        assert_eq!(s.sample(&mut logits), 1, "empty suppress ⇒ raw argmax");
    }

    #[test]
    fn repetition_penalty_suppresses_repeat() {
        // With a strong rep penalty + top_k=1, a previously-seen high-logit
        // token should be demoted below an unseen competitor.
        let mut s = Sampler::new(SamplingParams {
            temperature: 1.0,
            top_k: 1,
            top_p: 1.0,
            repetition_penalty: 4.0,
            frequency_penalty: 0.0,
            seed: Some(1),
            // ..Default::default() rather than another literal: the next field added to
            // SamplingParams should not break the test suite a second time.
            ..Default::default()
        });
        s.observe(1); // token 1 was generated before
        // Logit 1 (10.0) is highest but penalized /4 = 2.5; token 2 (3.0) wins.
        let mut logits = vec![0.0, 10.0, 3.0, 1.0];
        assert_eq!(s.sample(&mut logits), 2);
    }
}
