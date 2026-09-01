//! TOOL MASK — the calls that do not exist become UNREACHABLE, not merely wrong.
//!
//! ## What this replaces
//!
//! Tool calling in this system was: emit free text, then in the harness run a regex over it,
//! accept four different fence spellings, AST-parse, HEAL the model's typos with another
//! regex (`get _time()` -> `get_time()`), re-prompt on failure, and finally give up with
//! `"(tool loop exhausted)"` — which the operator saw on his screen. Five layers of hoping,
//! all of them there for one reason: the model emits free text, and free text can be wrong.
//!
//! A logit mask makes the wrong ones NOT EXIST. `recal(` is not a typo to be forgiven; it is
//! a token sequence the sampler cannot produce.
//!
//! ## DOMINO (arXiv 2403.06988) — and why this is a TOKEN trie, not a character matcher
//!
//! The paper's central finding is not about speed. It is that NAIVE CONSTRAINED DECODING
//! MAKES THE MODEL WORSE. Grammar terminals do not align with the subword vocabulary, so a
//! mask built over CHARACTERS forces the model off its natural token boundaries and task
//! accuracy measurably drops. The operator's worry — "we don't want to restrict the model's
//! ability to seem alive" — is exactly this, and it is real.
//!
//! So the trie here is built over `tokenizer.encode(name)` — THE MODEL'S OWN TOKENISATION of
//! each tool name. Every path through it is a sequence of tokens the model would naturally
//! have produced. It is never asked to spell a word one character at a time.
//!
//! ## And it should make tool turns FASTER
//!
//! Where the trie admits exactly ONE next token, that token needs no forward pass: the model
//! has no choice to make, so do not spend 262k logits asking it to make one. `forced()` is
//! that path. DOMINO reports up to ~2x from precisely this.
//!
//! ## The rule that matters most
//!
//! OUTSIDE A TOOL CALL, THIS MASKS NOTHING. `allowed()` returns `None` while she is talking,
//! and `None` means "unconstrained" — not "nothing is permitted". A grammar with an opinion
//! about her prose is how you get a model that can only fill in forms.

use std::collections::HashSet;

/// Where we are inside a tool call. Nothing outside `Name` constrains anything yet — the
/// name is the highest-value, lowest-risk constraint (it is what makes a hallucinated tool
/// unreachable), and it is the piece the argument grammar will hang off next.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// She is talking. The mask is OFF. This is the overwhelming majority of tokens.
    Prose,
    /// The fence is open: the next tokens must spell a tool she actually has.
    Name,
    /// The name is complete and `(` has been emitted — arguments are hers for now.
    Args,
}

pub struct ToolMask {
    /// One token-id sequence per tool name, in the MODEL'S OWN tokenisation.
    names: Vec<Vec<i32>>,
    /// The fence-open token sequence ("```tool_code\n"), likewise.
    fence: Vec<i32>,
    /// Rolling window of what has been generated, as token ids.
    hist: Vec<i32>,
    state: State,
    /// How many tokens of the name have been emitted so far (only meaningful in `Name`).
    depth: usize,
    /// Statistics — the whole point of shipping this behind a flag.
    pub masked_steps: usize,
    pub forced_steps: usize,
}

impl ToolMask {
    /// `encode` is the tokenizer's own encoder. Passing it in (rather than a &Tokenizer)
    /// keeps this module free of engine types, so it can be unit-tested with a toy vocab.
    pub fn new<F: Fn(&str) -> Vec<i32>>(tool_names: &[String], encode: F) -> Self {
        let names = tool_names
            .iter()
            .map(|n| encode(n))
            .filter(|v| !v.is_empty())
            .collect::<Vec<_>>();
        Self {
            names,
            fence: encode("```tool_code\n"),
            hist: Vec::new(),
            state: State::Prose,
            depth: 0,
            masked_steps: 0,
            forced_steps: 0,
        }
    }

    pub fn enabled(&self) -> bool {
        !self.names.is_empty() && !self.fence.is_empty()
    }

    /// Feed the token that was actually emitted. This is the ONLY state transition.
    pub fn push(&mut self, id: i32, piece: &[u8]) {
        self.hist.push(id);
        match self.state {
            State::Prose => {
                // The fence is matched on TOKENS, not on a string suffix: a string match
                // would be at the mercy of how the tokeniser happened to split the
                // backticks, which is the misalignment DOMINO warns about, in miniature.
                if self.hist.len() >= self.fence.len()
                    && self.hist[self.hist.len() - self.fence.len()..] == self.fence[..]
                {
                    self.state = State::Name;
                    self.depth = 0;
                }
            }
            State::Name => {
                self.depth += 1;
                // A complete name, followed by '(' in the same or the next token, ends the
                // constrained region. We look at the BYTES here (not the id) because '(' may
                // ride along on the tail of a name token.
                if piece.contains(&b'(') {
                    self.state = State::Args;
                } else if self.names.iter().all(|n| !self.prefix_ok(n)) {
                    // Belt and braces: if we have somehow left every legal path (which the
                    // mask should make impossible), stop constraining rather than deadlock
                    // the sampler into -inf everywhere. A mask that can wedge a turn is a
                    // worse bug than the one it fixes.
                    self.state = State::Prose;
                }
            }
            State::Args => {
                if piece.contains(&b'`') {
                    self.state = State::Prose; // fence closed; back to unconstrained
                }
            }
        }
    }

    fn prefix_ok(&self, name: &[i32]) -> bool {
        if self.depth > name.len() {
            return false;
        }
        let start = self.hist.len() - self.depth;
        self.hist[start..] == name[..self.depth]
    }

    /// The token ids the sampler may emit next.
    ///
    /// `None` = UNCONSTRAINED. That is the answer nearly every step, and it is the most
    /// important line in this file: while she is talking, the mask has no opinion.
    pub fn allowed(&mut self) -> Option<Vec<i32>> {
        if self.state != State::Name || !self.enabled() {
            return None;
        }
        let mut ids: HashSet<i32> = HashSet::new();
        for n in &self.names {
            if self.depth < n.len() && self.prefix_ok(n) {
                ids.insert(n[self.depth]);
            }
        }
        if ids.is_empty() {
            return None; // never wedge the sampler — see push()
        }
        self.masked_steps += 1;
        Some(ids.into_iter().collect())
    }

    /// THE FREE TOKEN. When exactly one continuation is legal the model has no choice, so
    /// there is nothing to ask it: skip the forward pass entirely and emit. This is where a
    /// constrained tool call stops being a tax and becomes a speedup.
    pub fn forced(&mut self) -> Option<i32> {
        let a = self.allowed()?;
        if a.len() == 1 {
            self.masked_steps -= 1; // it was not a mask, it was a certainty
            self.forced_steps += 1;
            return Some(a[0]);
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A toy tokeniser: one id per byte. Crude, but it exercises every transition, and the
    // real one is injected by the caller precisely so this can be tested without a GPU.
    fn enc(s: &str) -> Vec<i32> {
        s.bytes().map(|b| b as i32).collect()
    }

    fn feed(m: &mut ToolMask, s: &str) {
        for b in s.bytes() {
            m.push(b as i32, &[b]);
        }
    }

    #[test]
    fn prose_is_never_masked() {
        let mut m = ToolMask::new(&["recall".into(), "remember".into()], enc);
        feed(&mut m, "I've been thinking about your cat");
        assert!(m.allowed().is_none(), "she must be free to talk");
    }

    #[test]
    fn a_hallucinated_name_is_unreachable() {
        let mut m = ToolMask::new(&["recall".into(), "remember".into()], enc);
        feed(&mut m, "```tool_code\n");
        let a = m.allowed().expect("the fence is open: the name must be constrained");
        // both tools start with 'r', so 'r' is legal and nothing else is
        assert!(a.contains(&(b'r' as i32)));
        assert!(!a.contains(&(b'w' as i32)), "'websearch' cannot even begin");

        feed(&mut m, "rec");
        let a = m.allowed().unwrap();
        assert_eq!(a, vec![b'a' as i32], "only 'recall' survives 'rec'");
    }

    #[test]
    fn the_only_legal_token_is_free() {
        let mut m = ToolMask::new(&["recall".into(), "remember".into()], enc);
        feed(&mut m, "```tool_code\nrec");
        assert_eq!(m.forced(), Some(b'a' as i32), "no forward pass needed");
        assert_eq!(m.forced_steps, 1);
    }

    #[test]
    fn the_mask_lifts_once_the_call_begins() {
        let mut m = ToolMask::new(&["recall".into()], enc);
        feed(&mut m, "```tool_code\nrecall(");
        assert!(m.allowed().is_none(), "the ARGUMENTS are hers");
    }
}
