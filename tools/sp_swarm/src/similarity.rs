//! SP-SWARM L4 — C2-SimHash similarity OVERLAY (a secondary index, NOT the routing key).
//!
//! The exact SHA/C2 address (L1) does the routing + integrity; this rides on top purely for
//! "find similar memories" — a wrong hint costs nothing because the exact fetch (L2) verifies.
//! The signature IS the engine's proven C2: the sign of the frozen ±1 Rademacher projection of
//! a memory's global-K, packed to 256 bits ([u64;4]) — see `recall::Projection`/`recall::agree`.
//! Similarity = bit-agreement (256 − Hamming). This module holds only the INDEX + query (pure
//! std, no model); the engine computes the sigs.
//!
//! HONEST SCOPE: this proves the index MECHANICS (top-k Hamming ranking). Whether C2-Hamming is a
//! GOOD semantic discovery signal is a separate, falsifiable measurement (the boundary thesis has
//! shown structure-on-content signals can be weak) — gated separately before any network surface.

use std::collections::BTreeMap;
use std::path::Path;

/// A 256-bit C2 SimHash, 4 little-endian u64 words (== `recall::Projection::signature`).
pub type Sig = [u64; 4];

/// Hamming distance (0..=256): lower = more similar.
pub fn hamming(a: &Sig, b: &Sig) -> u32 {
    (0..4).map(|i| (a[i] ^ b[i]).count_ones()).sum()
}
/// Bit-agreement (256 − Hamming): higher = more similar. Matches `recall::agree`.
pub fn agree(a: &Sig, b: &Sig) -> u32 {
    256 - hamming(a, b)
}

pub fn sig_to_hex(s: &Sig) -> String {
    let mut b = [0u8; 32];
    for i in 0..4 { b[i * 8..i * 8 + 8].copy_from_slice(&s[i].to_le_bytes()); }
    hex::encode(b)
}
pub fn sig_from_hex(h: &str) -> Option<Sig> {
    let b = hex::decode(h.trim()).ok()?;
    if b.len() != 32 { return None; }
    let mut s = [0u64; 4];
    for i in 0..4 { s[i] = u64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().ok()?); }
    Some(s)
}

/// A local C2 similarity index: object address -> 256-bit sig. Query returns top-k nearest by
/// Hamming. Deterministic; ties broken by address for reproducibility.
#[derive(Default)]
pub struct C2Index {
    entries: Vec<(String, Sig)>,
}
impl C2Index {
    pub fn new() -> Self { C2Index { entries: Vec::new() } }
    pub fn len(&self) -> usize { self.entries.len() }
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }
    pub fn insert(&mut self, addr: String, sig: Sig) {
        if let Some(e) = self.entries.iter_mut().find(|(a, _)| *a == addr) { e.1 = sig; }
        else { self.entries.push((addr, sig)); }
    }
    /// Top-k nearest to `q` as (addr, hamming), ascending Hamming (nearest first).
    pub fn find_similar(&self, q: &Sig, k: usize) -> Vec<(String, u32)> {
        let mut v: Vec<(String, u32)> = self.entries.iter().map(|(a, s)| (a.clone(), hamming(q, s))).collect();
        v.sort_by(|x, y| x.1.cmp(&y.1).then(x.0.cmp(&y.0)));
        v.truncate(k);
        v
    }
}

/// Build a C2 index from a MEM-OKF store: scan full/<addr>.md and index every object that carries
/// a `mem_c2: <64hex>` frontmatter field (the objects with a captured latent signature — episodes).
/// Text/agent facts without a latent sig are simply absent from the similarity overlay (by design:
/// you can only find-similar over memories that HAVE a latent fingerprint).
pub fn index_store(root: &Path) -> C2Index {
    let mut idx = C2Index::new();
    let full = root.join("full");
    if let Ok(rd) = std::fs::read_dir(&full) {
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().to_string();
            let addr = match name.strip_suffix(".md") { Some(a) => a.to_string(), None => continue };
            if let Ok(text) = std::fs::read_to_string(e.path()) {
                let (fm, _body) = crate::parse_fm(&text);
                if let Some(c2) = fm.get("mem_c2").and_then(|h| sig_from_hex(h)) {
                    idx.insert(addr, c2);
                }
            }
        }
    }
    idx
}

/// Read all mem_c2 sigs keyed by addr (helper for gossip: advertise the C2 set you hold).
pub fn advertised_sigs(root: &Path) -> BTreeMap<String, Sig> {
    index_store(root).entries.into_iter().collect()
}
