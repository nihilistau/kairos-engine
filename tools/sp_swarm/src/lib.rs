//! SP-SWARM core — the Rust port of the proven Python L1/L2/L3 (tools/swarm_sync.py +
//! swarm_provenance.py in the lattice repo). Content addressing (L1), have/want replication
//! (L2), Ed25519 provenance (L3) — all byte-for-byte compatible with the Python prototype so
//! a Python node and a Rust node interoperate. The L0 libp2p transport lands behind the
//! `transport` feature as the next brick; everything here is transport-agnostic.
//!
//! Interop invariants (gated by tests/parity.rs against a pynacl-produced fixture):
//!   addr    = sha256(norm(body))[..16]                      (== Python hashlib)
//!   norm    = body.replace("\r\n","\n").trim() + "\n"       (== Python norm)
//!   payload = format!("{addr}\n") ++ norm(body)             (== Python signing_payload)
//!   sig     = Ed25519 over payload, verified by ed25519-dalek (== libsodium/pynacl)

use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

/// Canonical body normalization — MUST match Python `okf_mem.norm`.
pub fn norm(body: &str) -> String {
    format!("{}\n", body.replace("\r\n", "\n").trim())
}

/// Content address = first 16 hex chars of sha256(norm(body)) — MUST match Python `addr_of`.
pub fn addr_of(body: &str) -> String {
    let mut h = Sha256::new();
    h.update(norm(body).as_bytes());
    hex::encode(h.finalize())[..16].to_string()
}

/// Ed25519 signing payload — MUST match Python `swarm_provenance.signing_payload`.
pub fn signing_payload(addr: &str, body: &str) -> Vec<u8> {
    let mut p = format!("{addr}\n").into_bytes();
    p.extend_from_slice(norm(body).as_bytes());
    p
}

/// Split a stored object into (frontmatter map, body) — mirrors Python `okf_mem.parse_fm`
/// (regex `^---\n(.*?)\n---\n?(.*)$`). Non-frontmatter text returns ({}, text).
pub fn parse_fm(text: &str) -> (BTreeMap<String, String>, String) {
    // Normalize line endings on ingest — Python's text-mode read (okf_mem.read) translates
    // CRLF->LF on Windows, so a Python-on-Windows node writes CRLF files whose ADDRESS was
    // computed over LF. A Rust node MUST normalize on read or every address mismatches
    // (cross-platform interop; caught by the real-store parity test).
    let text = text.replace("\r\n", "\n");
    let text = text.as_str();
    let mut fm = BTreeMap::new();
    if let Some(rest) = text.strip_prefix("---\n") {
        if let Some(end) = rest.find("\n---") {
            for line in rest[..end].split('\n') {
                if let Some(i) = line.find(':') {
                    fm.insert(line[..i].trim().to_string(), line[i + 1..].trim().to_string());
                }
            }
            let after = &rest[end + 4..]; // skip "\n---"
            let body = after.strip_prefix('\n').unwrap_or(after); // the regex's `\n?`
            return (fm, body.to_string());
        }
    }
    (fm, text.to_string())
}

/// MEM-OKF's two address classes (grounded finding G-SWARM-REPLICATE-CONVERGE).
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Class {
    Content, // agent facts: addr == sha256(norm(body)) — tamper-evident by re-hash
    C2,      // episodes: addr == external C2 SimHash (frontmatter self-consistent)
    Bad,     // tampered content object, or malformed
}

pub fn classify(addr: &str, fm: &BTreeMap<String, String>, body: &str) -> Class {
    if addr_of(body) == addr {
        Class::Content
    } else if fm.get("mem_addr").map(|s| s.as_str()) == Some(addr)
        && fm.get("mem_kind").map(|s| s.as_str()) == Some("episode")
    {
        Class::C2
    } else {
        Class::Bad
    }
}

/// L3 provenance reject reasons (same strings as the Python gate).
#[derive(Debug, PartialEq, Eq)]
pub enum Reject {
    Unsigned,
    UntrustedSigner,
    SigInvalid,
    IntegrityFail,
    Missing,
}
impl Reject {
    pub fn as_str(&self) -> &'static str {
        match self {
            Reject::Unsigned => "unsigned",
            Reject::UntrustedSigner => "untrusted-signer",
            Reject::SigInvalid => "sig-invalid",
            Reject::IntegrityFail => "integrity-fail",
            Reject::Missing => "missing-on-remote",
        }
    }
}

/// Verify Ed25519 provenance against an invite-only roster (node_id -> verifying key).
/// MUST accept a signature produced by Python's `swarm_provenance.sign_object` (pynacl).
pub fn verify_provenance(
    addr: &str,
    fm: &BTreeMap<String, String>,
    body: &str,
    roster: &HashMap<String, VerifyingKey>,
) -> Result<(), Reject> {
    let signer = fm.get("mem_signer").ok_or(Reject::Unsigned)?;
    let sig_hex = fm.get("mem_sig").ok_or(Reject::Unsigned)?;
    if signer.is_empty() || sig_hex.is_empty() {
        return Err(Reject::Unsigned);
    }
    let vk = roster.get(signer).ok_or(Reject::UntrustedSigner)?;
    let raw = hex::decode(sig_hex).map_err(|_| Reject::SigInvalid)?;
    let sig = Signature::from_slice(&raw).map_err(|_| Reject::SigInvalid)?;
    vk.verify(&signing_payload(addr, body), &sig)
        .map_err(|_| Reject::SigInvalid)
}

/// Full verify-on-arrival = L2 (class) + optional L3 (provenance). This is what a `pull`
/// runs before committing a fetched object to the local store.
pub fn accept(
    addr: &str,
    text: &str,
    roster: Option<&HashMap<String, VerifyingKey>>,
) -> Result<Class, Reject> {
    let (fm, body) = parse_fm(text);
    let cls = classify(addr, &fm, &body);
    if cls == Class::Bad {
        return Err(Reject::IntegrityFail);
    }
    if let Some(r) = roster {
        verify_provenance(addr, &fm, &body, r)?;
    }
    Ok(cls)
}

/// L4 C2-SimHash similarity overlay (pure std; the index + query, engine computes the sigs).
pub mod similarity;

/// L0 network transport (QUIC + Ed25519 roster) — optional, behind the `transport` feature.
#[cfg(feature = "transport")]
pub mod transport;

// ---- L2 store seam (reads/writes the MEM-OKF full/ dir; transport-agnostic) ----
pub fn full_dir(root: &Path) -> PathBuf { root.join("full") }

/// The set of object addresses a node HOLDS (from full/<addr>.md).
pub fn have(root: &Path) -> HashSet<String> {
    let mut s = HashSet::new();
    if let Ok(rd) = fs::read_dir(full_dir(root)) {
        for e in rd.flatten() {
            let n = e.file_name().to_string_lossy().to_string();
            if let Some(a) = n.strip_suffix(".md") {
                s.insert(a.to_string());
            }
        }
    }
    s
}

/// Fetch `addrs` from remote into local, verifying each on arrival BEFORE write. Returns
/// (pulled, rejected:(addr,reason)). `roster=Some(..)` enforces L3 provenance.
pub fn pull(
    remote: &Path,
    local: &Path,
    addrs: &[String],
    roster: Option<&HashMap<String, VerifyingKey>>,
) -> (Vec<String>, Vec<(String, String)>) {
    let _ = fs::create_dir_all(full_dir(local));
    let _ = fs::create_dir_all(local.join("sum"));
    let (mut pulled, mut rejected) = (Vec::new(), Vec::new());
    for addr in addrs {
        let rf = full_dir(remote).join(format!("{addr}.md"));
        let text = match fs::read_to_string(&rf) {
            Ok(t) => t,
            Err(_) => { rejected.push((addr.clone(), Reject::Missing.as_str().into())); continue; }
        };
        match accept(addr, &text, roster) {
            Ok(_) => {
                let _ = fs::write(full_dir(local).join(format!("{addr}.md")), &text);
                let rs = remote.join("sum").join(format!("{addr}.md"));
                if let Ok(s) = fs::read_to_string(&rs) {
                    let _ = fs::write(local.join("sum").join(format!("{addr}.md")), s);
                }
                pulled.push(addr.clone());
            }
            Err(r) => rejected.push((addr.clone(), r.as_str().into())),
        }
    }
    (pulled, rejected)
}
