//! G-SWARM-TRANSPORT-QUIC (L0) — 2-node localhost gate over the real QUIC+rustls channel.
//! Proves the transport metrics the SWARM doc §8 requires (never the organism):
//!   - roster auth + encrypted round-trip: a rostered peer connects (QUIC/TLS 1.3) and pulls;
//!   - convergence (both directions): puller reaches the origin's clean set;
//!   - verify-on-arrival over the wire: a tampered object is rejected, never written (L3);
//!   - invite-only: an OFF-ROSTER peer's handshake is rejected (no object moves).
#![cfg(feature = "transport")]

use ed25519_dalek::{Signer, VerifyingKey};
use sp_swarm::transport::{bind, pull_from, serve, Identity};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

fn tmp(label: &str) -> PathBuf {
    let ns = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let d = std::env::temp_dir().join(format!("spswarm_{label}_{ns}"));
    std::fs::create_dir_all(d.join("full")).unwrap();
    d
}

/// Write a content-addressed object signed by `id` (mirror of Python swarm_provenance.sign_object).
fn write_signed(root: &PathBuf, body: &str, id: &Identity) -> String {
    let addr = sp_swarm::addr_of(body);
    let sig = id.sk.sign(&sp_swarm::signing_payload(&addr, body));
    let text = format!(
        "---\nmem_signer: {}\nmem_sig: {}\n---\n\n{}",
        id.node_id, hex::encode(sig.to_bytes()), sp_swarm::norm(body)
    );
    std::fs::write(root.join("full").join(format!("{addr}.md")), text).unwrap();
    addr
}

#[tokio::test]
async fn transport_gate() {
    // identities: A and B are rostered; C is NOT (invite-only test)
    let a = Arc::new(Identity::generate("node-A"));
    let b = Arc::new(Identity::generate("node-B"));
    let c = Identity::generate("node-C");
    let mut roster: HashMap<String, VerifyingKey> = HashMap::new();
    roster.insert("node-A".into(), a.verifying_key());
    roster.insert("node-B".into(), b.verifying_key());
    let roster = Arc::new(roster);

    // origin stores: B has 3 clean + 1 TAMPERED; A has 2 clean
    let a_root = tmp("origA");
    let b_root = tmp("origB");
    let b_clean: Vec<String> = (0..3).map(|i| write_signed(&b_root, &format!("B fact {i}: override code Z{i}-{i}{i}"), &b)).collect();
    let tampered = write_signed(&b_root, "B fact TAMPERED: secret payload", &b);
    { // corrupt the tampered object on disk AFTER signing -> signature no longer matches
        let p = b_root.join("full").join(format!("{tampered}.md"));
        let s = std::fs::read_to_string(&p).unwrap();
        std::fs::write(&p, format!("{s}\nTAMPERED-IN-TRANSIT")).unwrap();
    }
    let a_clean: Vec<String> = (0..2).map(|i| write_signed(&a_root, &format!("A fact {i}: rack A-{i} coolant type"), &a)).collect();

    // start both servers on localhost
    let b_ep = bind("127.0.0.1:0".parse().unwrap()).unwrap();
    let b_addr = b_ep.local_addr().unwrap();
    tokio::spawn(serve(b_ep, Arc::clone(&b), Arc::clone(&roster), b_root.clone()));
    let a_ep = bind("127.0.0.1:0".parse().unwrap()).unwrap();
    let a_addr = a_ep.local_addr().unwrap();
    tokio::spawn(serve(a_ep, Arc::clone(&a), Arc::clone(&roster), a_root.clone()));

    // (1)+(3) A' pulls B: converges to B's 3 clean; tampered rejected on arrival, not written
    let pull_a = tmp("pullA");
    let rep = pull_from(b_addr, &a, &roster, pull_a.clone()).await.expect("rostered pull must succeed");
    let got = sp_swarm::have(&pull_a);
    assert_eq!(rep.pulled.len(), 3, "should pull B's 3 clean objects, got {:?}", rep.pulled);
    for c in &b_clean { assert!(got.contains(c), "missing {c}"); }
    assert!(!got.contains(&tampered), "TAMPERED object must NOT be written");
    assert!(rep.rejected.iter().any(|(a, _)| a == &tampered), "tampered must be in rejected: {:?}", rep.rejected);
    eprintln!("[1] A<-B converge: pulled={} rejected={:?}", rep.pulled.len(), rep.rejected);

    // (2) reverse direction: B' pulls A -> converges to A's 2 clean
    let pull_b = tmp("pullB");
    let rep2 = pull_from(a_addr, &b, &roster, pull_b.clone()).await.expect("reverse rostered pull must succeed");
    let got2 = sp_swarm::have(&pull_b);
    assert_eq!(rep2.pulled.len(), 2, "should pull A's 2 clean objects");
    for c in &a_clean { assert!(got2.contains(c)); }
    eprintln!("[2] B<-A converge: pulled={}", rep2.pulled.len());

    // (4) invite-only: off-roster C is rejected -> no objects move
    let pull_c = tmp("pullC");
    let rc = pull_from(b_addr, &c, &roster, pull_c.clone()).await;
    assert!(rc.is_err(), "off-roster peer MUST be rejected (got {:?})", rc.map(|r| r.pulled.len()));
    assert_eq!(sp_swarm::have(&pull_c).len(), 0, "off-roster peer received nothing");
    eprintln!("[4] off-roster C rejected: {:?}", rc.err().map(|e| e.to_string()));

    for d in [a_root, b_root, pull_a, pull_b, pull_c] { let _ = std::fs::remove_dir_all(d); }
    eprintln!("==== G-SWARM-TRANSPORT-QUIC: GREEN — roster-auth + encrypted round-trip + bidirectional convergence + verify-on-arrival + off-roster reject ====");
}
