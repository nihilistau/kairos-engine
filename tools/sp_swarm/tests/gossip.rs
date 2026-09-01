//! G-SWARM-GOSSIP-DISCOVERY (L4 network) — a query on Node A discovers, via C2 shortlist gossip
//! over QUIC, an object held ONLY by Node B, exact-fetches + verifies it, and converges. Plus:
//! decoys A already holds are NOT re-fetched, and an off-roster peer's discovery is rejected.
#![cfg(feature = "transport")]

use ed25519_dalek::Signer;
use sp_swarm::similarity::{sig_to_hex, Sig};
use sp_swarm::transport::{bind, discover_similar, serve, Identity};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

fn tmp(l: &str) -> PathBuf {
    let ns = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let d = std::env::temp_dir().join(format!("spswarm_gossip_{l}_{ns}"));
    std::fs::create_dir_all(d.join("full")).unwrap();
    d
}
fn smix(mut x: u64) -> u64 { x = x.wrapping_add(0x9E3779B97F4A7C15); let mut z = x; z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9); z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB); z ^ (z >> 31) }
fn rand_sig(s: u64) -> Sig { [smix(s), smix(s ^ 0x11), smix(s ^ 0x22), smix(s ^ 0x33)] }
fn flip(s: &Sig, k: usize, seed: u64) -> Sig {
    let mut o = *s; let mut used = std::collections::HashSet::new(); let mut r = seed | 1; let mut d = 0;
    while d < k { r = smix(r); let b = (r % 256) as usize; if used.insert(b) { o[b / 64] ^= 1u64 << (b % 64); d += 1; } }
    o
}
/// content-addressed object signed by `id`, carrying a `mem_c2` sig (indexable for L4).
fn write_c2(root: &PathBuf, body: &str, id: &Identity, c2: &Sig) -> String {
    let addr = sp_swarm::addr_of(body);
    let sig = id.sk.sign(&sp_swarm::signing_payload(&addr, body));
    let text = format!("---\nmem_signer: {}\nmem_sig: {}\nmem_c2: {}\n---\n\n{}",
        id.node_id, hex::encode(sig.to_bytes()), sig_to_hex(c2), sp_swarm::norm(body));
    std::fs::write(root.join("full").join(format!("{addr}.md")), text).unwrap();
    addr
}

#[tokio::test]
async fn gossip_discovery_gate() {
    let a = Arc::new(Identity::generate("node-A"));
    let b = Arc::new(Identity::generate("node-B"));
    let c = Identity::generate("node-C"); // off-roster
    let mut roster: HashMap<String, _> = HashMap::new();
    roster.insert("node-A".into(), a.verifying_key());
    roster.insert("node-B".into(), b.verifying_key());
    let roster = Arc::new(roster);

    let a_root = tmp("a");
    let b_root = tmp("b");
    let s = rand_sig(0xBEEF);                                   // the target's C2 sig
    let target = write_c2(&b_root, "B-only memory: override code for node QRZ-91 is 7K2-XX", &b, &s);
    let far1 = flip(&s, 120, 7);
    let far2 = flip(&s, 118, 8);
    let d1 = write_c2(&b_root, "decoy one — unrelated content alpha", &b, &far1);
    let d2 = write_c2(&b_root, "decoy two — unrelated content beta", &b, &far2);
    // A ALREADY holds the two decoys (identical objects) — only the target is missing
    write_c2(&a_root, "decoy one — unrelated content alpha", &a, &far1);
    write_c2(&a_root, "decoy two — unrelated content beta", &a, &far2);

    let b_ep = bind("127.0.0.1:0".parse().unwrap()).unwrap();
    let b_addr = b_ep.local_addr().unwrap();
    tokio::spawn(serve(b_ep, Arc::clone(&b), Arc::clone(&roster), b_root.clone()));

    // A discovers with a NEAR query (4 bits off the target), k=5 (shortlist regime)
    let q = flip(&s, 4, 3);
    let rep = discover_similar(b_addr, &a, &roster, q, 5, a_root.clone()).await.expect("rostered discovery");
    let got = sp_swarm::have(&a_root);
    assert!(got.contains(&target), "A must discover+fetch B's target via C2 gossip");
    assert_eq!(rep.pulled, vec![target.clone()], "should fetch ONLY the missing similar object (decoys already held): {:?}", rep.pulled);
    eprintln!("[1] A discovered+verified the B-only object '{}' via C2 shortlist (decoys skipped)", target);

    // off-roster C is rejected — no discovery, no fetch
    let pc = tmp("c");
    let rc = discover_similar(b_addr, &c, &roster, q, 5, pc.clone()).await;
    assert!(rc.is_err(), "off-roster discovery MUST be rejected (got {:?})", rc.map(|r| r.pulled.len()));
    assert_eq!(sp_swarm::have(&pc).len(), 0);
    eprintln!("[2] off-roster C rejected: {:?}", rc.err().map(|e| e.to_string()));
    eprintln!("==== G-SWARM-GOSSIP-DISCOVERY: GREEN — C2 shortlist gossip -> exact-fetch verify -> converge (only-missing fetched); off-roster rejected ====");

    for d in [a_root, b_root, pc] { let _ = std::fs::remove_dir_all(d); }
}
