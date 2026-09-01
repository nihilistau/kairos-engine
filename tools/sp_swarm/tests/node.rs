//! G-SWARM-NODE — the integration orchestration gate. Proves the reusable `run_node` (what the
//! daemon spawns behind its `swarm` feature): persistent identity, roster-file parsing, and an
//! autonomous periodic serve+pull loop that converges two live nodes over QUIC. No the dense reference model, no daemon.
#![cfg(feature = "transport")]

use ed25519_dalek::Signer;
use sp_swarm::transport::{load_or_create_identity, load_roster, run_node, Identity, NodeConfig};
use std::collections::HashSet;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

fn tmp(label: &str) -> PathBuf {
    let ns = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let d = std::env::temp_dir().join(format!("spswarm_node_{label}_{ns}"));
    std::fs::create_dir_all(d.join("full")).unwrap();
    d
}
fn write_signed(root: &PathBuf, body: &str, id: &Identity) -> String {
    let addr = sp_swarm::addr_of(body);
    let sig = id.sk.sign(&sp_swarm::signing_payload(&addr, body));
    let text = format!("---\nmem_signer: {}\nmem_sig: {}\n---\n\n{}", id.node_id, hex::encode(sig.to_bytes()), sp_swarm::norm(body));
    std::fs::write(root.join("full").join(format!("{addr}.md")), text).unwrap();
    addr
}
fn free_port() -> u16 {
    let s = std::net::UdpSocket::bind("127.0.0.1:0").unwrap();
    let p = s.local_addr().unwrap().port();
    drop(s);
    p
}

#[tokio::test]
async fn node_gate() {
    // (a) persistent identity: same keyfile -> same pubkey across "reboots"
    let kf = tmp("k").join("node.key");
    let i1 = load_or_create_identity("node-A", &kf).unwrap();
    let i2 = load_or_create_identity("node-A", &kf).unwrap();
    assert_eq!(i1.verifying_key().to_bytes(), i2.verifying_key().to_bytes(), "persistent identity must be stable");

    let a = Arc::new(load_or_create_identity("node-A", &tmp("ka").join("a.key")).unwrap());
    let b = Arc::new(load_or_create_identity("node-B", &tmp("kb").join("b.key")).unwrap());

    // (b) roster file round-trip
    let rf = tmp("r").join("roster.txt");
    std::fs::write(&rf, format!(
        "# invite-only roster\nnode-A {}\nnode-B {}\n",
        hex::encode(a.verifying_key().to_bytes()), hex::encode(b.verifying_key().to_bytes()))).unwrap();
    let roster = Arc::new(load_roster(&rf).unwrap());
    assert_eq!(roster.len(), 2, "roster file must parse 2 peers");

    // (c) divergent stores, autonomous periodic sync -> convergence
    let root_a = tmp("a");
    let root_b = tmp("b");
    let a_addrs: Vec<String> = (0..2).map(|i| write_signed(&root_a, &format!("A node fact {i}"), &a)).collect();
    let b_addrs: Vec<String> = (0..3).map(|i| write_signed(&root_b, &format!("B node fact {i}"), &b)).collect();

    let (pa, pb) = (free_port(), free_port());
    let addr_a: SocketAddr = format!("127.0.0.1:{pa}").parse().unwrap();
    let addr_b: SocketAddr = format!("127.0.0.1:{pb}").parse().unwrap();
    let iv = Duration::from_millis(150);
    let ha = tokio::spawn(run_node(Arc::clone(&a), Arc::clone(&roster), NodeConfig { listen: addr_a, peers: vec![addr_b], root: root_a.clone(), interval: iv }));
    let hb = tokio::spawn(run_node(Arc::clone(&b), Arc::clone(&roster), NodeConfig { listen: addr_b, peers: vec![addr_a], root: root_b.clone(), interval: iv }));

    tokio::time::sleep(Duration::from_millis(1400)).await; // several sync cycles
    ha.abort();
    hb.abort();

    let (hav, hbv) = (sp_swarm::have(&root_a), sp_swarm::have(&root_b));
    let union: HashSet<String> = a_addrs.iter().chain(b_addrs.iter()).cloned().collect();
    for x in &union {
        assert!(hav.contains(x), "A missing {x}");
        assert!(hbv.contains(x), "B missing {x}");
    }
    assert_eq!(hav, hbv, "both nodes must converge to the union");
    eprintln!("==== G-SWARM-NODE: GREEN — persistent identity stable, roster file parsed, run_node autonomous periodic sync converged {} objects both directions ====", union.len());

    for d in [root_a, root_b] { let _ = std::fs::remove_dir_all(d); }
}
