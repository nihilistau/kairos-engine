//! SP-SWARM daemon integration — spawn the private memory mesh alongside the served the dense reference model.
//!
//! Default-off: `SP_SWARM` unset ⇒ `spawn_if_enabled` returns immediately (null floor; the daemon
//! behaves exactly as before). The mesh runs on its OWN QUIC port (`SP_SWARM_PORT`), separate from
//! the garner NTT-CRT coordinator (`--quic-port`). It uses a PERSISTENT Ed25519 identity from
//! `SP_SWARM_KEY` (NOT the daemon's ephemeral node key, which regenerates each boot) so the node's
//! roster entry is stable, and replicates the MEM-OKF store at `SP_OKF_ROOT` (or `SP_SWARM_ROOT`).
//! All reconciliation + provenance is the gated sp-swarm core (L1 re-hash, L2 have/want, L3 sig).
#![cfg(feature = "swarm")]

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

fn env(k: &str, d: &str) -> String { std::env::var(k).unwrap_or_else(|_| d.to_string()) }

/// Start the mesh node on a background task iff `SP_SWARM=1`. Safe no-op otherwise.
pub fn spawn_if_enabled() {
    if std::env::var("SP_SWARM").as_deref() != Ok("1") {
        return;
    }
    tokio::spawn(async move {
        use sp_swarm::transport::{load_or_create_identity, load_roster, run_node, NodeConfig};
        let node_id = env("SP_SWARM_NODE_ID", "node");
        let key = env("SP_SWARM_KEY", "swarm.key");
        let roster_path = env("SP_SWARM_ROSTER", "roster.txt");
        let root = std::env::var("SP_SWARM_ROOT")
            .or_else(|_| std::env::var("SP_OKF_ROOT"))
            .unwrap_or_else(|_| "memory-okf".to_string());
        let port = env("SP_SWARM_PORT", "7777");
        let listen = match format!("0.0.0.0:{port}").parse() {
            Ok(a) => a,
            Err(e) => { tracing::warn!("SP-SWARM: bad listen 0.0.0.0:{port}: {e} — not starting"); return; }
        };
        let id = match load_or_create_identity(&node_id, Path::new(&key)) {
            Ok(i) => Arc::new(i),
            Err(e) => { tracing::warn!("SP-SWARM: identity {key}: {e} — not starting"); return; }
        };
        let roster = match load_roster(Path::new(&roster_path)) {
            Ok(r) => Arc::new(r),
            Err(e) => { tracing::warn!("SP-SWARM: roster {roster_path}: {e} — not starting"); return; }
        };
        let peers: Vec<std::net::SocketAddr> = env("SP_SWARM_PEERS", "")
            .split(',').filter(|s| !s.is_empty()).filter_map(|s| s.parse().ok()).collect();
        let interval: u64 = std::env::var("SP_SWARM_INTERVAL_S").ok().and_then(|s| s.parse().ok()).unwrap_or(30);
        tracing::info!(
            "SP-SWARM: node '{node_id}' pubkey={} listen=0.0.0.0:{port} peers={peers:?} root={root} interval={interval}s roster_peers={}",
            hex::encode(id.verifying_key().to_bytes()), roster.len()
        );
        let cfg = NodeConfig { listen, peers, root: root.into(), interval: Duration::from_secs(interval) };
        if let Err(e) = run_node(id, roster, cfg).await {
            tracing::warn!("SP-SWARM: node exited: {e}");
        }
    });
}
