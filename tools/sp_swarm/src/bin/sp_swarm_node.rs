//! sp-swarm-node — standalone SP-SWARM mesh node (the deployable form of the integration).
//!
//! Serves the local MEM-OKF store to rostered peers and periodically pulls from them, over the
//! QUIC + Ed25519-roster transport. This is exactly what `sp-daemon` spawns behind its `swarm`
//! feature; run it standalone for a headless replication node (no the dense reference model).
//!
//! Config (env):
//!   SP_SWARM_NODE_ID   roster key for this node (default "node")
//!   SP_SWARM_KEY       persistent Ed25519 seed file (created on first run; default "swarm.key")
//!   SP_SWARM_ROSTER    invite-only roster file: "node_id <pubkey_hex>" per line (default "roster.txt")
//!   SP_SWARM_ROOT      MEM-OKF store dir to replicate (falls back to SP_OKF_ROOT, then "memory-okf")
//!   SP_SWARM_LISTEN    bind addr (default "0.0.0.0:0"; use a fixed port so peers can dial you)
//!   SP_SWARM_PEERS     comma-separated peer addrs to pull from (e.g. "10.0.0.2:7777,10.0.0.3:7777")
//!   SP_SWARM_INTERVAL_S pull cadence in seconds (default 30)

use sp_swarm::transport::{load_or_create_identity, load_roster, run_node, NodeConfig};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync + 'static>> {
    let env = |k: &str, d: &str| std::env::var(k).unwrap_or_else(|_| d.to_string());
    let node_id = env("SP_SWARM_NODE_ID", "node");
    let key = env("SP_SWARM_KEY", "swarm.key");
    let roster_path = env("SP_SWARM_ROSTER", "roster.txt");
    let root = std::env::var("SP_SWARM_ROOT")
        .or_else(|_| std::env::var("SP_OKF_ROOT"))
        .unwrap_or_else(|_| "memory-okf".to_string());
    let listen = env("SP_SWARM_LISTEN", "0.0.0.0:0");
    let peers = env("SP_SWARM_PEERS", "");
    let interval: u64 = std::env::var("SP_SWARM_INTERVAL_S").ok().and_then(|s| s.parse().ok()).unwrap_or(30);

    let id = Arc::new(load_or_create_identity(&node_id, Path::new(&key))?);
    eprintln!("sp-swarm-node id={node_id} pubkey={}", hex::encode(id.verifying_key().to_bytes()));
    let roster = Arc::new(load_roster(Path::new(&roster_path))?);
    let peers_v: Vec<std::net::SocketAddr> =
        peers.split(',').filter(|s| !s.is_empty()).filter_map(|s| s.parse().ok()).collect();
    let cfg = NodeConfig { listen: listen.parse()?, peers: peers_v, root: root.into(), interval: Duration::from_secs(interval) };
    eprintln!("sp-swarm-node listen={} peers={:?} root={} interval={interval}s roster_peers={}",
        cfg.listen, cfg.peers, cfg.root.display(), roster.len());
    run_node(id, roster, cfg).await
}
