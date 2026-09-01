//! SP-SWARM L0 — network transport (QUIC + Ed25519 invite-only roster).
//!
//! QUIC (quinn 0.11 + rustls 0.23/ring — the engine's proven, loopback-gated stack) provides
//! the encrypted-authenticated channel the SWARM doc §2 L0 calls for (TLS 1.3: X25519 key
//! agreement, ChaCha20-Poly1305/AES-GCM AEAD). On top of that channel, an application-level
//! **Ed25519 mutual challenge-response** authenticates each peer against an INVITE-ONLY ROSTER
//! (node_id -> verifying key): a peer whose id is not in the roster, or who cannot sign the
//! nonce, is rejected before any object moves. Objects then flow via a have/want pull that runs
//! `crate::accept` (L1 re-hash + L2 class + L3 provenance) on EVERY object BEFORE it is written.
//!
//! Not rust-libp2p: the engine already owns this QUIC transport, and libp2p's DHT/gossipsub are
//! overkill for a closed invite-only mesh with known peer addresses.

use crate::{accept, full_dir, have};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use quinn::{Endpoint, RecvStream, SendStream};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use std::collections::HashMap;
use std::fs;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

pub type TErr = Box<dyn std::error::Error + Send + Sync + 'static>;
pub type Roster = HashMap<String, VerifyingKey>;

/// A node identity: an Ed25519 keypair + a human node_id (roster key).
pub struct Identity {
    pub node_id: String,
    pub sk: SigningKey,
}
impl Identity {
    pub fn generate(node_id: &str) -> Self {
        let mut seed = [0u8; 32];
        getrandom::getrandom(&mut seed).expect("getrandom");
        Identity { node_id: node_id.to_string(), sk: SigningKey::from_bytes(&seed) }
    }
    pub fn verifying_key(&self) -> VerifyingKey { self.sk.verifying_key() }
}

// ---- QUIC/TLS config (copied verbatim from the proven tools/sp_daemon/src/network/quic_shard.rs;
//      self-signed cert = transport encryption; identity is the Ed25519 roster layer above) ----
#[derive(Debug)]
struct SkipServerVerification;
impl rustls::client::danger::ServerCertVerifier for SkipServerVerification {
    fn verify_server_cert(&self, _e: &CertificateDer<'_>, _i: &[CertificateDer<'_>],
        _s: &rustls::pki_types::ServerName<'_>, _o: &[u8], _n: rustls::pki_types::UnixTime)
        -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }
    fn verify_tls12_signature(&self, _m: &[u8], _c: &CertificateDer<'_>, _d: &rustls::DigitallySignedStruct)
        -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }
    fn verify_tls13_signature(&self, _m: &[u8], _c: &CertificateDer<'_>, _d: &rustls::DigitallySignedStruct)
        -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }
    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        rustls::crypto::ring::default_provider().signature_verification_algorithms.supported_schemes()
    }
}

fn server_config() -> Result<quinn::ServerConfig, TErr> {
    let ck = rcgen::generate_simple_self_signed(vec!["localhost".into()])?;
    let cert_der: CertificateDer<'static> = ck.cert.der().clone();
    let key_der = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(ck.key_pair.serialize_der()));
    let tls = rustls::ServerConfig::builder().with_no_client_auth().with_single_cert(vec![cert_der], key_der)?;
    let qs = quinn::crypto::rustls::QuicServerConfig::try_from(tls)?;
    let mut sc = quinn::ServerConfig::with_crypto(Arc::new(qs));
    let mut t = quinn::TransportConfig::default();
    t.max_idle_timeout(Some(quinn::VarInt::from_u32(30_000).into()));
    sc.transport_config(Arc::new(t));
    Ok(sc)
}
fn client_config() -> Result<quinn::ClientConfig, TErr> {
    let tls = rustls::ClientConfig::builder().dangerous()
        .with_custom_certificate_verifier(Arc::new(SkipServerVerification)).with_no_client_auth();
    let qc = quinn::crypto::rustls::QuicClientConfig::try_from(tls)?;
    Ok(quinn::ClientConfig::new(Arc::new(qc)))
}

// ---- length-prefixed framing (u32 LE len + payload) ----
async fn wr(s: &mut SendStream, b: &[u8]) -> Result<(), TErr> {
    s.write_all(&(b.len() as u32).to_le_bytes()).await?;
    s.write_all(b).await?;
    Ok(())
}
async fn rd(r: &mut RecvStream) -> Result<Vec<u8>, TErr> {
    let mut l = [0u8; 4];
    r.read_exact(&mut l).await?;
    let n = u32::from_le_bytes(l) as usize;
    if n > 64 * 1024 * 1024 { return Err("frame too large".into()); }
    let mut b = vec![0u8; n];
    r.read_exact(&mut b).await?;
    Ok(b)
}
fn nonce() -> [u8; 32] { let mut n = [0u8; 32]; getrandom::getrandom(&mut n).expect("getrandom"); n }

// ---- Ed25519 mutual roster auth over the encrypted stream ----
// Client proves it holds its key (signs server nonce) and that its id is rostered; server does
// the same. Either side rejects an unrostered id or a bad signature. Returns the peer's node_id.
async fn client_auth(s: &mut SendStream, r: &mut RecvStream, me: &Identity, roster: &Roster) -> Result<String, TErr> {
    let nc = nonce();
    wr(s, me.node_id.as_bytes()).await?;
    wr(s, &nc).await?;
    let peer_id = String::from_utf8(rd(r).await?)?;
    let ns = rd(r).await?;
    let sig_c = Signature::from_slice(&rd(r).await?)?;
    let vk = roster.get(&peer_id).ok_or("untrusted-signer")?;         // invite-only gate
    vk.verify(&nc, &sig_c).map_err(|_| "sig-invalid")?;               // server holds its key
    wr(s, &me.sk.sign(&ns).to_bytes()).await?;                        // prove we hold ours
    if rd(r).await? != b"OK" { return Err("peer-rejected".into()); }
    Ok(peer_id)
}
async fn server_auth(s: &mut SendStream, r: &mut RecvStream, me: &Identity, roster: &Roster) -> Result<String, TErr> {
    let peer_id = String::from_utf8(rd(r).await?)?;
    let nc = rd(r).await?;
    let vk = match roster.get(&peer_id) { Some(v) => *v, None => { let _ = wr(s, b"NO").await; return Err("untrusted-signer".into()); } };
    let ns = nonce();
    wr(s, me.node_id.as_bytes()).await?;
    wr(s, &ns).await?;
    wr(s, &me.sk.sign(&nc).to_bytes()).await?;                        // prove we hold ours
    let sig_s = Signature::from_slice(&rd(r).await?)?;
    if vk.verify(&ns, &sig_s).is_err() { let _ = wr(s, b"NO").await; return Err("sig-invalid".into()); }
    wr(s, b"OK").await?;
    Ok(peer_id)
}

fn have_set(root: &PathBuf) -> String {
    let mut v: Vec<String> = have(root).into_iter().collect();
    v.sort();
    v.join("\n")
}

/// Serve have/want to authenticated, rostered peers forever on `ep`.
pub async fn serve(ep: Endpoint, me: Arc<Identity>, roster: Arc<Roster>, root: PathBuf) {
    while let Some(inc) = ep.accept().await {
        let (me, roster, root) = (Arc::clone(&me), Arc::clone(&roster), root.clone());
        tokio::spawn(async move {
            let conn = match inc.await { Ok(c) => c, Err(_) => return };
            let (mut s, mut r) = match conn.accept_bi().await { Ok(x) => x, Err(_) => return };
            if server_auth(&mut s, &mut r, &me, &roster).await.is_err() { return; } // off-roster => drop
            loop {
                let cmd = match rd(&mut r).await { Ok(c) => c, Err(_) => break };
                if cmd == b"LIST" {
                    let _ = wr(&mut s, have_set(&root).as_bytes()).await;
                } else if cmd.starts_with(b"GET:") {
                    let addr = String::from_utf8_lossy(&cmd[4..]).to_string();
                    match fs::read(full_dir(&root).join(format!("{addr}.md"))) {
                        Ok(b) => { let _ = wr(&mut s, &b).await; }
                        Err(_) => { let _ = wr(&mut s, b"MISS").await; }
                    }
                } else if cmd.starts_with(b"SIM:") {
                    // L4 discovery: "SIM:<c2_hex>:<k>" -> top-k local candidates "addr hamming" per
                    // line (a HINT shortlist; the caller exact-fetches + verifies, per the C2
                    // semantic gate — C2 is a shortlist, not a top-1 oracle).
                    let q = String::from_utf8_lossy(&cmd[4..]).to_string();
                    let reply = match q.split_once(':') {
                        Some((hexs, kstr)) => match crate::similarity::sig_from_hex(hexs) {
                            Some(sig) => {
                                let k = kstr.parse::<usize>().unwrap_or(5).max(1);
                                crate::similarity::index_store(&root)
                                    .find_similar(&sig, k)
                                    .into_iter()
                                    .map(|(a, h)| format!("{a} {h}"))
                                    .collect::<Vec<_>>()
                                    .join("\n")
                            }
                            None => String::new(),
                        },
                        None => String::new(),
                    };
                    let _ = wr(&mut s, reply.as_bytes()).await;
                } else if cmd == b"DONE" { break; }
            }
        });
    }
}

pub struct SyncReport { pub pulled: Vec<String>, pub rejected: Vec<(String, String)> }

/// Connect to `server`, authenticate against the roster, and pull every object we lack —
/// verifying each on arrival (L1+L2+L3) before writing. Returns what was pulled / rejected.
pub async fn pull_from(server: SocketAddr, me: &Identity, roster: &Roster, local: PathBuf) -> Result<SyncReport, TErr> {
    let mut ep = Endpoint::client("127.0.0.1:0".parse().unwrap())?;
    ep.set_default_client_config(client_config()?);
    let conn = ep.connect(server, "localhost")?.await?;
    let (mut s, mut r) = conn.open_bi().await?;
    client_auth(&mut s, &mut r, me, roster).await?;            // Err if peer off-roster / bad sig
    wr(&mut s, b"LIST").await?;
    let hv = rd(&mut r).await?;
    let peer_have: std::collections::HashSet<String> =
        String::from_utf8_lossy(&hv).lines().filter(|x| !x.is_empty()).map(|x| x.to_string()).collect();
    let local_have = have(&local);
    let (mut pulled, mut rejected) = (Vec::new(), Vec::new());
    for addr in peer_have.difference(&local_have) {
        let mut cmd = b"GET:".to_vec();
        cmd.extend_from_slice(addr.as_bytes());
        wr(&mut s, &cmd).await?;
        let resp = rd(&mut r).await?;
        if resp == b"MISS" { rejected.push((addr.clone(), "missing-on-remote".into())); continue; }
        let text = String::from_utf8_lossy(&resp).to_string();
        match accept(addr, &text, Some(roster)) {            // L1 re-hash + L2 class + L3 provenance
            Ok(_) => {
                let _ = fs::create_dir_all(full_dir(&local));
                let _ = fs::write(full_dir(&local).join(format!("{addr}.md")), &text);
                pulled.push(addr.clone());
            }
            Err(e) => rejected.push((addr.clone(), e.as_str().into())),
        }
    }
    let _ = wr(&mut s, b"DONE").await;
    let _ = s.finish();
    conn.close(0u32.into(), b"done");
    Ok(SyncReport { pulled, rejected })
}

/// L4 discovery: ask a peer for the top-k C2-nearest candidates to `query`, then exact-fetch +
/// verify (`accept`) any we don't already hold. C2 is a shortlist HINT (semantic gate: recall@5
/// 0.885, recall@1 only 0.607) — correctness comes from the exact-fetch verify, so a wrong hint
/// costs only bandwidth. Use k>=5 (the shortlist regime the gate justified).
pub async fn discover_similar(
    server: SocketAddr, me: &Identity, roster: &Roster,
    query: crate::similarity::Sig, k: usize, local: PathBuf,
) -> Result<SyncReport, TErr> {
    let mut ep = Endpoint::client("127.0.0.1:0".parse().unwrap())?;
    ep.set_default_client_config(client_config()?);
    let conn = ep.connect(server, "localhost")?.await?;
    let (mut s, mut r) = conn.open_bi().await?;
    client_auth(&mut s, &mut r, me, roster).await?;
    let sim = format!("SIM:{}:{}", crate::similarity::sig_to_hex(&query), k.max(1));
    wr(&mut s, sim.as_bytes()).await?;
    let resp = rd(&mut r).await?;
    // candidate addresses, nearest first (peer already ranked by Hamming)
    let cands: Vec<String> = String::from_utf8_lossy(&resp)
        .lines().filter(|l| !l.is_empty())
        .filter_map(|l| l.split_whitespace().next().map(|a| a.to_string()))
        .collect();
    let local_have = have(&local);
    let (mut pulled, mut rejected) = (Vec::new(), Vec::new());
    for addr in cands {
        if local_have.contains(&addr) { continue; }
        let mut cmd = b"GET:".to_vec();
        cmd.extend_from_slice(addr.as_bytes());
        wr(&mut s, &cmd).await?;
        let obj = rd(&mut r).await?;
        if obj == b"MISS" { rejected.push((addr, "missing-on-remote".into())); continue; }
        let text = String::from_utf8_lossy(&obj).to_string();
        match accept(&addr, &text, Some(roster)) {   // exact-fetch verify (L1+L2+L3)
            Ok(_) => {
                let _ = fs::create_dir_all(full_dir(&local));
                let _ = fs::write(full_dir(&local).join(format!("{addr}.md")), &text);
                pulled.push(addr);
            }
            Err(e) => rejected.push((addr, e.as_str().into())),
        }
    }
    let _ = wr(&mut s, b"DONE").await;
    let _ = s.finish();
    conn.close(0u32.into(), b"done");
    Ok(SyncReport { pulled, rejected })
}

/// Bind a server endpoint on `addr` (e.g. "127.0.0.1:0"); read `.local_addr()` for the port.
pub fn bind(addr: SocketAddr) -> Result<Endpoint, TErr> {
    Ok(Endpoint::server(server_config()?, addr)?)
}

// ==== Integration orchestration (what the daemon calls) =========================================

/// Load a PERSISTENT identity from a 32-byte hex seed file, creating it on first run. The daemon's
/// own node_signing_key is EPHEMERAL (regenerated per boot) — the mesh needs a stable pubkey so a
/// node's roster entry survives reboots. `node_id` is the roster key (e.g. the hostname).
pub fn load_or_create_identity(node_id: &str, keyfile: &std::path::Path) -> Result<Identity, TErr> {
    let seed: [u8; 32] = if keyfile.exists() {
        let h = fs::read_to_string(keyfile)?;
        hex::decode(h.trim())?.try_into().map_err(|_| "bad key seed length")?
    } else {
        let mut s = [0u8; 32];
        getrandom::getrandom(&mut s).map_err(|e| format!("getrandom: {e}"))?;
        if let Some(p) = keyfile.parent() { let _ = fs::create_dir_all(p); }
        fs::write(keyfile, hex::encode(s))?;
        s
    };
    Ok(Identity { node_id: node_id.to_string(), sk: SigningKey::from_bytes(&seed) })
}

/// Roster file = one `node_id <pubkey_hex>` per line (comments `#`, blank lines skipped).
pub fn load_roster(path: &std::path::Path) -> Result<Roster, TErr> {
    let mut r = Roster::new();
    for line in fs::read_to_string(path)?.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') { continue; }
        let mut it = line.split_whitespace();
        if let (Some(id), Some(pk)) = (it.next(), it.next()) {
            let b: [u8; 32] = hex::decode(pk)?.try_into().map_err(|_| "bad pubkey length")?;
            r.insert(id.to_string(), VerifyingKey::from_bytes(&b)?);
        }
    }
    Ok(r)
}

pub struct NodeConfig {
    pub listen: SocketAddr,
    pub peers: Vec<SocketAddr>,
    pub root: PathBuf,
    pub interval: Duration,
}

/// The full mesh node: serve have/want to rostered peers on `listen`, and every `interval` pull
/// from each configured peer (verify-on-arrival). Runs forever; the daemon spawns this behind its
/// `swarm` feature when SP_SWARM=1. Default-off in the daemon = this is never spawned (null floor).
pub async fn run_node(id: Arc<Identity>, roster: Arc<Roster>, cfg: NodeConfig) -> Result<(), TErr> {
    let ep = bind(cfg.listen)?;
    tokio::spawn(serve(ep, Arc::clone(&id), Arc::clone(&roster), cfg.root.clone()));
    loop {
        for &p in &cfg.peers {
            match pull_from(p, &id, &roster, cfg.root.clone()).await {
                Ok(r) if !r.pulled.is_empty() || !r.rejected.is_empty() =>
                    eprintln!("swarm: {} pulled {} rejected {} from {p}", id.node_id, r.pulled.len(), r.rejected.len()),
                Ok(_) => {}
                Err(e) => eprintln!("swarm: {} pull {p} failed: {e}", id.node_id),
            }
        }
        tokio::time::sleep(cfg.interval).await;
    }
}

/// Idle transport-config keepalive knob (kept for parity with quic_shard; unused in tests).
pub const _KEEPALIVE: Duration = Duration::from_secs(30);
