// Phase 6-NET QUIC transport — wire types. TLS, endpoints, and loop added in Tasks 4-7.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use quinn::{Connection, Endpoint, RecvStream};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use dashmap::DashMap;
use tokio::sync::mpsc;

// ── Peer registry ─────────────────────────────────────────────────────────────

/// A QUIC peer registered in the DHT mesh.
/// Defined here (lib) so run_garner_loop can write it without depending on the
/// binary's state module.  Re-exported and used by AppState (binary side).
#[derive(Clone, Debug)]
pub struct ConnectedPeer {
    /// 0 = q1 shard (prime 1073738753), 1 = q2 shard (prime 1073732609).
    /// u8::MAX = shard unknown (connection accepted, no block received yet).
    pub shard_id: u8,
}

// ── Error type ────────────────────────────────────────────────────────────────

pub type ShardError = Box<dyn std::error::Error + Send + Sync + 'static>;
pub type Result<T> = std::result::Result<T, ShardError>;

// ── Wire types ────────────────────────────────────────────────────────────────

/// 64-byte stream header preceding each NTT residue payload.
/// All multi-byte fields are little-endian.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShardBlockHeader {
    pub seq_id:         u64,      //  0..8   global sequence counter
    pub token_pos:      u32,      //  8..12  token position in context
    pub layer_id:       u32,      // 12..16  transformer layer index
    pub prime_selector: u8,       // 16      0 = q1 (1073738753), 1 = q2 (1073732609)
    pub _pad:           [u8; 47], // 17..64  reserved zeros
}
const _: () = assert!(std::mem::size_of::<ShardBlockHeader>() == 64);

/// Residue block transmitted over one QUIC unidirectional stream.
pub struct ResidueBlock {
    pub header:   ShardBlockHeader,
    pub residues: Vec<u32>,
}

// ── Serialization (no serde, no protobuf) ─────────────────────────────────────

pub fn header_to_bytes(h: &ShardBlockHeader) -> [u8; 64] {
    let mut buf = [0u8; 64];
    buf[0..8].copy_from_slice(&h.seq_id.to_le_bytes());
    buf[8..12].copy_from_slice(&h.token_pos.to_le_bytes());
    buf[12..16].copy_from_slice(&h.layer_id.to_le_bytes());
    buf[16] = h.prime_selector;
    buf
}

pub fn header_from_bytes(b: &[u8; 64]) -> ShardBlockHeader {
    ShardBlockHeader {
        seq_id:         u64::from_le_bytes(b[0..8].try_into().unwrap()),
        token_pos:      u32::from_le_bytes(b[8..12].try_into().unwrap()),
        layer_id:       u32::from_le_bytes(b[12..16].try_into().unwrap()),
        prime_selector: b[16],
        _pad:           [0u8; 47],
    }
}

// ── TLS ───────────────────────────────────────────────────────────────────────

/// Dev-mode TLS verifier: accepts any server certificate.
/// INTEGRATION POINT: Replace with Phase 5 ed25519 dominance identity in a later phase.
/// When Phase 5 identity is implemented, swap this struct for one that verifies
/// the peer's ed25519 public key against the known lattice node registry.
#[derive(Debug)]
struct SkipServerVerification;

impl rustls::client::danger::ServerCertVerifier for SkipServerVerification {
    fn verify_server_cert(
        &self,
        _end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &rustls::pki_types::ServerName<'_>,
        _ocsp_response: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> std::result::Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> std::result::Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> std::result::Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        vec![
            rustls::SignatureScheme::RSA_PKCS1_SHA256,
            rustls::SignatureScheme::RSA_PKCS1_SHA384,
            rustls::SignatureScheme::RSA_PKCS1_SHA512,
            rustls::SignatureScheme::ECDSA_NISTP256_SHA256,
            rustls::SignatureScheme::ECDSA_NISTP384_SHA384,
            rustls::SignatureScheme::ECDSA_NISTP521_SHA512,
            rustls::SignatureScheme::RSA_PSS_SHA256,
            rustls::SignatureScheme::RSA_PSS_SHA384,
            rustls::SignatureScheme::RSA_PSS_SHA512,
            rustls::SignatureScheme::ED25519,
        ]
    }
}

pub fn make_server_config() -> Result<quinn::ServerConfig> {
    let ck = rcgen::generate_simple_self_signed(vec!["localhost".into()])?;
    let cert_der: CertificateDer<'static> = ck.cert.der().clone();
    let key_der = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(ck.key_pair.serialize_der()));

    let tls = rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(vec![cert_der], key_der)?;

    let quic_server = quinn::crypto::rustls::QuicServerConfig::try_from(tls)?;
    let mut server_config = quinn::ServerConfig::with_crypto(Arc::new(quic_server));

    let mut transport = quinn::TransportConfig::default();
    transport.keep_alive_interval(Some(Duration::from_secs(30)));
    transport.max_idle_timeout(Some(quinn::VarInt::from_u32(120_000).into()));
    server_config.transport_config(Arc::new(transport));

    Ok(server_config)
}

pub fn make_client_config() -> Result<quinn::ClientConfig> {
    let tls = rustls::ClientConfig::builder()
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(SkipServerVerification))
        .with_no_client_auth();

    let quic_client = quinn::crypto::rustls::QuicClientConfig::try_from(tls)?;
    let mut client_config = quinn::ClientConfig::new(Arc::new(quic_client));

    let mut transport = quinn::TransportConfig::default();
    transport.keep_alive_interval(Some(Duration::from_secs(30)));
    transport.max_idle_timeout(Some(quinn::VarInt::from_u32(120_000).into()));
    client_config.transport_config(Arc::new(transport));

    Ok(client_config)
}

// ── Coordinator ───────────────────────────────────────────────────────────────

pub struct SpQuicCoordinator {
    endpoint: Endpoint,
}

impl SpQuicCoordinator {
    /// Bind a QUIC server endpoint on `addr`.
    pub async fn bind(addr: SocketAddr) -> Result<Self> {
        let server_config = make_server_config()?;
        let endpoint = Endpoint::server(server_config, addr)?;
        Ok(Self { endpoint })
    }

    /// Accept the next incoming QUIC connection (blocks until one arrives).
    pub async fn accept_connection(&self) -> Result<Connection> {
        let incoming = self.endpoint.accept().await
            .ok_or("coordinator endpoint closed")?;
        let conn = incoming.await?;
        Ok(conn)
    }

    pub fn local_addr(&self) -> std::io::Result<SocketAddr> {
        self.endpoint.local_addr()
    }
}

// ── Worker ────────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct SpQuicWorker {
    connection: Connection,
}

impl SpQuicWorker {
    /// Dial the coordinator at `server_addr`, binding the client endpoint on
    /// `local_addr` (use port 0 for OS-assigned port).
    pub async fn connect(local_addr: SocketAddr, server_addr: SocketAddr) -> Result<Self> {
        let client_config = make_client_config()?;
        let mut endpoint = Endpoint::client(local_addr)?;
        endpoint.set_default_client_config(client_config);
        let conn = endpoint.connect(server_addr, "localhost")?.await?;
        Ok(Self { connection: conn })
    }

    /// Open a fresh unidirectional stream and write `block`.
    /// Each call to send_block opens its own stream ID — independent delivery,
    /// no HoL coupling between blocks.
    pub async fn send_block(&self, block: &ResidueBlock) -> Result<()> {
        let mut send = self.connection.open_uni().await?;

        send.write_all(&header_to_bytes(&block.header)).await?;
        for r in &block.residues {
            send.write_all(&r.to_le_bytes()).await?;
        }
        send.finish()?;
        Ok(())
    }
}

// ── Receive ───────────────────────────────────────────────────────────────────

/// Read all bytes from a unidirectional stream and decode a ResidueBlock.
/// First 64 bytes = ShardBlockHeader; remaining bytes / 4 = residues.
/// Max payload: 64 + 512 * 4 = 2112 bytes (N ≤ 512).
pub async fn recv_block(mut stream: RecvStream) -> Result<ResidueBlock> {
    let bytes = stream.read_to_end(64 + 512 * 4).await?;

    if bytes.len() < 64 {
        return Err(format!("stream too short: {} bytes (need ≥ 64)", bytes.len()).into());
    }
    if (bytes.len() - 64) % 4 != 0 {
        return Err("residue payload length not a multiple of 4".into());
    }

    let header = header_from_bytes(bytes[0..64].try_into().unwrap());
    let residues = bytes[64..]
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    Ok(ResidueBlock { header, residues })
}

// ── Garner assembly loop ──────────────────────────────────────────────────────

#[derive(Default)]
struct PendingBlock {
    q1:        Option<Vec<u32>>,
    q2:        Option<Vec<u32>>,
    token_pos: u32,
    layer_id:  u32,
}

pub struct GarnerResult {
    pub seq_id:    u64,
    pub token_pos: u32,
    pub layer_id:  u32,
    pub coeffs:    Vec<i64>,
}

/// Accept QUIC connections and streams from shard workers. When both q1 and q2
/// residues arrive for the same seq_id, call ntt_crt_recombine and send the
/// result on `results_tx`. Runs until the coordinator endpoint is dropped.
///
/// `peer_map` is updated on every connection event:
///   accept → insert ConnectedPeer { shard_id: u8::MAX }
///   first block → refine shard_id from block.header.prime_selector
///   close/error → remove entry (no ghost nodes in the UI)
pub async fn run_garner_loop(
    coordinator: SpQuicCoordinator,
    ntt_n: u32,
    results_tx: mpsc::Sender<GarnerResult>,
    peer_map: Arc<DashMap<SocketAddr, ConnectedPeer>>,
) -> Result<()> {
    let pending: Arc<DashMap<u64, PendingBlock>> = Arc::new(DashMap::new());

    loop {
        let conn = match coordinator.accept_connection().await {
            Ok(c) => c,
            Err(_) => break,
        };

        let remote_addr = conn.remote_address();
        // Register immediately; shard_id unknown until first block arrives.
        peer_map.insert(remote_addr, ConnectedPeer { shard_id: u8::MAX });

        // Explicit close-watcher: fires as soon as the connection terminates
        // (graceful or otherwise), regardless of whether any streams were opened.
        // The accept_uni loop below also calls remove on break — both are safe
        // because DashMap::remove is idempotent on already-absent keys.
        let peer_map_cleanup = Arc::clone(&peer_map);
        let conn_for_close = conn.clone();
        tokio::spawn(async move {
            conn_for_close.closed().await;
            peer_map_cleanup.remove(&remote_addr);
            tracing::info!("SP_INFO: QUIC peer disconnected: {remote_addr}");
        });

        let pending     = Arc::clone(&pending);
        let results_tx  = results_tx.clone();
        let peer_map_c  = Arc::clone(&peer_map);

        tokio::spawn(async move {
            loop {
                let stream = match conn.accept_uni().await {
                    Ok(s) => s,
                    Err(_) => break,
                };

                let pending    = Arc::clone(&pending);
                let results_tx = results_tx.clone();
                let peer_map_s = Arc::clone(&peer_map_c);

                tokio::spawn(async move {
                    let block = match recv_block(stream).await {
                        Ok(b) => b,
                        Err(_) => return,
                    };

                    // Refine shard_id from the first block seen from this peer.
                    peer_map_s.entry(remote_addr).and_modify(|p| {
                        if p.shard_id == u8::MAX {
                            p.shard_id = block.header.prime_selector;
                        }
                    });

                    let seq_id    = block.header.seq_id;
                    let prime_sel = block.header.prime_selector;

                    // Insert residue; check atomically if both primes arrived.
                    // DashMap entry() holds a shard lock for the duration of the block.
                    let both_arrived = {
                        let mut entry = pending.entry(seq_id).or_insert_with(|| PendingBlock {
                            q1: None,
                            q2: None,
                            token_pos: block.header.token_pos,
                            layer_id:  block.header.layer_id,
                        });
                        if prime_sel == 0 {
                            entry.q1 = Some(block.residues);
                        } else {
                            entry.q2 = Some(block.residues);
                        }
                        entry.q1.is_some() && entry.q2.is_some()
                    }; // shard lock released here

                    if both_arrived {
                        // Atomic remove — exactly one task wins if two race here.
                        if let Some((_, pb)) = pending.remove(&seq_id) {
                            if let (Some(q1), Some(q2)) = (pb.q1, pb.q2) {
                                if q1.len() != ntt_n as usize || q2.len() != ntt_n as usize {
                                    return;
                                }
                                let mut coeffs = vec![0i64; ntt_n as usize];
                                unsafe {
                                    use crate::ntt_ffi::{ntt_crt_recombine, ntt_free, ntt_init};
                                    let ctx = ntt_init(ntt_n);
                                    if ctx.is_null() { return; }
                                    ntt_crt_recombine(
                                        ctx,
                                        q1.as_ptr(),
                                        q2.as_ptr(),
                                        coeffs.as_mut_ptr(),
                                    );
                                    ntt_free(ctx);
                                }
                                let _ = results_tx.send(GarnerResult {
                                    seq_id,
                                    token_pos: pb.token_pos,
                                    layer_id:  pb.layer_id,
                                    coeffs,
                                }).await;
                            }
                        }
                    }
                });
            }
            // Connection closed or errored — remove from the active peer map
            // so the UI doesn't display ghost nodes.
            peer_map_c.remove(&remote_addr);
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_is_64_bytes() {
        assert_eq!(std::mem::size_of::<ShardBlockHeader>(), 64);
    }

    #[test]
    fn header_roundtrip() {
        let h = ShardBlockHeader {
            seq_id:         0xDEAD_BEEF_CAFE_1234,
            token_pos:      77,
            layer_id:       3,
            prime_selector: 1,
            _pad:           [0u8; 47],
        };
        let bytes = header_to_bytes(&h);
        let h2 = header_from_bytes(&bytes);
        assert_eq!(h2.seq_id,         h.seq_id);
        assert_eq!(h2.token_pos,      h.token_pos);
        assert_eq!(h2.layer_id,       h.layer_id);
        assert_eq!(h2.prime_selector, h.prime_selector);
    }

    #[test]
    fn tls_configs_construct_without_panic() {
        make_server_config().expect("server config");
        make_client_config().expect("client config");
    }

    // coordinator_binds_on_loopback is inline here because the integration test
    // path (tests/test_quic_shard.rs) is blocked by a pre-existing probe.rs linker
    // issue (same reason tls_configs_construct_without_panic is inline — see Task 4).
    #[tokio::test]
    async fn coordinator_binds_on_loopback() {
        let coord = SpQuicCoordinator::bind("127.0.0.1:0".parse().unwrap())
            .await
            .expect("coordinator bind failed");

        let addr = coord.local_addr().expect("local_addr");
        assert_eq!(addr.ip().to_string(), "127.0.0.1");
        assert_ne!(addr.port(), 0); // OS assigned a real port
    }

    // worker_connect_and_roundtrip is inline here (same probe linker issue
    // blocks tests/test_quic_shard.rs for async QUIC tests — see Task 4/5).
    #[tokio::test]
    async fn worker_connect_and_roundtrip() {
        use std::time::Duration;

        let coord = SpQuicCoordinator::bind("127.0.0.1:0".parse().unwrap())
            .await
            .expect("bind");
        let coord_addr = coord.local_addr().unwrap();

        let accept = tokio::spawn(async move {
            let conn = coord.accept_connection().await.expect("accept");
            let stream = conn.accept_uni().await.expect("uni");
            recv_block(stream).await.expect("recv_block")
        });

        let worker = SpQuicWorker::connect(
            "127.0.0.1:0".parse().unwrap(),
            coord_addr,
        )
        .await
        .expect("connect");

        let block = ResidueBlock {
            header: ShardBlockHeader {
                seq_id: 99,
                token_pos: 7,
                layer_id: 2,
                prime_selector: 0,
                _pad: [0u8; 47],
            },
            residues: (0u32..128).collect(),
        };
        worker.send_block(&block).await.expect("send_block");

        let received = tokio::time::timeout(Duration::from_secs(5), accept)
            .await
            .expect("timeout")
            .expect("task panic");

        assert_eq!(received.header.seq_id, 99);
        assert_eq!(received.header.prime_selector, 0);
        assert_eq!(received.residues.len(), 128);
        assert_eq!(received.residues[0], 0);
        assert_eq!(received.residues[127], 127);
    }

    /// M_NET_1: Three-node topology — coordinator accepts two independent worker
    /// connections and receives one stream from each.
    #[tokio::test]
    async fn m_net_1_topology_scaffold() {
        use std::time::Duration;

        let coord = SpQuicCoordinator::bind("127.0.0.1:8081".parse().unwrap())
            .await
            .expect("coordinator bind :8081");

        let accept = tokio::spawn(async move {
            // Connection from worker A (q1)
            let conn_a = coord.accept_connection().await.expect("accept conn_a");
            let stream_a = conn_a.accept_uni().await.expect("uni from A");
            let block_a = recv_block(stream_a).await.expect("block A");
            assert_eq!(block_a.header.prime_selector, 0, "worker A must send q1");
            assert_eq!(block_a.residues.len(), 128);

            // Connection from worker B (q2)
            let conn_b = coord.accept_connection().await.expect("accept conn_b");
            let stream_b = conn_b.accept_uni().await.expect("uni from B");
            let block_b = recv_block(stream_b).await.expect("block B");
            assert_eq!(block_b.header.prime_selector, 1, "worker B must send q2");
            assert_eq!(block_b.residues.len(), 128);
        });

        // Worker A: local :8082, connects to coord :8081
        let worker_a = SpQuicWorker::connect(
            "127.0.0.1:8082".parse().unwrap(),
            "127.0.0.1:8081".parse().unwrap(),
        ).await.expect("worker A connect");

        worker_a.send_block(&ResidueBlock {
            header: ShardBlockHeader { seq_id: 0, token_pos: 0, layer_id: 0,
                                       prime_selector: 0, _pad: [0; 47] },
            residues: vec![1u32; 128],
        }).await.expect("worker A send");

        // Worker B: local :8083, connects to coord :8081
        let worker_b = SpQuicWorker::connect(
            "127.0.0.1:8083".parse().unwrap(),
            "127.0.0.1:8081".parse().unwrap(),
        ).await.expect("worker B connect");

        worker_b.send_block(&ResidueBlock {
            header: ShardBlockHeader { seq_id: 0, token_pos: 0, layer_id: 0,
                                       prime_selector: 1, _pad: [0; 47] },
            residues: vec![2u32; 128],
        }).await.expect("worker B send");

        // Keep workers alive until acceptor completes
        tokio::time::timeout(Duration::from_secs(10), accept)
            .await
            .expect("M_NET_1 timed out")
            .expect("acceptor panicked");
        drop(worker_a);
        drop(worker_b);
    }

    /// M_NET_2: Garner reconstruction via QUIC must be bit-identical to local
    /// ntt_crt_recombine reference.
    #[tokio::test]
    async fn m_net_2_math_identity() {
        use crate::ntt_ffi::{ntt_crt_recombine, ntt_free, ntt_init};
        use std::time::Duration;
        use tokio::sync::mpsc;

        const Q1: u32 = 1073738753;
        const Q2: u32 = 1073732609;
        const N: usize = 128;

        let q1_residues: Vec<u32> = (0..N as u32).map(|i| i % Q1).collect();
        let q2_residues: Vec<u32> = (0..N as u32).map(|i| i % Q2).collect();

        // Scalar reference (direct FFI, no network)
        let expected: Vec<i64> = unsafe {
            let ctx = ntt_init(N as u32);
            assert!(!ctx.is_null());
            let mut out = vec![0i64; N];
            ntt_crt_recombine(ctx, q1_residues.as_ptr(), q2_residues.as_ptr(), out.as_mut_ptr());
            ntt_free(ctx);
            out
        };

        // Start coordinator on :8084
        let coord = SpQuicCoordinator::bind("127.0.0.1:8084".parse().unwrap())
            .await
            .expect("bind :8084");
        let (results_tx, mut results_rx) = mpsc::channel(4);
        tokio::spawn(run_garner_loop(coord, N as u32, results_tx, Arc::new(DashMap::new())));
        tokio::time::sleep(Duration::from_millis(20)).await;

        // Worker A sends q1 from :8085
        let wa = SpQuicWorker::connect(
            "127.0.0.1:8085".parse().unwrap(),
            "127.0.0.1:8084".parse().unwrap(),
        ).await.expect("worker A");
        wa.send_block(&ResidueBlock {
            header: ShardBlockHeader { seq_id: 42, token_pos: 10, layer_id: 5,
                                       prime_selector: 0, _pad: [0; 47] },
            residues: q1_residues,
        }).await.expect("send q1");

        // Worker B sends q2 from :8086
        let wb = SpQuicWorker::connect(
            "127.0.0.1:8086".parse().unwrap(),
            "127.0.0.1:8084".parse().unwrap(),
        ).await.expect("worker B");
        wb.send_block(&ResidueBlock {
            header: ShardBlockHeader { seq_id: 42, token_pos: 10, layer_id: 5,
                                       prime_selector: 1, _pad: [0; 47] },
            residues: q2_residues,
        }).await.expect("send q2");

        // Keep workers alive until result received
        let result = tokio::time::timeout(Duration::from_secs(5), results_rx.recv())
            .await
            .expect("M_NET_2 timeout — no reconstruction received")
            .expect("results channel closed");

        drop(wa);
        drop(wb);

        assert_eq!(result.seq_id, 42, "seq_id mismatch");
        assert_eq!(result.token_pos, 10);
        assert_eq!(result.layer_id, 5);
        assert_eq!(
            result.coeffs, expected,
            "M_NET_2 FAIL: Garner reconstruction via QUIC not bit-identical to scalar reference"
        );
    }

    /// M_NET_3: HoL bypass — block 1 (sent immediately) must arrive at coordinator
    /// within 100ms even though block 0 (independent stream) is delayed 200ms.
    ///
    /// Falsifiability: a buggy serialized-stream implementation would deliver block 1
    /// at ~200ms (behind block 0's delay), failing the 100ms timeout.
    #[tokio::test]
    async fn m_net_3_hol_bypass() {
        use std::time::Duration;

        let coord = SpQuicCoordinator::bind("127.0.0.1:8087".parse().unwrap())
            .await
            .expect("bind :8087");

        let accept = tokio::spawn(async move {
            let conn = coord.accept_connection().await.expect("accept");

            // Block 1 arrives first (t≈0ms); block 0 arrives second (t≈200ms)
            let stream1 = conn.accept_uni().await.expect("first uni stream");
            let block1 = tokio::time::timeout(
                Duration::from_millis(100),
                recv_block(stream1),
            )
            .await
            .expect("M_NET_3 FAIL: block 1 did not arrive within 100ms — HoL blocking detected")
            .expect("recv_block failed on block 1");
            assert_eq!(block1.header.seq_id, 1, "first stream must be seq_id=1 (immediate)");

            // Block 0 arrives after 200ms delay
            let stream0 = conn.accept_uni().await.expect("second uni stream");
            let block0 = tokio::time::timeout(
                Duration::from_millis(500),
                recv_block(stream0),
            )
            .await
            .expect("M_NET_3: block 0 did not arrive within 500ms")
            .expect("recv_block failed on block 0");
            assert_eq!(block0.header.seq_id, 0, "second stream must be seq_id=0 (delayed)");
        });

        // Single worker on :8088
        let worker = SpQuicWorker::connect(
            "127.0.0.1:8088".parse().unwrap(),
            "127.0.0.1:8087".parse().unwrap(),
        ).await.expect("worker connect");

        // Block 0 (seq_id=0): sleep 200ms then send. Stream ID opened AFTER sleep.
        let worker_clone = worker.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(200)).await;
            worker_clone.send_block(&ResidueBlock {
                header: ShardBlockHeader { seq_id: 0, token_pos: 0, layer_id: 0,
                                           prime_selector: 0, _pad: [0; 47] },
                residues: vec![0u32; 128],
            }).await.expect("send block 0");
        });

        // Block 1 (seq_id=1): sent immediately. Own stream ID, independent.
        worker.send_block(&ResidueBlock {
            header: ShardBlockHeader { seq_id: 1, token_pos: 1, layer_id: 0,
                                       prime_selector: 1, _pad: [0; 47] },
            residues: vec![1u32; 128],
        }).await.expect("send block 1");

        tokio::time::timeout(Duration::from_secs(3), accept)
            .await
            .expect("M_NET_3 overall timeout")
            .expect("acceptor panicked");
        // keep worker alive
    }

    // garner_loop_reconstructs_single_pair is inline here (same probe linker
    // issue blocks integration tests for async QUIC endpoints — see Task 4/5).
    #[tokio::test]
    async fn garner_loop_reconstructs_single_pair() {
        use crate::ntt_ffi::{ntt_crt_recombine, ntt_free, ntt_init};
        use std::time::Duration;

        const Q1: u32 = 1073738753;
        const Q2: u32 = 1073732609;
        const N: usize = 128;

        let q1: Vec<u32> = (0..N as u32).map(|i| i % Q1).collect();
        let q2: Vec<u32> = (0..N as u32).map(|i| i % Q2).collect();

        // Scalar reference (direct FFI, no network)
        let expected: Vec<i64> = unsafe {
            let ctx = ntt_init(N as u32);
            let mut out = vec![0i64; N];
            ntt_crt_recombine(ctx, q1.as_ptr(), q2.as_ptr(), out.as_mut_ptr());
            ntt_free(ctx);
            out
        };

        let coord = SpQuicCoordinator::bind("127.0.0.1:0".parse().unwrap())
            .await
            .expect("bind");
        let coord_addr = coord.local_addr().unwrap();

        let (tx, mut rx) = mpsc::channel(4);
        tokio::spawn(run_garner_loop(coord, N as u32, tx, Arc::new(DashMap::new())));
        tokio::time::sleep(Duration::from_millis(20)).await;

        let wa = SpQuicWorker::connect("127.0.0.1:0".parse().unwrap(), coord_addr)
            .await.expect("wa connect");
        wa.send_block(&ResidueBlock {
            header: ShardBlockHeader { seq_id: 7, token_pos: 0, layer_id: 0,
                                       prime_selector: 0, _pad: [0; 47] },
            residues: q1,
        }).await.expect("send q1");

        let wb = SpQuicWorker::connect("127.0.0.1:0".parse().unwrap(), coord_addr)
            .await.expect("wb connect");
        wb.send_block(&ResidueBlock {
            header: ShardBlockHeader { seq_id: 7, token_pos: 0, layer_id: 0,
                                       prime_selector: 1, _pad: [0; 47] },
            residues: q2,
        }).await.expect("send q2");

        // Keep wa, wb alive until result received — prevents FIN loss
        let result = tokio::time::timeout(Duration::from_secs(5), rx.recv())
            .await
            .expect("garner loop timeout")
            .expect("channel closed");

        drop(wa);
        drop(wb);

        assert_eq!(result.seq_id, 7);
        assert_eq!(result.coeffs, expected, "Garner reconstruction not bit-identical");
    }
}
