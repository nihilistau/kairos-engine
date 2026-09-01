// ring2_quic.rs — Trick #8 wire-closure: a remote peer as the ARM Ring-2 store.
//
// To the math-core, a peer is just another registered `sp_arm_ring2_backend`:
// the canonical decode calls the same write_block / read_block / read_batch
// fn pointers whether the bytes land on the Optane NVMe (ring2_arm_backend.c),
// the portable stdio store, or — here — a QUIC socket. The payload on the wire
// is the raw, untranslated u32 dual-prime residue block: the cache line IS the
// transport packet. A token recalled from NVMe and a token recalled from a
// peer are computationally indistinguishable to the math-core (proven by
// composition: T_GENKV_REGISTERED_BACKEND pins backend-indistinguishability;
// the gate below pins this backend's byte-exact contract over a real socket
// through the real C registry).
//
// Wire protocol (Trick #9 framing discipline — 64-byte LE header, one
// BIDIRECTIONAL stream per request, no HoL coupling between requests):
//   R2Req { magic "SPR2", op(0=read,1=write), which(0=K,1=V), len u32, off u64 }
//   write: header + len payload  -> resp: [status u8]
//   read:  header                -> resp: [status u8] + len payload (status==0)
// The server stores blocks keyed by (which, off) — the peer-side store.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex, OnceLock};
use quinn::Connection;
use tokio::runtime::Runtime;

use super::quic_shard::{make_client_config, make_server_config, Result, ShardError};

pub const R2_MAGIC: u32 = 0x5350_5232; // "SPR2"
pub const R2_HDR: usize = 64;

#[derive(Clone, Copy, Debug)]
pub struct R2Req {
    pub op: u8,     // 0 = read, 1 = write
    pub which: u8,  // 0 = K stream, 1 = V stream
    pub len: u32,   // block bytes
    pub off: u64,   // byte offset within the stream
}

pub fn req_to_bytes(r: &R2Req) -> [u8; R2_HDR] {
    let mut b = [0u8; R2_HDR];
    b[0..4].copy_from_slice(&R2_MAGIC.to_le_bytes());
    b[4] = r.op;
    b[5] = r.which;
    b[8..12].copy_from_slice(&r.len.to_le_bytes());
    b[16..24].copy_from_slice(&r.off.to_le_bytes());
    b
}

pub fn req_from_bytes(b: &[u8; R2_HDR]) -> Option<R2Req> {
    if u32::from_le_bytes(b[0..4].try_into().unwrap()) != R2_MAGIC {
        return None;
    }
    Some(R2Req {
        op: b[4],
        which: b[5],
        len: u32::from_le_bytes(b[8..12].try_into().unwrap()),
        off: u64::from_le_bytes(b[16..24].try_into().unwrap()),
    })
}

// ── server: the peer-side block store ────────────────────────────────────────

/// Serve Ring-2 blocks forever on `endpoint`. Each request is one bi-stream.
pub async fn run_ring2_server(endpoint: quinn::Endpoint) -> Result<()> {
    let store: Arc<dashmap::DashMap<(u8, u64), Vec<u8>>> = Arc::new(dashmap::DashMap::new());
    while let Some(incoming) = endpoint.accept().await {
        let conn = incoming.await?;
        let store = Arc::clone(&store);
        tokio::spawn(async move {
            loop {
                let (mut send, mut recv) = match conn.accept_bi().await {
                    Ok(s) => s,
                    Err(_) => break,
                };
                let store = Arc::clone(&store);
                tokio::spawn(async move {
                    let mut hdr = [0u8; R2_HDR];
                    if recv.read_exact(&mut hdr).await.is_err() { return; }
                    let req = match req_from_bytes(&hdr) { Some(r) => r, None => return };
                    match req.op {
                        1 => { // write: header + payload -> status
                            let mut buf = vec![0u8; req.len as usize];
                            if recv.read_exact(&mut buf).await.is_err() { return; }
                            store.insert((req.which, req.off), buf);
                            let _ = send.write_all(&[0u8]).await;
                            let _ = send.finish();
                        }
                        0 => { // read: header -> status + payload
                            match store.get(&(req.which, req.off)) {
                                Some(blk) if blk.len() == req.len as usize => {
                                    let _ = send.write_all(&[0u8]).await;
                                    let _ = send.write_all(&blk).await;
                                    let _ = send.finish();
                                }
                                _ => { let _ = send.write_all(&[1u8]).await; let _ = send.finish(); }
                            }
                        }
                        _ => {}
                    }
                });
            }
        });
    }
    Ok(())
}

/// Bind a Ring-2 server endpoint on `addr` (e.g. "127.0.0.1:0").
pub fn bind_ring2_server(addr: SocketAddr) -> Result<quinn::Endpoint> {
    let cfg = make_server_config()?;
    Ok(quinn::Endpoint::server(cfg, addr)?)
}

// ── client: the registered backend ───────────────────────────────────────────

pub struct Ring2QuicClient {
    rt: Runtime,
    conn: Connection,
}

impl Ring2QuicClient {
    /// Dial the peer. Owns a dedicated current-thread runtime so the C-ABI
    /// calls (synchronous, from decode threads) can block_on safely.
    pub fn connect(server: SocketAddr) -> Result<Self> {
        let rt = Runtime::new().map_err(|e| Box::new(e) as ShardError)?;
        let conn = rt.block_on(async {
            let cfg = make_client_config()?;
            let mut ep = quinn::Endpoint::client("0.0.0.0:0".parse().unwrap())?;
            ep.set_default_client_config(cfg);
            let c = ep.connect(server, "localhost")?.await?;
            Ok::<Connection, ShardError>(c)
        })?;
        Ok(Ring2QuicClient { rt, conn })
    }

    async fn one_write(conn: &Connection, which: u8, off: u64, src: &[u8]) -> Result<()> {
        let (mut send, mut recv) = conn.open_bi().await?;
        let req = R2Req { op: 1, which, len: src.len() as u32, off };
        send.write_all(&req_to_bytes(&req)).await?;
        send.write_all(src).await?;
        send.finish()?;
        let mut st = [0u8; 1];
        recv.read_exact(&mut st).await?;
        if st[0] != 0 { return Err("ring2 quic write rejected".into()); }
        Ok(())
    }

    async fn one_read(conn: &Connection, which: u8, off: u64, dst: &mut [u8]) -> Result<()> {
        let (mut send, mut recv) = conn.open_bi().await?;
        let req = R2Req { op: 0, which, len: dst.len() as u32, off };
        send.write_all(&req_to_bytes(&req)).await?;
        send.finish()?;
        let mut st = [0u8; 1];
        recv.read_exact(&mut st).await?;
        if st[0] != 0 { return Err("ring2 quic read miss".into()); }
        recv.read_exact(dst).await?;
        Ok(())
    }

    pub fn write_block(&self, which: u8, off: u64, src: &[u8]) -> Result<()> {
        self.rt.block_on(Self::one_write(&self.conn, which, off, src))
    }

    pub fn read_block(&self, which: u8, off: u64, dst: &mut [u8]) -> Result<()> {
        self.rt.block_on(Self::one_read(&self.conn, which, off, dst))
    }

    /// Batched scattered reads: all n requests in flight concurrently — the
    /// QUIC analog of the Optane IOCP queue-depth batch.
    pub fn read_batch(&self, reqs: Vec<(u8, u64, &mut [u8])>) -> Result<()> {
        self.rt.block_on(async {
            let futs = reqs
                .into_iter()
                .map(|(w, o, d)| Self::one_read(&self.conn, w, o, d));
            for r in futures::future::join_all(futs).await {
                r?;
            }
            Ok(())
        })
    }
}

// ── the C-registry trampolines ───────────────────────────────────────────────

/// Wire telemetry: counts of blocks that actually crossed the socket.
pub static NET_WRITES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static NET_READS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static NET_BATCHES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static NET_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

static CLIENT: OnceLock<Mutex<Option<Ring2QuicClient>>> = OnceLock::new();
fn client_cell() -> &'static Mutex<Option<Ring2QuicClient>> {
    CLIENT.get_or_init(|| Mutex::new(None))
}

unsafe extern "C" fn r2q_write(
    _h: *mut std::ffi::c_void, which: std::ffi::c_int, off: u64,
    src: *const std::ffi::c_void, len: usize,
) -> std::ffi::c_int {
    let g = client_cell().lock().unwrap();
    let Some(c) = g.as_ref() else { return 1 };
    let s = std::slice::from_raw_parts(src as *const u8, len);
    NET_WRITES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    NET_BYTES.fetch_add(len as u64, std::sync::atomic::Ordering::Relaxed);
    if c.write_block(which as u8, off, s).is_ok() { 0 } else { 1 }
}

unsafe extern "C" fn r2q_read(
    _h: *mut std::ffi::c_void, which: std::ffi::c_int, off: u64,
    dst: *mut std::ffi::c_void, len: usize,
) -> std::ffi::c_int {
    let g = client_cell().lock().unwrap();
    let Some(c) = g.as_ref() else { return 1 };
    let d = std::slice::from_raw_parts_mut(dst as *mut u8, len);
    NET_READS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    NET_BYTES.fetch_add(len as u64, std::sync::atomic::Ordering::Relaxed);
    if c.read_block(which as u8, off, d).is_ok() { 0 } else { 1 }
}

unsafe extern "C" fn r2q_read_batch(
    _h: *mut std::ffi::c_void, which: *const std::ffi::c_int, off: *const u64,
    dst: *const *mut std::ffi::c_void, len: usize, n: std::ffi::c_int,
) -> std::ffi::c_int {
    let g = client_cell().lock().unwrap();
    let Some(c) = g.as_ref() else { return 1 };
    let n = n as usize;
    let whichs = std::slice::from_raw_parts(which, n);
    let offs = std::slice::from_raw_parts(off, n);
    let dsts = std::slice::from_raw_parts(dst, n);
    let reqs: Vec<(u8, u64, &mut [u8])> = (0..n)
        .map(|i| {
            (whichs[i] as u8, offs[i],
             std::slice::from_raw_parts_mut(dsts[i] as *mut u8, len))
        })
        .collect();
    NET_BATCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    NET_READS.fetch_add(n as u64, std::sync::atomic::Ordering::Relaxed);
    NET_BYTES.fetch_add((n as u64) * (len as u64), std::sync::atomic::Ordering::Relaxed);
    if c.read_batch(reqs).is_ok() { 0 } else { 1 }
}

/// Dial `addr` and register the peer as THE ARM Ring-2 backend through the
/// real C registry (sp_arm_ring2_register). Borrowed semantics: the decode
/// never closes a registered backend; call unregister_ring2_quic to tear down.
pub fn register_ring2_quic(addr: SocketAddr) -> Result<()> {
    let client = Ring2QuicClient::connect(addr)?;
    *client_cell().lock().unwrap() = Some(client);
    let be = crate::ffi_l1::sp_arm_ring2_backend {
        handle: std::ptr::null_mut(),
        write_block: Some(r2q_write),
        read_block: Some(r2q_read),
        read_batch: Some(r2q_read_batch),
        alloc_aligned: None,   // network store: no direct-I/O alignment needs
        free_aligned: None,
        close: None,           // borrowed: WE own the teardown
        read_batch2: None,     // overlap n/a: read_batch already flies concurrent
                               // QUIC streams; the two-call fallback is correct
    };
    unsafe { crate::ffi_l1::sp_arm_ring2_register(&be) };
    tracing::info!("SP_INFO: ARM Ring-2 backend = QUIC peer {addr} (raw residue blocks on the wire)");
    Ok(())
}

pub fn unregister_ring2_quic() {
    unsafe { crate::ffi_l1::sp_arm_ring2_register(std::ptr::null()) };
    *client_cell().lock().unwrap() = None;
}

// ── the gate ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// M_NET_RING2: the Trick #8 contract gate. Registers the QUIC peer through
    /// the REAL C registry, fetches the backend struct back the way the decode
    /// does (sp_arm_ring2_registered), and drives the RETURNED fn pointers:
    /// K-residue-sized blocks (8 KB — NKV*2N u32, the fusion block) and
    /// V-f32-sized blocks (4 KB) round-trip byte-exact over the socket, scattered
    /// read_batch included. Plain #[test]: the C calls must come from a non-async
    /// thread (the client owns its runtime), exactly like a decode thread.
    #[test]
    fn m_net_ring2_residue_blocks_over_quic() {
        // server on its own runtime thread (quinn needs the reactor at bind time)
        let (addr_tx, addr_rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let rt = Runtime::new().unwrap();
            rt.block_on(async {
                let ep = bind_ring2_server("127.0.0.1:0".parse().unwrap()).expect("bind");
                addr_tx.send(ep.local_addr().expect("addr")).unwrap();
                let _ = run_ring2_server(ep).await;
            });
        });
        let addr: SocketAddr = addr_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("server addr");

        // register through the real C registry
        register_ring2_quic(addr).expect("register");
        let mut be = unsafe { std::mem::zeroed::<crate::ffi_l1::sp_arm_ring2_backend>() };
        let present = unsafe { crate::ffi_l1::sp_arm_ring2_registered(&mut be) };
        assert_eq!(present, 1, "backend visible through the C registry");

        const KBLK: usize = 8192; // NKV(8) * 2N(256) * u32 — the fusion K block
        const VBLK: usize = 4096; // KVD(1024) * f32      — the V plumbing block
        let kblock = |seed: u8| (0..KBLK).map(|i| (i as u8) ^ seed).collect::<Vec<u8>>();
        let vblock = |seed: u8| (0..VBLK).map(|i| (i as u8).wrapping_add(seed)).collect::<Vec<u8>>();

        // write through the registered fn pointers (as the decode does)
        let wf = be.write_block.expect("write_block");
        for s in 0..6u8 {
            let kb = kblock(s);
            let vb = vblock(s);
            let rc = unsafe { wf(be.handle, 0, (s as u64) * KBLK as u64,
                                 kb.as_ptr() as *const _, KBLK) };
            assert_eq!(rc, 0, "K residue block write over QUIC");
            let rc = unsafe { wf(be.handle, 1, (s as u64) * VBLK as u64,
                                 vb.as_ptr() as *const _, VBLK) };
            assert_eq!(rc, 0, "V block write over QUIC");
        }

        // single reads: byte-exact
        let rf = be.read_block.expect("read_block");
        let mut buf = vec![0u8; KBLK];
        let rc = unsafe { rf(be.handle, 0, 3 * KBLK as u64, buf.as_mut_ptr() as *mut _, KBLK) };
        assert_eq!(rc, 0);
        assert_eq!(buf, kblock(3), "K residue block round-trips byte-exact over QUIC");

        // scattered batched reads (the dedupe-stage shape): byte-exact
        let bf = be.read_batch.expect("read_batch");
        let order: [u64; 4] = [5, 1, 4, 0];
        let mut dsts: Vec<Vec<u8>> = (0..4).map(|_| vec![0u8; KBLK]).collect();
        let whichs = [0i32; 4];
        let offs: Vec<u64> = order.iter().map(|s| s * KBLK as u64).collect();
        let mut ptrs: Vec<*mut std::ffi::c_void> =
            dsts.iter_mut().map(|d| d.as_mut_ptr() as *mut _).collect();
        let rc = unsafe { bf(be.handle, whichs.as_ptr(), offs.as_ptr(),
                             ptrs.as_mut_ptr() as *const _, KBLK, 4) };
        assert_eq!(rc, 0, "scattered read_batch over QUIC");
        for (i, s) in order.iter().enumerate() {
            assert_eq!(dsts[i], kblock(*s as u8), "batched block {s} byte-exact");
        }

        // V stream independent
        let mut vb = vec![0u8; VBLK];
        let rc = unsafe { rf(be.handle, 1, 2 * VBLK as u64, vb.as_mut_ptr() as *mut _, VBLK) };
        assert_eq!(rc, 0);
        assert_eq!(vb, vblock(2), "V block round-trips byte-exact over QUIC");

        // teardown clears the registry
        unregister_ring2_quic();
        let present = unsafe { crate::ffi_l1::sp_arm_ring2_registered(&mut be) };
        assert_eq!(present, 0, "unregister clears the hook");
    }
}
