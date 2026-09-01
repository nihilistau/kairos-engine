/* mining.rs — Background Friedman Sieve mining loop (Phase 5 PoUW).
 *
 * Generates synthetic KSTE candidates, feeds them to sp_sieve_evaluate,
 * and mints ed25519-signed receipts on sieve-fold events.
 *
 * Candidate generation: random int16 LE values in the Tier-0 (bytes 8..19)
 * and Tier-1 (bytes 20..55) label regions; header bytes 0..7 are set to
 * the frozen v1 KSTE constants.  This produces valid-format KSTE trees that
 * the C dominance checks compare correctly.
 *
 * Yields to the tokio runtime (tokio::task::yield_now) between batches, and
 * sleeps longer when inference_active is set to avoid starving /v1/chat.
 */

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use ed25519_dalek::{Signer, SigningKey};
use rand::{RngCore, SeedableRng, rngs::StdRng};
use tracing::debug;

use crate::sieve_ffi::{sp_sieve_evaluate, SpKsteTree, SpSieveEvent};
use crate::state::{DaemonEvent, ReceiptRecord};

/// How many minted receipts stay resident. Only .len() and a tail are ever read, so
/// retention beyond this buys nothing — and at ~1 receipt/second an unbounded Vec grew
/// ~40 MB/day. 4096 is ~1.9 MB and a bit over an hour of history at the observed rate.
const RECEIPT_RING_CAP: usize = 4096;
const BATCH_SIZE:    usize = 64;
const FRONTIER_CAP:  usize = 256;
const LABEL_BYTES:   usize = 48; // Tier-0 (12) + Tier-1 (36)
const LABEL_OFF:     usize = 8;  // bytes 8..55 in the 64-byte KSTE tree

/// Build a synthetic KSTE tree from 48 random bytes in the label region.
/// Header bytes: version=1, branching=3, depth=3, reserved=0.
fn synthetic_kste(rng: &mut impl RngCore) -> SpKsteTree {
    let mut t = SpKsteTree { bytes: [0u8; 64] };
    t.bytes[0] = 1; // SP_KSTE_LAYOUT_VERSION
    t.bytes[1] = 3; // SP_KSTE_BRANCHING
    t.bytes[2] = 3; // SP_KSTE_DEPTH
    rng.fill_bytes(&mut t.bytes[LABEL_OFF..LABEL_OFF + LABEL_BYTES]);
    t
}

/// Pack the frozen 152-byte receipt wire format.
///
///   [  0..  7]  magic "SPRCPT01"
///   [  8.. 71]  kste_sig (64 bytes)
///   [ 72..103]  seq_hash (32 bytes, from sp_sieve_event_t.seq_hash)
///   [104..135]  pubkey   (32 bytes, ed25519 verifying key)
///   [136..143]  round    (uint64 LE)
///   [144..151]  minted_at_ns (uint64 LE)
fn pack_receipt(
    sig:          &[u8; 64],
    seq_hash:     &[u8; 32],
    pubkey:       &[u8; 32],
    round:        u64,
    minted_at_ns: u64,
) -> [u8; 152] {
    let mut buf = [0u8; 152];
    buf[..8].copy_from_slice(b"SPRCPT01");
    buf[8..72].copy_from_slice(sig);
    buf[72..104].copy_from_slice(seq_hash);
    buf[104..136].copy_from_slice(pubkey);
    buf[136..144].copy_from_slice(&round.to_le_bytes());
    buf[144..152].copy_from_slice(&minted_at_ns.to_le_bytes());
    buf
}

/// Background mining loop.  Never returns; intended for tokio::spawn.
pub async fn run_mining_loop(
    signing_key:     SigningKey,
    inference_active: Arc<AtomicBool>,
    receipt_store:   Arc<Mutex<Vec<ReceiptRecord>>>,
    pouw_minted:     Arc<std::sync::atomic::AtomicU64>,
    events_tx:       tokio::sync::broadcast::Sender<DaemonEvent>,
) {
    let pubkey: [u8; 32] = signing_key.verifying_key().to_bytes();
    let mut rng = StdRng::from_entropy();

    let mut frontier: Vec<SpKsteTree> = Vec::with_capacity(FRONTIER_CAP);
    frontier.resize(FRONTIER_CAP, SpKsteTree { bytes: [0u8; 64] });
    let mut frontier_n: usize = 0;

    let mut global_round: u64 = 0;

    loop {
        // Yield to the tokio runtime between every batch.
        tokio::task::yield_now().await;

        // Back off while inference is active so /v1/chat isn't starved.
        if inference_active.load(Ordering::Relaxed) {
            tokio::time::sleep(Duration::from_millis(50)).await;
            continue;
        }

        // Generate a batch of synthetic KSTE candidates.
        let mut batch: Vec<SpKsteTree> = (0..BATCH_SIZE)
            .map(|_| synthetic_kste(&mut rng))
            .collect();

        let mut events: Vec<SpSieveEvent> = vec![
            SpSieveEvent {
                sig:      SpKsteTree { bytes: [0u8; 64] },
                seq_hash: [0u8; 32],
                round:    0,
            };
            BATCH_SIZE
        ];
        let mut n_events: usize = 0;

        // SAFETY: all pointers are valid for the duration of the call;
        // frontier and batch buffers are correctly sized.
        let rc = unsafe {
            sp_sieve_evaluate(
                batch.as_mut_ptr(),
                BATCH_SIZE,
                frontier.as_mut_ptr(),
                &mut frontier_n,
                FRONTIER_CAP,
                events.as_mut_ptr(),
                &mut n_events,
            )
        };

        // SP_ESIEVE_FULL (-30): frontier saturated — reset and keep going.
        if rc == -30 {
            frontier_n = 0;
        }

        for i in 0..n_events {
            let ev = &events[i];
            let minted_at_ns = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64;

            let receipt = pack_receipt(
                &ev.sig.bytes,
                &ev.seq_hash,
                &pubkey,
                global_round,
                minted_at_ns,
            );

            let signature = signing_key.sign(&receipt);
            let sig_bytes: [u8; 64] = signature.to_bytes();

            let record = ReceiptRecord {
                payload_hex: hex_encode(&receipt),
                sig_hex:     hex_encode(&sig_bytes),
                round:       global_round,
            };

            debug!("sieve fold: round={} seq_hash={}", global_round,
                   &record.payload_hex[144..208]);

            {
                // ── THE STORE IS A RING NOW (2026-07-31) ──────────────────────────
                // It was an unbounded Vec that nothing ever trimmed. The only readers
                // are two `.len()` calls (telemetry) and a tail — so every record past
                // the most recent few thousand was pure retention: ~470 B each at
                // ~1/second is ~40 MB/day, forever, on a box whose spare RAM is
                // deliberately reserved for WDDM.
                //
                // The frontier count that telemetry reports is a CUMULATIVE total, so
                // trimming the Vec must not silently reset it — `minted_total` is kept
                // separately and is what the operator's "pouw receipts" figure means.
                let mut store = receipt_store.lock().unwrap();
                store.push(record.clone());
                if store.len() > RECEIPT_RING_CAP {
                    let drop_n = store.len() - RECEIPT_RING_CAP;
                    store.drain(0..drop_n);
                }
            }
            {
                // the cumulative total telemetry reports — NOT the ring's length
                pouw_minted.fetch_add(1, Ordering::Relaxed);
            }

            let _ = events_tx.send(DaemonEvent::Mint {
                receipt_hex: record.payload_hex.clone(),
                sig_hex:     record.sig_hex.clone(),
            });

            global_round += 1;
        }
    }
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().fold(String::with_capacity(bytes.len() * 2), |mut s, b| {
        use std::fmt::Write;
        let _ = write!(s, "{b:02x}");
        s
    })
}
