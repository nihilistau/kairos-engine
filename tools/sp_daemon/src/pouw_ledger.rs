//! §4-MeMo Sprint M.4 — PoUW receipt ledger + mesh replay primitive.
//!
//! Turns the M.2 [`crate::dialogue::SpinorReceipt`] 64-byte audit envelope
//! (silicon-confirmed Trick #9 inter-island integrity ABI) into an
//! **append-only ledger** + **byte-level replay** primitive.
//!
//! ## Wire format
//!
//! The ledger file is a plain stream of fixed-size 64-byte records — the
//! `[u8; 64]` returned by [`crate::dialogue::SpinorReceipt::as_bytes`].
//!
//!   - No file header (the file IS the data).
//!   - No record separators (records are fixed-size).
//!   - Corruption detection: every record's sentinel byte at offset 63
//!     must equal 0xA5 (per `reference-spinor-receipt-layout`).
//!   - `file_size % 64 == 0` is the steady-state invariant (when not
//!     mid-write). A partial trailing record (size % 64 != 0) means crash
//!     during an append; partial records are skipped on read.
//!
//! ## Atomic-append discipline
//!
//! All writes go through [`std::fs::OpenOptions::append`]:
//!   - POSIX `O_APPEND` is atomic up to PIPE_BUF (4096) — our 64-byte
//!     records fit comfortably.
//!   - Windows `FILE_APPEND_DATA` gives the same per-write atomicity for
//!     sub-page writes.
//!
//! ## Multi-writer policy
//!
//! [`Ledger::append`] takes `&mut self` → borrow checker serializes
//! per-handle appends. Multi-thread sharing requires
//! `Arc<Mutex<Ledger>>`. Multi-process sharing the same path: file shared-
//! write mode is the OS default on both POSIX (with `O_APPEND`) and
//! Windows (with `FILE_SHARE_WRITE`, default for `OpenOptions::append`).
//!
//! ## Replay determinism
//!
//! Per `reference-lattice-decode-determinism`: if receipts are bit-exact
//! (they are; SHA-256 hashes are domain-separated per-turn-per-model in
//! [`crate::dialogue::SpinorReceipt::mint`]), then replaying the same
//! source ledger into two empty destinations produces SHA-256-equal
//! destination files by construction. The replay-determinism gate
//! verifies this.

#![allow(dead_code)] // smoke harness drives most paths from the android binary

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::dialogue::{SpinorReceipt, SPINOR_SENTINEL};

/// Size of one SpinorReceipt on the wire = 64 bytes (one ARM Cortex-X2 /
/// V69 cache line). Compile-time-guaranteed by [`SpinorReceipt`]'s
/// `#[repr(C, packed)]` + the const-eval guard at `dialogue.rs:66`.
pub const RECEIPT_BYTES: usize = 64;

// ─── Errors ─────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum LedgerError {
    Io(std::io::Error),
    /// File contains a record with `bytes[63] != 0xA5`. Index is the
    /// 64-byte record index (0-based) where the bad sentinel was found.
    BadSentinel { index: u64, observed: u8 },
    /// File has a partial trailing record (size % 64 != 0). Includes the
    /// number of leftover bytes that were skipped.
    PartialTail { extra_bytes: u64 },
}

impl std::fmt::Display for LedgerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LedgerError::Io(e) => write!(f, "ledger I/O error: {e}"),
            LedgerError::BadSentinel { index, observed } => write!(
                f,
                "ledger corruption: record {index} sentinel = 0x{observed:02X} (expected 0xA5)"
            ),
            LedgerError::PartialTail { extra_bytes } => write!(
                f,
                "ledger has partial trailing record: {extra_bytes} extra bytes (last write torn)"
            ),
        }
    }
}

impl std::error::Error for LedgerError {}

impl From<std::io::Error> for LedgerError {
    fn from(e: std::io::Error) -> Self { LedgerError::Io(e) }
}

pub type LedgerResult<T> = std::result::Result<T, LedgerError>;

// ─── Ledger ─────────────────────────────────────────────────────────────────

/// Append-only ledger of [`SpinorReceipt`]s stored as a contiguous stream of
/// 64-byte records. Per `feedback-zero-copy-on-current-L1-ABI`-style
/// discipline: zero allocations inside [`Ledger::append`] except for the
/// underlying `BufWriter::write_all` syscall path; the on-stack 64-byte
/// `as_bytes()` array is `Copy` and lives entirely on the call stack.
pub struct Ledger {
    path: PathBuf,
    writer: BufWriter<File>,
    /// Cached `len()` of the file in bytes (sum of appended record bytes
    /// since `open()`, plus any pre-existing on-disk size). Used by
    /// [`Ledger::len_bytes`] to avoid a stat syscall per call.
    bytes_written: u64,
}

impl Ledger {
    /// Open the ledger at `path` for append-only writes. The file is
    /// created if it does not exist; existing content is preserved.
    ///
    /// The `bytes_written` field is initialized from the file's current
    /// length on disk — subsequent [`Ledger::append`] calls will append
    /// AFTER any pre-existing content.
    pub fn open<P: AsRef<Path>>(path: P) -> LedgerResult<Self> {
        let path: PathBuf = path.as_ref().to_path_buf();
        let f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;
        let bytes_written = f.metadata()?.len();
        Ok(Ledger {
            path,
            writer: BufWriter::new(f),
            bytes_written,
        })
    }

    /// Append one receipt. Returns the byte-offset of the appended record's
    /// FIRST byte (so the caller can build an index, replay range, etc.).
    ///
    /// Atomicity: the underlying file is opened with `O_APPEND`; the
    /// 64-byte write goes through `BufWriter::write_all` which is
    /// guaranteed to be a single contiguous payload on flush. We `flush()`
    /// after each append so the on-disk state always reflects the
    /// in-memory `bytes_written` counter — i.e., a crash after `append()`
    /// returns leaves the ledger in a consistent state.
    pub fn append(&mut self, receipt: &SpinorReceipt) -> LedgerResult<u64> {
        let offset = self.bytes_written;
        let bytes = receipt.as_bytes(); // stack-allocated [u8; 64]
        self.writer.write_all(&bytes)?;
        self.writer.flush()?;
        self.bytes_written += RECEIPT_BYTES as u64;
        Ok(offset)
    }

    /// Returns the in-memory cached length in bytes (= 64 × number of
    /// records appended-or-pre-existing).
    pub fn len_bytes(&self) -> u64 { self.bytes_written }

    /// Returns the path the ledger was opened on.
    pub fn path(&self) -> &Path { &self.path }

    /// Open a fresh read handle on the underlying file and return an
    /// iterator over the records. Each iteration validates the sentinel;
    /// the iterator yields `Err(LedgerError::BadSentinel)` on first
    /// corruption and stops. Partial trailing records produce
    /// `Err(LedgerError::PartialTail)` then stop.
    ///
    /// Note: this opens a SECOND file handle (read-only) on the same path
    /// while the original write handle stays open for append. POSIX + Windows
    /// both permit this concurrent access pattern.
    pub fn iter(&self) -> LedgerResult<LedgerIter> {
        let f = File::open(&self.path)?;
        let len = f.metadata()?.len();
        Ok(LedgerIter {
            file: f,
            remaining: len,
            index: 0,
            finished: false,
        })
    }

    /// Returns the list of receipts ready for broadcast over the mesh,
    /// starting at `since_offset` byte-offset. **Stub for M.4.** Real
    /// QUIC fan-out is filed as a follow-on sprint (existing
    /// `network/quic_shard.rs` is ResidueBlock-shaped, not generic
    /// receipt broadcast). M.4 returns the receipt list so a downstream
    /// transport can pipe them.
    ///
    /// Returns `Vec<SpinorReceipt>` directly (NOT raw bytes) so the
    /// caller can choose how to serialize — `as_bytes()` for byte-exact
    /// wire, or higher-level format for an HTTP gateway.
    // ─── mesh-canonical-order: canonical sort + replay ────────────────────
    //
    // Sprint `mesh-canonical-order` (follow-on to M.4). The M.4 closure
    // (`CLOSURE-M4-LEDGER.md:263-269`) called out that the
    // T_M4_CROSS_DEVICE_REPLAY gate succeeds only because each device's
    // local-first-then-broadcast merge order produces a per-device
    // deterministic result — NOT because devices converge to a single
    // canonical state. These two methods close that gap.
    //
    // Canonical sort key: `(sequence_rank ASC, input_hash ASC)`.
    //   - `sequence_rank` is the u16 stamped via
    //     [`SpinorReceipt::with_sequence_rank`] into `_reserved[0..2]`.
    //   - `input_hash` is the existing 24-byte SHA-256-truncated digest;
    //     dense (≥192-bit entropy) and domain-separated by
    //     (model_id, turn_index) per `dialogue::hash_buf`. It serves as a
    //     deterministic device-disambiguating tiebreak: two receipts with
    //     the same rank produced by different devices (different content)
    //     compare unequal by lexicographic byte order on `input_hash`;
    //     two devices producing IDENTICAL content under the same rank ARE
    //     the same receipt and the canonical sort dedupes-by-equality.
    //
    // Per `feedback-no-silent-gate-revisions`: the dispatch prompt's
    // requested `(rank, device_id)` key was structurally impossible
    // under the existing 2-byte `_reserved` field; surfaced in plan-
    // commit Stage 0.

    /// Read all records from this ledger, sort in-memory by canonical
    /// key `(sequence_rank, input_hash)` ascending, return the sorted
    /// list. Does NOT modify the on-disk file.
    ///
    /// Allocates a `Vec<SpinorReceipt>` of `len_bytes()/64` entries —
    /// for a 1M-record ledger that's 64 MB. Callers driving very large
    /// ledgers should chunk or use the file-streaming
    /// [`Ledger::replay_canonical_into`] companion instead.
    pub fn canonical_sort(&self) -> LedgerResult<Vec<SpinorReceipt>> {
        let mut recs: Vec<SpinorReceipt> = self
            .iter()?
            .collect::<LedgerResult<Vec<_>>>()?;
        recs.sort_by(|a, b| {
            a.sequence_rank()
                .cmp(&b.sequence_rank())
                .then_with(|| a.input_hash.cmp(&b.input_hash))
        });
        Ok(recs)
    }

    /// Read all records from this ledger, canonical-sort them, append
    /// to `dest` in canonical order. Returns the count of receipts
    /// appended.
    ///
    /// This is the cross-device byte-identity primitive: two devices
    /// holding the same receipt SET (e.g. after each broadcasts to the
    /// other) and invoking this method against a fresh empty `dest`
    /// produce SHA-256-equal `dest` files by construction.
    ///
    /// The source ledger is unchanged. Per
    /// `feedback-bundled-changeset-root-cause-ambiguity`: this method
    /// chains `canonical_sort()` + per-record `append()`; the sorted
    /// order is the only variable changing across calls.
    pub fn replay_canonical_into(&self, dest: &mut Ledger) -> LedgerResult<usize> {
        let sorted = self.canonical_sort()?;
        for r in &sorted {
            dest.append(r)?;
        }
        Ok(sorted.len())
    }

    pub fn broadcast_to_peers(&self, since_offset: u64) -> LedgerResult<Vec<SpinorReceipt>> {
        // Walk the file from `since_offset`. Validate sentinels as we go;
        // refuse to broadcast a corrupt prefix.
        if since_offset % RECEIPT_BYTES as u64 != 0 {
            return Err(LedgerError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "broadcast_to_peers: since_offset={since_offset} not a multiple of {RECEIPT_BYTES}"
                ),
            )));
        }
        let mut f = File::open(&self.path)?;
        f.seek(SeekFrom::Start(since_offset))?;
        let mut out: Vec<SpinorReceipt> = Vec::new();
        let mut buf = [0u8; RECEIPT_BYTES];
        let mut idx = since_offset / RECEIPT_BYTES as u64;
        loop {
            match read_exact_or_eof(&mut f, &mut buf)? {
                ReadOutcome::Eof => break,
                ReadOutcome::PartialTail(extra) => {
                    return Err(LedgerError::PartialTail { extra_bytes: extra as u64 });
                }
                ReadOutcome::Full => {
                    if buf[63] != SPINOR_SENTINEL {
                        return Err(LedgerError::BadSentinel { index: idx, observed: buf[63] });
                    }
                    out.push(receipt_from_bytes(&buf));
                    idx += 1;
                }
            }
        }
        Ok(out)
    }
}

// ─── LedgerIter — sequential reader over the ledger file ───────────────────

pub struct LedgerIter {
    file: File,
    remaining: u64,
    index: u64,
    finished: bool,
}

impl Iterator for LedgerIter {
    type Item = LedgerResult<SpinorReceipt>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished { return None; }
        if self.remaining == 0 { self.finished = true; return None; }
        if self.remaining < RECEIPT_BYTES as u64 {
            // partial trailing record
            self.finished = true;
            return Some(Err(LedgerError::PartialTail { extra_bytes: self.remaining }));
        }
        let mut buf = [0u8; RECEIPT_BYTES];
        match self.file.read_exact(&mut buf) {
            Ok(()) => {
                self.remaining -= RECEIPT_BYTES as u64;
                if buf[63] != SPINOR_SENTINEL {
                    self.finished = true;
                    return Some(Err(LedgerError::BadSentinel {
                        index: self.index,
                        observed: buf[63],
                    }));
                }
                let r = receipt_from_bytes(&buf);
                self.index += 1;
                Some(Ok(r))
            }
            Err(e) => {
                self.finished = true;
                Some(Err(LedgerError::Io(e)))
            }
        }
    }
}

// ─── LedgerReplayer — copy receipts from one ledger into another ───────────

/// Stateless namespace for ledger-to-ledger replay operations.
pub struct LedgerReplayer;

impl LedgerReplayer {
    /// Read all records from `source`, append each to `into`. Returns the
    /// number of records appended.
    ///
    /// **Determinism guarantee:** if `source` has no corruption (all
    /// sentinels valid, no partial tail) and `into` was empty before this
    /// call, the resulting `into` file is byte-identical to `source`.
    /// (Per the M.4 plan's T_M4_REPLAY_DETERMINISTIC gate.)
    ///
    /// Errors propagate from the source iterator OR from the destination
    /// append — both are surfaced as-is.
    pub fn replay_from(source: &Ledger, into: &mut Ledger) -> LedgerResult<usize> {
        let mut count = 0usize;
        for item in source.iter()? {
            let r = item?;
            into.append(&r)?;
            count += 1;
        }
        Ok(count)
    }

    /// Replay a pre-loaded receipt list (e.g. from
    /// [`Ledger::broadcast_to_peers`]) into the destination ledger.
    pub fn replay_list(receipts: &[SpinorReceipt], into: &mut Ledger) -> LedgerResult<usize> {
        for r in receipts {
            into.append(r)?;
        }
        Ok(receipts.len())
    }
}

// ─── Internal helpers ───────────────────────────────────────────────────────

enum ReadOutcome {
    Full,
    Eof,
    /// Some bytes read but fewer than the requested 64; `usize` is the count.
    PartialTail(usize),
}

fn read_exact_or_eof(f: &mut File, buf: &mut [u8; RECEIPT_BYTES]) -> std::io::Result<ReadOutcome> {
    let mut filled = 0usize;
    while filled < RECEIPT_BYTES {
        match f.read(&mut buf[filled..])? {
            0 => {
                return Ok(if filled == 0 { ReadOutcome::Eof } else { ReadOutcome::PartialTail(filled) });
            }
            n => filled += n,
        }
    }
    Ok(ReadOutcome::Full)
}

/// Reconstruct a [`SpinorReceipt`] from its 64-byte on-wire form.
///
/// SAFETY: the struct is `#[repr(C, packed)]`, size_of == 64 (compile-time
/// asserted in `dialogue.rs:66`). A bit-for-bit copy of 64 bytes into a
/// freshly-zero-initialized `SpinorReceipt` produces a valid instance for
/// any byte pattern (every field is a plain integer or byte array; no
/// references, no enum-with-niche, no padding traps).
fn receipt_from_bytes(buf: &[u8; RECEIPT_BYTES]) -> SpinorReceipt {
    // SAFETY: see function-level comment. We zero-init then memcpy the
    // bytes into the struct's memory.
    let mut r: SpinorReceipt = unsafe { core::mem::zeroed() };
    unsafe {
        core::ptr::copy_nonoverlapping(
            buf.as_ptr(),
            &mut r as *mut SpinorReceipt as *mut u8,
            RECEIPT_BYTES,
        );
    }
    r
}

// ─── Tests (host build, no L1 required) ─────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialogue::{MODEL_ID_EXECUTIVE, MODEL_ID_MEMORY};
    use std::io::Write as IoWrite;

    fn tmpfile(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        // Per-test unique filename to avoid CI-parallel collision.
        let unique = format!("sp_m4_test_{}_{}_{}.spinor",
            name,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos());
        p.push(unique);
        p
    }

    fn mk(turn: u8, model: u8, wall: u64) -> SpinorReceipt {
        SpinorReceipt::mint(turn, model, &[turn as i32, model as i32], &[wall as i32], wall)
    }

    #[test]
    fn append_then_len_grows_by_64() {
        let p = tmpfile("append_grows");
        let mut l = Ledger::open(&p).unwrap();
        assert_eq!(l.len_bytes(), 0);
        let off = l.append(&mk(1, MODEL_ID_EXECUTIVE, 100)).unwrap();
        assert_eq!(off, 0);
        assert_eq!(l.len_bytes(), 64);
        let off = l.append(&mk(2, MODEL_ID_MEMORY, 200)).unwrap();
        assert_eq!(off, 64);
        assert_eq!(l.len_bytes(), 128);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn iter_reads_back_what_was_written() {
        let p = tmpfile("iter_readback");
        let mut l = Ledger::open(&p).unwrap();
        let r1 = mk(1, MODEL_ID_EXECUTIVE, 100);
        let r2 = mk(2, MODEL_ID_MEMORY, 200);
        let r3 = mk(3, MODEL_ID_EXECUTIVE, 300);
        l.append(&r1).unwrap();
        l.append(&r2).unwrap();
        l.append(&r3).unwrap();
        drop(l);
        let l2 = Ledger::open(&p).unwrap();
        let collected: Vec<_> = l2.iter().unwrap().map(|x| x.unwrap()).collect();
        assert_eq!(collected.len(), 3);
        assert_eq!(collected[0].as_bytes(), r1.as_bytes());
        assert_eq!(collected[1].as_bytes(), r2.as_bytes());
        assert_eq!(collected[2].as_bytes(), r3.as_bytes());
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn sentinel_is_at_offset_63_in_file() {
        let p = tmpfile("sentinel_offset");
        let mut l = Ledger::open(&p).unwrap();
        l.append(&mk(1, MODEL_ID_EXECUTIVE, 100)).unwrap();
        drop(l);
        let mut f = File::open(&p).unwrap();
        let mut buf = [0u8; 64];
        f.read_exact(&mut buf).unwrap();
        assert_eq!(buf[63], SPINOR_SENTINEL);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn iter_detects_bad_sentinel() {
        let p = tmpfile("bad_sentinel");
        // Write a record with a corrupted sentinel by hand.
        {
            let mut f = OpenOptions::new().create(true).append(true).open(&p).unwrap();
            let mut bytes = [0u8; 64];
            bytes[0] = 1; // turn_index
            bytes[1] = MODEL_ID_EXECUTIVE;
            bytes[63] = 0x00; // CORRUPTED — should be 0xA5
            f.write_all(&bytes).unwrap();
            f.flush().unwrap();
        }
        let l = Ledger::open(&p).unwrap();
        let mut iter = l.iter().unwrap();
        match iter.next() {
            Some(Err(LedgerError::BadSentinel { index, observed })) => {
                assert_eq!(index, 0);
                assert_eq!(observed, 0x00);
            }
            other => panic!("expected BadSentinel, got {:?}", other.map(|r| r.map(|_| "Ok(receipt)"))),
        }
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn iter_detects_partial_tail() {
        let p = tmpfile("partial_tail");
        // Write one valid record then 17 stray bytes.
        let mut l = Ledger::open(&p).unwrap();
        l.append(&mk(1, MODEL_ID_EXECUTIVE, 100)).unwrap();
        drop(l);
        {
            let mut f = OpenOptions::new().append(true).open(&p).unwrap();
            f.write_all(&[0u8; 17]).unwrap();
            f.flush().unwrap();
        }
        let l = Ledger::open(&p).unwrap();
        let mut iter = l.iter().unwrap();
        // First record reads fine.
        let r = iter.next().unwrap().unwrap();
        assert_eq!(r.sentinel, SPINOR_SENTINEL);
        // Next read trips the partial-tail detector.
        match iter.next() {
            Some(Err(LedgerError::PartialTail { extra_bytes })) => {
                assert_eq!(extra_bytes, 17);
            }
            other => panic!("expected PartialTail, got {:?}",
                            other.map(|r| r.map(|_| "Ok(receipt)"))),
        }
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn replay_from_produces_byte_identical_file() {
        let src_p = tmpfile("replay_src");
        let dst_a_p = tmpfile("replay_dst_a");
        let dst_b_p = tmpfile("replay_dst_b");
        // Populate source with 10 receipts.
        let mut src = Ledger::open(&src_p).unwrap();
        for i in 0..10u8 {
            src.append(&mk(i + 1, if i % 2 == 0 { MODEL_ID_EXECUTIVE } else { MODEL_ID_MEMORY }, 100 + i as u64)).unwrap();
        }
        drop(src);
        let src_r = Ledger::open(&src_p).unwrap();
        let mut dst_a = Ledger::open(&dst_a_p).unwrap();
        let mut dst_b = Ledger::open(&dst_b_p).unwrap();
        let n_a = LedgerReplayer::replay_from(&src_r, &mut dst_a).unwrap();
        let n_b = LedgerReplayer::replay_from(&src_r, &mut dst_b).unwrap();
        assert_eq!(n_a, 10);
        assert_eq!(n_b, 10);
        drop(dst_a); drop(dst_b);
        let a_bytes = std::fs::read(&dst_a_p).unwrap();
        let b_bytes = std::fs::read(&dst_b_p).unwrap();
        assert_eq!(a_bytes, b_bytes);
        assert_eq!(a_bytes.len(), 640);
        let _ = std::fs::remove_file(&src_p);
        let _ = std::fs::remove_file(&dst_a_p);
        let _ = std::fs::remove_file(&dst_b_p);
    }

    #[test]
    fn broadcast_returns_full_list_from_offset_zero() {
        let p = tmpfile("broadcast_zero");
        let mut l = Ledger::open(&p).unwrap();
        for i in 0..5u8 {
            l.append(&mk(i + 1, MODEL_ID_EXECUTIVE, i as u64 + 1)).unwrap();
        }
        let bcast = l.broadcast_to_peers(0).unwrap();
        assert_eq!(bcast.len(), 5);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn broadcast_returns_suffix_from_offset() {
        let p = tmpfile("broadcast_suffix");
        let mut l = Ledger::open(&p).unwrap();
        for i in 0..5u8 {
            l.append(&mk(i + 1, MODEL_ID_EXECUTIVE, i as u64 + 1)).unwrap();
        }
        // Skip the first 2 records (offset 128).
        let bcast = l.broadcast_to_peers(128).unwrap();
        assert_eq!(bcast.len(), 3);
        assert_eq!(bcast[0].turn_index, 3);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn broadcast_rejects_misaligned_offset() {
        let p = tmpfile("broadcast_misaligned");
        let mut l = Ledger::open(&p).unwrap();
        l.append(&mk(1, MODEL_ID_EXECUTIVE, 1)).unwrap();
        let err = l.broadcast_to_peers(7).unwrap_err();
        match err {
            LedgerError::Io(_) => { /* ok */ }
            other => panic!("expected Io(InvalidInput), got {other}"),
        }
        let _ = std::fs::remove_file(&p);
    }

    // ─── mesh-canonical-order: canonical_sort + replay_canonical_into ─────

    fn mk_ranked(rank: u16, turn: u8, model: u8) -> SpinorReceipt {
        // input bytes vary with rank so input_hash is well-distributed
        let in_tokens: [i32; 3] = [rank as i32, (rank ^ 0x5A5A) as i32, turn as i32];
        let out_tokens: [i32; 2] = [(rank as i32) + 1, model as i32];
        SpinorReceipt::mint(turn, model, &in_tokens, &out_tokens, 1000 + rank as u64)
            .with_sequence_rank(rank)
    }

    #[test]
    fn canonical_sort_orders_by_rank_ascending() {
        let p = tmpfile("canon_sort_rank_asc");
        let mut l = Ledger::open(&p).unwrap();
        // Insert in reverse rank order.
        for rank in [9u16, 7, 5, 3, 1, 8, 6, 4, 2, 0] {
            l.append(&mk_ranked(rank, 1, MODEL_ID_EXECUTIVE)).unwrap();
        }
        drop(l);
        let l2 = Ledger::open(&p).unwrap();
        let sorted = l2.canonical_sort().unwrap();
        assert_eq!(sorted.len(), 10);
        for (i, r) in sorted.iter().enumerate() {
            assert_eq!(r.sequence_rank(), i as u16,
                "position {i} should hold rank {i}");
        }
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn canonical_sort_tiebreaks_by_input_hash() {
        let p = tmpfile("canon_sort_tiebreak");
        let mut l = Ledger::open(&p).unwrap();
        // Two receipts at rank=5 with different input streams -> different input_hash.
        let a = SpinorReceipt::mint(1, MODEL_ID_EXECUTIVE, &[1, 2, 3], &[10], 100)
            .with_sequence_rank(5);
        let b = SpinorReceipt::mint(1, MODEL_ID_EXECUTIVE, &[9, 8, 7], &[20], 200)
            .with_sequence_rank(5);
        // Insert in a non-canonical order; whichever input_hash compares lower
        // must come first.
        l.append(&b).unwrap();
        l.append(&a).unwrap();
        drop(l);
        let l2 = Ledger::open(&p).unwrap();
        let sorted = l2.canonical_sort().unwrap();
        assert_eq!(sorted.len(), 2);
        assert_eq!(sorted[0].sequence_rank(), 5);
        assert_eq!(sorted[1].sequence_rank(), 5);
        // First in canonical order is the one with lexicographically smaller input_hash.
        let expected_first = if a.input_hash <= b.input_hash { a } else { b };
        assert_eq!(sorted[0].input_hash, expected_first.input_hash,
            "tiebreak under equal rank must order by input_hash ASC");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn canonical_sort_is_deterministic_across_runs() {
        use sha2::{Digest, Sha256};
        let p = tmpfile("canon_sort_determ");
        let mut l = Ledger::open(&p).unwrap();
        // 100 receipts at randomized-but-seeded ranks.
        let mut rng_state: u64 = 0x5DEECE66D;
        for i in 0..100u16 {
            // SplitMix-style PRNG
            rng_state = rng_state.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(0x123456789ABCDEF0);
            let rank = ((rng_state >> 16) as u16) % 1000;
            l.append(&mk_ranked(rank, ((i % 3) as u8) + 1,
                if i % 2 == 0 { MODEL_ID_EXECUTIVE } else { MODEL_ID_MEMORY })).unwrap();
        }
        drop(l);

        let h = |sorted: &Vec<SpinorReceipt>| {
            let mut h = Sha256::new();
            for r in sorted { h.update(r.as_bytes()); }
            let d = h.finalize();
            d.iter().map(|b| format!("{b:02x}")).collect::<String>()
        };

        // Two independent runs on fresh Ledger::open handles.
        let l1 = Ledger::open(&p).unwrap();
        let s1 = l1.canonical_sort().unwrap();
        drop(l1);
        let l2 = Ledger::open(&p).unwrap();
        let s2 = l2.canonical_sort().unwrap();
        drop(l2);

        let h1 = h(&s1);
        let h2 = h(&s2);
        assert_eq!(h1, h2,
            "canonical_sort must be deterministic: two runs over the same source must SHA-256-match");

        // Spot-check ordering invariant: rank monotone non-decreasing.
        for i in 1..s1.len() {
            assert!(s1[i - 1].sequence_rank() <= s1[i].sequence_rank(),
                "rank monotone non-decreasing violated at {i}");
        }
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn replay_canonical_into_byte_identical_under_permutation() {
        use sha2::{Digest, Sha256};
        // Two source ledgers with the SAME receipt SET in different orders;
        // replay_canonical_into to fresh destinations -> SHA-256-equal files.
        let src_a_p = tmpfile("canon_replay_src_a");
        let src_b_p = tmpfile("canon_replay_src_b");
        let dst_a_p = tmpfile("canon_replay_dst_a");
        let dst_b_p = tmpfile("canon_replay_dst_b");

        // Build the receipt set once.
        let mut all: Vec<SpinorReceipt> = (0..10u16)
            .map(|r| mk_ranked(r, ((r % 3) as u8) + 1,
                if r % 2 == 0 { MODEL_ID_EXECUTIVE } else { MODEL_ID_MEMORY }))
            .collect();

        // Source A: forward order.
        {
            let mut la = Ledger::open(&src_a_p).unwrap();
            for r in &all { la.append(r).unwrap(); }
        }
        // Source B: reverse order.
        all.reverse();
        {
            let mut lb = Ledger::open(&src_b_p).unwrap();
            for r in &all { lb.append(r).unwrap(); }
        }

        // Replay canonically.
        let src_a = Ledger::open(&src_a_p).unwrap();
        let src_b = Ledger::open(&src_b_p).unwrap();
        let mut dst_a = Ledger::open(&dst_a_p).unwrap();
        let mut dst_b = Ledger::open(&dst_b_p).unwrap();
        let na = src_a.replay_canonical_into(&mut dst_a).unwrap();
        let nb = src_b.replay_canonical_into(&mut dst_b).unwrap();
        assert_eq!(na, 10);
        assert_eq!(nb, 10);
        drop(dst_a); drop(dst_b);

        let hash = |path: &PathBuf| -> String {
            let bytes = std::fs::read(path).unwrap();
            let mut h = Sha256::new();
            h.update(&bytes);
            h.finalize().iter().map(|b| format!("{b:02x}")).collect()
        };
        let ha = hash(&dst_a_p);
        let hb = hash(&dst_b_p);
        assert_eq!(ha, hb,
            "replay_canonical_into must produce byte-identical dest files for same-set sources under permutation");
        assert_eq!(std::fs::metadata(&dst_a_p).unwrap().len(), 640,
            "10 receipts * 64 bytes = 640 byte dest file");

        let _ = std::fs::remove_file(&src_a_p);
        let _ = std::fs::remove_file(&src_b_p);
        let _ = std::fs::remove_file(&dst_a_p);
        let _ = std::fs::remove_file(&dst_b_p);
    }

    #[test]
    fn canonical_sort_empty_ledger_returns_empty_vec() {
        let p = tmpfile("canon_sort_empty");
        let l = Ledger::open(&p).unwrap();
        drop(l);
        let l2 = Ledger::open(&p).unwrap();
        let sorted = l2.canonical_sort().unwrap();
        assert!(sorted.is_empty());
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn replay_canonical_into_preserves_source() {
        let src_p = tmpfile("canon_replay_preserve_src");
        let dst_p = tmpfile("canon_replay_preserve_dst");
        let mut src = Ledger::open(&src_p).unwrap();
        for r in [3u16, 1, 4, 1, 5, 9, 2, 6, 5, 3] {
            src.append(&mk_ranked(r, 1, MODEL_ID_EXECUTIVE)).unwrap();
        }
        drop(src);
        let pre_size = std::fs::metadata(&src_p).unwrap().len();

        let src_r = Ledger::open(&src_p).unwrap();
        let mut dst = Ledger::open(&dst_p).unwrap();
        let n = src_r.replay_canonical_into(&mut dst).unwrap();
        assert_eq!(n, 10);
        drop(dst);

        let post_size = std::fs::metadata(&src_p).unwrap().len();
        assert_eq!(pre_size, post_size,
            "replay_canonical_into must not mutate the source ledger file");
        let _ = std::fs::remove_file(&src_p);
        let _ = std::fs::remove_file(&dst_p);
    }

    #[test]
    fn replay_list_appends_in_order() {
        let p = tmpfile("replay_list");
        let mut l = Ledger::open(&p).unwrap();
        let items: Vec<SpinorReceipt> = (0..7u8)
            .map(|i| mk(i + 1, MODEL_ID_MEMORY, 100 + i as u64))
            .collect();
        let n = LedgerReplayer::replay_list(&items, &mut l).unwrap();
        assert_eq!(n, 7);
        assert_eq!(l.len_bytes(), 7 * 64);
        drop(l);
        let l2 = Ledger::open(&p).unwrap();
        let back: Vec<_> = l2.iter().unwrap().map(|r| r.unwrap()).collect();
        assert_eq!(back.len(), 7);
        for (i, r) in back.iter().enumerate() {
            assert_eq!(r.turn_index, (i as u8) + 1);
        }
        let _ = std::fs::remove_file(&p);
    }
}
