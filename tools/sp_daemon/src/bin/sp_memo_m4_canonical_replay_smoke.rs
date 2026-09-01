//! Sprint mesh-canonical-order — cross-device canonical-replay smoke harness.
//!
//! Drives the three substantive gates against a host filesystem:
//!
//!   T_MESH_RANK_PROTOCOL                u16 rank lands in _reserved[0..2]
//!                                       at on-wire offsets 61-62; sentinel
//!                                       at offset 63 preserved; all other
//!                                       receipt bytes unchanged.
//!
//!   T_MESH_CANONICAL_SORT_DETERMINISTIC sort N=100 receipts twice;
//!                                       SHA-256 of two sorted runs match.
//!
//!   T_MESH_CROSS_DEVICE_BYTE_IDENTITY   two simulated devices each mint
//!                                       half the receipt set; broadcast
//!                                       (M.4 stub) + replay_canonical_into
//!                                       on each; SHA-256 of both devices'
//!                                       canonical ledgers AND a third
//!                                       reference canonical ledger all
//!                                       three match.
//!
//! Per `feedback-no-silent-gate-revisions`: any FAIL is surfaced as
//! `gate FAILED` in JSON and as a non-zero exit code. No retry / tolerance
//! widening.
//!
//! Per `feedback-bundled-changeset-root-cause-ambiguity`: each gate
//! reports the variables that drove the outcome (sha hashes,
//! intermediate counts).
//!
//! ### CLI
//!
//!   sp_memo_m4_canonical_replay_smoke [--workdir DIR]
//!                                      [--report-json PATH]
//!
//! Host-only sprint: no L1 model, no device, pure data-layer.

use std::path::PathBuf;

use sha2::{Digest, Sha256};
use sp_daemon::dialogue::{SpinorReceipt, MODEL_ID_EXECUTIVE, MODEL_ID_MEMORY, SPINOR_SENTINEL};
use sp_daemon::pouw_ledger::Ledger;

// ─── CLI ─────────────────────────────────────────────────────────────────────

struct Args {
    workdir: PathBuf,
    report_json: Option<PathBuf>,
}

fn parse_args() -> Args {
    let mut workdir = std::env::temp_dir();
    let mut report_json: Option<PathBuf> = None;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--workdir" => {
                workdir = PathBuf::from(it.next().unwrap_or_else(|| {
                    eprintln!("--workdir requires a path argument");
                    std::process::exit(2);
                }));
            }
            "--report-json" => {
                report_json = Some(PathBuf::from(it.next().unwrap_or_else(|| {
                    eprintln!("--report-json requires a path argument");
                    std::process::exit(2);
                })));
            }
            "--help" | "-h" => {
                println!(
                    "usage: sp_memo_m4_canonical_replay_smoke [--workdir DIR] [--report-json PATH]"
                );
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(2);
            }
        }
    }
    let _ = std::fs::create_dir_all(&workdir);
    Args { workdir, report_json }
}

// ─── helpers ────────────────────────────────────────────────────────────────

fn sha256_file(path: &PathBuf) -> String {
    let bytes = std::fs::read(path).expect("read file for sha256");
    let mut h = Sha256::new();
    h.update(&bytes);
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

/// Mint a synthetic ranked receipt. Input tokens vary with (rank, device_tag)
/// so two devices' same-rank receipts diverge in input_hash, exercising the
/// canonical-sort tiebreak path.
fn mint_ranked(rank: u16, device_tag: u8) -> SpinorReceipt {
    let turn = ((rank % 3) as u8) + 1;
    let model = if rank % 2 == 0 { MODEL_ID_EXECUTIVE } else { MODEL_ID_MEMORY };
    let in_tokens: [i32; 4] = [
        rank as i32,
        device_tag as i32,
        (rank ^ 0xCAFE) as i32,
        (device_tag as i32) << 16,
    ];
    let out_tokens: [i32; 2] = [(rank as i32) + 1, device_tag as i32];
    let wall_us = 1_000 + (rank as u64) * 10 + device_tag as u64;
    SpinorReceipt::mint(turn, model, &in_tokens, &out_tokens, wall_us)
        .with_sequence_rank(rank)
}

fn tmp_path(workdir: &PathBuf, name: &str) -> PathBuf {
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let mut p = workdir.clone();
    p.push(format!("mesh_canonical_{name}_{pid}_{nanos}.spinor"));
    p
}

// ─── gates ──────────────────────────────────────────────────────────────────

struct GateResult {
    name: &'static str,
    pass: bool,
    detail: String,
}

fn gate_t_mesh_rank_protocol() -> GateResult {
    let base = SpinorReceipt::mint(1, MODEL_ID_EXECUTIVE, &[1, 2, 3], &[4, 5], 1234);
    let stamped = base.with_sequence_rank(42);

    let base_bytes = base.as_bytes();
    let stamped_bytes = stamped.as_bytes();

    let lo = stamped_bytes[61];
    let hi = stamped_bytes[62];
    let rank_bytes_match = lo == 0x2A && hi == 0x00; // 42 LE
    let set_get_match = stamped.sequence_rank() == 42;
    let sentinel_preserved = stamped_bytes[63] == SPINOR_SENTINEL;
    let other_reserved_bytes_unchanged = base_bytes[0..61] == stamped_bytes[0..61];

    let pass = rank_bytes_match
        && set_get_match
        && sentinel_preserved
        && other_reserved_bytes_unchanged;

    GateResult {
        name: "T_MESH_RANK_PROTOCOL",
        pass,
        detail: format!(
            "{{\"bytes_61_62\": \"{lo:02x} {hi:02x}\", \
              \"rank_bytes_match\": {rank_bytes_match}, \
              \"set_get_match\": {set_get_match}, \
              \"other_reserved_bytes_unchanged\": {other_reserved_bytes_unchanged}, \
              \"sentinel_preserved\": {sentinel_preserved}}}",
        ),
    }
}

fn gate_t_mesh_canonical_sort_deterministic(workdir: &PathBuf) -> GateResult {
    let p = tmp_path(workdir, "sort_determ");
    let mut l = Ledger::open(&p).expect("open ledger");
    // 100 receipts with seeded "random" ranks (SplitMix-style).
    let mut state: u64 = 0xDEAD_BEEF_CAFE_BABE;
    for i in 0..100u32 {
        state = state.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(0x1234_5678_9ABC_DEF0);
        let rank = ((state >> 16) as u16) % 1000;
        let device_tag = (i % 4) as u8;
        l.append(&mint_ranked(rank, device_tag)).expect("append");
    }
    drop(l);

    // Two independent sort runs on fresh handles.
    let l1 = Ledger::open(&p).expect("reopen 1");
    let s1 = l1.canonical_sort().expect("sort 1");
    let l2 = Ledger::open(&p).expect("reopen 2");
    let s2 = l2.canonical_sort().expect("sort 2");

    let h = |sorted: &Vec<SpinorReceipt>| -> String {
        let mut h = Sha256::new();
        for r in sorted {
            h.update(r.as_bytes());
        }
        h.finalize().iter().map(|b| format!("{b:02x}")).collect()
    };
    let h1 = h(&s1);
    let h2 = h(&s2);
    let sha256_match = h1 == h2;
    let runs = 2usize;

    let _ = std::fs::remove_file(&p);
    GateResult {
        name: "T_MESH_CANONICAL_SORT_DETERMINISTIC",
        pass: sha256_match && s1.len() == 100 && s2.len() == 100,
        detail: format!(
            "{{\"runs\": {runs}, \"n\": 100, \"sha256_run1\": \"{h1}\", \
              \"sha256_run2\": \"{h2}\", \"sha256_match\": {sha256_match}}}"
        ),
    }
}

fn gate_t_mesh_cross_device_byte_identity(workdir: &PathBuf) -> GateResult {
    // Device A: ranks 0, 2, 4, 6, 8 (even); device_tag 0xA.
    // Device B: ranks 1, 3, 5, 7, 9 (odd);  device_tag 0xB.
    // Each device's local ledger holds its OWN 5 receipts.
    // Each device "broadcasts" via M.4 broadcast_to_peers stub and appends
    // the peer's 5 receipts to its raw ledger -> raw ledger has all 10
    // in local-first order (NOT canonical).
    // Each device runs replay_canonical_into into a fresh canonical ledger.
    // Reference: build canonical ledger from minted 0..=9 directly.
    // Pass: all three canonical ledgers SHA-256-equal AND canonical order
    // observed via re-read is 0,1,2,3,4,5,6,7,8,9.

    let pa = tmp_path(workdir, "device_a_raw");
    let pb = tmp_path(workdir, "device_b_raw");
    let ca = tmp_path(workdir, "device_a_canonical");
    let cb = tmp_path(workdir, "device_b_canonical");
    let cref = tmp_path(workdir, "reference_canonical");

    let device_a_tag: u8 = 0xA;
    let device_b_tag: u8 = 0xB;

    // Device A mints ranks 0, 2, 4, 6, 8.
    let a_receipts: Vec<SpinorReceipt> =
        [0u16, 2, 4, 6, 8].iter().map(|&r| mint_ranked(r, device_a_tag)).collect();
    // Device B mints ranks 1, 3, 5, 7, 9.
    let b_receipts: Vec<SpinorReceipt> =
        [1u16, 3, 5, 7, 9].iter().map(|&r| mint_ranked(r, device_b_tag)).collect();

    // Device A local ledger: A first, then B's broadcast appended.
    {
        let mut la = Ledger::open(&pa).expect("open device_a raw");
        for r in &a_receipts { la.append(r).expect("append a-own to a"); }
    }
    // Device B local ledger: B first, then A's broadcast appended.
    {
        let mut lb = Ledger::open(&pb).expect("open device_b raw");
        for r in &b_receipts { lb.append(r).expect("append b-own to b"); }
    }
    // Simulate broadcast via M.4 stub: each device pulls peer's full
    // list and appends to its raw ledger. (Local-first-then-broadcast,
    // exactly the M.4 v1 merge order — which is NOT canonical.)
    {
        let lb_handle = Ledger::open(&pb).expect("reopen device_b for broadcast");
        let b_for_a = lb_handle.broadcast_to_peers(0).expect("b broadcast to a");
        let mut la = Ledger::open(&pa).expect("reopen device_a to receive");
        for r in &b_for_a { la.append(r).expect("append b->a"); }
    }
    {
        let la_handle = Ledger::open(&pa).expect("reopen device_a for broadcast");
        // IMPORTANT: take A's broadcast BEFORE A appended B's set — but for the
        // smoke purpose we want A's broadcast to contain only A's own 5.
        // Since we just appended b_for_a above, A's ledger now has 10 records.
        // For the smoke we want B's raw ledger to receive only A's 5.
        // Use broadcast_to_peers(0) and slice to the first 5 (A's own).
        let a_for_b_all = la_handle.broadcast_to_peers(0).expect("a broadcast to b");
        // A's ledger ordering: own-5 first, then peer-5 — take first 5.
        let a_for_b: Vec<SpinorReceipt> = a_for_b_all.into_iter().take(5).collect();
        let mut lb = Ledger::open(&pb).expect("reopen device_b to receive");
        for r in &a_for_b { lb.append(r).expect("append a->b"); }
    }

    // Both raw ledgers now have all 10 receipts, but in different orders.
    // Sanity: confirm sizes.
    let size_a = std::fs::metadata(&pa).expect("stat a").len();
    let size_b = std::fs::metadata(&pb).expect("stat b").len();
    let raw_sizes_match = size_a == 640 && size_b == 640;
    // Sanity: raw ledgers DIFFER (the canonical-order problem M.4 left open).
    let raw_a_sha = sha256_file(&pa);
    let raw_b_sha = sha256_file(&pb);
    let raw_devices_diverge = raw_a_sha != raw_b_sha;

    // Canonical replay each device into its OWN canonical ledger.
    {
        let src_a = Ledger::open(&pa).expect("reopen a for canonical");
        let mut dst_a = Ledger::open(&ca).expect("open a canonical");
        let n = src_a.replay_canonical_into(&mut dst_a).expect("replay a");
        assert_eq!(n, 10, "device A canonical replay must yield 10 receipts");
    }
    {
        let src_b = Ledger::open(&pb).expect("reopen b for canonical");
        let mut dst_b = Ledger::open(&cb).expect("open b canonical");
        let n = src_b.replay_canonical_into(&mut dst_b).expect("replay b");
        assert_eq!(n, 10, "device B canonical replay must yield 10 receipts");
    }
    // Reference: mint 0..=9 and append in rank order.
    {
        let mut all: Vec<SpinorReceipt> = Vec::with_capacity(10);
        for &r in &[0u16, 2, 4, 6, 8] { all.push(mint_ranked(r, device_a_tag)); }
        for &r in &[1u16, 3, 5, 7, 9] { all.push(mint_ranked(r, device_b_tag)); }
        all.sort_by(|x, y| {
            x.sequence_rank()
                .cmp(&y.sequence_rank())
                .then_with(|| x.input_hash.cmp(&y.input_hash))
        });
        let mut lref = Ledger::open(&cref).expect("open reference canonical");
        for r in &all { lref.append(r).expect("append ref"); }
    }

    let a_sha = sha256_file(&ca);
    let b_sha = sha256_file(&cb);
    let ref_sha = sha256_file(&cref);
    let all_three_match = a_sha == b_sha && b_sha == ref_sha;

    // Read back device A's canonical ledger and confirm rank order = 0..=9.
    let canonical_order_is_interleaved = {
        let l = Ledger::open(&ca).expect("reopen a canonical for verify");
        let recs: Vec<SpinorReceipt> = l.iter().expect("iter canonical")
            .collect::<Result<Vec<_>, _>>()
            .expect("collect canonical");
        recs.len() == 10
            && recs.iter().enumerate().all(|(i, r)| r.sequence_rank() == i as u16)
    };

    let pass = raw_sizes_match
        && raw_devices_diverge
        && all_three_match
        && canonical_order_is_interleaved;

    // Cleanup
    let _ = std::fs::remove_file(&pa);
    let _ = std::fs::remove_file(&pb);
    let _ = std::fs::remove_file(&ca);
    let _ = std::fs::remove_file(&cb);
    let _ = std::fs::remove_file(&cref);

    GateResult {
        name: "T_MESH_CROSS_DEVICE_BYTE_IDENTITY",
        pass,
        detail: format!(
            "{{\"raw_sizes_match\": {raw_sizes_match}, \
              \"raw_a_sha\": \"{raw_a_sha}\", \
              \"raw_b_sha\": \"{raw_b_sha}\", \
              \"raw_devices_diverge\": {raw_devices_diverge}, \
              \"device_a_sha\": \"{a_sha}\", \
              \"device_b_sha\": \"{b_sha}\", \
              \"reference_sha\": \"{ref_sha}\", \
              \"all_three_match\": {all_three_match}, \
              \"canonical_order_is_interleaved\": {canonical_order_is_interleaved}}}"
        ),
    }
}

fn main() {
    let args = parse_args();
    println!("# sp_memo_m4_canonical_replay_smoke — Sprint mesh-canonical-order");
    println!("# workdir = {}", args.workdir.display());

    let gates = vec![
        gate_t_mesh_rank_protocol(),
        gate_t_mesh_canonical_sort_deterministic(&args.workdir),
        gate_t_mesh_cross_device_byte_identity(&args.workdir),
    ];

    println!();
    println!("## Gates");
    let mut any_fail = false;
    for g in &gates {
        let verdict = if g.pass { "PASS" } else { "FAIL" };
        if !g.pass { any_fail = true; }
        println!("[{verdict}] {} :: {}", g.name, g.detail);
    }

    // Aggregate JSON report.
    let mut report = String::new();
    report.push_str("{\n  \"sprint\": \"mesh-canonical-order\",\n  \"gates\": [\n");
    for (i, g) in gates.iter().enumerate() {
        let comma = if i + 1 < gates.len() { "," } else { "" };
        report.push_str(&format!(
            "    {{\"name\": \"{}\", \"pass\": {}, \"detail\": {}}}{}\n",
            g.name, g.pass, g.detail, comma
        ));
    }
    report.push_str("  ],\n");
    report.push_str(&format!(
        "  \"all_pass\": {},\n  \"summary_sha256\": \"{}\"\n}}\n",
        !any_fail,
        sha256_bytes(report.as_bytes())
    ));

    if let Some(path) = args.report_json {
        std::fs::write(&path, report.as_bytes()).expect("write report json");
        println!("\n# JSON report written to {}", path.display());
    }

    println!();
    if any_fail {
        eprintln!("FAIL: one or more gates failed (see above).");
        std::process::exit(1);
    } else {
        println!("OK: all gates pass.");
    }
}
