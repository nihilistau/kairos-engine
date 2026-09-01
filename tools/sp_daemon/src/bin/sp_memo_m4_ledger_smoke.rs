//! §4-MeMo Sprint M.4 — PoUW receipt ledger + mesh replay smoke harness.
//!
//! Drives the four substantive M.4 gates against the host/android
//! filesystem. Pure orchestration-layer — does NOT load any L1 model;
//! synthesizes [`SpinorReceipt`]s via [`SpinorReceipt::mint`] with
//! deterministic synthetic token streams.
//!
//! ### Gates (no silent gate revisions per `feedback-no-silent-gate-revisions`)
//!
//!   T_M4_LEDGER_APPEND            N=1000 SpinorReceipts appended without
//!                                 error; file size = N*64 bytes
//!   T_M4_LEDGER_READ              N=1000 read back; sentinel 0xA5 on every
//!                                 record; byte-identical to source
//!   T_M4_REPLAY_DETERMINISTIC     replay source -> dest_a + source -> dest_b;
//!                                 SHA-256 of dest_a == SHA-256 of dest_b
//!   T_M4_CROSS_DEVICE_REPLAY      two simulated devices each mint half;
//!                                 broadcast (stub) + cross-replay; each
//!                                 device's final ledger matches an expected
//!                                 canonical ordering
//!
//! ### CLI
//!
//!   sp_memo_m4_ledger_smoke [--n 1000] \
//!                           [--workdir /data/local/tmp] \
//!                           [--report-json /data/local/tmp/m4_report.json]
//!
//! All gate output is hex digests + counts; no L1 ABI needed. Same binary
//! runs on host (cargo build --release) and android (build-android.bat
//! --bin sp_memo_m4_ledger_smoke).

use std::path::PathBuf;
use std::time::Instant;

use sha2::{Digest, Sha256};
use sp_daemon::dialogue::{SpinorReceipt, MODEL_ID_EXECUTIVE, MODEL_ID_MEMORY, SPINOR_SENTINEL};
use sp_daemon::pouw_ledger::{Ledger, LedgerReplayer};

// ─── Synthetic receipt minter ──────────────────────────────────────────────

/// Mint a deterministic synthetic receipt for index `i`. Distinguished by
/// `(turn_index, model_id, wall_us, token streams)` so 1000 distinct
/// receipts are produced. Each input/output hash will differ across `i`
/// due to the per-iter token mutation.
fn synth_receipt(i: u32) -> SpinorReceipt {
    let turn = ((i % 3) as u8) + 1;             // 1..3
    let model = if i % 2 == 0 { MODEL_ID_EXECUTIVE } else { MODEL_ID_MEMORY };
    let wall = (i as u64) * 1000 + 1234;
    let in_tokens: [i32; 3] = [i as i32, (i ^ 0xCAFE) as i32, (i.wrapping_mul(2654435761)) as i32];
    let out_tokens: [i32; 2] = [(i as i32) + 1, (i as i32) + 7];
    SpinorReceipt::mint(turn, model, &in_tokens, &out_tokens, wall)
}

// ─── /proc/self/status reader (android) ───────────────────────────────────

#[cfg(target_os = "android")]
fn vmrss_kb() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmRSS:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|n| n.parse::<u64>().ok())
        })
        .unwrap_or(0)
}

#[cfg(not(target_os = "android"))]
fn vmrss_kb() -> u64 { 0 }

// ─── SHA-256 over file contents ────────────────────────────────────────────

fn sha256_of_file(path: &PathBuf) -> Result<String, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let mut h = Sha256::new();
    h.update(&bytes);
    let digest = h.finalize();
    Ok(digest.iter().map(|b| format!("{b:02x}")).collect::<String>())
}

// ─── Tiny JSON emitter (same minimal pattern as M.2 smoke) ─────────────────

struct J(String);
impl J {
    fn new() -> Self { J("{".into()) }
    fn kv_u64(&mut self, k: &str, v: u64) -> &mut Self {
        self.comma_if();
        self.0.push_str(&format!("\"{k}\":{v}"));
        self
    }
    fn kv_i64(&mut self, k: &str, v: i64) -> &mut Self {
        self.comma_if();
        self.0.push_str(&format!("\"{k}\":{v}"));
        self
    }
    fn kv_str(&mut self, k: &str, v: &str) -> &mut Self {
        self.comma_if();
        let esc: String = v.chars().flat_map(|c| match c {
            '"' => vec!['\\', '"'],
            '\\' => vec!['\\', '\\'],
            '\n' => vec!['\\', 'n'],
            c => vec![c],
        }).collect();
        self.0.push_str(&format!("\"{k}\":\"{esc}\""));
        self
    }
    fn kv_bool(&mut self, k: &str, v: bool) -> &mut Self {
        self.comma_if();
        self.0.push_str(&format!("\"{k}\":{v}"));
        self
    }
    fn kv_f64(&mut self, k: &str, v: f64) -> &mut Self {
        self.comma_if();
        self.0.push_str(&format!("\"{k}\":{v}"));
        self
    }
    fn obj(&mut self, k: &str) -> &mut Self {
        self.comma_if();
        self.0.push_str(&format!("\"{k}\":{{"));
        self
    }
    fn end_obj(&mut self) -> &mut Self { self.0.push('}'); self }
    fn comma_if(&mut self) {
        let last = self.0.chars().last().unwrap_or('{');
        if last != '{' && last != '[' { self.0.push(','); }
    }
    fn finish(mut self) -> String { self.0.push('}'); self.0 }
}

// ─── Main ──────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Defaults: 1000 receipts; workdir per-platform; no report JSON.
    let mut n_records: u32 = 1000;
    let mut workdir = default_workdir();
    let mut report_json: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--n" => {
                n_records = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(1000);
                i += 2;
            }
            "--workdir" => {
                workdir = PathBuf::from(args.get(i + 1).cloned().unwrap_or_else(|| default_workdir().to_string_lossy().into_owned()));
                i += 2;
            }
            "--report-json" => {
                report_json = args.get(i + 1).cloned();
                i += 2;
            }
            other => {
                eprintln!("[M.4] unknown arg: {other}");
                i += 1;
            }
        }
    }

    // Ensure workdir exists.
    if let Err(e) = std::fs::create_dir_all(&workdir) {
        eprintln!("[M.4] FATAL: cannot create workdir {}: {e}", workdir.display());
        std::process::exit(2);
    }

    let mut json = J::new();
    json.kv_str("sprint", "M.4");
    json.kv_u64("n_records", n_records as u64);
    json.kv_str("workdir", &workdir.to_string_lossy());
    json.kv_u64("vmrss_start_kb", vmrss_kb());

    let mut fails: usize = 0;

    // ─── Gate 1: T_M4_LEDGER_APPEND ─────────────────────────────────────────
    eprintln!("\n[M.4] ═══ T_M4_LEDGER_APPEND (n={}) ═══", n_records);
    let main_path = workdir.join("m4_main.spinor");
    let _ = std::fs::remove_file(&main_path);

    // Pre-mint receipts AND record their on-wire bytes for the round-trip
    // check in Gate 2. The mint loop is OUTSIDE the timed append loop so the
    // append timing is pure I/O.
    let mut minted: Vec<SpinorReceipt> = Vec::with_capacity(n_records as usize);
    let mut minted_bytes: Vec<[u8; 64]> = Vec::with_capacity(n_records as usize);
    for k in 0..n_records {
        let r = synth_receipt(k);
        minted_bytes.push(r.as_bytes());
        minted.push(r);
    }

    let append_t0 = Instant::now();
    let mut per_append_us: Vec<u64> = Vec::with_capacity(n_records as usize);
    let mut appends_succeeded: u32 = 0;
    let mut append_err: Option<String> = None;
    {
        let mut led = match Ledger::open(&main_path) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("[M.4] FATAL: Ledger::open({}) failed: {e}", main_path.display());
                std::process::exit(2);
            }
        };
        for r in &minted {
            let t0 = Instant::now();
            match led.append(r) {
                Ok(_off) => {
                    appends_succeeded += 1;
                    per_append_us.push(t0.elapsed().as_micros() as u64);
                }
                Err(e) => {
                    append_err = Some(format!("{e}"));
                    break;
                }
            }
        }
        // Ledger dropped here → BufWriter flushed.
    }
    let append_wall_ms = append_t0.elapsed().as_millis();
    let file_size_bytes = std::fs::metadata(&main_path)
        .map(|m| m.len())
        .unwrap_or(0);

    let (p50, p99) = percentiles(&mut per_append_us, &[0.50, 0.99]);
    let append_pass = appends_succeeded == n_records
        && file_size_bytes == (n_records as u64) * 64
        && append_err.is_none();

    eprintln!("[M.4]   appends_succeeded   = {} / {}", appends_succeeded, n_records);
    eprintln!("[M.4]   file_size_bytes     = {} (expected {})", file_size_bytes, (n_records as u64) * 64);
    eprintln!("[M.4]   append_wall_ms      = {}", append_wall_ms);
    eprintln!("[M.4]   append_us_p50/p99   = {} / {}", p50, p99);
    if let Some(ref e) = append_err {
        eprintln!("[M.4]   append_err          = {e}");
    }
    eprintln!("[M.4]   T_M4_LEDGER_APPEND  {}", if append_pass { "PASS" } else { "FAIL" });
    if !append_pass { fails += 1; }

    json.obj("ledger_append")
        .kv_u64("appends_succeeded", appends_succeeded as u64)
        .kv_u64("file_size_bytes", file_size_bytes)
        .kv_u64("append_wall_ms", append_wall_ms as u64)
        .kv_u64("append_wall_us_p50", p50)
        .kv_u64("append_wall_us_p99", p99)
        .kv_str("append_err", append_err.as_deref().unwrap_or(""))
        .end_obj();

    // ─── Gate 2: T_M4_LEDGER_READ ───────────────────────────────────────────
    eprintln!("\n[M.4] ═══ T_M4_LEDGER_READ (n={}) ═══", n_records);
    let mut reads_succeeded: u32 = 0;
    let mut sentinel_failures: u32 = 0;
    let mut byte_divergences: u32 = 0;
    let mut read_err: Option<String> = None;
    {
        let led = match Ledger::open(&main_path) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("[M.4] FATAL: Ledger::open for read failed: {e}");
                std::process::exit(2);
            }
        };
        let mut idx: u32 = 0;
        match led.iter() {
            Ok(iter) => {
                for item in iter {
                    match item {
                        Ok(r) => {
                            reads_succeeded += 1;
                            if r.sentinel != SPINOR_SENTINEL {
                                sentinel_failures += 1;
                            }
                            if (idx as usize) < minted_bytes.len() {
                                if r.as_bytes() != minted_bytes[idx as usize] {
                                    byte_divergences += 1;
                                }
                            }
                            idx += 1;
                        }
                        Err(e) => {
                            read_err = Some(format!("at idx={idx}: {e}"));
                            break;
                        }
                    }
                }
            }
            Err(e) => { read_err = Some(format!("iter start: {e}")); }
        }
    }
    let read_pass = reads_succeeded == n_records
        && sentinel_failures == 0
        && byte_divergences == 0
        && read_err.is_none();

    eprintln!("[M.4]   reads_succeeded     = {} / {}", reads_succeeded, n_records);
    eprintln!("[M.4]   sentinel_failures   = {}", sentinel_failures);
    eprintln!("[M.4]   byte_divergences    = {}", byte_divergences);
    if let Some(ref e) = read_err {
        eprintln!("[M.4]   read_err            = {e}");
    }
    eprintln!("[M.4]   T_M4_LEDGER_READ    {}", if read_pass { "PASS" } else { "FAIL" });
    if !read_pass { fails += 1; }

    json.obj("ledger_read")
        .kv_u64("reads_succeeded", reads_succeeded as u64)
        .kv_u64("sentinel_failures", sentinel_failures as u64)
        .kv_u64("byte_divergences", byte_divergences as u64)
        .kv_str("read_err", read_err.as_deref().unwrap_or(""))
        .end_obj();

    // ─── Gate 3: T_M4_REPLAY_DETERMINISTIC ──────────────────────────────────
    eprintln!("\n[M.4] ═══ T_M4_REPLAY_DETERMINISTIC ═══");
    let dst_a_path = workdir.join("m4_dst_a.spinor");
    let dst_b_path = workdir.join("m4_dst_b.spinor");
    let _ = std::fs::remove_file(&dst_a_path);
    let _ = std::fs::remove_file(&dst_b_path);

    let mut replay_err: Option<String> = None;
    let mut replayed_a: usize = 0;
    let mut replayed_b: usize = 0;
    {
        let src = match Ledger::open(&main_path) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("[M.4] FATAL: open main for replay: {e}"); std::process::exit(2);
            }
        };
        let mut dst_a = match Ledger::open(&dst_a_path) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("[M.4] FATAL: open dst_a: {e}"); std::process::exit(2);
            }
        };
        let mut dst_b = match Ledger::open(&dst_b_path) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("[M.4] FATAL: open dst_b: {e}"); std::process::exit(2);
            }
        };
        match LedgerReplayer::replay_from(&src, &mut dst_a) {
            Ok(n) => replayed_a = n,
            Err(e) => replay_err = Some(format!("replay→a: {e}")),
        }
        if replay_err.is_none() {
            match LedgerReplayer::replay_from(&src, &mut dst_b) {
                Ok(n) => replayed_b = n,
                Err(e) => replay_err = Some(format!("replay→b: {e}")),
            }
        }
    }
    let dst_a_sha = sha256_of_file(&dst_a_path).unwrap_or_default();
    let dst_b_sha = sha256_of_file(&dst_b_path).unwrap_or_default();
    let main_sha = sha256_of_file(&main_path).unwrap_or_default();
    let sha_a_b_match = dst_a_sha == dst_b_sha && !dst_a_sha.is_empty();
    let sha_a_main_match = dst_a_sha == main_sha && !dst_a_sha.is_empty();
    let replay_pass = sha_a_b_match
        && sha_a_main_match
        && replayed_a == n_records as usize
        && replayed_b == n_records as usize
        && replay_err.is_none();

    eprintln!("[M.4]   replayed_a          = {}", replayed_a);
    eprintln!("[M.4]   replayed_b          = {}", replayed_b);
    eprintln!("[M.4]   dst_a_sha256        = {}", dst_a_sha);
    eprintln!("[M.4]   dst_b_sha256        = {}", dst_b_sha);
    eprintln!("[M.4]   main_sha256         = {}", main_sha);
    eprintln!("[M.4]   sha_a_b_match       = {}", sha_a_b_match);
    eprintln!("[M.4]   sha_a_main_match    = {} (replay equals source by construction)", sha_a_main_match);
    if let Some(ref e) = replay_err {
        eprintln!("[M.4]   replay_err          = {e}");
    }
    eprintln!("[M.4]   T_M4_REPLAY_DETERMINISTIC {}", if replay_pass { "PASS" } else { "FAIL" });
    if !replay_pass { fails += 1; }

    json.obj("replay_deterministic")
        .kv_u64("replayed_a", replayed_a as u64)
        .kv_u64("replayed_b", replayed_b as u64)
        .kv_str("dst_a_sha256", &dst_a_sha)
        .kv_str("dst_b_sha256", &dst_b_sha)
        .kv_str("main_sha256", &main_sha)
        .kv_bool("sha_a_b_match", sha_a_b_match)
        .kv_bool("sha_a_main_match", sha_a_main_match)
        .kv_str("replay_err", replay_err.as_deref().unwrap_or(""))
        .end_obj();

    // ─── Gate 4: T_M4_CROSS_DEVICE_REPLAY ───────────────────────────────────
    eprintln!("\n[M.4] ═══ T_M4_CROSS_DEVICE_REPLAY ═══");
    let device_a_path = workdir.join("m4_device_a.spinor");
    let device_b_path = workdir.join("m4_device_b.spinor");
    let reference_path = workdir.join("m4_reference.spinor");
    let reverse_reference_path = workdir.join("m4_reverse_reference.spinor");
    for p in [&device_a_path, &device_b_path, &reference_path, &reverse_reference_path] {
        let _ = std::fs::remove_file(p);
    }

    let half = n_records / 2;
    let mut cd_err: Option<String> = None;

    // Step 1+3: device A mints + appends 0..half locally.
    {
        let mut dev_a = match Ledger::open(&device_a_path) {
            Ok(l) => l, Err(e) => { cd_err = Some(format!("open device_a: {e}")); Ledger::open(&dst_a_path).unwrap() }
        };
        if cd_err.is_none() {
            for k in 0..half {
                if let Err(e) = dev_a.append(&minted[k as usize]) {
                    cd_err = Some(format!("device_a append k={k}: {e}")); break;
                }
            }
        }
    }
    // Step 2+4: device B mints + appends half..n locally.
    if cd_err.is_none() {
        let mut dev_b = match Ledger::open(&device_b_path) {
            Ok(l) => l, Err(e) => { cd_err = Some(format!("open device_b: {e}")); Ledger::open(&dst_b_path).unwrap() }
        };
        if cd_err.is_none() {
            for k in half..n_records {
                if let Err(e) = dev_b.append(&minted[k as usize]) {
                    cd_err = Some(format!("device_b append k={k}: {e}")); break;
                }
            }
        }
    }
    // Step 5: each device runs broadcast_to_peers(0) to emit its half.
    // Step 6+7: cross-replay.
    let mut bcast_a: Vec<SpinorReceipt> = Vec::new();
    let mut bcast_b: Vec<SpinorReceipt> = Vec::new();
    if cd_err.is_none() {
        match Ledger::open(&device_a_path).and_then(|l| l.broadcast_to_peers(0)) {
            Ok(v) => bcast_a = v,
            Err(e) => cd_err = Some(format!("device_a broadcast: {e}")),
        }
    }
    if cd_err.is_none() {
        match Ledger::open(&device_b_path).and_then(|l| l.broadcast_to_peers(0)) {
            Ok(v) => bcast_b = v,
            Err(e) => cd_err = Some(format!("device_b broadcast: {e}")),
        }
    }
    // Device A receives B's broadcast and appends to its own ledger.
    if cd_err.is_none() {
        let mut dev_a = Ledger::open(&device_a_path).unwrap();
        if let Err(e) = LedgerReplayer::replay_list(&bcast_b, &mut dev_a) {
            cd_err = Some(format!("device_a replay_list(bcast_b): {e}"));
        }
    }
    // Device B receives A's broadcast and appends to its own ledger.
    if cd_err.is_none() {
        let mut dev_b = Ledger::open(&device_b_path).unwrap();
        if let Err(e) = LedgerReplayer::replay_list(&bcast_a, &mut dev_b) {
            cd_err = Some(format!("device_b replay_list(bcast_a): {e}"));
        }
    }

    // Build canonical reference (0..n) and reverse-reference (half..n ++ 0..half).
    if cd_err.is_none() {
        let mut ref_l = Ledger::open(&reference_path).unwrap();
        for k in 0..n_records {
            ref_l.append(&minted[k as usize]).unwrap();
        }
    }
    if cd_err.is_none() {
        let mut rev_l = Ledger::open(&reverse_reference_path).unwrap();
        for k in half..n_records {
            rev_l.append(&minted[k as usize]).unwrap();
        }
        for k in 0..half {
            rev_l.append(&minted[k as usize]).unwrap();
        }
    }

    let dev_a_sha = sha256_of_file(&device_a_path).unwrap_or_default();
    let dev_b_sha = sha256_of_file(&device_b_path).unwrap_or_default();
    let ref_sha = sha256_of_file(&reference_path).unwrap_or_default();
    let rev_sha = sha256_of_file(&reverse_reference_path).unwrap_or_default();
    let dev_a_size = std::fs::metadata(&device_a_path).map(|m| m.len()).unwrap_or(0);
    let dev_b_size = std::fs::metadata(&device_b_path).map(|m| m.len()).unwrap_or(0);

    let dev_a_matches_ref = !dev_a_sha.is_empty() && dev_a_sha == ref_sha;
    let dev_b_matches_rev = !dev_b_sha.is_empty() && dev_b_sha == rev_sha;
    let all_match = dev_a_matches_ref && dev_b_matches_rev;
    let cd_pass = all_match
        && dev_a_size == (n_records as u64) * 64
        && dev_b_size == (n_records as u64) * 64
        && cd_err.is_none();

    eprintln!("[M.4]   device_a_size       = {} (expect {})", dev_a_size, (n_records as u64) * 64);
    eprintln!("[M.4]   device_b_size       = {} (expect {})", dev_b_size, (n_records as u64) * 64);
    eprintln!("[M.4]   device_a_sha256     = {}", dev_a_sha);
    eprintln!("[M.4]   device_b_sha256     = {}", dev_b_sha);
    eprintln!("[M.4]   reference_sha256    = {}", ref_sha);
    eprintln!("[M.4]   reverse_ref_sha256  = {}", rev_sha);
    eprintln!("[M.4]   device_a_matches_reference (A local + B replay = 0..n) = {}", dev_a_matches_ref);
    eprintln!("[M.4]   device_b_matches_reverse   (B local + A replay = half..n ++ 0..half) = {}", dev_b_matches_rev);
    eprintln!("[M.4]   all_match           = {}", all_match);
    if let Some(ref e) = cd_err {
        eprintln!("[M.4]   cross_device_err    = {e}");
    }
    eprintln!("[M.4]   T_M4_CROSS_DEVICE_REPLAY {}", if cd_pass { "PASS" } else { "FAIL" });
    if !cd_pass { fails += 1; }

    json.obj("cross_device_replay")
        .kv_u64("device_a_size_bytes", dev_a_size)
        .kv_u64("device_b_size_bytes", dev_b_size)
        .kv_str("device_a_final_sha256", &dev_a_sha)
        .kv_str("device_b_final_sha256", &dev_b_sha)
        .kv_str("reference_sha256", &ref_sha)
        .kv_str("reverse_reference_sha256", &rev_sha)
        .kv_bool("device_a_matches_reference", dev_a_matches_ref)
        .kv_bool("device_b_matches_reverse", dev_b_matches_rev)
        .kv_bool("all_match", all_match)
        .kv_str("cross_device_err", cd_err.as_deref().unwrap_or(""))
        .end_obj();

    // ─── Gates summary ──────────────────────────────────────────────────────
    json.obj("gates")
        .kv_str("T_M4_LEDGER_APPEND", if append_pass { "PASS" } else { "FAIL" })
        .kv_str("T_M4_LEDGER_READ", if read_pass { "PASS" } else { "FAIL" })
        .kv_str("T_M4_REPLAY_DETERMINISTIC", if replay_pass { "PASS" } else { "FAIL" })
        .kv_str("T_M4_CROSS_DEVICE_REPLAY", if cd_pass { "PASS" } else { "FAIL" })
        .end_obj();

    json.kv_u64("vmrss_end_kb", vmrss_kb());

    let final_json = json.finish();
    if let Some(path) = report_json.as_deref() {
        if let Err(e) = std::fs::write(path, &final_json) {
            eprintln!("[M.4] WARN: failed to write report-json to {path}: {e}");
        } else {
            eprintln!("[M.4] report JSON written to {path}");
        }
    }

    eprintln!("\n[M.4] ═══ SUMMARY ═══");
    eprintln!("[M.4]   T_M4_LEDGER_APPEND        {}", if append_pass { "PASS" } else { "FAIL" });
    eprintln!("[M.4]   T_M4_LEDGER_READ          {}", if read_pass { "PASS" } else { "FAIL" });
    eprintln!("[M.4]   T_M4_REPLAY_DETERMINISTIC {}", if replay_pass { "PASS" } else { "FAIL" });
    eprintln!("[M.4]   T_M4_CROSS_DEVICE_REPLAY  {}", if cd_pass { "PASS" } else { "FAIL" });

    if fails == 0 {
        eprintln!("[M.4] ALL GATES PASS");
        std::process::exit(0);
    } else {
        eprintln!("[M.4] {} gate(s) FAILED", fails);
        std::process::exit(1);
    }
}

// ─── Helpers ───────────────────────────────────────────────────────────────

#[cfg(target_os = "android")]
fn default_workdir() -> PathBuf { PathBuf::from("/data/local/tmp") }

#[cfg(not(target_os = "android"))]
fn default_workdir() -> PathBuf { std::env::temp_dir() }

/// Compute percentiles from a Vec<u64> of timings (in micros). Mutates
/// the input by sorting. Returns `(p_a, p_b)` for the requested fractions.
fn percentiles(samples: &mut Vec<u64>, fractions: &[f64; 2]) -> (u64, u64) {
    if samples.is_empty() { return (0, 0); }
    samples.sort_unstable();
    let n = samples.len();
    let idx0 = ((fractions[0] * n as f64) as usize).min(n - 1);
    let idx1 = ((fractions[1] * n as f64) as usize).min(n - 1);
    (samples[idx0], samples[idx1])
}
