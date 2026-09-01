//! Sprint ledger-autowire — /v1/dialogue + PoUW ledger autowire smoke
//! harness.
//!
//! Verifies the three substantive ledger-autowire gates by driving N=5
//! sequential dialogue invocations against a running daemon and asserting
//! that the PoUW ledger file accumulates 5 * 3 = 15 receipts (= 960 bytes)
//! that are byte-identical to the base64-decoded receipts returned in the
//! HTTP responses.
//!
//!   T_AUTOWIRE_LEDGER_GROWS         — pre→post delta == 5*3*64 == 960
//!   T_AUTOWIRE_RECEIPT_BYTE_IDENTITY — all 15*64 bytes byte-identical
//!                                      between ledger file + responses
//!   T_AUTOWIRE_NO_REGRESSION        — every dialogue HTTP 200 + 3
//!                                      receipts (the chat-integration
//!                                      contract still holds)
//!
//! CLI:
//!   sp_chat_ledger_autowire_smoke
//!         [--url http://127.0.0.1:8080/v1/dialogue]
//!         [--prompt "What is the capital of France?"]
//!         [--ledger-path <PATH>]          # required, file on the daemon side
//!         [--n 5]                         # number of dialogue invocations
//!         [--report-json PATH]
//!
//! REQUIREMENT: a running sp-daemon with BOTH --model AND --memo-model AND
//! --pouw-ledger-path configured. The path passed via --ledger-path MUST
//! match the path the daemon was started with (the smoke binary reads the
//! ledger file directly off disk). Both processes must see the same path
//! — colocate them on the same host (or push the smoke binary to the
//! daemon's host for android runs).
//!
//! Operator:
//!   sp-daemon start --model E.spm --tokenizer E.spt \
//!                   --memo-model M.spm --memo-tokenizer M.spt \
//!                   --pouw-ledger-path /tmp/autowire.spinor
//!   sp_chat_ledger_autowire_smoke --ledger-path /tmp/autowire.spinor \
//!                                  --report-json /tmp/autowire_report.json
//!
//! Hand-rolled HTTP/JSON/base64 mirror sp_chat_dialogue_smoke (no
//! new dependencies).

use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::Path;
use std::time::Duration;

// ─── Tiny JSON helpers (linear scans; we wrote the server side) ─────────

struct DialogueResp {
    response: String,
    receipts: Vec<String>,
    wall_ms: u64,
}

fn extract_string_field(body: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\":\"");
    let start = body.find(&pat)? + pat.len();
    let rest = &body[start..];
    let mut out = String::new();
    let mut chars = rest.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            if let Some(n) = chars.next() { out.push(c); out.push(n); }
        } else if c == '"' {
            return Some(out);
        } else {
            out.push(c);
        }
    }
    None
}

fn extract_u64_field(body: &str, key: &str) -> Option<u64> {
    let pat = format!("\"{key}\":");
    let start = body.find(&pat)? + pat.len();
    let rest = &body[start..];
    let end = rest.find(|c: char| !c.is_ascii_digit())?;
    rest[..end].parse().ok()
}

fn extract_string_array(body: &str, key: &str) -> Option<Vec<String>> {
    let pat = format!("\"{key}\":[");
    let start = body.find(&pat)? + pat.len();
    let rest = &body[start..];
    let end = rest.find(']')?;
    let inner = &rest[..end];
    let mut out = Vec::new();
    let mut chars = inner.chars();
    while let Some(c) = chars.next() {
        if c == '"' {
            let mut s = String::new();
            for cc in chars.by_ref() {
                if cc == '"' { break; }
                s.push(cc);
            }
            out.push(s);
        }
    }
    Some(out)
}

fn parse_response(body: &str) -> Option<DialogueResp> {
    Some(DialogueResp {
        response: extract_string_field(body, "response")?,
        receipts: extract_string_array(body, "receipts")?,
        wall_ms: extract_u64_field(body, "wall_ms")?,
    })
}

// ─── Hand-rolled base64 DECODE ───────────────────────────────────────────

fn base64_decode(input: &str) -> Result<Vec<u8>, String> {
    fn val(c: u8) -> Option<u8> {
        match c {
            b'A'..=b'Z' => Some(c - b'A'),
            b'a'..=b'z' => Some(c - b'a' + 26),
            b'0'..=b'9' => Some(c - b'0' + 52),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    }
    let bytes = input.as_bytes();
    if bytes.len() % 4 != 0 {
        return Err(format!("base64 len {} not multiple of 4", bytes.len()));
    }
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    let mut i = 0;
    while i < bytes.len() {
        let c0 = bytes[i];
        let c1 = bytes[i + 1];
        let c2 = bytes[i + 2];
        let c3 = bytes[i + 3];
        let v0 = val(c0).ok_or_else(|| format!("invalid b64 char {c0}"))?;
        let v1 = val(c1).ok_or_else(|| format!("invalid b64 char {c1}"))?;
        let n: u32 = ((v0 as u32) << 18) | ((v1 as u32) << 12);
        if c2 == b'=' {
            out.push((n >> 16) as u8);
            break;
        }
        let v2 = val(c2).ok_or_else(|| format!("invalid b64 char {c2}"))?;
        let n = n | ((v2 as u32) << 6);
        if c3 == b'=' {
            out.push((n >> 16) as u8);
            out.push((n >> 8) as u8);
            break;
        }
        let v3 = val(c3).ok_or_else(|| format!("invalid b64 char {c3}"))?;
        let n = n | v3 as u32;
        out.push((n >> 16) as u8);
        out.push((n >> 8) as u8);
        out.push(n as u8);
        i += 4;
    }
    Ok(out)
}

// ─── HTTP POST ───────────────────────────────────────────────────────────

#[derive(Debug)]
struct HttpResp {
    status: u16,
    body: String,
}

fn http_post(url: &str, body: &str, timeout: Duration) -> Result<HttpResp, String> {
    let rest = url.strip_prefix("http://").ok_or_else(|| format!("only http:// URLs supported (got {url})"))?;
    let slash = rest.find('/').ok_or_else(|| "URL missing path".to_string())?;
    let host_port = &rest[..slash];
    let path = &rest[slash..];

    let mut stream = TcpStream::connect_timeout(
        &host_port.parse().map_err(|e| format!("invalid addr {host_port}: {e}"))?,
        timeout,
    ).map_err(|e| format!("connect {host_port}: {e}"))?;
    stream.set_read_timeout(Some(timeout)).map_err(|e| e.to_string())?;
    stream.set_write_timeout(Some(timeout)).map_err(|e| e.to_string())?;

    let req = format!(
        "POST {path} HTTP/1.1\r\nHost: {host_port}\r\nContent-Type: application/json\r\nContent-Length: {len}\r\nConnection: close\r\n\r\n{body}",
        len = body.len(),
    );
    stream.write_all(req.as_bytes()).map_err(|e| format!("send: {e}"))?;

    let mut buf = Vec::new();
    stream.read_to_end(&mut buf).map_err(|e| format!("recv: {e}"))?;
    let raw = String::from_utf8_lossy(&buf).into_owned();

    let head_end = raw.find("\r\n\r\n").ok_or_else(|| "no header terminator".to_string())?;
    let head = &raw[..head_end];
    let body = &raw[head_end + 4..];
    let status_line = head.lines().next().ok_or_else(|| "empty response".to_string())?;
    let parts: Vec<&str> = status_line.split_whitespace().collect();
    if parts.len() < 2 { return Err(format!("bad status line: {status_line}")); }
    let status: u16 = parts[1].parse().map_err(|e| format!("status parse: {e}"))?;
    Ok(HttpResp { status, body: body.to_string() })
}

// ─── Ledger file size helper ─────────────────────────────────────────────

fn ledger_size_bytes(path: &Path) -> u64 {
    match std::fs::metadata(path) {
        Ok(m) => m.len(),
        Err(_) => 0, // absent file → treat as size 0 (daemon creates on first append)
    }
}

fn ledger_read_all(path: &Path) -> Result<Vec<u8>, String> {
    std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))
}

// ─── Main ────────────────────────────────────────────────────────────────

fn main() {
    let mut url = String::from("http://127.0.0.1:8080/v1/dialogue");
    let mut prompt = String::from("What is the capital of France?");
    let mut ledger_path: Option<String> = None;
    let mut n: usize = 5;
    let mut report_json: Option<String> = None;

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--url" => { url = args.get(i + 1).cloned().unwrap_or(url); i += 2; }
            "--prompt" => { prompt = args.get(i + 1).cloned().unwrap_or(prompt); i += 2; }
            "--ledger-path" => { ledger_path = args.get(i + 1).cloned(); i += 2; }
            "--n" => {
                let v = args.get(i + 1).map(|s| s.parse::<usize>().unwrap_or(5)).unwrap_or(5);
                n = v.max(1);
                i += 2;
            }
            "--report-json" => { report_json = args.get(i + 1).cloned(); i += 2; }
            "--help" | "-h" => {
                eprintln!("Usage: sp_chat_ledger_autowire_smoke [--url URL] [--prompt TEXT] --ledger-path PATH [--n N] [--report-json PATH]");
                std::process::exit(0);
            }
            other => { eprintln!("[autowire] unknown arg: {other}"); i += 1; }
        }
    }

    let ledger_path = match ledger_path {
        Some(p) => p,
        None => {
            eprintln!("[autowire] FAIL: --ledger-path is required (must match daemon's --pouw-ledger-path)");
            std::process::exit(2);
        }
    };
    let ledger_path = std::path::PathBuf::from(ledger_path);

    eprintln!("[autowire] ═══ ledger-autowire smoke ═══");
    eprintln!("[autowire]   url:         {url}");
    eprintln!("[autowire]   ledger:      {}", ledger_path.display());
    eprintln!("[autowire]   N dialogues: {n}");

    // ── Pre-snapshot ledger size ────────────────────────────────────────
    let pre_size = ledger_size_bytes(&ledger_path);
    eprintln!("[autowire]   pre_size:    {pre_size} bytes");

    // JSON-escape the prompt.
    let prompt_esc: String = prompt.chars().flat_map(|c| match c {
        '"' => vec!['\\', '"'],
        '\\' => vec!['\\', '\\'],
        '\n' => vec!['\\', 'n'],
        c => vec![c],
    }).collect();
    let body = format!("{{\"prompt\":\"{prompt_esc}\"}}");

    // Collect base64-decoded receipts from every successful dialogue.
    // `all_receipts_b64[i][j]` is the j-th receipt of the i-th dialogue.
    let mut all_receipts_decoded: Vec<Vec<Vec<u8>>> = Vec::with_capacity(n);
    let mut dialogue_status: Vec<u16> = Vec::with_capacity(n);
    let mut dialogue_response_heads: Vec<String> = Vec::with_capacity(n);
    let mut dialogue_walls_ms: Vec<u64> = Vec::with_capacity(n);
    let mut dialogue_receipt_counts: Vec<usize> = Vec::with_capacity(n);
    let mut transport_errors: Vec<String> = Vec::new();

    let wall_start = std::time::Instant::now();
    for d in 0..n {
        eprintln!("[autowire]   ── dialogue {} / {} ──", d + 1, n);
        let resp = match http_post(&url, &body, Duration::from_secs(300)) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("[autowire]     transport FAIL: {e}");
                transport_errors.push(format!("dialogue {}: {e}", d + 1));
                dialogue_status.push(0);
                dialogue_response_heads.push(String::new());
                dialogue_walls_ms.push(0);
                dialogue_receipt_counts.push(0);
                all_receipts_decoded.push(Vec::new());
                continue;
            }
        };

        dialogue_status.push(resp.status);
        let parsed = parse_response(&resp.body);
        let (head, wall_ms, receipts_b64) = match parsed {
            Some(d) => {
                let h: String = d.response.chars().take(48).collect();
                (h, d.wall_ms, d.receipts)
            }
            None => {
                eprintln!("[autowire]     JSON parse failed; raw 200 chars: {:?}", &resp.body.chars().take(200).collect::<String>());
                (String::new(), 0u64, Vec::new())
            }
        };
        eprintln!("[autowire]     http_status     = {}", resp.status);
        eprintln!("[autowire]     response_head   = {head:?}");
        eprintln!("[autowire]     response_wall_ms= {wall_ms}");
        eprintln!("[autowire]     receipts        = {} entries", receipts_b64.len());
        dialogue_response_heads.push(head);
        dialogue_walls_ms.push(wall_ms);
        dialogue_receipt_counts.push(receipts_b64.len());

        let mut decoded_for_this: Vec<Vec<u8>> = Vec::with_capacity(receipts_b64.len());
        for (ridx, r_b64) in receipts_b64.iter().enumerate() {
            match base64_decode(r_b64) {
                Ok(bytes) => {
                    eprintln!("[autowire]       receipt[{ridx}] {} bytes (sentinel={:02X})", bytes.len(), bytes.last().copied().unwrap_or(0));
                    decoded_for_this.push(bytes);
                }
                Err(e) => {
                    eprintln!("[autowire]       receipt[{ridx}] base64 decode FAIL: {e}");
                    decoded_for_this.push(Vec::new());
                }
            }
        }
        all_receipts_decoded.push(decoded_for_this);
    }
    let total_wall_ms = wall_start.elapsed().as_millis() as u64;
    eprintln!("[autowire]   total wall: {total_wall_ms} ms");

    // ── Post-snapshot ledger size ───────────────────────────────────────
    let post_size = ledger_size_bytes(&ledger_path);
    let delta = post_size.saturating_sub(pre_size);
    let expected: u64 = (n as u64) * 3 * 64;
    eprintln!("[autowire]   post_size:   {post_size} bytes");
    eprintln!("[autowire]   delta:       {delta} bytes (expected {expected})");

    // ── Gate 1: T_AUTOWIRE_LEDGER_GROWS ─────────────────────────────────
    eprintln!("\n[autowire] ═══ T_AUTOWIRE_LEDGER_GROWS ═══");
    eprintln!("[autowire]   pre_size  = {pre_size}");
    eprintln!("[autowire]   post_size = {post_size}");
    eprintln!("[autowire]   delta     = {delta}");
    eprintln!("[autowire]   expected  = {expected}");
    let ledger_grows_pass = delta == expected;
    eprintln!("[autowire]   T_AUTOWIRE_LEDGER_GROWS {}", if ledger_grows_pass { "PASS" } else { "FAIL" });

    // ── Gate 2: T_AUTOWIRE_RECEIPT_BYTE_IDENTITY ────────────────────────
    // Read the appended slice from the ledger (offset pre_size, length
    // delta), split into 64-byte records, and compare to the concatenation
    // of all decoded receipts in dialogue order.
    eprintln!("\n[autowire] ═══ T_AUTOWIRE_RECEIPT_BYTE_IDENTITY ═══");
    let mut receipts_compared = 0usize;
    let mut byte_divergences = 0usize;
    let mut identity_run_err: Option<String> = None;
    match ledger_read_all(&ledger_path) {
        Ok(all_bytes) => {
            if (all_bytes.len() as u64) < post_size {
                identity_run_err = Some(format!(
                    "ledger file shrank between size-check and read: post_size={post_size} read_len={}",
                    all_bytes.len()
                ));
            } else if delta == 0 {
                identity_run_err = Some("delta == 0 — no new records to compare".to_string());
            } else {
                let appended = &all_bytes[pre_size as usize..post_size as usize];
                // appended.len() should == delta == expected; chunks of 64.
                let mut cursor = 0usize;
                for (d_idx, dialog_receipts) in all_receipts_decoded.iter().enumerate() {
                    for (r_idx, decoded) in dialog_receipts.iter().enumerate() {
                        if decoded.len() != 64 {
                            eprintln!("[autowire]   receipt[{d_idx},{r_idx}] response-side decoded len = {} (skip)", decoded.len());
                            continue;
                        }
                        if cursor + 64 > appended.len() {
                            eprintln!("[autowire]   receipt[{d_idx},{r_idx}] ledger ran out at cursor={cursor} (appended.len={})", appended.len());
                            byte_divergences += 64;
                            continue;
                        }
                        let ledger_slice = &appended[cursor..cursor + 64];
                        receipts_compared += 1;
                        let mut local_divs = 0;
                        for k in 0..64 {
                            if ledger_slice[k] != decoded[k] { local_divs += 1; }
                        }
                        if local_divs == 0 {
                            eprintln!("[autowire]   receipt[{d_idx},{r_idx}] identity OK ({} bytes)", 64);
                        } else {
                            byte_divergences += local_divs;
                            eprintln!("[autowire]   receipt[{d_idx},{r_idx}] DIVERGED in {local_divs} of 64 bytes");
                            eprintln!("[autowire]     ledger head: {}", ledger_slice.iter().take(8).map(|b| format!("{b:02X}")).collect::<Vec<_>>().join(" "));
                            eprintln!("[autowire]     decoded head: {}", decoded.iter().take(8).map(|b| format!("{b:02X}")).collect::<Vec<_>>().join(" "));
                        }
                        cursor += 64;
                    }
                }
            }
        }
        Err(e) => {
            identity_run_err = Some(e);
        }
    }
    eprintln!("[autowire]   receipts_compared = {receipts_compared}");
    eprintln!("[autowire]   byte_divergences  = {byte_divergences}");
    if let Some(err) = &identity_run_err { eprintln!("[autowire]   identity_run_err = {err}"); }
    let identity_pass = identity_run_err.is_none() && byte_divergences == 0 && receipts_compared == n * 3;
    eprintln!("[autowire]   T_AUTOWIRE_RECEIPT_BYTE_IDENTITY {}", if identity_pass { "PASS" } else { "FAIL" });

    // ── Gate 3: T_AUTOWIRE_NO_REGRESSION ────────────────────────────────
    // Every dialogue invocation must have returned HTTP 200 with 3 receipts
    // — i.e. the chat-integration contract held under the new auto-append
    // wire-in. (We do NOT re-run sp_chat_dialogue_smoke as a separate
    // binary here — the regression check is the same shape: 200 + 3
    // receipts per dialogue. If this holds across N=5 invocations with the
    // ledger autowire active, the chat-integration gates by construction
    // still PASS.)
    eprintln!("\n[autowire] ═══ T_AUTOWIRE_NO_REGRESSION ═══");
    let mut bad_status = 0usize;
    let mut bad_receipt_count = 0usize;
    for d_idx in 0..n {
        if dialogue_status.get(d_idx).copied() != Some(200) { bad_status += 1; }
        if dialogue_receipt_counts.get(d_idx).copied() != Some(3) { bad_receipt_count += 1; }
    }
    eprintln!("[autowire]   dialogues_with_status_200          = {}", n - bad_status);
    eprintln!("[autowire]   dialogues_with_3_receipts          = {}", n - bad_receipt_count);
    eprintln!("[autowire]   transport_errors                   = {}", transport_errors.len());
    let no_regression_pass = bad_status == 0 && bad_receipt_count == 0 && transport_errors.is_empty();
    eprintln!("[autowire]   T_AUTOWIRE_NO_REGRESSION {}", if no_regression_pass { "PASS" } else { "FAIL" });

    // ── Report ───────────────────────────────────────────────────────────
    let mut fails = 0usize;
    if !ledger_grows_pass     { fails += 1; }
    if !identity_pass         { fails += 1; }
    if !no_regression_pass    { fails += 1; }

    let escape_path = ledger_path.display().to_string().replace('\\', "\\\\").replace('"', "\\\"");
    let identity_err_field = identity_run_err
        .clone()
        .map(|e| e.replace('\\', "\\\\").replace('"', "\\\""))
        .unwrap_or_default();
    let transport_err_concat = transport_errors.join("; ").replace('\\', "\\\\").replace('"', "\\\"");
    let report = format!(
        "{{\
\"sprint\":\"ledger-autowire\",\
\"url\":\"{url}\",\
\"ledger_path\":\"{escape_path}\",\
\"n_dialogues\":{n},\
\"pre_size\":{pre_size},\
\"post_size\":{post_size},\
\"delta\":{delta},\
\"expected\":{expected},\
\"receipts_compared\":{receipts_compared},\
\"byte_divergences\":{byte_divergences},\
\"identity_run_err\":\"{identity_err_field}\",\
\"dialogues_with_status_200\":{},\
\"dialogues_with_3_receipts\":{},\
\"transport_errors\":\"{transport_err_concat}\",\
\"total_wall_ms\":{total_wall_ms},\
\"gates\":{{\
\"T_AUTOWIRE_LEDGER_GROWS\":\"{}\",\
\"T_AUTOWIRE_RECEIPT_BYTE_IDENTITY\":\"{}\",\
\"T_AUTOWIRE_NO_REGRESSION\":\"{}\"\
}}\
}}",
        n - bad_status,
        n - bad_receipt_count,
        if ledger_grows_pass { "PASS" } else { "FAIL" },
        if identity_pass { "PASS" } else { "FAIL" },
        if no_regression_pass { "PASS" } else { "FAIL" },
    );

    if let Some(p) = report_json.as_deref() {
        if let Err(e) = std::fs::write(p, &report) {
            eprintln!("[autowire] WARN: report write to {p} failed: {e}");
        } else {
            eprintln!("\n[autowire] report JSON written to {p}");
        }
    }

    eprintln!("\n[autowire] ═══ SUMMARY ═══");
    eprintln!("[autowire]   T_AUTOWIRE_LEDGER_GROWS          {}", if ledger_grows_pass { "PASS" } else { "FAIL" });
    eprintln!("[autowire]   T_AUTOWIRE_RECEIPT_BYTE_IDENTITY {}", if identity_pass { "PASS" } else { "FAIL" });
    eprintln!("[autowire]   T_AUTOWIRE_NO_REGRESSION         {}", if no_regression_pass { "PASS" } else { "FAIL" });

    if fails == 0 {
        eprintln!("[autowire] ALL GATES PASS");
        std::process::exit(0);
    } else {
        eprintln!("[autowire] {fails} gate(s) FAILED");
        std::process::exit(1);
    }
}
