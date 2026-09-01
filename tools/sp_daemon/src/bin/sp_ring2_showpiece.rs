//! sp_ring2_showpiece — the Trick #8 finale: two processes, one wire, residue
//! blocks crossing it mid-decode.
//!
//!   PROCESS A (the peer store):   sp_ring2_showpiece --serve 127.0.0.1:47391
//!   PROCESS B (the decoder):      sp_ring2_showpiece --decode 127.0.0.1:47391 \
//!                                     <model.sp-model> <model.sp-tokenizer>
//!
//! Process B drives the CANONICAL math-core two-ring decode (qwen3_generate_kv
//! from the linked L1 static libs — the same single-source decode every tier
//! runs) twice on the same prompt:
//!   1. BASELINE: all knobs off — pure resident f32 attention.
//!   2. NETWORK FUSION: SP_NTT_KV (K cached as dual-prime residue blocks) ×
//!      SP_RECALL/SP_RING2 (6-slot Ring-1 forces eviction) × SP_RING2_DISK with
//!      the QUIC peer registered as THE Ring-2 backend — every cold K block is
//!      an 8 KB residue packet fetched from process A just-in-time.
//! The gate: the two token sequences are IDENTICAL, with nonzero wire traffic.
//!
//! Env knobs are set via the CRT's _putenv (Rust's set_var updates the Win32
//! env block, but the MSVC CRT getenv the math-core uses reads its own cached
//! copy — _putenv updates that copy).

use std::ffi::{c_char, c_int, c_void, CString};
use std::net::SocketAddr;
use std::sync::atomic::Ordering;

use sp_daemon::network::ring2_quic::{
    bind_ring2_server, register_ring2_quic, run_ring2_server, unregister_ring2_quic,
    NET_BATCHES, NET_BYTES, NET_READS, NET_WRITES,
};

// The canonical decode + loader surface, declared manually (opaque handles) so
// this binary does not depend on bindgen's struct naming. All symbols come from
// the math-core static libs build.rs already links.
extern "C" {
    fn sp_model_load(model: *const c_char, tok: *const c_char, out: *mut *mut c_void) -> c_int;
    fn sp_model_to_qwen3(m: *const c_void) -> *mut c_void;
    fn qwen3_generate_kv(m: *const c_void, seq: *mut i32, n_prompt: c_int,
                         n_gen: c_int, eos_id: c_int) -> c_int;
    fn _putenv(s: *const c_char) -> c_int;   /* MSVC CRT env (getenv's view) */
}

fn putenv(kv: &str) {
    let c = CString::new(kv).unwrap();
    unsafe { _putenv(c.as_ptr()) };
}

fn knobs_off() {
    for k in ["SP_NTT_KV=", "SP_RECALL_B=", "SP_RECALL_W=", "SP_RECALL_SINK=",
              "SP_RING2=", "SP_RING2_DISK=", "SP_RECALL_FUSE=", "SP_ENGINE_NTT_ATTN="] {
        putenv(k);                            /* "NAME=" removes from the CRT env */
    }
}

const NPROMPT: usize = 4;
const NGEN: usize = 12;
const PTOT: usize = NPROMPT + NGEN;

fn run_decode(model: *const c_void) -> Vec<i32> {
    let mut seq = vec![0i32; PTOT];
    seq[0] = 1; seq[1] = 2; seq[2] = 3; seq[3] = 4;
    let t0 = std::time::Instant::now();
    let n = unsafe {
        qwen3_generate_kv(model, seq.as_mut_ptr(), NPROMPT as c_int, NGEN as c_int, -1)
    };
    let dt = t0.elapsed().as_secs_f64();
    assert_eq!(n as usize, PTOT, "decode completed full length");
    eprintln!("    decode: {} tokens in {:.3}s = {:.2} tok/s", NGEN, dt, NGEN as f64 / dt);
    seq
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("--serve") => {
            let addr: SocketAddr = args[2].parse().expect("addr");
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let ep = bind_ring2_server(addr).expect("bind");
                eprintln!("[showpiece A] Ring-2 peer store SERVING on {addr} — raw residue blocks, SPR2 framing");
                let _ = run_ring2_server(ep).await;
            });
        }
        Some("--decode") => {
            let peer: SocketAddr = args[2].parse().expect("peer addr");
            let model_p = CString::new(args[3].as_str()).unwrap();
            let tok_p = CString::new(args[4].as_str()).unwrap();

            let mut spm: *mut c_void = std::ptr::null_mut();
            let rc = unsafe { sp_model_load(model_p.as_ptr(), tok_p.as_ptr(), &mut spm) };
            assert_eq!(rc, 0, "sp_model_load");
            let qm = unsafe { sp_model_to_qwen3(spm) };
            assert!(!qm.is_null(), "sp_model_to_qwen3");
            eprintln!("[showpiece B] model loaded via swivel (.sp-model OK_Q8 arena)");

            // 1. BASELINE — pure resident decode, no backend involved.
            knobs_off();
            eprintln!("[showpiece B] baseline decode (knobs off):");
            let base = run_decode(qm);

            // 2. NETWORK FUSION — peer registered as THE Ring-2 backend; 6-slot
            //    Ring-1 forces every cold K through the wire as residue blocks.
            register_ring2_quic(peer).expect("register peer");
            putenv("SP_NTT_KV=1");
            putenv("SP_RECALL_B=64");      // identity budget: every position attended
            putenv("SP_RECALL_W=4");       // 6-slot Ring-1 (2 sinks + 4 window)
            putenv("SP_RECALL_SINK=2");
            putenv("SP_RING2=1");
            putenv("SP_RING2_DISK=1");     // backend mode -> the registered QUIC peer
            eprintln!("[showpiece B] network-fusion decode (peer {peer} is Ring-2):");
            let net = run_decode(qm);
            knobs_off();
            unregister_ring2_quic();

            // 3. The verdict.
            let writes = NET_WRITES.load(Ordering::Relaxed);
            let reads = NET_READS.load(Ordering::Relaxed);
            let batches = NET_BATCHES.load(Ordering::Relaxed);
            let bytes = NET_BYTES.load(Ordering::Relaxed);
            eprintln!("[showpiece] wire traffic: {writes} block writes, {reads} block reads \
                       ({batches} concurrent batches), {:.1} KB of raw residue payload", bytes as f64 / 1024.0);
            eprintln!("[showpiece] baseline: {:?}", base);
            eprintln!("[showpiece] network:  {:?}", net);
            let identical = base == net;
            eprintln!("[showpiece] SEQUENCES IDENTICAL = {identical}   wire traffic nonzero = {}",
                      writes > 0 && reads > 0);
            assert!(identical, "network-fusion sequence must equal baseline");
            assert!(writes > 0 && reads > 0, "blocks must actually cross the wire");
            eprintln!("[showpiece] TRICK #8 LIVE: the cache line crossed the socket and the math never noticed.");
        }
        _ => {
            eprintln!("usage: sp_ring2_showpiece --serve <addr> | --decode <peer> <model.sp-model> <tok.sp-tokenizer>");
            std::process::exit(2);
        }
    }
}
