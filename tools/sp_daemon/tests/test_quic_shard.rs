use sp_daemon::network::quic_shard::{ShardBlockHeader, ResidueBlock};

#[test]
fn header_is_64_bytes() {
    assert_eq!(std::mem::size_of::<ShardBlockHeader>(), 64);
}

#[test]
fn header_roundtrip() {
    use sp_daemon::network::quic_shard::{header_to_bytes, header_from_bytes};

    let h = ShardBlockHeader {
        seq_id: 0xDEAD_BEEF_CAFE_1234,
        token_pos: 77,
        layer_id: 3,
        prime_selector: 1,
        _pad: [0u8; 47],
    };
    let bytes = header_to_bytes(&h);
    let h2 = header_from_bytes(&bytes);
    assert_eq!(h2.seq_id, h.seq_id);
    assert_eq!(h2.token_pos, h.token_pos);
    assert_eq!(h2.layer_id, h.layer_id);
    assert_eq!(h2.prime_selector, h.prime_selector);
}

use sp_daemon::ntt_ffi::{ntt_crt_recombine, ntt_free, ntt_init, NttCtxHandle};

const Q1: u32 = 1073738753;
const Q2: u32 = 1073732609;

#[test]
fn ntt_ffi_scalar_reference() {
    const N: usize = 128;
    let q1: Vec<u32> = (0..N as u32).map(|i| i % Q1).collect();
    let q2: Vec<u32> = (0..N as u32).map(|i| i % Q2).collect();

    let out: Vec<i64> = unsafe {
        let ctx = ntt_init(N as u32);
        assert!(!ctx.is_null(), "ntt_init returned null for N=128");
        let mut v = vec![0i64; N];
        ntt_crt_recombine(ctx, q1.as_ptr(), q2.as_ptr(), v.as_mut_ptr());
        ntt_free(ctx);
        v
    };

    assert_eq!(out[0], 0, "coeff[0] must be 0");
    assert_eq!(out[1], 1, "coeff[1] must be 1");
    let m_half: i64 = 1152908312643096577_i64 / 2;
    for &c in &out {
        assert!(c.abs() <= m_half, "coefficient out of CRT range: {}", c);
    }
}

#[test]
fn ntt_ctx_handle_drop() {
    unsafe {
        let ctx = ntt_init(128);
        assert!(!ctx.is_null());
        let _handle = NttCtxHandle(ctx);
        // _handle drops here — ntt_free called exactly once
    }
}

// tls_configs_construct_without_panic is an inline lib test in quic_shard.rs
// (integration test path blocked by pre-existing probe linker issue — sp_model_load etc.)

// coordinator_binds_on_loopback is an inline lib test in quic_shard.rs (Task 5)
// (same probe linker issue blocks integration tests for QUIC endpoints)

// garner_loop_reconstructs_single_pair is an inline lib test in quic_shard.rs (Task 7)
// (same probe linker issue blocks integration tests for async QUIC endpoints)
