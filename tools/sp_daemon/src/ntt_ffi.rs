//! ntt_ffi.rs — Manual FFI bindings for sp_ntt_crt (Phase 6-NET).
//! Mirrors ntt_crt.h. sp_ntt_crt is already linked via build.rs MODULES.

/// Opaque NTT context (mirrors ntt_ctx in ntt_crt.h).
pub enum NttCtx {}

extern "C" {
    /// Allocate context for transform length N ∈ {128, 256, 512}. Returns NULL
    /// for invalid N.
    pub fn ntt_init(n: u32) -> *mut NttCtx;

    /// Free a context. ntt_free(NULL) is a no-op.
    pub fn ntt_free(ctx: *mut NttCtx);

    /// Garner CRT: N residue pairs (x1 mod q1, x2 mod q2) → N signed centered
    /// coefficients in (-M/2, M/2], written to `out`.
    pub fn ntt_crt_recombine(
        ctx: *const NttCtx,
        x1:  *const u32,
        x2:  *const u32,
        out: *mut i64,
    );

    /// NTT-domain pointwise multiply:
    /// out1[i] = a1[i]*b1[i] mod q1, out2[i] = a2[i]*b2[i] mod q2.
    pub fn ntt_pointwise_mul(
        ctx:  *const NttCtx,
        a1: *const u32, a2: *const u32,
        b1: *const u32, b2: *const u32,
        out1: *mut u32, out2: *mut u32,
    );
}

/// RAII wrapper for ntt_ctx: calls ntt_free on drop.
///
/// Safety: ntt_ctx is read-only after ntt_init (per ntt_crt.h) — safe to share
/// across threads as Arc<NttCtxHandle>.
pub struct NttCtxHandle(pub *mut NttCtx);

unsafe impl Send for NttCtxHandle {}
unsafe impl Sync for NttCtxHandle {}

impl Drop for NttCtxHandle {
    fn drop(&mut self) {
        unsafe { ntt_free(self.0); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
