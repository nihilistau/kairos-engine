/*
 * ptx_ntt.cuh — PTX GF(p) Barrett butterfly for Shannon-Prime NTT.
 *
 * Frozen primes: q1=1073738753, q2=1073732609 (30-bit Proth primes).
 * Matches ntt_crt.c::modmul() byte-for-byte on integer kernels (M_PTX_1).
 *
 * PTX strategy (no mul.wide.u32 paired-register output — nvcc unreliable):
 *   1. mul.lo.u32 + mul.hi.u32 for the 64-bit product a*b
 *   2. shf.r.wrap.b32 for 64-bit barrel shifts (>> 29, >> 31)
 *   3. sub.u32 for r = lo - qlo (correct: r < 2^32 so r_hi=0, uint32 wrap OK)
 *   4. C-level conditional subtracts (nvcc emits setp/@pred on device)
 *
 * Barrett parameters (Q_BITS=30 per ntt_crt.c):
 *   qhat = ((x >> 29) * mu) >> 31
 *   mu   = floor(2^60 / q)    -- verified via BigInteger arithmetic
 *
 * Each asm() block has its own operand index space starting at 0.
 */
#pragma once
#include <cstdint>

#define PTX_NTT_Q1   1073738753u
#define PTX_NTT_Q2   1073732609u
/* floor(2^60 / PTX_NTT_Q1) = 1073744895  (verified: 1073744895*1073738753 = 2^60 - 9430041) */
/* floor(2^60 / PTX_NTT_Q2) = 1073751039  (verified: 1073751039*1073732609 = 2^60 - 84916225) */
#define PTX_MU_Q1    ((uint32_t)1073744895u)
#define PTX_MU_Q2    ((uint32_t)1073751039u)

/* ── C scalar reference — host + device, matches ntt_crt.c exactly ───── */

__host__ __device__ __forceinline__
uint32_t barrett_reduce32_ref(uint64_t x, uint32_t q, uint32_t mu) {
    uint64_t qhat = ((x >> 29u) * (uint64_t)mu) >> 31u;
    uint64_t r    = x - qhat * (uint64_t)q;
    if (r >= (uint64_t)q) r -= (uint64_t)q;
    if (r >= (uint64_t)q) r -= (uint64_t)q;
    return (uint32_t)r;
}

__host__ __device__ __forceinline__
uint32_t modmul_ref_q1(uint32_t a, uint32_t b) {
    return barrett_reduce32_ref((uint64_t)a * (uint64_t)b, PTX_NTT_Q1, PTX_MU_Q1);
}

__host__ __device__ __forceinline__
uint32_t modmul_ref_q2(uint32_t a, uint32_t b) {
    return barrett_reduce32_ref((uint64_t)a * (uint64_t)b, PTX_NTT_Q2, PTX_MU_Q2);
}

/* ── PTX Barrett (device, sm_75+) ────────────────────────────────────── */

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750

/*
 * ptx_modmul_q1: a * b mod q1 via PTX inline asm.
 *
 * Bounds analysis (why sub.u32 is correct):
 *   x = a*b < q1^2 < 2^60  => hi = x>>32 < 2^28
 *   r = x - qhat*q  where  0 <= r < 3*q1 < 2^32
 *   r_hi = 0  =>  sub.u32(lo, qlo) gives correct r regardless of borrow
 */
__device__ __forceinline__
uint32_t ptx_modmul_q1(uint32_t a, uint32_t b) {
    uint32_t lo, hi;
    asm volatile("mul.lo.u32 %0, %1, %2;" : "=r"(lo) : "r"(a), "r"(b));
    asm volatile("mul.hi.u32 %0, %1, %2;" : "=r"(hi) : "r"(a), "r"(b));

    /* x >> 29: shf.r.wrap.b32(low_src=lo, high_src=hi, 29) = bits[60:29] of (hi:lo) */
    uint32_t sh_lo;
    asm volatile("shf.r.wrap.b32 %0, %1, %2, 29;" : "=r"(sh_lo) : "r"(lo), "r"(hi));

    /* sh_lo * mu_q1 (64-bit product; sh_lo < 2^31, mu < 2^31 => product < 2^62) */
    uint32_t mlo, mhi;
    asm volatile("mul.lo.u32 %0, %1, %2;" : "=r"(mlo) : "r"(sh_lo), "r"(PTX_MU_Q1));
    asm volatile("mul.hi.u32 %0, %1, %2;" : "=r"(mhi) : "r"(sh_lo), "r"(PTX_MU_Q1));

    /* qhat = (mhi:mlo) >> 31 (lower 32 bits; result < 2^31) */
    uint32_t qhat;
    asm volatile("shf.r.wrap.b32 %0, %1, %2, 31;" : "=r"(qhat) : "r"(mlo), "r"(mhi));

    /* r = lo - (qhat*q1 & 0xFFFFFFFF); r_hi=0 so uint32 wrap gives correct r */
    uint32_t qlo, rlo;
    asm volatile("mul.lo.u32 %0, %1, %2;" : "=r"(qlo) : "r"(qhat), "r"(PTX_NTT_Q1));
    asm volatile("sub.u32 %0, %1, %2;"    : "=r"(rlo) : "r"(lo),   "r"(qlo));

    /* Barrett bound: r < 3*q1; at most 2 conditional subtracts */
    uint32_t res = rlo;
    if (res >= PTX_NTT_Q1) res -= PTX_NTT_Q1;
    if (res >= PTX_NTT_Q1) res -= PTX_NTT_Q1;
    return res;
}

__device__ __forceinline__
uint32_t ptx_modmul_q2(uint32_t a, uint32_t b) {
    uint32_t lo, hi;
    asm volatile("mul.lo.u32 %0, %1, %2;" : "=r"(lo) : "r"(a), "r"(b));
    asm volatile("mul.hi.u32 %0, %1, %2;" : "=r"(hi) : "r"(a), "r"(b));

    uint32_t sh_lo;
    asm volatile("shf.r.wrap.b32 %0, %1, %2, 29;" : "=r"(sh_lo) : "r"(lo), "r"(hi));

    uint32_t mlo, mhi;
    asm volatile("mul.lo.u32 %0, %1, %2;" : "=r"(mlo) : "r"(sh_lo), "r"(PTX_MU_Q2));
    asm volatile("mul.hi.u32 %0, %1, %2;" : "=r"(mhi) : "r"(sh_lo), "r"(PTX_MU_Q2));

    uint32_t qhat;
    asm volatile("shf.r.wrap.b32 %0, %1, %2, 31;" : "=r"(qhat) : "r"(mlo), "r"(mhi));

    uint32_t qlo, rlo;
    asm volatile("mul.lo.u32 %0, %1, %2;" : "=r"(qlo) : "r"(qhat), "r"(PTX_NTT_Q2));
    asm volatile("sub.u32 %0, %1, %2;"    : "=r"(rlo) : "r"(lo),   "r"(qlo));

    uint32_t res = rlo;
    if (res >= PTX_NTT_Q2) res -= PTX_NTT_Q2;
    if (res >= PTX_NTT_Q2) res -= PTX_NTT_Q2;
    return res;
}

/* NTT Cooley-Tukey butterfly: (a, b) -> (a + w*b, a - w*b) mod q */
__device__ __forceinline__
void ptx_butterfly_q1(uint32_t *a, uint32_t *b, uint32_t w) {
    uint32_t wb = ptx_modmul_q1(*b, w);
    uint32_t t  = *a + wb;              if (t >= PTX_NTT_Q1) t -= PTX_NTT_Q1;
    uint32_t u  = *a + PTX_NTT_Q1 - wb; if (u >= PTX_NTT_Q1) u -= PTX_NTT_Q1;
    *a = t; *b = u;
}

__device__ __forceinline__
void ptx_butterfly_q2(uint32_t *a, uint32_t *b, uint32_t w) {
    uint32_t wb = ptx_modmul_q2(*b, w);
    uint32_t t  = *a + wb;              if (t >= PTX_NTT_Q2) t -= PTX_NTT_Q2;
    uint32_t u  = *a + PTX_NTT_Q2 - wb; if (u >= PTX_NTT_Q2) u -= PTX_NTT_Q2;
    *a = t; *b = u;
}

#else  /* host / sm < 75: C fallback — same arithmetic, no PTX */

__host__ __device__ __forceinline__
uint32_t ptx_modmul_q1(uint32_t a, uint32_t b) { return modmul_ref_q1(a, b); }

__host__ __device__ __forceinline__
uint32_t ptx_modmul_q2(uint32_t a, uint32_t b) { return modmul_ref_q2(a, b); }

__host__ __device__ __forceinline__
void ptx_butterfly_q1(uint32_t *a, uint32_t *b, uint32_t w) {
    uint32_t wb = modmul_ref_q1(*b, w);
    uint32_t t  = *a + wb;               if (t >= PTX_NTT_Q1) t -= PTX_NTT_Q1;
    uint32_t u  = *a + PTX_NTT_Q1 - wb;  if (u >= PTX_NTT_Q1) u -= PTX_NTT_Q1;
    *a = t; *b = u;
}

__host__ __device__ __forceinline__
void ptx_butterfly_q2(uint32_t *a, uint32_t *b, uint32_t w) {
    uint32_t wb = modmul_ref_q2(*b, w);
    uint32_t t  = *a + wb;               if (t >= PTX_NTT_Q2) t -= PTX_NTT_Q2;
    uint32_t u  = *a + PTX_NTT_Q2 - wb;  if (u >= PTX_NTT_Q2) u -= PTX_NTT_Q2;
    *a = t; *b = u;
}

#endif  /* __CUDA_ARCH__ >= 750 */
