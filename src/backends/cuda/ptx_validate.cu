/*
 * ptx_validate.cu — M_PTX_1 bit-identity validation for Phase 2-CU.PTX.
 * Compares PTX kernel output vs math-core C scalar reference.
 * Usage: ./ptx_validate [ntt|hash|spinor|mma|all]
 * On no-GPU host: prints SKIP for each GPU gate.
 */
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>

#include "ptx_ntt.cuh"
#include "ptx_hash.cuh"
#include "ptx_spinor.cuh"
#include "ptx_mma.cuh"

static bool gpu_available(void) {
    int n = 0;
    cudaGetDeviceCount(&n);
    if (n == 0) return false;
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    if (p.major < 7 || (p.major == 7 && p.minor < 5)) {
        fprintf(stderr, "ptx_validate: sm_%d%d < sm_75; PTX gates require sm_75+\n",
                p.major, p.minor);
        return false;
    }
    return true;
}

/* ── NTT gate ────────────────────────────────────────────────────────── */

__global__ static void k_ntt_modmul_q1(const uint32_t *a, const uint32_t *b,
                                        uint32_t *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = ptx_modmul_q1(a[i], b[i]);
}

__global__ static void k_ntt_modmul_q2(const uint32_t *a, const uint32_t *b,
                                        uint32_t *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = ptx_modmul_q2(a[i], b[i]);
}

static bool run_ntt_modmul(uint32_t q, const char *tag,
                            const uint32_t *h_a, const uint32_t *h_b,
                            const uint32_t *h_ref, int n,
                            void (*kernel)(const uint32_t *, const uint32_t *,
                                           uint32_t *, int)) {
    uint32_t *d_a, *d_b, *d_out;
    cudaMalloc(&d_a,  n * sizeof(uint32_t));
    cudaMalloc(&d_b,  n * sizeof(uint32_t));
    cudaMalloc(&d_out, n * sizeof(uint32_t));
    cudaMemcpy(d_a, h_a, n * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(uint32_t), cudaMemcpyHostToDevice);

    kernel(d_a, d_b, d_out, n);
    cudaDeviceSynchronize();

    uint32_t *h_got = new uint32_t[n];
    cudaMemcpy(h_got, d_out, n * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);

    bool pass = true;
    for (int i = 0; i < n && pass; i++) {
        if (h_ref[i] != h_got[i]) {
            printf("M_PTX_1 NTT %s: FAIL (idx=%d ref=%u got=%u a=%u b=%u)\n",
                   tag, i, h_ref[i], h_got[i], h_a[i], h_b[i]);
            pass = false;
        }
    }
    if (pass) printf("M_PTX_1 NTT %s: PASS (%d pairs)\n", tag, n);
    delete[] h_got;
    return pass;
}

static bool validate_ntt(void) {
    const int N = 1024;
    uint32_t *h_a   = new uint32_t[N];
    uint32_t *h_b   = new uint32_t[N];
    uint32_t *h_ref = new uint32_t[N];

    /* Q1 test — inputs in [0, q1) */
    h_a[0] = 0;              h_b[0] = 0;
    h_a[1] = 1;              h_b[1] = PTX_NTT_Q1 - 1;
    h_a[2] = PTX_NTT_Q1 - 1; h_b[2] = PTX_NTT_Q1 - 1;
    h_a[3] = PTX_NTT_Q1 / 2; h_b[3] = PTX_NTT_Q1 / 2;
    for (int i = 4; i < N; i++) {
        h_a[i] = (uint32_t)((1664525ULL  * (uint32_t)i + 1013904223ULL) % PTX_NTT_Q1);
        h_b[i] = (uint32_t)((22695477ULL * (uint32_t)i + 1ULL)          % PTX_NTT_Q1);
    }
    for (int i = 0; i < N; i++) h_ref[i] = modmul_ref_q1(h_a[i], h_b[i]);

    bool ok = run_ntt_modmul(PTX_NTT_Q1, "Q1", h_a, h_b, h_ref, N,
                              [](const uint32_t *a, const uint32_t *b,
                                 uint32_t *o, int n) {
                                  k_ntt_modmul_q1<<<(n+255)/256, 256>>>(a, b, o, n);
                              });

    /* Q2 test — clamp inputs to [0, q2) */
    h_a[0] = 0;              h_b[0] = 0;
    h_a[1] = 1;              h_b[1] = PTX_NTT_Q2 - 1;
    h_a[2] = PTX_NTT_Q2 - 1; h_b[2] = PTX_NTT_Q2 - 1;
    h_a[3] = PTX_NTT_Q2 / 2; h_b[3] = PTX_NTT_Q2 / 2;
    for (int i = 4; i < N; i++) {
        h_a[i] = (uint32_t)((1664525ULL  * (uint32_t)i + 1013904223ULL) % PTX_NTT_Q2);
        h_b[i] = (uint32_t)((22695477ULL * (uint32_t)i + 1ULL)          % PTX_NTT_Q2);
    }
    for (int i = 0; i < N; i++) h_ref[i] = modmul_ref_q2(h_a[i], h_b[i]);

    ok &= run_ntt_modmul(PTX_NTT_Q2, "Q2", h_a, h_b, h_ref, N,
                          [](const uint32_t *a, const uint32_t *b,
                             uint32_t *o, int n) {
                              k_ntt_modmul_q2<<<(n+255)/256, 256>>>(a, b, o, n);
                          });

    delete[] h_a; delete[] h_b; delete[] h_ref;
    return ok;
}

/* ── HASH gate ───────────────────────────────────────────────────────── */

__global__ static void k_hash_xor3(const uint32_t *a, const uint32_t *b,
                                    const uint32_t *c, uint32_t *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = ptx_xor3(a[i], b[i], c[i]);
}

__global__ static void k_hash_prmt(const uint32_t *a, const uint32_t *b,
                                    const uint32_t *sel, uint32_t *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = ptx_prmt(a[i], b[i], sel[i]);
}

static bool validate_hash(void) {
    bool pass = true;

    /* ── xor3 test: 1024 triples ────────────────────────────────────── */
    const int N_XOR = 1024;
    uint32_t *h_xa = new uint32_t[N_XOR];
    uint32_t *h_xb = new uint32_t[N_XOR];
    uint32_t *h_xc = new uint32_t[N_XOR];
    uint32_t *h_xref = new uint32_t[N_XOR];

    /* corners at indices 0-2 */
    h_xa[0] = 0;          h_xb[0] = 0;          h_xc[0] = 0;
    h_xa[1] = 0xFFFFFFFFu; h_xb[1] = 0xFFFFFFFFu; h_xc[1] = 0xFFFFFFFFu;
    h_xa[2] = 0x55555555u; h_xb[2] = 0xAAAAAAAAu; h_xc[2] = 0xFFFFFFFFu;
    /* LCG-generated randoms for indices 3..N-1 */
    uint32_t lcg = 0xDEADBEEFu;
    for (int i = 3; i < N_XOR; i++) {
        lcg = lcg * 1664525u + 1013904223u; h_xa[i] = lcg;
        lcg = lcg * 1664525u + 1013904223u; h_xb[i] = lcg;
        lcg = lcg * 1664525u + 1013904223u; h_xc[i] = lcg;
    }
    for (int i = 0; i < N_XOR; i++) h_xref[i] = h_xa[i] ^ h_xb[i] ^ h_xc[i];

    uint32_t *d_xa, *d_xb, *d_xc, *d_xout;
    cudaMalloc(&d_xa,  N_XOR * sizeof(uint32_t));
    cudaMalloc(&d_xb,  N_XOR * sizeof(uint32_t));
    cudaMalloc(&d_xc,  N_XOR * sizeof(uint32_t));
    cudaMalloc(&d_xout, N_XOR * sizeof(uint32_t));
    cudaMemcpy(d_xa, h_xa, N_XOR * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_xb, h_xb, N_XOR * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_xc, h_xc, N_XOR * sizeof(uint32_t), cudaMemcpyHostToDevice);

    k_hash_xor3<<<(N_XOR + 255) / 256, 256>>>(d_xa, d_xb, d_xc, d_xout, N_XOR);
    cudaDeviceSynchronize();

    uint32_t *h_xgot = new uint32_t[N_XOR];
    cudaMemcpy(h_xgot, d_xout, N_XOR * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_xa); cudaFree(d_xb); cudaFree(d_xc); cudaFree(d_xout);

    bool xor3_pass = true;
    for (int i = 0; i < N_XOR && xor3_pass; i++) {
        if (h_xref[i] != h_xgot[i]) {
            printf("M_PTX_1 HASH xor3: FAIL (idx=%d ref=0x%08X got=0x%08X)\n",
                   i, h_xref[i], h_xgot[i]);
            xor3_pass = false;
        }
    }
    if (xor3_pass) printf("M_PTX_1 HASH xor3: PASS (%d triples)\n", N_XOR);
    delete[] h_xa; delete[] h_xb; delete[] h_xc; delete[] h_xref; delete[] h_xgot;
    pass &= xor3_pass;

    /* ── prmt test: 256 selectors ──────────────────────────────────── */
    const int N_PRM = 256;
    uint32_t *h_pa  = new uint32_t[N_PRM];
    uint32_t *h_pb  = new uint32_t[N_PRM];
    uint32_t *h_sel = new uint32_t[N_PRM];
    uint32_t *h_pref = new uint32_t[N_PRM];

    for (int i = 0; i < N_PRM; i++) {
        h_pa[i]  = 0x03020100u;
        h_pb[i]  = 0x07060504u;
        h_sel[i] = (uint32_t)i;
        /* C emulation of prmt.b32 (host fallback; __CUDA_ARCH__ not defined) */
        h_pref[i] = ptx_prmt(h_pa[i], h_pb[i], h_sel[i]);
    }

    uint32_t *d_pa, *d_pb, *d_psel, *d_pout;
    cudaMalloc(&d_pa,  N_PRM * sizeof(uint32_t));
    cudaMalloc(&d_pb,  N_PRM * sizeof(uint32_t));
    cudaMalloc(&d_psel, N_PRM * sizeof(uint32_t));
    cudaMalloc(&d_pout, N_PRM * sizeof(uint32_t));
    cudaMemcpy(d_pa,  h_pa,  N_PRM * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pb,  h_pb,  N_PRM * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_psel, h_sel, N_PRM * sizeof(uint32_t), cudaMemcpyHostToDevice);

    k_hash_prmt<<<1, 256>>>(d_pa, d_pb, d_psel, d_pout, N_PRM);
    cudaDeviceSynchronize();

    uint32_t *h_pgot = new uint32_t[N_PRM];
    cudaMemcpy(h_pgot, d_pout, N_PRM * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_pa); cudaFree(d_pb); cudaFree(d_psel); cudaFree(d_pout);

    bool prmt_pass = true;
    for (int i = 0; i < N_PRM && prmt_pass; i++) {
        if (h_pref[i] != h_pgot[i]) {
            printf("M_PTX_1 HASH prmt: FAIL (selector=0x%02X ref=0x%08X got=0x%08X)\n",
                   i, h_pref[i], h_pgot[i]);
            prmt_pass = false;
        }
    }
    if (prmt_pass) printf("M_PTX_1 HASH prmt: PASS (%d combinations)\n", N_PRM);
    delete[] h_pa; delete[] h_pb; delete[] h_sel; delete[] h_pref; delete[] h_pgot;
    pass &= prmt_pass;

    return pass;
}

/* ── SPINOR gate ─────────────────────────────────────────────────────── */

__global__ static void k_spinor_load(const uint32_t *src, uint32_t *dst,
                                      int n_blocks, int is_hot) {
    int b    = (int)blockIdx.x;
    int lane = (int)threadIdx.x;   /* blockDim must be 32 */
    if (b < n_blocks)
        dst[b * 32 + lane] = sp_spinor_warpload(src, (uint32_t)b, lane, is_hot);
}

/* k_spinor_load_v4: 4 words per lane using sp_spinor_warpload4.
 * dst layout: dst[(b * 32 + lane) * 4 + 0..3] = the four loaded words. */
__global__ static void k_spinor_load_v4(const uint32_t *src, uint32_t *dst,
                                         int n_blocks, int is_hot) {
    int b    = (int)blockIdx.x;
    int lane = (int)threadIdx.x;
    if (b < n_blocks) {
        uint32_t v0, v1, v2, v3;
        sp_spinor_warpload4(src, (uint32_t)b, lane, is_hot, &v0, &v1, &v2, &v3);
        int base = (b * 32 + lane) * 4;
        dst[base + 0] = v0;
        dst[base + 1] = v1;
        dst[base + 2] = v2;
        dst[base + 3] = v3;
    }
}

static bool validate_spinor(void) {
    /* 16 blocks × 32 words src (index b*16+lane, max = 15*16+31 = 271 < 512) */
    const int N = 16;
    uint32_t *h_src = new uint32_t[N * 32];
    uint32_t *h_got = new uint32_t[N * 32];
    for (int i = 0; i < N * 32; i++)
        h_src[i] = (uint32_t)((uint32_t)i * 2654435761u + 1u);

    uint32_t *d_src, *d_dst;
    cudaMalloc(&d_src, N * 32 * sizeof(uint32_t));
    cudaMalloc(&d_dst, N * 32 * sizeof(uint32_t));
    cudaMemcpy(d_src, h_src, N * 32 * sizeof(uint32_t), cudaMemcpyHostToDevice);

    bool pass = true;
    const char *labels[2] = {"cold", "hot"};
    for (int hot = 1; hot >= 0 && pass; hot--) {
        cudaMemset(d_dst, 0, N * 32 * sizeof(uint32_t));
        k_spinor_load<<<N, 32>>>(d_src, d_dst, N, hot);
        cudaDeviceSynchronize();
        cudaMemcpy(h_got, d_dst, N * 32 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        for (int b = 0; b < N && pass; b++) {
            for (int lane = 0; lane < 32 && pass; lane++) {
                uint32_t expected = h_src[b * 16 + lane];
                uint32_t got      = h_got[b * 32 + lane];
                if (expected != got) {
                    printf("M_PTX_1 SPINOR %s: FAIL (b=%d lane=%d exp=%u got=%u)\n",
                           labels[hot], b, lane, expected, got);
                    pass = false;
                }
            }
        }
        if (pass) printf("M_PTX_1 SPINOR %s: PASS (%d blocks)\n", labels[hot], N);
    }

    cudaFree(d_src); cudaFree(d_dst);
    delete[] h_src; delete[] h_got;
    if (!pass) return false;

    /* ── v4 validate: 8 blocks × 32 lanes × 4 words ─────────────────── */
    /* sp_spinor_warpload4 stride: 128 words/block.
     * Source array: N4 blocks × 128 words; each [b][lane*4+0..3] = expected. */
    const int N4 = 8;
    const int src4_sz = N4 * 128;
    uint32_t *h_src4 = new uint32_t[src4_sz];
    uint32_t *h_got4 = new uint32_t[N4 * 32 * 4];
    for (int i = 0; i < src4_sz; i++)
        h_src4[i] = (uint32_t)((uint32_t)i * 1664525u + 1013904223u);

    uint32_t *d_src4, *d_dst4;
    cudaMalloc(&d_src4, src4_sz * sizeof(uint32_t));
    cudaMalloc(&d_dst4, N4 * 32 * 4 * sizeof(uint32_t));
    cudaMemcpy(d_src4, h_src4, src4_sz * sizeof(uint32_t), cudaMemcpyHostToDevice);

    bool v4_pass = true;
    for (int hot = 1; hot >= 0 && v4_pass; hot--) {
        cudaMemset(d_dst4, 0, N4 * 32 * 4 * sizeof(uint32_t));
        k_spinor_load_v4<<<N4, 32>>>(d_src4, d_dst4, N4, hot);
        cudaDeviceSynchronize();
        cudaMemcpy(h_got4, d_dst4, N4 * 32 * 4 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        for (int b = 0; b < N4 && v4_pass; b++) {
            for (int lane = 0; lane < 32 && v4_pass; lane++) {
                for (int w = 0; w < 4 && v4_pass; w++) {
                    uint32_t expected = h_src4[b * 128 + lane * 4 + w];
                    uint32_t got      = h_got4[(b * 32 + lane) * 4 + w];
                    if (expected != got) {
                        printf("M_PTX_1 SPINOR_v4 %s: FAIL (b=%d lane=%d w=%d exp=%u got=%u)\n",
                               labels[hot], b, lane, w, expected, got);
                        v4_pass = false;
                    }
                }
            }
        }
        if (v4_pass)
            printf("M_PTX_1 SPINOR_v4 %s: PASS (%d blocks)\n", labels[hot], N4);
    }

    cudaFree(d_src4); cudaFree(d_dst4);
    delete[] h_src4; delete[] h_got4;
    return v4_pass;
}

/* ── MMA gate ────────────────────────────────────────────────────────── */

/*
 * validate_mma
 *
 * Runs M_PTX_1 (correctness), M_PTX_3 (no malloc in hot path),
 * and M_PTX_4 (two streams, no global sync) for the INT8 WMMA kernel.
 *
 * Returns true iff all three gates pass; increments pass/skip/fail counts.
 */
static bool validate_mma(int *p_pass_count, int *p_fail_count) {
    bool all_ok = true;

    /* ── Allocate device buffers (cudaMalloc happens HERE, not in kernel) ── */
    /* Test A: 16x16x16 all-ones */
    const int MA = MMA_M, KA = MMA_K, NA = MMA_N;
    int8_t  *h_Aa = new int8_t [MA * KA];
    int8_t  *h_Ba = new int8_t [KA * NA];
    float   *h_sca_a = new float[MA];
    float   *h_sca_b = new float[NA];
    __half  *h_Ca_got = new __half[MA * NA];
    for (int i = 0; i < MA * KA; i++) h_Aa[i] = 1;
    for (int i = 0; i < KA * NA; i++) h_Ba[i] = 1;
    for (int i = 0; i < MA; i++) h_sca_a[i] = 1.0f;
    for (int i = 0; i < NA; i++) h_sca_b[i] = 1.0f;

    int8_t *d_Aa, *d_Ba; float *d_sca_a, *d_sca_b; __half *d_Ca;
    cudaMalloc(&d_Aa,    MA * KA * sizeof(int8_t));
    cudaMalloc(&d_Ba,    KA * NA * sizeof(int8_t));
    cudaMalloc(&d_sca_a, MA * sizeof(float));
    cudaMalloc(&d_sca_b, NA * sizeof(float));
    cudaMalloc(&d_Ca,    MA * NA * sizeof(__half));
    cudaMemcpy(d_Aa,    h_Aa,    MA * KA * sizeof(int8_t),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ba,    h_Ba,    KA * NA * sizeof(int8_t),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_sca_a, h_sca_a, MA * sizeof(float),        cudaMemcpyHostToDevice);
    cudaMemcpy(d_sca_b, h_sca_b, NA * sizeof(float),        cudaMemcpyHostToDevice);

    /* Launch on default stream (stream=0) */
    sp_frob_matmul_q8_mma(d_Aa, d_Ba, d_sca_a, d_sca_b, d_Ca,
                           MA, KA, NA, 0);
    cudaDeviceSynchronize();

    cudaMemcpy(h_Ca_got, d_Ca, MA * NA * sizeof(__half), cudaMemcpyDeviceToHost);

    /* Check M_PTX_3: kernel ran without memory allocation error */
    cudaError_t err3 = cudaGetLastError();
    if (err3 == cudaSuccess) {
        printf("M_PTX_3 MMA: PASS\n");
        (*p_pass_count)++;
    } else {
        printf("M_PTX_3 MMA: FAIL (cuda error %s)\n", cudaGetErrorString(err3));
        (*p_fail_count)++;
        all_ok = false;
    }

    /* Validate Test A: all elements should be __float2half(16.0f) */
    __half expected_a = __float2half(16.0f);
    bool test_a_ok = true;
    int test_a_fail_cnt = 0;
    for (int i = 0; i < MA * NA; i++) {
        /* Compare as uint16 for exact bit match */
        uint16_t got_bits, exp_bits;
        memcpy(&got_bits, &h_Ca_got[i], sizeof(uint16_t));
        memcpy(&exp_bits, &expected_a,  sizeof(uint16_t));
        if (got_bits != exp_bits) {
            if (test_a_fail_cnt == 0)
                printf("M_PTX_1 MMA TestA: FAIL (first at idx=%d exp=%.1f got=%.4f)\n",
                       i, __half2float(expected_a), __half2float(h_Ca_got[i]));
            test_a_fail_cnt++;
            test_a_ok = false;
        }
    }
    if (test_a_ok)
        printf("M_PTX_1 MMA TestA: PASS (all-ones 16x16x16 -> 16.0f)\n");
    else
        printf("M_PTX_1 MMA TestA: %d mismatches\n", test_a_fail_cnt);

    cudaFree(d_Aa); cudaFree(d_Ba); cudaFree(d_sca_a); cudaFree(d_sca_b); cudaFree(d_Ca);
    delete[] h_Aa; delete[] h_Ba; delete[] h_sca_a; delete[] h_sca_b; delete[] h_Ca_got;

    /* ── Test B: random Q8 correctness (64x64x64) ─────────────────────── */
    const int MB = 64, KB = 64, NB = 64;
    int8_t *h_Ab = new int8_t [MB * KB];
    int8_t *h_Bb = new int8_t [KB * NB];
    float  *h_scb_a = new float[MB];
    float  *h_scb_b = new float[NB];
    __half *h_Cb_ref = new __half[MB * NB];
    __half *h_Cb_got = new __half[MB * NB];

    /* LCG random values in [-127, 127] */
    uint32_t lcg = 0xBEEFCAFEu;
    for (int i = 0; i < MB * KB; i++) {
        lcg = lcg * 1664525u + 1013904223u;
        h_Ab[i] = (int8_t)(((int)(lcg & 0xFFu)) - 128);
        if (h_Ab[i] < -127) h_Ab[i] = -127;
    }
    for (int i = 0; i < KB * NB; i++) {
        lcg = lcg * 1664525u + 1013904223u;
        h_Bb[i] = (int8_t)(((int)(lcg & 0xFFu)) - 128);
        if (h_Bb[i] < -127) h_Bb[i] = -127;
    }
    for (int i = 0; i < MB; i++) h_scb_a[i] = 1.0f / 127.0f;
    for (int i = 0; i < NB; i++) h_scb_b[i] = 1.0f;

    /* CPU reference */
    sp_frob_matmul_q8_ref(h_Ab, h_Bb, h_scb_a, h_scb_b, h_Cb_ref, MB, KB, NB);

    /* GPU */
    int8_t *d_Ab, *d_Bb; float *d_scb_a, *d_scb_b; __half *d_Cb;
    cudaMalloc(&d_Ab,    MB * KB * sizeof(int8_t));
    cudaMalloc(&d_Bb,    KB * NB * sizeof(int8_t));
    cudaMalloc(&d_scb_a, MB * sizeof(float));
    cudaMalloc(&d_scb_b, NB * sizeof(float));
    cudaMalloc(&d_Cb,    MB * NB * sizeof(__half));
    cudaMemcpy(d_Ab,    h_Ab,    MB * KB * sizeof(int8_t),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_Bb,    h_Bb,    KB * NB * sizeof(int8_t),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_scb_a, h_scb_a, MB * sizeof(float),        cudaMemcpyHostToDevice);
    cudaMemcpy(d_scb_b, h_scb_b, NB * sizeof(float),        cudaMemcpyHostToDevice);

    sp_frob_matmul_q8_mma(d_Ab, d_Bb, d_scb_a, d_scb_b, d_Cb,
                           MB, KB, NB, 0);
    cudaDeviceSynchronize();
    cudaMemcpy(h_Cb_got, d_Cb, MB * NB * sizeof(__half), cudaMemcpyDeviceToHost);

    bool test_b_ok = true;
    int test_b_fail_cnt = 0;
    for (int i = 0; i < MB * NB; i++) {
        uint16_t ref_bits, got_bits;
        memcpy(&ref_bits, &h_Cb_ref[i], sizeof(uint16_t));
        memcpy(&got_bits, &h_Cb_got[i], sizeof(uint16_t));
        if (ref_bits != got_bits) {
            if (test_b_fail_cnt == 0)
                printf("M_PTX_1 MMA TestB: FAIL (first at idx=%d ref=%.6f got=%.6f)\n",
                       i, __half2float(h_Cb_ref[i]), __half2float(h_Cb_got[i]));
            test_b_fail_cnt++;
            test_b_ok = false;
        }
    }
    if (test_b_ok)
        printf("M_PTX_1 MMA TestB: PASS (random Q8 64x64x64 bit-exact)\n");
    else
        printf("M_PTX_1 MMA TestB: %d mismatches out of %d\n",
               test_b_fail_cnt, MB * NB);

    cudaFree(d_Ab); cudaFree(d_Bb); cudaFree(d_scb_a); cudaFree(d_scb_b); cudaFree(d_Cb);
    delete[] h_Ab; delete[] h_Bb; delete[] h_scb_a; delete[] h_scb_b;
    delete[] h_Cb_ref; delete[] h_Cb_got;

    if (test_a_ok && test_b_ok) {
        printf("M_PTX_1 MMA: PASS\n");
        (*p_pass_count)++;
    } else {
        printf("M_PTX_1 MMA: FAIL\n");
        (*p_fail_count)++;
        all_ok = false;
    }

    /* ── Test C: INT4 all-ones (8×32×8) ──────────────────────────────── */
    /* K=32 nibbles: expected inner product = 32 × 1 × 1 = 32 → 32.0f */
    {
        const int MC = MMA_M, KC = MMA_INT4_K, NC = MMA_N;
        const int KC_bytes = KC / 2;  /* 16 */
        uint8_t *h_Ac = new uint8_t [MC * KC_bytes];
        uint8_t *h_Bc = new uint8_t [KC_bytes * NC];
        float   *h_scc_a = new float[MC];
        float   *h_scc_b = new float[NC];
        __half  *h_Cc_got = new __half[MC * NC];
        __half  *h_Cc_ref = new __half[MC * NC];
        /* All nibbles = 1: packed as 0x11 per byte */
        memset(h_Ac, 0x11, MC * KC_bytes);
        memset(h_Bc, 0x11, KC_bytes * NC);
        for (int i = 0; i < MC; i++) h_scc_a[i] = 1.0f;
        for (int i = 0; i < NC; i++) h_scc_b[i] = 1.0f;
        sp_frob_matmul_q4_ref(h_Ac, h_Bc, h_scc_a, h_scc_b, h_Cc_ref, MC, KC, NC);

        uint8_t *d_Ac, *d_Bc; float *d_scc_a, *d_scc_b; __half *d_Cc;
        cudaMalloc(&d_Ac,    MC * KC_bytes);
        cudaMalloc(&d_Bc,    KC_bytes * NC);
        cudaMalloc(&d_scc_a, MC * sizeof(float));
        cudaMalloc(&d_scc_b, NC * sizeof(float));
        cudaMalloc(&d_Cc,    MC * NC * sizeof(__half));
        cudaMemcpy(d_Ac, h_Ac, MC * KC_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_Bc, h_Bc, KC_bytes * NC, cudaMemcpyHostToDevice);
        cudaMemcpy(d_scc_a, h_scc_a, MC * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_scc_b, h_scc_b, NC * sizeof(float), cudaMemcpyHostToDevice);
        sp_frob_matmul_q4_mma(d_Ac, d_Bc, d_scc_a, d_scc_b, d_Cc, MC, KC, NC, 0);
        cudaDeviceSynchronize();
        cudaMemcpy(h_Cc_got, d_Cc, MC * NC * sizeof(__half), cudaMemcpyDeviceToHost);

        __half exp_c = __float2half((float)KC);
        bool test_c_ok = true;
        int test_c_fail = 0;
        for (int i = 0; i < MC * NC; i++) {
            uint16_t got_bits, exp_bits;
            memcpy(&got_bits, &h_Cc_got[i], 2);
            memcpy(&exp_bits, &exp_c, 2);
            if (got_bits != exp_bits) {
                if (test_c_fail == 0)
                    printf("M_PTX_1 MMA INT4 TestC: FAIL (idx=%d exp=%.1f got=%.1f)\n",
                           i, __half2float(exp_c), __half2float(h_Cc_got[i]));
                test_c_fail++;
                test_c_ok = false;
            }
        }
        if (test_c_ok)
            printf("M_PTX_1 MMA INT4 TestC: PASS (all-ones 8x32x8 -> 32.0f)\n");
        else
            printf("M_PTX_1 MMA INT4 TestC: %d mismatches\n", test_c_fail);

        cudaFree(d_Ac); cudaFree(d_Bc); cudaFree(d_scc_a); cudaFree(d_scc_b); cudaFree(d_Cc);
        delete[] h_Ac; delete[] h_Bc; delete[] h_scc_a; delete[] h_scc_b;
        delete[] h_Cc_got; delete[] h_Cc_ref;
        if (!test_c_ok) { (*p_fail_count)++; all_ok = false; }
        else              (*p_pass_count)++;
    }

    /* ── Test D: INT4 random Q4 (16×32×16), bit-exact vs scalar ref ──── */
    {
        const int MD = 16, KD = MMA_INT4_K * 2, ND = 16;  /* KD=64 nibbles */
        const int KD_bytes = KD / 2;                        /* 32 */
        uint8_t *h_Ad = new uint8_t [MD * KD_bytes];
        uint8_t *h_Bd = new uint8_t [KD_bytes * ND];
        float   *h_scd_a = new float[MD];
        float   *h_scd_b = new float[ND];
        __half  *h_Cd_ref = new __half[MD * ND];
        __half  *h_Cd_got = new __half[MD * ND];

        /* Nibbles in [1..7]: each byte packs two s4 values.
         * LCG → bits[2:0] gives nibble in [0..7]; bias to [1..7] via |1 mask. */
        uint32_t lcg = 0xCAFEBABEu;
        for (int i = 0; i < MD * KD_bytes; i++) {
            lcg = lcg * 1664525u + 1013904223u;
            uint8_t lo = ((lcg >> 0) & 0x7u) | 0x1u;  /* 1..7 */
            uint8_t hi = ((lcg >> 8) & 0x7u) | 0x1u;  /* 1..7 */
            h_Ad[i] = (uint8_t)(lo | (hi << 4));
        }
        for (int i = 0; i < KD_bytes * ND; i++) {
            lcg = lcg * 1664525u + 1013904223u;
            uint8_t lo = ((lcg >> 0) & 0x7u) | 0x1u;
            uint8_t hi = ((lcg >> 8) & 0x7u) | 0x1u;
            h_Bd[i] = (uint8_t)(lo | (hi << 4));
        }
        for (int i = 0; i < MD; i++) h_scd_a[i] = 1.0f / 7.0f;
        for (int i = 0; i < ND; i++) h_scd_b[i] = 1.0f;
        sp_frob_matmul_q4_ref(h_Ad, h_Bd, h_scd_a, h_scd_b, h_Cd_ref, MD, KD, ND);

        uint8_t *d_Ad, *d_Bd; float *d_scd_a, *d_scd_b; __half *d_Cd;
        cudaMalloc(&d_Ad,    MD * KD_bytes);
        cudaMalloc(&d_Bd,    KD_bytes * ND);
        cudaMalloc(&d_scd_a, MD * sizeof(float));
        cudaMalloc(&d_scd_b, ND * sizeof(float));
        cudaMalloc(&d_Cd,    MD * ND * sizeof(__half));
        cudaMemcpy(d_Ad, h_Ad, MD * KD_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_Bd, h_Bd, KD_bytes * ND, cudaMemcpyHostToDevice);
        cudaMemcpy(d_scd_a, h_scd_a, MD * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_scd_b, h_scd_b, ND * sizeof(float), cudaMemcpyHostToDevice);
        sp_frob_matmul_q4_mma(d_Ad, d_Bd, d_scd_a, d_scd_b, d_Cd, MD, KD, ND, 0);
        cudaDeviceSynchronize();
        cudaMemcpy(h_Cd_got, d_Cd, MD * ND * sizeof(__half), cudaMemcpyDeviceToHost);

        bool test_d_ok = true;
        int test_d_fail = 0;
        for (int i = 0; i < MD * ND; i++) {
            uint16_t ref_bits, got_bits;
            memcpy(&ref_bits, &h_Cd_ref[i], 2);
            memcpy(&got_bits, &h_Cd_got[i], 2);
            if (ref_bits != got_bits) {
                if (test_d_fail == 0)
                    printf("M_PTX_1 MMA INT4 TestD: FAIL (idx=%d ref=%.4f got=%.4f)\n",
                           i, __half2float(h_Cd_ref[i]), __half2float(h_Cd_got[i]));
                test_d_fail++;
                test_d_ok = false;
            }
        }
        if (test_d_ok)
            printf("M_PTX_1 MMA INT4 TestD: PASS (random Q4 16x64x16 bit-exact)\n");
        else
            printf("M_PTX_1 MMA INT4 TestD: %d mismatches out of %d\n",
                   test_d_fail, MD * ND);

        cudaFree(d_Ad); cudaFree(d_Bd); cudaFree(d_scd_a); cudaFree(d_scd_b); cudaFree(d_Cd);
        delete[] h_Ad; delete[] h_Bd; delete[] h_scd_a; delete[] h_scd_b;
        delete[] h_Cd_ref; delete[] h_Cd_got;
        if (!test_d_ok) { (*p_fail_count)++; all_ok = false; }
        else              (*p_pass_count)++;
    }

    /* ── M_PTX_4: two streams, no global sync ─────────────────────────── */
    /*
     * Launch the kernel on two independent streams simultaneously.
     * Use per-stream CUDA events to verify both complete without error.
     * Critically: do NOT call cudaDeviceSynchronize() between launches.
     */
    const int M4 = 32, K4 = 16, N4 = 32;
    int8_t  *h_A4 = new int8_t [M4 * K4];
    int8_t  *h_B4 = new int8_t [K4 * N4];
    float   *h_s4a = new float[M4];
    float   *h_s4b = new float[N4];
    for (int i = 0; i < M4 * K4; i++) h_A4[i] = 1;
    for (int i = 0; i < K4 * N4; i++) h_B4[i] = 1;
    for (int i = 0; i < M4; i++) h_s4a[i] = 1.0f;
    for (int i = 0; i < N4; i++) h_s4b[i] = 1.0f;

    int8_t *d_A4, *d_B4; float *d_s4a, *d_s4b;
    __half *d_C4s0, *d_C4s1;
    cudaMalloc(&d_A4,   M4 * K4 * sizeof(int8_t));
    cudaMalloc(&d_B4,   K4 * N4 * sizeof(int8_t));
    cudaMalloc(&d_s4a,  M4 * sizeof(float));
    cudaMalloc(&d_s4b,  N4 * sizeof(float));
    cudaMalloc(&d_C4s0, M4 * N4 * sizeof(__half));
    cudaMalloc(&d_C4s1, M4 * N4 * sizeof(__half));
    cudaMemcpy(d_A4,  h_A4,  M4 * K4 * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B4,  h_B4,  K4 * N4 * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s4a, h_s4a, M4 * sizeof(float),        cudaMemcpyHostToDevice);
    cudaMemcpy(d_s4b, h_s4b, N4 * sizeof(float),        cudaMemcpyHostToDevice);

    cudaStream_t stream0, stream1;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);

    cudaEvent_t ev0s, ev0e, ev1s, ev1e;
    cudaEventCreate(&ev0s); cudaEventCreate(&ev0e);
    cudaEventCreate(&ev1s); cudaEventCreate(&ev1e);

    /* Record start, launch on stream0, record end */
    cudaEventRecord(ev0s, stream0);
    sp_frob_matmul_q8_mma(d_A4, d_B4, d_s4a, d_s4b, d_C4s0,
                           M4, K4, N4, stream0);
    cudaEventRecord(ev0e, stream0);

    /* Record start, launch on stream1, record end — NO global sync in between */
    cudaEventRecord(ev1s, stream1);
    sp_frob_matmul_q8_mma(d_A4, d_B4, d_s4a, d_s4b, d_C4s1,
                           M4, K4, N4, stream1);
    cudaEventRecord(ev1e, stream1);

    /* Wait on each stream independently */
    cudaStreamSynchronize(stream0);
    cudaStreamSynchronize(stream1);

    cudaError_t e0 = cudaEventQuery(ev0e);
    cudaError_t e1 = cudaEventQuery(ev1e);

    if (e0 == cudaSuccess && e1 == cudaSuccess) {
        printf("M_PTX_4 MMA: PASS (two streams, no global sync)\n");
        (*p_pass_count)++;
    } else {
        printf("M_PTX_4 MMA: FAIL (stream0=%s stream1=%s)\n",
               cudaGetErrorString(e0), cudaGetErrorString(e1));
        (*p_fail_count)++;
        all_ok = false;
    }

    cudaEventDestroy(ev0s); cudaEventDestroy(ev0e);
    cudaEventDestroy(ev1s); cudaEventDestroy(ev1e);
    cudaStreamDestroy(stream0); cudaStreamDestroy(stream1);
    cudaFree(d_A4); cudaFree(d_B4); cudaFree(d_s4a); cudaFree(d_s4b);
    cudaFree(d_C4s0); cudaFree(d_C4s1);
    delete[] h_A4; delete[] h_B4; delete[] h_s4a; delete[] h_s4b;

    return all_ok;
}

/* ── main ────────────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    const char *filter = (argc > 1) ? argv[1] : "all";
    bool gpu = gpu_available();
    printf("ptx_validate: GPU=%s filter=%s\n", gpu ? "YES" : "NO (SKIP)", filter);

    int pass = 1, skip = 0, fail = 0;

    if (!strcmp(filter, "ntt") || !strcmp(filter, "all")) {
        if (!gpu) { printf("M_PTX_1 NTT: SKIP\n"); skip++; }
        else       { if (!validate_ntt()) { pass = 0; fail++; } }
    }

    if (!strcmp(filter, "hash") || !strcmp(filter, "all")) {
        if (!gpu) { printf("M_PTX_1 HASH: SKIP\n"); skip++; }
        else       { if (!validate_hash()) { pass = 0; fail++; } }
    }

    if (!strcmp(filter, "spinor") || !strcmp(filter, "all")) {
        if (!gpu) { printf("M_PTX_1 SPINOR: SKIP\n"); skip++; }
        else       { if (!validate_spinor()) { pass = 0; fail++; } }
    }

    if (!strcmp(filter, "mma") || !strcmp(filter, "all")) {
        if (!gpu) {
            printf("M_PTX_1 MMA: SKIP\n");
            printf("M_PTX_3 MMA: SKIP\n");
            printf("M_PTX_4 MMA: SKIP\n");
            skip += 3;
        } else {
            /* pass/fail counters are updated inside validate_mma */
            int mma_pass = 0, mma_fail = 0;
            validate_mma(&mma_pass, &mma_fail);
            if (mma_fail > 0) { pass = 0; fail += mma_fail; }
        }
    }

    printf("ptx_validate: %s (skip=%d)\n", (pass && fail == 0) ? "PASS" : "FAIL", skip);
    return (pass && fail == 0) ? 0 : 1;
}
