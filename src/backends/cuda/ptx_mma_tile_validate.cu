/*
 * ptx_mma_tile_validate.cu — M_PTX_MMA_TILE_1 three-way correctness sweep.
 *
 * Three-way bit-exact check per shape:
 *   1. sp_frob_matmul_q{8,4}_ref      (scalar host reference)   — B in [K][N]
 *   2. sp_frob_matmul_q{8,4}_mma      (single-instruction naive) — B in [K][N]
 *   3. sp_frob_matmul_q{8,4}_mma_tile (64x64 tiled kernel)       — B in [N][K]
 *
 * Shape sweep: (64,64,64), (256,256,256), (1024,1024,1024), (3072,8192,3072)
 * Exit code: 0 = all pass, 1 = any mismatch.
 */
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>

#include "ptx_mma_tile_int8.cuh"
#include "ptx_mma_tile_int4.cuh"

#ifndef __CUDACC__
int main() { printf("SKIP (not a CUDA build)\n"); return 0; }
#else

/* ── Test helpers ──────────────────────────────────────────────────────────── */

static uint32_t lcg(uint32_t &s) { s = s * 1664525u + 1013904223u; return s; }

static void fill_int8(int8_t *p, int n, uint32_t &seed) {
    for (int i = 0; i < n; i++) p[i] = (int8_t)((lcg(seed) & 0xFF) - 128);
}

/* Transpose B from [K][N] int8 to [N][K] int8 for the tile kernel. */
static void transpose_b_int8(const int8_t *src, int8_t *dst, int K, int N) {
    for (int k = 0; k < K; k++)
        for (int n = 0; n < N; n++)
            dst[n * K + k] = src[k * N + n];
}

/* Transpose Q4 packed B from [K_bytes][N] to [N][K_bytes] (byte-level). */
static void transpose_b_int4(const uint8_t *src, uint8_t *dst, int K_bytes, int N) {
    for (int k = 0; k < K_bytes; k++)
        for (int n = 0; n < N; n++)
            dst[n * K_bytes + k] = src[k * N + n];
}

static bool half_match(const __half *a, const __half *b, int n, float *max_diff) {
    *max_diff = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = fabsf(__half2float(a[i]) - __half2float(b[i]));
        if (diff > *max_diff) *max_diff = diff;
    }
    return *max_diff == 0.0f;
}

#define CUDA_CHECK(err) do { \
    cudaError_t _e = (err); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(_e)); \
        return 1; \
    } \
} while (0)

/* ── Per-shape INT8 test ───────────────────────────────────────────────────── */
static int test_int8(int M, int K, int N, const char *tag) {
    size_t szA = (size_t)M * K, szB = (size_t)K * N, szC = (size_t)M * N;

    int8_t  *hA = new int8_t[szA], *hB = new int8_t[szB];
    float   *hSA = new float[M], *hSB = new float[N];
    __half  *hRef = new __half[szC], *hNaive = new __half[szC], *hTile = new __half[szC];

    uint32_t seed = 0xDEAD0000u ^ (uint32_t)(M * 31 + K * 17 + N * 7);
    fill_int8(hA, (int)szA, seed);
    fill_int8(hB, (int)szB, seed);
    for (int i = 0; i < M; i++) hSA[i] = 0.5f + (lcg(seed) & 0xFFu) * (1.0f / 512.0f);
    for (int j = 0; j < N; j++) hSB[j] = 0.5f + (lcg(seed) & 0xFFu) * (1.0f / 512.0f);

    /* hB_NT: B transposed to [N][K] for the tile kernel */
    int8_t *hB_NT = new int8_t[szB];
    transpose_b_int8(hB, hB_NT, K, N);

    /* 1. Scalar host reference (reads B in [K][N]) */
    sp_frob_matmul_q8_ref(hA, hB, hSA, hSB, hRef, M, K, N);

    /* 2. Naive GPU kernel (reads B in [K][N]) */
    int8_t *dA, *dB, *dB_NT; float *dSA, *dSB; __half *dC;
    CUDA_CHECK(cudaMalloc(&dA,    szA * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&dB,    szB * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&dB_NT, szB * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&dSA,   M   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dSB,   N   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC,    szC * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(dA,    hA,    szA * sizeof(int8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB,    hB,    szB * sizeof(int8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB_NT, hB_NT, szB * sizeof(int8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dSA,   hSA,   M   * sizeof(float),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dSB,   hSB,   N   * sizeof(float),  cudaMemcpyHostToDevice));

    sp_frob_matmul_q8_mma(dA, dB, dSA, dSB, dC, M, K, N, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hNaive, dC, szC * sizeof(__half), cudaMemcpyDeviceToHost));

    /* 3. Tiled kernel (reads B_NT in [N][K]) */
    CUDA_CHECK(cudaMemset(dC, 0, szC * sizeof(__half)));
    CUDA_CHECK(sp_frob_matmul_q8_mma_tile(dA, dB_NT, dSA, dSB, dC, M, K, N, 0));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hTile, dC, szC * sizeof(__half), cudaMemcpyDeviceToHost));

    float d_ref_naive, d_ref_tile, d_naive_tile;
    bool ok_rn = half_match(hRef,   hNaive, (int)szC, &d_ref_naive);
    bool ok_rt = half_match(hRef,   hTile,  (int)szC, &d_ref_tile);
    bool ok_nt = half_match(hNaive, hTile,  (int)szC, &d_naive_tile);
    bool pass  = ok_rn && ok_rt && ok_nt;
    printf("  INT8 %s (%dx%dx%d): ref-naive=%.4g ref-tile=%.4g naive-tile=%.4g [%s]\n",
           tag, M, K, N, d_ref_naive, d_ref_tile, d_naive_tile, pass ? "PASS" : "FAIL");

    cudaFree(dA); cudaFree(dB); cudaFree(dB_NT);
    cudaFree(dSA); cudaFree(dSB); cudaFree(dC);
    delete[] hA; delete[] hB; delete[] hB_NT;
    delete[] hSA; delete[] hSB;
    delete[] hRef; delete[] hNaive; delete[] hTile;
    return pass ? 0 : 1;
}

/* ── Per-shape INT4 test ───────────────────────────────────────────────────── */
/*
 * K = nibble count (must be multiple of 2; ideally multiple of 64 for full tiles).
 * A_q4: uint8_t[M][K/2], B_q4: uint8_t[K/2][N].
 * Packing: byte b = packed[k>>1]; low nibble = even k, high nibble = odd k.
 * s4 values in [-7,7] (no clamping needed if source is already bounded).
 */
static void pack_row_a(const int8_t *row, uint8_t *dst, int K) {
    for (int k = 0; k < K; k += 2) {
        unsigned n0 = (unsigned)(row[k]   < 0 ? row[k]   + 16 : row[k])   & 0xFu;
        unsigned n1 = (unsigned)(row[k+1] < 0 ? row[k+1] + 16 : row[k+1]) & 0xFu;
        dst[k >> 1] = (uint8_t)(n0 | (n1 << 4));
    }
}

static void pack_b_kn(const int8_t *hB_int8, uint8_t *hBq4, int K, int N) {
    /* B[K][N] (int8) → B_q4[K/2][N] (packed nibbles, k-dimension packed) */
    for (int k = 0; k < K; k += 2) {
        for (int j = 0; j < N; j++) {
            unsigned n0 = (unsigned)(hB_int8[ k    * N + j] < 0 ?
                hB_int8[ k    * N + j] + 16 : hB_int8[ k    * N + j]) & 0xFu;
            unsigned n1 = (unsigned)(hB_int8[(k+1) * N + j] < 0 ?
                hB_int8[(k+1) * N + j] + 16 : hB_int8[(k+1) * N + j]) & 0xFu;
            hBq4[(k >> 1) * N + j] = (uint8_t)(n0 | (n1 << 4));
        }
    }
}

static void fill_int4_src(int8_t *p, int n, uint32_t &seed) {
    for (int i = 0; i < n; i++) {
        int v = (int)(lcg(seed) % 15) - 7;  /* -7..7 */
        p[i] = (int8_t)v;
    }
}

static int test_int4(int M, int K, int N, const char *tag) {
    /* K is nibble count; K_bytes is packed byte count */
    const int K_bytes = K / 2;
    size_t szA  = (size_t)M * K_bytes;
    size_t szB  = (size_t)K_bytes * N;
    size_t szC  = (size_t)M * N;

    int8_t  *hA_src = new int8_t[(size_t)M * K];
    int8_t  *hB_src = new int8_t[(size_t)K  * N];
    uint8_t *hAq4   = new uint8_t[szA];
    uint8_t *hBq4   = new uint8_t[szB];
    float   *hSA    = new float[M];
    float   *hSB    = new float[N];
    __half  *hRef   = new __half[szC];
    __half  *hNaive = new __half[szC];
    __half  *hTile  = new __half[szC];

    uint32_t seed = 0xBEEF0000u ^ (uint32_t)(M * 37 + K * 19 + N * 11);
    fill_int4_src(hA_src, M * K, seed);
    fill_int4_src(hB_src, K  * N, seed);
    for (int i = 0; i < M; i++) hSA[i] = 0.5f + (lcg(seed) & 0xFFu) * (1.0f / 512.0f);
    for (int j = 0; j < N; j++) hSB[j] = 0.5f + (lcg(seed) & 0xFFu) * (1.0f / 512.0f);

    /* Pack B in [K_bytes][N] for ref/naive; transpose to [N][K_bytes] for tile. */
    memset(hAq4, 0, szA); memset(hBq4, 0, szB);
    for (int i = 0; i < M; i++)
        pack_row_a(hA_src + i * K, hAq4 + i * K_bytes, K);
    pack_b_kn(hB_src, hBq4, K, N);

    uint8_t *hBq4_NT = new uint8_t[szB];
    transpose_b_int4(hBq4, hBq4_NT, K_bytes, N);

    /* 1. Scalar reference (reads B in [K_bytes][N]) */
    sp_frob_matmul_q4_ref(hAq4, hBq4, hSA, hSB, hRef, M, K, N);

    /* 2. Naive GPU kernel (reads B in [K_bytes][N]) */
    uint8_t *dAq4, *dBq4, *dBq4_NT; float *dSA, *dSB; __half *dC;
    CUDA_CHECK(cudaMalloc(&dAq4,    szA * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&dBq4,    szB * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&dBq4_NT, szB * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&dSA,     M   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dSB,     N   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC,      szC * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(dAq4,    hAq4,    szA * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBq4,    hBq4,    szB * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBq4_NT, hBq4_NT, szB * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dSA,     hSA,     M   * sizeof(float),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dSB,     hSB,     N   * sizeof(float),   cudaMemcpyHostToDevice));

    sp_frob_matmul_q4_mma(dAq4, dBq4, dSA, dSB, dC, M, K, N, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hNaive, dC, szC * sizeof(__half), cudaMemcpyDeviceToHost));

    /* 3. Tiled kernel (reads B_NT in [N][K_bytes]) */
    CUDA_CHECK(cudaMemset(dC, 0, szC * sizeof(__half)));
    CUDA_CHECK(sp_frob_matmul_q4_mma_tile(dAq4, dBq4_NT, dSA, dSB, dC, M, K, N, 0));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hTile, dC, szC * sizeof(__half), cudaMemcpyDeviceToHost));

    float d_rn, d_rt, d_nt;
    bool ok_rn = half_match(hRef,   hNaive, (int)szC, &d_rn);
    bool ok_rt = half_match(hRef,   hTile,  (int)szC, &d_rt);
    bool ok_nt = half_match(hNaive, hTile,  (int)szC, &d_nt);
    bool pass  = ok_rn && ok_rt && ok_nt;
    printf("  INT4 %s (%dx%dx%d): ref-naive=%.4g ref-tile=%.4g naive-tile=%.4g [%s]\n",
           tag, M, K, N, d_rn, d_rt, d_nt, pass ? "PASS" : "FAIL");

    cudaFree(dAq4); cudaFree(dBq4); cudaFree(dBq4_NT);
    cudaFree(dSA); cudaFree(dSB); cudaFree(dC);
    delete[] hA_src; delete[] hB_src; delete[] hAq4; delete[] hBq4; delete[] hBq4_NT;
    delete[] hSA; delete[] hSB; delete[] hRef; delete[] hNaive; delete[] hTile;
    return pass ? 0 : 1;
}

/* ── Main ──────────────────────────────────────────────────────────────────── */
int main() {
    int dev; if (cudaGetDevice(&dev) != cudaSuccess) { printf("SKIP (no GPU)\n"); return 0; }

    static const struct { int M, K, N; const char *tag; } shapes[] = {
        {  64,   64,   64, "tiny"    },
        { 256,  256,  256, "small"   },
        {1024, 1024, 1024, "medium"  },
        {3072, 8192, 3072, "qwen3-ffn"},
    };
    const int ns = (int)(sizeof(shapes) / sizeof(shapes[0]));

    int fail = 0;
    printf("=== M_PTX_MMA_TILE_1: INT8 ===\n");
    for (int i = 0; i < ns; i++) fail |= test_int8(shapes[i].M, shapes[i].K, shapes[i].N, shapes[i].tag);
    printf("=== M_PTX_MMA_TILE_1: INT4 ===\n");
    for (int i = 0; i < ns; i++) fail |= test_int4(shapes[i].M, shapes[i].K, shapes[i].N, shapes[i].tag);

    printf("\n%s\n", fail ? "FAIL" : "ALL PASS");
    return fail;
}

#endif /* __CUDACC__ */
