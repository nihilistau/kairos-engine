/*
 * ptx_mma_tile_bench.cu — M_PTX_MMA_TILE_2 throughput benchmark.
 *
 * Shape: (M,N,K) = (3072, 3072, 8192) — Qwen3-0.6B FFN prefill shape.
 * REPS: 5 warm-up (discarded) + 50 measured.
 * Reports: median / p90 / p99 ms and speedup vs cuBLAS HGEMM baseline.
 *
 * B layout: cuBLAS uses dB[K][N] (FP16 dequant); PTX tile uses dB_NT[N][K]
 * (pre-swizzled INT8) and dBq4_NT[N][K_bytes] (pre-swizzled INT4).
 *
 * INT8 baseline: k_dequant_i8_to_f16 × 2 + cublasHgemm (FP16, TENSOR_OP_MATH)
 * INT4 baseline: k_dequant_i4_to_f16 × 2 + cublasHgemm
 *
 * Run under ncu for tensor-core SOL:
 *   ncu --metrics sm__inst_executed_pipe_tensor.sum,dram__bytes_read.sum.per_second
 */
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <algorithm>

#include "ptx_mma_tile_int8.cuh"
#include "ptx_mma_tile_int4.cuh"

#ifndef __CUDACC__
int main() { printf("SKIP (not a CUDA build)\n"); return 0; }
#else

#define BENCH_M 3072
#define BENCH_K 8192
#define BENCH_N 3072
#define REPS     50
#define WARMUP    5

/* ── Dequant kernels (mirrors ptx_bench.cu convention) ────────────────────── */
__global__ static void k_dequant_i8_to_f16(const int8_t *in, __half *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = __float2half((float)in[idx]);
}

__global__ static void k_dequant_i4_to_f16(const uint8_t *in, __half *out, int n_nibbles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_nibbles) {
        uint8_t byte = in[idx >> 1];
        unsigned nib = (idx & 1) ? (byte >> 4u) : (byte & 0xFu);
        int8_t val = (nib >= 8u) ? (int8_t)(nib - 16) : (int8_t)nib;
        out[idx] = __float2half((float)val);
    }
}

#define CUBLAS_CHECK(x) do { cublasStatus_t _s=(x); \
    if(_s!=CUBLAS_STATUS_SUCCESS){ fprintf(stderr,"cuBLAS error %d at %s:%d\n",_s,__FILE__,__LINE__); return 1; } \
} while(0)
#define CUDA_CHECK(x) do { cudaError_t _e=(x); \
    if(_e!=cudaSuccess){ fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); return 1; } \
} while(0)

static float percentile(float *sorted, int n, float pct) {
    int idx = (int)(pct * (n - 1) / 100.0f + 0.5f);
    if (idx >= n) idx = n - 1;
    return sorted[idx];
}

/* ── Bench bodies ──────────────────────────────────────────────────────────── */
static int bench_int8(
    const int8_t *dA, const int8_t *dB, const int8_t *dB_NT,
    const float *dSA, const float *dSB,
    __half *dC_ptx, __half *dC_cbl,
    __half *dA_f16, __half *dB_f16,
    cublasHandle_t cbl)
{
    const __half alpha_h = __float2half(1.0f), beta_h = __float2half(0.0f);
    const int BLK = 256;
    const size_t szA = (size_t)BENCH_M * BENCH_K, szB = (size_t)BENCH_K * BENCH_N;

    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    float ms_cbl[REPS], ms_ptx[REPS];

    /* Warm-up */
    for (int r = 0; r < WARMUP; r++) {
        k_dequant_i8_to_f16<<<((int)szA+BLK-1)/BLK, BLK>>>(dA, dA_f16, (int)szA);
        k_dequant_i8_to_f16<<<((int)szB+BLK-1)/BLK, BLK>>>(dB, dB_f16, (int)szB);
        cublasHgemm(cbl, CUBLAS_OP_N, CUBLAS_OP_N, BENCH_N, BENCH_M, BENCH_K,
                    &alpha_h, dB_f16, BENCH_N, dA_f16, BENCH_K, &beta_h, dC_cbl, BENCH_N);
        sp_frob_matmul_q8_mma_tile(dA, dB_NT, dSA, dSB, dC_ptx, BENCH_M, BENCH_K, BENCH_N, 0);
    }
    cudaDeviceSynchronize();

    /* Measure baseline: dequant + cublasHgemm */
    for (int r = 0; r < REPS; r++) {
        cudaEventRecord(t0);
        k_dequant_i8_to_f16<<<((int)szA+BLK-1)/BLK, BLK>>>(dA, dA_f16, (int)szA);
        k_dequant_i8_to_f16<<<((int)szB+BLK-1)/BLK, BLK>>>(dB, dB_f16, (int)szB);
        cublasHgemm(cbl, CUBLAS_OP_N, CUBLAS_OP_N, BENCH_N, BENCH_M, BENCH_K,
                    &alpha_h, dB_f16, BENCH_N, dA_f16, BENCH_K, &beta_h, dC_cbl, BENCH_N);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        cudaEventElapsedTime(&ms_cbl[r], t0, t1);
    }

    /* Measure tiled INT8 kernel */
    for (int r = 0; r < REPS; r++) {
        cudaEventRecord(t0);
        sp_frob_matmul_q8_mma_tile(dA, dB_NT, dSA, dSB, dC_ptx, BENCH_M, BENCH_K, BENCH_N, 0);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        cudaEventElapsedTime(&ms_ptx[r], t0, t1);
    }

    std::sort(ms_cbl, ms_cbl + REPS); std::sort(ms_ptx, ms_ptx + REPS);
    float cbl_med = percentile(ms_cbl, REPS, 50);
    float ptx_med = percentile(ms_ptx, REPS, 50);
    float ptx_p90 = percentile(ms_ptx, REPS, 90);
    float ptx_p99 = percentile(ms_ptx, REPS, 99);
    printf("INT8 (%dx%dx%d): cuBLAS=%.2fms  tile_med=%.2fms p90=%.2fms p99=%.2fms  speedup=%.2fx\n",
           BENCH_M, BENCH_K, BENCH_N, cbl_med, ptx_med, ptx_p90, ptx_p99, cbl_med / ptx_med);
    printf("  gate: >=3x (physical ceiling ~2.8x on sm_75; see SESSION-PLAN ceiling table)\n");
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return 0;
}

static int bench_int4(
    const uint8_t *dAq4, const uint8_t *dBq4, const uint8_t *dBq4_NT,
    const float *dSA, const float *dSB,
    __half *dC_ptx, __half *dC_cbl,
    __half *dA_f16, __half *dB_f16,
    cublasHandle_t cbl)
{
    const __half alpha_h = __float2half(1.0f), beta_h = __float2half(0.0f);
    const int BLK = 256;
    /* INT4 nibble count = 2*K; for dequant we expand BENCH_M*2*BENCH_K nibbles */
    const int n_nib_A = BENCH_M * BENCH_K;   /* nibble count for A */
    const int n_nib_B = BENCH_K * BENCH_N;

    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    float ms_cbl[REPS], ms_ptx[REPS];

    for (int r = 0; r < WARMUP; r++) {
        k_dequant_i4_to_f16<<<(n_nib_A+BLK-1)/BLK, BLK>>>(dAq4, dA_f16, n_nib_A);
        k_dequant_i4_to_f16<<<(n_nib_B+BLK-1)/BLK, BLK>>>(dBq4, dB_f16, n_nib_B);
        cublasHgemm(cbl, CUBLAS_OP_N, CUBLAS_OP_N, BENCH_N, BENCH_M, BENCH_K,
                    &alpha_h, dB_f16, BENCH_N, dA_f16, BENCH_K, &beta_h, dC_cbl, BENCH_N);
        sp_frob_matmul_q4_mma_tile(dAq4, dBq4_NT, dSA, dSB, dC_ptx, BENCH_M, BENCH_K, BENCH_N, 0);
    }
    cudaDeviceSynchronize();

    for (int r = 0; r < REPS; r++) {
        cudaEventRecord(t0);
        k_dequant_i4_to_f16<<<(n_nib_A+BLK-1)/BLK, BLK>>>(dAq4, dA_f16, n_nib_A);
        k_dequant_i4_to_f16<<<(n_nib_B+BLK-1)/BLK, BLK>>>(dBq4, dB_f16, n_nib_B);
        cublasHgemm(cbl, CUBLAS_OP_N, CUBLAS_OP_N, BENCH_N, BENCH_M, BENCH_K,
                    &alpha_h, dB_f16, BENCH_N, dA_f16, BENCH_K, &beta_h, dC_cbl, BENCH_N);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        cudaEventElapsedTime(&ms_cbl[r], t0, t1);
    }

    for (int r = 0; r < REPS; r++) {
        cudaEventRecord(t0);
        sp_frob_matmul_q4_mma_tile(dAq4, dBq4_NT, dSA, dSB, dC_ptx, BENCH_M, BENCH_K, BENCH_N, 0);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        cudaEventElapsedTime(&ms_ptx[r], t0, t1);
    }

    std::sort(ms_cbl, ms_cbl + REPS); std::sort(ms_ptx, ms_ptx + REPS);
    float cbl_med = percentile(ms_cbl, REPS, 50);
    float ptx_med = percentile(ms_ptx, REPS, 50);
    float ptx_p90 = percentile(ms_ptx, REPS, 90);
    float ptx_p99 = percentile(ms_ptx, REPS, 99);
    printf("INT4 (%dx%dx%d): cuBLAS=%.2fms  tile_med=%.2fms p90=%.2fms p99=%.2fms  speedup=%.2fx\n",
           BENCH_M, BENCH_K, BENCH_N, cbl_med, ptx_med, ptx_p90, ptx_p99, cbl_med / ptx_med);
    printf("  gate: >=4x (achievable at ~75%% INT4 TC utilization on sm_75)\n");
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return 0;
}

/* ── Main ──────────────────────────────────────────────────────────────────── */
int main() {
    int dev; if (cudaGetDevice(&dev) != cudaSuccess) { printf("SKIP (no GPU)\n"); return 0; }

    const size_t szA   = (size_t)BENCH_M * BENCH_K;
    const size_t szB   = (size_t)BENCH_K * BENCH_N;
    const size_t szC   = (size_t)BENCH_M * BENCH_N;

    /* Allocate INT8 host data and scales */
    int8_t *hA = new int8_t[szA], *hB = new int8_t[szB];
    float  *hSA = new float[BENCH_M], *hSB = new float[BENCH_N];
    uint32_t seed = 0xABCD1234u;
    for (size_t i = 0; i < szA; i++) { seed=seed*1664525u+1013904223u; hA[i]=(int8_t)((seed>>24)&0xFF); }
    for (size_t i = 0; i < szB; i++) { seed=seed*1664525u+1013904223u; hB[i]=(int8_t)((seed>>24)&0xFF); }
    for (int i = 0; i < BENCH_M; i++) hSA[i] = 1.0f / 128.0f;
    for (int j = 0; j < BENCH_N; j++) hSB[j] = 1.0f / 128.0f;

    /* Transpose hB [K][N] → hB_NT [N][K] for PTX tile kernel */
    int8_t *hB_NT = new int8_t[szB];
    for (size_t k = 0; k < BENCH_K; k++)
        for (size_t n = 0; n < BENCH_N; n++)
            hB_NT[n * BENCH_K + k] = hB[k * BENCH_N + n];

    /* Pack INT4 nibbles */
    const size_t szAq4 = szA / 2, szBq4 = szB / 2;
    uint8_t *hAq4 = new uint8_t[szAq4], *hBq4 = new uint8_t[szBq4];
    memset(hAq4, 0, szAq4); memset(hBq4, 0, szBq4);
    for (size_t i = 0; i < BENCH_M; i++)
        for (size_t k = 0; k < BENCH_K; k += 2) {
            int8_t v0 = (int8_t)((hA[i*BENCH_K+k]   >> 4) & 0xF);
            int8_t v1 = (int8_t)((hA[i*BENCH_K+k+1] >> 4) & 0xF);
            hAq4[i*(BENCH_K/2) + k/2] = (uint8_t)((unsigned)v0 | ((unsigned)v1 << 4));
        }
    for (size_t k = 0; k < BENCH_K; k += 2)
        for (size_t j = 0; j < BENCH_N; j++) {
            int8_t v0 = (int8_t)((hB[k    *BENCH_N+j] >> 4) & 0xF);
            int8_t v1 = (int8_t)((hB[(k+1)*BENCH_N+j] >> 4) & 0xF);
            hBq4[(k/2)*BENCH_N + j] = (uint8_t)((unsigned)v0 | ((unsigned)v1 << 4));
        }

    /* Transpose hBq4 [K_bytes][N] → hBq4_NT [N][K_bytes] for PTX tile kernel */
    uint8_t *hBq4_NT = new uint8_t[szBq4];
    for (size_t k = 0; k < BENCH_K/2; k++)
        for (size_t n = 0; n < BENCH_N; n++)
            hBq4_NT[n * (BENCH_K/2) + k] = hBq4[k * BENCH_N + n];

    /* Allocate device buffers */
    int8_t *dA, *dB, *dB_NT;
    uint8_t *dAq4, *dBq4, *dBq4_NT;
    float *dSA, *dSB;
    __half *dC_ptx, *dC_cbl, *dA_f16, *dB_f16;
    CUDA_CHECK(cudaMalloc(&dA,      szA   * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&dB,      szB   * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&dB_NT,   szB   * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&dAq4,    szAq4 * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&dBq4,    szBq4 * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&dBq4_NT, szBq4 * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&dSA,     BENCH_M * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dSB,     BENCH_N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC_ptx,  szC * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dC_cbl,  szC * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dA_f16,  szA * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dB_f16,  szB * sizeof(__half)));

    CUDA_CHECK(cudaMemcpy(dA,      hA,      szA   * sizeof(int8_t),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB,      hB,      szB   * sizeof(int8_t),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB_NT,   hB_NT,   szB   * sizeof(int8_t),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dAq4,    hAq4,    szAq4 * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBq4,    hBq4,    szBq4 * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBq4_NT, hBq4_NT, szBq4 * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dSA,     hSA,     BENCH_M * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dSB,     hSB,     BENCH_N * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t cbl;
    CUBLAS_CHECK(cublasCreate(&cbl));
    CUBLAS_CHECK(cublasSetMathMode(cbl, CUBLAS_TENSOR_OP_MATH));

    bench_int8(dA, dB, dB_NT, dSA, dSB, dC_ptx, dC_cbl, dA_f16, dB_f16, cbl);
    /* Evict L2 (RTX 2060 = 4MB) between INT8 and INT4 bodies to avoid INT8
     * warmup passes from warming the output-tile mapping and biasing INT4 cuBLAS. */
    cudaDeviceSynchronize();
    cudaMemset(dB_f16, 0, szB * sizeof(__half));
    cudaDeviceSynchronize();
    bench_int4(dAq4, dBq4, dBq4_NT, dSA, dSB, dC_ptx, dC_cbl, dA_f16, dB_f16, cbl);

    cublasDestroy(cbl);
    cudaFree(dA); cudaFree(dB); cudaFree(dB_NT);
    cudaFree(dAq4); cudaFree(dBq4); cudaFree(dBq4_NT);
    cudaFree(dSA); cudaFree(dSB); cudaFree(dC_ptx); cudaFree(dC_cbl);
    cudaFree(dA_f16); cudaFree(dB_f16);
    delete[] hA; delete[] hB; delete[] hB_NT;
    delete[] hAq4; delete[] hBq4; delete[] hBq4_NT;
    delete[] hSA; delete[] hSB;
    return 0;
}

#endif /* __CUDACC__ */
