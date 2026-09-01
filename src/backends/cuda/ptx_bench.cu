/*
 * ptx_bench.cu — M_PTX_2 throughput benchmark for Phase 2-CU.PTX sub-phases.
 * Usage: ./ptx_bench [ntt|hash|spinor|mma|all]
 * On no-GPU host: prints SKIP for each gate.
 *
 * Honest baseline design:
 *   NTT  : runtime-q via uint64_t kernel parameter → forces software division (no compile-time Barrett)
 *   HASH : asm volatile xor.b32 × 2 (sequential dep chain) → baseline is 2-issue not fused lop3
 *   SPINOR: v4 vector loads (512 B/warp) vs ncu dram__bytes_read gate
 *   MMA  : dequant INT8→fp16 + cuBLAS HGEMM (tensor-op math mode) vs PTX INT8/INT4 MMA
 */
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <cublas_v2.h>

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
        fprintf(stderr, "ptx_bench: sm_%d%d < sm_75\n", p.major, p.minor);
        return false;
    }
    return true;
}

/* ── NTT throughput bench ───────────────────────────────────────────────── */

/*
 * Compute-bound NTT chain: 1 load, CHAIN Barrett reductions, 1 store.
 * CHAIN=64 pushes compute/bandwidth ratio to ~64:1 — compute bottleneck, not DRAM.
 * This isolates the per-instruction cost of Barrett (PTX) vs software div (baseline).
 */
#define NTT_CHAIN 64

__global__ static void k_bench_ntt_ptx(uint32_t *inout, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        uint32_t v = inout[i];
        for (int j = 0; j < NTT_CHAIN; j++)
            v = ptx_modmul_q1(v, v);
        inout[i] = v;
    }
}

/*
 * Honest baseline: q passed as runtime uint64_t so nvcc cannot emit compile-time
 * Barrett (IMAD immediate).  Forces software 64-bit division (IMUL.HI subroutine).
 * Confirm in SASS: no IMAD immediate 0x40000bff; expect IMUL.WIDE or UMULHI chain.
 */
__global__ __noinline__ static void k_bench_ntt_baseline(uint32_t *inout, int n, uint64_t q) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        uint64_t v = inout[i];
        for (int j = 0; j < NTT_CHAIN; j++)
            v = (v * v) % q;
        inout[i] = (uint32_t)v;
    }
}

static void bench_ntt(void) {
    const int N = 1 << 20;
    uint32_t *d_buf;
    cudaMalloc(&d_buf, N * sizeof(uint32_t));
    cudaMemset(d_buf, 0x01, N * sizeof(uint32_t));

    k_bench_ntt_ptx<<<(N+255)/256, 256>>>(d_buf, N);
    cudaDeviceSynchronize();

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    const int REPS = 20;

    cudaEventRecord(t0);
    for (int r = 0; r < REPS; r++)
        k_bench_ntt_ptx<<<(N+255)/256, 256>>>(d_buf, N);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms_ptx = 0;
    cudaEventElapsedTime(&ms_ptx, t0, t1);

    cudaEventRecord(t0);
    for (int r = 0; r < REPS; r++)
        k_bench_ntt_baseline<<<(N+255)/256, 256>>>(d_buf, N, (uint64_t)PTX_NTT_Q1);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms_base = 0;
    cudaEventElapsedTime(&ms_base, t0, t1);

    double ops       = (double)N * NTT_CHAIN * REPS;
    double bfly_ptx  = ops / (ms_ptx  * 1e-3);
    double bfly_base = ops / (ms_base * 1e-3);
    double speedup   = bfly_ptx / bfly_base;

    printf("NTT bench: PTX %.2e bfly/s  baseline(runtime-q) %.2e bfly/s  speedup=%.1fx\n",
           bfly_ptx, bfly_base, speedup);
    printf("M_PTX_2 NTT: %s (target >=2x; chain=%d, runtime-q baseline)\n",
           speedup >= 2.0 ? "PASS" : "NEEDS_TUNING", NTT_CHAIN);

    cudaFree(d_buf);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
}

/* ── HASH throughput bench ──────────────────────────────────────────────── */

/*
 * Compute-bound HASH chain: 1 load, CHAIN lop3/xor ops, 1 store.
 * Isolates instruction throughput: lop3 (1 issue) vs 2× xor.b32 (2 issues, dep chain).
 */
#define HASH_CHAIN 512

__global__ static void k_bench_hash_ptx(uint32_t *inout, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        uint32_t v = inout[i];
        uint32_t c = (uint32_t)i + 1u;
        for (int j = 0; j < HASH_CHAIN; j++) {
            v = ptx_xor3(v, v >> 1, c);
            c = c + 1u;
        }
        inout[i] = v;
    }
}

/*
 * Honest XOR-chain baseline: two asm volatile xor.b32 with sequential dep (t→r).
 * asm volatile prevents nvcc from fusing back to lop3.
 * Verify SASS: two XOR instructions, NOT LOP3.LUT.
 */
__global__ __noinline__ static void k_bench_hash_baseline(uint32_t *inout, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        uint32_t v = inout[i];
        uint32_t c = (uint32_t)i + 1u;
        for (int j = 0; j < HASH_CHAIN; j++) {
            uint32_t t, r;
            uint32_t tmp = v >> 1;
            asm volatile("xor.b32 %0, %1, %2;" : "=r"(t) : "r"(v), "r"(tmp));
            asm volatile("xor.b32 %0, %1, %2;" : "=r"(r) : "r"(t), "r"(c));
            v = r;
            c = c + 1u;
        }
        inout[i] = v;
    }
}

static void bench_hash(void) {
    const int N = 1 << 20;
    uint32_t *d_buf;
    cudaMalloc(&d_buf, N * sizeof(uint32_t));
    cudaMemset(d_buf, 0x01, N * sizeof(uint32_t));

    k_bench_hash_ptx<<<(N + 255) / 256, 256>>>(d_buf, N);
    cudaDeviceSynchronize();

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    const int REPS = 20;

    cudaEventRecord(t0);
    for (int r = 0; r < REPS; r++)
        k_bench_hash_ptx<<<(N + 255) / 256, 256>>>(d_buf, N);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms_ptx = 0;
    cudaEventElapsedTime(&ms_ptx, t0, t1);

    cudaEventRecord(t0);
    for (int r = 0; r < REPS; r++)
        k_bench_hash_baseline<<<(N + 255) / 256, 256>>>(d_buf, N);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms_base = 0;
    cudaEventElapsedTime(&ms_base, t0, t1);

    double ops      = (double)N * HASH_CHAIN * REPS;
    double ops_ptx  = ops / (ms_ptx  * 1e-3);
    double ops_base = ops / (ms_base * 1e-3);
    double speedup  = ops_ptx / ops_base;

    printf("HASH bench: lop3 %.2e ops/s  xor-chain %.2e ops/s  speedup=%.1fx\n",
           ops_ptx, ops_base, speedup);
    printf("M_PTX_2 HASH: %s (target >=3x; chain=%d, asm-xor-chain baseline)\n",
           speedup >= 3.0 ? "PASS" : "NEEDS_TUNING", HASH_CHAIN);

    cudaFree(d_buf);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
}

/* ── SPINOR v4 throughput bench ─────────────────────────────────────────── */

/*
 * v4 bench: 3 independent accumulator registers to hide load latency and
 * maximise in-flight DRAM requests per SM.  Each accum handles every-3rd block.
 * 512 bytes/warp per block × 960 warps × 3-way pipeline → ~1.4 MB in-flight.
 */
__global__ __noinline__ static void k_bench_spinor_v4(const uint32_t *src, uint32_t *sink,
                                                       int n_blocks, int is_hot) {
    int chunk = n_blocks / (int)gridDim.x;
    int start = (int)blockIdx.x * chunk;
    int end   = start + chunk;
    int lane  = threadIdx.x & 31;
    uint32_t acc0 = 0, acc1 = 0, acc2 = 0;
    uint32_t v0, v1, v2, v3;
    int b = start;
    for (; b + 2 < end; b += 3) {
        sp_spinor_warpload4(src, (uint32_t)(b + 0), lane, is_hot, &v0, &v1, &v2, &v3);
        acc0 ^= v0 ^ v1 ^ v2 ^ v3;
        sp_spinor_warpload4(src, (uint32_t)(b + 1), lane, is_hot, &v0, &v1, &v2, &v3);
        acc1 ^= v0 ^ v1 ^ v2 ^ v3;
        sp_spinor_warpload4(src, (uint32_t)(b + 2), lane, is_hot, &v0, &v1, &v2, &v3);
        acc2 ^= v0 ^ v1 ^ v2 ^ v3;
    }
    for (; b < end; b++) {
        sp_spinor_warpload4(src, (uint32_t)b, lane, is_hot, &v0, &v1, &v2, &v3);
        acc0 ^= v0 ^ v1 ^ v2 ^ v3;
    }
    uint32_t final_acc = acc0 ^ acc1 ^ acc2;
    if (lane == 0 && final_acc == 0xDEADBEEFu) *sink = final_acc;
}

static void bench_spinor(void) {
    /*
     * N_SUPER: spinor super-blocks, each 128 uint32 = 512 bytes.
     * Total buffer: 32768 × 512 = 16 MB = 5× RTX 2060 L2 (3 MB).
     * N_GRID: 960 = 32 warps/SM × 30 SMs — full sm_75 warp occupancy.
     * Gate: ncu dram__bytes_read.sum.per_second >= 286 GB/s (85% of 336 GB/s peak).
     * Run ncu externally: ncu --metrics dram__bytes_read.sum.per_second ./ptx_bench spinor
     */
    const int N_SUPER = 32768;
    const int N_GRID  = 960;
    const int REPS    = 20;

    uint32_t *d_src, *d_sink;
    cudaMalloc(&d_src,  (size_t)N_SUPER * 128 * sizeof(uint32_t));
    cudaMalloc(&d_sink, sizeof(uint32_t));
    cudaMemset(d_src, 0xAB, (size_t)N_SUPER * 128 * sizeof(uint32_t));

    /* warm-up */
    k_bench_spinor_v4<<<N_GRID, 32>>>(d_src, d_sink, N_SUPER, 0);
    k_bench_spinor_v4<<<N_GRID, 32>>>(d_src, d_sink, N_SUPER, 0);
    cudaDeviceSynchronize();

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);

    cudaEventRecord(t0);
    for (int r = 0; r < REPS; r++)
        k_bench_spinor_v4<<<N_GRID, 32>>>(d_src, d_sink, N_SUPER, 0);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms_v4 = 0;
    cudaEventElapsedTime(&ms_v4, t0, t1);

    /* bytes = N_SUPER super-blocks × 512 bytes/block × REPS */
    double bytes  = (double)N_SUPER * 512 * REPS;
    double bw_v4  = bytes / (ms_v4 * 1e-3) / 1e9;

    printf("SPINOR v4 bench: cold=%.1f GB/s (timer; L2 may inflate vs ncu metric)\n", bw_v4);
    printf("M_PTX_2 SPINOR: gate = ncu dram__bytes_read >= 286 GB/s (85%% SOL)\n");
    printf("  Run: ncu --metrics dram__bytes_read.sum.per_second ./ptx_bench spinor\n");
    printf("  Timer is advisory only; ncu metric is authoritative.\n");

    cudaFree(d_src); cudaFree(d_sink);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
}

/* ── MMA throughput bench (cuBLAS HGEMM baseline) ───────────────────────── */

/*
 * Dequant INT8 → fp16 for cuBLAS baseline input.
 * Applied per-element; scales not included (negligible vs matmul time).
 */
__global__ __noinline__ static void k_dequant_i8_to_f16(
    const int8_t *__restrict__ src, __half *__restrict__ dst, int n)
{
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) dst[i] = __float2half((float)src[i]);
}

/*
 * Dequant packed INT4 → fp16 (nibble-by-nibble expansion).
 * src: uint8[n_nibbles/2], low nibble = even index, high nibble = odd index.
 */
__global__ __noinline__ static void k_dequant_i4_to_f16(
    const uint8_t *__restrict__ src, __half *__restrict__ dst, int n_nibbles)
{
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n_nibbles) {
        uint8_t  byte = src[i >> 1];
        unsigned nib  = (i & 1) ? (byte >> 4u) : (byte & 0xFu);
        int8_t   val  = (nib >= 8u) ? (int8_t)(nib - 16) : (int8_t)nib;
        dst[i] = __float2half((float)val);
    }
}

static void bench_mma(void) {
    /*
     * M=K=N=512: compute-bound regime for both PTX kernel and cuBLAS HGEMM.
     * PTX grid = (64, 64) = 4096 warps; cuBLAS uses its own tiled implementation.
     * Gate INT8: PTX INT8 MMA >= 3x cuBLAS HGEMM fp16.
     * Gate INT4: PTX INT4 MMA >= 4x cuBLAS HGEMM fp16 (INT4 has 2x INT8 TC throughput).
     */
    const int M = 512, K = 512, N = 512;
    const int REPS = 50;

    /* ── Allocate buffers ── */
    int8_t  *d_A_i8, *d_B_i8;
    uint8_t *d_A_i4, *d_B_i4;    /* INT4 packed: M×K/2, K/2×N bytes */
    __half  *d_A_f16, *d_B_f16;  /* fp16 dequant scratch for cuBLAS */
    __half  *d_C_ptx, *d_C_cbl;  /* output buffers */
    float   *d_sca, *d_scb;

    cudaMalloc(&d_A_i8,  (size_t)M * K);
    cudaMalloc(&d_B_i8,  (size_t)K * N);
    cudaMalloc(&d_A_i4,  (size_t)M * (K / 2));
    cudaMalloc(&d_B_i4,  (size_t)(K / 2) * N);
    cudaMalloc(&d_A_f16, (size_t)M * K * sizeof(__half));
    cudaMalloc(&d_B_f16, (size_t)K * N * sizeof(__half));
    cudaMalloc(&d_C_ptx, (size_t)M * N * sizeof(__half));
    cudaMalloc(&d_C_cbl, (size_t)M * N * sizeof(__half));
    cudaMalloc(&d_sca,   (size_t)M * sizeof(float));
    cudaMalloc(&d_scb,   (size_t)N * sizeof(float));

    cudaMemset(d_A_i8, 0x01, (size_t)M * K);
    cudaMemset(d_B_i8, 0x01, (size_t)K * N);
    cudaMemset(d_A_i4, 0x11, (size_t)M * (K / 2));
    cudaMemset(d_B_i4, 0x11, (size_t)(K / 2) * N);

    float *h_sca = new float[M], *h_scb = new float[N];
    for (int i = 0; i < M; i++) h_sca[i] = 1.0f / 127.0f;
    for (int i = 0; i < N; i++) h_scb[i] = 1.0f;
    cudaMemcpy(d_sca, h_sca, M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scb, h_scb, N * sizeof(float), cudaMemcpyHostToDevice);
    delete[] h_sca; delete[] h_scb;

    /* ── cuBLAS setup with explicit tensor-op math mode ── */
    cublasHandle_t cbl;
    cublasCreate(&cbl);
    cublasSetMathMode(cbl, CUBLAS_TENSOR_OP_MATH);
    const __half alpha_h = __float2half(1.0f);
    const __half beta_h  = __float2half(0.0f);

    const int BLK = 256;

    /* Warm-up all paths */
    k_dequant_i8_to_f16<<<(M * K + BLK - 1) / BLK, BLK>>>(d_A_i8, d_A_f16, M * K);
    k_dequant_i8_to_f16<<<(K * N + BLK - 1) / BLK, BLK>>>(d_B_i8, d_B_f16, K * N);
    /* C = A × B in row-major: equiv to C^T = B^T × A^T in cuBLAS col-major */
    cublasHgemm(cbl, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                &alpha_h, d_B_f16, N, d_A_f16, K, &beta_h, d_C_cbl, N);
    sp_frob_matmul_q8_mma(d_A_i8, d_B_i8, d_sca, d_scb, d_C_ptx, M, K, N, 0);
    sp_frob_matmul_q4_mma(d_A_i4, d_B_i4, d_sca, d_scb, d_C_ptx, M, K, N, 0);
    cudaDeviceSynchronize();

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);

    /* ── INT8: time PTX MMA ── */
    cudaEventRecord(t0);
    for (int r = 0; r < REPS; r++)
        sp_frob_matmul_q8_mma(d_A_i8, d_B_i8, d_sca, d_scb, d_C_ptx, M, K, N, 0);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms_ptx_i8 = 0;
    cudaEventElapsedTime(&ms_ptx_i8, t0, t1);

    /* ── INT8: time cuBLAS baseline (dequant + HGEMM) ── */
    cudaEventRecord(t0);
    for (int r = 0; r < REPS; r++) {
        k_dequant_i8_to_f16<<<(M * K + BLK - 1) / BLK, BLK>>>(d_A_i8, d_A_f16, M * K);
        k_dequant_i8_to_f16<<<(K * N + BLK - 1) / BLK, BLK>>>(d_B_i8, d_B_f16, K * N);
        cublasHgemm(cbl, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                    &alpha_h, d_B_f16, N, d_A_f16, K, &beta_h, d_C_cbl, N);
    }
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms_cbl_i8 = 0;
    cudaEventElapsedTime(&ms_cbl_i8, t0, t1);

    double speedup_i8 = (double)(ms_cbl_i8 / ms_ptx_i8);
    printf("MMA INT8 bench: PTX=%.3f ms  cuBLAS-dq=%.3f ms  speedup=%.1fx (per %d reps)\n",
           ms_ptx_i8 / REPS, ms_cbl_i8 / REPS, speedup_i8, REPS);
    printf("M_PTX_2 MMA INT8: %s (target >=3x vs cuBLAS HGEMM+dequant)\n",
           speedup_i8 >= 3.0 ? "PASS" : "NEEDS_TUNING");

    /* ── INT4: time PTX MMA ── */
    cudaEventRecord(t0);
    for (int r = 0; r < REPS; r++)
        sp_frob_matmul_q4_mma(d_A_i4, d_B_i4, d_sca, d_scb, d_C_ptx, M, K, N, 0);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms_ptx_i4 = 0;
    cudaEventElapsedTime(&ms_ptx_i4, t0, t1);

    /* ── INT4: time cuBLAS baseline (dequant fp16 nibble-expand + HGEMM) ── */
    cudaEventRecord(t0);
    for (int r = 0; r < REPS; r++) {
        k_dequant_i4_to_f16<<<(M * K + BLK - 1) / BLK, BLK>>>(d_A_i4, d_A_f16, M * K);
        k_dequant_i4_to_f16<<<(K * N + BLK - 1) / BLK, BLK>>>(d_B_i4, d_B_f16, K * N);
        cublasHgemm(cbl, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                    &alpha_h, d_B_f16, N, d_A_f16, K, &beta_h, d_C_cbl, N);
    }
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms_cbl_i4 = 0;
    cudaEventElapsedTime(&ms_cbl_i4, t0, t1);

    double speedup_i4 = (double)(ms_cbl_i4 / ms_ptx_i4);
    printf("MMA INT4 bench: PTX=%.3f ms  cuBLAS-dq=%.3f ms  speedup=%.1fx (per %d reps)\n",
           ms_ptx_i4 / REPS, ms_cbl_i4 / REPS, speedup_i4, REPS);
    printf("M_PTX_2 MMA INT4: %s (target >=4x vs cuBLAS HGEMM+dequant)\n",
           speedup_i4 >= 4.0 ? "PASS" : "NEEDS_TUNING");

    cublasDestroy(cbl);
    cudaFree(d_A_i8); cudaFree(d_B_i8);
    cudaFree(d_A_i4); cudaFree(d_B_i4);
    cudaFree(d_A_f16); cudaFree(d_B_f16);
    cudaFree(d_C_ptx); cudaFree(d_C_cbl);
    cudaFree(d_sca);   cudaFree(d_scb);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
}

/* ── main ───────────────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    const char *filter = (argc > 1) ? argv[1] : "all";
    bool gpu = gpu_available();
    printf("ptx_bench: GPU=%s filter=%s\n", gpu ? "YES" : "NO (SKIP)", filter);

    if (!strcmp(filter, "ntt") || !strcmp(filter, "all")) {
        if (!gpu) printf("M_PTX_2 NTT: SKIP\n");
        else      bench_ntt();
    }
    if (!strcmp(filter, "hash") || !strcmp(filter, "all")) {
        if (!gpu) printf("M_PTX_2 HASH: SKIP\n");
        else      bench_hash();
    }
    if (!strcmp(filter, "spinor") || !strcmp(filter, "all")) {
        if (!gpu) printf("M_PTX_2 SPINOR: SKIP\n");
        else      bench_spinor();
    }
    if (!strcmp(filter, "mma") || !strcmp(filter, "all")) {
        if (!gpu) printf("M_PTX_2 MMA: SKIP\n");
        else      bench_mma();
    }

    return 0;
}
