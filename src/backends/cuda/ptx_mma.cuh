/*
 * ptx_mma.cuh — INT8 and INT4 tensor-core matmul via bare PTX inline asm.
 *
 * NO #include <mma.h>.  NO nvcuda::wmma references.  All matrix-multiply
 * operations use asm volatile("mma.sync.aligned...").
 *
 * INT8 path  (§17.3): mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32
 * INT4 path  (§17.3): mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32
 *
 * Fragment register layout (PTX ISA, sm_75, both INT8 and INT4):
 *   Thread T (0..31) in warp:
 *     A fragment : row = T/4,   k_byte_offset = (T%4)*4   (4 bytes = 4 INT8 or 8 nibbles)
 *     B fragment : col = T/4,   k_byte_offset = (T%4)*4   (col-major pattern from row-major B)
 *     D/C frag   : row = T/4,   col = (T%4)*2 and (T%4)*2+1
 *
 * Tile sizes:
 *   MMA_M=8, MMA_N=8, MMA_K=16  (INT8, one warp per 8×8 output tile)
 *   MMA_INT4_K=32               (INT4, same 8×8 output tile, double K depth)
 *
 * Gate coverage:
 *   M_PTX_1 INT8 : bit-exact vs sp_frob_matmul_q8_ref  (TestA all-ones, TestB random Q8)
 *   M_PTX_1 INT4 : bit-exact vs sp_frob_matmul_q4_ref  (TestC all-ones, TestD random Q4)
 *   M_PTX_2 MMA  : throughput vs cuBLAS HGEMM (INT8 >=3×, INT4 >=4×)
 *   M_PTX_3 MMA  : no cudaMalloc in kernel hot path
 *   M_PTX_4 MMA  : per-session CUDA stream isolation
 *
 * DO NOT add softmax / temperature / probability code.  Discrete argmax only.
 */
#pragma once
#include <cstdint>
#include <cstring>
#include <cuda_fp16.h>

#ifdef __CUDACC__

/* Tile constants (INT8 m8n8k16) */
#define MMA_M  8
#define MMA_N  8
#define MMA_K  16

/* INT4 depth constant (m8n8k32, same 8×8 output tile) */
#define MMA_INT4_K  32

/* ── PTX inline-asm wrappers ───────────────────────────────────────────── */

/* INT8 mma: d = a × b + c  (m8n8k16, row.col, s32 accumulator) */
static __device__ __forceinline__
void mma_s8_m8n8k16(int *d0, int *d1,
                    uint32_t a, uint32_t b,
                    int c0, int c1)
{
    asm volatile(
        "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
        "{%0,%1},{%2},{%3},{%4,%5};"
        : "=r"(*d0), "=r"(*d1)
        : "r"(a), "r"(b), "r"(c0), "r"(c1)
    );
}

/* INT4 mma: d = a × b + c  (m8n8k32, row.col, s32 accumulator) */
static __device__ __forceinline__
void mma_s4_m8n8k32(int *d0, int *d1,
                    uint32_t a, uint32_t b,
                    int c0, int c1)
{
    asm volatile(
        "mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 "
        "{%0,%1},{%2},{%3},{%4,%5};"
        : "=r"(*d0), "=r"(*d1)
        : "r"(a), "r"(b), "r"(c0), "r"(c1)
    );
}

/* ── INT8 kernel ───────────────────────────────────────────────────────── */

/* Forward declarations */
__global__ void sp_frob_matmul_q8_mma_kernel(
    const int8_t  *__restrict__ A,
    const int8_t  *__restrict__ B,
    const float   *__restrict__ scale_a,
    const float   *__restrict__ scale_b,
    __half        *__restrict__ C,
    int M, int K, int N);

inline void sp_frob_matmul_q8_mma(
    const int8_t  *A,
    const int8_t  *B,
    const float   *scale_a,
    const float   *scale_b,
    __half        *C,
    int M, int K, int N,
    cudaStream_t   stream)
{
    dim3 grid((unsigned)((N + MMA_N - 1) / MMA_N),
              (unsigned)((M + MMA_M - 1) / MMA_M));
    dim3 block(32);
    sp_frob_matmul_q8_mma_kernel<<<grid, block, 0, stream>>>(
        A, B, scale_a, scale_b, C, M, K, N);
}

/*
 * sp_frob_matmul_q8_mma_kernel
 *
 * One warp per 8×8 output tile.  Grid = (ceil(N/8), ceil(M/8)).
 * Uses PTX mma.sync.aligned.m8n8k16 — zero nvcuda::wmma references.
 *
 * A: int8[M][K]  row-major
 * B: int8[K][N]  row-major (staged to col-major via __shared__)
 * scale_a: float[M] per-row dequant scale
 * scale_b: float[N] per-col dequant scale
 * C: __half[M][N] row-major output
 *
 * M_PTX_3: only __shared__ used; no cudaMalloc in hot path.
 */
__global__ void sp_frob_matmul_q8_mma_kernel(
    const int8_t  *__restrict__ A,
    const int8_t  *__restrict__ B,
    const float   *__restrict__ scale_a,
    const float   *__restrict__ scale_b,
    __half        *__restrict__ C,
    int M, int K, int N)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    const int warp_m = (int)blockIdx.y;
    const int warp_n = (int)blockIdx.x;
    const int out_row = warp_m * MMA_M;
    const int out_col = warp_n * MMA_N;
    const int lane    = (int)threadIdx.x;

    /*
     * smem_a[8][16]: staging A tile in row-major.
     * smem_b[16][8]: staging B tile in row-major (read as col-major for mma).
     * 128 bytes each; 32 threads × 4 bytes/thread = 128 bytes per cooperative load.
     */
    __shared__ int8_t smem_a[MMA_M * MMA_K];
    __shared__ int8_t smem_b[MMA_K * MMA_N];

    /* Accumulator: starts at 0, accumulates over K-tiles. */
    int c0 = 0, c1 = 0;

    for (int k = 0; k < K; k += MMA_K) {
        /* ── Load smem_a (A tile: 8 rows × 16 cols = 128 bytes) ─────── */
        /*
         * Fragment layout: thread T owns row=T/4, k_start=(T%4)*4.
         * Load 4 consecutive bytes: A[out_row + T/4][k + (T%4)*4 .. +3].
         * Store into smem_a[row * MMA_K + k_start].
         */
        {
            const int a_row = lane / 4;        /* 0..7 */
            const int a_ko  = (lane % 4) * 4;  /* 0, 4, 8, 12 */
            const int gr    = out_row + a_row;
            for (int c = 0; c < 4; c++) {
                const int gk = k + a_ko + c;
                smem_a[a_row * MMA_K + a_ko + c] =
                    (gr < M && gk < K) ? A[gr * K + gk] : (int8_t)0;
            }
        }

        /* ── Load smem_b (B tile: 16 rows × 8 cols = 128 bytes) ─────── */
        /*
         * Load row-major from B[K][N]: thread T → row=T/2 in tile, col_start=(T%2)*4.
         * 32 threads × 4 bytes = 128 bytes, 2 threads per B-tile row.
         */
        {
            const int b_row = lane / 2;         /* 0..15 */
            const int b_cst = (lane % 2) * 4;   /* 0 or 4 */
            const int gk    = k + b_row;
            for (int c = 0; c < 4; c++) {
                const int gc = out_col + b_cst + c;
                smem_b[b_row * MMA_N + b_cst + c] =
                    (gk < K && gc < N) ? B[gk * N + gc] : (int8_t)0;
            }
        }

        __syncwarp();

        /* ── Build a_reg (4 int8 packed as uint32) ───────────────────── */
        uint32_t a_reg;
        memcpy(&a_reg, &smem_a[(lane / 4) * MMA_K + (lane % 4) * 4], 4);

        /* ── Build b_reg (col-major read: 4 bytes from smem_b column) ── */
        /*
         * Thread T owns col = T/4 in the N dimension; k_start = (T%4)*4.
         * Reads smem_b[k_start+0..3][col], which unpacks as col-major B.
         */
        {
            const int b_col = lane / 4;
            const int b_ko  = (lane % 4) * 4;
            int8_t bb[4] = {
                smem_b[(b_ko + 0) * MMA_N + b_col],
                smem_b[(b_ko + 1) * MMA_N + b_col],
                smem_b[(b_ko + 2) * MMA_N + b_col],
                smem_b[(b_ko + 3) * MMA_N + b_col]
            };
            uint32_t b_reg;
            memcpy(&b_reg, bb, 4);

            /* ── Execute INT8 mma.sync ───────────────────────────────── */
            int d0, d1;
            mma_s8_m8n8k16(&d0, &d1, a_reg, b_reg, c0, c1);
            c0 = d0;
            c1 = d1;
        }
    }

    /* ── Scatter accumulator → C with per-row/col scales ─────────────── */
    /*
     * Thread T owns output positions:
     *   row = out_row + T/4
     *   col0 = out_col + (T%4)*2
     *   col1 = out_col + (T%4)*2 + 1
     */
    const int d_row  = out_row + lane / 4;
    const int d_col0 = out_col + (lane % 4) * 2;
    const int d_col1 = d_col0 + 1;
    if (d_row < M && d_col0 < N)
        C[d_row * N + d_col0] = __float2half((float)c0 * scale_a[d_row] * scale_b[d_col0]);
    if (d_row < M && d_col1 < N)
        C[d_row * N + d_col1] = __float2half((float)c1 * scale_a[d_row] * scale_b[d_col1]);

#endif /* __CUDA_ARCH__ >= 750 */
}

/* ── INT4 kernel ───────────────────────────────────────────────────────── */
/*
 * Layout:
 *   A_q4: uint8[M][K/2]  — packed s4, low nibble = even K index, row-major
 *   B_q4: uint8[K/2][N]  — packed s4, low nibble = even K index, row-major
 *   (K must be even; SP_FROB_QMAX4=7, codes in [-7,7])
 *
 * Each K-tile covers MMA_INT4_K=32 nibbles = 16 bytes (MMA_INT4_KB=16).
 * smem layouts [8][16] and [16][8] are identical in size to the INT8 kernel.
 */

#define MMA_INT4_KB  (MMA_INT4_K / 2)  /* 16 bytes per K-tile */

__global__ void sp_frob_matmul_q4_mma_kernel(
    const uint8_t *__restrict__ A_q4,
    const uint8_t *__restrict__ B_q4,
    const float   *__restrict__ scale_a,
    const float   *__restrict__ scale_b,
    __half        *__restrict__ C,
    int M, int K, int N);

inline void sp_frob_matmul_q4_mma(
    const uint8_t *A_q4,
    const uint8_t *B_q4,
    const float   *scale_a,
    const float   *scale_b,
    __half        *C,
    int M, int K, int N,
    cudaStream_t   stream)
{
    dim3 grid((unsigned)((N + MMA_N - 1) / MMA_N),
              (unsigned)((M + MMA_M - 1) / MMA_M));
    dim3 block(32);
    sp_frob_matmul_q4_mma_kernel<<<grid, block, 0, stream>>>(
        A_q4, B_q4, scale_a, scale_b, C, M, K, N);
}

/*
 * sp_frob_matmul_q4_mma_kernel
 *
 * Same structure as the INT8 kernel but with packed nibbles and k_byte-indexed
 * K loop.  K is the number of s4 nibble elements; K/2 is the packed byte count.
 * M_PTX_3: only __shared__ used; no cudaMalloc in hot path.
 */
__global__ void sp_frob_matmul_q4_mma_kernel(
    const uint8_t *__restrict__ A_q4,
    const uint8_t *__restrict__ B_q4,
    const float   *__restrict__ scale_a,
    const float   *__restrict__ scale_b,
    __half        *__restrict__ C,
    int M, int K, int N)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    const int warp_m = (int)blockIdx.y;
    const int warp_n = (int)blockIdx.x;
    const int out_row = warp_m * MMA_M;
    const int out_col = warp_n * MMA_N;
    const int lane    = (int)threadIdx.x;
    const int K_bytes = K / 2;  /* packed byte count */

    /*
     * smem layout identical to INT8 (128 bytes each):
     *   smem_a[8][16]: 8 rows × 16 bytes (= 32 nibbles per row)
     *   smem_b[16][8]: 16 byte-rows × 8 cols
     */
    __shared__ uint8_t smem_a[MMA_M * MMA_INT4_KB];
    __shared__ uint8_t smem_b[MMA_INT4_KB * MMA_N];

    int c0 = 0, c1 = 0;

    /* k_byte: byte offset in K/2 dimension, steps by MMA_INT4_KB=16 */
    for (int k_byte = 0; k_byte < K_bytes; k_byte += MMA_INT4_KB) {
        /* ── Load smem_a ── */
        {
            const int a_row = lane / 4;
            const int a_bko = (lane % 4) * 4;
            const int gr    = out_row + a_row;
            for (int c = 0; c < 4; c++) {
                const int gkb = k_byte + a_bko + c;
                smem_a[a_row * MMA_INT4_KB + a_bko + c] =
                    (gr < M && gkb < K_bytes) ? A_q4[gr * K_bytes + gkb] : 0u;
            }
        }

        /* ── Load smem_b ── */
        {
            const int b_row = lane / 2;
            const int b_cst = (lane % 2) * 4;
            const int gkb   = k_byte + b_row;
            for (int c = 0; c < 4; c++) {
                const int gc = out_col + b_cst + c;
                smem_b[b_row * MMA_N + b_cst + c] =
                    (gkb < K_bytes && gc < N) ? B_q4[gkb * N + gc] : 0u;
            }
        }

        __syncwarp();

        uint32_t a_reg;
        memcpy(&a_reg, &smem_a[(lane / 4) * MMA_INT4_KB + (lane % 4) * 4], 4);

        {
            const int b_col = lane / 4;
            const int b_bko = (lane % 4) * 4;
            uint8_t bb[4] = {
                smem_b[(b_bko + 0) * MMA_N + b_col],
                smem_b[(b_bko + 1) * MMA_N + b_col],
                smem_b[(b_bko + 2) * MMA_N + b_col],
                smem_b[(b_bko + 3) * MMA_N + b_col]
            };
            uint32_t b_reg;
            memcpy(&b_reg, bb, 4);

            int d0, d1;
            mma_s4_m8n8k32(&d0, &d1, a_reg, b_reg, c0, c1);
            c0 = d0;
            c1 = d1;
        }
    }

    const int d_row  = out_row + lane / 4;
    const int d_col0 = out_col + (lane % 4) * 2;
    const int d_col1 = d_col0 + 1;
    if (d_row < M && d_col0 < N)
        C[d_row * N + d_col0] = __float2half((float)c0 * scale_a[d_row] * scale_b[d_col0]);
    if (d_row < M && d_col1 < N)
        C[d_row * N + d_col1] = __float2half((float)c1 * scale_a[d_row] * scale_b[d_col1]);

#endif /* __CUDA_ARCH__ >= 750 */
}

/* ── Scalar host references ────────────────────────────────────────────── */

/*
 * sp_frob_matmul_q8_ref — host-only scalar reference.
 * Matches GPU kernel fp ops order for bit-accurate comparison.
 */
inline void sp_frob_matmul_q8_ref(
    const int8_t *A, const int8_t *B,
    const float  *scale_a, const float *scale_b,
    __half *C, int M, int K, int N)
{
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += (int32_t)A[i * K + k] * (int32_t)B[k * N + j];
            C[i * N + j] = __float2half((float)acc * scale_a[i] * scale_b[j]);
        }
}

/*
 * sp_frob_matmul_q4_ref — host-only INT4 scalar reference.
 * A_q4: uint8[M][K/2], B_q4: uint8[K/2][N], K = nibble count (must be even).
 * Nibble packing: byte[idx/2] low nibble = idx%2==0, high nibble = idx%2==1.
 * s4 sign: nib >= 8 → value = nib - 16.
 */
static inline int8_t sp_q4_extract(const uint8_t *packed, int idx) {
    const uint8_t byte = packed[idx >> 1];
    const unsigned nib = (idx & 1) ? (byte >> 4u) : (byte & 0xFu);
    return (nib >= 8u) ? (int8_t)(nib - 16) : (int8_t)nib;
}

inline void sp_frob_matmul_q4_ref(
    const uint8_t *A_q4, const uint8_t *B_q4,
    const float   *scale_a, const float *scale_b,
    __half *C, int M, int K, int N)
{
    const int K_bytes = K / 2;
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++) {
                int8_t a_val = sp_q4_extract(A_q4 + i * K_bytes, k);
                /* B_q4[K/2][N]: packed byte at B_q4[(k/2)*N + j], nibble k%2 */
                int8_t b_val = sp_q4_extract(B_q4 + (k / 2) * N + j, k & 1);
                acc += (int32_t)a_val * (int32_t)b_val;
            }
            C[i * N + j] = __float2half((float)acc * scale_a[i] * scale_b[j]);
        }
}

#endif /* __CUDACC__ */
