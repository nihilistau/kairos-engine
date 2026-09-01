/*
 * ptx_mma_tile_common.cuh — shared constants, smem layout, and load helpers
 * for the 64×64 tiled INT8/INT4 MMA kernels (§17.3.TILE).
 *
 * Architecture: 64×64 block tile / 32×32 warp tile / 4×4 MMA grid / 4 warps.
 * Pipeline:     sm_75 — A tile staged to smem (ld.global.cg.v4.u32 + __syncthreads);
 *               B reads directly from global [N][K] via ld.global.nc.u32 (no smem).
 *
 * B-matrix contract (§17.3.TILE-2c): caller must provide B in [N][K] row-major.
 *   Frobenius arena stores packed codes as [N][K] already (rows=out_features,
 *   cols=in_features). CPU-side sp_transcode_b_q8/q4 handles any other source.
 *   Thread T at fragment (block_n, warp_n, mma_n, mma_k) reads B[nc][k0..k0+3]:
 *     nc = block_n*64 + warp_n*32 + mma_n*8 + (T>>2)
 *     k0 = k_tile *32 + mma_k *16 + (T&3 )*4  (4-byte aligned)
 *   Single ld.global.nc.u32 per fragment; routes through read-only cache.
 *
 * Assumes M, N multiples of SP_TILE_BLOCK_M/N and K multiple of SP_TILE_K_TILE.
 *
 * References:
 *   DeepEP ptx.cuh lines 1-10 (fallback macro pattern)
 *   DeepEP utils.cuh lines 177-181 (LD_NC_FUNC / SP_LD_WEIGHT_FUNC)
 *   ptx_spinor.cuh lines 36-50 (ld.global.cg.v4 discipline)
 *   ptx_mma.cuh lines 21-40 (mma_s8_m8n8k16, mma_s4_m8n8k32 — called, not redefined)
 */
#pragma once
#include <cstdint>
#include <cuda_fp16.h>

#ifdef __CUDACC__

/* ── Tile geometry ─────────────────────────────────────────────────────────── */
#define SP_TILE_BLOCK_M  64
#define SP_TILE_BLOCK_N  64
#define SP_TILE_K_TILE   32     /* INT8: 32 K-elements per smem stage;   */
                                /* INT4: 32 packed-nibble bytes = 64 nib  */
#define SP_TILE_WARPS    4
#define SP_TILE_THREADS  128    /* SP_TILE_WARPS * 32 */

/* Warp 2×2 arrangement: warp_id → (warp_m, warp_n) */
#define SP_TILE_WARP_M(wid) ((wid) >> 1)      /* 0,0,1,1 */
#define SP_TILE_WARP_N(wid) ((wid) & 1)       /* 0,1,0,1 */

/* ── Weight-side load macro ────────────────────────────────────────────────── */
/* sm_75: ld.global.nc routes B through the read-only/texture cache (non-coherent).  */
/* sm_80+: add L1::no_allocate + L2::256B eviction hints for streaming weight access.*/
#if !defined(DISABLE_WEIGHT_LD_NC) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  #define SP_LD_WEIGHT_FUNC "ld.global.nc.L1::no_allocate.L2::256B"
#else
  #define SP_LD_WEIGHT_FUNC "ld.global.nc"
#endif

/* ── A smem layout (B is NOT staged to smem — read directly from global) ───── */
typedef int8_t  sp_smem_a_t[SP_TILE_BLOCK_M][SP_TILE_K_TILE];

/* ── Cooperative A-tile load: 128 threads, 16 bytes each ──────────────────── */
/* A is the activation side; use ld.global.cg (L2-cached).                     */
/* Thread T: row = T>>1, col_byte = (T&1)<<4 (0 or 16). 2 threads per row.     */
__device__ __forceinline__
void sp_tile_load_a_int8(
    const int8_t * __restrict__ A_global,
    int8_t smem_a[SP_TILE_BLOCK_M][SP_TILE_K_TILE],
    int block_row, int k_tile,
    int M, int K, int thr_id)
{
    int row      = thr_id >> 1;
    int col_byte = (thr_id & 1) << 4;
    int g_row    = block_row * SP_TILE_BLOCK_M + row;
    int g_col    = k_tile   * SP_TILE_K_TILE  + col_byte;
    uint32_t v0, v1, v2, v3;
    if (g_row < M && (g_col + 15) < K) {
        const uint32_t *p = (const uint32_t *)(A_global + g_row * K + g_col);
        asm volatile ("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
            : "=r"(v0), "=r"(v1), "=r"(v2), "=r"(v3) : "l"(p));
    } else {
        v0 = v1 = v2 = v3 = 0;
    }
    uint32_t *s = (uint32_t *)(smem_a[row] + col_byte);
    s[0] = v0; s[1] = v1; s[2] = v2; s[3] = v3;
}

__device__ __forceinline__
void sp_tile_load_a_int4(
    const uint8_t * __restrict__ A_q4,
    uint8_t smem_a[SP_TILE_BLOCK_M][SP_TILE_K_TILE],
    int block_row, int k_tile,
    int M, int K_bytes, int thr_id)
{
    int row      = thr_id >> 1;
    int col_byte = (thr_id & 1) << 4;
    int g_row    = block_row * SP_TILE_BLOCK_M + row;
    int g_col    = k_tile   * SP_TILE_K_TILE  + col_byte;
    uint32_t v0, v1, v2, v3;
    if (g_row < M && (g_col + 15) < K_bytes) {
        const uint32_t *p = (const uint32_t *)(A_q4 + g_row * K_bytes + g_col);
        asm volatile ("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
            : "=r"(v0), "=r"(v1), "=r"(v2), "=r"(v3) : "l"(p));
    } else {
        v0 = v1 = v2 = v3 = 0;
    }
    uint32_t *s = (uint32_t *)(smem_a[row] + col_byte);
    s[0] = v0; s[1] = v1; s[2] = v2; s[3] = v3;
}

/* ── Fragment loads: A from smem, B direct from global [N][K] ─────────────── */
__device__ __forceinline__
uint32_t sp_tile_frag_a_int8(
    const int8_t smem_a[SP_TILE_BLOCK_M][SP_TILE_K_TILE],
    int warp_m, int mma_m, int mma_k, int lane)
{
    int r = warp_m * 32 + mma_m * 8 + (lane >> 2);
    int c = mma_k  * 16 + (lane & 3) * 4;
    return *(const uint32_t *)(smem_a[r] + c);
}

__device__ __forceinline__
uint32_t sp_tile_frag_a_int4(
    const uint8_t smem_a[SP_TILE_BLOCK_M][SP_TILE_K_TILE],
    int warp_m, int mma_m, int mma_k, int lane)
{
    int r = warp_m * 32 + mma_m * 8 + (lane >> 2);
    int c = mma_k  * 16 + (lane & 3) * 4;
    return *(const uint32_t *)(smem_a[r] + c);
}

/* B fragment: B_NT[N][K] global → ld.global.nc.u32 (read-only cache, no smem).
 * nc = block_n*64 + warp_n*32 + mma_n*8 + (lane>>2) — absolute N column.
 * k0 = k_tile *32 + mma_k *16 + (lane&3 )*4 — 4-byte aligned K offset.
 * 4 threads with same nc (lanes sharing n_col) issue the same load address →
 * hardware coalesces/broadcasts one cache-line read for the quad. */
__device__ __forceinline__
uint32_t sp_tile_frag_b_global_int8(
    const int8_t * __restrict__ B_NT,
    int block_n, int warp_n, int mma_n, int mma_k, int k_tile, int lane, int K)
{
    int nc = block_n * SP_TILE_BLOCK_N + warp_n * 32 + mma_n * 8 + (lane >> 2);
    int k0 = k_tile  * SP_TILE_K_TILE  + mma_k  * 16 + (lane & 3) * 4;
    const uint32_t *p = (const uint32_t *)(B_NT + (ptrdiff_t)nc * K + k0);
    uint32_t v;
    asm volatile (SP_LD_WEIGHT_FUNC ".u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}

__device__ __forceinline__
uint32_t sp_tile_frag_b_global_int4(
    const uint8_t * __restrict__ B_q4_NT,
    int block_n, int warp_n, int mma_n, int mma_k, int k_tile, int lane, int K_bytes)
{
    int nc = block_n * SP_TILE_BLOCK_N + warp_n * 32 + mma_n * 8 + (lane >> 2);
    int k0 = k_tile  * SP_TILE_K_TILE  + mma_k  * 16 + (lane & 3) * 4;
    const uint32_t *p = (const uint32_t *)(B_q4_NT + (ptrdiff_t)nc * K_bytes + k0);
    uint32_t v;
    asm volatile (SP_LD_WEIGHT_FUNC ".u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}

/* ── Scale epilogue helper ─────────────────────────────────────────────────── */
__device__ __forceinline__
void sp_tile_epilogue_mma(
    const int acc_d0, const int acc_d1,
    __half * __restrict__ C_global,
    const float * __restrict__ scale_a,
    const float * __restrict__ scale_b,
    int out_row, int out_col,
    int lane, int M, int N)
{
    int row  = out_row + (lane >> 2);
    int col0 = out_col + (lane & 3) * 2;
    int col1 = col0 + 1;
    if (row < M) {
        float sa = scale_a[row];
        if (col0 < N)
            C_global[row * N + col0] = __float2half(__int2float_rn(acc_d0) * sa * scale_b[col0]);
        if (col1 < N)
            C_global[row * N + col1] = __float2half(__int2float_rn(acc_d1) * sa * scale_b[col1]);
    }
}

#endif /* __CUDACC__ */
