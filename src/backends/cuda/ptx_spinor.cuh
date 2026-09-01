/*
 * ptx_spinor.cuh — PTX ld.global.cg/cs u32 load primitives for Spinor warp-loads.
 *
 * sp_spinor_block_t: 63 bytes (7 header + 55 body + 1 CRC), frozen.
 * Warp packing: 32 lanes × 4 bytes = 128 bytes = 2 Spinors + 2 slack bytes.
 * sp_spinor_warpload stride: 16 words (64 bytes) per block_idx step.
 *
 * Cache dispatch (runtime):
 *   hot  (in SWA window): ld.global.cg  — L2-cached, L1-bypassed
 *   cold (outside):       ld.global.cs  — streaming, L1+L2 bypass
 */
#pragma once
#include <cstdint>
#include <cstring>

#ifdef __CUDACC__

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750

__device__ __forceinline__
uint32_t ptx_ld_cg_u32(const void *addr) {
    uint32_t v;
    asm volatile ("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(addr));
    return v;
}

__device__ __forceinline__
uint32_t ptx_ld_cs_u32(const void *addr) {
    uint32_t v;
    asm volatile ("ld.global.cs.u32 %0, [%1];" : "=r"(v) : "l"(addr));
    return v;
}

__device__ __forceinline__
void ptx_ld4_cg(const uint32_t *addr,
                uint32_t *v0, uint32_t *v1, uint32_t *v2, uint32_t *v3) {
    asm volatile (
        "ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*v0), "=r"(*v1), "=r"(*v2), "=r"(*v3)
        : "l"(addr)
    );
}

__device__ __forceinline__
void ptx_ld4_cs(const uint32_t *addr,
                uint32_t *v0, uint32_t *v1, uint32_t *v2, uint32_t *v3) {
    asm volatile (
        "ld.global.cs.v4.u32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*v0), "=r"(*v1), "=r"(*v2), "=r"(*v3)
        : "l"(addr)
    );
}

/*
 * sp_spinor_warpload — one 4-byte word per lane from a Spinor block array.
 * Array stride: 16 uint32_t words (64 bytes) per block_idx.
 * Word loaded: base[block_idx * 16 + lane].
 */
__device__ __forceinline__
uint32_t sp_spinor_warpload(const uint32_t *base, uint32_t block_idx,
                             int lane, int is_hot) {
    const void *ptr = (const void *)(base + (size_t)block_idx * 16 + lane);
    return is_hot ? ptx_ld_cg_u32(ptr) : ptx_ld_cs_u32(ptr);
}

/*
 * sp_spinor_warpload4 — four 4-byte words per lane (v4 vector load, 16 B/lane = 512 B/warp).
 * Array stride: 128 uint32_t words (512 bytes) per block_idx.
 * Alignment: base + block_idx*128 + lane*4 is always 16-byte aligned since lane*4*4=lane*16 B.
 * Words loaded: base[block_idx*128 + lane*4 + 0..3].
 */
__device__ __forceinline__
void sp_spinor_warpload4(const uint32_t *base, uint32_t block_idx, int lane,
                          int is_hot,
                          uint32_t *v0, uint32_t *v1, uint32_t *v2, uint32_t *v3) {
    const uint32_t *ptr = base + (size_t)block_idx * 128 + (size_t)lane * 4;
    if (is_hot)
        ptx_ld4_cg(ptr, v0, v1, v2, v3);
    else
        ptx_ld4_cs(ptr, v0, v1, v2, v3);
}

#else  /* host / sm < 75 fallback */

__host__ __device__ __forceinline__
uint32_t ptx_ld_cg_u32(const void *addr) {
    uint32_t v; memcpy(&v, addr, 4); return v;
}
__host__ __device__ __forceinline__
uint32_t ptx_ld_cs_u32(const void *addr) {
    uint32_t v; memcpy(&v, addr, 4); return v;
}
__host__ __device__ __forceinline__
void ptx_ld4_cg(const uint32_t *addr,
                uint32_t *v0, uint32_t *v1, uint32_t *v2, uint32_t *v3) {
    *v0 = addr[0]; *v1 = addr[1]; *v2 = addr[2]; *v3 = addr[3];
}
__host__ __device__ __forceinline__
void ptx_ld4_cs(const uint32_t *addr,
                uint32_t *v0, uint32_t *v1, uint32_t *v2, uint32_t *v3) {
    *v0 = addr[0]; *v1 = addr[1]; *v2 = addr[2]; *v3 = addr[3];
}
__device__ __forceinline__
uint32_t sp_spinor_warpload(const uint32_t *base, uint32_t block_idx,
                             int lane, int is_hot) {
    (void)is_hot;
    return base[(size_t)block_idx * 16 + lane];
}
__host__ __device__ __forceinline__
void sp_spinor_warpload4(const uint32_t *base, uint32_t block_idx, int lane,
                          int is_hot,
                          uint32_t *v0, uint32_t *v1, uint32_t *v2, uint32_t *v3) {
    (void)is_hot;
    const uint32_t *ptr = base + (size_t)block_idx * 128 + (size_t)lane * 4;
    *v0 = ptr[0]; *v1 = ptr[1]; *v2 = ptr[2]; *v3 = ptr[3];
}

#endif /* __CUDA_ARCH__ >= 750 */
#endif /* __CUDACC__ */
