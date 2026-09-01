/*
 * ptx_hash.cuh — PTX lop3.b32 + prmt.b32 primitives for KSTE/XXH3 mixing.
 * M_PTX_1 gate: bit-identical vs C scalar for all functions below.
 *
 * Guard: PTX path active when __CUDA_ARCH__ >= 750 (sm_75/Turing+).
 * C fallback used on host and for sm < 75 device code.
 */
#pragma once
#include <cstdint>
#include <cstring>

/* ── LUT constants ────────────────────────────────────────────────────────
 * lop3.b32 truth tables: immLut bit (a<<2 | b<<1 | c) = output bit.
 * Enumeration: 000->0, 001->0, 010->0, 011->1, 100->1, 101->0, 110->1, 111->0
 *   binary 0b10010110 = 0x96   => XOR of three bits  (a^b^c)
 * Enumeration: 000->0, 001->0, 010->0, 011->1, 100->0, 101->1, 110->1, 111->1
 *   binary 0b11101000 = 0xE8   => majority(a,b,c)  = (a&b)|(b&c)|(a&c)
 */
#define PTX_LUT_XOR3     0x96u   /* a^b^c            — verified: 0b10010110 */
#define PTX_LUT_MAJORITY 0xE8u   /* majority(a,b,c)  — verified: 0b11101000 */

/* ── ptx_xor3 ─────────────────────────────────────────────────────────── */

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750

__device__ __forceinline__
uint32_t ptx_xor3(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t r;
    asm volatile ("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}

__device__ __forceinline__
uint32_t ptx_majority(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t r;
    asm volatile ("lop3.b32 %0, %1, %2, %3, 0xe8;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}

__device__ __forceinline__
uint32_t ptx_prmt(uint32_t a, uint32_t b, uint32_t selector) {
    uint32_t r;
    asm volatile ("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(selector));
    return r;
}

#else  /* host / sm < 75: C fallback — same semantics, no PTX */

__host__ __device__ __forceinline__
uint32_t ptx_xor3(uint32_t a, uint32_t b, uint32_t c) {
    return a ^ b ^ c;
}

__host__ __device__ __forceinline__
uint32_t ptx_majority(uint32_t a, uint32_t b, uint32_t c) {
    return (a & b) | (b & c) | (a & c);
}

/*
 * ptx_prmt C fallback — matches PTX prmt.b32 (default/b4e mode).
 * Each output byte i is selected by nibble i of selector:
 *   bits[2:0] = source byte index (0-3 from a, 4-7 from b)
 *   bit[3]    = sign-extension flag: 0=copy byte, 1=replicate MSB as 0x00/0xFF
 * Note: for test inputs with MSB=0, sign-ext produces 0x00.
 */
__host__ __device__ __forceinline__
uint32_t ptx_prmt(uint32_t a, uint32_t b, uint32_t selector) {
    const uint8_t *src_a = (const uint8_t *)&a;
    const uint8_t *src_b = (const uint8_t *)&b;
    uint8_t out[4];
    for (int i = 0; i < 4; i++) {
        uint8_t s   = (uint8_t)((selector >> (4 * i)) & 0xFu);
        uint8_t idx = s & 0x7u;
        uint8_t byte_val = (idx < 4) ? src_a[idx] : src_b[idx - 4];
        /* sign-extension: bit 3 set => replicate byte MSB */
        out[i] = (s & 0x8u) ? ((byte_val & 0x80u) ? 0xFFu : 0x00u) : byte_val;
    }
    uint32_t r;
    memcpy(&r, out, 4);
    return r;
}

#endif  /* __CUDA_ARCH__ >= 750 */

/* ── Wrappers (host + device; dispatch happens inside ptx_prmt/ptx_xor3) ─ */

__host__ __device__ __forceinline__
uint32_t ptx_kste_extract(uint32_t packed_word, uint32_t selector) {
    return ptx_prmt(packed_word, 0u, selector);
}

__host__ __device__ __forceinline__
uint32_t ptx_xxh3_mix3(uint32_t a, uint32_t b, uint32_t c) {
    return ptx_xor3(a, b, c);
}

/* ── sp_sieve_hash_ptx ───────────────────────────────────────────────────── *
 * KSTE sieve mixing round — GPU analog of sp_avx512_ternlog_kste_round.     *
 * Applies the XOR3 step (imm8=0x96) to one lane of the 16-lane KSTE state:  *
 *   result = state[lane] ^ state[(lane+1)%16] ^ state[(lane+5)%16]          *
 *                                                                             *
 * Launch with 16 threads per KSTE tree.  Caller gathers lane_plus1_val and  *
 * lane_plus5_val via __shfl_sync or shared memory before calling.           *
 *                                                                             *
 * CPU equivalent: sp_avx512_ternlog_kste_round (§18.4, imm8=0x96).          *
 * Phase 5 PoUW gate: M_POUW_2 / bench_sieve_hw.c.                           */
__host__ __device__ __forceinline__
uint32_t sp_sieve_hash_ptx(uint32_t lane_val,
                            uint32_t lane_plus1_val,
                            uint32_t lane_plus5_val) {
    return ptx_xor3(lane_val, lane_plus1_val, lane_plus5_val);
}
