/* moe_cpu_expert — step B of DESIGN-2026-09-28-hybrid-moe-decode: a routed expert computed on
 * the CPU from the pinned host arena, in the GPU decode path's arithmetic (int8 per-16
 * activation quant identical to k_quant_act_int8; OK_Q4B signed-nibble weights with per-32 f16
 * scales; exact int32 dots; float accumulation order differs -> relL2 ~2e-7 measured). */
#ifndef SP_MOE_CPU_EXPERT_H
#define SP_MOE_CPU_EXPERT_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* 1 if this CPU has AVX-512 F/BW/VNNI and the pool started; idempotent. nthreads<=0 -> 4. */
int  sp_moe_cpu_init(int nthreads);
/* BIT-EXACT pieces (2026-09-28, v2). The first cut matched the GPU to relL2 ~2e-6 and still
 * changed greedy text at token ~25, so the served path is now bitwise:
 *   sp_moe_cpu_quant == k_quant_act_int8 (codes AND scales)
 *   sp_moe_cpu_gemv  == k_gemv_q4b_dp4a_v2, including its FFMA contraction
 *     (t = fma(a0, s0, a1*s1) -- read from SASS; facc = fma(wbsc, t, facc)), its lane assignment (chunk c ->
 *     lane c % 32, in order) and its __shfl_down tree (16, 8, 4, 2, 1).
 * GELU is NOT reproduced here: CUDA tanhf uses MUFU.EX2/MUFU.RCP, hardware approximations the
 * CPU cannot match bit for bit, so the engine runs the CPU experts' GELU + requant on the GPU. */
int  sp_moe_cpu_quant(const float *x, int in, int8_t *qx, float *sx);
/* v3 (2026-09-29): the decode GELU, bit-exact. CUDA tanhf's two MUFU ops were characterised
 * exhaustively (engine/tools/moe_cpu_bench/mufu_test.cu): MUFU.RCP depends only on the mantissa
 * (0 mismatches over 2^23 mantissas x 24 binades) and MUFU.EX2(i+f) == MUFU.EX2(f)*2^i (0 over
 * all 33M floats in [1.70, 26.5]). So two 2^23-entry tables, built ONCE on this GPU, plus the
 * SASS op sequence of k_moe_gelu_split_batch replayed with fmaf, reproduce it: 0 of 16.8M
 * inputs differ. The tables are per-GPU-architecture facts: build them on the card that serves. */
void sp_moe_cpu_set_mufu_tables(const float *rcp_by_mantissa, const float *ex2_by_frac23);
int  sp_moe_cpu_gelu(const float *gu, int ffe, float *h);   /* h[c] = gelu(gu[c]) * gu[ffe+c] */
int  sp_moe_cpu_gemv(const uint8_t *codes, const uint16_t *bs, int in, int rows,
                     const int8_t *qx, const float *sx, float *y);

/* One expert, decode shape (v1, approximate GELU; kept for the bench). x[E] input (post pre_ffw_norm_2); gate_up: 2*FFE rows x E, row
 * stride E/2 bytes, scales [2*FFE][E/32]; down: E rows x FFE, stride FFE/2, scales [E][FFE/32].
 * out[E] = the raw down-projection (routing weight and expert scale NOT applied — same as the
 * GPU's per-expert output). Runs on the calling thread + the pool. E%32==0, FFE%32==0,
 * E<=4096, 2*FFE<=4096. Returns 0 on success. */
int  sp_moe_cpu_expert(const float *x, int E, int FFE,
                       const uint8_t *gu_codes, const uint16_t *gu_bs,
                       const uint8_t *dw_codes, const uint16_t *dw_bs,
                       float *out);
#ifdef __cplusplus
}
#endif
#endif
