/* moe_cpu_bench — step 2 of DESIGN-2026-09-28-hybrid-moe-decode (2026-09-28).
 *
 * One routed Gemma-4 expert, decode shape (n_tok = 1), computed two ways on the SAME OK_Q4B
 * bytes and the SAME input:
 *   GPU: verbatim copies of k_quant_act_int8, k_gemv_q4b_dp4a_v2, k_moe_gelu_split_batch
 *        (engine/src/backends/cuda/cuda_forward.cu) — the path decode serves today.
 *   CPU: AVX-512 VNNI. Same activation quant (per-16 maxabs/127, round-to-nearest-even,
 *        clamp +-127 -> the int8 codes are IDENTICAL), same signed-nibble weights, the int32
 *        dot per 16-block exact via vpdpbusd on (nib^8) with a -8*sum(q) correction; only the
 *        float accumulation order differs.
 * Reports relL2(CPU vs GPU) on the expert output, and CPU time per expert by thread count
 * with weights cold (a pool larger than the LLC, walked round-robin).
 *
 * Layout (engine OK_Q4B): row j codes at row_off[j], in/2 bytes, byte k = weights 2k (low
 * nibble) and 2k+1 (high); nibble is 4-bit two's complement; per-32 f16 block scale. */
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <immintrin.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <random>
#include <algorithm>
#include <functional>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include "../../src/backends/cuda/moe_cpu_expert.h"

static const int E = 2816, FFE = 704;

/* ---------------- GPU: verbatim from cuda_forward.cu ---------------- */
__global__ void k_quant_act_int8(const float *x, int n, int npad, signed char *qx, float *sxb) {
    const int nblk = npad >> 4;
    for (int b = threadIdx.x; b < nblk; b += blockDim.x) {
        const int base = b << 4;
        float m = 0.0f;
        for (int i = 0; i < 16; i++) { const int idx = base + i;
            const float a = (idx < n) ? fabsf(x[idx]) : 0.0f; if (a > m) m = a; }
        const float scale = m > 0.0f ? m * (1.0f / 127.0f) : 1.0f;
        sxb[b] = scale;
        const float inv = 1.0f / scale;
        for (int i = 0; i < 16; i++) { const int idx = base + i;
            float v = (idx < n) ? x[idx] * inv : 0.0f;
            int q = __float2int_rn(v); if (q > 127) q = 127; if (q < -127) q = -127;
            qx[idx] = (signed char)q; }
    }
}
__device__ __forceinline__ void sp_unpack8(int w, int &lo4, int &hi4) {
    int b0=w&0xFF, b1=(w>>8)&0xFF, b2=(w>>16)&0xFF, b3=(w>>24)&0xFF;
    #define SPX(byte,hi) (((((((hi)?((byte)>>4):(byte)))&0xF)^0x8)-0x8)&0xFF)
    lo4 = SPX(b0,0) | (SPX(b0,1)<<8) | (SPX(b1,0)<<16) | (SPX(b1,1)<<24);
    hi4 = SPX(b2,0) | (SPX(b2,1)<<8) | (SPX(b3,0)<<16) | (SPX(b3,1)<<24);
    #undef SPX
}
__global__ void k_gemv_q4b_dp4a_v2(const unsigned char *codes, const unsigned long long *row_off,
                                   const unsigned short *bscale, int bs_nblk, int in,
                                   const signed char *qx, const float *sxb, float *y, int out) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int j = blockIdx.x * (blockDim.x >> 5) + warp;
    if (j >= out) return;
    const int4 *wrow = (const int4 *)(codes + row_off[j]);
    const int4 *qxi  = (const int4 *)qx;
    const unsigned short *bs = bscale + (size_t)j * (size_t)bs_nblk;
    const int n32 = in >> 5;
    float facc = 0.0f;
    int a, b;
    for (int c = lane; c < n32; c += 32) {
        int4 wv = wrow[c];
        int4 q0 = qxi[2*c], q1 = qxi[2*c + 1];
        int a0 = 0, a1 = 0;
        sp_unpack8(wv.x, a, b); a0 = __dp4a(a, q0.x, a0); a0 = __dp4a(b, q0.y, a0);
        sp_unpack8(wv.y, a, b); a0 = __dp4a(a, q0.z, a0); a0 = __dp4a(b, q0.w, a0);
        sp_unpack8(wv.z, a, b); a1 = __dp4a(a, q1.x, a1); a1 = __dp4a(b, q1.y, a1);
        sp_unpack8(wv.w, a, b); a1 = __dp4a(a, q1.z, a1); a1 = __dp4a(b, q1.w, a1);
        float wbsc = __half2float(__ushort_as_half(__ldg(&bs[c])));
        facc += wbsc * ((float)a0 * sxb[2*c] + (float)a1 * sxb[2*c + 1]);
    }
    for (int o = 16; o > 0; o >>= 1) facc += __shfl_down_sync(0xffffffffu, facc, o);
    if (lane == 0) y[j] = facc;
}
__global__ void k_moe_gelu_split_batch(const float *gu, int ffe, int nrows, float *h) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nrows * ffe) return;
    int r = i / ffe, c = i % ffe;
    const float *g = gu + (size_t)r * (size_t)(2 * ffe);
    float x = g[c];
    const float k = 0.7978845608028654f;
    float th = tanhf(k * (x + 0.044715f * x * x * x));
    h[i] = 0.5f * x * (1.0f + th) * g[ffe + c];
}


static uint16_t f2h(float f) { return __half_as_ushort(__float2half(f)); }
int main(int argc, char **argv) {
    const int NPOOL = argc > 1 ? atoi(argv[1]) : 192;
    if (!sp_moe_cpu_init(4)) { printf("no AVX-512 VNNI\n"); return 1; }
    std::mt19937 rng(1234); std::normal_distribution<float> nd(0.0f, 1.0f);
    const size_t gub = (size_t)2 * FFE * (E / 2), gusb = (size_t)2 * FFE * (E / 32);
    const size_t dwb = (size_t)E * (FFE / 2), dwsb = (size_t)E * (FFE / 32);
    std::vector<uint8_t> codes((gub + dwb) * NPOOL); std::vector<uint16_t> scales((gusb + dwsb) * NPOOL);
    for (auto &c : codes) c = (uint8_t)(rng() & 0xFF);
    for (auto &s : scales) s = f2h(0.004f * (1.0f + 0.3f * fabsf(nd(rng))));
    std::vector<float> x(E); for (auto &v : x) v = nd(rng); for (int i = 0; i < 8; i++) x[rng() % E] *= 25.0f;
    std::vector<unsigned long long> ro_gu(2 * FFE), ro_dw(E);
    for (int j = 0; j < 2 * FFE; j++) ro_gu[j] = (unsigned long long)j * (E / 2);
    for (int j = 0; j < E; j++) ro_dw[j] = (unsigned long long)j * (FFE / 2);
    unsigned char *dgc, *ddc; unsigned short *dgs, *dds; unsigned long long *dro_g, *dro_d;
    float *dx, *dsx, *dgu, *dh, *dout; signed char *dqx;
    cudaMalloc(&dgc, gub); cudaMalloc(&ddc, dwb); cudaMalloc(&dgs, gusb * 2); cudaMalloc(&dds, dwsb * 2);
    cudaMalloc(&dro_g, ro_gu.size() * 8); cudaMalloc(&dro_d, ro_dw.size() * 8);
    cudaMalloc(&dx, E * 4); cudaMalloc(&dsx, E / 16 * 4); cudaMalloc(&dqx, E);
    cudaMalloc(&dgu, 2 * FFE * 4); cudaMalloc(&dh, FFE * 4); cudaMalloc(&dout, E * 4);
    cudaMemcpy(dro_g, ro_gu.data(), ro_gu.size() * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(dro_d, ro_dw.data(), ro_dw.size() * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x.data(), E * 4, cudaMemcpyHostToDevice);
    std::vector<int8_t> cq(E), gq(E); std::vector<float> csx(E / 16), gsx(E / 16);
    sp_moe_cpu_quant(x.data(), E, cq.data(), csx.data());
    std::vector<float> ggu(2 * FFE), cgu(2 * FFE), gout(E), cout_(E);
    long long bad_q = 0, bad_gu = 0, bad_dw = 0;
    for (int e = 0; e < 32; e++) {
        const uint8_t *gc = codes.data() + e * (gub + dwb), *dc = gc + gub;
        const uint16_t *gs = scales.data() + e * (gusb + dwsb), *ds = gs + gusb;
        cudaMemcpy(dgc, gc, gub, cudaMemcpyHostToDevice); cudaMemcpy(dgs, gs, gusb * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(ddc, dc, dwb, cudaMemcpyHostToDevice); cudaMemcpy(dds, ds, dwsb * 2, cudaMemcpyHostToDevice);
        k_quant_act_int8<<<1, 256>>>(dx, E, E, dqx, dsx);
        if (e == 0) { cudaMemcpy(gq.data(), dqx, E, cudaMemcpyDeviceToHost); cudaMemcpy(gsx.data(), dsx, E / 16 * 4, cudaMemcpyDeviceToHost);
            for (int i = 0; i < E; i++) bad_q += gq[i] != cq[i];
            for (int i = 0; i < E / 16; i++) bad_q += memcmp(&gsx[i], &csx[i], 4) != 0; }
        k_gemv_q4b_dp4a_v2<<<(2 * FFE + 7) / 8, 256>>>(dgc, dro_g, dgs, E / 32, E, dqx, dsx, dgu, 2 * FFE);
        cudaMemcpy(ggu.data(), dgu, 2 * FFE * 4, cudaMemcpyDeviceToHost);
        sp_moe_cpu_gemv(gc, gs, E, 2 * FFE, cq.data(), csx.data(), cgu.data());
        for (int i = 0; i < 2 * FFE; i++) bad_gu += memcmp(&ggu[i], &cgu[i], 4) != 0;
        /* GELU + requant stay on the GPU; the CPU takes the GPU's int8 h for the down GEMV */
        k_moe_gelu_split_batch<<<(FFE + 255) / 256, 256>>>(dgu, FFE, 1, dh);
        k_quant_act_int8<<<1, 256>>>(dh, FFE, FFE, dqx, dsx);
        std::vector<int8_t> hq(FFE); std::vector<float> hsx(FFE / 16);
        cudaMemcpy(hq.data(), dqx, FFE, cudaMemcpyDeviceToHost); cudaMemcpy(hsx.data(), dsx, FFE / 16 * 4, cudaMemcpyDeviceToHost);
        k_gemv_q4b_dp4a_v2<<<(E + 7) / 8, 256>>>(ddc, dro_d, dds, FFE / 32, FFE, dqx, dsx, dout, E);
        cudaMemcpy(gout.data(), dout, E * 4, cudaMemcpyDeviceToHost);
        sp_moe_cpu_gemv(dc, ds, FFE, E, hq.data(), hsx.data(), cout_.data());
        for (int i = 0; i < E; i++) bad_dw += memcmp(&gout[i], &cout_[i], 4) != 0;
    }
    printf("BITWISE (32 experts): input quant %lld mismatches; gate_up %lld / %d; down %lld / %d\n",
           bad_q, bad_gu, 32 * 2 * FFE, bad_dw, 32 * E);
    const double mb = (gub + dwb + (gusb + dwsb) * 2) / 1e6;
    std::vector<int8_t> hq(FFE, 3); std::vector<float> hsx(FFE / 16, 0.01f);
    for (int w = 0; w < 8; w++) { sp_moe_cpu_gemv(codes.data(), scales.data(), E, 2 * FFE, cq.data(), csx.data(), cgu.data()); }
    const int N = 3 * NPOOL;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < N; i++) {
        const uint8_t *gc = codes.data() + (size_t)(i % NPOOL) * (gub + dwb), *dc = gc + gub;
        const uint16_t *gs = scales.data() + (size_t)(i % NPOOL) * (gusb + dwsb), *ds = gs + gusb;
        sp_moe_cpu_gemv(gc, gs, E, 2 * FFE, cq.data(), csx.data(), cgu.data());
        sp_moe_cpu_gemv(dc, ds, FFE, E, hq.data(), hsx.data(), cout_.data());
    }
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / N;
    printf("exact path, 4 threads: %.3f ms/expert (gate_up + down, GELU excluded) = %.1f GB/s\n", ms, mb / ms);
    return 0;
}
