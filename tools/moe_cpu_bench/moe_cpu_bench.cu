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

/* ---------------- CPU ---------------- */
struct Tensor { const uint8_t *codes; const uint16_t *bs; int rows, in; };   /* row stride in/2 */
struct Expert { Tensor gu, dw; };

/* Activation prepared once per input vector, reused by every row of every expert that reads it. */
struct Act {
    int in, ng;                               /* ng = ceil(in/128) zmm groups */
    std::vector<int8_t>  qx;                  /* identical to the GPU's qx */
    std::vector<float>   sx;
    alignas(64) std::vector<__m512i> qe, qo, corr;   /* per group */
    std::vector<__m512>  S;                   /* per group, per lane */
    std::vector<__mmask64> lmask;             /* byte-load mask per group */
};
static void act_prepare(const float *x, int in, Act &A) {
    A.in = in; A.ng = (in + 127) / 128;
    const int npad = (in + 31) & ~31, nblk = npad >> 4;
    A.qx.assign(A.ng * 128, 0); A.sx.assign(A.ng * 8, 0.0f);
    for (int b = 0; b < nblk; b++) {
        float m = 0.0f;
        for (int i = 0; i < 16; i++) { int idx = b * 16 + i; float a = idx < in ? fabsf(x[idx]) : 0.0f; if (a > m) m = a; }
        const float scale = m > 0.0f ? m * (1.0f / 127.0f) : 1.0f;
        A.sx[b] = scale;
        const float inv = 1.0f / scale;
        for (int i = 0; i < 16; i++) { int idx = b * 16 + i;
            float v = idx < in ? x[idx] * inv : 0.0f;
            int q = (int)nearbyintf(v); if (q > 127) q = 127; if (q < -127) q = -127;
            A.qx[idx] = (int8_t)q; }
    }
    A.qe.resize(A.ng); A.qo.resize(A.ng); A.corr.resize(A.ng); A.S.resize(A.ng); A.lmask.resize(A.ng);
    for (int g = 0; g < A.ng; g++) {
        alignas(64) int8_t qe[64], qo[64]; alignas(64) int32_t cr[16]; alignas(64) float S[16];
        for (int p = 0; p < 64; p++) {
            const int chunk = 4 * g + p / 16, k = p % 16;
            const int w0 = 32 * chunk + 2 * k;
            qe[p] = w0 < in ? A.qx[w0] : 0; qo[p] = w0 + 1 < in ? A.qx[w0 + 1] : 0;
        }
        for (int l = 0; l < 16; l++) {
            int s = 0; for (int t = 0; t < 4; t++) s += qe[4 * l + t] + qo[4 * l + t];
            cr[l] = 8 * s;
            const int chunk = 4 * g + l / 4, blk = 2 * chunk + ((l % 4) >= 2);
            S[l] = (32 * chunk < in) ? A.sx[blk] : 0.0f;
        }
        A.qe[g] = _mm512_load_si512(qe); A.qo[g] = _mm512_load_si512(qo);
        A.corr[g] = _mm512_load_si512(cr); A.S[g] = _mm512_load_ps(S);
        const int bytes_left = in / 2 - 64 * g;               /* in%32==0 -> whole chunks */
        A.lmask[g] = bytes_left >= 64 ? ~0ULL : ((1ULL << bytes_left) - 1);
    }
}
static inline void gemv_rows(const Tensor &W, const Act &A, float *y, int r0, int r1) {
    const __m512i m0f = _mm512_set1_epi8(0x0F), m08 = _mm512_set1_epi8(0x08);
    const __m512i widx = _mm512_set_epi32(3,3,3,3, 2,2,2,2, 1,1,1,1, 0,0,0,0);
    const int rb = W.in / 2, nb = W.in / 32;
    for (int j = r0; j < r1; j++) {
        const uint8_t *row = W.codes + (size_t)j * rb;
        const uint16_t *bs = W.bs + (size_t)j * nb;
        __m512 facc = _mm512_setzero_ps();
        for (int g = 0; g < A.ng; g++) {
            const __mmask64 lm = A.lmask[g];
            __m512i v = _mm512_maskz_loadu_epi8(lm, row + 64 * g);
            __m512i lo = _mm512_xor_si512(_mm512_and_si512(v, m0f), m08);
            __m512i hi = _mm512_xor_si512(_mm512_and_si512(_mm512_srli_epi16(v, 4), m0f), m08);
            __m512i acc = _mm512_dpbusd_epi32(_mm512_setzero_si512(), lo, A.qe[g]);
            acc = _mm512_dpbusd_epi32(acc, hi, A.qo[g]);
            acc = _mm512_sub_epi32(acc, A.corr[g]);
            const int nch = std::min(4, nb - 4 * g);
            __m128i h4 = nch == 4 ? _mm_loadl_epi64((const __m128i *)(bs + 4 * g))
                                  : _mm_maskz_loadu_epi16((__mmask8)((1u << nch) - 1), bs + 4 * g);
            __m512 w = _mm512_permutexvar_ps(widx, _mm512_castps128_ps512(_mm_cvtph_ps(h4)));
            facc = _mm512_fmadd_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(acc), A.S[g]), w, facc);
        }
        y[j] = _mm512_reduce_add_ps(facc);
    }
}

/* spin pool: the calling thread is worker 0 */
struct Pool {
    int n; std::vector<std::thread> th; std::atomic<int> gen{0}, done{0}; std::atomic<bool> quit{false};
    std::function<void(int)> job;
    explicit Pool(int n_) : n(n_) {
        for (int i = 1; i < n; i++) th.emplace_back([this, i] {
            SetThreadAffinityMask(GetCurrentThread(), 1ull << (2 * i));   /* one per physical core */
            int seen = 0;
            while (!quit.load(std::memory_order_acquire)) {
                int gg = gen.load(std::memory_order_acquire);
                if (gg == seen) { _mm_pause(); continue; }
                seen = gg; job(i); done.fetch_add(1, std::memory_order_acq_rel);
            } });
    }
    void run(std::function<void(int)> f) {
        job = f; done.store(0); gen.fetch_add(1, std::memory_order_acq_rel);
        job(0);
        while (done.load(std::memory_order_acquire) < n - 1) _mm_pause();
    }
    ~Pool() { quit = true; for (auto &t : th) t.join(); }
};

static inline float gelu_mul(float x, float up) {
    const float k = 0.7978845608028654f;
    float th = tanhf(k * (x + 0.044715f * x * x * x));
    return 0.5f * x * (1.0f + th) * up;
}
/* one expert, rows split across the pool */
static void cpu_expert(Pool &P, const Expert &X, const Act &Ain, float *gu, float *h, Act &Ah, float *out) {
    const int R1 = X.gu.rows;
    P.run([&](int t) { int a = R1 * t / P.n, b = R1 * (t + 1) / P.n; gemv_rows(X.gu, Ain, gu, a, b); });
    for (int c = 0; c < FFE; c++) h[c] = gelu_mul(gu[c], gu[FFE + c]);
    act_prepare(h, FFE, Ah);
    const int R2 = X.dw.rows;
    P.run([&](int t) { int a = R2 * t / P.n, b = R2 * (t + 1) / P.n; gemv_rows(X.dw, Ah, out, a, b); });
}

static uint16_t f2h(float f) { return __half_as_ushort(__float2half(f)); }

int main(int argc, char **argv) {
    const int NPOOL = argc > 1 ? atoi(argv[1]) : 192;   /* 192 experts x 2.97 MB ~ 570 MB */
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const size_t gub = (size_t)2 * FFE * (E / 2), gusb = (size_t)2 * FFE * (E / 32);
    const size_t dwb = (size_t)E * (FFE / 2), dwsb = (size_t)E * (FFE / 32);
    std::vector<uint8_t> codes((gub + dwb) * NPOOL);
    std::vector<uint16_t> scales((gusb + dwsb) * NPOOL);
    for (auto &c : codes) c = (uint8_t)(rng() & 0xFF);
    for (auto &s : scales) s = f2h(0.004f * (1.0f + 0.3f * fabsf(nd(rng))));
    std::vector<Expert> X(NPOOL);
    for (int e = 0; e < NPOOL; e++) {
        X[e].gu = { codes.data() + e * (gub + dwb), scales.data() + e * (gusb + dwsb), 2 * FFE, E };
        X[e].dw = { codes.data() + e * (gub + dwb) + gub, scales.data() + e * (gusb + dwsb) + gusb, E, FFE };
    }
    std::vector<float> x(E);
    for (auto &v : x) v = nd(rng);
    for (int i = 0; i < 8; i++) x[rng() % E] *= 25.0f;    /* a few outliers, as residual streams have */

    /* ---- parity on 16 experts ---- */
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

    Pool P1(1);
    Act Ain, Ah; act_prepare(x.data(), E, Ain);
    std::vector<float> gu(2 * FFE), h(FFE), out(E), ref(E), refgu(2 * FFE);
    std::vector<signed char> gqx(E);
    double worst = 0.0, worst_gu = 0.0; int qx_mismatch = 0;
    for (int e = 0; e < 16; e++) {
        cudaMemcpy(dgc, X[e].gu.codes, gub, cudaMemcpyHostToDevice);
        cudaMemcpy(dgs, X[e].gu.bs, gusb * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(ddc, X[e].dw.codes, dwb, cudaMemcpyHostToDevice);
        cudaMemcpy(dds, X[e].dw.bs, dwsb * 2, cudaMemcpyHostToDevice);
        k_quant_act_int8<<<1, 256>>>(dx, E, E, dqx, dsx);
        k_gemv_q4b_dp4a_v2<<<(2 * FFE + 7) / 8, 256>>>(dgc, dro_g, dgs, E / 32, E, dqx, dsx, dgu, 2 * FFE);
        k_moe_gelu_split_batch<<<(FFE + 255) / 256, 256>>>(dgu, FFE, 1, dh);
        k_quant_act_int8<<<1, 256>>>(dh, FFE, FFE, dqx, dsx);
        k_gemv_q4b_dp4a_v2<<<(E + 7) / 8, 256>>>(ddc, dro_d, dds, FFE / 32, FFE, dqx, dsx, dout, E);
        cudaMemcpy(ref.data(), dout, E * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(refgu.data(), dgu, 2 * FFE * 4, cudaMemcpyDeviceToHost);
        if (e == 0) {   /* the activation codes must be identical, not close */
            k_quant_act_int8<<<1, 256>>>(dx, E, E, dqx, dsx);
            cudaMemcpy(gqx.data(), dqx, E, cudaMemcpyDeviceToHost);
            for (int i = 0; i < E; i++) qx_mismatch += gqx[i] != Ain.qx[i];
        }
        cpu_expert(P1, X[e], Ain, gu.data(), h.data(), Ah, out.data());
        double n = 0, d = 0, n2 = 0, d2 = 0;
        for (int i = 0; i < E; i++) { double t = (double)out[i] - ref[i]; n += t * t; d += (double)ref[i] * ref[i]; }
        for (int i = 0; i < 2 * FFE; i++) { double t = (double)gu[i] - refgu[i]; n2 += t * t; d2 += (double)refgu[i] * refgu[i]; }
        worst = std::max(worst, sqrt(n / d)); worst_gu = std::max(worst_gu, sqrt(n2 / d2));
    }
    printf("PARITY (16 experts): activation int8 codes mismatched %d/%d; worst relL2 gate_up %.3e, expert output %.3e\n",
           qx_mismatch, E, worst_gu, worst);

    /* ---- timing: cold weights, round-robin over the pool ---- */
    const double mb_per = (gub + dwb + (gusb + dwsb) * 2) / 1e6;
    for (int nt : {1, 2, 4, 6, 8}) {
        Pool P(nt);
        SetThreadAffinityMask(GetCurrentThread(), 1ull);
        for (int e = 0; e < 8; e++) cpu_expert(P, X[e], Ain, gu.data(), h.data(), Ah, out.data());
        const int N = 3 * NPOOL;
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < N; i++) cpu_expert(P, X[i % NPOOL], Ain, gu.data(), h.data(), Ah, out.data());
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / N;
        printf("threads %d: %.3f ms/expert  (%.1f GB/s effective on %.2f MB)\n", nt, ms, mb_per / ms, mb_per);
    }
    return 0;
}
