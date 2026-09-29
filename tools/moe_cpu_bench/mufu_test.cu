/* Can the CPU reproduce the decode GELU bit for bit? CUDA tanhf on sm_75 compiles to
 *   y2=|y|*2.88539; e=MUFU.EX2(y2); r=MUFU.RCP(e+1); th=fma(r,-2,1); ... (see bench_sass.txt)
 * Hypotheses tested here, exhaustively over the input ranges the GELU can produce:
 *   H1  MUFU.RCP depends only on the mantissa: rcp(m*2^k) == rcp(m)*2^-k  (d = e+1 in [6.6, 2.3e7])
 *   H2  MUFU.EX2 splits: ex2(i+f) == ex2(f)*2^i                            (x in [1.73, 26])
 * If both hold, two 2^23-entry tables (built once on the GPU) make the CPU GELU exact.
 * Last check: the full GELU expression, CPU-emulated with those tables, vs the GPU kernel. */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
__device__ __forceinline__ float mufu_rcp(float x) { float r; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x)); return r; }
__device__ __forceinline__ float mufu_ex2(float x) { float r; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x)); return r; }
__global__ void k_tables(float *trcp, float *tex2) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= (1u << 23)) return;
    trcp[i] = mufu_rcp(__uint_as_float(0x3f800000u | i));            /* m in [1,2) */
    tex2[i] = mufu_ex2((float)i * (1.0f / 8388608.0f));             /* f in [0,1), step 2^-23 */
}
__global__ void k_h1(const float *trcp, unsigned long long *bad) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= (1u << 23)) return;
    const float t = trcp[i];
    for (int k = 2; k <= 25; k++) {
        const float x = __uint_as_float(((unsigned)(127 + k) << 23) | i);
        const float r = mufu_rcp(x);
        const float want = __uint_as_float(__float_as_uint(t) - ((unsigned)k << 23));
        if (__float_as_uint(r) != __float_as_uint(want)) atomicAdd(bad, 1ull);
    }
}
__global__ void k_h2(const float *tex2, unsigned lo, unsigned n, unsigned long long *bad, unsigned long long *checked) {
    const unsigned j = blockIdx.x * blockDim.x + threadIdx.x; if (j >= n) return;
    const float x = __uint_as_float(lo + j);
    const int ip = (int)floorf(x); const float f = x - (float)ip;        /* exact for x < 2^24 */
    const unsigned fi = (unsigned)(f * 8388608.0f);
    if ((float)fi != f * 8388608.0f) return;                              /* f not on the 2^-23 grid (never, here) */
    const float want = __uint_as_float(__float_as_uint(tex2[fi]) + ((unsigned)ip << 23));
    const float r = mufu_ex2(x);
    atomicAdd(checked, 1ull);
    if (__float_as_uint(r) != __float_as_uint(want)) atomicAdd(bad, 1ull);
}
__global__ void k_gelu(const float *x, const float *up, float *h, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    float xv = x[i]; const float k = 0.7978845608028654f;
    float th = tanhf(k * (xv + 0.044715f * xv * xv * xv));
    h[i] = 0.5f * xv * (1.0f + th) * up[i];
}
/* CPU replica of the SASS in bench_sass.txt (k_moe_gelu_split_batch), with the two tables */
static std::vector<float> T_RCP, T_EX2;
static inline float u2f(unsigned u) { float f; memcpy(&f, &u, 4); return f; }
static inline unsigned f2u(float f) { unsigned u; memcpy(&u, &f, 4); return u; }
static float cpu_rcp(float d) { unsigned u = f2u(d); int e = (int)((u >> 23) & 0xFF) - 127; float t = T_RCP[u & 0x7FFFFF];
    return u2f(f2u(t) - ((unsigned)e << 23)); }
static float cpu_ex2(float x) { int ip = (int)floorf(x); float f = x - (float)ip; unsigned fi = (unsigned)(f * 8388608.0f);
    return u2f(f2u(T_EX2[fi]) + ((unsigned)ip << 23)); }
static float cpu_gelu(float x, float up) {
    float r3 = x * 0.044714998453855514526f;
    r3 = x * r3;
    r3 = fmaf(x, r3, x);
    const float r4 = x * 0.5f;
    const float y = r3 * 0.79788458347320556641f;
    const float ay = fabsf(y);
    float th;
    if (ay < 0.60000002384185791016f) {
        const float y2 = y * y;
        float p = fmaf(y2, u2f(0x3c80f082u), -0.052303962409496307373f);
        p = fmaf(y2, p, 0.1331529766321182251f);
        p = fmaf(y2, p, -0.33332768082618713379f);
        p = fmaf(y2, p, 0.0f);
        th = fmaf(y, p, y);
    } else {
        const float e = cpu_ex2(ay * 2.8853900432586669922f);
        const float r = cpu_rcp(e + 1.0f);
        float t = fmaf(r, -2.0f, 1.0f);
        if (ay >= 9.010913848876953125f) t = 1.0f;
        th = u2f((f2u(t) & 0x7FFFFFFFu) | (f2u(y) & 0x80000000u));
    }
    return (r4 * (th + 1.0f)) * up;
}
int main() {
    float *trcp, *tex2; unsigned long long *bad, *chk;
    cudaMalloc(&trcp, 4u << 23); cudaMalloc(&tex2, 4u << 23); cudaMalloc(&bad, 8); cudaMalloc(&chk, 8);
    k_tables<<<(1u << 23) / 256, 256>>>(trcp, tex2);
    cudaMemset(bad, 0, 8);
    k_h1<<<(1u << 23) / 256, 256>>>(trcp, bad);
    unsigned long long b = 0, c = 0; cudaMemcpy(&b, bad, 8, cudaMemcpyDeviceToHost);
    printf("H1 rcp(m*2^k)==rcp(m)*2^-k over 2^23 mantissas x k=2..25: %llu mismatches\n", b);
    cudaMemset(bad, 0, 8); cudaMemset(chk, 0, 8);
    const unsigned lo = f2u(1.70f), hi = f2u(26.5f);
    k_h2<<<(hi - lo + 255) / 256, 256>>>(tex2, lo, hi - lo, bad, chk);
    cudaMemcpy(&b, bad, 8, cudaMemcpyDeviceToHost); cudaMemcpy(&c, chk, 8, cudaMemcpyDeviceToHost);
    printf("H2 ex2(i+f)==ex2(f)*2^i over every float in [1.70, 26.5]: %llu mismatches of %llu\n", b, c);
    T_RCP.resize(1u << 23); T_EX2.resize(1u << 23);
    cudaMemcpy(T_RCP.data(), trcp, 4u << 23, cudaMemcpyDeviceToHost); cudaMemcpy(T_EX2.data(), tex2, 4u << 23, cudaMemcpyDeviceToHost);
    const int n = 1 << 24;
    std::vector<float> x(n), up(n), h(n);
    std::mt19937 r(11); std::normal_distribution<float> nd(0.f, 2.0f);
    for (int i = 0; i < n; i++) { x[i] = nd(r); up[i] = nd(r); }
    for (int i = 0; i < 4096; i++) x[i] = (i - 2048) * 0.01f;          /* dense sweep near 0 and the 0.6 switch */
    float *dx, *du, *dh; cudaMalloc(&dx, (size_t)n * 4); cudaMalloc(&du, (size_t)n * 4); cudaMalloc(&dh, (size_t)n * 4);
    cudaMemcpy(dx, x.data(), (size_t)n * 4, cudaMemcpyHostToDevice); cudaMemcpy(du, up.data(), (size_t)n * 4, cudaMemcpyHostToDevice);
    k_gelu<<<(n + 255) / 256, 256>>>(dx, du, dh, n);
    cudaMemcpy(h.data(), dh, (size_t)n * 4, cudaMemcpyDeviceToHost);
    long long mm = 0;
    for (int i = 0; i < n; i++) { float c2 = cpu_gelu(x[i], up[i]); if (f2u(c2) != f2u(h[i])) mm++; }
    printf("GELU, CPU replica with tables vs GPU kernel, %d inputs: %lld not bit-identical\n", n, mm);
    return 0;
}
