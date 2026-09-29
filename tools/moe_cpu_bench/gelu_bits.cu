/* Bitwise check: the GPU decode GELU (k_moe_gelu_split_batch's expression, CUDA tanhf) vs the
 * same expression on the CPU (MSVC tanhf). 4M inputs spanning the gate range seen in decode. */
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <random>
__global__ void k_gelu(const float *x, const float *up, float *h, float *th_out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    float xv = x[i];
    const float k = 0.7978845608028654f;
    float th = tanhf(k * (xv + 0.044715f * xv * xv * xv));
    th_out[i] = th;
    h[i] = 0.5f * xv * (1.0f + th) * up[i];
}
int main() {
    const int n = 1 << 22;
    std::vector<float> x(n), up(n), h(n), th(n);
    std::mt19937 r(7); std::normal_distribution<float> nd(0.f, 1.5f);
    for (int i = 0; i < n; i++) { x[i] = nd(r); up[i] = nd(r); }
    float *dx, *du, *dh, *dt; cudaMalloc(&dx, n*4); cudaMalloc(&du, n*4); cudaMalloc(&dh, n*4); cudaMalloc(&dt, n*4);
    cudaMemcpy(dx, x.data(), n*4, cudaMemcpyHostToDevice); cudaMemcpy(du, up.data(), n*4, cudaMemcpyHostToDevice);
    k_gelu<<<(n+255)/256, 256>>>(dx, du, dh, dt, n);
    cudaMemcpy(h.data(), dh, n*4, cudaMemcpyDeviceToHost); cudaMemcpy(th.data(), dt, n*4, cudaMemcpyDeviceToHost);
    long long bad_arg = 0, bad_th = 0, bad_h = 0, bad_h_fma = 0;
    for (int i = 0; i < n; i++) {
        float xv = x[i]; const float k = 0.7978845608028654f;
        float a = k * (xv + 0.044715f * xv * xv * xv);
        float a2 = k * fmaf(0.044715f * xv * xv, xv, xv);             /* contraction candidate */
        float t = tanhf(a), t2 = tanhf(a2);
        float hv = 0.5f * xv * (1.0f + t) * up[i];
        if (memcmp(&t, &th[i], 4)) bad_th++;
        if (memcmp(&t2, &th[i], 4)) bad_arg++;
        if (memcmp(&hv, &h[i], 4)) bad_h++;
        float hv2 = 0.5f * xv * (1.0f + t2) * up[i];
        if (memcmp(&hv2, &h[i], 4)) bad_h_fma++;
    }
    printf("n=%d  tanh mismatches: plain %lld (%.4f%%), fma-arg %lld (%.4f%%);  h mismatches: plain %lld, fma-arg %lld\n",
           n, bad_th, 100.0*bad_th/n, bad_arg, 100.0*bad_arg/n, bad_h, bad_h_fma);
    return 0;
}
