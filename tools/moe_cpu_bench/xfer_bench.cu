/* xfer_bench — Apex step 1, t_xfer (2026-10-20): one Gemma-4 26B expert (3,345,408 B, OK_Q4B codes +
 * scales) host -> device from PINNED memory, timed with events. Back-to-back (the link's steady rate)
 * and one-at-a-time with a sync between (what a lone miss pays). Pool of 64 experts, so the host
 * side is not cache-warm. Usage: xfer_bench.exe [reps=256] */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
int main(int argc, char **argv) {
    const int REPS = argc > 1 ? atoi(argv[1]) : 256;
    const size_t B = 3345408, NP = 64;
    unsigned char *h = nullptr, *d = nullptr;
    if (cudaMallocHost((void **)&h, B * NP) != cudaSuccess || cudaMalloc((void **)&d, B * 4) != cudaSuccess) {
        printf("alloc failed\n"); return 1; }
    memset(h, 0x5a, B * NP);
    cudaStream_t st; cudaStreamCreate(&st);
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    for (int i = 0; i < 16; i++) cudaMemcpyAsync(d + (i % 4) * B, h + (i % NP) * B, B, cudaMemcpyHostToDevice, st);
    cudaStreamSynchronize(st);
    cudaEventRecord(a, st);
    for (int i = 0; i < REPS; i++) cudaMemcpyAsync(d + (i % 4) * B, h + (i % NP) * B, B, cudaMemcpyHostToDevice, st);
    cudaEventRecord(b, st); cudaEventSynchronize(b);
    float ms = 0; cudaEventElapsedTime(&ms, a, b);
    printf("back-to-back: %.3f ms/expert (%.2f GB/s)\n", ms / REPS, (double)B * REPS / (ms * 1e6));
    double tot = 0;
    for (int i = 0; i < REPS; i++) {
        cudaEventRecord(a, st);
        cudaMemcpyAsync(d + (i % 4) * B, h + (i % NP) * B, B, cudaMemcpyHostToDevice, st);
        cudaEventRecord(b, st); cudaEventSynchronize(b);
        float m1 = 0; cudaEventElapsedTime(&m1, a, b); tot += m1;
    }
    printf("one at a time: %.3f ms/expert (%.2f GB/s)\n", tot / REPS, (double)B * REPS / (tot * 1e6));
    return 0;
}
