/* Does an H2D copy on a second stream run CONCURRENTLY with a kernel on this box (WDDM)?
 * Every decode trace so far shows ~0 copy/kernel overlap. This isolates the platform question
 * from the engine: a ~40 ms busy kernel on stream A, 40 x 2 MB H2D on stream B (issued from a
 * second host thread), from cudaMallocHost memory and from cudaHostRegister'd malloc memory. */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <chrono>
__global__ void k_spin(long long cycles, int *sink) {
    long long t0 = clock64(); while (clock64() - t0 < cycles) {} if (threadIdx.x == 0) sink[blockIdx.x] = 1;
}
static double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count(); }
int main(int argc, char **argv) {
    const size_t REG_GB = argc > 1 ? (size_t)atoi(argv[1]) : 0;
    int *sink; cudaMalloc(&sink, 4096 * 4);
    cudaStream_t sa, sb; cudaStreamCreateWithFlags(&sa, cudaStreamNonBlocking); cudaStreamCreateWithFlags(&sb, cudaStreamNonBlocking);
    const size_t B = 2 << 20; const int N = 40;
    void *dst; cudaMalloc(&dst, B * N);
    void *hp; cudaMallocHost(&hp, B * N);
    const size_t regb = REG_GB ? REG_GB << 30 : B * N;
    char *hrbase = (char *)malloc(regb);
    for (size_t o = 0; o < regb; o += 4096) hrbase[o] = 1;
    cudaError_t re = cudaHostRegister(hrbase, regb, cudaHostRegisterReadOnly);
    printf("registered %.2f GB: %s\n", regb / 1073741824.0, cudaGetErrorString(re));
    void *hr = hrbase + regb - B * N;
    int clk; cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0);
    const long long cyc = (long long)clk * 40;           /* ~40 ms at base clock (kHz * ms) */
    cudaEvent_t a0, a1, b0, b1; cudaEventCreate(&a0); cudaEventCreate(&a1); cudaEventCreate(&b0); cudaEventCreate(&b1);
    cudaEvent_t evk; cudaEventCreateWithFlags(&evk, cudaEventDisableTiming);
    for (int mode = 0; mode < 5; mode++) {
        void *src = mode == 1 ? hr : hp;
        for (int rep = 0; rep < 3; rep++) {
            cudaDeviceSynchronize();
            /* alone: copies */
            auto t = std::chrono::steady_clock::now();
            for (int i = 0; i < N; i++) cudaMemcpyAsync((char *)dst + i * B, (char *)src + i * B, B, cudaMemcpyHostToDevice, sb);
            cudaStreamSynchronize(sb); double copy_alone = ms_since(t);
            t = std::chrono::steady_clock::now();
            k_spin<<<68, 32, 0, sa>>>(cyc, sink); cudaStreamSynchronize(sa); double k_alone = ms_since(t);
            /* together */
            cudaDeviceSynchronize();
            t = std::chrono::steady_clock::now();
            double issue_ms = 0;
            k_spin<<<68, 32, 0, sa>>>(cyc, sink);
            if (mode == 3 || mode == 4) {
                /* the engine's pattern: the copy stream first waits on an event recorded on the
                 * compute stream (slot-reuse hazard), then the per-expert copies + an event each,
                 * and (mode 4) a short kernel queued on the compute stream that waits on that event. */
                auto ti = std::chrono::steady_clock::now();
                cudaEventRecord(evk, sa);
                cudaStreamWaitEvent(sb, evk, 0);
                for (int i = 0; i < N; i++) {
                    cudaMemcpyAsync((char *)dst + i * B, (char *)src + i * B, B, cudaMemcpyHostToDevice, sb);
                    if (mode == 4) { cudaEventRecord(b1, sb); cudaStreamWaitEvent(sa, b1, 0); k_spin<<<1, 32, 0, sa>>>(1000, sink); }
                }
                issue_ms = ms_since(ti);
            } else if (mode == 2) {   /* copies issued from THIS thread right after the launch */
                auto ti = std::chrono::steady_clock::now();
                for (int i = 0; i < N; i++) cudaMemcpyAsync((char *)dst + i * B, (char *)src + i * B, B, cudaMemcpyHostToDevice, sb);
                issue_ms = ms_since(ti);
            } else {
                std::thread th([&] { auto ti = std::chrono::steady_clock::now();
                    for (int i = 0; i < N; i++) cudaMemcpyAsync((char *)dst + i * B, (char *)src + i * B, B, cudaMemcpyHostToDevice, sb);
                    issue_ms = ms_since(ti); });
                th.join();
            }
            cudaDeviceSynchronize(); double both = ms_since(t);
            printf("%-28s rep %d: copies alone %6.1f ms, kernel alone %6.1f ms, together %6.1f ms (issue %5.1f ms) -> overlap %5.1f ms\n",
                   mode == 0 ? "cudaMallocHost, 2nd thread" : mode == 1 ? "HostRegister,   2nd thread" : mode == 2 ? "cudaMallocHost, same thread" : mode == 3 ? "wait-event then copies" : "interleaved evt+kernel",
                   rep, copy_alone, k_alone, both, issue_ms, copy_alone + k_alone - both);
        }
    }
    return 0;
}
