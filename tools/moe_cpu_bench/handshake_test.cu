/* P0 of DESIGN-2026-09-29-host-free-layer: how cheap is a GPU<->host handshake through MAPPED
 * pinned memory on this card under WDDM, compared with what the decode pays per layer today
 * (a cudaStreamSynchronize wake + fresh launches)?
 *
 *   T0  today's turnaround: launch a tiny kernel, cudaStreamSynchronize, repeat.
 *   T1  ping-pong: ONE kernel publishes seq=i to mapped memory, the host thread sees it and
 *       answers, the kernel spins until it sees the answer. Round trip per iteration.
 *   T2  the design's layer, 30 per "token", as one captured graph:
 *         publish(L) -> work(~90 us, the resident experts) -> wait(done >= L) -> combine(~5 us)
 *       the host thread polls the published seq, optionally burns CPU for the layer's "misses",
 *       and posts done=L. Overhead = graph time - pure GPU work. With and without a
 *       cudaStreamQuery after the launch (WDDM may hold the submission until the queue flushes).
 *   T3  zero-copy read of 22 KB (two CPU expert rows) from mapped memory inside a kernel.
 * Every GPU spin is bounded by %globaltimer (200 ms) and every host spin by 2 s, so a lost
 * handshake reports instead of hanging into a TDR. */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <algorithm>
typedef std::chrono::steady_clock clk;
static double us_since(clk::time_point t) { return std::chrono::duration<double, std::micro>(clk::now() - t).count(); }
__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
#define TIMEOUT_NS 50000000ull   /* 50 ms per wait; every kernel here stays far below the 2 s TDR */

__global__ void k_tiny(int *x) { if (threadIdx.x == 0) x[0]++; }

/* T1: one thread, N round trips; records the GPU-side ns per round trip */
__global__ void k_pingpong(volatile int *pub, volatile int *ans, int n, unsigned long long *rt, int *err) {
    /* WDDM kills any kernel that runs > 2 s (TDR), and this machine does not always recover from
     * one: the whole kernel is capped at 300 ms, not just each wait. */
    const unsigned long long tstart = gtime();
    for (int i = 1; i <= n; i++) {
        if (gtime() - tstart > 300000000ull) { *err = -i; return; }
        const unsigned long long t0 = gtime();
        *pub = i; __threadfence_system();
        while (*ans < i) { if (gtime() - t0 > TIMEOUT_NS) { *err = i; return; } }
        rt[i - 1] = gtime() - t0;
    }
}
/* T2 pieces */
__global__ void k_publish(volatile int *pub, int L) { if (threadIdx.x == 0 && blockIdx.x == 0) { *pub = L; __threadfence_system(); } }
__global__ void k_work(long long ns, float *sink) {       /* a full-card stand-in for the hit GEMVs */
    const unsigned long long t0 = gtime(); float a = threadIdx.x;
    while (gtime() - t0 < (unsigned long long)ns) a = a * 1.0001f + 1.0f;
    if (a == -1.0f) sink[0] = a;
}
__global__ void k_wait(volatile int *done, int L, int *err) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        const unsigned long long t0 = gtime();
        while (*done < L) { if (gtime() - t0 > TIMEOUT_NS) { *err = L; return; } }
    }
}
__global__ void k_combine(const float *rows, int n, float *acc) {  /* reads the CPU rows zero-copy */
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) acc[i] += rows[i];
}
static void pct(const char *name, std::vector<double> v) {
    std::sort(v.begin(), v.end()); const size_t n = v.size();
    printf("  %-44s p50 %8.1f  p90 %8.1f  p99 %8.1f  max %8.1f us  (n=%zu)\n", name,
           v[n / 2], v[n * 9 / 10], v[std::min(n - 1, n * 99 / 100)], v[n - 1], n);
}
int main() {
    cudaSetDeviceFlags(cudaDeviceMapHost);
    cudaDeviceProp pr; cudaGetDeviceProperties(&pr, 0);
    printf("%s, %d SMs, WDDM driver model=%d\n", pr.name, pr.multiProcessorCount, pr.tccDriver ? 0 : 1);
    int *hflags; cudaHostAlloc(&hflags, 4096, cudaHostAllocMapped); memset(hflags, 0, 4096);
    int *dflags; cudaHostGetDevicePointer(&dflags, hflags, 0);
    volatile int *h_pub = hflags, *h_ans = hflags + 32, *h_err = hflags + 64;
    int *d_pub = dflags, *d_ans = dflags + 32, *d_err = dflags + 64;
    const int E = 2816;
    float *hrows; cudaHostAlloc(&hrows, 2 * E * sizeof(float), cudaHostAllocMapped);
    float *drows; cudaHostGetDevicePointer(&drows, hrows, 0);
    for (int i = 0; i < 2 * E; i++) hrows[i] = 1.0f;
    float *acc; cudaMalloc(&acc, 2 * E * sizeof(float)); float *sink; cudaMalloc(&sink, 64);
    int *dx; cudaMalloc(&dx, 64);
    cudaStream_t st; cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);

    /* T0 */
    { std::vector<double> v;
      for (int i = 0; i < 2000; i++) { auto t = clk::now(); k_tiny<<<1, 32, 0, st>>>(dx); cudaStreamSynchronize(st); if (i >= 50) v.push_back(us_since(t)); }
      printf("T0 today's per-layer turnaround (launch + cudaStreamSynchronize):\n"); pct("launch -> sync returns", v);
      std::vector<double> w;
      for (int i = 0; i < 2000; i++) { auto t = clk::now(); k_tiny<<<1, 32, 0, st>>>(dx); if (i >= 50) w.push_back(us_since(t)); cudaStreamSynchronize(st); }
      pct("launch API alone", w); }

    /* T1 */
    { const int N = 500; unsigned long long *drt; cudaMalloc(&drt, N * 8);
      *h_pub = 0; *h_ans = 0; *h_err = 0;
      k_pingpong<<<1, 1, 0, st>>>(d_pub, d_ans, N, drt, d_err);
      cudaStreamQuery(st);                                     /* make sure it is submitted */
      auto tall = clk::now(); int lost = 0;
      for (int i = 1; i <= N; i++) {
          auto t = clk::now();
          while (*h_pub < i) { if (us_since(t) > 2e6) { lost = i; break; } }
          if (lost) break;
          *h_ans = i; std::atomic_thread_fence(std::memory_order_seq_cst);
      }
      cudaStreamSynchronize(st); const double wall = us_since(tall);
      std::vector<unsigned long long> rt(N); cudaMemcpy(rt.data(), drt, N * 8, cudaMemcpyDeviceToHost);
      std::vector<double> v; for (int i = 50; i < N; i++) if (rt[i]) v.push_back(rt[i] / 1000.0);
      if (v.empty()) v.push_back(-1);
      printf("T1 mapped-memory ping-pong (GPU -> host -> GPU), lost=%d gpu_err=%d, wall %.1f us/iter:\n", lost, *h_err, wall / N);
      pct("round trip, GPU clock", v); cudaFree(drt); }

    /* T2 */
    const int NL = 30; const long long WORK_NS = 90000;
    for (int variant = 0; variant < 4; variant++) {
        const bool query = variant & 1;             /* cudaStreamQuery after the launch */
        const bool misses = variant & 2;            /* host burns 110 us on every 3rd layer */
        cudaGraph_t g; cudaGraphExec_t ge;
        cudaStreamBeginCapture(st, cudaStreamCaptureModeRelaxed);
        for (int L = 1; L <= NL; L++) {
            k_publish<<<1, 32, 0, st>>>(d_pub, L);
            k_work<<<pr.multiProcessorCount * 2, 256, 0, st>>>(WORK_NS, sink);
            k_wait<<<1, 32, 0, st>>>(d_ans, L, d_err);
            k_combine<<<24, 256, 0, st>>>(drows, 2 * E, acc);
        }
        cudaStreamEndCapture(st, &g); cudaGraphInstantiate(&ge, g, 0);
        /* the pure-GPU reference: the same work without any handshake */
        cudaGraph_t g0; cudaGraphExec_t ge0;
        cudaStreamBeginCapture(st, cudaStreamCaptureModeRelaxed);
        for (int L = 1; L <= NL; L++) { k_work<<<pr.multiProcessorCount * 2, 256, 0, st>>>(WORK_NS, sink); k_combine<<<24, 256, 0, st>>>(drows, 2 * E, acc); }
        cudaStreamEndCapture(st, &g0); cudaGraphInstantiate(&ge0, g0, 0);
        std::vector<double> tok, ref, lat;
        for (int rep = 0; rep < 60; rep++) {
            auto t = clk::now(); cudaGraphLaunch(ge0, st); cudaStreamSynchronize(st); if (rep >= 5) ref.push_back(us_since(t));
            *h_pub = 0; *h_ans = 0; *h_err = 0; std::atomic_thread_fence(std::memory_order_seq_cst);
            t = clk::now(); cudaGraphLaunch(ge, st); if (query) cudaStreamQuery(st);
            int lost = 0;
            for (int L = 1; L <= NL; L++) {
                auto tw = clk::now();
                while (*h_pub < L) { if (us_since(tw) > 2e6) { lost = L; break; } }
                if (lost) break;
                auto seen = clk::now();
                if (misses && L % 3 == 0) { while (us_since(seen) < 110.0) {} }
                *h_ans = L; std::atomic_thread_fence(std::memory_order_seq_cst);
                if (rep >= 5 && L > 1) lat.push_back(std::chrono::duration<double, std::micro>(seen - tw).count());
            }
            cudaStreamSynchronize(st);
            if (lost || *h_err) { printf("  variant %d rep %d: LOST at L=%d gpu_err=%d\n", variant, rep, lost, *h_err); break; }
            if (rep >= 5) tok.push_back(us_since(t));
        }
        printf("T2 %d layers in one graph, %s, %s:\n", NL, query ? "cudaStreamQuery after launch" : "no query after launch",
               misses ? "host burns 110 us every 3rd layer" : "zero-miss layers");
        if (!tok.empty()) {
            pct("token with handshakes", tok); pct("token, same GPU work, no handshake", ref);
            std::vector<double> ov(tok.size()); for (size_t i = 0; i < tok.size(); i++) ov[i] = (tok[i] - ref[std::min(i, ref.size() - 1)]) / NL;
            pct("overhead per layer (paired)", ov);
            pct("host: wait for publish per layer", lat);
        }
        cudaGraphExecDestroy(ge); cudaGraphDestroy(g); cudaGraphExecDestroy(ge0); cudaGraphDestroy(g0);
    }

    /* T3 */
    { cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1); std::vector<double> v, w;
      float *dcopy; cudaMalloc(&dcopy, 2 * E * sizeof(float));
      for (int i = 0; i < 500; i++) {
          cudaEventRecord(e0, st); k_combine<<<24, 256, 0, st>>>(drows, 2 * E, acc); cudaEventRecord(e1, st);
          cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms, e0, e1); if (i >= 20) v.push_back(ms * 1000.0);
          cudaEventRecord(e0, st); k_combine<<<24, 256, 0, st>>>(dcopy, 2 * E, acc); cudaEventRecord(e1, st);
          cudaEventSynchronize(e1); cudaEventElapsedTime(&ms, e0, e1); if (i >= 20) w.push_back(ms * 1000.0);
      }
      printf("T3 combine over 2 x 2816 floats:\n"); pct("rows zero-copy from mapped host memory", v); pct("rows already in VRAM", w); }
    printf("cuda: %s\n", cudaGetErrorString(cudaGetLastError()));
    return 0;
}
