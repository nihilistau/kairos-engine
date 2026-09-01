/* cuda_backend.cu — CUDA backend bring-up (Phase 2-CU).
 *
 * CU.0: device query + the cudaError_t -> SP_ECUDA wrapping that honors the
 * frozen L1 ABI error surface (every failing CUDA call sets sp_last_error()).
 * The forward pass (gemma3_forward_cuda) and the kernels land in CU.1/CU.2.
 */
#include "sp_engine/cuda_backend.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

/* internal error setter (defined in src/common/sp_error.c) */
extern "C" void sp_set_error(const char *msg);

/* Wrap a failing cudaError_t: stash "CUDA: <where>: <cudaGetErrorString>" in the
 * thread-local error string and return SP_ECUDA. */
static sp_status sp_cuda_fail(cudaError_t e, const char *where) {
    char buf[512];
    std::snprintf(buf, sizeof(buf), "CUDA: %s: %s (%d)",
                  where, cudaGetErrorString(e), (int)e);
    sp_set_error(buf);
    return SP_ECUDA;
}

extern "C" int sp_cuda_device_count(void) {
    int n = 0;
    cudaError_t e = cudaGetDeviceCount(&n);
    if (e != cudaSuccess) { sp_cuda_fail(e, "cudaGetDeviceCount"); return 0; }
    return n;
}

extern "C" sp_status sp_cuda_device_info(int dev, char *name, int name_cap,
                                         int *sm_major, int *sm_minor) {
    cudaDeviceProp p;
    cudaError_t e = cudaGetDeviceProperties(&p, dev);
    if (e != cudaSuccess) return sp_cuda_fail(e, "cudaGetDeviceProperties");
    if (name && name_cap > 0) {
        std::strncpy(name, p.name, (size_t)name_cap - 1);
        name[name_cap - 1] = '\0';
    }
    if (sm_major) *sm_major = p.major;
    if (sm_minor) *sm_minor = p.minor;
    return SP_OK;
}
