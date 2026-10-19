/* vram_budget_win.c — this process's real VRAM budget on Windows (2026-10-19).
 *
 * WHY. On WDDM, cudaMemGetInfo reports free = 0 for this process (measured 2026-07-13, see
 * the batched-prefill guard in cuda_forward.cu), so the engine has no number for how much
 * VRAM it may still take. The binding limit on WDDM is the per-process BUDGET the OS hands
 * out, and it moves with everything else on the card (the desktop, an RDP session, a
 * browser). DXGI reports it: IDXGIAdapter3::QueryVideoMemoryInfo, local segment group,
 * Budget and CurrentUsage. Two cold-boot failures of the first batched prefill
 * (2026-10-19, `cublasSgemm FAILED status=13` then `out of memory`) carried no such number,
 * so this exists to make the next one carry it.
 *
 * dxgi.dll is loaded at run time, so nothing changes at link. The adapter is matched to
 * the CUDA device by LUID. Read-only; returns -1 when anything is unavailable, and every
 * caller treats -1 as "unknown". Non-Windows builds get the stub. */
#include <stdint.h>
#include <string.h>

#ifdef _WIN32
#define COBJMACROS
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <initguid.h>
#include <dxgi1_4.h>

static IDXGIAdapter3 *g_ad = NULL;
static int g_tried = 0;

int sp_vram_budget_mib(const unsigned char luid[8], long long *budget_mib, long long *usage_mib) {
    if (!g_tried) {
        g_tried = 1;
        HMODULE h = LoadLibraryA("dxgi.dll");
        if (!h) return -1;
        typedef HRESULT (WINAPI *create_fn)(REFIID, void **);
        create_fn cf = (create_fn)(void *)GetProcAddress(h, "CreateDXGIFactory1");
        IDXGIFactory1 *f = NULL;
        if (!cf || FAILED(cf(&IID_IDXGIFactory1, (void **)&f)) || !f) return -1;
        IDXGIAdapter1 *a = NULL;
        for (UINT i = 0; IDXGIFactory1_EnumAdapters1(f, i, &a) != DXGI_ERROR_NOT_FOUND; i++) {
            DXGI_ADAPTER_DESC1 d;
            if (SUCCEEDED(IDXGIAdapter1_GetDesc1(a, &d)) && memcmp(&d.AdapterLuid, luid, 8) == 0) {
                IDXGIAdapter1_QueryInterface(a, &IID_IDXGIAdapter3, (void **)&g_ad);
                IDXGIAdapter1_Release(a);
                break;
            }
            IDXGIAdapter1_Release(a);
        }
        IDXGIFactory1_Release(f);
    }
    if (!g_ad) return -1;
    DXGI_QUERY_VIDEO_MEMORY_INFO q;
    if (FAILED(IDXGIAdapter3_QueryVideoMemoryInfo(g_ad, 0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &q)))
        return -1;
    *budget_mib = (long long)(q.Budget >> 20);
    *usage_mib = (long long)(q.CurrentUsage >> 20);
    return 0;
}
#else
int sp_vram_budget_mib(const unsigned char luid[8], long long *budget_mib, long long *usage_mib) {
    (void)luid; (void)budget_mib; (void)usage_mib;
    return -1;
}
#endif
