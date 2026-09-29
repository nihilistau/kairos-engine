/* moe_cpu_expert.c — see moe_cpu_expert.h. Compiled with /arch:AVX512 (this TU only); the
 * entry points check CPUID first, so nothing here executes an AVX-512 instruction on a CPU
 * without it. Numerics mirror engine/tools/moe_cpu_bench (relL2 2.0e-7 vs the GPU kernels,
 * activation codes bit-identical). */
#include "moe_cpu_expert.h"
#ifndef _WIN32
/* The pool is Win32 (WaitOnAddress/SRWLOCK); elsewhere the path reports unavailable and the
 * engine stays on the GPU-only path. */
int sp_moe_cpu_init(int nthreads) { (void)nthreads; return 0; }
int sp_moe_cpu_expert(const float *x, int E, int FFE, const uint8_t *a, const uint16_t *b,
                      const uint8_t *c, const uint16_t *d, float *out) {
    (void)x; (void)E; (void)FFE; (void)a; (void)b; (void)c; (void)d; (void)out; return -1; }
int sp_moe_cpu_quant(const float *x, int in, int8_t *qx, float *sx) { (void)x; (void)in; (void)qx; (void)sx; return -1; }
void sp_moe_cpu_set_mufu_tables(const float *a, const float *b) { (void)a; (void)b; }
int sp_moe_cpu_gelu(const float *gu, int ffe, float *h) { (void)gu; (void)ffe; (void)h; return -1; }
int sp_moe_cpu_gemv(const uint8_t *codes, const uint16_t *bs, int in, int rows,
                    const int8_t *qx, const float *sx, float *y) {
    (void)codes; (void)bs; (void)in; (void)rows; (void)qx; (void)sx; (void)y; return -1; }
#else
#include <immintrin.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <intrin.h>
#pragma comment(lib, "Synchronization.lib")
#endif

#define MAXG 32                        /* in <= 4096 -> ceil(in/128) <= 32 */
typedef struct {
    int in, ng;
    __m512i qe[MAXG], qo[MAXG], corr[MAXG];
    __m512  S[MAXG];
    __mmask64 lm[MAXG];
    int8_t qx[MAXG * 128];
    float  sx[MAXG * 8];
} Act;

static void act_prepare(const float *x, int in, Act *A) {
    A->in = in; A->ng = (in + 127) / 128;
    const int npad = (in + 31) & ~31, nblk = npad >> 4;
    memset(A->qx, 0, sizeof(A->qx)); memset(A->sx, 0, sizeof(A->sx));
    for (int b = 0; b < nblk; b++) {
        float m = 0.0f;
        for (int i = 0; i < 16; i++) { int idx = b * 16 + i; float a = idx < in ? fabsf(x[idx]) : 0.0f; if (a > m) m = a; }
        const float scale = m > 0.0f ? m * (1.0f / 127.0f) : 1.0f;
        A->sx[b] = scale;
        const float inv = 1.0f / scale;
        for (int i = 0; i < 16; i++) { int idx = b * 16 + i;
            float v = idx < in ? x[idx] * inv : 0.0f;
            int q = (int)nearbyintf(v); if (q > 127) q = 127; if (q < -127) q = -127;
            A->qx[idx] = (int8_t)q; }
    }
    for (int g = 0; g < A->ng; g++) {
        __declspec(align(64)) int8_t qe[64], qo[64]; __declspec(align(64)) int32_t cr[16]; __declspec(align(64)) float S[16];
        for (int p = 0; p < 64; p++) {
            const int chunk = 4 * g + p / 16, k = p % 16, w0 = 32 * chunk + 2 * k;
            qe[p] = w0 < in ? A->qx[w0] : 0; qo[p] = w0 + 1 < in ? A->qx[w0 + 1] : 0;
        }
        for (int l = 0; l < 16; l++) {
            int s = 0; for (int t = 0; t < 4; t++) s += qe[4 * l + t] + qo[4 * l + t];
            cr[l] = 8 * s;
            const int chunk = 4 * g + l / 4, blk = 2 * chunk + ((l % 4) >= 2);
            S[l] = (32 * chunk < in) ? A->sx[blk] : 0.0f;
        }
        A->qe[g] = _mm512_load_si512(qe); A->qo[g] = _mm512_load_si512(qo);
        A->corr[g] = _mm512_load_si512(cr); A->S[g] = _mm512_load_ps(S);
        const int bytes_left = in / 2 - 64 * g;
        A->lm[g] = bytes_left >= 64 ? ~0ULL : ((1ULL << bytes_left) - 1);
    }
}

static void gemv_rows(const uint8_t *codes, const uint16_t *bsc, int in, const Act *A,
                      float *y, int r0, int r1) {
    const __m512i m0f = _mm512_set1_epi8(0x0F), m08 = _mm512_set1_epi8(0x08);
    const __m512i widx = _mm512_set_epi32(3,3,3,3, 2,2,2,2, 1,1,1,1, 0,0,0,0);
    const int rb = in / 2, nb = in / 32;
    for (int j = r0; j < r1; j++) {
        const uint8_t *row = codes + (size_t)j * rb;
        const uint16_t *bs = bsc + (size_t)j * nb;
        __m512 facc = _mm512_setzero_ps();
        for (int g = 0; g < A->ng; g++) {
            __m512i v = _mm512_maskz_loadu_epi8(A->lm[g], row + 64 * g);
            __m512i lo = _mm512_xor_si512(_mm512_and_si512(v, m0f), m08);
            __m512i hi = _mm512_xor_si512(_mm512_and_si512(_mm512_srli_epi16(v, 4), m0f), m08);
            __m512i acc = _mm512_dpbusd_epi32(_mm512_setzero_si512(), lo, A->qe[g]);
            acc = _mm512_dpbusd_epi32(acc, hi, A->qo[g]);
            acc = _mm512_sub_epi32(acc, A->corr[g]);
            const int nch = nb - 4 * g < 4 ? nb - 4 * g : 4;
            __m128i h4 = nch == 4 ? _mm_loadl_epi64((const __m128i *)(bs + 4 * g))
                                  : _mm_maskz_loadu_epi16((__mmask8)((1u << nch) - 1), bs + 4 * g);
            __m512 w = _mm512_permutexvar_ps(widx, _mm512_castps128_ps512(_mm_cvtph_ps(h4)));
            facc = _mm512_fmadd_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(acc), A->S[g]), w, facc);
        }
        y[j] = _mm512_reduce_add_ps(facc);
    }
}


/* ================= v2: bit-exact GEMV (see header) ================= */
typedef struct {
    int in, ng, n32;
    __m512i qe[MAXG], qo[MAXG], corr[MAXG];
    __mmask64 lm[MAXG];
    __declspec(align(64)) float s0[128], s1[128];   /* per chunk: sx[2c], sx[2c+1]; zero-padded */
} ActX;
static void actx_from_q(const int8_t *qx, const float *sx, int in, ActX *A) {
    A->in = in; A->ng = (in + 127) / 128; A->n32 = in >> 5;
    for (int c = 0; c < 128; c++) {
        A->s0[c] = c < A->n32 ? sx[2 * c] : 0.0f;
        A->s1[c] = c < A->n32 ? sx[2 * c + 1] : 0.0f;
    }
    for (int g = 0; g < A->ng; g++) {
        __declspec(align(64)) int8_t qe[64], qo[64]; __declspec(align(64)) int32_t cr[16];
        for (int p = 0; p < 64; p++) {
            const int chunk = 4 * g + p / 16, k = p % 16, w0 = 32 * chunk + 2 * k;
            qe[p] = w0 < in ? qx[w0] : 0; qo[p] = w0 + 1 < in ? qx[w0 + 1] : 0;
        }
        for (int l = 0; l < 16; l++) { int s = 0; for (int t = 0; t < 4; t++) s += qe[4*l+t] + qo[4*l+t]; cr[l] = 8 * s; }
        A->qe[g] = _mm512_load_si512(qe); A->qo[g] = _mm512_load_si512(qo); A->corr[g] = _mm512_load_si512(cr);
        const int bytes_left = in / 2 - 64 * g;
        A->lm[g] = bytes_left >= 64 ? ~0ULL : ((1ULL << bytes_left) - 1);
    }
}
static void gemv_rows_exact(const uint8_t *codes, const uint16_t *bsc, const ActX *A,
                            float *y, int r0, int r1) {
    const __m512i m0f = _mm512_set1_epi8(0x0F), m08 = _mm512_set1_epi8(0x08);
    /* after the adjacent-pair add, lanes 0,4,8,12 hold a0 of chunks 0..3, lanes 2,6,10,14 a1 */
    const __m512i pa0 = _mm512_set_epi32(0,0,0,0, 0,0,0,0, 0,0,0,0, 12,8,4,0);
    const __m512i pa1 = _mm512_set_epi32(0,0,0,0, 0,0,0,0, 0,0,0,0, 14,10,6,2);
    const int in = A->in, rb = in / 2, nb = A->n32;
    __declspec(align(64)) int32_t a0[128], a1[128];
    __declspec(align(64)) float lane[32];
    for (int j = r0; j < r1; j++) {
        const uint8_t *row = codes + (size_t)j * rb;
        const uint16_t *bs = bsc + (size_t)j * nb;
        for (int g = 0; g < A->ng; g++) {
            __m512i v = _mm512_maskz_loadu_epi8(A->lm[g], row + 64 * g);
            __m512i lo = _mm512_xor_si512(_mm512_and_si512(v, m0f), m08);
            __m512i hi = _mm512_xor_si512(_mm512_and_si512(_mm512_srli_epi16(v, 4), m0f), m08);
            __m512i acc = _mm512_dpbusd_epi32(_mm512_setzero_si512(), lo, A->qe[g]);
            acc = _mm512_dpbusd_epi32(acc, hi, A->qo[g]);
            acc = _mm512_sub_epi32(acc, A->corr[g]);
            acc = _mm512_add_epi32(acc, _mm512_shuffle_epi32(acc, (_MM_PERM_ENUM)0xB1));   /* swap pairs */
            _mm_store_si128((__m128i *)(a0 + 4 * g), _mm512_castsi512_si128(_mm512_permutexvar_epi32(pa0, acc)));
            _mm_store_si128((__m128i *)(a1 + 4 * g), _mm512_castsi512_si128(_mm512_permutexvar_epi32(pa1, acc)));
        }
        /* lanes: chunk c accumulates into lane c % 32, in increasing c — the GPU's order */
        __m512 f0 = _mm512_setzero_ps(), f1 = _mm512_setzero_ps();
        for (int cb = 0; cb < nb; cb += 32) {
            for (int h = 0; h < 2; h++) {
                const int c0 = cb + 16 * h;
                if (c0 >= nb) break;
                const int cnt = nb - c0 < 16 ? nb - c0 : 16;
                const __mmask16 m = (__mmask16)(cnt == 16 ? 0xFFFF : ((1u << cnt) - 1));
                __m512 x0 = _mm512_cvtepi32_ps(_mm512_maskz_loadu_epi32(m, a0 + c0));
                __m512 x1 = _mm512_cvtepi32_ps(_mm512_maskz_loadu_epi32(m, a1 + c0));
                __m512 s0 = _mm512_loadu_ps(A->s0 + c0), s1 = _mm512_loadu_ps(A->s1 + c0);
                __m512 w  = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(m, bs + c0));
                __m512 t  = _mm512_fmadd_ps(x0, s0, _mm512_mul_ps(x1, s1));   /* SASS: FMUL a1*s1 ; FFMA a0*s0 + that */
                if (h == 0) f0 = _mm512_mask_mov_ps(f0, m, _mm512_fmadd_ps(w, t, f0));
                else        f1 = _mm512_mask_mov_ps(f1, m, _mm512_fmadd_ps(w, t, f1));
            }
        }
        _mm512_store_ps(lane, f0); _mm512_store_ps(lane + 16, f1);
        for (int o = 16; o > 0; o >>= 1)
            for (int l = 0; l < o; l++) lane[l] = lane[l + o] + lane[l];
        y[j] = lane[0];
    }
}

/* ---------------- pool: spin briefly, then park on WaitOnAddress ---------------- */
typedef struct {
    const uint8_t *codes; const uint16_t *bs; int in, rows; const Act *A; const ActX *AX; float *y;
} Job;
static int g_n = 0, g_ok = -1;
static volatile LONG g_gen = 0, g_done = 0;
static Job g_job;
static HANDLE g_th[16];

static void run_share(int t) {
    const int a = g_job.rows * t / g_n, b = g_job.rows * (t + 1) / g_n;
    if (g_job.AX) gemv_rows_exact(g_job.codes, g_job.bs, g_job.AX, g_job.y, a, b);
    else          gemv_rows(g_job.codes, g_job.bs, g_job.in, g_job.A, g_job.y, a, b);
}
static DWORD WINAPI worker(LPVOID arg) {
    const int t = (int)(intptr_t)arg;
    LONG seen = 0;
    for (;;) {
        int spins = 0;
        while (g_gen == seen) {
            if (++spins < 200000) { _mm_pause(); continue; }          /* ~2-3 ms of spinning */
            LONG cmp = seen; WaitOnAddress(&g_gen, &cmp, sizeof(LONG), INFINITE); spins = 0;
        }
        seen = g_gen;
        run_share(t);
        InterlockedIncrement(&g_done);
    }
    return 0;
}
static void pool_run(const uint8_t *codes, const uint16_t *bs, int in, int rows, const Act *A, float *y) {
    g_job.codes = codes; g_job.bs = bs; g_job.in = in; g_job.rows = rows; g_job.A = A; g_job.AX = NULL; g_job.y = y;
    InterlockedExchange(&g_done, 0);
    InterlockedIncrement(&g_gen);
    WakeByAddressAll((PVOID)&g_gen);
    run_share(0);
    while (g_done < g_n - 1) _mm_pause();
}

static int cpu_has_vnni(void) {
    int r[4]; __cpuidex(r, 7, 0);
    const int avx512f = (r[1] >> 16) & 1, avx512bw = (r[1] >> 30) & 1, vnni = (r[2] >> 11) & 1;
    /* the OS must also save ZMM state */
    const unsigned long long xcr0 = _xgetbv(0);
    return avx512f && avx512bw && vnni && ((xcr0 & 0xE6) == 0xE6);
}

int sp_moe_cpu_init(int nthreads) {
    if (g_ok >= 0) return g_ok;
    g_ok = 0;
    if (!cpu_has_vnni()) return 0;
    if (nthreads <= 0) nthreads = 4;
    if (nthreads > 16) nthreads = 16;
    g_n = nthreads;
    for (int i = 1; i < g_n; i++) {
        g_th[i] = CreateThread(NULL, 0, worker, (LPVOID)(intptr_t)i, 0, NULL);
        if (!g_th[i]) { g_n = i; break; }
        SetThreadPriority(g_th[i], THREAD_PRIORITY_ABOVE_NORMAL);
    }
    g_ok = 1;
    return 1;
}

static Act g_ain, g_ah;          /* one caller at a time: g_cpu_lk */
static SRWLOCK g_cpu_lk = SRWLOCK_INIT;
static float g_gu[8192], g_h[4096];
int sp_moe_cpu_expert(const float *x, int E, int FFE,
                      const uint8_t *gu_codes, const uint16_t *gu_bs,
                      const uint8_t *dw_codes, const uint16_t *dw_bs, float *out) {
    if (g_ok != 1) return -1;
    if ((E & 31) || (FFE & 31) || E > 4096 || 2 * FFE > 4096) return -1;
    AcquireSRWLockExclusive(&g_cpu_lk);
    act_prepare(x, E, &g_ain);
    pool_run(gu_codes, gu_bs, E, 2 * FFE, &g_ain, g_gu);
    const float k = 0.7978845608028654f;
    for (int c = 0; c < FFE; c++) {
        const float xg = g_gu[c];
        const float th = tanhf(k * (xg + 0.044715f * xg * xg * xg));
        g_h[c] = 0.5f * xg * (1.0f + th) * g_gu[FFE + c];
    }
    act_prepare(g_h, FFE, &g_ah);
    pool_run(dw_codes, dw_bs, FFE, E, &g_ah, out);
    ReleaseSRWLockExclusive(&g_cpu_lk);
    return 0;
}
/* ---- v2 entry points ---- */
int sp_moe_cpu_quant(const float *x, int in, int8_t *qx, float *sx) {
    if ((in & 31) || in > 4096) return -1;
    const int nblk = in >> 4;                       /* npad == in for in % 32 == 0 */
    for (int b = 0; b < nblk; b++) {
        float m = 0.0f;
        for (int i = 0; i < 16; i++) { float a = fabsf(x[b * 16 + i]); if (a > m) m = a; }
        const float scale = m > 0.0f ? m * (1.0f / 127.0f) : 1.0f;
        sx[b] = scale;
        const float inv = 1.0f / scale;
        for (int i = 0; i < 16; i++) {
            int q = (int)nearbyintf(x[b * 16 + i] * inv); if (q > 127) q = 127; if (q < -127) q = -127;
            qx[b * 16 + i] = (int8_t)q; }
    }
    return 0;
}
static ActX g_ax;
int sp_moe_cpu_gemv(const uint8_t *codes, const uint16_t *bs, int in, int rows,
                    const int8_t *qx, const float *sx, float *y) {
    if (g_ok != 1 || (in & 31) || in > 4096) return -1;
    AcquireSRWLockExclusive(&g_cpu_lk);
    actx_from_q(qx, sx, in, &g_ax);
    g_job.codes = codes; g_job.bs = bs; g_job.in = in; g_job.rows = rows; g_job.A = NULL; g_job.AX = &g_ax; g_job.y = y;
    InterlockedExchange(&g_done, 0);
    InterlockedIncrement(&g_gen);
    WakeByAddressAll((PVOID)&g_gen);
    run_share(0);
    while (g_done < g_n - 1) _mm_pause();
    ReleaseSRWLockExclusive(&g_cpu_lk);
    return 0;
}
/* ---- v3: bit-exact GELU (see header) ---- */
static float *g_trcp = NULL, *g_tex2 = NULL;
void sp_moe_cpu_set_mufu_tables(const float *rcp_by_mantissa, const float *ex2_by_frac23) {
    const size_t n = (size_t)1 << 23;
    if (!g_trcp) g_trcp = (float *)malloc(n * sizeof(float));
    if (!g_tex2) g_tex2 = (float *)malloc(n * sizeof(float));
    if (!g_trcp || !g_tex2) { free(g_trcp); free(g_tex2); g_trcp = g_tex2 = NULL; return; }
    memcpy(g_trcp, rcp_by_mantissa, n * sizeof(float));
    memcpy(g_tex2, ex2_by_frac23, n * sizeof(float));
}
static __forceinline float u2f_(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static __forceinline uint32_t f2u_(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
/* k_moe_gelu_split_batch, op for op (SASS: FMUL, FMUL, FFMA, FMUL 0.5, FMUL k, |y|*2.885 ->
 * MUFU.EX2 -> FADD 1 -> MUFU.RCP -> FFMA(r,-2,1) -> FSEL(>=9.0109) -> copysign; |y|<0.6 -> odd
 * polynomial; then FADD 1, FMUL, FMUL). */
static __forceinline float gelu_exact(float x, float up) {
    float r3 = x * 0.044714998453855514526f;
    r3 = x * r3;
    r3 = fmaf(x, r3, x);
    const float r4 = x * 0.5f;
    const float y = r3 * 0.79788458347320556641f;
    const float ay = fabsf(y);
    float th;
    if (ay < 0.60000002384185791016f) {
        const float y2 = y * y;
        float p = fmaf(y2, u2f_(0x3c80f082u), -0.052303962409496307373f);
        p = fmaf(y2, p, 0.1331529766321182251f);
        p = fmaf(y2, p, -0.33332768082618713379f);
        p = fmaf(y2, p, 0.0f);
        th = fmaf(y, p, y);
    } else if (ay >= 9.010913848876953125f) {
        th = u2f_(f2u_(1.0f) | (f2u_(y) & 0x80000000u));
    } else if (ay != ay) {
        th = ay;                                           /* NaN propagates, as on the GPU */
    } else {
        const float x2 = ay * 2.8853900432586669922f;       /* in [1.73, 26.0] here */
        const int ip = (int)floorf(x2);
        const float f = x2 - (float)ip;                      /* exact */
        const float e = u2f_(f2u_(g_tex2[(uint32_t)(f * 8388608.0f)]) + ((uint32_t)ip << 23));
        const float d = e + 1.0f;
        const uint32_t du = f2u_(d);
        const int de = (int)((du >> 23) & 0xFF) - 127;
        const float r = u2f_(f2u_(g_trcp[du & 0x7FFFFFu]) - ((uint32_t)de << 23));
        const float t = fmaf(r, -2.0f, 1.0f);
        th = u2f_((f2u_(t) & 0x7FFFFFFFu) | (f2u_(y) & 0x80000000u));
    }
    return (r4 * (th + 1.0f)) * up;
}
int sp_moe_cpu_gelu(const float *gu, int ffe, float *h) {
    if (!g_trcp || !g_tex2) return -1;
    for (int c = 0; c < ffe; c++) h[c] = gelu_exact(gu[c], gu[ffe + c]);
    return 0;
}
#endif /* _WIN32 */
