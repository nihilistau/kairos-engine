/* sp_transcode.c — E_FMT_2 + E_FMT_3: GGUF -> .sp-model + .sp-tokenizer.
 * (PPT-LAT-SP-MODEL-v0 §9 + §7.) One-shot offline transcoder; not part of
 * libshannonprime.
 *
 * Per-tensor policy (§9):
 *   - matmul weights (attn q/k/v/o, ffn gate/up/down, LM head, token_embd):
 *     dequant to f32, re-quant into OK_Q8 (dtype 10) via sp_frob_pack_tensor
 *     (precision=8) -> per-row int8 codes + a sibling ".scale" tensor
 *     (FROBENIUS_SCALE_FP32, dtype 12). The codes bytes are exactly what the
 *     in-RAM arena packs (SP_FROB_ARENA_LAYOUT_VERSION=1), so the load path
 *     reconstructs a bit-identical arena.
 *   - norms (and any other tensor): copied as F32 (dequant F16->F32 if needed),
 *     so the loader's as_f32 path is a plain mmap+cast / owned-f32 copy.
 *
 * Data-region layout (§9 sibling adjacency): a weight's ".scale" tensor is
 * written immediately after it, no other tensor interposed. The tensor TABLE is
 * sorted by xxh64(name); the on-disk DATA order is parent-then-scale groups.
 *
 * Usage:
 *   sp_transcode <in.gguf> <out.sp-model> <out.sp-tokenizer> [--verify]
 */
#define _CRT_SECURE_NO_WARNINGS
#include "sp_engine/sp_model.h"
#include "sp_engine/gguf.h"
#include "sp_engine/model.h"     /* sp_dequant_row, sp_f16_to_f32 */
#include "sp/frobenius_lift.h"
#include "sp/sp_l1.h"            /* sp_arch_info, sp_precision (E_PARITY_3) */
#include "sp_hash.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

/* ── a growable in-memory tensor we are about to emit ── */
typedef struct {
    char     name[80];
    uint32_t dtype_id;
    uint32_t n_dims;
    uint64_t dims[8];
    uint32_t block_size;
    uint8_t *bytes;          /* owned payload; NULL after streamed-out */
    uint64_t size_bytes;
    uint64_t data_off;       /* assigned at layout time (rel. to data region) */
    uint8_t  blake3[32];     /* digest (streaming mode computes it at emit time) */
    int      streamed;       /* 1 = bytes already written to the temp data file */
} emit_tensor;

typedef struct { emit_tensor *t; int n, cap; } emit_list;

/* ── STREAMING data region (Phase 5: the 26B diffusion-gemma's ~14 GB OK_Q4B emit set
 * does not fit the commit limit when held all-in-RAM). When g_stream != NULL each add_*
 * writes its packed bytes straight to a temp data file (64-aligned, emit order) and frees
 * them immediately, so peak RAM is ~one tensor (largest ≈ a fused-expert tile). The temp
 * file is the verbatim data region (concatenated final-layout); main() appends it after the
 * header+table. Default (no --stream) = the original all-in-RAM path, byte-identical. */
static FILE    *g_stream    = NULL;   /* temp data file, or NULL = in-RAM mode */
static uint64_t g_stream_off = 0;     /* running data-region cursor (== next data_off base) */

static uint64_t align_up_(uint64_t x, uint64_t a) { return (x + a - 1) / a * a; }

/* Stream one filled emit_tensor's bytes to the temp data file: pad to 64, write, digest,
 * free. Sets e->data_off / e->blake3 / e->streamed. Returns 0, 1 on I/O error. */
static int emit_stream_one(emit_tensor *e) {
    static const uint8_t zpad[64] = {0};
    uint64_t aligned = align_up_(g_stream_off, SP_TENSOR_ALIGN);
    while (g_stream_off < aligned) {
        uint64_t w = aligned - g_stream_off; if (w > sizeof zpad) w = sizeof zpad;
        if (fwrite(zpad, 1, (size_t)w, g_stream) != w) return 1;
        g_stream_off += w;
    }
    e->data_off = g_stream_off;
    sp_blake3_256(e->bytes, (size_t)e->size_bytes, e->blake3);
    if (e->size_bytes && fwrite(e->bytes, 1, (size_t)e->size_bytes, g_stream) != e->size_bytes) return 1;
    g_stream_off += e->size_bytes;
    free(e->bytes); e->bytes = NULL; e->streamed = 1;
    return 0;
}

static emit_tensor *el_push(emit_list *L) {
    if (L->n == L->cap) {
        L->cap = L->cap ? L->cap * 2 : 64;
        L->t = (emit_tensor *)realloc(L->t, (size_t)L->cap * sizeof(emit_tensor));
        if (!L->t) { fprintf(stderr, "OOM tensor list\n"); exit(2); }
    }
    emit_tensor *e = &L->t[L->n++];
    memset(e, 0, sizeof *e);
    return e;
}
static void el_free(emit_list *L) { for (int i = 0; i < L->n; i++) free(L->t[i].bytes); free(L->t); }

/* ── SAFETENSORS DIRECT (gemma4): weight bytes from the official bf16 checkpoint ──
 * The 2026-06 gemma4 GGUF wave shipped corrupted tensor DATA while metadata,
 * tokenizer and rope_freqs stayed clean (gold-forward conviction: safetensors
 * 4.68 PPL vs GGUF 271-364 on identical arithmetic; see lattice
 * tests/gemma4_gold + CONTRACT-SPEED RESOLUTION 2026-06-07). With --st, every
 * gemma4-mapped tensor's VALUES come from model.safetensors; the GGUF supplies
 * structure, KV metadata, tokenizer, and rope_freqs only. Mapped tensors that
 * fail to resolve are a HARD ERROR — no silent fallback to poisoned bytes. */
typedef struct {
    char     name[128];          /* full HF name */
    int      is_bf16;            /* dtype: BF16 or F32 */
    uint64_t shape[4]; int n_shape;
    uint64_t off, len;           /* relative to data base */
} st_entry;
typedef struct {
    FILE *f;
    uint64_t data_base;
    st_entry *e; int n;
} st_ctx;

static st_ctx *st_open(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[st] cannot open %s\n", path); return NULL; }
    uint64_t hlen = 0;
    if (fread(&hlen, 8, 1, f) != 1 || hlen == 0 || hlen > (64u << 20)) { fclose(f); return NULL; }
    char *h = (char *)malloc((size_t)hlen + 1);
    if (!h || fread(h, 1, (size_t)hlen, f) != hlen) { free(h); fclose(f); return NULL; }
    h[hlen] = 0;
    st_ctx *st = (st_ctx *)calloc(1, sizeof *st);
    st->f = f; st->data_base = 8 + hlen;
    int cap = 0;
    /* linear scan of the header JSON: top-level "name":{...} pairs. HF tensor
     * names contain no escapes/quotes, so quote-scanning is exact. */
    const char *p = strchr(h, '{');
    if (p) p++;
    while (p && *p) {
        while (*p && *p != '"') p++;
        if (!*p) break;
        const char *k0 = ++p;
        while (*p && *p != '"') p++;
        if (!*p) break;
        size_t klen = (size_t)(p - k0);
        p++;                                   /* past closing quote */
        while (*p && *p != '{') p++;
        if (!*p) break;
        const char *v0 = p; int depth = 0;
        do { if (*p == '{') depth++; else if (*p == '}') depth--; p++; } while (*p && depth);
        if (klen >= 12 && strncmp(k0, "__metadata__", 12) == 0) continue;
        if (st->n == cap) { cap = cap ? cap * 2 : 1024; st->e = (st_entry *)realloc(st->e, (size_t)cap * sizeof(st_entry)); }
        st_entry *e = &st->e[st->n];
        memset(e, 0, sizeof *e);
        if (klen >= sizeof e->name) klen = sizeof e->name - 1;
        memcpy(e->name, k0, klen);
        char vb[512]; size_t vlen = (size_t)(p - v0);
        if (vlen >= sizeof vb) vlen = sizeof vb - 1;
        memcpy(vb, v0, vlen); vb[vlen] = 0;
        e->is_bf16 = strstr(vb, "\"BF16\"") != NULL;
        if (!e->is_bf16 && !strstr(vb, "\"F32\"")) continue;   /* skip exotic dtypes */
        const char *sh = strstr(vb, "\"shape\"");
        if (sh) { sh = strchr(sh, '[');
            if (sh) { sh++;
                while (e->n_shape < 4) {
                    char *end; uint64_t v = strtoull(sh, &end, 10);
                    if (end == sh) break;
                    e->shape[e->n_shape++] = v; sh = end;
                    while (*sh == ',' || *sh == ' ') sh++;
                    if (*sh == ']') break;
                } } }
        const char *of = strstr(vb, "\"data_offsets\"");
        if (!of) continue;
        of = strchr(of, '[');
        if (!of) continue;
        char *end; uint64_t a = strtoull(of + 1, &end, 10);
        while (*end == ',' || *end == ' ') end++;
        uint64_t b = strtoull(end, NULL, 10);
        e->off = a; e->len = b - a;
        st->n++;
    }
    free(h);
    fprintf(stderr, "[st] %s: %d tensors mapped\n", path, st->n);
    return st;
}
static void st_close(st_ctx *st) { if (st) { fclose(st->f); free(st->e); free(st); } }

/* gemma4 GGUF-name -> HF-name map. Returns 0 if unmapped (rope_freqs etc.). */
static int st_map_name(const char *gguf_name, char *out, size_t outsz) {
    unsigned L;
    char sub[64];
    if (strcmp(gguf_name, "token_embd.weight") == 0)
        return snprintf(out, outsz, "model.language_model.embed_tokens.weight"), 1;
    if (strcmp(gguf_name, "output_norm.weight") == 0)
        return snprintf(out, outsz, "model.language_model.norm.weight"), 1;
    if (sscanf(gguf_name, "blk.%u.%63s", &L, sub) != 2) return 0;
    static const struct { const char *g, *h; } M[] = {
        { "attn_norm.weight",            "input_layernorm.weight" },
        { "attn_q.weight",               "self_attn.q_proj.weight" },
        { "attn_k.weight",               "self_attn.k_proj.weight" },
        { "attn_v.weight",               "self_attn.v_proj.weight" },
        { "attn_output.weight",          "self_attn.o_proj.weight" },
        { "attn_q_norm.weight",          "self_attn.q_norm.weight" },
        { "attn_k_norm.weight",          "self_attn.k_norm.weight" },
        { "post_attention_norm.weight",  "post_attention_layernorm.weight" },
        { "ffn_norm.weight",             "pre_feedforward_layernorm.weight" },
        { "ffn_gate.weight",             "mlp.gate_proj.weight" },
        { "ffn_up.weight",               "mlp.up_proj.weight" },
        { "ffn_down.weight",             "mlp.down_proj.weight" },
        { "post_ffw_norm.weight",        "post_feedforward_layernorm.weight" },
        { "layer_output_scale.weight",   "layer_scalar" },        /* NO .weight suffix */
    };
    for (size_t i = 0; i < sizeof M / sizeof M[0]; i++)
        if (strcmp(sub, M[i].g) == 0)
            return snprintf(out, outsz, "model.language_model.layers.%u.%s", L, M[i].h), 1;
    return 0;
}
static const st_entry *st_find(const st_ctx *st, const char *gguf_name) {
    char hf[160];
    if (!st || !st_map_name(gguf_name, hf, sizeof hf)) return NULL;
    for (int i = 0; i < st->n; i++)
        if (strcmp(st->e[i].name, hf) == 0) return &st->e[i];
    fprintf(stderr, "[st] FATAL: %s maps to %s but it is absent from the safetensors\n", gguf_name, hf);
    return (const st_entry *)-1;                 /* sentinel: mapped-but-missing */
}
/* read elements [j*cols, (j+1)*cols) as f32 (bf16 widened or f32 verbatim) */
static int st_read_row(const st_ctx *st, const st_entry *e, int j, int cols, float *dst) {
    size_t esz = e->is_bf16 ? 2 : 4;
    uint64_t off = st->data_base + e->off + (uint64_t)j * (uint64_t)cols * esz;
#if defined(_WIN32)
    if (_fseeki64(st->f, (long long)off, SEEK_SET)) return 1;
#else
    if (fseeko(st->f, (off_t)off, SEEK_SET)) return 1;
#endif
    if (e->is_bf16) {
        uint16_t tmp[4096];
        int done = 0;
        while (done < cols) {
            int chunk = cols - done; if (chunk > 4096) chunk = 4096;
            if (fread(tmp, 2, (size_t)chunk, st->f) != (size_t)chunk) return 1;
            for (int c = 0; c < chunk; c++) {
                uint32_t u = (uint32_t)tmp[c] << 16;
                memcpy(&dst[done + c], &u, 4);
            }
            done += chunk;
        }
        return 0;
    }
    return fread(dst, 4, (size_t)cols, st->f) != (size_t)cols;
}
static st_ctx *g_st = NULL;                      /* active Safetensors Direct override */
static int g_st_hits = 0, g_st_misses = 0;

/* ── GGUF row reader for the Q8 packer (dequant F32/F16/Q8_0 -> f32) ── */
typedef struct { const uint8_t *base; uint32_t type; size_t rb; int cols;
                 const st_entry *se; } row_ctx;
static size_t ggml_row_bytes(uint32_t type, int n) {
    switch (type) {
        case GGML_T_F32:  return (size_t)n * 4;
        case GGML_T_F16:  return (size_t)n * 2;
        case GGML_T_Q4_0: return (size_t)(n / 32) * 18;   /* QAT-Q4_0 source */
        case GGML_T_Q5_0: return (size_t)(n / 32) * 22;   /* diffusion-gemma self_cond_down */
        case GGML_T_Q8_0: return (size_t)(n / 32) * 34;
        case GGML_T_Q4_K: return (size_t)(n / 256) * 144;  /* qwen35moe Q4_K_M source */
        case GGML_T_Q6_K: return (size_t)(n / 256) * 210;
        default:          return 0;
    }
}
static int get_row(void *ctx, int j, float *dst) {
    const row_ctx *g = (const row_ctx *)ctx;
    if (g->se)                                   /* Safetensors Direct */
        return st_read_row(g_st, g->se, j, g->cols, dst);
    return sp_dequant_row(g->base + (size_t)j * g->rb, g->type, g->cols, dst);
}

/* Resolve the value source for tensor W under the active override (or NULL).
 * Mapped-but-missing => hard error (-1 sentinel). Shape sanity: GGUF [cols,rows]
 * vs HF [rows,cols] (or 1-D / scalar) must agree element-wise. */
static const st_entry *st_source(const gguf_tensor *W, int *err) {
    *err = 0;
    if (!g_st) return NULL;
    const st_entry *se = st_find(g_st, W->name);
    if (se == (const st_entry *)-1) { *err = 1; return NULL; }
    if (!se) { g_st_misses++; return NULL; }     /* unmapped (rope_freqs) -> GGUF */
    uint64_t g_elems = 1, s_elems = 1;
    for (uint32_t d = 0; d < W->n_dims; d++) g_elems *= W->dims[d];
    for (int d = 0; d < se->n_shape; d++) s_elems *= se->shape[d];
    if (se->n_shape == 0) s_elems = se->len / (se->is_bf16 ? 2 : 4);  /* scalar */
    if (g_elems != s_elems) {
        fprintf(stderr, "[st] FATAL: %s element count %llu (gguf) != %llu (st)\n",
                W->name, (unsigned long long)g_elems, (unsigned long long)s_elems);
        *err = 1; return NULL;
    }
    if (W->n_dims >= 2 && se->n_shape >= 2 && W->dims[0] != se->shape[se->n_shape - 1]) {
        fprintf(stderr, "[st] FATAL: %s cols %llu (gguf) != %llu (st innermost)\n",
                W->name, (unsigned long long)W->dims[0],
                (unsigned long long)se->shape[se->n_shape - 1]);
        *err = 1; return NULL;
    }
    g_st_hits++;
    return se;
}

/* Add a matmul weight as OK_Q8 + ".scale" sibling (adjacent in the data region). */
static int add_q8(emit_list *L, const gguf_ctx *g, const gguf_tensor *W) {
    const uint8_t *base = (const uint8_t *)gguf_tensor_data(g, W);
    if (!base || W->n_dims < 2) { fprintf(stderr, "bad weight %s\n", W->name); return 1; }
    int serr; const st_entry *se = st_source(W, &serr);
    if (serr) return 1;
    int cols = (int)W->dims[0], rows = (int)W->dims[1];
    /* Rank-3 MoE expert tensors [cols, rows, n_expert]: pack as (rows*n_expert)
     * contiguous 2D Frobenius rows — the source row (e,r) is at element
     * (e*rows + r)*cols, so a single linear pass over total_rows is exact and the
     * existing 2D frobenius_lift applies per expert. The bridge slices expert e as
     * rows [e*rows, (e+1)*rows). */
    int ne2 = (W->n_dims >= 3) ? (int)W->dims[2] : 1;
    int total_rows = rows * ne2;
    size_t rb = ggml_row_bytes(W->type, cols);
    if (rb == 0 && !se) { fprintf(stderr, "unsupported src type for %s\n", W->name); return 1; }
    row_ctx ctx = { base, W->type, rb, cols, se };
    sp_frob_packed_tensor pt;
    if (sp_frob_pack_tensor(total_rows, cols, 8, 0.0f, get_row, &ctx, &pt, NULL)) {
        fprintf(stderr, "pack failed %s\n", W->name); return 1;
    }
    /* OK_Q8 codes (total_rows*cols int8, row-major) */
    emit_tensor *q = el_push(L);
    snprintf(q->name, sizeof q->name, "%s", W->name);
    q->dtype_id = SP_DT_OK_Q8; q->n_dims = W->n_dims;
    q->dims[0] = (uint64_t)cols; q->dims[1] = (uint64_t)rows;
    if (W->n_dims >= 3) q->dims[2] = (uint64_t)ne2;
    q->block_size = 1; q->size_bytes = (uint64_t)total_rows * cols;
    q->bytes = (uint8_t *)malloc(q->size_bytes ? q->size_bytes : 1);
    if (!q->bytes) { sp_frob_packed_free(&pt); return 1; }
    memcpy(q->bytes, pt.codes, q->size_bytes);
    if (g_stream && emit_stream_one(q)) { sp_frob_packed_free(&pt); return 1; }
    /* ".scale" sibling (rows fp32) — pushed immediately after, so it is adjacent
     * in the data region after layout (§9). */
    emit_tensor *s = el_push(L);
    snprintf(s->name, sizeof s->name, "%s.scale", W->name);
    s->dtype_id = SP_DT_FROBENIUS_SCALE_FP32; s->n_dims = 1; s->dims[0] = (uint64_t)total_rows;
    s->block_size = 4; s->size_bytes = (uint64_t)total_rows * sizeof(float);
    s->bytes = (uint8_t *)malloc(s->size_bytes ? s->size_bytes : 1);
    if (!s->bytes) { sp_frob_packed_free(&pt); return 1; }
    memcpy(s->bytes, pt.row_scale, s->size_bytes);
    if (g_stream && emit_stream_one(s)) { sp_frob_packed_free(&pt); return 1; }
    sp_frob_packed_free(&pt);
    return 0;
}

/* Add a matmul weight as OK_Q4 (nibble-packed, 2 codes/byte) + ".scale" sibling.
 * The REDUCING codec for k-quant-source models (C1): ~0.5 B/weight on disk. Per-row
 * Frobenius scale; codes in [-7,7] two's-complement; low nibble = even col, high = odd.
 * Layout is exactly what build_packed_q4 (bridge) reads. Rank-3 expert tensors
 * [cols,rows,n_expert] pack as (rows*n_expert) rows (bridge slices expert e). */
static int add_q4(emit_list *L, const gguf_ctx *g, const gguf_tensor *W) {
    const uint8_t *base = (const uint8_t *)gguf_tensor_data(g, W);
    if (!base || W->n_dims < 2) { fprintf(stderr, "bad weight %s\n", W->name); return 1; }
    int serr; const st_entry *se = st_source(W, &serr);
    if (serr) return 1;
    int cols = (int)W->dims[0], rows = (int)W->dims[1];
    int ne2 = (W->n_dims >= 3) ? (int)W->dims[2] : 1;
    int total_rows = rows * ne2;
    size_t rb = ggml_row_bytes(W->type, cols);
    if (rb == 0 && !se) { fprintf(stderr, "unsupported src type for %s\n", W->name); return 1; }
    size_t nib_cols = ((size_t)cols + 1u) / 2u;

    float  *wrow  = (float *)malloc((size_t)cols * sizeof(float));
    int8_t *codes = (int8_t *)malloc((size_t)cols);
    float  *scales = (float *)malloc((size_t)total_rows * sizeof(float));
    if (!wrow || !codes || !scales) { free(wrow); free(codes); free(scales); return 1; }

    emit_tensor *q = el_push(L);
    snprintf(q->name, sizeof q->name, "%s", W->name);
    q->dtype_id = SP_DT_OK_Q4; q->n_dims = W->n_dims;
    q->dims[0] = (uint64_t)cols; q->dims[1] = (uint64_t)rows;
    if (W->n_dims >= 3) q->dims[2] = (uint64_t)ne2;
    q->block_size = 1; q->size_bytes = (uint64_t)total_rows * nib_cols;
    q->bytes = (uint8_t *)malloc(q->size_bytes ? q->size_bytes : 1);
    if (!q->bytes) { free(wrow); free(codes); free(scales); return 1; }
    for (int r = 0; r < total_rows; r++) {
        if (se ? st_read_row(g_st, se, r, cols, wrow)
               : sp_dequant_row(base + (size_t)r * rb, W->type, cols, wrow)) {
            free(wrow); free(codes); free(scales); return 1;
        }
        float s = sp_frob_row_scale(wrow, cols);
        scales[r] = s;
        for (int c = 0; c < cols; c++) codes[c] = sp_frob_quant1_q4(wrow[c], s);
        sp_frob_q4_pack(codes, cols, q->bytes + (size_t)r * nib_cols);
    }
    if (g_stream && emit_stream_one(q)) { free(wrow); free(codes); free(scales); return 1; }
    /* ".scale" sibling (total_rows fp32). NOTE: el_push may realloc — q is filled
     * above and not touched after this point. */
    emit_tensor *sc = el_push(L);
    snprintf(sc->name, sizeof sc->name, "%s.scale", W->name);
    sc->dtype_id = SP_DT_FROBENIUS_SCALE_FP32; sc->n_dims = 1; sc->dims[0] = (uint64_t)total_rows;
    sc->block_size = 4; sc->size_bytes = (uint64_t)total_rows * sizeof(float);
    sc->bytes = (uint8_t *)malloc(sc->size_bytes ? sc->size_bytes : 1);
    if (!sc->bytes) { free(wrow); free(codes); free(scales); return 1; }
    memcpy(sc->bytes, scales, (size_t)sc->size_bytes);
    if (g_stream && emit_stream_one(sc)) { free(wrow); free(codes); free(scales); return 1; }
    free(wrow); free(codes); free(scales);
    return 0;
}

/* Add a matmul weight as OK_Q4B (SPEC OK_Q4B): int4 codes [-7,7] nibble-packed
 * (low nibble = even col, identical packing to OK_Q4) + PER-32-BLOCK f16 scales
 * in a ".bscale" sibling [rows * ceil(cols/32)]. Scale discipline: s = maxabs/7
 * computed f32, ROUNDED THROUGH f16, codes quantized against the STORED scale. */
static int add_q4b(emit_list *L, const gguf_ctx *g, const gguf_tensor *W) {
    const uint8_t *base = (const uint8_t *)gguf_tensor_data(g, W);
    if (!base || W->n_dims < 2) { fprintf(stderr, "bad weight %s\n", W->name); return 1; }
    int serr; const st_entry *se = st_source(W, &serr);
    if (serr) return 1;
    int cols = (int)W->dims[0], rows = (int)W->dims[1];
    int ne2 = (W->n_dims >= 3) ? (int)W->dims[2] : 1;
    int total_rows = rows * ne2;
    size_t rb = ggml_row_bytes(W->type, cols);
    if (rb == 0 && !se) { fprintf(stderr, "unsupported src type for %s\n", W->name); return 1; }
    size_t nib_cols = ((size_t)cols + 1u) / 2u;
    int nblk = (cols + 31) / 32;

    float  *wrow  = (float *)malloc((size_t)cols * sizeof(float));
    int8_t *codes = (int8_t *)malloc((size_t)cols);
    if (!wrow || !codes) { free(wrow); free(codes); return 1; }

    emit_tensor *q = el_push(L);
    snprintf(q->name, sizeof q->name, "%s", W->name);
    q->dtype_id = SP_DT_OK_Q4B; q->n_dims = W->n_dims;
    q->dims[0] = (uint64_t)cols; q->dims[1] = (uint64_t)rows;
    if (W->n_dims >= 3) q->dims[2] = (uint64_t)ne2;
    q->block_size = 1; q->size_bytes = (uint64_t)total_rows * nib_cols;
    q->bytes = (uint8_t *)malloc(q->size_bytes ? q->size_bytes : 1);
    uint16_t *bs = (uint16_t *)malloc((size_t)total_rows * nblk * sizeof(uint16_t));
    if (!q->bytes || !bs) { free(wrow); free(codes); free(bs); return 1; }

    for (int r = 0; r < total_rows; r++) {
        if (se ? st_read_row(g_st, se, r, cols, wrow)
               : sp_dequant_row(base + (size_t)r * rb, W->type, cols, wrow)) {
            free(wrow); free(codes); free(bs); return 1;
        }
        for (int b = 0; b < nblk; b++) {
            int c0 = b * 32, c1 = c0 + 32 < cols ? c0 + 32 : cols;
            float ma = 0.0f;
            for (int c = c0; c < c1; c++) { float a = fabsf(wrow[c]); if (a > ma) ma = a; }
            float s = ma / 7.0f;
            uint16_t sh = sp_f32_to_f16(s);
            float sf = sp_f16_to_f32(sh);          /* the STORED scale */
            bs[(size_t)r * nblk + b] = sh;
            if (sf == 0.0f) { for (int c = c0; c < c1; c++) codes[c] = 0; continue; }
            for (int c = c0; c < c1; c++) {
                float v = wrow[c] / sf;
                int  k = (int)lrintf(v);
                if (k >  7) k =  7;
                if (k < -7) k = -7;
                codes[c] = (int8_t)k;
            }
        }
        sp_frob_q4_pack(codes, cols, q->bytes + (size_t)r * nib_cols);
    }
    /* stream the codes tensor out now (frees q->bytes) BEFORE the next el_push may
     * realloc L->t and dangle `q`. No-op in RAM mode. */
    if (g_stream && emit_stream_one(q)) { free(wrow); free(codes); free(bs); return 1; }
    /* ".bscale" sibling (f16, row-major blocks), adjacent per §9. */
    emit_tensor *sc = el_push(L);
    snprintf(sc->name, sizeof sc->name, "%s.bscale", W->name);
    sc->dtype_id = SP_DT_BLOCK_SCALE_FP16; sc->n_dims = 2;
    sc->dims[0] = (uint64_t)nblk; sc->dims[1] = (uint64_t)total_rows;
    sc->block_size = 2; sc->size_bytes = (uint64_t)total_rows * nblk * 2u;
    sc->bytes = (uint8_t *)malloc(sc->size_bytes ? sc->size_bytes : 1);
    if (!sc->bytes) { free(wrow); free(codes); free(bs); return 1; }
    memcpy(sc->bytes, bs, (size_t)sc->size_bytes);
    if (g_stream && emit_stream_one(sc)) { free(wrow); free(codes); free(bs); return 1; }
    free(wrow); free(codes); free(bs);
    return 0;
}

/* Add a tensor as F32 (norms etc.): dequant from F32/F16 to F32. */
static int add_f32(emit_list *L, const gguf_ctx *g, const gguf_tensor *W) {
    const uint8_t *base = (const uint8_t *)gguf_tensor_data(g, W);
    if (!base) { fprintf(stderr, "no data %s\n", W->name); return 1; }
    int serr; const st_entry *se = st_source(W, &serr);
    if (serr) return 1;
    int n = (int)W->n_elements;
    emit_tensor *e = el_push(L);
    snprintf(e->name, sizeof e->name, "%s", W->name);
    e->dtype_id = SP_DT_F32; e->n_dims = W->n_dims;
    for (uint32_t d = 0; d < W->n_dims && d < 8; d++) e->dims[d] = W->dims[d];
    e->block_size = 4; e->size_bytes = (uint64_t)n * sizeof(float);
    e->bytes = (uint8_t *)malloc(e->size_bytes ? e->size_bytes : 1);
    if (!e->bytes) return 1;
    if (se ? st_read_row(g_st, se, 0, n, (float *)e->bytes)
           : sp_dequant_row(base, W->type, n, (float *)e->bytes)) {
        fprintf(stderr, "dequant f32 failed %s\n", W->name); return 1;
    }
    if (g_stream && emit_stream_one(e)) return 1;
    return 0;
}

static uint64_t align_up(uint64_t x, uint64_t a) { return (x + a - 1) / a * a; }

/* qsort comparator: tensor entries by name_hash ascending. */
static int cmp_entry(const void *a, const void *b) {
    uint64_t ha = ((const sp_tensor_entry *)a)->name_hash;
    uint64_t hb = ((const sp_tensor_entry *)b)->name_hash;
    return (ha > hb) - (ha < hb);
}

/* ── tokenizer extraction (E_FMT_3, §7) ──
 * GGUF stores parsed tokens/scores/merges, not the original SentencePiece proto.
 * v0 .sp-tokenizer blob is a self-describing serialization of those arrays; the
 * L1 loader only needs a stable SHA-256 over the whole file. Blob layout (LE):
 *   u32 magic 'SPTB' | u32 type_id | u32 vocab | u32 n_merges
 *   tokens:  vocab * (u32 len, bytes)
 *   scores:  vocab * f32   (SPM only; 0 entries if absent)
 *   merges:  n_merges * (u32 len, bytes)
 */
static uint8_t *build_tok_blob(const gguf_ctx *g, uint64_t *blob_size_out,
                               uint32_t *type_id_out, uint32_t *vocab_out,
                               uint32_t bos_eos_pad_unk[4]) {
    const gguf_kv *tk = gguf_find_kv(g, "tokenizer.ggml.tokens");
    if (!tk || tk->type != GGUF_T_ARRAY || tk->arr_type != GGUF_T_STRING) {
        fprintf(stderr, "no tokenizer.ggml.tokens\n"); return NULL;
    }
    uint64_t nv = tk->arr_len;
    const char **tp = (const char **)malloc((size_t)nv * sizeof(char *));
    uint64_t    *tl = (uint64_t *)malloc((size_t)nv * sizeof(uint64_t));
    if (!tp || !tl || gguf_kv_str_array(g, tk, tp, tl, nv) != nv) {
        fprintf(stderr, "tokens read failed\n"); free(tp); free(tl); return NULL;
    }
    /* family tag derivation (#115): tokenizer.ggml.model + tokenizer.ggml.pre.
     * Unknown family = HARD ERROR naming the string — a silently mis-tagged blob
     * would be the tokenizer twin of the GGUF weight corruption (every id valid,
     * every id wrong). Mirrors llama-vocab.cpp:1894 (model=="gemma4") + :2005
     * (pre=="gemma4") + :1930-1931 (unknown model throws). */
    const char *model = gguf_get_str(g, "tokenizer.ggml.model");
    const char *pre   = gguf_get_str(g, "tokenizer.ggml.pre");
    int spm = (model && strcmp(model, "llama") == 0);
    if (spm) {
        *type_id_out = SP_TOK_SENTENCEPIECE;
    } else if (model && (strcmp(model, "gemma4") == 0 ||
               (strcmp(model, "gpt2") == 0 && pre && strcmp(pre, "gemma4") == 0))) {
        *type_id_out = SP_TOK_GEMMA4_BPE;
    } else if (model && strcmp(model, "gpt2") == 0) {
        *type_id_out = SP_TOK_BPE_GPT2;
    } else {
        fprintf(stderr, "build_tok_blob: unknown tokenizer family '%s' "
                        "(tokenizer.ggml.model%s%s) — refusing silent GPT2 tag\n",
                model ? model : "(missing)", pre ? ", pre=" : "", pre ? pre : "");
        free(tp); free(tl); return NULL;
    }

    const gguf_kv *sk = gguf_find_kv(g, "tokenizer.ggml.scores");
    const float *scores = (spm && sk && sk->type == GGUF_T_ARRAY &&
                           sk->arr_type == GGUF_T_FLOAT32 && sk->arr_len == nv)
                          ? (const float *)sk->arr_data : NULL;

    const gguf_kv *mk = gguf_find_kv(g, "tokenizer.ggml.merges");
    uint64_t nm = 0; const char **mp = NULL; uint64_t *ml = NULL;
    if (mk && mk->type == GGUF_T_ARRAY && mk->arr_type == GGUF_T_STRING && mk->arr_len > 0) {
        nm = mk->arr_len;
        mp = (const char **)malloc((size_t)nm * sizeof(char *));
        ml = (uint64_t *)malloc((size_t)nm * sizeof(uint64_t));
        if (!mp || !ml || gguf_kv_str_array(g, mk, mp, ml, nm) != nm) { nm = 0; }
    }

    uint64_t sz = 16;                            /* magic+type+vocab+n_merges */
    for (uint64_t i = 0; i < nv; i++) sz += 4 + tl[i];
    sz += scores ? nv * 4 : 0;
    for (uint64_t i = 0; i < nm; i++) sz += 4 + ml[i];

    uint8_t *blob = (uint8_t *)malloc(sz);
    if (!blob) { free(tp); free(tl); free(mp); free(ml); return NULL; }
    uint8_t *w = blob;
    #define PUT32(v) do { uint32_t _v=(uint32_t)(v); memcpy(w,&_v,4); w+=4; } while(0)
    PUT32(0x42545053u /*'SPTB'*/); PUT32(*type_id_out); PUT32((uint32_t)nv); PUT32((uint32_t)nm);
    for (uint64_t i = 0; i < nv; i++) { PUT32(tl[i]); memcpy(w, tp[i], (size_t)tl[i]); w += tl[i]; }
    if (scores) { memcpy(w, scores, (size_t)nv * 4); w += nv * 4; }
    for (uint64_t i = 0; i < nm; i++) { PUT32(ml[i]); memcpy(w, mp[i], (size_t)ml[i]); w += ml[i]; }
    #undef PUT32

    *vocab_out = (uint32_t)nv;
    *blob_size_out = sz;
    /* special-token ids */
    uint64_t v;
    bos_eos_pad_unk[0] = gguf_get_u64(g, "tokenizer.ggml.bos_token_id", &v) ? (uint32_t)v : 0xFFFFFFFFu;
    bos_eos_pad_unk[1] = gguf_get_u64(g, "tokenizer.ggml.eos_token_id", &v) ? (uint32_t)v : 0xFFFFFFFFu;
    bos_eos_pad_unk[2] = gguf_get_u64(g, "tokenizer.ggml.padding_token_id", &v) ? (uint32_t)v : 0xFFFFFFFFu;
    bos_eos_pad_unk[3] = gguf_get_u64(g, "tokenizer.ggml.unknown_token_id", &v) ? (uint32_t)v : 0xFFFFFFFFu;
    free(tp); free(tl); free(mp); free(ml);
    return blob;
}

/* Write the .sp-tokenizer file; sets sha32 to its SHA-256. */
static int write_tokenizer(const char *path, const gguf_ctx *g,
                           uint32_t *vocab_out, uint8_t sha32[32]) {
    uint64_t blob_size; uint32_t type_id, vocab, special[4];
    uint8_t *blob = build_tok_blob(g, &blob_size, &type_id, &vocab, special);
    if (!blob) return 1;

    sp_tok_header h; memset(&h, 0, sizeof h);
    h.magic = SP_TOK_MAGIC; h.version_major = SP_MODEL_VER_MAJOR; h.version_minor = SP_MODEL_VER_MINOR;
    h.header_size = SP_TOK_HEADER_SIZE; h.type_id = type_id; h.vocab_size = vocab;
    h.bos_token = special[0]; h.eos_token = special[1]; h.pad_token = special[2]; h.unk_token = special[3];
    h.blob_offset = SP_TOK_HEADER_SIZE; h.blob_size = blob_size;
    h.header_crc32 = sp_crc32(&h, SP_TOK_CRC_COVER);

    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path); free(blob); return 1; }
    int ok = (fwrite(&h, 1, SP_TOK_HEADER_SIZE, f) == SP_TOK_HEADER_SIZE) &&
             (fwrite(blob, 1, (size_t)blob_size, f) == blob_size);
    fclose(f); free(blob);
    if (!ok) { fprintf(stderr, "tokenizer write short\n"); return 1; }

    /* SHA-256 over the entire produced file. */
    f = fopen(path, "rb");
    if (!f) return 1;
    sp_sha256_ctx sc; sp_sha256_init(&sc);
    uint8_t buf[65536]; size_t r;
    while ((r = fread(buf, 1, sizeof buf, f)) > 0) sp_sha256_update(&sc, buf, r);
    fclose(f);
    sp_sha256_final(&sc, sha32);
    *vocab_out = vocab;
    return 0;
}

/* Populate sp_arch_info into the 256-byte arch_struct (PPT-LAT-SP-MODEL-v0 §3).
 * Reads GGUF metadata directly from the already-open gguf_ctx — bypasses
 * qwen3_load so Qwen2.5 GGUFs (which omit attention.key_length) are handled. */
static int fill_arch_struct(const gguf_ctx *g, uint8_t arch_struct[256],
                            uint32_t *arch_id, uint32_t *arch_size, uint32_t *vocab) {
    const char *arch_str = gguf_get_str(g, "general.architecture");
    if (!arch_str) { fprintf(stderr, "fill_arch_struct: missing general.architecture\n"); return 1; }
    int is_gemma3 = (strcmp(arch_str, "gemma3") == 0);
    int is_gemma4 = (strcmp(arch_str, "gemma4") == 0);
    int is_qwen25 = (strcmp(arch_str, "qwen2")  == 0);
    int is_qwen3  = (strcmp(arch_str, "qwen3")  == 0);
    int is_qwen36 = (strcmp(arch_str, "qwen35moe") == 0);
    int is_dg     = (strcmp(arch_str, "diffusion-gemma") == 0);  /* Phase 5: gemma4 MoE backbone */
    int is_g4a    = (strcmp(arch_str, "gemma4-assistant") == 0); /* EAGLE/MTP draft head */
    if (!is_gemma3 && !is_gemma4 && !is_qwen25 && !is_qwen3 && !is_qwen36 && !is_dg && !is_g4a) {
        fprintf(stderr, "fill_arch_struct: unsupported arch '%s'\n", arch_str); return 1;
    }
    /* diffusion-gemma reuses the gemma4 per-layer SWA/global geometry block; its
     * backbone KV keys mirror gemma4.* under the diffusion-gemma.* prefix. */
    int is_gemma4_like = is_gemma4 || is_dg;
    /* Gemma-4 MoE (26B-A4B, "128x2.6B"): general.architecture is plain "gemma4",
     * but the GGUF carries gemma4.expert_count / expert_used_count /
     * expert_feed_forward_length. The BACKBONE is the same graph diffusion-gemma
     * runs (attention + dense shared FFN + routed experts); the difference is the
     * DECODER — this one is ordinary autoregressive, so it gets its own arch id and
     * must NOT inherit diffusion-gemma's REQUIRED diffusion.canvas_length. Probed
     * here (not in the g4 block below) because arch_id is decided before it. */
    uint64_t g4_nexp_probe = 0;
    int is_g4moe = is_gemma4 &&
                   gguf_get_u64(g, "gemma4.expert_count", &g4_nexp_probe) && g4_nexp_probe > 0;

    char key[128];
    /* ARCH_KEY: build "arch.suffix" into local `key` and return it. */
    #define ARCH_KEY(suf) (snprintf(key, sizeof key, "%s." suf, arch_str), (const char *)key)

    uint64_t n_embd = 0, n_layers = 0, n_head = 0, n_head_kv = 0;
    uint64_t context_length = 0, n_ff = 0, head_dim = 0, swa_window = 0;
    float rope_freq_base = 1e6f, rms_eps = 1e-5f;

    if (!gguf_get_u64(g, ARCH_KEY("embedding_length"),  &n_embd)  || n_embd   == 0)
        { fprintf(stderr, "fill_arch_struct: missing %s.embedding_length\n",  arch_str); return 1; }
    if (!gguf_get_u64(g, ARCH_KEY("block_count"),       &n_layers) || n_layers == 0)
        { fprintf(stderr, "fill_arch_struct: missing %s.block_count\n",       arch_str); return 1; }
    if (!gguf_get_u64(g, ARCH_KEY("attention.head_count"), &n_head) || n_head  == 0)
        { fprintf(stderr, "fill_arch_struct: missing %s.attention.head_count\n", arch_str); return 1; }
    if (!gguf_get_u64(g, ARCH_KEY("attention.head_count_kv"), &n_head_kv)) n_head_kv = n_head;
    /* head_dim optional in Qwen2.5 GGUFs — derive from hidden_dim / n_heads. */
    if (!gguf_get_u64(g, ARCH_KEY("attention.key_length"), &head_dim) || head_dim == 0)
        head_dim = n_embd / n_head;
    gguf_get_u64(g, ARCH_KEY("context_length"),              &context_length);
    gguf_get_u64(g, ARCH_KEY("attention.sliding_window"),    &swa_window);
    gguf_get_u64(g, ARCH_KEY("feed_forward_length"),         &n_ff);
    gguf_get_f32(g, ARCH_KEY("rope.freq_base"),              &rope_freq_base);
    gguf_get_f32(g, ARCH_KEY("attention.layer_norm_rms_epsilon"), &rms_eps);
    #undef ARCH_KEY

    /* vocab from token_embd.weight dims[1]; fallback to tokenizer array length. */
    uint64_t n_vocab = 0;
    const gguf_tensor *embd = gguf_find_tensor(g, "token_embd.weight");
    if (embd && embd->n_dims >= 2) n_vocab = embd->dims[1];
    if (n_vocab == 0) {
        const gguf_kv *tk = gguf_find_kv(g, "tokenizer.ggml.tokens");
        if (tk && tk->type == GGUF_T_ARRAY) n_vocab = tk->arr_len;
    }
    if (n_vocab == 0) { fprintf(stderr, "fill_arch_struct: cannot determine vocab size\n"); return 1; }

    /* tied embeddings: output.weight absent → tied. */
    int tied = (gguf_find_tensor(g, "output.weight") == NULL) ? 1 : 0;
    /* QK norms: blk.0.attn_q_norm.weight present → has_qk_norm. */
    int has_qk_norm = (gguf_find_tensor(g, "blk.0.attn_q_norm.weight") != NULL) ? 1 : 0;

    sp_arch_info ai;
    memset(&ai, 0, sizeof ai);
    ai.arch_id          = is_g4a    ? (uint32_t)SP_ARCH_ID_GEMMA4_ASSISTANT :
                          is_gemma3 ? (uint32_t)SP_ARCH_ID_GEMMA3 :
                          is_g4moe  ? (uint32_t)SP_ARCH_ID_GEMMA4_MOE :   /* BEFORE plain gemma4 */
                          is_gemma4 ? (uint32_t)SP_ARCH_ID_GEMMA4 :
                          is_dg     ? (uint32_t)SP_ARCH_ID_DIFFUSION_GEMMA :
                          is_qwen25 ? (uint32_t)SP_ARCH_ID_QWEN25 :
                          is_qwen36 ? (uint32_t)SP_ARCH_ID_QWEN36 : (uint32_t)SP_ARCH_ID_QWEN3;
    ai.vocab_size       = (uint32_t)n_vocab;
    ai.hidden_dim       = (uint32_t)n_embd;
    ai.n_layers         = (uint32_t)n_layers;
    ai.n_heads          = (uint32_t)n_head;
    ai.n_kv_heads       = (uint32_t)n_head_kv;
    ai.head_dim         = (uint32_t)head_dim;
    ai.max_context      = (uint32_t)context_length;
    ai.swa_window       = (uint32_t)swa_window;
    ai.rope_freq_base   = rope_freq_base;
    ai.ffn_variant      = (is_gemma3 || is_gemma4_like || is_g4a) ? 1u : 0u;   /* 1=GeGLU(gemma3/4/dg/g4a), 0=SwiGLU(qwen3/qwen25) */
    ai.norm_variant     = (is_gemma3 || is_gemma4_like || is_g4a) ? 1u : 0u;   /* 1=sandwich(gemma3/4/dg/g4a), 0=pre-norm(qwen3/qwen25) */
    ai.tied_embeddings  = (uint32_t)tied;
    ai.has_qk_norm      = (uint32_t)has_qk_norm;
    ai.n_ff             = (uint32_t)n_ff;
    ai.rms_eps          = rms_eps;
    ai.preferred_precision = (uint32_t)SP_PRECISION_FP16;

    /* ── Gemma4 g4_* fields (mirror qwen3_load): per-layer SWA geometry + AltUp +
     * shared-KV + softcap + period. head_dim above holds key_length (GLOBAL); the
     * per-layer head_dim/n_ff differences are recovered at load time from the
     * emitted per-layer tensor dims. ── */
    if (is_gemma4_like) {
        uint64_t gv = 0; float gf = 0.0f;
        char g4key[128];
        /* G4K(suf): build "<arch_str>.suf" — gemma4.* OR diffusion-gemma.* (same shape). */
        #define G4K(suf) (snprintf(g4key, sizeof g4key, "%s." suf, arch_str), (const char *)g4key)
        uint64_t hd_swa = 0;
        gguf_get_u64(g, G4K("attention.key_length_swa"), &hd_swa);
        ai.g4_hd_swa  = (uint32_t)(hd_swa ? hd_swa : head_dim);
        ai.g4_nh_swa  = (uint32_t)n_head;      /* n_head constant across layer types  */
        ai.g4_nkv_swa = (uint32_t)n_head_kv;   /* n_head_kv constant                   */
        ai.g4_rope_base_swa = gguf_get_f32(g, G4K("rope.freq_base_swa"), &gf) ? gf : 1e4f;
        if (gguf_get_u64(g, G4K("embedding_length_per_layer_input"), &gv)) ai.g4_n_embd_per_layer = (uint32_t)gv;
        ai.g4_logit_softcap = gguf_get_f32(g, G4K("final_logit_softcapping"), &gf) ? gf : 0.0f;
        if (gguf_get_u64(g, G4K("attention.shared_kv_layers"), &gv))
            ai.g4_n_kv_from_start = (n_layers > gv) ? (uint32_t)(n_layers - gv) : (uint32_t)n_layers;
        else
            ai.g4_n_kv_from_start = (uint32_t)n_layers;
        {
            const gguf_kv *kv = gguf_find_kv(g, G4K("attention.sliding_window_pattern"));
            uint32_t period = 0;
            if (kv && kv->type == GGUF_T_ARRAY && kv->arr_type == GGUF_T_BOOL &&
                kv->arr_len == n_layers && kv->arr_data) {
                const uint8_t *pat = (const uint8_t *)kv->arr_data;  /* 1=SWA, 0=global */
                int first_global = -1;
                for (uint64_t L = 0; L < n_layers; L++) if (!pat[L]) { first_global = (int)L; break; }
                if (first_global >= 0) {
                    period = (uint32_t)first_global + 1u;
                    for (uint64_t L = 0; L < n_layers; L++)
                        if ((int)((L % period) == period - 1u) != (int)(!pat[L])) { period = 0; break; }
                }
            }
            if (period == 0) { fprintf(stderr, "fill_arch_struct: gemma4 non-periodic SWA pattern\n"); return 1; }
            ai.g4_swa_period = period;
        }
        /* Per-layer head_count_kv: the DENSE gemma-4 (12B) emits an ARRAY —
         * e.g. 8 kv-heads on SWA layers, 1 on globals (the E-series E2B carries
         * a scalar, handled above). The arch struct expresses this two-class
         * geometry as n_kv_heads (GLOBAL) + g4_nkv_swa (SWA); derive both from
         * the array via the SWA period and REQUIRE per-type uniformity. */
        {
            const gguf_kv *kv = gguf_find_kv(g, G4K("attention.head_count_kv"));
            if (kv && kv->type == GGUF_T_ARRAY && kv->arr_data && kv->arr_len == n_layers &&
                (kv->arr_type == GGUF_T_UINT32 || kv->arr_type == GGUF_T_INT32)) {
                const uint32_t *hk = (const uint32_t *)kv->arr_data;   /* i32/u32: same width */
                uint32_t nkv_g = 0, nkv_s = 0; int two_class = 1;
                for (uint64_t L = 0; L < n_layers; L++) {
                    const int global = ((L % ai.g4_swa_period) == ai.g4_swa_period - 1u);
                    if (global) { if (!nkv_g) nkv_g = hk[L]; else if (nkv_g != hk[L]) two_class = 0; }
                    else        { if (!nkv_s) nkv_s = hk[L]; else if (nkv_s != hk[L]) two_class = 0; }
                }
                if (!two_class || !nkv_g || !nkv_s) {
                    fprintf(stderr, "fill_arch_struct: gemma4 per-layer head_count_kv is not "
                                    "two-class uniform (global/SWA)\n");
                    return 1;
                }
                ai.n_kv_heads = nkv_g;
                ai.g4_nkv_swa = nkv_s;
                fprintf(stderr, "    [arch] gemma4 per-layer kv-heads: global=%u swa=%u (period %u)\n",
                        nkv_g, nkv_s, ai.g4_swa_period);
            }
        }
        if (ai.n_ff == 0) {   /* feed_forward_length is a per-layer array; n_ff scalar fallback = layer 0 */
            const gguf_tensor *fg0 = gguf_find_tensor(g, "blk.0.ffn_gate.weight");
            if (fg0 && fg0->n_dims >= 2) ai.n_ff = (uint32_t)fg0->dims[1];
        }
        ai.has_qk_norm = 1u;

        /* ── The routed-MoE FFN geometry, shared by BOTH Gemma-4 MoE decoders:
         *      diffusion-gemma   (arch 9)  — block masked-diffusion decoder
         *      gemma4 + experts  (arch 11) — ordinary AUTOREGRESSIVE decoder (26B-A4B)
         * The gemma4 backbone read above is identical for both; only the expert
         * counts are new, and they reuse the q36_* expert slots (the contract:
         * backbone MoE geometry lives there). The GGUF key SUFFIXES are the same
         * for both — G4K() just prefixes them with the arch string, so
         * "gemma4.expert_count" and "diffusion-gemma.expert_count" both resolve.
         *
         * The DIFFUSION SURFACE (dg_*, canvas_length REQUIRED) belongs to arch 9
         * ALONE: an AR model has no canvas, and requiring one here would reject a
         * perfectly good 26B-A4B. ── */
        if (is_dg || is_g4moe) {
            uint64_t dv = 0; float df = 0.0f;
            if (gguf_get_u64(g, G4K("expert_count"), &dv))               ai.q36_n_expert      = (uint32_t)dv;
            if (gguf_get_u64(g, G4K("expert_used_count"), &dv))          ai.q36_n_expert_used = (uint32_t)dv;
            if (gguf_get_u64(g, G4K("expert_feed_forward_length"), &dv)) ai.q36_n_ff_exp      = (uint32_t)dv;
            ai.q36_rope_dim  = gguf_get_u64(g, G4K("rope.dimension_count"), &dv) ? (uint32_t)dv : 0u;
            ai.q36_rope_base = rope_freq_base;
            if (ai.q36_n_expert == 0 || ai.q36_n_expert_used == 0) {
                fprintf(stderr, "fill_arch_struct: %s missing expert_count/used\n", arch_str); return 1;
            }
            /* The fused expert tensor is [n_embd, 2*n_ff_exp, n_expert] (gate+up
             * concatenated; the forward slices it). If expert_feed_forward_length
             * disagrees with the tensor, the forward would slice at the wrong
             * offset and produce plausible garbage — so CHECK it here, where it is
             * still cheap, rather than discovering it as a quality regression. */
            {
                const gguf_tensor *gu = gguf_find_tensor(g, "blk.0.ffn_gate_up_exps.weight");
                if (gu && gu->n_dims >= 3) {
                    if (gu->dims[1] != (uint64_t)ai.q36_n_ff_exp * 2u ||
                        gu->dims[2] != (uint64_t)ai.q36_n_expert) {
                        fprintf(stderr, "fill_arch_struct: %s ffn_gate_up_exps is [%llu,%llu,%llu] but "
                                        "expert_ff=%u (expect dim1=2*%u=%u) / expert_count=%u\n",
                                arch_str, (unsigned long long)gu->dims[0],
                                (unsigned long long)gu->dims[1], (unsigned long long)gu->dims[2],
                                ai.q36_n_ff_exp, ai.q36_n_ff_exp, ai.q36_n_ff_exp * 2u, ai.q36_n_expert);
                        return 1;
                    }
                }
            }
        }
        if (is_dg) {
            uint64_t dv = 0; float df = 0.0f;
            /* diffusion.* (NOT arch-prefixed) — canvas_length REQUIRED; eb_* optional
             * (this conversion wave omits them; 0 = unspecified, sampler defaults @N4). */
            if (!gguf_get_u64(g, "diffusion.canvas_length", &dv) || dv == 0) {
                fprintf(stderr, "fill_arch_struct: diffusion-gemma missing REQUIRED diffusion.canvas_length\n");
                return 1;
            }
            ai.dg_canvas_length = (uint32_t)dv;
            if (gguf_get_u64(g, "diffusion.eb_max_steps", &dv)) ai.dg_eb_max_steps = (uint32_t)dv;
            if (gguf_get_f32(g, "diffusion.eb_t_min", &df))                ai.dg_eb_t_min                = df;
            if (gguf_get_f32(g, "diffusion.eb_t_max", &df))                ai.dg_eb_t_max                = df;
            if (gguf_get_f32(g, "diffusion.eb_entropy_bound", &df))        ai.dg_eb_entropy_bound        = df;
            if (gguf_get_f32(g, "diffusion.eb_stability_threshold", &df))  ai.dg_eb_stability_threshold  = df;
            if (gguf_get_f32(g, "diffusion.eb_confidence_threshold", &df)) ai.dg_eb_confidence_threshold = df;
            if (gguf_get_u64(g, "diffusion.shift_logits", &dv)) ai.dg_shift_logits = (uint32_t)dv;
            fprintf(stderr, "    [arch] diffusion-gemma: canvas_length=%u experts=%u/%u expert_ff=%u "
                            "eb_max_steps=%u entropy_bound=%g\n",
                    ai.dg_canvas_length, ai.q36_n_expert, ai.q36_n_expert_used,
                    ai.q36_n_ff_exp, ai.dg_eb_max_steps, ai.dg_eb_entropy_bound);
        } else if (is_g4moe) {
            /* Global-attention layers may omit attn_v entirely (V aliases K —
             * llama.cpp's "use_alternative_attention"). Report it, because a loader
             * that silently binds NULL there is the failure we would spend a day on. */
            int n_v = 0;
            for (uint64_t L = 0; L < n_layers; L++) {
                char tn[64]; snprintf(tn, sizeof tn, "blk.%llu.attn_v.weight", (unsigned long long)L);
                if (gguf_find_tensor(g, tn)) n_v++;
            }
            fprintf(stderr, "    [arch] gemma4-MoE (AR): experts=%u/%u expert_ff=%u dense_ff=%u; "
                            "attn_v present on %d/%llu layers (%llu global layers alias V:=K)\n",
                    ai.q36_n_expert, ai.q36_n_expert_used, ai.q36_n_ff_exp, ai.n_ff,
                    n_v, (unsigned long long)n_layers, (unsigned long long)(n_layers - (uint64_t)n_v));
        }
        #undef G4K
    }

    /* ── gemma4-assistant: the EAGLE/MTP DRAFT head. NOT a standalone LM — 4 tiny
     * layers (hidden 1024) hung off the 12B residual stream: nextn.pre_projection
     * [2*3840 -> 1024] consumes concat(target_hidden, token_embed), the 4 layers run,
     * nextn.post_projection [1024 -> 3840] returns to the 12B residual dim, and the
     * SHARED 12B embd+head produce logits. It carries the gemma4 GeGLU+sandwich
     * surface but a SMALL fixed geometry with NO sliding_window_pattern array, so it
     * gets its own fill (the is_gemma4_like SWA-period detection would hard-error).
     * head_dim above = key_length (GLOBAL, 512); g4_hd_swa = key_length_swa (256). ── */
    if (is_g4a) {
        uint64_t gv = 0; float gf = 0.0f;
        #define G4AK(suf) (snprintf(key, sizeof key, "gemma4-assistant." suf), (const char *)key)
        uint64_t hd_swa = 0;
        gguf_get_u64(g, G4AK("attention.key_length_swa"), &hd_swa);
        ai.g4_hd_swa  = (uint32_t)(hd_swa ? hd_swa : head_dim);
        ai.g4_nh_swa  = (uint32_t)n_head;                 /* n_head constant across layer types */
        ai.g4_rope_base_swa = gguf_get_f32(g, G4AK("rope.freq_base_swa"), &gf) ? gf : 1e4f;
        /* per-layer attention.head_count_kv is an ARRAY ([8,8,8,1]): the SWA layers
         * carry the larger count, the trailing global layer the smaller. Read both
         * ends directly (NO period detection — this head has no SWA pattern array). */
        {
            const gguf_kv *kv = gguf_find_kv(g, G4AK("attention.head_count_kv"));
            if (kv && kv->type == GGUF_T_ARRAY && kv->arr_data && kv->arr_len >= 1 &&
                (kv->arr_type == GGUF_T_UINT32 || kv->arr_type == GGUF_T_INT32)) {
                const uint32_t *hk = (const uint32_t *)kv->arr_data;
                ai.g4_nkv_swa = hk[0];                     /* SWA kv-heads (8)    */
                ai.n_kv_heads = hk[kv->arr_len - 1];       /* global kv-heads (1) */
            } else {
                ai.g4_nkv_swa = (uint32_t)n_head_kv;
            }
        }
        /* nextn/MTP: number of draft prediction blocks (== block_count here, 4). */
        if (gguf_get_u64(g, G4AK("nextn_predict_layers"), &gv)) ai.q36_nextn_predict_layers = (uint32_t)gv;
        ai.has_qk_norm = 1u;                               /* blk.*.attn_q_norm present */
        if (ai.n_ff == 0) {                                /* feed_forward_length scalar fallback */
            const gguf_tensor *fg0 = gguf_find_tensor(g, "blk.0.ffn_gate.weight");
            if (fg0 && fg0->n_dims >= 2) ai.n_ff = (uint32_t)fg0->dims[1];
        }
        fprintf(stderr, "    [arch] gemma4-assistant (EAGLE/MTP draft): layers=%u hidden=%u nh=%u "
                        "kv(swa=%u global=%u) hd(global=%u swa=%u) nextn=%u\n",
                ai.n_layers, ai.hidden_dim, ai.n_heads, ai.g4_nkv_swa, ai.n_kv_heads,
                ai.head_dim, ai.g4_hd_swa, ai.q36_nextn_predict_layers);
        #undef G4AK
    }

    /* ── qwen35moe q36_* fields (mirror qwen3_load): GDN geometry + MoE params +
     * IMRoPE sections. Base head_dim/n_heads/n_kv_heads hold the FULL-ATTN geometry. ── */
    if (is_qwen36) {
        uint64_t qv = 0; float qf2 = 0.0f;
        ai.q36_full_attn_interval = gguf_get_u64(g, "qwen35moe.full_attention_interval", &qv) ? (uint32_t)qv : 4u;
        if (gguf_get_u64(g, "qwen35moe.expert_count", &qv))                      ai.q36_n_expert      = (uint32_t)qv;
        if (gguf_get_u64(g, "qwen35moe.expert_used_count", &qv))                 ai.q36_n_expert_used = (uint32_t)qv;
        if (gguf_get_u64(g, "qwen35moe.expert_feed_forward_length", &qv))        ai.q36_n_ff_exp      = (uint32_t)qv;
        if (gguf_get_u64(g, "qwen35moe.expert_shared_feed_forward_length", &qv)) ai.q36_n_ff_shexp    = (uint32_t)qv;
        ai.q36_expert_weights_scale = gguf_get_f32(g, "qwen35moe.expert_weights_scale", &qf2) ? qf2 : 1.0f;
        if (gguf_get_u64(g, "qwen35moe.ssm.conv_kernel", &qv))    ai.q36_gdn_conv_k    = (uint32_t)qv;
        if (gguf_get_u64(g, "qwen35moe.ssm.state_size", &qv))     ai.q36_gdn_state     = (uint32_t)qv;
        if (gguf_get_u64(g, "qwen35moe.ssm.group_count", &qv))    ai.q36_gdn_n_k_heads = (uint32_t)qv;
        if (gguf_get_u64(g, "qwen35moe.ssm.time_step_rank", &qv)) ai.q36_gdn_n_v_heads = (uint32_t)qv;
        if (gguf_get_u64(g, "qwen35moe.ssm.inner_size", &qv))     ai.q36_gdn_inner     = (uint32_t)qv;
        ai.q36_rope_dim  = gguf_get_u64(g, "qwen35moe.rope.dimension_count", &qv) ? (uint32_t)qv : 64u;
        ai.q36_rope_base = rope_freq_base;
        {
            const gguf_kv *kv = gguf_find_kv(g, "qwen35moe.rope.dimension_sections");
            if (kv && kv->type == GGUF_T_ARRAY && kv->arr_type == GGUF_T_INT32 && kv->arr_data) {
                const int32_t *sec = (const int32_t *)kv->arr_data;
                for (int s = 0; s < 4; s++) ai.q36_rope_sections[s] = (s < (int)kv->arr_len) ? sec[s] : 0;
            }
        }
        if (gguf_get_u64(g, "qwen35moe.nextn_predict_layers", &qv)) ai.q36_nextn_predict_layers = (uint32_t)qv;
        ai.has_qk_norm = 1u;
        if (ai.n_ff == 0) ai.n_ff = ai.q36_n_ff_exp;   /* nonzero sentinel (MoE has no dense FFN) */
    }

    memset(arch_struct, 0, 256);
    memcpy(arch_struct, &ai, sizeof ai);
    *arch_id   = ai.arch_id;
    *arch_size = (uint32_t)sizeof(sp_arch_info);
    *vocab     = ai.vocab_size;
    return 0;
}

/* Decide per-tensor policy from the GGUF tensor name. Matmul weights -> Q8;
 * everything else (norms) -> F32. token_embd -> Q8 (the engine packs it in the
 * arena when SP_ARENA_EMBED). LM head "output.weight" (untied) -> Q8. */
static int is_matmul_weight(const char *name) {
    const char *bn = strrchr(name, '.');
    if (!bn) return 0;
    /* name forms: blk.N.<sub>.weight or token_embd.weight / output.weight */
    if (strstr(name, "_norm.weight")) return 0;   /* attn_norm, ffn_norm, q/k_norm, post_* */
    if (strcmp(name, "token_embd.weight") == 0) return 1;
    if (strcmp(name, "output.weight") == 0) return 1;
    if (strstr(name, "attn_q.weight") || strstr(name, "attn_k.weight") ||
        strstr(name, "attn_v.weight") || strstr(name, "attn_output.weight") ||
        strstr(name, "ffn_gate.weight") || strstr(name, "ffn_up.weight") ||
        strstr(name, "ffn_down.weight")) return 1;
    /* Gemma4 AltUp matmul weights (per-layer inp_gate/proj + the model-global
     * per_layer_token_embd / per_layer_model_proj). rope_freqs + layer_output_scale
     * stay F32 (freq table / scalar); *_norm caught above. */
    if (strstr(name, "inp_gate.weight") || strstr(name, "proj.weight") ||
        strstr(name, "projection.weight") ||   /* gemma4-assistant nextn.pre/post_projection */
        strstr(name, "per_layer_token_embd.weight")) return 1;
    /* qwen35moe (Qwen3.6): GDN + MoE matmul weights -> Q8. The router gates
     * (ffn_gate_inp.weight, ffn_gate_inp_shexp.weight) and ssm_conv1d/a/dt/norm
     * stay F32 (discrete top-k cliff / conv kernel / scalars; *_norm caught above).
     * ffn_{gate,up,down}_exps are rank-3 [cols,rows,n_expert] (handled in add_q8). */
    if (strstr(name, "attn_qkv.weight")  || strstr(name, "attn_gate.weight") ||
        strstr(name, "ssm_alpha.weight") || strstr(name, "ssm_beta.weight")  ||
        strstr(name, "ssm_out.weight")   ||
        strstr(name, "ffn_gate_exps.weight") || strstr(name, "ffn_up_exps.weight") ||
        strstr(name, "ffn_down_exps.weight") ||
        strstr(name, "ffn_gate_shexp.weight") || strstr(name, "ffn_up_shexp.weight") ||
        strstr(name, "ffn_down_shexp.weight")) return 1;
    /* diffusion-gemma (Phase 5): the FUSED MoE expert tensor ffn_gate_up_exps
     * [cols,1408,n_expert] (gate+up concatenated; the forward slices it at N1) is a
     * rank-3 expert matmul -> Q-transcode (rank-3 path in add_q4b/add_q8). The
     * decoder self-conditioning gated MLP weights (self_cond_{gate,up,down}.weight)
     * are matmuls. self_cond_pre_norm.weight is F32 (caught by _norm above);
     * ffn_gate_inp.weight (router) + *.scale stay F32 (NOT matched here). */
    if (strstr(name, "ffn_gate_up_exps.weight")) return 1;
    if (strcmp(name, "self_cond_gate.weight") == 0 ||
        strcmp(name, "self_cond_up.weight")   == 0 ||
        strcmp(name, "self_cond_down.weight") == 0) return 1;
    return 0;
}

int main(int argc, char **argv) {
    /* --tok-only (#115): extract ONLY the .sp-tokenizer (family-tagged blob)
     * from a GGUF — no tensor pass, works on vocab-only GGUFs. */
    if (argc == 4 && strcmp(argv[1], "--tok-only") == 0) {
        gguf_ctx *gt = gguf_open(argv[2]);
        if (!gt) { fprintf(stderr, "cannot open GGUF %s\n", argv[2]); return 1; }
        uint8_t sha[32]; uint32_t vocab = 0;
        int rc = write_tokenizer(argv[3], gt, &vocab, sha);
        gguf_close(gt);
        if (rc) return 1;
        fprintf(stderr, "tok-only: wrote %s (vocab %u)\n", argv[3], vocab);
        return 0;
    }
    if (argc < 4) {
        fprintf(stderr, "usage: %s <in.gguf> <out.sp-model> <out.sp-tokenizer> "
                        "[--verify] [--q4|--q8|--q4b] [--st <model.safetensors>]\n"
                        "       %s --tok-only <in.gguf> <out.sp-tokenizer>\n",
                argv[0], argv[0]);
        return 2;
    }
    const char *in = argv[1], *out_model = argv[2], *out_tok = argv[3];
    int verify = 0;
    int stream = 0;                              /* --stream: low-RAM (data region to temp file) */
    const char *st_path = NULL;
    for (int a = 4; a < argc; a++) {
        if      (strcmp(argv[a], "--verify") == 0) verify = 1;
        else if (strcmp(argv[a], "--stream") == 0) stream = 1;
        else if (strcmp(argv[a], "--st") == 0 && a + 1 < argc) st_path = argv[++a];
    }

    gguf_ctx *g = gguf_open(in);
    if (!g) { fprintf(stderr, "cannot open GGUF %s\n", in); return 1; }

    /* --stream: write the data region incrementally to a temp file beside the output so
     * peak RAM is ~one packed tensor (the 26B diffusion-gemma's ~14 GB OK_Q4B emit set
     * exceeds the commit limit if held all-in-RAM). Default off = the all-in-RAM path. */
    char stream_tmp[1200];
    if (stream) {
        snprintf(stream_tmp, sizeof stream_tmp, "%s.datatmp", out_model);
        g_stream = fopen(stream_tmp, "wb+");
        if (!g_stream) { fprintf(stderr, "cannot open stream temp %s\n", stream_tmp); gguf_close(g); return 1; }
        g_stream_off = 0;
        fprintf(stderr, "[stream] low-RAM mode: data region -> %s\n", stream_tmp);
    }

    if (st_path) {                               /* Safetensors Direct (gemma4) */
        g_st = st_open(st_path);
        if (!g_st) { gguf_close(g); return 1; }
        fprintf(stderr, "[st] SAFETENSORS DIRECT: weight values from %s; "
                        "GGUF supplies structure/KV/tokenizer/rope_freqs only\n", st_path);
    }

    /* 1. tokenizer -> .sp-tokenizer + its SHA-256 */
    uint8_t tok_sha[32]; uint32_t tok_vocab = 0;
    if (write_tokenizer(out_tok, g, &tok_vocab, tok_sha)) { gguf_close(g); return 1; }

    /* 2. arch_struct from the already-open GGUF context */
    uint8_t arch_struct[256]; uint32_t arch_id, arch_size, arch_vocab;
    if (fill_arch_struct(g, arch_struct, &arch_id, &arch_size, &arch_vocab)) { gguf_close(g); return 1; }
    if (arch_vocab != tok_vocab)
        fprintf(stderr, "warning: model vocab %u != tokenizer vocab %u\n", arch_vocab, tok_vocab);

    /* 3. transcode every tensor. token_embd FIRST (§9 access-frequency order),
     * then per-layer blocks in GGUF order, then output_norm. */
    emit_list L = {0};
    int rc = 0;
    /* Codec by source (C1: the converter REDUCES — never emit a wider codec than the
     * source). Sub-Q8 k-quant source (e.g. Q4_K_M) -> OK_Q4; Q8_0/F16/F32 -> OK_Q8.
     * Decided from token_embd's source type; overridable via --q4 / --q8. */
    int use_q4 = 0;
    {
        const gguf_tensor *te = gguf_find_tensor(g, "token_embd.weight");
        if (te) switch (te->type) {
            case GGML_T_Q4_0: case GGML_T_Q4_1: case GGML_T_Q4_K:
            case GGML_T_Q5_0: case GGML_T_Q5_1: case GGML_T_Q5_K:
            case GGML_T_Q6_K: use_q4 = 1; break;
            default: use_q4 = 0;
        }
        if (g_st) use_q4 = 0;                    /* bf16 source: default full-width OK_Q8 */
        for (int a = 4; a < argc; a++) {
            if      (strcmp(argv[a], "--q4") == 0)      use_q4 = 1;
            else if (strcmp(argv[a], "--q8") == 0)      use_q4 = 0;
            else if (strcmp(argv[a], "--q4b") == 0)     use_q4 = 2;  /* block-scaled, all */
            else if (strcmp(argv[a], "--q4b-ffn") == 0) use_q4 = 3;  /* RECIPE B1: Q4B on
                ffn_gate/ffn_up only, OK_Q8 rest — 12B sim'd at PPL 5.13 (+9.6% vs gold
                4.68), ~8.9 GB, fits the 2060-12GB (lattice CONTRACT-SPEED SPEC OK_Q4B) */
        }
        fprintf(stderr, "transcode codec: matmul weights -> %s\n",
                use_q4 == 3 ? "MIXED: OK_Q4B ffn_gate/up + OK_Q8 rest (recipe B1)" :
                use_q4 == 2 ? "OK_Q4B (per-32 f16 block scales)" :
                use_q4      ? "OK_Q4 (reducing)" : "OK_Q8");
    }
    for (uint64_t i = 0; i < gguf_n_tensors(g) && rc == 0; i++) {
        const gguf_tensor *W = gguf_tensor_at(g, i);
        if (is_matmul_weight(W->name)) {
            /* "Q4B EVERYWHERE, Q8 AT THE HEAD." token_embd is BOTH the embedding
             * gather source and (when tied) the LM head. The CUDA gather kernel
             * k_embed_packed_at reads row_scale[tok] with NO block-scale path, and
             * OK_Q4B leaves row_scale NULL by design (bscale governs) — so a Q4B
             * token_embd is a NULL deref on the FIRST token of the FIRST layer.
             * tools/model/repack_q4b.py already encodes exactly this rule ("keeping
             * token_embd.weight as OK_Q8 ... the Q4B embedding gather path wasn't
             * implemented"); --q4b silently ignored it and produced a model that
             * loads, verifies, and then faults on its first embed. Honour it here so
             * the container can never be minted in that state. */
            const int is_embed_src = (strcmp(W->name, "token_embd.weight") == 0);
            int q4b_this = !is_embed_src &&
                           ((use_q4 == 2) ||
                            (use_q4 == 3 && (strstr(W->name, "ffn_gate.weight") ||
                                             strstr(W->name, "ffn_up.weight"))));
            if (is_embed_src && use_q4 == 2)
                fprintf(stderr, "    [codec] token_embd.weight -> OK_Q8 (NOT Q4B): the CUDA "
                                "embed gather has no block-scale path\n");
            rc = q4b_this     ? add_q4b(&L, g, W)
               : use_q4 == 1  ? add_q4(&L, g, W) : add_q8(&L, g, W);
        } else rc = add_f32(&L, g, W);
    }
    if (rc) { el_free(&L); gguf_close(g); return 1; }

    /* 4. assign data offsets in emit order (parent-then-scale already adjacent),
     * each tensor 64-aligned. In stream mode emit_stream_one already set data_off
     * (and wrote the bytes to the temp file); data_region_size == the temp cursor. */
    uint64_t data_region_size;
    if (g_stream) {
        data_region_size = g_stream_off;
    } else {
        uint64_t data_cursor = 0;
        for (int i = 0; i < L.n; i++) {
            data_cursor = align_up(data_cursor, SP_TENSOR_ALIGN);
            L.t[i].data_off = data_cursor;
            data_cursor += L.t[i].size_bytes;
        }
        data_region_size = data_cursor;
    }

    /* 5. build the tensor table, sort by name_hash. Keep a parallel index into L
     * via the data_off (entries carry offset; data write follows emit order). */
    sp_tensor_entry *tbl = (sp_tensor_entry *)calloc((size_t)L.n, sizeof(sp_tensor_entry));
    if (!tbl) { el_free(&L); gguf_close(g); return 1; }
    for (int i = 0; i < L.n; i++) {
        sp_tensor_entry *e = &tbl[i];
        snprintf(e->name, sizeof e->name, "%s", L.t[i].name);
        e->dtype_id = L.t[i].dtype_id; e->n_dims = L.t[i].n_dims;
        for (int d = 0; d < 8; d++) e->dims[d] = L.t[i].dims[d];
        e->offset_in_data = L.t[i].data_off; e->size_bytes = L.t[i].size_bytes;
        e->block_size = L.t[i].block_size;
        e->block_count = e->block_size ? (uint32_t)(e->size_bytes / e->block_size) : 0;
        if (g_stream) memcpy(e->blake3, L.t[i].blake3, 32);   /* computed at emit time */
        else          sp_blake3_256(L.t[i].bytes, (size_t)L.t[i].size_bytes, e->blake3);
        e->name_hash = sp_xxh64(e->name, strlen(e->name), 0);
    }
    qsort(tbl, (size_t)L.n, sizeof(sp_tensor_entry), cmp_entry);

    /* 6. header */
    uint64_t table_off = SP_HEADER_SIZE;
    uint64_t table_end = table_off + (uint64_t)L.n * SP_TENSOR_ENTRY_SIZE;
    uint64_t data_off  = align_up(table_end, SP_DATA_REGION_ALIGN);
    uint64_t file_size = data_off + data_region_size;

    sp_model_header h; memset(&h, 0, sizeof h);
    h.magic = SP_MODEL_MAGIC; h.version_major = SP_MODEL_VER_MAJOR; h.version_minor = SP_MODEL_VER_MINOR;
    h.header_size = SP_HEADER_SIZE; h.arch_id = arch_id;
    h.arch_struct_size = arch_size; h.arch_struct_capacity = 256;
    memcpy(h.arch_struct, arch_struct, 256);
    memcpy(h.tokenizer_hash, tok_sha, 32);
    h.vocab_size = arch_vocab; h.tensor_count = (uint32_t)L.n;
    h.tensor_table_offset = table_off; h.tensor_data_offset = data_off;
    h.file_size = file_size; h.created_unix_seconds = (uint64_t)time(NULL);
    h.transcoded_from = sp_xxh64(in, strlen(in), 0);
    h.header_crc32 = sp_crc32(&h, SP_HEADER_CRC_COVER);

    /* 7. write the file: header | table | pad | data (emit order) */
    FILE *f = fopen(out_model, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", out_model); free(tbl); el_free(&L); gguf_close(g); return 1; }
    int ok = (fwrite(&h, 1, SP_HEADER_SIZE, f) == SP_HEADER_SIZE) &&
             (fwrite(tbl, sizeof(sp_tensor_entry), (size_t)L.n, f) == (size_t)L.n);
    /* pad to data_off */
    static const uint8_t zeros[65536] = {0};
    uint64_t pos = table_end;
    while (ok && pos < data_off) {
        uint64_t want = data_off - pos; if (want > sizeof zeros) want = sizeof zeros;
        ok = fwrite(zeros, 1, (size_t)want, f) == want; pos += want;
    }
    if (g_stream) {
        /* data region already on disk (verbatim, 64-aligned) in the temp file — copy it
         * in wholesale. The temp file's byte layout == what the in-RAM loop would emit. */
        if (fflush(g_stream) != 0) ok = 0;
        if (ok && fseek(g_stream, 0, SEEK_SET) != 0) ok = 0;
        uint8_t *cpbuf = (uint8_t *)malloc(1u << 22);   /* 4 MiB copy buffer */
        if (!cpbuf) ok = 0;
        uint64_t copied = 0;
        while (ok && copied < data_region_size) {
            size_t want = (size_t)((data_region_size - copied) < (1u << 22) ? (data_region_size - copied) : (1u << 22));
            size_t got = fread(cpbuf, 1, want, g_stream);
            if (got == 0) { ok = 0; break; }
            if (fwrite(cpbuf, 1, got, f) != got) { ok = 0; break; }
            copied += got;
        }
        free(cpbuf);
    } else {
        /* data region in emit order, 64-aligned per tensor */
        uint64_t dpos = 0;
        for (int i = 0; ok && i < L.n; i++) {
            uint64_t aligned = align_up(dpos, SP_TENSOR_ALIGN);
            while (ok && dpos < aligned) { uint64_t w = aligned - dpos; if (w > sizeof zeros) w = sizeof zeros; ok = fwrite(zeros,1,(size_t)w,f)==w; dpos += w; }
            ok = fwrite(L.t[i].bytes, 1, (size_t)L.t[i].size_bytes, f) == L.t[i].size_bytes;
            dpos += L.t[i].size_bytes;
        }
    }
    fclose(f);
    if (g_stream) { fclose(g_stream); g_stream = NULL; remove(stream_tmp); }
    if (!ok) { fprintf(stderr, "model write short\n"); free(tbl); el_free(&L); gguf_close(g); return 1; }

    fprintf(stderr, "[sp_transcode] %s -> %s (%u tensors, %llu bytes) + %s\n",
            in, out_model, h.tensor_count, (unsigned long long)file_size, out_tok);
    if (g_st)
        fprintf(stderr, "[st] value sources: %d tensors from safetensors, %d from GGUF (unmapped)\n",
                g_st_hits, g_st_misses);

    /* 8. --verify: sibling adjacency (§9) + reload sanity. */
    if (verify) {
        int warned = 0;
        /* For each Q8 weight in emit order, the next emit tensor must be its .scale
         * and they must be data-adjacent (parent end == scale start, both 64-aligned
         * boundaries notwithstanding — adjacency means no THIRD tensor between). */
        for (int i = 0; i + 1 < L.n; i++) {
            if (L.t[i].dtype_id == SP_DT_OK_Q8) {
                char want[80]; snprintf(want, sizeof want, "%s.scale", L.t[i].name);
                if (strcmp(L.t[i+1].name, want) != 0) {
                    fprintf(stderr, "[verify] WARN: %s scale not adjacent (next is %s)\n", L.t[i].name, L.t[i+1].name);
                    warned = 1;
                }
            }
        }
        sp_model *vm = NULL;
        sp_status st = sp_model_load(out_model, out_tok, &vm);
        if (st != SP_OK) { fprintf(stderr, "[verify] sp_model_load failed: %d (%s)\n", st, sp_last_error()); rc = 1; }
        else { fprintf(stderr, "[verify] load OK, %u tensors%s\n", sp_model_tensor_count(vm), warned ? " (adjacency warnings above)" : ""); sp_model_unload(vm); }
    }

    free(tbl); el_free(&L); gguf_close(g);
    st_close(g_st); g_st = NULL;
    return rc;
}
