/* tokenizer.c — GGUF/.sp-tokenizer vocab load + family dispatch.
 *
 * Families (tokenizer_internal.h; unknown family = HARD ERROR, never a silent
 * fallback — SPEC-gemma4-tokenizer-dispatch.md):
 *   GPT2_BPE   ("gpt2")   — byte-level BPE, Qwen family (this file);
 *   SPM        ("llama")  — SentencePiece bigram-merge, Gemma3 (this file);
 *   GEMMA4_BPE ("gemma4") — 514k-merge U+2581-piece BPE (gemma4_bpe.c, #115).
 *
 * GPT2 ENCODE (UTF-8 -> token IDs): the Qwen2 pipeline —
 *   1. optional special-token pre-split (CONTROL/USER_DEFINED surfaces, longest
 *      match first), with the gaps tokenized as ordinary text;
 *   2. the Qwen2 pre-tokenizer regex split (hand-coded; classes \p{L}/\p{N}/\s
 *      come from the generated unicode_ranges.h);
 *   3. GPT2 byte-level encode of each piece;
 *   4. greedy lowest-rank BPE over tokenizer.ggml.merges;
 *   5. token->id lookup.
 * Validated to reproduce stock llama.cpp IDs byte-for-byte (see tools/oracle/
 * bpe_proto.py and the TOK_ENCODE test). See sp_engine/tokenizer.h. */
#include "tokenizer_internal.h"
#include "sp_engine/sp_model.h"   /* sp_tok_header + sp_tok_type_id (blob lane) */
#include "unicode_ranges.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- string hash map (open addressing, linear probe): key = bytes -> int64 --
 * (typedefs live in tokenizer_internal.h; shared with gemma4_bpe.c) */

static uint64_t fnv1a(const char *s, uint32_t n) {
    uint64_t h = 1469598103934665603ull;
    for (uint32_t i = 0; i < n; i++) { h ^= (unsigned char)s[i]; h *= 1099511628211ull; }
    return h;
}
static int hmap_init(sp_hmap *m, size_t want) {
    size_t cap = 16; while (cap < want * 2) cap <<= 1;
    m->e = (sp_hent *)calloc(cap, sizeof *m->e);
    if (!m->e) return 0;
    m->mask = cap - 1; return 1;
}
static void hmap_free(sp_hmap *m) { free(m->e); m->e = NULL; m->mask = 0; }
/* insert; first writer for a key wins (keeps the lowest merge rank). */
static void hmap_put(sp_hmap *m, const char *k, uint32_t kl, int64_t v) {
    size_t i = fnv1a(k, kl) & m->mask;
    while (m->e[i].used) {
        if (m->e[i].klen == kl && memcmp(m->e[i].key, k, kl) == 0) return;
        i = (i + 1) & m->mask;
    }
    m->e[i].key = k; m->e[i].klen = kl; m->e[i].val = v; m->e[i].used = 1;
}
static int64_t hmap_get(const sp_hmap *m, const char *k, uint32_t kl, int64_t dflt) {
    size_t i = fnv1a(k, kl) & m->mask;
    while (m->e[i].used) {
        if (m->e[i].klen == kl && memcmp(m->e[i].key, k, kl) == 0) return m->e[i].val;
        i = (i + 1) & m->mask;
    }
    return dflt;
}

/* (struct sp_tokenizer + sp_special live in tokenizer_internal.h) */

int64_t sp_tok_vocab_lookup(const sp_tokenizer *t, const char *bytes, uint32_t len) {
    return hmap_get(&t->vocab, bytes, len, -1);
}

/* GPT2 bytes_to_unicode: printable bytes (33..126,161..172,174..255) map to
 * themselves; the remaining 68 map to codepoints 256.. in order. */
static void build_byte_maps(sp_tokenizer *t) {
    int printable[256]; for (int b = 0; b < 256; b++) printable[b] = 0;
    for (int b = 33;  b <= 126; b++) printable[b] = 1;
    for (int b = 161; b <= 172; b++) printable[b] = 1;
    for (int b = 174; b <= 255; b++) printable[b] = 1;
    int byte_to_cp[256], n = 0;
    for (int b = 0; b < 256; b++) byte_to_cp[b] = printable[b] ? b : (256 + n++);
    int mx = 0;
    for (int b = 0; b < 256; b++) if (byte_to_cp[b] > mx) mx = byte_to_cp[b];
    t->max_cp = mx;
    for (int c = 0; c <= mx; c++) t->cp_to_byte[c] = -1;
    for (int b = 0; b < 256; b++) {
        int cp = byte_to_cp[b];
        t->cp_to_byte[cp] = b;
        if (cp < 0x80) { t->bcp[b][0] = (uint8_t)cp; t->bcp_len[b] = 1; }
        else { t->bcp[b][0] = (uint8_t)(0xC0 | (cp >> 6)); t->bcp[b][1] = (uint8_t)(0x80 | (cp & 0x3F)); t->bcp_len[b] = 2; }
    }
}

static int spec_cmp_desc(const void *a, const void *b) {
    uint32_t la = ((const sp_special *)a)->len, lb = ((const sp_special *)b)->len;
    return (la < lb) - (la > lb);   /* longer first */
}

sp_tokenizer *sp_tokenizer_load(const gguf_ctx *g) { return sp_tokenizer_load_ex(g, 0); }

sp_tokenizer *sp_tokenizer_load_ex(const gguf_ctx *g, int own) {
    const gguf_kv *kv = gguf_find_kv(g, "tokenizer.ggml.tokens");
    if (!kv || kv->type != GGUF_T_ARRAY || kv->arr_type != GGUF_T_STRING) return NULL;
    uint64_t nv = kv->arr_len;
    if (nv == 0 || nv > 100000000ull) return NULL;

    sp_tokenizer *t = (sp_tokenizer *)calloc(1, sizeof *t);
    if (!t) return NULL;
    t->n_vocab = (uint32_t)nv;
    t->tok = (const char **)malloc((size_t)nv * sizeof(char *));
    t->len = (uint64_t *)malloc((size_t)nv * sizeof(uint64_t));
    if (!t->tok || !t->len) { sp_tokenizer_free(t); return NULL; }
    if (gguf_kv_str_array(g, kv, t->tok, t->len, nv) != nv) { sp_tokenizer_free(t); return NULL; }
    build_byte_maps(t);

    /* owning mode: copy the token bytes into an owned blob and repoint tok[] so
     * the tokenizer no longer borrows the GGUF mapping. */
    if (own) {
        uint64_t total = 0;
        for (uint64_t i = 0; i < nv; i++) total += t->len[i];
        t->tok_blob = (char *)malloc((size_t)(total ? total : 1));
        if (!t->tok_blob) { sp_tokenizer_free(t); return NULL; }
        uint64_t off = 0;
        for (uint64_t i = 0; i < nv; i++) {
            memcpy(t->tok_blob + off, t->tok[i], (size_t)t->len[i]);
            t->tok[i] = t->tok_blob + off;
            off += t->len[i];
        }
    }

    /* vocab map: token bytes -> id (keys are owned when own, else into the mapping) */
    if (!hmap_init(&t->vocab, (size_t)nv)) { sp_tokenizer_free(t); return NULL; }
    for (uint32_t i = 0; i < t->n_vocab; i++)
        hmap_put(&t->vocab, t->tok[i], (uint32_t)t->len[i], (int64_t)i);

    /* family dispatch — BEFORE the merge tables (gemma4 keys merges by id).
     * Unknown family = HARD ERROR naming the string; never a silent GPT2
     * fallback (SPEC-gemma4-tokenizer-dispatch.md / #115). */
    {
        const char *model = gguf_get_str(g, "tokenizer.ggml.model");
        const char *pre   = gguf_get_str(g, "tokenizer.ggml.pre");
        if (model && strcmp(model, "llama") == 0) {
            t->family = SP_TOKFAM_SPM;
        } else if (model && (strcmp(model, "gemma4") == 0 ||
                   (strcmp(model, "gpt2") == 0 && pre && strcmp(pre, "gemma4") == 0))) {
            /* llama-vocab.cpp:1894 (model=="gemma4") + :2005 (pre=="gemma4") */
            t->family = SP_TOKFAM_GEMMA4_BPE;
        } else if (model && strcmp(model, "gpt2") == 0) {
            t->family = SP_TOKFAM_GPT2_BPE;
        } else {
            fprintf(stderr, "sp_tokenizer: unknown tokenizer family '%s' "
                            "(tokenizer.ggml.model%s%s) — refusing silent GPT2 fallback\n",
                    model ? model : "(missing)",
                    pre ? ", pre=" : "", pre ? pre : "");
            sp_tokenizer_free(t); return NULL;
        }
        t->spm = (t->family == SP_TOKFAM_SPM);
    }

    /* merge tables. GEMMA4: hashed id-pair -> rank (gemma4_bpe.c). Other
     * families: "A B" merge-line bytes -> rank (line index), unchanged. */
    const gguf_kv *mkv = gguf_find_kv(g, "tokenizer.ggml.merges");
    if (t->family == SP_TOKFAM_GEMMA4_BPE) {
        if (!mkv || mkv->type != GGUF_T_ARRAY || mkv->arr_type != GGUF_T_STRING ||
            mkv->arr_len == 0) {
            fprintf(stderr, "sp_tokenizer(gemma4): tokenizer.ggml.merges missing/empty\n");
            sp_tokenizer_free(t); return NULL;
        }
        uint64_t nm = mkv->arr_len;
        const char **mp = (const char **)malloc((size_t)nm * sizeof(char *));
        uint64_t    *ml = (uint64_t *)malloc((size_t)nm * sizeof(uint64_t));
        int ok = mp && ml && gguf_kv_str_array(g, mkv, mp, ml, nm) == nm &&
                 sp_g4_build(t, mp, ml, nm) == 0;
        free((void *)mp); free(ml);   /* id tables built; merge bytes not retained */
        if (!ok) { sp_tokenizer_free(t); return NULL; }
    } else if (mkv && mkv->type == GGUF_T_ARRAY && mkv->arr_type == GGUF_T_STRING && mkv->arr_len > 0) {
        uint64_t nm = mkv->arr_len;
        const char **mp = (const char **)malloc((size_t)nm * sizeof(char *));
        uint64_t    *ml = (uint64_t *)malloc((size_t)nm * sizeof(uint64_t));
        if (mp && ml && gguf_kv_str_array(g, mkv, mp, ml, nm) == nm && hmap_init(&t->merge, (size_t)nm)) {
            const char **key = mp;
            if (own) {                 /* copy merge bytes into an owned blob, key off it */
                uint64_t total = 0;
                for (uint64_t i = 0; i < nm; i++) total += ml[i];
                t->merge_blob = (char *)malloc((size_t)(total ? total : 1));
                if (t->merge_blob) {
                    uint64_t off = 0;
                    for (uint64_t i = 0; i < nm; i++) {
                        memcpy(t->merge_blob + off, mp[i], (size_t)ml[i]);
                        mp[i] = t->merge_blob + off;   /* reuse mp[] as the owned-pointer array */
                        off += ml[i];
                    }
                }
            }
            for (uint64_t i = 0; i < nm; i++) hmap_put(&t->merge, key[i], (uint32_t)ml[i], (int64_t)i);
        }
        free((void *)mp); free(ml);   /* the pointer arrays are scratch; keys live in mapping or blob */
    }

    /* special surfaces: token_type CONTROL(3) or USER_DEFINED(4) */
    const gguf_kv *tt = gguf_find_kv(g, "tokenizer.ggml.token_type");
    if (tt && tt->type == GGUF_T_ARRAY &&
        (tt->arr_type == GGUF_T_INT32 || tt->arr_type == GGUF_T_UINT32) &&
        tt->arr_len == nv) {
        const int32_t *types = (const int32_t *)tt->arr_data;
        uint32_t cnt = 0;
        for (uint64_t i = 0; i < nv; i++) if (types[i] == 3 || types[i] == 4) cnt++;
        if (cnt) {
            t->spec = (sp_special *)malloc(cnt * sizeof *t->spec);
            if (t->spec) {
                uint32_t j = 0;
                for (uint32_t i = 0; i < t->n_vocab; i++)
                    if (types[i] == 3 || types[i] == 4) {
                        t->spec[j].surf = t->tok[i];
                        t->spec[j].len  = (uint32_t)t->len[i];
                        t->spec[j].id   = (int32_t)i;
                        j++;
                    }
                t->n_spec = j;
                qsort(t->spec, t->n_spec, sizeof *t->spec, spec_cmp_desc);
            }
        }
    }

    /* special token ids + BOS policy (family set above). */
    t->bos_id = -1; t->eos_id = -1; t->pad_id = -1; t->unk_id = -1;
    { uint64_t b; if (gguf_get_u64(g, "tokenizer.ggml.bos_token_id", &b)) t->bos_id = (int32_t)b; }
    { uint64_t a; if (gguf_get_u64(g, "tokenizer.ggml.add_bos_token", &a)) t->add_bos = (int)a; }
    { uint64_t v; if (gguf_get_u64(g, "tokenizer.ggml.eos_token_id", &v))     t->eos_id = (int32_t)v; }
    { uint64_t v; if (gguf_get_u64(g, "tokenizer.ggml.padding_token_id", &v)) t->pad_id = (int32_t)v; }
    { uint64_t v; if (gguf_get_u64(g, "tokenizer.ggml.unknown_token_id", &v)) t->unk_id = (int32_t)v; }
    if (t->family == SP_TOKFAM_GEMMA4_BPE) {
        t->add_bos = 1;   /* FORCED for gemma4 (llama-vocab.cpp:2338-2344, PR #21500) */
        if (t->bos_id < 0 || (uint32_t)t->bos_id >= t->n_vocab) {
            fprintf(stderr, "sp_tokenizer(gemma4): bos_token_id missing/out of range\n");
            sp_tokenizer_free(t); return NULL;
        }
    }

    /* SPM ("llama") path: Gemma3 etc. need the unigram scores + byte-fallback
     * tokens. (BPE models leave spm=0 and never touch these.) */
    if (t->spm) {
        const gguf_kv *sk = gguf_find_kv(g, "tokenizer.ggml.scores");
        if (!sk || sk->type != GGUF_T_ARRAY || sk->arr_type != GGUF_T_FLOAT32 ||
            sk->arr_len != nv) { sp_tokenizer_free(t); return NULL; }
        t->scores = (const float *)sk->arr_data;
        if (own) {
            t->scores_blob = (float *)malloc((size_t)nv * sizeof(float));
            if (!t->scores_blob) { sp_tokenizer_free(t); return NULL; }
            memcpy(t->scores_blob, t->scores, (size_t)nv * sizeof(float));
            t->scores = t->scores_blob;
        }
        /* byte-fallback table: byte b -> id of "<0xXX>" (must all resolve) */
        for (int b = 0; b < 256; b++) {
            char nm[8]; int nl = snprintf(nm, sizeof nm, "<0x%02X>", b);
            t->byte_tok[b] = (int32_t)hmap_get(&t->vocab, nm, (uint32_t)nl, -1);
            if (t->byte_tok[b] < 0) { sp_tokenizer_free(t); return NULL; }
        }
    }
    return t;
}

void sp_tokenizer_free(sp_tokenizer *t) {
    if (!t) return;
    hmap_free(&t->vocab); hmap_free(&t->merge);
    sp_g4_free(t);
    free(t->spec); free((void *)t->tok); free(t->len);
    free(t->tok_blob); free(t->merge_blob); free(t->scores_blob);
    free(t->file_blob);
    free(t);
}

uint32_t sp_tokenizer_vocab_size(const sp_tokenizer *t) { return t ? t->n_vocab : 0; }

/* ---- decode (unchanged) ----------------------------------------------------- */
static void put(char *buf, size_t cap, size_t *pos, unsigned char b) {
    if (*pos + 1 < cap) buf[*pos] = (char)b;
    (*pos)++;
}

long sp_tokenizer_decode(const sp_tokenizer *t, const int32_t *ids, int n,
                         char *buf, size_t cap) {
    if (!t || (!ids && n > 0) || (!buf && cap > 0)) return -1;
    if (t->family == SP_TOKFAM_GEMMA4_BPE) return sp_g4_decode(t, ids, n, buf, cap);
    size_t pos = 0;
    for (int k = 0; k < n; k++) {
        int32_t id = ids[k];
        if (id < 0 || (uint32_t)id >= t->n_vocab) continue;   /* skip OOR */
        const char *s = t->tok[id];
        uint64_t L = t->len[id];
        uint64_t i = 0;
        while (i < L) {
            unsigned char c0 = (unsigned char)s[i];
            uint32_t cp; uint64_t adv;
            if (c0 < 0x80)                                  { cp = c0; adv = 1; }
            else if ((c0 & 0xE0) == 0xC0 && i + 1 < L)      { cp = ((uint32_t)(c0 & 0x1F) << 6) | (s[i+1] & 0x3F); adv = 2; }
            else if ((c0 & 0xF0) == 0xE0 && i + 2 < L)      { cp = ((uint32_t)(c0 & 0x0F) << 12) | ((uint32_t)(s[i+1] & 0x3F) << 6) | (s[i+2] & 0x3F); adv = 3; }
            else if ((c0 & 0xF8) == 0xF0 && i + 3 < L)      { cp = ((uint32_t)(c0 & 0x07) << 18) | ((uint32_t)(s[i+1] & 0x3F) << 12) | ((uint32_t)(s[i+2] & 0x3F) << 6) | (s[i+3] & 0x3F); adv = 4; }
            else                                            { cp = c0; adv = 1; }
            int byte = (cp <= (uint32_t)t->max_cp) ? t->cp_to_byte[cp] : -1;
            if (byte >= 0) put(buf, cap, &pos, (unsigned char)byte);
            else for (uint64_t j = 0; j < adv; j++) put(buf, cap, &pos, (unsigned char)s[i + j]);
            i += adv;
        }
    }
    if (cap > 0) buf[pos < cap ? pos : cap - 1] = '\0';
    return (long)pos;
}

/* ---- encode helpers --------------------------------------------------------- */
#define CR 0x0Du
#define LF 0x0Au
#define SP 0x20u

static int in_ranges(const uint32_t (*r)[2], size_t n, uint32_t cp) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t mid = (lo + hi) >> 1;
        if (cp < r[mid][0]) hi = mid;
        else if (cp > r[mid][1]) lo = mid + 1;
        else return 1;
    }
    return 0;
}
static int is_L(uint32_t c) { return in_ranges(sp_uni_letter, sp_uni_letter_n, c); }
static int is_N(uint32_t c) { return in_ranges(sp_uni_number, sp_uni_number_n, c); }
static int is_S(uint32_t c) { return in_ranges(sp_uni_space,  sp_uni_space_n,  c); }
/* class for ` ?[^\s\p{L}\p{N}]+`: not whitespace, letter, or number. */
static int is_C4(uint32_t c) { return !is_S(c) && !is_L(c) && !is_N(c); }
static uint32_t lc(uint32_t c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; }

/* decode one UTF-8 codepoint at s[i..len); writes *cp, returns byte advance. */
static size_t cp_decode(const unsigned char *s, size_t len, size_t i, uint32_t *cp) {
    unsigned char c0 = s[i];
    if (c0 < 0x80)                              { *cp = c0; return 1; }
    if ((c0 & 0xE0) == 0xC0 && i + 1 < len)     { *cp = ((uint32_t)(c0 & 0x1F) << 6) | (s[i+1] & 0x3F); return 2; }
    if ((c0 & 0xF0) == 0xE0 && i + 2 < len)     { *cp = ((uint32_t)(c0 & 0x0F) << 12) | ((uint32_t)(s[i+1] & 0x3F) << 6) | (s[i+2] & 0x3F); return 3; }
    if ((c0 & 0xF8) == 0xF0 && i + 3 < len)     { *cp = ((uint32_t)(c0 & 0x07) << 18) | ((uint32_t)(s[i+1] & 0x3F) << 12) | ((uint32_t)(s[i+2] & 0x3F) << 6) | (s[i+3] & 0x3F); return 4; }
    *cp = c0; return 1;   /* malformed: one byte */
}

/* Qwen2 pre-tokenizer: given the codepoint array cp[0..n), return the end index
 * (exclusive) of the pre-token that starts at position i (always > i). Mirrors
 * the alternation
 *   (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])
 *   |[^\r\n\p{L}\p{N}]?\p{L}+ | \p{N}
 *   | ?[^\s\p{L}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
 * with ordered (first-match-wins) alternation and the \s+(?!\S) backtrack. */
static size_t next_piece(const uint32_t *cp, size_t n, size_t i) {
    /* ALT1 — contraction suffixes */
    if (cp[i] == 0x27 && i + 1 < n) {
        uint32_t c1 = lc(cp[i + 1]);
        if (i + 2 < n) {
            uint32_t c2 = lc(cp[i + 2]);
            if (((c1 == 'r' || c1 == 'v') && c2 == 'e') || (c1 == 'l' && c2 == 'l')) return i + 3;
        }
        if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') return i + 2;
    }
    /* ALT2 — [^\r\n\p{L}\p{N}]? \p{L}+ */
    {
        size_t k = i;
        if (cp[k] != CR && cp[k] != LF && !is_L(cp[k]) && !is_N(cp[k])) {
            if (k + 1 < n && is_L(cp[k + 1])) k++;   /* greedy optional prefix, needs a letter next */
        }
        if (k < n && is_L(cp[k])) {
            while (k < n && is_L(cp[k])) k++;
            return k;
        }
    }
    /* ALT3 — \p{N} (single) */
    if (is_N(cp[i])) return i + 1;
    /* ALT4 —  ?[^\s\p{L}\p{N}]+[\r\n]* */
    {
        size_t k = i;
        if (cp[k] == SP) {
            if (k + 1 < n && is_C4(cp[k + 1])) k++;  /* consume the optional leading space */
            else goto alt4_done;                     /* lone space is \s, not class C4 */
        }
        if (k < n && is_C4(cp[k])) {
            while (k < n && is_C4(cp[k])) k++;
            while (k < n && (cp[k] == CR || cp[k] == LF)) k++;
            return k;
        }
    }
alt4_done:;
    /* ALT5/6/7 — whitespace run (cp[i] is whitespace if we reach here) */
    {
        size_t e = i; while (e < n && is_S(cp[e])) e++;
        size_t lastnl = (size_t)-1;
        for (size_t k = i; k < e; k++) if (cp[k] == CR || cp[k] == LF) lastnl = k;
        if (lastnl != (size_t)-1) return lastnl + 1;   /* ALT5: \s*[\r\n]+         */
        if (e == n)              return e;             /* ALT6: \s+ at end-of-text */
        if (e - i >= 2)          return e - 1;         /* ALT6: \s+(?!\S) backtrack */
        return e;                                      /* ALT7: single \s before \S */
    }
}

/* append id to out (capacity max_out); always advance *cnt so the caller can
 * detect truncation (return value > max_out). */
static void emit(int32_t *out, int max_out, long *cnt, int32_t id) {
    if (*cnt < max_out) out[*cnt] = id;
    (*cnt)++;
}

/* BPE-encode one pre-token (raw bytes [rb,rb+rn)) into out, using scratch
 * buffers: enc[] (byte-level UTF-8, >= 2*rn), sym off/len arrays (>= rn),
 * key[] (>= enc capacity + 1). */
static int bpe_piece(const sp_tokenizer *t, const unsigned char *rb, size_t rn,
                     unsigned char *enc, uint32_t *soff, uint32_t *slen, unsigned char *key,
                     int32_t *out, int max_out, long *cnt) {
    /* byte-level encode; one symbol per source byte */
    size_t ne = 0, ns = 0;
    for (size_t i = 0; i < rn; i++) {
        unsigned char b = rb[i];
        soff[ns] = (uint32_t)ne;
        slen[ns] = t->bcp_len[b];
        for (uint8_t j = 0; j < t->bcp_len[b]; j++) enc[ne++] = t->bcp[b][j];
        ns++;
    }
    /* greedy lowest-rank merges */
    while (ns > 1) {
        int64_t best = -1; size_t bi = 0;
        for (size_t i = 0; i + 1 < ns; i++) {
            uint32_t la = slen[i], lb = slen[i + 1];
            memcpy(key, enc + soff[i], la);
            key[la] = ' ';
            memcpy(key + la + 1, enc + soff[i + 1], lb);
            int64_t r = hmap_get(&t->merge, (const char *)key, la + 1 + lb, -1);
            if (r >= 0 && (best < 0 || r < best)) { best = r; bi = i; }
        }
        if (best < 0) break;
        slen[bi] += slen[bi + 1];                       /* symbols are contiguous in enc */
        for (size_t k = bi + 1; k + 1 < ns; k++) { soff[k] = soff[k + 1]; slen[k] = slen[k + 1]; }
        ns--;
    }
    /* map final symbols to ids */
    for (size_t i = 0; i < ns; i++) {
        int64_t id = hmap_get(&t->vocab, (const char *)(enc + soff[i]), slen[i], -1);
        if (id < 0) return -1;                          /* should not happen for byte-level BPE */
        emit(out, max_out, cnt, (int32_t)id);
    }
    return 0;
}

/* tokenize a raw text range [s,s+len) (no specials) into out. */
static int encode_text(const sp_tokenizer *t, const unsigned char *s, size_t len,
                       uint32_t *cps, unsigned char *enc, uint32_t *soff, uint32_t *slen,
                       unsigned char *key, int32_t *out, int max_out, long *cnt) {
    if (len == 0) return 0;
    /* decode UTF-8 -> codepoints, remembering each codepoint's byte span */
    size_t ncp = 0;
    size_t boff_buf_cap = len + 1;
    uint32_t *boff = (uint32_t *)malloc(boff_buf_cap * sizeof(uint32_t));
    if (!boff) return -1;
    for (size_t i = 0; i < len; ) {
        uint32_t c; size_t adv = cp_decode(s, len, i, &c);
        cps[ncp] = c; boff[ncp] = (uint32_t)i; ncp++;
        i += adv;
    }
    boff[ncp] = (uint32_t)len;   /* sentinel: byte end of last codepoint */
    int rc = 0;
    for (size_t i = 0; i < ncp; ) {
        size_t e = next_piece(cps, ncp, i);             /* [i,e) in codepoints */
        size_t rstart = boff[i], rend = boff[e];        /* -> raw byte span    */
        rc = bpe_piece(t, s + rstart, rend - rstart, enc, soff, slen, key, out, max_out, cnt);
        if (rc) break;
        i = e;
    }
    free(boff);
    return rc;
}

/* ===== SPM (SentencePiece "llama") encode — Gemma3 etc. ====================== *
 * Greedy bigram-merge by unigram score, byte-parity with llama.cpp's
 * llm_tokenizer_spm_session. Normalisation: spaces -> U+2581 (no space prefix on
 * Gemma, add_space_prefix=0). Byte fallback for symbols not in the vocab. Since a
 * merge only ever forms a vocab token, resegment is just token-or-byte-fallback
 * (the oracle's rev_merge recursion is dead for non-"unused" tokens). */

static int utf8_len(unsigned char c0) {
    if (c0 < 0x80) return 1;
    if ((c0 & 0xE0) == 0xC0) return 2;
    if ((c0 & 0xF0) == 0xE0) return 3;
    if ((c0 & 0xF8) == 0xF0) return 4;
    return 1;   /* malformed lead byte: one byte */
}

typedef struct { int prev, next; const char *text; int n; } spm_sym;
typedef struct { int left, right; float score; int size; } spm_bigram;

/* max-heap priority: higher score first; tie -> smaller left index (matches the
 * oracle comparator (l.score<r.score)||(l.score==r.score && l.left>r.left)). */
static int spm_hi(const spm_bigram *a, const spm_bigram *b) {
    return (a->score > b->score) || (a->score == b->score && a->left < b->left);
}
static void spm_push(spm_bigram *h, int *hn, spm_bigram bg) {
    int i = (*hn)++; h[i] = bg;
    while (i > 0) { int p = (i - 1) >> 1; if (!spm_hi(&h[i], &h[p])) break;
        spm_bigram tmp = h[i]; h[i] = h[p]; h[p] = tmp; i = p; }
}
static spm_bigram spm_pop(spm_bigram *h, int *hn) {
    spm_bigram top = h[0]; h[0] = h[--(*hn)];
    int i = 0;
    for (;;) { int l = 2*i+1, r = 2*i+2, m = i;
        if (l < *hn && spm_hi(&h[l], &h[m])) m = l;
        if (r < *hn && spm_hi(&h[r], &h[m])) m = r;
        if (m == i) break;
        spm_bigram tmp = h[i]; h[i] = h[m]; h[m] = tmp; i = m; }
    return top;
}

/* Encode one raw-text fragment [s,len) (no specials) into out via SPM merge. */
static int spm_fragment(const sp_tokenizer *t, const unsigned char *s, size_t len,
                        int32_t *out, int max_out, long *cnt) {
    if (len == 0) return 0;
    /* escape: every ' ' -> U+2581 (E2 96 81); other bytes copied. */
    unsigned char *esc = (unsigned char *)malloc(3 * len + 1);
    if (!esc) return -1;
    size_t el = 0;
    for (size_t i = 0; i < len; i++) {
        if (s[i] == 0x20) { esc[el++] = 0xE2; esc[el++] = 0x96; esc[el++] = 0x81; }
        else esc[el++] = s[i];
    }
    /* split into UTF-8-char symbols (doubly linked over an array). */
    spm_sym *sym = (spm_sym *)malloc((el + 1) * sizeof *sym);
    spm_bigram *heap = (spm_bigram *)malloc((4 * el + 16) * sizeof *heap);
    if (!sym || !heap) { free(esc); free(sym); free(heap); return -1; }
    int ns = 0;
    for (size_t o = 0; o < el; ) {
        int L = utf8_len(esc[o]); if (o + (size_t)L > el) L = (int)(el - o);
        sym[ns].text = (const char *)(esc + o); sym[ns].n = L;
        sym[ns].prev = ns - 1; sym[ns].next = (o + (size_t)L >= el) ? -1 : ns + 1;
        o += (size_t)L; ns++;
    }
    int hn = 0;
    #define SPM_TRY(LFT, RGT) do { \
        int l_ = (LFT), r_ = (RGT); \
        if (l_ >= 0 && r_ >= 0) { \
            int nn = sym[l_].n + sym[r_].n; \
            int64_t id_ = hmap_get(&t->vocab, sym[l_].text, (uint32_t)nn, -1); \
            if (id_ >= 0) { spm_bigram bg; bg.left = l_; bg.right = r_; \
                bg.score = t->scores[id_]; bg.size = nn; spm_push(heap, &hn, bg); } \
        } } while (0)
    for (int i = 1; i < ns; i++) SPM_TRY(i - 1, i);
    while (hn > 0) {
        spm_bigram bg = spm_pop(heap, &hn);
        spm_sym *Lf = &sym[bg.left], *Rt = &sym[bg.right];
        if (Lf->n == 0 || Rt->n == 0 || Lf->n + Rt->n != bg.size) continue;
        Lf->n += Rt->n; Rt->n = 0;
        Lf->next = Rt->next;
        if (Rt->next >= 0) sym[Rt->next].prev = bg.left;
        SPM_TRY(Lf->prev, bg.left);
        SPM_TRY(bg.left, Lf->next);
    }
    #undef SPM_TRY
    for (int i = 0; i != -1; i = sym[i].next) {
        int64_t id = hmap_get(&t->vocab, sym[i].text, (uint32_t)sym[i].n, -1);
        if (id >= 0) emit(out, max_out, cnt, (int32_t)id);
        else for (int j = 0; j < sym[i].n; j++)
            emit(out, max_out, cnt, t->byte_tok[(unsigned char)sym[i].text[j]]);
    }
    free(esc); free(sym); free(heap);
    return 0;
}

/* SPM top level: optional BOS, special-token partition, SPM merge on the gaps. */
static long spm_encode(const sp_tokenizer *t, const unsigned char *s, size_t text_len,
                       int parse_special, int32_t *out, int max_out) {
    long cnt = 0; int rc = 0;
    if (t->add_bos && t->bos_id >= 0) emit(out, max_out, &cnt, t->bos_id);
    size_t tstart = 0, i = 0;
    while (i < text_len) {
        int hit = 0;
        if (parse_special && t->n_spec) {
            for (uint32_t k = 0; k < t->n_spec; k++) {
                uint32_t sl = t->spec[k].len;
                if (sl && i + sl <= text_len && memcmp(s + i, t->spec[k].surf, sl) == 0) {
                    if (i > tstart && (rc = spm_fragment(t, s + tstart, i - tstart, out, max_out, &cnt))) goto done;
                    emit(out, max_out, &cnt, t->spec[k].id);
                    i += sl; tstart = i; hit = 1; break;
                }
            }
        }
        if (!hit) i++;
    }
    if (tstart < text_len) rc = spm_fragment(t, s + tstart, text_len - tstart, out, max_out, &cnt);
done:
    return rc ? -1 : cnt;
}

long sp_tokenizer_encode(const sp_tokenizer *t, const char *text, size_t text_len,
                         int parse_special, int32_t *out, int max_out) {
    if (!t || (!text && text_len > 0) || (!out && max_out > 0)) return -1;
    if (text_len == 0) return 0;
    const unsigned char *s = (const unsigned char *)text;
    if (t->family == SP_TOKFAM_GEMMA4_BPE)
        return sp_g4_encode(t, s, text_len, parse_special, out, max_out);
    if (t->spm) return spm_encode(t, s, text_len, parse_special, out, max_out);

    /* scratch sized to the whole input (codepoints <= bytes; enc <= 2*bytes) */
    uint32_t      *cps  = (uint32_t *)malloc(text_len * sizeof(uint32_t));
    unsigned char *enc  = (unsigned char *)malloc(2 * text_len + 8);
    uint32_t      *soff = (uint32_t *)malloc((text_len + 1) * sizeof(uint32_t));
    uint32_t      *slen = (uint32_t *)malloc((text_len + 1) * sizeof(uint32_t));
    unsigned char *key  = (unsigned char *)malloc(2 * text_len + 16);
    if (!cps || !enc || !soff || !slen || !key) {
        free(cps); free(enc); free(soff); free(slen); free(key); return -1;
    }

    long cnt = 0; int rc = 0;
    size_t tstart = 0, i = 0;
    while (i < text_len) {
        int hit = 0;
        if (parse_special && t->n_spec) {
            for (uint32_t k = 0; k < t->n_spec; k++) {       /* sorted longest-first */
                uint32_t sl = t->spec[k].len;
                if (sl && i + sl <= text_len && memcmp(s + i, t->spec[k].surf, sl) == 0) {
                    if (i > tstart) {
                        rc = encode_text(t, s + tstart, i - tstart, cps, enc, soff, slen, key, out, max_out, &cnt);
                        if (rc) goto done;
                    }
                    emit(out, max_out, &cnt, t->spec[k].id);
                    i += sl; tstart = i; hit = 1; break;
                }
            }
        }
        if (!hit) i++;
    }
    if (tstart < text_len)
        rc = encode_text(t, s + tstart, text_len - tstart, cps, enc, soff, slen, key, out, max_out, &cnt);
done:
    free(cps); free(enc); free(soff); free(slen); free(key);
    return rc ? -1 : cnt;
}

/* ===== .sp-tokenizer file loader — the blob lane (#115) ====================== *
 * Parses the SPTK header + SPTB blob written by sp_transcode build_tok_blob and
 * dispatches on type_id (the on-disk family tag):
 *   0 SENTENCEPIECE -> SPM lane;  2 BPE_GPT2 -> GPT2 lane (legacy values keep
 *   their old meaning);  4 GEMMA4_BPE -> gemma4 lane (#115).
 * Any other value = HARD ERROR naming it — never a silent fallback.
 * The blob does NOT serialize token_type, so no special surfaces are available:
 * parse_special is inert on this lane (n_spec==0), and add_bos comes from the
 * family (GEMMA4 forced true per llama PR #21500; legacy families 0 — the blob
 * carries no add_bos flag). SHA-256 pairing with a .sp-model is sp_model_load's
 * job; this loader checks magics + bounds. */

static int tf_u32(const uint8_t *blob, uint64_t bs, uint64_t *pos, uint32_t *v) {
    if (*pos + 4 > bs) return 1;
    memcpy(v, blob + *pos, 4); *pos += 4;
    return 0;
}

sp_tokenizer *sp_tokenizer_load_tokfile(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "sp_tokenizer: cannot open %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END); long fsz = ftell(f); fseek(f, 0, SEEK_SET);
    if (fsz < (long)(SP_TOK_HEADER_SIZE + 16)) {
        fprintf(stderr, "sp_tokenizer: %s too short for SPTK header\n", path);
        fclose(f); return NULL;
    }
    uint8_t *buf = (uint8_t *)malloc((size_t)fsz);
    if (!buf || fread(buf, 1, (size_t)fsz, f) != (size_t)fsz) {
        fprintf(stderr, "sp_tokenizer: read failed %s\n", path);
        free(buf); fclose(f); return NULL;
    }
    fclose(f);

    sp_tok_header hdr; memcpy(&hdr, buf, sizeof hdr);
    if (hdr.magic != SP_TOK_MAGIC || hdr.header_size != SP_TOK_HEADER_SIZE ||
        hdr.blob_offset + hdr.blob_size > (uint64_t)fsz || hdr.blob_size < 16) {
        fprintf(stderr, "sp_tokenizer: %s bad SPTK header\n", path);
        free(buf); return NULL;
    }
    const uint8_t *blob = buf + hdr.blob_offset;
    uint64_t bs = hdr.blob_size, pos = 0;
    uint32_t magic = 0, type_id = 0, nv = 0, nm = 0;
    if (tf_u32(blob, bs, &pos, &magic) || magic != 0x42545053u /*'SPTB'*/ ||
        tf_u32(blob, bs, &pos, &type_id) || tf_u32(blob, bs, &pos, &nv) ||
        tf_u32(blob, bs, &pos, &nm)) {
        fprintf(stderr, "sp_tokenizer: %s bad SPTB blob header\n", path);
        free(buf); return NULL;
    }
    int family;
    switch (type_id) {
        case SP_TOK_SENTENCEPIECE: family = SP_TOKFAM_SPM;        break;
        case SP_TOK_BPE_GPT2:      family = SP_TOKFAM_GPT2_BPE;   break;
        case SP_TOK_GEMMA4_BPE:    family = SP_TOKFAM_GEMMA4_BPE; break;
        default:
            fprintf(stderr, "sp_tokenizer: %s unknown tokenizer family type_id=%u "
                            "— refusing silent fallback\n", path, type_id);
            free(buf); return NULL;
    }
    if (nv == 0 || nv != hdr.vocab_size) {
        fprintf(stderr, "sp_tokenizer: %s blob vocab %u != header vocab %u\n",
                path, nv, hdr.vocab_size);
        free(buf); return NULL;
    }

    sp_tokenizer *t = (sp_tokenizer *)calloc(1, sizeof *t);
    if (!t) { free(buf); return NULL; }
    t->file_blob = buf;
    t->family = family;
    t->spm = (family == SP_TOKFAM_SPM);
    t->n_vocab = nv;
    t->tok = (const char **)malloc((size_t)nv * sizeof(char *));
    t->len = (uint64_t *)malloc((size_t)nv * sizeof(uint64_t));
    if (!t->tok || !t->len) { sp_tokenizer_free(t); return NULL; }
    for (uint32_t i = 0; i < nv; i++) {
        uint32_t L = 0;
        if (tf_u32(blob, bs, &pos, &L) || pos + L > bs) {
            fprintf(stderr, "sp_tokenizer: %s truncated token[%u]\n", path, i);
            sp_tokenizer_free(t); return NULL;
        }
        t->tok[i] = (const char *)(blob + pos);
        t->len[i] = L;
        pos += L;
    }
    build_byte_maps(t);
    if (!hmap_init(&t->vocab, (size_t)nv)) { sp_tokenizer_free(t); return NULL; }
    for (uint32_t i = 0; i < nv; i++)
        hmap_put(&t->vocab, t->tok[i], (uint32_t)t->len[i], (int64_t)i);

    /* SPM: f32 scores follow the tokens (copy out — blob floats are unaligned) */
    if (t->spm) {
        if (pos + (uint64_t)nv * 4 > bs) {
            fprintf(stderr, "sp_tokenizer: %s truncated scores\n", path);
            sp_tokenizer_free(t); return NULL;
        }
        t->scores_blob = (float *)malloc((size_t)nv * sizeof(float));
        if (!t->scores_blob) { sp_tokenizer_free(t); return NULL; }
        memcpy(t->scores_blob, blob + pos, (size_t)nv * sizeof(float));
        t->scores = t->scores_blob;
        pos += (uint64_t)nv * 4;
        for (int b = 0; b < 256; b++) {
            char nmb[8]; int nl = snprintf(nmb, sizeof nmb, "<0x%02X>", b);
            t->byte_tok[b] = (int32_t)hmap_get(&t->vocab, nmb, (uint32_t)nl, -1);
            if (t->byte_tok[b] < 0) {
                fprintf(stderr, "sp_tokenizer: %s SPM byte token <0x%02X> missing\n", path, b);
                sp_tokenizer_free(t); return NULL;
            }
        }
    }

    /* merges */
    if (nm > 0) {
        const char **mp = (const char **)malloc((size_t)nm * sizeof(char *));
        uint64_t    *ml = (uint64_t *)malloc((size_t)nm * sizeof(uint64_t));
        if (!mp || !ml) { free((void *)mp); free(ml); sp_tokenizer_free(t); return NULL; }
        int ok = 1;
        for (uint32_t i = 0; i < nm; i++) {
            uint32_t L = 0;
            if (tf_u32(blob, bs, &pos, &L) || pos + L > bs) {
                fprintf(stderr, "sp_tokenizer: %s truncated merge[%u]\n", path, i);
                ok = 0; break;
            }
            mp[i] = (const char *)(blob + pos);
            ml[i] = L;
            pos += L;
        }
        if (ok) {
            if (family == SP_TOKFAM_GEMMA4_BPE) {
                ok = (sp_g4_build(t, mp, ml, nm) == 0);
            } else if (family == SP_TOKFAM_GPT2_BPE) {
                ok = hmap_init(&t->merge, (size_t)nm);
                if (ok) for (uint32_t i = 0; i < nm; i++)
                    hmap_put(&t->merge, mp[i], (uint32_t)ml[i], (int64_t)i);
            }   /* SPM: merges unused */
        }
        free((void *)mp); free(ml);
        if (!ok) { sp_tokenizer_free(t); return NULL; }
    } else if (family == SP_TOKFAM_GEMMA4_BPE) {
        fprintf(stderr, "sp_tokenizer: %s gemma4 blob has no merges\n", path);
        sp_tokenizer_free(t); return NULL;
    }

    /* specials from the header (0xFFFFFFFF = absent) */
    t->bos_id = (hdr.bos_token == 0xFFFFFFFFu) ? -1 : (int32_t)hdr.bos_token;
    t->eos_id = (hdr.eos_token == 0xFFFFFFFFu) ? -1 : (int32_t)hdr.eos_token;
    t->pad_id = (hdr.pad_token == 0xFFFFFFFFu) ? -1 : (int32_t)hdr.pad_token;
    t->unk_id = (hdr.unk_token == 0xFFFFFFFFu) ? -1 : (int32_t)hdr.unk_token;
    if (family == SP_TOKFAM_GEMMA4_BPE) {
        t->add_bos = 1;   /* FORCED (llama-vocab.cpp:2338-2344, PR #21500) */
        if (t->bos_id < 0 || (uint32_t)t->bos_id >= nv) {
            fprintf(stderr, "sp_tokenizer: %s gemma4 bos_token missing/out of range\n", path);
            sp_tokenizer_free(t); return NULL;
        }
    }
    return t;
}
