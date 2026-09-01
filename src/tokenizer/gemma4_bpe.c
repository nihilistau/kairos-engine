/* gemma4_bpe.c — the GEMMA4_BPE tokenizer lane (issue #115).
 *
 * Reference (Stage-0, read verbatim before coding — see the #115 plan commit):
 * llama.cpp b8861 src/llama-vocab.cpp
 *   :496-505  PRE_TYPE_GEMMA4: regex {"[^\n]+|[\n]+"}, byte_encode=false
 *   :3140-3151 normalizer: per raw fragment, escape_whitespaces ->
 *              ' ' -> U+2581 (E2 96 81) BEFORE tokenize (:3038-3040)
 *   :576-587  ignore_merges NOT set for gemma4; all-'\n' word in vocab ->
 *              atomic symbol (PR #21343)
 *   :589-599  initial symbols = UTF-8 chars, len = min(remaining,
 *              lookup{1,..,1,2,2,3,4}[lead>>4])
 *   :263-277  bigram heap: lowest rank first, tie -> lowest LEFT index
 *   :605-632  pop loop; outdated bigram skipped when either side changed
 *              (string equality :616 == both side ids unchanged, id-keyed)
 *   :660-676  byte fallback: "<0xNN>" UPPERCASE hex via text_to_token
 *   :2328-2344 add_bos FORCED true for gemma4 (PR #21500)
 *
 * The 514k merges are looked up through a hash table built ONCE at load:
 * u64 key (left_id<<32)|right_id -> rank, plus rank -> result-id. Id-keying is
 * exactly equivalent to llama's string-keying iff every merge side/result is a
 * vocab token (true for HF-valid gemma4 data); sp_g4_build HARD-ERRORS
 * otherwise instead of risking silent divergence.
 *
 * Decode: <0xNN> byte tokens reassembled, U+2581 -> ' ', BOS/EOS/PAD/UNK
 * skipped. NOTE the inherent SPM-style aliasing (shared with llama.cpp): a
 * literal U+2581 in the input decodes back as ' '. */
#include "tokenizer_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- pair-rank hash (u64 open addressing, linear probe) -------------------- */
static size_t g4_slot(uint64_t k, size_t mask) {
    k ^= k >> 33; k *= 0xFF51AFD7ED558CCDull;     /* splitmix-style finalizer */
    k ^= k >> 33; k *= 0xC4CEB9FE1A85EC53ull;
    k ^= k >> 33;
    return (size_t)k & mask;
}
static void g4_pair_put(sp_tokenizer *t, uint64_t key, int32_t rank) {
    size_t i = g4_slot(key, t->g4_pairmask);
    while (t->g4_pairkey[i] != SP_G4_EMPTY_KEY) {
        if (t->g4_pairkey[i] == key) return;       /* first writer (lowest rank) wins */
        i = (i + 1) & t->g4_pairmask;
    }
    t->g4_pairkey[i] = key; t->g4_pairrank[i] = rank;
}
static int32_t g4_pair_rank(const sp_tokenizer *t, int32_t l, int32_t r) {
    uint64_t key = ((uint64_t)(uint32_t)l << 32) | (uint32_t)r;
    size_t i = g4_slot(key, t->g4_pairmask);
    while (t->g4_pairkey[i] != SP_G4_EMPTY_KEY) {
        if (t->g4_pairkey[i] == key) return t->g4_pairrank[i];
        i = (i + 1) & t->g4_pairmask;
    }
    return -1;
}

int sp_g4_build(sp_tokenizer *t, const char **mp, const uint64_t *ml, uint64_t nm) {
    if (nm == 0 || nm > 0x7FFFFFFFull) {
        fprintf(stderr, "sp_tokenizer(gemma4): merge count %llu invalid\n",
                (unsigned long long)nm);
        return 1;
    }
    size_t cap = 16; while (cap < (size_t)nm * 2) cap <<= 1;
    t->g4_pairkey  = (uint64_t *)malloc(cap * sizeof(uint64_t));
    t->g4_pairrank = (int32_t *)malloc(cap * sizeof(int32_t));
    t->g4_result   = (int32_t *)malloc((size_t)nm * sizeof(int32_t));
    char *cat = (char *)malloc(512);   /* scratch for left+right concat */
    size_t cat_cap = 512;
    if (!t->g4_pairkey || !t->g4_pairrank || !t->g4_result || !cat) {
        free(cat); sp_g4_free(t);
        fprintf(stderr, "sp_tokenizer(gemma4): OOM building merge tables\n");
        return 1;
    }
    for (size_t i = 0; i < cap; i++) t->g4_pairkey[i] = SP_G4_EMPTY_KEY;
    t->g4_pairmask = cap - 1;
    t->g4_n_merges = (uint32_t)nm;

    uint64_t inert = 0;
    for (uint64_t i = 0; i < nm; i++) {
        const char *line = mp[i];
        uint32_t L = (uint32_t)ml[i];
        /* split at the first ' ' at index >= 1 (llama-vocab.cpp:1910) */
        uint32_t sp = 0;
        for (uint32_t k = 1; k < L; k++) if (line[k] == ' ') { sp = k; break; }
        if (sp == 0 || sp + 1 >= L) {              /* no separator: inert (llama
                                                      emplaces ("","") — never fires) */
            t->g4_result[i] = -1; inert++; continue;
        }
        int64_t lid = sp_tok_vocab_lookup(t, line, sp);
        int64_t rid = sp_tok_vocab_lookup(t, line + sp + 1, L - sp - 1);
        if (lid < 0 || rid < 0) {
            fprintf(stderr, "sp_tokenizer(gemma4): merge %llu side not in vocab "
                            "(id-keyed rank table cannot represent it) — HARD ERROR\n",
                    (unsigned long long)i);
            sp_g4_free(t); free(cat); return 1;
        }
        uint32_t cl = (sp) + (L - sp - 1);
        if (cl + 1 > cat_cap) {
            cat_cap = cl + 64; char *nc = (char *)realloc(cat, cat_cap);
            if (!nc) { sp_g4_free(t); free(cat); return 1; }
            cat = nc;
        }
        memcpy(cat, line, sp); memcpy(cat + sp, line + sp + 1, L - sp - 1);
        int64_t res = sp_tok_vocab_lookup(t, cat, cl);
        if (res < 0) {
            fprintf(stderr, "sp_tokenizer(gemma4): merge %llu result not in vocab "
                            "— HARD ERROR\n", (unsigned long long)i);
            sp_g4_free(t); free(cat); return 1;
        }
        t->g4_result[i] = (int32_t)res;
        g4_pair_put(t, ((uint64_t)(uint32_t)lid << 32) | (uint32_t)rid, (int32_t)i);
    }
    free(cat);
    if (inert)
        fprintf(stderr, "sp_tokenizer(gemma4): %llu merge line(s) without separator "
                        "(inert, llama-parity)\n", (unsigned long long)inert);

    /* byte-fallback table: byte b -> id of "<0xNN>" (all 256 must resolve) */
    for (int b = 0; b < 256; b++) {
        char nm2[8]; int nl = snprintf(nm2, sizeof nm2, "<0x%02X>", b);
        int64_t id = sp_tok_vocab_lookup(t, nm2, (uint32_t)nl);
        if (id < 0) {
            fprintf(stderr, "sp_tokenizer(gemma4): byte token <0x%02X> missing\n", b);
            sp_g4_free(t); return 1;
        }
        t->byte_tok[b] = (int32_t)id;
    }
    return 0;
}

void sp_g4_free(sp_tokenizer *t) {
    free(t->g4_pairkey);  t->g4_pairkey  = NULL;
    free(t->g4_pairrank); t->g4_pairrank = NULL;
    free(t->g4_result);   t->g4_result   = NULL;
    t->g4_pairmask = 0; t->g4_n_merges = 0;
}

/* ---- encode ----------------------------------------------------------------- */

/* UTF-8 char length from the lead byte — llama's unicode_len_utf8 lookup
 * {1,1,1,1,1,1,1,1,1,1,1,1,2,2,3,4}[lead>>4] (NOT a validity check; invalid
 * leads 0xF8-0xFF advance 4, continuation bytes advance 1 — mirror exactly). */
static uint32_t g4_utf8_len(unsigned char lead) {
    static const uint8_t lut[16] = {1,1,1,1,1,1,1,1,1,1,1,1,2,2,3,4};
    return lut[lead >> 4];
}

typedef struct { int prev, next; uint32_t off, n; int32_t id; } g4_sym;
typedef struct { int32_t rank; int left, right; int32_t lid, rid; } g4_bigram;

/* min-heap: lowest rank first; tie -> lowest LEFT index (llama-vocab.cpp:263-277:
 * priority_queue with less = l.rank > r.rank || (== && l.left > r.left), so the
 * popped top is min-rank / min-left). */
static int g4_before(const g4_bigram *a, const g4_bigram *b) {
    return (a->rank < b->rank) || (a->rank == b->rank && a->left < b->left);
}
typedef struct { g4_bigram *e; int n, cap; } g4_heap;
static int g4_heap_push(g4_heap *h, g4_bigram bg) {
    if (h->n == h->cap) {
        int nc = h->cap ? h->cap * 2 : 64;
        g4_bigram *ne = (g4_bigram *)realloc(h->e, (size_t)nc * sizeof *ne);
        if (!ne) return 1;
        h->e = ne; h->cap = nc;
    }
    int i = h->n++;
    h->e[i] = bg;
    while (i > 0) {
        int p = (i - 1) >> 1;
        if (!g4_before(&h->e[i], &h->e[p])) break;
        g4_bigram tmp = h->e[i]; h->e[i] = h->e[p]; h->e[p] = tmp; i = p;
    }
    return 0;
}
static g4_bigram g4_heap_pop(g4_heap *h) {
    g4_bigram top = h->e[0];
    h->e[0] = h->e[--h->n];
    int i = 0;
    for (;;) {
        int l = 2 * i + 1, r = 2 * i + 2, m = i;
        if (l < h->n && g4_before(&h->e[l], &h->e[m])) m = l;
        if (r < h->n && g4_before(&h->e[r], &h->e[m])) m = r;
        if (m == i) break;
        g4_bigram tmp = h->e[i]; h->e[i] = h->e[m]; h->e[m] = tmp; i = m;
    }
    return top;
}

static void g4_emit(int32_t *out, int max_out, long *cnt, int32_t id) {
    if (*cnt < max_out) out[*cnt] = id;
    (*cnt)++;
}

/* push (left,right) as a candidate bigram if both ids resolve to a ranked pair */
static int g4_try_pair(const sp_tokenizer *t, const g4_sym *sym, int l, int r,
                       g4_heap *h) {
    if (l < 0 || r < 0) return 0;
    if (sym[l].id < 0 || sym[r].id < 0) return 0;   /* non-vocab symbol: no merge
                                                       can name it (build enforced) */
    int32_t rank = g4_pair_rank(t, sym[l].id, sym[r].id);
    if (rank < 0) return 0;
    g4_bigram bg; bg.rank = rank; bg.left = l; bg.right = r;
    bg.lid = sym[l].id; bg.rid = sym[r].id;
    return g4_heap_push(h, bg);
}

/* BPE over one pre-token (piece) [s,s+n): esc'd bytes, no '\n' mixing. */
static int g4_piece(const sp_tokenizer *t, const unsigned char *s, size_t n,
                    int is_nl_run, int32_t *out, int max_out, long *cnt) {
    /* gemma4 fix (llama-vocab.cpp:580-587, PR #21343): an all-'\n' piece that is
     * itself a vocab token is atomic. */
    if (is_nl_run) {
        int64_t id = sp_tok_vocab_lookup(t, (const char *)s, (uint32_t)n);
        if (id >= 0) { g4_emit(out, max_out, cnt, (int32_t)id); return 0; }
    }
    /* initial symbols = UTF-8 chars (llama-vocab.cpp:589-599) */
    g4_sym *sym = (g4_sym *)malloc((n + 1) * sizeof *sym);
    if (!sym) return -1;
    int ns = 0;
    for (size_t o = 0; o < n; ) {
        uint32_t L = g4_utf8_len(s[o]);
        if (o + L > n) L = (uint32_t)(n - o);
        sym[ns].off = (uint32_t)o; sym[ns].n = L;
        sym[ns].id = (int32_t)sp_tok_vocab_lookup(t, (const char *)s + o, L);
        sym[ns].prev = ns - 1;
        sym[ns].next = (o + L >= n) ? -1 : ns + 1;
        o += L; ns++;
    }
    g4_heap heap = {0};
    int rc = 0;
    for (int i = 1; i < ns && !rc; i++) rc = g4_try_pair(t, sym, i - 1, i, &heap);
    while (!rc && heap.n > 0) {
        g4_bigram bg = g4_heap_pop(&heap);
        g4_sym *L = &sym[bg.left], *R = &sym[bg.right];
        /* outdated-bigram skip (llama-vocab.cpp:611-618): consumed or changed.
         * Id equality == string equality (ids are vocab-unique; a merge always
         * changes the symbol's id). */
        if (L->n == 0 || R->n == 0 || L->id != bg.lid || R->id != bg.rid) continue;
        L->n += R->n;
        L->id = t->g4_result[bg.rank];
        R->n = 0; R->id = -1;
        L->next = R->next;
        if (R->next >= 0) sym[R->next].prev = bg.left;
        rc = g4_try_pair(t, sym, L->prev, bg.left, &heap);
        if (!rc) rc = g4_try_pair(t, sym, bg.left, L->next, &heap);
    }
    /* emit: vocab symbols by id; the rest byte-fallback <0xNN>
     * (llama-vocab.cpp:650-681) */
    if (!rc && ns > 0) {
        for (int i = 0; i != -1; i = sym[i].next) {
            if (sym[i].n == 0) continue;
            if (sym[i].id >= 0) g4_emit(out, max_out, cnt, sym[i].id);
            else for (uint32_t b = 0; b < sym[i].n; b++)
                g4_emit(out, max_out, cnt, t->byte_tok[s[sym[i].off + b]]);
        }
    }
    free(heap.e); free(sym);
    return rc ? -1 : 0;
}

/* one raw-text fragment (between specials): escape spaces, split on newline
 * runs ("[^\n]+|[\n]+", llama-vocab.cpp:501-503), BPE each piece. */
static int g4_fragment(const sp_tokenizer *t, const unsigned char *s, size_t len,
                       int32_t *out, int max_out, long *cnt) {
    if (len == 0) return 0;
    /* normalizer (llama-vocab.cpp:3144-3146 + :3038-3040): ' ' -> U+2581 */
    unsigned char *esc = (unsigned char *)malloc(3 * len + 1);
    if (!esc) return -1;
    size_t el = 0;
    for (size_t i = 0; i < len; i++) {
        if (s[i] == 0x20) { esc[el++] = 0xE2; esc[el++] = 0x96; esc[el++] = 0x81; }
        else esc[el++] = s[i];
    }
    int rc = 0;
    for (size_t o = 0; o < el && !rc; ) {
        int isnl = (esc[o] == '\n');
        size_t e = o + 1;
        while (e < el && (esc[e] == '\n') == isnl) e++;
        rc = g4_piece(t, esc + o, e - o, isnl, out, max_out, cnt);
        o = e;
    }
    free(esc);
    return rc;
}

long sp_g4_encode(const sp_tokenizer *t, const unsigned char *s, size_t text_len,
                  int parse_special, int32_t *out, int max_out) {
    long cnt = 0; int rc = 0;
    /* add_bos is FORCED true for gemma4 (llama-vocab.cpp:2338-2344, PR #21500);
     * the loader guarantees bos_id >= 0. */
    if (t->add_bos && t->bos_id >= 0) g4_emit(out, max_out, &cnt, t->bos_id);
    size_t tstart = 0, i = 0;
    while (i < text_len) {
        int hit = 0;
        if (parse_special && t->n_spec) {
            for (uint32_t k = 0; k < t->n_spec; k++) {     /* sorted longest-first */
                uint32_t sl = t->spec[k].len;
                if (sl && i + sl <= text_len && memcmp(s + i, t->spec[k].surf, sl) == 0) {
                    if (i > tstart &&
                        (rc = g4_fragment(t, s + tstart, i - tstart, out, max_out, &cnt)))
                        goto done;
                    g4_emit(out, max_out, &cnt, t->spec[k].id);
                    i += sl; tstart = i; hit = 1; break;
                }
            }
        }
        if (!hit) i++;
    }
    if (tstart < text_len)
        rc = g4_fragment(t, s + tstart, text_len - tstart, out, max_out, &cnt);
done:
    return rc ? -1 : cnt;
}

/* ---- decode ----------------------------------------------------------------- */

static void g4_put(char *buf, size_t cap, size_t *pos, unsigned char b) {
    if (*pos + 1 < cap) buf[*pos] = (char)b;
    (*pos)++;
}
static int g4_hex(unsigned char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    return -1;
}

long sp_g4_decode(const sp_tokenizer *t, const int32_t *ids, int n,
                  char *buf, size_t cap) {
    size_t pos = 0;
    for (int k = 0; k < n; k++) {
        int32_t id = ids[k];
        if (id < 0 || (uint32_t)id >= t->n_vocab) continue;   /* skip OOR */
        if (id == t->bos_id || id == t->eos_id ||
            id == t->pad_id || id == t->unk_id) continue;     /* skip specials */
        const unsigned char *s = (const unsigned char *)t->tok[id];
        uint64_t L = t->len[id];
        /* byte token "<0xNN>" -> raw byte */
        if (L == 6 && s[0] == '<' && s[1] == '0' && s[2] == 'x' && s[5] == '>') {
            int hi = g4_hex(s[3]), lo = g4_hex(s[4]);
            if (hi >= 0 && lo >= 0) { g4_put(buf, cap, &pos, (unsigned char)((hi << 4) | lo)); continue; }
        }
        for (uint64_t i = 0; i < L; ) {
            if (i + 2 < L && s[i] == 0xE2 && s[i+1] == 0x96 && s[i+2] == 0x81) {
                g4_put(buf, cap, &pos, 0x20); i += 3;          /* U+2581 -> space */
            } else {
                g4_put(buf, cap, &pos, s[i]); i += 1;
            }
        }
    }
    if (cap > 0) buf[pos < cap ? pos : cap - 1] = '\0';
    return (long)pos;
}
