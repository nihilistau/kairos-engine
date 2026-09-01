/* tokenizer_internal.h — engine-internal sharing between tokenizer.c (loaders,
 * dispatch, GPT2/SPM lanes) and gemma4_bpe.c (the GEMMA4_BPE lane, issue #115).
 * NOT a public header. The public ABI stays sp_engine/tokenizer.h.
 *
 * Family dispatch (SPEC-gemma4-tokenizer-dispatch.md): the .sp-tokenizer
 * type_id / GGUF tokenizer.ggml.model(+.pre) select exactly one lane; an
 * unknown family is a HARD ERROR naming the family — never a silent GPT2
 * fallback (a wrong dispatch would be silent mis-tokenization: every id
 * valid, every id wrong). */
#ifndef SP_ENGINE_TOKENIZER_INTERNAL_H
#define SP_ENGINE_TOKENIZER_INTERNAL_H

#include "sp_engine/tokenizer.h"

/* ---- string hash map (open addressing, linear probe): key bytes -> int64 ---
 * Definitions live in tokenizer.c (sp_tok_hmap_*). */
typedef struct { const char *key; uint32_t klen; int64_t val; uint8_t used; } sp_hent;
typedef struct { sp_hent *e; size_t mask; } sp_hmap;

/* special-token surface (for parse_special pre-split) */
typedef struct { const char *surf; uint32_t len; int32_t id; } sp_special;

/* tokenizer family — dispatch tag (NOT the on-disk type_id; the mapping from
 * sp_tok_type_id / tokenizer.ggml.model happens in the loaders). */
typedef enum {
    SP_TOKFAM_GPT2_BPE   = 0,  /* GPT2 byte-level BPE ("gpt2": Qwen family)     */
    SP_TOKFAM_SPM        = 1,  /* SentencePiece bigram ("llama": Gemma3 family) */
    SP_TOKFAM_GEMMA4_BPE = 2   /* gemma4 514k-merge U+2581-piece BPE (#115)     */
} sp_tok_family;

struct sp_tokenizer {
    uint32_t      n_vocab;
    const char  **tok;          /* n_vocab pointers into the GGUF mapping / blob      */
    uint64_t     *len;          /* n_vocab token byte-lengths                         */
    int           cp_to_byte[512]; /* GPT2 byte-level inverse: codepoint -> byte      */
    int           max_cp;
    /* encode side */
    uint8_t       bcp[256][2];  /* byte -> UTF-8 of its byte-level codepoint           */
    uint8_t       bcp_len[256]; /* 1 or 2                                              */
    sp_hmap       vocab;        /* token-string bytes -> id                            */
    sp_hmap       merge;        /* "A B" merge-line bytes -> rank (GPT2 lane)          */
    sp_special   *spec;         /* special surfaces, sorted longest-first              */
    uint32_t      n_spec;
    char         *tok_blob;     /* owned vocab bytes (owning mode); NULL if borrowing  */
    char         *merge_blob;   /* owned merge bytes (owning mode); NULL if borrowing  */
    /* SPM (SentencePiece "llama") path — Gemma3 etc. */
    int           spm;          /* 1 if family == SP_TOKFAM_SPM                        */
    const float  *scores;       /* n_vocab unigram scores (mapping ptr, or owned blob) */
    float        *scores_blob;  /* owned scores copy (owning mode); NULL if borrowing  */
    int32_t       byte_tok[256];/* byte-fallback: byte -> "<0xXX>" token id (-1 NA);
                                   SPM + GEMMA4 lanes                                  */
    int           add_bos;      /* tokenizer.ggml.add_bos_token (auto-prepend BOS)     */
    int32_t       bos_id;       /* tokenizer.ggml.bos_token_id                         */

    /* ---- #115 additions (zero impact on the GPT2/SPM lanes) ---- */
    int           family;       /* sp_tok_family                                       */
    int32_t       eos_id, pad_id, unk_id;  /* decode-skip set for the gemma4 lane      */
    uint8_t      *file_blob;    /* owned .sp-tokenizer file bytes (blob lane); NULL    */
    /* gemma4 merge tables (built once at load; never linear-scanned):
     *   pair hash  u64 key (left_id<<32)|right_id -> rank
     *   result map rank -> merged token id                                            */
    uint64_t     *g4_pairkey;   /* open addressing; SP_G4_EMPTY_KEY = empty slot       */
    int32_t      *g4_pairrank;
    size_t        g4_pairmask;
    int32_t      *g4_result;
    uint32_t      g4_n_merges;
};

#define SP_G4_EMPTY_KEY 0xFFFFFFFFFFFFFFFFull

/* tokenizer.c: vocab lookup (token bytes -> id, -1 absent). */
int64_t sp_tok_vocab_lookup(const sp_tokenizer *t, const char *bytes, uint32_t len);

/* gemma4_bpe.c — the GEMMA4_BPE lane.
 * sp_g4_build: build pair-rank + result tables from the raw merge lines
 * ("LEFT RIGHT", split at the first space at index >= 1, mirroring
 * llama-vocab.cpp:1910). HARD-ERRORS (returns nonzero, message on stderr) if a
 * non-inert merge side or result does not resolve in the vocab — an id-keyed
 * table cannot represent such merges, so refusing is the honest move (HF-valid
 * gemma4 data has none). Also builds byte_tok[] (<0xNN>, all 256 required). */
int  sp_g4_build(sp_tokenizer *t, const char **merges, const uint64_t *mlens,
                 uint64_t n_merges);
void sp_g4_free(sp_tokenizer *t);
long sp_g4_encode(const sp_tokenizer *t, const unsigned char *s, size_t text_len,
                  int parse_special, int32_t *out, int max_out);
long sp_g4_decode(const sp_tokenizer *t, const int32_t *ids, int n,
                  char *buf, size_t cap);

#endif /* SP_ENGINE_TOKENIZER_INTERNAL_H */
