/* ppl.h — perplexity over a text corpus, replicating llama.cpp/tools/perplexity
 * (the default, non-stride path). Tokenizes the whole corpus once (BOS added),
 * splits into non-overlapping n_ctx chunks, re-anchors each chunk with BOS at
 * position 0, and scores positions [n_ctx/2, n_ctx-1) (each scored token has
 * >= n_ctx/2 tokens of context). PPL = exp(mean NLL). Used to close T_FRO_4
 * (Gemma3-1B Frobenius-Q8 PPL within 0.1% of the engine's own f32 PPL). */
#ifndef SP_ENGINE_PPL_H
#define SP_ENGINE_PPL_H

#include "sp_engine/model.h"
#include "sp_engine/tokenizer.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Compute perplexity of `text` under model `m` (arch-dispatched: gemma3/qwen3
 * forward) using tokenizer `tok`. n_ctx is the context-window chunk size (e.g.
 * 512). Writes *ppl (and *ppl_stderr, *n_scored if non-NULL). Honors the matmul
 * env knobs (SP_ENGINE_FROB / SP_ARENA / SP_CPU_SCALAR). Returns 0 on success,
 * nonzero on error (too few tokens, alloc failure, forward failure). */
int sp_perplexity(const qwen3_model *m, const sp_tokenizer *tok,
                  const char *text, size_t text_len, int n_ctx,
                  double *ppl, double *ppl_stderr, long *n_scored);

#ifdef __cplusplus
}
#endif
#endif /* SP_ENGINE_PPL_H */
