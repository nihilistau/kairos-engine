/* sp_daemon_cuda_glue.c -- C glue between sp_l1.h's sp_forward_dispatch_fn ABI
 * and the engine's CUDA PTX whole-forward entry points (cuda_forward.cu:497,724).
 *
 * Sprint WIRE-CUDA: symmetric to sp_daemon_hex_glue.c. The daemon's host
 * binary, when SP_DAEMON_BACKEND=cuda is set, calls
 *   sp_session_register_forward_backend(s, NULL, sp_daemon_cuda_forward).
 * On each subsequent sp_prefill_chunk, the math-core session routes through
 * this glue (sp_l1.h:#6) which forwards to the engine's CUDA backend.
 *
 * Arch routing: the CUDA backend exports BOTH gemma3_forward_cuda (CU.1-4)
 * and qwen3_forward_cuda (CU.5) plus qwen3_forward_cuda_ex (CU.5+E_CU_6
 * KV-tree variant, not used by daemon). The glue inspects m->cfg.arch and
 * routes accordingly. This is wider than the hex glue (gemma3-only) because
 * the CUDA backend already supports both arches in ctest (M_QWEN3_CUDA
 * passes; M_GEMMA3_CUDA has a known OOM bug on this host -- see ctest log).
 *
 * Type-safety: the L1 ABI passes the qwen3_model pointer as `const void *`
 * to keep sp_l1.h independent of sp/model.h (which lives in the engine
 * include tree). Cast back here. Same arch enum (SP_ARCH_GEMMA3 /
 * SP_ARCH_QWEN3) on both math-core and engine sides.
 *
 * Error handling: gemma3_forward_cuda / qwen3_forward_cuda return 0 on
 * success, non-zero on error. sp_prefill_chunk maps non-zero to
 * SP_EBADSTATE + preserves sp_last_error via the sp_set_error calls already
 * inside the CUDA backend.
 *
 * No kernel-name shim: the CUDA backend's cuda_forward.cu calls math-core's
 * sp_matmul / sp_embed_row / sp_as_f32 directly via forward_dispatch.h --
 * the same symbols the daemon already links from libsp_forward_dispatch.a.
 * cpu_overlay.c is irrelevant on CUDA.
 *
 * Lifecycle hook: sp_cuda_model_release frees device-resident weights cached
 * by the engine's static (g_w in cuda_forward.cu:729). Called at daemon
 * shutdown via release_for_model (cuda_forward_dispatch.rs).
 */
#include "sp_engine/cuda_backend.h"   /* gemma3_forward_cuda + qwen3_forward_cuda + sp_cuda_model_release */
#include "sp/forward_dispatch.h"      /* sp_matmul / sp_embed_row / sp_as_f32 / sp_kernels_read_env */
#include "sp/model.h"                 /* qwen3_model + sp_arch_t + SP_ARCH_GEMMA3 / SP_ARCH_QWEN3 + gguf_tensor */
#include <stdint.h>
#include <stdlib.h>              /* malloc/free + _putenv_s */

extern void sp_set_error(const char *msg);

/* The L1 forward-dispatch ABI from sp/sp_l1.h #6. Forward-declared here so the
 * glue stays free of the math-core include path (the daemon Rust side bindgens
 * sp_l1.h directly; this C TU just needs the signature to match). */
typedef int (*sp_forward_dispatch_fn_t)(
    void *handle, const void *qm_opaque,
    const int32_t *tokens, int n_tok, float *logits);

/* Forward-dispatch entry point exposed to Rust via #[link]. The L1 contract
 * (sp_l1.h:#6) guarantees:
 *   - handle is the opaque pointer the daemon passed at register time. Ignored
 *     here -- the CUDA backend caches its device-resident weight blob via
 *     a static (g_w in cuda_forward.cu) keyed on the model pointer.
 *   - qm_opaque is the session-owned qwen3_model* (math-core's sp_session.c
 *     reconstructs it via sp_model_to_gemma3 / sp_model_to_qwen3; pointer
 *     remains valid for the session's lifetime).
 *   - tokens/n_tok/logits per the standard engine forward signature
 *     (src/forward/ppl.c:19).
 * Returns 0 on success, non-zero on error.
 *
 * Arch routing:
 *   SP_ARCH_GEMMA3 -> gemma3_forward_cuda  (CU.1-4)
 *   SP_ARCH_QWEN3  -> qwen3_forward_cuda   (CU.5)
 *   else           -> sp_set_error + return -1
 *
 * SP_ARCH_QWEN25 is not yet covered by the CUDA backend (qwen25_forward_cuda
 * does not exist in cuda_forward.cu); routing to qwen3_forward_cuda would
 * mismatch the arch check inside the kernel and fail; surface as error here. */
int sp_daemon_cuda_forward(void *handle,
                           const void *qm_opaque,
                           const int32_t *tokens,
                           int n_tok,
                           float *logits) {
    (void)handle;  /* opaque; the engine owns its own statics */
    const qwen3_model *qm = (const qwen3_model *)qm_opaque;
    if (!qm) { sp_set_error("cuda: NULL qwen3_model"); return -1; }
    switch (qm->cfg.arch) {
        case SP_ARCH_GEMMA3:
            return gemma3_forward_cuda(qm, tokens, n_tok, logits);
        case SP_ARCH_GEMMA4_MOE:   /* ADR-013: same backbone forward, MoE FFN inside */
        case SP_ARCH_GEMMA4:
            /* byte-exact bridge step 1 (CONTRACT-BYTEEXACT §8): route the real
             * Gemma-4-dense model forward through the universal crate's register_forward_backend
             * hook. Same prefill-only contract + signature as gemma3_forward_cuda;
             * entry exported in cuda_backend.h:77. Needs the wire_cuda build to gate. */
            return gemma4_forward_cuda(qm, tokens, n_tok, logits);
        case SP_ARCH_QWEN3:
            return qwen3_forward_cuda(qm, tokens, n_tok, logits);
        default:
            sp_set_error("cuda: unsupported arch (only SP_ARCH_GEMMA3 / SP_ARCH_GEMMA4 / SP_ARCH_QWEN3)");
            return -1;
    }
}

/* Lifecycle hook: when the daemon tears down its cuda backend (or the model
 * unloads), release the cached device-resident weight blob. Mirrors
 * sp_hexagon_model_release. */
void sp_daemon_cuda_release(const void *qm_opaque) {
    const qwen3_model *qm = (const qwen3_model *)qm_opaque;
    if (qm) sp_cuda_model_release(qm);
}

/* ── Sprint WIRE-CUDA-DECODE-GEMMA4: persistent-KV decode glue ──────────────
 *
 * The prefill dispatcher above (sp_daemon_cuda_forward) is PREFILL-ONLY: it
 * re-runs the full forward over the accumulated history and, for a dense reference model OK_Q4B
 * model, the tied full-vocab head is reserved for the DECODE path, so driving
 * decode through it trips `g4 probe: FULL head needs the f32 embd`
 * (cuda_forward.cu:1627). These six glue fns adapt the new
 * sp_kvdecode_dispatch_fn table (WIRE-CUDA-DECODE-GEMMA4.md §2) onto the
 * `gemma4_kv_*` C ABI already compiled into THIS lib from cuda_forward.cu
 * (declared inline in tests/test_gemma4_cuda.c:65-79; defined extern "C" at
 * cuda_forward.cu ~3478-3960). The opaque handle is an `sp_g4_kv*`.
 *
 * gemma4_kv_* is the engine null floor (byte-untouched). The ONLY additive
 * engine symbol this glue needs is the logits-returning decode step (§3.1
 * option A): gemma4_kv_decode_logits — NOT yet in cuda_forward.cu. Until it
 * lands at INTEGRATION the `step` body below is a TODO(WIRE-CUDA-DECODE) stub
 * that surfaces a clear error rather than silently arg-maxing.
 *
 * Forward-declared here (the glue stays free of the engine include path beyond
 * cuda_backend.h; these match the test-file prototypes exactly). sp_g4_kv is
 * opaque on this side — we only ever pass the pointer through. */
typedef struct sp_g4_kv sp_g4_kv;
extern sp_g4_kv *gemma4_kv_open(const qwen3_model *m, int Pmax);
extern int   gemma4_kv_prefill(sp_g4_kv *s, const int32_t *toks, int n);
extern int   gemma4_kv_prefill_batched(sp_g4_kv *s, const int32_t *toks, int n);  /* #41 batch prefill */
extern int   gemma4_kv_prefill_batched_from(sp_g4_kv *s, const int32_t *toks, int n, int P);  /* #41b suffix batch */
extern int   gemma4_kv_rewind(sp_g4_kv *s, int delta);
extern int   gemma4_kv_reset(sp_g4_kv *s);
extern int   gemma4_kv_reset_cold(sp_g4_kv *s);
extern int   gemma4_kv_shear(sp_g4_kv *s, int P);   /* KAIROS P1c-2 prefix shear */
/* PREFIX SNAPSHOT / RESTORE (2026-07-13) — the 164-second hello.
 * The shear is O(1) but only legal while the SWA ring has never wrapped, and the operator's
 * preamble (2517 tok) is LONGER than ring_W (2048), so the ring wraps inside the preamble and
 * the shear can never fire at this config. Copying the bytes works regardless of wrapping,
 * because it does not reason about wrapping at all. ~540 MiB, ~90 ms, vs 164,000 ms. */
/* ONE-SHOT: this session is decoded once and released, never continued. It is the precondition
 * gemma4_kv_prefill_batched actually needs — and the guard was reading a PROCESS-WIDE
 * SP_PERSIST_KV instead, so the chat path's need for persistence disqualified the batched path
 * for the judge/reflect calls that have nothing to continue. 78 s to produce one YES/NO token. */
extern int   gemma4_kv_set_oneshot(sp_g4_kv *s, int on);
/* SCRATCH: ring-off, one-shot, sized to its own prompt. A judge call must not be handed the
 * process's 2048-slot SWA ring (~490 MB it never reads) — on a full card that spills to host
 * memory and slows EVERY caller, including the conversation this route exists to protect. */
extern sp_g4_kv *gemma4_kv_open_scratch(const qwen3_model *m, int Pmax);
extern long  gemma4_kv_prefix_bytes(const sp_g4_kv *s, int P);
extern long  gemma4_kv_snapshot_prefix(const sp_g4_kv *s, void *host, long cap, int P);
extern int   gemma4_kv_restore_prefix(sp_g4_kv *s, const void *host, long cap, int P);
extern int   gemma4_kv_pos(const sp_g4_kv *s);
extern void  gemma4_kv_close(sp_g4_kv *s);
/* WIRE-CUDA-DECODE-GEMMA4 §3.1.A — additive logits-returning step (now defined
 * in cuda_forward.cu, declared in sp_engine/cuda_backend.h). Forward K/V at the
 * live dpos, D2H the post-head full-vocab logits row [n_vocab], advance dpos. */
extern int gemma4_kv_decode_logits(sp_g4_kv *s, int32_t token, float *logits);
/* CONTRACT-CHAT-FULLSTACK B1 — per-session byte-exact "auditable mode" setter. */
extern int gemma4_kv_byteexact_set(sp_g4_kv *s, int on);
/* CONTRACT-CHAT-FULLSTACK B2 (§6d-b) — episode replay into the live turn. */
extern int gemma4_kv_replay(sp_g4_kv *s, const char *epdir, int npos, int zero);
/* B3-v10 ABLATION GATE — memset-zero k specific episode positions (base+pos[i]) for the
 * thermodynamic-knockout re-score; restored by gemma4_kv_rewind (transient). */
extern int gemma4_kv_ablate_rows(sp_g4_kv *s, int base, const int *pos, int k);
/* CONTRACT-CHAT-FULLSTACK B5 (§6e) — the single latent entry seam.
 *   inject_tokens: TEXT via the residual seam, bit-identical to prefill by construction.
 *   inject_seq:    GENERIC residual-frame channel (audio/memory) at a placeholder. */
extern int gemma4_kv_inject_tokens(sp_g4_kv *s, const int32_t *toks, int n);
/* G-INT-2-FIX — attenuated inject for the LIVE recall path (constant-budget K scale). */
extern int gemma4_kv_inject_tokens_atten(sp_g4_kv *s, const int32_t *toks, int n);
extern int gemma4_kv_inject_seq(sp_g4_kv *s, const float *embs, int n_frames, int ph_token);
/* CONTRACT-CHAT-FULLSTACK B3 (AUTONOMOUS RECALL) — read global-owner K [0,npos)
 * out of the resident cache for the daemon's C2 query-signature. Returns the
 * number of global layers written, or -1. Cache byte-untouched (read-only D2H). */
extern int gemma4_kv_read_global_k(const sp_g4_kv *s, float *out, int npos);
extern int gemma4_kv_global_kvd(const sp_g4_kv *s);
extern int gemma4_kv_geometry(const sp_g4_kv *s, int *out, int n);
/* CONTRACT-CHAT-FULLSTACK B3-v2 (q·K AUTONOMOUS RECALL) — run one non-committing
 * forward of `token` at the live dpos and read the last-token GLOBAL-layer query
 * (post-RoPE) into `out` (packed [n_global][g_nh*g_hd] row-major). dpos is rolled
 * back; the cache is unchanged for the caller's subsequent replay + decode. Returns
 * the number of global layers written (>0), or -1. */
extern int gemma4_kv_read_global_q(sp_g4_kv *s, int token, float *out);

/* ── B4 NIGHTSHIFT Option-2 PROVENANCE FIX — batched-forward episode capture ──
 *
 * The curator built ep.k/ep.v/ep.mf via the BATCHED one-shot forward
 * `gemma4_decode_cuda` (cuda_forward.cu) with SP_XBAR_RECALL_WRITE pointing at
 * the episode dir (writer ~cuda_forward.cu:2642). The W_c head trained on THAT
 * provenance. The previous NIGHTSHIFT capture used a per-token
 * gemma4_kv_prefill + read_global_k (scratch session), which evolves the gemma4
 * AltUp/PLE residual differently => structurally-divergent K => live episodes
 * super-attract => foreign-reject fails. This entry captures a live episode
 * through the SAME batched path so the on-disk ep.k is byte-compatible with the
 * curated registry, and the deployed W_c head works with ZERO retraining.
 *
 *   qm_opaque : the session's borrowed qwen3_model* (const void*).
 *   tokens    : the raw episode token ids (no chat template — match the curator).
 *   n         : token count.
 *   out_dir   : the episode dir to write ep.k/ep.v/ep.mf into (must exist).
 *
 * gemma4_decode_cuda allocates its OWN scratch cache and reuses the model's
 * cached device weights, so this adds NO 9GB reload — just a small per-call
 * forward. n_gen=0 => forward-only over seq[0..n). We set SP_XBAR_RECALL_WRITE
 * tightly around the one call and UNSET it immediately after (the capture is
 * serialized under the daemon's kvdecode Mutex; the chat path uses g4_kv_step,
 * not gemma4_decode_cuda, so there is no concurrent reader of the env). Returns
 * gemma4_decode_cuda's rc (0 on success). */
extern int gemma4_decode_cuda(const qwen3_model *m, int32_t *seq,
                              int n_prompt, int n_gen, int eos_id);

int sp_daemon_cuda_kvcapture_batched(const void *qm_opaque, const int32_t *tokens,
                                     int n, const char *out_dir) {
    const qwen3_model *qm = (const qwen3_model *)qm_opaque;
    if (!qm || !tokens || n <= 0 || !out_dir) {
        sp_set_error("cuda kvcapture_batched: bad args");
        return -1;
    }
    /* gemma4_decode_cuda wants a MUTABLE int32_t* seq; copy the const input. */
    int32_t *seq = (int32_t *)malloc((size_t)n * sizeof(int32_t));
    if (!seq) { sp_set_error("cuda kvcapture_batched: seq OOM"); return -1; }
    for (int i = 0; i < n; i++) seq[i] = tokens[i];

    /* Point the curator's WRITE path at out_dir, run the batched forward, then
     * unset tightly. _putenv_s is the Windows env setter (this is the host CUDA
     * build); empty value clears the var. */
    /* E2 (2026-08-29 audit): a pre-existing operator/curator export of this var was
     * clobbered forever by the first live capture — save and restore it. */
    char saved_wr[1024] = {0};
    { const char *prev = getenv("SP_XBAR_RECALL_WRITE");
      if (prev && prev[0]) snprintf(saved_wr, sizeof saved_wr, "%s", prev); }
    _putenv_s("SP_XBAR_RECALL_WRITE", out_dir);
    int rc = gemma4_decode_cuda(qm, seq, n, /*n_gen*/0, /*eos*/1);
    _putenv_s("SP_XBAR_RECALL_WRITE", saved_wr[0] ? saved_wr : "");

    free(seq);
    /* gemma4_decode_cuda returns the valid token count `n` (= n_prompt for
     * n_gen=0) on SUCCESS and -1 on error (cuda_forward.cu: `int rc=-1, n=n_prompt;`
     * then `rc = n`). Normalize to 0=success / -1=failure so the Rust caller's
     * `rc == 0` check is correct (a non-negative count is NOT a failure). */
    return (rc < 0) ? -1 : 0;
}

/* open(qm, pmax) -> sp_g4_kv* (as void*). NULL on failure (sp_last_error set by
 * gemma4_kv_open). pmax = max resident position budget. Requires the model to
 * be SP_ARCH_GEMMA4 and SP_CUDA_DECODE_INT8=1 in the env (tied head). */
void *sp_daemon_cuda_kvdecode_open(const void *qm_opaque, int pmax) {
    const qwen3_model *qm = (const qwen3_model *)qm_opaque;
    if (!qm) { sp_set_error("cuda kvdecode open: NULL qwen3_model"); return 0; }
    if (qm->cfg.arch != SP_ARCH_GEMMA4 && qm->cfg.arch != SP_ARCH_GEMMA4_MOE) {
        /* ADR-013: the MoE model shares the whole persistent-KV decode (ring, shear,
         * persist); only its FFN differs. Refusing it here silently demotes the
         * daemon to the math-core CPU reference decode - correct, but ~100x slower,
         * and it never touches the CUDA MoE path at all. */
        sp_set_error("cuda kvdecode open: persistent-KV decode is SP_ARCH_GEMMA4 / GEMMA4_MOE only");
        return 0;
    }
    return (void *)gemma4_kv_open(qm, pmax);
}

/* prefill(handle, tokens, n): ingest history into the resident cache. 0 ok. */
int sp_daemon_cuda_kvdecode_prefill(void *handle, const int32_t *tokens, int n_tok) {
    sp_g4_kv *s = (sp_g4_kv *)handle;
    if (!s || !tokens || n_tok <= 0) { sp_set_error("cuda kvdecode prefill: bad args"); return -1; }
    return gemma4_kv_prefill(s, tokens, n_tok);
}

/* #41 batch prefill: one n-wide batched forward that sinks K/V into the resident
 * cache (CONTRACT-BATCH-PREFILL). Cold + ring-off + full-cache only; FLOAT (chat
 * speed mode, not byte-exact). 0 ok, -1 on precondition fail (caller falls back). */
int sp_daemon_cuda_kvdecode_prefill_batched(void *handle, const int32_t *tokens, int n_tok) {
    sp_g4_kv *s = (sp_g4_kv *)handle;
    if (!s || !tokens || n_tok <= 0) { sp_set_error("cuda kvdecode prefill_batched: bad args"); return -1; }
    return gemma4_kv_prefill_batched(s, tokens, n_tok);
}

/* #41b suffix batch: batched forward of ONLY the suffix at absolute positions
 * [from, from+n), extending a live cache standing exactly at `from`. The engine
 * enforces dpos==from and errors otherwise (caller falls back). The caller OWNS the
 * continuation-safe snapshot round-trip — see the routes.rs dispatch. 0 ok. */
int sp_daemon_cuda_kvdecode_prefill_batched_from(void *handle, const int32_t *tokens,
                                                 int n_tok, int from) {
    sp_g4_kv *s = (sp_g4_kv *)handle;
    if (!s || !tokens || n_tok <= 0 || from < 0) { sp_set_error("cuda kvdecode prefill_batched_from: bad args"); return -1; }
    return gemma4_kv_prefill_batched_from(s, tokens, n_tok, from);
}

/* decode_step(handle, token, logits): forward ONE token at the live dpos, write
 * the full-vocab logits row [n_vocab] for the NEXT position, advance dpos.
 * Wired (WIRE-CUDA-DECODE §7.1/§3.1.A) to the additive gemma4_kv_decode_logits in
 * cuda_forward.cu — the logits-returning sibling of the null-floor argmax-only
 * gemma4_kv_decode. L2 owns sampling over the returned row. 0 on success. */
int sp_daemon_cuda_kvdecode_step(void *handle, int32_t token, float *logits) {
    sp_g4_kv *s = (sp_g4_kv *)handle;
    if (!s || !logits) { sp_set_error("cuda kvdecode step: bad args"); return -1; }
    return gemma4_kv_decode_logits(s, token, logits);
}

/* rewind(handle, n): O(1) cold-evict (dpos -= n). 0 ok. */
int sp_daemon_cuda_kvdecode_rewind(void *handle, int n) {
    sp_g4_kv *s = (sp_g4_kv *)handle;
    if (!s || n < 0) { sp_set_error("cuda kvdecode rewind: bad args"); return -1; }
    return gemma4_kv_rewind(s, n);
}

/* reset(handle): CONTRACT-CHAT-FULLSTACK B2 RING-FIX — clean per-request reset
 * to dpos=0 WITHOUT journal replay. gemma4_kv_rewind(pos) replays the SWA-owner
 * undo-journal in reverse and reads jK/jV[L]+j*kvd for j up to pos-1, which is
 * OUT OF BOUNDS past the flat Jmax*kvd journal once pos>Jmax (the diagnosed B2
 * ring-reset bug). gemma4_kv_reset just zeroes the counters (dpos/commit_pos/
 * jcur); stale ring slots are never read because the next turn writes them
 * fresh in position order. The chat path resets each request via this. 0 ok. */
int sp_daemon_cuda_kvdecode_reset(void *handle) {
    sp_g4_kv *s = (sp_g4_kv *)handle;
    if (!s) { sp_set_error("cuda kvdecode reset: NULL handle"); return -1; }
    return gemma4_kv_reset(s);
}

/* reset_cold(handle): G-INT-2-FIX — like reset() but ADDITIONALLY zeroes every
 * owner K/V cache (and the SWA undo-journal), so a reconstruction truly starts
 * cold. Used by the B3-JUDGE branch after the nested judge forward (which advances
 * dpos past the prompt anchor and leaves judge K/V in slots >= n-1); the plain
 * reset()+prefill(head) leaves that residue resident, and once a PICK injects memory
 * the synthesis window sweeps over it -> prompt-echo. Byte-identical to the normal
 * null-floor reset for the standard path (the zeroed slots are never read after a
 * fresh prefill). 0 ok. */
int sp_daemon_cuda_kvdecode_reset_cold(void *handle) {
    sp_g4_kv *s = (sp_g4_kv *)handle;
    if (!s) { sp_set_error("cuda kvdecode reset_cold: NULL handle"); return -1; }
    return gemma4_kv_reset_cold(s);
}

/* KAIROS P1c-2 prefix shear: O(1) position restore to P when the SWA ring has
 * never wrapped this residency (slots [0..P) still positional — see the verb's
 * contract in cuda_forward.cu). 0 ok; -1 = wrapped/bad-args (caller falls back
 * to the full re-prefill floor). */
int sp_daemon_cuda_kvdecode_shear(void *handle, int P) {
    sp_g4_kv *s = (sp_g4_kv *)handle;
    if (!s) { sp_set_error("cuda kvdecode shear: NULL handle"); return -1; }
    return gemma4_kv_shear(s, P);
}

/* ── PREFIX SNAPSHOT / RESTORE (2026-07-13) ──────────────────────────────────────
 * The shear restores a prefix in O(1) but is only legal while the SWA ring has never
 * wrapped. The operator's preamble is 2517 tokens and ring_W is 2048 — THE RING WRAPS
 * INSIDE THE PREAMBLE — so the shear can never legally fire at his config, and the
 * fallback is a full re-prefill: 164 SECONDS to say "Hello! How are you today?".
 *
 * Copying the bytes works regardless of wrapping, because it does not reason about
 * wrapping at all: it restores the cache to the exact state it was in. Byte-exact by
 * construction. ~540 MiB over PCIe, ~90 ms.
 *
 * The blob is host memory owned by the CALLER (Rust). The engine never allocates it,
 * so there is no hidden lifetime and nothing to leak on a session teardown. */
void *sp_daemon_cuda_kvdecode_open_scratch(const void *qm, int Pmax) {
    return (void *)gemma4_kv_open_scratch((const qwen3_model *)qm, Pmax);
}

int sp_daemon_cuda_kvdecode_set_oneshot(void *handle, int on) {
    sp_g4_kv *s = (sp_g4_kv *)handle;
    if (!s) { sp_set_error("cuda kvdecode set_oneshot: NULL handle"); return -1; }
    return gemma4_kv_set_oneshot(s, on);
}

long sp_daemon_cuda_kvdecode_prefix_bytes(void *handle, int P) {
    sp_g4_kv *s = (sp_g4_kv *)handle;
    if (!s) { sp_set_error("cuda kvdecode prefix_bytes: NULL handle"); return -1; }
    return gemma4_kv_prefix_bytes(s, P);
}

long sp_daemon_cuda_kvdecode_snapshot_prefix(void *handle, void *host, long cap, int P) {
    sp_g4_kv *s = (sp_g4_kv *)handle;
    if (!s) { sp_set_error("cuda kvdecode snapshot_prefix: NULL handle"); return -1; }
    return gemma4_kv_snapshot_prefix(s, host, cap, P);
}

int sp_daemon_cuda_kvdecode_restore_prefix(void *handle, const void *host, long cap, int P) {
    sp_g4_kv *s = (sp_g4_kv *)handle;
    if (!s) { sp_set_error("cuda kvdecode restore_prefix: NULL handle"); return -1; }
    return gemma4_kv_restore_prefix(s, host, cap, P);
}

/* position(handle): current dpos, -1 on NULL. */
int sp_daemon_cuda_kvdecode_position(const void *handle) {
    const sp_g4_kv *s = (const sp_g4_kv *)handle;
    return gemma4_kv_pos(s);
}

/* close(handle): free the resident cache. NULL-safe. */
void sp_daemon_cuda_kvdecode_close(void *handle) {
    sp_g4_kv *s = (sp_g4_kv *)handle;
    if (s) gemma4_kv_close(s);
}

/* byteexact(handle, on): CONTRACT-CHAT-FULLSTACK B1 — toggle per-session
 * byte-exact "auditable mode" on the resident KV-decode cache. on!=0 routes the
 * islands+attention through the exact-integer (dual-prime CRT-NTT) substrate;
 * on==0 restores the float Stage-A path (byte-identical null floor). The chat
 * path holds the cache Mutex, sets on=1 at request start, on=0 at end. 0 ok. */
int sp_daemon_cuda_kvdecode_byteexact(void *handle, int on) {
    sp_g4_kv *s = (sp_g4_kv *)handle;
    if (!s) { sp_set_error("cuda kvdecode byteexact: NULL handle"); return -1; }
    return gemma4_kv_byteexact_set(s, on);
}

/* kv_flags(handle, flags): CONTRACT-CUDA-KV-FOUNDATION — set the KV codec flags
 * (bit0 = SP_KV_SPINOR) on the resident decode cache. flags==0 = float null floor. */
int sp_daemon_cuda_kvdecode_kv_flags(void *handle, unsigned int flags) {
    sp_g4_kv *s = (sp_g4_kv *)handle;
    if (!s) { sp_set_error("cuda kvdecode kv_flags: NULL handle"); return -1; }
    return gemma4_kv_set_kv_flags(s, flags);
}

/* replay(handle, epdir, npos, zero): CONTRACT-CHAT-FULLSTACK B2 (§6d-b) — recall a
 * stored episode's owner K/V into the resident cache at [dpos,dpos+npos) and advance
 * dpos (SP_REPLAY into a live turn). zero!=0 = the zeroed reject control. The chat
 * path holds the cache Mutex and calls this before decode when the turn names an
 * episode; reject = gemma4_kv_rewind(npos). 0 ok. */
int sp_daemon_cuda_kvdecode_replay(void *handle, const char *epdir, int npos, int zero) {
    sp_g4_kv *s = (sp_g4_kv *)handle;
    if (!s || !epdir || npos <= 0) { sp_set_error("cuda kvdecode replay: bad args"); return -1; }
    return gemma4_kv_replay(s, epdir, npos, zero);
}

/* ablate(handle, base, pos, k): B3-v10 — knock out k episode positions for the ablation
 * gate's knockout re-score. base = the episode's anchor (dpos at replay time). 0 ok. */
int sp_daemon_cuda_kvdecode_ablate(void *handle, int base, const int *pos, int k) {
    sp_g4_kv *s = (sp_g4_kv *)handle;
    if (!s) { sp_set_error("cuda kvdecode ablate: bad handle"); return -1; }
    return gemma4_kv_ablate_rows(s, base, pos, k);
}

/* inject_tokens(handle, toks, n): CONTRACT-CHAT-FULLSTACK B5 (§6e) — TEXT through
 * the single latent entry seam. Per token, stage embd[id]*sqrt(E) into the inject
 * buffer and step the real id; the residual entering layer 0 is bit-identical to
 * gemma4_kv_prefill(&id,1) (the B5 parity proof). 0 ok. */
int sp_daemon_cuda_kvdecode_inject_tokens(void *handle, const int32_t *toks, int n) {
    sp_g4_kv *s = (sp_g4_kv *)handle;
    if (!s || !toks || n <= 0) { sp_set_error("cuda kvdecode inject_tokens: bad args"); return -1; }
    return gemma4_kv_inject_tokens(s, toks, n);
}

/* inject_tokens_atten(handle, toks, n): G-INT-2-FIX — the LIVE recall inject seam.
 * Same residual entry as inject_tokens but the natively-minted memory K is scaled by
 * the constant-budget alpha (SP_REPLAY_MTARGET, default 42) so a recalled episode
 * BINDS instead of HIJACKING. Used by the B3-JUDGE/B3-WC live-recall PICK. 0 ok. */
int sp_daemon_cuda_kvdecode_inject_tokens_atten(void *handle, const int32_t *toks, int n) {
    sp_g4_kv *s = (sp_g4_kv *)handle;
    if (!s || !toks || n <= 0) { sp_set_error("cuda kvdecode inject_tokens_atten: bad args"); return -1; }
    return gemma4_kv_inject_tokens_atten(s, toks, n);
}

/* inject_frames(handle, embs, n_frames, ph_token): CONTRACT-CHAT-FULLSTACK B5 (§6e) —
 * the GENERIC residual-frame channel. Inject n_frames raw E-float residual vectors at
 * n_frames consecutive positions, each minted at ph_token. This is the seam the AUDIO
 * (EAR/KAI-3) and MEMORY (decoded episode residuals) sources feed through. 0 ok. */
int sp_daemon_cuda_kvdecode_inject_frames(void *handle, const float *embs, int n_frames, int ph_token) {
    sp_g4_kv *s = (sp_g4_kv *)handle;
    if (!s || !embs || n_frames <= 0) { sp_set_error("cuda kvdecode inject_frames: bad args"); return -1; }
    return gemma4_kv_inject_seq(s, embs, n_frames, ph_token);
}

/* read_global_k(handle, out, npos): CONTRACT-CHAT-FULLSTACK B3 (AUTONOMOUS RECALL)
 * — read the GLOBAL-owner K rows [0,npos) out of the resident cache into `out`
 * (packed [n_global][npos][g_kvd] row-major, global layers ascending), so the
 * daemon can compute the live query's C2 256-bit signature and match it against
 * the episode registry. Returns the number of global layers written (>0), or -1.
 * The cache is byte-untouched (read-only D2H). */
/* THE ROW STRIDE read_global_k WRITES AT — ask, do not reconstruct. Callers were sizing
 * their buffers from g_hd alone, which is right only while g_nkv==1 (the dense reference model). It is 2 on
 * gemma4-26b-a4b, so those buffers were half the size the engine writes into. */
int sp_daemon_cuda_kvdecode_geometry(const void *handle, int *out, int n) {
    const sp_g4_kv *s = (const sp_g4_kv *)handle;
    if (!s || !out) { sp_set_error("cuda kvdecode geometry: bad args"); return -1; }
    return gemma4_kv_geometry(s, out, n);
}

int sp_daemon_cuda_kvdecode_global_kvd(const void *handle) {
    const sp_g4_kv *s = (const sp_g4_kv *)handle;
    if (!s) { sp_set_error("cuda kvdecode global_kvd: NULL handle"); return -1; }
    return gemma4_kv_global_kvd(s);
}

int sp_daemon_cuda_kvdecode_read_global_k(const void *handle, float *out, int npos) {
    const sp_g4_kv *s = (const sp_g4_kv *)handle;
    if (!s || !out || npos <= 0) { sp_set_error("cuda kvdecode read_global_k: bad args"); return -1; }
    return gemma4_kv_read_global_k(s, out, npos);
}

/* read_global_q(handle, token, out): CONTRACT-CHAT-FULLSTACK B3-v2 — the q·K
 * autonomous-recall selector. Forward `token` at the live dpos through the resident
 * step, capture the last-token global-layer query into `out` (packed
 * [n_global][g_nh*g_hd]), roll dpos back. The daemon scores each registry episode by
 * q·K(ep.k). Returns the number of global layers written (>0), or -1. */
int sp_daemon_cuda_kvdecode_read_global_q(void *handle, int token, float *out) {
    sp_g4_kv *s = (sp_g4_kv *)handle;
    if (!s || !out) { sp_set_error("cuda kvdecode read_global_q: bad args"); return -1; }
    return gemma4_kv_read_global_q(s, token, out);
}

/* ── Engine-kernel shim over math-core forward_dispatch ────────────────────
 *
 * cuda_forward.cu reaches for `sp_engine/kernels.h` entry points by unprefixed
 * names (as_f32, and indirectly matmul/embed_row via the engine's host-side
 * weight-upload path). In the daemon-link build we DO NOT link cpu_overlay.c
 * (it would duplicate sp_kernels_read_env + qwen3_q4_stats already present in
 * math-core's libsp_forward_dispatch). Instead, expose the engine names as
 * thin aliases over math-core's sp_* variants. Same TU as the L1 dispatcher
 * so the link graph is one extra .o, matching the WIRE-HEX glue shape. */

int matmul(const qwen3_model *m, const gguf_tensor *W,
           const float *X, int n_tok, int in, int out, float *Y) {
    return sp_matmul(m, W, X, n_tok, in, out, Y);
}

int embed_row(const qwen3_model *m, int32_t tok, int E, float *dst) {
    return sp_embed_row(m, tok, E, dst);
}

const float *as_f32(const qwen3_model *m, const gguf_tensor *t) {
    return sp_as_f32(m, t);
}

/* sp_kernels_read_env: math-core's forward_dispatch.c already exports this
 * name; cuda_forward.cu's call site resolves to the math-core symbol directly,
 * no shim entry needed here. Mirrors the WIRE-HEX glue (sp_daemon_hex_glue.c). */
