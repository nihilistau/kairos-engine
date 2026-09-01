# WIRE-CUDA-DECODE-GEMMA4 — contract addendum

**Status:** SCAFFOLD (compiling stubs, feature-gated). Device wiring is the
INTEGRATION step (see §7). Gate name: **`G-WIRE-CUDA-DECODE-GEMMA4`**.

**Parent contract:** `<path-to>\shannon-prime-lattice\papers\CONTRACT-BYTEEXACT-forward.md` §8
(the byte-exact CUDA forward bridge) + the engine L1-ABI pattern in
`include/sp/sp_l1.h` §6 (`sp_session_register_forward_backend`).

**Predecessor:** engine commit `eee3aac` — `G-WIRE-CUDA-GEMMA4 GREEN` proved the
universal daemon driving `gemma4_forward_cuda` for **PREFILL** through the
`sp_session_register_forward_backend` hook (the `sp_forward_dispatch_fn` ABI).

---

## 1. The problem this addendum solves

`sp_forward_dispatch_fn` (sp_l1.h §6) is **PREFILL-ONLY by contract** — it re-runs
the full forward over the *accumulated history* per call (the engine's ppl-style
usage). Full dense token-by-token DECODE through that hook is two failures at once:

1. **Performance.** Re-running the entire history per generated token is
   O(N²) — devastating, the exact thing the persistent-KV path exists to delete.

2. **Correctness — the actual blocker tonight.** A dense reference model OK_Q4B model keeps only
   packed weight *codes*; the tied full-vocab LM head is materialized to f32 only
   inside the DECODE path. Driving DECODE through the prefill hook trips the guard
   at `src/backends/cuda/cuda_forward.cu:1627`:

   ```
   -4: g4 probe: FULL head needs the f32 embd
   ```

   i.e. `gemma4_forward_cuda` (the prefill entry) deliberately reserves the
   full-vocab head for `gemma4_decode_cuda` / the persistent-KV
   `gemma4_kv_*` family. Prefill never needs all-position full-vocab logits; only
   the decode path does, and only at the live position.

The fix is a **second L1 verb** dedicated to persistent-KV decode, mirroring the
already-frozen `gemma4_kv_*` C ABI (defined `extern "C"` in `cuda_forward.cu`
~3478–3960; declared today inline in `tests/test_gemma4_cuda.c:65-79`).

---

## 2. The new verb — `sp_session_register_kvdecode_backend`

Symmetric to `sp_session_register_forward_backend` but bound to a **stateful,
session-resident KV handle** and a step-wise dispatch table (not a single
stateless forward function). Proposed L1-ABI surface (to be added to a FUTURE
frozen revision of `sp/sp_l1.h` §6b — NOT added in this scaffold; see §6):

```c
/* Opaque per-session KV-decode handle. On the CUDA backend this wraps an
 * sp_g4_kv* (cuda_forward.cu). Owned by the backend; lifetime tied to the
 * registration (open at register, close at unregister/destroy). */
typedef struct sp_kvdecode_handle sp_kvdecode_handle;

/* Step-wise persistent-KV decode dispatch table. Every fn returns 0 on
 * success, non-zero on error (sp_last_error carries detail). The backend owns
 * the device-resident cache across calls — this is the whole point.
 *
 *   open    (qm_opaque, pmax)            -> *handle      : alloc resident KV (dpos=0)
 *   prefill (handle, tokens, n_tok)                      : ingest history, store K/V, dpos+=n
 *   decode_step (handle, token, logits)                  : forward ONE token at the live
 *                                                          dpos, write full-vocab logits
 *                                                          [n_vocab] for the NEXT position
 *   rewind  (handle, n)                                  : O(1) cold-evict dpos -= n
 *   position(handle)                     -> int          : current dpos
 *   close   (handle)                                     : free the resident cache
 */
typedef struct sp_kvdecode_dispatch_fn {
    int  (*open)       (const void *qm_opaque, int pmax, sp_kvdecode_handle **out);
    int  (*prefill)    (sp_kvdecode_handle *h, const int32_t *tokens, int n_tok);
    int  (*decode_step)(sp_kvdecode_handle *h, int32_t token, float *logits);
    int  (*rewind)     (sp_kvdecode_handle *h, int n);
    int  (*position)   (const sp_kvdecode_handle *h);
    void (*close)      (sp_kvdecode_handle *h);
} sp_kvdecode_dispatch_fn;

/* Register a persistent-KV decode backend for this session. After registration,
 * sp_decode_step (the L1 single-token path) routes through dt->decode_step on
 * the session-resident handle instead of math-core's reference decode.
 *
 * `handle` is the backend-opaque KV handle (created by dt->open at register
 * time, or NULL to let L1 call dt->open lazily on first decode). Pass NULL
 * dt to unregister (L1 calls dt->close on the live handle).
 *
 * Thread-safety: caller-serialized with all &mut sp_session ops (the L2
 * Mutex<SpSession> guard). NOT safe concurrent with a decode on the session.
 * Returns SP_OK; SP_EBADARG on NULL session. */
sp_status sp_session_register_kvdecode_backend(
    sp_session *s,
    sp_kvdecode_handle *handle,
    const sp_kvdecode_dispatch_fn *dt);

/* Read-back accessors (NULL dt => no kvdecode backend; fall back to reference). */
sp_kvdecode_handle *sp_session_kvdecode_backend_handle(const sp_session *s);
const sp_kvdecode_dispatch_fn *sp_session_kvdecode_backend_dt(const sp_session *s);
```

### Why a dispatch TABLE, not a single fn

The prefill verb is a pure function `(model, tokens) -> logits`; one pointer
suffices. Decode is a **lifecycle**: open → prefill → N×decode_step → rewind →
close, all sharing one device-resident cache. A struct-of-fn-ptrs keeps the L1
side ABI-stable (one registration call) while letting the backend express the
full `gemma4_kv_*` lifecycle. This mirrors the math-core `sp_arm_ring2_backend`
stdio-ABI shape (a vtable of operations over a stateful store).

---

## 3. Mapping onto the existing `gemma4_kv_*` ABI

The CUDA backend ABI already exists and is the null floor (byte-untouched). The
new dispatch table is a **thin adapter** — one glue fn per row:

| `sp_kvdecode_dispatch_fn` row | engine `gemma4_kv_*` (cuda_forward.cu) | notes |
|---|---|---|
| `open(qm, pmax, &h)`        | `gemma4_kv_open(m, Pmax)` -> `sp_g4_kv*` | needs `SP_CUDA_DECODE_INT8=1` for the tied head (see kv_open:3670) |
| `prefill(h, toks, n)`       | `gemma4_kv_prefill(s, toks, n)` | stores K/V at `[dpos, dpos+n)` |
| `decode_step(h, tok, log)`  | **gap — see §3.1** | `gemma4_kv_decode` does internal argmax, returns no logits |
| `rewind(h, n)`              | `gemma4_kv_rewind(s, n)` | O(1) `dpos -= n` (KAI-1b slot==pos inverse) |
| `position(h)`              | `gemma4_kv_pos(s)` | returns `dpos_host` |
| `close(h)`                 | `gemma4_kv_close(s)` | frees the resident cache |

`sp_g4_kv*` IS the natural `sp_kvdecode_handle*` — the cast is in the glue.

### 3.1 The one real ABI gap — `decode_step` logits

`gemma4_kv_decode(s, n_gen, out)` runs `g4_kv_step(s, /*do_head=*/1)` internally
and **greedily argmaxes** — it writes *token IDs* into `out`, never exposing the
full-vocab logits. The daemon's `sp_decode_step` contract wants the **logits**
back so L2 owns sampling (temperature / top-p / the speculative-decode verify).

Two ways to close this at INTEGRATION time (decision deferred to the wiring
agent; both are additive, neither touches the null-floor `gemma4_decode_cuda`):

- **(A) preferred — a logits-returning kv step.** Add a sibling
  `gemma4_kv_decode_logits(sp_g4_kv *s, int32_t token, float *logits)` next to
  `gemma4_kv_decode` in `cuda_forward.cu`: it runs the SAME `g4_kv_step` but
  D2H-copies the post-head logits row `[n_vocab]` instead of the argmax token,
  and advances `dpos`. This is a small, additive `extern "C"` symbol — the
  existing `gemma4_kv_decode` stays byte-identical (its own null floor). The
  glue's `decode_step` calls this directly.

- **(B) fallback — argmax-only step.** If logits-out is descoped, `decode_step`
  can wrap `gemma4_kv_decode(s, 1, &tok)` and the glue synthesizes a one-hot
  logits row (or L2 takes the token directly via a narrower verb
  `decode_step_argmax`). This loses L2-side sampling but unblocks a greedy dense decode
  decode tonight. **The scaffold marks (A) as the chosen target** in
  `TODO(WIRE-CUDA-DECODE)`.

---

## 4. Files in this scaffold (all additive, feature `wire_cuda_backend`)

1. **`src/cuda_kvdecode_dispatch.rs`** — Rust trampoline, `#![cfg(feature =
   "wire_cuda_backend")]`, mirroring `cuda_forward_dispatch.rs`. Declares the
   `extern "C"` link surface for the C glue fns, a `register_with_session`
   entry (stubbed against the future L1 verb), a dispatch counter, and
   `release_for_model`. Real device calls marked `TODO(WIRE-CUDA-DECODE)`.

2. **`c_backend_cuda/sp_daemon_cuda_glue.c`** — appended C glue:
   `sp_daemon_cuda_kvdecode_{open,prefill,step,rewind,pos,close}` routing to the
   `gemma4_kv_*` symbols already compiled into the lib. The `step` fn carries
   the §3.1 `TODO(WIRE-CUDA-DECODE)` for the logits-out symbol.

3. **`src/state.rs`** — an `sp_g4_kv*` handle slot on `AppState`
   (`cuda_kvdecode_handle`), `cfg(feature = "wire_cuda_backend")`, mirroring the
   android `dsp_session` Option<Mutex<>> shape. A `Send`-wrapper newtype guards
   the raw device pointer.

4. **`src/lib.rs`** — `#[cfg(feature = "wire_cuda_backend")] pub mod
   cuda_kvdecode_dispatch;` next to `cuda_forward_dispatch`.

No change to: `cuda_forward.cu` kernels/decode, the byte-exact crate
(`tools/sp_dsp_smoke`), `build.rs`, `Cargo.toml` features (the existing
`wire_cuda_backend` feature covers this module — no new feature needed), or any
other committed work.

---

## 5. Runtime activation (INTEGRATION, not tonight)

Mirror the `daemon.rs` WIRE-CUDA block: gate on `SP_DAEMON_BACKEND=cuda` +
a decode opt-in (proposed `SP_DAEMON_KVDECODE=1`) so the new verb only
binds when explicitly requested — keeping the prefill-bridge default untouched.
On register: glue `open(qm, pmax)` allocates the resident `sp_g4_kv`; store the
handle in `AppState.cuda_kvdecode_handle`; `sp_session_register_kvdecode_backend`
points `sp_decode_step` at the glue. On shutdown: `close`.

---

## 6. Decisions / deferrals recorded

- **The L1 verb is NOT added to `sp/sp_l1.h` in this scaffold.** That header is
  the frozen ABI inside the `lib/shannon-prime-system` submodule (and the
  byte-exact crate's bindgen input); adding a verb there is a frozen-ABI change
  requiring its own upstream commit + bindgen regen + the no-silent-gate-revision
  surfacing. This scaffold designs the verb here and stubs the Rust/glue around
  it so the integration agent lands the header change deliberately. The Rust
  `register_with_session` therefore compiles against a **local placeholder**
  (returns `Ok(())` after the glue `open`, with the real
  `sp_session_register_kvdecode_backend` call behind `TODO(WIRE-CUDA-DECODE)`).

- **`decode_step` targets the logits-out symbol (§3.1 option A).** The
  argmax-only fallback (B) is documented but not the default.

- **No new Cargo feature.** `wire_cuda_backend` already links the lib that
  carries `gemma4_kv_*`; the kvdecode module rides the same feature.

- **Null floor preserved.** `gemma4_decode_cuda` and `gemma4_kv_decode` stay
  byte-identical; the only new engine symbol at integration is the additive
  `gemma4_kv_decode_logits` (option A).

---

## 7. Remaining INTEGRATION steps → `G-WIRE-CUDA-DECODE-GEMMA4`

1. Add `gemma4_kv_decode_logits(sp_g4_kv*, int32_t, float*)` to
   `cuda_forward.cu` (additive `extern "C"`; reuse `g4_kv_step` + a logits D2H).
   Declare it in the glue's local prototypes.
2. Add `sp_session_register_kvdecode_backend` + the `sp_kvdecode_dispatch_fn`
   table + accessors to `lib/shannon-prime-system/include/sp/sp_l1.h` §6b and its
   implementation in the math-core `sp_session.c`; route `sp_decode_step` through
   `dt->decode_step` when registered. Regenerate the daemon bindgen. Commit
   upstream (submodule) first.
3. Fill the `TODO(WIRE-CUDA-DECODE)` bodies in `sp_daemon_cuda_glue.c` (call the
   real `gemma4_kv_*` symbols) and `cuda_kvdecode_dispatch.rs` (call the real
   `sp_session_register_kvdecode_backend`).
4. Add the `daemon.rs` startup block (`SP_DAEMON_BACKEND=cuda` +
   `SP_DAEMON_KVDECODE=1`) + the `AppState.cuda_kvdecode_handle` population +
   shutdown `close`.
5. Build: `build-host-cuda-backend.bat` then
   `cargo build --features wire_cuda_backend --release`.
6. **Gate `G-WIRE-CUDA-DECODE-GEMMA4`:** the universal daemon, `SP_DAEMON_BACKEND=cuda`
   `SP_DAEMON_KVDECODE=1`, prefills a prompt + decodes ≥32 tokens of a real
   the dense reference model through `sp_decode_step` -> glue -> `gemma4_kv_decode_logits`, and
   the produced token stream is **identical** to the reference
   `gemma4_kv_decode` argmax sequence (the null-floor oracle) — proving the L1
   kvdecode verb drives the resident-KV dense decode bit-for-bit, with VRAM flat
   across the decode (O(1) cache, the KAI-1b property).
