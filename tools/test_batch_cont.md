# Manual gate: `SP_KV_PREFILL_BATCH_CONT` (warm full re-batch)

**Why manual:** proving restored-prefix length + a subsequent continuation needs a
live `sp_g4_kv` + model weights. Offline gates cover the flag/threshold contract
(`cargo test -p sp-daemon batch_cont_prefill`). This script is the residual live leg.

**Design under test:** full re-batch (not offset). See `batch_cont_prefill.rs` and
the `BATCH-PREFILL-CONT:` branch in `routes.rs`.

## Build

From the worktree root (or `engine/tools/sp_daemon/`):

```bat
cd engine\tools\sp_daemon
set CARGO_TARGET_DIR=target-wirecuda
cargo build --features wire_cuda_backend --release
```

(Requires the host CUDA backend lib already built via `build-host-cuda-backend.bat`
/ the usual wire-cuda stack.)

## Offline gate (no GPU)

The pure helpers live in the lib crate and do not need CUDA. Prefer `--lib`
(or `--test batch_cont_prefill` with `wire_cuda_backend` so the package's
CUDA-gated bins resolve). Without the feature, plain `cargo test` also tries
to compile `sp-daemon` bin paths that import `cuda_kvdecode_dispatch` and fails
for an unrelated reason.

```bat
cd engine\tools\sp_daemon
set SP_SYSTEM_INCLUDE=...\core\include
set SP_SYSTEM_BUILD_DIR=...\engine\build-cpu\lib\shannon-prime-system
set LIBCLANG_PATH=C:\Program Files\LLVM\bin
cargo test -p sp-daemon --lib batch_cont_prefill -- --nocapture
```

With the wire-cuda target (same env as the release build):

```bat
cargo test --features wire_cuda_backend --target-dir target-wirecuda --lib batch_cont_prefill -- --nocapture
cargo test --features wire_cuda_backend --target-dir target-wirecuda --test batch_cont_prefill -- --nocapture
```

Expect all tests green. This proves:

- flag default-off leaves `want_batch_cont` false
- threshold miss (flag on, short suffix) declines → per-token fall-through predicate
- threshold hit + flag on arms the path; `single_entry` never does
- `prefill_from == 0` never takes the CONT arm (cold stays on `SP_KV_PREFILL_BATCH`)

## Live A/B (daemon + GPU)

1. Start the daemon with the production profile as usual, **without** the cont flag.
   Run two turns in one session so turn 2 is a warm continuation with a large suffix
   (800–1400 tokens of new prompt head is typical after a long reply).

2. In `var/daemon.log` (or the daemon stderr), find the warm turn:

   ```
   TURN-PHASE: prefill N tok in M ms (...)
   ```

   Note `N`, `M`, and that there is **no** `BATCH-PREFILL-CONT:` line. This is the
   per-token baseline.

3. Restart with the flag armed (and leave cold batch alone or arm it separately):

   ```bat
   set SP_KV_PREFILL_BATCH_CONT=1
   ```

   Repeat the same two-turn session shape.

4. Pass criteria for the warm turn:

   - A log line of the form:
     ```
     BATCH-PREFILL-CONT: T tokens full re-batch + snapshot-restored (mode=full-rebatch, warm_suffix=S, prefix=P) in M ms (X.X ms/tok)
     ```
     with `T = P + S`, and `M/T` in the same ballpark as cold batch (~10–20 ms/tok),
     not the per-token ~80 ms/tok band — **when** the threshold says it should fire
     (`S > full_rebatch_min_suffix(P)`).
   - OR, if a precondition fails (VRAM, engine decline):
     ```
     BATCH-PREFILL-CONT declined (...) -> per-token suffix prefill (S tok, prefix P kept)
     ```
     and the turn still completes (fall-through). That is also a pass for the
     safety contract.
   - Decode continues; the reply is coherent (no empty second turn à la G-PK2-BATCHRING).

5. **State contract (when introspection is available):** after a successful
   `BATCH-PREFILL-CONT` line, the next short continuation in the same session must
   not empty. If you have a debug path that reads `prefix_bytes` / position, the
   restored length after the cont prefill must equal `T` (full head). Byte-identical
   logits vs pure per-token are **not** claimed (FLOAT batch mode, same as cold
   `SP_KV_PREFILL_BATCH`).

6. **Flag-off regression:** unset `SP_KV_PREFILL_BATCH_CONT`, restart, confirm no
   `BATCH-PREFILL-CONT:` lines appear on warm turns (cold `BATCH-PREFILL:` may still
   appear if `SP_KV_PREFILL_BATCH=1`).

## Residual risk

- Offline tests do not load CUDA; they cannot assert cache bytes or logits.
- Full re-batch recomputes the prefix with the FLOAT batched kernels; if the live
  prefix was minted under byte-exact or per-token float, contents are not
  bit-identical — only length/position + continuation safety. Same doctrine as the
  existing cold batch + snapshot round-trip.
