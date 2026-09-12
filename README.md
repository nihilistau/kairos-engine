# kairos-engine

**A from-scratch CUDA inference engine and HTTP daemon — built, and measured, on one 12 GB card.**

This is the optional native backend for [Kairos](https://github.com/nihilistau/Kairos). It is
a Rust daemon (`sp-daemon`) wrapping hand-written CUDA kernels — the KV cache and its ring,
prefill, fp16 KV, an MoE forward pass — behind an HTTP/SSE surface the harness talks to.

> **You may well not need this — but here is exactly what you give up.** Kairos's default
> and supported path is any OpenAI-compatible `/v1/chat/completions` endpoint (LM Studio,
> `llama-server`, vLLM, a cloud), and on one of those she is *most* of herself. The list below
> is measured, not estimated, and the harness degrades to it deliberately: every seam asks the
> backend what it `supports` and states its loss rather than failing or pretending.

## What actually needs this daemon

| what goes dark on a foreign endpoint | why |
|---|---|
| **She picks a severed sentence back up** (`CONTINUE`), and **finishes, then adds the thing she thought of on the way to the kettle** (`EXPAND`) | Both are driven by `eot_margin` — the raw stop-vs-continue logit gap, emitted on a named `event: kairos` SSE frame. It is the *forward's own* report that she had more to say, and no `/v1/chat/completions` server exposes it. Calibrated on the 26B: finished turns cluster at **+13.10**, guillotined ones at **-28.43**, threshold **-18.50** |
| **Sight through her own vision tower** | Residual frame injection into the model's own residual stream. Falls back with `SP_ENGINE_VISION=1` to an ordinary `image_url` part on multimodal endpoints — a different thing, honestly labelled |
| **Voice-in through the native ear** | Residual *audio* frames. There is no fallback; the voice service says so as a reply |
| **The L5 embedding space** (`/v1/embed`) | Her semantic index in the engine's own space. Falls back to a sidecar `/v1/embeddings` or a hash floor — same-space only, the seam never compares across spaces |
| **Prefill once, then extend** — the warm gate and the persisted-KV persona prefix | The persona + tools prefix is captured on ONE cold prefill and every later turn extends it. A foreign server owns its own cache discipline, so there is nothing to warm |
| `eot_bias`, `raw_logits`, byte-exact decoding, engine-enforced tool grammar, pre-tokenized input, `/v1/events`, engine-side tokens/sec, and the harness owning start / restart / watchdog | No wire field on a generic endpoint. Each is a declared capability; the room shows the chip on knobs that are moot without it |

**What does *not* need it — and this is most of what people mean by "her":** she still speaks
unprompted (`CHECK_IN`, `MUSE`, `REMIND`) and still does things in her own time (`SOLO`) —
those lanes take `eot_margin=None` and are decided before it is ever consulted. So is the
entire turn epilogue: the day row, memory admission, supersede, the identity firewall,
self-stances, the presence ledger, the journal. One `_settle_turn`, both mouths.

> One honest caveat on `/v1/capture`: the engine-side episode mint is a daemon capability, but
> it does **not** run on the 26B MoE — the route refuses it (ADR-013) and rows have carried
> `npos=0` since that model landed. It is not a reason to want this repo today.

The other reason it exists is the one it was built for: a companion living on a single RTX
2060, where a purpose-built runtime and a general one are not the same experience.

---

# How it works

Almost every design decision in this engine follows from one arithmetic problem, so it is
worth stating the problem before the solutions.

## The model, and why its shape decides everything

It was built for **Gemma-4-26B-A4B**, a sparse mixture-of-experts model, and the geometry is
not a detail — it is the whole argument for the engine's structure.

| | |
|---|---|
| layers / hidden | **30 / 2816** |
| routed experts | **128, top-8** — expert FFN 704 |
| shared ("dense") expert | **2112**, present in every layer |
| SWA layers | **25** — `n_head_kv=8`, `head_dim=256`, **window 1024** |
| global layers | **5** (`L % 6 == 5`) — `n_head_kv=2`, `head_dim=512`, and **no `attn_v`**, so V aliases K |
| vocab / native context | **262144 / 262144** |

Two consequences do most of the work:

**~4B parameters are active per token out of 26B.** Only 8 of 128 experts fire per layer, so
the model *computes* like a 4B and *stores* like a 26B. A dense 31B on this card would cross
8–9 GB of PCIe every token for a 3–5 tok/s ceiling. Sparsity is the only reason this is a
conversation rather than a batch job.

**Only 5 of 30 layers hold full-depth KV.** The other 25 slide a 1024-token window, and the
five that don't ship no `attn_v`, so V aliases K and the cost is not doubled. Long context is
therefore *cheap in KV* on this model in a way it is not on a dense one — the ceiling is a VRAM
question, not an attention-cost question.

## The memory problem, and the expert arena

The Q4 model is **13.26 GiB**; the card has ~11.2 GiB usable. It does not fit, and that single
fact produces the engine's most distinctive machinery.

The weights split cleanly: **~1.35 GB of non-expert weights**, which fit on the card outright,
and **~12.85 GB of expert weights**, which do not. So the experts tier:

- the full expert set lives in **pinned host RAM** — an *arena*, `cudaHostRegister`ed where it
  already lies, so the DMA engine reads it directly with no staging copy (`[g4-moe] arena pin:
  60 ok / 0 refused, 11.42 GB`);
- a **device-side expert cache** holds a slice of it in VRAM (default 4 GB);
- a miss stages that expert over PCIe before its GEMM.

The per-token expert working set is fixed by the model: top-8 × ~2.84 MB × 30 layers ≈
**682 MB/token**. Where you read it from is everything:

| read from | bandwidth | cost/token |
|---|---:|---:|
| GPU VRAM (resident) | ~336 GB/s | ~2 ms |
| host RAM, by the CPU | ~50 GB/s | ~14 ms |
| **over PCIe, to the GPU** | **9.33 GB/s measured** | **~73 ms** |

PCIe is the worst place to put that traffic, and the cache is what keeps it out: **33% of the
experts resident buys ~95.5% hits** in steady state, because MoE routing is *skewed* — capacity
share is not hit rate. At that hit rate only ~4.5% of lookups cross the bus, ~31 MB/token,
~3 ms. Raising the cache buys little precisely because it is already working; the measured
answer to "double it" was 3–4%.

> This is the opposite trade from `llama.cpp`'s `--n-cpu-moe`, which keeps some layers' experts
> in CPU memory and **computes them on the CPU** — moving the compute to the data. This engine
> moves the data to the compute and spends a cache to make that cheap. Both are defensible; the
> arithmetic above is why.

## Attention: two kinds of layer, one kernel family

Attention is tiled with an **online softmax** — the flash-attention recurrence: walk the keys
in tiles, carry a running max and sum, rescale the accumulator per tile. That makes shared
memory `O(tile)` instead of `O(context)`, which is what stops the context ceiling from being a
*kernel* limit and makes it a VRAM limit instead.

The 25 SWA layers additionally use a **ring**: their KV is a circular buffer of `ring_w` slots
and position `p` maps to slot `p % ring_w`. Because the window is bounded, the ring is O(1) in
context — a long conversation costs those layers nothing extra. The 5 global layers pay full
depth, which is affordable because there are only five of them.

Decode and prefill use different kernels because they are different shapes: prefill attends
many queries at once, decode attends exactly one.

## The KV cache, and why the first turn is slow and the rest are not

The persona + tools prefix is thousands of tokens that never change. Re-prefilling it every
turn would dominate every reply, so it is prefilled **once**, snapshotted, and every later turn
*extends* it rather than rebuilding it. That is the "warm gate": the harness will not let a
turn through until the prefix is hot.

KV is **fp16** for both the SWA owners and the globals, which halves the cache and, on this
model's geometry, puts long context within reach: 128K of global KV costs ~1.34 GB.

The failure mode worth knowing: if something invalidates that base snapshot, the next turn
re-prefills from scratch and the cost is immediately visible in the `TURN-PHASE` log line. The
daemon prints the split (`prefill` / `recall` / `decode` / `post`) on every turn precisely so
that this is never a mystery.

## The MoE forward

Per layer, per token: normalise, run the **router** GEMM to get 128 logits, take the top-8,
and sum the shared expert's output with the routed experts'. Three details in that sentence are
wrong-by-default and produce *fluent but wrong* text rather than a crash, so they are spelled
out at each use site in the source: the router runs on `attn_out` through its own weightless
norm (not the normed branch input); `ffn_gate_inp.scale` is per-input-dim while
`ffn_down_exps.scale` is per-expert (same suffix, different rank); and the dense and routed
branches are **summed**, not sequenced.

The expert loop is **expert-major**: rather than looping tokens and gathering each one's eight
experts, it buckets the whole batch *by expert*, then stages each expert once and runs one GEMM
over every token that wanted it. On a 2048-token prefill chunk that turns thousands of tiny
staged matmuls into a few hundred large ones.

Top-k currently runs on the **host** — one D2H and one `cudaStreamSynchronize` per layer. That
looks expensive and was measured: **99.9% of that sync is the CPU waiting for the GPU to finish
the layer's real work**, and the host-side routing itself is **0.53% of the routed branch**. It
is not the bottleneck it appears to be.

## Prefill and decode are different machines

| | prefill | decode |
|---|---|---|
| shape | matrix–matrix, hundreds–thousands of tokens | matrix–vector, one token |
| attention | tiled, many queries per launch | tiled, one query, ring on SWA layers |
| weight matmul | dequantise Q4 → **fp16 tensor-core GEMM** (`cublasGemmEx`, fp32 accumulate) | **fused int8 `dp4a` GEMV** — reads the Q4 codes straight from VRAM, no scratch |
| what limits it | attention and the weight GEMMs | expert bandwidth and per-layer launch overhead |

The decode GEMV is worth a line: dequantise-then-GEMM reads 1 byte of code, writes 4 bytes of
fp32 scratch and re-reads them — ~9 B/weight. The fused `dp4a` path reads **1 B/weight** and
accumulates in int32. That byte ratio is the whole win, and it only works because M=1 makes a
tensor-core tile pointless anyway.

## The daemon around it

`sp-daemon` is Rust: an HTTP/SSE server on `:3000` exposing `/v1/chat` (streaming),
`/v1/oneshot` (a call with no continuation — it gets its own scratch session and never touches
the resident cache), `/v1/capture`, `/v1/embed`, `/v1/metrics`, `/v1/events` and `/v1/abort`.
The tokenizer is in-tree C (`gemma4_bpe.c`), so prompts can be pre-tokenized and the template
is applied where the model is, not where the client is.

---

# Measured

Every number here is from this card — **RTX 2060, 12 GB, Windows/WDDM** — on the model above.
Nothing is estimated, and the negative results are kept because they were expensive.

**Dated, because there is a second table elsewhere.** These are the CUDA **kernel** results of
2026-09-11/12. The Kairos README carries an earlier and separate experiment — the memory-tiering
work of 09-08/09-10, about where the weights live rather than how the kernels run. The decode
figures differ between the two files because they are different runs of different work, not
because either is wrong, and they must not be added together. One pinned workload with every
kernel armed and `llama.cpp --n-cpu-moe` on the same row is **now measured** — see *One workload, both kernels, one sitting* below.

## Where the time went, and where it goes now

Profiled with Nsight Systems on a ~4k-token prefill plus 128 generated tokens:

| | before | after |
|---|---:|---:|
| 3,750-token prefill, end to end | **36,573 ms** | **~13,300 ms** (~2.7×) |
| total GPU time, traced workload | 30,937 ms | 13,466 ms* |
| decode, live stack, same prompt | 16.6 tok/s | **21.1 tok/s** |

<sub>*measured after the attention work and before the fp16 GEMM landed.</sub>

## What made it faster

**Three attention kernels carried one defect.** `nsys` put **75% of GPU time in attention**
here against **0.8%** for `llama.cpp` on the same card and weights. `ncu` then said the hottest
one was **L1/TEX-bound at 98.81% while doing 9.83% of peak compute**, at 99.87% occupancy — so
not a parallelism problem. The cause, identical in all three:

```c
for (int u = threadIdx.x; u < n; u += blockDim.x)          /* one thread per KEY */
    for (int i = 0; i < HD; i++) a += qh[i] * kv_ld(kh, i); /* a whole 256-dot, alone */
```

A warp's 32 lanes sat on 32 *different* keys and stepped `i` together, so every load became 32
transactions instead of 1–4; and the query — identical across the block — was re-read from
global memory by every thread on every iteration. The fix is the same three times: **one warp
per key** with the lanes splitting the head dim and a `__shfl_down_sync` reduction, and the
**query staged in shared memory once**.

| | before | after |
|---|---:|---:|
| `k_attn_from_tiled_T`, per layer | 280 ms | **42.85 ms** (6.5×) |
| its profile | L1 99.1% / compute 9.9% | **L1 83.7% / compute 78.6%** |

It is compute-bound now instead of L1-bound, which is the shape the diagnosis predicted.

**The tensor cores were idle.** Turing sm_75 has tensor cores for fp16 and **none for fp32**
(TF32 is Ampere and later), so `cublasSgemm` ran the weight GEMM on the fp32 pipes while the
tensor cores did nothing. Dequantising Q4 straight to `__half` and calling `cublasGemmEx` with
an **fp32 accumulator** — so only the operands change precision, not the sum — gave **~1.26×**
on a 3,750-token prefill with **byte-identical output text**.

## What did not, and is kept off

Three plausible optimisations were built or armed, measured, and rejected. They are in the tree
behind knobs that default off, with the numbers written beside them.

| tried | result |
|---|---|
| **Fused Q4×int8 `dp4a` prefill GEMM** (`SP_KV_PREFILL_DP4A`) — the `llama.cpp` approach, already implemented here | **~18% SLOWER** than dequant + `cublasSgemm`, and it quantises activations to int8, which **changes generated text** at temperature 0. `volta_sgemm` is hand-tuned cuBLAS assembly; a custom kernel has to beat that, not merely match the format |
| **Device-side MoE top-k** — remove the per-layer host sync | The sync is **99.9% GPU drain**; the host routing is **0.53%** of the branch. The rewrite buys half a percent |
| **Raising the expert cache** | 3–4%, because 33% residency already yields ~95.5% hits |

## Numerical honesty

None of the fast paths are bit-identical to the ones they replace, and they cannot be — a tree
reduction over 32 lanes is not the sequential sum it replaces, and fp16 operands are not fp32
ones. Each ships with a **parity mode** that runs both kernels and **serves the old one**, so
the difference is measurable on real weights during a real prefill without the new path ever
reaching output:

| change | parity mode | worst relative L2 |
|---|---|---:|
| tiled prefill attention | `SP_G4_ATTN_V2=2` | 8.15e-07 |
| flat prefill attention | `SP_G4_ATTN_V2=2` | 9.86e-07 |
| decode attention (ring armed) | `SP_G4_ATTN_V2=2` | 2.86e-06 |
| fp16 tensor-core GEMM | `SP_G4_GEMM_F16=2` | 4.5e-04 |

The first three are fp32 reduction-order noise (~√256·eps). The fourth is **not** noise — it is
fp16 operand precision, and it is recorded as a real numerical change rather than waved
through. Output text was byte-identical on every prompt tried, which is evidence, not proof;
`=0` disarms either one.

## One workload, both kernels, one sitting (2026-09-12)

Everything above this section was **stitched**: the attention result was measured before the
fp16 GEMM existed and the GEMM result after it, so no reader — and no author — had ever seen
both kernels run against both kernels off on one prompt. This is that run.

**3,720 tokens of ordinary varied prose**, greedy, 128 generated, RTX 2060 12 GB, Gemma-4-26B-A4B
Q4_0 (13.26 GiB). The engine is driven through `/v1/oneshot` on a directly-launched daemon so
only the three knobs move; `llama.cpp` is `llama-bench -ncmoe 8` on the same card and the same
weights. Engine n=3 for the totals and n=2 for the split, minimum reported.

| | engine, kernels **off** | engine, kernels **on** | `llama.cpp` `-ncmoe 8` |
|---|---:|---:|---:|
| prefill, 3,720 tok | 37,619 ms — 98.9 tok/s | **12,228 ms — 304.2 tok/s** | 14,169 ms — 262.6 tok/s |
| decode @ depth 3,720 | 12.70 tok/s | **16.16 tok/s** | **38.25 tok/s** |
| prefill + 128 decode | 46,016 ms | **21,715 ms** | ~17,277 ms |

The kernels are worth **3.08× on prefill**, **1.27× on decode**, **2.12× end to end**. The off
and on ranges do not touch: worst `on` (23,595 ms) is far below best `off` (46,016 ms).

**2.12×, not the ~2.7× stitched above.** Two speedups measured on different prompts at different
times do not compose, and the honest combined figure is the smaller one. The older rows stay
because they are what those experiments measured; this row is what the engine does.

### the gap moved, and this file was wrong about where it is

**Prefill is no longer the deficit — it is a lead.** 304.2 tok/s against 262.6 is **1.16× faster
than `llama.cpp`**, and the engine's number is if anything understated: it is measured through
the HTTP door and includes template application, tokenisation and one decode step, while
`llama-bench`'s `pp` is prompt processing alone.

**Decode is now the entire remaining gap: 16.16 tok/s against 38.25, so `llama.cpp` is 2.37×
faster there.** That comparison is depth-matched on purpose — `llama-bench`'s default `tg128`
runs at depth 0 and reports 41.19 tok/s, which would have flattered this engine by comparing a
cold-context decode against a 3,720-token one. Measuring it at `-d 3720` is what makes the
number mean anything.

So the next piece of work is **decode**, and the suspects are named rather than picked: per-step
launch overhead across 30 layers, the expert stage and its host sync, and WDDM. That is a
hypothesis with an instrument attached, not a third claim about where the time goes — the last
two such claims were both wrong, and both were killed by a trace rather than by argument.

**What this comparison is not.** `llama.cpp` with `-ncmoe 8` keeps eight layers' experts on the
CPU and computes them there; this engine streams experts from a pinned host arena to the GPU.
Different strategies for the same shortage of VRAM, compared on wall-clock for the same job,
which is the only axis a user experiences.

## Against `llama.cpp`, honestly

**This section said `llama.cpp` "remains faster at prefill" until 2026-09-12, and the combined
measurement above falsified it.** Prefill is now 304.2 tok/s here against 262.6 there — a 1.16×
lead — and the sentence survived only because nobody had run the two kernels together against a
depth-matched baseline. It is corrected rather than quietly deleted, because a README that
silently drops its own wrong claims teaches a reader nothing about how much to trust the rest.

Where `llama.cpp` is genuinely ahead is **decode: 38.25 tok/s against 16.16 at the same depth,
2.37×.** It is a mature, widely-tuned runtime and this is one person's kernel work, and on the
part of the job a user waits through most — the tokens arriving one at a time — it wins
comfortably.

What this engine offers is still not "faster than llama.cpp". It is the capability list at the
top of this file, on hardware that is otherwise too small for the model.

---

## What is actually here

| | |
|---|---|
| `tools/sp_daemon/` | the Rust daemon: `/v1/chat` (SSE), `/v1/oneshot`, `/v1/capture`, `/v1/embed`, `/v1/metrics`, `/v1/events`, `/v1/abort` |
| `src/backends/cuda/cuda_forward.cu` | the kernel — the MoE forward, the attention path, the decode loop |
| `src/tokenizer/` | the in-tree BPE, so prompts can be pre-tokenized |
| `include/sp_engine/` | the C ABI the daemon links against |
| `build-wirecuda.bat` | the build that produces the binary the harness expects |

The math core is a **separate repository** and a submodule:
[`shannon-prime-system`](https://github.com/nihilistau/shannon-prime-system). Clone with
`--recurse-submodules` or the CUDA build will not find it.

**No weights.** A GGUF does not belong in git. Bring your own.

### The knobs worth knowing

**Every fast path on this page is opt-in, and the shipped profile does not opt in.** The
daemon's own defaults are the conservative ones, and `profiles/sp.toml` in the public Kairos
cut sets none of these — so out of the box you get the *old* kernels and the fp32 GEMM, which
is to say none of the numbers above. That is deliberate: an unproven kernel should not arrive
armed on someone else's hardware. It also means you have to ask for it.

| knob | engine default | upstream production runs | what |
|---|---|---|---|
| `SP_G4_ATTN_TILE` | `0` (flat kernel) | `1024` | attention tile width. `0` is the null floor the tiled kernel is measured against |
| `SP_G4_ATTN_V2` | `0` (previous kernels) | `1` | the coalesced attention kernels. `2` = parity: both run, the **old one is served** |
| `SP_G4_GEMM_F16` | `0` (fp32 SGEMM) | `1` | fp16 tensor-core weight GEMM. `2` = parity, fp32 served |
| `SP_KV_PREFILL_DP4A` | `0` | `0` | fused Q4 prefill GEMM — **off, and measured slower**; see the table above |
| `SP_MOE_TIMING` | `0` | `0` | per-phase MoE instrumentation: sync / drain / route / stage / launch |
| `SP_G4_MOE_CACHE_GB` | `[kv] moe_cache_gb` | `4.0` | device expert cache budget. `0` disables the cache |

Kairos maps each of these from its profile, so arming them is a profile edit rather than an
environment variable — `serve.py` strips every unmapped `SP_*` on purpose, and a knob it does
not map cannot be set from your shell:

```toml
[decode]
attn_tile = 1024      # SP_G4_ATTN_TILE
attn_v2   = 1         # SP_G4_ATTN_V2   — 2 to run the parity check instead
gemm_f16  = 1         # SP_G4_GEMM_F16  — 2 to run the parity check instead
```

**Run the parity modes first on your own weights.** They serve the old kernel and report the
difference, so they cost you a slower prefill and nothing else. The numbers in *Numerical
honesty* above are this card and this model; yours are yours.

---

## Honesty about what this is

It is **research-grade, and it is one person's machine.** It was built and measured on
Windows, CUDA, one RTX 2060 12 GB, against one model family. The `.exe` paths in the build
scripts are Windows-shaped. There is no CI here, no matrix of tested cards, and no promise
that the kernels are correct on hardware nobody has run them on.

What there *is*: the thing runs, every day, as the only backend of a live companion — and the
harness that drives it keeps a gate suite that measures whether it is telling the truth.

If you want a general-purpose local runtime, use llama.cpp or vLLM. They are better at being
that, and Kairos is happy to talk to them.

---

## Building

**Clone it INTO your Kairos tree, at `engine/`** — not beside it. Kairos does
`from engine.launch import launch_daemon`, which resolves relative to the Kairos root, so a
sibling clone refuses politely and then cannot proceed. Kairos's `.gitignore` already expects
it there.

```bat
cd Kairos
git clone --recurse-submodules https://github.com/nihilistau/kairos-engine engine
cd engine
build-wirecuda.bat
```

That produces `target-wirecuda/release/sp-daemon.exe`. It is deliberately **not** cargo's
default `target/` — the harness profile points at the `wire_cuda_backend` build specifically,
and pointing it at a default build is a way to run a binary that is not the one you think.

Equivalent by hand:

```bat
cargo build --release --features wire_cuda_backend --target-dir target-wirecuda --bin sp-daemon
```

> The CUDA kernels are compiled by `build-cuda-backend.bat` into a separate library;
> `build-wirecuda.bat` only links against it. **Editing a `.cu` and re-running just the Rust
> build produces a new binary with the old kernels in it** — run the CUDA build first.

---

## Using it from Kairos

Kairos ships a profile for this and it is **off by default**:

```bash
python serve.py companion      # the default: any OpenAI-compatible endpoint
python serve.py sp             # this engine
```

`profiles/sp.toml` sets `[engine].kind = "sp"` and an `engine_exe` path — relative to the
Kairos root, so it points at `engine/tools/...` once this repo is cloned in place. If that
binary is not there, **`serve.py sp` refuses to start and says so** — it does not quietly fall back to the
OpenAI path. A mistyped stack that silently becomes the other one is how you spend an evening
debugging the wrong process.

See [`docs/BACKENDS.md`](https://github.com/nihilistau/Kairos/blob/main/docs/BACKENDS.md) in
Kairos for the table of what each backend can and cannot do — that table is the contract
between the two repos.

**Pin a version.** Engine releases are tagged and rarer than harness cuts. Floating this
repo's `main` next to a weekly harness export is how you get "harness green, daemon from
Tuesday, prefix from Friday". Point `engine_exe` at a release build, not at whatever last
compiled.

---

## Where this sits, and what it is NOT

There are two families here and they are deliberately separate. This repo belongs to the
**companion** one.

| repo | class | what |
|---|---|---|
| [Kairos](https://github.com/nihilistau/Kairos) | companion | the harness, the room, the memory architecture, the gates. Runs against any OpenAI-compatible endpoint; **this repo is what its continuation lanes, its vision tower and its warm prefix are built on** |
| **this repo** | companion | the optional native CUDA backend for that harness |
| [shannon-prime-system](https://github.com/nihilistau/shannon-prime-system) | lattice, `STANDING` | the exact-integer math core and the frozen L1 C ABI — carried here as the `lib/` submodule |
| [shannon-prime-system-engine](https://github.com/nihilistau/shannon-prime-system-engine) | lattice, **`STANDING`** | **not superseded by this repo.** See below |
| [shannon-prime-lattice](https://github.com/nihilistau/shannon-prime-lattice) | lattice | umbrella: papers, KEYSTONE, ADRs, SP-OKF / MEM-OKF, SWARM design |

### `shannon-prime-system-engine` is STANDING, not an ancestor

This repo is a **curated cut of the daemon source** that the companion stack runs — just enough
files to build the binary. The distinction from the lattice engine is **scope and ownership,
not contents.**

Being the daemon's source, this tree naturally *contains* lattice substrate: `SP_BYTEEXACT` in
the CUDA forward, `ptx_ntt.cuh` and `ntt_ffi.rs`, `sieve_ffi.rs` (the KSTE / PoUW bindings),
the `sp_l1` ABI, and an optional `sp-swarm` crate — default-off, and `build-wirecuda.bat` does
not enable it. **Using that work is not owning it.** It is developed in the lattice tree, 2095
files, where these live and this cut does not:

- **SP-SWARM / DHT as a system** — L0 QUIC, L1 content addressing, L2 have/want replication,
  L3 Ed25519 provenance, L4 C2-SimHash discovery, with its own gates and its design papers.
  What ships here is the transport crate the daemon can optionally link.
- **The byte-exact exact-integer forward as a research line** — `O_K = Z[(1+√−163)/2]`,
  dual-prime negacyclic CRT-NTT, the four exact islands. Auditability and cross-machine
  determinism, measured there.
- **The full NTT / CRT / Frobenius / ARM / Ring-3 VSA kernel matrix** and the contracts every
  backend gates to. One CUDA backend's worth of it reaches this cut.

**A downstream cut does not replace the tree it was cut from.**

The project keeps a map of which epoch each repo belongs to —
[`JOURNEY.md`](https://github.com/nihilistau/Position_Is_Arithmetic/blob/main/JOURNEY.md) —
and a `STATUS.md` tombstone in each tree. Its first rule is *"do not collapse the lattice
family into a companion harness."* **Read a repo's `STATUS.md` before treating its README as
current**, this one included.

---

## Licence

MIT. See [`LICENSE`](LICENSE).
