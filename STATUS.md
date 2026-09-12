# STATUS — kairos-engine

**Date:** 2026-09-12  
**Class:** `LIVE` — the optional native CUDA backend for the **companion** stack.

**Version:** `0.6.0` — the first tagged cut. Derived from **Shannon-Prime-Engine**, and the
number is a claim about lineage and maturity rather than a semver contract: it is one person
and one model's engine, research grade, one card, one OS.

What it does have is duration. It is the **only** backend of a live companion stack and has
served that stack continuously for months — which is a different kind of evidence from a
benchmark, and the one this repository can actually stand behind. Treat uptime claims as the
operator's experience of it: nothing here publishes an uptime receipt.

## Where the remaining gap is (2026-09-12, measured)

**Prefill is no longer behind `llama.cpp` — it is ahead.** On one workload in one sitting,
3,720 tokens, same card, same Q4_0 weights:

| | engine, kernels off | engine, kernels on | `llama.cpp -ncmoe 8` |
|---|---:|---:|---:|
| prefill 3,720 tok | 98.9 tok/s | **304.2 tok/s** | 262.6 tok/s |
| decode @ depth 3,720 | 12.70 tok/s | **16.16 tok/s** | **38.25 tok/s** |
| prefill + 128 decode | 46,016 ms | **21,715 ms** | ~17,277 ms |

The kernels are worth **2.12× end to end** — *not* the ~2.7× that appeared here previously,
which was two speedups measured on different prompts at different times and added together.
Prefill is a **1.16× lead**; decode is a **2.37× deficit**, depth-matched. **Decode is the whole remaining gap**, and it has been traced: in one 38.4 s call the GPU spent
15.1 s in kernels and **13.5 s receiving 81.4 GB of expert weights across PCIe** in 295,952
copies, at ~6.0 GB/s against a 9.33 GB/s link. It is the bus, not the kernels — which is also
exactly why `llama.cpp -ncmoe 8` wins decode: it computes those experts on the CPU and never
sends them. Candidates are in the README; none is claimed as *the* answer yet.

**A fix landed for it** (`SP_G4_MOE_OVERLAP=1`, decode only): expert staging moved to a second
CUDA stream with a per-expert event, so a layer's copies run beside its arithmetic. **1.078× on
decode, n=5, ranges non-overlapping, output byte-identical.** Coalescing the copies — the
obvious fix — was measured and rejected first: 67% of them are under 64 KB and they are 0.01%
of the bytes. The remaining ceiling is a routing dependency rather than a scheduling one.

## CI (2026-09-12)

There is CI now, and there had never been any. Linux, no GPU, fresh clone — it proves only that
the tree builds for somebody who is not the author. That bar was not being cleared: the first
run found `build-core-cpu.bat` pointing at a directory that does not exist in a clone, a math
core benchmark that does not link on glibc, and a documented Cargo profile
(`--no-default-features`) that could not compile because one call site imported a feature-gated
module unconditionally. The last of those is fixed; the first two are recorded here rather than
papered over.

## Recent — the attention kernels, 2026-09-12

Three kernels carried the same defect and it was worth about half the engine's GPU time. An
`nsys` trace against `llama.cpp` on the same card, same weights (Gemma-4-26B-A4B Q4_0 QAT,
RTX 2060 12 GB), same ~4k-prefill workload put **75% of GPU time in attention** here against
**0.8%** there. `ncu` then said the hot one was **L1/TEX-bound at 98.81% while doing 9.83% of
peak compute**, at 99.87% occupancy — so not a parallelism problem.

The cause, in each of `k_attn_from_tiled_T`, `k_attn` and `k_attn_decode_win_tiled_T`:

```c
for (int u = threadIdx.x; u < n; u += blockDim.x)      /* one thread per KEY */
    for (int i = 0; i < HD; i++) a += qh[i] * kv_ld(kh, i);   /* a whole 256-dot, alone */
```

A warp's 32 lanes sat on 32 *different* keys and stepped `i` together, so every load became 32
transactions instead of 1–4; and `qh` — identical across the block — was re-read from global
by every thread on every iteration. The fix is the same three times: **one warp per key** with
the lanes splitting the head dim and a `__shfl_down_sync` reduction, and the **query staged in
shared memory once**. Tiling, the online-softmax recurrence and the AV loop are untouched.

| | before | after |
|---|---:|---:|
| `k_attn_from_tiled_T`, per layer (ncu, same launch) | 280 ms | **42.85 ms** |
| its profile | L1 99.1% / compute 9.9% | **L1 83.7% / compute 78.6%** |
| prefill, 3,750 tokens | 36,573 ms | **18,205 ms** (2.01×) |
| decode, live stack, same prompt | 16.6 tok/s | **21.1 tok/s** (1.27×) |
| total GPU, traced workload | 30,937 ms | 13,466 ms |

It is compute-bound now instead of L1-bound, which is the shape the diagnosis predicted.

**Numerics.** Not bit-identical and cannot be: a tree reduction over 32 lanes is not the
sequential sum it replaces. `SP_G4_ATTN_V2=2` is a **parity mode** that runs both kernels every
layer and *serves the old one*, so the difference is measured on real weights during a real
prefill without the new path ever reaching output. Worst relative L2 over all layers:
**8.15e-07** (tiled prefill), **9.86e-07** (flat prefill), **2.86e-06** (decode, ring armed) —
fp32 reduction-order noise, ~√256·eps.

`SP_G4_ATTN_V2`: `0` the previous kernels, `1` the new ones, `2` parity. Disarming is one value.

**Still open:** `volta_sgemm_128x64_tn` is the largest single line at 24.2%, beside
`k_dequant_arena_q4b` — this engine dequantises Q4 to fp32 and runs cuBLAS **fp32** SGEMM,
where llama.cpp's hottest kernel `mul_mat_vec_q<Q4_0>` multiplies the Q4 bytes directly
against q8_1 activations.

**I first wrote that this was most of the remaining gap. It is not — tested 2026-09-12.** The
direct path is already in this tree: `gemm_q4b_dp4a_batched` + `k_gemm_q4b_dp4a`, wired into
the batched prefill behind `SP_KV_PREFILL_DP4A`, complete and with a clean decline. Measured
over three pairs on a 3,750-token prefill it is **~18% SLOWER** than dequant + SGEMM (every
"on" run above every "off" run), and because it quantises activations to int8 it **changes the
output text** at temperature 0 — a real precision reduction, not the reduction-order noise the
attention change produced. The knob being off is correct.

Quantised matmul is not inherently faster than fp32 SGEMM here. `volta_sgemm_128x64_tn` is
hand-tuned cuBLAS assembly; `k_gemm_q4b_dp4a` is a custom kernel. llama.cpp's advantage is not
the weight format, it is that `mul_mat_vec_q` is a very good kernel — swapping format without
matching the tuning loses.

**fp16 with tensor cores WAS the lever — tested and armed 2026-09-12.** Turing sm_75 has
tensor cores for fp16 and none for fp32, so `cublasSgemm` ran on the fp32 pipes while they sat
idle. `k_dequant_arena_q4b_h` writes `__half` and `cublasGemmEx(CUDA_R_16F, …,
CUBLAS_COMPUTE_32F)` does the GEMM; **the accumulator stays fp32**, so only the operands change
precision, and the dequant emits half directly rather than converting afterwards (a second
`in*out` pass would spend most of what the format saves).

| 3,750-token prefill | fp32 | fp16 |
|---|---:|---:|
| | 15,750 / 17,815 ms | **13,021 / 13,650 ms** |

**~1.26×, with byte-identical output text** over four runs at temperature 0 — the distinction
from the int8 path above, which was slower *and* changed what the model said. Parity worst
**relL2 4.5e-04**, which is fp16 operand precision and is recorded as a real numerical change
rather than noise.

`SP_G4_GEMM_F16`: `0` fp32, `1` fp16 tensor-core, `2` parity (both run, fp32 served).

**Zoo map:** [JOURNEY.md](https://github.com/nihilistau/Position_Is_Arithmetic/blob/main/JOURNEY.md)

This tree is in the **companion** family, not the lattice family. It is a curated cut of the
`sp-daemon` source that [Kairos](https://github.com/nihilistau/Kairos) runs when you choose
`[engine].kind = "sp"` — 151 files, fresh history, scrubbed. Kairos does **not** require it:
its default and supported path is any OpenAI-compatible `/v1/chat/completions` endpoint.

What lives here:

- `tools/sp_daemon` — the Rust HTTP/SSE daemon on `:3000`
- `src/backends/cuda` — the CUDA kernels, the MoE forward, the decode loop
- `include/sp_engine` — the C ABI the daemon links against
- `lib/shannon-prime-system` — **submodule** to the math core

**This repo does not supersede
[shannon-prime-system-engine](https://github.com/nihilistau/shannon-prime-system-engine),
which is `STANDING`** — and the reason is scope, not contents.

This is a **narrow cut**: 152 files, just enough to build the daemon the companion runs. It is
built ON the lattice substrate and therefore *contains* plenty of it — `SP_BYTEEXACT` in the
CUDA forward, `ptx_ntt.cuh` and `ntt_ffi.rs`, `sieve_ffi.rs` (KSTE / PoUW), the `sp_l1` ABI
bindings, and an optional `sp-swarm` crate (default-off; `build-wirecuda.bat` does not enable
it). What it is not is the place that work *lives*. That is the lattice tree — **2095 files**:
the full backend matrix, SP-SWARM's L0-L4 with its own gates, the memory agency, NIGHTSHIFT,
EAGLE / MTP, and the research history that produced all of it.

**A downstream cut does not replace the tree it was cut from.** JOURNEY.md rule 2: *do not
collapse the lattice family into a companion harness.*

**No weights.** A GGUF does not belong in git; `.gitignore` refuses `*.gguf`, `*.safetensors`
and `*.bin`. Bring your own.

Upstream commit: see `ENGINE-SOURCE.txt`.
