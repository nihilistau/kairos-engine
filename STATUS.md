# STATUS — kairos-engine

**Date:** 2026-09-12  
**Class:** `LIVE` — the optional native CUDA backend for the **companion** stack.

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

**The untested option is fp16 with tensor cores**: dequantise to `__half` rather than `float`
and run `cublasGemmEx(CUDA_R_16F, …, CUBLAS_COMPUTE_32F)`. Turing sm_75 has tensor cores for
fp16 and none for fp32, so the engine currently uses none of them; this would halve the
dequant write bandwidth and put the GEMM on hardware that is idle today. That is the next
experiment, and unlike the one above it has not been tried.

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
