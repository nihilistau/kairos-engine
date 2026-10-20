# Changelog — kairos-engine

What changed in the engine, by the day it changed. Numbers are from one card (RTX 2060 12 GB,
Windows/WDDM, a PCIe 3.0 link negotiated at **x8**) on Gemma-4-26B-A4B. Every knob below is
**off by default** in the daemon; the README's knob table says what upstream production arms.
Earlier history (2026-09) is in the README's dated sections and in `STATUS.md`.

## 2026-10-20 — decode attention, the CPU and the cache, then prefill

All bench numbers: 4,204-token prompt, greedy, a fresh daemon per arm, interleaved rounds.

### Decode

- **Split-K decode attention** (`SP_G4_ATTN_SPLIT`; 1 serves, 2 = parity). Flash-decoding: the
  grid is (KV head x position chunk), each block serves every query head of its KV head (each
  K/V row is read once), the softmax is a warp-shuffle reduction, and a combine merges the
  chunks. It replaces one block per query head (16 blocks on a 30-SM card, softmax on thread 0).
  Graph-safe: fixed grid, chunk derived from the device position.
  - parity relL2 <= 1.9e-06 at 4.2k and 9k depth, 1.4e-06 with the SWA ring armed;
  - decode 36.6 -> 32.8 ms/token at 4.2k, 63.2 -> 55.1 ms/token at 9k;
  - greedy output moves at near-tie words (a reduction-order change).
- **CPU expert threads 2 -> 4** (`SP_G4_MOE_CPU_THREADS`), 3 rounds, n=9: 30.83 -> 29.30
  ms/token, 4 winning every round; 6 is no better than 4. Output identical.
- **Expert cache 5.5 -> 6.0 GB** (`SP_G4_MOE_CACHE_GB`), 3 rounds, n=9: 30.05 -> 28.19 ms/token,
  winning every round. A 5k-token one-shot still fits; **6.5 GB does not** (the first batched
  prefill is declined on VRAM). Output identical.

### Prefill

- **Prefill copy overlap** (`SP_G4_MOE_OVERLAP_PREFILL`, needs `SP_G4_MOE_OVERLAP`). nsys showed
  a prefill's expert copies (4.4 s of 14.3 s) overlapping **0.0 s** of compute: the copy-stream
  pipeline was decode-only. It now runs for the batched prefill too (`G4_MOE_OVL_MAX` 32 -> 128).
  4.2k prefill 16.43 -> 13.95 s (n=9), output byte-identical in 27 runs. `SP_MOE_BUCKET_PIN` on
  top of it cancels the gain.
- **Prefill attention as tensor-core GEMMs** (`SP_G4_ATTN_PF_GEMM`; 1 serves, 2 = parity).
  S = QK^T as a batched fp16 GEMM (fp32 accumulate), a masked fp32 softmax, then O = PV, per
  block of queries and per KV head, on all three prefill attention paths.
  - 4.2k prefill 13.79 -> 6.49 s (n=9); 9k with the ring 40.7 -> 13.6 s (3.0x); a 5k one-shot
    18.2 -> 8.4 s;
  - parity relL2 <= 5.7e-04 (Q and P are fp16 operands, the precision `SP_G4_GEMM_F16`
    already accepted);
  - peak VRAM +~90 MB, in one shared scratch pool capped at 64 MB for S + P;
  - greedy output moves at near-tie words (identical at 9k over 64 tokens).
- **Eight-wide Q4 -> fp16 dequant** (`SP_G4_DEQ_H8`; 1 serves, 2 = bitwise parity). The prefill
  weight GEMMs dequantise each expert first; the old kernel ran one thread per weight at
  ~74 GB/s. This one takes 8 weights per thread (one code load, one scale, one 16-byte store),
  with the same arithmetic per weight, so the output is bit-identical by construction.
  4.2k prefill 6.40 -> 5.95 s (n=9, 3 rounds), a 5k one-shot 8.3 -> 7.7 s; bitwise parity 0 of
  2.69e11 weights differ over 86,001 calls; the output hash is identical in every run.

### Measured and not built

- **The GPU link is PCIe 3.0 x8** on this machine (`nvidia-smi` `pcie.link.width.current = 8`,
  max 16). A pinned 3.35 MB expert copy is 0.54 ms (6.16 GB/s, `tools/moe_cpu_bench/xfer_bench.cu`).
  Every miss upload and prefill copy pays it.
- **Apex-style load-aware miss assignment** (CPU-compute low-load misses, copy high-load ones):
  break-even n* ~= 190 tokens at x8 (CPU 0.16 ms per expert at 4 threads plus ~5 us/token,
  estimated). 70-85% of a prefill's staged experts sit below it, but they are 20-40% of the work.
  With the copies now overlapped, what is left to win is small; deferred.
- **Next-layer prefetch** (`SP_G4_PF_ORACLE`, test-only instrument): layer L+1's router applied
  to layer L's attention output predicts L+1's misses. The predicted top-8 covers ~60% of misses
  at ~31-37 uploads/token (two-thirds unused); a probability threshold of 0.02 covers ~50% at
  ~19 uploads (44% useful). Worth ~1.5 ms/token at most; not built.
- **Host-free layer P2** (`SP_G4_HF_P2`, off): correct (15,360 layers served, 0 fallbacks) but
  not faster (24.7 vs 26.6 tok/s). The host tail it targeted turned out to be mostly CPU expert
  compute (~7 ms/token at 2 threads), not round-trip overhead. Paused after P2.3.

## 2026-10-19

- **Against llama.cpp 0.7.0's `--moe-cache-mib`:** this engine is 1.30-1.37x faster at every
  cache size; `llama.cpp -ncmoe 14` (no cache) was 1.17x ahead on decode at 34.4 tok/s and 1.36x on
  prefill. See `STATUS.md`.
- **The batched fallback** consumes the pending CUDA error, so a declined batched prefill no
  longer poisons the next call; VRAM budget (DXGI) and commit-headroom lines are logged at each
  batched prefill.
