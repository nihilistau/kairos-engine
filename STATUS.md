# STATUS — kairos-engine

**Date:** 2026-09-02  
**Class:** `LIVE` — the optional native CUDA backend for the **companion** stack.

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
