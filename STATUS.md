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
which is `STANDING`.** That tree holds unique lattice substrate that is not here and is not a
companion feature: SP-SWARM / DHT (QUIC, content addressing, Ed25519 provenance, C2
discovery), the byte-exact exact-integer forward (`SP_BYTEEXACT`), the NTT / CRT / Frobenius
kernels, and the frozen L1 C ABI. JOURNEY.md rule 2: *do not collapse the lattice family into
a companion harness.*

**No weights.** A GGUF does not belong in git; `.gitignore` refuses `*.gguf`, `*.safetensors`
and `*.bin`. Bring your own.

Upstream commit: see `ENGINE-SOURCE.txt`.
