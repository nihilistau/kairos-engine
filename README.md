# kairos-engine

**A from-scratch CUDA inference engine and HTTP daemon, built for one 12 GB card.**

This is the optional native backend for [Kairos](https://github.com/nihilistau/Kairos). It is
a Rust daemon (`sp-daemon`) wrapping hand-written CUDA kernels — the KV cache and its ring,
prefill, fp16 KV, an MoE forward pass — behind an HTTP/SSE surface the harness talks to.

> **You almost certainly do not need this.** Kairos runs against any OpenAI-compatible
> `/v1/chat/completions` endpoint — LM Studio, `llama-server`, vLLM, a cloud provider — and
> that is its default and its supported path. This repo exists because the companion it was
> written for runs on a single RTX 2060, where the difference between a general runtime and a
> purpose-built one is the difference between a four-second reply and a forty-second one.

---

## What is actually here

| | |
|---|---|
| `tools/sp_daemon/` | the Rust daemon: `/v1/chat` (SSE), `/v1/oneshot`, `/v1/capture`, `/v1/embed`, `/v1/metrics`, `/v1/events`, `/v1/abort` |
| `src/backends/cuda/cuda_forward.cu` | the kernel — the MoE forward, the attention path, the decode loop |
| `include/sp_engine/` | the C ABI the daemon links against |
| `build-wirecuda.bat` | the build that produces the binary the harness expects |

The math core is a **separate repository** and a submodule:
[`shannon-prime-system`](https://github.com/nihilistau/shannon-prime-system). Clone with
`--recurse-submodules` or the CUDA build will not find it.

**No weights.** A GGUF does not belong in git. Bring your own.

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
| [Kairos](https://github.com/nihilistau/Kairos) | companion | the harness, the room, the memory architecture, the gates. Talks to any OpenAI-compatible endpoint and **does not need this** |
| **this repo** | companion | the optional native CUDA backend for that harness |
| [shannon-prime-system](https://github.com/nihilistau/shannon-prime-system) | lattice, `STANDING` | the exact-integer math core and the frozen L1 C ABI — carried here as the `lib/` submodule |
| [shannon-prime-system-engine](https://github.com/nihilistau/shannon-prime-system-engine) | lattice, **`STANDING`** | **not superseded by this repo.** See below |
| [shannon-prime-lattice](https://github.com/nihilistau/shannon-prime-lattice) | lattice | umbrella: papers, KEYSTONE, ADRs, SP-OKF / MEM-OKF, SWARM design |

### `shannon-prime-system-engine` is STANDING, not an ancestor

This repo is a **curated cut of the daemon source** that the companion stack runs. It is not a
replacement for the lattice engine, and it does not own that work. The lattice tree holds
unique substrate that is not here and is not a companion feature:

- **SP-SWARM / DHT** — L0 QUIC, L1 content addressing, L2 have/want replication, L3 Ed25519
  provenance, L4 C2-SimHash discovery.
- **The byte-exact exact-integer forward** — `O_K = Z[(1+√−163)/2]`, dual-prime negacyclic
  CRT-NTT, the four exact islands, `SP_BYTEEXACT`. Auditability and cross-machine determinism.
- **NTT / CRT / Frobenius / ARM / Ring-3 VSA** kernels and contracts, and the frozen L1 C ABI
  every backend gates to.

The project keeps a map of which epoch each repo belongs to —
[`JOURNEY.md`](https://github.com/nihilistau/Position_Is_Arithmetic/blob/main/JOURNEY.md) —
and a `STATUS.md` tombstone in each tree. Its first rule is *"do not collapse the lattice
family into a companion harness."* **Read a repo's `STATUS.md` before treating its README as
current**, this one included.

---

## Licence

MIT. See [`LICENSE`](LICENSE).
