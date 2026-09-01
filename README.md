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
| **Prefill once, then extend** — the warm gate and the persisted-KV persona prefix | The persona + tools prefix is captured on ONE cold prefill (~5 minutes) and every later turn extends it. A foreign server owns its own cache discipline, so there is nothing to warm |
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
| [Kairos](https://github.com/nihilistau/Kairos) | companion | the harness, the room, the memory architecture, the gates. Runs against any OpenAI-compatible endpoint; **this repo is what its continuation lanes, its vision tower and its warm prefix are built on** |
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
