# AGENTS.md — kairos-engine

**The orientation file for anyone — human or agent — working in this repo.**

This is the optional native backend for
[Kairos](https://github.com/nihilistau/Kairos): a Rust daemon (`sp-daemon`) over hand-written
CUDA kernels, serving an HTTP/SSE surface on `:3000`. Kairos runs without it — its default and
supported path is any OpenAI-compatible endpoint — but "optional" is not "inconsequential".
**Four things exist only on this side of the wire:** the `eot_margin` that drives her
continuation lanes (`CONTINUE` / `EXPAND`), residual frame injection for sight and voice-in,
the L5 embedding space, and the persisted-KV warm prefix. Her unprompted speech, her own-time
acts and the whole turn epilogue do not need any of it. The README's table is the measured
list; `docs/BACKENDS.md` in the Kairos repo is the contract.

> **This is a curated export.** The source of truth is a private research tree; this repo is
> the curated subset that actually builds the daemon, cut with fresh history and scrubbed.
> `ENGINE-SOURCE.txt` names the upstream commit. Fixes are welcome as PRs; they get carried
> back upstream by hand.

---

## 0. THE ONE THING TO KNOW BEFORE YOU CHANGE A KERNEL

**The daemon is not the authority on anything except tokens.**

Memory rules, recall, admission, supersede, the identity firewall, what a fact is and who
said it — all of that lives in the *harness*, on the other side of the HTTP boundary, and it
got there the hard way. The daemon once had its own auto-capture (`growth = true`), which
stored whole user turns as facts if they passed a word count and mentioned a person. One
17-turn conversation put 17 rows in, including *"yes, we lose lips, sink ships."*

Two authorities decided what a memory was — the daemon's word-count-and-a-pronoun and the
harness's lifecycle rules — and the daemon won every time, **because it wrote first.**

So: the daemon's memory writers are off in every profile and refused at boot. If you are
adding a feature here that stores, judges, or interprets *meaning*, it is almost certainly in
the wrong repo. Serve tokens; let the harness decide what they were.

---

## 1. THE SHAPE

```
tools/sp_daemon/          the Rust daemon — routes, sampler, KV, the HTTP/SSE surface
  src/routes.rs           every /v1/* verb the harness calls
  src/recall.rs           the engine-side episode selection (OFF on the live profile)
  src/sampler.rs          decode, eot margin, repeat handling
src/backends/cuda/        the kernels
  cuda_forward.cu         the MoE forward and the decode loop — the big one
include/sp_engine/        the C ABI the daemon links against
lib/shannon-prime-system  SUBMODULE -> the math core (clone --recurse-submodules)
build-wirecuda.bat        the build that produces the binary the harness expects
```

`build-wirecuda.bat` writes to `target-wirecuda/`, **not** cargo's default `target/`. The
harness profile points at the `wire_cuda_backend` build specifically; a default build is a
different binary with different features, and pointing a profile at it is how you spend an
evening debugging a process that is not the one you changed.

---

## 2. THE SEAM

Kairos talks to this over HTTP and the contract is
[`docs/BACKENDS.md`](https://github.com/nihilistau/Kairos/blob/main/docs/BACKENDS.md) in that
repo — a table of what each backend can and cannot do. The harness asks a backend what it
`supports` and degrades rather than assuming; if you add a capability here, that table is
where it becomes real.

The verbs: `/v1/chat` (SSE, with a named `event: kairos` carrying the stop-vs-continue
margin), `/v1/oneshot`, `/v1/capture`, `/v1/embed`, `/v1/metrics`, `/v1/events`, `/v1/abort`.

**Version pinning matters.** Engine releases are tagged and rarer than harness cuts. A
floating `main` next to a weekly harness export gives you "harness green, daemon from Tuesday,
prefix from Friday" — three components that were each fine and were never tested together.

---

## 3. HARDWARE HONESTY

Built and measured on **Windows, CUDA, one RTX 2060 12 GB**, against one model family. The
`.exe` paths are Windows-shaped. There is no CI, no tested-card matrix, and no claim that the
kernels are correct on hardware nobody has run them on.

If a kernel is wrong on your card, that is entirely plausible and a bug report with the card
and the failing shape is genuinely useful.

---

## 4. WEIGHTS

Not here, and not ever. A GGUF does not belong in git — `.gitignore` refuses `*.gguf`,
`*.safetensors` and `*.bin` so that stays true by accident as well as on purpose. Bring your
own.

---

## 5. THE OTHER REPOS

**Two families, deliberately separate.** This repo is in the COMPANION one; the lattice family
is not its ancestor and this repo does not supersede any of it.

| repo | family / class | what |
|---|---|---|
| [Kairos](https://github.com/nihilistau/Kairos) | companion | the harness, the room, the memory architecture, the gates |
| **this repo** | companion | the optional native CUDA backend |
| [shannon-prime-system](https://github.com/nihilistau/shannon-prime-system) | lattice, `STANDING` | the math core — the `lib/` submodule here |
| [shannon-prime-system-engine](https://github.com/nihilistau/shannon-prime-system-engine) | lattice, **`STANDING`** | **NOT superseded by this repo.** Holds unique substrate: SP-SWARM / DHT (QUIC, Ed25519, C2 discovery), the byte-exact exact-integer forward (`SP_BYTEEXACT`), the NTT/CRT kernels, the frozen L1 C ABI |
| [shannon-prime-lattice](https://github.com/nihilistau/shannon-prime-lattice) | lattice | umbrella: papers, KEYSTONE, ADRs, SP-OKF / MEM-OKF, SWARM design |
| [shannon-prime-engine](https://github.com/nihilistau/shannon-prime-engine) | lattice, `HISTORICAL` | the FIRST reference engine — the PPT-ARM line (Friedman sieve, CRT-NTT). A different codebase: no `sp_daemon`, no kernels shared with this tree |

── DO NOT COLLAPSE THE LATTICE FAMILY INTO A COMPANION HARNESS ─────────────────────────
That is rule 2 of the project's own repo map,
[`JOURNEY.md`](https://github.com/nihilistau/Position_Is_Arithmetic/blob/main/JOURNEY.md),
and it is aimed at exactly the agent reading this. Each tree carries a `STATUS.md` tombstone
naming its class; **read it before treating a README as current.** The first draft of this
file got that wrong — it described the lattice engine as a superseded ancestor of this one.
It is neither: it is `STANDING`, it holds work this repo does not have, and the two lines are
not a succession.
