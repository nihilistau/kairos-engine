# AGENTS.md — kairos-engine

**The orientation file for anyone — human or agent — working in this repo.**

This is the optional native backend for
[Kairos](https://github.com/nihilistau/Kairos): a Rust daemon (`sp-daemon`) over hand-written
CUDA kernels, serving an HTTP/SSE surface on `:3000`. Kairos does not need it — its default
and supported path is any OpenAI-compatible endpoint — and nothing here is required to run a
companion.

> **This is a curated export.** The source of truth is a private research tree; this repo is
> the 173-file subset that actually builds the daemon, cut with fresh history and scrubbed.
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
core/                     SUBMODULE -> shannon-prime-system (the math core)
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

| repo | what |
|---|---|
| [Kairos](https://github.com/nihilistau/Kairos) | the harness, the room, the memory architecture, the gates |
| [shannon-prime-system](https://github.com/nihilistau/shannon-prime-system) | the math core — the `core/` submodule here |
| `shannon-prime-engine`, `shannon-prime-system-engine` | earlier, larger snapshots. Not maintained; this is the curated source in use |
