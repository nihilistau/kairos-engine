# sp-daemon — Shannon Prime L3 HTTP/SSE Daemon

Phase 2-L3.CORE scaffold. Wraps the frozen L1 C ABI (`libshannonprime`) in a
long-lived HTTP server on `127.0.0.1:8080`. The canonical UX boundary for all
four frontends (mobile, desktop, watch, CLI).

**Repo location chosen:** `shannon-prime-system-engine/tools/sp_daemon/` alongside
`sp_transcode`, because build.rs uses direct relative paths to the engine's
math-core headers and build artefacts without cross-repo discovery.

## Build

Two env vars required at build time:

| Variable | Purpose | Example |
|---|---|---|
| `SP_SYSTEM_INCLUDE` | math-core include dir | `../../lib/shannon-prime-system/include` |
| `SP_SYSTEM_BUILD_DIR` | math-core build dir root (see below) | see below |
| `LIBCLANG_PATH` | libclang for bindgen (Windows) | `C:\Program Files\LLVM\bin` |

### SP_SYSTEM_BUILD_DIR layouts

**Engine-embedded (Windows, default):**
```
SP_SYSTEM_BUILD_DIR=<engine-root>/build-cpu/lib/shannon-prime-system
```
Libs at `{dir}/core/{module}/sp_{module}.lib`

**Standalone math-core (Linux / MinGW):**
```
SP_SYSTEM_BUILD_DIR=/path/to/shannon-prime-system/build
```
Libs at `{dir}/core/{module}/libsp_{module}.a`

### Quick build (Windows MSVC, relative from this directory)

```bat
set SP_SYSTEM_BUILD_DIR=..\..\build-cpu\lib\shannon-prime-system
set SP_SYSTEM_INCLUDE=..\..\lib\shannon-prime-system\include
set LIBCLANG_PATH=C:\Program Files\LLVM\bin
cargo build --release
```

### Quick build (Linux gcc)

```sh
export SP_SYSTEM_BUILD_DIR=/path/to/shannon-prime-system/build
export SP_SYSTEM_INCLUDE=/path/to/shannon-prime-system/include
cargo build --release
```

## Usage

```
sp-daemon start --model <path>.spm --tokenizer <path>.spt
sp-daemon stop
sp-daemon reload   # no-op v0
```

Env var shortcuts: `SP_MODEL_PATH`, `SP_TOKENIZER_PATH`.

**Parity fixture** (math-core repo root):
```
sp-daemon start \
  --model    /path/to/shannon-prime-system/fx_q4.spm \
  --tokenizer /path/to/shannon-prime-system/fx_q4.spt
```

## Routes (CORE)

| Route | Method | Response |
|---|---|---|
| `/v1/metrics` | GET | JSON with `tokens_per_sec`, `ram_svm_bytes`, `peers`, `phase`, `session_pos` |

`session_pos` is the live `sp_session_position` read — proof the FFI handle works.
Placeholder values for `tokens_per_sec`/`ram_svm_bytes`/`peers` land in VERBS.

## Security

The daemon binds **only** to `127.0.0.1:8080`. Binding to `0.0.0.0` is
explicitly refused (single-user developer-device assumption,
`PPT-LAT-Roadmap §14.3.1`). LAN exposure and TLS are v1+ scope.

## aarch64-android

Cross-compiling for `aarch64-linux-android` type-checks but skips the link
step (`cargo:rustc-cfg=sp_no_link`) because no Android-built math-core libs
exist yet. Real device link is Phase 2-L3.FG scope.

## Phase map

| Sub-phase | Status |
|---|---|
| CORE (this crate) | ✓ closed |
| VERBS | subsequent session |
| SSE | subsequent session |
| FG (foreground service) | subsequent session |
| AUTH (bearer token) | subsequent session |
