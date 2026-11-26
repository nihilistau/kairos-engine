# sp_transcode — GGUF to `.sp` for the native engine

Converts a GGUF model into the two files sp-daemon loads: `<name>.sp-model` (the weights,
re-quantised) and `<name>.sp-tokenizer`. You need it only for the native engine. llama.cpp
reads GGUF directly.

## Build (Windows)

```
engine\build-transcode.bat
```

This uses the same toolchain and math core as the daemon's core build (`build-core-cpu.bat`):
clang-cl with Ninja in Release mode, `env-cuda.bat` for vcvars, and `resolve-core.bat` for
the core. It builds only the core libraries the tool uses. Output goes to
`engine\tools\sp_transcode\build\sp_transcode.exe`, which is where the setup centre
(`harness/setup/installer.py`) looks after `SP_TRANSCODE_EXE` and before `PATH`.

On another platform, run `cmake -S engine/tools/sp_transcode -B <build> -DSP_CORE=<core checkout>`
and then `cmake --build <build> --target sp_transcode`.

## Use

```
sp_transcode <in.gguf> <out.sp-model> <out.sp-tokenizer> --q4b --stream --verify
sp_transcode --tok-only <in.gguf> <out.sp-tokenizer>
```

| flag | what it does |
|---|---|
| `--q4b` | matmul weights become OK_Q4B (per-32 f16 block scales). `token_embd` stays OK_Q8 because the CUDA embed gather has no block-scale path. This is the codec gemma4-MoE runs on. |
| `--q4b-ffn` | OK_Q4B on ffn gate/up only, OK_Q8 everywhere else (recipe B1) |
| `--q4` / `--q8` | all OK_Q4 / all OK_Q8 |
| `--stream` | streams the data region through a temp file, so peak memory is about one tensor. A 26B needs this on a 32 GB machine. The output is the same bytes. |
| `--verify` | after writing, checks that each weight sits next to its `.scale` tensor and reloads the file through `sp_model_load` |
| `--st <file>` | takes the weights from a safetensors file instead |

## Provenance

The tool was brought across on 2026-11-26 from the retired C engine
(`shannon-prime-system-engine/tools/sp_transcode`, source dated 2026-09-02). That engine
linked it against its whole `sp_engine` library. Every symbol the tool uses (GGUF parse,
dequant, Frobenius pack, the `.sp-model` loader, the hashes) lives in the math core, so this
build links the core alone.

The check: the gemma4-26B-A4B HauhauCS GGUF transcoded with `--q4b --stream` gives the model
she runs, byte for byte in all 14,557,496,192 bytes except 16 header bytes. Those are the
creation timestamp (`created_unix_seconds`) and the header checksum computed over it. The
tokenizer is byte-identical.
