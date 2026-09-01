@echo off
REM ── kairos engine build (G-KAIROS-P1a → G-CLEAN-BUILD) ─────────────────────
REM Builds sp-daemon.exe with the wire_cuda_backend feature (RTX 2060 sm_75).
REM
REM Layout mirrors the staging engine INSIDE engine/ so build.rs runs UNPATCHED
REM (engine_root = manifest/../.. = this directory): tools/sp_daemon (crate),
REM tools/sp_swarm (path dep), include/sp_engine (C headers), src/tokenizer,
REM src/backends/cuda (kernels), lib/shannon-prime-system (junction -> ..\core).
REM
REM G-CLEAN-BUILD (2026-07-11): ALL staging artifact tethers CUT —
REM   headers   core\include                            (submodule)
REM   math-core engine\build-cpu\...                    (build-core-cpu.bat)
REM   CUDA lib  engine\build-host-cuda-backend\...      (build-cuda-backend.bat)
REM   env pins  engine\scripts\env\env-cuda.bat         (migrated copy)
REM Prereq: run build-core-cpu.bat + build-cuda-backend.bat once first.
call "%~dp0scripts\env\env-cuda.bat"
if errorlevel 1 exit /b 1

set "SP_SYSTEM_INCLUDE=%~dp0..\core\include"
set "SP_SYSTEM_BUILD_DIR=%~dp0build-cpu\lib\shannon-prime-system"
set "SP_CUDA_BACKEND_DIR=%~dp0build-host-cuda-backend"

cd /d "%~dp0tools\sp_daemon"
cargo build --release --features wire_cuda_backend --target-dir target-wirecuda --bin sp-daemon
echo EXITCODE=%ERRORLEVEL%
exit /b %ERRORLEVEL%
