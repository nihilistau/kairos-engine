@echo off
REM THE RELEASE BUILD (2026-11-26): the same sp-daemon as build-wirecuda.bat, built into its own
REM target dir (target-release, so the binary her profile runs is never touched) with every
REM build-machine path remapped out of the binary. A plain build bakes in the builder's paths
REM (panic locations name %USERPROFILE%\.cargo\registry\... and the engine checkout), and a
REM release binary is published. Check with: strings sp-daemon.exe ^| findstr /i "%USERNAME%"
REM Paths containing spaces cannot be remapped through RUSTFLAGS; build from a path without them.
REM Then package-release.bat collects the zip.
setlocal
call "%~dp0scripts\env\resolve-core.bat" || goto :err
call "%~dp0scripts\env\env-cuda.bat" || goto :err

set "SP_SYSTEM_INCLUDE=%SP_CORE%\include"
set "SP_SYSTEM_BUILD_DIR=%~dp0build-cpu\lib\shannon-prime-system"
set "SP_CUDA_BACKEND_DIR=%~dp0build-host-cuda-backend"

set "_ENG=%~dp0"
set "_ENG=%_ENG:~0,-1%"
for %%I in ("%SP_CORE%") do set "_CORE=%%~fI"
set "RUSTFLAGS=--remap-path-prefix=%USERPROFILE%\.cargo=cargo --remap-path-prefix=%USERPROFILE%\.rustup=rustup --remap-path-prefix=%_ENG%=engine --remap-path-prefix=%_CORE%=core"

cd /d "%~dp0tools\sp_daemon"
cargo build --release --features wire_cuda_backend --target-dir target-release --bin sp-daemon || goto :err
echo RELEASE BUILD OK: %~dp0tools\sp_daemon\target-release\release\sp-daemon.exe
endlocal & exit /b 0
:err
echo [build-release] FAILED
endlocal & exit /b 1
