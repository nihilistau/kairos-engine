@echo off
REM ============================================================================
REM  Sprint WIRE-CPU -- build the daemon-linkable CPU AVX-512 backend static lib
REM  for the HOST target. Consumed by sp_daemon's build.rs when the
REM  `wire_cpu_backend` Cargo feature is set.
REM
REM  Output: <engine>\build-host-cpu-backend\Release\sp_cpu_daemon_backend.lib
REM
REM  Companion to scripts/build/build-cpu.bat (which builds the engine's own
REM  sp_engine.lib + math-core .lib archives). Run AFTER build-cpu.bat so the
REM  math-core archives the daemon links transitively are present.
REM
REM  Required env (from scripts\env\env-cpu.bat):
REM    VS2019 Build Tools x64 host/target activated
REM    cmake + ninja on PATH (or fall back to NMake)
REM ============================================================================
setlocal

REM Engine root = two levels up from tools/sp_daemon/
set "ENGINE=%~dp0..\.."
pushd "%ENGINE%" >nul
for %%I in (.) do set "ENGINE_ABS=%%~fI"
popd >nul
set "ENGINE=%ENGINE_ABS%"

REM Verify the CPU backend source files exist (Phase 2-CPU.AVX deliverables).
if not exist "%ENGINE%\src\backends\cpu\cpu_overlay.c" (
    echo ERROR: cpu_overlay.c not found at %ENGINE%\src\backends\cpu\
    exit /b 1
)
if not exist "%ENGINE%\src\backends\cpu\avx512\avx512_vnni.c" (
    echo ERROR: AVX-512 sources not found at %ENGINE%\src\backends\cpu\avx512\
    exit /b 1
)

REM Activate VS2019 BT if not already (idempotent; env-cpu.bat handles re-entry).
if "%VCToolsVersion%"=="" (
    call "%ENGINE%\scripts\env\env-cpu.bat" || exit /b 1
)

set "BUILD_DIR=%ENGINE%\build-host-cpu-backend"

REM Prefer Ninja (parallel + fast). NMake fallback if missing.
where ninja >nul 2>&1
if errorlevel 1 (
    set "SP_GEN=NMake Makefiles"
) else (
    set "SP_GEN=Ninja"
)

cmake -S "%ENGINE%\tools\sp_daemon\c_backend_cpu" -B "%BUILD_DIR%" -G "%SP_GEN%" ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DSP_CPU_DAEMON_WITH_AVX2=ON ^
  -DSP_CPU_DAEMON_WITH_AVX512=ON ^
  || exit /b 1

cmake --build "%BUILD_DIR%" --config Release || exit /b 1

echo.
echo Built CPU daemon backend static lib:
if exist "%BUILD_DIR%\Release\sp_cpu_daemon_backend.lib" (
    dir /b "%BUILD_DIR%\Release\sp_cpu_daemon_backend.lib"
) else if exist "%BUILD_DIR%\sp_cpu_daemon_backend.lib" (
    dir /b "%BUILD_DIR%\sp_cpu_daemon_backend.lib"
) else (
    dir /b "%BUILD_DIR%\libsp_cpu_daemon_backend.a" 2>nul
)

endlocal
