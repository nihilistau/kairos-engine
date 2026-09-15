@echo off
REM ── G-CLEAN-BUILD leg 2: the CUDA backend static lib built FROM KAIROS ──────
REM Sources: engine\src\backends\cuda\*.cu (migrated), c_backend_cuda glue,
REM math-core xbar_episode.c (via engine\lib\shannon-prime-system, resolved by
REM scripts\env\resolve-core.bat: the submodule itself in a clone, a junction to
REM ..\core in the source tree — the CMakeLists ENGINE_ROOT layout is preserved
REM verbatim either way).
setlocal
set "ENGINE=%~dp0"
REM This script used to inline the junction itself. It was the ONLY one of the three that
REM happened to work in a clone -- its `if not exist` skipped the mklink because the
REM submodule was already there -- and it was right by luck, not by rule. Same call as the
REM other two now, so there is one answer to "where is the core" (2026-09-15).
call "%ENGINE%scripts\env\resolve-core.bat" || goto :err
call "%ENGINE%scripts\env\env-cuda.bat" || goto :err

set "SRC_DIR=%ENGINE%tools\sp_daemon\c_backend_cuda"
set "BUILD_DIR=%ENGINE%build-host-cuda-backend"

cmake -S "%SRC_DIR%" -B "%BUILD_DIR%" -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_C_COMPILER=cl ^
  -DCMAKE_CUDA_COMPILER="%SP_PIN_CUDA_ROOT%/bin/nvcc.exe" ^
  -DCMAKE_CUDA_ARCHITECTURES="%SP_CUDA_ARCH%" ^
  -DCMAKE_CUDA_FLAGS="--use-local-env" || goto :err
cmake --build "%BUILD_DIR%" --config Release || goto :err

echo CUDA BACKEND BUILD OK
dir /b "%BUILD_DIR%\sp_cuda_daemon_backend.lib" 2>nul
endlocal & exit /b 0
:err
echo [build-cuda-backend] FAILED
endlocal & exit /b 1
