@echo off
REM ── G-CLEAN-BUILD leg 2: the CUDA backend static lib built FROM KAIROS ──────
REM Sources: engine\src\backends\cuda\*.cu (migrated), c_backend_cuda glue,
REM core\ submodule xbar_episode.c (via the engine\lib\shannon-prime-system
REM junction -> ..\core, created by this script if missing — the CMakeLists
REM ENGINE_ROOT layout is preserved verbatim).
setlocal
set "ENGINE=%~dp0"
if not exist "%ENGINE%lib\shannon-prime-system" (
  mkdir "%ENGINE%lib" 2>nul
  mklink /J "%ENGINE%lib\shannon-prime-system" "%ENGINE%..\core" || goto :err
)
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
