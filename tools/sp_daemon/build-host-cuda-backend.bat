@echo off
REM ============================================================================
REM  Sprint WIRE-CUDA -- build the daemon-linkable CUDA backend static lib
REM  for host (x86_64 Windows MSVC). Consumed by sp_daemon's build.rs when
REM  CARGO_FEATURE_WIRE_CUDA_BACKEND=1 (via --features wire_cuda_backend).
REM
REM  Output: <engine>/build-host-cuda-backend/sp_cuda_daemon_backend.lib
REM
REM  Companion to build-android-hex-backend.bat. Symmetric WIRE-HEX shape.
REM
REM  Required env (sourced by env-cuda.bat):
REM    SP_PIN_CUDA_ROOT  C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2
REM    SP_PIN_VS_BUILDTOOLS  VS2019 BT path (vcvars64.bat)
REM    SP_CUDA_ARCH      75 (RTX 2060 = Turing); set via env-cuda.bat
REM
REM  --use-local-env is mandatory on VS2019 BuildTools (nvcc's internal
REM  vcvars detection fails without a full VS install; the flag makes nvcc
REM  inherit the parent vcvars64 shell env).
REM ============================================================================
setlocal

REM engine root = two levels up from tools/sp_daemon/
set "ENGINE=%~dp0..\.."

REM Source the CUDA env (vcvars64 + CUDA on PATH + SP_CUDA_ARCH=75).
call "%ENGINE%\scripts\env\env-cuda.bat" || goto :err

set "SRC_DIR=%ENGINE%\tools\sp_daemon\c_backend_cuda"
set "BUILD_DIR=%ENGINE%\build-host-cuda-backend"

cmake -S "%SRC_DIR%" -B "%BUILD_DIR%" -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_C_COMPILER=cl ^
  -DCMAKE_CUDA_COMPILER="%SP_PIN_CUDA_ROOT%/bin/nvcc.exe" ^
  -DCMAKE_CUDA_ARCHITECTURES="%SP_CUDA_ARCH%" ^
  -DCMAKE_CUDA_FLAGS="--use-local-env" || goto :err

cmake --build "%BUILD_DIR%" --config Release || goto :err

echo.
echo Built CUDA backend static lib:
dir /b "%BUILD_DIR%\sp_cuda_daemon_backend.lib" 2>nul
dir /b "%BUILD_DIR%\libsp_cuda_daemon_backend.a" 2>nul

endlocal
endlocal & exit /b 0

:err
echo [build-host-cuda-backend] FAILED
endlocal
exit /b 1
