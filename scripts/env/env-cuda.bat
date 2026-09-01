@echo off
REM ---------------------------------------------------------------------
REM env-cuda.bat -- CUDA backend environment (VS2019 BT + CUDA 13.2 + Ninja).
REM Pinned toolchain (CUDA version in env-common.bat). ASCII only; goto-based
REM error handling (the VS path contains "(x86)", which breaks if-blocks).
REM
REM Host reality (2026-05-22): RTX 2060 = sm_75 (Turing); CUDA 13.2 on PATH;
REM VS2019 BuildTools only (no full VS install). nvcc's registry-based VS
REM lookup fails with BuildTools-only, so the build passes --use-local-env
REM (see build-cuda.bat) to make nvcc inherit this vcvars64 shell env.
REM ---------------------------------------------------------------------

call "%~dp0env-common.bat"

set "_VCVARS=%SP_PIN_VS_BUILDTOOLS%\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%_VCVARS%" goto :no_vcvars
if not exist "%SP_PIN_CUDA_ROOT%\bin\nvcc.exe" goto :no_cuda

call "%_VCVARS%" >nul

set "CUDA_PATH=%SP_PIN_CUDA_ROOT%"
set "CUDAToolkit_ROOT=%SP_PIN_CUDA_ROOT%"
set "PATH=%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%PATH%"

REM Target SM architecture. RTX 2060 = sm_75 (Turing). Older SMs unsupported.
set SP_CUDA_ARCH=75

set SP_BACKEND=cuda
set SP_BUILD_DIR=%SP_ENGINE%\build-cuda

echo [env-cuda] VS2019 BT + CUDA %SP_PIN_CUDA_VERSION% activated.  SP_CUDA_ARCH=%SP_CUDA_ARCH%  SP_BUILD_DIR=%SP_BUILD_DIR%
goto :eof

:no_vcvars
echo [env-cuda] ERROR: VS2019 Build Tools vcvars64.bat not found at:
echo           %_VCVARS%
echo           Install VS2019 Build Tools or correct SP_PIN_VS_BUILDTOOLS in env-common.bat.
exit /b 1

:no_cuda
echo [env-cuda] ERROR: CUDA %SP_PIN_CUDA_VERSION% nvcc not found at:
echo           %SP_PIN_CUDA_ROOT%\bin\nvcc.exe
echo           Install CUDA Toolkit %SP_PIN_CUDA_VERSION% or correct SP_PIN_CUDA_VERSION in env-common.bat.
exit /b 1
