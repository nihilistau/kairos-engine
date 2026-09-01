@echo off
REM ── G-CLEAN-BUILD leg 1: math-core static libs built FROM THE KAIROS SUBMODULE ──
REM Output layout: engine\build-cpu\lib\shannon-prime-system\core\<m>\sp_<m>.lib
REM (byte-compatible with what build.rs SP_SYSTEM_BUILD_DIR expects — the same
REM shape staging's engine-root cmake produced). Toolchain mirrors staging
REM build-cpu: clang-cl + Ninja Release (env-cuda supplies vcvars64 + PATH).
setlocal
set "ENGINE=%~dp0"
call "%ENGINE%scripts\env\env-cuda.bat" || goto :err

cmake -S "%ENGINE%..\core" -B "%ENGINE%build-cpu\lib\shannon-prime-system" -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_C_COMPILER="C:/Program Files/LLVM/bin/clang-cl.exe" ^
  -DCMAKE_CXX_COMPILER="C:/Program Files/LLVM/bin/clang-cl.exe" ^
  -DSP_SYSTEM_BUILD_TESTS=OFF || goto :err
cmake --build "%ENGINE%build-cpu\lib\shannon-prime-system" --config Release || goto :err

echo CORE-CPU BUILD OK
endlocal & exit /b 0
:err
echo [build-core-cpu] FAILED
endlocal & exit /b 1
