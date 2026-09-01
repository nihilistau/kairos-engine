@echo off
REM ---------------------------------------------------------------------
REM env-cpu.bat -- CPU backend environment (clang-cl MSVC-ABI + Ninja).
REM Uses the VS2019 Build Tools x64 headers/libs (via vcvars64) but the COMPILER
REM is clang-cl, NOT cl.exe -- the math-core uses __int128 / C11 stdatomic / GCC-isms
REM that cl.exe cannot compile (C4235 "__int128 not supported"). ASCII only; goto-based
REM error handling (the VS path contains "(x86)", which breaks if-blocks).
REM ---------------------------------------------------------------------

call "%~dp0env-common.bat"

set "_VCVARS=%SP_PIN_VS_BUILDTOOLS%\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%_VCVARS%" goto :no_vcvars

call "%_VCVARS%" >nul

REM Compiler = clang-cl (MSVC ABI, links into the windows-msvc cargo daemon) -- NOT cl.exe.
REM The math-core uses __int128 (core/exact_islands/exact_islands.c), C11 <stdatomic.h>
REM and other GCC/Clang constructs that MSVC cl.exe CANNOT compile (error C4235
REM "'__int128' keyword not supported on this architecture"). clang-cl supports them AND
REM emits MSVC-ABI .lib archives the daemon links. PROVEN 2026-06-28: clang-cl
REM -fsyntax-only exact_islands.c == OK ; cl.exe == C4235/C2059. *** DO NOT revert this to
REM cl.exe -- that is the recurring drift that breaks clean-build. *** vcvars (above) still
REM provides the MSVC headers/libs that clang-cl consumes via INCLUDE/LIB. See
REM papers/BUILD-ENV-TOOLCHAIN.md (lattice) + okf 'clean-build RED on MSVC'.
set "PATH=C:\Program Files\LLVM\bin;%PATH%"
set "CC=clang-cl"
set "CXX=clang-cl"

REM AVX feature flags. The engine has explicit AVX2 and AVX512 code paths
REM gated by CMake options; /arch:AVX2 is the floor and AVX512 is emitted on
REM specific TUs (see src/backends/cpu/CMakeLists.txt).
set SP_CPU_AVX2=1
set SP_CPU_AVX512=1

set SP_BACKEND=cpu
set SP_BUILD_DIR=%SP_ENGINE%\build-cpu

echo [env-cpu] clang-cl (MSVC-ABI) activated via VS2019 %VCToolsVersion% env.  SP_BACKEND=cpu  SP_BUILD_DIR=%SP_BUILD_DIR%
goto :eof

:no_vcvars
echo [env-cpu] ERROR: VS2019 Build Tools vcvars64.bat not found at:
echo           %_VCVARS%
echo           Install VS2019 Build Tools or correct SP_PIN_VS_BUILDTOOLS in env-common.bat.
exit /b 1
