@echo off
REM ---------------------------------------------------------------------
REM shannon-prime-system-engine  env-common.bat
REM Shared environment for all backends. Sourced by env-cpu / env-cuda /
REM env-vulkan / env-hexagon.bat. EDIT WITH CAUTION -- paths are pinned.
REM (ASCII only, no parenthesised if-blocks: paths contain "(x86)".)
REM ---------------------------------------------------------------------

REM Repo roots
set SP_REPO_ROOT=<path-to>\shannon-prime-repos
set SP_LATTICE=%SP_REPO_ROOT%\shannon-prime-lattice
set SP_SYSTEM=%SP_REPO_ROOT%\shannon-prime-system
set SP_ENGINE=%SP_REPO_ROOT%\shannon-prime-system-engine

REM Toolchain pins. Any change here is a project decision, not a session decision.
REM CUDA 13.2 pin (2026-05-22, 2-CU): dev host has 13.2 on PATH + RTX 2060 (sm_75);
REM roadmap 8.3 amended to 13.2 + sm_75. The old 12.4/sm_86-89 line is retired.
REM
REM CORRECTION-OF-CORRECTION 2026-06-02: per docs/BUILD-ENV.md, the canonical CPU
REM backend is MinGW gcc 15.2 (the `build/` dir), NOT MSVC; MSVC cannot build the
REM CPU backend (GCC __attribute__((target))/__atomic_/<stdatomic.h>) and that is a
REM KNOWN Tier-3-deferred fact. SP_PIN_VS_BUILDTOOLS is the CUDA HOST compiler pin
REM only -> keep it at VS2019 BT (CUDA tightly pinned per BUILD-ENV). A prior step
REM this session wrongly pointed it at VS18 while chasing an MSVC CPU build; reverted.
set SP_PIN_VS_BUILDTOOLS=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools
REM Tier-3 MSVC-parity toolchain (separate; NOT the CUDA host). VS18 BuildTools on D:
REM (MSVC v14.50, cl 19.50, ships <stdatomic.h>). Used only for the tracked de-GCC
REM MSVC-parity build, never for CPU(=MinGW) or CUDA(=VS2019) production.
set SP_PIN_VS2022_BUILDTOOLS=D:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools
set SP_PIN_CUDA_VERSION=13.2
set SP_PIN_CUDA_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v%SP_PIN_CUDA_VERSION%
set SP_PIN_VULKAN_MIN=1.3.250
set SP_PIN_HEXAGON_SDK=C:\Qualcomm\Hexagon_SDK\5.5.6.0
set SP_PIN_NINJA_MIN=1.10
set SP_PIN_CMAKE_MIN=3.20
set SP_PIN_GIT_USR_BIN=C:\Program Files\Git\usr\bin

REM Build directory naming convention (per backend, parallel coexistence):
REM   build-cpu  build-cpu-dbg  build-cuda  build-vulkan  build-hexagon
set SP_BUILD_TYPE_DEFAULT=Release
set SP_GENERATOR=Ninja

where cmake >nul 2>&1 || echo [env-common] WARNING: cmake not on PATH (need ^>= %SP_PIN_CMAKE_MIN%).
where ninja >nul 2>&1 || echo [env-common] WARNING: ninja not on PATH (need ^>= %SP_PIN_NINJA_MIN%).

echo [env-common] paths pinned: SP_REPO_ROOT=%SP_REPO_ROOT%
echo [env-common] toolchain pins: CPU=MinGW-gcc-15.2 (build/), CUDA-host=VS2019 BT + CUDA %SP_PIN_CUDA_VERSION%, Tier3-MSVC-parity=VS18 (D:), Vulkan ^>= %SP_PIN_VULKAN_MIN%, Hexagon SDK %SP_PIN_HEXAGON_SDK%
