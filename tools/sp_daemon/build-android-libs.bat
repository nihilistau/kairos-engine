@echo off
REM ============================================================================
REM  Phase 2-L3.FG-CROSS-COMPILE — build the 17 math-core static libs for
REM  aarch64-linux-android, consumed by sp_daemon's android link step
REM  (build.rs, SP_SYSTEM_BUILD_DIR=build-android-libs when sp_no_link is OFF).
REM
REM  The math-core core/ is x86-free by design (forward_kernels.c:3), so this is
REM  plain NDK targeting via the android.toolchain.cmake — no NEON port. Output:
REM    <engine>/build-android-libs/core/<module>/libsp_<module>.a   (17 archives)
REM
REM  Run from anywhere; paths are resolved relative to the engine root (this
REM  script lives in tools/sp_daemon/). NDK: android-ndk-r27d, API 21, arm64-v8a.
REM ============================================================================
setlocal

set "NDK=D:\Files\Android\android-ndk-r27d"
set "TOOLCHAIN=%NDK%\build\cmake\android.toolchain.cmake"
REM engine root = two levels up from tools/sp_daemon/
set "ENGINE=%~dp0..\.."

if not exist "%TOOLCHAIN%" (
    echo ERROR: NDK CMake toolchain not found at %TOOLCHAIN%
    echo Set NDK in this script to your android-ndk-r27d install.
    exit /b 1
)

cmake -S "%ENGINE%\lib\shannon-prime-system" -B "%ENGINE%\build-android-libs" -G Ninja ^
  -DCMAKE_TOOLCHAIN_FILE="%TOOLCHAIN%" ^
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-21 ^
  -DSP_SYSTEM_BUILD_TESTS=OFF -DSP_UBSAN=OFF -DCMAKE_BUILD_TYPE=Release || exit /b 1

cmake --build "%ENGINE%\build-android-libs" --config Release || exit /b 1

echo.
echo Built math-core libs for aarch64-linux-android:
dir /b "%ENGINE%\build-android-libs\core\*\libsp_*.a"

endlocal
