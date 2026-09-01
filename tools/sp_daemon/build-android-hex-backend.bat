@echo off
REM ============================================================================
REM  Sprint WIRE-HEX — build the daemon-linkable Hexagon backend static lib
REM  for aarch64-linux-android. Consumed by sp_daemon's build.rs when
REM  SP_DAEMON_LINK_HEX=1 is set.
REM
REM  Output: <engine>/build-android-hex-backend/libsp_hex_daemon_backend.a
REM
REM  Companion to build-android-libs.bat (math-core libs). Run AFTER the
REM  math-core libs are built so the daemon can link both layers.
REM
REM  Required env:
REM    HEXAGON_SDK_ROOT  pinned to C:\Qualcomm\Hexagon_SDK\5.5.6.0 by
REM                      scripts\env\env-hexagon.bat
REM    NDK               android-ndk-r27d at D:\Files\Android\android-ndk-r27d
REM ============================================================================
setlocal

set "NDK=D:\Files\Android\android-ndk-r27d"
set "TOOLCHAIN=%NDK%\build\cmake\android.toolchain.cmake"
REM engine root = two levels up from tools/sp_daemon/
set "ENGINE=%~dp0..\.."

if not exist "%TOOLCHAIN%" (
    echo ERROR: NDK CMake toolchain not found at %TOOLCHAIN%
    exit /b 1
)

if "%HEXAGON_SDK_ROOT%"=="" (
    REM Fall back to the engine's pinned path (env-common.bat).
    set "HEXAGON_SDK_ROOT=C:\Qualcomm\Hexagon_SDK\5.5.6.0"
)
if not exist "%HEXAGON_SDK_ROOT%\ipc\fastrpc\qaic\WinNT\qaic.exe" (
    echo ERROR: HEXAGON_SDK_ROOT does not contain qaic.exe: %HEXAGON_SDK_ROOT%
    exit /b 1
)

cmake -S "%ENGINE%\tools\sp_daemon\c_backend" -B "%ENGINE%\build-android-hex-backend" -G Ninja ^
  -DCMAKE_TOOLCHAIN_FILE="%TOOLCHAIN%" ^
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-21 ^
  -DCMAKE_BUILD_TYPE=Release || exit /b 1

cmake --build "%ENGINE%\build-android-hex-backend" --config Release || exit /b 1

echo.
echo Built hex backend static lib:
dir /b "%ENGINE%\build-android-hex-backend\libsp_hex_daemon_backend.a"

endlocal
