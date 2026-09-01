@echo off
REM ---------------------------------------------------------------------
REM env-hexagon.bat -- Hexagon V69 HTP backend environment (Phase 2-HX).
REM Targets Snapdragon 8 Gen 1 (SM8450 "taro") V69 HTP, Galaxy S22 Ultra.
REM Host builds on Windows (Hexagon SDK 5.5.6.0); the engine + FastRPC stub
REM cross-compile to Android aarch64 (NDK r25c) and RUN ON THE PHONE, which
REM FastRPCs to the cDSP. ASCII only; goto-based error handling (the SDK +
REM VS paths contain "(x86)" / parens which break parenthesised if-blocks).
REM ---------------------------------------------------------------------

call "%~dp0env-common.bat"

if not exist "%SP_PIN_HEXAGON_SDK%\setup_sdk_env.cmd" goto :no_sdk
if not exist "%SP_PIN_GIT_USR_BIN%\sh.exe" goto :no_sh

set HEXAGON_SDK_ROOT=%SP_PIN_HEXAGON_SDK%
set HEXAGON_TOOLS_VER=8.7.06

REM Source the SDK environment (sets hex tools, gow, cmake-3.22, NDK, qaic).
REM setup_sdk_env.cmd has a PATH-cleanup `for %%a in (%PATH%)` loop that splits
REM unquoted spaced PATH entries (e.g. "C:\Program Files (x86)\Microsoft Visual
REM Studio") into bare tokens and tries to run them -> a flurry of "'M' is not
REM recognized" + a nonzero residual errorlevel. Both are BENIGN: every var we
REM consume (hex tools, gow make, cmake-3.22, NDK, qaic-path-via-CMake-patch) is
REM still set correctly afterward. We DON'T prepend the spaced Git\usr\bin path
REM BEFORE this call (that would add yet another spaced entry to the loop), and
REM we `ver >nul` after to reset errorlevel so a caller's
REM `call env-hexagon.bat || exit /b 1` doesn't abort on the benign noise.
REM (The full output is suppressed; "Failed to install QAIC" is also harmless --
REM the WinNT qaic.exe is present and hexagon_fun.cmake is patched to use it.)
call "%SP_PIN_HEXAGON_SDK%\setup_sdk_env.cmd" 1>nul 2>nul
ver >nul

REM Git for Windows sh.exe AFTER the SDK env: the SDK's qaic *.cmd / make steps
REM need a POSIX shell. Prepended last so the setup_sdk_env loop above never saw
REM this spaced entry.
set "PATH=%SP_PIN_GIT_USR_BIN%;%PATH%"

REM Android NDK r25c (the host-side cross toolchain for the on-phone runner).
set ANDROID_NDK_ROOT=%SP_PIN_HEXAGON_SDK%\tools\android-ndk-r25c
set ANDROID_NDK=%ANDROID_NDK_ROOT%

REM FastRPC rpcmem alloc size MUST equal the IDL length parameter EXACTLY.
REM Over-allocating returns AEE_EUNSUPPORTED (rc=0x4e) and the bridge silently
REM zero-fills -> model runs but PPL is garbage. See project_hexagon_silent_fallback.
set SP_FASTRPC_STRICT_ALLOC=1

REM freethedsp shim (opt-in via env var on device run).
set SP_FREETHEDSP=0

REM ADB: pt-latest (v36) handles Android 13+; old adbs were disabled.
set SP_ADB=D:\Files\Android\pt-latest\platform-tools\adb.exe
if not exist "%SP_ADB%" set SP_ADB=adb
set SP_ADB_SERIAL=R5CT22445JA
set SP_DEVICE_DIR=/data/local/tmp/sp22u

set SP_BACKEND=hexagon
set SP_BUILD_DIR=%SP_ENGINE%\build-hexagon
set SP_HEXAGON_TARGET=v69
set SP_HEXAGON_TOOLS_VARIANT=toolv87

echo [env-hexagon] Hexagon SDK %SP_PIN_HEXAGON_SDK% activated for target %SP_HEXAGON_TARGET% (%SP_HEXAGON_TOOLS_VARIANT%).
echo [env-hexagon] NDK=%ANDROID_NDK_ROOT%
echo [env-hexagon] SP_BUILD_DIR=%SP_BUILD_DIR%  SP_FASTRPC_STRICT_ALLOC=1  SP_FREETHEDSP=%SP_FREETHEDSP%
echo [env-hexagon] device=%SP_ADB_SERIAL%  dir=%SP_DEVICE_DIR%
goto :eof

:no_sdk
echo [env-hexagon] ERROR: Hexagon SDK not found at:
echo           %SP_PIN_HEXAGON_SDK%\setup_sdk_env.cmd
echo           Install Hexagon SDK 5.5.6.0 or correct SP_PIN_HEXAGON_SDK in env-common.bat.
exit /b 1

:no_sh
echo [env-hexagon] ERROR: Git for Windows sh.exe not found at:
echo           %SP_PIN_GIT_USR_BIN%\sh.exe
echo           The SDK's qaic/make steps need a POSIX shell. Install Git for Windows.
exit /b 1
