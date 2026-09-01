@echo off
REM ============================================================================
REM  Sprint WIRE-VULKAN -- build the daemon-linkable Vulkan backend static lib
REM  for the host toolchain (Windows VS2019 + Vulkan SDK). Consumed by
REM  sp_daemon's build.rs when --features wire_vulkan_backend is on.
REM
REM  Output: <engine>/build-host-vulkan-backend/Release/sp_vulkan_daemon_backend.lib
REM           (or libsp_vulkan_daemon_backend.a on MinGW/Linux variants)
REM
REM  Companion to scripts/build/build-vulkan.bat (full engine + tests). This
REM  script builds ONLY the daemon-link static lib via the c_backend
REM  CMakeLists's SP_DAEMON_BUILD_VULKAN_BACKEND branch.
REM
REM  Required env (set by scripts/env/env-vulkan.bat):
REM    VS2019 Build Tools vcvars64 (compiler)
REM    VULKAN_SDK pinned (glslc + Vulkan loader + headers)
REM ============================================================================
setlocal

REM engine root = two levels up from tools/sp_daemon/
set "ENGINE=%~dp0..\.."

REM Activate VS + Vulkan environment (vcvars64 + VULKAN_SDK + glslc on PATH).
call "%ENGINE%\scripts\env\env-vulkan.bat" || exit /b 1

if "%VULKAN_SDK%"=="" (
    echo ERROR: VULKAN_SDK not set after env-vulkan.bat
    exit /b 1
)
if not exist "%VULKAN_SDK%\Bin\glslc.exe" (
    echo ERROR: glslc.exe not found at %VULKAN_SDK%\Bin
    exit /b 1
)

set "BUILD_DIR=%ENGINE%\build-host-vulkan-backend"

REM SP_DAEMON_BUILD_HEX_BACKEND=OFF skips the Hexagon SDK probe at the top of
REM tools/sp_daemon/c_backend/CMakeLists.txt — the host Vulkan-only build path
REM doesn't need HEXAGON_SDK_ROOT (cross-compile to aarch64-android is a
REM different toolchain via build-android-hex-backend.bat).
cmake -S "%ENGINE%\tools\sp_daemon\c_backend" -B "%BUILD_DIR%" -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DSP_DAEMON_BUILD_HEX_BACKEND=OFF ^
  -DSP_DAEMON_BUILD_VULKAN_BACKEND=ON || exit /b 1

cmake --build "%BUILD_DIR%" --config Release || exit /b 1

echo.
echo Built Vulkan backend static lib:
dir /b "%BUILD_DIR%\sp_vulkan_daemon_backend.lib" 2>nul
dir /b "%BUILD_DIR%\libsp_vulkan_daemon_backend.a" 2>nul

endlocal
