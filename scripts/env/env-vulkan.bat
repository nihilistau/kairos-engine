@echo off
REM ---------------------------------------------------------------------
REM env-vulkan.bat -- Vulkan compute backend environment (VS2019 BT +
REM Vulkan SDK + glslc + Ninja). ASCII only; goto-based error handling
REM (the VS path contains "(x86)", which breaks parenthesised if-blocks).
REM
REM Host reality (2026-05-23): NVIDIA RTX 2060 (Vulkan-capable); Vulkan SDK
REM 1.4.341.1 (VULKAN_SDK set globally by the installer); glslc on PATH.
REM GLSL compute shaders compile to SPIR-V at BUILD time via glslc.
REM ---------------------------------------------------------------------

call "%~dp0env-common.bat"

set "_VCVARS=%SP_PIN_VS_BUILDTOOLS%\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%_VCVARS%" goto :no_vcvars
if "%VULKAN_SDK%"=="" goto :no_sdk
if not exist "%VULKAN_SDK%\Bin\glslc.exe" goto :no_glslc

call "%_VCVARS%" >nul

set "PATH=%VULKAN_SDK%\Bin;%PATH%"
set SP_BACKEND=vulkan
set SP_BUILD_DIR=%SP_ENGINE%\build-vulkan
set SP_VULKAN_SUBGROUP_OPS=1

echo [env-vulkan] VS2019 BT + Vulkan SDK %VULKAN_SDK% activated.
echo [env-vulkan] glslc on PATH.  SP_BUILD_DIR=%SP_BUILD_DIR%
goto :eof

:no_vcvars
echo [env-vulkan] ERROR: VS2019 Build Tools vcvars64.bat not found at:
echo           %_VCVARS%
echo           Install VS2019 Build Tools or correct SP_PIN_VS_BUILDTOOLS in env-common.bat.
exit /b 1

:no_sdk
echo [env-vulkan] ERROR: VULKAN_SDK not set.
echo           Install the Vulkan SDK ^>= %SP_PIN_VULKAN_MIN% from https://vulkan.lunarg.com/ and rerun.
exit /b 1

:no_glslc
echo [env-vulkan] ERROR: glslc.exe not found at %VULKAN_SDK%\Bin
exit /b 1
