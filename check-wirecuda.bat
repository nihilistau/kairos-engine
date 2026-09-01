@echo off
REM cargo check twin of build-wirecuda.bat (2026-08-29): same env, same feature,
REM no exe link — safe to run while the daemon's exe is locked. Used to verify
REM engine edits compile before the one stop-build-boot at the end of a session.
call "%~dp0scripts\env\env-cuda.bat"
if errorlevel 1 exit /b 1
set "SP_SYSTEM_INCLUDE=%~dp0..\core\include"
set "SP_SYSTEM_BUILD_DIR=%~dp0build-cpu\lib\shannon-prime-system"
set "SP_CUDA_BACKEND_DIR=%~dp0build-host-cuda-backend"
cd /d "%~dp0tools\sp_daemon"
cargo check --release --features wire_cuda_backend --target-dir target-wirecuda --bin sp-daemon
echo EXITCODE=%ERRORLEVEL%
exit /b %ERRORLEVEL%
