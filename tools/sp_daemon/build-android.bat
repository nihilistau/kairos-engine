@echo off
REM ============================================================================
REM  §3-HX Sprint J.5 — sp_daemon aarch64-linux-android cross-compile helper.
REM
REM  cargo discovers .cargo/config.toml by walking UP from the working dir, so
REM  this script must run cargo with tools/sp_daemon as the CWD (it cd's here).
REM  The [env] block in .cargo/config.toml supplies CC/CXX/AR for the cc-rs
REM  deps (ring, esaxx-rs); the exports below are belt-and-suspenders so a build
REM  launched from a different CWD still resolves the toolchain.
REM
REM  NDK: android-ndk-r27d.  No bindgen sysroot needed (build.rs skips bindgen
REM  on android).  Usage:  build-android.bat            (release sp-daemon)
REM                        build-android.bat --bin foo  (extra cargo args pass through)
REM ============================================================================
setlocal

set "NDK_BIN=D:\Files\Android\android-ndk-r27d\toolchains\llvm\prebuilt\windows-x86_64\bin"
set "CC_aarch64_linux_android=%NDK_BIN%\aarch64-linux-android21-clang.cmd"
set "CXX_aarch64_linux_android=%NDK_BIN%\aarch64-linux-android21-clang++.cmd"
set "AR_aarch64_linux_android=%NDK_BIN%\llvm-ar.exe"

if not exist "%CC_aarch64_linux_android%" (
    echo ERROR: NDK clang not found at %CC_aarch64_linux_android%
    echo Set NDK_BIN in this script to your android-ndk-r27d toolchain bin dir.
    exit /b 1
)

cd /d "%~dp0"

if "%~1"=="" (
    cargo build --target aarch64-linux-android --release --bin sp-daemon
) else (
    cargo build --target aarch64-linux-android --release %*
)

endlocal
