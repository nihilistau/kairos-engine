@echo off
REM -- WHERE IS THE MATH CORE? One answer, for all three build scripts. -----------
REM
REM 2026-09-15, found by the CI that had never run before. The core has TWO layouts:
REM
REM   source tree : a sibling directory, engine\..\core, with engine\lib\shannon-prime-system
REM                 a junction pointing at it (build-cuda-backend.bat made it)
REM   a clone     : the submodule lib/shannon-prime-system, and NO ..\core at all
REM
REM build-core-cpu.bat and build-wirecuda.bat both spelled it `..\core`, so a fresh clone
REM died in cmake on a path that tree never had. The giveaway is that the comments in all
REM three scripts ALREADY called lib\shannon-prime-system the mechanism -- the invariant was
REM written down in three places and enforced in one of them (AGENTS.md section 0: an
REM invariant enforced in one of two paths is enforced in neither).
REM
REM lib\shannon-prime-system is the name that is correct in BOTH layouts, so that is the name
REM everything uses now. This resolves it, creates the junction when only the sibling exists,
REM and when neither exists names BOTH places it looked -- because "cannot find ..\core" is
REM unreadable to the one person who matters here, someone who never had a ..\core.
REM
REM Sets SP_CORE for the caller. Deliberately no setlocal: a `call`ed .bat shares the
REM caller's environment, which is how scripts\env\env-cuda.bat already works.

set "SP_CORE="
set "_ENG=%~dp0..\..\"

REM 1. the name that works everywhere -- submodule in a clone, junction in the source tree
if exist "%_ENG%lib\shannon-prime-system\CMakeLists.txt" set "SP_CORE=%_ENG%lib\shannon-prime-system"
if defined SP_CORE goto :found

REM 2. only the sibling exists: make the junction the rest of the build expects
if not exist "%_ENG%..\core\CMakeLists.txt" goto :missing
mkdir "%_ENG%lib" 2>nul
mklink /J "%_ENG%lib\shannon-prime-system" "%_ENG%..\core" >nul
if errorlevel 1 goto :nolink
set "SP_CORE=%_ENG%lib\shannon-prime-system"
goto :found

:missing
REM %%~fI normalises the path. Printing the raw "...\scripts\env\..\..\..\core" spelling
REM makes the reader parse relative segments to work out which directory was even meant.
for %%I in ("%_ENG%lib\shannon-prime-system") do set "_P1=%%~fI"
for %%I in ("%_ENG%..\core") do set "_P2=%%~fI"
echo [resolve-core] CANNOT FIND THE MATH CORE. Both places were checked:
echo [resolve-core]   %_P1%
echo [resolve-core]       ^-- the submodule. This is the layout a clone has.
echo [resolve-core]   %_P2%
echo [resolve-core]       ^-- the sibling directory. Only the source tree has this.
set "_P1="
set "_P2="
echo [resolve-core].
echo [resolve-core] In a clone this means the submodule was never fetched. Run:
echo [resolve-core]   git submodule update --init --recursive
set "_ENG="
exit /b 1

:nolink
echo [resolve-core] mklink could not create the junction:
echo [resolve-core]   %_ENG%lib\shannon-prime-system  ^-^>  %_ENG%..\core
echo [resolve-core] A junction needs an elevated shell or Developer Mode enabled.
set "_ENG="
exit /b 1

:found
set "_ENG="
exit /b 0
