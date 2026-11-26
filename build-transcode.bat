@echo off
REM sp_transcode — GGUF -> .sp, built standalone against the math core (2026-11-26).
REM Output: engine\tools\sp_transcode\build\sp_transcode.exe, where the setup centre looks.
REM Toolchain is build-core-cpu.bat's: clang-cl + Ninja Release (env-cuda supplies vcvars64).
setlocal
set "ENGINE=%~dp0"
call "%ENGINE%scripts\env\resolve-core.bat" || goto :err
call "%ENGINE%scripts\env\env-cuda.bat" || goto :err

cmake -S "%ENGINE%tools\sp_transcode" -B "%ENGINE%tools\sp_transcode\build" -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_C_COMPILER="C:/Program Files/LLVM/bin/clang-cl.exe" ^
  -DCMAKE_CXX_COMPILER="C:/Program Files/LLVM/bin/clang-cl.exe" ^
  -DSP_CORE="%SP_CORE%" || goto :err
cmake --build "%ENGINE%tools\sp_transcode\build" --config Release --target sp_transcode || goto :err

echo TRANSCODE BUILD OK: %ENGINE%tools\sp_transcode\build\sp_transcode.exe
endlocal & exit /b 0
:err
echo [build-transcode] FAILED
endlocal & exit /b 1
