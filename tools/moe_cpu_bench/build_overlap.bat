@echo off
call "%~dp0..\..\scripts\env\env-cuda.bat" || exit /b 1
cd /d "%~dp0"
nvcc -O3 -arch=sm_75 --use-local-env -o overlap_test.exe overlap_test.cu
