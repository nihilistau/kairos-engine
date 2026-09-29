@echo off
call "%~dp0..\..\scripts\env\env-cuda.bat" || exit /b 1
cd /d "%~dp0"
nvcc -O3 -arch=sm_75 --use-local-env -Xcompiler "/O2 /arch:AVX512 /EHsc" -o moe_cpu_bench.exe moe_cpu_bench.cu
