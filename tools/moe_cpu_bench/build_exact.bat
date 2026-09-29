@echo off
call "%~dp0..\..\scripts\env\env-cuda.bat" || exit /b 1
cd /d "%~dp0"
nvcc -O3 -arch=sm_75 --use-local-env -Xcompiler "/O2 /arch:AVX512 /EHsc" -o moe_exact_bench.exe moe_exact_bench.cu ..\..\src\backends\cuda\moe_cpu_expert.c -lSynchronization
