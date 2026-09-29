@echo off
call "%~dp0..\..\scripts\env\env-cuda.bat" || exit /b 1
cd /d "%~dp0"
nvcc -O3 -arch=sm_75 --use-local-env -Xcompiler "/O2 /arch:AVX512" -o gelu_bits.exe gelu_bits.cu
cuobjdump -sass -fun k_gelu gelu_bits.exe > gelu_sass.txt
cuobjdump -sass moe_cpu_bench.exe > bench_sass.txt
