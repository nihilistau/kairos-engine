@echo off
call "%~dp0..\..\scripts\env\env-cuda.bat" || exit /b 1
cd /d "%~dp0"
nvcc -O3 -arch=sm_75 --use-local-env -Xcompiler "/O2 /fp:precise" -o mufu_test.exe mufu_test.cu
cuobjdump -sass mufu_test.exe > mufu_sass.txt
