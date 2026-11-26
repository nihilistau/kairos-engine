# sp_transcode — GGUF -> .sp-model + .sp-tokenizer, standalone (2026-11-26).
#
# Brought across from the retired C engine (shannon-prime-system-engine/tools/sp_transcode,
# source dated 2026-09-02: --q4b, gemma4-MoE detection, --stream). That engine linked the tool
# against its whole sp_engine library; every symbol the tool actually uses lives in the math
# core, so this builds it against the core alone — the same submodule sp-daemon builds from.
#
#   cmake -S engine/tools/sp_transcode -B engine/tools/sp_transcode/build -DSP_CORE=<core>
#   (build-transcode.bat does this with the house toolchain and resolve-core.bat)
cmake_minimum_required(VERSION 3.20)
project(sp_transcode LANGUAGES C CXX)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

set(SP_CORE "${CMAKE_CURRENT_SOURCE_DIR}/../../lib/shannon-prime-system" CACHE PATH "the math core")
if(NOT EXISTS "${SP_CORE}/CMakeLists.txt")
    message(FATAL_ERROR "math core not found at ${SP_CORE} — pass -DSP_CORE=<core checkout>")
endif()
set(SP_SYSTEM_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory("${SP_CORE}" core EXCLUDE_FROM_ALL)

add_executable(sp_transcode sp_transcode.c)
target_include_directories(sp_transcode PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../include"     # sp_engine/{sp_model,gguf,model}.h
    "${SP_CORE}/include/sp")                        # "sp_hash.h"
target_link_libraries(sp_transcode PRIVATE
    sp_frobenius sp_gguf sp_weight_dtype sp_io_hash sp_io_format sp_model sp_arena)
if(MSVC)
    target_compile_options(sp_transcode PRIVATE /W3)
endif()
