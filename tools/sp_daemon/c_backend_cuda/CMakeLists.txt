# ============================================================================
# Sprint WIRE-CUDA -- daemon-linkable CUDA PTX backend static library.
#
# Builds `libsp_cuda_daemon_backend.lib` (Windows / MSVC) or
# `libsp_cuda_daemon_backend.a` (Linux / GNU) consumed by sp_daemon's build.rs
# link step when the `wire_cuda_backend` Cargo feature is on.
#
# Companion to the engine's SP_ENGINE_WITH_CUDA sp_engine_cuda target (in
# src/CMakeLists.txt:99-119) but compiled STANDALONE so the daemon can link
# it WITHOUT pulling in the entire sp_engine library (which would duplicate
# qwen3_forward, gemma3_forward, etc. symbols the math-core forward path
# already provides via sp_session).
#
# Sources:
#   src/backends/cuda/cuda_backend.cu   -- sp_cuda_device_count, info, sp_set_error wrap
#   src/backends/cuda/cuda_forward.cu   -- gemma3_forward_cuda + qwen3_forward_cuda
#   sp_daemon_cuda_glue.c               -- C glue: L1 forward dispatcher
#                                          (sp_daemon_cuda_forward -> arch-switch
#                                          -> gemma3_forward_cuda / qwen3_forward_cuda)
#
# Math-core archives (sp_arena, sp_frobenius, sp_weight_dtype, sp_model,
# sp_gguf, sp_io_format, sp_forward_dispatch) are linked by sp_daemon's
# build.rs already (the standard host link). This lib stays light: it
# doesn't depend on the engine's other backends or the forward orchestration.
# The CUDA TUs DO consume math-core's sp_matmul / sp_embed_row / sp_as_f32
# directly via forward_dispatch.h; those symbols resolve at sp-daemon link
# time from the math-core archives.
#
# Build target:    x86_64 Windows MSVC (host, RTX 2060 sm_75)
#                  also supports Linux GCC if a CUDA-capable host runs Linux
# Link surface:    CUDA::cudart + CUDA::cublas
# Output path:     build-host-cuda-backend/sp_cuda_daemon_backend.lib (MSVC)
#                  build-host-cuda-backend/libsp_cuda_daemon_backend.a (GCC)
# ============================================================================

cmake_minimum_required(VERSION 3.20)
project(sp_cuda_daemon_backend C CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Engine root = three levels up from this CMakeLists (tools/sp_daemon/c_backend/).
get_filename_component(ENGINE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)

find_package(CUDAToolkit REQUIRED)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # RTX 2060 = sm_75 (Turing). Set explicitly via -DCMAKE_CUDA_ARCHITECTURES=75
    # from the build script for clarity. Default here covers the engine's
    # production set (75=Turing, 80=Ampere, 90=Hopper).
    set(CMAKE_CUDA_ARCHITECTURES "75;80;90")
endif()

# ── The static library the daemon links ──
#
# WIRE-CUDA-GEMMA4: cuda_forward.cu's gemma4_decode_cuda (the XBAR P3 decode
# harness) references the math-core XBAR episode-manifest serializer
# (sp_xbar_manifest_serial_size / _serialize / _deserialize / _free from
# lib/shannon-prime-system/core/xbar/xbar_episode.c). That TU is a standalone,
# self-contained C file (includes only sp/xbar_episode.h + libc — no model, no
# other module deps) and is NOT carried by any math-core build-cpu module, so the
# daemon's build.rs MODULES loop never links it. Compile it directly into this
# daemon-specific backend lib (the established home for engine-side glue) so the
# 4 sp_xbar_manifest_* symbols resolve at the daemon's binary link step.
add_library(sp_cuda_daemon_backend STATIC
    ${ENGINE_ROOT}/src/backends/cuda/cuda_backend.cu
    ${ENGINE_ROOT}/src/backends/cuda/cuda_forward.cu
    ${ENGINE_ROOT}/lib/shannon-prime-system/core/xbar/xbar_episode.c
    ${CMAKE_CURRENT_SOURCE_DIR}/sp_daemon_cuda_glue.c
)

set_target_properties(sp_cuda_daemon_backend PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    # CUDA-side OUTPUT_NAME stays plain so MSVC writes sp_cuda_daemon_backend.lib
    # and GCC writes libsp_cuda_daemon_backend.a (sp_daemon's build.rs uses
    # static=sp_cuda_daemon_backend, matching either convention).
)

target_include_directories(sp_cuda_daemon_backend PUBLIC
    ${ENGINE_ROOT}/include
    ${ENGINE_ROOT}/lib/shannon-prime-system/include
)

target_compile_definitions(sp_cuda_daemon_backend PRIVATE
    SP_ENGINE_WITH_CUDA=1
)

# ── Runtime-library alignment with cargo's cc-rs default (LIBCMT / static MT) ─
#
# cc-rs (used by tokenizers/esaxx-rs to build esaxx.lib for the sp-daemon Rust
# binary) defaults to /MT on MSVC (LIBCMT directive emitted into .objs). nvcc's
# default is /MD (MSVCRT) for its C++ runtime — which produces LNK2038
# "RuntimeLibrary mismatch ('MD_DynamicRelease' doesn't match 'MT_StaticRelease')"
# at the rust binary link step. Force /MT here so both sides of the daemon link
# graph agree. The math-core C libs (sp_session, sp_arena, ...) are pure C and
# emit no _RuntimeLibrary directive at all, so they don't enter this conflict —
# only C++/CUDA TUs do.
if(MSVC)
    set_property(TARGET sp_cuda_daemon_backend PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    # CMAKE_MSVC_RUNTIME_LIBRARY only applies to host (cl) TUs; CUDA needs
    # the explicit -Xcompiler flag.
    target_compile_options(sp_cuda_daemon_backend PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/MT>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-Xcompiler=/MTd>
    )
endif()

# CUDA runtime + cuBLAS. The daemon's final link picks these up via build.rs
# (rustc-link-lib=cudart_static, cublas_static, etc.). We declare them PUBLIC
# here so the static lib's link interface advertises them, though they're
# resolved at the binary link step.
target_link_libraries(sp_cuda_daemon_backend PUBLIC
    CUDA::cudart CUDA::cublas)

message(STATUS "sp_cuda_daemon_backend: CUDA PTX backend, daemon-link variant")
message(STATUS "  sources: cuda_backend.cu + cuda_forward.cu + sp_daemon_cuda_glue.c")
message(STATUS "  CUDA arch: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  output:    ${CMAKE_CURRENT_BINARY_DIR}/(lib)sp_cuda_daemon_backend.(lib|a)")

# G-PERF-ATTN (2026-07-12): register/spill visibility. Nobody had ever seen the bx
# attention kernels' register budget -- and __launch_bounds__(1024) caps Turing at 64.
target_compile_options(sp_cuda_daemon_backend PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>)
