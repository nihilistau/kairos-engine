# ============================================================================
# Sprint WIRE-CPU — daemon-linkable CPU AVX-512 backend static library.
#
# Builds `libsp_cpu_daemon_backend.a` (Linux/MinGW) or `sp_cpu_daemon_backend.lib`
# (MSVC) for the HOST target, consumed by sp_daemon's build.rs link step when
# the `wire_cpu_backend` Cargo feature is on.
#
# Symmetric to tools/sp_daemon/c_backend/CMakeLists.txt (the hex variant) but
# with two key differences:
#   1. HOST target. No Android cross-compile. No NDK toolchain. No FastRPC.
#   2. INCLUDES cpu_overlay.c (engine kernel surface incl. AVX2 dot_f32 and
#      the matmul/embed_row/rmsnorm/rope_neox engine symbols). The hex
#      variant deliberately OMITTED cpu_overlay.c because hex compute runs
#      on the cDSP; for CPU, cpu_overlay.c IS the value-add.
#
# Symbol-collision resolution: engine cpu sources export the same
# `gemma3_forward` / `qwen3_forward` / `qwen25_forward` /
# `sp_kernels_read_env` / `qwen3_q4_stats` names as math-core's
# `libsp_forward.lib` + `libsp_forward_dispatch.lib`. The daemon already
# links math-core (build.rs MODULES table). To avoid duplicate-symbol link
# errors we compile each engine cpu TU with -D macros renaming the
# colliding names to *_cpu_impl variants; sp_daemon_cpu_glue.c then
# re-exposes them under their final "_cpu" suffix.
#
# The non-colliding engine names stay un-renamed (`matmul`, `embed_row`,
# `dot_f32`, `rmsnorm`, `rmsnorm_head`, `rope_neox`, `kernels_attn_head`)
# because they are UNIQUE to the engine -- math-core uses `sp_matmul`,
# `sp_embed_row`, `sp_dot_f32` etc. The engine cpu_forward.c / cpu_gemma3.c
# CALL these unprefixed names from cpu_overlay.c; routing them through
# math-core's sp_* variants would shed the entire CPU vectorization value.
#
# Build target:    host (whatever the cargo build target_os is)
# Output path:     build-host-cpu-backend/libsp_cpu_daemon_backend.a (Linux/MinGW)
#                  build-host-cpu-backend/Release/sp_cpu_daemon_backend.lib (MSVC)
# ============================================================================

cmake_minimum_required(VERSION 3.20)
project(sp_cpu_daemon_backend C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Engine root = three levels up from this CMakeLists (tools/sp_daemon/c_backend/).
get_filename_component(ENGINE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)

# ── Symbol-rename macros applied to the engine cpu TUs ────────────────────
# These -D defines preprocess every occurrence of the LHS to the RHS inside
# the engine sources. The renamed bodies land under the *_cpu_impl symbol
# names which sp_daemon_cpu_glue.c forwards to its public *_cpu entries.
set(SP_CPU_RENAMES
    -Dgemma3_forward=gemma3_forward_cpu_impl
    -Dqwen3_forward=qwen3_forward_cpu_impl
    -Dqwen3_forward_ex=qwen3_forward_ex_cpu_impl
    -Dqwen3_generate=qwen3_generate_cpu_impl
    -Dqwen3_generate_kv=qwen3_generate_kv_cpu_impl
    -Dsp_kernels_read_env=sp_cpu_kernels_read_env
    -Dqwen3_q4_stats=sp_cpu_q4_stats
)

# ── The static library the daemon links ──
add_library(sp_cpu_daemon_backend STATIC
    ${ENGINE_ROOT}/src/backends/cpu/cpu_overlay.c
    ${ENGINE_ROOT}/src/backends/cpu/cpu_forward.c
    ${ENGINE_ROOT}/src/backends/cpu/cpu_gemma3.c
    ${ENGINE_ROOT}/src/backends/cpu/cpu_generate.c
    ${CMAKE_CURRENT_SOURCE_DIR}/sp_daemon_cpu_glue.c
)

target_include_directories(sp_cpu_daemon_backend PUBLIC
    ${ENGINE_ROOT}/include
    ${ENGINE_ROOT}/lib/shannon-prime-system/include
)

# Apply the renames to ALL engine cpu TUs (NOT the glue, which uses the
# unrenamed *_cpu_impl declarations directly).
set_source_files_properties(
    ${ENGINE_ROOT}/src/backends/cpu/cpu_overlay.c
    ${ENGINE_ROOT}/src/backends/cpu/cpu_forward.c
    ${ENGINE_ROOT}/src/backends/cpu/cpu_gemma3.c
    ${ENGINE_ROOT}/src/backends/cpu/cpu_generate.c
    PROPERTIES COMPILE_OPTIONS "${SP_CPU_RENAMES}"
)

# ── AVX2 / AVX-512 codegen flags ──
# AVX2 default ON (cpu_overlay.c dot_f32 uses _mm256_fmadd_ps; compiles under
# MSVC + GCC/Clang).
# AVX-512 sources (avx512_vnni.c, avx512_spinor.c, avx512_ternlog.c,
# avx512_ifma.c, avx512_persist.c) use __attribute__((target("avx512f"))) +
# GCC-specific __atomic_* builtins; MSVC does not parse these. The canonical
# engine MSVC build (<path-to>\shannon-prime-system-engine\build-cpu)
# verified 2026-06-01 to contain ONLY avx512_dispatch.c.obj from the avx512/
# directory -- the other 5 TUs failed silently in the engine's own MSVC build
# pipeline and are NOT in libsp_engine.lib. They were tested under a different
# toolchain (Linux/Clang per Phase 2-CPU.AVX). So on MSVC we drop them; on
# GCC/Clang we include all of them, plus avx512_dispatch.c which is portable.
option(SP_CPU_DAEMON_WITH_AVX2   "Compile cpu_overlay.c with AVX2"          ON)
if(MSVC)
    option(SP_CPU_DAEMON_WITH_AVX512 "Add AVX-512 primitive TUs (sp_avx512_*)" OFF)
else()
    option(SP_CPU_DAEMON_WITH_AVX512 "Add AVX-512 primitive TUs (sp_avx512_*)" ON)
endif()

if(SP_CPU_DAEMON_WITH_AVX2)
    target_compile_definitions(sp_cpu_daemon_backend PRIVATE SP_ENGINE_AVX2=1)
    if(MSVC)
        target_compile_options(sp_cpu_daemon_backend PRIVATE /arch:AVX2)
    else()
        target_compile_options(sp_cpu_daemon_backend PRIVATE -mavx2 -mfma)
    endif()
endif()

if(SP_CPU_DAEMON_WITH_AVX512)
    # avx512_dispatch.c is portable (uses __builtin_cpu_supports OR MSVC __cpuidex);
    # we keep it for the future-proof CPU-feature detection. The intrinsic-bearing
    # TUs follow only on non-MSVC toolchains per the comment above.
    target_sources(sp_cpu_daemon_backend PRIVATE
        ${ENGINE_ROOT}/src/backends/cpu/avx512/avx512_dispatch.c
    )
    if(NOT MSVC)
        target_sources(sp_cpu_daemon_backend PRIVATE
            ${ENGINE_ROOT}/src/backends/cpu/avx512/avx512_spinor.c
            ${ENGINE_ROOT}/src/backends/cpu/avx512/avx512_vnni.c
            ${ENGINE_ROOT}/src/backends/cpu/avx512/avx512_ifma.c
            ${ENGINE_ROOT}/src/backends/cpu/avx512/avx512_ternlog.c
            ${ENGINE_ROOT}/src/backends/cpu/avx512/avx512_persist.c
        )
    endif()
    target_compile_definitions(sp_cpu_daemon_backend PRIVATE SP_ENGINE_AVX512=1)
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        set_source_files_properties(
            ${ENGINE_ROOT}/src/backends/cpu/avx512/avx512_dispatch.c
            ${ENGINE_ROOT}/src/backends/cpu/avx512/avx512_spinor.c
            ${ENGINE_ROOT}/src/backends/cpu/avx512/avx512_vnni.c
            ${ENGINE_ROOT}/src/backends/cpu/avx512/avx512_ifma.c
            ${ENGINE_ROOT}/src/backends/cpu/avx512/avx512_ternlog.c
            PROPERTIES COMPILE_OPTIONS
            "-mavx512f;-mavx512vnni;-mavx512ifma;-mavx512bw;-mavx512dq;-mavx512vpopcntdq;-mavx512vbmi2"
        )
        set_source_files_properties(
            ${ENGINE_ROOT}/src/backends/cpu/avx512/avx512_persist.c
            PROPERTIES COMPILE_OPTIONS "-mavx512f;-mwaitpkg"
        )
    elseif(MSVC)
        # Only avx512_dispatch.c is in the source list under MSVC; the other
        # TUs are gated out in the NOT MSVC block above. avx512_dispatch.c
        # itself uses __cpuidex (intrin.h) and does NOT need /arch:AVX512.
    endif()
endif()

# MSVC: drop the harmless C4996 (secure-CRT) noise the engine sources
# already gate via _CRT_SECURE_NO_WARNINGS in their own headers.
if(MSVC)
    target_compile_definitions(sp_cpu_daemon_backend PRIVATE _CRT_SECURE_NO_WARNINGS)
    target_compile_options(sp_cpu_daemon_backend PRIVATE /wd4244 /wd4267)
endif()

message(STATUS "sp_cpu_daemon_backend: CPU AVX-512 backend, daemon-link variant")
message(STATUS "  sources: cpu_overlay.c + cpu_forward.c + cpu_gemma3.c + cpu_generate.c + glue")
message(STATUS "  AVX2 = ${SP_CPU_DAEMON_WITH_AVX2}, AVX512 = ${SP_CPU_DAEMON_WITH_AVX512}")
message(STATUS "  output: ${CMAKE_BINARY_DIR}/(Release|.)/sp_cpu_daemon_backend.(lib|a)")
