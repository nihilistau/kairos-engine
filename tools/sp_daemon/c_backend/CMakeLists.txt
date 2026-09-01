# ============================================================================
# Sprint WIRE-HEX — daemon-linkable Hexagon V69 backend static library.
#
# Builds `libsp_hex_daemon_backend.a` for aarch64-android, consumed by
# sp_daemon's build.rs link step when SP_DAEMON_BACKEND=hex is intended.
# Companion to the engine's own SP_ENGINE_WITH_HEXAGON sp_engine target
# (src/backends/hexagon/CMakeLists.txt:76-87) but compiled STANDALONE so
# the daemon can link it WITHOUT pulling in the entire sp_engine library
# (which would duplicate qwen3_forward, gemma3_forward, etc. symbols
# the math-core forward path already provides via sp_session).
#
# Sources:
#   src/backends/hexagon/sp_hex_host.c   — gemma3_forward_hexagon entry
#   sp_hex_stub.c                        — qaic-generated FastRPC client stub
#   sp_daemon_hex_glue.c                 — C glue: (a) L1 forward dispatcher
#                                          (sp_daemon_hex_forward → gemma3_forward_hexagon)
#                                          and (b) engine-kernel SHIM (matmul/embed_row/
#                                          as_f32) routed to math-core's sp_* variants.
#                                          cpu_overlay.c deliberately NOT linked —
#                                          duplicates sp_kernels_read_env + qwen3_q4_stats
#                                          already exported by math-core's
#                                          libsp_forward_dispatch.a (forward_dispatch.c:30,40).
#                                          See the glue header for the resolution rationale.
#
# Math-core archives (sp_arena, sp_frobenius, sp_weight_dtype, sp_model,
# sp_gguf, sp_io_format) are linked by sp_daemon's build.rs ALREADY (the
# L3.FG cross-compile output). This lib stays light: it doesn't depend on
# the engine's other backends or the forward orchestration.
#
# Build target:    aarch64-linux-android (NDK r27d, API 21, arm64-v8a)
# Link surface:    libcdsprpc.so + rpcmem.a from Hexagon SDK 5.5.6.0
# Output path:     build-android-hex-backend/libsp_hex_daemon_backend.a
# ============================================================================

cmake_minimum_required(VERSION 3.20)
project(sp_hex_daemon_backend C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Engine root = three levels up from this CMakeLists (tools/sp_daemon/c_backend/).
get_filename_component(ENGINE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)

# Sprint WIRE-VULKAN (2026-06-02 Stage 1 follow-up): the hex backend was the
# original purpose of this CMakeLists. WIRE-VULKAN added a second STATIC
# target gated by SP_DAEMON_BUILD_VULKAN_BACKEND. To let the host Vulkan
# build path operate without the Hexagon SDK installed (the operator runs
# build-host-vulkan-backend.bat on a Windows + Vulkan-SDK host without
# HEXAGON_SDK_ROOT set), we also gate the hex block. Default = ON so the
# existing WIRE-HEX build flow (build-android-hex-backend.bat) is byte-for-
# byte unchanged. build-host-vulkan-backend.bat passes
# -DSP_DAEMON_BUILD_HEX_BACKEND=OFF to skip the Hexagon SDK probe.
option(SP_DAEMON_BUILD_HEX_BACKEND
    "Build sp_hex_daemon_backend (aarch64-android Hexagon V69 backend daemon-link variant)"
    ON)

if(SP_DAEMON_BUILD_HEX_BACKEND)
    # ── Hexagon SDK paths (mirrors src/backends/hexagon/CMakeLists.txt:32-36) ──
    if(NOT DEFINED ENV{HEXAGON_SDK_ROOT})
        message(FATAL_ERROR "HEXAGON_SDK_ROOT not set — source env-hexagon.bat or pass -DHEXAGON_SDK_ROOT=... (or -DSP_DAEMON_BUILD_HEX_BACKEND=OFF if building only the Vulkan/CPU/etc. variant)")
    endif()
    set(HEXSDK $ENV{HEXAGON_SDK_ROOT})

    set(SP_HEX_RPCMEM_INC  "${HEXSDK}/ipc/fastrpc/rpcmem/inc")
    set(SP_HEX_STDDEF_INC  "${HEXSDK}/incs/stddef")
    set(SP_HEX_FASTRPC_INC "${HEXSDK}/incs")
    set(SP_HEX_RPCMEM_LIB  "${HEXSDK}/ipc/fastrpc/rpcmem/prebuilt/android_aarch64/rpcmem.a")
    set(SP_HEX_CDSPRPC_LIB "${HEXSDK}/ipc/fastrpc/remote/ship/android_aarch64/libcdsprpc.so")

    # ── qaic codegen for sp_hex.idl → host stub + header ──
    set(SP_HEX_IDL  "${ENGINE_ROOT}/src/backends/hexagon/inc/sp_hex.idl")
    set(SP_HEX_GEN  "${CMAKE_CURRENT_BINARY_DIR}/idl_gen")
    set(SP_HEX_QAIC "${HEXSDK}/ipc/fastrpc/qaic/WinNT/qaic.exe")
    file(MAKE_DIRECTORY ${SP_HEX_GEN})

    add_custom_command(
        OUTPUT ${SP_HEX_GEN}/sp_hex.h ${SP_HEX_GEN}/sp_hex_stub.c ${SP_HEX_GEN}/sp_hex_skel.c
        COMMAND "${SP_HEX_QAIC}" -mdll -o ${SP_HEX_GEN}
                -I${HEXSDK}/incs/ -I${HEXSDK}/incs/stddef/ ${SP_HEX_IDL}
        DEPENDS ${SP_HEX_IDL}
        COMMENT "qaic sp_hex.idl -> host stub + header (daemon)"
        VERBATIM)
    add_custom_target(sp_hex_daemon_idl
        DEPENDS ${SP_HEX_GEN}/sp_hex.h ${SP_HEX_GEN}/sp_hex_stub.c)

    # ── The static library the daemon links ──
    add_library(sp_hex_daemon_backend STATIC
        ${ENGINE_ROOT}/src/backends/hexagon/sp_hex_host.c
        # cpu_overlay.c omitted — math-core's libsp_forward_dispatch.a already
        # exports sp_kernels_read_env + qwen3_q4_stats; the engine-name kernel
        # surface (matmul/embed_row/as_f32) is shimmed in sp_daemon_hex_glue.c.
        ${SP_HEX_GEN}/sp_hex_stub.c
        ${CMAKE_CURRENT_SOURCE_DIR}/sp_daemon_hex_glue.c
    )
    add_dependencies(sp_hex_daemon_backend sp_hex_daemon_idl)

    target_include_directories(sp_hex_daemon_backend PUBLIC
        ${ENGINE_ROOT}/include
        ${ENGINE_ROOT}/lib/shannon-prime-system/include
        ${ENGINE_ROOT}/src/backends/hexagon
        ${SP_HEX_GEN}
        ${SP_HEX_RPCMEM_INC}
        ${SP_HEX_STDDEF_INC}
        ${SP_HEX_FASTRPC_INC}
    )

    # cpu_overlay was conditionally AVX-gated in its production build; for the daemon
    # backend we deliberately turn it OFF (sp_hex_host pushes the compute to the
    # cDSP; the host side only needs portable C). Mirrors build-hexagon.bat:40 which
    # passes -DSP_ENGINE_WITH_AVX2=OFF for the on-phone test_ppl variant.
    target_compile_definitions(sp_hex_daemon_backend PRIVATE
        SP_ENGINE_WITH_HEXAGON=1
    )

    # The daemon links cdsprpc + rpcmem at the binary level (build.rs cargo:rustc-link-lib).
    # We just need them on the include path for the stub TU; the static lib is
    # unresolved-symbol clean apart from these (resolved at sp-daemon link time).

    message(STATUS "sp_hex_daemon_backend: hexagon V69 backend, daemon-link variant")
    message(STATUS "  sources: sp_hex_host.c + cpu_overlay.c + qaic stub + glue")
    message(STATUS "  output:  ${CMAKE_CURRENT_BINARY_DIR}/libsp_hex_daemon_backend.a")
endif()

# ============================================================================
# Sprint WIRE-VULKAN — daemon-linkable Vulkan compute backend static library.
#
# Builds `libsp_vulkan_daemon_backend.{a,lib}` for the host toolchain (Windows
# VS2019 + Vulkan SDK; Linux GCC/Clang + libvulkan). Consumed by sp_daemon's
# build.rs link step when SP_DAEMON_BACKEND=vulkan is intended.
#
# Companion to the engine's own SP_ENGINE_WITH_VULKAN sp_engine_vulkan target
# (src/CMakeLists.txt:160-181) but compiled STANDALONE so the daemon can link
# it WITHOUT pulling in the entire sp_engine library (which would duplicate
# qwen3_forward / gemma3_forward / matmul / embed_row / as_f32 symbols the
# math-core forward path already provides via sp_session).
#
# Sources (same as src/CMakeLists.txt:160-163 sp_engine_vulkan):
#   src/backends/vulkan/vulkan_backend.cpp  -- instance/device/queue lifecycle
#   src/backends/vulkan/vulkan_forward.cpp  -- gemma3_forward_vulkan + qwen3_forward_vulkan
#                                              entry points + the SPIR-V dispatch path
#   sp_daemon_vulkan_glue.c                 -- L1 forward dispatcher
#                                              (sp_daemon_vulkan_forward -> arch-route to
#                                              gemma3_forward_vulkan / qwen3_forward_vulkan)
#
# Shaders: 12 GLSL compute shaders under src/backends/vulkan/shaders/ are
# compiled to SPIR-V .spv.h C-initializer headers via glslc, then #include'd
# directly by vulkan_forward.cpp (no runtime .spv loading). Same custom_command
# pattern as src/CMakeLists.txt:142-158.
#
# The daemon binary links Vulkan::Vulkan (the loader: vulkan-1.lib on Windows,
# libvulkan.so on Linux/macOS) at sp-daemon link time via build.rs's
# wire_vulkan_backend block.
#
# Build target:   host (x86_64 / aarch64 desktop)
# Output path:    build-host-vulkan-backend/libsp_vulkan_daemon_backend.{a,lib}
# Activation:     -DSP_DAEMON_BUILD_VULKAN_BACKEND=ON
# ============================================================================

option(SP_DAEMON_BUILD_VULKAN_BACKEND
    "Build sp_vulkan_daemon_backend (host Vulkan backend daemon-link variant)"
    OFF)

if(SP_DAEMON_BUILD_VULKAN_BACKEND)
    enable_language(CXX)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    find_package(Vulkan REQUIRED)
    if(NOT Vulkan_GLSLC_EXECUTABLE)
        find_program(Vulkan_GLSLC_EXECUTABLE glslc
            HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin")
    endif()
    if(NOT Vulkan_GLSLC_EXECUTABLE)
        message(FATAL_ERROR "sp_vulkan_daemon_backend: glslc not found "
                            "(need the Vulkan SDK on PATH)")
    endif()

    # ── glslc: compile every .comp to a .spv.h C-initializer header ──
    # Same set as src/CMakeLists.txt:142-144 (kept in sync MANUALLY; if the
    # engine's shader set grows, mirror the new name here AND #include in
    # vulkan_forward.cpp upstream).
    set(SP_VK_SHADER_DIR ${ENGINE_ROOT}/src/backends/vulkan/shaders)
    set(SP_VK_SPV_DIR    ${CMAKE_CURRENT_BINARY_DIR}/vk_spv)
    file(MAKE_DIRECTORY ${SP_VK_SPV_DIR})
    set(SP_VK_SHADERS
        gemm embed_scale rmsnorm rmsnorm_head rope attn attn_ntt
        gelu_mul silu_mul add dequant_arena round_f16)
    set(SP_VK_SPV_HEADERS)
    foreach(s ${SP_VK_SHADERS})
        set(_src ${SP_VK_SHADER_DIR}/${s}.comp)
        set(_hdr ${SP_VK_SPV_DIR}/${s}.spv.h)
        add_custom_command(
            OUTPUT ${_hdr}
            COMMAND ${Vulkan_GLSLC_EXECUTABLE} -O --target-env=vulkan1.1
                    -mfmt=c -o ${_hdr} ${_src}
            DEPENDS ${_src}
            COMMENT "glslc ${s}.comp -> ${s}.spv.h (daemon)"
            VERBATIM)
        list(APPEND SP_VK_SPV_HEADERS ${_hdr})
    endforeach()
    add_custom_target(sp_vk_daemon_shaders DEPENDS ${SP_VK_SPV_HEADERS})

    # ── The static library the daemon links ──
    add_library(sp_vulkan_daemon_backend STATIC
        ${ENGINE_ROOT}/src/backends/vulkan/vulkan_backend.cpp
        ${ENGINE_ROOT}/src/backends/vulkan/vulkan_forward.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sp_daemon_vulkan_glue.c
    )
    add_dependencies(sp_vulkan_daemon_backend sp_vk_daemon_shaders)
    set_source_files_properties(${ENGINE_ROOT}/src/backends/vulkan/vulkan_forward.cpp
        PROPERTIES OBJECT_DEPENDS "${SP_VK_SPV_HEADERS}")

    target_include_directories(sp_vulkan_daemon_backend PUBLIC
        ${ENGINE_ROOT}/include
        ${ENGINE_ROOT}/lib/shannon-prime-system/include
        ${ENGINE_ROOT}/src/backends/vulkan
        ${SP_VK_SPV_DIR}
    )

    target_compile_definitions(sp_vulkan_daemon_backend PRIVATE
        SP_ENGINE_WITH_VULKAN=1
    )

    # The Vulkan loader is resolved at sp-daemon binary link time via
    # build.rs's cargo:rustc-link-lib=vulkan-1 (Windows) / vulkan (Linux).
    # Here we just need the headers on the include path for both TUs (the
    # Vulkan::Vulkan target provides Vulkan_INCLUDE_DIRS).
    target_link_libraries(sp_vulkan_daemon_backend PUBLIC Vulkan::Vulkan)

    message(STATUS "sp_vulkan_daemon_backend: Vulkan backend, daemon-link variant")
    message(STATUS "  sources: vulkan_backend.cpp + vulkan_forward.cpp + 12 SPIR-V shaders + glue")
    message(STATUS "  glslc:   ${Vulkan_GLSLC_EXECUTABLE}")
    message(STATUS "  output:  ${CMAKE_CURRENT_BINARY_DIR}/sp_vulkan_daemon_backend.{a,lib}")
endif()
