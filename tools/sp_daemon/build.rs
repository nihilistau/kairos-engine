use std::env;
use std::path::PathBuf;

// Math-core modules in link order (high-level → primitive).
// Each entry: (subdir_name, link_lib_name).
// Paths: {SP_SYSTEM_BUILD_DIR}/core/{subdir}/[lib]{link_lib_name}.[a|lib]
const MODULES: &[(&str, &str)] = &[
    ("arm",              "sp_arm"),
    ("session",          "sp_session"),
    ("forward",          "sp_forward"),
    ("forward_dispatch", "sp_forward_dispatch"),
    ("forward_kernels",  "sp_forward_kernels"),
    ("model",            "sp_model"),
    ("arena",            "sp_arena"),
    ("frobenius",        "sp_frobenius"),
    ("sieve",            "sp_sieve"),
    ("kste",             "sp_kste"),
    ("vht2",             "sp_vht2"),
    ("poly_ring",        "sp_poly_ring"),
    ("ntt_crt",          "sp_ntt_crt"),
    ("ok_arith",         "sp_ok_arith"),
    ("gguf",             "sp_gguf"),
    ("io_format",        "sp_io_format"),
    ("io_hash",          "sp_io_hash"),
    ("weight_dtype",     "sp_weight_dtype"),
];

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    // tools/sp_daemon/ → engine root is ../..
    let engine_root = manifest_dir.parent().unwrap().parent().unwrap();

    // ── Headers ────────────────────────────────────────────────────────────
    // sp_l1.h lives in the math-core submodule include tree.
    // Override with SP_SYSTEM_INCLUDE for out-of-tree builds.
    let include_dir = env::var("SP_SYSTEM_INCLUDE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| engine_root.join("lib/shannon-prime-system/include"));

    println!("cargo:rerun-if-env-changed=SP_SYSTEM_INCLUDE");
    println!("cargo:rerun-if-env-changed=SP_SYSTEM_BUILD_DIR");
    println!("cargo:rerun-if-changed={}", include_dir.join("sp/sp_l1.h").display());
    println!("cargo:rerun-if-changed={}", include_dir.join("sp/sp_model.h").display());
    println!("cargo:rerun-if-changed={}", include_dir.join("sp/sp_status.h").display());
    // M.5 (routing): KSTE encoder header is the routing primitive input.
    println!("cargo:rerun-if-changed={}", include_dir.join("sp/kste.h").display());

    // Phase 2-L3.FG: android now bindgens + links the math-core C ABI (the
    // early sp_no_link skip is removed). bindgen runs on the host but targets
    // aarch64 via BINDGEN_EXTRA_CLANG_ARGS_aarch64-linux-android (--target +
    // --sysroot, set in .cargo/config.toml).

    // ── bindgen ────────────────────────────────────────────────────────────
    // Bindgen the frozen L1 header. sp_l1.h includes sp_model.h + sp_status.h
    // transitively; bindgen sees all three through -I. M.5 routing also needs
    // sp_kste_encode + sp_kste_tree_t from sp/kste.h which is NOT transitively
    // included from sp_l1.h, so we add it via a synthesized wrapper header in
    // OUT_DIR.
    // Requires libclang. On Windows set:
    //   LIBCLANG_PATH=C:\Program Files\LLVM\bin
    // Issue #115: the engine's C tokenizer (sp_engine/tokenizer.h) is the
    // PROVEN gemma4 514k-merge BPE encoder (T_G4_TOK_PARITY: byte-for-byte vs
    // the llama oracle). The daemon FFIs to it for gemma4 (.sp-tokenizer
    // type_id=4) rather than reimplementing it in Rust. The engine include tree
    // is at {engine}/include; override with SP_ENGINE_INCLUDE.
    let engine_include_dir = env::var("SP_ENGINE_INCLUDE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| engine_root.join("include"));
    let tok_src_dir = engine_root.join("src/tokenizer");
    println!("cargo:rerun-if-env-changed=SP_ENGINE_INCLUDE");
    println!("cargo:rerun-if-changed={}", engine_include_dir.join("sp_engine/tokenizer.h").display());
    println!("cargo:rerun-if-changed={}", tok_src_dir.join("tokenizer.c").display());
    println!("cargo:rerun-if-changed={}", tok_src_dir.join("gemma4_bpe.c").display());

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let wrapper_path = out_dir.join("sp_bindgen_wrapper.h");
    std::fs::write(
        &wrapper_path,
        "/* Auto-generated bindgen wrapper. */\n\
         #include \"sp/sp_l1.h\"\n#include \"sp/arm.h\"\n\
         /* M.5 (routing): KSTE encoder + Tier-0/Tier-1 dominance API. */\n\
         #include \"sp/kste.h\"\n\
         /* Issue #115: the engine C gemma4 BPE tokenizer ABI. */\n\
         #include \"sp_engine/tokenizer.h\"\n",
    ).expect("could not write sp_bindgen_wrapper.h");

    let bindings = bindgen::Builder::default()
        .header(wrapper_path.to_string_lossy())
        .clang_arg(format!("-I{}", include_dir.display()))
        .clang_arg(format!("-I{}", engine_include_dir.display()))
        .allowlist_function("sp_.*")
        .allowlist_type("sp_.*")
        .allowlist_var("SP_.*")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("bindgen failed — set LIBCLANG_PATH if libclang is not on PATH");

    bindings
        .write_to_file(out_dir.join("sp_bindings.rs"))
        .expect("could not write sp_bindings.rs");

    // ── Issue #115: compile the engine C gemma4 BPE tokenizer ──────────────
    // Two TUs implement the proven encoder:
    //   src/tokenizer/tokenizer.c   — loaders + family dispatch + GPT2/SPM lanes
    //   src/tokenizer/gemma4_bpe.c  — the GEMMA4_BPE lane (514k-merge, #115)
    // tokenizer.c's GGUF-lane references (gguf_open/find_kv/get_str/get_u64/
    // kv_str_array/close) resolve from the math-core `sp_gguf` archive ALREADY
    // linked by the MODULES loop below (identical C ABI; the engine and
    // math-core gguf.h are the same surface). sp_tokenizer_load_tokfile — the
    // path the daemon calls — reads the .sp-tokenizer file directly and pulls
    // no gguf symbols. sp_tok_header / SP_TOK_* are header-only (sp_model.h).
    // The static lib is linked into the daemon so routes/tokenizer.rs can FFI
    // sp_tokenizer_{load_tokfile,encode,decode,free,vocab_size}.
    cc::Build::new()
        .file(tok_src_dir.join("tokenizer.c"))
        .file(tok_src_dir.join("gemma4_bpe.c"))
        .include(&engine_include_dir)        // sp_engine/{tokenizer,gguf,sp_model}.h
        .include(&tok_src_dir)               // tokenizer_internal.h + unicode_ranges.h
        .define("_CRT_SECURE_NO_WARNINGS", None)
        .warnings(false)
        .compile("sp_engine_tokenizer");

    // ── Link ───────────────────────────────────────────────────────────────
    // Phase 2-L3.FG: aarch64-android now links the cross-compiled math-core
    // (build-android-libs.bat → build-android-libs/core/<m>/libsp_<m>.a). The
    // old sp_no_link skip is gone; android falls through to the link loop below.
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    // SP_SYSTEM_BUILD_DIR: root of the math-core build directory.
    //
    //   Engine-embedded (Windows MinGW / MSVC, engine build-cpu):
    //     {engine}/build-cpu/lib/shannon-prime-system/  → core/{m}/sp_{m}.lib
    //   aarch64-android (Phase 2-L3.FG, build-android-libs.bat):
    //     {engine}/build-android-libs/                   → core/{m}/libsp_{m}.a
    //
    // Both resolve libs under the same core/{module}/ sub-path; only the file
    // prefix/extension differs, which Cargo derives from the target.
    let build_dir = env::var("SP_SYSTEM_BUILD_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            if target_os == "android" {
                engine_root.join("build-android-libs")
            } else {
                // Default: engine's MinGW build-cpu embedded submodule (Windows).
                engine_root.join("build-cpu/lib/shannon-prime-system")
            }
        });

    // CARGO_CFG_TARGET_ENV differentiates MSVC ("msvc") from MinGW/Linux ("gnu"/"").
    // On MSVC, cargo:rustc-link-lib=static stops propagating to [[bin]] targets when
    // the [lib] in this package (lib.rs) does not itself reference the C FFI symbols.
    // cargo:rustc-link-arg bypasses that boundary and reaches the binary link step
    // directly with a full absolute path — no /LIBPATH lookup required.
    let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();

    // DESIGN-NO-EXACT-PROFILE §4b: the integer memory-ring math-core archives are linked only
    // under the default `exact` feature. The standard-FP profile (--no-default-features) drops them
    // — they are dormant in the served path (recall is FP cosine); the Rust NTT FFI that references
    // their symbols (ntt_ffi / ntt_hex_dispatch) is cfg-gated out in lockstep.
    let exact = env::var("CARGO_FEATURE_EXACT").is_ok();
    const RING_MODULES: &[&str] = &["ntt_crt", "poly_ring", "ok_arith", "frobenius"];

    for (module_dir, lib_name) in MODULES {
        if !exact && RING_MODULES.contains(module_dir) {
            continue; // FP profile: do not link the integer memory-ring archives
        }
        let search = build_dir.join("core").join(module_dir);
        println!("cargo:rustc-link-search=native={}", search.display());
        if target_env == "msvc" {
            let lib_path = search.join(format!("{lib_name}.lib"));
            println!("cargo:rustc-link-arg={}", lib_path.display());
        } else {
            println!("cargo:rustc-link-lib=static={lib_name}");
        }
    }

    if target_os == "linux" || target_os == "android" {
        println!("cargo:rustc-link-lib=m");
    }

    // ── Sprint WIRE-HEX: link the daemon-callable Hexagon V69 backend ──
    //
    // When the `wire_hex_backend` Cargo feature is on AND we're cross-compiling
    // for android, link the standalone static lib built by
    // `tools/sp_daemon/build-android-hex-backend.bat`:
    //   build-android-hex-backend/libsp_hex_daemon_backend.a
    //
    // That archive contains:
    //   - src/backends/hexagon/sp_hex_host.c (gemma3_forward_hexagon)
    //   - src/backends/cpu/cpu_overlay.c     (matmul/embed_row/as_f32/sp_kernels_read_env)
    //   - generated sp_hex_stub.c            (qaic FastRPC client stub)
    //   - tools/sp_daemon/c_backend/sp_daemon_hex_glue.c (the §6 dispatcher)
    //
    // Plus the FastRPC runtime libs from Hexagon SDK 5.5.6.0:
    //   libcdsprpc.so   (intra-device IPC; provides remote_handle64_* + remote_session_control)
    //   rpcmem.a        (rpcmem_alloc/free/init)
    //
    // Build order: `build-android-libs.bat` first (math-core archives the
    // hex glue depends on transitively), then `build-android-hex-backend.bat`,
    // then `cargo build --target aarch64-linux-android --features wire_hex_backend`.
    // ── Sprint WIRE-VULKAN: link the daemon-callable Vulkan backend ──
    //
    // When the `wire_vulkan_backend` Cargo feature is on (host-side; no
    // target_os gate — Vulkan is desktop GPU compute on Windows / Linux /
    // macOS), link the standalone static lib built by
    // `tools/sp_daemon/build-host-vulkan-backend.bat`:
    //   build-host-vulkan-backend/sp_vulkan_daemon_backend.{a,lib}
    //
    // That archive contains:
    //   - src/backends/vulkan/vulkan_backend.cpp  (instance/device/queue lifecycle)
    //   - src/backends/vulkan/vulkan_forward.cpp  (gemma3_forward_vulkan + qwen3_forward_vulkan)
    //   - 12 SPIR-V compute shader .spv.h embeds  (compiled by glslc at build time)
    //   - tools/sp_daemon/c_backend/sp_daemon_vulkan_glue.c  (the §6 dispatcher)
    //
    // Plus the Vulkan loader at sp-daemon link time:
    //   vulkan-1 (Windows: vulkan-1.lib -> vulkan-1.dll)
    //   vulkan   (Linux/macOS: libvulkan.so / libvulkan.dylib)
    //
    // Build order: `build-host-vulkan-backend.bat` (or the symmetric
    // `cmake -B build-host-vulkan-backend -S tools/sp_daemon/c_backend
    //  -DSP_DAEMON_BUILD_VULKAN_BACKEND=ON` on Linux) first, then
    // `cargo build --features wire_vulkan_backend` builds the daemon with
    // the dispatcher wired in.
    let wire_vulkan = env::var("CARGO_FEATURE_WIRE_VULKAN_BACKEND").is_ok();
    if wire_vulkan {
        let vk_lib_dir = env::var("SP_VULKAN_BACKEND_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| engine_root.join("build-host-vulkan-backend"));
        // Archive name differs per platform (.lib MSVC; .a GNU/Clang/MinGW).
        let vk_archive_msvc = vk_lib_dir.join("sp_vulkan_daemon_backend.lib");
        let vk_archive_gnu  = vk_lib_dir.join("libsp_vulkan_daemon_backend.a");
        if !vk_archive_msvc.exists() && !vk_archive_gnu.exists() {
            panic!(
                "WIRE-VULKAN: Vulkan backend archive not found at {} (neither {} nor {}) — run build-host-vulkan-backend.bat first",
                vk_lib_dir.display(),
                vk_archive_msvc.file_name().unwrap().to_string_lossy(),
                vk_archive_gnu.file_name().unwrap().to_string_lossy(),
            );
        }
        println!("cargo:rustc-link-search=native={}", vk_lib_dir.display());
        if target_env == "msvc" && vk_archive_msvc.exists() {
            // MSVC: pass the .lib by absolute path through rustc-link-arg
            // (same pattern as the math-core MODULES loop above).
            println!("cargo:rustc-link-arg={}", vk_archive_msvc.display());
        } else {
            println!("cargo:rustc-link-lib=static=sp_vulkan_daemon_backend");
        }

        // Vulkan loader. On Windows the import lib is vulkan-1.lib (no
        // VULKAN_SDK on PATH at run time — only the DLL needs to be found
        // by the dynamic loader). On Linux it's libvulkan.so.
        let vulkan_loader_lib = if target_os == "windows" || target_env == "msvc" {
            "vulkan-1"
        } else {
            "vulkan"
        };
        println!("cargo:rustc-link-lib=dylib={vulkan_loader_lib}");

        // Vulkan SDK path lets us add the loader's link search dir on
        // Windows (where the system PATH may not include $VULKAN_SDK\Lib).
        if let Ok(vk_sdk) = env::var("VULKAN_SDK") {
            let lib_subdir = if target_env == "msvc" { "Lib" } else { "lib" };
            println!("cargo:rustc-link-search=native={vk_sdk}/{lib_subdir}");
        }

        // C++ standard library: vulkan_forward.cpp + vulkan_backend.cpp are
        // C++17 TUs; link the appropriate C++ runtime.
        if target_env == "msvc" {
            // MSVC links msvcprt implicitly via the static archive's symbols.
        } else if target_os == "macos" {
            println!("cargo:rustc-link-lib=dylib=c++");
        } else {
            // GNU/Clang on Linux / MinGW: stdc++.
            println!("cargo:rustc-link-lib=dylib=stdc++");
        }

        println!("cargo:rerun-if-env-changed=SP_VULKAN_BACKEND_DIR");
        println!("cargo:rerun-if-env-changed=VULKAN_SDK");
        println!("cargo:warning=WIRE-VULKAN: linking sp_vulkan_daemon_backend + {vulkan_loader_lib}");
    }

    let wire_hex = env::var("CARGO_FEATURE_WIRE_HEX_BACKEND").is_ok();
    if wire_hex && target_os == "android" {
        let hex_lib_dir = env::var("SP_HEX_BACKEND_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| engine_root.join("build-android-hex-backend"));
        let hex_archive = hex_lib_dir.join("libsp_hex_daemon_backend.a");
        if !hex_archive.exists() {
            panic!(
                "WIRE-HEX: hex backend archive not found at {} — run build-android-hex-backend.bat first",
                hex_archive.display()
            );
        }
        println!("cargo:rustc-link-search=native={}", hex_lib_dir.display());
        println!("cargo:rustc-link-lib=static=sp_hex_daemon_backend");

        // FastRPC runtime libs (Hexagon SDK 5.5.6.0). HEXAGON_SDK_ROOT is set
        // by scripts/env/env-hexagon.bat; this build step expects it present.
        let hex_sdk = env::var("HEXAGON_SDK_ROOT")
            .unwrap_or_else(|_| String::from("C:/Qualcomm/Hexagon_SDK/5.5.6.0"));
        let rpcmem_dir  = format!("{hex_sdk}/ipc/fastrpc/rpcmem/prebuilt/android_aarch64");
        let cdsprpc_dir = format!("{hex_sdk}/ipc/fastrpc/remote/ship/android_aarch64");
        println!("cargo:rustc-link-search=native={rpcmem_dir}");
        println!("cargo:rustc-link-search=native={cdsprpc_dir}");
        // rpcmem.a is a non-standard archive name (not librpcmem.a). Use an
        // explicit -l:rpcmem.a directive so the GNU/Clang linker accepts it.
        println!("cargo:rustc-link-arg=-l:rpcmem.a");
        println!("cargo:rustc-link-lib=dylib=cdsprpc");
        println!("cargo:rerun-if-env-changed=SP_HEX_BACKEND_DIR");
        println!("cargo:rerun-if-env-changed=HEXAGON_SDK_ROOT");
        println!("cargo:warning=WIRE-HEX: linking libsp_hex_daemon_backend.a + libcdsprpc.so + rpcmem.a");
    }

    // ── Sprint WIRE-CPU: link the daemon-callable CPU AVX-512 backend ──
    //
    // When the `wire_cpu_backend` Cargo feature is on, link the standalone
    // static lib built by `tools/sp_daemon/build-host-cpu-backend.bat`:
    //   build-host-cpu-backend/Release/sp_cpu_daemon_backend.lib   (MSVC)
    //   build-host-cpu-backend/sp_cpu_daemon_backend.lib           (MSVC, single-config)
    //   build-host-cpu-backend/libsp_cpu_daemon_backend.a          (MinGW / Linux)
    //
    // That archive contains:
    //   - src/backends/cpu/cpu_overlay.c     (matmul/embed_row/dot_f32/rmsnorm/etc.)
    //   - src/backends/cpu/cpu_forward.c     (qwen3_forward_cpu_impl, renamed)
    //   - src/backends/cpu/cpu_gemma3.c      (gemma3_forward_cpu_impl, renamed)
    //   - src/backends/cpu/cpu_generate.c    (qwen3_generate_cpu_impl, renamed)
    //   - src/backends/cpu/avx512/avx512_dispatch.c (g_avx512_caps CPU feature probe)
    //   - On non-MSVC: avx512_{vnni,spinor,ifma,ternlog,persist}.c (intrinsics)
    //   - tools/sp_daemon/c_backend_cpu/sp_daemon_cpu_glue.c (the L1 §6 dispatcher)
    //
    // The math-core archives are linked by the MODULES loop above; the engine
    // cpu sources were renamed at compile time to avoid duplicate-symbol with
    // math-core's reference forwards (see CMakeLists-cpu.txt header).
    //
    // HOST target — no FastRPC libs needed (unlike WIRE-HEX which links cdsprpc).
    // Build order: scripts/build/build-cpu.bat first (math-core .lib archives),
    // then build-host-cpu-backend.bat, then
    // `cargo build --features wire_cpu_backend --release`.
    let wire_cpu = env::var("CARGO_FEATURE_WIRE_CPU_BACKEND").is_ok();
    if wire_cpu {
        let cpu_lib_dir = env::var("SP_CPU_BACKEND_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| engine_root.join("build-host-cpu-backend"));

        // CMake puts the lib in either {dir}/Release/{name}.lib (multi-config
        // generators like Visual Studio) or {dir}/{name}.lib (Ninja, single-config).
        // We don't know which generator was used; probe and pick.
        let cpu_archive_msvc_release = cpu_lib_dir.join("Release").join("sp_cpu_daemon_backend.lib");
        let cpu_archive_msvc_flat    = cpu_lib_dir.join("sp_cpu_daemon_backend.lib");
        let cpu_archive_unix         = cpu_lib_dir.join("libsp_cpu_daemon_backend.a");

        let (search_dir, link_name): (PathBuf, &str) = if cpu_archive_msvc_flat.exists() {
            (cpu_lib_dir.clone(), "sp_cpu_daemon_backend")
        } else if cpu_archive_msvc_release.exists() {
            (cpu_lib_dir.join("Release"), "sp_cpu_daemon_backend")
        } else if cpu_archive_unix.exists() {
            (cpu_lib_dir.clone(), "sp_cpu_daemon_backend")
        } else {
            panic!(
                "WIRE-CPU: backend archive not found under {} (tried {{,Release/}}sp_cpu_daemon_backend.lib + libsp_cpu_daemon_backend.a) — run build-host-cpu-backend.bat first",
                cpu_lib_dir.display()
            );
        };
        println!("cargo:rustc-link-search=native={}", search_dir.display());
        if target_env == "msvc" {
            // Same MSVC quirk as the math-core libs above: use a full-path
            // -link arg so the binary link step receives the lib regardless
            // of intermediate /LIBPATH lookup gaps.
            let lib_path = search_dir.join(format!("{link_name}.lib"));
            println!("cargo:rustc-link-arg={}", lib_path.display());
        } else {
            println!("cargo:rustc-link-lib=static={link_name}");
        }
        println!("cargo:rerun-if-env-changed=SP_CPU_BACKEND_DIR");
        println!("cargo:warning=WIRE-CPU: linking sp_cpu_daemon_backend.lib (host CPU AVX2/AVX-512 backend)");
    }

    // ── Sprint WIRE-CUDA: link the daemon-callable CUDA PTX backend ──
    //
    // When the `wire_cuda_backend` Cargo feature is on (host x86_64; CUDA is
    // host-only), link the standalone static lib built by
    // `tools/sp_daemon/build-host-cuda-backend.bat`:
    //   build-host-cuda-backend/sp_cuda_daemon_backend.lib  (MSVC)
    //   build-host-cuda-backend/libsp_cuda_daemon_backend.a (GNU)
    //
    // That archive contains:
    //   - src/backends/cuda/cuda_backend.cu   (device mgmt + error mapping)
    //   - src/backends/cuda/cuda_forward.cu   (gemma3_forward_cuda + qwen3_forward_cuda)
    //   - tools/sp_daemon/c_backend_cuda/sp_daemon_cuda_glue.c (the §6 dispatcher)
    //
    // Plus CUDA runtime libs (cudart + cublas) discovered via CUDA_PATH.
    //
    // Build order: build-host-cuda-backend.bat first (which itself calls the
    // engine's env-cuda.bat for vcvars64 + CUDA on PATH + SP_CUDA_ARCH), then
    // `cargo build --features wire_cuda_backend --release`.
    let wire_cuda = env::var("CARGO_FEATURE_WIRE_CUDA_BACKEND").is_ok();
    if wire_cuda {
        let cuda_lib_dir = env::var("SP_CUDA_BACKEND_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| engine_root.join("build-host-cuda-backend"));
        // MSVC: sp_cuda_daemon_backend.lib (no "lib" prefix).
        // GNU/Linux: libsp_cuda_daemon_backend.a.
        let cuda_archive = if target_env == "msvc" {
            cuda_lib_dir.join("sp_cuda_daemon_backend.lib")
        } else {
            cuda_lib_dir.join("libsp_cuda_daemon_backend.a")
        };
        if !cuda_archive.exists() {
            panic!(
                "WIRE-CUDA: CUDA backend archive not found at {} — run build-host-cuda-backend.bat first",
                cuda_archive.display()
            );
        }
        println!("cargo:rustc-link-search=native={}", cuda_lib_dir.display());
        if target_env == "msvc" {
            // Match the MSVC pattern used for math-core libs above (absolute path,
            // bypasses /LIBPATH lookup quirks when the lib doesn't reference any
            // symbol the host crate uses directly).
            println!("cargo:rustc-link-arg={}", cuda_archive.display());
        } else {
            println!("cargo:rustc-link-lib=static=sp_cuda_daemon_backend");
        }

        // CUDA runtime libs. CUDA_PATH is set by scripts/env/env-cuda.bat
        // (NVIDIA's standard env var); fall back to the canonical
        // SP_PIN_CUDA_ROOT pin used by the engine for build reproducibility.
        let cuda_root = env::var("CUDA_PATH")
            .or_else(|_| env::var("SP_PIN_CUDA_ROOT"))
            .unwrap_or_else(|_| {
                // Default for VS2019 BT + CUDA 13.2 host build.
                String::from("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2")
            });
        if target_env == "msvc" {
            // Use rustc-link-arg with absolute .lib paths (same bypass the MODULES use
            // above): cargo:rustc-link-lib does NOT propagate to standalone [[bin]]
            // targets on MSVC, leaving cudart/cudadevrt unresolved for the gate/probe
            // bins. cudadevrt carries the __cudaRegister* fatbin glue for the device
            // code embedded in the static backend lib.
            println!("cargo:rustc-link-search=native={cuda_root}/lib/x64");
            for l in ["cudart", "cudadevrt", "cublas", "cublasLt"] {
                println!("cargo:rustc-link-arg={cuda_root}/lib/x64/{l}.lib");
            }
        } else {
            // Linux: cudart + cublas dylibs from CUDA_PATH/lib64.
            println!("cargo:rustc-link-search=native={cuda_root}/lib64");
            println!("cargo:rustc-link-lib=dylib=cudart");
            println!("cargo:rustc-link-lib=dylib=cublas");
            println!("cargo:rustc-link-lib=dylib=cublasLt");
        }
        println!("cargo:rerun-if-env-changed=SP_CUDA_BACKEND_DIR");
        println!("cargo:rerun-if-env-changed=CUDA_PATH");
        println!("cargo:rerun-if-env-changed=SP_PIN_CUDA_ROOT");
        println!("cargo:warning=WIRE-CUDA: linking sp_cuda_daemon_backend + cudart + cublas + cublasLt");
    }
}
