//! §3-HX Sprint A — FastRPC bridge for the Hexagon V69 cDSP.
//!
//! Dynamic loader for `libcdsprpc.so` (Android aarch64; userspace FastRPC).
//! Admission via `DSPRPC_CONTROL_UNSIGNED_MODULE` (Path B per
//! `reference-signed-pd-developer-path`) — Sam's S22U has no testsig on
//! `/vendor/etc/`, so signed-PD admission (Path A) is unavailable.
//!
//! Pattern source: `remote.h:793,803,811,840` (Hexagon SDK 5.5.6.0).
//! Prior-cohort validation: `shannon_prime_hexagon.c:93-100,148-154`.
//!
//! This module is compiled only on `target_os = "android"`. Host builds
//! (Windows / Linux x86) skip it via #[cfg].

#![cfg(target_os = "android")]

use std::ffi::{c_int, c_void, CString};
use std::os::raw::c_char;

use libloading::{Library, Symbol};

// ── Constants (from remote.h) ──────────────────────────────────────────────

/// CDSP_DOMAIN_ID — V69 cDSP on Snapdragon. remote.h:125.
pub const CDSP_DOMAIN_ID: c_int = 3;

/// DSPRPC_CONTROL_UNSIGNED_MODULE — session_control req ID. remote.h:641.
pub const DSPRPC_CONTROL_UNSIGNED_MODULE: u32 = 2;

/// AEE_ERPC = AEE_EOFFSET (0x80000400 on hex) + 0x200.
/// Surfaced to host as 0x80000600 on signature mismatch / admission failure.
pub const AEE_ERPC: c_int = 0x8000_0600u32 as i32;

// ── Wire types (#[repr(C)] from remote.h:161-210) ──────────────────────────

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RemoteBuf {
    pub pv:   *mut c_void,
    pub nlen: usize,
}

#[repr(C)]
pub union RemoteArg {
    pub buf: RemoteBuf,
    pub h:   u32,
}

/// `struct remote_rpc_control_unsigned_module` — remote.h:423-428.
#[repr(C)]
struct RemoteRpcControlUnsignedModule {
    domain: c_int,
    enable: c_int,
}

// ── Function pointer types ─────────────────────────────────────────────────

type FnSessionControl = unsafe extern "C" fn(req: u32,
                                              data: *mut c_void,
                                              datalen: u32) -> c_int;

// Sprint A uses remote_handle64 (multi-domain) per SDK S22U pattern —
// matches the qaic-emitted "<iface>_skel_handle_invoke" symbol.  remote_handle
// (single-domain, u32) is a separate ABI.
type FnHandleOpen = unsafe extern "C" fn(name: *const c_char,
                                          ph: *mut u64) -> c_int;

type FnHandleInvoke = unsafe extern "C" fn(h: u64,
                                            scalars: u32,
                                            pra: *mut RemoteArg) -> c_int;

type FnHandleClose = unsafe extern "C" fn(h: u64) -> c_int;

// rpcmem_* from libcdsprpc.so (rpcmem.h:186,216,223).
// `int size` per the header; max single allocation is 2 GB. For >2 GB use
// rpcmem_alloc2 (not wired in Sprint B).
type FnRpcMemAlloc = unsafe extern "C" fn(heapid: c_int, flags: u32, size: c_int) -> *mut c_void;
type FnRpcMemFree  = unsafe extern "C" fn(p: *mut c_void);

// ── rpcmem constants (rpcmem.h) ────────────────────────────────────────────

/// V69 cDSP via SMMU (non-contiguous physical memory). rpcmem.h:89.
pub const RPCMEM_HEAP_ID_SYSTEM: c_int = 25;
/// ION_FLAG_CACHED equivalent. rpcmem.h:52.
pub const RPCMEM_DEFAULT_FLAGS:  u32 = 1;
/// Pre-map at allocation time; recommended for latency-critical FastRPC calls.
/// rpcmem.h:62.
pub const RPCMEM_TRY_MAP_STATIC: u32 = 0x0400_0000;

// ── Errors ──────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum SpErr {
    /// `Library::new("libcdsprpc.so")` failed.
    LibLoad(String),
    /// `Library::get(<symbol>)` failed for the named symbol.
    Symbol(&'static str),
    /// `remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, ...)` returned non-zero.
    UnsignedPdReject(c_int),
    /// `remote_handle_open(...)` returned 0x80000600 (AEE_ERPC).
    ///
    /// Per `reference-signed-pd-developer-path`, five usual causes:
    ///   1. Skel not signed AND no Path B/C admission negotiated
    ///   2. Skel signed with wrong dev key
    ///   3. Skel pushed to path NOT in ADSP_LIBRARY_PATH
    ///   4. vendor.fastrpc.process.attrs=0x8 forces Unsigned Sandbox
    ///   5. Skel built for wrong DSP arch (-mv66 vs V69 device)
    SignatureMismatch(c_int),
    /// `remote_handle_open(...)` returned other non-zero.
    HandleOpen(c_int),
    /// `remote_handle_invoke(...)` returned non-zero.
    Invoke(c_int),
    /// `rpcmem_alloc(...)` returned NULL for the requested size.
    RpcMemAlloc(usize),
    /// Diagnostic error from a higher-level consumer (Sprint J model
    /// loader: missing tensor, dtype mismatch, file IO, etc.).
    Other(String),
}

// ── Scalars helper (REMOTE_SCALARS_MAKE — remote.h:113) ────────────────────

/// Build the `dwScalars` u32 for `remote_handle_invoke`.
///
/// Layout: `(method:5 << 24) | (n_in:8 << 16) | (n_out:8 << 8)`.
/// `method` = IDL method index (qaic-generated; 0 for the first method).
/// `n_in`   = number of input buffers in `args` (counted from front).
/// `n_out`  = number of output buffers in `args` (counted after inputs).
/// `args.len()` MUST equal `n_in + n_out`.
pub fn make_scalars(method: u32, n_in: u32, n_out: u32) -> u32 {
    ((method & 0x1f) << 24) | ((n_in & 0xff) << 16) | ((n_out & 0xff) << 8)
}

// ── FastRpcSession ─────────────────────────────────────────────────────────

/// One FastRPC session = one loaded `libcdsprpc.so` + one open `remote_handle`.
///
/// Lifecycle: `new` → load libcdsprpc.so → resolve symbols →
///   `remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE)` →
///   `remote_handle_open(skel_uri)` → return.
/// `Drop` calls `remote_handle_close` (best-effort; errors logged not panicked).
pub struct FastRpcSession {
    // Keep the library loaded for the lifetime of the session — symbol
    // pointers below are only valid while `_lib` is held.
    _lib:           Library,
    fn_invoke:      FnHandleInvoke,
    fn_close:       FnHandleClose,
    fn_rpcmem_alloc: FnRpcMemAlloc,
    fn_rpcmem_free:  FnRpcMemFree,
    handle:         u64,
}

impl FastRpcSession {
    /// Open an Unsigned-PD FastRPC session and load `skel_uri`.
    ///
    /// `skel_uri` is the qaic-generated URI plus the domain selector,
    /// e.g. `"file:///libshannonprime_echo_skel.so?_dom=cdsp"`.
    ///
    /// Per Path B: enables Unsigned PD on CDSP_DOMAIN_ID=3 via
    /// `remote_session_control` BEFORE the handle open. If the device's
    /// libcdsprpc.so version is old enough that the session-control symbol
    /// is absent, the loader returns `SpErr::Symbol("remote_session_control")`.
    pub fn new(skel_uri: &str) -> Result<Self, SpErr> {
        // 1. Load libcdsprpc.so. On aarch64-android the linker finds it via
        //    /vendor/lib64 / /system/lib64 by default.
        let lib = unsafe {
            Library::new("libcdsprpc.so")
                .map_err(|e| SpErr::LibLoad(format!("libcdsprpc.so: {e}")))?
        };

        // 2. Resolve the four symbols we need.
        let fn_session: Symbol<FnSessionControl> = unsafe {
            lib.get(b"remote_session_control\0")
                .map_err(|_| SpErr::Symbol("remote_session_control"))?
        };
        let fn_open: Symbol<FnHandleOpen> = unsafe {
            lib.get(b"remote_handle64_open\0")
                .map_err(|_| SpErr::Symbol("remote_handle64_open"))?
        };
        let fn_invoke: Symbol<FnHandleInvoke> = unsafe {
            lib.get(b"remote_handle64_invoke\0")
                .map_err(|_| SpErr::Symbol("remote_handle64_invoke"))?
        };
        let fn_close: Symbol<FnHandleClose> = unsafe {
            lib.get(b"remote_handle64_close\0")
                .map_err(|_| SpErr::Symbol("remote_handle64_close"))?
        };
        // rpcmem_* are exported from libcdsprpc.so — no separate librpcmem on
        // the device.  Per rpcmem.h:127, when linked to libcdsprpc.so the
        // rpcmem_init/_deinit calls are not required (the .so initializes on
        // first use), so we resolve only alloc + free.
        let fn_rpcmem_alloc: Symbol<FnRpcMemAlloc> = unsafe {
            lib.get(b"rpcmem_alloc\0")
                .map_err(|_| SpErr::Symbol("rpcmem_alloc"))?
        };
        let fn_rpcmem_free: Symbol<FnRpcMemFree> = unsafe {
            lib.get(b"rpcmem_free\0")
                .map_err(|_| SpErr::Symbol("rpcmem_free"))?
        };

        // Bind the resolved symbols to raw fn pointers we can store past
        // the `Symbol` scope.
        let fn_session_raw:      FnSessionControl = *fn_session;
        let fn_open_raw:         FnHandleOpen     = *fn_open;
        let fn_invoke_raw:       FnHandleInvoke   = *fn_invoke;
        let fn_close_raw:        FnHandleClose    = *fn_close;
        let fn_rpcmem_alloc_raw: FnRpcMemAlloc    = *fn_rpcmem_alloc;
        let fn_rpcmem_free_raw:  FnRpcMemFree     = *fn_rpcmem_free;
        // Drop the Symbol wrappers; raw pointers stay valid as long as
        // `lib` is held (and lib moves into Self below).
        drop(fn_session);
        drop(fn_open);
        drop(fn_invoke);
        drop(fn_close);
        drop(fn_rpcmem_alloc);
        drop(fn_rpcmem_free);

        // 3. Enable Unsigned PD on CDSP. Path B admission gate.
        let mut ctrl = RemoteRpcControlUnsignedModule {
            domain: CDSP_DOMAIN_ID,
            enable: 1,
        };
        let rc = unsafe {
            fn_session_raw(
                DSPRPC_CONTROL_UNSIGNED_MODULE,
                &mut ctrl as *mut _ as *mut c_void,
                std::mem::size_of::<RemoteRpcControlUnsignedModule>() as u32,
            )
        };
        if rc != 0 {
            return Err(SpErr::UnsignedPdReject(rc));
        }

        // 4. Open the skel handle (remote_handle64 = u64).
        let uri_c = CString::new(skel_uri)
            .map_err(|_| SpErr::HandleOpen(-1))?;
        let mut handle: u64 = 0;
        let rc = unsafe { fn_open_raw(uri_c.as_ptr(), &mut handle as *mut u64) };
        if rc != 0 {
            return Err(if rc == AEE_ERPC {
                SpErr::SignatureMismatch(rc)
            } else {
                SpErr::HandleOpen(rc)
            });
        }

        Ok(FastRpcSession {
            _lib: lib,
            fn_invoke:        fn_invoke_raw,
            fn_close:         fn_close_raw,
            fn_rpcmem_alloc:  fn_rpcmem_alloc_raw,
            fn_rpcmem_free:   fn_rpcmem_free_raw,
            handle,
        })
    }

    /// Invoke a method on the open handle. `args.len()` must equal
    /// `n_in + n_out` as encoded in `scalars`.
    pub fn invoke(&self, scalars: u32, args: &mut [RemoteArg]) -> Result<(), SpErr> {
        let rc = unsafe { (self.fn_invoke)(self.handle, scalars, args.as_mut_ptr()) };
        if rc == 0 { Ok(()) } else { Err(SpErr::Invoke(rc)) }
    }

    /// §3-HX Sprint B — allocate a zero-copy `DmaBuffer` of exactly `size`
    /// bytes from heap `RPCMEM_HEAP_ID_SYSTEM` (V69 cDSP via SMMU).
    /// Flags = `RPCMEM_DEFAULT_FLAGS | RPCMEM_TRY_MAP_STATIC` (cached +
    /// pre-mapped for low-latency FastRPC calls).
    ///
    /// CRITICAL CONTRACT: the `size` passed here must EXACTLY equal the
    /// IDL `Len` parameter when this buffer is later passed to `invoke`.
    /// Off-by-one (allocating extra "safety pad") → `AEE_EUNSUPPORTED`
    /// silent failure at invoke time per
    /// `reference-hexagon-working-setup` §"Exact rpcmem size MATCH".
    pub fn alloc_dma(&self, size: usize) -> Result<DmaBuffer<'_>, SpErr> {
        if size == 0 || size > i32::MAX as usize {
            return Err(SpErr::RpcMemAlloc(size));
        }
        let flags = RPCMEM_DEFAULT_FLAGS | RPCMEM_TRY_MAP_STATIC;
        let p = unsafe {
            (self.fn_rpcmem_alloc)(RPCMEM_HEAP_ID_SYSTEM, flags, size as c_int)
        };
        if p.is_null() {
            return Err(SpErr::RpcMemAlloc(size));
        }
        Ok(DmaBuffer {
            ptr:      p as *mut u8,
            len:      size,
            fn_free:  self.fn_rpcmem_free,
            _phantom: std::marker::PhantomData,
        })
    }
}

/// §3-HX Sprint B — RPC-memory backed buffer. Backing is an ION dma-buf
/// allocated by `libcdsprpc.so::rpcmem_alloc` — zero-copy across the
/// ARM/DSP boundary (FastRPC sees these as an ION fd and skips the
/// marshal copy).
///
/// Drop calls `rpcmem_free`. The owning `FastRpcSession`'s `Library` MUST
/// outlive any `DmaBuffer` it issued — `DmaBuffer` borrows the
/// `rpcmem_free` fn ptr from the session via `PhantomData<&FastRpcSession>`.
pub struct DmaBuffer<'sess> {
    ptr:      *mut u8,
    len:      usize,
    fn_free:  FnRpcMemFree,
    _phantom: std::marker::PhantomData<&'sess FastRpcSession>,
}

impl<'sess> DmaBuffer<'sess> {
    pub fn as_ptr(&self) -> *const u8 { self.ptr }
    pub fn as_mut_ptr(&mut self) -> *mut u8 { self.ptr }
    pub fn len(&self) -> usize { self.len }
    pub fn is_empty(&self) -> bool { self.len == 0 }
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl<'sess> Drop for DmaBuffer<'sess> {
    fn drop(&mut self) {
        unsafe { (self.fn_free)(self.ptr as *mut c_void) };
    }
}

impl Drop for FastRpcSession {
    fn drop(&mut self) {
        // Best-effort close. Errors are logged but cannot be propagated
        // out of Drop. The OS will reap the handle when the process exits
        // regardless.
        let rc = unsafe { (self.fn_close)(self.handle) };
        if rc != 0 {
            eprintln!("[sp-daemon] FastRpcSession::drop: remote_handle_close \
                       returned {rc:#x} (handle={})", self.handle);
        }
    }
}
