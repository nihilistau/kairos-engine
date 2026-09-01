//! Safe Rust wrappers around the L1 opaque handles (sp_model, sp_session).
//!
//! Ownership rules from the L1 ABI contract (sp_l1.h):
//!   - sp_model is immutable after load → Send + Sync (many sessions per model).
//!   - sp_session is single-thread mutable → Send but NOT Sync.
//!   - cancel_flag storage must outlive sp_session_destroy → held in Arc.
use std::ffi::{CStr, CString};
use std::os::raw::c_int;
use std::ptr;
use std::sync::atomic::AtomicI32;
use std::sync::Arc;

use crate::ffi;

// ── SpModel ────────────────────────────────────────────────────────────────

pub struct SpModel(*mut ffi::sp_model);

// SAFETY: sp_model is immutable after sp_model_load returns.
unsafe impl Send for SpModel {}
unsafe impl Sync for SpModel {}

impl Drop for SpModel {
    fn drop(&mut self) {
        unsafe { ffi::sp_model_unload(self.0) };
    }
}

impl SpModel {
    pub fn load(model_path: &str, tok_path: &str) -> Result<Self, String> {
        let model_c = CString::new(model_path).map_err(|e| e.to_string())?;
        let tok_c = CString::new(tok_path).map_err(|e| e.to_string())?;
        let mut ptr: *mut ffi::sp_model = ptr::null_mut();
        let status =
            unsafe { ffi::sp_model_load(model_c.as_ptr(), tok_c.as_ptr(), &mut ptr) };
        if status == ffi::sp_status_SP_OK {
            Ok(SpModel(ptr))
        } else {
            let detail = unsafe { std::ffi::CStr::from_ptr(ffi::sp_last_error()) }
                .to_string_lossy()
                .into_owned();
            Err(format!("sp_model_load → status={status}: {detail}"))
        }
    }

    pub fn as_ptr(&self) -> *const ffi::sp_model {
        self.0 as *const _
    }

    pub fn arch_info(&self) -> Result<ffi::sp_arch_info, String> {
        let mut info: ffi::sp_arch_info = unsafe { std::mem::zeroed() };
        let status = unsafe { ffi::sp_model_arch(self.0, &mut info) };
        if status == ffi::sp_status_SP_OK {
            Ok(info)
        } else {
            Err(format!("sp_model_arch → status={status}"))
        }
    }

    /// Returns a slice into the tokenizer mmap held by the model.
    /// The slice lifetime is tied to &self — safe because SpModel owns the mmap.
    pub fn tokenizer_blob(&self) -> Result<&[u8], String> {
        let mut size: u64 = 0;
        let ptr = unsafe { crate::ffi::sp_model_tokenizer_blob(self.0 as *const _, &mut size) };
        if ptr.is_null() {
            return Err("sp_model_tokenizer_blob returned NULL".to_string());
        }
        Ok(unsafe { std::slice::from_raw_parts(ptr as *const u8, size as usize) })
    }
}

// ── SpSession ──────────────────────────────────────────────────────────────

pub struct SpSession {
    ptr: *mut ffi::sp_session,
    /// Keeps the cancel_flag allocation live for the session's lifetime
    /// (L1 contract: cancel_flag storage must outlive sp_session_destroy).
    _cancel_flag: Arc<AtomicI32>,
}

// SAFETY: session is Send (single thread at a time, enforced by Mutex<SpSession>
// in AppState). NOT Sync — callers must not share &SpSession across threads.
unsafe impl Send for SpSession {}

impl Drop for SpSession {
    fn drop(&mut self) {
        unsafe { ffi::sp_session_destroy(self.ptr) };
    }
}

impl SpSession {
    /// Create a session over an already-loaded model.
    ///
    /// `cancel_flag` is L2-owned. The raw pointer is passed to L1 and must
    /// remain valid until this `SpSession` is dropped. The `Arc` guarantees that.
    ///
    /// cancel_flag type: `volatile int *` in C (sp_l1.h:128).
    /// Rust side: `Arc<AtomicI32>` — AtomicI32 has the same layout as c_int;
    /// L1 only reads the flag with relaxed ordering (header guarantee), so the
    /// cast to *mut c_int is sound.
    pub fn create(model: &SpModel, cancel_flag: Arc<AtomicI32>) -> Result<Self, String> {
        let cfg = ffi::sp_session_config {
            max_context: 0,       // 0 = arch default
            deterministic: 1,     // bit-exact reductions for CORE
            arm_bank_kb: 0,
            sieve_capacity: 0,
            flags: 0,
            precision_override: 0, // defer to arch_info.preferred_precision
        };
        let mut ptr: *mut ffi::sp_session = ptr::null_mut();
        // SAFETY: cancel_flag Arc keeps the allocation alive until drop.
        let cancel_raw = cancel_flag.as_ptr() as *mut std::os::raw::c_int;
        let status = unsafe {
            ffi::sp_session_create(model.as_ptr(), &cfg, cancel_raw, &mut ptr)
        };
        if status == ffi::sp_status_SP_OK {
            Ok(SpSession { ptr, _cancel_flag: cancel_flag })
        } else {
            let detail = unsafe { std::ffi::CStr::from_ptr(ffi::sp_last_error()) }
                .to_string_lossy()
                .into_owned();
            Err(format!("sp_session_create → status={status}: {detail}"))
        }
    }

    /// Current sequence position (number of tokens consumed so far).
    /// Reads sp_session_position — the FFI proof call in /v1/metrics.
    pub fn position(&self) -> Result<usize, String> {
        let mut pos: usize = 0;
        let status = unsafe { ffi::sp_session_position(self.ptr, &mut pos) };
        if status == ffi::sp_status_SP_OK {
            Ok(pos)
        } else {
            Err(format!("sp_session_position → status={status}"))
        }
    }

    /// Deep-copy this session into a new independent session (the spec-decode fork).
    /// `cancel_flag` is an L2-owned atomic for the child session.
    pub fn clone_session(&self, cancel_flag: Arc<AtomicI32>) -> Result<Self, String> {
        let cancel_raw = cancel_flag.as_ptr() as *mut c_int;
        let mut out: *mut ffi::sp_session = ptr::null_mut();
        let status = unsafe { ffi::sp_session_clone(self.ptr, cancel_raw, &mut out) };
        if status == ffi::sp_status_SP_OK {
            Ok(SpSession { ptr: out, _cancel_flag: cancel_flag })
        } else {
            let detail = unsafe { CStr::from_ptr(ffi::sp_last_error()) }
                .to_string_lossy()
                .into_owned();
            Err(format!("sp_session_clone → status={status}: {detail}"))
        }
    }

    /// Prefill a chunk of tokens; writes the last token's logits into `logits`.
    pub fn prefill_chunk(&mut self, tokens: &[i32], logits: &mut [f32]) -> Result<(), String> {
        let status = unsafe {
            ffi::sp_prefill_chunk(
                self.ptr,
                tokens.as_ptr(),
                tokens.len(),
                logits.as_mut_ptr(),
                logits.len(),
            )
        };
        if status == ffi::sp_status_SP_OK {
            Ok(())
        } else {
            let detail = unsafe { CStr::from_ptr(ffi::sp_last_error()) }
                .to_string_lossy()
                .into_owned();
            Err(format!("sp_prefill_chunk → status={status}: {detail}"))
        }
    }

    /// Decode one token; overwrites `logits` with the next-position logits.
    pub fn decode_step(&mut self, token: i32, logits: &mut [f32]) -> Result<(), String> {
        let status = unsafe {
            ffi::sp_decode_step(self.ptr, token, logits.as_mut_ptr(), logits.len())
        };
        if status == ffi::sp_status_SP_OK {
            Ok(())
        } else {
            let detail = unsafe { CStr::from_ptr(ffi::sp_last_error()) }
                .to_string_lossy()
                .into_owned();
            Err(format!("sp_decode_step → status={status}: {detail}"))
        }
    }

    /// §4-NTT Sprint NTT.5b — escape hatch to call L1 functions that aren't
    /// otherwise wrapped on the safe surface. Currently used only by
    /// `sp_session_register_compute_backend` in daemon.rs at startup.
    ///
    /// Safety: the returned pointer is only valid while &mut self is held
    /// (i.e. for the duration of the borrow); the SpSession Drop calls
    /// sp_session_destroy. Callers must NOT retain the pointer past the
    /// borrow's lifetime, and must NOT call any L1 fn that mutates session
    /// state from another thread concurrently.
    pub fn raw_ptr(&mut self) -> *mut ffi::sp_session {
        self.ptr
    }

    /// Roll back n_tokens positions in the KV cache (O(1) ring-pointer decrement).
    ///
    /// Corollary T8.1: state at P−n after rewind from P is byte-identical to
    /// state at P−n having never visited P. Called on the draft session when the
    /// target rejects at position k < P, with n = P − 1 − k.
    pub fn rewind(&mut self, n_tokens: usize) -> Result<(), String> {
        let status = unsafe { ffi::sp_session_rewind(self.ptr, n_tokens) };
        if status == ffi::sp_status_SP_OK {
            Ok(())
        } else {
            let detail = unsafe { CStr::from_ptr(ffi::sp_last_error()) }
                .to_string_lossy()
                .into_owned();
            Err(format!("sp_session_rewind({n_tokens}) → status={status}: {detail}"))
        }
    }
}
