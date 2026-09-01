//! NORTHSTAR serve — the qwen36 (Qwen3.6-35B-A3B GDN+MoE hybrid) chat lane.
//!
//! The gemma4 chat path runs on L1 sessions + the resident kvdecode cache; the
//! 35B hybrid instead decodes through the math-core `qwen36_state` persistent
//! decode (O(1) recurrent GDN state + windowed attn K/V, G-MOE-STATE-PARITY)
//! with the GPU hybrid hooks (dense-resident + expert-resident + pinned
//! streaming, G-MOE-GPU4-PINNED 6.073 tok/s / 337x) booted ONCE at daemon start
//! via `sp_q36gpu_boot` (CONTRACT-QWEN36-SERVE S1, engine c12d1ea).
//!
//! v1 scope (the contract's G-QWEN36-SERVE): greedy argmax decode, fresh state
//! per request (prefill-by-stepping, ~0.16 s/token — long histories pay; batch
//! prefill is a pre-scoped follow-up), one turn at a time (`gate` mutex — the
//! GDN/OMP/CUDA-stream plumbing is process-wide). Sampling knobs come after the
//! gate is GREEN.

use std::os::raw::{c_int, c_void};
use std::sync::Mutex;

// qwen36_state_* live in core/forward (sp/model.h) but do NOT carry the `sp_`
// prefix, so they fall outside build.rs's bindgen allowlist_function("sp_.*");
// declared manually here. sp_model_to_qwen36 is bindgen-visible but redeclared
// with an opaque pointer to avoid threading the generated qwen3_model type.
extern "C" {
    fn sp_model_to_qwen36(m: *const crate::ffi::sp_model) -> *mut c_void;
    fn qwen36_state_new(m: *const c_void, max_pos: c_int) -> *mut c_void;
    fn qwen36_state_free(st: *mut c_void);
    fn qwen36_step(m: *const c_void, st: *mut c_void, token: i32, logits: *mut f32) -> c_int;
}

// The one-call GPU boot (dense uploads + expert residency under budget +
// streaming table + hook registration) — compiled into the CUDA backend lib
// only, so it exists only under the wire_cuda_backend feature.
#[cfg(feature = "wire_cuda_backend")]
extern "C" {
    fn sp_q36gpu_boot(m: *const c_void, moe_gb: f64, stream_on: c_int) -> *mut c_void;
}

pub struct Qwen36Lane {
    qm: *mut c_void,
    /// GPU boot handle (kept for the process lifetime; null = CPU-only lane).
    #[allow(dead_code)]
    gpu: *mut c_void,
    pmax: i32,
    pub n_vocab: usize,
    /// Serializes turns: the recurrent state is per-request, but the OMP pool +
    /// CUDA stream + pinned staging buffers are process-wide singletons.
    pub gate: Mutex<()>,
}

// The raw pointers are owned process-wide and only dereferenced under `gate`.
unsafe impl Send for Qwen36Lane {}
unsafe impl Sync for Qwen36Lane {}

impl Qwen36Lane {
    /// Boot the lane once at daemon start (arch_id == SP_ARCH_ID_QWEN36 == 8).
    /// Env: SP_Q36_GPU=1 boots the GPU hybrid (SP_Q36_GPU_MOE_GB expert budget,
    /// SP_Q36_GPU_STREAM=1 pinned streaming for the rump); SP_Q36_PMAX context.
    pub fn boot(model: &crate::session::SpModel, n_vocab: usize) -> Result<Self, String> {
        let qm = unsafe { sp_model_to_qwen36(model.as_ptr()) };
        if qm.is_null() {
            return Err("sp_model_to_qwen36 returned null (not a qwen36 container?)".into());
        }
        let pmax: i32 = std::env::var("SP_Q36_PMAX").ok()
            .and_then(|s| s.parse().ok()).unwrap_or(4096);

        #[allow(unused_mut)]
        let mut gpu: *mut c_void = std::ptr::null_mut();
        #[cfg(feature = "wire_cuda_backend")]
        if std::env::var("SP_Q36_GPU").ok().as_deref() == Some("1") {
            let moe_gb: f64 = std::env::var("SP_Q36_GPU_MOE_GB").ok()
                .and_then(|s| s.parse().ok()).unwrap_or(9.9);
            let stream_on: c_int = if std::env::var("SP_Q36_GPU_STREAM").ok().as_deref() == Some("0") { 0 } else { 1 };
            gpu = unsafe { sp_q36gpu_boot(qm, moe_gb, stream_on) };
            if gpu.is_null() {
                tracing::warn!("sp_q36gpu_boot failed — qwen36 lane continues CPU-only");
            }
        }
        tracing::info!("qwen36 lane booted: pmax={} gpu={}", pmax, !gpu.is_null());
        Ok(Self { qm, gpu, pmax, n_vocab, gate: Mutex::new(()) })
    }

    /// One greedy turn. Steps the whole prompt (prefill-by-stepping), then
    /// greedy-decodes up to `max_new` tokens. `on_tok` receives each generated
    /// id BEFORE eos filtering of the NEXT step; return false to stop (client
    /// cancel). Returns (generated ids, decode-phase tok/s).
    pub fn run_turn(
        &self,
        prompt: &[i32],
        max_new: usize,
        eos: &[i32],
        mut on_tok: impl FnMut(i32) -> bool,
    ) -> Result<(Vec<i32>, f64), String> {
        if prompt.is_empty() {
            return Err("empty prompt".into());
        }
        if prompt.len() as i32 + max_new as i32 >= self.pmax {
            return Err(format!(
                "prompt {} + max_tokens {} exceeds SP_Q36_PMAX {}",
                prompt.len(), max_new, self.pmax
            ));
        }
        let _g = self.gate.lock().unwrap();
        let st = unsafe { qwen36_state_new(self.qm, self.pmax) };
        if st.is_null() {
            return Err("qwen36_state_new failed".into());
        }
        let mut logits = vec![0f32; self.n_vocab];
        // Prefill: step every prompt token; the LAST step's logits row selects
        // the first generated token.
        for &t in prompt {
            let rc = unsafe { qwen36_step(self.qm, st, t, logits.as_mut_ptr()) };
            if rc != 0 {
                unsafe { qwen36_state_free(st) };
                return Err(format!("qwen36_step rc={rc} during prefill"));
            }
        }
        let mut out = Vec::with_capacity(max_new);
        let t0 = std::time::Instant::now();
        loop {
            // Greedy argmax over the current logits row.
            let mut best = 0usize;
            let mut bv = f32::NEG_INFINITY;
            for (i, &v) in logits.iter().enumerate() {
                if v > bv { bv = v; best = i; }
            }
            let next = best as i32;
            if eos.contains(&next) { break; }
            out.push(next);
            if !on_tok(next) { break; }
            if out.len() >= max_new { break; }
            let rc = unsafe { qwen36_step(self.qm, st, next, logits.as_mut_ptr()) };
            if rc != 0 {
                unsafe { qwen36_state_free(st) };
                return Err(format!("qwen36_step rc={rc} during decode"));
            }
        }
        let dt = t0.elapsed().as_secs_f64();
        unsafe { qwen36_state_free(st) };
        let tokps = if dt > 0.0 { out.len() as f64 / dt } else { 0.0 };
        Ok((out, tokps))
    }
}
