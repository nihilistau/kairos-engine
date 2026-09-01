/* sieve_ffi.rs — Manual FFI bindings for sp_sieve (Phase 5 PoUW).
 *
 * Mirrors the types and function from include/sp/sp_sieve.h and
 * include/sp/sp_status.h.  Kept small and explicit to avoid adding
 * a second bindgen header invocation in build.rs.
 */

/// 64-byte packed KSTE tree (mirrors sp_kste_tree_t).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SpKsteTree {
    pub bytes: [u8; 64],
}

/// Sieve-fold event emitted by sp_sieve_evaluate (mirrors sp_sieve_event_t).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SpSieveEvent {
    pub sig:      SpKsteTree,
    pub seq_hash: [u8; 32],
    pub round:    u32,
}

extern "C" {
    /// Evaluate a batch of KSTE candidates against a Pareto frontier.
    /// Returns 0 (SP_OK), -30 (SP_ESIEVE_FULL), or -3 (SP_EBADARG).
    pub fn sp_sieve_evaluate(
        candidates:   *const SpKsteTree,
        n:            usize,
        frontier:     *mut SpKsteTree,
        frontier_n:   *mut usize,
        frontier_cap: usize,
        events_out:   *mut SpSieveEvent,
        n_events_out: *mut usize,
    ) -> i32;
}
