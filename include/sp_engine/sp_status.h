/* sp_status.h — the frozen L1 ABI status surface (PPT-LAT-L1-ABI-v0 Appendix A
 * section 7, tag lat-phase2-contract-frozen). The CUDA backend wraps any
 * cudaError_t / cublasStatus_t into SP_ECUDA and stashes the human-readable
 * detail in a thread-local string retrievable via sp_last_error().
 *
 * This is the ABI's signed-int status enum: SP_OK = 0, negative = error,
 * positive reserved for future soft signals. Defined here (engine-side) so the
 * backends share one definition until the full sp_session driver lands; the
 * values match the contract exactly and must not be renumbered.
 */
#ifndef SP_ENGINE_SP_STATUS_H
#define SP_ENGINE_SP_STATUS_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    SP_OK                =   0,

    /* Generic */
    SP_ENOMEM            =  -1,
    SP_ECANCEL           =  -2,
    SP_EBADARG           =  -3,
    SP_EBADSTATE         =  -4,
    SP_EUNSUPPORTED      =  -5,
    SP_EIO               =  -6,

    /* Model load / arch */
    SP_EBADFORMAT        = -10,
    SP_EBADARCH          = -11,
    SP_ETOKENIZER_HASH   = -12,
    SP_EVOCAB            = -13,

    /* Discrete-algebra layer — the "lost the algebraic invariant" surface */
    SP_ESPINOR_BADBLOCK  = -20,
    SP_EVHT2_DOMAIN      = -21,
    SP_EMOBIUS_PERM      = -22,
    SP_EOK_NORM          = -23,
    SP_EFROBENIUS_QUANT  = -24,
    SP_ENTT_OVERFLOW     = -25,
    SP_ERING_DEGREE      = -26,

    /* Lattice / framework features */
    SP_ESIEVE_FULL       = -30,
    SP_EARM_BANK_FULL    = -31,
    SP_EDOMINANCE_CYCLE  = -32,
    SP_ECONTEXT_FULL     = -33,

    /* Backend */
    SP_ECUDA             = -40,    /* wraps any cudaError_t; sp_last_error has detail */
    SP_EVULKAN           = -41,
    SP_EHVX              = -42,
    SP_EBACKEND_OOM      = -43
} sp_status;

/* Thread-local error detail set by the last failing L1/backend call on this
 * thread. Pointer valid until the next such call on the same thread. */
const char *sp_last_error(void);

#ifdef __cplusplus
}
#endif
#endif /* SP_ENGINE_SP_STATUS_H */
