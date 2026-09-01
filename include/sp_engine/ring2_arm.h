/* ring2_arm.h — Stage C: the engine's Optane NO_BUFFERING+IOCP Ring-2 store
 * wrapped as the math-core ARM backend (sp/arm.h sp_arm_ring2_backend) and
 * registered through the L1 hook, so the CANONICAL math-core decode
 * (qwen3_generate_kv / qwen3_ppl_decode — single source of truth) drives the
 * physical Optane drive. The engine supplies what the portable core must not
 * know about: sector-aligned landing buffers (FILE_FLAG_NO_BUFFERING) via the
 * alloc_aligned hook, and queue-depth amortization via read_batch (IOCP). */
#ifndef SP_ENGINE_RING2_ARM_H
#define SP_ENGINE_RING2_ARM_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Open the Optane store and register it as the ARM Ring-2 backend.
 *   dir            store directory (e.g. "F:\\"); the K/V files live here.
 *   bytes_per_file presized capacity per stream (>= n_layers*P*KVD*4 of the
 *                  largest run this process will serve).
 *   block_bytes    spill granularity == KVD*sizeof(float) of the model
 *                  (must be sector-multiple; 4096 for Qwen3-0.6B).
 * Returns 0 on success. The store stays open until _unregister. */
int sp_ring2_optane_register(const char *dir, size_t bytes_per_file, size_t block_bytes);

/* Dual-block-size variant (the NTT fusion): the K stream carries dual-prime
 * residue blocks (NKV*2N*u32) while V stays f32 (KVD*4) — two independent
 * stores (distinct filename prefixes under `dir`), routed by stream. */
int sp_ring2_optane_register2(const char *dir,
                              size_t bytes_k, size_t blk_k,
                              size_t bytes_v, size_t blk_v);

/* Split-device variant: K-stream store under dir_k, V-stream store under
 * dir_v — two independent NVMe controllers / IOCP queues. Beast Canyon
 * topology: K (heavy 8 KB residue stream) on the CPU-attached slot (F:),
 * V (4 KB f32) on the PCH-attached drive (E:). */
int sp_ring2_optane_register_split(const char *dir_k, const char *dir_v,
                                   size_t bytes_k, size_t blk_k,
                                   size_t bytes_v, size_t blk_v);

/* Env-driven variant: SP_RING2_OPTANE_DIR (required to act; absent => no-op 1),
 * SP_RING2_OPTANE_DIR_V (V-stream device; default = DIR),
 * SP_RING2_OPTANE_BYTES[_K/_V] (per-stream presize; default 1 GiB),
 * SP_RING2_OPTANE_BLOCK[_K/_V] (fusion dual sizes; default 4096). */
int sp_ring2_optane_register_env(void);

/* Unregister + close the store (prints the read-latency stats). Safe if not registered. */
void sp_ring2_optane_unregister(void);

#ifdef __cplusplus
}
#endif
#endif /* SP_ENGINE_RING2_ARM_H */
