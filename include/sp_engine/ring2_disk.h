/* ring2_disk.h — C2.1 Step 2b: physical Optane-backed Ring-2 byte store.
 *
 * Swaps the Step-2a mock RAM Ring-2 for two real files (K, V) on a fast
 * byte-addressable drive (Optane E:/F:). Spilled (layer,token) KV blocks are
 * written at a deterministic 4 KB-aligned offset; the recall router fetches the
 * scattered top-B blocks back with blocking reads.
 *
 * v0 (this file): SYNCHRONOUS blocking reads, plumbed through the overlapped-
 * offset mechanism so they are thread-safe when issued concurrently from the OMP
 * attention head loop (each call carries its own offset + caller-supplied event,
 * no shared file pointer). Opened FILE_FLAG_NO_BUFFERING so reads hit the device,
 * NOT the OS page cache — otherwise a <RAM-sized KV file is served from DRAM and
 * the latency measurement is theater. The deterministic offset is already a
 * multiple of the 4 KB block, satisfying NO_BUFFERING's sector alignment.
 *
 * v1 (later): batch the scattered recall reads as outstanding overlapped requests
 * to exploit the device's random-read queue depth (read-with-read concurrency —
 * the only latency-hiding lever available, since the decode thread has no compute
 * to overlap the I/O against). Not implemented here (one variable at a time).
 */
#ifndef SP_ENGINE_RING2_DISK_H
#define SP_ENGINE_RING2_DISK_H
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ring2_disk ring2_disk;

/* Per-thread read scratch: an aligned 4 KB (block_bytes) buffer + an OS event/
 * overlapped handle, so concurrent reads from the head loop never share state.
 * Allocate one per OMP thread (ring2_disk_scratch_new) and free at thread exit. */
typedef struct ring2_scratch ring2_scratch;

/* Open (create/truncate) the K and V store files under `dir` (e.g. "E:\\"),
 * each presized to `bytes_per_file`. `block_bytes` is the spill granularity
 * (KVD*sizeof(float)); must be a multiple of the device sector size for the
 * unbuffered path. Returns NULL on failure. */
ring2_disk *ring2_disk_open(const char *dir, size_t bytes_per_file, size_t block_bytes);

/* Spill one block to file `which` (0=K, 1=V) at byte offset `off` (a multiple of
 * block_bytes). `src` must be block_bytes long; it is copied through an internal
 * aligned bounce buffer (NO_BUFFERING requires an aligned source). Single-writer
 * (called from the sequential prefill/decode loop, before the parallel region).
 * Returns 0 on success. */
int ring2_disk_write(ring2_disk *r, int which, size_t off, const void *src);

/* Blocking read of one block from file `which` at byte offset `off` into the
 * scratch's aligned buffer; on success returns a pointer to block_bytes of data
 * (valid until the next read on the same scratch), or NULL on failure. Thread-safe
 * across distinct scratches. */
const void *ring2_disk_read(ring2_disk *r, int which, size_t off, ring2_scratch *sc);

ring2_scratch *ring2_disk_scratch_new(ring2_disk *r);
void ring2_disk_scratch_free(ring2_scratch *sc);

/* v1b: batched async read via an I/O Completion Port. Submit all `n` requests as
 * overlapped reads (no 64-handle WaitForMultipleObjects cap), then drain the
 * completion port — exploits the device's random-read queue depth. Each `dst`
 * must be sector-aligned (use ring2_disk_alloc_aligned) and hold block_bytes.
 * Blocking: returns only when the whole batch has landed. 0 on success.
 * (POSIX fallback: serial pread loop.) */
typedef struct { int which; size_t off; void *dst; } ring2_req;
int ring2_disk_read_batch(ring2_disk *r, const ring2_req *reqs, int n);

/* Sector-aligned (4 KB) allocation for direct-IO staging buffers. */
void *ring2_disk_alloc_aligned(size_t bytes);
void  ring2_disk_free_aligned(void *p);

size_t ring2_disk_block_bytes(const ring2_disk *r);
/* Aggregate read counters for the latency report (total blocking reads + nanoseconds). */
void ring2_disk_stats(const ring2_disk *r, unsigned long long *n_reads, double *read_seconds);
void ring2_disk_close(ring2_disk *r);

#ifdef __cplusplus
}
#endif
#endif /* SP_ENGINE_RING2_DISK_H */
