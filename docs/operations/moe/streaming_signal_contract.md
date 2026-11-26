# Streaming Signal Contract for the MoE Grouped GEMMs

**Experimental.** This page is for whoever implements the kernel that **feeds** a grouped GEMM (for example an
expert-parallel dispatch) or **consumes** its output (for example an expert-parallel combine), so that the two run
concurrently instead of one after the other. The kernels stay separate. They share only a few int32 entries in device
memory: the readiness signals.

| Direction | GEMM option | GEMM role | Signal unit | Scheduler |
|---|---|---|---|---|
| producer → GEMM | `streaming_signal_capacity` | consumer | one cluster M tile of A (256 rows) | `MoEStreamingInputBlockSignalTileScheduler` |
| GEMM → consumer | `streaming_output_notify=StreamingOutputNotifyConfig(StreamingOutputNotifyMode.N_CHUNK, ...)` | producer | one N chunk of D columns | `MoEStreamingOutputChunkSignalTileScheduler` |

Kernels: `MoEGroupedGemmBf16Kernel` (`unfused/`, both directions) and `MoEGroupedGemmGluBiasBf16Kernel` (`glu/`,
input direction only), BF16 on SM100-class GPUs. Both directions need `use_dynamic_sched=True`. Every option is off by
default; with the defaults the generated code is the same as without this feature.

## 1. Input direction: producer → GEMM

### Workspace layout

The kernel's workspace (`get_workspace_bytes()`) holds the signal entries. Hand `signal[]` to the producer as a view
of that workspace (zero-copy). Offsets come from `moe_kernel_helpers.streaming_signal_workspace_offsets(kernel)`.

| Byte offset | Contents | Written by | Read by |
|---|---|---|---|
| `desc` | claim counter (1 × int32) | GEMM | GEMM |
| `desc + 4` | `signal[streaming_signal_capacity]` (int32) | **producer** | GEMM |
| `desc + S` | scheduler state | GEMM | GEMM |

- `desc = kernel.get_desc_workspace_bytes()`: 0 for dense weights, TMA descriptors for discrete weights.
- `S = streaming_scheduler_state_offset(capacity) = round_up(4 + 4 * capacity, 128)`, so the producer-written signals
  and the GEMM's own state never share a 128-byte line.
- State size: `4 * (2 * E + 1 + 3 * capacity)` bytes, where E is the local expert count: two int32 entries per expert,
  one claim count and a 3-entry batch record per M tile.

### `signal[t]`

| Value | Meaning |
|---|---|
| `-1` | M tile `t` not ready |
| `t` | M tile `t` complete: its rows `[t * M, (t + 1) * M)` of A (M = cluster tile M), **padding rows included**, are in memory |

- `t` ranges over `[0, padded_offsets[E - 1] / M)`, at most `streaming_signal_capacity` M tiles. Later
  entries stay -1 and the GEMM never waits on them.
- The position is the M tile id. The producer publishes no arrival order; the GEMM picks its own order, running ready
  M tiles in expert-consecutive batches of up to 16 M tiles.
- Readiness is `signal[t] >= 0`. **Reset with `0xFF` bytes (-1), never with zeros**: 0 reads as "M tile 0 ready" and
  every `t >= 0` as ready.

### M tile size and capacity

| Quantity | Rule |
|---|---|
| M tile | The cluster tile M: 256 rows for the 2-CTA 256×256 tile with cluster (2, 1). It must equal the producer's row alignment: each expert's padded row count is a multiple of it, so no M tile spans two experts. |
| `streaming_signal_capacity` | `>= ceil(allocated receive rows / M)`. Derive it from the receive buffer's **allocated** rows, not the routed rows. |
| `padded_offsets` | **Inclusive** cumulative sum of padded rows per local expert: `offs[e]` is expert e's one-past-last row and `offs[E-1]` is the total. An exclusive "start" array passed by mistake shifts every expert by one and looks plausible. |
| A rows | `<= (2^31 - 1) * M`, since `t` is int32. |
| linear tile indices | `capacity × ceil(N / cluster tile N) < 2^31`, checked at build time. N must be known at build time (dense B). |

### Memory ordering (producer side)

1. Write all rows of M tile `t`, padding included.
2. If they were written with TMA / `cp.async.bulk`, wait for completion with `cp.async.bulk.wait_group 0` (not the
   `.read` variant, which only frees the source).
3. Make every writer's stores happen-before the publishing thread (`__syncthreads()`; across blocks, a per-M-tile
   `acq_rel` counter whose last arrival publishes).
4. Store `t` to `signal[t]` with a gpu-scope release
   (`cuda::atomic_ref<int, cuda::thread_scope_device>::store(t, memory_order_release)`).

The GEMM does a gpu-scope acquire load of `signal[t]` and then a `fence.proxy.async.global` before its TMA loads, so the
producer does not need a proxy fence of its own.

### Reset and termination

- **Who resets:** the caller, before every launch, ordered before both the producer and the GEMM. Use
  `moe_kernel_helpers.reset_streaming_signal_workspace(workspace, kernel)`: counter and state 0, `signal[]` -1. With
  `all_complete=True` it sets `signal[t] = t`, which runs the GEMM with no producer (tests, warm-up).
- **No GEMM-side reset:** the GEMM never resets the signal workspace, and on this path its helper kernel does not reset
  the claim counter either.
- **Unbounded wait:** waits have no timeout; a bounded wait could return with the M tile not ready. A cluster stops once
  its claim index passes the last tile (every M tile times `ceil(N / cluster tile N)`), before it waits on anything.
- **Two GEMM launches on one counter:** a base kernel and a second kernel launched as its PDL dependent can share one
  workspace and claim from the same counter, and each tile is still computed once. Reset that workspace **once**,
  before the base.

### Liveness

- The producer must be able to run while the GEMM waits. Give the GEMM fewer clusters than the device holds
  (`max_active_clusters`), or launch it as the producer's programmatic dependent (section 4). Either way, the producer
  keeps the SMs it needs.
- The producer must not wait on the GEMM.

## 2. Device layout (`device_layout=True`)

Here the producer also writes the per-expert offsets while the GEMM is already running, so the host never reads the
routing.

| Item | Contract |
|---|---|
| buffer | `layout`: `E + 1` int32 entries; the GEMM's `padded_offsets` is the view `layout[1:E+1]` |
| contents | `layout[0:E]` = **exclusive** expert starts, `layout[E]` = padded total, so `layout[1:E+1]` is the inclusive form the GEMM takes |
| reset | the caller sets all `E + 1` entries to -1 before the producer starts, together with the `signal[]` reset |
| producer | writes all `E + 1` entries (gpu-scope release) before it publishes any `signal[]` entry |
| GEMM | every CTA's scheduler warp acquire-waits, without bound, until all `E + 1` entries are `>= 0` |

- The GEMM reads `layout[0]`, one entry **before** `padded_offsets`. A standalone `padded_offsets` tensor therefore
  reads out of bounds.
- With `device_layout`, no CTA exits early on a host-side total of 0, and every warp role runs in every CTA. A total of
  0 ends the first claim in every CTA of the cluster.

## 3. Output direction: GEMM → consumer (`N_CHUNK`)

`StreamingOutputNotifyConfig(mode=StreamingOutputNotifyMode.N_CHUNK, n_chunk_size=C, n_chunk_capacity, caller_reset=False, clusters_beside_consumer)`

### `signal[u]`

- Location: the last `4 × n_chunk_capacity` bytes of the GEMM workspace. Get it from
  `streaming_output_notify_offsets(kernel)["signal"]` and hand it to the consumer zero-copy.
- `signal[]` is int32, one entry per N chunk `u` in `[0, ceil(N / C))`. Chunk `u` is D columns
  `[u × C, min((u + 1) × C, N))` of **every** padded row.

| Value | Meaning |
|---|---|
| `-1 - k` (k ≥ 0) | chunk incomplete, `k` CTA tiles counted so far |
| `u` | chunk complete: every row of those columns is in global memory |

- A tile counts only after all its D stores have reached global memory: `cp.async.bulk.wait_group 0`, then one
  `acq_rel` gpu-scope add per finished run of tiles.
- The add that completes the chunk is followed by a gpu-scope release store of `signal[u] = u`.
- **The consumer waits for exactly `u`**, with a gpu-scope acquire load, then reads the chunk with coherent loads.
  An in-progress value is always negative, so it is never mistaken for a publication.
- The GEMM claims tiles N chunk → every M tile → the chunk's N tiles, so chunks complete roughly in order. The consumer
  still waits on each `signal[u]`.

### Sizes

| Quantity | Rule |
|---|---|
| `C` (`n_chunk_size`) | A multiple of the CTA tile N, fixed at build time. Give the consumer the same C; a consumer chunk wider than the GEMM's corrupts data silently. |
| last chunk | May be partial when C does not divide N; it counts only its own tiles. |
| `n_chunk_capacity` | `>= streaming_output_notify_capacity(C, N) = ceil(N / C)`, checked before launch. The device indexes `signal[]` with no bound check. |
| N | `<= (2^31 - 1) * C`, since `u` is int32; known at build time (dense B). |

### Reset

- Default: the GEMM call enqueues a one-block kernel ahead of itself that sets `signal[] = -1`. With `use_pdl=True`
  that reset is the kernel the GEMM depends on, so every GEMM thread runs `griddepcontrol.wait` at entry, before any
  count lands on `signal[]` and before it lets a dependent consumer launch. The consumer therefore never sees a
  previous launch's `signal[u] == u`.
- `caller_reset=True`: the caller runs `streaming_output_notify.reset(workspace_ptr, kernel.get_workspace_bytes(), stream)`
  before each GEMM, on the GEMM's stream. The caller then owns the ordering: the reset's stores must be visible before
  the GEMM starts counting, which holds when the reset completes before the GEMM launches. The constructor rejects
  `use_pdl=True` here (see §4). The consumer is still the GEMM's programmatic dependent (see below).

### Extra clusters and liveness

- The GEMM launches on every cluster of the device. `clusters_beside_consumer` (≥ 1) is how many clusters fit beside
  the consumer's SMs; clusters at or past that rank are extras.
- An extra stops claiming once its last claim is at or past `max(chunk 0 tiles − grid clusters, 0)`. That frees its
  SMs at about the end of chunk 0. Every tile is still claimed exactly once.
- **The consumer must be the GEMM's programmatic dependent on the same stream.** Build the GEMM with
  `launch_pdl_dependents_at_entry=True` and launch the consumer with PDL. Otherwise the consumer can take SMs the
  persistent GEMM still needs, and both hang.
- A GEMM with no rows (`padded_offsets[E-1] == 0`) exits before publishing anything. The consumer must not wait on
  `signal[]` when it received no rows.

## 4. Programmatic dependent launch (PDL) options

| Option | Effect | Use |
|---|---|---|
| `use_pdl=True` | The GEMM launches as the programmatic dependent of the kernel before it on the stream, so it can start before that kernel ends; at exit every thread waits for that grid (`griddepcontrol.wait`). With `N_CHUNK` and its own reset, that kernel is the reset, and the wait moves to entry (see §3, Reset). The constructor rejects it whenever the GEMM call enqueues its helper kernel right before the GEMM (discrete weights, or the dynamic scheduler without `streaming_signal_capacity`), other than that own reset: the GEMM reads what the helper writes (B descriptors, the claim counter) before its exit wait. | The GEMM runs beside its producer on one stream (`streaming_signal_capacity`), or `N_CHUNK` with its own reset. |
| `launch_pdl_dependents_at_entry=True` (plain and GLU kernels) | The GEMM lets its own dependent launch at entry. | A second GEMM on the same counter, an `N_CHUNK` consumer, or the next GEMM of a chain takes the SMs this grid leaves. |

## 5. Sketches

Producer of `signal[]` (CUDA C++; one block per M tile):

```cpp
__global__ void produce_m_tiles(const Row* src, Row* a, int* signal, int m_tile_rows) {
    int t = blockIdx.x;
    copy_rows(src, a, t * m_tile_rows, m_tile_rows);  // padding rows included
    __syncthreads();                              // every thread's stores before the publisher
    if (threadIdx.x == 0) {
        cuda::atomic_ref<int, cuda::thread_scope_device>(signal[t]).store(t, cuda::memory_order_release);
    }
}
```

Consumer of an `N_CHUNK` GEMM (launched as the GEMM's PDL dependent):

```cpp
__global__ void consume_chunks(const int* signal, const T* d, int chunk_count) {
    for (int u = 0; u < chunk_count; ++u) {
        if (threadIdx.x == 0) {
            cuda::atomic_ref<const int, cuda::thread_scope_device> entry(signal[u]);
            while (entry.load(cuda::memory_order_acquire) != u) {}
        }
        __syncthreads();
        send_columns(d, u);  // coherent loads of columns [u*C, min((u+1)*C, N)) of every row
    }
    // The per-chunk acquires make D visible; they do not complete the PDL dependency on the GEMM grid.
    cudaGridDependencySynchronize();  // also when chunk_count == 0
}
```

### Single-GPU self-test

Pattern from `test/python/gemm/cutedsl/test_grouped_gemm_streaming_signal.py`:

- **Same bits:** compare against the stock dynamic scheduler, bit for bit. Call `reset_streaming_signal_workspace(...,
  all_complete=True)` so every M tile is ready, and compare `C` and `D` with a launch built with `use_dynamic_sched=True`
  only.
- **Early reads are caught:** fill A with NaN and every `signal[t]` with -1. Enqueue the producer (a sleep, then per
  M tile: copy the rows, then set `signal[t] = t`) on a side stream, and launch the GEMM last with few clusters. Any
  early read puts NaN in D.
- **Module-loading deadlock:** with lazy module loading, the first launch of a kernel waits for running kernels to
  finish. A producer launched after a GEMM that is already spinning therefore deadlocks. To avoid it:
  - run the GEMM once first, which loads its module;
  - enqueue every producer launch before the GEMM's launch.

  `CUDA_MODULE_LOADING=EAGER` also avoids it.
- **`N_CHUNK`:** after the GEMM, `signal[u] == u` for every `u < ceil(N / C)`, and D matches the stock run bit for bit.

## 6. Constraints

- Cluster shapes: the input direction (`streaming_signal_capacity`) supports only (2, 1) with the 2-CTA MMA; a (1, 1)
  cluster is **not supported** (the scheduler broadcasts each claim into a peer CTA's shared memory and asserts at
  least two CTAs). `N_CHUNK` supports (2, 1) with the 2-CTA MMA and (1, 1) with the 1-CTA MMA. Other shapes raise
  `ValueError`, because a CTA could exit while a peer still arrives on its shared memory.
- `streaming_signal_capacity` and `N_CHUNK` cannot be combined in one kernel; both need `use_dynamic_sched=True`.
- N must be known at build time (dense B).
- `device_layout` requires `streaming_signal_capacity`.
- Covered today: the BF16 plain and GLU grouped GEMM forward kernels. Other recipes are not wired up yet.

## Acknowledgements

The streaming signal contract and its schedulers were designed and developed by Wentao Guo, Phuong Nguyen,
Zhongbo Zhu, Subhadeep Bhattacharya, Fei Wu, and Jimmy Zhang.
