# BSA JAX / torch / GPU benchmark

The small-input API overhead is substantial in both frameworks. Eager JAX
backward also recompiles XLA on repeated calls; use `jax.jit` for the current
implementation. These measurements do **not** qualify production performance.

## Reproduce

Use the edited frontend, a CUDA-enabled JAX installation, torch, CuTeDSL and
`cuda-bindings`. This comparison intentionally imports torch; the BSA JAX runtime
and its isolated tests remain torch-free. Select one idle SM100 GPU:

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python benchmark/bsa/benchmark_jax.py --output results.json
```

For separate Python profiles, add `--cases small --layouts bhsd --cpu-profile`.
The script rejects other GPU processes before and after each measurement phase;
this is a contention check, not a reservation mechanism.

## Workloads

BF16 MHA, sparse block size 128, compact BHSD and BSHD layouts. Fixed-count
int32 metadata selects unique random KV blocks independently per query block.
Inputs and sparse patterns share a seed across frameworks. Backward uses each
framework's own forward O/LSE, computed outside the timed region.

| Case | B | H | Sq | Sk | D | Active KV blocks/query block | Backward bucket size |
|---|---:|---:|---:|---:|---:|---:|---|
| small | 1 | 2 | 256 | 512 | 64 | 2 | default, one bucket |
| medium | 2 | 8 | 2048 | 4096 | 128 | 8 | default, one bucket |
| multi_bucket | 1 | 8 | 16384 | 16384 | 128 | 16 | 64 query blocks, two buckets |

## Method

- API measurements: median of seven batches of 100 calls, wall time including
  one final device synchronization per batch. Dispatch time, sample ranges and
  individual samples are retained in JSON. This measures amortized time per
  operation under repeated asynchronous submission, not isolated call latency.
- Initial JIT compilation, five warmup calls per mode, inputs, initial forward
  results and parity checks are outside timing. Recurring compilation within
  an API invocation remains included, even after warmup.
- **Raw sequence**: CUDA-event device time for a captured torch BSA invocation,
  repeated 32 times through serial child graph nodes sharing the same captured
  buffers. Seven samples each launch that graph 100 times, after a 250 ms replay
  warmup. This amortizes Python submission and reuses existing kernels directly
  through their captured launches. It includes GPU graph scheduling, all required
  initialization, CSR preparation and backward pre/main/post kernels. It is
  neither an isolated attention-kernel duration nor a profiler sum of kernels.
- **Torch graph**: wall time for a single captured invocation per Python replay,
  with the same captured buffers. It diagnoses host API cost eliminated by
  capture. Eager APIs still allocate their own outputs/workspaces.
- **JAX CB requested**: `jax.jit` with
  `xla_gpu_enable_command_buffer=FUSION,CUSTOM_CALL` and
  `xla_gpu_graph_min_graph_size=1`. This is an opt-in compiler configuration;
  these measurements do not establish which operations XLA actually captured.
- JAX jit and CB-requested outputs are checked against torch, and captured
  outputs are checked after raw and single-graph replays (`atol=rtol=0.03`).
  These parity checks supplement the existing independent numerical tests.

Raw captures aggressively reuse buffers. API allocation, cache state, CuTeDSL
specialization, graph scheduling and unlocked clocks can differ across modes.
Consequently **API minus raw is a comparison gap, not a precise attribution of
framework overhead**; small negative gaps must not be interpreted as speedups
over identical kernel execution. Inputs remain fixed, so these measurements
cover a synthetic workload with warm caches. Mode order is fixed and
CPU affinity/clocks are not locked; retain the sample ranges when interpreting
small differences. Custom-VJP end-to-end gradient timing, variable-count rows,
compilation latency and peak workspace are outside this sweep.

## Measured environment

SM100 (CC 10.0), NVIDIA Graphics Device, 183359 MiB, host
`cudnn-dev-batten-22-04`, GPU 0. Driver 580.159.03; sampled SM/memory clocks
1830/4000 MHz. Python 3.14.7, JAX/jaxlib 0.11.1, torch 2.14.0+cu130,
CuTeDSL 4.7.0, cuda-bindings 13.3.1, TVM-FFI 0.1.13.post3.

Runtime source commit: `3faa2f03c3f9fc1c1e1b78f994cbc06e1572294b`.
The run imported this checkout's Python sources via `PYTHONPATH=$PWD/python`,
using the released frontend 1.28.0 extension. The shared-backend header mismatch
still prevents a complete matching editable source build; see the draft PR's
qualification notes. The final sweep ran after the competing KDA job finished.

## Results

Median µs/op. Full samples, ranges, settings and GPU identity:
[results_sm100.json](results_sm100.json). Δ = API − raw sequence.

| Case / layout | Direction | Raw sequence | JAX jit | JAX Δ | Torch eager | Torch Δ |
|---|---|---:|---:|---:|---:|---:|
| small / bhsd | forward | 5.7 | 38.5 | +32.7 | 52.8 | +47.1 |
| small / bhsd | backward | 19.8 | 66.0 | +46.2 | 123.5 | +103.7 |
| small / bshd | forward | 5.5 | 41.8 | +36.3 | 62.6 | +57.0 |
| small / bshd | backward | 20.0 | 71.8 | +51.8 | 129.8 | +109.8 |
| medium / bhsd | forward | 24.9 | 42.1 | +17.1 | 96.7 | +71.8 |
| medium / bhsd | backward | 94.6 | 111.0 | +16.4 | 124.3 | +29.7 |
| medium / bshd | forward | 25.2 | 39.2 | +14.0 | 62.8 | +37.5 |
| medium / bshd | backward | 95.6 | 110.3 | +14.8 | 129.6 | +34.0 |
| multi_bucket / bhsd | forward | 186.7 | 170.3 | -16.4 | 163.2 | -23.6 |
| multi_bucket / bhsd | backward | 682.4 | 679.9 | -2.5 | 666.2 | -16.2 |
| multi_bucket / bshd | forward | 192.6 | 178.2 | -14.4 | 188.4 | -4.2 |
| multi_bucket / bshd | backward | 700.6 | 678.5 | -22.1 | 681.2 | -19.4 |

Diagnostic modes (µs/op except eager JAX backward, shown in ms):

| Case / layout | Direction | JAX CB requested, µs | Torch graph, µs | JAX eager |
|---|---|---:|---:|---:|
| small / bhsd | forward | 67.1 | 6.2 | 73.9 µs |
| small / bhsd | backward | 37.7 | 20.6 | 15.26 ms |
| small / bshd | forward | 41.7 | 6.2 | 82.0 µs |
| small / bshd | backward | 57.4 | 22.6 | 15.77 ms |
| medium / bhsd | forward | 46.4 | 22.7 | 86.1 µs |
| medium / bhsd | backward | 99.6 | 97.3 | 16.60 ms |
| medium / bshd | forward | 42.8 | 22.6 | 93.1 µs |
| medium / bshd | backward | 99.6 | 96.9 | 16.70 ms |
| multi_bucket / bhsd | forward | 177.0 | 160.8 | 176.7 µs |
| multi_bucket / bhsd | backward | 678.4 | 653.4 | 18.91 ms |
| multi_bucket / bshd | forward | 178.6 | 173.4 | 170.8 µs |
| multi_bucket / bshd | backward | 673.3 | 662.8 | 18.99 ms |

Large-case variability is material: multi_bucket BSHD backward torch-graph
samples range from 561.3 to 715.6 µs/op.
Do not use the small negative gaps in the main table to rank kernel speed.

## Overhead findings

Torch eager is abnormally expensive for the small cases. Separate untimed
[cProfile output](profile_sm100.txt) shows warmed forward calls still construct
kernel objects, compute dynamic tensor cache keys, validate inputs and allocate
outputs. Backward additionally allocates metadata/workspaces and makes four
CuTeDSL launches for the single-bucket case. The profile supports host-side
setup/dispatch as a contributor; it does not assign accurate microseconds to
each component. CUDA-graph replay removes most of the small-case torch gap.

Eager JAX backward is a more severe issue: the separate profile records **ten
XLA backend compilations for ten warmed calls**. The initialized-output wrapper
in `python/cudnn/jax/call.py` reconstructs the native CuTeDSL callable on each
invocation; CuTeDSL 4.7.0 creates a fresh inner jitted function for that callable.
This is the likely cache-identity cause. Outer `jax.jit` avoids recurring Python
wrapper execution. Cached BSA call builders alone do not prevent this problem.

Before performance qualification, cache a stable initialized-output callee and
verify repeated eager calls stop compiling; profile the remaining jitted FFI
cost; and evaluate torch launch-plan/workspace reuse against existing API
semantics. These are follow-up runtime changes, not hidden benchmark shortcuts.
Repeat the comparison with representative sparse metadata and a qualified
source build. No runtime implementation was changed for this benchmark.
