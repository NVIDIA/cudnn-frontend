# Simplified sparse attention benchmark

Run against an installed `nvidia-cudnn-frontend` wheel with PyTorch and an
SM100 or SM103 GPU. No separate attention package is required.

```sh
python benchmark/simplified_sparse_attention/benchmark_simplified_sparse_attention.py \
  --query-length 1024 --key-length 4096 --heads 16 --topk 512 --mode training
python benchmark/simplified_sparse_attention/benchmark_simplified_sparse_attention.py \
  --query-length 1024 --key-length 131072 --heads 32 --topk 2048 --mode forward --reserve
XLA_PYTHON_CLIENT_PREALLOCATE=false python benchmark/simplified_sparse_attention/benchmark_simplified_sparse_attention.py \
  --framework jax --query-length 1024 --key-length 4096 --heads 16 --topk 512 --mode training
```

The benchmark uses BF16, causal row bounds, and the public API. CUDA events
measure eager forward or forward plus backward after warmup. Forward includes
the auxiliary loss and precomputed indexer gradients. JIT compilation and
optional reservation are outside timing. CSV output includes runtime versions,
latency, peak allocated memory, and `cuda_graph=False`.

`--framework jax` measures JIT execution with wall time and a completion wait
per call (`timing=wall_sync`). PyTorch uses CUDA events (`timing=cuda_events`).
Compare these timing methods with care. Both use the same PyTorch-generated
BF16 inputs; JAX receives them through DLPack before warmup. PyTorch peak-memory
statistics do not measure XLA allocations, so that CSV field is empty for JAX.
JAX has no reserve option. Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` when sharing
the GPU with other work.

External CUDA graph capture is unsupported. PyTorch's internal conditional
repair graphs do not make the public call capture compatible. JAX uses an XLA
repair loop and opts its FFI calls out of CUDA command-buffer capture.

For Phase 1 indexer warmup, select the fused kernel or the chunked PyTorch path:

```sh
python benchmark/simplified_sparse_attention/benchmark_simplified_sparse_attention.py \
  --phase 1 --phase1-backend cutedsl --query-length 65 --key-length 4097 --heads 32 --mode training
```

`--phase1-backend torch` selects the chunked path. Phase 1 timing excludes
teacher attention/LSE preparation and includes the scalar KL plus precomputed
indexer gradients. Training also applies the upstream loss gradient.
`topk` and reservation do not apply; Phase 1 has no JAX entry point or external
CUDA graph support. Peak memory still includes benchmark input allocations.

Measured on 2026-11-09 with the installed frontend 1.33.0 wheel, SM100
(148 SMs), driver 610.11, PyTorch 2.13.0+cu130, cuDNN 9.27.0, and
CuTe DSL 4.8.0 compiling with CUDA 13.4. Each case used three warmups and
20 timed calls. These are single-run samples on a shared GPU.

| Query / key rows | Heads / top-k | Mode | Latency (ms) | Peak allocated memory (MiB) |
| --- | --- | --- | --- | --- |
| 1024 / 4096 | 16 / 512 | Forward | 0.513 | 93.2 |
| 1024 / 4096 | 16 / 512 | Forward + backward | 1.175 | 111.8 |
| 1024 / 131072 | 32 / 2048 | Forward, reserved | 2.373 | 452.5 |

Peak memory includes inputs, returned warmup tensors, and persistent repair
state. It is not an incremental workspace measurement. SM103 performance
has not been measured.

JAX/JAXlib 0.11.0 samples from the same SM100 runtime and the rebuilt wheel
use wall time with a completion wait for each call, three warmups, and 20
timed calls. These include host dispatch and synchronization costs. They are
not direct speed comparisons with the PyTorch CUDA-event samples above.

| Query / key rows | Heads / top-k | JAX mode | Wall latency (ms) |
| --- | --- | --- | --- |
| 1024 / 4096 | 16 / 512 | Forward | 2.951 |
| 1024 / 4096 | 16 / 512 | Forward + backward | 3.441 |
| 1024 / 131072 | 32 / 2048 | Forward | 6.914 |

Phase 1 samples at `b1a9c4a`, with the installed wheel on the SM100 runtime
above (CUDA 13.4 compiler), three warmups and 20 timed calls:

| Query / key rows | Heads | Backend | Mode | Latency (ms) | Peak allocated memory (MiB) |
| --- | --- | --- | --- | --- | --- |
| 65 / 4097 | 32 | CuTe | Forward + backward | 0.722 | 45.2 |
| 65 / 4097 | 32 | PyTorch chunked | Forward + backward | 6.320 | 76.6 |

These single-run samples exclude teacher LSE preparation and use a shared GPU.
They do not qualify SM103 performance or CUDA graph capture.
