# DSA Sparse Attention Backward Benchmark

Microbenchmark for the DeepSeek Sparse Attention (DSA) backward kernel in the
cuDNN Frontend CuTe DSL package, driven through the public
`cudnn.DSA.sparse_attention_backward_wrapper` API. The wrapper dispatches to
the Hopper (SM90) or Blackwell (SM100) implementation based on the active
CUDA device.

## What is measured

Inputs are flat FlashMLA-shaped tensors: `q (S_q, H, d_qk)`, a shared
`kv (S_kv, d_qk)` buffer (K = V), and per-query global top-k indices
`topk_idxs (S_q, topk)` with unique random indices per query row. The forward
`out`/`lse` consumed by the backward kernel come from a chunked PyTorch
reference, since the production forward (FlashMLA) is out of scope for this
repository.

Each timed iteration is one full wrapper call — gradient-buffer zeroing,
workspace allocation, and the preprocess/backward/convert kernels — i.e. the
cost a training step pays per backward invocation. Timing uses a single
CUDA-event window around `--repeat` iterations and reports the average.

Reported TFLOPS use the 5-matmul model of the backward pass (recompute S, dV,
dP, dQ, dK):

```
FLOPs = 2 * S_q * H * topk * (3 * d_qk + 2 * d_v)
```

## Requirements

- Hopper (SM90) or Blackwell (SM100) GPU
- PyTorch with CUDA support
- `pip install nvidia-cudnn-frontend[cutedsl]` (or a development install of
  this repository's `python/` package with the `cutedsl` extra)

## How to run

Default sweep (`seqlens 4096,8192 x topks 128,512,1024,2048`, bf16,
`d_qk = d_v = 512`, 64 heads):

```bash
python benchmark_dsa_sparse_attention_backward.py
```

Custom shapes and CSV output:

```bash
python benchmark_dsa_sparse_attention_backward.py --seqlens 4096,8192,16384 --topks 512,2048 --csv results.csv
python benchmark_dsa_sparse_attention_backward.py --head-dim 576   # 512 value dims + 64 RoPE dims
```

Options:

- `--seqlens` — comma-separated total query lengths; `seqlen_kv = seqlen_q`
  for every config. Configs with `topk > seqlen_kv` are skipped.
- `--topks` — comma-separated top-k values.
- `--nheads` — number of query heads (default 64).
- `--head-dim` — QK head dim, `512` or `576`; `head_dim_v` is derived (512).
- `--dtype` — `bfloat16` (default) or `float16`.
- `--no-attn-sink` — disable the attention sink (passes `-inf` sink logits).
- `--no-topk-length` — omit the `topk_length` tensor. Kernels with and
  without `topk_length` are different compiled variants; the default
  benchmarks the `topk_length` variant with every row at the full top-k
  count.
- `--warmup` / `--repeat` — iterations per config (defaults 10 / 50; the
  first warmup iteration also triggers kernel compilation).
- `--csv` — write results to a CSV file.

## Results

### B200

Generated on an NVIDIA B200 with the default
sweep settings (`nheads=64`, `d_qk = d_v = 512`, bf16, attention sink and
`topk_length` enabled, `warmup=10`, `repeat=50`), using `torch 2.12.1`,
`nvidia-cutlass-dsl 4.5.2`, and `nvidia-cudnn-frontend` built from this
repository.

| seqlen_q | seqlen_kv | topk | BWD ms | BWD TFLOPS |
|---------:|----------:|-----:|-------:|-----------:|
|     4096 |      4096 |  128 |  0.563 |     305.09 |
|     4096 |      4096 |  512 |  1.243 |     552.87 |
|     4096 |      4096 | 1024 |  2.198 |     625.35 |
|     4096 |      4096 | 2048 |  4.168 |     659.46 |
|     8192 |      8192 |  128 |  1.094 |     313.97 |
|     8192 |      8192 |  512 |  2.489 |     552.16 |
|     8192 |      8192 | 1024 |  4.538 |     605.76 |
|     8192 |      8192 | 2048 |  8.562 |     642.06 |

### B300

Generated on an NVIDIA B300 (`nheads=64`, bf16, attention sink and
`topk_length` enabled, `warmup=10`, `repeat=50`), using `torch 2.13.0+cu130`,
`nvidia-cutlass-dsl 4.5.2`, CUDA 13.0, and `nvidia-cudnn-frontend` built from
this repository.

`d_qk = d_v = 512`:

| seqlen_q | seqlen_kv | topk | BWD ms | BWD TFLOPS |
|---------:|----------:|-----:|-------:|-----------:|
|     1024 |      1024 |  512 |  0.344 |     500.08 |
|     2048 |      2048 |  512 |  0.648 |     530.63 |
|     4096 |      4096 |  512 |  1.238 |     555.26 |
|     8192 |      8192 |  512 |  2.406 |     571.31 |
|    16384 |     16384 |  512 |  4.789 |     574.04 |
|    32768 |     32768 |  512 | 10.415 |     527.85 |
|     2048 |      2048 | 2048 |  1.993 |     689.57 |
|     4096 |      4096 | 2048 |  3.949 |     696.08 |
|     8192 |      8192 | 2048 |  7.914 |     694.63 |
|    16384 |     16384 | 2048 | 15.906 |     691.24 |
|    32768 |     32768 | 2048 | 33.351 |     659.37 |

`d_qk = 576` (512 value dims + 64 RoPE dims, `d_v = 512`):

| seqlen_q | seqlen_kv | topk | BWD ms | BWD TFLOPS |
|---------:|----------:|-----:|-------:|-----------:|
|     2048 |      2048 | 2048 |  2.638 |     560.12 |
|     4096 |      4096 | 2048 |  5.214 |     566.76 |
|     8192 |      8192 | 2048 | 10.312 |     573.13 |
|    16384 |     16384 | 2048 | 20.620 |     573.21 |
|    32768 |     32768 | 2048 | 47.195 |     500.89 |

### Tensor-pipe (MMA) utilization

Nsight Compute speed-of-light numbers for the main warp-specialized backward
kernel (`kernel_cutlass_bwd_*`, 97–98% of the pass at `topk=2048`), collected
on the same B300 with
`ncu --profile-from-start off -k regex:kernel_cutlass_bwd --metrics
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`. ncu locks
clocks to base during collection, so percentages are comparable across
configs while the wall-clock numbers above reflect boost clocks. All tensor
pipe activity is HMMA.

| d_qk | topk | seqlen 1k | 2k | 4k | 8k | 16k | 32k |
|-----:|-----:|----------:|-----:|-----:|-----:|-----:|-----:|
|  512 |  512 |      36.7 | 37.5 | 38.3 | 38.7 | 38.7 | 38.4 |
|  512 | 2048 |         — | 43.7 | 44.1 | 44.2 | 44.5 | 44.3 |
|  576 | 2048 |         — | 36.8 | 36.9 | 37.1 | 37.2 | 34.2 |

Observations:

- MMA utilization is essentially **sequence-length invariant** at a fixed
  top-k: the grid parallelizes over query tokens, so utilization is set by
  the per-CTA inner loop (`topk / 64` MMA iterations amortizing a fixed
  prologue/epilogue), not by problem size. `topk` is the utilization knob.
- The kernel is memory-pipe-heavy rather than tensor-bound: L1TEX/memory
  speed-of-light (62–68%) exceeds SM (49–57%) and MMA (34–45%) everywhere;
  the top-k gather of KV rows plus fp32 dKV accumulation is the busiest pipe.
- At `seqlen 32k` the KV working set (32k x 512 x 2B = 32 MB) exceeds L2 and
  DRAM utilization jumps (1.8% -> 12.4% at `d_qk=512/topk=2048`,
  6.5% -> 22.7% at `topk=512`, 1.6% -> 15.1% at `d_qk=576`), which is the
  wall-clock dip in the tables above.
- The 576-wide latent runs ~7 SOL points below 512 at the same top-k: the
  non-128-aligned 64-dim RoPE tail takes separate tail-tile MMAs.

## Profiling

`profile` mode runs a single warmed-up backward call (using the first value
of `--seqlens` and the last value of `--topks`) wrapped in
`cudaProfilerStart/Stop` and an NVTX range, so nsys/ncu capture only the
kernels of interest:

```bash
nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop -o dsa_bwd \
  python benchmark_dsa_sparse_attention_backward.py profile --seqlens 8192 --topks 2048

ncu --profile-from-start off -o dsa_bwd \
  python benchmark_dsa_sparse_attention_backward.py profile --seqlens 8192 --topks 2048
```
