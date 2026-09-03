# DSA Sparse Attention Benchmarks

## Sparse Attention Forward

`benchmark_dsa_sparse_attention_forward.py` benchmarks the public SM100
`cudnn.DSA.sparse_attention_forward_wrapper` API for H64/D512 or D576 and the
H128/D512 small-top-k Prefill kernel. The default shape is the H64 production
anchor `S_q=4096, S_kv=5120, K=640, indexer_topk=512`.

Two cuDNN Frontend timings are reported:

- **DSA execute path (preallocated)** — the concrete CuTe kernel is
  compiled before timing, all enabled output buffers are reused, and each
  iteration calls the same `SparseAttentionForward.execute()` object.
- **DSA public wrapper** — each iteration calls
  `sparse_attention_forward_wrapper`, including any GPU work caused by output
  allocation and logical-K padding/layout normalization.

Both timings use CUDA events on the current stream after independent warmup.
The execute measurement is not raw kernel-only time: it still includes the
lightweight public execution adapter, and for a non-multiple-of-64 logical K
it also includes internal index padding. CUDA events measure GPU elapsed time;
Python/host-side wrapper and allocation overhead is not separately captured.
The script prints both logical and padded K so that distinction is visible.

Run the default H64/D512 case or select another frozen specialization:

```bash
python benchmark_dsa_sparse_attention_forward.py
python benchmark_dsa_sparse_attention_forward.py --heads 64 --head-dim 576 --topk 640 --indexer-topk 512
python benchmark_dsa_sparse_attention_forward.py --heads 128 --head-dim 512 --topk 1152 --indexer-topk 1024
```

Common options are `--seqlen-q`, `--seqlen-kv`, `--heads`, `--head-dim`,
`--topk`, `--indexer-topk`, `--warmup`, `--repeat`, `--seed`,
`--no-attn-sink`, and `--use-topk-length`. Forward
requires an SM100-family GPU, BF16 inputs, PyTorch with CUDA support, and the
cuDNN Frontend `[cutedsl]` dependencies.

## Sparse Attention Backward

Microbenchmark for the DeepSeek Sparse Attention (DSA) backward kernel in the
cuDNN Frontend CuTe DSL package, driven through the public
`cudnn.DSA.sparse_attention_backward_wrapper` API. The wrapper dispatches to
the Hopper (SM90) or Blackwell (SM100) implementation based on the active
CUDA device.

### What is measured

Inputs are flat MQA tensors: `q (S_q, H, d_qk)`, a shared
`kv (S_kv, d_qk)` buffer (K = V), and per-query global top-k indices
`topk_idxs (S_q, topk)` with unique random indices per query row. The forward
`out`/`lse` consumed by the backward kernel come from a chunked PyTorch
reference so that this benchmark isolates backward and does not include a
forward launch.

Each timed iteration is one full wrapper call — gradient-buffer zeroing,
workspace allocation, and the preprocess/backward/convert kernels — i.e. the
cost a training step pays per backward invocation. Timing uses a single
CUDA-event window around `--repeat` iterations and reports the average.

Reported TFLOPS use the 5-matmul model of the backward pass (recompute S, dV,
dP, dQ, dK):

```
FLOPs = 2 * S_q * H * topk * (3 * d_qk + 2 * d_v)
```

### Requirements

- Hopper (SM90) or Blackwell (SM100) GPU
- PyTorch with CUDA support
- `pip install nvidia-cudnn-frontend[cutedsl]` (or a development install of
  this repository's `python/` package with the `cutedsl` extra)

### How to run

Default sweep (`seqlens 4096,8192 x topks 128,512,1024,2048`, bf16,
`d_qk = d_v = 512`, 64 heads):

```bash
python benchmark_dsa_sparse_attention_backward.py
```

Custom shapes and CSV output:

```bash
python benchmark_dsa_sparse_attention_backward.py --seqlens 4096,8192,16384 --topks 512,2048 --csv results.csv
python benchmark_dsa_sparse_attention_backward.py --head-dim 576   # 512 value dims + 64 RoPE dims
python benchmark_dsa_sparse_attention_backward.py --nheads 16 --head-dim 576  # SM100 H16/D576 M128 backend
python benchmark_dsa_sparse_attention_backward.py --nheads 128  # SM100 (10, 0) BF16 H128/D512 two-CTA backend
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

### Results

#### B200

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

### Profiling

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
