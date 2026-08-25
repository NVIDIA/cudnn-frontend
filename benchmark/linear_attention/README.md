# Linear Attention Benchmark

## Introduction

This directory contains benchmarking tools for linear attention operations (GDN/KDA/GDN-2) across various backends. The benchmarks target training use cases with support for forward and backward passes.

## Contents

- `Dockerfile` - Docker container setup for running benchmarks
- `benchmark_single_linear_attention.py` - Single linear attention benchmark script
- `plot_results.py` - Renders the charts under `results/` from a sweep CSV
- `results/<variant>/<gpu>/` - Dated sweep CSVs and the charts rendered from them

## Quick Start

### 1. Build Docker Container

```bash
docker build -t cudnn_linear_attention_benchmark .

docker run -it --gpus all --rm cudnn_linear_attention_benchmark
```

### 2. Run Benchmarks

```bash
# cuDNN Frontend (BF16, GDN, forward + backward)
python benchmark_single_linear_attention.py \
    --batch_size 1 --seqlen 8192 \
    --num_q_heads 8 --num_kv_heads 64 --head_dim 128 \
    --la_backend cudnn --variant gdn --data_type bfloat16 \
    --skip_ref --fwd_bwd

# cuDNN Frontend (KDA, backward pass only)
python benchmark_single_linear_attention.py \
    --batch_size 1 --seqlen 8192 \
    --num_q_heads 16 --num_kv_heads 16 --head_dim 128 \
    --la_backend cudnn --variant kda --data_type bfloat16 \
    --skip_ref --profile_pass bwd

# cuDNN Frontend (GDN-2, forward pass)
python benchmark_single_linear_attention.py \
    --batch_size 1 --seqlen 8192 \
    --num_q_heads 16 --num_kv_heads 16 --head_dim 128 \
    --la_backend cudnn --variant gdn2 --data_type bfloat16 \
    --skip_ref --profile_pass fwd

# GQA (q-heads grouped over v-heads, backward pass only)
python benchmark_single_linear_attention.py \
    --batch_size 1 --seqlen 8192 \
    --num_q_heads 64 --num_kv_heads 8 --head_dim_qk 128 --head_dim_vo 128 \
    --la_backend cudnn --variant gdn --data_type bfloat16 \
    --skip_ref --profile_pass bwd

# FLA (flash-linear-attention) comparison point
python benchmark_single_linear_attention.py \
    --batch_size 1 --seqlen 8192 \
    --num_q_heads 8 --num_kv_heads 64 --head_dim 128 \
    --la_backend fla --variant gdn --data_type bfloat16 \
    --skip_ref --fwd_bwd

# FlashQLA (TileLang) comparison point (gdn variant only)
python benchmark_single_linear_attention.py \
    --batch_size 1 --seqlen 8192 \
    --num_q_heads 8 --num_kv_heads 64 --head_dim 128 \
    --la_backend flash_qla --variant gdn --data_type bfloat16 \
    --skip_ref --fwd_bwd

# FlashKDA comparison point (kda variant only, forward only, bf16)
python benchmark_single_linear_attention.py \
    --batch_size 1 --seqlen 8192 \
    --num_q_heads 32 --num_kv_heads 32 --head_dim 128 \
    --la_backend flash_kda --variant kda --data_type bfloat16 \
    --skip_ref

# Input initial state and dump state for every chunk
python benchmark_single_linear_attention.py \
    --batch_size 1 --seqlen 8192 \
    --num_q_heads 8 --num_kv_heads 64 --head_dim 128 \
    --la_backend cudnn --variant gdn --data_type bfloat16 \
    --skip_ref --fwd_bwd --initial_state --store_on
```

Run `python benchmark_single_linear_attention.py --help` for all options.

The `kda` and `gdn2` variants fuse q/k L2 normalization in-kernel on every backend; `gdn` runs unfused.

## Supported Backends

| Backend | Description |
|---------|-------------|
| `cudnn` | cuDNN (native, via the cuDNN Frontend torch custom ops) |
| `fla`   | FLA (flash-linear-attention, Triton; `gdn`, `kda`, and `gdn2`) |
| `flash_qla` | FlashQLA (TileLang fused GDN kernels, `gdn` variant only) |
| `flash_kda` | FlashKDA (`kda` forward variant only) |

The cuDNN backend routes through the pygraph engines: FROST (Cutlass DSL) on SM100-class devices, the cuTile engines elsewhere.

## Results

Forward and backward TFLOPS at batch 4 over the sequence-length sweep, one
chart per variant, rendered by `plot_results.py` from the dated CSVs under
`results/<variant>/<gpu>/`. The `cudnn (state on)` bars dump the per-chunk
state-checkpoint series in the forward pass and reuse it in the backward
pass. Runs were captured on GB200 and GB300 (GB300 results shown below).

### GB300 - GDN
![GDN on GB300](results/gdn/gb300/gdn_fixed_batch_flops.png)
- `batch=4; num_q_heads=64; num_kv_heads=64; head_dim=128; seqlen 2048-32768; bf16`

### GB300 - KDA
![KDA on GB300](results/kda/gb300/kda_fixed_batch_flops.png)
- `batch=4; num_q_heads=64; num_kv_heads=64; head_dim=128; seqlen 2048-32768; bf16`

### GB300 - GDN-2
![GDN-2 on GB300](results/gdn2/gb300/gdn2_fixed_batch_flops.png)
- `batch=4; num_q_heads=64; num_kv_heads=64; head_dim=128; seqlen 2048-32768; bf16`