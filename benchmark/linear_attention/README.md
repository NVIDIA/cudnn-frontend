# Linear Attention Benchmark

## Introduction

This directory contains benchmarking tools for linear attention operations (GDN/KDA/GDN-2/GDP/Mamba-2) across various backends. The benchmarks target training use cases with support for forward and backward passes.

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

# cuDNN Frontend (GDP, n Householder sub-tokens per token; k/v/beta carry
# seqlen * num_householder rows)
python benchmark_single_linear_attention.py \
    --batch_size 1 --seqlen 8192 \
    --num_q_heads 64 --num_kv_heads 64 --head_dim_qk 128 --head_dim_vo 64 \
    --la_backend cudnn --variant gdp --num_householder 3 --data_type bfloat16 \
    --skip_ref --fwd_bwd

# cuDNN Frontend (Mamba-2 SSD: dt is the decay exponent and the write strength; B / C at the
# group count, x at the head count; state dim N = 128, head dim P = 64)
python benchmark_single_linear_attention.py \
    --batch_size 4 --seqlen 8192 \
    --num_q_heads 64 --num_kv_heads 64 --head_dim_qk 128 --head_dim_vo 64 \
    --la_backend cudnn --variant mamba2 --data_type bfloat16 \
    --skip_ref --fwd_bwd

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

Every variant, except Mamba-2, applies the q/k L2 normalization; pass `--no_qk_l2norm` to opt out.

The decay gates are all ones (alpha = 1, log gate 0) for every variant and backend, so the measured schedule does not depend on the input draw; kda and gdn2 feed the raw safe-gate logit of that gate, whose in-kernel `lower_bound * sigmoid` transform reproduces alpha = 1 exactly in fp32. The write strengths (beta, and w for GDN-2) stay random.  Mamba-2 feeds a raw dt of -30 (A = -1, zero dt_bias): softplus makes dt vanish, so its decay exp(A dt) is 1 in fp32 as well.

## Supported Backends

| Backend | Description |
|---------|-------------|
| `cudnn` | cuDNN (native, via the cuDNN Frontend torch custom ops) |
| `fla`   | FLA (flash-linear-attention, Triton; `gdn`, `kda`, `gdn2`, `gdp`, and `mamba2` through `chunk_simple_gla` with the Mamba-2 pre-scaling of k by dt and g = A dt) |
| `flash_qla` | FlashQLA (TileLang fused GDN kernels, `gdn` variant only) |
| `flash_kda` | FlashKDA (`kda` forward variant only) |

The cuDNN backend routes through the pygraph engines: FROST (Cutlass DSL) on SM100-SM103 and SM107, the cuTile engines elsewhere. `gdn2`, `gdp` and `mamba2` are FROST-only.

Default head counts are per variant: `kda` 96, `gdp` 40, 64 otherwise (`gdn`, `gdn2`, `mamba2`), the counts the tracked sweeps run at; the low-occupancy sweep passes 16 explicitly.  `--d_skip` adds the Mamba-2 D skip (`y += D x`, a per-head `D`; `--d_per_channel` makes it `[heads, P]`): the cudnn op fuses it and produces `dD`, the fla path adds it in torch inside the timed region.  Mamba-2 only.

## Results

Forward and backward TFLOPS, rendered by `plot_results.py` from the dated CSVs
under `results/<variant>/<gpu>/`. Three sweeps per variant: batch 4 over the
sequence length (`<variant>_fixed_batch_*`), sequence length 8192 over the
batch (`<variant>_fixed_seq_*`), and the low-occupancy point batch 1 with 16
heads over the sequence length (`<variant>_low_bh_*`, the regime the exact
piece chain serves). The `cudnn (state on)` bars dump the per-chunk
state-checkpoint series in the forward pass and reuse it in the backward
pass. Runs were captured on GB200 and GB300 (GB300 results shown below).

### GB300 - GDN
![GDN on GB300](results/gdn/gb300/gdn_fixed_batch_flops.webp)
![GDN on GB300, batch 1](results/gdn/gb300/gdn_low_bh_flops.webp)

### GB300 - KDA
![KDA on GB300](results/kda/gb300/kda_fixed_batch_flops.webp)
![KDA on GB300, batch 1](results/kda/gb300/kda_low_bh_flops.webp)

### GB300 - GDN-2
![GDN-2 on GB300](results/gdn2/gb300/gdn2_fixed_batch_flops.webp)
![GDN-2 on GB300, batch 1](results/gdn2/gb300/gdn2_low_bh_flops.webp)

### GB300 - GDP
![GDP on GB300](results/gdp/gb300/gdp_fixed_batch_flops.webp)
![GDP on GB300, batch 1](results/gdp/gb300/gdp_low_bh_flops.webp)

### GB300 - Mamba-2
![Mamba-2 on GB300](results/mamba2/gb300/mamba2_fixed_batch_flops.webp)
![Mamba-2 on GB300, batch 1](results/mamba2/gb300/mamba2_low_bh_flops.webp)
