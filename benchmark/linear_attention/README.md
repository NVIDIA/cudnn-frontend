# Linear Attention Benchmark

## Introduction

This directory contains benchmarking tools for linear attention operations (Gated DeltaNet and its variants) across various backends. The benchmarks target training use cases with support for forward and backward passes, grouped-value attention (GVA), and the per-sequence recurrent state ports (initial state in, final state out).

## Contents

- `Dockerfile` - Docker container setup for running benchmarks
- `benchmark_single_linear_attention.py` - Single linear attention benchmark script

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

# cuDNN Frontend (GDN-2, forward only)
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

# Recurrent state ports: seed with an initial state and request the final
# state (its gradient feeds the backward pass)
python benchmark_single_linear_attention.py \
    --batch_size 1 --seqlen 8192 \
    --num_q_heads 8 --num_kv_heads 64 --head_dim 128 \
    --la_backend cudnn --variant gdn --data_type bfloat16 \
    --skip_ref --fwd_bwd --initial_state --store_on
```

Run `python benchmark_single_linear_attention.py --help` for all options.

Dropping `--skip_ref` validates the forward output against FLA (the same way the SDPA benchmark validates against FlashAttention 4).

## Supported Backends

| Backend | Description |
|---------|-------------|
| `cudnn` | cuDNN (native, via the cuDNN Frontend torch custom ops) |
| `fla`   | FLA (flash-linear-attention, Triton) |

The cuDNN backend routes through the pygraph engines: FROST (Cutlass DSL) on SM100-class devices, the cuTile engines elsewhere. Both passes run through autograd, exactly like a training step.

## Supported Variants

| Variant | Description |
|---------|-------------|
| `gdn`   | Gated DeltaNet: scalar per-token decay and write strength |
| `kda`   | Kimi Delta Attention: per-key-channel decay |
| `gdn2`  | Gated DeltaNet v2: channel-wise decay/erase/write gates (forward only, cuDNN only) |

The benchmark runs `kda` and `gdn2` with the in-kernel q/k L2 normalization off (`use_qk_l2norm_in_kernel=False`) on every backend, for an apples-to-apples comparison.

## Notes

- Head convention: `--num_q_heads` counts the query/key heads and `--num_kv_heads` counts the value heads; the gates, output, and recurrent state live at `max(num_q_heads, num_kv_heads)` heads. Both grouping directions are supported for `gdn`: grouped-value attention (`num_kv_heads > num_q_heads`, v-heads grouped over q-heads) and GQA (`num_q_heads > num_kv_heads`, q-heads grouped over v-heads, e.g. `--num_q_heads 64 --num_kv_heads 8`). The two counts must be equal or one a multiple of the other; `kda` and `gdn2` support the GVA direction only.
- The cuDNN ops use the THD (token-packed) layout internally; the benchmark expresses the dense batch as `cu_seqlens = [0, T, 2T, ...]`.
- `--initial_state` provides a per-sequence fp32 recurrent state (its gradient is produced in the backward pass); `--store_on` requests the per-sequence final state from the forward pass and feeds its gradient in the backward pass. Both are once-per-kernel I/O ports (one `[head_dim_qk, head_dim_vo]` tile per sequence per state head).
- Performance is measured with the torch profiler (device time of the matched kernels), with a 256 MB L2 flush before each timed iteration and the median reported. TFLOPS use the chunked-BMM FLOPs model documented in the script's `flops()`.
