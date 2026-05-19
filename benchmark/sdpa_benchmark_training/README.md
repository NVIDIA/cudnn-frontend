# Scaled Dot Product Attention Benchmark

## Introduction

This directory contains benchmarking tools for Scaled Dot Product Attention (SDPA) operations across various backends. The benchmarks target training use cases with support for causal masking and grouped query attention (GQA).

## Contents

- `Dockerfile` - Docker container setup for running benchmarks
- `benchmark_single_sdpa.py` - Single SDPA benchmark script
- `configs/` - Benchmark configuration files
  - `llama.py` - Llama 3.1 GQA benchmarks (causal + non-causal)
  - `dsv3.py` - DeepSeek V3 MLA benchmarks (asymmetric head dims, causal + non-causal)
  - `kimiK26.py` - Kimi-K2.6 MLA benchmarks (asymmetric head dims, causal + non-causal)
  - `wan22.py` - Wan 2.2 A14B video DiT self-attention benchmarks (bidirectional, no mask)
  - `ltx2.py` - LTX-2 video DiT self-attention benchmarks (bidirectional, no mask)
  - `gpt_oss.py` - GPT-OSS sliding-window-attention GQA benchmarks (causal, SWA=128)
  - `qwen35.py` - Qwen 3.5 GQA benchmarks (head_dim=256, causal, bf16 fwd-only — Blackwell bwd/fp8/fa4 limits)
- `runner.py` - Configuration-based benchmark runner
- `config_types.py` - Data types for benchmark configuration
- `charts.py` - Chart generation utilities
- `../results/` - Benchmark outputs (CSV and charts)

## Quick Start

### 1. Build Docker Container

```bash
docker build -t cudnn_attention_benchmark .

docker run -it --gpus all --rm cudnn_attention_benchmark
```

### 2. Run Benchmarks

```bash
# Run Llama 3.1 benchmark suite
python -m benchmark.sdpa_benchmark_training.runner --config llama

# Run DeepSeek V3 benchmark suite
python -m benchmark.sdpa_benchmark_training.runner --config dsv3

# Run Kimi-K2.6 benchmark suite
python -m benchmark.sdpa_benchmark_training.runner --config kimiK26

# Run GPT-OSS benchmark suite (sliding window attention, W=128)
python -m benchmark.sdpa_benchmark_training.runner --config gpt_oss

# Run Wan 2.2 A14B benchmark suite
python -m benchmark.sdpa_benchmark_training.runner --config wan22

# Run LTX-2 benchmark suite
python -m benchmark.sdpa_benchmark_training.runner --config ltx2

# Run Qwen 3.5 benchmark suite (cuDNN bf16 fwd only at head_dim=256)
python -m benchmark.sdpa_benchmark_training.runner --config qwen35

# Dry run (show what would be executed)
python -m benchmark.sdpa_benchmark_training.runner --config llama --dry-run

# Filter by backend
python -m benchmark.sdpa_benchmark_training.runner --config llama --backend cudnn

# Filter by data type
python -m benchmark.sdpa_benchmark_training.runner --config llama --dtype bfloat16
```

## Configuration-Based Benchmarking

### Creating Custom Configurations

1. Copy the template:
   ```bash
   cp configs/llama.py configs/my_config.py
   ```

2. Edit your config:
   ```python
   from ..config_types import ModelPreset, BenchmarkConfig

   MY_MODEL = ModelPreset(
       name="my_model",
       num_q_heads=32,
       num_kv_heads=8,
       head_dim=128,
   )

   CONFIG = BenchmarkConfig(
       name="my_benchmark",
       models=[MY_MODEL],
       seqlens=[(4096, 4096), (8192, 8192)],
       backends=["cudnn", "flash_attention_4"],
       data_types=["bfloat16", "fp8"],
       attn_masks=["top_left", "no_mask"],
       profile_pass="fwd",  # "fwd", "bwd", or "both"
       num_iterations=10,
   )
   ```

3. Run:
   ```bash
   python -m benchmark.sdpa_benchmark_training.runner --config my_config
   ```

### Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `models` | List of `ModelPreset` to benchmark | Required |
| `seqlens` | List of `(q_seqlen, kv_seqlen)` tuples | Required |
| `backends` | Backends to compare | `["cudnn"]` |
| `data_types` | Data types to test | `["bfloat16"]` |
| `attn_masks` | Attention masks (`top_left`, `no_mask`) | `["top_left"]` |
| `profile_pass` | Which pass to profile (`fwd`, `bwd`, `both`) | `"fwd"` |
| `batch_size` | Batch size | `1` |
| `num_iterations` | Iterations per benchmark | `10` |
| `deterministic_bwd` | Deterministic modes for backward | `[False]` |
| `sliding_window_size` | Sliding window attention size (requires `top_left` mask) | `None` |

### Model Presets

Standard model:
```python
LLAMA3_1 = ModelPreset(
    name="llama3.1",
    num_q_heads=64,
    num_kv_heads=8,
    head_dim=128,
)
```

Asymmetric head dimensions (DeepSeek V3):
```python
DSV3 = ModelPreset(
    name="dsv3",
    num_q_heads=128,
    num_kv_heads=128,
    head_dim_qk=192,  # Q/K head dimension
    head_dim_vo=128,  # V/O head dimension
)
```

Sliding window attention (GPT-OSS):
```python
GPT_OSS = ModelPreset(
    name="gpt_oss",
    num_q_heads=64,
    num_kv_heads=8,  # GQA with 8:1 ratio
    head_dim=64,
)

CONFIG = BenchmarkConfig(
    ...
    sliding_window_size=128,  # Look back 128 tokens from diagonal (requires top_left mask)
)
```

### Output

The runner produces (in `benchmark/results/`):
- **CSV**: `<config>_<timestamp>.csv` — one row per (backend, dtype, mask, seqlen, profile_pass, deterministic_bwd)
- **Charts**: Separate chart per mask type:
  - `<config>_top_left.png` (causal)
  - `<config>_no_mask.png` (non-causal)
  - `<config>_<mask>_det_overhead.png` — bwd bf16 only, comparing deterministic vs non-deterministic for cuDNN and FAv4 side-by-side
- Main charts filter to `deterministic_bwd=False`; the det-overhead chart shows both modes
- Backend legends include the cuDNN backend version (e.g. `cudnn 9.22.0 (BF16)`)

## Single Benchmark Script

For running individual benchmarks:

```bash
# cuDNN Frontend (BF16)
python benchmark_single_sdpa.py \
    --batch_size 1 --q_seqlen 8192 --kv_seqlen 8192 \
    --num_q_heads 64 --num_kv_heads 8 --head_dim 128 \
    --sdpa_backend cudnn --data_type bfloat16 \
    --attn_mask top_left --fwd_bwd

# cuDNN Frontend (FP8)
python benchmark_single_sdpa.py \
    --batch_size 1 --q_seqlen 8192 --kv_seqlen 8192 \
    --num_q_heads 64 --num_kv_heads 8 --head_dim 128 \
    --sdpa_backend cudnn --data_type fp8 \
    --attn_mask top_left --fwd_bwd

# FlashAttention 4
python benchmark_single_sdpa.py \
    --batch_size 1 --q_seqlen 8192 --kv_seqlen 8192 \
    --num_q_heads 64 --num_kv_heads 8 --head_dim 128 \
    --sdpa_backend flash_attention_4 --data_type bfloat16 \
    --attn_mask top_left --fwd_bwd
```

Run `python benchmark_single_sdpa.py --help` for all options.

## Comparing With `test_repro --perf`

The benchmark script above is useful for standalone SDPA timing. To compare those numbers against the Python SDPA test harness, use `pytest ...::test_repro --perf`.

`--perf` enables timing-only execution for `test_repro`. In this mode, the test prints the median `graph.execute` time and skips the reference path so the measurement matches the benchmark workflow more closely.

### Timing Methods

- `--timing_method cupti`: Uses `torch.profiler` device time. This is the default and is the closest match to `benchmark_single_sdpa.py`.
- `--timing_method events`: Uses CUDA events for comparison.

### Example: Causal Training Case

```bash
# benchmark (CUPTI-style timing)
python benchmark/sdpa_benchmark_training/benchmark_single_sdpa.py \
    --batch_size 1 --q_seqlen 8192 --kv_seqlen 8192 \
    --num_q_heads 64 --num_kv_heads 8 --head_dim 128 \
    --sdpa_backend cudnn --data_type bfloat16 \
    --attn_mask top_left --skip_ref --fwd_bwd

# test_repro --perf (CUPTI by default)
pytest -vv -s test/python/test_mhas_v2.py::test_repro --perf --repro "{
    'data_type': 'torch.bfloat16',
    'is_infer': False,
    'is_padding': False, 'is_alibi': None, 'is_bias': None, 'is_dropout': None,
    'with_sink_token': False, 'with_score_max': False, 'with_score_sum_exp': False,
    'batches': 1, 'd_qk': 128, 'd_v': 128,
    's_q': 8192, 's_kv': 8192,
    'h_q': 64, 'h_k': 8, 'h_v': 8,
    'shape_q': (1, 64, 8192, 128), 'stride_q': (67108864, 128, 8192, 1),
    'shape_k': (1, 8, 8192, 128),  'stride_k': (8388608, 128, 1024, 1),
    'shape_v': (1, 8, 8192, 128),  'stride_v': (8388608, 128, 1024, 1),
    'shape_o': (1, 64, 8192, 128), 'stride_o': (67108864, 128, 8192, 1),
    'diag_align': 'cudnn.diagonal_alignment.TOP_LEFT',
    'right_bound': 0,
    'rng_data_seed': 42, 'rng_geom_seed': 42,
}"

# test_repro --perf with CUDA events instead of CUPTI
pytest -vv -s test/python/test_mhas_v2.py::test_repro --perf --timing_method events --repro "{
    'data_type': 'torch.bfloat16',
    'is_infer': False,
    'is_padding': False, 'is_alibi': None, 'is_bias': None, 'is_dropout': None,
    'with_sink_token': False, 'with_score_max': False, 'with_score_sum_exp': False,
    'batches': 1, 'd_qk': 128, 'd_v': 128,
    's_q': 8192, 's_kv': 8192,
    'h_q': 64, 'h_k': 8, 'h_v': 8,
    'shape_q': (1, 64, 8192, 128), 'stride_q': (67108864, 128, 8192, 1),
    'shape_k': (1, 8, 8192, 128),  'stride_k': (8388608, 128, 1024, 1),
    'shape_v': (1, 8, 8192, 128),  'stride_v': (8388608, 128, 1024, 1),
    'shape_o': (1, 64, 8192, 128), 'stride_o': (67108864, 128, 8192, 1),
    'diag_align': 'cudnn.diagonal_alignment.TOP_LEFT',
    'right_bound': 0,
    'rng_data_seed': 42, 'rng_geom_seed': 42,
}"
```

## Programmatic Usage

```python
from benchmark.sdpa_benchmark_training import (
    BenchmarkRunner,
    BenchmarkConfig,
    ModelPreset,
    load_config,
)

# Load existing config
config = load_config("llama")

# Or create programmatically
config = BenchmarkConfig(
    name="custom",
    models=[ModelPreset("test", 64, 8, 128)],
    seqlens=[(4096, 4096)],
    backends=["cudnn"],
)

runner = BenchmarkRunner()
results = runner.run_config(config)
runner.save_csv(results, config)
```

## Supported Backends

| Backend | Description |
|---------|-------------|
| `cudnn` | cuDNN (native, via cuDNN Frontend) |
| `flash_attention_4` | FlashAttention 4 |
| `flash_attention_3` | FlashAttention 3 |
| `pyt_flash_attention` | PyTorch FlashAttention |
| `pyt_cudnn` | PyTorch cuDNN backend |
| `pyt_efficient_attention` | PyTorch xFormers |

## Benchmark Results

Results are organized by `<config>/<gpu>/`. The plots compare cuDNN against FAv4 across BF16, MXFP8, and FP8 (cuDNN-only for FP8/MXFP8). Per-config layout:

```
results/<config>/<gpu>/
    <config>_<timestamp>.csv
    <config>_<mask>.png                  # main fwd+bwd chart, non-deterministic only
    <config>_<mask>_det_overhead.png     # bwd bf16: det vs non-det comparison
```

Runs were captured on Lyris GB200 and GB300 with cuDNN 9.22.0 and FAv4 4.0.0b10.

### GB300 - Llama 3.1 Causal (top_left)
![Llama 3.1 Causal on GB300](results/llama3.1/gb300/llama3.1_top_left.png)
- `batch=2; num_q_heads=64; num_kv_heads=8; head_dim=128`

### GB300 - Llama 3.1 Non-Causal (no_mask)
![Llama 3.1 Non-Causal on GB300](results/llama3.1/gb300/llama3.1_no_mask.png)
- `batch=2; num_q_heads=64; num_kv_heads=8; head_dim=128`

### GB300 - DeepSeek V3 (MLA, asymmetric head dims)
![DeepSeek V3 Causal on GB300](results/dsv3/gb300/dsv3_top_left.png)
![DeepSeek V3 Non-Causal on GB300](results/dsv3/gb300/dsv3_no_mask.png)
- `batch=2; num_q_heads=128; num_kv_heads=128; head_dim_qk=192; head_dim_vo=128`

### GB300 - Kimi-K2.6 (MLA, asymmetric head dims)
![Kimi-K2.6 Causal on GB300](results/kimiK26/gb300/kimiK26_top_left.png)
![Kimi-K2.6 Non-Causal on GB300](results/kimiK26/gb300/kimiK26_no_mask.png)
- `batch=2; num_q_heads=64; num_kv_heads=64; head_dim_qk=192; head_dim_vo=128`

### GB300 - Wan 2.2 (video DiT, bidirectional)
![Wan 2.2 Non-Causal on GB300](results/wan22/gb300/wan22_no_mask.png)
- `batch=1; num_q_heads=40; num_kv_heads=40; head_dim=128`

### GB300 - LTX-2 (video DiT, bidirectional)
![LTX-2 Non-Causal on GB300](results/ltx2/gb300/ltx2_no_mask.png)
- `batch=1; num_q_heads=32; num_kv_heads=32; head_dim=128`

### GB300 - GPT-OSS (sliding window attention, W=128)
![GPT-OSS Causal on GB300](results/gpt_oss/gb300/gpt_oss_top_left.png)
- `batch=2; num_q_heads=128; num_kv_heads=128; head_dim=64; sliding_window_size=128`

### GB300 - Qwen 3.5 (head_dim=256, fwd only)
![Qwen 3.5 Causal on GB300](results/qwen35/gb300/qwen35_top_left.png)
- `batch=2; num_q_heads=32; num_kv_heads=2; head_dim=256` — cuDNN BF16 fwd only at head_dim=256 on Blackwell

GB200 results are available under the same layout at `results/<config>/gb200/`.
