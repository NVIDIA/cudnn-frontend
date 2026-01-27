"""
Llama 3.1 SDPA Benchmark Configuration

Benchmarks Llama 3.1 405B-style GQA attention with both causal and non-causal masks.
Includes forward and backward pass benchmarking with deterministic mode options.

Usage:
    python -m benchmark.sdpa_benchmark_training.runner --config llama
    python -m benchmark.sdpa_benchmark_training.runner --config llama --dry-run
"""

from ..config_types import ModelPreset, BenchmarkConfig

LLAMA3_1 = ModelPreset(
    name="llama3.1",
    num_q_heads=64,
    num_kv_heads=8,
    head_dim=128,
)

CONFIG = BenchmarkConfig(
    name="llama3.1",
    models=[LLAMA3_1],
    seqlens=[
        (32768, 32768),
        (16384, 16384),
        (8192, 8192),
        (4096, 4096),
        (2048, 2048),
    ],
    backends=["cudnn", "flash_attention_4"],
    data_types=["bfloat16", "fp8"],
    attn_masks=["top_left", "no_mask"],  # Both causal and non-causal
    profile_pass="both",  # Forward and backward
    deterministic_bwd=[False],
    batch_size=1,
    num_iterations=10,
    output_dir="results",
)
