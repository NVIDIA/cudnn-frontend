"""
GPT OSS SDPA Benchmark Configuration

Benchmarks GPT-style attention with sliding window attention (SWA) and GQA.
Uses a 128-token sliding window for local attention.

Usage:
    python -m benchmark.sdpa_benchmark_training.runner --config gpt_oss
    python -m benchmark.sdpa_benchmark_training.runner --config gpt_oss --dry-run
"""

from ..config_types import ModelPreset, BenchmarkConfig

GPT_OSS = ModelPreset(
    name="gpt_oss",
    num_q_heads=64,
    num_kv_heads=8,  # GQA with 8:1 ratio
    head_dim=64,
)

CONFIG = BenchmarkConfig(
    name="gpt_oss",
    models=[GPT_OSS],
    seqlens=[
        (32768, 32768),
        (16384, 16384),
        (8192, 8192),
        (4096, 4096),
        (2048, 2048),
    ],
    backends=["cudnn", "flash_attention_4"],
    data_types=["bfloat16", "fp8", "mxfp8"],
    attn_masks=["top_left"],  # Causal with sliding window
    profile_pass="both",  # Forward and backward
    deterministic_bwd=[False],
    sliding_window_size=1024,  # 1024-token sliding window
    batch_size=2,
    num_iterations=10,
    output_dir="results",
)
