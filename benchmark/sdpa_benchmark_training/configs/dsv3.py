"""
DeepSeek V3 SDPA Benchmark Configuration

Benchmarks DeepSeek V3-style MHA with asymmetric head dimensions.
Only causal (top_left) mask - no non-causal benchmarks needed.
Includes forward and backward pass benchmarking with deterministic mode options.

Usage:
    python -m benchmark.sdpa_benchmark_training.runner --config dsv3
    python -m benchmark.sdpa_benchmark_training.runner --config dsv3 --dry-run
"""

from ..config_types import ModelPreset, BenchmarkConfig

DSV3 = ModelPreset(
    name="dsv3",
    num_q_heads=128,
    num_kv_heads=128,
    head_dim_qk=192,
    head_dim_vo=128,
)

CONFIG = BenchmarkConfig(
    name="dsv3",
    models=[DSV3],
    seqlens=[
        (32768, 32768),
        (16384, 16384),
        (8192, 8192),
        (4096, 4096),
        (2048, 2048),
    ],
    backends=["cudnn", "flash_attention_4"],
    data_types=["bfloat16", "fp8"],
    attn_masks=["top_left"],  # Causal only
    profile_pass="both",  # Forward and backward
    deterministic_bwd=[True],
    batch_size=1,
    num_iterations=10,
    output_dir="results",
)
