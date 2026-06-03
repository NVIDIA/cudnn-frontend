"""
Kimi-K2.6 SDPA Benchmark Configuration

Benchmarks Kimi-K2.6 text transformer attention. Kimi-K2.6 uses Multi-head
Latent Attention (same family as DeepSeek V3) with asymmetric head dims:

    head_dim_qk = qk_nope_head_dim + qk_rope_head_dim = 128 + 64 = 192
    head_dim_vo = v_head_dim                          = 128

with 64 Q heads and 64 K/V heads (MHA, not GQA). Source:
``Kimi-K2.6/config.json`` (text_config).

This shape is the one cuDNN explicitly supports past head_dim=128 on both
fwd AND bwd (requires cuDNN 9.19+), so full training benchmarking works.

Usage:
    python -m benchmark.sdpa_benchmark_training.runner --config kimiK26
    python -m benchmark.sdpa_benchmark_training.runner --config kimiK26 --dry-run
"""

from ..config_types import ModelPreset, BenchmarkConfig

KIMI_K26 = ModelPreset(
    name="kimiK26",
    num_q_heads=64,
    num_kv_heads=64,
    head_dim_qk=192,
    head_dim_vo=128,
)

CONFIG = BenchmarkConfig(
    name="kimiK26",
    models=[KIMI_K26],
    seqlens=[
        (32768, 32768),
        (16384, 16384),
        (8192, 8192),
        (4096, 4096),
        (2048, 2048),
    ],
    backends=["cudnn", "flash_attention_4"],
    data_types=["bfloat16", "fp8", "mxfp8"],
    attn_masks=["top_left", "no_mask"],
    profile_pass="both",  # Forward and backward
    deterministic_bwd=[False, True],
    batch_size=2,
    num_iterations=10,
    output_dir="results",
)
