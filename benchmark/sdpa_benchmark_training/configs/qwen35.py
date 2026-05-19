"""
Qwen 3.5 SDPA Benchmark Configuration

Benchmarks Qwen 3.5-style GQA attention with causal (top_left) mask only.
32 Q heads and 2 KV heads (16:1 GQA) with head_dim 256.
Forward-only pass with bfloat16 (backward blocked at head_dim=256 on Blackwell).

Usage:
    python -m benchmark.sdpa_benchmark_training.runner --config qwen35
    python -m benchmark.sdpa_benchmark_training.runner --config qwen35 --dry-run
"""

from ..config_types import ModelPreset, BenchmarkConfig

QWEN35 = ModelPreset(
    name="qwen35",
    num_q_heads=32,
    num_kv_heads=2,
    head_dim=256,
)

CONFIG = BenchmarkConfig(
    name="qwen35",
    models=[QWEN35],
    seqlens=[
        (32768, 32768),
        (16384, 16384),
        (8192, 8192),
        (4096, 4096),
        (2048, 2048),
    ],
    backends=["cudnn", "flash_attention_4"],
    # Blackwell limits at head_dim=256: cuDNN bwd rejects head_dim>128 at
    # graph_bwd.validate(), and fa4's sm100 forward kernel asserts on tmem
    # exhaustion for head_dim=256 regardless of batch. Restrict to cuDNN fwd.
    # fa4 rows are kept (they fail) to document the sm100 kernel limitation.
    data_types=["bfloat16"],
    attn_masks=["top_left"],  # Causal only
    profile_pass="fwd",  # Forward-only (bwd blocked at head_dim=256)
    deterministic_bwd=[False, True],
    batch_size=1,
    num_iterations=10,
    output_dir="results",
)
