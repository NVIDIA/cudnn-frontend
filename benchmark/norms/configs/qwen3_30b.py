"""
Qwen3 30B Norm Benchmark Configuration

Usage:
    python -m benchmark.norms.runner --config qwen3_30b
    python -m benchmark.norms.runner --config qwen3_30b --dry-run
"""

from ..config_types import NormPreset, NormBenchmarkConfig

QWEN3_30B_4096X2048 = NormPreset(
    name="qwen3-30b-4096x2048",
    norm_type="rms_norm",
    N=4096,
    C=2048,
    epsilon=1e-6,
    has_bias=False,
)

QWEN3_30B_131072X128 = NormPreset(
    name="qwen3-30b-131072x128",
    norm_type="rms_norm",
    N=131072,
    C=128,
    epsilon=1e-6,
    has_bias=False,
)

QWEN3_30B_16384X128 = NormPreset(
    name="qwen3-30b-16384x128",
    norm_type="rms_norm",
    N=16384,
    C=128,
    epsilon=1e-6,
    has_bias=False,
)

CONFIG = NormBenchmarkConfig(
    name="qwen3_30b",
    norms=[QWEN3_30B_4096X2048, QWEN3_30B_131072X128, QWEN3_30B_16384X128],
    backends=["cudnn", "quack", "pytorch", "torch_compile"],
    data_types=["bfloat16"],
    profile_pass="both",
    num_iterations=20,
    output_dir="results",
)
