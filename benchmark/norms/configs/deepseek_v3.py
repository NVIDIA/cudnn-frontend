"""
DeepSeek V3 Norm Benchmark Configuration

Usage:
    python -m benchmark.norms.runner --config deepseek_v3
    python -m benchmark.norms.runner --config deepseek_v3 --dry-run
"""

from ..config_types import NormPreset, NormBenchmarkConfig

DEEPSEEK_V3_2048X4096 = NormPreset(
    name="deepseek-v3-2048x4096",
    norm_type="rms_norm",
    N=2048,
    C=4096,
    epsilon=1e-6,
    has_bias=False,
)

DEEPSEEK_V3_131072X128 = NormPreset(
    name="deepseek-v3-131072x128",
    norm_type="rms_norm",
    N=131072,
    C=128,
    epsilon=1e-6,
    has_bias=False,
)

DEEPSEEK_V3_8192X128 = NormPreset(
    name="deepseek-v3-8192x128",
    norm_type="rms_norm",
    N=8192,
    C=128,
    epsilon=1e-6,
    has_bias=False,
)

CONFIG = NormBenchmarkConfig(
    name="deepseek_v3",
    norms=[DEEPSEEK_V3_2048X4096, DEEPSEEK_V3_131072X128, DEEPSEEK_V3_8192X128],
    backends=["cudnn", "pytorch", "torch_compile"],
    data_types=["bfloat16"],
    profile_pass="both",
    num_iterations=20,
    output_dir="results",
)
