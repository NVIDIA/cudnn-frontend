"""
Qwen3 235B Norm Benchmark Configuration

Usage:
    python -m benchmark.norms.runner --config qwen3_235b
    python -m benchmark.norms.runner --config qwen3_235b --dry-run
"""

from ..config_types import NormPreset, NormBenchmarkConfig

QWEN3_235B = NormPreset(
    name="qwen3-235b",
    norm_type="rms_norm",
    N=8192,
    C=5120,
    epsilon=1e-5,
    has_bias=False,
)

CONFIG = NormBenchmarkConfig(
    name="qwen3_235b",
    norms=[QWEN3_235B],
    backends=["cudnn", "pytorch", "torch_compile"],
    data_types=["bfloat16"],
    profile_pass="both",
    num_iterations=20,
    output_dir="results",
)
