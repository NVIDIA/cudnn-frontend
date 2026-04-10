"""
Mixtral 8x22B Norm Benchmark Configuration

Usage:
    python -m benchmark.norms.runner --config mixtral_8x22b
    python -m benchmark.norms.runner --config mixtral_8x22b --dry-run
"""

from ..config_types import NormPreset, NormBenchmarkConfig

MIXTRAL_8X22B = NormPreset(
    name="mixtral-8x22b",
    norm_type="rms_norm",
    N=4096,
    C=6144,
    epsilon=1e-5,
    has_bias=False,
)

CONFIG = NormBenchmarkConfig(
    name="mixtral_8x22b",
    norms=[MIXTRAL_8X22B],
    backends=["cudnn", "quack", "pytorch", "torch_compile"],
    data_types=["bfloat16"],
    profile_pass="both",
    num_iterations=20,
    output_dir="results",
)
