"""
Mixtral 8x7B Norm Benchmark Configuration

Usage:
    python -m benchmark.norms.runner --config mixtral_8x7b
    python -m benchmark.norms.runner --config mixtral_8x7b --dry-run
"""

from ..config_types import NormPreset, NormBenchmarkConfig

MIXTRAL_8X7B = NormPreset(
    name="mixtral-8x7b",
    norm_type="rms_norm",
    N=8192,
    C=4096,
    epsilon=1e-5,
    has_bias=False,
)

CONFIG = NormBenchmarkConfig(
    name="mixtral_8x7b",
    norms=[MIXTRAL_8X7B],
    backends=["cudnn", "pytorch", "torch_compile"],
    data_types=["bfloat16"],
    profile_pass="both",
    num_iterations=20,
    output_dir="results",
)
