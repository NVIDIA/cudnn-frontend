"""
NemotronH 56B Norm Benchmark Configuration

Usage:
    python -m benchmark.norms.runner --config nemotronh_56b
    python -m benchmark.norms.runner --config nemotronh_56b --dry-run
"""

from ..config_types import NormPreset, NormBenchmarkConfig

NEMOTRONH_56B = NormPreset(
    name="nemotronh-56b",
    norm_type="rms_norm",
    N=4096,
    C=8192,
    epsilon=1e-5,
    has_bias=False,
)

CONFIG = NormBenchmarkConfig(
    name="nemotronh_56b",
    norms=[NEMOTRONH_56B],
    backends=["cudnn", "quack", "pytorch", "torch_compile"],
    data_types=["bfloat16"],
    profile_pass="both",
    num_iterations=20,
    output_dir="results",
)
