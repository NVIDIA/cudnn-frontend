"""
Llama 3 70B Norm Benchmark Configuration

Usage:
    python -m benchmark.norms.runner --config llama3_70b
    python -m benchmark.norms.runner --config llama3_70b --dry-run
"""

from ..config_types import NormPreset, NormBenchmarkConfig

LLAMA3_70B = NormPreset(
    name="llama3-70b",
    norm_type="rms_norm",
    N=8192,
    C=8192,
    epsilon=1e-5,
    has_bias=False,
)

CONFIG = NormBenchmarkConfig(
    name="llama3_70b",
    norms=[LLAMA3_70B],
    backends=["cudnn", "quack", "pytorch", "torch_compile"],
    data_types=["bfloat16"],
    profile_pass="both",
    num_iterations=20,
    output_dir="results",
)
