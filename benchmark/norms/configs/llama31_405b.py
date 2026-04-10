"""
Llama 3.1 405B Norm Benchmark Configuration

Usage:
    python -m benchmark.norms.runner --config llama31_405b
    python -m benchmark.norms.runner --config llama31_405b --dry-run
"""

from ..config_types import NormPreset, NormBenchmarkConfig

LLAMA31_405B = NormPreset(
    name="llama31-405b",
    norm_type="rms_norm",
    N=4096,
    C=16384,
    epsilon=1e-5,
    has_bias=False,
)

CONFIG = NormBenchmarkConfig(
    name="llama31_405b",
    norms=[LLAMA31_405B],
    backends=["cudnn", "quack", "pytorch", "torch_compile"],
    data_types=["bfloat16"],
    profile_pass="both",
    num_iterations=20,
    output_dir="results",
)
