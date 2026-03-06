"""
Llama 3 8B Norm Benchmark Configuration

Usage:
    python -m benchmark.norms.runner --config llama3_8b
    python -m benchmark.norms.runner --config llama3_8b --dry-run
"""

from ..config_types import NormPreset, NormBenchmarkConfig

LLAMA3_8B = NormPreset(
    name="llama3-8b",
    norm_type="rms_norm",
    N=16384,
    C=4096,
    epsilon=1e-5,
    has_bias=False,
)

CONFIG = NormBenchmarkConfig(
    name="llama3_8b",
    norms=[LLAMA3_8B],
    backends=["cudnn", "pytorch", "torch_compile"],
    data_types=["bfloat16"],
    profile_pass="both",
    num_iterations=20,
    output_dir="results",
)
