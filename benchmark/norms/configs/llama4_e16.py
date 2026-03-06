"""
Llama 4 E16 Norm Benchmark Configuration

Usage:
    python -m benchmark.norms.runner --config llama4_e16
    python -m benchmark.norms.runner --config llama4_e16 --dry-run
"""

from ..config_types import NormPreset, NormBenchmarkConfig

LLAMA4_E16 = NormPreset(
    name="llama4-e16",
    norm_type="rms_norm",
    N=8192,
    C=5120,
    epsilon=1e-5,
    has_bias=False,
)

CONFIG = NormBenchmarkConfig(
    name="llama4_e16",
    norms=[LLAMA4_E16],
    backends=["cudnn", "pytorch", "torch_compile"],
    data_types=["bfloat16"],
    profile_pass="both",
    num_iterations=20,
    output_dir="results",
)
