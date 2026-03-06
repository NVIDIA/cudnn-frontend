"""
GPT-3 175B Norm Benchmark Configuration

Usage:
    python -m benchmark.norms.runner --config gpt3_175b
    python -m benchmark.norms.runner --config gpt3_175b --dry-run
"""

from ..config_types import NormPreset, NormBenchmarkConfig

GPT3_175B = NormPreset(
    name="gpt3-175b",
    norm_type="layer_norm",
    N=1024,
    C=12288,
    epsilon=1e-5,
    has_bias=True,
)

CONFIG = NormBenchmarkConfig(
    name="gpt3_175b",
    norms=[GPT3_175B],
    backends=["cudnn", "pytorch", "torch_compile"],
    data_types=["bfloat16"],
    profile_pass="both",
    num_iterations=20,
    output_dir="results",
)
