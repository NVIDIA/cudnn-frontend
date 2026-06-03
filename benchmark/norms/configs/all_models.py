"""
All Models Norm Benchmark Configuration

Combines all unique norm presets from every model config into a single suite.

Usage:
    python -m benchmark.norms.runner --config all_models
    python -m benchmark.norms.runner --config all_models --dry-run
"""

from ..config_types import NormBenchmarkConfig

from .llama3_8b import LLAMA3_8B
from .llama3_70b import LLAMA3_70B
from .llama31_405b import LLAMA31_405B
from .llama4_e16 import LLAMA4_E16
from .gpt3_175b import GPT3_175B
from .mixtral_8x7b import MIXTRAL_8X7B
from .mixtral_8x22b import MIXTRAL_8X22B
from .nemotronh_56b import NEMOTRONH_56B
from .deepseek_v3 import DEEPSEEK_V3_2048X4096, DEEPSEEK_V3_131072X128, DEEPSEEK_V3_8192X128
from .qwen3_235b import QWEN3_235B
from .qwen3_30b import QWEN3_30B_4096X2048, QWEN3_30B_131072X128, QWEN3_30B_16384X128

CONFIG = NormBenchmarkConfig(
    name="all_models",
    norms=[
        LLAMA3_8B,
        LLAMA3_70B,
        LLAMA31_405B,
        LLAMA4_E16,
        GPT3_175B,
        MIXTRAL_8X7B,
        MIXTRAL_8X22B,
        NEMOTRONH_56B,
        DEEPSEEK_V3_2048X4096,
        DEEPSEEK_V3_131072X128,
        DEEPSEEK_V3_8192X128,
        QWEN3_235B,
        QWEN3_30B_4096X2048,
        QWEN3_30B_131072X128,
        QWEN3_30B_16384X128,
    ],
    backends=["cudnn", "quack", "pytorch", "torch_compile"],
    data_types=["bfloat16"],
    profile_pass="both",
    num_iterations=20,
    output_dir="results",
)
