"""
Norm Benchmark Package

This package provides a flexible benchmark configuration system for
normalization operations (RMSNorm, LayerNorm).

Usage:
    python -m benchmark.norms.runner --config llama3_8b
    python -m benchmark.norms.runner --config llama3_8b --dry-run
"""

from .config_types import NormPreset, NormBenchmarkConfig, NormBenchmarkResult
from .configs import load_config, list_configs
from .runner import NormBenchmarkRunner

__all__ = [
    "NormPreset",
    "NormBenchmarkConfig",
    "NormBenchmarkResult",
    "NormBenchmarkRunner",
    "load_config",
    "list_configs",
]
