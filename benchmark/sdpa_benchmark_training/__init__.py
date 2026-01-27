"""
SDPA Benchmark Training Package

This package provides a flexible benchmark configuration system for
Scaled Dot Product Attention (SDPA) operations.

Usage:
    # Run benchmarks from command line
    python -m benchmark.sdpa_benchmark_training.runner --config mlperf

    # Dry run to see what would be executed
    python -m benchmark.sdpa_benchmark_training.runner --config mlperf --dry-run

    # Import and use programmatically
    from benchmark.sdpa_benchmark_training import (
        BenchmarkRunner,
        BenchmarkConfig,
        BenchmarkResult,
        ModelPreset,
        load_config,
    )

    config = load_config("mlperf")
    runner = BenchmarkRunner()
    results = runner.run_config(config)
    runner.save_csv(results, config)
"""

from .config_types import ModelPreset, BenchmarkConfig, BenchmarkResult
from .configs import load_config, list_configs
from .runner import BenchmarkRunner

__all__ = [
    "ModelPreset",
    "BenchmarkConfig",
    "BenchmarkResult",
    "BenchmarkRunner",
    "load_config",
    "list_configs",
]
