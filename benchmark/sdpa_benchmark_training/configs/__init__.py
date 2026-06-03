"""
Benchmark configuration loading utilities.

This module provides functions to load benchmark configurations by name.
"""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config_types import BenchmarkConfig


def load_config(name: str) -> "BenchmarkConfig":
    """
    Load a benchmark configuration by name.

    Configurations are Python modules in the configs directory.
    Each module should define a CONFIG variable of type BenchmarkConfig.

    Args:
        name: Name of the config (without .py extension)

    Returns:
        BenchmarkConfig instance

    Raises:
        ValueError: If config not found or doesn't define CONFIG

    Example:
        config = load_config("mlperf")
        print(config.name)  # "mlperf"
    """
    try:
        module = importlib.import_module(f".{name}", package=__package__)
    except ModuleNotFoundError:
        raise ValueError(f"Config '{name}' not found. " f"Create a file at configs/{name}.py with a CONFIG variable.")

    if not hasattr(module, "CONFIG"):
        raise ValueError(f"Config module '{name}' must define a CONFIG variable of type BenchmarkConfig")

    return module.CONFIG


def list_configs() -> list:
    """
    List available config names.

    Returns:
        List of config names (without .py extension)
    """
    import os
    from pathlib import Path

    configs_dir = Path(__file__).parent
    configs = []

    for f in configs_dir.iterdir():
        if f.suffix == ".py" and f.stem != "__init__":
            configs.append(f.stem)

    return sorted(configs)
