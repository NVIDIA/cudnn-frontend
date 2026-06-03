"""
Benchmark configuration loading utilities.

This module provides functions to load benchmark configurations by name.
"""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config_types import NormBenchmarkConfig


def load_config(name: str) -> "NormBenchmarkConfig":
    """
    Load a benchmark configuration by name.

    Args:
        name: Name of the config (without .py extension)

    Returns:
        NormBenchmarkConfig instance
    """
    try:
        module = importlib.import_module(f".{name}", package=__package__)
    except ModuleNotFoundError:
        raise ValueError(f"Config '{name}' not found. Create a file at configs/{name}.py with a CONFIG variable.")

    if not hasattr(module, "CONFIG"):
        raise ValueError(f"Config module '{name}' must define a CONFIG variable of type NormBenchmarkConfig")

    return module.CONFIG


def list_configs() -> list:
    """List available config names."""
    from pathlib import Path

    configs_dir = Path(__file__).parent
    configs = []

    for f in configs_dir.iterdir():
        if f.suffix == ".py" and f.stem != "__init__":
            configs.append(f.stem)

    return sorted(configs)
