# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Optional-dependency boundary shared by all JAX operation adapters."""

from importlib import import_module

_INSTALL_HINT = "pip install 'nvidia-cudnn-frontend[jax]'"
_OPTIONAL_DEPENDENCY_PREFIXES = ("cuda", "cutlass", "jax")


def _require_dependency(module_name: str):
    try:
        return import_module(module_name)
    except ModuleNotFoundError as error:
        missing_name = error.name
        if missing_name is None or not any(missing_name == prefix or missing_name.startswith(f"{prefix}.") for prefix in _OPTIONAL_DEPENDENCY_PREFIXES):
            raise
        raise ImportError(f"cuDNN JAX APIs require {module_name!r}. " f"Install the optional JAX dependencies with `{_INSTALL_HINT}`.") from error


jax = _require_dependency("jax")
cutlass_jax = _require_dependency("cutlass.jax")

if not cutlass_jax.is_available():
    minimum_version = ".".join(str(part) for part in cutlass_jax.CUTE_DSL_MIN_SUPPORTED_JAX_VERSION)
    installed_version = getattr(jax, "__version__", "unknown")
    raise ImportError(
        f"CUTLASS JAX support is unavailable with JAX {installed_version}; "
        f"the minimum supported JAX version is {minimum_version}. "
        f"Install the optional JAX dependencies with `{_INSTALL_HINT}`."
    )

from .api_base import JaxApiBase, JaxTensorDesc

__all__ = ["JaxApiBase", "JaxTensorDesc"]
