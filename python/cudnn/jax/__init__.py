# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Public optional-dependency boundary for JAX operation APIs."""

_INSTALL_HINT = "pip install 'nvidia-cudnn-frontend[jax]'"
_OPTIONAL_DEPENDENCY_PREFIXES = ("cuda", "cutlass", "jax")

try:
    import cutlass.jax as _cutlass_jax
    import jax as _jax
except ModuleNotFoundError as error:
    missing_name = error.name
    if missing_name is None or not any(missing_name == prefix or missing_name.startswith(f"{prefix}.") for prefix in _OPTIONAL_DEPENDENCY_PREFIXES):
        raise
    raise ImportError(f"cuDNN JAX APIs require the optional JAX dependencies. Install them with `{_INSTALL_HINT}`.") from error

if not _cutlass_jax.is_available():
    minimum_version = ".".join(str(part) for part in _cutlass_jax.CUTE_DSL_MIN_SUPPORTED_JAX_VERSION)
    installed_version = getattr(_jax, "__version__", "unknown")
    raise ImportError(
        f"CUTLASS JAX support is unavailable with JAX {installed_version}; "
        f"the minimum supported JAX version is {minimum_version}. "
        f"Install the optional JAX dependencies with `{_INSTALL_HINT}`."
    )

from .._jax import JaxApiBase, JaxTensorDesc, disable_device_compatibility_checks
from ..rmsnorm_rht_amax.jax import (
    RmsNormRhtAmaxSm100,
    rmsnorm_rht_amax_sm100,
)

__all__ = [
    "JaxApiBase",
    "JaxTensorDesc",
    "disable_device_compatibility_checks",
    "RmsNormRhtAmaxSm100",
    "rmsnorm_rht_amax_sm100",
]
