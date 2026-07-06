# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Pytest support for optional JAX tests."""

from importlib import import_module

import pytest

_JAX_TEST_PREFIX = "test_jax_"
_JAX_INSTALL_HINT = "pip install 'nvidia-cudnn-frontend[jax]'"


def _requested_module_is_missing(module_name, error):
    missing_name = error.name
    return missing_name is not None and (
        module_name == missing_name or module_name.startswith(f"{missing_name}.")
    )


def _jax_test_skip_reason():
    try:
        jax = import_module("jax")
    except ModuleNotFoundError as error:
        if not _requested_module_is_missing("jax", error):
            raise
        return f"JAX tests require JAX. Install it with `{_JAX_INSTALL_HINT}`."

    try:
        cutlass_jax = import_module("cutlass.jax")
    except ModuleNotFoundError as error:
        if not _requested_module_is_missing("cutlass.jax", error):
            raise
        return (
            "JAX tests require CUTLASS JAX support. "
            f"Install it with `{_JAX_INSTALL_HINT}`."
        )

    is_available = getattr(cutlass_jax, "is_available", None)
    if not callable(is_available):
        raise AttributeError("cutlass.jax must define is_available()")
    if is_available():
        return None

    installed_version = getattr(jax, "__version__", "unknown")
    minimum_version_info = getattr(
        cutlass_jax,
        "CUTE_DSL_MIN_SUPPORTED_JAX_VERSION",
        None,
    )
    if minimum_version_info is None:
        return f"CUTLASS JAX support is unavailable with JAX {installed_version}."

    minimum_version = ".".join(str(part) for part in minimum_version_info)
    return (
        f"CUTLASS JAX support is unavailable with JAX {installed_version}; "
        f"the minimum supported JAX version is {minimum_version}."
    )


def pytest_collection_modifyitems(config, items):
    del config
    jax_items = [item for item in items if item.path.name.startswith(_JAX_TEST_PREFIX)]
    if not jax_items:
        return

    reason = _jax_test_skip_reason()
    if reason is None:
        return

    skip = pytest.mark.skip(reason=reason)
    for item in jax_items:
        item.add_marker(skip)
