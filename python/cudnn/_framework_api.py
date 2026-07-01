# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Helpers for operation packages with optional Torch and JAX APIs."""

from __future__ import annotations

import importlib.util
from importlib import import_module
from typing import Any, Callable, MutableMapping, Sequence, Tuple


def make_framework_api(
    module_globals: MutableMapping[str, Any],
    *,
    torch_exports: Sequence[str],
    jax_exports: Sequence[str],
) -> Tuple[list[str], Callable[[str], Any]]:
    """Create ``__all__`` and ``__getattr__`` for a dual-framework package.

    ``api.py`` is the Torch API and ``jax.py`` is the JAX API. The unqualified
    package surface prefers Torch when it is installed, falls back to JAX in a
    JAX-only installation, and is empty when neither framework is installed.
    Explicit ``api`` and ``jax`` attributes remain independently lazy.
    """

    package_name = module_globals["__name__"]
    if importlib.util.find_spec("torch") is not None:
        default_module = ".api"
        selected_exports = tuple(torch_exports)
    elif importlib.util.find_spec("jax") is not None:
        default_module = ".jax"
        selected_exports = tuple(jax_exports)
    else:
        default_module = None
        selected_exports = ()

    def get_attribute(name: str) -> Any:
        if name in {"api", "jax"}:
            value = import_module(f".{name}", package_name)
        elif default_module is not None and name in selected_exports:
            value = getattr(import_module(default_module, package_name), name)
        else:
            raise AttributeError(f"module {package_name!r} has no attribute {name!r}")

        module_globals[name] = value
        return value

    return list(selected_exports), get_attribute


__all__ = ["make_framework_api"]
