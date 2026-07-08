# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Lazy exports for frontend-only operation packages."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Callable, MutableMapping, Sequence, Tuple


def make_operation_api(
    module_globals: MutableMapping[str, Any],
    *,
    exports: Sequence[str],
) -> Tuple[list[str], Callable[[str], Any]]:
    """Create ``__all__`` and ``__getattr__`` for an operation package.

    Unqualified symbols always come from the sibling ``api.py`` module. The
    sibling ``jax.py`` module is available only through the explicit ``jax``
    attribute. Dependency availability does not change how a name is routed.
    """

    package_name = module_globals["__name__"]
    exported_names = tuple(exports)

    def get_attribute(name: str) -> Any:
        if name in {"api", "jax"}:
            value = import_module(f".{name}", package_name)
        elif name in exported_names:
            value = getattr(import_module(".api", package_name), name)
        else:
            raise AttributeError(f"module {package_name!r} has no attribute {name!r}")

        module_globals[name] = value
        return value

    return list(exported_names), get_attribute


__all__ = ["make_operation_api"]
