# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Reusable lazy exports for framework-specific operation packages."""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping, Sequence
from importlib import import_module
from typing import Any


def make_operation_api(
    module_globals: MutableMapping[str, Any],
    *,
    exports: Mapping[str, Sequence[str]],
    submodules: Sequence[str] = (),
) -> tuple[list[str], Callable[[str], Any], Callable[[], list[str]]]:
    """Create ``__all__``, ``__getattr__``, and ``__dir__`` for an operation.

    ``exports`` maps sibling module names to the public symbols provided by
    each module. ``submodules`` lists sibling modules that are themselves
    available as lazy attributes. Resolved values are cached in the package's
    globals, following Python's normal module import behavior.
    """

    package_name = module_globals["__name__"]
    export_routes: dict[str, str] = {}
    for module_name, names in exports.items():
        for name in names:
            if name in export_routes:
                raise ValueError(f"Operation export {name!r} is provided by more than one module")
            export_routes[name] = module_name

    lazy_submodules = frozenset(submodules)

    def get_attribute(name: str) -> Any:
        if name in lazy_submodules:
            value = import_module(f".{name}", package_name)
        elif name in export_routes:
            value = getattr(import_module(f".{export_routes[name]}", package_name), name)
        else:
            raise AttributeError(f"module {package_name!r} has no attribute {name!r}")

        module_globals[name] = value
        return value

    def list_attributes() -> list[str]:
        return sorted((*module_globals, *export_routes, *lazy_submodules))

    return sorted(export_routes), get_attribute, list_attributes


__all__ = ["make_operation_api"]
