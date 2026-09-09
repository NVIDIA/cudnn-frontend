# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend-independent GEMM operation contracts (torch custom-op wrappers)."""

from typing import Any

# The op modules import torch; keep that dependency lazy so ``import
# cudnn.gemm.ops`` stays frontend-only and torch is imported only when a
# specific operation is first accessed. Mirrors cudnn/gemm/__init__.py.
_LAZY_EXPORTS = {
    "moe_grouped_matmul": ("cudnn.gemm.ops.moe_grouped_matmul", "moe_grouped_matmul"),
    # Both semantic entry points share the existing implementation module.
    "situ_mlp": ("cudnn.gemm.ops.swiglu_mlp", "situ_mlp"),
    "swiglu_mlp": ("cudnn.gemm.ops.swiglu_mlp", "swiglu_mlp"),
    "swiglu_moe": ("cudnn.gemm.ops.swiglu_moe", "swiglu_moe"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    import importlib

    module = importlib.import_module(module_name)
    # Multiple semantic entry points can share one implementation module. Import
    # machinery writes a same-named module onto this package while loading it;
    # install every export from this module as its callable so either
    # order in ``from cudnn.gemm.ops import situ_mlp, swiglu_mlp`` is stable.
    for export_name, (export_module_name, export_attr_name) in _LAZY_EXPORTS.items():
        if export_module_name == module_name:
            globals()[export_name] = getattr(module, export_attr_name)
    return globals()[name]


__all__ = list(_LAZY_EXPORTS)
