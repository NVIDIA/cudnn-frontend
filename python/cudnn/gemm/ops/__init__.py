# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend-independent GEMM operation contracts (torch custom-op wrappers)."""

from typing import Any

# The op modules import torch; keep that dependency lazy so ``import
# cudnn.gemm.ops`` stays frontend-only and torch is imported only when a
# specific operation is first accessed. Mirrors cudnn/gemm/__init__.py.
_LAZY_EXPORTS = {
    "moe_grouped_matmul": ("cudnn.gemm.ops.moe_grouped_matmul", "moe_grouped_matmul"),
    "swiglu_mlp": ("cudnn.gemm.ops.swiglu_mlp", "swiglu_mlp"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    import importlib

    value = getattr(importlib.import_module(module_name), attr_name)
    globals()[name] = value
    return value


__all__ = list(_LAZY_EXPORTS)
