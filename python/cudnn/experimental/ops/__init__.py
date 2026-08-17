# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import sys
from typing import Any

from .sdpa import scaled_dot_product_attention

# moe_grouped_matmul / swiglu_mlp live with the rest of the GEMM family in
# cudnn.gemm.ops (their modules import torch). Expose them here lazily so that
# importing this package does not eagerly pull in those kernel modules; the
# submodule aliases (``cudnn.experimental.ops.<name>``) are registered on first
# access, so ``from cudnn.experimental.ops import <name>`` keeps resolving.
_LAZY_ALIASES = {
    "moe_grouped_matmul": "cudnn.gemm.ops.moe_grouped_matmul",
    "swiglu_mlp": "cudnn.gemm.ops.swiglu_mlp",
}


def __getattr__(name: str) -> Any:
    try:
        target = _LAZY_ALIASES[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    module = importlib.import_module(target)
    sys.modules[f"{__name__}.{name}"] = module
    value = getattr(module, name)
    globals()[name] = value
    return value


__all__ = [
    "scaled_dot_product_attention",
    "moe_grouped_matmul",
    "swiglu_mlp",
]
