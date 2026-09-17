# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import sys
from typing import Any

# Implementations live with their owning operation families. Expose them here
# lazily so importing this experimental compatibility package does not pull in
# PyTorch until a specific operation is requested.
_LAZY_ALIASES = {
    "moe_grouped_matmul": "cudnn.gemm.ops.moe_grouped_matmul",
    "swiglu_mlp": "cudnn.gemm.ops.swiglu_mlp",
    "rms_norm": "cudnn.ops.norm.rmsnorm",
    "layer_norm": "cudnn.ops.norm.layernorm",
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
    "moe_grouped_matmul",
    "swiglu_mlp",
    "rms_norm",
    "layer_norm",
]
