# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Torch API exports for RMSNorm + RHT + amax.

The exports are lazy so importing the framework-neutral CuTe kernel from the
JAX adapter does not import Torch.
"""

from importlib import import_module
from typing import Any

__all__ = [
    "RmsNormRhtAmaxSm100",
    "best_num_threads",
    "pick_rows_per_cta",
    "rmsnorm_rht_amax_wrapper_sm100",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(".api", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
