# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental PyTorch custom operators built on cuDNN.

Names are resolved lazily so importing this package does not require PyTorch;
accessing one without PyTorch installed raises
:class:`cudnn.TorchNotAvailableError`.
"""

from typing import Any

from cudnn._deps import torch_dep

__all__ = [
    "scaled_dot_product_attention",
    "moe_grouped_matmul",
]

# Bind from the implementation modules rather than the same-named facade
# submodules: importing ``cudnn.experimental.ops.moe_grouped_matmul`` would set
# it as an attribute of this package and shadow the function of the same name.
_SOURCE = {
    "scaled_dot_product_attention": ("._sdpa_torch", "cudnn.experimental.ops.sdpa"),
    "moe_grouped_matmul": ("._moe_grouped_matmul_torch", "cudnn.experimental.ops.moe_grouped_matmul"),
}


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)

    module_name, feature = _SOURCE[name]
    torch_dep.require(feature)

    import importlib

    module = importlib.import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
