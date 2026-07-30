# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PyTorch custom operators built on cuDNN.

Names are resolved lazily so importing this package does not require PyTorch;
accessing one without PyTorch installed raises
:class:`cudnn.TorchNotAvailableError`.
"""

from typing import Any

from cudnn._deps import torch_dep

_FEATURE = "cudnn.ops.causal_conv1d"

__all__ = ["causal_conv1d", "causal_conv1d_nwh", "b2b_causal_conv1d"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)

    torch_dep.require(_FEATURE)

    # Bind from the implementation module rather than the same-named facade
    # submodule: importing ``cudnn.ops.causal_conv1d`` would set it as an
    # attribute of this package and shadow the function of the same name.
    from . import _causal_conv1d_torch as _impl

    value = getattr(_impl, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
