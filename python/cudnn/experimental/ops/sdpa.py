# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import-safe facade for the PyTorch scaled-dot-product-attention custom operators.

Importing this module never requires PyTorch. The public callables are resolved
on first attribute access, which imports the sibling implementation module
(:mod:`cudnn.experimental.ops._sdpa_torch`) and thereby registers the operator schemas, fake
implementations and autograd formulas exactly once.

Without PyTorch, touching any public name raises
:class:`cudnn.TorchNotAvailableError`.
"""

from typing import Any

from cudnn._deps import torch_dep

_FEATURE = "cudnn.experimental.ops.sdpa"

__all__ = ["scaled_dot_product_attention", "sdpa_fwd_d256", "sdpa_bwd_d256"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)

    # Raise the precise missing-framework error before the implementation module
    # (whose body is guarded on the same probe) would silently define nothing.
    torch_dep.require(_FEATURE)

    from . import _sdpa_torch as _impl

    value = getattr(_impl, name)
    globals()[name] = value  # cache: subsequent lookups skip __getattr__
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
