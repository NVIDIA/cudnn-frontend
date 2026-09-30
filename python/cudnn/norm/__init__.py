# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cudnn.norm: normalization ops backed by sm_100 CUTLASS-primitive kernels.

Five variants (LayerNorm, RMSNorm, GroupNorm, InstanceNorm, BatchNorm), both
fprop and bprop, bf16 / fp16 / fp32 I/O. Kernels are JIT-compiled with the
CUTLASS CuTe DSL using CUTLASS primitives (``cutlass.primitives`` / ``nvvm``
cp.async staging, ``SmemAllocator`` reductions) and the shared
``cudnn.frost.tile_dsl`` library — mirroring ``cudnn.sdpa`` / ``cudnn.gemm.frost``.

Direct (non-graph) API::

    from cudnn.norm import NormVariant, norm_fprop, norm_bprop

    y, mean, rstd = norm_fprop(NormVariant.LAYER_NORM, x, gamma, beta, normalized_shape=[D])
    dx, dgamma, dbeta = norm_bprop(NormVariant.LAYER_NORM, dy, x, gamma, mean, rstd,
                                   normalized_shape=[D])

The cuDNN graph engines that lower ``NORM_FWD`` / ``NORM_BWD`` nodes onto these
kernels live in :mod:`cudnn.norm.fprop.engines` / :mod:`cudnn.norm.bprop.engines`
and register with :mod:`cudnn.frost`.
"""

from typing import Any

_LAZY_EXPORTS = {
    "NormVariant": ("cudnn.norm.config_sm100", "NormVariant"),
    "norm_fprop": ("cudnn.norm.fprop.api", "norm_fprop"),
    "norm_bprop": ("cudnn.norm.bprop.api", "norm_bprop"),
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
