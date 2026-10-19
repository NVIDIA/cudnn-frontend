# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""sm_100 norm forward: torch-tensor API + per-flavor CUTLASS-primitive kernels.

    from cudnn.norm.fprop import norm_fprop
    y, mean, rstd = norm_fprop(NormVariant.LAYER_NORM, x, gamma, beta, normalized_shape=[D])

Kernels live under :mod:`cudnn.norm.fprop.kernels`; the cuDNN graph engine that
lowers onto them lives in :mod:`cudnn.norm.fprop.engines`.
"""

from .._lazy import lazy_exports

# Imported lazily: the graph engine lives in this package, and importing it used
# to drag ``api`` -- and every kernel, and the CuTe DSL -- in before the engine
# had a chance to decline. See cudnn.norm._lazy.
_LAZY_EXPORTS = {"norm_fprop": ("cudnn.norm.fprop.api", "norm_fprop")}

__getattr__ = lazy_exports(__name__, _LAZY_EXPORTS, globals())

__all__ = ["norm_fprop"]
