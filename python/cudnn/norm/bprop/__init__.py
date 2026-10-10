# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""sm_100 norm backward: torch-tensor API + per-flavor CUTLASS-primitive kernels.

    from cudnn.norm.bprop import norm_bprop
    dx, dgamma, dbeta = norm_bprop(NormVariant.LAYER_NORM, dy, x, gamma, mean, rstd,
                                   normalized_shape=[D])

Kernels live under :mod:`cudnn.norm.bprop.kernels`; the cuDNN graph engine that
lowers onto them lives in :mod:`cudnn.norm.bprop.engines`.
"""

from .api import norm_bprop

__all__ = ["norm_bprop"]
