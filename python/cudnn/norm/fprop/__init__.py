# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""sm_100 norm forward: torch-tensor API + per-flavor CUTLASS-primitive kernels.

    from cudnn.norm.fprop import norm_fprop
    y, mean, rstd = norm_fprop(NormVariant.LAYER_NORM, x, gamma, beta, normalized_shape=[D])

Kernels live under :mod:`cudnn.norm.fprop.kernels`; the cuDNN graph engine that
lowers onto them lives in :mod:`cudnn.norm.fprop.engines`.
"""

from .api import norm_fprop

__all__ = ["norm_fprop"]
