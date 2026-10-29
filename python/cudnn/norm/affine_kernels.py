# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared host entries for the shared row kernels' affine specializations."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute

from .fprop.kernels.layernorm_sm100 import _ln_fwd_kernel
from .bprop.kernels.layernorm_sm100 import _ln_bwd_kernel


@cute.jit
def forward(
    x,
    gamma,
    beta,
    y,
    mean,
    rstd,
    eps: cutlass.Float32,
    r: cutlass.Constexpr,
    d: cutlass.Constexpr,
    v: cutlass.Constexpr,
    bt: cutlass.Constexpr,
    eb: cutlass.Constexpr,
    stage: cutlass.Constexpr,
    vec: cutlass.Constexpr,
    has_mean: cutlass.Constexpr,
    has_beta: cutlass.Constexpr,
    affine: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    _ln_fwd_kernel(x, y, gamma, beta, mean, rstd, eps, d, v, bt, eb, stage, vec, has_mean, has_beta, affine).launch(
        stream=stream, grid=(r, 1, 1), block=(bt, 1, 1)
    )


@cute.jit
def backward(
    dy,
    x,
    gamma,
    mean,
    rstd,
    dx,
    dgamma,
    dbeta,
    r: cutlass.Constexpr,
    d: cutlass.Constexpr,
    v: cutlass.Constexpr,
    bt: cutlass.Constexpr,
    eb: cutlass.Constexpr,
    stage: cutlass.Constexpr,
    vec: cutlass.Constexpr,
    has_mean: cutlass.Constexpr,
    has_beta: cutlass.Constexpr,
    affine: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    _ln_bwd_kernel(dy, x, gamma, mean, rstd, dx, dgamma, dbeta, d, v, bt, eb, stage, vec, has_mean, has_beta, affine).launch(
        stream=stream, grid=(r, 1, 1), block=(bt, 1, 1)
    )
