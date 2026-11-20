# Copyright (c) 2025, Tri Dao.
# SPDX-License-Identifier: Apache-2.0
"""Minimal SwiGLU activation helpers adapted from quack-kernels 0.5.0."""

import math
from typing import Tuple

import cutlass.cute as cute
from cutlass import Float32, const_expr
from cutlass.cutlass_dsl import dsl_user_op

F32_or_F32x2 = Float32 | Tuple[Float32, Float32]


@dsl_user_op
def sigmoid(x: F32_or_F32x2, *, loc=None, ip=None) -> F32_or_F32x2:
    if const_expr(not isinstance(x, tuple)):
        return cute.arch.rcp_approx(1.0 + cute.math.exp(-x, fastmath=True), loc=loc, ip=ip)
    else:
        log2_e = math.log2(math.e)
        neg_x = cute.arch.mul_packed_f32x2(x, (-log2_e, -log2_e))
        exp_neg_x = (
            cute.math.exp2(neg_x[0], fastmath=True),
            cute.math.exp2(neg_x[1], fastmath=True),
        )
        denom = cute.arch.add_packed_f32x2(exp_neg_x, (1.0, 1.0))
        return cute.arch.rcp_approx(denom[0]), cute.arch.rcp_approx(denom[1])


@dsl_user_op
def silu(x: F32_or_F32x2, *, loc=None, ip=None) -> F32_or_F32x2:
    """
    silu(x) = x * sigmoid(x) = x * rcp(1 + exp(-x)).
    """
    if const_expr(not isinstance(x, tuple)):
        return x * sigmoid(x)
    else:
        return cute.arch.mul_packed_f32x2(x, sigmoid(x))


@dsl_user_op
def swiglu(x: F32_or_F32x2, y: F32_or_F32x2, *, loc=None, ip=None) -> F32_or_F32x2:
    if const_expr(not isinstance(x, tuple)):
        return silu(x) * y
    else:
        return cute.arch.mul_packed_f32x2(silu(x), y)


@dsl_user_op
def dswiglu(
    x: F32_or_F32x2,
    y: F32_or_F32x2,
    dout: F32_or_F32x2,
    *,
    loc=None,
    ip=None,
) -> Tuple[F32_or_F32x2, F32_or_F32x2, F32_or_F32x2]:
    """
    SwiGLU backward pass: computes gradients w.r.t. x (gate) and y (up projection)
    Given: swiglu_out = silu(x) * y, and dout = grad w.r.t. swiglu_out
    Returns: (dx, dy, swiglu_out) where dx = dout * y * d_silu(x), dy = dout * silu(x)

    d_silu(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))

    This has been optimized to use fewer instructions (i.e. we expand things out
    to use FFMA instead of FADD and FMUL).
    """
    if const_expr(not isinstance(x, tuple)):
        sigmoid_x = sigmoid(x)
        silu_x = x * sigmoid_x  # FMUL
        silu_x_dout = silu_x * dout  # FMUL
        #   d_silu(x) * dout
        # = sigmoid_x * (1 + x * (1 - sigmoid_x)) * dout
        # = (sigmoid_x + sigmoid_x * x * (1 - sigmoid_x)) * dout
        # = (sigmoid_x + silu_x * (1 - sigmoid_x)) * dout
        # = (sigmoid_x + silu_x - silu_x * sigmoid_x) * dout
        # = (sigmoid_x - silu_x * sigmoid_x) * dout + silu_x * dout
        d_silu_x_dout = (sigmoid_x - silu_x * sigmoid_x) * dout + silu_x_dout  # FFMA, FFMA
        dx = d_silu_x_dout * y  # FMUL
        dy = silu_x_dout
        swiglu_out = silu_x * y  # FMUL
        return dx, dy, swiglu_out
    else:
        # Compute sigmoid(x) and silu(x)
        sigmoid_x = sigmoid(x)
        silu_x = cute.arch.mul_packed_f32x2(x, sigmoid_x)
        silu_x_dout = cute.arch.mul_packed_f32x2(silu_x, dout)
        # d_silu(x) * dout = (sigmoid_x - silu_x * sigmoid_x) * dout + silu_x * dout
        sigmoid_x_minus_silu_x_sigmoid_x = cute.arch.fma_packed_f32x2(sigmoid_x, (-silu_x[0], -silu_x[1]), sigmoid_x)
        d_silu_x_dout = cute.arch.fma_packed_f32x2(sigmoid_x_minus_silu_x_sigmoid_x, dout, silu_x_dout)
        dx = cute.arch.mul_packed_f32x2(d_silu_x_dout, y)
        dy = silu_x_dout
        swiglu_out = cute.arch.mul_packed_f32x2(silu_x, y)
        return dx, dy, swiglu_out
