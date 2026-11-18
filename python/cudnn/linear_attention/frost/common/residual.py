# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The delta-rule residual operand ``Y = beta * (W .* V - projection)`` formed in FP32 from the FP32 MMA projection.

The projection ``K decay @ S`` is already FP32 in TMEM.  BF16 paths form the write-gated value, the residual and the beta
scaling in FP32 and round once into the packed Y operand; FP16 paths retain their packed arithmetic.  KDA scales by the
per-token ``beta`` (its ``W`` is one); GDN-2 scales by the per-value write gate ``W`` (its beta is folded into K decay).
Register-only: the caller still owns the projection-ready wait and the packed-Y publication.
"""

import cutlass
import cutlass.cute as cute

from cudnn.frost.tile_dsl.pointwise import f16x2_to_f32, fadd2, ffma2, fmul2, fp32_to_fp16, mul_f16x2, sub_f16x2


@cute.jit
def residual_pair(value_word, projection0=None, projection1=None, *, dtype: cutlass.Constexpr):
    """Return two FP32 residuals ``V - projection`` for the backward dBeta accumulation.

    ``value_word`` packs two values of compile-time ``dtype`` (BF16 or FP16).  Supply both FP32 projection components, or
    omit both for a zero-state projection.  BF16 subtracts without an intermediate cast; FP16 preserves the rounded
    projection and packed subtraction before widening the residual.
    """
    if cutlass.const_expr(projection0 is None):
        return f16x2_to_f32(value_word, dtype=dtype)
    if cutlass.const_expr(dtype == cutlass.BFloat16):
        v0, v1 = f16x2_to_f32(value_word, dtype=dtype)
        return fadd2(v0, v1, -projection0, -projection1)
    projection_word = fp32_to_fp16(projection0, projection1, dtype=dtype)
    return f16x2_to_f32(sub_f16x2(value_word, projection_word, dtype), dtype=dtype)


@cute.jit
def residual_operand(value_word, projection0=None, projection1=None, beta0=None, beta1=None, *, w_word=None, dtype: cutlass.Constexpr):
    """Pack ``Y = beta * (W .* V - projection)`` for two io-dtype elements.

    ``value_word`` (and ``w_word`` when given) pack two values of compile-time ``dtype``; the projection and beta
    components are FP32 registers.  Omit the projections for a zero state, the betas where beta is already folded into the
    operands (GDN-2, prep-fed KDA), ``w_word`` where there is no write gate (KDA).  BF16 rounds once after the FP32
    arithmetic; FP16 rounds the projection and beta to FP16 and keeps its packed multiply and subtraction.
    """
    if cutlass.const_expr(dtype == cutlass.BFloat16):
        v0, v1 = f16x2_to_f32(value_word, dtype=dtype)
        if cutlass.const_expr(w_word is not None):
            w0, w1 = f16x2_to_f32(w_word, dtype=dtype)
            if cutlass.const_expr(projection0 is None):
                r0, r1 = fmul2(w0, w1, v0, v1)
            else:
                r0, r1 = ffma2(w0, w1, v0, v1, -projection0, -projection1)
        elif cutlass.const_expr(projection0 is None):
            r0, r1 = v0, v1
        else:
            r0, r1 = fadd2(v0, v1, -projection0, -projection1)
        if cutlass.const_expr(beta0 is not None):
            r0, r1 = fmul2(r0, r1, beta0, beta1)
        return fp32_to_fp16(r0, r1, dtype=dtype)
    result = value_word
    if cutlass.const_expr(w_word is not None):
        result = mul_f16x2(w_word, result, dtype)
    if cutlass.const_expr(projection0 is not None):
        result = sub_f16x2(result, fp32_to_fp16(projection0, projection1, dtype=dtype), dtype)
    if cutlass.const_expr(beta0 is not None):
        result = mul_f16x2(fp32_to_fp16(beta0, beta1, dtype=dtype), result, dtype)
    return result
