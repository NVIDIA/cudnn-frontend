# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scalar pointwise DAG lowering shared by vector and scalar norm stores."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm


@cutlass.dsl_user_op
def _rounded_binary(lhs, rhs, mode, *, loc=None, ip=None):
    """Keep graph-edge FP32 rounding opaque to cross-node FMA contraction."""
    return cutlass.Float32(
        llvm.inline_asm(
            cutlass.Float32.mlir_type,
            [lhs.ir_value(loc=loc, ip=ip), rhs.ir_value(loc=loc, ip=ip)],
            f"{mode}.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def apply_epilogue(norm, aux, linear, spec: cutlass.Constexpr):
    # spec contains only prepared integer layouts and dtype/op semantics.
    norm_type, layouts, ops = spec
    values = (norm.to(norm_type),)
    for a in cutlass.range_constexpr(len(layouts)):
        offset = cutlass.Int64(0)
        for axis in cutlass.range_constexpr(len(layouts[a])):
            pitch, extent, stride = layouts[a][axis]
            offset = offset + (linear // pitch % extent) * stride
        values = values + (aux[a][offset],)
    for i in cutlass.range_constexpr(len(ops)):
        mode, lhs, rhs, compute_type, output_type = ops[i]
        left = values[lhs].to(compute_type)
        right = values[rhs].to(compute_type)
        value = _rounded_binary(left, right, mode)
        # Round at every graph edge; no reassociation of add/mul across casts.
        values = values + (value.to(output_type),)
    return values[-1]


@cute.jit
def forward(
    x,
    y,
    gamma,
    beta,
    mean,
    rstd,
    aux,
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
    spec: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    from cudnn.norm.fprop.kernels.layernorm_sm100 import _ln_fwd_kernel

    _ln_fwd_kernel(x, y, gamma, beta, mean, rstd, eps, d, v, bt, eb, stage, vec, has_mean, has_beta, epilogue=spec, epilogue_inputs=aux).launch(
        stream=stream, grid=(r, 1, 1), block=(bt, 1, 1)
    )
