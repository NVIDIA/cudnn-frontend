# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit FP32 arithmetic and aligned vector access for Engram."""

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm


@cutlass.dsl_user_op
def _copysign(magnitude, sign, *, loc=None, ip=None):
    return cutlass.Float32(
        llvm.inline_asm(
            cutlass.Float32.mlir_type,
            [cutlass.Float32(magnitude).ir_value(loc=loc, ip=ip), cutlass.Float32(sign).ir_value(loc=loc, ip=ip)],
            "{ .reg .b32 m,s; mov.b32 m,$1; mov.b32 s,$2; and.b32 m,m,0x7fffffff; and.b32 s,s,0x80000000; or.b32 m,m,s; mov.b32 $0,m; }",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cutlass.dsl_user_op
def _mul(a, b, *, loc=None, ip=None):
    return cutlass.Float32(
        llvm.inline_asm(
            cutlass.Float32.mlir_type,
            [cutlass.Float32(a).ir_value(loc=loc, ip=ip), cutlass.Float32(b).ir_value(loc=loc, ip=ip)],
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cutlass.dsl_user_op
def _add(a, b, *, loc=None, ip=None):
    return cutlass.Float32(
        llvm.inline_asm(
            cutlass.Float32.mlir_type,
            [cutlass.Float32(a).ir_value(loc=loc, ip=ip), cutlass.Float32(b).ir_value(loc=loc, ip=ip)],
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _source_sum(accumulator):
    value = _add(_add(_add(accumulator[0], accumulator[1]), accumulator[2]), accumulator[3])
    for step in cutlass.range_constexpr(5):
        value = _add(value, cute.arch.shuffle_sync_bfly(value, offset=16 >> step))
    return value


@cute.jit
def _load(tensor, offset, vec: cutlass.Constexpr):
    offset = cute.assume(offset, divby=vec)
    result = cute.make_rmem_tensor(vec, tensor.element_type)
    cute.autovec_copy(cute.make_tensor(tensor.iterator + offset, cute.make_layout(vec)), result)
    return result


@cute.jit
def _store(value, tensor, offset, vec: cutlass.Constexpr):
    offset = cute.assume(offset, divby=vec)
    cute.autovec_copy(value, cute.make_tensor(tensor.iterator + offset, cute.make_layout(vec)))
