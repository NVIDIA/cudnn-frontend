# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import cutlass.cute as cute
from cutlass import Float32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op


@dsl_user_op
def atomic_add_fp32x4(a: Float32, b: Float32, c: Float32, d: Float32, ptr: cute.Pointer, *, loc=None, ip=None) -> None:
    """Reduce four FP32 values to a 16-byte-aligned global address on SM90+."""
    llvm.inline_asm(
        None,
        [ptr.toint(loc=loc, ip=ip).ir_value(), *[Float32(x).ir_value(loc=loc, ip=ip) for x in (a, b, c, d)]],
        "red.global.add.v4.f32 [$0], {$1, $2, $3, $4};",
        "l,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
