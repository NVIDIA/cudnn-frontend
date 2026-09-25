# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BF16 tail64 rotation with explicit FP32 multiply/FMA rounding.

The arithmetic follows DeepSeek's complex RoPE convention. The vectorized
full-copy/tail implementation is authored for cuDNN Frontend.
"""

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm


@cutlass.dsl_user_op
def _source_real(a, c, b, s, *, loc=None, ip=None):
    return cutlass.Float32(
        llvm.inline_asm(
            cutlass.Float32.mlir_type,
            [v.ir_value(loc=loc, ip=ip) for v in (a, c, b, s)],
            "{ .reg .f32 p; mul.rn.f32 p,$3,$4; neg.f32 p,p; fma.rn.f32 $0,$1,$2,p; }",
            "=f,f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cutlass.dsl_user_op
def _source_imag(a, c, b, s, *, loc=None, ip=None):
    return cutlass.Float32(
        llvm.inline_asm(
            cutlass.Float32.mlir_type,
            [v.ir_value(loc=loc, ip=ip) for v in (a, c, b, s)],
            "{ .reg .f32 p; mul.rn.f32 p,$3,$4; fma.rn.f32 $0,$1,$2,p; }",
            "=f,f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.kernel
def _frost_tail_rope(
    x: cute.Tensor,
    cosine: cute.Tensor,
    sine: cute.Tensor,
    out: cute.Tensor,
    elements: cutlass.Int64,
    h: cutlass.Constexpr,
    d: cutlass.Constexpr,
    transpose: cutlass.Constexpr,
    vector: cutlass.Constexpr,
):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    at = cute.assume((cutlass.Int64(bid) * 128 + tid) * vector, divby=vector)
    if at < elements:
        values = cute.make_rmem_tensor(vector, cutlass.BFloat16)
        cute.autovec_copy(cute.make_tensor(x.iterator + at, cute.make_layout(vector)), values)
        dim = at % d
        if dim >= d - 64:
            table_at = cute.assume(at // (h * d) * 32 + (dim - (d - 64)) // 2, divby=vector // 2)
            cosines = cute.make_rmem_tensor(vector // 2, cutlass.Float32)
            sines = cute.make_rmem_tensor(vector // 2, cutlass.Float32)
            cute.autovec_copy(cute.make_tensor(cosine.iterator + table_at, cute.make_layout(vector // 2)), cosines)
            cute.autovec_copy(cute.make_tensor(sine.iterator + table_at, cute.make_layout(vector // 2)), sines)
            for pair in cutlass.range_constexpr(vector // 2):
                a, b = values[2 * pair].to(cutlass.Float32), values[2 * pair + 1].to(cutlass.Float32)
                c, s = cosines[pair], sines[pair]
                if cutlass.const_expr(transpose):
                    s = -s
                values[2 * pair] = _source_real(a, c, b, s).to(cutlass.BFloat16)
                values[2 * pair + 1] = _source_imag(b, c, a, s).to(cutlass.BFloat16)
        cute.autovec_copy(values, cute.make_tensor(out.iterator + at, cute.make_layout(vector)))


_frost_tail_rope.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def launch(
    x: cute.Tensor,
    cosine: cute.Tensor,
    sine: cute.Tensor,
    out: cute.Tensor,
    stream,
    elements: cutlass.Int64,
    h: cutlass.Constexpr,
    d: cutlass.Constexpr,
    transpose: cutlass.Constexpr,
    vector: cutlass.Constexpr,
):
    # CuTe shape-leaf unpacking can narrow i64 extents to i32. Keep the
    # element count scalar through grid division and the device bounds check.
    _frost_tail_rope(x, cosine, sine, out, elements, h, d, transpose, vector).launch(
        grid=((elements + 128 * vector - 1) // (128 * vector), 1, 1), block=(128, 1, 1), stream=stream
    )
