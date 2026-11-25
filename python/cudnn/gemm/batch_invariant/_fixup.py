# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from flashinfer/gemm/batch_invariant/fixup.py.

"""Fold workspace partials in increasing chunk order and apply the graph's epilogue."""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32
from cutlass.cute.math import RoundingMode

_THREADS = 256
_VEC = 4


@cute.jit
def reduce_partials(chunks: cutlass.Constexpr, partials, out, scale_a, scale_b, bias, stream):
    m, n, _ = out.shape
    rows = cutlass.min(m, Int32(65535))
    _fold(chunks, partials, out, scale_a, scale_b, bias, rows).launch(grid=((n - 1) // (_THREADS * _VEC) + 1, rows, 1), block=(_THREADS, 1, 1), stream=stream)


@cute.kernel
def _fold(chunks: cutlass.Constexpr, partials, out, scale_a, scale_b, bias, rows: Int32):
    bx, by, _ = cute.arch.block_idx()
    col0 = (bx * _THREADS + cute.arch.thread_idx()[0]) * _VEC
    m, n, _ = out.shape
    scale = None
    if cutlass.const_expr(scale_a is not None):
        scale = scale_a[0]
        if cutlass.const_expr(scale_b is not None):
            scale = cute.math.mul(scale, scale_b[0], rounding=RoundingMode.NEAREST_EVEN)
    # Every supported output row contains a multiple of eight elements.
    if col0 < n:
        for i in cutlass.range((m - 1 - by) // rows + 1, unroll=1):
            row = by + i * rows
            for j in cutlass.range_constexpr(_VEC):
                col = col0 + j
                acc = Float32(0)
                for chunk in cutlass.range_constexpr(chunks):
                    acc = cute.math.add(acc, partials[row, col, chunk], rounding=RoundingMode.NEAREST_EVEN)
                if cutlass.const_expr(scale is not None):
                    acc = cute.math.mul(acc, scale, rounding=RoundingMode.NEAREST_EVEN)
                if cutlass.const_expr(bias is not None):
                    acc = cute.math.add(acc, bias[col].to(Float32), rounding=RoundingMode.NEAREST_EVEN)
                out[row, col, 0] = acc.to(out.element_type)


_fold.set_name_prefix("cudnn", remove_cutlass_symbol=True)
