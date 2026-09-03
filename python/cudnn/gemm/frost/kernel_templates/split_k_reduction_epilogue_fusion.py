# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Split-K reduction epilogue: kernel 2 of the two-kernel split-K scheme.

A regular importable module (like ``_tile_helpers``), not a rendered template:
any main-GEMM template that stores fp32 partials (``sm100_matmul`` today)
imports the kernel and passes its formerly-injected constants —
``split_k_slices``, ``splitk_reduce_elems``, the store dtype, ``USE_PDL`` —
as ``Constexpr`` arguments at the launch site.

Each thread owns one ``splitk_reduce_elems``-wide fp32 group of one output row
and sums it across the S partials in a fixed order — bitwise deterministic by
construction — as four independent add chains unrolled at trace time (the
support gate bounds S at 32), then applies the store cast and writes D.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.experimental.primitives as nvvm

SPLITK_REDUCE_THREADS = 256


@cute.kernel
def _splitk_reduce_kernel(
    m: cutlass.Int64,
    n: cutlass.Int64,
    batch: cutlass.Int64,
    out_stride_m: cutlass.Int64,
    out_stride_n: cutlass.Int64,
    out_stride_l: cutlass.Int64,
    mD: cute.Tensor,
    mSplitK_partials: cute.Tensor,
    split_k_slices: cutlass.Constexpr,
    splitk_reduce_elems: cutlass.Constexpr,
    cd_dtype: cutlass.Constexpr,
    use_pdl: cutlass.Constexpr,
) -> None:
    row = cutlass.Int64(cute.arch.block_idx()[0])
    col = (cutlass.Int64(cute.arch.block_idx()[1]) * SPLITK_REDUCE_THREADS + cute.arch.thread_idx()[0]) * splitk_reduce_elems
    batch_idx = cutlass.Int64(cute.arch.block_idx()[2])
    if cutlass.const_expr(use_pdl):
        nvvm.griddepcontrol("wait")
    if col < n:
        elems_per_partial = m * n
        partials_ptr = mSplitK_partials.iterator.raw_ptr()
        # partials layout: [batch*split][M][N]
        first_partial_offset = (batch_idx * split_k_slices * m + row) * n + col

        chain_0 = (partials_ptr + first_partial_offset).load(count=splitk_reduce_elems, alignment=splitk_reduce_elems * 4)
        chain_1 = cutlass.full_like(chain_0, 0.0)
        chain_2 = cutlass.full_like(chain_0, 0.0)
        chain_3 = cutlass.full_like(chain_0, 0.0)
        for s in cutlass.range_constexpr(1, split_k_slices):
            _p = (partials_ptr + first_partial_offset + s * elems_per_partial).load(count=splitk_reduce_elems, alignment=splitk_reduce_elems * 4)
            if cutlass.const_expr(s % 4 == 0):
                chain_0 = chain_0 + _p
            elif cutlass.const_expr(s % 4 == 1):
                chain_1 = chain_1 + _p
            elif cutlass.const_expr(s % 4 == 2):
                chain_2 = chain_2 + _p
            else:
                chain_3 = chain_3 + _p
        acc = (chain_0 + chain_1) + (chain_2 + chain_3)
        d_offset = batch_idx * out_stride_l + row * out_stride_m + col * out_stride_n
        (mD.iterator.raw_ptr() + d_offset).store(acc.to(cd_dtype), alignment=splitk_reduce_elems * cd_dtype.width // 8)


_splitk_reduce_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)
