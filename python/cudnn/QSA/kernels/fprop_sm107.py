# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Rubin QSA forward with native SM107 compilation and FP32 saved output."""

import cutlass
import cutlass.cute as cute
from cutlass.experimental.cuda import tensor_map as tmap
from cutlass.experimental import primitives as nvvm

from .fprop_sm100 import _fprop_body, SOFTMAX_WARPS, GATHER_WARPS


@cute.kernel
def frost_qsa_fprop_sm107(
    q_desc: cutlass.GridConstant[tmap.TensorMap],
    k_desc: cutlass.GridConstant[tmap.TensorMap],
    v_desc: cutlass.GridConstant[tmap.TensorMap],
    k_contiguous_desc: cutlass.GridConstant[tmap.TensorMap],
    v_contiguous_desc: cutlass.GridConstant[tmap.TensorMap],
    out: cute.Tensor,
    out_fp32: cute.Tensor,
    lse: cute.Tensor,
    ids: cute.Tensor,
    counts: cute.Tensor,
    lo: cute.Tensor,
    hi: cute.Tensor,
    first: cute.Tensor,
    last: cute.Tensor,
    mode,
    scale: cutlass.Float32,
    group_size: cutlass.Constexpr[int],
):
    if cutlass.const_expr(mode is None):
        _fprop_body(q_desc, k_desc, v_desc, k_contiguous_desc, v_contiguous_desc, out, out_fp32, lse, ids, counts, lo, hi, first, last, scale, group_size)
    else:
        first_query = cute.arch.block_idx()[0] * 8
        if first_query < out.shape[0]:
            if mode[cutlass.Int64(first_query // 128)] != 0:
                for step in cutlass.range(8, unroll=1):
                    query = first_query + step
                    if query < out.shape[0]:
                        _fprop_body(
                            q_desc,
                            k_desc,
                            v_desc,
                            k_contiguous_desc,
                            v_contiguous_desc,
                            out,
                            out_fp32,
                            lse,
                            ids,
                            counts,
                            lo,
                            hi,
                            first,
                            last,
                            scale,
                            group_size,
                            query,
                        )
                        nvvm.barrier_cta_sync()
                if cute.arch.warp_idx() == SOFTMAX_WARPS + GATHER_WARPS:
                    nvvm.tcgen05_relinquish_alloc_permit(group=nvvm.CTAGroup.CTA_1)


frost_qsa_fprop_sm107.set_name_prefix("cudnn", remove_cutlass_symbol=True)


def compile_kernel(dtype, query_heads: int, kv_heads: int, arch: int, gated=False):
    """Compile the native Rubin entry point."""
    from .fprop_sm100 import compile_kernel as compile_forward

    if arch != 107:
        raise ValueError(f"Rubin QSA forward requires SM107, got SM{arch}")
    return compile_forward(dtype, query_heads, kv_heads, arch, gated)


def launch(compiled, q, k, v, out, out_fp32, lse, block_ids, block_counts, lo, hi, block_first, block_last, scale, stream, mode=None):
    """Launch with caller-owned outputs and compact metadata."""
    compiled(q, k, v, out, out_fp32, lse, block_ids, block_counts, lo, hi, block_first, block_last, mode, cutlass.Float32(scale), stream)
