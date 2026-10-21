# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Rubin QSA forward with native SM107 compilation and FP32 saved output."""

import cutlass
import cutlass.cute as cute
from cutlass.experimental.cuda import tensor_map as tmap

from .fprop_sm100 import _fprop_body


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
    scale: cutlass.Float32,
    group_size: cutlass.Constexpr[int],
):
    _fprop_body(q_desc, k_desc, v_desc, k_contiguous_desc, v_contiguous_desc, out, out_fp32, lse, ids, counts, lo, hi, first, last, scale, group_size)


frost_qsa_fprop_sm107.set_name_prefix("cudnn", remove_cutlass_symbol=True)


def compile_kernel(dtype, query_heads: int, kv_heads: int, arch: int):
    """Compile the native Rubin entry point."""
    from .fprop_sm100 import compile_kernel as compile_forward

    if arch != 107:
        raise ValueError(f"Rubin QSA forward requires SM107, got SM{arch}")
    return compile_forward(dtype, query_heads, kv_heads, arch)


def launch(compiled, q, k, v, out, out_fp32, lse, block_ids, block_counts, lo, hi, block_first, block_last, scale, stream):
    """Launch with caller-owned outputs and compact metadata."""
    compiled(q, k, v, out, out_fp32, lse, block_ids, block_counts, lo, hi, block_first, block_last, cutlass.Float32(scale), stream)
