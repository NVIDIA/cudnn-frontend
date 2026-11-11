# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from FlashAttention's sliced WGMMA schedule; see THIRD_PARTY_LICENSES.txt.

from typing import Optional
import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Int32, const_expr
from cudnn.flex_attention._compat.sm90_utils import gemm


@cute.jit
def gemm_slice(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    zero_init: Boolean,
    A_idx: Optional[Int32] = None,
    B_idx: Optional[Int32] = None,
    wg_wait: cutlass.Constexpr[int] = -1,
    M_slice: cutlass.Constexpr[int] = 0,
):
    """Issue one M slice of a swapped-operand WGMMA, preserving full C layout."""
    a = tCrA if const_expr(A_idx is None) else tCrA[None, None, None, A_idx]
    b = tCrB if const_expr(B_idx is None) else tCrB[None, None, None, B_idx]
    mma_m = cute.size(acc, mode=[1])
    assert mma_m == 2
    c_slice = cute.logical_divide(acc, (None, mma_m))[None, (M_slice, None), None]
    b_slice = cute.logical_divide(b, (None, mma_m))[None, (M_slice, None), None]
    gemm(tiled_mma, c_slice, a, b_slice, zero_init=zero_init, wg_wait=wg_wait, swap_AB=True)
