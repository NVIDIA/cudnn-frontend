# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Warp-level FP32 amax reductions that also assemble on SM110 (Jetson Thor).

``redux.sync.max.f32`` (``cute.arch.warp_redux_sync`` with ``kind="fmax"``) is
not available on ``sm_110a`` even though the block-scaled ``tcgen05`` MMA is, so
a kernel that reaches it fails in ptxas (#1686).  Every FP32 amax reduction in
the CuTeDSL GEMM kernels goes through :func:`warp_amax_with_nan`, which keeps
the single-instruction path on other targets and unrolls a butterfly shuffle on
SM110.  Both forms propagate NaN and reduce across the full warp.
"""

import cutlass
import cutlass.cute as cute


def target_needs_shuffle_amax() -> bool:
    """True when the current compile target cannot assemble ``redux.f32``."""
    from cutlass.cutlass_dsl import CuTeDSL

    arch = CuTeDSL._get_dsl().get_arch_enum()
    return (int(arch.major), int(arch.minor)) == (11, 0)


@cute.jit
def warp_amax_with_nan(value):
    """Full-warp FP32 max with NaN propagation; ``redux.f32`` or an SM110 shuffle."""
    if cutlass.const_expr(target_needs_shuffle_amax()):
        for shift in cutlass.range_constexpr(5):
            value = cute.arch.fmax(value, cute.arch.shuffle_sync_bfly(value, offset=1 << shift), nan=True)
    else:
        value = cute.arch.warp_redux_sync(value=value, kind="fmax", mask_and_clamp=0xFFFFFFFF, nan=True)
    return value
