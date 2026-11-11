# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""In-house `cutlass.block` compat shim.

The public nvidia-cutlass-dsl wheel (4.7.x) does not yet expose the
``cutlass.block`` namespace that the vendored filtered-top-k kernels import for
their block-level TMA loads. This is a small clean-room reimplementation of the
ONE entry those kernels use -- ``block_copy`` -- built only on public CuTe-DSL
APIs (``cpasync.tma_partition`` + ``cute.copy``), so no proprietary/EULA code is
vendored. The ``__init__`` shim aliases this module to ``cutlass.block`` before
the kernels import.

Scope: only the TMA global->smem (single-CTA, non-multicast) copy is
implemented, which is the only path the top-k kernels exercise (all call sites
are ``block_copy(tma_atom, group_modes(gmem, 0, 2), group_modes(smem, 0, 2),
tma_bar_ptr=...)``). A non-TMA copy op raises NotImplementedError rather than
silently doing the wrong thing.
"""

from typing import Any, Optional

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.nvgpu.cpasync.copy import TmaCopyOp


def block_copy(
    tiled_copy,
    src: cute.Tensor,
    dst: cute.Tensor,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
    **kwargs: Any,
) -> None:
    """Block-level copy of a tile whose first mode is the TMA tile.

    ``src``/``dst`` arrive already grouped as ``(TMATile, Rest...)`` (the caller
    does ``cute.group_modes(t, 0, 2)``). For a TMA global->smem load this
    partitions both operands for a single CTA (no cluster multicast) and issues
    the copy; ``tma_bar_ptr`` (the completion mbarrier) rides through ``kwargs``
    to ``cute.copy``.
    """
    if cutlass.const_expr(not isinstance(tiled_copy.op, TmaCopyOp)):
        raise NotImplementedError(f"block_copy compat shim supports TMA copies only; got " f"{type(tiled_copy.op).__name__}")
    # tma_partition returns (smem_partitioned, gmem_partitioned); a load copies
    # gmem -> smem.
    s_part, g_part = cpasync.tma_partition(tiled_copy, 0, cute.make_layout(1), dst, src)
    cute.copy(tiled_copy, g_part, s_part, loc=loc, ip=ip, **kwargs)


__all__ = ["block_copy"]
