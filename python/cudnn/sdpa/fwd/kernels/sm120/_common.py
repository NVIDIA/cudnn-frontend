# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""PTX helpers shared by the SM120 SDPA-forward kernels.

``sm120/prefill_f16.py``, ``sm120/prefill_fp8.py``, ``sm120/prefill_mxfp8.py``,
and the D=256 specialization ``sm120/prefill_d256_f16.py`` are self-contained
kernels that differ in head-tile geometry, MMA shape, and warp roles; the
lane-level primitives below are shared between them (``sfo_atom_offset`` by the
two block-scaled-O kernels, FP8 and MXFP8) and live here so a fix lands once.
"""

from typing import Type

import cutlass
import cutlass.cute as cute

from cutlass.experimental import primitives as prims

SCHED_L2_BUDGET_BYTES = 50 * 1024 * 1024


@cute.jit
def nvvm_threadquad_reduction_max(val: cutlass.Float32) -> cutlass.Float32:
    """Butterfly thread-quad (4 lanes) reduction max via shfl.sync.bfly."""
    val = cute.arch.fmax(
        val,
        prims.shfl_sync(
            thread_mask=0xFFFFFFFF,
            val=val,
            offset=2,
            mask_and_clamp=0x1F,
            kind=prims.Shfl.BFLY,
        ),
    )
    val = cute.arch.fmax(
        val,
        prims.shfl_sync(
            thread_mask=0xFFFFFFFF,
            val=val,
            offset=1,
            mask_and_clamp=0x1F,
            kind=prims.Shfl.BFLY,
        ),
    )
    return val


@cute.jit
def nvvm_threadquad_reduction_sum(val: cutlass.Float32) -> cutlass.Float32:
    """Butterfly thread-quad (4 lanes) reduction sum via shfl.sync.bfly."""
    val = val + prims.shfl_sync(
        thread_mask=0xFFFFFFFF,
        val=val,
        offset=2,
        mask_and_clamp=0x1F,
        kind=prims.Shfl.BFLY,
    )
    val = val + prims.shfl_sync(
        thread_mask=0xFFFFFFFF,
        val=val,
        offset=1,
        mask_and_clamp=0x1F,
        kind=prims.Shfl.BFLY,
    )
    return val


@cute.jit
def pack_to_i32(
    src: tuple,
    dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
) -> cutlass.Int32:
    """Pack four 8-bit or two 16-bit values into one 32-bit register."""
    vals = cutlass.Vector.from_elements(src, dtype)
    return vals.bitcast(cutlass.Int32)[0]


def ceil_div(a: int, b: int) -> int:
    """Return the ceiling division of a by b."""
    return (a + b - 1) // b


def sfo_atom_offset(plane, r, c, cols):
    """Byte offset of scale (r, c) in a [rows, cols] SF_O matrix stored in the F8_128x4 atom
    order at ``plane``: ``plane + (r//128)*128*cols + (c//4)*512 + (r%32)*16 + ((r//32)%4)*4 + c%4``."""
    off = plane + (r >> cutlass.Int32(7)) * (cols << cutlass.Int32(7)) + (c >> cutlass.Int32(2)) * cutlass.Int32(512)
    return off + (r & cutlass.Int32(31)) * cutlass.Int32(16) + ((r >> cutlass.Int32(5)) & cutlass.Int32(3)) * cutlass.Int32(4) + (c & cutlass.Int32(3))
