# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""PTX helpers shared by the SM120 SDPA-forward kernels.

``sm120/prefill_f16.py``, ``sm120/prefill_fp8.py``, ``sm120/prefill_mxfp8.py``,
and the D=256 specialization ``sm120/prefill_d256_f16.py`` are self-contained
kernels that differ in head-tile geometry, MMA shape, and warp roles; the
lane-level primitives below are shared between them (``sfo_atom_offset`` by the
two block-scaled-O kernels, FP8 and MXFP8; the PackGQA tile geometry by these and
the three D=512 templates) and live here so a fix lands once.
"""

from typing import NamedTuple, Type

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


class PackedTileGeometry(NamedTuple):
    """How one Q tile's rows map to tokens, shared by every SM120 template.

    A PackGQA tile holds ``tile_tokens = q_tile // G`` whole tokens of the G query
    heads sharing one KV head, token-major (row r: token ``r // G``, head ``r % G``).
    When G does not divide the tile, the rows past ``live_rows = tile_tokens * G``
    are dead: zero Q, never stored, outside Amax_O (G = 6 at 64 rows: ten tokens in
    60 live rows). ``diag_shifted``: the causal diagonal can straddle one more KV
    tile than a tile-aligned top-left one, when it is translated (bottom-right or a
    right band) or when Q tiles start off the KV grid.
    """

    tile_tokens: int
    live_rows: int
    dead_rows: bool
    diag_shifted: bool


def packed_tile_geometry(q_tile: int, kv_tile: int, pack_gqa: bool, qh_per_kh: int, *, bottom_right: bool, window_right: int) -> PackedTileGeometry:
    """The :class:`PackedTileGeometry` of a ``q_tile`` x ``kv_tile`` CTA tile."""
    tile_tokens = q_tile // qh_per_kh if pack_gqa else q_tile
    live_rows = tile_tokens * qh_per_kh if pack_gqa else q_tile
    off_grid = tile_tokens % kv_tile != 0 and kv_tile % tile_tokens != 0
    return PackedTileGeometry(tile_tokens, live_rows, live_rows != q_tile, bottom_right or window_right > 0 or off_grid)


@cute.jit
def zero_dead_q_rows(frag, row_lo, live_rows: cutlass.Constexpr[int]):
    """Zero a packed tile's dead rows in one ``ldmatrix.x4`` A fragment of a TMA-staged
    Q tile: the packed TMA box holds only the live rows, so theirs are stale slab bytes.
    Registers 0 and 2 hold row ``row_lo`` (the warp's first row + ``lane // 4``),
    registers 1 and 3 that row + 8."""
    keep_lo = cutlass.Int32(-1) if row_lo < live_rows else cutlass.Int32(0)
    keep_hi = cutlass.Int32(-1) if row_lo + 8 < live_rows else cutlass.Int32(0)
    return cutlass.Vector.from_elements((frag[0] & keep_lo, frag[1] & keep_hi, frag[2] & keep_lo, frag[3] & keep_hi), cutlass.Int32)


def sfo_atom_offset(plane, r, c, cols):
    """Byte offset of scale (r, c) in a [rows, cols] SF_O matrix stored in the F8_128x4 atom
    order at ``plane``: ``plane + (r//128)*128*cols + (c//4)*512 + (r%32)*16 + ((r//32)%4)*4 + c%4``."""
    off = plane + (r >> cutlass.Int32(7)) * (cols << cutlass.Int32(7)) + (c >> cutlass.Int32(2)) * cutlass.Int32(512)
    return off + (r & cutlass.Int32(31)) * cutlass.Int32(16) + ((r >> cutlass.Int32(5)) & cutlass.Int32(3)) * cutlass.Int32(4) + (c & cutlass.Int32(3))
