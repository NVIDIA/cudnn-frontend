# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""PTX helpers shared by the SM120 SDPA-forward kernels.

``sm120/prefill_f16.py``, ``sm120/prefill_fp8.py``, ``sm120/prefill_mxfp8.py``,
the D=256 specialization ``sm120/prefill_d256_f16.py`` and the D=512 flavors
``sm120/prefill_d512_f16.py``, ``sm120/prefill_d512_fp8.py`` and
``sm120/prefill_d512_mxfp8.py`` are self-contained
kernels that differ in head-tile geometry, MMA shape, and warp roles; the
lane-level primitives below are shared between them (``sfo_atom_offset`` by the
two block-scaled-O kernels, FP8 and MXFP8; ``mxfp8_sf_coords`` by the two MXFP8
kernels) and live here so a fix lands once.
"""

from types import SimpleNamespace
from typing import Optional, Type

import cutlass
import cutlass.cute as cute

from cutlass.experimental import primitives as prims
from cudnn.frost.tile_dsl.sf_layout import SF_ATOM_BYTES, SF_ATOM_ROWS

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


@cute.jit
def mxfp8_sf_coords(
    sf_q: cute.Tensor,
    sf_k: cute.Tensor,
    sf_v: cute.Tensor,
    seq_kv_lens: Optional[cute.Tensor],
    batch_idx: cutlass.Int32,
    kv_head_idx: cutlass.Int32,
    num_heads_q: cutlass.Int32,
    num_heads_kv: cutlass.Int32,
    lane: cutlass.Int32,
    v_sf_tile_bytes: cutlass.Constexpr[int],
    thd_varlen: cutlass.Constexpr[bool],
) -> SimpleNamespace:
    """Scale-tile coordinates of one (sequence, head) unit of an MXFP8 kernel;
    every lane of the warp calls it.

    Dense scale tensors are (B, H, tiles, tile_bytes). THD packs each sequence's
    tiles after the earlier sequences' ceil(len / 128) tiles, which the lanes
    split and a butterfly sums. Dense V planes are B * H_kv * tiles atoms apart;
    THD keeps a tile's planes contiguous. The Q scale tile of (head, token) is
    ``(q_bh0 + head) * q_tiles + q_tile0 + token // 128``.
    """
    sf_q_bh0 = batch_idx * num_heads_q
    sf_kv_bh0 = batch_idx * num_heads_kv
    sf_q_tile0 = cutlass.Int32(0)
    sf_kv_tile0 = cutlass.Int32(0)
    if cutlass.const_expr(thd_varlen):
        sf_q_bh0 = cutlass.Int32(0)
        sf_kv_bh0 = cutlass.Int32(0)
        n_batch_meta = (seq_kv_lens.shape[0] - 4) // 4
        meta_sf = cutlass.make_array_view(seq_kv_lens)
        for prev in cutlass.range(lane, batch_idx, cute.arch.WARP_SIZE, unroll=1):
            len_q_prev = cutlass.Int32(meta_sf[n_batch_meta + prev + 1]) - cutlass.Int32(meta_sf[n_batch_meta + prev])
            len_kv_prev = cutlass.Int32(meta_sf[2 * n_batch_meta + 2 + prev]) - cutlass.Int32(meta_sf[2 * n_batch_meta + 1 + prev])
            sf_q_tile0 = sf_q_tile0 + ceil_div(len_q_prev, SF_ATOM_ROWS)
            sf_kv_tile0 = sf_kv_tile0 + ceil_div(len_kv_prev, SF_ATOM_ROWS)
        for i in cutlass.range_constexpr(5):
            sf_q_tile0 = sf_q_tile0 + prims.shfl_sync(thread_mask=0xFFFFFFFF, val=sf_q_tile0, offset=16 >> i, mask_and_clamp=0x1F, kind=prims.Shfl.BFLY)
            sf_kv_tile0 = sf_kv_tile0 + prims.shfl_sync(thread_mask=0xFFFFFFFF, val=sf_kv_tile0, offset=16 >> i, mask_and_clamp=0x1F, kind=prims.Shfl.BFLY)
    sf_kv_tiles = cutlass.Int32(sf_k.shape[2])
    v_sf_tile_stride = v_sf_tile_bytes if cutlass.const_expr(thd_varlen) else SF_ATOM_BYTES
    v_sf_plane_stride = cutlass.Int64(SF_ATOM_BYTES)
    if cutlass.const_expr(not thd_varlen):
        v_sf_plane_stride = cutlass.Int64(sf_v.shape[0]) * cutlass.Int64(sf_v.shape[1]) * cutlass.Int64(sf_v.shape[2]) * SF_ATOM_BYTES
    return SimpleNamespace(
        q_ptr=sf_q.iterator.raw_ptr(),
        k_ptr=sf_k.iterator.raw_ptr(),
        v_ptr=sf_v.iterator.raw_ptr(),
        q_bh0=sf_q_bh0,
        q_tiles=cutlass.Int32(sf_q.shape[2]),
        q_tile0=sf_q_tile0,
        kv_tile_base=(cutlass.Int64(sf_kv_bh0) + cutlass.Int64(kv_head_idx)) * sf_kv_tiles + sf_kv_tile0,
        v_tile_stride=v_sf_tile_stride,
        v_plane_stride=v_sf_plane_stride,
    )
