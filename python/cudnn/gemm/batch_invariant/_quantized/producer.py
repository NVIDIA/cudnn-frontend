# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted for cuDNN Frontend from FlashInfer's
# flashinfer/gemm/batch_invariant/blocks/producer.py (e2d163fb3d46).
"""TMA producers stream operand tiles in ascending K order."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute


@cute.jit
def tma_produce_ktiles(
    k_tile_cnt: cutlass.Int32,
    ab_producer,
    tma_atom_a: cute.CopyAtom,
    tma_atom_b: cute.CopyAtom,
    tAgA_slice: cute.Tensor,
    tBgB_slice: cute.Tensor,
    tAsA: cute.Tensor,
    tBsB: cute.Tensor,
    a_mcast_mask,
    b_mcast_mask,
):
    """The k-tiles of one tile (or of one chunk of a tile), ascending, into the ring. Returns the producer (loop-carried)."""
    ab_producer.reset()
    peek_empty = ab_producer.try_acquire()
    for _ in cutlass.range(0, k_tile_cnt, 1, unroll=1):
        handle = ab_producer.acquire_and_advance(peek_empty)  # slot empty; leader arms the tx barrier
        cute.copy(
            tma_atom_a,
            tAgA_slice[(None, handle.count)],
            tAsA[(None, handle.index)],
            tma_bar_ptr=handle.barrier,
            mcast_mask=a_mcast_mask,
        )
        cute.copy(
            tma_atom_b,
            tBgB_slice[(None, handle.count)],
            tBsB[(None, handle.index)],
            tma_bar_ptr=handle.barrier,
            mcast_mask=b_mcast_mask,
        )
        peek_empty = cutlass.Boolean(1)
        if handle.count + 1 < k_tile_cnt:
            peek_empty = ab_producer.try_acquire()
    return ab_producer


@cute.jit
def tma_producer_loop(
    atom_thr_size: cutlass.Constexpr,
    k_tile_cnt: cutlass.Int32,
    tile_sched,
    ab_producer,
    tma_atom_a: cute.CopyAtom,
    tma_atom_b: cute.CopyAtom,
    tAgA: cute.Tensor,
    tBgB: cute.Tensor,
    tAsA: cute.Tensor,
    tBsB: cute.Tensor,
    a_mcast_mask,
    b_mcast_mask,
    chunks=1,
    cc: cutlass.Constexpr = False,
):
    """``cc``: the in-CTA chunk fold (static); ``chunks`` is then its count, a runtime value (the operands' batch extent)."""
    work_tile = tile_sched.initial_work_tile_info()
    while work_tile.is_valid_tile:
        coord = work_tile.tile_idx
        tAgA_slice = tAgA[(None, coord[0] // atom_thr_size, None, coord[2])]
        tBgB_slice = tBgB[(None, coord[1], None, coord[2])]
        ab_producer = tma_produce_ktiles(
            k_tile_cnt,
            ab_producer,
            tma_atom_a,
            tma_atom_b,
            tAgA_slice,
            tBgB_slice,
            tAsA,
            tBsB,
            a_mcast_mask,
            b_mcast_mask,
        )
        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()
    ab_producer.tail()
