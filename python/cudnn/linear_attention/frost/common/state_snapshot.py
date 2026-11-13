# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Copy the states an aliased launch reads into scratch, so its exact twin can redo it.

A ``STEEP_FAST`` launch writes its final state before its ``STEEP_EXACT`` twin runs (common/steep_decay.py).  When that
output aliases the state the launch reads (``overwrite_initial_state``, a ``state_indices`` pool, or the backward's
``d_initial_state`` over ``d_final_state``), the plan first copies the read state into a dense workspace snapshot and
both launches read the copy.  ``indices`` gathers pool rows in sequence order; sequences that name the same pool row all
read its value from before the call.  The source may have any strides with a unit innermost stride and 16-byte aligned
``[.., v, :]`` rows, the layout the kernels' own state loads require."""

from typing import Optional

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda

from cudnn.frost.tile_dsl.barrier import wait_on_dependent_grids
from cudnn.frost.tile_dsl.tma import ld_global_v4, st_global_v4

USE_PDL = True

BLOCK = 256


@cute.kernel
def frost_state_snapshot(mSrc: cute.Tensor, mDst: cute.Tensor, mIdx: Optional[cute.Tensor], line_chunks: cutlass.Int64) -> None:
    if cutlass.const_expr(USE_PDL):
        wait_on_dependent_grids()
    tidx, _, _ = cute.arch.thread_idx()
    bidx = cute.arch.block_idx()[0]
    chunk = cutlass.Int64(cutlass.Int32(bidx)) * cutlass.Int64(BLOCK) + cutlass.Int64(cutlass.Int32(tidx))
    lines = cutlass.Int64(mDst.shape[0]) * cutlass.Int64(mDst.shape[1]) * cutlass.Int64(mDst.shape[2])
    if chunk < lines * line_chunks:
        line = chunk // line_chunks
        word = chunk - line * line_chunks
        rows_per_seq = cutlass.Int64(mDst.shape[1]) * cutlass.Int64(mDst.shape[2])
        seq = line // rows_per_seq
        rem = line - seq * rows_per_seq
        head_wide = rem // cutlass.Int64(mDst.shape[2])
        head = head_wide.to(cutlass.Int32)
        v_row = (rem - head_wide * cutlass.Int64(mDst.shape[2])).to(cutlass.Int32)
        slot = seq.to(cutlass.Int32)
        if cutlass.const_expr(mIdx is not None):
            slot = cutlass.Int32(mIdx[slot])
        elem_bytes = cutlass.const_expr(mSrc.element_type.width // 8)
        offset = cutlass.Int64(mSrc.layout((slot, head, v_row, 0)))
        src = mSrc.iterator.toint() + offset * cutlass.Int64(elem_bytes) + word * cutlass.Int64(16)
        dst = mDst.iterator.toint() + chunk * cutlass.Int64(16)
        w0, w1, w2, w3 = ld_global_v4(src, cutlass.Int32)
        st_global_v4(dst, (w0, w1, w2, w3), cutlass.Int32)


frost_state_snapshot.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def snapshot_states(mSrc: cute.Tensor, mDst: cute.Tensor, mIdx: Optional[cute.Tensor], stream: cuda.CUstream) -> None:
    """``mDst[i] = mSrc[mIdx[i] if mIdx else i]`` for every row of the dense ``mDst``; issued from another kernel's host."""
    elem_bytes = cutlass.const_expr(mSrc.element_type.width // 8)
    line_chunks = cutlass.Int64(mDst.shape[3]) * cutlass.Int64(elem_bytes) // cutlass.Int64(16)
    total = cutlass.Int64(mDst.shape[0]) * cutlass.Int64(mDst.shape[1]) * cutlass.Int64(mDst.shape[2]) * line_chunks
    grid_x = ((total + cutlass.Int64(BLOCK - 1)) // cutlass.Int64(BLOCK)).to(cutlass.Int32)
    frost_state_snapshot(mSrc, mDst, mIdx, line_chunks).launch(grid=(grid_x, 1, 1), block=(BLOCK, 1, 1), stream=stream, use_pdl=USE_PDL)
