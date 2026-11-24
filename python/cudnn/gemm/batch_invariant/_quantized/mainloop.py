# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted for cuDNN Frontend from FlashInfer's
# flashinfer/gemm/batch_invariant/blocks/mainloop.py (e2d163fb3d46).
"""Issue MMA instructions in ascending K order with an accumulator reset for each chunk."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import tcgen05


@cute.jit
def mma_fold(
    lowering,
    k_tile_cnt: cutlass.Int32,
    is_leader_cta,
    tiled_mma: cute.TiledMma,
    ab_consumer,
    acc_pipeline,
    acc_state,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    tCtAcc_base: cute.Tensor,
):
    """One ``kstep`` fold (a tile, or one chunk of a tile) into the accumulator stage ``acc_state`` points at, then the
    stage's commit to the epilogue. Returns the participants it advanced (the DSL carries a loop's objects only through
    names the loop body stores)."""
    tCtAcc = tCtAcc_base[(None, None, None, acc_state.index)]
    ab_consumer.reset()
    peek_full = ab_consumer.try_wait() if is_leader_cta else cutlass.Boolean(1)
    if is_leader_cta:
        acc_pipeline.producer_acquire(acc_state)  # the epilogue has drained this TMEM buffer
    for k_tile in range(k_tile_cnt):  # blocks ascending (the lowering's outer loop)
        if is_leader_cta:
            handle = ab_consumer.wait_and_advance(peek_full)
            for j in lowering.inner_steps():  # the lowering's issue order inside a block
                tiled_mma.set(tcgen05.Field.ACCUMULATE, lowering.accumulate(k_tile, j))  # off on the fold's first step only
                crd = (None, None, j, handle.index)
                cute.gemm(tiled_mma, tCtAcc, tCrA[crd], tCrB[crd], tCtAcc)  # one fold step
            handle.release()
            if handle.count + 1 < k_tile_cnt:
                peek_full = ab_consumer.try_wait()
            else:
                peek_full = cutlass.Boolean(1)
    if is_leader_cta:
        acc_pipeline.producer_commit(acc_state)  # accumulator complete: hand to the epilogue
    acc_state.advance()
    return ab_consumer, acc_state


@cute.jit
def mma_loop(
    lowering,
    num_acc_stage: cutlass.Constexpr,
    k_tile_cnt: cutlass.Int32,
    is_leader_cta,
    tile_sched,
    tiled_mma: cute.TiledMma,
    ab_consumer,
    acc_pipeline,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    tCtAcc_base: cute.Tensor,
    chunks=1,
    cc: cutlass.Constexpr = False,
):
    """``cc``: the in-CTA chunk fold (static); ``chunks`` is then the class's chunk count, a runtime value (``k_tile_cnt``
    is the k-tiles of ONE chunk and the fold is issued per chunk, chunks ascending, each committed to the epilogue)."""
    assert cute.size(tCrA, mode=[2]) == lowering.inner, "one smem stage must hold exactly the lowering's inner steps"
    acc_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, num_acc_stage)
    work_tile = tile_sched.initial_work_tile_info()
    while work_tile.is_valid_tile:
        if cutlass.const_expr(not cc):
            ab_consumer, acc_state = mma_fold(
                lowering,
                k_tile_cnt,
                is_leader_cta,
                tiled_mma,
                ab_consumer,
                acc_pipeline,
                acc_state,
                tCrA,
                tCrB,
                tCtAcc_base,
            )
        else:
            for _c in cutlass.range(0, chunks, 1, unroll=1):  # chunks ascending, each a fold from +0 into the next stage
                ab_consumer, acc_state = mma_fold(
                    lowering,
                    k_tile_cnt,
                    is_leader_cta,
                    tiled_mma,
                    ab_consumer,
                    acc_pipeline,
                    acc_state,
                    tCrA,
                    tCrB,
                    tCtAcc_base,
                )
        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()
    acc_pipeline.producer_tail(acc_state)
