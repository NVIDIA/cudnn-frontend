# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Read the FP32 accumulator from TMEM, apply the scalar scale, convert and TMA-store."""

from __future__ import annotations

from typing import Optional

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.utils.blackwell_helpers import get_smem_store_op, get_tmem_load_op
from cutlass.utils.gemm.sm100 import transform_partitioned_tensor_layout

from cudnn.frost.tile_dsl.barrier import named_barrier_sync_unaligned

from .arithmetic import CuteF32
from .operands import ACC_DTYPE, BAR_EPILOGUE, EPILOGUE_WARPS


def apply_chain_pre_cvt(chain, frag, alpha):
    """The chain's ops before the conversion, per element, in fp32 through ``CuteF32``; the conversion itself is the
    vector ``.to(out dtype)`` that follows in the caller."""
    if chain.pre:
        for i in range(cute.size(frag)):
            frag[i] = chain.apply_pre(CuteF32, frag[i], alpha)


@cute.jit
def epilogue_loop(
    st,
    chain,
    atom_thr_size: cutlass.Constexpr,
    tidx: cutlass.Int32,
    warp_idx: cutlass.Int32,
    tile_sched,
    tma_atom_c: cute.CopyAtom,
    tCtAcc_base: cute.Tensor,
    sC: cute.Tensor,
    tCgC_base: cute.Tensor,
    epi_tile: cute.Tile,
    acc_pipeline,
    alpha,
    chunks=1,
    mBias: Optional[cute.Tensor] = None,
    tCcC_base: Optional[cute.Tensor] = None,
    cc: cutlass.Constexpr = False,
):
    """``chain`` is the specification's epilogue chain (``spec.EpilogueChain``) and ``alpha`` its runtime scalar (None for
    the unscaled chain). ``cc`` (static) with ``chunks`` its runtime count: the in-CTA chunk fold (module docstring) -- the mainloop commits one accumulator
    stage per chunk, the first ``chunks - 1`` are folded into the TMEM running sum (``st.sum_stage``) and released, the
    last one receives the fold and is stored through the ordinary subtile loop below (the chain once, after the fold).
    ``mBias`` (the (N,) bias) with ``tCcC_base`` (the output coordinates partitioned like C): the biased chain, the
    bias of each element's column through the same ``apply_pre`` before the conversion -- after the whole fold, chunked
    or not, so the bias enters exactly once."""
    # ((ATOM_M, MMA_M), (ATOM_N, MMA_N), ...)
    tCgC = transform_partitioned_tensor_layout(tCgC_base)
    tCtAcc = transform_partitioned_tensor_layout(tCtAcc_base)
    # TMEM -> registers: one epilogue subtile per copy
    copy_atom_t2r = get_tmem_load_op(
        st.cta_tile_shape_mnk,
        st.c_layout,
        st.c_dtype,
        ACC_DTYPE,
        epi_tile,
        st.use_2cta_instrs,
    )
    tAcc_epi = cute.flat_divide(tCtAcc, epi_tile)  # (EPI_M, EPI_N, SUB_M, SUB_N, STAGE)
    tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])
    thr_t2r = tiled_copy_t2r.get_slice(tidx)
    tTR_tAcc_base = thr_t2r.partition_S(tAcc_epi)
    tCgC_epi = cute.flat_divide(tCgC, epi_tile)  # (EPI_M, EPI_N, SUB_M, SUB_N, RestM, RestN, RestL)
    tTR_gC = thr_t2r.partition_D(tCgC_epi)
    tTR_rAcc = cute.make_rmem_tensor(tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, ACC_DTYPE)
    tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, st.c_dtype)
    # registers -> smem
    copy_atom_r2s = get_smem_store_op(st.c_layout, st.c_dtype, ACC_DTYPE, tiled_copy_t2r)
    tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
    thr_r2s = tiled_copy_r2s.get_slice(tidx)
    tRS_sC = thr_r2s.partition_D(sC)
    tRS_rC = tiled_copy_r2s.retile(tTR_rC)
    # smem -> global (TMA store)
    bSG_sC, bSG_gC_all = cpasync.tma_partition(
        tma_atom_c,
        0,
        cute.make_layout(1),
        cute.group_modes(sC, 0, 2),
        cute.group_modes(tCgC_epi, 0, 2),
    )
    acc_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, st.num_acc_stage)
    c_pipeline = pipeline.PipelineTmaStore.create(
        num_stages=st.num_c_stage,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 32 * len(EPILOGUE_WARPS)),
    )
    work_tile = tile_sched.initial_work_tile_info()
    while work_tile.is_valid_tile:
        coord = work_tile.tile_idx
        mnl = (coord[0] // atom_thr_size, coord[1], coord[2])
        tile_sched.advance_to_next_work()  # pre-advance: the executed count includes this tile
        work_tile = tile_sched.get_current_work()
        num_tiles_executed = tile_sched.num_tiles_executed
        bSG_gC = bSG_gC_all[(None, None, None, *mnl)]
        tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_state.index)]
        acc_pipeline.consumer_wait(acc_state)  # the fold of this tile (its last chunk) is complete
        tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
        bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))
        subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
        num_prev_subtiles = num_tiles_executed * subtile_cnt
        tTR_cC = None
        for s in range(subtile_cnt):
            cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, s)], tTR_rAcc)
            apply_chain_pre_cvt(chain, tTR_rAcc, alpha)  # the chain's fp32 ops (mul.rn alpha when scaled)
            acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
            tRS_rC.store(acc_vec.to(st.c_dtype))  # the chain's cvt.rn <out>, exactly once
            c_buffer = (num_prev_subtiles + s) % st.num_c_stage
            cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, c_buffer)])
            cute.arch.fence_proxy("async.shared", space="cta")
            named_barrier_sync_unaligned(BAR_EPILOGUE, 32 * len(EPILOGUE_WARPS))
            if warp_idx == EPILOGUE_WARPS[0]:
                cute.copy(tma_atom_c, bSG_sC[(None, c_buffer)], bSG_gC[(None, s)])
                c_pipeline.producer_commit()
                c_pipeline.producer_acquire()
            named_barrier_sync_unaligned(BAR_EPILOGUE, 32 * len(EPILOGUE_WARPS))
        named_barrier_sync_unaligned(BAR_EPILOGUE, 32 * len(EPILOGUE_WARPS))
        with cute.arch.elect_one():
            acc_pipeline.consumer_release(acc_state)  # TMEM buffer free for the next fold
        acc_state.advance()
    c_pipeline.producer_tail()
