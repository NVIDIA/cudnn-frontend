# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Native packed D192/V128 half backward with linear FP32 dQ scratch."""

import cutlass
import cutlass.cute as cute
from cutlass.experimental import primitives as nvvm

from cudnn.frost.tile_dsl.barrier import PipelineState, advance
from cudnn.frost.tile_dsl.handles import tma_slice_runtime_desc
from cudnn.frost.tile_dsl.thd import TENSOR_MAP_QWORDS, thd_port_origin
from cudnn.frost.tile_dsl.tma import tma_tensormap_acquire, tma_reduce_add_subtile
from cudnn.frost.tile_dsl.mma import mma_ss
from cudnn.frost.tile_dsl.scheduler import Sched, read_clc_payload
from cudnn.frost.tile_dsl.tma import tma_load_subtiles, tma_load_tile, tma_store_commit, tma_store_tile, tma_store_wait

from cudnn.sdpa.bwd.config_sm100_d192_accum import CFG
from .bprop_d192_accum_f16 import (
    _view,
    _bars,
    _operand,
    _mma_desc,
    _wait,
    _wait_tcgen,
    _commit,
    _credit,
    _tmem_load,
    _stat_publish,
    _p_values,
    _ds_values,
    _store_half_row,
    _store_dq_rows,
    _native_epilogue,
    _ts,
    _service_regs,
    _scheduler_loop,
)

MAP_Q, MAP_K, MAP_V, MAP_DO, MAP_DQ, MAP_DK, MAP_DV = range(7)

LOG2E = 1.4426950408889634
DQ_ROWS_PER_WARP = 16


@cute.jit
def _next_tile(sched, bars, state):
    _wait(bars.sched_full[state.idx], state.phase)
    x, yz, valid = read_clc_payload(sched, state.idx * 8, warp_broadcast=True)
    return x, yz & 0xFFFF, (yz >> 16) & 0xFFFF, valid, advance(state, CFG.SCHED_STAGES)


@cute.jit
def _bounds(block, meta, batch, n_batch, causal: cutlass.Constexpr[bool]):
    data = cutlass.make_array_view(meta)
    q_start = cutlass.Int32(data[n_batch + batch])
    seq_q = cutlass.Int32(data[n_batch + batch + 1]) - q_start
    seq_k = cutlass.Int32(data[batch])
    lo = block * (CFG.M // CFG.N) if cutlass.const_expr(causal) else cutlass.Int32(0)
    hi = (seq_q + CFG.N - 1) // CFG.N
    if block * CFG.M >= seq_k:
        lo = cutlass.Int32(0)
        hi = cutlass.Int32(0)
    return lo, hi, seq_q, seq_k, q_start


@cute.jit
def _stat_load(lse, delta, row, group, qbase, sq, batch, head, q_start, origins, stats_origin: cutlass.Constexpr[int]):
    val = cutlass.Float32(0.0)
    if row < CFG.N and qbase + row < sq:
        local_row = qbase + row
        if group == 0:
            token = thd_port_origin(origins, stats_origin, batch, q_start) + cutlass.Int64(local_row)
            if cutlass.const_expr(len(lse.shape) == 2):
                val = cutlass.Float32(lse[token, head])
            else:
                val = cutlass.Float32(lse[0, head, token])
        else:
            val = cutlass.Float32(delta[0, head, cutlass.Int64(q_start) + local_row])
    return val


@cute.jit
def _softmax_loop(
    slab,
    bars,
    sched,
    tmem_base,
    lse,
    delta,
    origins,
    stats_origin: cutlass.Constexpr[int],
    block,
    head,
    batch,
    meta,
    n_batch,
    scale,
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
):
    nvvm.setmaxregister(CFG.SOFTMAX_REGS, nvvm.SetMaxRegisterAction.INCREASE)
    tid = cute.arch.thread_idx()[0]
    row = tid % CFG.M
    group = tid // CFG.M
    state = PipelineState.start()
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        if cutlass.const_expr(not causal):
            _credit(bars.sched_empty[sched_state.idx])
        lo, hi, sq, skv, q_start = _bounds(block, meta, batch, n_batch, causal)
        if lo < hi:
            _stat_publish(
                slab, row, group, lo % 2, lo * CFG.N, sq, scale, _stat_load(lse, delta, row, group, lo * CFG.N, sq, batch, head, q_start, origins, stats_origin)
            )
        for qtile in cutlass.range(lo, hi, 1, unroll=1):
            stat_next = _stat_load(lse, delta, row, group, (qtile + 1) * CFG.N, sq, batch, head, q_start, origins, stats_origin)
            _wait_tcgen(bars.s_done, state.phase)
            p = _p_values(slab, tmem_base, row, group, block, qtile, lo, skv, scale, causal, dtype)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
            bars.p_ready.arrive()
            _wait_tcgen(bars.dp_done, state.phase)
            ds_words = _ds_values(slab, tmem_base, group, qtile, p, scale, dtype)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
            bars.ds_ready.arrive()
            _store_half_row(slab, CFG.DS, row, group, ds_words)
            nvvm.fence_proxy("async.shared", space="cta")
            bars.ds_smem.arrive()
            if qtile + 1 < hi:
                _stat_publish(slab, row, group, (qtile + 1) % 2, (qtile + 1) * CFG.N, sq, scale, stat_next)
            state = advance(state, 1)
        if cutlass.const_expr(causal):
            _credit(bars.sched_empty[sched_state.idx])
        block, head, batch, valid, sched_state = _next_tile(sched, bars, sched_state)


@cute.jit
def _epilogue_loop(
    slab,
    bars,
    sched,
    tmem_base,
    block,
    head,
    batch,
    meta,
    n_batch,
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
):
    nvvm.setmaxregister(CFG.EPILOGUE_REGS, nvvm.SetMaxRegisterAction.DECREASE)
    tid = cute.arch.thread_idx()[0]
    row = tid % CFG.M
    lane = tid % 32
    dq_row = (tid // 32) % 4 * DQ_ROWS_PER_WARP + lane
    dq_lane = lane < DQ_ROWS_PER_WARP
    dq_state = PipelineState.start()
    stage = PipelineState.start()
    empty_state = PipelineState.start(phase=1)
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit(bars.sched_empty[sched_state.idx])
        lo, hi, sq, skv, q_start = _bounds(block, meta, batch, n_batch, causal)
        for _ in cutlass.range(lo, hi, 1, unroll=1):
            for _ in cutlass.range_constexpr(CFG.DQ_CHUNKS):
                _wait_tcgen(bars.dq_done, dq_state.phase)
                vals = _tmem_load(tmem_base + CFG.TMEM_DQ, 64)
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
                bars.dq_read.arrive()
                _wait(bars.dq_empty[stage.idx], empty_state.phase)
                if dq_lane:
                    _store_dq_rows(slab, stage.idx, dq_row, vals)
                nvvm.fence_proxy("async.shared", space="cta")
                bars.dq_full[stage.idx].arrive()
                dq_state = advance(dq_state, 1)
                stage = advance(stage, CFG.DQ_STAGES)
                empty_state = advance(empty_state, CFG.DQ_STAGES)
        if lo >= hi:
            _wait_tcgen(bars.dq_done, dq_state.phase)
            dq_state = advance(dq_state, 1)
        _native_epilogue(slab, tmem_base, bars, row, lo < hi, dtype, False)
        _native_epilogue(slab, tmem_base, bars, row, lo < hi, dtype, True)
        block, head, batch, valid, sched_state = _next_tile(sched, bars, sched_state)


@cute.jit
def _mma_loop(
    slab,
    bars,
    sched,
    tmem_base,
    block,
    head,
    batch,
    meta,
    n_batch,
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
):
    score = _mma_desc(dtype, CFG.M, CFG.N, CFG.DQK)
    dpdesc = _mma_desc(dtype, CFG.M, CFG.N, CFG.DV)
    dvdesc = _mma_desc(dtype, CFG.M, CFG.DV, CFG.N, btranspose=True)
    dkdesc = _mma_desc(dtype, CFG.M, CFG.DQK, CFG.N, btranspose=True)
    dqdesc = _mma_desc(dtype, CFG.N, CFG.DQ_CHUNK, CFG.M, atranspose=True, btranspose=True)
    kd = _operand(slab, CFG.K, dtype, CFG.M, CFG.DQK).desc()
    vd = _operand(slab, CFG.V, dtype, CFG.M, CFG.DV).desc()
    dst = _operand(slab, CFG.DS, dtype, CFG.M, CFG.N, transpose=True).desc()
    s = nvvm.make_tmem_ptr(tmem_base + CFG.TMEM_S, cutlass.Float32)
    dp = nvvm.make_tmem_ptr(tmem_base + CFG.TMEM_DP, cutlass.Float32)
    dq = nvvm.make_tmem_ptr(tmem_base + CFG.TMEM_DQ, cutlass.Float32)
    dv = nvvm.make_tmem_ptr(tmem_base + CFG.TMEM_DV, cutlass.Float32)
    dk = nvvm.make_tmem_ptr(tmem_base + CFG.TMEM_DK, cutlass.Float32)
    q_state = PipelineState.start()
    do_state = PipelineState.start()
    kv_state = PipelineState.start()
    ready_state = PipelineState.start()
    read_state = PipelineState.start(phase=1)
    out_empty = PipelineState.start(phase=1)
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit(bars.sched_empty[sched_state.idx])
        lo, hi, sq, skv, q_start = _bounds(block, meta, batch, n_batch, causal)
        # Empty work still acknowledges the loader's K-buffer acquisition.
        # Otherwise its zero-output store can lap the loader's k_empty wait.
        _wait_tcgen(bars.k_full, kv_state.phase)
        if lo < hi:
            _wait_tcgen(bars.q_full[q_state.idx], q_state.phase)
            qd = _operand(slab, CFG.Q + q_state.idx * CFG.Q_BYTES, dtype, CFG.N, CFG.DQK).desc()
            mma_ss(score, kd, qd, s)
            _commit(bars.s_done)
        _wait_tcgen(bars.v_full, kv_state.phase)
        kv_state = advance(kv_state, 1)
        _wait_tcgen(bars.dvk_empty, out_empty.phase)
        out_empty = advance(out_empty, 1)
        for qtile in cutlass.range(lo, hi, 1, unroll=1):
            dod = _operand(slab, CFG.DO + do_state.idx * CFG.DO_BYTES, dtype, CFG.N, CFG.DV).desc()
            dot = _operand(slab, CFG.DO + do_state.idx * CFG.DO_BYTES, dtype, CFG.N, CFG.DV, transpose=True).desc()
            for half in cutlass.range_constexpr(2):
                _wait_tcgen(bars.do_full[do_state.idx * 2 + half], do_state.phase)
                mma_ss(dpdesc, vd, dod, dp, k_start=half * 4, k_count=4, accumulate=half > 0)
            _commit(bars.dp_done)
            _wait_tcgen(bars.p_ready, ready_state.phase)
            _ts(dvdesc, tmem_base + CFG.TMEM_S, dot, dv, qtile > lo)
            _commit(bars.do_empty[do_state.idx])
            q_next = advance(q_state, CFG.Q_STAGES)
            _wait_tcgen(bars.ds_smem, ready_state.phase)
            _wait_tcgen(bars.dq_read, read_state.phase)
            read_state = advance(read_state, 1)
            mma_ss(dqdesc, dst, _operand(slab, CFG.K, dtype, CFG.M, CFG.DQ_CHUNK, transpose=True).desc(), dq)
            _commit(bars.dq_done)
            if qtile + 1 < hi:
                _wait_tcgen(bars.q_full[q_next.idx], q_next.phase)
                qd_next = _operand(slab, CFG.Q + q_next.idx * CFG.Q_BYTES, dtype, CFG.N, CFG.DQK).desc()
                mma_ss(score, kd, qd_next, s)
                _commit(bars.s_done)
            _wait_tcgen(bars.dq_read, read_state.phase)
            read_state = advance(read_state, 1)
            mma_ss(dqdesc, dst, _operand(slab, CFG.K + CFG.M * CFG.SWIZZLE_BYTES, dtype, CFG.M, CFG.DQ_CHUNK, transpose=True).desc(), dq)
            _commit(bars.dq_done)
            _wait_tcgen(bars.ds_ready, ready_state.phase)
            qt = _operand(slab, CFG.Q + q_state.idx * CFG.Q_BYTES, dtype, CFG.N, CFG.DQK, transpose=True).desc()
            _ts(dkdesc, tmem_base + CFG.TMEM_DP, qt, dk, qtile > lo)
            _commit(bars.q_empty[q_state.idx])
            _wait_tcgen(bars.dq_read, read_state.phase)
            read_state = advance(read_state, 1)
            mma_ss(dqdesc, dst, _operand(slab, CFG.K + 2 * CFG.M * CFG.SWIZZLE_BYTES, dtype, CFG.M, CFG.DQ_CHUNK, transpose=True).desc(), dq)
            _commit(bars.dq_done)
            q_state = q_next
            do_state = advance(do_state, CFG.DO_STAGES)
            ready_state = advance(ready_state, 1)
        if lo >= hi:
            _commit(bars.dq_done)
        block, head, batch, valid, sched_state = _next_tile(sched, bars, sched_state)


@cute.jit
def _load_loop(
    slab,
    bars,
    sched,
    maps,
    block,
    head,
    batch,
    meta,
    n_batch,
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
):
    q_state = PipelineState.start(phase=1)
    do_state = PipelineState.start(phase=1)
    kv_state = PipelineState.start(phase=1)
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit(bars.sched_empty[sched_state.idx])
        lo, hi, sq, skv, q_start = _bounds(block, meta, batch, n_batch, causal)
        q_desc = _map_ptr(maps, batch, n_batch, MAP_Q)
        k_desc = _map_ptr(maps, batch, n_batch, MAP_K)
        v_desc = _map_ptr(maps, batch, n_batch, MAP_V)
        do_desc = _map_ptr(maps, batch, n_batch, MAP_DO)
        _wait(bars.k_empty, kv_state.phase)
        kv_state = advance(kv_state, 1)
        if lo < hi:
            tma_tensormap_acquire(q_desc)
            tma_tensormap_acquire(k_desc)
            tma_tensormap_acquire(v_desc)
            tma_tensormap_acquire(do_desc)
            bars.k_full.arrive(n_bytes=CFG.K_BYTES, pred=nvvm.elect_sync())
            tma_load_tile(
                _operand(slab, CFG.K, dtype, CFG.M, CFG.DQK), tma_slice_runtime_desc(k_desc, 0, block * CFG.M, head, 0), bars.k_full.smem_ptr, acquire=False
            )
            bars.v_full.arrive(n_bytes=CFG.V_BYTES, pred=nvvm.elect_sync())
            tma_load_tile(
                _operand(slab, CFG.V, dtype, CFG.M, CFG.DV), tma_slice_runtime_desc(v_desc, 0, block * CFG.M, head, 0), bars.v_full.smem_ptr, acquire=False
            )
        else:
            bars.k_full.arrive(n_bytes=0, pred=nvvm.elect_sync())
            bars.v_full.arrive(n_bytes=0, pred=nvvm.elect_sync())
        for qtile in cutlass.range(lo, hi, 1, unroll=1):
            _wait(bars.q_empty[q_state.idx], q_state.phase)
            bars.q_full[q_state.idx].arrive(n_bytes=CFG.Q_BYTES, pred=nvvm.elect_sync())
            tma_load_tile(
                _operand(slab, CFG.Q + q_state.idx * CFG.Q_BYTES, dtype, CFG.N, CFG.DQK),
                tma_slice_runtime_desc(q_desc, 0, qtile * CFG.N, head, 0),
                bars.q_full[q_state.idx].smem_ptr,
                acquire=False,
            )
            _wait(bars.do_empty[do_state.idx], do_state.phase)
            for half in cutlass.range_constexpr(2):
                bars.do_full[do_state.idx * 2 + half].arrive(n_bytes=CFG.DO_BYTES // 2, pred=nvvm.elect_sync())
            for half in cutlass.range_constexpr(2):
                tma_load_subtiles(
                    _operand(slab, CFG.DO + do_state.idx * CFG.DO_BYTES, dtype, CFG.N, CFG.DV),
                    tma_slice_runtime_desc(do_desc, 0, qtile * CFG.N, head, 0),
                    bars.do_full[do_state.idx * 2 + half].smem_ptr,
                    half,
                    1,
                    acquire=False,
                )
            q_state = advance(q_state, CFG.Q_STAGES)
            do_state = advance(do_state, CFG.DO_STAGES)
        block, head, batch, valid, sched_state = _next_tile(sched, bars, sched_state)


@cute.jit
def _store_loop(
    slab,
    bars,
    sched,
    maps,
    block,
    head,
    batch,
    meta,
    n_batch,
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
):
    full_state = PipelineState.start()
    out_state = PipelineState.start()
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit(bars.sched_empty[sched_state.idx])
        lo, hi, sq, skv, q_start = _bounds(block, meta, batch, n_batch, causal)
        dq_desc = _map_ptr(maps, batch, n_batch, MAP_DQ)
        dk_desc = _map_ptr(maps, batch, n_batch, MAP_DK)
        dv_desc = _map_ptr(maps, batch, n_batch, MAP_DV)
        if lo < hi:
            tma_tensormap_acquire(dq_desc)
        for qtile in cutlass.range(lo, hi, 1, unroll=1):
            for chunk in cutlass.range_constexpr(CFG.DQ_CHUNKS):
                _wait(bars.dq_full[full_state.idx], full_state.phase)
                staged = _operand(slab, CFG.DQ_STAGE + full_state.idx * CFG.DQ_STAGE_BYTES, cutlass.Float32, CFG.N, CFG.DQ_CHUNK)
                for sub in cutlass.range_constexpr(2):
                    tma_reduce_add_subtile(staged, tma_slice_runtime_desc(dq_desc, chunk * CFG.DQ_CHUNK, qtile * CFG.N, head, 0), sub, acquire=False)
                tma_store_commit()
                tma_store_wait(0)
                bars.dq_empty[full_state.idx].arrive()
                full_state = advance(full_state, CFG.DQ_STAGES)
        _wait(bars.dv_full, out_state.phase)
        if block * CFG.M < skv:
            tma_store_tile(_operand(slab, CFG.K, dtype, CFG.M, CFG.DV), tma_slice_runtime_desc(dv_desc, 0, block * CFG.M, head, 0))
        _wait(bars.dk_full, out_state.phase)
        if block * CFG.M < skv:
            tma_store_tile(_operand(slab, CFG.K + CFG.M * CFG.DV * CFG.BPE, dtype, CFG.M, CFG.DQK), tma_slice_runtime_desc(dk_desc, 0, block * CFG.M, head, 0))
        out_state = advance(out_state, 1)
        tma_store_commit()
        tma_store_wait(0)
        bars.dvk_empty.arrive()
        bars.k_empty.arrive()
        block, head, batch, valid, sched_state = _next_tile(sched, bars, sched_state)


@cute.jit
def _map_ptr(maps, batch, n_batch, slot: cutlass.Constexpr[int]):
    return (maps.iterator.raw_ptr() + (cutlass.Int64(slot) * n_batch + batch) * TENSOR_MAP_QWORDS).tospace(cutlass.AddressSpace.generic)


@cute.kernel
def bprop_d192_accum_thd_kernel(
    maps: cute.Tensor,
    lse: cute.Tensor,
    delta: cute.Tensor,
    meta: cute.Tensor,
    origins: cute.Tensor,
    n_batch: cutlass.Int32,
    stats_origin: cutlass.Constexpr[int],
    scale: cutlass.Float32,
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
):
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    x, y, z = cute.arch.block_idx()
    block, head, batch = x, y, z
    slab = cutlass.Array(cutlass.Uint8, CFG.SMEM_BYTES, alignment=1024, space=cutlass.AddressSpace.smem)
    bars = _bars(slab)
    tmem_slot = _view(slab, CFG.TMEM_HANDLE, cutlass.Int32, 1, alignment=4)
    sched_words = _view(slab, CFG.SCHED_PAYLOAD, cutlass.Int32, 16, alignment=16)
    sched = Sched(bars.sched_full.base_ptr, bars.sched_empty.base_ptr, sched_words, x, y, z)
    if warp == 0:
        # All 512 TMEM columns belong to this CTA; use the portable allocation form.
        nvvm.tcgen05_alloc(tmem_slot, cutlass.Int32(CFG.TMEM_COLS), group=nvvm.CTAGroup.CTA_1)
        nvvm.tcgen05_relinquish_alloc_permit(group=nvvm.CTAGroup.CTA_1)
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
        if nvvm.elect_sync():
            for i in cutlass.range_constexpr(len(bars)):
                for stage in cutlass.range_constexpr(bars[i].stages):
                    bars[i][stage].init()
    nvvm.fence_mbarrier_init()
    nvvm.barrier_cta_sync()
    tmem_base = tmem_slot.load()
    # setmaxnreg is warpgroup-aligned: the four service warps must release
    # registers at the same instruction before their individual role branches.
    if warp >= 12:
        _service_regs()
    if warp < 8:
        _softmax_loop(slab, bars, sched, tmem_base, lse, delta, origins, stats_origin, block, head, batch, meta, n_batch, scale, causal, dtype)
    elif warp < 12:
        _epilogue_loop(slab, bars, sched, tmem_base, block, head, batch, meta, n_batch, causal, dtype)
    elif warp == 12:
        _mma_loop(slab, bars, sched, tmem_base, block, head, batch, meta, n_batch, causal, dtype)
    elif warp == 13:
        _load_loop(slab, bars, sched, maps, block, head, batch, meta, n_batch, causal, dtype)
    elif warp == 14:
        _store_loop(slab, bars, sched, maps, block, head, batch, meta, n_batch, causal, dtype)
    else:
        _scheduler_loop(bars, sched)
    nvvm.barrier_cta_sync()
    if warp == 0:
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
        nvvm.tcgen05_dealloc(nvvm.make_tmem_ptr(tmem_slot.load(), cutlass.Int8), cutlass.Int32(CFG.TMEM_COLS), group=nvvm.CTAGroup.CTA_1)


bprop_d192_accum_thd_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def host(
    maps,
    lse,
    delta,
    meta,
    origins,
    n_batch,
    nheads,
    max_skv,
    stats_origin: cutlass.Constexpr[int],
    scale,
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
    stream,
):
    """Launch the native THD body on a host-declared KV envelope, never live totals."""
    nk = cute.math.max((max_skv + CFG.M - 1) // CFG.M, cutlass.Int32(1))
    bprop_d192_accum_thd_kernel(maps, lse, delta, meta, origins, n_batch, stats_origin, scale, causal, dtype).launch(
        grid=(nk, nheads, n_batch), block=(CFG.THREADS, 1, 1), stream=stream, min_blocks_per_mp=1
    )
