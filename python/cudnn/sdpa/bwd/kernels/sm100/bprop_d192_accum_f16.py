# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Dense D_QK = 192 / D_V = 128 half backward with FP32 dQ accumulation and native dK/dV stores."""

from typing import NamedTuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import arith
from cutlass.base_dsl.typing import Pointer
from cutlass.experimental import primitives as nvvm
from cutlass.experimental.cuda import tensor_map as tmap

from cudnn.frost.tile_dsl.barrier import MBarrier, PipelineState, Producer, advance, named_barrier_fence
from cudnn.frost.tile_dsl.handles import GmemTileTma, MmaDesc, SmemTile
from cudnn.frost.tile_dsl.mma import mma_ss, mma_ts_step
from cudnn.frost.tile_dsl.pointwise import ffma2, fmul2, fp32_to_fp16
from cudnn.frost.tile_dsl.scheduler import Sched, read_clc_payload
from cudnn.frost.tile_dsl.tma import st_shared_v4, tma_load_subtiles, tma_load_tile, tma_store_commit, tma_store_tile, tma_store_wait

from cudnn.sdpa.bwd.config_sm100_d192_accum import BAR_LAYOUT, CFG

LOG2E = 1.4426950408889634
DQ_ROWS_PER_WARP = 16


class Bars(NamedTuple):
    k_empty: MBarrier
    k_full: MBarrier
    v_full: MBarrier
    do_empty: MBarrier
    do_full: MBarrier
    q_empty: MBarrier
    q_full: MBarrier
    s_done: MBarrier
    dp_done: MBarrier
    dq_done: MBarrier
    dq_read: MBarrier
    p_ready: MBarrier
    ds_ready: MBarrier
    ds_smem: MBarrier
    dq_empty: MBarrier
    dq_full: MBarrier
    dv_full: MBarrier
    dk_full: MBarrier
    dvk_empty: MBarrier
    sched_full: MBarrier
    sched_empty: MBarrier


def _view(slab, offset, dtype, count, alignment=16):
    ptr = Pointer(slab.data_ptr() + offset, dtype=dtype)
    return cutlass.Array(ptr.ir_value(), shape=count, dtype=dtype, alignment=alignment)


def _bars(slab):
    kinds = {"mma": Producer.MMA_COMMIT, "tma": Producer.TMA_LOAD, "thread": Producer.THREAD}
    return Bars(
        *(
            MBarrier(_view(slab, off, cutlass.Int64, slots), stages=slots, init_count=count, producer=kinds[kind], try_wait=True)
            for _, off, slots, count, kind in BAR_LAYOUT
        )
    )


def _operand(slab, offset, dtype, rows, cols, *, transpose=False):
    bpe = 4 if dtype == cutlass.Float32 else 2
    box = CFG.SWIZZLE_BYTES // bpe
    return SmemTile(
        base=_view(slab, offset, dtype, rows * cols, alignment=1024),
        elems_per_stage=rows * cols,
        leading_byte_offset=rows * CFG.SWIZZLE_BYTES if transpose else 16,
        stride_byte_offset=8 * CFG.SWIZZLE_BYTES,
        layout=CFG.SWIZZLE_LAYOUT,
        tma_loads_per_tile=cols // box,
        tma_granu_elems=box,
        tma_subtile_stride_elems=rows * box,
        desc_version=CFG.DESC_VERSION,
    )


def _mma_desc(dtype, m, n, k, *, atranspose=False, btranspose=False):
    return MmaDesc(
        M=m,
        N=n,
        K=k,
        bpe_a=CFG.BPE,
        bpe_b=CFG.BPE,
        tile_k_hw=CFG.K_HW,
        atranspose=atranspose,
        btranspose=btranspose,
        cta_group=1,
        idesc=nvvm.Tcgen05InstrDesc.build(
            c_dtype=cutlass.Float32, a_dtype=dtype, b_dtype=dtype, m_dim=m, n_dim=n, a_major=int(atranspose), b_major=int(btranspose)
        ),
        kind=nvvm.Tcgen05MMAKind.F16,
    )


@cute.jit
def _wait(bar, phase):
    bar.wait(phase)
    nvvm.bar_warp_sync(cute.arch.FULL_MASK)


@cute.jit
def _wait_tcgen(bar, phase):
    _wait(bar, phase)
    nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)


@cute.jit
def _commit(bar):
    if nvvm.elect_sync():
        bar.arrive(cta_group=1)


@cute.jit
def _credit(bar):
    if nvvm.elect_sync():
        bar.arrive()


@cute.jit
def _tmem_load(address, n: cutlass.Constexpr[int], dtype: cutlass.Constexpr = cutlass.Float32):
    return nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(address, dtype), num=n)


@cute.jit
def _tmem_store(address, values):
    nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(address, cutlass.Int32), values)


@cute.jit
def tma_reduce_subtile(smem_tile, gmem_slice, subtile: cutlass.Constexpr[int]):
    """One elected source-sized FP32 add; caller owns commit and wait/credit."""
    desc = gmem_slice.tma_desc.get_ptr()
    coords = [gmem_slice.coord_d + cutlass.Int32(subtile * smem_tile.tma_granu_elems)] + list(gmem_slice.coords[1:])
    src = smem_tile.base.subview(subtile * smem_tile.tma_subtile_stride_elems)
    if nvvm.elect_sync():
        nvvm.cp_async_bulk_tensor_reduce(desc, src, "add", coords)


@cute.jit
def _decode_flat(tile, nheads, nk, swizzle: cutlass.Constexpr[int], batch_heads: cutlass.Constexpr[int]):
    group = tile // (swizzle * nk)
    inner = tile % (swizzle * nk)
    full = group < batch_heads // swizzle
    divisor = cutlass.Int32(arith.select(full.ir_value(), cutlass.Int32(swizzle).ir_value(), cutlass.Int32(max(batch_heads % swizzle, 1)).ir_value()))
    block = inner // divisor
    residual = inner % divisor
    hb = group * swizzle + residual
    return block, hb % nheads, hb // nheads


@cute.jit
def _next_tile(sched, bars, state, nheads, nk, causal: cutlass.Constexpr[bool], swizzle: cutlass.Constexpr[int], batch_heads: cutlass.Constexpr[int]):
    _wait(bars.sched_full[state.idx], state.phase)
    x, yz, valid = read_clc_payload(sched, state.idx * 8, warp_broadcast=True)
    if cutlass.const_expr(causal):
        block, head, batch = _decode_flat(x, nheads, nk, swizzle, batch_heads)
    else:
        block = x
        head = yz & 0xFFFF
        batch = (yz >> 16) & 0xFFFF
    return block, head, batch, valid, advance(state, CFG.SCHED_STAGES)


def _tensor_map(tensor, rows, box):
    return tmap.create_tensor_map_tiled_from_view(tensor, box_dims=(1, rows, 1, box), stride_order=(3, 1, 2, 0), swizzle=tmap.TensorMapSwizzle.s128b)


@cute.jit
def _bounds(block, sq, causal: cutlass.Constexpr[bool]):
    lo = cutlass.Int32(0)
    if cutlass.const_expr(causal):
        lo = block * (CFG.M // CFG.N)
    return lo, (sq + CFG.N - 1) // CFG.N


@cute.jit
def _stat_load(lse, delta, row, group, qbase, sq, batch, head):
    val = cutlass.Float32(0.0)
    if row < CFG.N:
        idx = qbase + row
        clamped = cutlass.Int32(arith.select((idx < sq).ir_value(), idx.ir_value(), (sq - 1).ir_value()))
        if group == 0:
            val = cutlass.Float32(lse[batch, head, clamped])
        else:
            val = cutlass.Float32(delta[batch, head, clamped])
    return val


@cute.jit
def _stat_publish(slab, row, group, slot, qbase, sq, scale, val):
    if row < CFG.N:
        if group == 0:
            valid = qbase + row < sq
            stat = -cutlass.Float32(LOG2E) * val
            if val == cutlass.Float32(float("-inf")):
                stat = cutlass.Float32(0.0)
            stat = cutlass.Float32(arith.select(valid.ir_value(), stat.ir_value(), cutlass.Float32(float("-inf")).ir_value()))
            _view(slab, CFG.STATS + slot * 256, cutlass.Float32, CFG.N).subview(row).store(stat)
        else:
            _view(slab, CFG.DELTA + slot * 256, cutlass.Float32, CFG.N).subview(row).store(-scale * val)
    named_barrier_fence(CFG.SOFTMAX_BARRIER, CFG.SOFTMAX_LANES)


@cute.jit
def _masked_scores(scores, row, group, block, qtile, skv, causal: cutlass.Constexpr[bool]):
    neg_inf = cutlass.Float32(float("-inf"))
    kr = block * CFG.M + row
    key_valid = kr < skv
    out = []
    for i in cutlass.range_constexpr(32):
        v = scores[i]
        if cutlass.const_expr(causal):
            qc = qtile * CFG.N + group * 32 + i
            v = cutlass.Float32(arith.select((kr <= qc).ir_value(), v.ir_value(), neg_inf.ir_value()))
        v = cutlass.Float32(arith.select(key_valid.ir_value(), v.ir_value(), neg_inf.ir_value()))
        out.append(v)
    return cutlass.Vector.from_elements(tuple(out), cutlass.Float32)


@cute.jit
def _p_exps(slab, scores, slot, group, scale, dtype: cutlass.Constexpr):
    p = []
    packed = []
    scaled = scale * cutlass.Float32(LOG2E)
    for chunk in cutlass.range_constexpr(8):
        stat = _view(slab, CFG.STATS + slot * 256, cutlass.Float32, CFG.N).subview(group * 32 + chunk * 4).load(vector_size=4, alignment=16)
        for pair in cutlass.range_constexpr(2):
            i = chunk * 4 + pair * 2
            e0, e1 = ffma2(scaled, scaled, scores[i], scores[i + 1], stat[pair * 2], stat[pair * 2 + 1])
            p0 = cute.math.exp2(e0, fastmath=True)
            p1 = cute.math.exp2(e1, fastmath=True)
            p.extend((p0, p1))
            packed.append(fp32_to_fp16(p0, p1, dtype=dtype))
    return cutlass.Vector.from_elements(tuple(p), cutlass.Float32), cutlass.Vector.from_elements(tuple(packed), cutlass.Int32)


@cute.jit
def _p_values(slab, tmem_base, row, group, block, qtile, lo, skv, scale, causal: cutlass.Constexpr[bool], dtype: cutlass.Constexpr):
    slot = qtile % 2
    address = tmem_base + CFG.TMEM_S + group * 32
    scores = _tmem_load(address, 32)
    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
    masked = (block + 1) * CFG.M > skv
    if cutlass.const_expr(causal):
        masked = masked | (qtile < lo + CFG.M // CFG.N)
    if masked:
        scores = _masked_scores(scores, row, group, block, qtile, skv, causal)
    p, words = _p_exps(slab, scores, slot, group, scale, dtype)
    _tmem_store(address, words)
    return p


@cute.jit
def _ds_values(slab, tmem_base, group, qtile, p, scale, dtype: cutlass.Constexpr):
    slot = qtile % 2
    address = tmem_base + CFG.TMEM_DP + group * 32
    dp = _tmem_load(address, 32)
    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
    packed = []
    for chunk in cutlass.range_constexpr(8):
        stat = _view(slab, CFG.DELTA + slot * 256, cutlass.Float32, CFG.N).subview(group * 32 + chunk * 4).load(vector_size=4, alignment=16)
        for pair in cutlass.range_constexpr(2):
            i = chunk * 4 + pair * 2
            d0, d1 = ffma2(scale, scale, dp[i], dp[i + 1], stat[pair * 2], stat[pair * 2 + 1])
            d0, d1 = fmul2(d0, d1, p[i], p[i + 1])
            packed.append(fp32_to_fp16(d0, d1, dtype=dtype))
    words = cutlass.Vector.from_elements(tuple(packed), cutlass.Int32)
    _tmem_store(address, words)
    return words


@cute.jit
def _st16(slab, off, values, i: cutlass.Constexpr[int], dtype: cutlass.Constexpr):
    st_shared_v4(_view(slab, off, dtype, 4).data_ptr(), (values[i], values[i + 1], values[i + 2], values[i + 3]), dtype)


@cute.jit
def _store_half_row(slab, base, row, group, words):
    for vec in cutlass.range_constexpr(4):
        chunk = group * 4 + vec
        off = base + row * CFG.SWIZZLE_BYTES + ((row % 8) ^ chunk) * 16
        _st16(slab, off, words, vec * 4, cutlass.Int32)


@cute.jit
def _softmax_loop(
    slab,
    bars,
    sched,
    tmem_base,
    lse,
    delta,
    block,
    head,
    batch,
    sq,
    skv,
    nk,
    nheads,
    batch_heads: cutlass.Constexpr[int],
    swizzle: cutlass.Constexpr[int],
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
        lo, hi = _bounds(block, sq, causal)
        if lo < hi:
            _stat_publish(slab, row, group, lo % 2, lo * CFG.N, sq, scale, _stat_load(lse, delta, row, group, lo * CFG.N, sq, batch, head))
        for qtile in cutlass.range(lo, hi, 1, unroll=1):
            stat_next = _stat_load(lse, delta, row, group, (qtile + 1) * CFG.N, sq, batch, head)
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
        block, head, batch, valid, sched_state = _next_tile(sched, bars, sched_state, nheads, nk, causal, swizzle, batch_heads)


@cute.jit
def _store_row(slab, byte_offset, row, words, chunks: cutlass.Constexpr[int]):
    for vec in cutlass.range_constexpr(chunks):
        off = byte_offset + row * CFG.SWIZZLE_BYTES + ((row % 8) ^ vec) * 16
        _st16(slab, off, words, vec * 4, cutlass.Int32)


@cute.jit
def _store_dq_rows(slab, stage, row, values):
    for sub in cutlass.range_constexpr(2):
        base = CFG.DQ_STAGE + stage * CFG.DQ_STAGE_BYTES + sub * CFG.SLICE_BYTES
        for vec in cutlass.range_constexpr(8):
            off = base + row * CFG.SWIZZLE_BYTES + ((row % 8) ^ vec) * 16
            _st16(slab, off, values, sub * 32 + vec * 4, cutlass.Float32)


@cute.jit
def _native_epilogue(slab, tmem_base, bars, row, live, dtype: cutlass.Constexpr, dk: cutlass.Constexpr[bool]):
    plane = CFG.TMEM_DK if cutlass.const_expr(dk) else CFG.TMEM_DV
    cols = CFG.DQK if cutlass.const_expr(dk) else CFG.DV
    base = CFG.K + (CFG.M * CFG.DV * CFG.BPE if cutlass.const_expr(dk) else 0)
    for half in cutlass.range_constexpr(cols // 64):
        vals = cutlass.Vector.from_elements(tuple(cutlass.Float32(0.0) for _ in range(64)), cutlass.Float32)
        if live:
            vals = _tmem_load(tmem_base + plane + half * 64, 64)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
        words = cutlass.Vector.from_elements(tuple(fp32_to_fp16(vals[2 * i], vals[2 * i + 1], dtype=dtype) for i in range(32)), cutlass.Int32)
        _store_row(slab, base + half * CFG.M * CFG.SWIZZLE_BYTES, row, words, 8)
    nvvm.fence_proxy("async.shared", space="cta")
    nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
    if cutlass.const_expr(dk):
        bars.dk_full.arrive()
    else:
        bars.dv_full.arrive()


@cute.jit
def _epilogue_loop(
    slab,
    bars,
    sched,
    tmem_base,
    block,
    head,
    batch,
    sq,
    nk,
    nheads,
    batch_heads: cutlass.Constexpr[int],
    swizzle: cutlass.Constexpr[int],
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
        lo, hi = _bounds(block, sq, causal)
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
        block, head, batch, valid, sched_state = _next_tile(sched, bars, sched_state, nheads, nk, causal, swizzle, batch_heads)


@cute.jit
def _ts(desc, plane, bdesc, dst, accum):
    for k in cutlass.range_constexpr(CFG.N // CFG.K_HW):
        mma_ts_step(
            desc, nvvm.make_tmem_ptr(plane + (k // 2) * 16, cutlass.Int8), bdesc, dst, k_idx=k, accumulate=accum if cutlass.const_expr(k == 0) else True
        )


@cute.jit
def _service_regs():
    nvvm.setmaxregister(CFG.SERVICE_REGS, nvvm.SetMaxRegisterAction.DECREASE)


@cute.jit
def _mma_loop(
    slab,
    bars,
    sched,
    tmem_base,
    block,
    head,
    batch,
    sq,
    nk,
    nheads,
    batch_heads: cutlass.Constexpr[int],
    swizzle: cutlass.Constexpr[int],
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
        lo, hi = _bounds(block, sq, causal)
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
        block, head, batch, valid, sched_state = _next_tile(sched, bars, sched_state, nheads, nk, causal, swizzle, batch_heads)


@cute.jit
def _load_loop(
    slab,
    bars,
    sched,
    q_desc,
    k_desc,
    v_desc,
    do_desc,
    block,
    head,
    batch,
    sq,
    nk,
    nheads,
    batch_heads: cutlass.Constexpr[int],
    swizzle: cutlass.Constexpr[int],
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
        lo, hi = _bounds(block, sq, causal)
        _wait(bars.k_empty, kv_state.phase)
        kv_state = advance(kv_state, 1)
        if lo < hi:
            bars.k_full.arrive(n_bytes=CFG.K_BYTES, pred=nvvm.elect_sync())
            tma_load_tile(_operand(slab, CFG.K, dtype, CFG.M, CFG.DQK), GmemTileTma(k_desc)(0, block * CFG.M, head, batch), bars.k_full.smem_ptr)
            bars.v_full.arrive(n_bytes=CFG.V_BYTES, pred=nvvm.elect_sync())
            tma_load_tile(_operand(slab, CFG.V, dtype, CFG.M, CFG.DV), GmemTileTma(v_desc)(0, block * CFG.M, head, batch), bars.v_full.smem_ptr)
        else:
            bars.k_full.arrive(n_bytes=0, pred=nvvm.elect_sync())
            bars.v_full.arrive(n_bytes=0, pred=nvvm.elect_sync())
        for qtile in cutlass.range(lo, hi, 1, unroll=1):
            _wait(bars.q_empty[q_state.idx], q_state.phase)
            bars.q_full[q_state.idx].arrive(n_bytes=CFG.Q_BYTES, pred=nvvm.elect_sync())
            tma_load_tile(
                _operand(slab, CFG.Q + q_state.idx * CFG.Q_BYTES, dtype, CFG.N, CFG.DQK),
                GmemTileTma(q_desc)(0, qtile * CFG.N, head, batch),
                bars.q_full[q_state.idx].smem_ptr,
            )
            _wait(bars.do_empty[do_state.idx], do_state.phase)
            for half in cutlass.range_constexpr(2):
                bars.do_full[do_state.idx * 2 + half].arrive(n_bytes=CFG.DO_BYTES // 2, pred=nvvm.elect_sync())
            for half in cutlass.range_constexpr(2):
                tma_load_subtiles(
                    _operand(slab, CFG.DO + do_state.idx * CFG.DO_BYTES, dtype, CFG.N, CFG.DV),
                    GmemTileTma(do_desc)(0, qtile * CFG.N, head, batch),
                    bars.do_full[do_state.idx * 2 + half].smem_ptr,
                    half,
                    1,
                )
            q_state = advance(q_state, CFG.Q_STAGES)
            do_state = advance(do_state, CFG.DO_STAGES)
        block, head, batch, valid, sched_state = _next_tile(sched, bars, sched_state, nheads, nk, causal, swizzle, batch_heads)


@cute.jit
def _store_loop(
    slab,
    bars,
    sched,
    dq_desc,
    dk_desc,
    dv_desc,
    block,
    head,
    batch,
    sq,
    nk,
    nheads,
    batch_heads: cutlass.Constexpr[int],
    swizzle: cutlass.Constexpr[int],
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
):
    full_state = PipelineState.start()
    out_state = PipelineState.start()
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit(bars.sched_empty[sched_state.idx])
        lo, hi = _bounds(block, sq, causal)
        for qtile in cutlass.range(lo, hi, 1, unroll=1):
            for chunk in cutlass.range_constexpr(CFG.DQ_CHUNKS):
                _wait(bars.dq_full[full_state.idx], full_state.phase)
                staged = _operand(slab, CFG.DQ_STAGE + full_state.idx * CFG.DQ_STAGE_BYTES, cutlass.Float32, CFG.N, CFG.DQ_CHUNK)
                for sub in cutlass.range_constexpr(2):
                    tma_reduce_subtile(staged, GmemTileTma(dq_desc)(chunk * CFG.DQ_CHUNK, qtile * CFG.N, head, batch), sub)
                tma_store_commit()
                tma_store_wait(0)
                bars.dq_empty[full_state.idx].arrive()
                full_state = advance(full_state, CFG.DQ_STAGES)
        _wait(bars.dv_full, out_state.phase)
        tma_store_tile(_operand(slab, CFG.K, dtype, CFG.M, CFG.DV), GmemTileTma(dv_desc)(0, block * CFG.M, head, batch))
        _wait(bars.dk_full, out_state.phase)
        tma_store_tile(_operand(slab, CFG.K + CFG.M * CFG.DV * CFG.BPE, dtype, CFG.M, CFG.DQK), GmemTileTma(dk_desc)(0, block * CFG.M, head, batch))
        out_state = advance(out_state, 1)
        tma_store_commit()
        tma_store_wait(0)
        bars.dvk_empty.arrive()
        bars.k_empty.arrive()
        block, head, batch, valid, sched_state = _next_tile(sched, bars, sched_state, nheads, nk, causal, swizzle, batch_heads)


@cute.jit
def _scheduler_loop(bars, sched):
    state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _wait(bars.sched_empty[state.idx], state.phase)
        if nvvm.elect_sync():
            bars.sched_full[state.idx].arrive(n_bytes=16)
            nvvm.clusterlaunchcontrol_try_cancel(sched.tile_id_smem.subview(state.idx * 8), bars.sched_full[state.idx].smem_ptr, multicast=1)
        nvvm.fence_proxy("async.shared", space="cta")
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
        _wait(bars.sched_full[state.idx], state.phase)
        _, _, valid = read_clc_payload(sched, state.idx * 8, warp_broadcast=True)
        state = advance(state, CFG.SCHED_STAGES)


@cute.kernel
def bprop_d192_accum_kernel(
    q_desc: cutlass.GridConstant[tmap.TensorMap],
    k_desc: cutlass.GridConstant[tmap.TensorMap],
    v_desc: cutlass.GridConstant[tmap.TensorMap],
    do_desc: cutlass.GridConstant[tmap.TensorMap],
    dq_desc: cutlass.GridConstant[tmap.TensorMap],
    dk_desc: cutlass.GridConstant[tmap.TensorMap],
    dv_desc: cutlass.GridConstant[tmap.TensorMap],
    lse: cute.Tensor,
    delta: cute.Tensor,
    sq: cutlass.Int32,
    skv: cutlass.Int32,
    nheads: cutlass.Int32,
    batch_heads: cutlass.Constexpr[int],
    scale: cutlass.Float32,
    causal: cutlass.Constexpr[bool],
    swizzle: cutlass.Constexpr[int],
    dtype: cutlass.Constexpr,
):
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    x, y, z = cute.arch.block_idx()
    nk = (skv + CFG.M - 1) // CFG.M
    if cutlass.const_expr(causal):
        block, head, batch = _decode_flat(x, nheads, nk, swizzle, batch_heads)
    else:
        block, head, batch = x, y, z
    slab = cutlass.Array(cutlass.Uint8, CFG.SMEM_BYTES, alignment=1024, space=cutlass.AddressSpace.smem)
    bars = _bars(slab)
    tmem_slot = _view(slab, CFG.TMEM_HANDLE, cutlass.Int32, 1, alignment=4)
    sched_words = _view(slab, CFG.SCHED_PAYLOAD, cutlass.Int32, 16, alignment=16)
    sched = Sched(bars.sched_full.base_ptr, bars.sched_empty.base_ptr, sched_words, x, y, z)
    if warp == 0:
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
        _softmax_loop(slab, bars, sched, tmem_base, lse, delta, block, head, batch, sq, skv, nk, nheads, batch_heads, swizzle, scale, causal, dtype)
    elif warp < 12:
        _epilogue_loop(slab, bars, sched, tmem_base, block, head, batch, sq, nk, nheads, batch_heads, swizzle, causal, dtype)
    elif warp == 12:
        _mma_loop(slab, bars, sched, tmem_base, block, head, batch, sq, nk, nheads, batch_heads, swizzle, causal, dtype)
    elif warp == 13:
        _load_loop(slab, bars, sched, q_desc, k_desc, v_desc, do_desc, block, head, batch, sq, nk, nheads, batch_heads, swizzle, causal, dtype)
    elif warp == 14:
        _store_loop(slab, bars, sched, dq_desc, dk_desc, dv_desc, block, head, batch, sq, nk, nheads, batch_heads, swizzle, causal, dtype)
    else:
        _scheduler_loop(bars, sched)
    nvvm.barrier_cta_sync()
    if warp == 0:
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
        nvvm.tcgen05_dealloc(nvvm.make_tmem_ptr(tmem_slot.load(), cutlass.Int8), cutlass.Int32(CFG.TMEM_COLS), group=nvvm.CTAGroup.CTA_1)


bprop_d192_accum_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def host(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    do: cute.Tensor,
    lse: cute.Tensor,
    delta: cute.Tensor,
    dq: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    scale: cutlass.Float32,
    causal: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    if cutlass.const_expr(q.element_type not in (cutlass.BFloat16, cutlass.Float16)):
        raise ValueError("dense d192 accumulation requires BF16/FP16")
    if cutlass.const_expr(q.shape[3] != CFG.DQK or k.shape[3] != CFG.DQK or v.shape[3] != CFG.DV or do.shape[3] != CFG.DV or q.shape[2] != k.shape[2]):
        raise ValueError("dense d192 accumulation requires dense MHA with D_QK 192 and D_V 128")
    if cutlass.const_expr(q.shape[1] <= 0 or k.shape[1] <= 0):
        raise ValueError("dense d192 accumulation requires positive Sq/Skv")
    nk = (k.shape[1] + CFG.M - 1) // CFG.M
    heads_per_l2 = (50 * 1024 * 1024) // (q.shape[1] * (CFG.DQK + CFG.DV) * CFG.BPE)
    swizzle = 1 << (heads_per_l2.bit_length() - 1) if heads_per_l2 > 0 else 1
    grid = (nk * q.shape[2] * q.shape[0], 1, 1) if causal else (nk, q.shape[2], q.shape[0])
    bprop_d192_accum_kernel(
        _tensor_map(q, CFG.N, CFG.HEAD_BOX),
        _tensor_map(k, CFG.M, CFG.HEAD_BOX),
        _tensor_map(v, CFG.M, CFG.HEAD_BOX),
        _tensor_map(do, CFG.N, CFG.HEAD_BOX),
        _tensor_map(dq, CFG.N, CFG.DQ_BOX),
        _tensor_map(dk, CFG.M, CFG.HEAD_BOX),
        _tensor_map(dv, CFG.M, CFG.HEAD_BOX),
        lse,
        delta,
        q.shape[1],
        k.shape[1],
        q.shape[2],
        q.shape[0] * q.shape[2],
        scale,
        causal,
        swizzle,
        q.element_type,
    ).launch(grid=grid, block=(CFG.THREADS, 1, 1), stream=stream, min_blocks_per_mp=1)
