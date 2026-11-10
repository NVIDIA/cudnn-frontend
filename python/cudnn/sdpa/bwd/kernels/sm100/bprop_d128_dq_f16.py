# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Dense d64 / d128 dQ with native stores: the query-owner half of the deterministic two-kernel backward.

One CTA owns a 128-row query tile and walks its key tiles once: S = Q K^T and dP = dO V^T are recomputed, dS is
packed into the dP TMEM plane, and dQ accumulates in TMEM until the epilogue converts and stores its rows. Nothing
reduces through memory, so the summation order is fixed. The S plane is released as soon as its scores are in
registers, and K keeps three stages so the next S never waits behind the dQ step that frees a stage.

Sequence lengths need not be tile multiples: the tail Q / K / V / dO boxes zero-fill, the tail dQ rows are not
stored, the tail query rows publish -inf and the tail key columns are masked before the exponent.
"""

from typing import NamedTuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import arith
from cutlass.base_dsl.typing import Pointer
from cutlass.experimental import primitives as nvvm
from cutlass.experimental.cuda import tensor_map as tmap

from cudnn.frost.tile_dsl.barrier import MBarrier, PipelineState, Producer, advance
from cudnn.frost.tile_dsl.handles import GmemTileTma, MmaDesc, SmemTile
from cudnn.frost.tile_dsl.mma import mma_ss, mma_ts_step
from cudnn.frost.tile_dsl.pointwise import ffma2, fmul2, fp32_to_fp16
from cudnn.frost.tile_dsl.scheduler import Sched, read_clc_payload
from cudnn.frost.tile_dsl.tma import tma_load_tile

from cudnn.sdpa.bwd.config_sm100_d128_dq import make_bar_layout, make_config

HEAD_DIM = globals().get("FROST_TEMPLATE_PARAMS", 128)
CFG = make_config(HEAD_DIM)
BAR_LAYOUT = make_bar_layout(CFG)
LOG2E = 1.4426950408889634


class Bars(NamedTuple):
    q_empty: MBarrier
    q_full: MBarrier
    do_empty: MBarrier
    do_full: MBarrier
    k_empty: MBarrier
    k_full: MBarrier
    v_empty: MBarrier
    v_full: MBarrier
    s_done: MBarrier
    dp_done: MBarrier
    dq_done: MBarrier
    s_free: MBarrier
    ds_ready: MBarrier
    dq_read: MBarrier
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


def _operand(slab, offset, dtype, *, transpose=False):
    return SmemTile(
        base=_view(slab, offset, dtype, CFG.M * CFG.D, alignment=1024),
        elems_per_stage=CFG.M * CFG.D,
        leading_byte_offset=CFG.M * CFG.SWIZZLE_BYTES if transpose else 16,
        stride_byte_offset=8 * CFG.SWIZZLE_BYTES,
        layout=CFG.SWIZZLE_LAYOUT,
        tma_loads_per_tile=CFG.D // CFG.HEAD_BOX,
        tma_granu_elems=CFG.HEAD_BOX,
        tma_subtile_stride_elems=CFG.M * CFG.HEAD_BOX,
        desc_version=CFG.DESC_VERSION,
    )


def _mma_desc(dtype, *, btranspose=False, N=CFG.N, K=CFG.D):
    return MmaDesc(
        M=CFG.M,
        N=N,
        K=K,
        bpe_a=CFG.BPE,
        bpe_b=CFG.BPE,
        tile_k_hw=CFG.K_HW,
        atranspose=False,
        btranspose=btranspose,
        cta_group=1,
        idesc=nvvm.Tcgen05InstrDesc.build(c_dtype=cutlass.Float32, a_dtype=dtype, b_dtype=dtype, m_dim=CFG.M, n_dim=N, a_major=0, b_major=int(btranspose)),
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
def _decode_flat(tile, nheads, nq, swizzle: cutlass.Constexpr[int], batch_heads: cutlass.Constexpr[int]):
    group = tile // (swizzle * nq)
    inner = tile % (swizzle * nq)
    block = cutlass.Int32(0)
    residual = cutlass.Int32(0)
    if group < batch_heads // swizzle:
        block = inner // swizzle
        residual = inner % swizzle
    else:
        block = inner // max(batch_heads % swizzle, 1)
        residual = inner % max(batch_heads % swizzle, 1)
    hb = group * swizzle + residual
    return block, hb % nheads, hb // nheads


@cute.jit
def _next_tile(sched, bars, state, nheads, nq, causal: cutlass.Constexpr[bool], swizzle: cutlass.Constexpr[int], batch_heads: cutlass.Constexpr[int]):
    _wait(bars.sched_full[state.idx], state.phase)
    x, yz, valid = read_clc_payload(sched, state.idx * 8, warp_broadcast=True)
    if cutlass.const_expr(causal):
        block, head, batch = _decode_flat(x, nheads, nq, swizzle, batch_heads)
        qtile = nq - 1 - block
    else:
        qtile = x
        head = yz & 0xFFFF
        batch = (yz >> 16) & 0xFFFF
    return qtile, head, batch, valid, advance(state, CFG.SCHED_STAGES)


def _tensor_map(tensor, box):
    return tmap.create_tensor_map_tiled_from_view(tensor, box_dims=(1, CFG.M, 1, box), stride_order=(3, 1, 2, 0), swizzle=tmap.TensorMapSwizzle.s128b)


@cute.jit
def _kv_range(qtile, skv, causal: cutlass.Constexpr[bool], window: cutlass.Constexpr[int] = 0):
    lo = cutlass.Int32(0)
    if cutlass.const_expr(window > 0):
        lo = cute.math.max(cutlass.Int32(0), (qtile * CFG.M - (window - 1)) // CFG.N)
    hi = (skv + CFG.N - 1) // CFG.N
    if cutlass.const_expr(causal):
        hi = cute.math.min(hi, (qtile * CFG.M + CFG.M - 1) // CFG.N + 1)
    return lo, hi


@cute.jit
def _row_stats(lse, delta, batch, head, q, sq, scale):
    # Tail rows past S_q read the last row (in bounds) and publish -inf: their zero-filled Q / dO yield exp2(-inf) = 0.
    valid = q < sq
    clamped = cutlass.Int32(arith.select(valid.ir_value(), q.ir_value(), (sq - 1).ir_value()))
    val = cutlass.Float32(lse[batch, head, clamped])
    stat = -cutlass.Float32(LOG2E) * val
    if val == cutlass.Float32(float("-inf")):
        stat = cutlass.Float32(0.0)
    stat = cutlass.Float32(arith.select(valid.ir_value(), stat.ir_value(), cutlass.Float32(float("-inf")).ir_value()))
    dstat = -scale * cutlass.Float32(delta[batch, head, clamped])
    return stat, dstat


@cute.jit
def _exp_pairs(scores, stat, scaled):
    p = []
    for pair in cutlass.range_constexpr(32):
        e0, e1 = ffma2(scaled, scaled, scores[2 * pair], scores[2 * pair + 1], stat, stat)
        p.append(cute.math.exp2(e0, fastmath=True))
        p.append(cute.math.exp2(e1, fastmath=True))
    return cutlass.Vector.from_elements(tuple(p), cutlass.Float32)


@cute.jit
def _masked_scores(scores, group, q, ktile, skv, causal: cutlass.Constexpr[bool], window: cutlass.Constexpr[int]):
    # Every enabled select is applied; on a cell the select cannot mask it keeps the score, so the body is only
    # entered on a tile that can hold a masked cell.
    neg_inf = cutlass.Float32(float("-inf"))
    out = []
    for i in cutlass.range_constexpr(64):
        v = scores[i]
        kc = ktile * CFG.N + group * 64 + i
        if cutlass.const_expr(causal):
            v = cutlass.Float32(arith.select((kc <= q).ir_value(), v.ir_value(), neg_inf.ir_value()))
        if cutlass.const_expr(window > 0):
            v = cutlass.Float32(arith.select((kc + (window - 1) >= q).ir_value(), v.ir_value(), neg_inf.ir_value()))
        v = cutlass.Float32(arith.select((kc < skv).ir_value(), v.ir_value(), neg_inf.ir_value()))
        out.append(v)
    return cutlass.Vector.from_elements(tuple(out), cutlass.Float32)


@cute.jit
def _scores(tmem_base, group):
    address = tmem_base + CFG.TMEM_S + group * 64
    a = _tmem_load(address, 32)
    b = _tmem_load(address + 32, 32)
    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
    return cutlass.Vector.from_elements(tuple(a[i] for i in range(32)) + tuple(b[i] for i in range(32)), cutlass.Float32)


@cute.jit
def _p_values(scores, group, q, qtile, ktile, skv, stat, scale, causal: cutlass.Constexpr[bool], window: cutlass.Constexpr[int] = 0):
    scaled = scale * cutlass.Float32(LOG2E)
    # Only a tile that can hold a masked cell pays the selects: the key tail (zero-filled key rows score 0, which
    # would exponentiate to a live probability), the causal diagonal, the window's left edge.
    masked = (ktile + 1) * CFG.N > skv
    if cutlass.const_expr(causal):
        masked = masked | ((ktile + 1) * CFG.N - 1 > qtile * CFG.M)
    if cutlass.const_expr(window > 0):
        masked = masked | (ktile * CFG.N < qtile * CFG.M + CFG.M - window)
    p = cutlass.Vector.from_elements(tuple(cutlass.Float32(0.0) for _ in range(64)), cutlass.Float32)
    if masked:
        p = _exp_pairs(_masked_scores(scores, group, q, ktile, skv, causal, window), stat, scaled)
    else:
        p = _exp_pairs(scores, stat, scaled)
    return p


@cute.jit
def _ds_values(tmem_base, group, p, dstat, scale, dtype: cutlass.Constexpr):
    address = tmem_base + CFG.TMEM_DP + group * 64
    a = _tmem_load(address, 32)
    b = _tmem_load(address + 32, 32)
    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
    dp = cutlass.Vector.from_elements(tuple(a[i] for i in range(32)) + tuple(b[i] for i in range(32)), cutlass.Float32)
    packed = []
    for pair in cutlass.range_constexpr(32):
        d0, d1 = ffma2(scale, scale, dp[2 * pair], dp[2 * pair + 1], dstat, dstat)
        d0, d1 = fmul2(d0, d1, p[2 * pair], p[2 * pair + 1])
        packed.append(fp32_to_fp16(d0, d1, dtype=dtype))
    _tmem_store(address, cutlass.Vector.from_elements(tuple(packed), cutlass.Int32))


@cute.jit
def _softmax_loop(
    slab,
    bars,
    sched,
    tmem_base,
    lse,
    delta,
    qtile,
    head,
    batch,
    sq,
    skv,
    nq,
    nheads,
    batch_heads: cutlass.Constexpr[int],
    swizzle: cutlass.Constexpr[int],
    scale,
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
    window: cutlass.Constexpr[int] = 0,
):
    nvvm.setmaxregister(CFG.SOFTMAX_REGS, nvvm.SetMaxRegisterAction.INCREASE)
    tid = cute.arch.thread_idx()[0]
    row = tid % CFG.M
    group = tid // CFG.M
    phase = cutlass.Int32(0)
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit(bars.sched_empty[sched_state.idx])
        lo, hi = _kv_range(qtile, skv, causal, window)
        q = qtile * CFG.M + row
        stat, dstat = _row_stats(lse, delta, batch, head, q, sq, scale)
        for ktile in cutlass.range(lo, hi, 1, unroll=1):
            _wait_tcgen(bars.s_done, phase)
            scores = _scores(tmem_base, group)
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
            bars.s_free.arrive()
            p = _p_values(scores, group, q, qtile, ktile, skv, stat, scale, causal, window)
            _wait_tcgen(bars.dp_done, phase)
            _ds_values(tmem_base, group, p, dstat, scale, dtype)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
            bars.ds_ready.arrive()
            phase = phase ^ 1
        qtile, head, batch, valid, sched_state = _next_tile(sched, bars, sched_state, nheads, nq, causal, swizzle, batch_heads)


@cute.jit
def _store_row(dq, batch, head, q, half: cutlass.Constexpr[int], words):
    base = (
        cutlass.Int64(batch) * cutlass.Int64(dq.stride[0])
        + cutlass.Int64(q) * cutlass.Int64(dq.stride[1])
        + cutlass.Int64(head) * cutlass.Int64(dq.stride[2])
        + half * 64
    )
    for vec in cutlass.range_constexpr(8):
        values = cutlass.Vector.from_elements(tuple(words[vec * 4 + k] for k in range(4)), cutlass.Int32)
        Pointer(cutlass.make_array_view(dq).data_ptr() + base + vec * 8, dtype=cutlass.Int32).store(values, alignment=16)


@cute.jit
def _epilogue_loop(
    bars,
    sched,
    tmem_base,
    dq,
    qtile,
    head,
    batch,
    sq,
    skv,
    nq,
    nheads,
    batch_heads: cutlass.Constexpr[int],
    swizzle: cutlass.Constexpr[int],
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
    window: cutlass.Constexpr[int] = 0,
):
    nvvm.setmaxregister(CFG.EPILOGUE_REGS, nvvm.SetMaxRegisterAction.INCREASE)
    row = cute.arch.thread_idx()[0] % CFG.M
    done_phase = cutlass.Int32(0)
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit(bars.sched_empty[sched_state.idx])
        lo, hi = _kv_range(qtile, skv, causal, window)
        live = lo < hi
        q = qtile * CFG.M + row
        _wait_tcgen(bars.dq_done, done_phase)
        done_phase = done_phase ^ 1
        for half in cutlass.range_constexpr(CFG.D // 64):
            vals = cutlass.Vector.from_elements(tuple(cutlass.Float32(0.0) for _ in range(64)), cutlass.Float32)
            if live:
                vals = _tmem_load(tmem_base + CFG.TMEM_DQ + half * 64, 64)
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
            words = cutlass.Vector.from_elements(tuple(fp32_to_fp16(vals[2 * i], vals[2 * i + 1], dtype=dtype) for i in range(32)), cutlass.Int32)
            if q < sq:
                _store_row(dq, batch, head, q, half, words)
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
        bars.dq_read.arrive()
        qtile, head, batch, valid, sched_state = _next_tile(sched, bars, sched_state, nheads, nq, causal, swizzle, batch_heads)


@cute.jit
def _ts(desc, plane, bdesc, dst, accum):
    for k in cutlass.range_constexpr(CFG.N // CFG.K_HW):
        mma_ts_step(
            desc, nvvm.make_tmem_ptr(plane + (k // 4) * 32, cutlass.Int8), bdesc, dst, k_idx=k, accumulate=accum if cutlass.const_expr(k == 0) else True
        )


@cute.jit
def _mma_loop(
    slab,
    bars,
    sched,
    tmem_base,
    qtile,
    head,
    batch,
    skv,
    nq,
    nheads,
    batch_heads: cutlass.Constexpr[int],
    swizzle: cutlass.Constexpr[int],
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
    window: cutlass.Constexpr[int] = 0,
):
    nvvm.setmaxregister(CFG.SERVICE_REGS, nvvm.SetMaxRegisterAction.DECREASE)
    score = _mma_desc(dtype)
    dqdesc = _mma_desc(dtype, btranspose=True, N=CFG.D, K=CFG.N)
    qd = _operand(slab, CFG.Q, dtype).desc()
    dod = _operand(slab, CFG.DO, dtype).desc()
    s = nvvm.make_tmem_ptr(tmem_base + CFG.TMEM_S, cutlass.Float32)
    dp = nvvm.make_tmem_ptr(tmem_base + CFG.TMEM_DP, cutlass.Float32)
    dq = nvvm.make_tmem_ptr(tmem_base + CFG.TMEM_DQ, cutlass.Float32)
    k_state = PipelineState.start()
    v_state = PipelineState.start()
    free_state = PipelineState.start()
    ready_state = PipelineState.start()
    read_state = PipelineState.start(phase=1)
    io_phase = cutlass.Int32(0)
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit(bars.sched_empty[sched_state.idx])
        lo, hi = _kv_range(qtile, skv, causal, window)
        if lo < hi:
            _wait_tcgen(bars.q_full, io_phase)
            _wait_tcgen(bars.k_full[k_state.idx], k_state.phase)
            mma_ss(score, qd, _operand(slab, CFG.K + k_state.idx * CFG.OPERAND_BYTES, dtype).desc(), s)
            _commit(bars.s_done)
            if hi - lo == 1:
                _commit(bars.q_empty)
            _wait_tcgen(bars.do_full, io_phase)
            io_phase = io_phase ^ 1
        for ktile in cutlass.range(lo, hi, 1, unroll=1):
            _wait_tcgen(bars.v_full[v_state.idx], v_state.phase)
            mma_ss(score, dod, _operand(slab, CFG.V + v_state.idx * CFG.OPERAND_BYTES, dtype).desc(), dp)
            _commit(bars.dp_done)
            _commit(bars.v_empty[v_state.idx])
            if ktile + 1 == hi:
                _commit(bars.do_empty)
            k_next = advance(k_state, CFG.K_STAGES)
            _wait_tcgen(bars.s_free, free_state.phase)
            free_state = advance(free_state, 1)
            if ktile + 1 < hi:
                _wait_tcgen(bars.k_full[k_next.idx], k_next.phase)
                mma_ss(score, qd, _operand(slab, CFG.K + k_next.idx * CFG.OPERAND_BYTES, dtype).desc(), s)
                _commit(bars.s_done)
                if ktile + 2 == hi:
                    _commit(bars.q_empty)
            _wait_tcgen(bars.ds_ready, ready_state.phase)
            ready_state = advance(ready_state, 1)
            if ktile == lo:
                _wait_tcgen(bars.dq_read, read_state.phase)
                read_state = advance(read_state, 1)
            kt = _operand(slab, CFG.K + k_state.idx * CFG.OPERAND_BYTES, dtype, transpose=True).desc()
            _ts(dqdesc, tmem_base + CFG.TMEM_DP, kt, dq, ktile > lo)
            _commit(bars.k_empty[k_state.idx])
            k_state = k_next
            v_state = advance(v_state, CFG.V_STAGES)
        if lo >= hi:
            _wait_tcgen(bars.dq_read, read_state.phase)
            read_state = advance(read_state, 1)
        _commit(bars.dq_done)
        qtile, head, batch, valid, sched_state = _next_tile(sched, bars, sched_state, nheads, nq, causal, swizzle, batch_heads)


@cute.jit
def _load_loop(
    slab,
    bars,
    sched,
    q_desc,
    k_desc,
    v_desc,
    do_desc,
    qtile,
    head,
    batch,
    skv,
    nq,
    nheads,
    batch_heads: cutlass.Constexpr[int],
    swizzle: cutlass.Constexpr[int],
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
    window: cutlass.Constexpr[int] = 0,
    ratio: cutlass.Constexpr[int] = 1,
):
    nvvm.setmaxregister(CFG.SERVICE_REGS, nvvm.SetMaxRegisterAction.DECREASE)
    k_state = PipelineState.start(phase=1)
    v_state = PipelineState.start(phase=1)
    io_phase = cutlass.Int32(1)
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit(bars.sched_empty[sched_state.idx])
        lo, hi = _kv_range(qtile, skv, causal, window)
        kv_head = head // ratio
        if lo < hi:
            _wait(bars.q_empty, io_phase)
            bars.q_full.arrive(n_bytes=CFG.OPERAND_BYTES, pred=nvvm.elect_sync())
            tma_load_tile(_operand(slab, CFG.Q, dtype), GmemTileTma(q_desc)(0, qtile * CFG.M, head, batch), bars.q_full.smem_ptr)
            _wait(bars.do_empty, io_phase)
            bars.do_full.arrive(n_bytes=CFG.OPERAND_BYTES, pred=nvvm.elect_sync())
            tma_load_tile(_operand(slab, CFG.DO, dtype), GmemTileTma(do_desc)(0, qtile * CFG.M, head, batch), bars.do_full.smem_ptr)
            io_phase = io_phase ^ 1
        for ktile in cutlass.range(lo, hi, 1, unroll=1):
            _wait(bars.k_empty[k_state.idx], k_state.phase)
            bars.k_full[k_state.idx].arrive(n_bytes=CFG.OPERAND_BYTES, pred=nvvm.elect_sync())
            tma_load_tile(
                _operand(slab, CFG.K + k_state.idx * CFG.OPERAND_BYTES, dtype),
                GmemTileTma(k_desc)(0, ktile * CFG.N, kv_head, batch),
                bars.k_full[k_state.idx].smem_ptr,
            )
            k_state = advance(k_state, CFG.K_STAGES)
            _wait(bars.v_empty[v_state.idx], v_state.phase)
            bars.v_full[v_state.idx].arrive(n_bytes=CFG.OPERAND_BYTES, pred=nvvm.elect_sync())
            tma_load_tile(
                _operand(slab, CFG.V + v_state.idx * CFG.OPERAND_BYTES, dtype),
                GmemTileTma(v_desc)(0, ktile * CFG.N, kv_head, batch),
                bars.v_full[v_state.idx].smem_ptr,
            )
            v_state = advance(v_state, CFG.V_STAGES)
        qtile, head, batch, valid, sched_state = _next_tile(sched, bars, sched_state, nheads, nq, causal, swizzle, batch_heads)


@cute.jit
def _follow_loop(bars, sched, nheads, nq, causal: cutlass.Constexpr[bool], swizzle: cutlass.Constexpr[int], batch_heads: cutlass.Constexpr[int]):
    """A spare service warp: it only keeps the scheduler credits balanced."""
    nvvm.setmaxregister(CFG.SERVICE_REGS, nvvm.SetMaxRegisterAction.DECREASE)
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit(bars.sched_empty[sched_state.idx])
        _, _, _, valid, sched_state = _next_tile(sched, bars, sched_state, nheads, nq, causal, swizzle, batch_heads)


@cute.jit
def _scheduler_loop(bars, sched):
    nvvm.setmaxregister(CFG.SERVICE_REGS, nvvm.SetMaxRegisterAction.DECREASE)
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
def bprop_d128_dq_kernel(
    q_desc: cutlass.GridConstant[tmap.TensorMap],
    k_desc: cutlass.GridConstant[tmap.TensorMap],
    v_desc: cutlass.GridConstant[tmap.TensorMap],
    do_desc: cutlass.GridConstant[tmap.TensorMap],
    dq: cute.Tensor,
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
    window: cutlass.Constexpr[int] = 0,
    ratio: cutlass.Constexpr[int] = 1,
):
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    x, y, z = cute.arch.block_idx()
    nq = (sq + CFG.M - 1) // CFG.M
    if cutlass.const_expr(causal):
        block, head, batch = _decode_flat(x, nheads, nq, swizzle, batch_heads)
        qtile = nq - 1 - block
    else:
        qtile, head, batch = x, y, z
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
    if warp < 8:
        _softmax_loop(slab, bars, sched, tmem_base, lse, delta, qtile, head, batch, sq, skv, nq, nheads, batch_heads, swizzle, scale, causal, dtype, window)
    elif warp < 12:
        _epilogue_loop(bars, sched, tmem_base, dq, qtile, head, batch, sq, skv, nq, nheads, batch_heads, swizzle, causal, dtype, window)
    elif warp == 12:
        _mma_loop(slab, bars, sched, tmem_base, qtile, head, batch, skv, nq, nheads, batch_heads, swizzle, causal, dtype, window)
    elif warp == 13:
        _load_loop(slab, bars, sched, q_desc, k_desc, v_desc, do_desc, qtile, head, batch, skv, nq, nheads, batch_heads, swizzle, causal, dtype, window, ratio)
    elif warp == 14:
        _follow_loop(bars, sched, nheads, nq, causal, swizzle, batch_heads)
    else:
        _scheduler_loop(bars, sched)
    nvvm.barrier_cta_sync()
    if warp == 0:
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
        nvvm.tcgen05_dealloc(nvvm.make_tmem_ptr(tmem_slot.load(), cutlass.Int8), cutlass.Int32(CFG.TMEM_COLS), group=nvvm.CTAGroup.CTA_1)


bprop_d128_dq_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def host(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    do: cute.Tensor,
    lse: cute.Tensor,
    delta: cute.Tensor,
    dq: cute.Tensor,
    scale: cutlass.Float32,
    causal: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
    window: cutlass.Constexpr[int] = 0,
    ratio: cutlass.Constexpr[int] = 1,
):
    if cutlass.const_expr(q.element_type not in (cutlass.BFloat16, cutlass.Float16)):
        raise ValueError("dense d128 dQ requires BF16/FP16")
    if cutlass.const_expr(q.shape[3] != CFG.D or k.shape[3] != CFG.D or v.shape[3] != CFG.D or dq.shape[3] != CFG.D):
        raise ValueError("dense d128 dQ requires the configured head dimension")
    if cutlass.const_expr(q.shape[2] != k.shape[2] * ratio):
        raise ValueError("dense d128 dQ requires H_q = ratio * H_kv")
    if cutlass.const_expr(q.shape[1] <= 0 or k.shape[1] <= 0):
        raise ValueError("dense d128 dQ requires positive Sq/Skv")
    nq = (q.shape[1] + CFG.M - 1) // CFG.M
    heads_per_l2 = (50 * 1024 * 1024) // (k.shape[1] * (CFG.D + CFG.D) * CFG.BPE)
    swizzle = 1 << (heads_per_l2.bit_length() - 1) if heads_per_l2 > 0 else 1
    grid = (nq * q.shape[2] * q.shape[0], 1, 1) if causal else (nq, q.shape[2], q.shape[0])
    bprop_d128_dq_kernel(
        _tensor_map(q, CFG.HEAD_BOX),
        _tensor_map(k, CFG.HEAD_BOX),
        _tensor_map(v, CFG.HEAD_BOX),
        _tensor_map(do, CFG.HEAD_BOX),
        dq,
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
        window,
        ratio,
    ).launch(grid=grid, block=(CFG.THREADS, 1, 1), stream=stream, min_blocks_per_mp=1)
