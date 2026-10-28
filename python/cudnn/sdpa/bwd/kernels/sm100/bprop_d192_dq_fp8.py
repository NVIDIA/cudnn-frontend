# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Dense per-tensor FP8 dQ for D_QK in {128, 192} with D_V = 128: the query-owner half of the two-kernel backward."""

from typing import NamedTuple, Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import arith
from cutlass.experimental import primitives as nvvm
from cutlass.experimental.cuda import tensor_map as tmap

from cudnn.frost.tile_dsl.barrier import MBarrier, PipelineState, Producer, advance
from cudnn.frost.tile_dsl.handles import GmemTileTma
from cudnn.frost.tile_dsl.mma import mma_ss
from cudnn.frost.tile_dsl.pointwise import abs_max_tree, amax_to_ue8m0_rp, exp2_emul_pair, ffma2, fmax_f32, fmul2, fp32_to_fp16, opaque_f32_zero
from cudnn.frost.tile_dsl.scheduler import Sched, read_clc_payload
from cudnn.frost.tile_dsl.tma import tma_load_tile, tma_store_commit, tma_store_tile, tma_store_wait

from cudnn.sdpa.bwd.config_sm100_d192_dq_fp8 import make_config
from cudnn.sdpa.bwd.kernels.sm100.bprop_d192_dkdv_fp8 import (
    FP8,
    HALF_DTYPES,
    LOG2E,
    _bpe,
    _commit,
    _credit,
    _decode_flat,
    _fold_amax,
    _mma_desc,
    _operand,
    _pack_fp8,
    _scalar,
    _scale_pairs,
    _service_regs,
    _sf_copy,
    _sf_map,
    _sf_ptr,
    _sf_tile,
    _store_chunks,
    _tensor_map,
    _tmem_load,
    _view,
    _wait,
    _wait_tcgen,
)

_PARAMS = globals().get("FROST_TEMPLATE_PARAMS", (192, 128))
MX = bool(_PARAMS[3]) if len(_PARAMS) > 3 else False
WINDOW = int(_PARAMS[4]) if len(_PARAMS) > 4 else 0
CFG, BAR_LAYOUT = make_config(int(_PARAMS[0]), int(_PARAMS[1]), MX)
TS = bool(CFG.T_SHARED)
OUT_SPLIT = bool(CFG.OUT_SPLIT)
EXP2_FMA_SPLIT = bool(_PARAMS[2]) if len(_PARAMS) > 2 else False


def _emulated(pair: int) -> bool:
    return EXP2_FMA_SPLIT and pair % 8 >= 6


class Bars(NamedTuple):
    q_empty: MBarrier
    q_full: MBarrier
    do_empty: MBarrier
    do_full: MBarrier
    k_empty: MBarrier
    k_full: MBarrier
    kt_empty: MBarrier
    kt_full: MBarrier
    k_used: MBarrier
    v_empty: MBarrier
    v_full: MBarrier
    s_done: MBarrier
    dp_done: MBarrier
    dq_done: MBarrier
    s_free: MBarrier
    dp_empty: MBarrier
    ds_full: MBarrier
    ds_empty: MBarrier
    dq_read: MBarrier
    out_full: MBarrier
    out_empty: MBarrier
    sched_full: MBarrier
    sched_empty: MBarrier


def _bars(slab):
    kinds = {"mma": Producer.MMA_COMMIT, "tma": Producer.TMA_LOAD, "thread": Producer.THREAD}
    return Bars(
        *(
            MBarrier(_view(slab, off, cutlass.Int64, slots), stages=slots, init_count=count, producer=kinds[kind], try_wait=True)
            for _, off, slots, count, kind in BAR_LAYOUT
        )
    )


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


@cute.jit
def _kv_range(qtile, skv, causal: cutlass.Constexpr[bool]):
    lo = cutlass.Int32(0)
    if cutlass.const_expr(WINDOW > 0):
        lo = cute.math.max(cutlass.Int32(0), (qtile * CFG.M - (WINDOW - 1)) // CFG.N)
    hi = (skv + CFG.N - 1) // CFG.N
    if cutlass.const_expr(causal):
        hi = cute.math.min(hi, qtile + 1)
    return lo, hi


@cute.jit
def _row_stats(lse, delta, batch, head, q, sq, delta_scale, lse_shift):
    valid = q < sq
    clamped = cutlass.Int32(arith.select(valid.ir_value(), q.ir_value(), (sq - 1).ir_value()))
    val = cutlass.Float32(lse[batch, head, clamped])
    stat = -cutlass.Float32(LOG2E) * val + lse_shift
    if val == cutlass.Float32(float("-inf")):
        stat = lse_shift
    stat = cutlass.Float32(arith.select(valid.ir_value(), stat.ir_value(), cutlass.Float32(float("-inf")).ir_value()))
    dstat = -delta_scale * cutlass.Float32(delta[batch, head, clamped])
    return stat, dstat


@cute.jit
def _masked_scores(
    scores, q, group, ktile, skv, causal_mask: cutlass.Constexpr[bool], key_mask: cutlass.Constexpr[bool], window_mask: cutlass.Constexpr[bool] = False
):
    neg_inf = cutlass.Float32(float("-inf"))
    out = []
    for i in cutlass.range_constexpr(64):
        v = scores[i]
        kc = ktile * CFG.N + group * 64 + i
        if cutlass.const_expr(causal_mask):
            v = cutlass.Float32(arith.select((kc <= q).ir_value(), v.ir_value(), neg_inf.ir_value()))
        if cutlass.const_expr(window_mask):
            v = cutlass.Float32(arith.select((kc + (WINDOW - 1) >= q).ir_value(), v.ir_value(), neg_inf.ir_value()))
        if cutlass.const_expr(key_mask):
            v = cutlass.Float32(arith.select((kc < skv).ir_value(), v.ir_value(), neg_inf.ir_value()))
        out.append(v)
    return cutlass.Vector.from_elements(tuple(out), cutlass.Float32)


@cute.jit
def _softmax_tile(
    slab,
    bars,
    tmem_base,
    q,
    group,
    skv,
    ktile,
    stat,
    dstat,
    s_scale,
    dp_scale,
    phase,
    ring_idx,
    ring_phase,
    amax,
    fold: cutlass.Constexpr[bool],
    causal_mask: cutlass.Constexpr[bool],
    key_mask: cutlass.Constexpr[bool],
    window_mask: cutlass.Constexpr[bool] = False,
):
    _wait_tcgen(bars.s_done, phase)
    a = _tmem_load(tmem_base + CFG.TMEM_S + group * 64, 32)
    b = _tmem_load(tmem_base + CFG.TMEM_S + group * 64 + 32, 32)
    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
    nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
    bars.s_free.arrive()
    scores = cutlass.Vector.from_elements(tuple(a[i] for i in range(32)) + tuple(b[i] for i in range(32)), cutlass.Float32)
    if cutlass.const_expr(causal_mask or key_mask or window_mask):
        scores = _masked_scores(scores, q, group, ktile, skv, causal_mask, key_mask, window_mask)
    scaled = s_scale * cutlass.Float32(LOG2E)
    exponents = []
    for pair in cutlass.range_constexpr(32):
        e0, e1 = ffma2(scaled, scaled, scores[2 * pair], scores[2 * pair + 1], stat, stat)
        exponents.extend((e0, e1))
    p = []
    for pair in cutlass.range_constexpr(32):
        if cutlass.const_expr(_emulated(pair)):
            p.extend(exp2_emul_pair(exponents[2 * pair], exponents[2 * pair + 1]))
        else:
            p.append(cute.math.exp2(exponents[2 * pair], fastmath=True))
            p.append(cute.math.exp2(exponents[2 * pair + 1], fastmath=True))
    _wait_tcgen(bars.dp_done, phase)
    a = _tmem_load(tmem_base + CFG.TMEM_DP + group * 64, 32)
    b = _tmem_load(tmem_base + CFG.TMEM_DP + group * 64 + 32, 32)
    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
    nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
    bars.dp_empty.arrive()
    dp = cutlass.Vector.from_elements(tuple(a[i] for i in range(32)) + tuple(b[i] for i in range(32)), cutlass.Float32)
    ds = []
    for pair in cutlass.range_constexpr(32):
        d0, d1 = ffma2(dp_scale, dp_scale, dp[2 * pair], dp[2 * pair + 1], dstat, dstat)
        d0, d1 = fmul2(d0, d1, p[2 * pair], p[2 * pair + 1])
        ds.extend((d0, d1))
    if cutlass.const_expr(fold):
        amax = fmax_f32(amax, abs_max_tree(ds))
    ds_sf = cutlass.Int32(0)
    if cutlass.const_expr(MX):
        e0, inv0 = amax_to_ue8m0_rp(abs_max_tree(ds[:32]))
        e1, inv1 = amax_to_ue8m0_rp(abs_max_tree(ds[32:]))
        ds_words = _pack_fp8(_scale_pairs(ds[:32], inv0) + _scale_pairs(ds[32:], inv1))
        ds_sf = cutlass.Int32(e0) | (cutlass.Int32(e1) << 8)
    else:
        ds_words = _pack_fp8(ds)
    _wait(bars.ds_empty[ring_idx], ring_phase ^ 1)
    row = q % CFG.M
    _store_chunks(slab, CFG.DS + ring_idx * CFG.RING_BYTES, row, ds_words, group * 4, 4)
    if cutlass.const_expr(MX):
        _view(slab, CFG.SF_DS + ring_idx * 512 + (row % 32) * 16 + (row // 32) * 4 + group * 2, cutlass.Uint16, 1, alignment=2).subview(0).store(
            cutlass.Uint16(ds_sf)
        )
    nvvm.fence_proxy("async.shared", space="cta")
    bars.ds_full[ring_idx].arrive()
    next_idx = ring_idx + 1
    wrap = next_idx == CFG.DS_STAGES
    ring_idx = cutlass.Int32(arith.select(wrap.ir_value(), cutlass.Int32(0).ir_value(), next_idx.ir_value()))
    ring_phase = cutlass.Int32(arith.select(wrap.ir_value(), (ring_phase ^ 1).ir_value(), ring_phase.ir_value()))
    return phase ^ 1, ring_idx, ring_phase, amax


@cute.jit
def _softmax_loop(
    slab,
    bars,
    sched,
    tmem_base,
    lse,
    delta,
    descale_q,
    descale_k,
    descale_v,
    descale_s,
    descale_do,
    scale_s_t,
    scale_dp_t,
    amax_dp: Optional[cute.Tensor],
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
):
    nvvm.setmaxregister(CFG.SOFTMAX_REGS, nvvm.SetMaxRegisterAction.INCREASE)
    tid = cute.arch.thread_idx()[0]
    row = tid % CFG.M
    group = tid // CFG.M
    if cutlass.const_expr(MX):
        s_scale = scale
        scale_dp = cutlass.Float32(1.0)
        lse_shift = cutlass.Float32(8.0)
        delta_scale = scale * cutlass.Float32(1.0 / 256.0)
        dp_scale = delta_scale
    else:
        s_scale = scale * _scalar(descale_q) * _scalar(descale_k)
        scale_s = _scalar(scale_s_t)
        scale_dp = _scalar(scale_dp_t)
        lse_shift = cute.math.log2(scale_s)
        delta_scale = scale * _scalar(descale_s) * scale_dp
        dp_scale = delta_scale * _scalar(descale_v) * _scalar(descale_do)
    fold = cutlass.const_expr(amax_dp is not None)
    amax = opaque_f32_zero()
    phase = cutlass.Int32(0)
    ring_idx = cutlass.Int32(0)
    ring_phase = cutlass.Int32(0)
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit(bars.sched_empty[sched_state.idx])
        q = qtile * CFG.M + row
        lo, hi = _kv_range(qtile, skv, causal)
        stat, dstat = _row_stats(lse, delta, batch, head, q, sq, delta_scale, lse_shift)
        edge = lo
        if cutlass.const_expr(WINDOW > 0):
            edge = cute.math.max(lo, cute.math.min(hi - 1, (qtile * CFG.M + CFG.M - WINDOW + CFG.N - 1) // CFG.N))
            for ktile in cutlass.range(lo, edge, 1, unroll=1):
                phase, ring_idx, ring_phase, amax = _softmax_tile(
                    slab, bars, tmem_base, q, group, skv, ktile, stat, dstat, s_scale, dp_scale, phase, ring_idx, ring_phase, amax, fold, False, False, True
                )
        for ktile in cutlass.range(edge, hi - 1, 1, unroll=1):
            phase, ring_idx, ring_phase, amax = _softmax_tile(
                slab, bars, tmem_base, q, group, skv, ktile, stat, dstat, s_scale, dp_scale, phase, ring_idx, ring_phase, amax, fold, False, False
            )
        for ktile in cutlass.range(hi - 1, hi, 1, unroll=1):
            phase, ring_idx, ring_phase, amax = _softmax_tile(
                slab, bars, tmem_base, q, group, skv, ktile, stat, dstat, s_scale, dp_scale, phase, ring_idx, ring_phase, amax, fold, causal, True, WINDOW > 0
            )
        qtile, head, batch, valid, sched_state = _next_tile(sched, bars, sched_state, nheads, nq, causal, swizzle, batch_heads)
    if cutlass.const_expr(fold):
        _fold_amax(amax * (cutlass.Float32(1.0) / scale_dp), amax_dp)


@cute.jit
def _epilogue_loop(
    slab,
    bars,
    sched,
    tmem_base,
    descale_k,
    descale_dp,
    scale_dq_t,
    amax_dq: Optional[cute.Tensor],
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
    grad_dtype: cutlass.Constexpr,
):
    nvvm.setmaxregister(CFG.EPILOGUE_REGS, nvvm.SetMaxRegisterAction.DECREASE)
    row = cute.arch.thread_idx()[0] % CFG.M
    fp8_out = cutlass.const_expr(grad_dtype == FP8)
    split = cutlass.const_expr(OUT_SPLIT and not fp8_out)
    factor = opaque_f32_zero() + cutlass.Float32(1.0) if MX else _scalar(descale_dp) * _scalar(descale_k)
    out_scale = cutlass.Float32(1.0)
    if cutlass.const_expr(fp8_out):
        out_scale = _scalar(scale_dq_t)
    fold = cutlass.const_expr(amax_dq is not None)
    amax = opaque_f32_zero()
    done_phase = cutlass.Int32(0)
    empty_phase = cutlass.Int32(1)
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit(bars.sched_empty[sched_state.idx])
        _wait_tcgen(bars.dq_done, done_phase)
        done_phase = done_phase ^ 1
        _wait(bars.out_empty, empty_phase)
        empty_phase = empty_phase ^ 1
        for third in cutlass.range_constexpr(CFG.DQK // 64):
            vals = _tmem_load(tmem_base + CFG.TMEM_DQ + third * 64, 64)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
            if cutlass.const_expr(third == CFG.DQK // 64 - 1):
                nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
                bars.dq_read.arrive()
            if cutlass.const_expr(split and third == 2):
                nvvm.fence_proxy("async.shared", space="cta")
                bars.out_full.arrive()
                _wait(bars.out_empty, empty_phase)
                empty_phase = empty_phase ^ 1
            scaled = _scale_pairs([vals[i] for i in range(64)], factor)
            if cutlass.const_expr(fold):
                amax = fmax_f32(amax, abs_max_tree(scaled))
            if cutlass.const_expr(fp8_out and CFG.QK_SWZ == CFG.NARROW_SWIZZLE_BYTES):
                _store_chunks(
                    slab, CFG.OUT + third * CFG.M * CFG.NARROW_SWIZZLE_BYTES, row, _pack_fp8(_scale_pairs(scaled, out_scale)), 0, 4, CFG.NARROW_SWIZZLE_BYTES
                )
            elif cutlass.const_expr(fp8_out):
                _store_chunks(slab, CFG.OUT + (third // 2) * CFG.M * CFG.SWIZZLE_BYTES, row, _pack_fp8(_scale_pairs(scaled, out_scale)), (third % 2) * 4, 4)
            else:
                words = cutlass.Vector.from_elements(tuple(fp32_to_fp16(scaled[2 * i], scaled[2 * i + 1], dtype=grad_dtype) for i in range(32)), cutlass.Int32)
                _store_chunks(slab, CFG.OUT + (third % (2 if split else 3)) * CFG.M * CFG.SWIZZLE_BYTES, row, words, 0, 8)
        nvvm.fence_proxy("async.shared", space="cta")
        bars.out_full.arrive()
        qtile, head, batch, valid, sched_state = _next_tile(sched, bars, sched_state, nheads, nq, causal, swizzle, batch_heads)
    if cutlass.const_expr(fold):
        _fold_amax(amax, amax_dq)


@cute.jit
def _k_desc(slab, stage):
    return _operand(slab, CFG.K + stage * CFG.K_STAGE_BYTES, FP8, CFG.N, CFG.DQK, swz=CFG.QK_SWZ).desc()


@cute.jit
def _k_scales(slab, tmem_base, idx):
    _sf_copy(tmem_base, CFG.TMEM_SF_K + idx * CFG.SF_COLS_QK, _sf_tile(slab, CFG.SF_K + idx * CFG.SF_QK_BYTES, CFG.SF_QK_BYTES).desc(), CFG.SF_COLS_QK // 4)


@cute.jit
def _kt_scales(slab, tmem_base, idx):
    _sf_copy(tmem_base, CFG.TMEM_SF_KT + idx * CFG.SF_COLS_QK, _sf_tile(slab, CFG.SF_KT + idx * CFG.SF_KT_BYTES, CFG.SF_KT_BYTES).desc(), CFG.SF_COLS_QK // 4)


@cute.jit
def _v_desc(slab, stage):
    return _operand(slab, CFG.V + stage * CFG.V_BYTES, FP8, CFG.N, CFG.DV, swz=CFG.V_SWZ).desc()


@cute.jit
def _mma_loop(
    slab,
    bars,
    sched,
    tmem_base,
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
):
    score = _mma_desc(CFG.M, CFG.N, CFG.DQK, mx=MX)
    dpdesc = _mma_desc(CFG.M, CFG.N, CFG.DV, mx=MX)
    dqdesc = _mma_desc(CFG.M, CFG.DQK, CFG.N, btranspose=True, mx=MX)
    sf_q = _sf_ptr(tmem_base, CFG.TMEM_SF_Q) if MX else None
    sf_do = _sf_ptr(tmem_base, CFG.TMEM_SF_DO) if MX else None
    qd = _operand(slab, CFG.Q, FP8, CFG.M, CFG.DQK, swz=CFG.QK_SWZ).desc()
    dod = _operand(slab, CFG.DO, FP8, CFG.M, CFG.DV, swz=CFG.V_SWZ).desc()
    s = nvvm.make_tmem_ptr(tmem_base + CFG.TMEM_S, cutlass.Float32)
    dp = nvvm.make_tmem_ptr(tmem_base + CFG.TMEM_DP, cutlass.Float32)
    dq = nvvm.make_tmem_ptr(tmem_base + CFG.TMEM_DQ, cutlass.Float32)
    k_state = PipelineState.start()
    kt_state = PipelineState.start()
    v_state = PipelineState.start()
    ring = PipelineState.start()
    sfree_state = PipelineState.start()
    dpempty_state = PipelineState.start()
    read_state = PipelineState.start(phase=1)
    io_phase = cutlass.Int32(0)
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit(bars.sched_empty[sched_state.idx])
        lo, hi = _kv_range(qtile, skv, causal)
        _wait_tcgen(bars.q_full, io_phase)
        _wait_tcgen(bars.do_full, io_phase)
        if cutlass.const_expr(MX):
            _sf_copy(tmem_base, CFG.TMEM_SF_Q, _sf_tile(slab, CFG.SF_Q, CFG.SF_QK_BYTES).desc(), CFG.SF_COLS_QK // 4)
            _sf_copy(tmem_base, CFG.TMEM_SF_DO, _sf_tile(slab, CFG.SF_DO, CFG.SF_V_BYTES).desc(), 1)
        io_phase = io_phase ^ 1
        _wait_tcgen(bars.k_full[k_state.idx], k_state.phase)
        if cutlass.const_expr(MX):
            _k_scales(slab, tmem_base, k_state.idx)
        mma_ss(
            score, qd, _k_desc(slab, k_state.idx), s, tmem_sf_a=sf_q, tmem_sf_b=_sf_ptr(tmem_base, CFG.TMEM_SF_K + k_state.idx * CFG.SF_COLS_QK) if MX else None
        )
        _commit(bars.s_done)
        if cutlass.const_expr(TS):
            _commit(bars.k_used[k_state.idx])
        elif cutlass.const_expr(MX):
            _commit(bars.k_empty[k_state.idx])
        if hi - lo == 1:
            _commit(bars.q_empty)
        _wait_tcgen(bars.v_full[v_state.idx], v_state.phase)
        if cutlass.const_expr(MX):
            _sf_copy(tmem_base, CFG.TMEM_SF_V + v_state.idx * CFG.SF_COLS_V, _sf_tile(slab, CFG.SF_V + v_state.idx * CFG.SF_V_BYTES, CFG.SF_V_BYTES).desc(), 1)
        mma_ss(
            dpdesc,
            dod,
            _v_desc(slab, v_state.idx),
            dp,
            tmem_sf_a=sf_do,
            tmem_sf_b=_sf_ptr(tmem_base, CFG.TMEM_SF_V + v_state.idx * CFG.SF_COLS_V) if MX else None,
        )
        _commit(bars.dp_done)
        _commit(bars.v_empty[v_state.idx])
        if hi - lo == 1:
            _commit(bars.do_empty)
        for ktile in cutlass.range(lo, hi, 1, unroll=1):
            k_next = advance(k_state, CFG.K_STAGES)
            v_next = advance(v_state, CFG.V_STAGES)
            _wait_tcgen(bars.s_free, sfree_state.phase)
            sfree_state = advance(sfree_state, 1)
            if ktile + 1 < hi:
                _wait_tcgen(bars.k_full[k_next.idx], k_next.phase)
                if cutlass.const_expr(MX):
                    _k_scales(slab, tmem_base, k_next.idx)
                mma_ss(
                    score,
                    qd,
                    _k_desc(slab, k_next.idx),
                    s,
                    tmem_sf_a=sf_q,
                    tmem_sf_b=_sf_ptr(tmem_base, CFG.TMEM_SF_K + k_next.idx * CFG.SF_COLS_QK) if MX else None,
                )
                _commit(bars.s_done)
                if cutlass.const_expr(TS):
                    _commit(bars.k_used[k_next.idx])
                elif cutlass.const_expr(MX):
                    _commit(bars.k_empty[k_next.idx])
                if ktile + 2 == hi:
                    _commit(bars.q_empty)
            _wait_tcgen(bars.dp_empty, dpempty_state.phase)
            dpempty_state = advance(dpempty_state, 1)
            if ktile + 1 < hi:
                _wait_tcgen(bars.v_full[v_next.idx], v_next.phase)
                if cutlass.const_expr(MX):
                    _sf_copy(
                        tmem_base, CFG.TMEM_SF_V + v_next.idx * CFG.SF_COLS_V, _sf_tile(slab, CFG.SF_V + v_next.idx * CFG.SF_V_BYTES, CFG.SF_V_BYTES).desc(), 1
                    )
                mma_ss(
                    dpdesc,
                    dod,
                    _v_desc(slab, v_next.idx),
                    dp,
                    tmem_sf_a=sf_do,
                    tmem_sf_b=_sf_ptr(tmem_base, CFG.TMEM_SF_V + v_next.idx * CFG.SF_COLS_V) if MX else None,
                )
                _commit(bars.dp_done)
                _commit(bars.v_empty[v_next.idx])
                if ktile + 2 == hi:
                    _commit(bars.do_empty)
            _wait_tcgen(bars.ds_full[ring.idx], ring.phase)
            if ktile == lo:
                _wait_tcgen(bars.dq_read, read_state.phase)
                read_state = advance(read_state, 1)
            if cutlass.const_expr(TS):
                _wait_tcgen(bars.kt_full[k_state.idx], k_state.phase)
                _kt_scales(slab, tmem_base, k_state.idx)
                _sf_copy(tmem_base, CFG.TMEM_SF_DS + ring.idx * CFG.SF_COLS_V, _sf_tile(slab, CFG.SF_DS + ring.idx * 512, 512).desc(), 1)
                kt = _operand(slab, CFG.K + k_state.idx * CFG.K_STAGE_BYTES, FP8, CFG.N, CFG.DQK, transpose=True, swz=CFG.QK_SWZ).desc()
            elif cutlass.const_expr(MX):
                _wait_tcgen(bars.kt_full[kt_state.idx], kt_state.phase)
                _kt_scales(slab, tmem_base, kt_state.idx)
                _sf_copy(tmem_base, CFG.TMEM_SF_DS + ring.idx * CFG.SF_COLS_V, _sf_tile(slab, CFG.SF_DS + ring.idx * 512, 512).desc(), 1)
                kt = _operand(slab, CFG.KT + kt_state.idx * CFG.K_BYTES, FP8, CFG.N, CFG.DQK, transpose=True, swz=CFG.QK_SWZ).desc()
            else:
                kt = _operand(slab, CFG.K + k_state.idx * CFG.K_STAGE_BYTES, FP8, CFG.N, CFG.DQK, transpose=True, swz=CFG.QK_SWZ).desc()
            sf_ds = _sf_ptr(tmem_base, CFG.TMEM_SF_DS + ring.idx * CFG.SF_COLS_V) if MX else None
            sf_kt = _sf_ptr(tmem_base, CFG.TMEM_SF_KT + (k_state.idx if TS else kt_state.idx) * CFG.SF_COLS_QK) if MX else None
            mma_ss(
                dqdesc,
                _operand(slab, CFG.DS + ring.idx * CFG.RING_BYTES, FP8, CFG.M, CFG.N).desc(),
                kt,
                dq,
                tmem_sf_a=sf_ds,
                tmem_sf_b=sf_kt,
                accumulate=ktile > lo,
            )
            _commit(bars.ds_empty[ring.idx])
            if cutlass.const_expr(MX and not TS):
                _commit(bars.kt_empty[kt_state.idx])
                kt_state = advance(kt_state, CFG.KT_STAGES)
            else:
                _commit(bars.k_empty[k_state.idx])
            k_state = k_next
            v_state = v_next
            ring = advance(ring, CFG.DS_STAGES)
        _commit(bars.dq_done)
        qtile, head, batch, valid, sched_state = _next_tile(sched, bars, sched_state, nheads, nq, causal, swizzle, batch_heads)


@cute.jit
def _load_sf_t(slab, offset, desc, tile, plane, bar):
    for g in cutlass.range_constexpr(CFG.SF_COLS_QK // 4):
        tma_load_tile(_sf_tile(slab, offset + g * 512, 512), GmemTileTma(desc)(0, 0, tile, plane, g), bar)


@cute.jit
def _load_kt(slab, bars, kt_desc, sf_kt_desc, k_idx, k_phase, ktile, khead, batch, plane):
    _wait(bars.k_used[k_idx], k_phase ^ 1)
    bars.kt_full[k_idx].arrive(n_bytes=CFG.KT_TX, pred=nvvm.elect_sync())
    tma_load_tile(
        _operand(slab, CFG.K + k_idx * CFG.K_STAGE_BYTES, FP8, CFG.N, CFG.DQK, swz=CFG.QK_SWZ),
        GmemTileTma(kt_desc)(0, ktile * CFG.N, khead, batch),
        bars.kt_full[k_idx].smem_ptr,
    )
    _load_sf_t(slab, CFG.SF_KT + k_idx * CFG.SF_KT_BYTES, sf_kt_desc, ktile, plane, bars.kt_full[k_idx].smem_ptr)


@cute.jit
def _load_loop(
    slab,
    bars,
    sched,
    q_desc,
    k_desc,
    v_desc,
    do_desc,
    kt_desc,
    sf_q_desc,
    sf_do_desc,
    sf_k_desc,
    sf_kt_desc,
    sf_v_desc,
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
    ratio: cutlass.Constexpr[int] = 1,
):
    k_state = PipelineState.start(phase=1)
    kt_state = PipelineState.start(phase=1)
    v_state = PipelineState.start(phase=1)
    io_phase = cutlass.Int32(1)
    sched_state = PipelineState.start()
    pending = cutlass.Int32(0)
    pk_idx, pk_phase, pktile = cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0)
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit(bars.sched_empty[sched_state.idx])
        lo, hi = _kv_range(qtile, skv, causal)
        plane = batch * (nheads // ratio) + head // ratio
        _wait(bars.q_empty, io_phase)
        bars.q_full.arrive(n_bytes=CFG.Q_TX, pred=nvvm.elect_sync())
        tma_load_tile(_operand(slab, CFG.Q, FP8, CFG.M, CFG.DQK, swz=CFG.QK_SWZ), GmemTileTma(q_desc)(0, qtile * CFG.M, head, batch), bars.q_full.smem_ptr)
        if cutlass.const_expr(MX):
            tma_load_tile(_sf_tile(slab, CFG.SF_Q, CFG.SF_QK_BYTES), GmemTileTma(sf_q_desc)(0, 0, qtile, head, batch), bars.q_full.smem_ptr)
        _wait(bars.do_empty, io_phase)
        bars.do_full.arrive(n_bytes=CFG.DO_TX, pred=nvvm.elect_sync())
        tma_load_tile(_operand(slab, CFG.DO, FP8, CFG.M, CFG.DV, swz=CFG.V_SWZ), GmemTileTma(do_desc)(0, qtile * CFG.M, head, batch), bars.do_full.smem_ptr)
        if cutlass.const_expr(MX):
            tma_load_tile(_sf_tile(slab, CFG.SF_DO, CFG.SF_V_BYTES), GmemTileTma(sf_do_desc)(0, 0, qtile, head, batch), bars.do_full.smem_ptr)
        io_phase = io_phase ^ 1
        for ktile in cutlass.range(lo, hi, 1, unroll=1):
            _wait(bars.k_empty[k_state.idx], k_state.phase)
            bars.k_full[k_state.idx].arrive(n_bytes=CFG.K_TX, pred=nvvm.elect_sync())
            k_base = CFG.K + k_state.idx * CFG.K_STAGE_BYTES
            k_prev = k_state
            tma_load_tile(
                _operand(slab, k_base, FP8, CFG.N, CFG.DQK, swz=CFG.QK_SWZ),
                GmemTileTma(k_desc)(0, ktile * CFG.N, head // ratio, batch),
                bars.k_full[k_state.idx].smem_ptr,
            )
            if cutlass.const_expr(MX):
                tma_load_tile(
                    _sf_tile(slab, CFG.SF_K + k_state.idx * CFG.SF_QK_BYTES, CFG.SF_QK_BYTES),
                    GmemTileTma(sf_k_desc)(0, 0, ktile, head // ratio, batch),
                    bars.k_full[k_state.idx].smem_ptr,
                )
            k_state = advance(k_state, CFG.K_STAGES)
            _wait(bars.v_empty[v_state.idx], v_state.phase)
            bars.v_full[v_state.idx].arrive(n_bytes=CFG.V_TX, pred=nvvm.elect_sync())
            tma_load_tile(
                _operand(slab, CFG.V + v_state.idx * CFG.V_BYTES, FP8, CFG.N, CFG.DV, swz=CFG.V_SWZ),
                GmemTileTma(v_desc)(0, ktile * CFG.N, head // ratio, batch),
                bars.v_full[v_state.idx].smem_ptr,
            )
            if cutlass.const_expr(MX):
                tma_load_tile(
                    _sf_tile(slab, CFG.SF_V + v_state.idx * CFG.SF_V_BYTES, CFG.SF_V_BYTES),
                    GmemTileTma(sf_v_desc)(0, 0, ktile, head // ratio, batch),
                    bars.v_full[v_state.idx].smem_ptr,
                )
            v_state = advance(v_state, CFG.V_STAGES)
            if cutlass.const_expr(TS):
                if pending != 0:
                    _load_kt(slab, bars, kt_desc, sf_kt_desc, pk_idx, pk_phase, pktile, head // ratio, batch, plane)
                pending = cutlass.Int32(1)
                pk_idx, pk_phase, pktile = k_prev.idx, k_prev.phase, ktile
            elif cutlass.const_expr(MX):
                _wait(bars.kt_empty[kt_state.idx], kt_state.phase)
                bars.kt_full[kt_state.idx].arrive(n_bytes=CFG.KT_TX, pred=nvvm.elect_sync())
                tma_load_tile(
                    _operand(slab, CFG.KT + kt_state.idx * CFG.K_BYTES, FP8, CFG.N, CFG.DQK, swz=CFG.QK_SWZ),
                    GmemTileTma(kt_desc)(0, ktile * CFG.N, head // ratio, batch),
                    bars.kt_full[kt_state.idx].smem_ptr,
                )
                _load_sf_t(slab, CFG.SF_KT + kt_state.idx * CFG.SF_KT_BYTES, sf_kt_desc, ktile, plane, bars.kt_full[kt_state.idx].smem_ptr)
                kt_state = advance(kt_state, CFG.KT_STAGES)
        if cutlass.const_expr(TS):
            if pending != 0:
                _load_kt(slab, bars, kt_desc, sf_kt_desc, pk_idx, pk_phase, pktile, head // ratio, batch, plane)
            pending = cutlass.Int32(0)
        qtile, head, batch, valid, sched_state = _next_tile(sched, bars, sched_state, nheads, nq, causal, swizzle, batch_heads)


@cute.jit
def _store_loop(
    slab,
    bars,
    sched,
    dq_desc,
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
    grad_dtype: cutlass.Constexpr,
):
    out_swz = CFG.QK_SWZ if cutlass.const_expr(grad_dtype == FP8) else CFG.SWIZZLE_BYTES
    split = cutlass.const_expr(OUT_SPLIT and grad_dtype != FP8)
    full_phase = cutlass.Int32(0)
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit(bars.sched_empty[sched_state.idx])
        _wait(bars.out_full, full_phase)
        full_phase = full_phase ^ 1
        if cutlass.const_expr(split):
            tma_store_tile(_operand(slab, CFG.OUT, grad_dtype, CFG.M, 128, swz=out_swz), GmemTileTma(dq_desc)(0, qtile * CFG.M, head, batch))
            tma_store_commit()
            tma_store_wait(0)
            bars.out_empty.arrive()
            _wait(bars.out_full, full_phase)
            full_phase = full_phase ^ 1
            tma_store_tile(_operand(slab, CFG.OUT, grad_dtype, CFG.M, CFG.DQK - 128, swz=out_swz), GmemTileTma(dq_desc)(128, qtile * CFG.M, head, batch))
        else:
            tma_store_tile(_operand(slab, CFG.OUT, grad_dtype, CFG.M, CFG.DQK, swz=out_swz), GmemTileTma(dq_desc)(0, qtile * CFG.M, head, batch))
        tma_store_commit()
        tma_store_wait(0)
        bars.out_empty.arrive()
        qtile, head, batch, valid, sched_state = _next_tile(sched, bars, sched_state, nheads, nq, causal, swizzle, batch_heads)


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
def bprop_d192_dq_fp8_kernel(
    q_desc: cutlass.GridConstant[tmap.TensorMap],
    k_desc: cutlass.GridConstant[tmap.TensorMap],
    v_desc: cutlass.GridConstant[tmap.TensorMap],
    do_desc: cutlass.GridConstant[tmap.TensorMap],
    dq_desc: cutlass.GridConstant[tmap.TensorMap],
    kt_desc: cutlass.GridConstant[tmap.TensorMap],
    sf_q_desc: cutlass.GridConstant[tmap.TensorMap],
    sf_do_desc: cutlass.GridConstant[tmap.TensorMap],
    sf_k_desc: cutlass.GridConstant[tmap.TensorMap],
    sf_kt_desc: cutlass.GridConstant[tmap.TensorMap],
    sf_v_desc: cutlass.GridConstant[tmap.TensorMap],
    lse: cute.Tensor,
    delta: cute.Tensor,
    descale_q: Optional[cute.Tensor],
    descale_k: Optional[cute.Tensor],
    descale_v: Optional[cute.Tensor],
    descale_s: Optional[cute.Tensor],
    scale_s: Optional[cute.Tensor],
    descale_do: Optional[cute.Tensor],
    descale_dp: Optional[cute.Tensor],
    scale_dq: Optional[cute.Tensor],
    scale_dp: Optional[cute.Tensor],
    amax_dq: Optional[cute.Tensor],
    amax_dp: Optional[cute.Tensor],
    sq: cutlass.Int32,
    skv: cutlass.Int32,
    nheads: cutlass.Int32,
    batch_heads: cutlass.Constexpr[int],
    scale: cutlass.Float32,
    causal: cutlass.Constexpr[bool],
    swizzle: cutlass.Constexpr[int],
    grad_dtype: cutlass.Constexpr,
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
    # All four service warps must execute the same warpgroup-aligned
    # register release before diverging into their individual roles.
    if warp >= 12:
        _service_regs()
    if warp < 8:
        _softmax_loop(
            slab,
            bars,
            sched,
            tmem_base,
            lse,
            delta,
            descale_q,
            descale_k,
            descale_v,
            descale_s,
            descale_do,
            scale_s,
            scale_dp,
            amax_dp,
            qtile,
            head,
            batch,
            sq,
            skv,
            nq,
            nheads,
            batch_heads,
            swizzle,
            scale,
            causal,
        )
    elif warp < 12:
        _epilogue_loop(
            slab,
            bars,
            sched,
            tmem_base,
            descale_k,
            descale_dp,
            scale_dq,
            amax_dq,
            qtile,
            head,
            batch,
            sq,
            skv,
            nq,
            nheads,
            batch_heads,
            swizzle,
            causal,
            grad_dtype,
        )
    elif warp == 12:
        _mma_loop(slab, bars, sched, tmem_base, qtile, head, batch, sq, skv, nq, nheads, batch_heads, swizzle, causal)
    elif warp == 13:
        _load_loop(
            slab,
            bars,
            sched,
            q_desc,
            k_desc,
            v_desc,
            do_desc,
            kt_desc,
            sf_q_desc,
            sf_do_desc,
            sf_k_desc,
            sf_kt_desc,
            sf_v_desc,
            qtile,
            head,
            batch,
            sq,
            skv,
            nq,
            nheads,
            batch_heads,
            swizzle,
            causal,
            ratio,
        )
    elif warp == 14:
        _store_loop(slab, bars, sched, dq_desc, qtile, head, batch, sq, skv, nq, nheads, batch_heads, swizzle, causal, grad_dtype)
    else:
        _scheduler_loop(bars, sched)
    nvvm.barrier_cta_sync()
    if warp == 0:
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
        nvvm.tcgen05_dealloc(nvvm.make_tmem_ptr(tmem_slot.load(), cutlass.Int8), cutlass.Int32(CFG.TMEM_COLS), group=nvvm.CTAGroup.CTA_1)


bprop_d192_dq_fp8_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def host(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    do: cute.Tensor,
    lse: cute.Tensor,
    delta: cute.Tensor,
    dq: cute.Tensor,
    descale_q: Optional[cute.Tensor],
    descale_k: Optional[cute.Tensor],
    descale_v: Optional[cute.Tensor],
    descale_s: Optional[cute.Tensor],
    scale_s: Optional[cute.Tensor],
    descale_do: Optional[cute.Tensor],
    descale_dp: Optional[cute.Tensor],
    scale_dq: Optional[cute.Tensor],
    scale_dp: Optional[cute.Tensor],
    amax_dq: Optional[cute.Tensor],
    amax_dp: Optional[cute.Tensor],
    scale: cutlass.Float32,
    causal: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
    ratio: cutlass.Constexpr[int] = 1,
    kt: Optional[cute.Tensor] = None,
    sf_q: Optional[cute.Tensor] = None,
    sf_do: Optional[cute.Tensor] = None,
    sf_k: Optional[cute.Tensor] = None,
    sf_kt: Optional[cute.Tensor] = None,
    sf_v: Optional[cute.Tensor] = None,
):
    if cutlass.const_expr(q.element_type != FP8 or k.element_type != FP8 or v.element_type != FP8 or do.element_type != FP8):
        raise ValueError("dense d192 fp8 dQ requires E4M3 Q/K/V/dO")
    grad_dtype = dq.element_type
    if cutlass.const_expr(grad_dtype not in (FP8,) + HALF_DTYPES):
        raise ValueError("dense d192 fp8 dQ requires an E4M3, BF16 or FP16 dQ")
    if cutlass.const_expr(
        q.shape[3] != CFG.DQK
        or k.shape[3] != CFG.DQK
        or v.shape[3] != CFG.DV
        or do.shape[3] != CFG.DV
        or dq.shape[3] != CFG.DQK
        or q.shape[2] != k.shape[2] * ratio
    ):
        raise ValueError("dense fp8 dQ requires the configured D_QK / D_V with H_q = ratio * H_kv")
    if cutlass.const_expr(q.shape[1] <= 0 or k.shape[1] <= 0):
        raise ValueError("dense d192 fp8 dQ requires positive Sq/Skv")
    nq = (q.shape[1] + CFG.M - 1) // CFG.M
    heads_per_l2 = (50 * 1024 * 1024) // (k.shape[1] * (CFG.DQK + CFG.DV) * CFG.BPE)
    swizzle = 1 << (heads_per_l2.bit_length() - 1) if heads_per_l2 > 0 else 1
    grid = (nq * q.shape[2] * q.shape[0], 1, 1) if causal else (nq, q.shape[2], q.shape[0])
    k_map = _tensor_map(k, CFG.N, CFG.QK_BOX, CFG.QK_SWZ)
    if cutlass.const_expr(MX):
        kt_map = _tensor_map(kt, CFG.N, CFG.QK_BOX, CFG.QK_SWZ)
        sf_maps = (_sf_map(sf_q), _sf_map(sf_do), _sf_map(sf_k), _sf_map(sf_kt), _sf_map(sf_v))
    else:
        kt_map = k_map
        sf_maps = (k_map,) * 5
    bprop_d192_dq_fp8_kernel(
        _tensor_map(q, CFG.M, CFG.QK_BOX, CFG.QK_SWZ),
        k_map,
        _tensor_map(v, CFG.N, CFG.V_BOX, CFG.V_SWZ),
        _tensor_map(do, CFG.M, CFG.V_BOX, CFG.V_SWZ),
        (
            _tensor_map(dq, CFG.M, CFG.QK_BOX, CFG.QK_SWZ)
            if cutlass.const_expr(grad_dtype == FP8)
            else _tensor_map(dq, CFG.M, CFG.SWIZZLE_BYTES // _bpe(grad_dtype))
        ),
        kt_map,
        sf_maps[0],
        sf_maps[1],
        sf_maps[2],
        sf_maps[3],
        sf_maps[4],
        lse,
        delta,
        descale_q,
        descale_k,
        descale_v,
        descale_s,
        scale_s,
        descale_do,
        descale_dp,
        scale_dq,
        scale_dp,
        amax_dq,
        amax_dp,
        q.shape[1],
        k.shape[1],
        q.shape[2],
        q.shape[0] * q.shape[2],
        scale,
        causal,
        swizzle,
        grad_dtype,
        ratio,
    ).launch(grid=grid, block=(CFG.THREADS, 1, 1), stream=stream, min_blocks_per_mp=1)
