# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""The deterministic pair's dK/dV owner; dQ is produced by kernel_dq.py.

The worker roles are fixed to the dK/dV pipeline. The width, grouped-head
ratio and q-range split are internal variants of this same implementation.
Configuration-bound device helpers remain local to the rendered module.
"""

from typing import NamedTuple, Optional
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
from cudnn.frost.tile_dsl.tma import tma_load_subtiles, tma_load_tile, tma_store_commit, tma_store_tile, tma_store_wait
from cudnn.sdpa.bwd.config_sm100_d128_accum import make_bar_layout, make_config
from cudnn.sdpa.bwd.kernels.sm100._d128_device_common import (
    _commit,
    _credit_accum,
    _decode_flat,
    _tmem_load,
    _tmem_store,
    _view,
    _wait_accum,
    _wait_tcgen_accum,
)

_PARAMS = globals().get("FROST_TEMPLATE_PARAMS", 128)
HEAD_DIM = int(_PARAMS) if isinstance(_PARAMS, int) else int(_PARAMS[0])
DQ_OUT = False
RATIO = 1 if isinstance(_PARAMS, int) or len(_PARAMS) < 2 else int(_PARAMS[1])
SPLIT = 1 if isinstance(_PARAMS, int) or len(_PARAMS) < 3 else int(_PARAMS[2])
if SPLIT not in (1, 2, 4):
    raise ValueError("the q-range split must be 1, 2 or 4")
CFG_ACCUM = make_config(HEAD_DIM)
BAR_LAYOUT_ACCUM = make_bar_layout(CFG_ACCUM)
LOG2E = 1.4426950408889634
_DO_STEPS = CFG_ACCUM.D // CFG_ACCUM.K_HW // CFG_ACCUM.DO_SUBTILES


class BarsAccum(NamedTuple):
    k_empty: MBarrier
    k_full: MBarrier
    v_empty: MBarrier
    v_full: MBarrier
    do_empty: MBarrier
    do_full: MBarrier
    q_empty: MBarrier
    q_full: MBarrier
    s_done: MBarrier
    dp_done: MBarrier
    dq_done: MBarrier
    dp_empty: MBarrier
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


def _bars_accum(slab):
    kinds = {"mma": Producer.MMA_COMMIT, "tma": Producer.TMA_LOAD, "thread": Producer.THREAD}
    return BarsAccum(
        *(
            MBarrier(_view(slab, off, cutlass.Int64, slots), stages=slots, init_count=count, producer=kinds[kind], try_wait=True)
            for _, off, slots, count, kind in BAR_LAYOUT_ACCUM
        )
    )


def _operand_accum(slab, offset, dtype, *, transpose=False, fp32=False):
    box = CFG_ACCUM.DQ_BOX if fp32 else CFG_ACCUM.HEAD_BOX
    return SmemTile(
        base=_view(slab, offset, dtype, CFG_ACCUM.M * CFG_ACCUM.D, alignment=1024),
        elems_per_stage=CFG_ACCUM.M * CFG_ACCUM.D,
        leading_byte_offset=CFG_ACCUM.M * CFG_ACCUM.SWIZZLE_BYTES if transpose else 16,
        stride_byte_offset=8 * CFG_ACCUM.SWIZZLE_BYTES,
        layout=CFG_ACCUM.SWIZZLE_LAYOUT,
        tma_loads_per_tile=CFG_ACCUM.D // box,
        tma_granu_elems=box,
        tma_subtile_stride_elems=CFG_ACCUM.M * box,
        desc_version=CFG_ACCUM.DESC_VERSION,
    )


def _mma_desc_accum(dtype, *, atranspose=False, btranspose=False, N=CFG_ACCUM.N, K=CFG_ACCUM.D):
    return MmaDesc(
        M=CFG_ACCUM.M,
        N=N,
        K=K,
        bpe_a=CFG_ACCUM.BPE,
        bpe_b=CFG_ACCUM.BPE,
        tile_k_hw=CFG_ACCUM.K_HW,
        atranspose=atranspose,
        btranspose=btranspose,
        cta_group=1,
        idesc=nvvm.Tcgen05InstrDesc.build(
            c_dtype=cutlass.Float32, a_dtype=dtype, b_dtype=dtype, m_dim=CFG_ACCUM.M, n_dim=N, a_major=int(atranspose), b_major=int(btranspose)
        ),
        kind=nvvm.Tcgen05MMAKind.F16,
    )


@cute.jit
def _next_tile_accum(sched, bars, state, nheads, nk, causal: cutlass.Constexpr[bool], swizzle: cutlass.Constexpr[int], batch_heads: cutlass.Constexpr[int]):
    _wait_accum(bars.sched_full[state.idx], state.phase)
    x, yz, valid = read_clc_payload(sched, state.idx * 8, warp_broadcast=True)
    if cutlass.const_expr(causal):
        block, head, batch = _decode_flat(x, nheads, nk, swizzle, batch_heads)
    else:
        block = x
        head = yz & 0xFFFF
        batch = (yz >> 16) & 0xFFFF
    return block, head, batch, valid, advance(state, CFG_ACCUM.SCHED_STAGES)


def _tensor_map_accum(tensor, box):
    # Logical TMA coordinates (d, sequence, head, batch), source-proved order.
    return tmap.create_tensor_map_tiled_from_view(tensor, box_dims=(1, CFG_ACCUM.M, 1, box), stride_order=(3, 1, 2, 0), swizzle=tmap.TensorMapSwizzle.s128b)


@cute.jit
def _bounds_accum(block, sq, causal: cutlass.Constexpr[bool], window: cutlass.Constexpr[int] = 0):
    lo = cutlass.Int32(0)
    if cutlass.const_expr(causal):
        lo = block
    hi = (sq + CFG_ACCUM.N - 1) // CFG_ACCUM.N
    if cutlass.const_expr(window > 0):
        hi = cute.math.min(hi, (block * CFG_ACCUM.M + CFG_ACCUM.M + window - 1 + CFG_ACCUM.N - 1) // CFG_ACCUM.N)
    return lo, hi


@cute.jit
def _stats_accum(slab, lse, delta, row, group, qbase, sq, batch, head, scale):
    # Every one of256 lanes participates in both named barriers. Each group
    # writes a different128-float vector, one row per lane.
    named_barrier_fence(CFG_ACCUM.SOFTMAX_BARRIER, CFG_ACCUM.SOFTMAX_LANES)
    idx = qbase + row
    valid = idx < sq
    if group == 0:
        # Tail rows past S_q read the last Stats row (in bounds) and publish
        # -inf: their zero-filled Q / dO columns then yield exp2(-inf) = 0.
        clamped = cutlass.Int32(arith.select(valid.ir_value(), idx.ir_value(), (sq - 1).ir_value()))
        val = cutlass.Float32(lse[batch, head, clamped])
        stat = -cutlass.Float32(LOG2E) * val
        if val == cutlass.Float32(float("-inf")):
            stat = cutlass.Float32(0.0)
        stat = cutlass.Float32(arith.select(valid.ir_value(), stat.ir_value(), cutlass.Float32(float("-inf")).ir_value()))
        _view(slab, CFG_ACCUM.STATS, cutlass.Float32, CFG_ACCUM.N).subview(row).store(stat)
    else:
        # delta spans the rounded S_q; dot_do_o_kernel zeroes its tail rows.
        stat = -scale * cutlass.Float32(delta[batch, head, idx])
        _view(slab, CFG_ACCUM.DELTA, cutlass.Float32, CFG_ACCUM.N).subview(row).store(stat)
    named_barrier_fence(CFG_ACCUM.SOFTMAX_BARRIER, CFG_ACCUM.SOFTMAX_LANES)


@cute.jit
def _ds_values_accum(slab, tmem_base, group, p, scale, dtype: cutlass.Constexpr):
    address = tmem_base + 128 + group * 64
    a = _tmem_load(address, 32)
    b = _tmem_load(address + 32, 32)
    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
    dp = cutlass.Vector.from_elements(tuple(a[i] for i in range(32)) + tuple(b[i] for i in range(32)), cutlass.Float32)
    packed = []
    for chunk in cutlass.range_constexpr(16):
        stat = _view(slab, CFG_ACCUM.DELTA, cutlass.Float32, CFG_ACCUM.N).subview(group * 64 + chunk * 4).load(vector_size=4, alignment=16)
        for pair in cutlass.range_constexpr(2):
            i = chunk * 4 + pair * 2
            d0, d1 = ffma2(scale, scale, dp[i], dp[i + 1], stat[pair * 2], stat[pair * 2 + 1])
            d0, d1 = fmul2(d0, d1, p[i], p[i + 1])
            packed.append(fp32_to_fp16(d0, d1, dtype=dtype))
    _tmem_store(address, cutlass.Vector.from_elements(tuple(packed), cutlass.Int32))


@cute.jit
def _store_words_accum(slab, byte_offset, row, words):
    for vec in cutlass.range_constexpr(8):
        off = byte_offset + row * CFG_ACCUM.SWIZZLE_BYTES + ((row % 8) ^ vec) * 16
        values = cutlass.Vector.from_elements(tuple(words[vec * 4 + k] for k in range(4)), cutlass.Int32)
        Pointer(slab.data_ptr() + off, dtype=cutlass.Int32).store(values, alignment=16)


@cute.jit
def _store_dq_slice_accum(slab, row, values, index: cutlass.Constexpr[int]):
    for vec in cutlass.range_constexpr(8):
        off = CFG_ACCUM.ALIAS + index * CFG_ACCUM.SLICE_BYTES + row * CFG_ACCUM.SWIZZLE_BYTES + ((row % 8) ^ vec) * 16
        v = cutlass.Vector.from_elements(tuple(values[index * 32 + vec * 4 + k] for k in range(4)), cutlass.Float32)
        Pointer(slab.data_ptr() + off, dtype=cutlass.Float32).store(v, alignment=16)


@cute.jit
def _native_epilogue_accum(slab, tmem_base, bars, row, live, empty_phase, dtype: cutlass.Constexpr, dk: cutlass.Constexpr[bool]):
    plane = 256 + CFG_ACCUM.D if cutlass.const_expr(dk) else 256
    base = CFG_ACCUM.ALIAS + (CFG_ACCUM.OPERAND_BYTES if cutlass.const_expr(dk) else 0)
    last_slice = (CFG_ACCUM.OPERAND_BYTES * (2 if cutlass.const_expr(dk) else 1)) // CFG_ACCUM.SLICE_BYTES - 1
    for half in cutlass.range_constexpr(CFG_ACCUM.D // 64):
        vals = cutlass.Vector.from_elements(tuple(cutlass.Float32(0.0) for _ in range(64)), cutlass.Float32)
        if live:
            vals = _tmem_load(tmem_base + plane + half * 64, 64)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
        words = cutlass.Vector.from_elements(tuple(fp32_to_fp16(vals[2 * i], vals[2 * i + 1], dtype=dtype) for i in range(32)), cutlass.Int32)
        if cutlass.const_expr(half == 0):
            _wait_accum(bars.dq_empty[last_slice], empty_phase)
        _store_words_accum(slab, base + half * CFG_ACCUM.SLICE_BYTES, row, words)
    nvvm.fence_proxy("async.shared", space="cta")
    nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
    if cutlass.const_expr(dk):
        bars.dk_full.arrive()
    else:
        bars.dv_full.arrive()


@cute.jit
def _ts_accum(desc, plane, bdesc, dst, accum):
    for k in cutlass.range_constexpr(CFG_ACCUM.N // CFG_ACCUM.K_HW):
        mma_ts_step(
            desc, nvvm.make_tmem_ptr(plane + (k // 4) * 32, cutlass.Int8), bdesc, dst, k_idx=k, accumulate=accum if cutlass.const_expr(k == 0) else True
        )


@cute.jit
def _scheduler_loop_accum(bars, sched):
    nvvm.setmaxregister(CFG_ACCUM.SERVICE_REGS, nvvm.SetMaxRegisterAction.DECREASE)
    state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _wait_accum(bars.sched_empty[state.idx], state.phase)
        if nvvm.elect_sync():
            bars.sched_full[state.idx].arrive(n_bytes=16)
            nvvm.clusterlaunchcontrol_try_cancel(sched.tile_id_smem.subview(state.idx * 8), bars.sched_full[state.idx].smem_ptr, multicast=1)
        nvvm.fence_proxy("async.shared", space="cta")
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
        _wait_accum(bars.sched_full[state.idx], state.phase)
        _, _, valid = read_clc_payload(sched, state.idx * 8, warp_broadcast=True)
        state = advance(state, CFG_ACCUM.SCHED_STAGES)


@cute.jit
def _pair_exp(scores, slab, group, scale):
    scaled = scale * cutlass.Float32(LOG2E)
    p = []
    for chunk in cutlass.range_constexpr(16):
        stat = _view(slab, CFG_ACCUM.STATS, cutlass.Float32, CFG_ACCUM.N).subview(group * 64 + chunk * 4).load(vector_size=4, alignment=16)
        for pair in cutlass.range_constexpr(2):
            i = chunk * 4 + pair * 2
            e0, e1 = ffma2(scaled, scaled, scores[i], scores[i + 1], stat[pair * 2], stat[pair * 2 + 1])
            p.append(cute.math.exp2(e0, fastmath=True))
            p.append(cute.math.exp2(e1, fastmath=True))
    return cutlass.Vector.from_elements(tuple(p), cutlass.Float32)


@cute.jit
def _pair_masked_scores(scores, row, group, block, qtile, skv, causal: cutlass.Constexpr[bool], window: cutlass.Constexpr[int]):
    # Every enabled select is applied; a cell the select cannot mask keeps its score, so the body is entered only on
    # a tile that can hold a masked cell.
    neg_inf = cutlass.Float32(float("-inf"))
    kr = block * CFG_ACCUM.M + row
    out = []
    for i in cutlass.range_constexpr(64):
        v = scores[i]
        qc = qtile * CFG_ACCUM.N + group * 64 + i
        if cutlass.const_expr(causal):
            v = cutlass.Float32(arith.select((kr <= qc).ir_value(), v.ir_value(), neg_inf.ir_value()))
        if cutlass.const_expr(window > 0):
            v = cutlass.Float32(arith.select((kr + (window - 1) >= qc).ir_value(), v.ir_value(), neg_inf.ir_value()))
        v = cutlass.Float32(arith.select((kr < skv).ir_value(), v.ir_value(), neg_inf.ir_value()))
        out.append(v)
    return cutlass.Vector.from_elements(tuple(out), cutlass.Float32)


@cute.jit
def _pair_p_values(
    slab, tmem_base, row, group, block, qtile, lo, skv, scale, causal: cutlass.Constexpr[bool], dtype: cutlass.Constexpr, window: cutlass.Constexpr[int] = 0
):
    address = tmem_base + group * 64
    a = _tmem_load(address, 32)
    b = _tmem_load(address + 32, 32)
    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
    scores = cutlass.Vector.from_elements(tuple(a[i] for i in range(32)) + tuple(b[i] for i in range(32)), cutlass.Float32)
    masked = (block + 1) * CFG_ACCUM.M > skv
    if cutlass.const_expr(causal):
        masked = masked | (qtile == lo)
    if cutlass.const_expr(window > 0):
        masked = masked | ((qtile + 1) * CFG_ACCUM.N - 1 - block * CFG_ACCUM.M >= window)
    p = cutlass.Vector.from_elements(tuple(cutlass.Float32(0.0) for _ in range(64)), cutlass.Float32)
    if masked:
        p = _pair_exp(_pair_masked_scores(scores, row, group, block, qtile, skv, causal, window), slab, group, scale)
    else:
        p = _pair_exp(scores, slab, group, scale)
    packed = []
    for pair in cutlass.range_constexpr(32):
        packed.append(fp32_to_fp16(p[2 * pair], p[2 * pair + 1], dtype=dtype))
    _tmem_store(address, cutlass.Vector.from_elements(tuple(packed), cutlass.Int32))
    return p


@cute.jit
def _pair_bounds(block, sq, causal: cutlass.Constexpr[bool], window: cutlass.Constexpr[int] = 0):
    """The KV tile and this split's q-tile range: tile ``block`` = KV tile * SPLIT + split index."""
    kv = block // SPLIT
    part = block % SPLIT
    lo, hi = _bounds_accum(kv, sq, causal, window)
    n = hi - lo
    return kv, lo + (part * n) // SPLIT, lo + ((part + 1) * n) // SPLIT


@cute.jit
def _pair_partial_epilogue(slab, tmem_base, bars, row, live, empty_state):
    """fp32 dV then dK of this split through the dq slice ring (one credit phase per plane); returns the ring state."""
    for plane in cutlass.range_constexpr(2):
        vals = cutlass.Vector.from_elements(tuple(cutlass.Float32(0.0) for _ in range(CFG_ACCUM.D)), cutlass.Float32)
        if live:
            if cutlass.const_expr(CFG_ACCUM.D == 128):
                a = _tmem_load(tmem_base + 256 + plane * CFG_ACCUM.D, 64)
                b = _tmem_load(tmem_base + 256 + plane * CFG_ACCUM.D + 64, 64)
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                vals = cutlass.Vector.from_elements(tuple(a[i] for i in range(64)) + tuple(b[i] for i in range(64)), cutlass.Float32)
            else:
                a = _tmem_load(tmem_base + 256 + plane * CFG_ACCUM.D, 64)
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                vals = cutlass.Vector.from_elements(tuple(a[i] for i in range(64)), cutlass.Float32)
        for i in cutlass.range_constexpr(CFG_ACCUM.DQ_SLICES):
            _wait_accum(bars.dq_empty[i], empty_state.phase)
            _store_dq_slice_accum(slab, row, vals, i)
            nvvm.fence_proxy("async.shared", space="cta")
            bars.dq_full[i].arrive()
        empty_state = advance(empty_state, 1)
    return empty_state


@cute.jit
def _pair_softmax_loop(
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
    window: cutlass.Constexpr[int] = 0,
):
    nvvm.setmaxregister(CFG_ACCUM.SOFTMAX_REGS, nvvm.SetMaxRegisterAction.INCREASE)
    tid = cute.arch.thread_idx()[0]
    row = tid % CFG_ACCUM.M
    group = tid // CFG_ACCUM.M
    state = PipelineState.start()
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        if cutlass.const_expr(not causal):
            _credit_accum(bars.sched_empty[sched_state.idx])
        kv, lo, hi = _pair_bounds(block, sq, causal, window)
        if lo < hi:
            _stats_accum(slab, lse, delta, row, group, lo * CFG_ACCUM.N, sq, batch, head * RATIO, scale)
        for g in cutlass.range(0, RATIO, 1, unroll=1):
            qh = head * RATIO + g
            for qtile in cutlass.range(lo, hi, 1, unroll=1):
                _wait_tcgen_accum(bars.s_done, state.phase)
                p = _pair_p_values(slab, tmem_base, row, group, kv, qtile, lo, skv, scale, causal, dtype, window)
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
                bars.p_ready.arrive()
                _wait_tcgen_accum(bars.dp_done, state.phase)
                _ds_values_accum(slab, tmem_base, group, p, scale, dtype)
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
                bars.ds_ready.arrive()
                if qtile + 1 < hi:
                    _stats_accum(slab, lse, delta, row, group, (qtile + 1) * CFG_ACCUM.N, sq, batch, qh, scale)
                elif g + 1 < RATIO:
                    _stats_accum(slab, lse, delta, row, group, lo * CFG_ACCUM.N, sq, batch, qh + 1, scale)
                state = advance(state, 1)
        if cutlass.const_expr(causal):
            _credit_accum(bars.sched_empty[sched_state.idx])
        block, head, batch, valid, sched_state = _next_tile_accum(sched, bars, sched_state, nheads, nk, causal, swizzle, batch_heads)


@cute.jit
def _pair_epilogue_loop(
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
    window: cutlass.Constexpr[int] = 0,
):
    nvvm.setmaxregister(CFG_ACCUM.EPILOGUE_REGS, nvvm.SetMaxRegisterAction.INCREASE)
    row = cute.arch.thread_idx()[0] % CFG_ACCUM.M
    dq_state = PipelineState.start()
    empty_state = PipelineState.start()
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit_accum(bars.sched_empty[sched_state.idx])
        kv, lo, hi = _pair_bounds(block, sq, causal, window)
        # dq_done is committed once per KV tile after its last dK MMA: it orders the readout below after the MMAs.
        _wait_tcgen_accum(bars.dq_done, dq_state.phase)
        dq_state = advance(dq_state, 1)
        if cutlass.const_expr(SPLIT > 1):
            empty_state = _pair_partial_epilogue(slab, tmem_base, bars, row, lo < hi, empty_state)
        else:
            for i in cutlass.range_constexpr(CFG_ACCUM.DQ_SLICES):
                _wait_accum(bars.dq_empty[i], empty_state.phase)
            _native_epilogue_accum(slab, tmem_base, bars, row, lo < hi, empty_state.phase, dtype, False)
            _native_epilogue_accum(slab, tmem_base, bars, row, lo < hi, empty_state.phase, dtype, True)
            empty_state = advance(empty_state, 1)
        block, head, batch, valid, sched_state = _next_tile_accum(sched, bars, sched_state, nheads, nk, causal, swizzle, batch_heads)


@cute.jit
def _pair_mma_loop(
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
    window: cutlass.Constexpr[int] = 0,
):
    nvvm.setmaxregister(CFG_ACCUM.SERVICE_REGS, nvvm.SetMaxRegisterAction.DECREASE)
    score = _mma_desc_accum(dtype)
    grad = _mma_desc_accum(dtype, btranspose=True, N=CFG_ACCUM.D, K=CFG_ACCUM.N)
    kd = _operand_accum(slab, CFG_ACCUM.K, dtype).desc()
    vd = _operand_accum(slab, CFG_ACCUM.V, dtype).desc()
    dod = _operand_accum(slab, CFG_ACCUM.DO, dtype).desc()
    dot = _operand_accum(slab, CFG_ACCUM.DO, dtype, transpose=True).desc()
    s = nvvm.make_tmem_ptr(tmem_base, cutlass.Float32)
    dp = nvvm.make_tmem_ptr(tmem_base + 128, cutlass.Float32)
    dv = nvvm.make_tmem_ptr(tmem_base + 256, cutlass.Float32)
    dk = nvvm.make_tmem_ptr(tmem_base + 256 + CFG_ACCUM.D, cutlass.Float32)
    q_state = PipelineState.start()
    do_state = PipelineState.start()
    kv_state = PipelineState.start()
    ready_state = PipelineState.start()
    out_empty = PipelineState.start(phase=1)
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit_accum(bars.sched_empty[sched_state.idx])
        kv, lo, hi = _pair_bounds(block, sq, causal, window)
        if lo < hi:
            _wait_tcgen_accum(bars.k_full, kv_state.phase)
            _wait_tcgen_accum(bars.q_full[q_state.idx], q_state.phase)
            qd = _operand_accum(slab, CFG_ACCUM.Q + q_state.idx * CFG_ACCUM.OPERAND_BYTES, dtype).desc()
            mma_ss(score, kd, qd, s)
            _commit(bars.s_done)
            _wait_tcgen_accum(bars.v_full, kv_state.phase)
        _wait_tcgen_accum(bars.dvk_empty, out_empty.phase)
        out_empty = advance(out_empty, 1)
        first = cutlass.Int32(1)
        for g in cutlass.range(0, RATIO, 1, unroll=1):
            for qtile in cutlass.range(lo, hi, 1, unroll=1):
                for half in cutlass.range_constexpr(CFG_ACCUM.DO_SUBTILES):
                    _wait_tcgen_accum(bars.do_full[half], do_state.phase)
                    mma_ss(score, vd, dod, dp, k_start=half * _DO_STEPS, k_count=_DO_STEPS, accumulate=half > 0)
                _commit(bars.dp_done)
                _wait_tcgen_accum(bars.p_ready, ready_state.phase)
                _ts_accum(grad, tmem_base, dot, dv, first == 0)
                _commit(bars.do_empty)
                q_next = advance(q_state, CFG_ACCUM.Q_STAGES)
                if (qtile + 1 < hi) | (g + 1 < RATIO):
                    _wait_tcgen_accum(bars.q_full[q_next.idx], q_next.phase)
                    qd_next = _operand_accum(slab, CFG_ACCUM.Q + q_next.idx * CFG_ACCUM.OPERAND_BYTES, dtype).desc()
                    mma_ss(score, kd, qd_next, s)
                    _commit(bars.s_done)
                _wait_tcgen_accum(bars.ds_ready, ready_state.phase)
                qt = _operand_accum(slab, CFG_ACCUM.Q + q_state.idx * CFG_ACCUM.OPERAND_BYTES, dtype, transpose=True).desc()
                _ts_accum(grad, tmem_base + 128, qt, dk, first == 0)
                _commit(bars.q_empty[q_state.idx])
                first = cutlass.Int32(0)
                q_state = q_next
                do_state = advance(do_state, 1)
                ready_state = advance(ready_state, 1)
        if lo < hi:
            _commit(bars.k_empty)
            _commit(bars.v_empty)
            kv_state = advance(kv_state, 1)
        _commit(bars.dq_done)
        block, head, batch, valid, sched_state = _next_tile_accum(sched, bars, sched_state, nheads, nk, causal, swizzle, batch_heads)


@cute.jit
def _pair_load_loop(
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
    window: cutlass.Constexpr[int] = 0,
):
    nvvm.setmaxregister(CFG_ACCUM.SERVICE_REGS, nvvm.SetMaxRegisterAction.DECREASE)
    q_state = PipelineState.start(phase=1)
    do_state = PipelineState.start(phase=1)
    kv_state = PipelineState.start(phase=1)
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit_accum(bars.sched_empty[sched_state.idx])
        kv, lo, hi = _pair_bounds(block, sq, causal, window)
        if lo < hi:
            _wait_accum(bars.k_empty, kv_state.phase)
            bars.k_full.arrive(n_bytes=CFG_ACCUM.OPERAND_BYTES, pred=nvvm.elect_sync())
            tma_load_tile(_operand_accum(slab, CFG_ACCUM.K, dtype), GmemTileTma(k_desc)(0, kv * CFG_ACCUM.M, head, batch), bars.k_full.smem_ptr)
            _wait_accum(bars.v_empty, kv_state.phase)
            bars.v_full.arrive(n_bytes=CFG_ACCUM.OPERAND_BYTES, pred=nvvm.elect_sync())
            tma_load_tile(_operand_accum(slab, CFG_ACCUM.V, dtype), GmemTileTma(v_desc)(0, kv * CFG_ACCUM.M, head, batch), bars.v_full.smem_ptr)
            kv_state = advance(kv_state, 1)
        for g in cutlass.range(0, RATIO, 1, unroll=1):
            qh = head * RATIO + g
            for qtile in cutlass.range(lo, hi, 1, unroll=1):
                _wait_accum(bars.q_empty[q_state.idx], q_state.phase)
                bars.q_full[q_state.idx].arrive(n_bytes=CFG_ACCUM.OPERAND_BYTES, pred=nvvm.elect_sync())
                tma_load_tile(
                    _operand_accum(slab, CFG_ACCUM.Q + q_state.idx * CFG_ACCUM.OPERAND_BYTES, dtype),
                    GmemTileTma(q_desc)(0, qtile * CFG_ACCUM.N, qh, batch),
                    bars.q_full[q_state.idx].smem_ptr,
                )
                _wait_accum(bars.do_empty, do_state.phase)
                for half in cutlass.range_constexpr(CFG_ACCUM.DO_SUBTILES):
                    bars.do_full[half].arrive(n_bytes=CFG_ACCUM.SLICE_BYTES, pred=nvvm.elect_sync())
                for half in cutlass.range_constexpr(CFG_ACCUM.DO_SUBTILES):
                    tma_load_subtiles(
                        _operand_accum(slab, CFG_ACCUM.DO, dtype), GmemTileTma(do_desc)(0, qtile * CFG_ACCUM.N, qh, batch), bars.do_full[half].smem_ptr, half, 1
                    )
                q_state = advance(q_state, CFG_ACCUM.Q_STAGES)
                do_state = advance(do_state, 1)
        block, head, batch, valid, sched_state = _next_tile_accum(sched, bars, sched_state, nheads, nk, causal, swizzle, batch_heads)


@cute.jit
def _pair_store_loop(
    slab,
    bars,
    sched,
    dk_desc,
    dv_desc,
    block,
    head,
    batch,
    nk,
    nheads,
    batch_heads: cutlass.Constexpr[int],
    swizzle: cutlass.Constexpr[int],
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
):
    nvvm.setmaxregister(CFG_ACCUM.SERVICE_REGS, nvvm.SetMaxRegisterAction.DECREASE)
    out_state = PipelineState.start()
    full_state = PipelineState.start()
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit_accum(bars.sched_empty[sched_state.idx])
        for i in cutlass.range_constexpr(CFG_ACCUM.DQ_SLICES):
            bars.dq_empty[i].arrive()
        kv = block // SPLIT
        if cutlass.const_expr(SPLIT > 1):
            # fp32 partials of this split: plane = batch * SPLIT + split index of the partial tensors
            plane = batch * SPLIT + block % SPLIT
            for rnd in cutlass.range_constexpr(2):
                for i in cutlass.range_constexpr(CFG_ACCUM.DQ_SLICES):
                    _wait_accum(bars.dq_full[i], full_state.phase)
                if cutlass.const_expr(rnd == 0):
                    tma_store_tile(_operand_accum(slab, CFG_ACCUM.ALIAS, cutlass.Float32, fp32=True), GmemTileTma(dv_desc)(0, kv * CFG_ACCUM.M, head, plane))
                else:
                    tma_store_tile(_operand_accum(slab, CFG_ACCUM.ALIAS, cutlass.Float32, fp32=True), GmemTileTma(dk_desc)(0, kv * CFG_ACCUM.M, head, plane))
                tma_store_commit()
                tma_store_wait(0)
                if cutlass.const_expr(rnd == 0):
                    # the next tile's bootstrap credits the slices after the second store
                    for i in cutlass.range_constexpr(CFG_ACCUM.DQ_SLICES):
                        bars.dq_empty[i].arrive()
                full_state = advance(full_state, 1)
        else:
            _wait_accum(bars.dv_full, out_state.phase)
            tma_store_tile(_operand_accum(slab, CFG_ACCUM.ALIAS, dtype), GmemTileTma(dv_desc)(0, kv * CFG_ACCUM.M, head, batch))
            _wait_accum(bars.dk_full, out_state.phase)
            tma_store_tile(_operand_accum(slab, CFG_ACCUM.ALIAS + CFG_ACCUM.OPERAND_BYTES, dtype), GmemTileTma(dk_desc)(0, kv * CFG_ACCUM.M, head, batch))
            out_state = advance(out_state, 1)
            tma_store_commit()
            tma_store_wait(0)
        bars.dvk_empty.arrive()
        block, head, batch, valid, sched_state = _next_tile_accum(sched, bars, sched_state, nheads, nk, causal, swizzle, batch_heads)


@cute.kernel
def bprop_d128_accum_kernel(
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
    window: cutlass.Constexpr[int] = 0,
):
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    x, y, z = cute.arch.block_idx()
    nk = (skv + CFG_ACCUM.M - 1) // CFG_ACCUM.M
    if cutlass.const_expr(causal):
        block, head, batch = _decode_flat(x, nheads, nk, swizzle, batch_heads)
    else:
        block, head, batch = x, y, z
    slab = cutlass.Array(cutlass.Uint8, CFG_ACCUM.SMEM_BYTES, alignment=1024, space=cutlass.AddressSpace.smem)
    bars = _bars_accum(slab)
    tmem_slot = _view(slab, CFG_ACCUM.TMEM_HANDLE, cutlass.Int32, 1, alignment=4)
    sched_words = _view(slab, CFG_ACCUM.SCHED_PAYLOAD, cutlass.Int32, 16, alignment=16)
    sched = Sched(bars.sched_full.base_ptr, bars.sched_empty.base_ptr, sched_words, x, y, z)
    if warp == 0:
        # All 512 TMEM columns belong to this CTA; use the portable allocation form.
        nvvm.tcgen05_alloc(tmem_slot, cutlass.Int32(CFG_ACCUM.TMEM_COLS), group=nvvm.CTAGroup.CTA_1)
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
        _pair_softmax_loop(
            slab, bars, sched, tmem_base, lse, delta, block, head, batch, sq, skv, nk, nheads, batch_heads, swizzle, scale, causal, dtype, window
        )
    elif warp < 12:
        _pair_epilogue_loop(slab, bars, sched, tmem_base, block, head, batch, sq, nk, nheads, batch_heads, swizzle, causal, dtype, window)
    elif warp == 12:
        _pair_mma_loop(slab, bars, sched, tmem_base, block, head, batch, sq, nk, nheads, batch_heads, swizzle, causal, dtype, window)
    elif warp == 13:
        _pair_load_loop(slab, bars, sched, q_desc, k_desc, v_desc, do_desc, block, head, batch, sq, nk, nheads, batch_heads, swizzle, causal, dtype, window)
    elif warp == 14:
        _pair_store_loop(slab, bars, sched, dk_desc, dv_desc, block, head, batch, nk, nheads, batch_heads, swizzle, causal, dtype)
    else:
        _scheduler_loop_accum(bars, sched)
    nvvm.barrier_cta_sync()
    if warp == 0:
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
        nvvm.tcgen05_dealloc(nvvm.make_tmem_ptr(tmem_slot.load(), cutlass.Int8), cutlass.Int32(CFG_ACCUM.TMEM_COLS), group=nvvm.CTAGroup.CTA_1)


bprop_d128_accum_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def host_accum(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    do: cute.Tensor,
    lse: cute.Tensor,
    delta: cute.Tensor,
    dq: Optional[cute.Tensor],
    dk: cute.Tensor,
    dv: cute.Tensor,
    scale: cutlass.Float32,
    causal: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
    window: cutlass.Constexpr[int] = 0,
):
    if cutlass.const_expr(q.element_type not in (cutlass.BFloat16, cutlass.Float16)):
        raise ValueError("dense d128 accumulation requires BF16/FP16")
    if cutlass.const_expr(q.shape[3] != CFG_ACCUM.D or k.shape[3] != CFG_ACCUM.D or q.shape[2] != k.shape[2] * RATIO):
        raise ValueError("dense d128 accumulation requires dense MHA d128")
    if cutlass.const_expr(q.shape[1] <= 0 or k.shape[1] <= 0):
        raise ValueError("dense d128 accumulation requires positive Sq/Skv")
    if cutlass.const_expr(SPLIT > 1 and (causal or window > 0)):
        raise ValueError("the q-range split serves unmasked tiles only")
    if cutlass.const_expr(SPLIT > 1 and (dk.element_type != cutlass.Float32 or dv.element_type != cutlass.Float32)):
        raise ValueError("the q-range split writes fp32 dK / dV partials")
    nk = (k.shape[1] + CFG_ACCUM.M - 1) // CFG_ACCUM.M
    heads_per_l2 = (50 * 1024 * 1024) // (q.shape[1] * (CFG_ACCUM.D + CFG_ACCUM.D) * CFG_ACCUM.BPE * RATIO)
    swizzle = 1 << (heads_per_l2.bit_length() - 1) if heads_per_l2 > 0 else 1
    grid = (nk * k.shape[2] * q.shape[0], 1, 1) if causal else (nk * SPLIT, k.shape[2], q.shape[0])
    out_box = CFG_ACCUM.DQ_BOX if SPLIT > 1 else CFG_ACCUM.HEAD_BOX
    bprop_d128_accum_kernel(
        _tensor_map_accum(q, CFG_ACCUM.HEAD_BOX),
        _tensor_map_accum(k, CFG_ACCUM.HEAD_BOX),
        _tensor_map_accum(v, CFG_ACCUM.HEAD_BOX),
        _tensor_map_accum(do, CFG_ACCUM.HEAD_BOX),
        _tensor_map_accum(dq, CFG_ACCUM.DQ_BOX) if DQ_OUT else _tensor_map_accum(dk, out_box),
        _tensor_map_accum(dk, out_box),
        _tensor_map_accum(dv, out_box),
        lse,
        delta,
        q.shape[1],
        k.shape[1],
        k.shape[2],
        q.shape[0] * k.shape[2],
        scale,
        causal,
        swizzle,
        q.element_type,
        window,
    ).launch(grid=grid, block=(CFG_ACCUM.THREADS, 1, 1), stream=stream, min_blocks_per_mp=1)
