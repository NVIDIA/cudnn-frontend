# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Native packed THD MHA d128 backward with FP32 dQ accumulation.

Sequence-local tensor maps clip every input and output tail, including caller
ports with independent ragged origins. The dense five-product pipeline and all
33 mbarriers retain their lane counts. Natural CLC visits a declared envelope;
invalid owners take the complete zero-trip protocol without touching a map.
"""

from typing import NamedTuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import arith
from cutlass.base_dsl.typing import Pointer
from cutlass.experimental import primitives as nvvm
from cutlass.experimental.cuda import tensor_map as tmap

from cudnn.frost.tile_dsl.barrier import MBarrier, PipelineState, Producer, advance, named_barrier_fence
from cudnn.frost.tile_dsl.handles import MmaDesc, SmemTile, tma_slice_runtime_desc
from cudnn.frost.tile_dsl.mask import apply_mask_words, band_mask_words
from cudnn.frost.tile_dsl.thd import TENSOR_MAP_QWORDS, thd_port_origin
from cudnn.frost.tile_dsl.mma import mma_ss, mma_ts_step
from cudnn.frost.tile_dsl.pointwise import ffma2, fmul2, fp32_to_fp16
from cudnn.frost.tile_dsl.scheduler import Sched, read_clc_payload
from cudnn.frost.tile_dsl.tma import (
    tma_load_subtiles,
    tma_load_tile,
    tma_reduce_add_subtile,
    tma_store_commit,
    tma_store_tile,
    tma_store_wait,
    tma_tensormap_acquire,
)

from ...config_sm100_d128_accum import BAR_LAYOUT, CFG

LOG2E = 1.4426950408889634
MAP_Q, MAP_K, MAP_V, MAP_DO, MAP_DQ, MAP_DK, MAP_DV = range(7)
MAP_COUNT = 7


class Bars(NamedTuple):
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


def _operand(slab, offset, dtype, *, transpose=False, fp32=False):
    box = CFG.DQ_BOX if fp32 else CFG.HEAD_BOX
    bpe = 4 if fp32 else CFG.BPE
    return SmemTile(
        base=_view(slab, offset, dtype, CFG.M * CFG.D, alignment=1024),
        elems_per_stage=CFG.M * CFG.D,
        leading_byte_offset=CFG.M * CFG.SWIZZLE_BYTES if transpose else 16,
        stride_byte_offset=8 * CFG.SWIZZLE_BYTES,
        layout=CFG.SWIZZLE_LAYOUT,
        tma_loads_per_tile=CFG.D // box,
        tma_granu_elems=box,
        tma_subtile_stride_elems=CFG.M * box,
        desc_version=CFG.DESC_VERSION,
    )


def _mma_desc(dtype, *, atranspose=False, btranspose=False):
    return MmaDesc(
        M=CFG.M,
        N=CFG.N,
        K=CFG.D,
        bpe_a=CFG.BPE,
        bpe_b=CFG.BPE,
        tile_k_hw=CFG.K_HW,
        atranspose=atranspose,
        btranspose=btranspose,
        cta_group=1,
        idesc=nvvm.Tcgen05InstrDesc.build(
            c_dtype=cutlass.Float32, a_dtype=dtype, b_dtype=dtype, m_dim=CFG.M, n_dim=CFG.N, a_major=int(atranspose), b_major=int(btranspose)
        ),
        kind=nvvm.Tcgen05MMAKind.F16,
    )


@cute.jit
def _wait(bar, phase):
    bar.wait(phase)
    # All full-warp consumers reconverge before any elected credit/issue.
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
def _map_ptr(maps, batch, n_batch, slot: cutlass.Constexpr[int]):
    return (maps.iterator.raw_ptr() + (cutlass.Int64(slot) * n_batch + batch) * TENSOR_MAP_QWORDS).tospace(cutlass.AddressSpace.generic)


@cute.jit
def _next_tile(sched, bars, state):
    _wait(bars.sched_full[state.idx], state.phase)
    x, yz, valid = read_clc_payload(sched, state.idx * 8, warp_broadcast=True)
    return x, yz & 0xFFFF, (yz >> 16) & 0xFFFF, valid, advance(state, CFG.SCHED_STAGES)


def _tensor_map(tensor, box):
    # Logical TMA coordinates (d, sequence, head, batch), source-proved order.
    return tmap.create_tensor_map_tiled_from_view(tensor, box_dims=(1, CFG.M, 1, box), stride_order=(3, 1, 2, 0), swizzle=tmap.TensorMapSwizzle.s128b)


@cute.jit
def _bounds(block, meta, batch, n_batch, causal: cutlass.Constexpr[bool]):
    data = cutlass.make_array_view(meta)
    q_start = cutlass.Int32(data[n_batch + batch])
    seq_q = cutlass.Int32(data[n_batch + batch + 1]) - q_start
    seq_k = cutlass.Int32(data[batch])
    lo = block if cutlass.const_expr(causal) else cutlass.Int32(0)
    hi = (seq_q + CFG.N - 1) // CFG.N
    if block * CFG.M >= seq_k:
        lo = cutlass.Int32(0)
        hi = cutlass.Int32(0)
    return lo, hi, seq_q, seq_k, q_start


@cute.jit
def _stats(slab, lse, delta, row, group, qbase, batch, head, scale, q_start, seq_q, origins, stats_origin: cutlass.Constexpr[int]):
    # Full 256-lane named barriers also cover inactive tail rows. Invalid rows
    # never dereference a Stats/Delta address, including an all-empty carrier.
    named_barrier_fence(CFG.SOFTMAX_BARRIER, CFG.SOFTMAX_LANES)
    local_row = qbase + row
    if group == 0:
        stat = cutlass.Float32(0.0)
        if local_row < seq_q:
            token = thd_port_origin(origins, stats_origin, batch, q_start) + cutlass.Int64(local_row)
            if cutlass.const_expr(len(lse.shape) == 2):
                val = cutlass.Float32(lse[token, head])
            else:
                val = cutlass.Float32(lse[0, head, token])
            stat = -cutlass.Float32(LOG2E) * val
            if val == cutlass.Float32(float("-inf")):
                stat = cutlass.Float32(0.0)
        _view(slab, CFG.STATS, cutlass.Float32, CFG.N).subview(row).store(stat)
    else:
        stat = cutlass.Float32(0.0)
        if local_row < seq_q:
            stat = -scale * cutlass.Float32(delta[0, head, cutlass.Int64(q_start) + local_row])
        _view(slab, CFG.DELTA, cutlass.Float32, CFG.N).subview(row).store(stat)
    named_barrier_fence(CFG.SOFTMAX_BARRIER, CFG.SOFTMAX_LANES)


@cute.jit
def _p_values(slab, tmem_base, row, group, block, qtile, seq_q, seq_k, scale, causal: cutlass.Constexpr[bool], dtype: cutlass.Constexpr):
    address = tmem_base + group * 64
    a = _tmem_load(address, 32)
    b = _tmem_load(address + 32, 32)
    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
    scores = cutlass.Vector.from_elements(tuple(a[i] for i in range(32)) + tuple(b[i] for i in range(32)), cutlass.Float32)
    values = []
    scaled = scale * cutlass.Float32(LOG2E)
    for chunk in cutlass.range_constexpr(16):
        stat = _view(slab, CFG.STATS, cutlass.Float32, CFG.N).subview(group * 64 + chunk * 4).load(vector_size=4, alignment=16)
        for pair in cutlass.range_constexpr(2):
            i = chunk * 4 + pair * 2
            e0, e1 = ffma2(scaled, scaled, scores[i], scores[i + 1], stat[pair * 2], stat[pair * 2 + 1])
            values.extend((cute.math.exp2(e0, fastmath=True), cute.math.exp2(e1, fastmath=True)))
    # KV-major transpose of the shared mask band: Q columns >= key for causal,
    # below the actual Q length, and entirely dead for a padded KV row. Select
    # exact zero AFTER exp2 (also safe for a no-key row whose LSE is -inf).
    key = block * CFG.M + row
    qbase = qtile * CFG.N + group * 64
    lo = key if cutlass.const_expr(causal) else None
    hi = cutlass.Int32(arith.select((key < seq_k).ir_value(), seq_q.ir_value(), cutlass.Int32(0).ir_value()))
    words = band_mask_words(lo, hi, qbase, 64)
    p = apply_mask_words(cutlass.Vector.from_elements(tuple(values), cutlass.Float32), words, mask_value=0.0, n_cols=64)
    packed = cutlass.Vector.from_elements(tuple(fp32_to_fp16(p[2 * i], p[2 * i + 1], dtype=dtype) for i in range(32)), cutlass.Int32)
    _tmem_store(address, packed)
    return p


@cute.jit
def _ds_values(slab, tmem_base, group, p, scale, dtype: cutlass.Constexpr):
    address = tmem_base + 128 + group * 64
    a = _tmem_load(address, 32)
    b = _tmem_load(address + 32, 32)
    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
    dp = cutlass.Vector.from_elements(tuple(a[i] for i in range(32)) + tuple(b[i] for i in range(32)), cutlass.Float32)
    packed = []
    for chunk in cutlass.range_constexpr(16):
        stat = _view(slab, CFG.DELTA, cutlass.Float32, CFG.N).subview(group * 64 + chunk * 4).load(vector_size=4, alignment=16)
        for pair in cutlass.range_constexpr(2):
            i = chunk * 4 + pair * 2
            d0, d1 = ffma2(scale, scale, dp[i], dp[i + 1], stat[pair * 2], stat[pair * 2 + 1])
            d0, d1 = fmul2(d0, d1, p[i], p[i + 1])
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
        lo, hi, seq_q, seq_k, q_start = _bounds(block, meta, batch, n_batch, causal)
        if lo < hi:
            _stats(slab, lse, delta, row, group, lo * CFG.N, batch, head, scale, q_start, seq_q, origins, stats_origin)
        for qtile in cutlass.range(lo, hi, 1, unroll=1):
            _wait_tcgen(bars.s_done, state.phase)
            p = _p_values(slab, tmem_base, row, group, block, qtile, seq_q, seq_k, scale, causal, dtype)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
            bars.p_ready.arrive()
            _wait_tcgen(bars.dp_done, state.phase)
            _ds_values(slab, tmem_base, group, p, scale, dtype)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
            bars.ds_ready.arrive()
            if qtile + 1 < hi:
                _stats(slab, lse, delta, row, group, (qtile + 1) * CFG.N, batch, head, scale, q_start, seq_q, origins, stats_origin)
            state = advance(state, 1)
        if cutlass.const_expr(causal):
            _credit(bars.sched_empty[sched_state.idx])
        block, head, batch, valid, sched_state = _next_tile(sched, bars, sched_state)


@cute.jit
def _store_words(slab, byte_offset, row, words):
    for vec in cutlass.range_constexpr(8):
        off = byte_offset + row * CFG.SWIZZLE_BYTES + ((row % 8) ^ vec) * 16
        values = cutlass.Vector.from_elements(tuple(words[vec * 4 + k] for k in range(4)), cutlass.Int32)
        Pointer(slab.data_ptr() + off, dtype=cutlass.Int32).store(values, alignment=16)


@cute.jit
def _store_dq_slice(slab, row, values, index: cutlass.Constexpr[int]):
    for vec in cutlass.range_constexpr(8):
        off = CFG.ALIAS + index * CFG.SLICE_BYTES + row * CFG.SWIZZLE_BYTES + ((row % 8) ^ vec) * 16
        v = cutlass.Vector.from_elements(tuple(values[index * 32 + vec * 4 + k] for k in range(4)), cutlass.Float32)
        Pointer(slab.data_ptr() + off, dtype=cutlass.Float32).store(v, alignment=16)


@cute.jit
def _native_epilogue(slab, tmem_base, bars, row, live, empty_phase, dtype: cutlass.Constexpr, dk: cutlass.Constexpr[bool]):
    plane = 384 if cutlass.const_expr(dk) else 256
    base = CFG.ALIAS + (CFG.OPERAND_BYTES if cutlass.const_expr(dk) else 0)
    last_slice = 3 if cutlass.const_expr(dk) else 1
    for half in cutlass.range_constexpr(2):
        vals = cutlass.Vector.from_elements(tuple(cutlass.Float32(0.0) for _ in range(64)), cutlass.Float32)
        if live:
            vals = _tmem_load(tmem_base + plane + half * 64, 64)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
        words = cutlass.Vector.from_elements(tuple(fp32_to_fp16(vals[2 * i], vals[2 * i + 1], dtype=dtype) for i in range(32)), cutlass.Int32)
        if cutlass.const_expr(half == 0):
            _wait(bars.dq_empty[last_slice], empty_phase)
        _store_words(slab, base + half * CFG.SLICE_BYTES, row, words)
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
    meta,
    n_batch,
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
):
    nvvm.setmaxregister(CFG.EPILOGUE_REGS, nvvm.SetMaxRegisterAction.INCREASE)
    row = cute.arch.thread_idx()[0] % CFG.M
    ds_state = PipelineState.start()
    dq_state = PipelineState.start()
    empty_state = PipelineState.start()
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit(bars.sched_empty[sched_state.idx])
        lo, hi, seq_q, seq_k, q_start = _bounds(block, meta, batch, n_batch, causal)
        for qtile in cutlass.range(lo, hi, 1, unroll=1):
            _wait_tcgen(bars.ds_ready, ds_state.phase)
            ds0 = _tmem_load(tmem_base + 128, 32, cutlass.Int32)
            ds1 = _tmem_load(tmem_base + 192, 32, cutlass.Int32)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
            _wait(bars.dq_empty[0], empty_state.phase)
            _store_words(slab, CFG.ALIAS, row, ds0)
            _wait(bars.dq_empty[1], empty_state.phase)
            _store_words(slab, CFG.ALIAS + CFG.SLICE_BYTES, row, ds1)
            nvvm.fence_proxy("async.shared", space="cta")
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
            bars.ds_smem.arrive()
            _wait_tcgen(bars.dq_done, dq_state.phase)
            dq0 = _tmem_load(tmem_base + 128, 64)
            dq1 = _tmem_load(tmem_base + 192, 64)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
            bars.dp_empty.arrive()
            vals = cutlass.Vector.from_elements(tuple(dq0[i] for i in range(64)) + tuple(dq1[i] for i in range(64)), cutlass.Float32)
            for i in cutlass.range_constexpr(4):
                _wait(bars.dq_empty[i], empty_state.phase)
                _store_dq_slice(slab, row, vals, i)
                nvvm.fence_proxy("async.shared", space="cta")
                bars.dq_full[i].arrive()
            ds_state = advance(ds_state, 1)
            dq_state = advance(dq_state, 1)
            empty_state = advance(empty_state, 1)
        if lo >= hi:
            _wait_tcgen(bars.dq_done, dq_state.phase)
            dq_state = advance(dq_state, 1)
        _native_epilogue(slab, tmem_base, bars, row, lo < hi, empty_state.phase, dtype, False)
        _native_epilogue(slab, tmem_base, bars, row, lo < hi, empty_state.phase, dtype, True)
        empty_state = advance(empty_state, 1)
        block, head, batch, valid, sched_state = _next_tile(sched, bars, sched_state)


@cute.jit
def _ts(desc, plane, bdesc, dst, accum):
    for k in cutlass.range_constexpr(CFG.D // CFG.K_HW):
        mma_ts_step(
            desc, nvvm.make_tmem_ptr(plane + (k // 4) * 32, cutlass.Int8), bdesc, dst, k_idx=k, accumulate=accum if cutlass.const_expr(k == 0) else True
        )


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
    nvvm.setmaxregister(CFG.SERVICE_REGS, nvvm.SetMaxRegisterAction.DECREASE)
    score = _mma_desc(dtype)
    grad = _mma_desc(dtype, btranspose=True)
    dqdesc = _mma_desc(dtype, atranspose=True, btranspose=True)
    kd = _operand(slab, CFG.K, dtype).desc()
    vd = _operand(slab, CFG.V, dtype).desc()
    dod = _operand(slab, CFG.DO, dtype).desc()
    dot = _operand(slab, CFG.DO, dtype, transpose=True).desc()
    dst = _operand(slab, CFG.ALIAS, dtype, transpose=True).desc()
    kt = _operand(slab, CFG.K, dtype, transpose=True).desc()
    s = nvvm.make_tmem_ptr(tmem_base, cutlass.Float32)
    dp = nvvm.make_tmem_ptr(tmem_base + 128, cutlass.Float32)
    dv = nvvm.make_tmem_ptr(tmem_base + 256, cutlass.Float32)
    dk = nvvm.make_tmem_ptr(tmem_base + 384, cutlass.Float32)
    q_state = PipelineState.start()
    do_state = PipelineState.start()
    kv_state = PipelineState.start()
    ready_state = PipelineState.start()
    dp_empty = PipelineState.start(phase=1)
    out_empty = PipelineState.start(phase=1)
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit(bars.sched_empty[sched_state.idx])
        lo, hi, seq_q, seq_k, q_start = _bounds(block, meta, batch, n_batch, causal)
        if lo < hi:
            _wait_tcgen(bars.k_full, kv_state.phase)
            _wait_tcgen(bars.q_full[q_state.idx], q_state.phase)
            qd = _operand(slab, CFG.Q + q_state.idx * CFG.OPERAND_BYTES, dtype).desc()
            mma_ss(score, kd, qd, s)
            _commit(bars.s_done)
            _wait_tcgen(bars.v_full, kv_state.phase)
        _wait_tcgen(bars.dvk_empty, out_empty.phase)
        out_empty = advance(out_empty, 1)
        for qtile in cutlass.range(lo, hi, 1, unroll=1):
            _wait_tcgen(bars.dp_empty, dp_empty.phase)
            _wait_tcgen(bars.do_full[0], do_state.phase)
            mma_ss(score, vd, dod, dp, k_start=0, k_count=4)
            _wait_tcgen(bars.do_full[1], do_state.phase)
            mma_ss(score, vd, dod, dp, k_start=4, k_count=4, accumulate=True)
            _commit(bars.dp_done)
            _wait_tcgen(bars.p_ready, ready_state.phase)
            _ts(grad, tmem_base, dot, dv, qtile > lo)
            _commit(bars.do_empty)
            q_next = advance(q_state, CFG.Q_STAGES)
            if qtile + 1 < hi:
                _wait_tcgen(bars.q_full[q_next.idx], q_next.phase)
                qd_next = _operand(slab, CFG.Q + q_next.idx * CFG.OPERAND_BYTES, dtype).desc()
                mma_ss(score, kd, qd_next, s)
                _commit(bars.s_done)
            _wait_tcgen(bars.ds_ready, ready_state.phase)
            qt = _operand(slab, CFG.Q + q_state.idx * CFG.OPERAND_BYTES, dtype, transpose=True).desc()
            _ts(grad, tmem_base + 128, qt, dk, qtile > lo)
            _commit(bars.q_empty[q_state.idx])
            _wait_tcgen(bars.ds_smem, ready_state.phase)
            mma_ss(dqdesc, dst, kt, dp)
            _commit(bars.dq_done)
            q_state = q_next
            do_state = advance(do_state, 1)
            ready_state = advance(ready_state, 1)
            dp_empty = advance(dp_empty, 1)
        if lo < hi:
            _commit(bars.k_empty)
            _commit(bars.v_empty)
            kv_state = advance(kv_state, 1)
        else:
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
    nvvm.setmaxregister(CFG.SERVICE_REGS, nvvm.SetMaxRegisterAction.DECREASE)
    q_state = PipelineState.start(phase=1)
    do_state = PipelineState.start(phase=1)
    kv_state = PipelineState.start(phase=1)
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit(bars.sched_empty[sched_state.idx])
        lo, hi, seq_q, seq_k, q_start = _bounds(block, meta, batch, n_batch, causal)
        q_desc = _map_ptr(maps, batch, n_batch, MAP_Q)
        k_desc = _map_ptr(maps, batch, n_batch, MAP_K)
        v_desc = _map_ptr(maps, batch, n_batch, MAP_V)
        do_desc = _map_ptr(maps, batch, n_batch, MAP_DO)
        if lo < hi:
            tma_tensormap_acquire(q_desc)
            tma_tensormap_acquire(k_desc)
            tma_tensormap_acquire(v_desc)
            tma_tensormap_acquire(do_desc)
            _wait(bars.k_empty, kv_state.phase)
            bars.k_full.arrive(n_bytes=CFG.OPERAND_BYTES, pred=nvvm.elect_sync())
            tma_load_tile(_operand(slab, CFG.K, dtype), tma_slice_runtime_desc(k_desc, 0, block * CFG.M, head, 0), bars.k_full.smem_ptr, acquire=False)
            _wait(bars.v_empty, kv_state.phase)
            bars.v_full.arrive(n_bytes=CFG.OPERAND_BYTES, pred=nvvm.elect_sync())
            tma_load_tile(_operand(slab, CFG.V, dtype), tma_slice_runtime_desc(v_desc, 0, block * CFG.M, head, 0), bars.v_full.smem_ptr, acquire=False)
            kv_state = advance(kv_state, 1)
        for qtile in cutlass.range(lo, hi, 1, unroll=1):
            _wait(bars.q_empty[q_state.idx], q_state.phase)
            bars.q_full[q_state.idx].arrive(n_bytes=CFG.OPERAND_BYTES, pred=nvvm.elect_sync())
            tma_load_tile(
                _operand(slab, CFG.Q + q_state.idx * CFG.OPERAND_BYTES, dtype),
                tma_slice_runtime_desc(q_desc, 0, qtile * CFG.N, head, 0),
                bars.q_full[q_state.idx].smem_ptr,
                acquire=False,
            )
            _wait(bars.do_empty, do_state.phase)
            for half in cutlass.range_constexpr(2):
                bars.do_full[half].arrive(n_bytes=CFG.SLICE_BYTES, pred=nvvm.elect_sync())
            for half in cutlass.range_constexpr(2):
                tma_load_subtiles(
                    _operand(slab, CFG.DO, dtype),
                    tma_slice_runtime_desc(do_desc, 0, qtile * CFG.N, head, 0),
                    bars.do_full[half].smem_ptr,
                    half,
                    1,
                    acquire=False,
                )
            q_state = advance(q_state, CFG.Q_STAGES)
            do_state = advance(do_state, 1)
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
    nvvm.setmaxregister(CFG.SERVICE_REGS, nvvm.SetMaxRegisterAction.DECREASE)
    dq_state = PipelineState.start()
    out_state = PipelineState.start()
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit(bars.sched_empty[sched_state.idx])
        for i in cutlass.range_constexpr(4):
            bars.dq_empty[i].arrive()  # all32 lanes, one bootstrap per KV tile
        lo, hi, seq_q, seq_k, q_start = _bounds(block, meta, batch, n_batch, causal)
        dq_desc = _map_ptr(maps, batch, n_batch, MAP_DQ)
        if lo < hi:
            tma_tensormap_acquire(dq_desc)
        for qtile in cutlass.range(lo, hi, 1, unroll=1):
            for i in cutlass.range_constexpr(4):
                _wait(bars.dq_full[i], dq_state.phase)
                tma_reduce_add_subtile(
                    _operand(slab, CFG.ALIAS, cutlass.Float32, fp32=True), tma_slice_runtime_desc(dq_desc, 0, qtile * CFG.N, head, 0), i, acquire=False
                )
                tma_store_commit()
            for i in cutlass.range_constexpr(4):
                tma_store_wait(3 - i)
                bars.dq_empty[i].arrive()  # all32 after corresponding source read
            dq_state = advance(dq_state, 1)
        _wait(bars.dv_full, out_state.phase)
        # Invalid envelope owners still consume both native publications and
        # return every source credit, but never acquire/use an empty map.
        if block * CFG.M < seq_k:
            dv_desc = _map_ptr(maps, batch, n_batch, MAP_DV)
            tma_store_tile(_operand(slab, CFG.ALIAS, dtype), tma_slice_runtime_desc(dv_desc, 0, block * CFG.M, head, 0))
        _wait(bars.dk_full, out_state.phase)
        if block * CFG.M < seq_k:
            dk_desc = _map_ptr(maps, batch, n_batch, MAP_DK)
            tma_store_tile(_operand(slab, CFG.ALIAS + CFG.OPERAND_BYTES, dtype), tma_slice_runtime_desc(dk_desc, 0, block * CFG.M, head, 0))
        out_state = advance(out_state, 1)
        tma_store_commit()
        tma_store_wait(0)
        bars.dvk_empty.arrive()
        block, head, batch, valid, sched_state = _next_tile(sched, bars, sched_state)


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
def bprop_d128_accum_thd_kernel(
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


bprop_d128_accum_thd_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


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
    bprop_d128_accum_thd_kernel(maps, lse, delta, meta, origins, n_batch, stats_origin, scale, causal, dtype).launch(
        grid=(nk, nheads, n_batch), block=(CFG.THREADS, 1, 1), stream=stream, min_blocks_per_mp=1
    )
