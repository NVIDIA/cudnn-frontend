# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""D192 accumulation helpers shared by its dense and packed entries."""

from typing import NamedTuple
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import arith
from cutlass.base_dsl.typing import Pointer
from cutlass.cute.arch.nvvm_wrappers import inline_ptx
from cutlass.experimental import primitives as nvvm
from cutlass.experimental.cuda import tensor_map as tmap
from cudnn.frost.tile_dsl.barrier import MBarrier, PipelineState, Producer, advance
from cudnn.frost.tile_dsl.handles import MmaDesc, SmemTile
from cudnn.frost.tile_dsl.mma import mma_ts_step
from cudnn.frost.tile_dsl.pointwise import ffma2, fmul2, fp32_to_fp16
from cudnn.frost.tile_dsl.scheduler import read_clc_payload
from cudnn.frost.tile_dsl.tma import st_shared_v4
from cudnn.sdpa.bwd.kernels.sm100.d192_f16_accum.config import BAR_LAYOUT, CFG

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


def _tensor_map(tensor, rows, box):
    return tmap.create_tensor_map_tiled_from_view(tensor, box_dims=(1, rows, 1, box), stride_order=(3, 1, 2, 0), swizzle=tmap.TensorMapSwizzle.s128b)


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
    # Only the softmax warps participate; bar.sync implies CTA-wide convergence.
    # Retain the inline scheduling fence without that aligned-CTA promise.
    inline_ptx(f"barrier.sync {CFG.SOFTMAX_BARRIER}, {CFG.SOFTMAX_LANES};", write_only_types=[], read_only_args=[])


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
def _ts(desc, plane, bdesc, dst, accum):
    for k in cutlass.range_constexpr(CFG.N // CFG.K_HW):
        mma_ts_step(
            desc, nvvm.make_tmem_ptr(plane + (k // 2) * 16, cutlass.Int8), bdesc, dst, k_idx=k, accumulate=accum if cutlass.const_expr(k == 0) else True
        )


@cute.jit
def _service_regs():
    nvvm.setmaxregister(CFG.SERVICE_REGS, nvvm.SetMaxRegisterAction.DECREASE)


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
