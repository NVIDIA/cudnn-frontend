# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Dense per-tensor FP8 dK / dV for D_QK in {128, 192} with D_V = 128: the key-owner half of the two-kernel backward."""

from typing import NamedTuple, Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import arith
from cutlass.base_dsl.typing import Pointer
from cutlass.experimental import primitives as nvvm
from cutlass.experimental.cuda import tensor_map as tmap

from cudnn.frost.tile_dsl.barrier import MBarrier, PipelineState, Producer, advance
from cudnn.frost.tile_dsl.handles import GmemTileTma, MmaDesc, SmemTile
from cudnn.frost.tile_dsl.mma import mma_ss, mma_ts
from cudnn.frost.tile_dsl.pointwise import (
    abs_max_tree,
    amax_to_ue8m0_rp,
    exp2_emul_pair,
    ffma2,
    fmax_f32,
    fmul2,
    fp32_to_fp16,
    fp32_to_fp8_pack,
    opaque_f32_zero,
    opaque_i32_zero,
)
from cudnn.frost.tile_dsl.scheduler import Sched, read_clc_payload
from cudnn.frost.tile_dsl.tma import st_shared_v4, tma_load_tile, tma_store_commit, tma_store_tile, tma_store_wait

from cudnn.sdpa.bwd.config_sm100_d192_dkdv_fp8 import make_config

_PARAMS = globals().get("FROST_TEMPLATE_PARAMS", (192, 128))
MX = bool(_PARAMS[3]) if len(_PARAMS) > 3 else False
WINDOW = int(_PARAMS[4]) if len(_PARAMS) > 4 else 0
CFG, BAR_LAYOUT = make_config(int(_PARAMS[0]), int(_PARAMS[1]), MX)
TS = bool(CFG.T_SHARED)
P_SMEM = bool(CFG.P_SMEM)
EXP2_FMA_SPLIT = bool(_PARAMS[2]) if len(_PARAMS) > 2 else False


def _emulated(pair: int) -> bool:
    return EXP2_FMA_SPLIT and pair % 8 >= 6


LOG2E = 1.4426950408889634
FP8 = cutlass.Float8E4M3FN
HALF_DTYPES = (cutlass.BFloat16, cutlass.Float16)


class Bars(NamedTuple):
    k_empty: MBarrier
    k_full: MBarrier
    v_full: MBarrier
    do_empty: MBarrier
    do_full: MBarrier
    q_empty: MBarrier
    q_full: MBarrier
    qt_empty: MBarrier
    qt_full: MBarrier
    q_used: MBarrier
    do_used: MBarrier
    dot_full: MBarrier
    s_done: MBarrier
    dp_done: MBarrier
    dvk_done: MBarrier
    s_free: MBarrier
    dp_empty: MBarrier
    p_full: MBarrier
    p_empty: MBarrier
    ds_full: MBarrier
    ds_empty: MBarrier
    dv_full: MBarrier
    dk_full: MBarrier
    dvk_empty: MBarrier
    sched_full: MBarrier
    sched_empty: MBarrier
    stats_full: MBarrier


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


def _bpe(dtype):
    if dtype == cutlass.Float32:
        return 4
    if dtype in HALF_DTYPES:
        return 2
    return 1


def _box(dtype):
    return CFG.SWIZZLE_BYTES // _bpe(dtype)


def _operand(slab, offset, dtype, rows, cols, *, transpose=False, swz=CFG.SWIZZLE_BYTES):
    box = swz // _bpe(dtype)
    return SmemTile(
        base=_view(slab, offset, dtype, rows * cols, alignment=1024),
        elems_per_stage=rows * cols,
        leading_byte_offset=rows * swz if transpose else 16,
        stride_byte_offset=8 * swz,
        layout=CFG.SWIZZLE_LAYOUT if swz == CFG.SWIZZLE_BYTES else CFG.NARROW_SWIZZLE_LAYOUT,
        tma_loads_per_tile=cols // box,
        tma_granu_elems=box,
        tma_subtile_stride_elems=rows * box,
        desc_version=CFG.DESC_VERSION,
    )


def _mma_desc(m, n, k, *, atranspose=False, btranspose=False, mx=MX):
    if mx:
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
            idesc=nvvm.Tcgen05MxInstrDesc.build(a_dtype=FP8, b_dtype=FP8, m_dim=m, n_dim=n, k_dim=0, a_major=int(atranspose), b_major=int(btranspose)),
            kind=nvvm.MMABlockScaleKind.MXF8F6F4,
            is_block_scale=True,
            sf_blocks_per_step=CFG.K_HW // 32,
            scale_vec_size=nvvm.Tcgen05MMAScaleVecSize.BLOCK32,
        )
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
            c_dtype=cutlass.Float32, a_dtype=FP8, b_dtype=FP8, m_dim=m, n_dim=n, a_major=int(atranspose), b_major=int(btranspose)
        ),
        kind=nvvm.Tcgen05MMAKind.F8F6F4,
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
def _scalar(tensor):
    return cutlass.Float32(cutlass.make_array_view(tensor)[0])


@cute.jit
def _fold_amax(value, tensor):
    for step in cutlass.range_constexpr(5):
        peer = nvvm.shfl_sync(thread_mask=0xFFFFFFFF, val=value, offset=1 << step, mask_and_clamp=0x1F, kind=nvvm.Shfl.BFLY)
        value = fmax_f32(value, peer)
    if cute.arch.thread_idx()[0] % 32 == 0:
        nvvm.atomicrmw(nvvm.AtomicOp.MAX, Pointer(tensor.iterator.raw_ptr(), dtype=cutlass.Int32), value.bitcast(cutlass.Int32))


def _pack_fp8(values):
    words = []
    for chunk in range(len(values) // 16):
        packed = fp32_to_fp8_pack(values[chunk * 16 : chunk * 16 + 16], dtype=FP8)
        words.extend(packed[j] for j in range(4))
    return cutlass.Vector.from_elements(tuple(words), cutlass.Int32)


def _scale_pairs(values, factor):
    scaled = []
    for i in range(0, len(values), 2):
        a, b = fmul2(values[i], values[i + 1], factor, factor)
        scaled.extend((a, b))
    return scaled


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


def _sf_map(tensor):
    return tmap.create_tensor_map_tiled_from_view(
        tensor, box_dims=(1, 1, 1, tensor.shape[3], 128), stride_order=(4, 3, 2, 1, 0), swizzle=tmap.TensorMapSwizzle.none
    )


def _tensor_map(tensor, rows, box, swz=CFG.SWIZZLE_BYTES):
    swizzle = tmap.TensorMapSwizzle.s128b if swz == CFG.SWIZZLE_BYTES else tmap.TensorMapSwizzle.s64b
    return tmap.create_tensor_map_tiled_from_view(tensor, box_dims=(1, rows, 1, box), stride_order=(3, 1, 2, 0), swizzle=swizzle)


@cute.jit
def _bounds(block, sq, causal: cutlass.Constexpr[bool]):
    lo = cutlass.Int32(0)
    if cutlass.const_expr(causal):
        lo = block * (CFG.M // CFG.N)
    hi = (sq + CFG.N - 1) // CFG.N
    if cutlass.const_expr(WINDOW > 0):
        hi = cute.math.min(hi, (block * CFG.M + CFG.M + WINDOW - 1 + CFG.N - 1) // CFG.N)
    return lo, hi


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
def _stat_publish(slab, bars, row, group, slot, qbase, sq, delta_scale, lse_shift, val):
    if row < CFG.N:
        if group == 0:
            valid = qbase + row < sq
            stat = -cutlass.Float32(LOG2E) * val + lse_shift
            if val == cutlass.Float32(float("-inf")):
                stat = lse_shift
            stat = cutlass.Float32(arith.select(valid.ir_value(), stat.ir_value(), cutlass.Float32(float("-inf")).ir_value()))
            _view(slab, CFG.STATS + slot * 512, cutlass.Float32, CFG.N).subview(row).store(stat)
        else:
            _view(slab, CFG.DELTA + slot * 512, cutlass.Float32, CFG.N).subview(row).store(-delta_scale * val)
    bars.stats_full[slot].arrive()


@cute.jit
def _masked_scores(
    scores, row, group, block, qtile, skv, causal_mask: cutlass.Constexpr[bool], key_mask: cutlass.Constexpr[bool], window_mask: cutlass.Constexpr[bool] = False
):
    neg_inf = cutlass.Float32(float("-inf"))
    kr = block * CFG.M + row
    key_valid = kr < skv
    out = []
    for i in cutlass.range_constexpr(64):
        v = scores[i]
        qc = qtile * CFG.N + group * 64 + i
        if cutlass.const_expr(causal_mask):
            v = cutlass.Float32(arith.select((kr <= qc).ir_value(), v.ir_value(), neg_inf.ir_value()))
        if cutlass.const_expr(window_mask):
            v = cutlass.Float32(arith.select((kr + (WINDOW - 1) >= qc).ir_value(), v.ir_value(), neg_inf.ir_value()))
        if cutlass.const_expr(key_mask):
            v = cutlass.Float32(arith.select(key_valid.ir_value(), v.ir_value(), neg_inf.ir_value()))
        out.append(v)
    return cutlass.Vector.from_elements(tuple(out), cutlass.Float32)


@cute.jit
def _p_values(
    slab,
    scores,
    row,
    group,
    block,
    qtile,
    skv,
    slot,
    s_scale,
    causal_mask: cutlass.Constexpr[bool],
    key_mask: cutlass.Constexpr[bool],
    window_mask: cutlass.Constexpr[bool] = False,
):
    if cutlass.const_expr(causal_mask or key_mask or window_mask):
        scores = _masked_scores(scores, row, group, block, qtile, skv, causal_mask, key_mask, window_mask)
    p = []
    scaled = s_scale * cutlass.Float32(LOG2E)
    for half in cutlass.range_constexpr(2):
        stats = [
            _view(slab, CFG.STATS + slot * 512, cutlass.Float32, CFG.N).subview(group * 64 + (half * 8 + c) * 4).load(vector_size=4, alignment=16)
            for c in range(8)
        ]
        exponents = []
        for c in cutlass.range_constexpr(8):
            for pair in cutlass.range_constexpr(2):
                i = half * 32 + c * 4 + pair * 2
                e0, e1 = ffma2(scaled, scaled, scores[i], scores[i + 1], stats[c][pair * 2], stats[c][pair * 2 + 1])
                exponents.extend((e0, e1))
        for pair in cutlass.range_constexpr(16):
            if cutlass.const_expr(_emulated(pair)):
                p.extend(exp2_emul_pair(exponents[2 * pair], exponents[2 * pair + 1]))
            else:
                p.append(cute.math.exp2(exponents[2 * pair], fastmath=True))
                p.append(cute.math.exp2(exponents[2 * pair + 1], fastmath=True))
    return cutlass.Vector.from_elements(tuple(p), cutlass.Float32), _pack_fp8(p)


@cute.jit
def _ds_values(slab, dp, group, slot, p, dp_scale, amax, fold: cutlass.Constexpr[bool]):
    ds = []
    for chunk in cutlass.range_constexpr(16):
        stat = _view(slab, CFG.DELTA + slot * 512, cutlass.Float32, CFG.N).subview(group * 64 + chunk * 4).load(vector_size=4, alignment=16)
        for pair in cutlass.range_constexpr(2):
            i = chunk * 4 + pair * 2
            d0, d1 = ffma2(dp_scale, dp_scale, dp[i], dp[i + 1], stat[pair * 2], stat[pair * 2 + 1])
            d0, d1 = fmul2(d0, d1, p[i], p[i + 1])
            ds.extend((d0, d1))
    if cutlass.const_expr(fold):
        amax = fmax_f32(amax, abs_max_tree(ds))
    if cutlass.const_expr(MX):
        e0, inv0 = amax_to_ue8m0_rp(abs_max_tree(ds[:32]))
        e1, inv1 = amax_to_ue8m0_rp(abs_max_tree(ds[32:]))
        scaled = _scale_pairs(ds[:32], inv0) + _scale_pairs(ds[32:], inv1)
        return amax, _pack_fp8(scaled), cutlass.Int32(e0) | (cutlass.Int32(e1) << 8)
    return amax, _pack_fp8(ds), cutlass.Int32(0)


@cute.jit
def _st16(slab, off, values, i: cutlass.Constexpr[int], dtype: cutlass.Constexpr):
    st_shared_v4(_view(slab, off, dtype, 4).data_ptr(), (values[i], values[i + 1], values[i + 2], values[i + 3]), dtype)


@cute.jit
def _store_chunks(slab, byte_offset, row, words, first: cutlass.Constexpr[int], count: cutlass.Constexpr[int], swz: cutlass.Constexpr[int] = CFG.SWIZZLE_BYTES):
    for vec in cutlass.range_constexpr(count):
        chunk = first + vec
        if cutlass.const_expr(swz == CFG.SWIZZLE_BYTES):
            off = byte_offset + row * swz + ((row % 8) ^ chunk) * 16
        else:
            off = byte_offset + row * swz + (((row // 2) % 4) ^ chunk) * 16
        _st16(slab, off, words, vec * 4, cutlass.Int32)


@cute.jit
def _softmax_tile(
    slab,
    bars,
    tmem_base,
    lse,
    delta,
    row,
    group,
    block,
    head,
    batch,
    sq,
    skv,
    hi,
    qtile,
    s_scale,
    dp_scale,
    delta_scale,
    lse_shift,
    phase,
    ring_idx,
    ring_phase,
    stats,
    amax,
    fold: cutlass.Constexpr[bool],
    causal_mask: cutlass.Constexpr[bool],
    key_mask: cutlass.Constexpr[bool],
    window_mask: cutlass.Constexpr[bool] = False,
):
    stat_next = _stat_load(lse, delta, row, group, (qtile + 1) * CFG.N, sq, batch, head)
    _wait_tcgen(bars.s_done, phase)
    address = tmem_base + CFG.TMEM_S + group * 64
    a = _tmem_load(address, 32)
    b = _tmem_load(address + 32, 32)
    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
    nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
    bars.s_free.arrive()
    scores = cutlass.Vector.from_elements(tuple(a[i] for i in range(32)) + tuple(b[i] for i in range(32)), cutlass.Float32)
    _wait(bars.stats_full[stats.idx], stats.phase)
    p, p_words = _p_values(slab, scores, row, group, block, qtile, skv, stats.idx, s_scale, causal_mask, key_mask, window_mask)
    if cutlass.const_expr(P_SMEM):
        _wait(bars.p_empty[ring_idx], ring_phase ^ 1)
        _store_chunks(slab, CFG.P + ring_idx * CFG.RING_BYTES, row, p_words, group * 4, 4)
        nvvm.fence_proxy("async.shared", space="cta")
        bars.p_full[ring_idx].arrive()
    else:
        _wait_tcgen(bars.p_empty[ring_idx], ring_phase ^ 1)
        _tmem_store(tmem_base + CFG.TMEM_P + ring_idx * CFG.P_COLS + group * (CFG.P_COLS // 2), p_words)
        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
        bars.p_full[ring_idx].arrive()
    _wait_tcgen(bars.dp_done, phase)
    a = _tmem_load(address, 32)
    b = _tmem_load(address + 32, 32)
    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
    nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
    bars.dp_empty.arrive()
    dp = cutlass.Vector.from_elements(tuple(a[i] for i in range(32)) + tuple(b[i] for i in range(32)), cutlass.Float32)
    amax, ds_words, ds_sf = _ds_values(slab, dp, group, stats.idx, p, dp_scale, amax, fold)
    _wait(bars.ds_empty[ring_idx], ring_phase ^ 1)
    _store_chunks(slab, CFG.DS + ring_idx * CFG.RING_BYTES, row, ds_words, group * 4, 4)
    if cutlass.const_expr(MX):
        _view(slab, CFG.SF_DS + ring_idx * 512 + (row % 32) * 16 + (row // 32) * 4 + group * 2, cutlass.Uint16, 1, alignment=2).subview(0).store(
            cutlass.Uint16(ds_sf)
        )
    nvvm.fence_proxy("async.shared", space="cta")
    bars.ds_full[ring_idx].arrive()
    stats = advance(stats, 2)
    if qtile + 1 < hi:
        _stat_publish(slab, bars, row, group, stats.idx, (qtile + 1) * CFG.N, sq, delta_scale, lse_shift, stat_next)
    next_idx = ring_idx + 1
    wrap = next_idx == CFG.P_STAGES
    ring_idx = cutlass.Int32(arith.select(wrap.ir_value(), cutlass.Int32(0).ir_value(), next_idx.ir_value()))
    ring_phase = cutlass.Int32(arith.select(wrap.ir_value(), (ring_phase ^ 1).ir_value(), ring_phase.ir_value()))
    return phase ^ 1, ring_idx, ring_phase, stats, amax


@cute.jit
def _softmax_tiles(
    slab,
    bars,
    tmem_base,
    lse,
    delta,
    row,
    group,
    block,
    head,
    batch,
    sq,
    skv,
    lo,
    hi,
    s_scale,
    dp_scale,
    delta_scale,
    lse_shift,
    phase,
    ring_idx,
    ring_phase,
    stats,
    amax,
    fold: cutlass.Constexpr[bool],
    causal: cutlass.Constexpr[bool],
    key_mask: cutlass.Constexpr[bool],
):
    diag_end = lo
    if cutlass.const_expr(causal):
        diag_end = cute.math.min(lo + CFG.M // CFG.N, hi)
        for qtile in cutlass.range(lo, diag_end, 1, unroll=1):
            phase, ring_idx, ring_phase, stats, amax = _softmax_tile(
                slab,
                bars,
                tmem_base,
                lse,
                delta,
                row,
                group,
                block,
                head,
                batch,
                sq,
                skv,
                hi,
                qtile,
                s_scale,
                dp_scale,
                delta_scale,
                lse_shift,
                phase,
                ring_idx,
                ring_phase,
                stats,
                amax,
                fold,
                True,
                key_mask,
                WINDOW > 0,
            )
    edge_start = hi
    if cutlass.const_expr(WINDOW > 0):
        edge_start = cute.math.max(diag_end, cute.math.min(hi, (block * CFG.M + WINDOW + CFG.N) // CFG.N - 1))
    for qtile in cutlass.range(diag_end, edge_start, 1, unroll=1):
        phase, ring_idx, ring_phase, stats, amax = _softmax_tile(
            slab,
            bars,
            tmem_base,
            lse,
            delta,
            row,
            group,
            block,
            head,
            batch,
            sq,
            skv,
            hi,
            qtile,
            s_scale,
            dp_scale,
            delta_scale,
            lse_shift,
            phase,
            ring_idx,
            ring_phase,
            stats,
            amax,
            fold,
            False,
            key_mask,
        )
    if cutlass.const_expr(WINDOW > 0):
        for qtile in cutlass.range(edge_start, hi, 1, unroll=1):
            phase, ring_idx, ring_phase, stats, amax = _softmax_tile(
                slab,
                bars,
                tmem_base,
                lse,
                delta,
                row,
                group,
                block,
                head,
                batch,
                sq,
                skv,
                hi,
                qtile,
                s_scale,
                dp_scale,
                delta_scale,
                lse_shift,
                phase,
                ring_idx,
                ring_phase,
                stats,
                amax,
                fold,
                False,
                key_mask,
                True,
            )
    return phase, ring_idx, ring_phase, stats, amax


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
    ratio: cutlass.Constexpr[int] = 1,
):
    nvvm.setmaxregister(CFG.SOFTMAX_REGS, nvvm.SetMaxRegisterAction.INCREASE)
    tid = cute.arch.thread_idx()[0]
    row = tid % CFG.M
    group = tid // CFG.M
    if cutlass.const_expr(MX):
        s_scale = scale
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
    stats = PipelineState.start()
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        if cutlass.const_expr(not causal):
            _credit(bars.sched_empty[sched_state.idx])
        lo, hi = _bounds(block, sq, causal)
        for r in cutlass.range(0, ratio, 1, unroll=1):
            hq = head * ratio + r
            if lo < hi:
                _stat_publish(
                    slab, bars, row, group, stats.idx, lo * CFG.N, sq, delta_scale, lse_shift, _stat_load(lse, delta, row, group, lo * CFG.N, sq, batch, hq)
                )
            if (block + 1) * CFG.M > skv:
                phase, ring_idx, ring_phase, stats, amax = _softmax_tiles(
                    slab,
                    bars,
                    tmem_base,
                    lse,
                    delta,
                    row,
                    group,
                    block,
                    hq,
                    batch,
                    sq,
                    skv,
                    lo,
                    hi,
                    s_scale,
                    dp_scale,
                    delta_scale,
                    lse_shift,
                    phase,
                    ring_idx,
                    ring_phase,
                    stats,
                    amax,
                    fold,
                    causal,
                    True,
                )
            else:
                phase, ring_idx, ring_phase, stats, amax = _softmax_tiles(
                    slab,
                    bars,
                    tmem_base,
                    lse,
                    delta,
                    row,
                    group,
                    block,
                    hq,
                    batch,
                    sq,
                    skv,
                    lo,
                    hi,
                    s_scale,
                    dp_scale,
                    delta_scale,
                    lse_shift,
                    phase,
                    ring_idx,
                    ring_phase,
                    stats,
                    amax,
                    fold,
                    causal,
                    False,
                )
        if cutlass.const_expr(causal):
            _credit(bars.sched_empty[sched_state.idx])
        block, head, batch, valid, sched_state = _next_tile(sched, bars, sched_state, nheads, nk, causal, swizzle, batch_heads)
    if cutlass.const_expr(fold):
        _fold_amax(amax * (cutlass.Float32(1.0) / scale_dp), amax_dp)


@cute.jit
def _native_epilogue(
    slab, tmem_base, bars, row, live, grad_dtype: cutlass.Constexpr, dk: cutlass.Constexpr[bool], factor, out_scale, amax, fold: cutlass.Constexpr[bool]
):
    plane = CFG.TMEM_DK if cutlass.const_expr(dk) else CFG.TMEM_DV
    cols = CFG.DQK if cutlass.const_expr(dk) else CFG.DV
    fp8_out = cutlass.const_expr(grad_dtype == FP8)
    base = CFG.OUT + (CFG.M * CFG.DV * _bpe(grad_dtype) if cutlass.const_expr(dk) else 0)
    for half in cutlass.range_constexpr(cols // 64):
        vals = cutlass.Vector.from_elements(tuple(cutlass.Float32(0.0) for _ in range(64)), cutlass.Float32)
        if live:
            vals = _tmem_load(tmem_base + plane + half * 64, 64)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
        scaled = _scale_pairs([vals[i] for i in range(64)], factor)
        if cutlass.const_expr(fold):
            amax = fmax_f32(amax, abs_max_tree(scaled))
        if cutlass.const_expr(fp8_out):
            words = _pack_fp8(_scale_pairs(scaled, out_scale))
            if cutlass.const_expr(dk and CFG.QK_SWZ == CFG.NARROW_SWIZZLE_BYTES):
                _store_chunks(slab, base + half * CFG.M * CFG.NARROW_SWIZZLE_BYTES, row, words, 0, 4, CFG.NARROW_SWIZZLE_BYTES)
            else:
                _store_chunks(slab, base + (half // 2) * CFG.M * CFG.SWIZZLE_BYTES, row, words, (half % 2) * 4, 4)
        else:
            words = cutlass.Vector.from_elements(tuple(fp32_to_fp16(scaled[2 * i], scaled[2 * i + 1], dtype=grad_dtype) for i in range(32)), cutlass.Int32)
            _store_chunks(slab, base + half * CFG.M * CFG.SWIZZLE_BYTES, row, words, 0, 8)
    nvvm.fence_proxy("async.shared", space="cta")
    nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
    if cutlass.const_expr(dk):
        bars.dk_full.arrive()
    else:
        bars.dv_full.arrive()
    return amax


@cute.jit
def _epilogue_loop(
    slab,
    bars,
    sched,
    tmem_base,
    descale_q,
    descale_s,
    descale_do,
    descale_dp,
    scale_dk_t,
    scale_dv_t,
    amax_dk: Optional[cute.Tensor],
    amax_dv: Optional[cute.Tensor],
    block,
    head,
    batch,
    sq,
    nk,
    nheads,
    batch_heads: cutlass.Constexpr[int],
    swizzle: cutlass.Constexpr[int],
    causal: cutlass.Constexpr[bool],
    grad_dtype: cutlass.Constexpr,
):
    nvvm.setmaxregister(CFG.EPILOGUE_REGS, nvvm.SetMaxRegisterAction.DECREASE)
    tid = cute.arch.thread_idx()[0]
    row = tid % CFG.M
    if cutlass.const_expr(MX):
        dk_factor = opaque_f32_zero() + cutlass.Float32(1.0)
        dv_factor = opaque_f32_zero() + cutlass.Float32(1.0 / 256.0)
    else:
        dk_factor = _scalar(descale_dp) * _scalar(descale_q)
        dv_factor = _scalar(descale_s) * _scalar(descale_do)
    dk_out = cutlass.Float32(1.0)
    dv_out = cutlass.Float32(1.0)
    if cutlass.const_expr(grad_dtype == FP8):
        dk_out = _scalar(scale_dk_t)
        dv_out = _scalar(scale_dv_t)
    fold_k = cutlass.const_expr(amax_dk is not None)
    fold_v = cutlass.const_expr(amax_dv is not None)
    amax_k = opaque_f32_zero()
    amax_v = opaque_f32_zero()
    done_phase = cutlass.Int32(0)
    out_empty = PipelineState.start(phase=1)
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit(bars.sched_empty[sched_state.idx])
        lo, hi = _bounds(block, sq, causal)
        _wait_tcgen(bars.dvk_done, done_phase)
        done_phase = done_phase ^ 1
        _wait(bars.dvk_empty, out_empty.phase)
        out_empty = advance(out_empty, 1)
        amax_v = _native_epilogue(slab, tmem_base, bars, row, lo < hi, grad_dtype, False, dv_factor, dv_out, amax_v, fold_v)
        amax_k = _native_epilogue(slab, tmem_base, bars, row, lo < hi, grad_dtype, True, dk_factor, dk_out, amax_k, fold_k)
        block, head, batch, valid, sched_state = _next_tile(sched, bars, sched_state, nheads, nk, causal, swizzle, batch_heads)
    if cutlass.const_expr(fold_v):
        _fold_amax(amax_v, amax_dv)
    if cutlass.const_expr(fold_k):
        _fold_amax(amax_k, amax_dk)


@cute.jit
def _service_regs():
    nvvm.setmaxregister(CFG.SERVICE_REGS, nvvm.SetMaxRegisterAction.DECREASE)


@cute.jit
def _ring(slab, base, idx, transpose: cutlass.Constexpr[bool]):
    return _operand(slab, base + idx * CFG.RING_BYTES, FP8, CFG.M, CFG.N, transpose=transpose).desc()


def _sf_tile(slab, offset, nbytes: cutlass.Constexpr[int]):
    return SmemTile(
        base=_view(slab, offset, cutlass.Int8, nbytes, alignment=128),
        elems_per_stage=nbytes,
        leading_byte_offset=16,
        stride_byte_offset=128,
        layout=0,
        desc_version=CFG.DESC_VERSION,
    )


@cute.jit
def _sf_ptr(tmem_base, col):
    return nvvm.make_tmem_ptr(tmem_base + col, cutlass.Int8)


@cute.jit
def _sf_copy(tmem_base, col, smem_desc, groups: cutlass.Constexpr[int]):
    if nvvm.elect_sync():
        for g in cutlass.range_constexpr(groups):
            nvvm.tcgen05_cp(
                nvvm.Tcgen05CpShape.SHAPE_32X128B,
                _sf_ptr(tmem_base, col + 4 * g),
                smem_desc + 32 * g,
                group=nvvm.CTAGroup.CTA_1,
                multicast=nvvm.Tcgen05CpMulticast.WARPX4,
            )


@cute.jit
def _q_scales(slab, tmem_base, idx):
    _sf_copy(tmem_base, CFG.TMEM_SF_Q + idx * CFG.SF_COLS_QK, _sf_tile(slab, CFG.SF_Q + idx * CFG.SF_QK_BYTES, CFG.SF_QK_BYTES).desc(), CFG.SF_COLS_QK // 4)


@cute.jit
def _qt_scales(slab, tmem_base, idx):
    _sf_copy(tmem_base, CFG.TMEM_SF_QT + idx * CFG.SF_COLS_QK, _sf_tile(slab, CFG.SF_QT + idx * CFG.SF_QT_BYTES, CFG.SF_QT_BYTES).desc(), CFG.SF_COLS_QK // 4)


@cute.jit
def _do_scales(slab, tmem_base, idx):
    _sf_copy(tmem_base, CFG.TMEM_SF_DO + idx * CFG.SF_COLS_V, _sf_tile(slab, CFG.SF_DO + idx * CFG.SF_V_BYTES, CFG.SF_V_BYTES).desc(), 1)
    if cutlass.const_expr(not TS):
        _sf_copy(tmem_base, CFG.TMEM_SF_DOT + idx * CFG.SF_COLS_V, _sf_tile(slab, CFG.SF_DOT + idx * CFG.SF_V_BYTES, CFG.SF_V_BYTES).desc(), 1)


@cute.jit
def _dot_scales(slab, tmem_base, idx):
    _sf_copy(tmem_base, CFG.TMEM_SF_DOT + idx * CFG.SF_COLS_V, _sf_tile(slab, CFG.SF_DOT + idx * CFG.SF_V_BYTES, CFG.SF_V_BYTES).desc(), 1)


@cute.jit
def _load_sf_t(slab, offset, desc, tile, plane, bar):
    for g in cutlass.range_constexpr(CFG.SF_COLS_QK // 4):
        tma_load_tile(_sf_tile(slab, offset + g * 512, 512), GmemTileTma(desc)(0, 0, tile, plane, g), bar)


@cute.jit
def _load_t(slab, bars, qt_desc, dot_desc, sf_qt_desc, sf_dot_desc, q_idx, q_phase, do_idx, do_phase, qtile, plane, hq, batch):
    _wait(bars.q_used[q_idx], q_phase ^ 1)
    bars.qt_full[q_idx].arrive(n_bytes=CFG.QT_TX, pred=nvvm.elect_sync())
    tma_load_tile(
        _operand(slab, CFG.Q + q_idx * CFG.Q_STAGE_BYTES, FP8, CFG.N, CFG.DQK, swz=CFG.QK_SWZ),
        GmemTileTma(qt_desc)(0, qtile * CFG.N, hq, batch),
        bars.qt_full[q_idx].smem_ptr,
    )
    _load_sf_t(slab, CFG.SF_QT + q_idx * CFG.SF_QT_BYTES, sf_qt_desc, qtile, plane, bars.qt_full[q_idx].smem_ptr)
    _wait(bars.do_used[do_idx], do_phase ^ 1)
    bars.dot_full[do_idx].arrive(n_bytes=CFG.DOT_TX, pred=nvvm.elect_sync())
    tma_load_tile(
        _operand(slab, CFG.DO + do_idx * CFG.DO_STAGE_BYTES, FP8, CFG.N, CFG.DV, swz=CFG.V_SWZ),
        GmemTileTma(dot_desc)(0, qtile * CFG.N, hq, batch),
        bars.dot_full[do_idx].smem_ptr,
    )
    tma_load_tile(
        _sf_tile(slab, CFG.SF_DOT + do_idx * CFG.SF_V_BYTES, CFG.SF_V_BYTES), GmemTileTma(sf_dot_desc)(0, 0, qtile, plane, 0), bars.dot_full[do_idx].smem_ptr
    )


@cute.jit
def _grad_products(slab, bars, tmem_base, do_state, q_state, qt_state, ring, dvdesc, dkdesc, dv, dk, accumulate):
    _wait_tcgen(bars.p_full[ring.idx], ring.phase)
    if cutlass.const_expr(TS):
        _wait_tcgen(bars.dot_full[do_state.idx], do_state.phase)
        _dot_scales(slab, tmem_base, do_state.idx)
    dot = _operand(
        slab, CFG.DO + do_state.idx * CFG.DO_STAGE_BYTES + (CFG.DO_BYTES if (MX and not TS) else 0), FP8, CFG.N, CFG.DV, transpose=True, swz=CFG.V_SWZ
    ).desc()
    sf_p = _sf_ptr(tmem_base, CFG.TMEM_SF_P) if MX else None
    sf_dot = _sf_ptr(tmem_base, CFG.TMEM_SF_DOT + do_state.idx * CFG.SF_COLS_V) if MX else None
    if cutlass.const_expr(P_SMEM):
        mma_ss(dvdesc, _ring(slab, CFG.P, ring.idx, False), dot, dv, tmem_sf_a=sf_p, tmem_sf_b=sf_dot, accumulate=accumulate)
    else:
        mma_ts(
            dvdesc,
            nvvm.make_tmem_ptr(tmem_base + CFG.TMEM_P + ring.idx * CFG.P_COLS, cutlass.Int8),
            dot,
            dv,
            tmem_sf_a=sf_p,
            tmem_sf_b=sf_dot,
            accumulate=accumulate,
        )
    _commit(bars.do_empty[do_state.idx])
    _commit(bars.p_empty[ring.idx])
    _wait_tcgen(bars.ds_full[ring.idx], ring.phase)
    if cutlass.const_expr(TS):
        _wait_tcgen(bars.qt_full[q_state.idx], q_state.phase)
        _qt_scales(slab, tmem_base, q_state.idx)
        _sf_copy(tmem_base, CFG.TMEM_SF_DS + ring.idx * CFG.SF_COLS_V, _sf_tile(slab, CFG.SF_DS + ring.idx * 512, 512).desc(), 1)
        qt = _operand(slab, CFG.Q + q_state.idx * CFG.Q_STAGE_BYTES, FP8, CFG.N, CFG.DQK, transpose=True, swz=CFG.QK_SWZ).desc()
    elif cutlass.const_expr(MX):
        _wait_tcgen(bars.qt_full[qt_state.idx], qt_state.phase)
        _qt_scales(slab, tmem_base, qt_state.idx)
        _sf_copy(tmem_base, CFG.TMEM_SF_DS + ring.idx * CFG.SF_COLS_V, _sf_tile(slab, CFG.SF_DS + ring.idx * 512, 512).desc(), 1)
        qt = _operand(slab, CFG.QT + qt_state.idx * CFG.Q_BYTES, FP8, CFG.N, CFG.DQK, transpose=True, swz=CFG.QK_SWZ).desc()
    else:
        qt = _operand(slab, CFG.Q + q_state.idx * CFG.Q_STAGE_BYTES, FP8, CFG.N, CFG.DQK, transpose=True, swz=CFG.QK_SWZ).desc()
    sf_ds = _sf_ptr(tmem_base, CFG.TMEM_SF_DS + ring.idx * CFG.SF_COLS_V) if MX else None
    sf_qt = _sf_ptr(tmem_base, CFG.TMEM_SF_QT + (q_state.idx if TS else qt_state.idx) * CFG.SF_COLS_QK) if MX else None
    mma_ss(dkdesc, _ring(slab, CFG.DS, ring.idx, False), qt, dk, tmem_sf_a=sf_ds, tmem_sf_b=sf_qt, accumulate=accumulate)
    if cutlass.const_expr(MX and not TS):
        _commit(bars.qt_empty[qt_state.idx])
    else:
        _commit(bars.q_empty[q_state.idx])
    _commit(bars.ds_empty[ring.idx])


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
    ratio: cutlass.Constexpr[int] = 1,
):
    score = _mma_desc(CFG.M, CFG.N, CFG.DQK)
    dpdesc = _mma_desc(CFG.M, CFG.N, CFG.DV)
    dvdesc = _mma_desc(CFG.M, CFG.DV, CFG.N, btranspose=True)
    dkdesc = _mma_desc(CFG.M, CFG.DQK, CFG.N, btranspose=True)
    kd = _operand(slab, CFG.K, FP8, CFG.M, CFG.DQK, swz=CFG.QK_SWZ).desc()
    vd = _operand(slab, CFG.V, FP8, CFG.M, CFG.DV, swz=CFG.V_SWZ).desc()
    s = nvvm.make_tmem_ptr(tmem_base + CFG.TMEM_S, cutlass.Float32)
    dp = nvvm.make_tmem_ptr(tmem_base + CFG.TMEM_DP, cutlass.Float32)
    dv = nvvm.make_tmem_ptr(tmem_base + CFG.TMEM_DV, cutlass.Float32)
    dk = nvvm.make_tmem_ptr(tmem_base + CFG.TMEM_DK, cutlass.Float32)
    sf_k = _sf_ptr(tmem_base, CFG.TMEM_SF_K) if MX else None
    sf_v = _sf_ptr(tmem_base, CFG.TMEM_SF_V) if MX else None
    if cutlass.const_expr(MX):
        lane = cute.arch.thread_idx()[0] % 32
        one = opaque_i32_zero() + cutlass.Int32(0x7F7F7F7F)
        st_shared_v4(_view(slab, CFG.SF_P + lane * 16, cutlass.Int32, 4).data_ptr(), (one, one, one, one), cutlass.Int32)
        nvvm.fence_proxy("async.shared", space="cta")
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
        _sf_copy(tmem_base, CFG.TMEM_SF_P, _sf_tile(slab, CFG.SF_P, 512).desc(), 1)
    q_state = PipelineState.start()
    qt_state = PipelineState.start()
    do_state = PipelineState.start()
    kv_state = PipelineState.start()
    ring = PipelineState.start()
    sfree_state = PipelineState.start()
    dpempty_state = PipelineState.start()
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit(bars.sched_empty[sched_state.idx])
        lo, hi = _bounds(block, sq, causal)
        if lo < hi:
            _wait_tcgen(bars.k_full, kv_state.phase)
            if cutlass.const_expr(MX):
                _sf_copy(tmem_base, CFG.TMEM_SF_K, _sf_tile(slab, CFG.SF_K, CFG.SF_QK_BYTES).desc(), CFG.SF_COLS_QK // 4)
            _wait_tcgen(bars.q_full[q_state.idx], q_state.phase)
            if cutlass.const_expr(MX):
                _q_scales(slab, tmem_base, q_state.idx)
            qd = _operand(slab, CFG.Q + q_state.idx * CFG.Q_STAGE_BYTES, FP8, CFG.N, CFG.DQK, swz=CFG.QK_SWZ).desc()
            mma_ss(score, kd, qd, s, tmem_sf_a=sf_k, tmem_sf_b=_sf_ptr(tmem_base, CFG.TMEM_SF_Q + q_state.idx * CFG.SF_COLS_QK) if MX else None)
            _commit(bars.s_done)
            if cutlass.const_expr(TS):
                _commit(bars.q_used[q_state.idx])
            elif cutlass.const_expr(MX):
                _commit(bars.q_empty[q_state.idx])
            _wait_tcgen(bars.v_full, kv_state.phase)
            if cutlass.const_expr(MX):
                _sf_copy(tmem_base, CFG.TMEM_SF_V, _sf_tile(slab, CFG.SF_V, CFG.SF_V_BYTES).desc(), 1)
            kv_state = advance(kv_state, 1)
        total = ratio * (hi - lo)
        t = cutlass.Int32(0)
        prev_q = q_state
        prev_qt = qt_state
        prev_do = do_state
        prev_ring = ring
        for _ in cutlass.range(0, ratio, 1, unroll=1):
            for qtile in cutlass.range(lo, hi, 1, unroll=1):
                q_next = advance(q_state, CFG.Q_STAGES)
                _wait_tcgen(bars.do_full[do_state.idx], do_state.phase)
                if cutlass.const_expr(MX):
                    _do_scales(slab, tmem_base, do_state.idx)
                dod = _operand(slab, CFG.DO + do_state.idx * CFG.DO_STAGE_BYTES, FP8, CFG.N, CFG.DV, swz=CFG.V_SWZ).desc()
                _wait_tcgen(bars.s_free, sfree_state.phase)
                sfree_state = advance(sfree_state, 1)
                mma_ss(dpdesc, vd, dod, dp, tmem_sf_a=sf_v, tmem_sf_b=_sf_ptr(tmem_base, CFG.TMEM_SF_DO + do_state.idx * CFG.SF_COLS_V) if MX else None)
                _commit(bars.dp_done)
                if cutlass.const_expr(TS):
                    _commit(bars.do_used[do_state.idx])
                if t > 0:
                    _grad_products(slab, bars, tmem_base, prev_do, prev_q, prev_qt, prev_ring, dvdesc, dkdesc, dv, dk, t > 1)
                _wait_tcgen(bars.dp_empty, dpempty_state.phase)
                dpempty_state = advance(dpempty_state, 1)
                if t + 1 < total:
                    _wait_tcgen(bars.q_full[q_next.idx], q_next.phase)
                    if cutlass.const_expr(MX):
                        _q_scales(slab, tmem_base, q_next.idx)
                    qd_next = _operand(slab, CFG.Q + q_next.idx * CFG.Q_STAGE_BYTES, FP8, CFG.N, CFG.DQK, swz=CFG.QK_SWZ).desc()
                    mma_ss(score, kd, qd_next, s, tmem_sf_a=sf_k, tmem_sf_b=_sf_ptr(tmem_base, CFG.TMEM_SF_Q + q_next.idx * CFG.SF_COLS_QK) if MX else None)
                    _commit(bars.s_done)
                    if cutlass.const_expr(TS):
                        _commit(bars.q_used[q_next.idx])
                    elif cutlass.const_expr(MX):
                        _commit(bars.q_empty[q_next.idx])
                prev_q = q_state
                prev_qt = qt_state
                prev_do = do_state
                prev_ring = ring
                q_state = q_next
                if cutlass.const_expr(MX and not TS):
                    qt_state = advance(qt_state, CFG.QT_STAGES)
                do_state = advance(do_state, CFG.DO_STAGES)
                ring = advance(ring, CFG.P_STAGES)
                t = t + 1
        if lo < hi:
            _grad_products(slab, bars, tmem_base, prev_do, prev_q, prev_qt, prev_ring, dvdesc, dkdesc, dv, dk, total > 1)
        _commit(bars.dvk_done)
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
    qt_desc,
    dot_desc,
    sf_q_desc,
    sf_qt_desc,
    sf_k_desc,
    sf_v_desc,
    sf_do_desc,
    sf_dot_desc,
    block,
    head,
    batch,
    sq,
    nk,
    nheads,
    batch_heads: cutlass.Constexpr[int],
    swizzle: cutlass.Constexpr[int],
    causal: cutlass.Constexpr[bool],
    ratio: cutlass.Constexpr[int] = 1,
):
    q_state = PipelineState.start(phase=1)
    qt_state = PipelineState.start(phase=1)
    do_state = PipelineState.start(phase=1)
    kv_state = PipelineState.start(phase=1)
    sched_state = PipelineState.start()
    pending = cutlass.Int32(0)
    pq_idx, pq_phase, pd_idx, pd_phase = cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0)
    pqtile, phq, pplane = cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0)
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit(bars.sched_empty[sched_state.idx])
        lo, hi = _bounds(block, sq, causal)
        _wait(bars.k_empty, kv_state.phase)
        kv_state = advance(kv_state, 1)
        if lo < hi:
            bars.k_full.arrive(n_bytes=CFG.K_TX, pred=nvvm.elect_sync())
            tma_load_tile(
                _operand(slab, CFG.K, FP8, CFG.M, CFG.DQK, swz=CFG.QK_SWZ),
                GmemTileTma(k_desc)(0, block * CFG.M, head, batch),
                bars.k_full.smem_ptr,
            )
            if cutlass.const_expr(MX):
                tma_load_tile(_sf_tile(slab, CFG.SF_K, CFG.SF_QK_BYTES), GmemTileTma(sf_k_desc)(0, 0, block, head, batch), bars.k_full.smem_ptr)
            bars.v_full.arrive(n_bytes=CFG.V_TX, pred=nvvm.elect_sync())
            tma_load_tile(_operand(slab, CFG.V, FP8, CFG.M, CFG.DV, swz=CFG.V_SWZ), GmemTileTma(v_desc)(0, block * CFG.M, head, batch), bars.v_full.smem_ptr)
            if cutlass.const_expr(MX):
                tma_load_tile(_sf_tile(slab, CFG.SF_V, CFG.SF_V_BYTES), GmemTileTma(sf_v_desc)(0, 0, block, head, batch), bars.v_full.smem_ptr)
        for r in cutlass.range(0, ratio, 1, unroll=1):
            hq = head * ratio + r
            for qtile in cutlass.range(lo, hi, 1, unroll=1):
                _wait(bars.q_empty[q_state.idx], q_state.phase)
                bars.q_full[q_state.idx].arrive(n_bytes=CFG.Q_TX, pred=nvvm.elect_sync())
                q_base = CFG.Q + q_state.idx * CFG.Q_STAGE_BYTES
                tma_load_tile(
                    _operand(slab, q_base, FP8, CFG.N, CFG.DQK, swz=CFG.QK_SWZ),
                    GmemTileTma(q_desc)(0, qtile * CFG.N, hq, batch),
                    bars.q_full[q_state.idx].smem_ptr,
                )
                if cutlass.const_expr(MX):
                    tma_load_tile(
                        _sf_tile(slab, CFG.SF_Q + q_state.idx * CFG.SF_QK_BYTES, CFG.SF_QK_BYTES),
                        GmemTileTma(sf_q_desc)(0, 0, qtile, hq, batch),
                        bars.q_full[q_state.idx].smem_ptr,
                    )
                _wait(bars.do_empty[do_state.idx], do_state.phase)
                bars.do_full[do_state.idx].arrive(n_bytes=CFG.DO_TX, pred=nvvm.elect_sync())
                do_base = CFG.DO + do_state.idx * CFG.DO_STAGE_BYTES
                tma_load_tile(
                    _operand(slab, do_base, FP8, CFG.N, CFG.DV, swz=CFG.V_SWZ),
                    GmemTileTma(do_desc)(0, qtile * CFG.N, hq, batch),
                    bars.do_full[do_state.idx].smem_ptr,
                )
                if cutlass.const_expr(MX):
                    tma_load_tile(
                        _sf_tile(slab, CFG.SF_DO + do_state.idx * CFG.SF_V_BYTES, CFG.SF_V_BYTES),
                        GmemTileTma(sf_do_desc)(0, 0, qtile, hq, batch),
                        bars.do_full[do_state.idx].smem_ptr,
                    )
                if cutlass.const_expr(TS):
                    if pending != 0:
                        _load_t(slab, bars, qt_desc, dot_desc, sf_qt_desc, sf_dot_desc, pq_idx, pq_phase, pd_idx, pd_phase, pqtile, pplane, phq, batch)
                    pending = cutlass.Int32(1)
                    pq_idx, pq_phase, pd_idx, pd_phase = q_state.idx, q_state.phase, do_state.idx, do_state.phase
                    pqtile, phq = qtile, hq
                    pplane = batch * (nheads * ratio) + hq
                elif cutlass.const_expr(MX):
                    tma_load_tile(
                        _operand(slab, do_base + CFG.DO_BYTES, FP8, CFG.N, CFG.DV, swz=CFG.V_SWZ),
                        GmemTileTma(dot_desc)(0, qtile * CFG.N, hq, batch),
                        bars.do_full[do_state.idx].smem_ptr,
                    )
                    tma_load_tile(
                        _sf_tile(slab, CFG.SF_DOT + do_state.idx * CFG.SF_V_BYTES, CFG.SF_V_BYTES),
                        GmemTileTma(sf_dot_desc)(0, 0, qtile, batch * (nheads * ratio) + hq, 0),
                        bars.do_full[do_state.idx].smem_ptr,
                    )
                    _wait(bars.qt_empty[qt_state.idx], qt_state.phase)
                    bars.qt_full[qt_state.idx].arrive(n_bytes=CFG.QT_TX, pred=nvvm.elect_sync())
                    tma_load_tile(
                        _operand(slab, CFG.QT + qt_state.idx * CFG.Q_BYTES, FP8, CFG.N, CFG.DQK, swz=CFG.QK_SWZ),
                        GmemTileTma(qt_desc)(0, qtile * CFG.N, hq, batch),
                        bars.qt_full[qt_state.idx].smem_ptr,
                    )
                    _load_sf_t(
                        slab, CFG.SF_QT + qt_state.idx * CFG.SF_QT_BYTES, sf_qt_desc, qtile, batch * (nheads * ratio) + hq, bars.qt_full[qt_state.idx].smem_ptr
                    )
                    qt_state = advance(qt_state, CFG.QT_STAGES)
                q_state = advance(q_state, CFG.Q_STAGES)
                do_state = advance(do_state, CFG.DO_STAGES)
        if cutlass.const_expr(TS):
            if pending != 0:
                _load_t(slab, bars, qt_desc, dot_desc, sf_qt_desc, sf_dot_desc, pq_idx, pq_phase, pd_idx, pd_phase, pqtile, pplane, phq, batch)
            pending = cutlass.Int32(0)
        block, head, batch, valid, sched_state = _next_tile(sched, bars, sched_state, nheads, nk, causal, swizzle, batch_heads)


@cute.jit
def _store_loop(
    slab,
    bars,
    sched,
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
    grad_dtype: cutlass.Constexpr,
):
    dk_swz = CFG.QK_SWZ if cutlass.const_expr(grad_dtype == FP8) else CFG.SWIZZLE_BYTES
    out_state = PipelineState.start()
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit(bars.sched_empty[sched_state.idx])
        _wait(bars.dv_full, out_state.phase)
        tma_store_tile(_operand(slab, CFG.OUT, grad_dtype, CFG.M, CFG.DV), GmemTileTma(dv_desc)(0, block * CFG.M, head, batch))
        _wait(bars.dk_full, out_state.phase)
        tma_store_tile(
            _operand(slab, CFG.OUT + CFG.M * CFG.DV * _bpe(grad_dtype), grad_dtype, CFG.M, CFG.DQK, swz=dk_swz),
            GmemTileTma(dk_desc)(0, block * CFG.M, head, batch),
        )
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
def bprop_d192_dkdv_fp8_kernel(
    q_desc: cutlass.GridConstant[tmap.TensorMap],
    k_desc: cutlass.GridConstant[tmap.TensorMap],
    v_desc: cutlass.GridConstant[tmap.TensorMap],
    do_desc: cutlass.GridConstant[tmap.TensorMap],
    dk_desc: cutlass.GridConstant[tmap.TensorMap],
    dv_desc: cutlass.GridConstant[tmap.TensorMap],
    qt_desc: cutlass.GridConstant[tmap.TensorMap],
    dot_desc: cutlass.GridConstant[tmap.TensorMap],
    sf_q_desc: cutlass.GridConstant[tmap.TensorMap],
    sf_qt_desc: cutlass.GridConstant[tmap.TensorMap],
    sf_k_desc: cutlass.GridConstant[tmap.TensorMap],
    sf_v_desc: cutlass.GridConstant[tmap.TensorMap],
    sf_do_desc: cutlass.GridConstant[tmap.TensorMap],
    sf_dot_desc: cutlass.GridConstant[tmap.TensorMap],
    lse: cute.Tensor,
    delta: cute.Tensor,
    descale_q: Optional[cute.Tensor],
    descale_k: Optional[cute.Tensor],
    descale_v: Optional[cute.Tensor],
    descale_s: Optional[cute.Tensor],
    scale_s: Optional[cute.Tensor],
    descale_do: Optional[cute.Tensor],
    descale_dp: Optional[cute.Tensor],
    scale_dk: Optional[cute.Tensor],
    scale_dv: Optional[cute.Tensor],
    scale_dp: Optional[cute.Tensor],
    amax_dk: Optional[cute.Tensor],
    amax_dv: Optional[cute.Tensor],
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
            block,
            head,
            batch,
            sq,
            skv,
            nk,
            nheads,
            batch_heads,
            swizzle,
            scale,
            causal,
            ratio,
        )
    elif warp < 12:
        _epilogue_loop(
            slab,
            bars,
            sched,
            tmem_base,
            descale_q,
            descale_s,
            descale_do,
            descale_dp,
            scale_dk,
            scale_dv,
            amax_dk,
            amax_dv,
            block,
            head,
            batch,
            sq,
            nk,
            nheads,
            batch_heads,
            swizzle,
            causal,
            grad_dtype,
        )
    elif warp == 12:
        _mma_loop(slab, bars, sched, tmem_base, block, head, batch, sq, nk, nheads, batch_heads, swizzle, causal, ratio)
    elif warp == 13:
        _load_loop(
            slab,
            bars,
            sched,
            q_desc,
            k_desc,
            v_desc,
            do_desc,
            qt_desc,
            dot_desc,
            sf_q_desc,
            sf_qt_desc,
            sf_k_desc,
            sf_v_desc,
            sf_do_desc,
            sf_dot_desc,
            block,
            head,
            batch,
            sq,
            nk,
            nheads,
            batch_heads,
            swizzle,
            causal,
            ratio,
        )
    elif warp == 14:
        _store_loop(slab, bars, sched, dk_desc, dv_desc, block, head, batch, sq, nk, nheads, batch_heads, swizzle, causal, grad_dtype)
    else:
        _scheduler_loop(bars, sched)
    nvvm.barrier_cta_sync()
    if warp == 0:
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
        nvvm.tcgen05_dealloc(nvvm.make_tmem_ptr(tmem_slot.load(), cutlass.Int8), cutlass.Int32(CFG.TMEM_COLS), group=nvvm.CTAGroup.CTA_1)


bprop_d192_dkdv_fp8_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def host(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    do: cute.Tensor,
    lse: cute.Tensor,
    delta: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    descale_q: Optional[cute.Tensor],
    descale_k: Optional[cute.Tensor],
    descale_v: Optional[cute.Tensor],
    descale_s: Optional[cute.Tensor],
    scale_s: Optional[cute.Tensor],
    descale_do: Optional[cute.Tensor],
    descale_dp: Optional[cute.Tensor],
    scale_dk: Optional[cute.Tensor],
    scale_dv: Optional[cute.Tensor],
    scale_dp: Optional[cute.Tensor],
    amax_dk: Optional[cute.Tensor],
    amax_dv: Optional[cute.Tensor],
    amax_dp: Optional[cute.Tensor],
    scale: cutlass.Float32,
    causal: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
    ratio: cutlass.Constexpr[int] = 1,
    qt: Optional[cute.Tensor] = None,
    dot: Optional[cute.Tensor] = None,
    sf_q: Optional[cute.Tensor] = None,
    sf_qt: Optional[cute.Tensor] = None,
    sf_k: Optional[cute.Tensor] = None,
    sf_v: Optional[cute.Tensor] = None,
    sf_do: Optional[cute.Tensor] = None,
    sf_dot: Optional[cute.Tensor] = None,
):
    if cutlass.const_expr(q.element_type != FP8 or k.element_type != FP8 or v.element_type != FP8 or do.element_type != FP8):
        raise ValueError("dense d192 fp8 dK / dV requires E4M3 Q/K/V/dO")
    grad_dtype = dk.element_type
    if cutlass.const_expr(grad_dtype not in (FP8,) + HALF_DTYPES or dv.element_type != grad_dtype):
        raise ValueError("dense d192 fp8 dK / dV requires E4M3, BF16 or FP16 dK/dV of one dtype")
    if cutlass.const_expr(grad_dtype == FP8 and CFG.DV != 128):
        raise ValueError("dense fp8 dK / dV writes E4M3 gradients at D_V 128 only")
    if cutlass.const_expr(
        q.shape[3] != CFG.DQK
        or k.shape[3] != CFG.DQK
        or v.shape[3] != CFG.DV
        or do.shape[3] != CFG.DV
        or dk.shape[3] != CFG.DQK
        or dv.shape[3] != CFG.DV
        or q.shape[2] != k.shape[2] * ratio
        or dk.shape[2] != k.shape[2]
    ):
        raise ValueError("dense fp8 dK / dV requires the configured D_QK / D_V with H_q = ratio * H_kv and per-KV-head dK/dV")
    if cutlass.const_expr(q.shape[1] <= 0 or k.shape[1] <= 0):
        raise ValueError("dense d192 fp8 dK / dV requires positive Sq/Skv")
    nk = (k.shape[1] + CFG.M - 1) // CFG.M
    heads_per_l2 = (50 * 1024 * 1024) // (q.shape[1] * (CFG.DQK + CFG.DV) * CFG.BPE)
    swizzle = 1 << (heads_per_l2.bit_length() - 1) if heads_per_l2 > 0 else 1
    grid = (nk * k.shape[2] * q.shape[0], 1, 1) if causal else (nk, k.shape[2], q.shape[0])
    q_map = _tensor_map(q, CFG.N, CFG.QK_BOX, CFG.QK_SWZ)
    do_map = _tensor_map(do, CFG.N, CFG.V_BOX, CFG.V_SWZ)
    if cutlass.const_expr(MX):
        qt_map = _tensor_map(qt, CFG.N, CFG.QK_BOX, CFG.QK_SWZ)
        dot_map = _tensor_map(dot, CFG.N, CFG.V_BOX, CFG.V_SWZ)
        sf_maps = (_sf_map(sf_q), _sf_map(sf_qt), _sf_map(sf_k), _sf_map(sf_v), _sf_map(sf_do), _sf_map(sf_dot))
    else:
        qt_map = q_map
        dot_map = do_map
        sf_maps = (q_map,) * 6
    bprop_d192_dkdv_fp8_kernel(
        q_map,
        _tensor_map(k, CFG.M, CFG.QK_BOX, CFG.QK_SWZ),
        _tensor_map(v, CFG.M, CFG.V_BOX, CFG.V_SWZ),
        do_map,
        _tensor_map(dk, CFG.M, CFG.QK_BOX, CFG.QK_SWZ) if cutlass.const_expr(grad_dtype == FP8) else _tensor_map(dk, CFG.M, _box(grad_dtype)),
        _tensor_map(dv, CFG.M, _box(grad_dtype)),
        qt_map,
        dot_map,
        sf_maps[0],
        sf_maps[1],
        sf_maps[2],
        sf_maps[3],
        sf_maps[4],
        sf_maps[5],
        lse,
        delta,
        descale_q,
        descale_k,
        descale_v,
        descale_s,
        scale_s,
        descale_do,
        descale_dp,
        scale_dk,
        scale_dv,
        scale_dp,
        amax_dk,
        amax_dv,
        amax_dp,
        q.shape[1],
        k.shape[1],
        k.shape[2],
        q.shape[0] * k.shape[2],
        scale,
        causal,
        swizzle,
        grad_dtype,
        ratio,
    ).launch(grid=grid, block=(CFG.THREADS, 1, 1), stream=stream, min_blocks_per_mp=1)
