# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Dense d128 per-tensor FP8 backward with FP32 dQ accumulation and native dK/dV stores."""

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
from cudnn.frost.tile_dsl.mma import mma_ss
from cudnn.frost.tile_dsl.pointwise import abs_max_tree, ffma2, fmax_f32, fmul2, fp32_to_fp16, fp32_to_fp8_pack, opaque_f32_zero
from cudnn.frost.tile_dsl.scheduler import Sched, read_clc_payload
from cudnn.frost.tile_dsl.tma import tma_load_tile, tma_store_commit, tma_store_tile, tma_store_wait

from cudnn.sdpa.bwd.config_sm100_d128_accum_fp8 import make_bar_layout, make_config

_PARAMS = globals().get("FROST_TEMPLATE_PARAMS", 128)
HEAD_DIM, WINDOW = (_PARAMS, 0) if isinstance(_PARAMS, int) else (int(_PARAMS[0]), int(_PARAMS[1]))
CFG = make_config(HEAD_DIM)
BAR_LAYOUT = make_bar_layout(CFG)

LOG2E = 1.4426950408889634
FP8 = cutlass.Float8E4M3FN
HALF_DTYPES = (cutlass.BFloat16, cutlass.Float16)


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
    s_free: MBarrier
    dp_empty: MBarrier
    p_full: MBarrier
    p_empty: MBarrier
    ds_full: MBarrier
    ds_empty: MBarrier
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


def _bpe(dtype):
    if dtype == cutlass.Float32:
        return 4
    if dtype in HALF_DTYPES:
        return 2
    return 1


def _box(dtype):
    return CFG.SWIZZLE_BYTES // _bpe(dtype)


def _dk_stage(grad_dtype):
    return CFG.SLICE_BYTES * ((CFG.M * CFG.D * _bpe(grad_dtype) + CFG.SLICE_BYTES - 1) // CFG.SLICE_BYTES)


def _out_swz(grad_dtype):
    return min(CFG.D * _bpe(grad_dtype), CFG.SWIZZLE_BYTES)


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


def _mma_desc(m, n, k, *, atranspose=False, btranspose=False):
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


def _tensor_map(tensor, box, swz=CFG.SWIZZLE_BYTES):
    swizzle = tmap.TensorMapSwizzle.s128b if swz == CFG.SWIZZLE_BYTES else tmap.TensorMapSwizzle.s64b
    return tmap.create_tensor_map_tiled_from_view(tensor, box_dims=(1, CFG.M, 1, box), stride_order=(3, 1, 2, 0), swizzle=swizzle)


@cute.jit
def _bounds(block, sq, causal: cutlass.Constexpr[bool]):
    lo = cutlass.Int32(0)
    if cutlass.const_expr(causal):
        lo = block
    hi = (sq + CFG.N - 1) // CFG.N
    if cutlass.const_expr(WINDOW > 0):
        hi = cute.math.min(hi, (block * CFG.M + CFG.M + WINDOW - 1 + CFG.N - 1) // CFG.N)
    return lo, hi


@cute.jit
def _stats(slab, lse, delta, row, group, qbase, sq, batch, head, delta_scale, lse_shift):
    named_barrier_fence(CFG.SOFTMAX_BARRIER, CFG.SOFTMAX_LANES)
    idx = qbase + row
    valid = idx < sq
    if group == 0:
        clamped = cutlass.Int32(arith.select(valid.ir_value(), idx.ir_value(), (sq - 1).ir_value()))
        val = cutlass.Float32(lse[batch, head, clamped])
        stat = -cutlass.Float32(LOG2E) * val + lse_shift
        if val == cutlass.Float32(float("-inf")):
            stat = lse_shift
        stat = cutlass.Float32(arith.select(valid.ir_value(), stat.ir_value(), cutlass.Float32(float("-inf")).ir_value()))
        _view(slab, CFG.STATS, cutlass.Float32, CFG.N).subview(row).store(stat)
    else:
        stat = -delta_scale * cutlass.Float32(delta[batch, head, idx])
        _view(slab, CFG.DELTA, cutlass.Float32, CFG.N).subview(row).store(stat)
    named_barrier_fence(CFG.SOFTMAX_BARRIER, CFG.SOFTMAX_LANES)


@cute.jit
def _masked_scores(
    scores, row, group, block, qtile, skv, causal_mask: cutlass.Constexpr[bool], key_mask: cutlass.Constexpr[bool], window_mask: cutlass.Constexpr[bool]
):
    neg_inf = cutlass.Float32(float("-inf"))
    kr = block * CFG.M + row
    key_valid = kr < skv
    out = []
    for i in cutlass.range_constexpr(64):
        v = scores[i]
        if cutlass.const_expr(causal_mask or window_mask):
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
    s_scale,
    causal_mask: cutlass.Constexpr[bool],
    key_mask: cutlass.Constexpr[bool],
    window_mask: cutlass.Constexpr[bool],
):
    if cutlass.const_expr(causal_mask or key_mask or window_mask):
        scores = _masked_scores(scores, row, group, block, qtile, skv, causal_mask, key_mask, window_mask)
    p = []
    scaled = s_scale * cutlass.Float32(LOG2E)
    for chunk in cutlass.range_constexpr(16):
        stat = _view(slab, CFG.STATS, cutlass.Float32, CFG.N).subview(group * 64 + chunk * 4).load(vector_size=4, alignment=16)
        for pair in cutlass.range_constexpr(2):
            i = chunk * 4 + pair * 2
            e0, e1 = ffma2(scaled, scaled, scores[i], scores[i + 1], stat[pair * 2], stat[pair * 2 + 1])
            p0 = cute.math.exp2(e0, fastmath=True)
            p1 = cute.math.exp2(e1, fastmath=True)
            p.extend((p0, p1))
    return cutlass.Vector.from_elements(tuple(p), cutlass.Float32), _pack_fp8(p)


@cute.jit
def _ds_values(slab, dp, group, p, dp_scale, amax, fold: cutlass.Constexpr[bool]):
    ds = []
    for chunk in cutlass.range_constexpr(16):
        stat = _view(slab, CFG.DELTA, cutlass.Float32, CFG.N).subview(group * 64 + chunk * 4).load(vector_size=4, alignment=16)
        for pair in cutlass.range_constexpr(2):
            i = chunk * 4 + pair * 2
            d0, d1 = ffma2(dp_scale, dp_scale, dp[i], dp[i + 1], stat[pair * 2], stat[pair * 2 + 1])
            d0, d1 = fmul2(d0, d1, p[i], p[i + 1])
            ds.extend((d0, d1))
    if cutlass.const_expr(fold):
        amax = fmax_f32(amax, abs_max_tree(ds))
    return amax, _pack_fp8(ds)


@cute.jit
def _store_half_row(slab, base, row, group, words):
    for vec in cutlass.range_constexpr(4):
        chunk = group * 4 + vec
        off = base + row * CFG.SWIZZLE_BYTES + ((row % 8) ^ chunk) * 16
        values = cutlass.Vector.from_elements(tuple(words[vec * 4 + k] for k in range(4)), cutlass.Int32)
        Pointer(slab.data_ptr() + off, dtype=cutlass.Int32).store(values, alignment=16)


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
    amax,
    fold: cutlass.Constexpr[bool],
    causal_mask: cutlass.Constexpr[bool],
    key_mask: cutlass.Constexpr[bool],
    window_mask: cutlass.Constexpr[bool],
):
    _wait_tcgen(bars.s_done, phase)
    address = tmem_base + group * 64
    a = _tmem_load(address, 32)
    b = _tmem_load(address + 32, 32)
    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
    nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
    bars.s_free.arrive()
    scores = cutlass.Vector.from_elements(tuple(a[i] for i in range(32)) + tuple(b[i] for i in range(32)), cutlass.Float32)
    p, p_words = _p_values(slab, scores, row, group, block, qtile, skv, s_scale, causal_mask, key_mask, window_mask)
    _wait(bars.p_empty[ring_idx], ring_phase ^ 1)
    _store_half_row(slab, CFG.P + ring_idx * CFG.RING_BYTES, row, group, p_words)
    nvvm.fence_proxy("async.shared", space="cta")
    bars.p_full[ring_idx].arrive()
    _wait_tcgen(bars.dp_done, phase)
    a = _tmem_load(address + 128, 32)
    b = _tmem_load(address + 160, 32)
    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
    nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
    dp = cutlass.Vector.from_elements(tuple(a[i] for i in range(32)) + tuple(b[i] for i in range(32)), cutlass.Float32)
    amax, ds_words = _ds_values(slab, dp, group, p, dp_scale, amax, fold)
    _wait(bars.ds_empty[ring_idx], ring_phase ^ 1)
    _store_half_row(slab, CFG.DS + ring_idx * CFG.RING_BYTES, row, group, ds_words)
    nvvm.fence_proxy("async.shared", space="cta")
    bars.ds_full[ring_idx].arrive()
    if qtile + 1 < hi:
        _stats(slab, lse, delta, row, group, (qtile + 1) * CFG.N, sq, batch, head, delta_scale, lse_shift)
    next_idx = ring_idx + 1
    wrap = next_idx == CFG.P_STAGES
    ring_idx = cutlass.Int32(arith.select(wrap.ir_value(), cutlass.Int32(0).ir_value(), next_idx.ir_value()))
    ring_phase = cutlass.Int32(arith.select(wrap.ir_value(), (ring_phase ^ 1).ir_value(), ring_phase.ir_value()))
    return phase ^ 1, ring_idx, ring_phase, amax


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
    amax,
    fold: cutlass.Constexpr[bool],
    causal: cutlass.Constexpr[bool],
    key_mask: cutlass.Constexpr[bool],
):
    diag_end = lo
    if cutlass.const_expr(causal):
        diag_end = cute.math.min(lo + 1, hi)
        for qtile in cutlass.range(lo, diag_end, 1, unroll=1):
            phase, ring_idx, ring_phase, amax = _softmax_tile(
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
        phase, ring_idx, ring_phase, amax = _softmax_tile(
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
            amax,
            fold,
            False,
            key_mask,
            False,
        )
    if cutlass.const_expr(WINDOW > 0):
        for qtile in cutlass.range(edge_start, hi, 1, unroll=1):
            phase, ring_idx, ring_phase, amax = _softmax_tile(
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
                amax,
                fold,
                False,
                key_mask,
                True,
            )
    return phase, ring_idx, ring_phase, amax


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
):
    nvvm.setmaxregister(CFG.SOFTMAX_REGS, nvvm.SetMaxRegisterAction.INCREASE)
    tid = cute.arch.thread_idx()[0]
    row = tid % CFG.M
    group = tid // CFG.M
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
        if cutlass.const_expr(not causal):
            _credit(bars.sched_empty[sched_state.idx])
        lo, hi = _bounds(block, sq, causal)
        if lo < hi:
            _stats(slab, lse, delta, row, group, lo * CFG.N, sq, batch, head, delta_scale, lse_shift)
        if (block + 1) * CFG.M > skv:
            phase, ring_idx, ring_phase, amax = _softmax_tiles(
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
                amax,
                fold,
                causal,
                True,
            )
        else:
            phase, ring_idx, ring_phase, amax = _softmax_tiles(
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
def _store_chunks(slab, byte_offset, row, words, first: cutlass.Constexpr[int], count: cutlass.Constexpr[int], swz: cutlass.Constexpr[int] = 128):
    for vec in cutlass.range_constexpr(count):
        chunk = first + vec
        if cutlass.const_expr(swz == 128):
            off = byte_offset + row * swz + ((row % 8) ^ chunk) * 16
        else:
            off = byte_offset + row * swz + (((row // 2) % 4) ^ chunk) * 16
        values = cutlass.Vector.from_elements(tuple(words[vec * 4 + k] for k in range(4)), cutlass.Int32)
        Pointer(slab.data_ptr() + off, dtype=cutlass.Int32).store(values, alignment=16)


@cute.jit
def _store_words(slab, byte_offset, row, words):
    _store_chunks(slab, byte_offset, row, words, 0, 8, 128)


@cute.jit
def _store_dq_slice(slab, row, values, index: cutlass.Constexpr[int]):
    for vec in cutlass.range_constexpr(8):
        off = CFG.ALIAS + index * CFG.SLICE_BYTES + row * CFG.SWIZZLE_BYTES + ((row % 8) ^ vec) * 16
        v = cutlass.Vector.from_elements(tuple(values[index * 32 + vec * 4 + k] for k in range(4)), cutlass.Float32)
        Pointer(slab.data_ptr() + off, dtype=cutlass.Float32).store(v, alignment=16)


@cute.jit
def _native_epilogue(
    slab,
    tmem_base,
    bars,
    row,
    live,
    empty_phase,
    grad_dtype: cutlass.Constexpr,
    dk: cutlass.Constexpr[bool],
    factor,
    out_scale,
    amax,
    fold: cutlass.Constexpr[bool],
):
    plane = CFG.TMEM_DK if cutlass.const_expr(dk) else CFG.TMEM_DV
    fp8_out = cutlass.const_expr(grad_dtype == FP8)
    base = CFG.ALIAS + (_dk_stage(grad_dtype) if cutlass.const_expr(dk) else 0)
    last_slice = (base - CFG.ALIAS + CFG.M * CFG.D * _bpe(grad_dtype) - 1) // CFG.SLICE_BYTES
    true_values = []
    for half in cutlass.range_constexpr(CFG.D // 64):
        vals = cutlass.Vector.from_elements(tuple(cutlass.Float32(0.0) for _ in range(64)), cutlass.Float32)
        if live:
            vals = _tmem_load(tmem_base + plane + half * 64, 64)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
        scaled = _scale_pairs([vals[i] for i in range(64)], factor)
        if cutlass.const_expr(fold):
            amax = fmax_f32(amax, abs_max_tree(scaled))
        if cutlass.const_expr(fp8_out):
            true_values.extend(scaled)
        else:
            words = cutlass.Vector.from_elements(tuple(fp32_to_fp16(scaled[2 * i], scaled[2 * i + 1], dtype=grad_dtype) for i in range(32)), cutlass.Int32)
            if cutlass.const_expr(half == 0):
                _wait(bars.dq_empty[last_slice], empty_phase)
            _store_words(slab, base + half * CFG.SLICE_BYTES, row, words)
    if cutlass.const_expr(fp8_out):
        words = _pack_fp8(_scale_pairs(true_values, out_scale))
        _wait(bars.dq_empty[last_slice], empty_phase)
        _store_chunks(slab, base, row, words, 0, CFG.D // 16, _out_swz(FP8))
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
    nvvm.setmaxregister(CFG.EPILOGUE_REGS, nvvm.SetMaxRegisterAction.INCREASE)
    row = cute.arch.thread_idx()[0] % CFG.M
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
    dq_state = PipelineState.start()
    empty_state = PipelineState.start()
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit(bars.sched_empty[sched_state.idx])
        lo, hi = _bounds(block, sq, causal)
        for _ in cutlass.range(lo, hi, 1, unroll=1):
            _wait_tcgen(bars.dq_done, dq_state.phase)
            halves = []
            for half in cutlass.range_constexpr(CFG.D // 64):
                halves.append(_tmem_load(tmem_base + 128 + half * 64, 64))
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
            bars.dp_empty.arrive()
            vals = cutlass.Vector.from_elements(tuple(h[i] for h in halves for i in range(64)), cutlass.Float32)
            for i in cutlass.range_constexpr(CFG.DQ_SLICES):
                _wait(bars.dq_empty[i], empty_state.phase)
                _store_dq_slice(slab, row, vals, i)
                nvvm.fence_proxy("async.shared", space="cta")
                bars.dq_full[i].arrive()
            dq_state = advance(dq_state, 1)
            empty_state = advance(empty_state, 1)
        if lo >= hi:
            _wait_tcgen(bars.dq_done, dq_state.phase)
            dq_state = advance(dq_state, 1)
        for i in cutlass.range_constexpr(CFG.DQ_SLICES):
            _wait(bars.dq_empty[i], empty_state.phase)
        amax_v = _native_epilogue(slab, tmem_base, bars, row, lo < hi, empty_state.phase, grad_dtype, False, dv_factor, dv_out, amax_v, fold_v)
        amax_k = _native_epilogue(slab, tmem_base, bars, row, lo < hi, empty_state.phase, grad_dtype, True, dk_factor, dk_out, amax_k, fold_k)
        empty_state = advance(empty_state, 1)
        block, head, batch, valid, sched_state = _next_tile(sched, bars, sched_state, nheads, nk, causal, swizzle, batch_heads)
    if cutlass.const_expr(fold_v):
        _fold_amax(amax_v, amax_dv)
    if cutlass.const_expr(fold_k):
        _fold_amax(amax_k, amax_dk)


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
):
    nvvm.setmaxregister(CFG.SERVICE_REGS, nvvm.SetMaxRegisterAction.DECREASE)
    score = _mma_desc(CFG.M, CFG.N, CFG.D)
    grad = _mma_desc(CFG.M, CFG.D, CFG.N, btranspose=True)
    dqdesc = _mma_desc(CFG.N, CFG.D, CFG.M, atranspose=True, btranspose=True)
    kd = _operand(slab, CFG.K, FP8, CFG.M, CFG.D, swz=CFG.IN_SWZ).desc()
    vd = _operand(slab, CFG.V, FP8, CFG.M, CFG.D, swz=CFG.IN_SWZ).desc()
    dod = _operand(slab, CFG.DO, FP8, CFG.N, CFG.D, swz=CFG.IN_SWZ).desc()
    dot = _operand(slab, CFG.DO, FP8, CFG.N, CFG.D, transpose=True, swz=CFG.IN_SWZ).desc()
    kt = _operand(slab, CFG.K, FP8, CFG.M, CFG.D, transpose=True, swz=CFG.IN_SWZ).desc()
    s = nvvm.make_tmem_ptr(tmem_base, cutlass.Float32)
    dp = nvvm.make_tmem_ptr(tmem_base + 128, cutlass.Float32)
    dv = nvvm.make_tmem_ptr(tmem_base + CFG.TMEM_DV, cutlass.Float32)
    dk = nvvm.make_tmem_ptr(tmem_base + CFG.TMEM_DK, cutlass.Float32)
    q_state = PipelineState.start()
    do_state = PipelineState.start()
    kv_state = PipelineState.start()
    ring = PipelineState.start()
    sfree_state = PipelineState.start()
    dp_empty = PipelineState.start(phase=1)
    out_empty = PipelineState.start(phase=1)
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit(bars.sched_empty[sched_state.idx])
        lo, hi = _bounds(block, sq, causal)
        if lo < hi:
            _wait_tcgen(bars.k_full, kv_state.phase)
            _wait_tcgen(bars.q_full[q_state.idx], q_state.phase)
            qd = _operand(slab, CFG.Q + q_state.idx * CFG.OPERAND_BYTES, FP8, CFG.N, CFG.D, swz=CFG.IN_SWZ).desc()
            mma_ss(score, kd, qd, s)
            _commit(bars.s_done)
            _wait_tcgen(bars.v_full, kv_state.phase)
        _wait_tcgen(bars.dvk_empty, out_empty.phase)
        out_empty = advance(out_empty, 1)
        for qtile in cutlass.range(lo, hi, 1, unroll=1):
            _wait_tcgen(bars.dp_empty, dp_empty.phase)
            _wait_tcgen(bars.do_full, do_state.phase)
            mma_ss(score, vd, dod, dp)
            _commit(bars.dp_done)
            _wait_tcgen(bars.p_full[ring.idx], ring.phase)
            pd = _operand(slab, CFG.P + ring.idx * CFG.RING_BYTES, FP8, CFG.M, CFG.N).desc()
            mma_ss(grad, pd, dot, dv, accumulate=qtile > lo)
            _commit(bars.do_empty)
            _commit(bars.p_empty[ring.idx])
            q_next = advance(q_state, CFG.Q_STAGES)
            _wait_tcgen(bars.s_free, sfree_state.phase)
            sfree_state = advance(sfree_state, 1)
            if qtile + 1 < hi:
                _wait_tcgen(bars.q_full[q_next.idx], q_next.phase)
                qd_next = _operand(slab, CFG.Q + q_next.idx * CFG.OPERAND_BYTES, FP8, CFG.N, CFG.D, swz=CFG.IN_SWZ).desc()
                mma_ss(score, kd, qd_next, s)
                _commit(bars.s_done)
            _wait_tcgen(bars.ds_full[ring.idx], ring.phase)
            dsd = _operand(slab, CFG.DS + ring.idx * CFG.RING_BYTES, FP8, CFG.M, CFG.N).desc()
            dst = _operand(slab, CFG.DS + ring.idx * CFG.RING_BYTES, FP8, CFG.M, CFG.N, transpose=True).desc()
            mma_ss(dqdesc, dst, kt, dp)
            _commit(bars.dq_done)
            qt = _operand(slab, CFG.Q + q_state.idx * CFG.OPERAND_BYTES, FP8, CFG.N, CFG.D, transpose=True, swz=CFG.IN_SWZ).desc()
            mma_ss(grad, dsd, qt, dk, accumulate=qtile > lo)
            _commit(bars.q_empty[q_state.idx])
            _commit(bars.ds_empty[ring.idx])
            q_state = q_next
            do_state = advance(do_state, 1)
            dp_empty = advance(dp_empty, 1)
            ring = advance(ring, CFG.P_STAGES)
        if lo < hi:
            _commit(bars.k_empty)
            _commit(bars.v_empty)
            kv_state = advance(kv_state, 1)
        else:
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
    ratio: cutlass.Constexpr[int] = 1,
):
    nvvm.setmaxregister(CFG.SERVICE_REGS, nvvm.SetMaxRegisterAction.DECREASE)
    q_state = PipelineState.start(phase=1)
    do_state = PipelineState.start(phase=1)
    kv_state = PipelineState.start(phase=1)
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit(bars.sched_empty[sched_state.idx])
        lo, hi = _bounds(block, sq, causal)
        if lo < hi:
            _wait(bars.k_empty, kv_state.phase)
            bars.k_full.arrive(n_bytes=CFG.OPERAND_BYTES, pred=nvvm.elect_sync())
            tma_load_tile(
                _operand(slab, CFG.K, FP8, CFG.M, CFG.D, swz=CFG.IN_SWZ), GmemTileTma(k_desc)(0, block * CFG.M, head // ratio, batch), bars.k_full.smem_ptr
            )
            _wait(bars.v_empty, kv_state.phase)
            bars.v_full.arrive(n_bytes=CFG.OPERAND_BYTES, pred=nvvm.elect_sync())
            tma_load_tile(
                _operand(slab, CFG.V, FP8, CFG.M, CFG.D, swz=CFG.IN_SWZ), GmemTileTma(v_desc)(0, block * CFG.M, head // ratio, batch), bars.v_full.smem_ptr
            )
            kv_state = advance(kv_state, 1)
        for qtile in cutlass.range(lo, hi, 1, unroll=1):
            _wait(bars.q_empty[q_state.idx], q_state.phase)
            bars.q_full[q_state.idx].arrive(n_bytes=CFG.OPERAND_BYTES, pred=nvvm.elect_sync())
            tma_load_tile(
                _operand(slab, CFG.Q + q_state.idx * CFG.OPERAND_BYTES, FP8, CFG.N, CFG.D, swz=CFG.IN_SWZ),
                GmemTileTma(q_desc)(0, qtile * CFG.N, head, batch),
                bars.q_full[q_state.idx].smem_ptr,
            )
            _wait(bars.do_empty, do_state.phase)
            bars.do_full.arrive(n_bytes=CFG.OPERAND_BYTES, pred=nvvm.elect_sync())
            tma_load_tile(_operand(slab, CFG.DO, FP8, CFG.N, CFG.D, swz=CFG.IN_SWZ), GmemTileTma(do_desc)(0, qtile * CFG.N, head, batch), bars.do_full.smem_ptr)
            q_state = advance(q_state, CFG.Q_STAGES)
            do_state = advance(do_state, 1)
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
    grad_dtype: cutlass.Constexpr,
):
    nvvm.setmaxregister(CFG.SERVICE_REGS, nvvm.SetMaxRegisterAction.DECREASE)
    dq_state = PipelineState.start()
    out_state = PipelineState.start()
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit(bars.sched_empty[sched_state.idx])
        for i in cutlass.range_constexpr(CFG.DQ_SLICES):
            bars.dq_empty[i].arrive()
        lo, hi = _bounds(block, sq, causal)
        for qtile in cutlass.range(lo, hi, 1, unroll=1):
            for i in cutlass.range_constexpr(CFG.DQ_SLICES):
                _wait(bars.dq_full[i], dq_state.phase)
                tma_reduce_subtile(_operand(slab, CFG.ALIAS, cutlass.Float32, CFG.M, CFG.D), GmemTileTma(dq_desc)(0, qtile * CFG.N, head, batch), i)
                tma_store_commit()
            for i in cutlass.range_constexpr(CFG.DQ_SLICES):
                tma_store_wait(CFG.DQ_SLICES - 1 - i)
                bars.dq_empty[i].arrive()
            dq_state = advance(dq_state, 1)
        _wait(bars.dv_full, out_state.phase)
        tma_store_tile(_operand(slab, CFG.ALIAS, grad_dtype, CFG.M, CFG.D, swz=_out_swz(grad_dtype)), GmemTileTma(dv_desc)(0, block * CFG.M, head, batch))
        _wait(bars.dk_full, out_state.phase)
        tma_store_tile(
            _operand(slab, CFG.ALIAS + _dk_stage(grad_dtype), grad_dtype, CFG.M, CFG.D, swz=_out_swz(grad_dtype)),
            GmemTileTma(dk_desc)(0, block * CFG.M, head, batch),
        )
        out_state = advance(out_state, 1)
        tma_store_commit()
        tma_store_wait(0)
        bars.dvk_empty.arrive()
        block, head, batch, valid, sched_state = _next_tile(sched, bars, sched_state, nheads, nk, causal, swizzle, batch_heads)


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
def bprop_d128_accum_fp8_kernel(
    q_desc: cutlass.GridConstant[tmap.TensorMap],
    k_desc: cutlass.GridConstant[tmap.TensorMap],
    v_desc: cutlass.GridConstant[tmap.TensorMap],
    do_desc: cutlass.GridConstant[tmap.TensorMap],
    dq_desc: cutlass.GridConstant[tmap.TensorMap],
    dk_desc: cutlass.GridConstant[tmap.TensorMap],
    dv_desc: cutlass.GridConstant[tmap.TensorMap],
    lse: cute.Tensor,
    delta: cute.Tensor,
    descale_q: cute.Tensor,
    descale_k: cute.Tensor,
    descale_v: cute.Tensor,
    descale_s: cute.Tensor,
    scale_s: cute.Tensor,
    descale_do: cute.Tensor,
    descale_dp: cute.Tensor,
    scale_dk: cute.Tensor,
    scale_dv: cute.Tensor,
    scale_dp: cute.Tensor,
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
        _mma_loop(slab, bars, sched, tmem_base, block, head, batch, sq, nk, nheads, batch_heads, swizzle, causal)
    elif warp == 13:
        _load_loop(slab, bars, sched, q_desc, k_desc, v_desc, do_desc, block, head, batch, sq, nk, nheads, batch_heads, swizzle, causal, ratio)
    elif warp == 14:
        _store_loop(slab, bars, sched, dq_desc, dk_desc, dv_desc, block, head, batch, sq, nk, nheads, batch_heads, swizzle, causal, grad_dtype)
    else:
        _scheduler_loop(bars, sched)
    nvvm.barrier_cta_sync()
    if warp == 0:
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
        nvvm.tcgen05_dealloc(nvvm.make_tmem_ptr(tmem_slot.load(), cutlass.Int8), cutlass.Int32(CFG.TMEM_COLS), group=nvvm.CTAGroup.CTA_1)


bprop_d128_accum_fp8_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


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
    descale_q: cute.Tensor,
    descale_k: cute.Tensor,
    descale_v: cute.Tensor,
    descale_s: cute.Tensor,
    scale_s: cute.Tensor,
    descale_do: cute.Tensor,
    descale_dp: cute.Tensor,
    scale_dk: cute.Tensor,
    scale_dv: cute.Tensor,
    scale_dp: cute.Tensor,
    amax_dk: Optional[cute.Tensor],
    amax_dv: Optional[cute.Tensor],
    amax_dp: Optional[cute.Tensor],
    scale: cutlass.Float32,
    causal: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
    ratio: cutlass.Constexpr[int] = 1,
):
    if cutlass.const_expr(q.element_type != FP8 or k.element_type != FP8 or v.element_type != FP8 or do.element_type != FP8):
        raise ValueError("dense d128 fp8 accumulation requires E4M3 Q/K/V/dO")
    grad_dtype = dk.element_type
    if cutlass.const_expr(grad_dtype not in (FP8,) + HALF_DTYPES or dv.element_type != grad_dtype):
        raise ValueError("dense d128 fp8 accumulation requires E4M3, BF16 or FP16 dK/dV of one dtype")
    if cutlass.const_expr(q.shape[3] != CFG.D or k.shape[3] != CFG.D or q.shape[2] != k.shape[2] * ratio or dk.shape[2] != q.shape[2]):
        raise ValueError("dense d128 fp8 accumulation requires d128 with H_q = ratio * H_kv and per-Q-head dK/dV")
    if cutlass.const_expr(q.shape[1] <= 0 or k.shape[1] <= 0):
        raise ValueError("dense d128 fp8 accumulation requires positive Sq/Skv")
    nk = (k.shape[1] + CFG.M - 1) // CFG.M
    heads_per_l2 = (50 * 1024 * 1024) // (q.shape[1] * (CFG.D + CFG.D) * CFG.BPE)
    swizzle = 1 << (heads_per_l2.bit_length() - 1) if heads_per_l2 > 0 else 1
    grid = (nk * q.shape[2] * q.shape[0], 1, 1) if causal else (nk, q.shape[2], q.shape[0])
    bprop_d128_accum_fp8_kernel(
        _tensor_map(q, CFG.HEAD_BOX, CFG.IN_SWZ),
        _tensor_map(k, CFG.HEAD_BOX, CFG.IN_SWZ),
        _tensor_map(v, CFG.HEAD_BOX, CFG.IN_SWZ),
        _tensor_map(do, CFG.HEAD_BOX, CFG.IN_SWZ),
        _tensor_map(dq, CFG.DQ_BOX),
        _tensor_map(dk, _out_swz(grad_dtype) // _bpe(grad_dtype), _out_swz(grad_dtype)),
        _tensor_map(dv, _out_swz(grad_dtype) // _bpe(grad_dtype), _out_swz(grad_dtype)),
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
        q.shape[2],
        q.shape[0] * q.shape[2],
        scale,
        causal,
        swizzle,
        grad_dtype,
        ratio,
    ).launch(grid=grid, block=(CFG.THREADS, 1, 1), stream=stream, min_blocks_per_mp=1)
