# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
# Adapted from sdpa/fwd/kernels/sm107/sparse_d256_f16.py.
"""QSA forward for SM100/103: gathered four-token blocks and packed GQA.

Each CTA processes one query, or up to 32 in gated tiles, for one KV head.
Sixteen gather warps fill a three-stage K/V ring. One MMA warp overlaps scores with the
current softmax and value product. The two softmax groups split the head
columns; a group of at most 12 heads uses eight columns and four columns
(a 6-head group occupies the first six; the padded Q rows stay zero).
The compact block list is streamed from global memory; its width has no
compile-time limit. All token addresses use the supplied block spans.

Barrier counts: Q full=1 TMA issuer; KV full=16 TMA issuers; KV empty,
score full and value done=1 MMA issuer; score empty, P full and TMEM
release=256 softmax lanes. Empty rings start at phase 1; full rings at 0.
"""

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import arith
from cutlass.base_dsl.typing import Pointer
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.experimental import primitives as nvvm
from cutlass.experimental.cuda import tensor_map as tmap
from cuda.bindings import driver as cuda

from cudnn.frost.compiled_cache import compile_cached, template_key
from cudnn.frost.tile_dsl.barrier import MBarrier, PipelineState, Producer, advance
from cudnn.frost.tile_dsl.handles import GmemTileTma, MmaDesc, SmemTile
from cudnn.frost.tile_dsl.mma import mma_ss
from cudnn.frost.tile_dsl.pointwise import fp32_to_fp16
from cudnn.frost.tile_dsl.tma import TMA_L2_EVICT_LAST, opaque_i64, tma_gather4, tma_load_tile
from cudnn.frost.tile_dsl.tmem import tmem_alloc, tmem_dealloc

I32, I64, F32 = cutlass.Int32, cutlass.Int64, cutlass.Float32
D, NQ, TILE_N = 256, 16, 128
KV_STAGES, S_STAGES = 3, 2
SOFTMAX_WARPS, GATHER_WARPS, THREADS = 8, 16, 832
SOFTMAX_LANES = 256
LOG2E, LN2 = 1.4426950408889634, 0.6931471805599453
FROST_SOURCE_DIGEST = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
_FPROP_SM107_SOURCE_DIGEST = hashlib.sha256(Path(__file__).with_name("fprop_sm107.py").read_bytes()).hexdigest()[:16]


class FpropBars(NamedTuple):
    q_full: object
    kv_full: object
    kv_empty: object
    s_full: object
    s_empty: object
    p_full: object
    value_done: object
    release: object


def _fprop_bars():
    def alloc(n):
        return cutlass.Array(cutlass.Int64, n, alignment=16, space=cutlass.AddressSpace.smem)

    return FpropBars(
        MBarrier(alloc(1), stages=1, init_count=1, producer=Producer.TMA_LOAD),
        MBarrier(alloc(KV_STAGES), stages=KV_STAGES, init_count=GATHER_WARPS, producer=Producer.TMA_LOAD),
        MBarrier(alloc(KV_STAGES), stages=KV_STAGES, init_count=1, producer=Producer.MMA_COMMIT),
        MBarrier(alloc(S_STAGES), stages=S_STAGES, init_count=1, producer=Producer.MMA_COMMIT),
        MBarrier(alloc(S_STAGES), stages=S_STAGES, init_count=SOFTMAX_LANES, producer=Producer.THREAD),
        MBarrier(alloc(S_STAGES), stages=S_STAGES, init_count=SOFTMAX_LANES, producer=Producer.THREAD),
        MBarrier(alloc(S_STAGES), stages=S_STAGES, init_count=1, producer=Producer.MMA_COMMIT),
        MBarrier(alloc(1), stages=1, init_count=SOFTMAX_LANES, producer=Producer.THREAD),
    )


@cute.jit
def _fprop_select_i32(p, a, b):
    return I32(arith.select(p.ir_value(), I32(a).ir_value(), I32(b).ir_value()))


@cute.jit
def _fprop_select_f32(p, a, b):
    return F32(arith.select(p.ir_value(), F32(a).ir_value(), F32(b).ir_value()))


@cute.jit
def _fprop_key_span(ids, first, last, query, entry, count):
    begin, end = I32(-1), I32(-1)
    if entry < count:
        block = I32(ids[(I64(query), I64(entry))])
        if (block >= 0) & (block < first.shape[0]):
            begin = I32(first[block])
            end = I32(last[block])
    return begin, end


@cute.jit
def _fprop_gather(desc, contiguous_desc, storage, bars, state, tile, query, head, ids, first, last, count, low, high, total):
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx()) - SOFTMAX_WARPS
    lane = cute.arch.thread_idx()[0] % 32
    begin, end = I32(-1), I32(-1)
    if lane < 2:
        entry = tile * 32 + warp * 2 + lane
        begin, end = _fprop_key_span(ids, first, last, query, entry, count)
    starts, ends = [], []
    for block in cutlass.range_constexpr(2):
        starts.append(I32(nvvm.shfl_sync(0xFFFFFFFF, begin, block, 31, kind=nvvm.Shfl.IDX)))
        ends.append(I32(nvvm.shfl_sync(0xFFFFFFFF, end, block, 31, kind=nvvm.Shfl.IDX)))
    contiguous = (starts[0] >= 0) & (starts[0] >= low) & (starts[0] + 7 <= high) & (starts[0] + 7 < total)
    for block in cutlass.range_constexpr(2):
        contiguous = contiguous & (starts[block] == starts[0] + block * 4) & (ends[block] >= starts[block] + 3)
    bars.kv_empty[state.idx].wait(state.phase ^ 1)
    hint = opaque_i64(TMA_L2_EVICT_LAST)
    if nvvm.elect_sync():
        bars.kv_full[state.idx].arrive(n_bytes=4096)
    if contiguous:
        if nvvm.elect_sync():
            for box in cutlass.range_constexpr(4):
                nvvm.cp_async_bulk_tensor_shared_cta_global(
                    storage.subview(state.idx * (TILE_N * D) + box * (TILE_N * 64) + warp * 8 * 64),
                    contiguous_desc.get_ptr(),
                    [head * D + box * 64, starts[0]],
                    bars.kv_full[state.idx].smem_ptr,
                    l2_cache_hint=hint,
                )
    else:
        for block in cutlass.range_constexpr(2):
            if nvvm.elect_sync():
                rows = []
                for r in cutlass.range_constexpr(4):
                    token = starts[block] + r
                    valid = (starts[block] >= 0) & (token <= ends[block]) & (token >= low) & (token <= high) & (token < total)
                    rows.append(_fprop_select_i32(valid, token, -1))
                quad_offset = (warp * 2 + block) * 4 * 64
                for box in cutlass.range_constexpr(4):
                    tma_gather4(
                        desc,
                        storage.subview(state.idx * (TILE_N * D) + box * (TILE_N * 64) + quad_offset),
                        bars.kv_full[state.idx].smem_ptr,
                        head * D + box * 64,
                        rows[0],
                        rows[1],
                        rows[2],
                        rows[3],
                        l2_hint=hint,
                    )
    return advance(state, KV_STAGES)


@cute.jit
def _fprop_gather_loop(k_desc, v_desc, k_contiguous_desc, v_contiguous_desc, storage, bars, ids, first, last, query, head, count, low, high, total, tiles):
    state = PipelineState.start()
    state = _fprop_gather(k_desc, k_contiguous_desc, storage, bars, state, 0, query, head, ids, first, last, count, low, high, total)
    for tile in cutlass.range(tiles, unroll=1):
        if tile + 1 < tiles:
            state = _fprop_gather(k_desc, k_contiguous_desc, storage, bars, state, tile + 1, query, head, ids, first, last, count, low, high, total)
        state = _fprop_gather(v_desc, v_contiguous_desc, storage, bars, state, tile, query, head, ids, first, last, count, low, high, total)
    for stage in cutlass.range_constexpr(KV_STAGES):
        bars.kv_empty[state.idx].wait(state.phase ^ 1)
        state = advance(state, KV_STAGES)


@cute.jit
def _fprop_mma(sq, sk, sv, sp, tmem, bars, tiles, dtype: cutlass.Constexpr, retain_permit: cutlass.Constexpr = False):
    if cutlass.const_expr(retain_permit):
        nvvm.tcgen05_alloc(tmem, I32(64), group=nvvm.CTAGroup.CTA_1)
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
    else:
        tmem_alloc(tmem, 64, nvvm.CTAGroup.CTA_1)
    nvvm.barrier_cta_arrive(1, SOFTMAX_LANES + 32)
    tm = nvvm.make_tmem_ptr(tmem.load(), cutlass.Int8)
    qk = MmaDesc(
        M=128,
        N=16,
        K=256,
        bpe_a=2,
        bpe_b=2,
        tile_k_hw=16,
        cta_group=1,
        idesc=nvvm.Tcgen05InstrDesc.build(c_dtype=F32, a_dtype=dtype, b_dtype=dtype, n_dim=16, m_dim=128, k_dim=0),
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    pv = MmaDesc(
        M=128,
        N=16,
        K=128,
        bpe_a=2,
        bpe_b=2,
        tile_k_hw=16,
        atranspose=True,
        btranspose=True,
        cta_group=1,
        idesc=nvvm.Tcgen05InstrDesc.build(c_dtype=F32, a_dtype=dtype, b_dtype=dtype, n_dim=16, m_dim=128, a_major=1, b_major=1, k_dim=0),
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    kv, score, probability = PipelineState.start(), PipelineState.start(), PipelineState.start()
    bars.q_full.wait(0)
    bars.kv_full[kv.idx].wait(kv.phase)
    bars.s_empty[score.idx].wait(score.phase ^ 1)
    nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
    mma_ss(qk, sk[kv.idx].desc(), sq.desc(), tm.subview(score.idx * 16))
    elected = nvvm.elect_sync()
    bars.s_full[score.idx].arrive(cta_group=1, pred=elected)
    bars.kv_empty[kv.idx].arrive(cta_group=1, pred=elected)
    kv, score = advance(kv, KV_STAGES), advance(score, S_STAGES)
    for tile in cutlass.range(tiles, unroll=1):
        if tile + 1 < tiles:
            bars.kv_full[kv.idx].wait(kv.phase)
            bars.s_empty[score.idx].wait(score.phase ^ 1)
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
            mma_ss(qk, sk[kv.idx].desc(), sq.desc(), tm.subview(score.idx * 16))
            elected = nvvm.elect_sync()
            bars.s_full[score.idx].arrive(cta_group=1, pred=elected)
            bars.kv_empty[kv.idx].arrive(cta_group=1, pred=elected)
            kv, score = advance(kv, KV_STAGES), advance(score, S_STAGES)
        bars.kv_full[kv.idx].wait(kv.phase)
        bars.p_full[probability.idx].wait(probability.phase)
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
        for half in cutlass.range_constexpr(2):
            mma_ss(pv, sv[kv.idx].shifted(half * 2 * TILE_N * 64).desc(), sp[probability.idx].desc(), tm.subview(32 + half * 16), accumulate=tile > 0)
        elected = nvvm.elect_sync()
        bars.value_done[probability.idx].arrive(cta_group=1, pred=elected)
        bars.kv_empty[kv.idx].arrive(cta_group=1, pred=elected)
        kv, probability = advance(kv, KV_STAGES), advance(probability, S_STAGES)
    bars.release.wait(0)
    tmem_dealloc(tmem, 64, nvvm.CTAGroup.CTA_1)


@cute.jit
def _fprop_warp_max(value):
    for shift in cutlass.range_constexpr(5):
        value = cute.math.max(value, F32(nvvm.shfl_sync(0xFFFFFFFF, value, 16 >> shift, 31, kind=nvvm.Shfl.BFLY)), ftz=True)
    return value


@cute.jit
def _fprop_warp_sum(value):
    for shift in cutlass.range_constexpr(5):
        value = value + F32(nvvm.shfl_sync(0xFFFFFFFF, value, 16 >> shift, 31, kind=nvvm.Shfl.BFLY))
    return value


@cute.jit
def _fprop_softmax(
    tmem,
    bars,
    scratch,
    sp,
    out: cute.Tensor,
    out_fp32: cute.Tensor,
    lse: cute.Tensor,
    ids: cute.Tensor,
    first: cute.Tensor,
    last: cute.Tensor,
    query,
    head,
    count,
    low,
    high,
    total,
    tiles,
    scale,
    group_size: cutlass.Constexpr,
    ncols: cutlass.Constexpr,
):
    nvvm.barrier_cta_sync(barrier_id=1, thread_count=SOFTMAX_LANES + 32)
    tid = cute.arch.thread_idx()[0]
    warp, group, lane = tid // 32, tid // 128, tid % 128
    column = group * 8
    red = Pointer(scratch.data_ptr(), dtype=F32)
    base = tmem.load()
    neg_inf, zero, one = F32(float("-inf")), F32(0), F32(1)
    m = cutlass.Vector.from_elements(tuple(F32(float("-inf")) for _ in range(ncols)), F32)
    norm = cutlass.Vector.from_elements(tuple(F32(0) for _ in range(ncols)), F32)
    score, value = PipelineState.start(), PipelineState.start()
    for tile in cutlass.range(tiles, unroll=1):
        bars.s_full[score.idx].wait(score.phase)
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
        raw = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(base + score.idx * 16 + column, F32), num=ncols)
        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
        bars.s_empty[score.idx].arrive()
        begin, end = _fprop_key_span(ids, first, last, query, tile * 32 + lane // 4, count)
        token = begin + lane % 4
        valid = (begin >= 0) & (token <= end) & (token >= low) & (token <= high) & (token < total)
        scores = []
        red_base = score.idx * 64 + warp * 8
        for col in cutlass.range_constexpr(ncols):
            s = _fprop_select_f32(valid, F32(raw[col]) * scale * F32(LOG2E), neg_inf)
            scores.append(s)
            warp_max = _fprop_warp_max(s)
            if tid % 32 == 0:
                (red + red_base + col).store(warp_max)
        nvvm.barrier_cta_sync(barrier_id=2 + group, thread_count=128)
        maxima, sums, alphas, probabilities = [], [], [], []
        for col in cutlass.range_constexpr(ncols):
            maximum = F32(m[col])
            for other_warp in cutlass.range_constexpr(4):
                maximum = cute.math.max(maximum, F32((red + score.idx * 64 + group * 32 + other_warp * 8 + col).load()), ftz=True)
            safe_old = _fprop_select_f32(F32(m[col]) == neg_inf, zero, F32(m[col]))
            safe_new = _fprop_select_f32(maximum == neg_inf, zero, maximum)
            alpha = cute.math.exp2(cute.math.min(safe_old - safe_new, zero, ftz=True), fastmath=True)
            probability = cute.math.exp2(scores[col] - safe_new, fastmath=True)
            maxima.append(maximum)
            sums.append(F32(norm[col]) * alpha + probability)
            alphas.append(alpha)
            probabilities.append(probability)
        m, norm = cutlass.Vector.from_elements(tuple(maxima), F32), cutlass.Vector.from_elements(tuple(sums), F32)
        words = []
        for col in cutlass.range_constexpr(ncols // 2):
            words.append(fp32_to_fp16(probabilities[2 * col], probabilities[2 * col + 1], dtype=out.element_type))
        for col in cutlass.range_constexpr(ncols // 2, 4):
            words.append(I32(0))
        packed = cutlass.Vector.from_elements(tuple(words), I32)
        p_ptr = Pointer(sp.subview(score.idx * (TILE_N * NQ // 2) + lane * 8 + group * 4).data_ptr(), dtype=I32)
        p_ptr.store_swizzled(packed, cutlass.Swizzle(1, 4, 3), alignment=16)
        nvvm.fence_proxy("async.shared", space="cta")
        if tile > 0:
            bars.value_done[value.idx].wait(value.phase)
            value = advance(value, S_STAGES)
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
            needs_rescale = alphas[0] != one
            for col in cutlass.range_constexpr(1, ncols):
                needs_rescale = needs_rescale | (alphas[col] != one)
            if needs_rescale:
                for half in cutlass.range_constexpr(2):
                    op = nvvm.make_tmem_ptr(base + 32 + half * 16 + column, F32)
                    old = nvvm.tcgen05_ld("32x32b", op, num=ncols)
                    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                    updated = cutlass.Vector.from_elements(tuple(F32(old[col]) * alphas[col] for col in range(ncols)), F32)
                    nvvm.tcgen05_st("32x32b", op, updated)
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
        bars.p_full[score.idx].arrive()
        score = advance(score, S_STAGES)
    bars.value_done[value.idx].wait(value.phase)
    nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
    for col in cutlass.range_constexpr(ncols):
        warp_sum = _fprop_warp_sum(F32(norm[col]))
        if tid % 32 == 0:
            (red + 128 + warp * 8 + col).store(warp_sum)
    nvvm.barrier_cta_sync(barrier_id=2 + group, thread_count=128)
    inverses, empty = [], []
    for col in cutlass.range_constexpr(ncols):
        denominator = zero
        for other_warp in cutlass.range_constexpr(4):
            denominator = denominator + F32((red + 128 + group * 32 + other_warp * 8 + col).load())
        dead = denominator == zero
        safe_denominator = _fprop_select_f32(dead, one, denominator)
        inverses.append(one / safe_denominator)
        empty.append(dead)
        stats = _fprop_select_f32(dead, neg_inf, F32(m[col]) * F32(LN2) + cute.math.log(safe_denominator, fastmath=True))
        if (lane == col) & (column + col < group_size):
            lse[(I64(head * group_size + column + col), I64(query))] = stats
    for half in cutlass.range_constexpr(2):
        raw_o = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(base + 32 + half * 16 + column, F32), num=ncols)
        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
        for col in cutlass.range_constexpr(ncols):
            if column + col < group_size:
                result = _fprop_select_f32(empty[col], zero, F32(raw_o[col]) * inverses[col])
                coord = (I64(query), I64(head * group_size + column + col), I64(lane + half * 128))
                out_fp32[coord] = result
                out[coord] = result.to(out.element_type)
    nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
    bars.release.arrive()


@cute.jit
def _fprop_body(
    q_desc,
    k_desc,
    v_desc,
    k_contiguous_desc,
    v_contiguous_desc,
    out: cute.Tensor,
    out_fp32: cute.Tensor,
    lse: cute.Tensor,
    ids: cute.Tensor,
    counts: cute.Tensor,
    lo: cute.Tensor,
    hi: cute.Tensor,
    first: cute.Tensor,
    last: cute.Tensor,
    scale,
    group_size: cutlass.Constexpr,
    query_override=None,
):
    query, head, _ = cute.arch.block_idx()
    if cutlass.const_expr(query_override is not None):
        query = I32(query_override)
    tid = cute.arch.thread_idx()[0]
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    dtype = out.element_type
    count, low, high = I32(counts[query]), I32(lo[query]), I32(hi[query])
    tiles = cute.math.max((count + 31) // 32, 1)
    sq_raw = cutlass.Array(dtype, NQ * D, alignment=1024, space=cutlass.AddressSpace.smem)
    sp_raw = cutlass.Array(I32, S_STAGES * TILE_N * NQ // 2, alignment=1024, space=cutlass.AddressSpace.smem)
    kv_raw = cutlass.Array(dtype, KV_STAGES * TILE_N * D, alignment=1024, space=cutlass.AddressSpace.smem)
    scratch = cutlass.Array(F32, 192, alignment=16, space=cutlass.AddressSpace.smem)
    tmem = cutlass.Array(I32, 1, alignment=16, space=cutlass.AddressSpace.smem)
    sq = SmemTile(
        base=sq_raw,
        elems_per_stage=NQ * D,
        leading_byte_offset=0,
        stride_byte_offset=1024,
        layout=2,
        tma_loads_per_tile=4,
        tma_granu_elems=64,
        tma_subtile_stride_elems=NQ * 64,
    )
    sk = SmemTile(base=kv_raw, elems_per_stage=TILE_N * D, stages=KV_STAGES, leading_byte_offset=0, stride_byte_offset=1024, layout=2)
    sv = SmemTile(base=kv_raw, elems_per_stage=TILE_N * D, stages=KV_STAGES, leading_byte_offset=TILE_N * 128, stride_byte_offset=1024, layout=2)
    sp = SmemTile(base=sp_raw, elems_per_stage=TILE_N * NQ // 2, stages=S_STAGES, leading_byte_offset=0, stride_byte_offset=256, layout=6)
    bars = _fprop_bars()
    if warp == 0:
        if nvvm.elect_sync():
            bars.q_full.init()
            bars.release.init()
            for stage in cutlass.range_constexpr(KV_STAGES):
                bars.kv_full[stage].init()
                bars.kv_empty[stage].init()
            for stage in cutlass.range_constexpr(S_STAGES):
                bars.s_full[stage].init()
                bars.s_empty[stage].init()
                bars.p_full[stage].init()
                bars.value_done[stage].init()
    if warp < SOFTMAX_WARPS:
        zero = cutlass.Vector.from_elements((I32(0), I32(0), I32(0), I32(0)), I32)
        for chunk in cutlass.range(tid, NQ * D // 8, SOFTMAX_LANES, unroll=1):
            Pointer(sq_raw.subview(chunk * 8).data_ptr(), dtype=I32).store(zero, alignment=16)
        nvvm.fence_proxy("async.shared", space="cta")
    nvvm.fence_mbarrier_init()
    nvvm.barrier_cta_sync()
    if warp < SOFTMAX_WARPS:
        if cutlass.const_expr(group_size <= 12):
            if warp < 4:
                _fprop_softmax(
                    tmem, bars, scratch, sp_raw, out, out_fp32, lse, ids, first, last, query, head, count, low, high, out.shape[0], tiles, scale, group_size, 8
                )
            else:
                _fprop_softmax(
                    tmem, bars, scratch, sp_raw, out, out_fp32, lse, ids, first, last, query, head, count, low, high, out.shape[0], tiles, scale, group_size, 4
                )
        else:
            _fprop_softmax(
                tmem, bars, scratch, sp_raw, out, out_fp32, lse, ids, first, last, query, head, count, low, high, out.shape[0], tiles, scale, group_size, 8
            )
    elif warp < SOFTMAX_WARPS + GATHER_WARPS:
        _fprop_gather_loop(
            k_desc, v_desc, k_contiguous_desc, v_contiguous_desc, kv_raw, bars, ids, first, last, query, head, count, low, high, out.shape[0], tiles
        )
    elif warp == SOFTMAX_WARPS + GATHER_WARPS:
        _fprop_mma(sq, sk, sv, sp, tmem, bars, tiles, dtype, cutlass.const_expr(query_override is not None))
    elif warp == SOFTMAX_WARPS + GATHER_WARPS + 1:
        bars.q_full.arrive(n_bytes=group_size * D * 2, pred=nvvm.elect_sync())
        tma_load_tile(sq, GmemTileTma(q_desc)(I32(0), head * group_size, query), bars.q_full.smem_ptr, cta_group=1)
        bars.q_full.wait(0)


@cute.kernel
def frost_qsa_fprop_sm100(
    q_desc: cutlass.GridConstant[tmap.TensorMap],
    k_desc: cutlass.GridConstant[tmap.TensorMap],
    v_desc: cutlass.GridConstant[tmap.TensorMap],
    k_contiguous_desc: cutlass.GridConstant[tmap.TensorMap],
    v_contiguous_desc: cutlass.GridConstant[tmap.TensorMap],
    out: cute.Tensor,
    out_fp32: cute.Tensor,
    lse: cute.Tensor,
    ids: cute.Tensor,
    counts: cute.Tensor,
    lo: cute.Tensor,
    hi: cute.Tensor,
    first: cute.Tensor,
    last: cute.Tensor,
    mode,
    scale: F32,
    group_size: cutlass.Constexpr[int],
):
    if cutlass.const_expr(mode is None):
        _fprop_body(q_desc, k_desc, v_desc, k_contiguous_desc, v_contiguous_desc, out, out_fp32, lse, ids, counts, lo, hi, first, last, scale, group_size)
    else:
        first_query = cute.arch.block_idx()[0] * 32
        if first_query < out.shape[0]:
            if mode[I64(first_query // 128)] != 0:
                for step in cutlass.range(32, unroll=1):
                    query = first_query + step
                    if query < out.shape[0]:
                        _fprop_body(
                            q_desc,
                            k_desc,
                            v_desc,
                            k_contiguous_desc,
                            v_contiguous_desc,
                            out,
                            out_fp32,
                            lse,
                            ids,
                            counts,
                            lo,
                            hi,
                            first,
                            last,
                            scale,
                            group_size,
                            query,
                        )
                        nvvm.barrier_cta_sync()
                if cute.arch.warp_idx() == SOFTMAX_WARPS + GATHER_WARPS:
                    nvvm.tcgen05_relinquish_alloc_permit(group=nvvm.CTAGroup.CTA_1)


frost_qsa_fprop_sm100.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _fprop_host(q, k, v, out, out_fp32, lse, ids, counts, lo, hi, first, last, mode, scale: F32, stream: cuda.CUstream, arch: cutlass.Constexpr[int]):
    group_size = cutlass.const_expr(q.shape[1] // k.shape[1])
    q_desc = tmap.create_tensor_map_tiled(
        global_address=q.iterator.toint(),
        dtype=q.element_type,
        global_dims=(q.shape[2], q.shape[1], q.shape[0]),
        global_strides=(I64(q.stride[1]) // 8, I64(q.stride[0]) // 8),
        box_dims=(64, group_size, 1),
        swizzle=tmap.TensorMapSwizzle.s128b,
        l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
    )
    k_desc = tmap.create_tensor_map_tiled(
        global_address=k.iterator.toint(),
        dtype=k.element_type,
        global_dims=(k.shape[1] * D, k.shape[0]),
        global_strides=(I64(k.stride[0]) // 8,),
        box_dims=(64, 1),
        swizzle=tmap.TensorMapSwizzle.s128b,
        l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
    )
    v_desc = tmap.create_tensor_map_tiled(
        global_address=v.iterator.toint(),
        dtype=v.element_type,
        global_dims=(v.shape[1] * D, v.shape[0]),
        global_strides=(I64(v.stride[0]) // 8,),
        box_dims=(64, 1),
        swizzle=tmap.TensorMapSwizzle.s128b,
        l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
    )
    k_contiguous_desc = tmap.create_tensor_map_tiled(
        global_address=k.iterator.toint(),
        dtype=k.element_type,
        global_dims=(k.shape[1] * D, k.shape[0]),
        global_strides=(I64(k.stride[0]) // 8,),
        box_dims=(64, 8),
        swizzle=tmap.TensorMapSwizzle.s128b,
        l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
    )
    v_contiguous_desc = tmap.create_tensor_map_tiled(
        global_address=v.iterator.toint(),
        dtype=v.element_type,
        global_dims=(v.shape[1] * D, v.shape[0]),
        global_strides=(I64(v.stride[0]) // 8,),
        box_dims=(64, 8),
        swizzle=tmap.TensorMapSwizzle.s128b,
        l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
    )
    if cutlass.const_expr(arch == 107):
        from .fprop_sm107 import frost_qsa_fprop_sm107

        frost_qsa_fprop_sm107(
            q_desc, k_desc, v_desc, k_contiguous_desc, v_contiguous_desc, out, out_fp32, lse, ids, counts, lo, hi, first, last, mode, scale, group_size
        ).launch(
            grid=(cute.ceil_div(q.shape[0], 8) if cutlass.const_expr(mode is not None) else q.shape[0], k.shape[1], 1),
            block=(THREADS, 1, 1),
            min_blocks_per_mp=1,
            stream=stream,
        )
    else:
        frost_qsa_fprop_sm100(
            q_desc, k_desc, v_desc, k_contiguous_desc, v_contiguous_desc, out, out_fp32, lse, ids, counts, lo, hi, first, last, mode, scale, group_size
        ).launch(
            grid=(cute.ceil_div(q.shape[0], 32) if cutlass.const_expr(mode is not None) else q.shape[0], k.shape[1], 1),
            block=(THREADS, 1, 1),
            min_blocks_per_mp=1,
            stream=stream,
        )


@lru_cache(maxsize=None)
def _compile_args(dtype, query_heads, kv_heads, gated):
    """Reuse storage-free signatures; the shared cache owns CUDA objects."""
    total, capacity, blocks = cute.sym_int64(), cute.sym_int64(), cute.sym_int64()

    def tensor(dt, shape, alignment=16):
        return make_fake_compact_tensor(dt, shape, stride_order=tuple(range(len(shape) - 1, -1, -1)), assumed_align=alignment)

    q = tensor(dtype, (total, query_heads, D))
    kv = tensor(dtype, (total, kv_heads, D))
    out_fp32 = tensor(F32, (total, query_heads, D))
    lse = tensor(F32, (query_heads, total), 4)
    ids = tensor(I32, (total, capacity), 4)
    rows, spans = tensor(I32, (total,), 4), tensor(I32, (blocks,), 4)
    return (
        q,
        kv,
        kv,
        q,
        out_fp32,
        lse,
        ids,
        rows,
        rows,
        rows,
        spans,
        spans,
        tensor(I32, (cute.sym_int64(),), 4) if gated else None,
        F32(1),
        make_fake_stream(use_tvm_ffi_env_stream=False),
    )


def compile_kernel(dtype, query_heads: int, kv_heads: int, arch: int, gated=False):
    """Compile dynamic extents with the common device-aware artifact cache."""
    key = template_key(globals(), {**locals(), "sm107_source_digest": _FPROP_SM107_SOURCE_DIGEST if arch == 107 else None}, "compile_kernel")
    if arch not in (100, 103, 107):
        raise ValueError(f"QSA forward requires SM100, SM103 or SM107, got SM{arch}")
    if query_heads % kv_heads or not 1 <= query_heads // kv_heads <= NQ:
        raise ValueError("QSA forward requires a GQA group in [1,16]")
    if dtype not in (cutlass.BFloat16, cutlass.Float16):
        raise ValueError("QSA forward requires BF16 or FP16")
    return compile_cached(
        _fprop_host,
        *_compile_args(dtype, query_heads, kv_heads, gated),
        arch,
        options=f"--enable-tvm-ffi --gpu-arch=sm_{arch}a",
        cache_key=key,
        symbol="qsa_forward",
    )


def launch(compiled, q, k, v, out, out_fp32, lse, block_ids, block_counts, lo, hi, block_first, block_last, scale, stream, mode=None):
    """Launch with caller-owned outputs and compact metadata; no allocation."""
    compiled(q, k, v, out, out_fp32, lse, block_ids, block_counts, lo, hi, block_first, block_last, mode, F32(scale), stream)
