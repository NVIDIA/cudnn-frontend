# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Dense d128 M128/N128 backward, one launch and no workspace.

CTA384: compute0..7 form two128-thread groups, each owns row=tid%128 and
64 columns; loader8, MMA9, stats10+11, output/TMEM control11. Input has two
paired Q/dO or K/V stages; O and stats each have one stage. KV retains64
original FP32 P values per lane through dP readiness; native copies feed
gapped MMA-TS dV/dK. Q dS remains SW128 shared MMA-SS A. K16 only.

SMEM bytes: resident0[0,32768), resident1[32768,65536), paired input0
[65536,131072), paired input1[131072,196608), auxiliary[196608,229376)
(Q prologue O then dS, or KV O), stats[229376,230400). Sixteen aligned
barrier arrays contain18 slots at230400:230656; Int32 TMEM handle230656.
Dynamic230660 plus separate driver allowance1024 gives231684 bytes.
Each128x128 operand uses TWO SW128 TMA boxes[128,64], head coordinates0/64,
half offsets0/8192. Gradient B is MN-major LBO16384,stride1024; score and
Q dS A are K-major LBO0,stride1024. Generic Q dS stores use64-byte alignment.

TMEM: S[0,128), DP[128,256), G0[256,384), G1[384,512). Packed P halves
occupy[0,32),[64,96); dS occupies[128,160),[192,224). Groupg chunkc loads
FP32 atplane+64*g+c then LOAD-waits before storing native atplane+64*g+c/2.
Groups never compact into each other's unread source. All256 lanes STORE-
wait and BEFORE_THREAD_SYNC fence before P/dS publication. Consumers use
AFTER_THREAD_SYNC after readiness/completion waits before tcgen05 operations. TS A raw Int8 steps0/8/16/24/64/72/80/88 use
existing mma_ts_step; gradient C offsets are raw before Float32 conversion.

LOCAL CTA1 barrier ledger: field(depth,producer,count; issuers -> waiters):
 resident_full(1,TMA,1; loader8 elect -> all384)
 q_stats_ready(1,THREAD,64; Q stats10+11 bare -> Q compute256,loader8)
 input_full(2,TMA,1 each; loader8 elect -> MMA9,KV stats10+11)
 input_empty(2,MMA,1 each; MMA9 elect final gradient -> loader8)
 o_full(1,TMA,1; KV loader8 elect -> KV stats10+11)
 o_empty(1,THREAD,64; KV stats10+11 bare -> KV loader8)
 stats_full(1,THREAD,64; KV stats10+11 bare -> KV compute256)
 stats_empty(1,THREAD,256; KV compute0..7 bare -> KV stats10+11)
 s_full(1,MMA,1; MMA9 elect S or Q S+DP -> compute256)
 dp_full(1,MMA,1; KV MMA9 elect DP -> KV compute256)
 p_full(1,THREAD,256; KV compute0..7 bare after STORE wait -> MMA9)
 ds_full(1,THREAD,256; compute0..7 bare after STORE wait/Q fence -> MMA9)
 q_score_empty(1,THREAD,256; Q compute0..7 bare -> Q MMA9)
 q_ds_empty(1,MMA,1; Q MMA9 elect final gradient -> Q compute256)
 final_grad(1,MMA,1; MMA9 elect ONCE iff nonempty -> compute256 iff nonempty)
 out_ready(1,THREAD,256; compute0..7 bare output/fence -> control11)
All18 init sites are warp0 AND elect, then fence_mbarrier_init and CTA384
join before use. Full starts phase0, empty waiters phase1, independent typed
states advance once per event. No synthetic arrivals or role-local CTA join.

KV schedule S0 then DP(j),DV(j),S(j+1),DK(j). Same-issuer/group MMA A-read
before later D-write protects DV->nextS and DK->nextDP WAR. Cross-warp
packed stores still wait cumulative S/DP completion (including old DV/DK),
never merely issuer order. Input release follows DK completion and includes
both stats readers transitively through stats_full -> DS -> ds_full -> DK.
O releases only after all64 stats threads finish both rows into four private
FP32 scalars per lane. Stats then wait stats_empty before publishing those
scalars; early O release cannot overwrite the still-live statistics slab. Q shared dS reuse
requires q_ds_empty completion; q_score_empty only releases S/DP reads.

Every tail wait advances: loader input_empty twice and KV o_empty once;
both KV stats warps stats_empty once; Q MMA q_score_empty once; Q compute
q_ds_empty once. Nonempty compute waits one-shot final_grad before output.
Zero-trip still waits resident+CTA join, drains initial empties, writes zero
outputs without TMEM gradient reads, completes output TMA and final CTA384
join before warp11 collective512-column deallocation.
"""

from typing import NamedTuple
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import arith
from cutlass.experimental import primitives as nvvm
from cutlass.experimental.cuda import tensor_map as tmap
import cuda.bindings.driver as cuda
from cudnn.frost.tile_dsl.barrier import MBarrier, PipelineState, Producer, advance
from cudnn.frost.tile_dsl.handles import GmemTileTma, MmaDesc, SmemTile
from cudnn.frost.tile_dsl.mask import apply_mask_words, band_mask_words
from cudnn.frost.tile_dsl.mma import mma_ss, mma_ts_step
from cudnn.frost.tile_dsl.pointwise import f16x2_to_f32, row_reduction_pair, tmem_load_tile
from cudnn.frost.tile_dsl.tma import tma_load_tile, tma_store_commit, tma_store_tile, tma_store_wait
from cudnn.frost.tile_dsl.tmem import tmem_alloc, tmem_dealloc
from cudnn.sdpa.bwd.config_sm100_d128 import CFG

_SWIZZLE = cutlass.Swizzle(3, 4, 3)
_LOG2E = 1.4426950408889634


def _operand(raw, offset, rows, *, transposed=False, cols=CFG.HEAD_DIM):
    return SmemTile(
        base=raw.subview(offset),
        elems_per_stage=rows * cols,
        leading_byte_offset=rows * CFG.SWIZZLE_BYTES if transposed else 0,
        stride_byte_offset=8 * CFG.SWIZZLE_BYTES,
        layout=CFG.SMEM_LAYOUT,
        tma_loads_per_tile=cols // CFG.HEAD_GRANULE,
        tma_granu_elems=CFG.HEAD_GRANULE,
        tma_subtile_stride_elems=rows * CFG.HEAD_GRANULE,
        desc_version=CFG.DESC_VERSION,
    )


def _mma_desc(dtype, *, gradient=False):
    n = CFG.HEAD_DIM if gradient else CFG.TILE_N
    k = CFG.TILE_N if gradient else CFG.HEAD_DIM
    return MmaDesc(
        M=CFG.TILE_M,
        N=n,
        K=k,
        bpe_a=CFG.BPE,
        bpe_b=CFG.BPE,
        tile_k_hw=CFG.TILE_K_HW,
        btranspose=gradient,
        cta_group=CFG.CTA_MMA,
        idesc=nvvm.Tcgen05InstrDesc.build(c_dtype=cutlass.Float32, a_dtype=dtype, b_dtype=dtype, m_dim=CFG.TILE_M, n_dim=n, b_major=int(gradient)),
        kind=nvvm.Tcgen05MMAKind.F16,
    )


def _input(slab, stage, operand, *, transposed=False):
    return _operand(slab, CFG.INPUT + stage * CFG.INPUT_STAGE_ELEMS + operand * CFG.INPUT_OPERAND_ELEMS, CFG.TILE_N, transposed=transposed)


@cute.jit
def _wait_service(bar, state):
    bar[state.idx].wait(state.phase)
    nvvm.bar_warp_sync(cute.arch.FULL_MASK)


@cute.jit
def _read_stats_row(s_o, s_do, lse, batch, head, q_base, sq, rows: cutlass.Constexpr[int], row):
    delta = cutlass.Float32(0.0)
    for chunk in cutlass.range_constexpr(CFG.HEAD_DIM // CFG.HEAD_GRANULE):
        off = chunk * rows * CFG.HEAD_GRANULE + row * CFG.HEAD_GRANULE
        o_half = s_o.base.subview(off).data_ptr().load_swizzled(_SWIZZLE, alignment=128, count=CFG.HEAD_GRANULE)
        do_half = s_do.base.subview(off).data_ptr().load_swizzled(_SWIZZLE, alignment=128, count=CFG.HEAD_GRANULE)
        # Preserve the SW128 load and logical element order while expanding
        # packed pairs directly, avoiding generic BF16 vector conversion.
        o_words = o_half.bitcast(cutlass.Int32)
        do_words = do_half.bitcast(cutlass.Int32)
        o = cutlass.Vector.from_elements(tuple(v for i in range(CFG.HEAD_GRANULE // 2) for v in f16x2_to_f32(o_words[i], dtype=o_half.dtype)), cutlass.Float32)
        do = cutlass.Vector.from_elements(
            tuple(v for i in range(CFG.HEAD_GRANULE // 2) for v in f16x2_to_f32(do_words[i], dtype=do_half.dtype)), cutlass.Float32
        )
        pair = row_reduction_pair(o * do)
        delta += pair[0] + pair[1]
    lse_val = cutlass.Float32(float("inf"))
    if q_base + row < sq:
        lse_val = lse[batch, head, q_base + row]
    return lse_val, delta


@cute.jit
def _write_stats(s_o, s_do, stats, lse, batch, head, q_base, sq, rows: cutlass.Constexpr[int]):
    stats_thread = cute.arch.thread_idx()[0] - CFG.STATS_WARP * CFG.WARP_LANES
    for row_group in cutlass.range_constexpr(rows // CFG.STATS_THREADS):
        row = stats_thread + row_group * CFG.STATS_THREADS
        lse_val, delta = _read_stats_row(s_o, s_do, lse, batch, head, q_base, sq, rows, row)
        stats.subview(row).store(lse_val)
        stats.subview(rows + row).store(delta)


@cute.jit
def _probability_masked(
    s, dp, stats, owner_base, stream_base, sq, skv, scale, causal: cutlass.Constexpr[bool], q_owner: cutlass.Constexpr[bool], col: cutlass.Constexpr[int]
):
    tid = cute.arch.thread_idx()[0] % CFG.TILE_M
    if cutlass.const_expr(q_owner):
        lse = stats.subview(tid).load()
        delta = stats.subview(CFG.Q_STATS_DELTA + tid).load()
        hi = skv
        if cutlass.const_expr(causal):
            hi = cute.math.min(hi, owner_base + tid + 1)
        live = (owner_base + tid < sq) & (lse != cutlass.Float32(float("-inf")))
        hi = cutlass.Int32(arith.select(live.ir_value(), hi.ir_value(), stream_base.ir_value()))
        words = band_mask_words(None, hi, stream_base + col, CFG.POINTWISE_COLS)
        ds_live = skv > 1
        if cutlass.const_expr(causal):
            ds_live = ds_live & (owner_base + tid > 0)
        ds_hi = cutlass.Int32(arith.select(ds_live.ir_value(), hi.ir_value(), stream_base.ir_value()))
        ds_words = band_mask_words(None, ds_hi, stream_base + col, CFG.POINTWISE_COLS)
    else:
        lse = cutlass.Vector.from_elements(tuple(stats.subview(col + i).load() for i in range(CFG.POINTWISE_COLS)), cutlass.Float32)
        delta = cutlass.Vector.from_elements(tuple(stats.subview(CFG.KV_STATS_DELTA + col + i).load() for i in range(CFG.POINTWISE_COLS)), cutlass.Float32)
        lo = owner_base + tid if cutlass.const_expr(causal) else None
        hi = cutlass.Int32(arith.select((owner_base + tid < skv).ir_value(), sq.ir_value(), stream_base.ir_value()))
        words = band_mask_words(lo, hi, stream_base + col, CFG.POINTWISE_COLS)
        ds_lo = cute.math.max(lo, cutlass.Int32(1)) if cutlass.const_expr(causal) else None
        ds_hi = cutlass.Int32(arith.select((skv > 1).ir_value(), hi.ir_value(), stream_base.ir_value()))
        ds_words = band_mask_words(ds_lo, ds_hi, stream_base + col, CFG.POINTWISE_COLS)
    p = cute.math.exp2((s * scale - lse) * _LOG2E, fastmath=True)
    p = apply_mask_words(p, words, mask_value=0.0, n_cols=CFG.POINTWISE_COLS)
    # A row with exactly one legal key has softmax derivative exactly zero.
    # Select zero rather than retaining dP-versus-delta reduction roundoff;
    # P remains unchanged because that key still contributes to dV.
    ds = apply_mask_words((dp - delta) * p * scale, ds_words, mask_value=0.0, n_cols=CFG.POINTWISE_COLS)
    return p, ds


@cute.jit
def _probability_full(s, dp, stats, scale, q_owner: cutlass.Constexpr[bool], col: cutlass.Constexpr[int]):
    tid = cute.arch.thread_idx()[0] % CFG.TILE_M
    if cutlass.const_expr(q_owner):
        lse = stats.subview(tid).load()
        delta = stats.subview(CFG.Q_STATS_DELTA + tid).load()
    else:
        lse = cutlass.Vector.from_elements(tuple(stats.subview(col + i).load() for i in range(CFG.POINTWISE_COLS)), cutlass.Float32)
        delta = cutlass.Vector.from_elements(tuple(stats.subview(CFG.KV_STATS_DELTA + col + i).load() for i in range(CFG.POINTWISE_COLS)), cutlass.Float32)
    # Full rectangles exclude SKV1 and causal query0, so both original
    # masks are all-one. KV keeps its original nonfinite-LSE behavior.
    p = cute.math.exp2((s * scale - lse) * _LOG2E, fastmath=True)
    ds = (dp - delta) * p * scale
    return p, ds


@cute.jit
def _probability(
    s, dp, stats, owner_base, stream_base, sq, skv, scale, causal: cutlass.Constexpr[bool], q_owner: cutlass.Constexpr[bool], col: cutlass.Constexpr[int]
):
    tid = cute.arch.thread_idx()[0] % CFG.TILE_M
    if cutlass.const_expr(q_owner):
        full = (owner_base + CFG.TILE_M <= sq) & (stream_base + CFG.TILE_N <= skv)
        if cutlass.const_expr(causal):
            full = full & (stream_base + CFG.TILE_N - 1 <= owner_base)
        lse = stats.subview(tid).load()
        # All32 lanes participate even on geometric tails. A sentinel in any
        # lane sends that whole warp through the original zero-masking path.
        lse_live = nvvm.vote_sync(cute.arch.FULL_MASK, lse != cutlass.Float32(float("-inf")), nvvm.VoteSync.ALL)
        full = full & lse_live
    else:
        full = (owner_base + CFG.TILE_M <= skv) & (stream_base + CFG.TILE_N <= sq)
        if cutlass.const_expr(causal):
            full = full & (stream_base >= owner_base + CFG.TILE_M - 1)
    # Seed staged-control-flow result types; both branches overwrite them.
    p, ds = s, dp
    if full:
        p, ds = _probability_full(s, dp, stats, scale, q_owner, col)
    else:
        p, ds = _probability_masked(s, dp, stats, owner_base, stream_base, sq, skv, scale, causal, q_owner, col)
    return p, ds


@cute.jit
def _kv_full(owner_base, stream_base, sq, skv, causal: cutlass.Constexpr[bool]):
    full = (owner_base + CFG.TILE_M <= skv) & (stream_base + CFG.TILE_N <= sq)
    if cutlass.const_expr(causal):
        full = full & (stream_base >= owner_base + CFG.TILE_M - 1)
    return full


@cute.jit
def _kv_probability_p(s, stats, owner_base, stream_base, sq, skv, scale, causal: cutlass.Constexpr[bool], col: cutlass.Constexpr[int]):
    tid = cute.arch.thread_idx()[0] % CFG.TILE_M
    lse = cutlass.Vector.from_elements(tuple(stats.subview(col + i).load() for i in range(CFG.POINTWISE_COLS)), cutlass.Float32)
    p = s
    if _kv_full(owner_base, stream_base, sq, skv, causal):
        p = cute.math.exp2((s * scale - lse) * _LOG2E, fastmath=True)
    else:
        lo = owner_base + tid if cutlass.const_expr(causal) else None
        hi = cutlass.Int32(arith.select((owner_base + tid < skv).ir_value(), sq.ir_value(), stream_base.ir_value()))
        words = band_mask_words(lo, hi, stream_base + col, CFG.POINTWISE_COLS)
        p = cute.math.exp2((s * scale - lse) * _LOG2E, fastmath=True)
        p = apply_mask_words(p, words, mask_value=0.0, n_cols=CFG.POINTWISE_COLS)
    return p


@cute.jit
def _kv_probability_ds(dp, p, stats, owner_base, stream_base, sq, skv, scale, causal: cutlass.Constexpr[bool], col: cutlass.Constexpr[int]):
    tid = cute.arch.thread_idx()[0] % CFG.TILE_M
    delta = cutlass.Vector.from_elements(tuple(stats.subview(CFG.KV_STATS_DELTA + col + i).load() for i in range(CFG.POINTWISE_COLS)), cutlass.Float32)
    ds = dp
    if _kv_full(owner_base, stream_base, sq, skv, causal):
        ds = (dp - delta) * p * scale
    else:
        lo = owner_base + tid if cutlass.const_expr(causal) else None
        hi = cutlass.Int32(arith.select((owner_base + tid < skv).ir_value(), sq.ir_value(), stream_base.ir_value()))
        ds_lo = cute.math.max(lo, cutlass.Int32(1)) if cutlass.const_expr(causal) else None
        ds_hi = cutlass.Int32(arith.select((skv > 1).ir_value(), hi.ir_value(), stream_base.ir_value()))
        ds_words = band_mask_words(ds_lo, ds_hi, stream_base + col, CFG.POINTWISE_COLS)
        ds = apply_mask_words((dp - delta) * p * scale, ds_words, mask_value=0.0, n_cols=CFG.POINTWISE_COLS)
    return ds


@cute.jit
def _kv_p_chunk(
    score_base, stats, owner_base, stream_base, sq, skv, scale, causal: cutlass.Constexpr[bool], dtype: cutlass.Constexpr, col: cutlass.Constexpr[int]
):
    s = tmem_load_tile(score_base + CFG.TMEM_S + col, num_elems=CFG.POINTWISE_COLS, ld_num=CFG.POINTWISE_COLS).vec
    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
    p = _kv_probability_p(s, stats, owner_base, stream_base, sq, skv, scale, causal, col)
    packed_col = (col // CFG.GROUP_COLS) * CFG.GROUP_COLS + (col % CFG.GROUP_COLS) * CFG.BPE // CFG.TMEM_WORD_BYTES
    nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(score_base + CFG.TMEM_S + packed_col, cutlass.Float32), p.to(dtype))
    return p  # The original FP32 vector remains live; native conversion is a copy.


@cute.jit
def _kv_ds_chunk(
    score_base, p, stats, owner_base, stream_base, sq, skv, scale, causal: cutlass.Constexpr[bool], dtype: cutlass.Constexpr, col: cutlass.Constexpr[int]
):
    dp = tmem_load_tile(score_base + CFG.TMEM_DP + col, num_elems=CFG.POINTWISE_COLS, ld_num=CFG.POINTWISE_COLS).vec
    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
    ds = _kv_probability_ds(dp, p, stats, owner_base, stream_base, sq, skv, scale, causal, col)
    packed_col = (col // CFG.GROUP_COLS) * CFG.GROUP_COLS + (col % CFG.GROUP_COLS) * CFG.BPE // CFG.TMEM_WORD_BYTES
    nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(score_base + CFG.TMEM_DP + packed_col, cutlass.Float32), ds.to(dtype))


class Bars(NamedTuple):
    resident_full: MBarrier
    q_stats_ready: MBarrier
    input_full: MBarrier
    input_empty: MBarrier
    o_full: MBarrier
    o_empty: MBarrier
    stats_full: MBarrier
    stats_empty: MBarrier
    s_full: MBarrier
    dp_full: MBarrier
    p_full: MBarrier
    ds_full: MBarrier
    q_score_empty: MBarrier
    q_ds_empty: MBarrier
    final_grad: MBarrier
    out_ready: MBarrier


def make_bars():
    def make(stages, count, producer):
        return MBarrier(cutlass.Array(cutlass.Int64, stages, alignment=16, space=cutlass.AddressSpace.smem), stages=stages, init_count=count, producer=producer)

    return Bars(
        make(1, CFG.ONE_LANE, Producer.TMA_LOAD),
        make(1, CFG.STATS_ARRIVERS, Producer.THREAD),
        make(CFG.INPUT_STAGES, CFG.ONE_LANE, Producer.TMA_LOAD),
        make(CFG.INPUT_STAGES, CFG.ONE_LANE, Producer.MMA_COMMIT),
        make(1, CFG.ONE_LANE, Producer.TMA_LOAD),
        make(1, CFG.STATS_ARRIVERS, Producer.THREAD),
        make(1, CFG.STATS_ARRIVERS, Producer.THREAD),
        make(1, CFG.COMPUTE_ARRIVERS, Producer.THREAD),
        make(1, CFG.ONE_LANE, Producer.MMA_COMMIT),
        make(1, CFG.ONE_LANE, Producer.MMA_COMMIT),
        make(1, CFG.COMPUTE_ARRIVERS, Producer.THREAD),
        make(1, CFG.COMPUTE_ARRIVERS, Producer.THREAD),
        make(1, CFG.COMPUTE_ARRIVERS, Producer.THREAD),
        make(1, CFG.ONE_LANE, Producer.MMA_COMMIT),
        make(1, CFG.ONE_LANE, Producer.MMA_COMMIT),
        make(1, CFG.COMPUTE_ARRIVERS, Producer.THREAD),
    )


def _q_ds(slab):
    return _operand(slab, CFG.AUX, CFG.TILE_M, cols=CFG.TILE_N)


@cute.jit
def _mma_ts_gapped(desc, plane, desc_b, dest, accumulate):
    for k in cutlass.range_constexpr(CFG.TILE_N // CFG.TILE_K_HW):
        # mma_ts_step adds the packed K16 stride. Shift the origin past each
        # group gap: logical group width minus its native packed width.
        mma_ts_step(
            desc,
            nvvm.make_tmem_ptr(
                plane + (k // (CFG.GROUP_COLS // CFG.TILE_K_HW)) * (CFG.GROUP_COLS - CFG.GROUP_COLS * CFG.BPE // CFG.TMEM_WORD_BYTES), cutlass.Int8
            ),
            desc_b,
            dest,
            k_idx=k,
            accumulate=accumulate if cutlass.const_expr(k == 0) else True,
        )


@cute.jit
def _issue_kv_s(slab, bars, tmem_base, input_state, full_state, dtype: cutlass.Constexpr):
    _wait_service(bars.input_full, input_state)
    nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
    mma_ss(
        _mma_desc(dtype),
        _operand(slab, CFG.RESIDENT0, CFG.TILE_M).desc(),
        _input(slab, input_state.idx, 0).desc(),
        nvvm.make_tmem_ptr(tmem_base + CFG.TMEM_S, cutlass.Float32),
    )
    bars.s_full[full_state.idx].arrive(cta_group=CFG.CTA_MMA, pred=nvvm.elect_sync())
    return advance(input_state, CFG.INPUT_STAGES), advance(full_state, CFG.SINGLE_STAGE)


@cute.jit
def _mma_kv(slab, bars, tmem_base, lo, hi, dtype: cutlass.Constexpr):
    input_state = PipelineState.start()
    input_empty_state = PipelineState.start()
    s_state = PipelineState.start()
    dp_state = PipelineState.start()
    p_state = PipelineState.start()
    ds_state = PipelineState.start()
    score = _mma_desc(dtype)
    grad = _mma_desc(dtype, gradient=True)
    if hi > lo:
        input_state, s_state = _issue_kv_s(slab, bars, tmem_base, input_state, s_state, dtype)
    for tile in cutlass.range(lo, hi, 1, unroll=1):
        mma_ss(
            score,
            _operand(slab, CFG.RESIDENT1, CFG.TILE_M).desc(),
            _input(slab, input_empty_state.idx, 1).desc(),
            nvvm.make_tmem_ptr(tmem_base + CFG.TMEM_DP, cutlass.Float32),
        )
        bars.dp_full[dp_state.idx].arrive(cta_group=CFG.CTA_MMA, pred=nvvm.elect_sync())
        dp_state = advance(dp_state, CFG.SINGLE_STAGE)
        _wait_service(bars.p_full, p_state)
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
        p_state = advance(p_state, CFG.SINGLE_STAGE)
        _mma_ts_gapped(
            grad,
            tmem_base + CFG.TMEM_S,
            _input(slab, input_empty_state.idx, 1, transposed=True).desc(),
            nvvm.make_tmem_ptr(tmem_base + CFG.TMEM_G1, cutlass.Float32),
            tile > lo,
        )
        if tile + 1 < hi:
            input_state, s_state = _issue_kv_s(slab, bars, tmem_base, input_state, s_state, dtype)
        _wait_service(bars.ds_full, ds_state)
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
        ds_state = advance(ds_state, CFG.SINGLE_STAGE)
        _mma_ts_gapped(
            grad,
            tmem_base + CFG.TMEM_DP,
            _input(slab, input_empty_state.idx, 0, transposed=True).desc(),
            nvvm.make_tmem_ptr(tmem_base + CFG.TMEM_G0, cutlass.Float32),
            tile > lo,
        )
        bars.input_empty[input_empty_state.idx].arrive(cta_group=CFG.CTA_MMA, pred=nvvm.elect_sync())
        input_empty_state = advance(input_empty_state, CFG.INPUT_STAGES)
    if hi > lo:
        bars.final_grad.arrive(cta_group=CFG.CTA_MMA, pred=nvvm.elect_sync())


@cute.jit
def _mma_q(slab, bars, tmem_base, lo, hi, dtype: cutlass.Constexpr):
    input_state = PipelineState.start()
    input_empty_state = PipelineState.start()
    empty_state = PipelineState.start(phase=1)
    s_state = PipelineState.start()
    ds_state = PipelineState.start()
    ds_empty_state = PipelineState.start()
    score = _mma_desc(dtype)
    grad = _mma_desc(dtype, gradient=True)
    for tile in cutlass.range(lo, hi, 1, unroll=1):
        _wait_service(bars.input_full, input_state)
        _wait_service(bars.q_score_empty, empty_state)
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
        mma_ss(
            score,
            _operand(slab, CFG.RESIDENT0, CFG.TILE_M).desc(),
            _input(slab, input_state.idx, 0).desc(),
            nvvm.make_tmem_ptr(tmem_base + CFG.TMEM_S, cutlass.Float32),
        )
        mma_ss(
            score,
            _operand(slab, CFG.RESIDENT1, CFG.TILE_M).desc(),
            _input(slab, input_state.idx, 1).desc(),
            nvvm.make_tmem_ptr(tmem_base + CFG.TMEM_DP, cutlass.Float32),
        )
        bars.s_full[s_state.idx].arrive(cta_group=CFG.CTA_MMA, pred=nvvm.elect_sync())
        input_state = advance(input_state, CFG.INPUT_STAGES)
        empty_state = advance(empty_state, CFG.SINGLE_STAGE)
        s_state = advance(s_state, CFG.SINGLE_STAGE)
        _wait_service(bars.ds_full, ds_state)
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
        mma_ss(
            grad,
            _q_ds(slab).desc(),
            _input(slab, input_empty_state.idx, 0, transposed=True).desc(),
            nvvm.make_tmem_ptr(tmem_base + CFG.TMEM_G0, cutlass.Float32),
            accumulate=tile > lo,
        )
        bars.input_empty[input_empty_state.idx].arrive(cta_group=CFG.CTA_MMA, pred=nvvm.elect_sync())
        bars.q_ds_empty[ds_empty_state.idx].arrive(cta_group=CFG.CTA_MMA, pred=nvvm.elect_sync())
        ds_state = advance(ds_state, CFG.SINGLE_STAGE)
        input_empty_state = advance(input_empty_state, CFG.INPUT_STAGES)
        ds_empty_state = advance(ds_empty_state, CFG.SINGLE_STAGE)
    if hi > lo:
        bars.final_grad.arrive(cta_group=CFG.CTA_MMA, pred=nvvm.elect_sync())
    _wait_service(bars.q_score_empty, empty_state)
    empty_state = advance(empty_state, CFG.SINGLE_STAGE)


@cute.jit
def _stage_gradient(tmem_base, tmem_offset: cutlass.Constexpr[int], smem, live, dtype: cutlass.Constexpr, group: cutlass.Constexpr[int]):
    row = cute.arch.thread_idx()[0] % CFG.TILE_M
    values = cutlass.Vector.from_elements(tuple(cutlass.Float32(0.0) for _ in range(CFG.GROUP_COLS)), cutlass.Float32)
    if live:
        values = tmem_load_tile(tmem_base + tmem_offset + group * CFG.GROUP_COLS, num_elems=CFG.GROUP_COLS).vec
        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
    smem.base.subview(group * CFG.TILE_M * CFG.GROUP_COLS + row * CFG.GROUP_COLS).data_ptr().store_swizzled(values.to(dtype), alignment=128, swizzle=_SWIZZLE)


@cute.jit
def _compute_kv(
    slab, stats, bars, tmem_base, owner_base, lo, hi, sq, skv, scale, causal: cutlass.Constexpr[bool], dtype: cutlass.Constexpr, group: cutlass.Constexpr[int]
):
    s_state = PipelineState.start()
    dp_state = PipelineState.start()
    stats_full_state = PipelineState.start()
    stats_empty_state = PipelineState.start()
    p_state = PipelineState.start()
    ds_state = PipelineState.start()
    for tile in cutlass.range(lo, hi, 1, unroll=1):
        bars.s_full[s_state.idx].wait(s_state.phase)
        bars.stats_full[stats_full_state.idx].wait(stats_full_state.phase)
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
        p0 = _kv_p_chunk(tmem_base, stats, owner_base, tile * CFG.TILE_N, sq, skv, scale, causal, dtype, group * CFG.GROUP_COLS)
        p1 = _kv_p_chunk(tmem_base, stats, owner_base, tile * CFG.TILE_N, sq, skv, scale, causal, dtype, group * CFG.GROUP_COLS + CFG.POINTWISE_COLS)
        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
        bars.p_full[p_state.idx].arrive()
        bars.dp_full[dp_state.idx].wait(dp_state.phase)
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
        _kv_ds_chunk(tmem_base, p0, stats, owner_base, tile * CFG.TILE_N, sq, skv, scale, causal, dtype, group * CFG.GROUP_COLS)
        _kv_ds_chunk(tmem_base, p1, stats, owner_base, tile * CFG.TILE_N, sq, skv, scale, causal, dtype, group * CFG.GROUP_COLS + CFG.POINTWISE_COLS)
        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
        bars.stats_empty[stats_empty_state.idx].arrive()
        bars.ds_full[ds_state.idx].arrive()
        s_state = advance(s_state, CFG.SINGLE_STAGE)
        dp_state = advance(dp_state, CFG.SINGLE_STAGE)
        stats_full_state = advance(stats_full_state, CFG.SINGLE_STAGE)
        stats_empty_state = advance(stats_empty_state, CFG.SINGLE_STAGE)
        p_state = advance(p_state, CFG.SINGLE_STAGE)
        ds_state = advance(ds_state, CFG.SINGLE_STAGE)
    if hi > lo:
        bars.final_grad.wait(PipelineState.start().phase)
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
    _stage_gradient(tmem_base, CFG.TMEM_G0, _operand(slab, CFG.RESIDENT0, CFG.TILE_M), hi > lo, dtype, group)
    _stage_gradient(tmem_base, CFG.TMEM_G1, _operand(slab, CFG.RESIDENT1, CFG.TILE_M), hi > lo, dtype, group)
    nvvm.fence_proxy("async.shared", space="cta")
    nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
    bars.out_ready.arrive()


@cute.jit
def _compute_q(
    slab, stats, bars, tmem_base, owner_base, lo, hi, sq, skv, scale, causal: cutlass.Constexpr[bool], dtype: cutlass.Constexpr, group: cutlass.Constexpr[int]
):
    row = cute.arch.thread_idx()[0] % CFG.TILE_M
    s_state = PipelineState.start()
    empty_state = PipelineState.start()
    ds_state = PipelineState.start()
    ds_empty_state = PipelineState.start(phase=1)
    bars.q_stats_ready.wait(PipelineState.start().phase)
    for tile in cutlass.range(lo, hi, 1, unroll=1):
        bars.s_full[s_state.idx].wait(s_state.phase)
        bars.q_ds_empty[ds_empty_state.idx].wait(ds_empty_state.phase)
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
        for chunk in cutlass.range_constexpr(CFG.GROUP_COLS // CFG.POINTWISE_COLS):
            col = group * CFG.GROUP_COLS + chunk * CFG.POINTWISE_COLS
            s = tmem_load_tile(tmem_base + CFG.TMEM_S + col, num_elems=CFG.POINTWISE_COLS, ld_num=CFG.POINTWISE_COLS).vec
            dp = tmem_load_tile(tmem_base + CFG.TMEM_DP + col, num_elems=CFG.POINTWISE_COLS, ld_num=CFG.POINTWISE_COLS).vec
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
            p, ds = _probability(s, dp, stats, owner_base, tile * CFG.TILE_N, sq, skv, scale, causal, True, col)
            _q_ds(slab).base.subview(group * CFG.TILE_M * CFG.GROUP_COLS + row * CFG.GROUP_COLS + chunk * CFG.POINTWISE_COLS).data_ptr().store_swizzled(
                ds.to(dtype), alignment=CFG.POINTWISE_COLS * CFG.BPE, swizzle=_SWIZZLE
            )
        nvvm.fence_proxy("async.shared", space="cta")
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
        bars.q_score_empty[empty_state.idx].arrive()
        bars.ds_full[ds_state.idx].arrive()
        s_state = advance(s_state, CFG.SINGLE_STAGE)
        empty_state = advance(empty_state, CFG.SINGLE_STAGE)
        ds_state = advance(ds_state, CFG.SINGLE_STAGE)
        ds_empty_state = advance(ds_empty_state, CFG.SINGLE_STAGE)
    bars.q_ds_empty[ds_empty_state.idx].wait(ds_empty_state.phase)
    ds_empty_state = advance(ds_empty_state, CFG.SINGLE_STAGE)
    if hi > lo:
        bars.final_grad.wait(PipelineState.start().phase)
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
    _stage_gradient(tmem_base, CFG.TMEM_G0, _operand(slab, CFG.RESIDENT0, CFG.TILE_M), hi > lo, dtype, group)
    nvvm.fence_proxy("async.shared", space="cta")
    nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
    bars.out_ready.arrive()


@cute.jit
def _tma_warp(slab, bars, stream0_desc, stream1_desc, o_desc, lo, hi, head, batch, q_owner: cutlass.Constexpr[bool]):
    empty_state = PipelineState.start(phase=1)
    full_state = PipelineState.start()
    o_empty_state = PipelineState.start(phase=1)
    o_full_state = PipelineState.start()
    if cutlass.const_expr(q_owner):
        _wait_service(bars.q_stats_ready, PipelineState.start())
    for tile in cutlass.range(lo, hi, 1, unroll=1):
        _wait_service(bars.input_empty, empty_state)
        slot = full_state.idx
        bars.input_full[slot].arrive(n_bytes=2 * CFG.TILE_N * CFG.HEAD_DIM * CFG.BPE, pred=nvvm.elect_sync())
        tma_load_tile(_input(slab, slot, 0), GmemTileTma(stream0_desc)(0, head, tile * CFG.TILE_N, batch), bars.input_full[slot].smem_ptr)
        tma_load_tile(_input(slab, slot, 1), GmemTileTma(stream1_desc)(0, head, tile * CFG.TILE_N, batch), bars.input_full[slot].smem_ptr)
        empty_state = advance(empty_state, CFG.INPUT_STAGES)
        full_state = advance(full_state, CFG.INPUT_STAGES)
        if cutlass.const_expr(not q_owner):
            _wait_service(bars.o_empty, o_empty_state)
            bars.o_full[o_full_state.idx].arrive(n_bytes=CFG.TILE_N * CFG.HEAD_DIM * CFG.BPE, pred=nvvm.elect_sync())
            tma_load_tile(_operand(slab, CFG.AUX, CFG.TILE_N), GmemTileTma(o_desc)(0, head, tile * CFG.TILE_N, batch), bars.o_full[o_full_state.idx].smem_ptr)
            o_empty_state = advance(o_empty_state, CFG.SINGLE_STAGE)
            o_full_state = advance(o_full_state, CFG.SINGLE_STAGE)
    for _ in cutlass.range_constexpr(CFG.INPUT_STAGES):
        _wait_service(bars.input_empty, empty_state)
        empty_state = advance(empty_state, CFG.INPUT_STAGES)
    if cutlass.const_expr(not q_owner):
        _wait_service(bars.o_empty, o_empty_state)
        o_empty_state = advance(o_empty_state, CFG.SINGLE_STAGE)


@cute.jit
def _stats_warps(slab, stats, bars, lse, owner_base, lo, hi, batch, head, sq, q_owner: cutlass.Constexpr[bool]):
    if cutlass.const_expr(q_owner):
        _write_stats(_operand(slab, CFG.AUX, CFG.TILE_M), _operand(slab, CFG.RESIDENT1, CFG.TILE_M), stats, lse, batch, head, owner_base, sq, CFG.TILE_M)
        bars.q_stats_ready.arrive()
    else:
        row0 = cute.arch.thread_idx()[0] - CFG.STATS_WARP * CFG.WARP_LANES
        row1 = row0 + CFG.STATS_THREADS
        input_state = PipelineState.start()
        empty_state = PipelineState.start(phase=1)
        full_state = PipelineState.start()
        o_full_state = PipelineState.start()
        o_empty_state = PipelineState.start()
        for tile in cutlass.range(lo, hi, 1, unroll=1):
            _wait_service(bars.input_full, input_state)
            _wait_service(bars.o_full, o_full_state)
            lse0, delta0 = _read_stats_row(
                _operand(slab, CFG.AUX, CFG.TILE_N), _input(slab, input_state.idx, 1), lse, batch, head, tile * CFG.TILE_N, sq, CFG.TILE_N, row0
            )
            lse1, delta1 = _read_stats_row(
                _operand(slab, CFG.AUX, CFG.TILE_N), _input(slab, input_state.idx, 1), lse, batch, head, tile * CFG.TILE_N, sq, CFG.TILE_N, row1
            )
            bars.o_empty[o_empty_state.idx].arrive()
            _wait_service(bars.stats_empty, empty_state)
            stats.subview(row0).store(lse0)
            stats.subview(CFG.TILE_N + row0).store(delta0)
            stats.subview(row1).store(lse1)
            stats.subview(CFG.TILE_N + row1).store(delta1)
            bars.stats_full[full_state.idx].arrive()
            input_state = advance(input_state, CFG.INPUT_STAGES)
            empty_state = advance(empty_state, CFG.SINGLE_STAGE)
            full_state = advance(full_state, CFG.SINGLE_STAGE)
            o_full_state = advance(o_full_state, CFG.SINGLE_STAGE)
            o_empty_state = advance(o_empty_state, CFG.SINGLE_STAGE)
        _wait_service(bars.stats_empty, empty_state)
        empty_state = advance(empty_state, CFG.SINGLE_STAGE)


@cute.jit
def _control_warp(slab, bars, out0_desc, out1_desc, owner_base, head, batch, q_owner: cutlass.Constexpr[bool]):
    _wait_service(bars.out_ready, PipelineState.start())
    if nvvm.elect_sync():
        tma_store_tile(_operand(slab, CFG.RESIDENT0, CFG.TILE_M), GmemTileTma(out0_desc)(0, head, owner_base, batch))
        if cutlass.const_expr(not q_owner):
            tma_store_tile(_operand(slab, CFG.RESIDENT1, CFG.TILE_M), GmemTileTma(out1_desc)(0, head, owner_base, batch))
        tma_store_commit()
        tma_store_wait(0)
    nvvm.bar_warp_sync(cute.arch.FULL_MASK)


@cute.jit
def _owner(
    resident0_desc,
    resident1_desc,
    o_desc,
    stream0_desc,
    stream1_desc,
    out0_desc,
    out1_desc,
    lse,
    slab,
    stats,
    bars,
    tmem_base,
    owner_base,
    batch,
    head,
    sq,
    skv,
    scale,
    causal: cutlass.Constexpr[bool],
    q_owner: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
):
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    if warp == CFG.TMA_WARP:
        nbytes = (3 if cutlass.const_expr(q_owner) else 2) * CFG.TILE_M * CFG.HEAD_DIM * CFG.BPE
        bars.resident_full.arrive(n_bytes=nbytes, pred=nvvm.elect_sync())
        tma_load_tile(_operand(slab, CFG.RESIDENT0, CFG.TILE_M), GmemTileTma(resident0_desc)(0, head, owner_base, batch), bars.resident_full.smem_ptr)
        tma_load_tile(_operand(slab, CFG.RESIDENT1, CFG.TILE_M), GmemTileTma(resident1_desc)(0, head, owner_base, batch), bars.resident_full.smem_ptr)
        if cutlass.const_expr(q_owner):
            tma_load_tile(_operand(slab, CFG.AUX, CFG.TILE_M), GmemTileTma(o_desc)(0, head, owner_base, batch), bars.resident_full.smem_ptr)
    bars.resident_full.wait(PipelineState.start().phase)
    nvvm.barrier_cta_sync()
    lo = cutlass.Int32(0)
    if cutlass.const_expr(q_owner):
        hi = cute.ceil_div(skv, CFG.TILE_N)
        if cutlass.const_expr(causal):
            hi = cute.math.min(hi, cute.ceil_div(owner_base + CFG.TILE_M, CFG.TILE_N))
    else:
        hi = cute.ceil_div(sq, CFG.TILE_N)
        if cutlass.const_expr(causal):
            lo = cute.math.min(owner_base // CFG.TILE_N, hi)
    if warp < CFG.COMPUTE_WARPS:
        if warp < CFG.COMPUTE_WARPS // 2:
            if cutlass.const_expr(q_owner):
                _compute_q(slab, stats, bars, tmem_base, owner_base, lo, hi, sq, skv, scale, causal, dtype, 0)
            else:
                _compute_kv(slab, stats, bars, tmem_base, owner_base, lo, hi, sq, skv, scale, causal, dtype, 0)
        else:
            if cutlass.const_expr(q_owner):
                _compute_q(slab, stats, bars, tmem_base, owner_base, lo, hi, sq, skv, scale, causal, dtype, 1)
            else:
                _compute_kv(slab, stats, bars, tmem_base, owner_base, lo, hi, sq, skv, scale, causal, dtype, 1)
    elif warp == CFG.TMA_WARP:
        _tma_warp(slab, bars, stream0_desc, stream1_desc, o_desc, lo, hi, head, batch, q_owner)
    elif warp == CFG.MMA_WARP:
        if cutlass.const_expr(q_owner):
            _mma_q(slab, bars, tmem_base, lo, hi, dtype)
        else:
            _mma_kv(slab, bars, tmem_base, lo, hi, dtype)
    elif (warp >= CFG.STATS_WARP) & (warp < CFG.STATS_WARP + CFG.STATS_WARPS):
        # Each stats warp owns independent state and its stats-empty tail wait.
        # The full count includes all 64 writers, bounding each waiter's reuse.
        _stats_warps(slab, stats, bars, lse, owner_base, lo, hi, batch, head, sq, q_owner)
        if warp == CFG.CONTROL_WARP:
            _control_warp(slab, bars, out0_desc, out1_desc, owner_base, head, batch, q_owner)


@cute.kernel
def bprop_d128_kernel(
    q128: cutlass.GridConstant[tmap.TensorMap],
    k128: cutlass.GridConstant[tmap.TensorMap],
    v128: cutlass.GridConstant[tmap.TensorMap],
    o128: cutlass.GridConstant[tmap.TensorMap],
    do128: cutlass.GridConstant[tmap.TensorMap],
    q64: cutlass.GridConstant[tmap.TensorMap],
    k64: cutlass.GridConstant[tmap.TensorMap],
    v64: cutlass.GridConstant[tmap.TensorMap],
    o64: cutlass.GridConstant[tmap.TensorMap],
    do64: cutlass.GridConstant[tmap.TensorMap],
    dq: cutlass.GridConstant[tmap.TensorMap],
    dk: cutlass.GridConstant[tmap.TensorMap],
    dv: cutlass.GridConstant[tmap.TensorMap],
    lse: cute.Tensor,
    sq: cutlass.Int32,
    skv: cutlass.Int32,
    scale: cutlass.Float32,
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
):
    flat = cute.arch.block_idx()[0]
    q_tiles = cute.ceil_div(sq, CFG.TILE_M)
    owners = q_tiles + cute.ceil_div(skv, CFG.TILE_M)
    if cutlass.const_expr(causal):
        # Head/batch-fastest ranks interleave KV ascending with Q descending.
        batch_heads = lse.shape[0] * lse.shape[1]
        rank = flat // batch_heads
        hb = flat % batch_heads
        head = hb % lse.shape[1]
        batch = hb // lse.shape[1]
        kv_tiles = owners - q_tiles
        common = cute.math.min(q_tiles, kv_tiles)
        owner = cutlass.Int32(0)
        if rank < 2 * common:
            j = rank // 2
            if (rank & 1) != 0:
                owner = q_tiles - 1 - j
            else:
                owner = q_tiles + j
        elif q_tiles > kv_tiles:
            owner = q_tiles - 1 - common - (rank - 2 * common)
        else:
            owner = q_tiles + common + (rank - 2 * common)
    else:
        owner = flat % owners
        head = (flat // owners) % lse.shape[1]
        batch = flat // (owners * lse.shape[1])
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    slab = cutlass.Array(dtype, CFG.SLAB_ELEMS, alignment=1024, space=cutlass.AddressSpace.smem)
    stats = cutlass.Array(cutlass.Float32, CFG.STATS_ELEMS, alignment=1024, space=cutlass.AddressSpace.smem)
    bars = make_bars()
    tmem_slot = cutlass.Array(cutlass.Int32, 1, alignment=16, space=cutlass.AddressSpace.smem)
    if warp == 0:
        if nvvm.elect_sync():
            for i in cutlass.range_constexpr(len(bars)):
                for stage in cutlass.range_constexpr(bars[i].stages):
                    bars[i][stage].init()
    if warp == CFG.CONTROL_WARP:
        tmem_alloc(tmem_slot, CFG.TMEM_COLS, nvvm.CTAGroup.CTA_1)
    nvvm.fence_mbarrier_init()
    nvvm.barrier_cta_sync()
    tmem_base = tmem_slot.load()
    if owner < q_tiles:
        _owner(q128, do128, o128, k64, v64, dq, dv, lse, slab, stats, bars, tmem_base, owner * CFG.TILE_M, batch, head, sq, skv, scale, causal, True, dtype)
    else:
        _owner(
            k128,
            v128,
            o64,
            q64,
            do64,
            dk,
            dv,
            lse,
            slab,
            stats,
            bars,
            tmem_base,
            (owner - q_tiles) * CFG.TILE_M,
            batch,
            head,
            sq,
            skv,
            scale,
            causal,
            False,
            dtype,
        )
    nvvm.barrier_cta_sync()
    if warp == CFG.CONTROL_WARP:
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
        tmem_dealloc(tmem_slot, CFG.TMEM_COLS, nvvm.CTAGroup.CTA_1)


bprop_d128_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


def _tensor_map(tensor, rows):
    return tmap.create_tensor_map_tiled_from_view(
        tensor, box_dims=(1, rows, 1, CFG.HEAD_GRANULE), stride_order=(3, 2, 1, 0), swizzle=tmap.TensorMapSwizzle.s128b
    )


@cute.jit
def host(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    o: cute.Tensor,
    do: cute.Tensor,
    lse: cute.Tensor,
    dq: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    scale: cutlass.Float32,
    causal: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    if cutlass.const_expr(q.element_type not in (cutlass.BFloat16, cutlass.Float16)):
        raise ValueError("dense d128 backward supports only BF16 and FP16")
    if cutlass.const_expr(q.shape[3] != CFG.HEAD_DIM or k.shape[3] != CFG.HEAD_DIM):
        raise ValueError("dense d128 backward requires head dimension 128")
    if cutlass.const_expr(q.shape[2] != k.shape[2]):
        raise ValueError("dense d128 backward requires equal Q and KV head counts")
    if cutlass.const_expr(q.shape[1] <= 0 or k.shape[1] <= 0):
        raise ValueError("dense d128 backward requires positive sequence lengths")
    q_tiles = cute.ceil_div(q.shape[1], CFG.TILE_M)
    kv_tiles = cute.ceil_div(k.shape[1], CFG.TILE_M)
    bprop_d128_kernel(
        _tensor_map(q, CFG.TILE_M),
        _tensor_map(k, CFG.TILE_M),
        _tensor_map(v, CFG.TILE_M),
        _tensor_map(o, CFG.TILE_M),
        _tensor_map(do, CFG.TILE_M),
        _tensor_map(q, CFG.TILE_N),
        _tensor_map(k, CFG.TILE_N),
        _tensor_map(v, CFG.TILE_N),
        _tensor_map(o, CFG.TILE_N),
        _tensor_map(do, CFG.TILE_N),
        _tensor_map(dq, CFG.TILE_M),
        _tensor_map(dk, CFG.TILE_M),
        _tensor_map(dv, CFG.TILE_M),
        lse,
        q.shape[1],
        k.shape[1],
        scale,
        causal,
        q.element_type,
    ).launch(grid=((q_tiles + kv_tiles) * q.shape[2] * q.shape[0], 1, 1), block=(CFG.THREADS, 1, 1), stream=stream)


_host = host
