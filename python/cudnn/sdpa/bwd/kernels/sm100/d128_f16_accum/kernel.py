# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""One-CTA D64/D128 accumulation with dense and packed layout entries."""

from typing import NamedTuple, Optional
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import arith
from cutlass.base_dsl.typing import Pointer
from cutlass.experimental import primitives as nvvm
from cutlass.experimental.cuda import tensor_map as tmap
from cudnn.frost.tile_dsl.barrier import MBarrier, PipelineState, Producer, advance, named_barrier_fence
from cudnn.frost.tile_dsl.handles import GmemTileTma, MmaDesc, SmemTile, tma_slice_runtime_desc
from cudnn.frost.tile_dsl.mask import apply_mask_words, band_mask_words
from cudnn.frost.tile_dsl.mma import mma_ss, mma_ts_step
from cudnn.frost.tile_dsl.pointwise import ffma2, fmul2, fp32_to_fp16
from cudnn.frost.tile_dsl.scheduler import Sched, read_clc_payload
from cudnn.frost.tile_dsl.thd import THD_SETUP_THREADS, thd_port_origin
from cudnn.frost.tile_dsl.tma import (
    tma_load_subtiles,
    tma_load_tile,
    tma_reduce_add_subtile,
    tma_store_commit,
    tma_store_tile,
    tma_store_wait,
    tma_tensormap_acquire,
)
from cudnn.sdpa.bwd.config_sm100_d128_accum import BAR_LAYOUT as BAR_LAYOUT_ACCUM_D128
from cudnn.sdpa.bwd.config_sm100_d128_accum import CFG as CFG_ACCUM_D128
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
from cudnn.sdpa.bwd.kernels.bprop_chain_common import dkv_reduce_host
from cudnn.sdpa.bwd.kernels.sm100._d128_chain import ROWS_PER_CTA, THREADS, cast_dq_host, delta_zero_host
from cudnn.sdpa.bwd.kernels.bprop_thd_reduce import dkv_reduce_thd_origins_host
from cudnn.sdpa.bwd.kernels.sm100._d128_thd_chain import (
    MAP_DK_THD,
    MAP_DO_THD,
    MAP_DQ_THD,
    MAP_DV_THD,
    MAP_K_THD,
    MAP_Q_THD,
    MAP_V_THD,
    _map_ptr,
    _origin_row,
    cast_dq_thd_d128_kernel,
    delta_zero_thd_d128_kernel,
    setup_thd_d128_kernel,
)

# ======================================================================================================================
# section 1: the shared chain kernels (delta / dQ zero, the dQ cast) and the row constants
# (was kernels/sm100/bprop_d128_helpers.py)
# ======================================================================================================================
"""Dense d128 delta, dQ initialization and native output conversion.

Both launches cover dQ's 128-row-rounded sequence: tail rows past S_q zero
their dQ accumulator rows and delta, and the conversion skips them.
"""

_PARAMS = globals().get("FROST_TEMPLATE_PARAMS", 128)
HEAD_DIM = int(_PARAMS) if isinstance(_PARAMS, int) else int(_PARAMS[0])
DQ_OUT = True
if not isinstance(_PARAMS, int) and not bool(_PARAMS[1]):
    raise ValueError("the deterministic dK/dV rendering is in d128_f16_pair/kernel_dkv.py")
RATIO = 1 if isinstance(_PARAMS, int) or len(_PARAMS) < 3 else int(_PARAMS[2])
SPLIT = 1 if isinstance(_PARAMS, int) or len(_PARAMS) < 4 else int(_PARAMS[3])
if (RATIO != 1 or SPLIT != 1) and DQ_OUT:
    raise ValueError("grouped query heads and q-range splits are served by the dK / dV-only rendering")
if SPLIT not in (1, 2, 4):
    raise ValueError("the q-range split must be 1, 2 or 4")

# ======================================================================================================================
# section 3: the dense MHA ACCUMULATION body (3-launch chain, fp32 dQ workspace)
# (was kernels/sm100/bprop_d128_accum_f16.py)
# ======================================================================================================================
"""Dense d128 backward with FP32 dQ accumulation and native dK/dV stores.

The five-product pipeline retains 33 live mbarriers. Native output TMA stores
are warp-uniform, dQ reductions elected; both return 32-lane credits.

Sequence lengths need not be tile multiples: the tail Q / K / V / dO boxes
zero-fill, the tail dK / dV stores and dQ reductions clip, the tail Stats rows
publish -inf and the tail key rows are masked before the exponent.
"""

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
def tma_reduce_subtile(smem_tile, gmem_slice, subtile: cutlass.Constexpr[int]):
    """One elected source-sized FP32 add; caller owns commit and wait/credit."""
    desc = gmem_slice.tma_desc.get_ptr()
    coords = [gmem_slice.coord_d + cutlass.Int32(subtile * smem_tile.tma_granu_elems)] + list(gmem_slice.coords[1:])
    src = smem_tile.base.subview(subtile * smem_tile.tma_subtile_stride_elems)
    if nvvm.elect_sync():
        nvvm.cp_async_bulk_tensor_reduce(desc, src, "add", coords)


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
def _p_values_accum(
    slab, tmem_base, row, group, block, qtile, lo, skv, scale, causal: cutlass.Constexpr[bool], dtype: cutlass.Constexpr, window: cutlass.Constexpr[int] = 0
):
    address = tmem_base + group * 64
    a = _tmem_load(address, 32)
    b = _tmem_load(address + 32, 32)
    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
    scores = cutlass.Vector.from_elements(tuple(a[i] for i in range(32)) + tuple(b[i] for i in range(32)), cutlass.Float32)
    p = []
    packed = []
    scaled = scale * cutlass.Float32(LOG2E)
    neg_inf = cutlass.Float32(float("-inf"))
    kr = block * CFG_ACCUM.M + row
    # The tail KV tile's zero-filled key rows (kr >= skv) must not exponentiate:
    # exp2(-stat) overflows for a very negative LSE, and inf times their zero K
    # would put NaN into dQ.  Only the last KV tile pays the select.
    key_tail = (block + 1) * CFG_ACCUM.M > skv
    key_valid = kr < skv
    if cutlass.const_expr(window > 0):
        window_edge = (qtile + 1) * CFG_ACCUM.N - 1 - block * CFG_ACCUM.M >= window
    for chunk in cutlass.range_constexpr(16):
        stat = _view(slab, CFG_ACCUM.STATS, cutlass.Float32, CFG_ACCUM.N).subview(group * 64 + chunk * 4).load(vector_size=4, alignment=16)
        for pair in cutlass.range_constexpr(2):
            i = chunk * 4 + pair * 2
            s0, s1 = scores[i], scores[i + 1]
            qc = qtile * CFG_ACCUM.N + group * 64 + i
            if cutlass.const_expr(causal):
                if qtile == lo:
                    s0 = cutlass.Float32(arith.select((kr <= qc).ir_value(), s0.ir_value(), neg_inf.ir_value()))
                    s1 = cutlass.Float32(arith.select((kr <= qc + 1).ir_value(), s1.ir_value(), neg_inf.ir_value()))
            if cutlass.const_expr(window > 0):
                if window_edge:
                    s0 = cutlass.Float32(arith.select((kr + (window - 1) >= qc).ir_value(), s0.ir_value(), neg_inf.ir_value()))
                    s1 = cutlass.Float32(arith.select((kr + (window - 1) >= qc + 1).ir_value(), s1.ir_value(), neg_inf.ir_value()))
            if key_tail:
                s0 = cutlass.Float32(arith.select(key_valid.ir_value(), s0.ir_value(), neg_inf.ir_value()))
                s1 = cutlass.Float32(arith.select(key_valid.ir_value(), s1.ir_value(), neg_inf.ir_value()))
            e0, e1 = ffma2(scaled, scaled, s0, s1, stat[pair * 2], stat[pair * 2 + 1])
            p0 = cute.math.exp2(e0, fastmath=True)
            p1 = cute.math.exp2(e1, fastmath=True)
            p.extend((p0, p1))
            packed.append(fp32_to_fp16(p0, p1, dtype=dtype))
    _tmem_store(address, cutlass.Vector.from_elements(tuple(packed), cutlass.Int32))
    return cutlass.Vector.from_elements(tuple(p), cutlass.Float32)


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
def _softmax_loop_accum(
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
        lo, hi = _bounds_accum(block, sq, causal, window)
        if lo < hi:
            _stats_accum(slab, lse, delta, row, group, lo * CFG_ACCUM.N, sq, batch, head, scale)
        for qtile in cutlass.range(lo, hi, 1, unroll=1):
            _wait_tcgen_accum(bars.s_done, state.phase)
            p = _p_values_accum(slab, tmem_base, row, group, block, qtile, lo, skv, scale, causal, dtype, window)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
            bars.p_ready.arrive()
            _wait_tcgen_accum(bars.dp_done, state.phase)
            _ds_values_accum(slab, tmem_base, group, p, scale, dtype)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
            bars.ds_ready.arrive()
            if qtile + 1 < hi:
                _stats_accum(slab, lse, delta, row, group, (qtile + 1) * CFG_ACCUM.N, sq, batch, head, scale)
            state = advance(state, 1)
        if cutlass.const_expr(causal):
            _credit_accum(bars.sched_empty[sched_state.idx])
        block, head, batch, valid, sched_state = _next_tile_accum(sched, bars, sched_state, nheads, nk, causal, swizzle, batch_heads)


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
def _epilogue_loop_accum(
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
    ds_state = PipelineState.start()
    dq_state = PipelineState.start()
    empty_state = PipelineState.start()
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit_accum(bars.sched_empty[sched_state.idx])
        lo, hi = _bounds_accum(block, sq, causal, window)
        for _ in cutlass.range(lo, hi, 1, unroll=1):
            _wait_tcgen_accum(bars.ds_ready, ds_state.phase)
            ds0 = _tmem_load(tmem_base + 128, 32, cutlass.Int32)
            ds1 = _tmem_load(tmem_base + 192, 32, cutlass.Int32)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
            _wait_accum(bars.dq_empty[0], empty_state.phase)
            _store_words_accum(slab, CFG_ACCUM.ALIAS, row, ds0)
            _wait_accum(bars.dq_empty[1], empty_state.phase)
            _store_words_accum(slab, CFG_ACCUM.ALIAS + CFG_ACCUM.SLICE_BYTES, row, ds1)
            nvvm.fence_proxy("async.shared", space="cta")
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
            bars.ds_smem.arrive()
            _wait_tcgen_accum(bars.dq_done, dq_state.phase)
            dq0 = _tmem_load(tmem_base + 128, 64)
            if cutlass.const_expr(CFG_ACCUM.D == 128):
                dq1 = _tmem_load(tmem_base + 192, 64)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
            bars.dp_empty.arrive()
            if cutlass.const_expr(CFG_ACCUM.D == 128):
                vals = cutlass.Vector.from_elements(tuple(dq0[i] for i in range(64)) + tuple(dq1[i] for i in range(64)), cutlass.Float32)
            else:
                vals = cutlass.Vector.from_elements(tuple(dq0[i] for i in range(64)), cutlass.Float32)
            for i in cutlass.range_constexpr(CFG_ACCUM.DQ_SLICES):
                _wait_accum(bars.dq_empty[i], empty_state.phase)
                _store_dq_slice_accum(slab, row, vals, i)
                nvvm.fence_proxy("async.shared", space="cta")
                bars.dq_full[i].arrive()
            ds_state = advance(ds_state, 1)
            dq_state = advance(dq_state, 1)
            empty_state = advance(empty_state, 1)
        if lo >= hi:
            _wait_tcgen_accum(bars.dq_done, dq_state.phase)
            dq_state = advance(dq_state, 1)
        # Observe every terminal source-credit phase before the next CLC
        # owner bootstraps these barriers again. Native dV/dK alone wait on
        # slices 1/3, leaving slices 0/2 unobserved after the final dQ or a
        # zero-trip owner. Each completed mbarrier phase needs a successful
        # wait before an arrival in its next phase.
        for i in cutlass.range_constexpr(CFG_ACCUM.DQ_SLICES):
            _wait_accum(bars.dq_empty[i], empty_state.phase)
        _native_epilogue_accum(slab, tmem_base, bars, row, lo < hi, empty_state.phase, dtype, False)
        _native_epilogue_accum(slab, tmem_base, bars, row, lo < hi, empty_state.phase, dtype, True)
        empty_state = advance(empty_state, 1)
        block, head, batch, valid, sched_state = _next_tile_accum(sched, bars, sched_state, nheads, nk, causal, swizzle, batch_heads)


@cute.jit
def _ts_accum(desc, plane, bdesc, dst, accum):
    for k in cutlass.range_constexpr(CFG_ACCUM.N // CFG_ACCUM.K_HW):
        mma_ts_step(
            desc, nvvm.make_tmem_ptr(plane + (k // 4) * 32, cutlass.Int8), bdesc, dst, k_idx=k, accumulate=accum if cutlass.const_expr(k == 0) else True
        )


@cute.jit
def _mma_loop_accum(
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
    dqdesc = _mma_desc_accum(dtype, atranspose=True, btranspose=True, N=CFG_ACCUM.D, K=CFG_ACCUM.M)
    kd = _operand_accum(slab, CFG_ACCUM.K, dtype).desc()
    vd = _operand_accum(slab, CFG_ACCUM.V, dtype).desc()
    dod = _operand_accum(slab, CFG_ACCUM.DO, dtype).desc()
    dot = _operand_accum(slab, CFG_ACCUM.DO, dtype, transpose=True).desc()
    dst = _operand_accum(slab, CFG_ACCUM.ALIAS, dtype, transpose=True).desc()
    kt = _operand_accum(slab, CFG_ACCUM.K, dtype, transpose=True).desc()
    s = nvvm.make_tmem_ptr(tmem_base, cutlass.Float32)
    dp = nvvm.make_tmem_ptr(tmem_base + 128, cutlass.Float32)
    dv = nvvm.make_tmem_ptr(tmem_base + 256, cutlass.Float32)
    dk = nvvm.make_tmem_ptr(tmem_base + 256 + CFG_ACCUM.D, cutlass.Float32)
    q_state = PipelineState.start()
    do_state = PipelineState.start()
    kv_state = PipelineState.start()
    ready_state = PipelineState.start()
    dp_empty = PipelineState.start(phase=1)
    out_empty = PipelineState.start(phase=1)
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit_accum(bars.sched_empty[sched_state.idx])
        lo, hi = _bounds_accum(block, sq, causal, window)
        if lo < hi:
            _wait_tcgen_accum(bars.k_full, kv_state.phase)
            _wait_tcgen_accum(bars.q_full[q_state.idx], q_state.phase)
            qd = _operand_accum(slab, CFG_ACCUM.Q + q_state.idx * CFG_ACCUM.OPERAND_BYTES, dtype).desc()
            mma_ss(score, kd, qd, s)
            _commit(bars.s_done)
            _wait_tcgen_accum(bars.v_full, kv_state.phase)
        _wait_tcgen_accum(bars.dvk_empty, out_empty.phase)
        out_empty = advance(out_empty, 1)
        for qtile in cutlass.range(lo, hi, 1, unroll=1):
            _wait_tcgen_accum(bars.dp_empty, dp_empty.phase)
            for half in cutlass.range_constexpr(CFG_ACCUM.DO_SUBTILES):
                _wait_tcgen_accum(bars.do_full[half], do_state.phase)
                mma_ss(score, vd, dod, dp, k_start=half * _DO_STEPS, k_count=_DO_STEPS, accumulate=half > 0)
            _commit(bars.dp_done)
            _wait_tcgen_accum(bars.p_ready, ready_state.phase)
            _ts_accum(grad, tmem_base, dot, dv, qtile > lo)
            _commit(bars.do_empty)
            q_next = advance(q_state, CFG_ACCUM.Q_STAGES)
            if qtile + 1 < hi:
                _wait_tcgen_accum(bars.q_full[q_next.idx], q_next.phase)
                qd_next = _operand_accum(slab, CFG_ACCUM.Q + q_next.idx * CFG_ACCUM.OPERAND_BYTES, dtype).desc()
                mma_ss(score, kd, qd_next, s)
                _commit(bars.s_done)
            _wait_tcgen_accum(bars.ds_ready, ready_state.phase)
            qt = _operand_accum(slab, CFG_ACCUM.Q + q_state.idx * CFG_ACCUM.OPERAND_BYTES, dtype, transpose=True).desc()
            _ts_accum(grad, tmem_base + 128, qt, dk, qtile > lo)
            _commit(bars.q_empty[q_state.idx])
            _wait_tcgen_accum(bars.ds_smem, ready_state.phase)
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
        block, head, batch, valid, sched_state = _next_tile_accum(sched, bars, sched_state, nheads, nk, causal, swizzle, batch_heads)


@cute.jit
def _load_loop_accum(
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
        lo, hi = _bounds_accum(block, sq, causal, window)
        if lo < hi:
            _wait_accum(bars.k_empty, kv_state.phase)
            bars.k_full.arrive(n_bytes=CFG_ACCUM.OPERAND_BYTES, pred=nvvm.elect_sync())
            tma_load_tile(_operand_accum(slab, CFG_ACCUM.K, dtype), GmemTileTma(k_desc)(0, block * CFG_ACCUM.M, head, batch), bars.k_full.smem_ptr)
            _wait_accum(bars.v_empty, kv_state.phase)
            bars.v_full.arrive(n_bytes=CFG_ACCUM.OPERAND_BYTES, pred=nvvm.elect_sync())
            tma_load_tile(_operand_accum(slab, CFG_ACCUM.V, dtype), GmemTileTma(v_desc)(0, block * CFG_ACCUM.M, head, batch), bars.v_full.smem_ptr)
            kv_state = advance(kv_state, 1)
        for qtile in cutlass.range(lo, hi, 1, unroll=1):
            _wait_accum(bars.q_empty[q_state.idx], q_state.phase)
            bars.q_full[q_state.idx].arrive(n_bytes=CFG_ACCUM.OPERAND_BYTES, pred=nvvm.elect_sync())
            tma_load_tile(
                _operand_accum(slab, CFG_ACCUM.Q + q_state.idx * CFG_ACCUM.OPERAND_BYTES, dtype),
                GmemTileTma(q_desc)(0, qtile * CFG_ACCUM.N, head, batch),
                bars.q_full[q_state.idx].smem_ptr,
            )
            _wait_accum(bars.do_empty, do_state.phase)
            for half in cutlass.range_constexpr(CFG_ACCUM.DO_SUBTILES):
                bars.do_full[half].arrive(n_bytes=CFG_ACCUM.SLICE_BYTES, pred=nvvm.elect_sync())
            for half in cutlass.range_constexpr(CFG_ACCUM.DO_SUBTILES):
                tma_load_subtiles(
                    _operand_accum(slab, CFG_ACCUM.DO, dtype), GmemTileTma(do_desc)(0, qtile * CFG_ACCUM.N, head, batch), bars.do_full[half].smem_ptr, half, 1
                )
            q_state = advance(q_state, CFG_ACCUM.Q_STAGES)
            do_state = advance(do_state, 1)
        block, head, batch, valid, sched_state = _next_tile_accum(sched, bars, sched_state, nheads, nk, causal, swizzle, batch_heads)


# DIAGNOSTIC lever ``diag_delete_dq_reduce_add`` (``Constexpr[bool]``, default False = the shipped kernel, rendered PTX-identical):
# the store warp skips the four fp32 dQ reduce-add issues per q tile and nothing else (every dq_full wait, commit, wait_group and
# dq_empty arrive stays: the barrier table is unchanged).  Armed, the OUTPUT IS WRONG BY DESIGN (dQ stays zero) -- a timing-only
# ceiling for the L2 reduce-add term; the dense MHA entry never arms it, the grouped dense kernel threads it from its adapter record.
# The THD twin's header (section 5 of this module) carries the full statement.
@cute.jit
def _store_loop_accum(
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
    diag_delete_dq_reduce_add: cutlass.Constexpr[bool] = False,
    window: cutlass.Constexpr[int] = 0,
):
    nvvm.setmaxregister(CFG_ACCUM.SERVICE_REGS, nvvm.SetMaxRegisterAction.DECREASE)
    dq_state = PipelineState.start()
    out_state = PipelineState.start()
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit_accum(bars.sched_empty[sched_state.idx])
        for i in cutlass.range_constexpr(CFG_ACCUM.DQ_SLICES):
            bars.dq_empty[i].arrive()  # all32 lanes, one bootstrap per KV tile
        lo, hi = _bounds_accum(block, sq, causal, window)
        for qtile in cutlass.range(lo, hi, 1, unroll=1):
            for i in cutlass.range_constexpr(CFG_ACCUM.DQ_SLICES):
                _wait_accum(bars.dq_full[i], dq_state.phase)
                # DIAGNOSTIC D2 (default off = the shipped code): only the reduce-add issue is deleted; commit / wait / arrives stay.
                if cutlass.const_expr(not diag_delete_dq_reduce_add):
                    tma_reduce_subtile(
                        _operand_accum(slab, CFG_ACCUM.ALIAS, cutlass.Float32, fp32=True), GmemTileTma(dq_desc)(0, qtile * CFG_ACCUM.N, head, batch), i
                    )
                tma_store_commit()
            for i in cutlass.range_constexpr(CFG_ACCUM.DQ_SLICES):
                tma_store_wait(CFG_ACCUM.DQ_SLICES - 1 - i)
                bars.dq_empty[i].arrive()  # all32 after corresponding source read
            dq_state = advance(dq_state, 1)
        _wait_accum(bars.dv_full, out_state.phase)
        # Exact backend distinction: native TMA store is warp-uniform, no elect.
        tma_store_tile(_operand_accum(slab, CFG_ACCUM.ALIAS, dtype), GmemTileTma(dv_desc)(0, block * CFG_ACCUM.M, head, batch))
        _wait_accum(bars.dk_full, out_state.phase)
        tma_store_tile(_operand_accum(slab, CFG_ACCUM.ALIAS + CFG_ACCUM.OPERAND_BYTES, dtype), GmemTileTma(dk_desc)(0, block * CFG_ACCUM.M, head, batch))
        out_state = advance(out_state, 1)
        tma_store_commit()
        tma_store_wait(0)
        bars.dvk_empty.arrive()
        block, head, batch, valid, sched_state = _next_tile_accum(sched, bars, sched_state, nheads, nk, causal, swizzle, batch_heads)


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
        _softmax_loop_accum(
            slab, bars, sched, tmem_base, lse, delta, block, head, batch, sq, skv, nk, nheads, batch_heads, swizzle, scale, causal, dtype, window
        )
    elif warp < 12:
        _epilogue_loop_accum(slab, bars, sched, tmem_base, block, head, batch, sq, nk, nheads, batch_heads, swizzle, causal, dtype, window)
    elif warp == 12:
        _mma_loop_accum(slab, bars, sched, tmem_base, block, head, batch, sq, nk, nheads, batch_heads, swizzle, causal, dtype, window)
    elif warp == 13:
        _load_loop_accum(slab, bars, sched, q_desc, k_desc, v_desc, do_desc, block, head, batch, sq, nk, nheads, batch_heads, swizzle, causal, dtype, window)
    elif warp == 14:
        _store_loop_accum(slab, bars, sched, dq_desc, dk_desc, dv_desc, block, head, batch, sq, nk, nheads, batch_heads, swizzle, causal, dtype, False, window)
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


# ======================================================================================================================
# section 4: the THD chain kernels (setup / delta_zero / cast_dq) and the THD chain host
# (was kernels/sm100/bprop_d128_accum_thd_helpers.py)
# ======================================================================================================================
"""Native THD metadata/maps, Delta+dQ initialization, and dQ conversion.

All caller ports retain their independent ragged origins. Compact prefixes
address only the internal Delta and FP32 dQ regions. Lengths remain on device;
launch bounds and physical capacities are runtime declaration metadata.

Grouped chain, op-level field ``dkv_accum = "fp32_reduce"``: ``delta_zero_thd_d128_kernel`` also zero-fills the per-KV-head fp32 dK / dV
accumulators over each sequence's live kv rows and ``cast_dq_thd_d128_kernel`` also converts them once to the IO dtype at each output's own
ragged origin -- both through the Optional trailing tensors (``None`` = dropped from the signature: the MHA chain and the default grouped
chain render byte-identically); the grid's row extent then covers max(max_sq, max_skv) rows and z = Hq (the kv heads are the first Hkv).
"""


@cute.jit
def host_thd(
    q,
    k,
    v,
    o,
    do,
    lse,
    dq,
    dk,
    dv,
    dq_accum,
    delta,
    meta,
    origins,
    maps,
    q_lens,
    kv_lens,
    ro,
    origin_specs: cutlass.Constexpr,
    lens_form,
    batch: cutlass.Constexpr[int],
    max_sq,
    max_skv,
    scale,
    causal: cutlass.Constexpr[bool],
    stream,
    diag_delete_dq_reduce_add: cutlass.Constexpr[bool] = False,
):
    """Four declared native launches; no host reads, tensor copies or allocation."""
    setup_thd_d128_kernel(
        _tensor_map_thd(q, CFG_ACCUM_D128.HEAD_BOX),
        _tensor_map_thd(k, CFG_ACCUM_D128.HEAD_BOX),
        _tensor_map_thd(v, CFG_ACCUM_D128.HEAD_BOX),
        _tensor_map_thd(do, CFG_ACCUM_D128.HEAD_BOX),
        _tensor_map_thd(dq_accum, CFG_ACCUM_D128.DQ_BOX),
        _tensor_map_thd(dk, CFG_ACCUM_D128.HEAD_BOX),
        _tensor_map_thd(dv, CFG_ACCUM_D128.HEAD_BOX),
        q,
        k,
        v,
        do,
        dq_accum,
        dk,
        dv,
        meta,
        origins,
        maps,
        q_lens,
        kv_lens,
        *ro,
        origin_specs,
        lens_form,
        cutlass.Int32(batch),
    ).launch(grid=(1, 1, 1), block=(THD_SETUP_THREADS, 1, 1), stream=stream)
    q_grid = cute.math.max((max_sq + ROWS_PER_CTA - 1) // ROWS_PER_CTA, cutlass.Int32(1))
    delta_zero_thd_d128_kernel(o, do, delta, dq_accum, meta, origins, cutlass.Int32(batch), origin_specs).launch(
        grid=(q_grid, batch, q.shape[2]),
        block=(THREADS, 1, 1),
        stream=stream,
    )
    host_accum_thd(
        maps,
        lse,
        delta,
        meta,
        origins,
        cutlass.Int32(batch),
        q.shape[2],
        max_skv,
        _origin_row(origin_specs, 8),
        scale,
        causal,
        q.element_type,
        stream,
        diag_delete_dq_reduce_add,
    )
    cast_dq_thd_d128_kernel(dq_accum, dq, meta, origins, cutlass.Int32(batch), _origin_row(origin_specs, 5), q.element_type).launch(
        grid=(q_grid, batch, q.shape[2]),
        block=(THREADS, 1, 1),
        stream=stream,
    )


# ======================================================================================================================
# section 5: the packed THD MHA accumulation body
# (was kernels/sm100/bprop_d128_accum_thd_f16.py)
# ======================================================================================================================
"""Native packed THD MHA d128 backward with FP32 dQ accumulation.

Sequence-local tensor maps clip every input and output tail, including caller
ports with independent ragged origins. The dense five-product pipeline and all
33 mbarriers retain their lane counts. Natural CLC visits a declared envelope;
invalid owners take the complete zero-trip protocol without touching a map.

DIAGNOSTIC lever ``diag_delete_dq_reduce_add`` (``Constexpr[bool]``, default False = the shipped kernel, rendered
PTX-identical; pinned by ``test_sdpa_bwd_d128_diag_dq_reduce_add_sm100.py``): the store warp skips the four
``tma_reduce_add_subtile`` issues of the fp32 dQ slices per q tile and nothing else -- every ``dq_full`` wait,
``tma_store_commit``, ``tma_store_wait`` and ``dq_empty`` arrive stays, so the barrier table (arrive / wait sites and
their lane sums) is unchanged.  Armed, the OUTPUT IS WRONG BY DESIGN (dQ stays zero): a timing-only ceiling for the
L2 reduce-add term, never selected by a graph, never shipped armed.  Threaded from the adapter record
(``SdpaBwdDslSm100D128.diag_delete_dq_reduce_add``) through ``compile_host`` -> ``host`` -> the kernel.

Op-level field ``dkv_accum = "fp32_reduce"`` (``Constexpr[bool]`` ``dkv_fp32`` on ``_epilogue_loop`` / ``_store_loop``, default False = the
shipped kernel, rendered PTX-identical; set by the GROUPED THD kernel only, threaded from ``SdpaBwdDslSm100D128.dkv_accum`` through the grouped
chain; the MHA kernel never sets it): the epilogue stages the fp32 dV then dK accumulator tiles as the four [M x DQ_BOX] fp32 SW128 slices of
the ALIAS slab (the dQ staging image, ``_native_epilogue_fp32``) instead of the bf16 pack, and the store warp reduce-adds each slice
(``tma_reduce_add_subtile`` add.f32, the dQ slices' primitive) into the per-KV-head fp32 workspace maps carried by the MAP_DV / MAP_DK slots
(kv head = head // ratio).  Barrier rows touched: dv_full / dk_full keep their lanes and sites (the published bytes are fp32 slices); dq_empty
gains ONE release per item on both sides -- the store warp arrives dq_empty[i] after the dV slice i source read (``tma_store_wait(3 - i)``, P7)
and the epilogue's dK round waits dq_empty[i] at that phase before overwriting slice i (the dK staging's WAR edge), advancing its empty_state
once more per item; the bootstrap, the terminal observation, dvk_empty and every other row are unchanged.  A zero-trip owner stages zeros and
issues no reduce-add (the workspace is zero-filled); every commit / wait / arrive is kept.  A ROUNDING CHANGE at the chain level (fp32 group
accumulation in hardware order replaces 8 bf16 partials summed in a fixed order), an op attribute: the test module
``test_sdpa_bwd_d128_dkv_accum_sm100.py`` pins the default rendering and the fp32 census.
"""


MAP_COUNT_THD = 7


class BarsThd(NamedTuple):
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


def _bars_thd(slab):
    kinds = {"mma": Producer.MMA_COMMIT, "tma": Producer.TMA_LOAD, "thread": Producer.THREAD}
    return BarsThd(
        *(
            MBarrier(_view(slab, off, cutlass.Int64, slots), stages=slots, init_count=count, producer=kinds[kind], try_wait=True)
            for _, off, slots, count, kind in BAR_LAYOUT_ACCUM_D128
        )
    )


def _operand_thd(slab, offset, dtype, *, transpose=False, fp32=False):
    box = CFG_ACCUM_D128.DQ_BOX if fp32 else CFG_ACCUM_D128.HEAD_BOX
    return SmemTile(
        base=_view(slab, offset, dtype, CFG_ACCUM_D128.M * CFG_ACCUM_D128.D, alignment=1024),
        elems_per_stage=CFG_ACCUM_D128.M * CFG_ACCUM_D128.D,
        leading_byte_offset=CFG_ACCUM_D128.M * CFG_ACCUM_D128.SWIZZLE_BYTES if transpose else 16,
        stride_byte_offset=8 * CFG_ACCUM_D128.SWIZZLE_BYTES,
        layout=CFG_ACCUM_D128.SWIZZLE_LAYOUT,
        tma_loads_per_tile=CFG_ACCUM_D128.D // box,
        tma_granu_elems=box,
        tma_subtile_stride_elems=CFG_ACCUM_D128.M * box,
        desc_version=CFG_ACCUM_D128.DESC_VERSION,
    )


def _mma_desc_thd(dtype, *, atranspose=False, btranspose=False):
    return MmaDesc(
        M=CFG_ACCUM_D128.M,
        N=CFG_ACCUM_D128.N,
        K=CFG_ACCUM_D128.D,
        bpe_a=CFG_ACCUM_D128.BPE,
        bpe_b=CFG_ACCUM_D128.BPE,
        tile_k_hw=CFG_ACCUM_D128.K_HW,
        atranspose=atranspose,
        btranspose=btranspose,
        cta_group=1,
        idesc=nvvm.Tcgen05InstrDesc.build(
            c_dtype=cutlass.Float32,
            a_dtype=dtype,
            b_dtype=dtype,
            m_dim=CFG_ACCUM_D128.M,
            n_dim=CFG_ACCUM_D128.N,
            a_major=int(atranspose),
            b_major=int(btranspose),
        ),
        kind=nvvm.Tcgen05MMAKind.F16,
    )


@cute.jit
def _wait_thd(bar, phase):
    bar.wait(phase)
    # All full-warp consumers reconverge before any elected credit/issue.
    nvvm.bar_warp_sync(cute.arch.FULL_MASK)


@cute.jit
def _wait_tcgen_thd(bar, phase):
    _wait_thd(bar, phase)
    nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)


@cute.jit
def _credit_thd(bar):
    if nvvm.elect_sync():
        bar.arrive()


@cute.jit
def _next_tile_thd(sched, bars, state):
    _wait_thd(bars.sched_full[state.idx], state.phase)
    x, yz, valid = read_clc_payload(sched, state.idx * 8, warp_broadcast=True)
    return x, yz & 0xFFFF, (yz >> 16) & 0xFFFF, valid, advance(state, CFG_ACCUM_D128.SCHED_STAGES)


def _tensor_map_thd(tensor, box):
    # Logical TMA coordinates (d, sequence, head, batch), source-proved order.
    return tmap.create_tensor_map_tiled_from_view(
        tensor, box_dims=(1, CFG_ACCUM_D128.M, 1, box), stride_order=(3, 1, 2, 0), swizzle=tmap.TensorMapSwizzle.s128b
    )


@cute.jit
def _bounds_thd(block, meta, batch, n_batch, causal: cutlass.Constexpr[bool]):
    data = cutlass.make_array_view(meta)
    q_start = cutlass.Int32(data[n_batch + batch])
    seq_q = cutlass.Int32(data[n_batch + batch + 1]) - q_start
    seq_k = cutlass.Int32(data[batch])
    lo = block if cutlass.const_expr(causal) else cutlass.Int32(0)
    hi = (seq_q + CFG_ACCUM_D128.N - 1) // CFG_ACCUM_D128.N
    if block * CFG_ACCUM_D128.M >= seq_k:
        lo = cutlass.Int32(0)
        hi = cutlass.Int32(0)
    return lo, hi, seq_q, seq_k, q_start


@cute.jit
def _stats_thd(slab, lse, delta, row, group, qbase, batch, head, scale, q_start, seq_q, origins, stats_origin: cutlass.Constexpr[int]):
    # Full 256-lane named barriers also cover inactive tail rows. Invalid rows
    # never dereference a Stats/Delta address, including an all-empty carrier.
    named_barrier_fence(CFG_ACCUM_D128.SOFTMAX_BARRIER, CFG_ACCUM_D128.SOFTMAX_LANES)
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
        _view(slab, CFG_ACCUM_D128.STATS, cutlass.Float32, CFG_ACCUM_D128.N).subview(row).store(stat)
    else:
        stat = cutlass.Float32(0.0)
        if local_row < seq_q:
            stat = -scale * cutlass.Float32(delta[0, head, cutlass.Int64(q_start) + local_row])
        _view(slab, CFG_ACCUM_D128.DELTA, cutlass.Float32, CFG_ACCUM_D128.N).subview(row).store(stat)
    named_barrier_fence(CFG_ACCUM_D128.SOFTMAX_BARRIER, CFG_ACCUM_D128.SOFTMAX_LANES)


@cute.jit
def _p_values_thd(slab, tmem_base, row, group, block, qtile, seq_q, seq_k, scale, causal: cutlass.Constexpr[bool], dtype: cutlass.Constexpr):
    address = tmem_base + group * 64
    a = _tmem_load(address, 32)
    b = _tmem_load(address + 32, 32)
    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
    scores = cutlass.Vector.from_elements(tuple(a[i] for i in range(32)) + tuple(b[i] for i in range(32)), cutlass.Float32)
    values = []
    scaled = scale * cutlass.Float32(LOG2E)
    for chunk in cutlass.range_constexpr(16):
        stat = _view(slab, CFG_ACCUM_D128.STATS, cutlass.Float32, CFG_ACCUM_D128.N).subview(group * 64 + chunk * 4).load(vector_size=4, alignment=16)
        for pair in cutlass.range_constexpr(2):
            i = chunk * 4 + pair * 2
            e0, e1 = ffma2(scaled, scaled, scores[i], scores[i + 1], stat[pair * 2], stat[pair * 2 + 1])
            values.extend((cute.math.exp2(e0, fastmath=True), cute.math.exp2(e1, fastmath=True)))
    # KV-major transpose of the shared mask band: Q columns >= key for causal,
    # below the actual Q length, and entirely dead for a padded KV row. Select
    # exact zero AFTER exp2 (also safe for a no-key row whose LSE is -inf).
    key = block * CFG_ACCUM_D128.M + row
    qbase = qtile * CFG_ACCUM_D128.N + group * 64
    lo = key if cutlass.const_expr(causal) else None
    hi = cutlass.Int32(arith.select((key < seq_k).ir_value(), seq_q.ir_value(), cutlass.Int32(0).ir_value()))
    words = band_mask_words(lo, hi, qbase, 64)
    p = apply_mask_words(cutlass.Vector.from_elements(tuple(values), cutlass.Float32), words, mask_value=0.0, n_cols=64)
    packed = cutlass.Vector.from_elements(tuple(fp32_to_fp16(p[2 * i], p[2 * i + 1], dtype=dtype) for i in range(32)), cutlass.Int32)
    _tmem_store(address, packed)
    return p


@cute.jit
def _ds_values_thd(slab, tmem_base, group, p, scale, dtype: cutlass.Constexpr):
    address = tmem_base + 128 + group * 64
    a = _tmem_load(address, 32)
    b = _tmem_load(address + 32, 32)
    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
    dp = cutlass.Vector.from_elements(tuple(a[i] for i in range(32)) + tuple(b[i] for i in range(32)), cutlass.Float32)
    packed = []
    for chunk in cutlass.range_constexpr(16):
        stat = _view(slab, CFG_ACCUM_D128.DELTA, cutlass.Float32, CFG_ACCUM_D128.N).subview(group * 64 + chunk * 4).load(vector_size=4, alignment=16)
        for pair in cutlass.range_constexpr(2):
            i = chunk * 4 + pair * 2
            d0, d1 = ffma2(scale, scale, dp[i], dp[i + 1], stat[pair * 2], stat[pair * 2 + 1])
            d0, d1 = fmul2(d0, d1, p[i], p[i + 1])
            packed.append(fp32_to_fp16(d0, d1, dtype=dtype))
    _tmem_store(address, cutlass.Vector.from_elements(tuple(packed), cutlass.Int32))


@cute.jit
def _softmax_loop_thd(
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
    nvvm.setmaxregister(CFG_ACCUM_D128.SOFTMAX_REGS, nvvm.SetMaxRegisterAction.INCREASE)
    tid = cute.arch.thread_idx()[0]
    row = tid % CFG_ACCUM_D128.M
    group = tid // CFG_ACCUM_D128.M
    state = PipelineState.start()
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        if cutlass.const_expr(not causal):
            _credit_thd(bars.sched_empty[sched_state.idx])
        lo, hi, seq_q, seq_k, q_start = _bounds_thd(block, meta, batch, n_batch, causal)
        if lo < hi:
            _stats_thd(slab, lse, delta, row, group, lo * CFG_ACCUM_D128.N, batch, head, scale, q_start, seq_q, origins, stats_origin)
        for qtile in cutlass.range(lo, hi, 1, unroll=1):
            _wait_tcgen_thd(bars.s_done, state.phase)
            p = _p_values_thd(slab, tmem_base, row, group, block, qtile, seq_q, seq_k, scale, causal, dtype)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
            bars.p_ready.arrive()
            _wait_tcgen_thd(bars.dp_done, state.phase)
            _ds_values_thd(slab, tmem_base, group, p, scale, dtype)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
            bars.ds_ready.arrive()
            if qtile + 1 < hi:
                _stats_thd(slab, lse, delta, row, group, (qtile + 1) * CFG_ACCUM_D128.N, batch, head, scale, q_start, seq_q, origins, stats_origin)
            state = advance(state, 1)
        if cutlass.const_expr(causal):
            _credit_thd(bars.sched_empty[sched_state.idx])
        block, head, batch, valid, sched_state = _next_tile_thd(sched, bars, sched_state)


@cute.jit
def _store_words_thd(slab, byte_offset, row, words):
    for vec in cutlass.range_constexpr(8):
        off = byte_offset + row * CFG_ACCUM_D128.SWIZZLE_BYTES + ((row % 8) ^ vec) * 16
        values = cutlass.Vector.from_elements(tuple(words[vec * 4 + k] for k in range(4)), cutlass.Int32)
        Pointer(slab.data_ptr() + off, dtype=cutlass.Int32).store(values, alignment=16)


@cute.jit
def _store_dq_slice_thd(slab, row, values, index: cutlass.Constexpr[int]):
    for vec in cutlass.range_constexpr(8):
        off = CFG_ACCUM_D128.ALIAS + index * CFG_ACCUM_D128.SLICE_BYTES + row * CFG_ACCUM_D128.SWIZZLE_BYTES + ((row % 8) ^ vec) * 16
        v = cutlass.Vector.from_elements(tuple(values[index * 32 + vec * 4 + k] for k in range(4)), cutlass.Float32)
        Pointer(slab.data_ptr() + off, dtype=cutlass.Float32).store(v, alignment=16)


@cute.jit
def _native_epilogue_thd(slab, tmem_base, bars, row, live, empty_phase, dtype: cutlass.Constexpr, dk: cutlass.Constexpr[bool]):
    plane = 384 if cutlass.const_expr(dk) else 256
    base = CFG_ACCUM_D128.ALIAS + (CFG_ACCUM_D128.OPERAND_BYTES if cutlass.const_expr(dk) else 0)
    last_slice = 3 if cutlass.const_expr(dk) else 1
    for half in cutlass.range_constexpr(2):
        vals = cutlass.Vector.from_elements(tuple(cutlass.Float32(0.0) for _ in range(64)), cutlass.Float32)
        if live:
            vals = _tmem_load(tmem_base + plane + half * 64, 64)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
        words = cutlass.Vector.from_elements(tuple(fp32_to_fp16(vals[2 * i], vals[2 * i + 1], dtype=dtype) for i in range(32)), cutlass.Int32)
        if cutlass.const_expr(half == 0):
            _wait_thd(bars.dq_empty[last_slice], empty_phase)
        _store_words_thd(slab, base + half * CFG_ACCUM_D128.SLICE_BYTES, row, words)
    nvvm.fence_proxy("async.shared", space="cta")
    nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
    if cutlass.const_expr(dk):
        bars.dk_full.arrive()
    else:
        bars.dv_full.arrive()


@cute.jit
def _native_epilogue_fp32(slab, tmem_base, bars, row, live, empty_phase, dk: cutlass.Constexpr[bool]):
    """dkv_accum = fp32_reduce (the grouped THD kernel only): the lane's kv row of the fp32 dV (TMEM plane CFG_ACCUM_D128.DV_COL) or dK (CFG_ACCUM_D128.DK_COL)
    accumulator as the FOUR [M x DQ_BOX] fp32 SW128 slices of the ALIAS slab (the dQ staging image, `_store_dq_slice`), published through
    dv_full / dk_full for the store warp's reduce-adds.  The dV round's WAR (the last dQ slice reads) is the caller's terminal dq_empty waits;
    the dK round waits dq_empty[i] (the store warp's credit after its dV slice i source read) before overwriting slice i -- the SAME ring
    object, one more completion per item on both sides.  A zero-trip owner stages zeros (its reduce-adds are skipped by the store warp)."""
    # ALIAS 64 KiB = one fp32 tensor (4 dQ-slice boxes): dV round, dK round; dq_empty x4
    plane = CFG_ACCUM_D128.DK_COL if cutlass.const_expr(dk) else CFG_ACCUM_D128.DV_COL
    lo_vals = cutlass.Vector.from_elements(tuple(cutlass.Float32(0.0) for _ in range(CFG_ACCUM_D128.D_HALF)), cutlass.Float32)
    hi_vals = cutlass.Vector.from_elements(tuple(cutlass.Float32(0.0) for _ in range(CFG_ACCUM_D128.D_HALF)), cutlass.Float32)
    if live:
        lo_vals = _tmem_load(tmem_base + plane, CFG_ACCUM_D128.D_HALF)
        hi_vals = _tmem_load(tmem_base + plane + CFG_ACCUM_D128.D_HALF, CFG_ACCUM_D128.D_HALF)
        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
    vals = cutlass.Vector.from_elements(
        tuple(lo_vals[i] for i in range(CFG_ACCUM_D128.D_HALF)) + tuple(hi_vals[i] for i in range(CFG_ACCUM_D128.D_HALF)), cutlass.Float32
    )
    for i in cutlass.range_constexpr(CFG_ACCUM_D128.DKV_ACC_SLICES):
        if cutlass.const_expr(dk):
            _wait_thd(bars.dq_empty[i], empty_phase)  # WAR: the store warp read the dV slice out of these bytes (its post-read credit)
        _store_dq_slice_thd(slab, row, vals, i)
    nvvm.fence_proxy("async.shared", space="cta")
    nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
    if cutlass.const_expr(dk):
        bars.dk_full.arrive()
    else:
        bars.dv_full.arrive()


@cute.jit
def _epilogue_loop_thd(
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
    dkv_fp32: cutlass.Constexpr[bool] = False,
):
    nvvm.setmaxregister(CFG_ACCUM_D128.EPILOGUE_REGS, nvvm.SetMaxRegisterAction.INCREASE)
    row = cute.arch.thread_idx()[0] % CFG_ACCUM_D128.M
    ds_state = PipelineState.start()
    dq_state = PipelineState.start()
    empty_state = PipelineState.start()
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit_thd(bars.sched_empty[sched_state.idx])
        lo, hi, seq_q, seq_k, q_start = _bounds_thd(block, meta, batch, n_batch, causal)
        for _ in cutlass.range(lo, hi, 1, unroll=1):
            _wait_tcgen_thd(bars.ds_ready, ds_state.phase)
            ds0 = _tmem_load(tmem_base + 128, 32, cutlass.Int32)
            ds1 = _tmem_load(tmem_base + 192, 32, cutlass.Int32)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
            _wait_thd(bars.dq_empty[0], empty_state.phase)
            _store_words_thd(slab, CFG_ACCUM_D128.ALIAS, row, ds0)
            _wait_thd(bars.dq_empty[1], empty_state.phase)
            _store_words_thd(slab, CFG_ACCUM_D128.ALIAS + CFG_ACCUM_D128.SLICE_BYTES, row, ds1)
            nvvm.fence_proxy("async.shared", space="cta")
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
            bars.ds_smem.arrive()
            _wait_tcgen_thd(bars.dq_done, dq_state.phase)
            dq0 = _tmem_load(tmem_base + 128, 64)
            dq1 = _tmem_load(tmem_base + 192, 64)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
            bars.dp_empty.arrive()
            vals = cutlass.Vector.from_elements(tuple(dq0[i] for i in range(64)) + tuple(dq1[i] for i in range(64)), cutlass.Float32)
            for i in cutlass.range_constexpr(4):
                _wait_thd(bars.dq_empty[i], empty_state.phase)
                _store_dq_slice_thd(slab, row, vals, i)
                nvvm.fence_proxy("async.shared", space="cta")
                bars.dq_full[i].arrive()
            ds_state = advance(ds_state, 1)
            dq_state = advance(dq_state, 1)
            empty_state = advance(empty_state, 1)
        if lo >= hi:
            _wait_tcgen_thd(bars.dq_done, dq_state.phase)
            dq_state = advance(dq_state, 1)
        # Observe every terminal source-credit phase before the next CLC
        # owner bootstraps these barriers again. Native dV/dK alone wait on
        # slices 1/3, leaving slices 0/2 unobserved after the final dQ or a
        # zero-trip owner. Each completed mbarrier phase needs a successful
        # wait before an arrival in its next phase.
        for i in cutlass.range_constexpr(4):
            _wait_thd(bars.dq_empty[i], empty_state.phase)
        if cutlass.const_expr(dkv_fp32):
            # dkv_accum = fp32_reduce: dV then dK as fp32 slices over the ALIAS slab -- two ring rounds per item (the store warp credits dq_empty
            # after the dV slices' source reads; the dK round waits them at the advanced phase)
            _native_epilogue_fp32(slab, tmem_base, bars, row, lo < hi, empty_state.phase, False)
            empty_state = advance(empty_state, 1)
            _native_epilogue_fp32(slab, tmem_base, bars, row, lo < hi, empty_state.phase, True)
        else:
            _native_epilogue_thd(slab, tmem_base, bars, row, lo < hi, empty_state.phase, dtype, False)
            _native_epilogue_thd(slab, tmem_base, bars, row, lo < hi, empty_state.phase, dtype, True)
        empty_state = advance(empty_state, 1)
        block, head, batch, valid, sched_state = _next_tile_thd(sched, bars, sched_state)


@cute.jit
def _ts_thd(desc, plane, bdesc, dst, accum):
    for k in cutlass.range_constexpr(CFG_ACCUM_D128.D // CFG_ACCUM_D128.K_HW):
        mma_ts_step(
            desc, nvvm.make_tmem_ptr(plane + (k // 4) * 32, cutlass.Int8), bdesc, dst, k_idx=k, accumulate=accum if cutlass.const_expr(k == 0) else True
        )


@cute.jit
def _mma_loop_thd(
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
    nvvm.setmaxregister(CFG_ACCUM_D128.SERVICE_REGS, nvvm.SetMaxRegisterAction.DECREASE)
    score = _mma_desc_thd(dtype)
    grad = _mma_desc_thd(dtype, btranspose=True)
    dqdesc = _mma_desc_thd(dtype, atranspose=True, btranspose=True)
    kd = _operand_thd(slab, CFG_ACCUM_D128.K, dtype).desc()
    vd = _operand_thd(slab, CFG_ACCUM_D128.V, dtype).desc()
    dod = _operand_thd(slab, CFG_ACCUM_D128.DO, dtype).desc()
    dot = _operand_thd(slab, CFG_ACCUM_D128.DO, dtype, transpose=True).desc()
    dst = _operand_thd(slab, CFG_ACCUM_D128.ALIAS, dtype, transpose=True).desc()
    kt = _operand_thd(slab, CFG_ACCUM_D128.K, dtype, transpose=True).desc()
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
        _credit_thd(bars.sched_empty[sched_state.idx])
        lo, hi, seq_q, seq_k, q_start = _bounds_thd(block, meta, batch, n_batch, causal)
        if lo < hi:
            _wait_tcgen_thd(bars.k_full, kv_state.phase)
            _wait_tcgen_thd(bars.q_full[q_state.idx], q_state.phase)
            qd = _operand_thd(slab, CFG_ACCUM_D128.Q + q_state.idx * CFG_ACCUM_D128.OPERAND_BYTES, dtype).desc()
            mma_ss(score, kd, qd, s)
            _commit(bars.s_done)
            _wait_tcgen_thd(bars.v_full, kv_state.phase)
        _wait_tcgen_thd(bars.dvk_empty, out_empty.phase)
        out_empty = advance(out_empty, 1)
        for qtile in cutlass.range(lo, hi, 1, unroll=1):
            _wait_tcgen_thd(bars.dp_empty, dp_empty.phase)
            _wait_tcgen_thd(bars.do_full[0], do_state.phase)
            mma_ss(score, vd, dod, dp, k_start=0, k_count=4)
            _wait_tcgen_thd(bars.do_full[1], do_state.phase)
            mma_ss(score, vd, dod, dp, k_start=4, k_count=4, accumulate=True)
            _commit(bars.dp_done)
            _wait_tcgen_thd(bars.p_ready, ready_state.phase)
            _ts_thd(grad, tmem_base, dot, dv, qtile > lo)
            _commit(bars.do_empty)
            q_next = advance(q_state, CFG_ACCUM_D128.Q_STAGES)
            if qtile + 1 < hi:
                _wait_tcgen_thd(bars.q_full[q_next.idx], q_next.phase)
                qd_next = _operand_thd(slab, CFG_ACCUM_D128.Q + q_next.idx * CFG_ACCUM_D128.OPERAND_BYTES, dtype).desc()
                mma_ss(score, kd, qd_next, s)
                _commit(bars.s_done)
            _wait_tcgen_thd(bars.ds_ready, ready_state.phase)
            qt = _operand_thd(slab, CFG_ACCUM_D128.Q + q_state.idx * CFG_ACCUM_D128.OPERAND_BYTES, dtype, transpose=True).desc()
            _ts_thd(grad, tmem_base + 128, qt, dk, qtile > lo)
            _commit(bars.q_empty[q_state.idx])
            _wait_tcgen_thd(bars.ds_smem, ready_state.phase)
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
        block, head, batch, valid, sched_state = _next_tile_thd(sched, bars, sched_state)


@cute.jit
def _load_loop_thd(
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
    nvvm.setmaxregister(CFG_ACCUM_D128.SERVICE_REGS, nvvm.SetMaxRegisterAction.DECREASE)
    q_state = PipelineState.start(phase=1)
    do_state = PipelineState.start(phase=1)
    kv_state = PipelineState.start(phase=1)
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit_thd(bars.sched_empty[sched_state.idx])
        lo, hi, seq_q, seq_k, q_start = _bounds_thd(block, meta, batch, n_batch, causal)
        q_desc = _map_ptr(maps, batch, n_batch, MAP_Q_THD)
        k_desc = _map_ptr(maps, batch, n_batch, MAP_K_THD)
        v_desc = _map_ptr(maps, batch, n_batch, MAP_V_THD)
        do_desc = _map_ptr(maps, batch, n_batch, MAP_DO_THD)
        if lo < hi:
            tma_tensormap_acquire(q_desc)
            tma_tensormap_acquire(k_desc)
            tma_tensormap_acquire(v_desc)
            tma_tensormap_acquire(do_desc)
            _wait_thd(bars.k_empty, kv_state.phase)
            bars.k_full.arrive(n_bytes=CFG_ACCUM_D128.OPERAND_BYTES, pred=nvvm.elect_sync())
            tma_load_tile(
                _operand_thd(slab, CFG_ACCUM_D128.K, dtype),
                tma_slice_runtime_desc(k_desc, 0, block * CFG_ACCUM_D128.M, head, 0),
                bars.k_full.smem_ptr,
                acquire=False,
            )
            _wait_thd(bars.v_empty, kv_state.phase)
            bars.v_full.arrive(n_bytes=CFG_ACCUM_D128.OPERAND_BYTES, pred=nvvm.elect_sync())
            tma_load_tile(
                _operand_thd(slab, CFG_ACCUM_D128.V, dtype),
                tma_slice_runtime_desc(v_desc, 0, block * CFG_ACCUM_D128.M, head, 0),
                bars.v_full.smem_ptr,
                acquire=False,
            )
            kv_state = advance(kv_state, 1)
        for qtile in cutlass.range(lo, hi, 1, unroll=1):
            _wait_thd(bars.q_empty[q_state.idx], q_state.phase)
            bars.q_full[q_state.idx].arrive(n_bytes=CFG_ACCUM_D128.OPERAND_BYTES, pred=nvvm.elect_sync())
            tma_load_tile(
                _operand_thd(slab, CFG_ACCUM_D128.Q + q_state.idx * CFG_ACCUM_D128.OPERAND_BYTES, dtype),
                tma_slice_runtime_desc(q_desc, 0, qtile * CFG_ACCUM_D128.N, head, 0),
                bars.q_full[q_state.idx].smem_ptr,
                acquire=False,
            )
            _wait_thd(bars.do_empty, do_state.phase)
            for half in cutlass.range_constexpr(2):
                bars.do_full[half].arrive(n_bytes=CFG_ACCUM_D128.SLICE_BYTES, pred=nvvm.elect_sync())
            for half in cutlass.range_constexpr(2):
                tma_load_subtiles(
                    _operand_thd(slab, CFG_ACCUM_D128.DO, dtype),
                    tma_slice_runtime_desc(do_desc, 0, qtile * CFG_ACCUM_D128.N, head, 0),
                    bars.do_full[half].smem_ptr,
                    half,
                    1,
                    acquire=False,
                )
            q_state = advance(q_state, CFG_ACCUM_D128.Q_STAGES)
            do_state = advance(do_state, 1)
        block, head, batch, valid, sched_state = _next_tile_thd(sched, bars, sched_state)


@cute.jit
def _store_loop_thd(
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
    diag_delete_dq_reduce_add: cutlass.Constexpr[bool] = False,
    dkv_fp32: cutlass.Constexpr[bool] = False,
    ratio: cutlass.Constexpr[int] = 1,
):
    nvvm.setmaxregister(CFG_ACCUM_D128.SERVICE_REGS, nvvm.SetMaxRegisterAction.DECREASE)
    dq_state = PipelineState.start()
    out_state = PipelineState.start()
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit_thd(bars.sched_empty[sched_state.idx])
        for i in cutlass.range_constexpr(4):
            bars.dq_empty[i].arrive()  # all32 lanes, one bootstrap per KV tile
        lo, hi, seq_q, seq_k, q_start = _bounds_thd(block, meta, batch, n_batch, causal)
        dq_desc = _map_ptr(maps, batch, n_batch, MAP_DQ_THD)
        if lo < hi:
            tma_tensormap_acquire(dq_desc)
        for qtile in cutlass.range(lo, hi, 1, unroll=1):
            for i in cutlass.range_constexpr(4):
                _wait_thd(bars.dq_full[i], dq_state.phase)
                # DIAGNOSTIC D2 (default off = the shipped code): the armed rendering deletes ONLY the reduce-add issue;
                # the commit, the wait_group and both dq_* arrives stay, so the barrier table is unchanged.
                if cutlass.const_expr(not diag_delete_dq_reduce_add):
                    tma_reduce_add_subtile(
                        _operand_thd(slab, CFG_ACCUM_D128.ALIAS, cutlass.Float32, fp32=True),
                        tma_slice_runtime_desc(dq_desc, 0, qtile * CFG_ACCUM_D128.N, head, 0),
                        i,
                        acquire=False,
                    )
                tma_store_commit()
            for i in cutlass.range_constexpr(4):
                tma_store_wait(3 - i)
                bars.dq_empty[i].arrive()  # all32 after corresponding source read
            dq_state = advance(dq_state, 1)
        _wait_thd(bars.dv_full, out_state.phase)
        if cutlass.const_expr(dkv_fp32):
            # dkv_accum = fp32_reduce: four fp32 reduce-adds per tensor from the ALIAS slices into the per-KV-head workspace maps (the MAP_DV / MAP_DK
            # slots carry fp32 [M x DQ_BOX]-box maps; kv head = head // ratio); the per-slice bulk groups credit dq_empty[i] after the dV slice's
            # source read (the dK staging's WAR edge, P7), then dK.  A zero-trip owner issues nothing (the workspace is zero-filled) and never
            # acquires a map, but keeps every commit / wait / arrive.
            live = lo < hi
            dv_desc = _map_ptr(maps, batch, n_batch, MAP_DV_THD)
            if live:
                tma_tensormap_acquire(dv_desc)
            for i in cutlass.range_constexpr(CFG_ACCUM_D128.DKV_ACC_SLICES):
                if live:
                    tma_reduce_add_subtile(
                        _operand_thd(slab, CFG_ACCUM_D128.ALIAS, cutlass.Float32, fp32=True),
                        tma_slice_runtime_desc(dv_desc, 0, block * CFG_ACCUM_D128.M, head // ratio, 0),
                        i,
                        acquire=False,
                    )
                tma_store_commit()
            for i in cutlass.range_constexpr(CFG_ACCUM_D128.DKV_ACC_SLICES):
                tma_store_wait(CFG_ACCUM_D128.DKV_ACC_SLICES - 1 - i)
                bars.dq_empty[i].arrive()  # all32 after the dV slice's source read: the epilogue's dK round may overwrite slice i
            _wait_thd(bars.dk_full, out_state.phase)
            dk_desc = _map_ptr(maps, batch, n_batch, MAP_DK_THD)
            if live:
                tma_tensormap_acquire(dk_desc)
            for i in cutlass.range_constexpr(CFG_ACCUM_D128.DKV_ACC_SLICES):
                if live:
                    tma_reduce_add_subtile(
                        _operand_thd(slab, CFG_ACCUM_D128.ALIAS, cutlass.Float32, fp32=True),
                        tma_slice_runtime_desc(dk_desc, 0, block * CFG_ACCUM_D128.M, head // ratio, 0),
                        i,
                        acquire=False,
                    )
                tma_store_commit()
        else:
            # Invalid envelope owners still consume both native publications and
            # return every source credit, but never acquire/use an empty map.
            if block * CFG_ACCUM_D128.M < seq_k:
                dv_desc = _map_ptr(maps, batch, n_batch, MAP_DV_THD)
                tma_store_tile(_operand_thd(slab, CFG_ACCUM_D128.ALIAS, dtype), tma_slice_runtime_desc(dv_desc, 0, block * CFG_ACCUM_D128.M, head, 0))
            _wait_thd(bars.dk_full, out_state.phase)
            if block * CFG_ACCUM_D128.M < seq_k:
                dk_desc = _map_ptr(maps, batch, n_batch, MAP_DK_THD)
                tma_store_tile(
                    _operand_thd(slab, CFG_ACCUM_D128.ALIAS + CFG_ACCUM_D128.OPERAND_BYTES, dtype),
                    tma_slice_runtime_desc(dk_desc, 0, block * CFG_ACCUM_D128.M, head, 0),
                )
        out_state = advance(out_state, 1)
        tma_store_commit()
        tma_store_wait(0)
        bars.dvk_empty.arrive()
        block, head, batch, valid, sched_state = _next_tile_thd(sched, bars, sched_state)


@cute.jit
def _scheduler_loop_thd(bars, sched):
    nvvm.setmaxregister(CFG_ACCUM_D128.SERVICE_REGS, nvvm.SetMaxRegisterAction.DECREASE)
    state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _wait_thd(bars.sched_empty[state.idx], state.phase)
        if nvvm.elect_sync():
            bars.sched_full[state.idx].arrive(n_bytes=16)
            nvvm.clusterlaunchcontrol_try_cancel(sched.tile_id_smem.subview(state.idx * 8), bars.sched_full[state.idx].smem_ptr, multicast=1)
        nvvm.fence_proxy("async.shared", space="cta")
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
        _wait_thd(bars.sched_full[state.idx], state.phase)
        _, _, valid = read_clc_payload(sched, state.idx * 8, warp_broadcast=True)
        state = advance(state, CFG_ACCUM_D128.SCHED_STAGES)


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
    diag_delete_dq_reduce_add: cutlass.Constexpr[bool] = False,
):
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    x, y, z = cute.arch.block_idx()
    block, head, batch = x, y, z
    slab = cutlass.Array(cutlass.Uint8, CFG_ACCUM_D128.SMEM_BYTES, alignment=1024, space=cutlass.AddressSpace.smem)
    bars = _bars_thd(slab)
    tmem_slot = _view(slab, CFG_ACCUM_D128.TMEM_HANDLE, cutlass.Int32, 1, alignment=4)
    sched_words = _view(slab, CFG_ACCUM_D128.SCHED_PAYLOAD, cutlass.Int32, 16, alignment=16)
    sched = Sched(bars.sched_full.base_ptr, bars.sched_empty.base_ptr, sched_words, x, y, z)
    if warp == 0:
        # All 512 TMEM columns belong to this CTA; use the portable allocation form.
        nvvm.tcgen05_alloc(tmem_slot, cutlass.Int32(CFG_ACCUM_D128.TMEM_COLS), group=nvvm.CTAGroup.CTA_1)
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
        _softmax_loop_thd(slab, bars, sched, tmem_base, lse, delta, origins, stats_origin, block, head, batch, meta, n_batch, scale, causal, dtype)
    elif warp < 12:
        _epilogue_loop_thd(slab, bars, sched, tmem_base, block, head, batch, meta, n_batch, causal, dtype)
    elif warp == 12:
        _mma_loop_thd(slab, bars, sched, tmem_base, block, head, batch, meta, n_batch, causal, dtype)
    elif warp == 13:
        _load_loop_thd(slab, bars, sched, maps, block, head, batch, meta, n_batch, causal, dtype)
    elif warp == 14:
        _store_loop_thd(slab, bars, sched, maps, block, head, batch, meta, n_batch, causal, dtype, diag_delete_dq_reduce_add)
    else:
        _scheduler_loop_thd(bars, sched)
    nvvm.barrier_cta_sync()
    if warp == 0:
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
        nvvm.tcgen05_dealloc(nvvm.make_tmem_ptr(tmem_slot.load(), cutlass.Int8), cutlass.Int32(CFG_ACCUM_D128.TMEM_COLS), group=nvvm.CTAGroup.CTA_1)


bprop_d128_accum_thd_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def host_accum_thd(
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
    diag_delete_dq_reduce_add: cutlass.Constexpr[bool] = False,
):
    """Launch the native THD body on a host-declared KV envelope, never live totals."""
    nk = cute.math.max((max_skv + CFG_ACCUM_D128.M - 1) // CFG_ACCUM_D128.M, cutlass.Int32(1))
    bprop_d128_accum_thd_kernel(maps, lse, delta, meta, origins, n_batch, stats_origin, scale, causal, dtype, diag_delete_dq_reduce_add).launch(
        grid=(nk, nheads, n_batch), block=(CFG_ACCUM_D128.THREADS, 1, 1), stream=stream, min_blocks_per_mp=1
    )


# ======================================================================================================================
# section 6: the grouped chain hosts (dense, THD)
# (was kernels/sm100/bprop_d128_accum_grouped_helpers.py)
# ======================================================================================================================
"""Dense/THD grouped d128 chains with native partials and a final FP32 fold."""


@cute.jit
def host_grouped_dense(
    q,
    k,
    v,
    o,
    do,
    lse,
    dq,
    dk,
    dv,
    dq_accum,
    delta,
    dk_part,
    dv_part,
    scale,
    causal: cutlass.Constexpr[bool],
    stream,
    diag_delete_dq_reduce_add: cutlass.Constexpr[bool] = False,
):
    """Four native launches; uniquely owned partials are folded in FP32."""
    delta_zero_host(o, do, delta, dq_accum, stream)
    main_dense_host(q, k, v, do, lse, delta, dq_accum, dk_part, dv_part, scale, causal, stream, diag_delete_dq_reduce_add)
    cast_dq_host(dq_accum, dq, q.element_type, stream)
    dkv_reduce_host(dk_part, dv_part, dk, dv, CFG_ACCUM_D128.D, CFG_ACCUM_D128.D, q.shape[2] // k.shape[2], q.element_type, False, stream)


@cute.jit
def host_grouped_thd(
    q,
    k,
    v,
    o,
    do,
    lse,
    dq,
    dk,
    dv,
    dq_accum,
    delta,
    dk_part,
    dv_part,
    meta,
    origins,
    maps,
    q_lens,
    kv_lens,
    ro,
    origin_specs: cutlass.Constexpr,
    lens_form,
    batch: cutlass.Constexpr[int],
    max_sq,
    max_skv,
    scale,
    causal: cutlass.Constexpr[bool],
    stream,
    diag_delete_dq_reduce_add: cutlass.Constexpr[bool] = False,
    dkv_fp32: cutlass.Constexpr[bool] = False,
    dk_acc=None,
    dv_acc=None,
):
    """Five native launches, with compact partials and independent final origins; FOUR under ``dkv_fp32`` (dkv_accum = fp32_reduce): the
    per-KV-head fp32 dK / dV accumulators ``dk_acc`` / ``dv_acc`` take the MAP_DK / MAP_DV slots as fp32 [M x DQ_BOX]-box maps (compact KV
    origins, the sequence-local clip), delta_zero zero-fills them over the kv rows, the main kernel reduce-adds every owner's fp32 tile,
    cast_dq converts them once to the IO dtype at each output's own origin, and the fold is not launched."""
    if cutlass.const_expr(dkv_fp32):
        dk_map, dv_map = _tensor_map_thd(dk_acc, CFG_ACCUM_D128.DQ_BOX), _tensor_map_thd(dv_acc, CFG_ACCUM_D128.DQ_BOX)
        dk_ws, dv_ws = dk_acc, dv_acc
    else:
        dk_map, dv_map = _tensor_map_thd(dk_part, CFG_ACCUM_D128.HEAD_BOX), _tensor_map_thd(dv_part, CFG_ACCUM_D128.HEAD_BOX)
        dk_ws, dv_ws = dk_part, dv_part
    setup_thd_d128_kernel(
        _tensor_map_thd(q, CFG_ACCUM_D128.HEAD_BOX),
        _tensor_map_thd(k, CFG_ACCUM_D128.HEAD_BOX),
        _tensor_map_thd(v, CFG_ACCUM_D128.HEAD_BOX),
        _tensor_map_thd(do, CFG_ACCUM_D128.HEAD_BOX),
        _tensor_map_thd(dq_accum, CFG_ACCUM_D128.DQ_BOX),
        dk_map,
        dv_map,
        q,
        k,
        v,
        do,
        dq_accum,
        dk_ws,
        dv_ws,
        meta,
        origins,
        maps,
        q_lens,
        kv_lens,
        *ro,
        origin_specs,
        lens_form,
        cutlass.Int32(batch),
        partial_dkv=True,
    ).launch(grid=(1, 1, 1), block=(THD_SETUP_THREADS, 1, 1), stream=stream)
    q_grid = cute.math.max((max_sq + ROWS_PER_CTA - 1) // ROWS_PER_CTA, cutlass.Int32(1))
    if cutlass.const_expr(dkv_fp32):
        # the row extent covers the q rows (Delta, dQ zero) AND the kv rows (the fp32 dK / dV zero fill / convert); z = Hq, kv heads = the first Hkv
        kv_grid = cute.math.max((max_skv + ROWS_PER_CTA - 1) // ROWS_PER_CTA, cutlass.Int32(1))
        rows_grid = cute.math.max(q_grid, kv_grid)
        delta_zero_thd_d128_kernel(o, do, delta, dq_accum, meta, origins, cutlass.Int32(batch), origin_specs, dk_acc, dv_acc).launch(
            grid=(rows_grid, batch, q.shape[2]), block=(THREADS, 1, 1), stream=stream
        )
    else:
        delta_zero_thd_d128_kernel(o, do, delta, dq_accum, meta, origins, cutlass.Int32(batch), origin_specs).launch(
            grid=(q_grid, batch, q.shape[2]), block=(THREADS, 1, 1), stream=stream
        )
    main_thd_host(
        maps,
        lse,
        delta,
        meta,
        origins,
        cutlass.Int32(batch),
        q.shape[2],
        max_skv,
        _origin_row(origin_specs, 8),
        scale,
        causal,
        q.element_type,
        q.shape[2] // k.shape[2],
        stream,
        diag_delete_dq_reduce_add,
        dkv_fp32,
    )
    if cutlass.const_expr(dkv_fp32):
        cast_dq_thd_d128_kernel(
            dq_accum,
            dq,
            meta,
            origins,
            cutlass.Int32(batch),
            _origin_row(origin_specs, 5),
            q.element_type,
            dk_acc,
            dv_acc,
            dk,
            dv,
            _origin_row(origin_specs, 6),
            _origin_row(origin_specs, 7),
        ).launch(grid=(rows_grid, batch, q.shape[2]), block=(THREADS, 1, 1), stream=stream)
    else:
        cast_dq_thd_d128_kernel(dq_accum, dq, meta, origins, cutlass.Int32(batch), _origin_row(origin_specs, 5), q.element_type).launch(
            grid=(q_grid, batch, q.shape[2]), block=(THREADS, 1, 1), stream=stream
        )
        dkv_reduce_thd_origins_host(
            dk_part,
            dv_part,
            dk,
            dv,
            meta,
            origins,
            batch,
            max_skv,
            _origin_row(origin_specs, 6),
            _origin_row(origin_specs, 7),
            CFG_ACCUM_D128.D,
            q.shape[2] // k.shape[2],
            q.element_type,
            stream,
        )


# ======================================================================================================================
# section 7: the grouped bodies (dense, THD): the MHA roles with the grouped loaders
# (was kernels/sm100/bprop_d128_accum_grouped_f16.py)
# ======================================================================================================================
"""Grouped d128 backward: one Q-head owner, native dK/dV partials.

The qualified MHA math, epilogue, store and scheduler roles are imported
unchanged. Only the loaders map Q heads to shared K/V heads. Separate fold
kernels consume the uniquely owned native partials after this main pass.

Op-level field ``dkv_accum = "fp32_reduce"`` (``Constexpr[bool]`` ``dkv_fp32`` on the THD kernel and ``main_thd_host``, default False = the
shipped chain, rendered PTX-identical): the epilogue / store roles of the packed body take their fp32 store path (the section 5
header), reduce-adding each owner's fp32 dK / dV tile into the per-KV-head workspace maps (kv head = head // ratio) instead of storing a
bf16 partial; the fold launch is then not needed.
"""


@cute.jit
def _load_dense(
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
    ratio: cutlass.Constexpr[int],
):
    nvvm.setmaxregister(CFG_ACCUM_D128.SERVICE_REGS, nvvm.SetMaxRegisterAction.DECREASE)
    q_state = PipelineState.start(phase=1)
    do_state = PipelineState.start(phase=1)
    kv_state = PipelineState.start(phase=1)
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit_accum(bars.sched_empty[sched_state.idx])
        lo, hi = _bounds_accum(block, sq, causal)
        if lo < hi:
            _wait_accum(bars.k_empty, kv_state.phase)
            bars.k_full.arrive(n_bytes=CFG_ACCUM_D128.OPERAND_BYTES, pred=nvvm.elect_sync())
            tma_load_tile(
                _operand_accum(slab, CFG_ACCUM_D128.K, dtype), GmemTileTma(k_desc)(0, block * CFG_ACCUM_D128.M, head // ratio, batch), bars.k_full.smem_ptr
            )
            _wait_accum(bars.v_empty, kv_state.phase)
            bars.v_full.arrive(n_bytes=CFG_ACCUM_D128.OPERAND_BYTES, pred=nvvm.elect_sync())
            tma_load_tile(
                _operand_accum(slab, CFG_ACCUM_D128.V, dtype), GmemTileTma(v_desc)(0, block * CFG_ACCUM_D128.M, head // ratio, batch), bars.v_full.smem_ptr
            )
            kv_state = advance(kv_state, 1)
        for qtile in cutlass.range(lo, hi, 1, unroll=1):
            _wait_accum(bars.q_empty[q_state.idx], q_state.phase)
            bars.q_full[q_state.idx].arrive(n_bytes=CFG_ACCUM_D128.OPERAND_BYTES, pred=nvvm.elect_sync())
            tma_load_tile(
                _operand_accum(slab, CFG_ACCUM_D128.Q + q_state.idx * CFG_ACCUM_D128.OPERAND_BYTES, dtype),
                GmemTileTma(q_desc)(0, qtile * CFG_ACCUM_D128.N, head, batch),
                bars.q_full[q_state.idx].smem_ptr,
            )
            _wait_accum(bars.do_empty, do_state.phase)
            for half in cutlass.range_constexpr(2):
                bars.do_full[half].arrive(n_bytes=CFG_ACCUM_D128.SLICE_BYTES, pred=nvvm.elect_sync())
            for half in cutlass.range_constexpr(2):
                tma_load_subtiles(
                    _operand_accum(slab, CFG_ACCUM_D128.DO, dtype),
                    GmemTileTma(do_desc)(0, qtile * CFG_ACCUM_D128.N, head, batch),
                    bars.do_full[half].smem_ptr,
                    half,
                    1,
                )
            q_state = advance(q_state, CFG_ACCUM_D128.Q_STAGES)
            do_state = advance(do_state, 1)
        block, head, batch, valid, sched_state = _next_tile_accum(sched, bars, sched_state, nheads, nk, causal, swizzle, batch_heads)


@cute.jit
def _load_thd(
    slab, bars, sched, maps, block, head, batch, meta, n_batch, causal: cutlass.Constexpr[bool], dtype: cutlass.Constexpr, ratio: cutlass.Constexpr[int]
):
    nvvm.setmaxregister(CFG_ACCUM_D128.SERVICE_REGS, nvvm.SetMaxRegisterAction.DECREASE)
    q_state = PipelineState.start(phase=1)
    do_state = PipelineState.start(phase=1)
    kv_state = PipelineState.start(phase=1)
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit_thd(bars.sched_empty[sched_state.idx])
        lo, hi, seq_q, seq_k, q_start = _bounds_thd(block, meta, batch, n_batch, causal)
        q_desc = _map_ptr(maps, batch, n_batch, MAP_Q_THD)
        k_desc = _map_ptr(maps, batch, n_batch, MAP_K_THD)
        v_desc = _map_ptr(maps, batch, n_batch, MAP_V_THD)
        do_desc = _map_ptr(maps, batch, n_batch, MAP_DO_THD)
        if lo < hi:
            tma_tensormap_acquire(q_desc)
            tma_tensormap_acquire(k_desc)
            tma_tensormap_acquire(v_desc)
            tma_tensormap_acquire(do_desc)
            _wait_thd(bars.k_empty, kv_state.phase)
            bars.k_full.arrive(n_bytes=CFG_ACCUM_D128.OPERAND_BYTES, pred=nvvm.elect_sync())
            tma_load_tile(
                _operand_thd(slab, CFG_ACCUM_D128.K, dtype),
                tma_slice_runtime_desc(k_desc, 0, block * CFG_ACCUM_D128.M, head // ratio, 0),
                bars.k_full.smem_ptr,
                acquire=False,
            )
            _wait_thd(bars.v_empty, kv_state.phase)
            bars.v_full.arrive(n_bytes=CFG_ACCUM_D128.OPERAND_BYTES, pred=nvvm.elect_sync())
            tma_load_tile(
                _operand_thd(slab, CFG_ACCUM_D128.V, dtype),
                tma_slice_runtime_desc(v_desc, 0, block * CFG_ACCUM_D128.M, head // ratio, 0),
                bars.v_full.smem_ptr,
                acquire=False,
            )
            kv_state = advance(kv_state, 1)
        for qtile in cutlass.range(lo, hi, 1, unroll=1):
            _wait_thd(bars.q_empty[q_state.idx], q_state.phase)
            bars.q_full[q_state.idx].arrive(n_bytes=CFG_ACCUM_D128.OPERAND_BYTES, pred=nvvm.elect_sync())
            tma_load_tile(
                _operand_thd(slab, CFG_ACCUM_D128.Q + q_state.idx * CFG_ACCUM_D128.OPERAND_BYTES, dtype),
                tma_slice_runtime_desc(q_desc, 0, qtile * CFG_ACCUM_D128.N, head, 0),
                bars.q_full[q_state.idx].smem_ptr,
                acquire=False,
            )
            _wait_thd(bars.do_empty, do_state.phase)
            for half in cutlass.range_constexpr(2):
                bars.do_full[half].arrive(n_bytes=CFG_ACCUM_D128.SLICE_BYTES, pred=nvvm.elect_sync())
            for half in cutlass.range_constexpr(2):
                tma_load_subtiles(
                    _operand_thd(slab, CFG_ACCUM_D128.DO, dtype),
                    tma_slice_runtime_desc(do_desc, 0, qtile * CFG_ACCUM_D128.N, head, 0),
                    bars.do_full[half].smem_ptr,
                    half,
                    1,
                    acquire=False,
                )
            q_state = advance(q_state, CFG_ACCUM_D128.Q_STAGES)
            do_state = advance(do_state, 1)
        block, head, batch, valid, sched_state = _next_tile_thd(sched, bars, sched_state)


@cute.kernel
def bprop_d128_accum_grouped_dense_kernel(
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
    ratio: cutlass.Constexpr[int],
    diag_delete_dq_reduce_add: cutlass.Constexpr[bool] = False,
):
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    x, y, z = cute.arch.block_idx()
    nk = skv // CFG_ACCUM_D128.M
    if cutlass.const_expr(causal):
        block, head, batch = _decode_flat(x, nheads, nk, swizzle, batch_heads)
    else:
        block, head, batch = (x, y, z)
    slab = cutlass.Array(cutlass.Uint8, CFG_ACCUM_D128.SMEM_BYTES, alignment=1024, space=cutlass.AddressSpace.smem)
    bars = _bars_accum(slab)
    tmem_slot = _view(slab, CFG_ACCUM_D128.TMEM_HANDLE, cutlass.Int32, 1, alignment=4)
    sched_words = _view(slab, CFG_ACCUM_D128.SCHED_PAYLOAD, cutlass.Int32, 16, alignment=16)
    sched = Sched(bars.sched_full.base_ptr, bars.sched_empty.base_ptr, sched_words, x, y, z)
    if warp == 0:
        nvvm.tcgen05_alloc(tmem_slot, cutlass.Int32(CFG_ACCUM_D128.TMEM_COLS), group=nvvm.CTAGroup.CTA_1)
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
        _softmax_loop_accum(slab, bars, sched, tmem_base, lse, delta, block, head, batch, sq, skv, nk, nheads, batch_heads, swizzle, scale, causal, dtype)
    elif warp < 12:
        _epilogue_loop_accum(slab, bars, sched, tmem_base, block, head, batch, sq, nk, nheads, batch_heads, swizzle, causal, dtype)
    elif warp == 12:
        _mma_loop_accum(slab, bars, sched, tmem_base, block, head, batch, sq, nk, nheads, batch_heads, swizzle, causal, dtype)
    elif warp == 13:
        _load_dense(slab, bars, sched, q_desc, k_desc, v_desc, do_desc, block, head, batch, sq, nk, nheads, batch_heads, swizzle, causal, dtype, ratio)
    elif warp == 14:
        _store_loop_accum(
            slab, bars, sched, dq_desc, dk_desc, dv_desc, block, head, batch, sq, nk, nheads, batch_heads, swizzle, causal, dtype, diag_delete_dq_reduce_add
        )
    else:
        _scheduler_loop_accum(bars, sched)
    nvvm.barrier_cta_sync()
    if warp == 0:
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
        nvvm.tcgen05_dealloc(nvvm.make_tmem_ptr(tmem_slot.load(), cutlass.Int8), cutlass.Int32(CFG_ACCUM_D128.TMEM_COLS), group=nvvm.CTAGroup.CTA_1)


bprop_d128_accum_grouped_dense_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.kernel
def bprop_d128_accum_grouped_thd_kernel(
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
    ratio: cutlass.Constexpr[int],
    diag_delete_dq_reduce_add: cutlass.Constexpr[bool] = False,
    dkv_fp32: cutlass.Constexpr[bool] = False,
):
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    x, y, z = cute.arch.block_idx()
    block, head, batch = (x, y, z)
    slab = cutlass.Array(cutlass.Uint8, CFG_ACCUM_D128.SMEM_BYTES, alignment=1024, space=cutlass.AddressSpace.smem)
    bars = _bars_thd(slab)
    tmem_slot = _view(slab, CFG_ACCUM_D128.TMEM_HANDLE, cutlass.Int32, 1, alignment=4)
    sched_words = _view(slab, CFG_ACCUM_D128.SCHED_PAYLOAD, cutlass.Int32, 16, alignment=16)
    sched = Sched(bars.sched_full.base_ptr, bars.sched_empty.base_ptr, sched_words, x, y, z)
    if warp == 0:
        nvvm.tcgen05_alloc(tmem_slot, cutlass.Int32(CFG_ACCUM_D128.TMEM_COLS), group=nvvm.CTAGroup.CTA_1)
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
        _softmax_loop_thd(slab, bars, sched, tmem_base, lse, delta, origins, stats_origin, block, head, batch, meta, n_batch, scale, causal, dtype)
    elif warp < 12:
        _epilogue_loop_thd(slab, bars, sched, tmem_base, block, head, batch, meta, n_batch, causal, dtype, dkv_fp32)
    elif warp == 12:
        _mma_loop_thd(slab, bars, sched, tmem_base, block, head, batch, meta, n_batch, causal, dtype)
    elif warp == 13:
        _load_thd(slab, bars, sched, maps, block, head, batch, meta, n_batch, causal, dtype, ratio)
    elif warp == 14:
        _store_loop_thd(slab, bars, sched, maps, block, head, batch, meta, n_batch, causal, dtype, diag_delete_dq_reduce_add, dkv_fp32, ratio)
    else:
        _scheduler_loop_thd(bars, sched)
    nvvm.barrier_cta_sync()
    if warp == 0:
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
        nvvm.tcgen05_dealloc(nvvm.make_tmem_ptr(tmem_slot.load(), cutlass.Int8), cutlass.Int32(CFG_ACCUM_D128.TMEM_COLS), group=nvvm.CTAGroup.CTA_1)


bprop_d128_accum_grouped_thd_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def main_dense_host(
    q,
    k,
    v,
    do,
    lse,
    delta,
    dq_accum,
    dk_part,
    dv_part,
    scale,
    causal: cutlass.Constexpr[bool],
    stream,
    diag_delete_dq_reduce_add: cutlass.Constexpr[bool] = False,
):
    """Qualified dense geometry, with the partial output head axis equal to Hq."""
    if cutlass.const_expr(q.element_type not in (cutlass.BFloat16, cutlass.Float16)):
        raise ValueError("grouped d128 accumulation requires BF16/FP16")
    if cutlass.const_expr(q.shape[3] != CFG_ACCUM_D128.D or k.shape[3] != CFG_ACCUM_D128.D or q.shape[2] <= k.shape[2] or q.shape[2] % k.shape[2]):
        raise ValueError("grouped d128 accumulation requires integral Hq/Hkv > 1 and d128")
    if cutlass.const_expr(q.shape[1] <= 0 or k.shape[1] <= 0 or q.shape[1] % CFG_ACCUM_D128.N or k.shape[1] % CFG_ACCUM_D128.M):
        raise ValueError("dense grouped d128 accumulation requires positive Sq/Skv multiples of 128")
    nk = k.shape[1] // CFG_ACCUM_D128.M
    heads_per_l2 = (50 * 1024 * 1024) // (q.shape[1] * (CFG_ACCUM_D128.D + CFG_ACCUM_D128.D) * CFG_ACCUM_D128.BPE)
    swizzle = 1 << (heads_per_l2.bit_length() - 1) if heads_per_l2 > 0 else 1
    grid = (nk * q.shape[2] * q.shape[0], 1, 1) if causal else (nk, q.shape[2], q.shape[0])
    bprop_d128_accum_grouped_dense_kernel(
        _tensor_map_accum(q, CFG_ACCUM_D128.HEAD_BOX),
        _tensor_map_accum(k, CFG_ACCUM_D128.HEAD_BOX),
        _tensor_map_accum(v, CFG_ACCUM_D128.HEAD_BOX),
        _tensor_map_accum(do, CFG_ACCUM_D128.HEAD_BOX),
        _tensor_map_accum(dq_accum, CFG_ACCUM_D128.DQ_BOX),
        _tensor_map_accum(dk_part, CFG_ACCUM_D128.HEAD_BOX),
        _tensor_map_accum(dv_part, CFG_ACCUM_D128.HEAD_BOX),
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
        q.shape[2] // k.shape[2],
        diag_delete_dq_reduce_add,
    ).launch(grid=grid, block=(CFG_ACCUM_D128.THREADS, 1, 1), stream=stream, min_blocks_per_mp=1)


@cute.jit
def main_thd_host(
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
    ratio: cutlass.Constexpr[int],
    stream,
    diag_delete_dq_reduce_add: cutlass.Constexpr[bool] = False,
    dkv_fp32: cutlass.Constexpr[bool] = False,
):
    """One owner per Q head and declared KV-envelope tile; no host length read."""
    nk = cute.math.max((max_skv + CFG_ACCUM_D128.M - 1) // CFG_ACCUM_D128.M, cutlass.Int32(1))
    bprop_d128_accum_grouped_thd_kernel(
        maps,
        lse,
        delta,
        meta,
        origins,
        n_batch,
        stats_origin,
        scale,
        causal,
        dtype,
        ratio,
        diag_delete_dq_reduce_add,
        dkv_fp32,
    ).launch(grid=(nk, nheads, n_batch), block=(CFG_ACCUM_D128.THREADS, 1, 1), stream=stream, min_blocks_per_mp=1)
