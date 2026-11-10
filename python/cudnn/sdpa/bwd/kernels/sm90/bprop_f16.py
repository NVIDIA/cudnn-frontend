# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Hopper dense d64 / d128 backward: the key-owner dK / dV kernel and the query-owner dQ kernel.

Each CTA runs one TMA producer warpgroup and two WGMMA consumer warpgroups that own 64 rows each of
the CTA's tile. The dK / dV kernel keeps one 128-row key tile resident, streams 64-row Q / dO tiles
with their Stats and delta, and accumulates dK and dV in registers; P and dS stay in registers as the A
operands of the dV and dK WGMMAs. The dQ kernel keeps one 128-row query tile resident, streams 64-row
K / V tiles and accumulates dQ in registers. Every gradient tile has one owner and a fixed summation
order, so the pair is deterministic without a dQ workspace.

Sequence lengths need not be tile multiples: tail loads zero-fill, tail stores clip, tail query columns
and key rows are masked before the exponent. Grouped heads walk the group's query heads inside the dK / dV
kernel; the dQ kernel maps a query head to its key head.
"""

from typing import NamedTuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import arith
from cutlass.base_dsl.typing import Pointer
from cutlass.experimental import primitives as prims
from cutlass.experimental.cuda import tensor_map as tmap

from cudnn.frost.tile_dsl.barrier import MBarrier, PipelineState, Producer, advance
from cudnn.frost.tile_dsl.handles import GmemTileTma, SmemTile
from cudnn.frost.tile_dsl.tma import tma_load_tile, tma_store_commit, tma_store_tile, tma_store_wait
from cudnn.frost.tile_dsl.wgmma import wgmma_m64nNk16_f32, wgmma_smem_desc
from cudnn.sdpa.fwd.kernels.sm90._common_hopper import store_fragment

_PARAMS = globals().get("FROST_TEMPLATE_PARAMS", 128)
HEAD_DIM = int(_PARAMS) if isinstance(_PARAMS, int) else int(_PARAMS[0])
RATIO = 1 if isinstance(_PARAMS, int) or len(_PARAMS) < 2 else int(_PARAMS[1])
if HEAD_DIM not in (64, 128):
    raise ValueError("the Hopper backward pair serves head dimensions 64 and 128")
if RATIO < 1:
    raise ValueError("the query head ratio must be positive")

D = HEAD_DIM
BPE = 2
SLABS = D // 64  # 64-column SW128 slabs per row block
OWNED_ROWS = 128  # rows of the CTA's resident tile (two warpgroups x 64)
STEP_ROWS = 64  # rows of one streamed tile
STAGES = 3
THREADS = 384
PRODUCER_REGS = 24
CONSUMER_REGS = 240
LOG2E = 1.4426950408889634
RESIDENT_BYTES = OWNED_ROWS * D * BPE
STEP_BYTES = STEP_ROWS * D * BPE
RING = 2 * RESIDENT_BYTES  # the streamed ring follows the two resident tiles
BARS = RING + STAGES * 2 * STEP_BYTES
# one 64-row block of a slab (64 rows x 128 B) and the 16-byte descriptor units of one k16 step along rows
ROWS64_BYTES = 64 * 128
K16_ROW_UNITS = 16 * 128 // 16
_BAR_SLOTS = (
    ("a_full", 1, 1, "tma"),  # the first resident tile (K or Q)
    ("b_full", 1, 1, "tma"),  # the second resident tile (V or dO)
    ("x_full", STAGES, 1, "tma"),  # streamed tile 1 (Q or K)
    ("y_full", STAGES, 1, "tma"),  # streamed tile 2 (dO or V)
    ("x_empty", STAGES, 2, "thread"),
    ("y_empty", STAGES, 2, "thread"),
)
BAR_BYTES = 8 * sum(slots for _, slots, _, _ in _BAR_SLOTS)
SMEM_BYTES = (BARS + BAR_BYTES + 1023) // 1024 * 1024
if SMEM_BYTES > 227 * 1024:
    raise ValueError("the Hopper backward pair exceeds the shared memory carveout")


class Bars(NamedTuple):
    a_full: MBarrier
    b_full: MBarrier
    x_full: MBarrier
    y_full: MBarrier
    x_empty: MBarrier
    y_empty: MBarrier


def _view(slab, offset, dtype, count, alignment=16):
    ptr = Pointer(slab.data_ptr() + offset, dtype=dtype)
    return cutlass.Array(ptr.ir_value(), shape=count, dtype=dtype, alignment=alignment)


def _bars(slab):
    kinds = {"tma": Producer.TMA_LOAD, "thread": Producer.THREAD}
    bars, off = [], BARS
    for _, slots, count, kind in _BAR_SLOTS:
        bars.append(MBarrier(_view(slab, off, cutlass.Int64, slots), stages=slots, init_count=count, producer=kinds[kind], try_wait=True))
        off += 8 * slots
    return Bars(*bars)


def _tile(slab, offset, dtype, rows):
    return SmemTile(
        base=_view(slab, offset, dtype, rows * D, alignment=1024),
        elems_per_stage=rows * D,
        leading_byte_offset=16,
        stride_byte_offset=1024,
        layout=2,
        tma_loads_per_tile=SLABS,
        tma_granu_elems=64,
        tma_subtile_stride_elems=rows * 64,
    )


def _tensor_map(tensor, rows):
    return tmap.create_tensor_map_tiled_from_view(tensor, box_dims=(1, rows, 1, 64), stride_order=(3, 1, 2, 0), swizzle=tmap.TensorMapSwizzle.s128b)


@cute.jit
def _desc(region, rows: cutlass.Constexpr[int], mn_major: cutlass.Constexpr[bool] = False):
    """K-major: 16-byte core matrices at a 1024-byte row-group stride; MN-major: the next 64-column slab is ``rows`` rows away."""
    leading = rows * 128 if cutlass.const_expr(mn_major) else 16
    return wgmma_smem_desc(region, leading_byte_offset=leading, stride_byte_offset=1024)


@cute.jit
def _wait(bar, phase):
    bar.wait(phase)
    prims.bar_warp_sync(cute.arch.FULL_MASK)


@cute.jit
def _select(keep, value):
    return cutlass.Float32(arith.select(keep.ir_value(), value.ir_value(), cutlass.Float32(0.0).ir_value()))


@cute.jit
def _zeros(n: cutlass.Constexpr[int]):
    return cutlass.Vector.from_elements(tuple(cutlass.Float32(0.0) for _ in range(n)), cutlass.Float32)


def _kmajor_offset(k, rows):
    """Descriptor offset (16-byte units) of k16 step ``k`` of a ``rows``-row K-major operand: slab ``k // 4``, 32 bytes per step inside it."""
    return (k // 4) * (rows * 128 // 16) + (k % 4) * 2


@cute.jit
def _mma_ss(cs, a_desc, b_desc, a_rows: cutlass.Constexpr[int], b_rows: cutlass.Constexpr[int], dtype: cutlass.Constexpr):
    """One m64 n64 contraction over the head dimension; returns the C vector (the caller commits)."""
    prims.wgmma_fence_aligned()
    for k in cutlass.range_constexpr(D // 16):
        cs = wgmma_m64nNk16_f32(cs, a_desc + _kmajor_offset(k, a_rows), b_desc + _kmajor_offset(k, b_rows), dtype, accumulate=k != 0)
    prims.wgmma_commit_group_sync_aligned()
    return cs


@cute.jit
def _mma_rs(acc, words, b_desc_mn, dtype: cutlass.Constexpr):
    """Accumulate a 64-row register A operand (four k16 steps over the streamed rows) against an MN-major B operand."""
    prims.wgmma_fence_aligned()
    for k in cutlass.range_constexpr(STEP_ROWS // 16):
        acc = wgmma_m64nNk16_f32(acc, words[4 * k : 4 * k + 4], b_desc_mn + K16_ROW_UNITS * k, dtype, transpose_b=True, accumulate=True)
    prims.wgmma_commit_group_sync_aligned()
    return acc


@cute.jit
def _epilogue(acc, slab, region_offset, tid, bar_id: cutlass.Constexpr[int], dtype: cutlass.Constexpr, desc, row0, head, batch):
    """Stage the warpgroup's 64 x D accumulator through SW128 shared memory and TMA-store it (rows past the sequence clip)."""
    region = _view(slab, region_offset, dtype, STEP_ROWS * D, alignment=1024)
    words = acc.to(dtype).bitcast(cutlass.Uint32)
    store_fragment(words, region, tid)
    prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
    prims.barrier_cta_sync(bar_id, thread_count=cute.arch.THREADS_PER_WARPGROUP)
    if tid < cute.arch.WARP_SIZE:
        if prims.elect_sync():
            tma_store_tile(_tile(slab, region_offset, dtype, STEP_ROWS), GmemTileTma(desc)(0, row0, head, batch))
            tma_store_commit()
            tma_store_wait(0)


# ----------------------------------------------------------------------------------------------------------------------
# the key-owner dK / dV kernel
# ----------------------------------------------------------------------------------------------------------------------


@cute.jit
def _dkdv_bounds(kv_tile, sq, causal: cutlass.Constexpr[bool]):
    lo = cutlass.Int32(0)
    if cutlass.const_expr(causal):
        lo = kv_tile * (OWNED_ROWS // STEP_ROWS)
    hi = (sq + STEP_ROWS - 1) // STEP_ROWS
    return lo, hi


@cute.jit
def _dkdv_producer(slab, bars, k_desc, v_desc, q_desc, do_desc, kv_tile, kv_head, batch, sq, dtype: cutlass.Constexpr, causal: cutlass.Constexpr[bool]):
    bars.a_full.arrive(n_bytes=RESIDENT_BYTES, pred=prims.elect_sync())
    tma_load_tile(_tile(slab, 0, dtype, OWNED_ROWS), GmemTileTma(k_desc)(0, kv_tile * OWNED_ROWS, kv_head, batch), bars.a_full.smem_ptr)
    bars.b_full.arrive(n_bytes=RESIDENT_BYTES, pred=prims.elect_sync())
    tma_load_tile(_tile(slab, RESIDENT_BYTES, dtype, OWNED_ROWS), GmemTileTma(v_desc)(0, kv_tile * OWNED_ROWS, kv_head, batch), bars.b_full.smem_ptr)
    lo, hi = _dkdv_bounds(kv_tile, sq, causal)
    state = PipelineState.start(phase=1)
    for g in cutlass.range(0, RATIO, 1, unroll=1):
        head = kv_head * RATIO + g
        for q_tile in cutlass.range(lo, hi, 1, unroll=1):
            _wait(bars.x_empty[state.idx], state.phase)
            bars.x_full[state.idx].arrive(n_bytes=STEP_BYTES, pred=prims.elect_sync())
            tma_load_tile(
                _tile(slab, RING + state.idx * 2 * STEP_BYTES, dtype, STEP_ROWS),
                GmemTileTma(q_desc)(0, q_tile * STEP_ROWS, head, batch),
                bars.x_full[state.idx].smem_ptr,
            )
            _wait(bars.y_empty[state.idx], state.phase)
            bars.y_full[state.idx].arrive(n_bytes=STEP_BYTES, pred=prims.elect_sync())
            tma_load_tile(
                _tile(slab, RING + state.idx * 2 * STEP_BYTES + STEP_BYTES, dtype, STEP_ROWS),
                GmemTileTma(do_desc)(0, q_tile * STEP_ROWS, head, batch),
                bars.y_full[state.idx].smem_ptr,
            )
            state = advance(state, STAGES)


@cute.jit
def _dkdv_step(
    slab, bars, state, tid, kd, vd, lse, delta, batch, head, kv_row0, q0, sq, skv, scale, dv, dk, dtype: cutlass.Constexpr, causal: cutlass.Constexpr[bool]
):
    _wait(bars.x_full[state.idx], state.phase)
    _wait(bars.y_full[state.idx], state.phase)
    q_region = _view(slab, RING + state.idx * 2 * STEP_BYTES, dtype, STEP_ROWS * D, alignment=1024)
    do_region = _view(slab, RING + state.idx * 2 * STEP_BYTES + STEP_BYTES, dtype, STEP_ROWS * D, alignment=1024)
    qd = _desc(q_region, STEP_ROWS)
    dod = _desc(do_region, STEP_ROWS)
    qdt = _desc(q_region, STEP_ROWS, mn_major=True)
    dodt = _desc(do_region, STEP_ROWS, mn_major=True)
    cs = _mma_ss(_zeros(32), kd, qd, OWNED_ROWS, STEP_ROWS, dtype)
    cdp = _mma_ss(_zeros(32), vd, dod, OWNED_ROWS, STEP_ROWS, dtype)
    scale_log2 = scale * cutlass.Float32(LOG2E)
    row_base = kv_row0 + 16 * (tid // cute.arch.WARP_SIZE) + (tid % cute.arch.WARP_SIZE) // 4
    col_base = q0 + 2 * (tid % 4)
    # the 16 column Stats of this thread, loaded while the WGMMAs run
    stats = []
    for c in cutlass.range_constexpr(16):
        q = col_base + (c % 2) + 8 * (c // 2)
        stats.append(cutlass.Float32(lse[batch, head, cute.math.min(q, sq - 1)]) * cutlass.Float32(LOG2E))
    prims.wgmma_wait_group_sync_aligned(0)
    p, ds = [], []
    for i in cutlass.range_constexpr(32):
        q = col_base + (i % 2) + 8 * (i // 4)
        kv = row_base + 8 * ((i // 2) % 2)
        qc = cute.math.min(q, sq - 1)
        stat = stats[(i % 2) + 2 * (i // 4)]
        value = cute.math.exp2(cs[i] * scale_log2 - stat, fastmath=True)
        value = _select(q < sq, value)
        value = _select(kv < skv, value)
        if cutlass.const_expr(causal):
            value = _select(q >= kv, value)
        p.append(value)
        ds.append(value * (cdp[i] - cutlass.Float32(delta[batch, head, qc])) * scale)
    p_words = cutlass.Vector.from_elements(tuple(p), cutlass.Float32).to(dtype).bitcast(cutlass.Uint32)
    ds_words = cutlass.Vector.from_elements(tuple(ds), cutlass.Float32).to(dtype).bitcast(cutlass.Uint32)
    # both register A operands stay live until the final wait: the WGMMAs read them asynchronously
    dv = _mma_rs(dv, p_words, dodt, dtype)
    dk = _mma_rs(dk, ds_words, qdt, dtype)
    prims.wgmma_wait_group_sync_aligned(0)
    prims.bar_warp_sync(cute.arch.FULL_MASK)
    if tid < cute.arch.WARP_SIZE:
        if prims.elect_sync():
            bars.x_empty[state.idx].arrive()
            bars.y_empty[state.idx].arrive()
    return dv, dk


@cute.jit
def _dkdv_consumer(
    slab, bars, w, tid, lse, delta, dk_desc, dv_desc, kv_tile, kv_head, batch, sq, skv, scale, dtype: cutlass.Constexpr, causal: cutlass.Constexpr[bool]
):
    lo, hi = _dkdv_bounds(kv_tile, sq, causal)
    dv_acc = cutlass.Array(cutlass.Float32, D // 2, alignment=16)
    dk_acc = cutlass.Array(cutlass.Float32, D // 2, alignment=16)
    dv_acc.store(_zeros(D // 2))
    dk_acc.store(_zeros(D // 2))
    kv_row0 = kv_tile * OWNED_ROWS + w * 64
    if lo < hi:
        _wait(bars.a_full, cutlass.Int32(0))
        _wait(bars.b_full, cutlass.Int32(0))
        kd = _desc(_view(slab, w * ROWS64_BYTES, dtype, OWNED_ROWS * D, alignment=1024), OWNED_ROWS)
        vd = _desc(_view(slab, RESIDENT_BYTES + w * ROWS64_BYTES, dtype, OWNED_ROWS * D, alignment=1024), OWNED_ROWS)
        state = PipelineState.start()
        for g in cutlass.range(0, RATIO, 1, unroll=1):
            head = kv_head * RATIO + g
            for q_tile in cutlass.range(lo, hi, 1, unroll=1):
                dv, dk = _dkdv_step(
                    slab,
                    bars,
                    state,
                    tid,
                    kd,
                    vd,
                    lse,
                    delta,
                    batch,
                    head,
                    kv_row0,
                    q_tile * STEP_ROWS,
                    sq,
                    skv,
                    scale,
                    dv_acc.load(0, D // 2),
                    dk_acc.load(0, D // 2),
                    dtype,
                    causal,
                )
                dv_acc.store(dv)
                dk_acc.store(dk)
                state = advance(state, STAGES)
    # both consumer warpgroups have finished reading the ring before it is reused as the output staging
    prims.barrier_cta_sync(1, thread_count=2 * cute.arch.THREADS_PER_WARPGROUP)
    if w == 0:
        _epilogue(dv_acc.load(0, D // 2), slab, RING, tid, 2, dtype, dv_desc, kv_row0, kv_head, batch)
        _epilogue(dk_acc.load(0, D // 2), slab, RING + STEP_BYTES, tid, 2, dtype, dk_desc, kv_row0, kv_head, batch)
    else:
        _epilogue(dv_acc.load(0, D // 2), slab, RING + 2 * STEP_BYTES, tid, 3, dtype, dv_desc, kv_row0, kv_head, batch)
        _epilogue(dk_acc.load(0, D // 2), slab, RING + 3 * STEP_BYTES, tid, 3, dtype, dk_desc, kv_row0, kv_head, batch)


@cute.kernel
def bprop_sm90_dkdv_kernel(
    q_desc: cutlass.GridConstant[tmap.TensorMap],
    k_desc: cutlass.GridConstant[tmap.TensorMap],
    v_desc: cutlass.GridConstant[tmap.TensorMap],
    do_desc: cutlass.GridConstant[tmap.TensorMap],
    dk_desc: cutlass.GridConstant[tmap.TensorMap],
    dv_desc: cutlass.GridConstant[tmap.TensorMap],
    lse: cute.Tensor,
    delta: cute.Tensor,
    sq: cutlass.Int32,
    skv: cutlass.Int32,
    scale: cutlass.Float32,
    dtype: cutlass.Constexpr,
    causal: cutlass.Constexpr[bool],
):
    kv_tile, kv_head, batch = cute.arch.block_idx()
    slab = cutlass.Array(cutlass.Uint8, SMEM_BYTES, alignment=1024, space=cutlass.AddressSpace.smem)
    bars = _bars(slab)
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    if warp == 0:
        if prims.elect_sync():
            for i in cutlass.range_constexpr(len(bars)):
                for stage in cutlass.range_constexpr(bars[i].stages):
                    bars[i][stage].init()
    prims.fence_mbarrier_init()
    prims.barrier_cta_sync()
    wg = warp // cute.arch.WARPS_PER_WARPGROUP
    tid = cute.arch.thread_idx()[0] % cute.arch.THREADS_PER_WARPGROUP
    if wg == 0:
        prims.setmaxregister(PRODUCER_REGS, prims.SetMaxRegisterAction.DECREASE)
        if warp == 0:
            _dkdv_producer(slab, bars, k_desc, v_desc, q_desc, do_desc, kv_tile, kv_head, batch, sq, dtype, causal)
    else:
        prims.setmaxregister(CONSUMER_REGS, prims.SetMaxRegisterAction.INCREASE)
        _dkdv_consumer(slab, bars, wg - 1, tid, lse, delta, dk_desc, dv_desc, kv_tile, kv_head, batch, sq, skv, scale, dtype, causal)


bprop_sm90_dkdv_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def dkdv_host(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    do: cute.Tensor,
    lse: cute.Tensor,
    delta: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    scale: cutlass.Float32,
    causal: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    if cutlass.const_expr(q.element_type not in (cutlass.BFloat16, cutlass.Float16)):
        raise ValueError("the Hopper backward pair requires BF16 / FP16")
    if cutlass.const_expr(q.shape[3] != D or k.shape[3] != D or q.shape[2] != k.shape[2] * RATIO):
        raise ValueError("the Hopper backward pair requires its configured head dimension and head ratio")
    nk = (k.shape[1] + OWNED_ROWS - 1) // OWNED_ROWS
    bprop_sm90_dkdv_kernel(
        _tensor_map(q, STEP_ROWS),
        _tensor_map(k, OWNED_ROWS),
        _tensor_map(v, OWNED_ROWS),
        _tensor_map(do, STEP_ROWS),
        _tensor_map(dk, STEP_ROWS),
        _tensor_map(dv, STEP_ROWS),
        lse,
        delta,
        q.shape[1],
        k.shape[1],
        scale,
        q.element_type,
        causal,
    ).launch(grid=(nk, k.shape[2], q.shape[0]), block=(THREADS, 1, 1), stream=stream)


# ----------------------------------------------------------------------------------------------------------------------
# the query-owner dQ kernel
# ----------------------------------------------------------------------------------------------------------------------


@cute.jit
def _dq_bound(q_tile, skv, causal: cutlass.Constexpr[bool]):
    hi = (skv + STEP_ROWS - 1) // STEP_ROWS
    if cutlass.const_expr(causal):
        hi = cute.math.min(hi, (q_tile + 1) * (OWNED_ROWS // STEP_ROWS))
    return hi


@cute.jit
def _dq_producer(slab, bars, q_desc, do_desc, k_desc, v_desc, q_tile, head, batch, skv, dtype: cutlass.Constexpr, causal: cutlass.Constexpr[bool]):
    kv_head = head // RATIO
    bars.a_full.arrive(n_bytes=RESIDENT_BYTES, pred=prims.elect_sync())
    tma_load_tile(_tile(slab, 0, dtype, OWNED_ROWS), GmemTileTma(q_desc)(0, q_tile * OWNED_ROWS, head, batch), bars.a_full.smem_ptr)
    bars.b_full.arrive(n_bytes=RESIDENT_BYTES, pred=prims.elect_sync())
    tma_load_tile(_tile(slab, RESIDENT_BYTES, dtype, OWNED_ROWS), GmemTileTma(do_desc)(0, q_tile * OWNED_ROWS, head, batch), bars.b_full.smem_ptr)
    hi = _dq_bound(q_tile, skv, causal)
    state = PipelineState.start(phase=1)
    for kv_tile in cutlass.range(0, hi, 1, unroll=1):
        _wait(bars.x_empty[state.idx], state.phase)
        bars.x_full[state.idx].arrive(n_bytes=STEP_BYTES, pred=prims.elect_sync())
        tma_load_tile(
            _tile(slab, RING + state.idx * 2 * STEP_BYTES, dtype, STEP_ROWS),
            GmemTileTma(k_desc)(0, kv_tile * STEP_ROWS, kv_head, batch),
            bars.x_full[state.idx].smem_ptr,
        )
        _wait(bars.y_empty[state.idx], state.phase)
        bars.y_full[state.idx].arrive(n_bytes=STEP_BYTES, pred=prims.elect_sync())
        tma_load_tile(
            _tile(slab, RING + state.idx * 2 * STEP_BYTES + STEP_BYTES, dtype, STEP_ROWS),
            GmemTileTma(v_desc)(0, kv_tile * STEP_ROWS, kv_head, batch),
            bars.y_full[state.idx].smem_ptr,
        )
        state = advance(state, STAGES)


@cute.jit
def _dq_step(slab, bars, state, tid, qd, dod, stats, dstats, q_row0, kv0, skv, scale, dq, dtype: cutlass.Constexpr, causal: cutlass.Constexpr[bool]):
    _wait(bars.x_full[state.idx], state.phase)
    _wait(bars.y_full[state.idx], state.phase)
    k_region = _view(slab, RING + state.idx * 2 * STEP_BYTES, dtype, STEP_ROWS * D, alignment=1024)
    v_region = _view(slab, RING + state.idx * 2 * STEP_BYTES + STEP_BYTES, dtype, STEP_ROWS * D, alignment=1024)
    kd = _desc(k_region, STEP_ROWS)
    vd = _desc(v_region, STEP_ROWS)
    kdt = _desc(k_region, STEP_ROWS, mn_major=True)
    cs = _mma_ss(_zeros(32), qd, kd, OWNED_ROWS, STEP_ROWS, dtype)
    cdp = _mma_ss(_zeros(32), dod, vd, OWNED_ROWS, STEP_ROWS, dtype)
    prims.wgmma_wait_group_sync_aligned(0)
    scale_log2 = scale * cutlass.Float32(LOG2E)
    row_base = q_row0 + 16 * (tid // cute.arch.WARP_SIZE) + (tid % cute.arch.WARP_SIZE) // 4
    col_base = kv0 + 2 * (tid % 4)
    ds = []
    for i in cutlass.range_constexpr(32):
        half = (i // 2) % 2
        kv = col_base + (i % 2) + 8 * (i // 4)
        q = row_base + 8 * half
        value = cute.math.exp2(cs[i] * scale_log2 - stats[half], fastmath=True)
        value = _select(kv < skv, value)
        if cutlass.const_expr(causal):
            value = _select(kv <= q, value)
        ds.append(value * (cdp[i] - dstats[half]) * scale)
    ds_vec = cutlass.Vector.from_elements(tuple(ds), cutlass.Float32)
    dq = _mma_rs(dq, ds_vec.to(dtype).bitcast(cutlass.Uint32), kdt, dtype)
    prims.wgmma_wait_group_sync_aligned(0)
    prims.bar_warp_sync(cute.arch.FULL_MASK)
    if tid < cute.arch.WARP_SIZE:
        if prims.elect_sync():
            bars.x_empty[state.idx].arrive()
            bars.y_empty[state.idx].arrive()
    return dq


@cute.jit
def _dq_consumer(slab, bars, w, tid, lse, delta, dq_desc, q_tile, head, batch, sq, skv, scale, dtype: cutlass.Constexpr, causal: cutlass.Constexpr[bool]):
    hi = _dq_bound(q_tile, skv, causal)
    dq_acc = cutlass.Array(cutlass.Float32, D // 2, alignment=16)
    dq_acc.store(_zeros(D // 2))
    q_row0 = q_tile * OWNED_ROWS + w * 64
    row_base = q_row0 + 16 * (tid // cute.arch.WARP_SIZE) + (tid % cute.arch.WARP_SIZE) // 4
    # the two owned rows' Stats (log2 units) and delta; rows past S_q exponentiate to zero and are not stored
    stat_values, dstat_values = [], []
    for half in cutlass.range_constexpr(2):
        q = row_base + 8 * half
        qc = cute.math.min(q, sq - 1)
        stat = cutlass.Float32(lse[batch, head, qc]) * cutlass.Float32(LOG2E)
        stat = cutlass.Float32(arith.select((q < sq).ir_value(), stat.ir_value(), cutlass.Float32(float("inf")).ir_value()))
        stat_values.append(stat)
        dstat_values.append(cutlass.Float32(delta[batch, head, qc]))
    stats = cutlass.Vector.from_elements(tuple(stat_values), cutlass.Float32)
    dstats = cutlass.Vector.from_elements(tuple(dstat_values), cutlass.Float32)
    _wait(bars.a_full, cutlass.Int32(0))
    _wait(bars.b_full, cutlass.Int32(0))
    qd = _desc(_view(slab, w * ROWS64_BYTES, dtype, OWNED_ROWS * D, alignment=1024), OWNED_ROWS)
    dod = _desc(_view(slab, RESIDENT_BYTES + w * ROWS64_BYTES, dtype, OWNED_ROWS * D, alignment=1024), OWNED_ROWS)
    state = PipelineState.start()
    for kv_tile in cutlass.range(0, hi, 1, unroll=1):
        dq = _dq_step(slab, bars, state, tid, qd, dod, stats, dstats, q_row0, kv_tile * STEP_ROWS, skv, scale, dq_acc.load(0, D // 2), dtype, causal)
        dq_acc.store(dq)
        state = advance(state, STAGES)
    prims.barrier_cta_sync(1, thread_count=2 * cute.arch.THREADS_PER_WARPGROUP)
    if w == 0:
        _epilogue(dq_acc.load(0, D // 2), slab, RING, tid, 2, dtype, dq_desc, q_row0, head, batch)
    else:
        _epilogue(dq_acc.load(0, D // 2), slab, RING + STEP_BYTES, tid, 3, dtype, dq_desc, q_row0, head, batch)


@cute.kernel
def bprop_sm90_dq_kernel(
    q_desc: cutlass.GridConstant[tmap.TensorMap],
    k_desc: cutlass.GridConstant[tmap.TensorMap],
    v_desc: cutlass.GridConstant[tmap.TensorMap],
    do_desc: cutlass.GridConstant[tmap.TensorMap],
    dq_desc: cutlass.GridConstant[tmap.TensorMap],
    lse: cute.Tensor,
    delta: cute.Tensor,
    sq: cutlass.Int32,
    skv: cutlass.Int32,
    scale: cutlass.Float32,
    dtype: cutlass.Constexpr,
    causal: cutlass.Constexpr[bool],
):
    q_tile, head, batch = cute.arch.block_idx()
    if cutlass.const_expr(causal):
        q_tile = cute.arch.grid_dim()[0] - 1 - q_tile
    slab = cutlass.Array(cutlass.Uint8, SMEM_BYTES, alignment=1024, space=cutlass.AddressSpace.smem)
    bars = _bars(slab)
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    if warp == 0:
        if prims.elect_sync():
            for i in cutlass.range_constexpr(len(bars)):
                for stage in cutlass.range_constexpr(bars[i].stages):
                    bars[i][stage].init()
    prims.fence_mbarrier_init()
    prims.barrier_cta_sync()
    wg = warp // cute.arch.WARPS_PER_WARPGROUP
    tid = cute.arch.thread_idx()[0] % cute.arch.THREADS_PER_WARPGROUP
    if wg == 0:
        prims.setmaxregister(PRODUCER_REGS, prims.SetMaxRegisterAction.DECREASE)
        if warp == 0:
            _dq_producer(slab, bars, q_desc, do_desc, k_desc, v_desc, q_tile, head, batch, skv, dtype, causal)
    else:
        prims.setmaxregister(CONSUMER_REGS, prims.SetMaxRegisterAction.INCREASE)
        _dq_consumer(slab, bars, wg - 1, tid, lse, delta, dq_desc, q_tile, head, batch, sq, skv, scale, dtype, causal)


bprop_sm90_dq_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def dq_host(
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
):
    if cutlass.const_expr(q.element_type not in (cutlass.BFloat16, cutlass.Float16)):
        raise ValueError("the Hopper backward pair requires BF16 / FP16")
    if cutlass.const_expr(q.shape[3] != D or k.shape[3] != D or q.shape[2] != k.shape[2] * RATIO):
        raise ValueError("the Hopper backward pair requires its configured head dimension and head ratio")
    nq = (q.shape[1] + OWNED_ROWS - 1) // OWNED_ROWS
    bprop_sm90_dq_kernel(
        _tensor_map(q, OWNED_ROWS),
        _tensor_map(k, STEP_ROWS),
        _tensor_map(v, STEP_ROWS),
        _tensor_map(do, OWNED_ROWS),
        _tensor_map(dq, STEP_ROWS),
        lse,
        delta,
        q.shape[1],
        k.shape[1],
        scale,
        q.element_type,
        causal,
    ).launch(grid=(nq, q.shape[2], q.shape[0]), block=(THREADS, 1, 1), stream=stream)
