# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Hopper dense d64 / d128 backward: one key-owner kernel with an ordered dQ relay.

Each CTA keeps one 128-row key tile resident and streams 64-row Q / dO tiles with their Stats through a
three-stage TMA ring. Two WGMMA consumer warpgroups own 64 key rows each: they recompute S^T and dP^T, keep
P^T and dS^T in registers as the A operands of dV += P^T dO and dK += dS^T Q, stage dS^T in shared memory
and contract it against the resident key tile into the step's fp32 dQ partial. A relay warp adds every
partial into the fp32 dQ accumulator with a TMA reduction once the query tile's turn counter reaches the
key tile index, so every query tile accumulates its key tiles in index order. A CTA claims its key tile
from a per-head ticket when it starts, so the tiles it waits for always belong to CTAs that are already
running; causal CTAs walk the query tiles from the sequence end down to their diagonal, unmasked CTAs walk
them upward.

Sequence lengths need not be tile multiples: tail loads zero-fill, tail stores clip, tail query columns
and key rows are masked before the exponent. Grouped heads walk the group's query heads inside every query step.
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
from cudnn.frost.tile_dsl.tma import tma_load_tile, tma_reduce_add_subtile, tma_store_commit, tma_store_tile, tma_store_wait
from cudnn.frost.tile_dsl.wgmma import wgmma_m64nNk16_f32, wgmma_smem_desc
from cudnn.sdpa.fwd.kernels.sm90._common_hopper import store_fragment

_PARAMS = globals().get("FROST_TEMPLATE_PARAMS", 128)
HEAD_DIM = int(_PARAMS) if isinstance(_PARAMS, int) else int(_PARAMS[0])
RATIO = 1 if isinstance(_PARAMS, int) or len(_PARAMS) < 2 else int(_PARAMS[1])
if HEAD_DIM not in (64, 128):
    raise ValueError("the Hopper backward kernel serves head dimensions 64 and 128")
if RATIO < 1:
    raise ValueError("the query head ratio must be positive")

D = HEAD_DIM
BPE = 2
SLABS = D // 64  # 64-column SW128 slabs per row block
OWNED_ROWS = 128  # rows of the CTA's resident tile (two warpgroups x 64)
STEP_ROWS = 64  # rows of one streamed tile
STAGES = 3
THREADS = 384
PRODUCER_REGS = 32
CONSUMER_REGS = 232
LOG2E = 1.4426950408889634
RESIDENT_BYTES = OWNED_ROWS * D * BPE
STEP_BYTES = STEP_ROWS * D * BPE
SLAB_BYTES = OWNED_ROWS * 128  # one 64-column slab of a resident tile
HALF_DS_BYTES = 64 * STEP_ROWS * BPE  # one warpgroup's 64 key rows of the dS^T tile
DS_BYTES = 2 * HALF_DS_BYTES  # the dS^T tile: 128 key rows x 64 query columns
DQ_SLICES = D // 32  # 32-column fp32 SW128 slices of the dQ partial
SLICE_BYTES = STEP_ROWS * 128
DQS_BYTES = DQ_SLICES * SLICE_BYTES
RING = 2 * RESIDENT_BYTES  # the streamed ring follows the two resident tiles
DS = RING + STAGES * 2 * STEP_BYTES  # two dS^T buffers follow the ring
DQS = DS + 2 * DS_BYTES  # the fp32 dQ partial follows them
STATS = DQS + DQS_BYTES  # per stage: the step's 64 LSE values (log2 units), then its 64 delta values
STATS_BYTES = 2 * STEP_ROWS * 4
BARS = STATS + STAGES * STATS_BYTES
# one 64-row block of a slab (64 rows x 128 B) and the 16-byte descriptor units of one k16 step along rows
ROWS64_BYTES = 64 * 128
K16_ROW_UNITS = 16 * 128 // 16
_BAR_SLOTS = (
    ("a_full", 1, 1, "tma"),  # the resident K tile
    ("b_full", 1, 1, "tma"),  # the resident V tile
    ("x_full", STAGES, 1, "tma"),  # streamed Q
    ("y_full", STAGES, 1, "tma"),  # streamed dO
    ("s_full", STAGES, 1, "thread"),  # the stats loader warp
    ("x_empty", STAGES, 2, "thread"),
    ("y_empty", STAGES, 2, "thread"),
    ("dq_full", 1, 4, "thread"),  # one lane per warp of the folding warpgroup, once per step
    ("dq_empty", 1, 1, "thread"),  # the relay warp, once per step
    ("fold_full", 1, 4, "thread"),  # d64: warpgroup 0 staged its partial
)
BAR_BYTES = 8 * sum(slots for _, slots, _, _ in _BAR_SLOTS)
TICKET = BARS + BAR_BYTES  # the claimed key tile index
SMEM_BYTES = (TICKET + 16 + 1023) // 1024 * 1024
if SMEM_BYTES > 227 * 1024:
    raise ValueError("the Hopper backward kernel exceeds the shared memory carveout")


class Bars(NamedTuple):
    a_full: MBarrier
    b_full: MBarrier
    x_full: MBarrier
    y_full: MBarrier
    s_full: MBarrier
    x_empty: MBarrier
    y_empty: MBarrier
    dq_full: MBarrier
    dq_empty: MBarrier
    fold_full: MBarrier


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


def _dq_tile(slab):
    return SmemTile(
        base=_view(slab, DQS, cutlass.Float32, STEP_ROWS * D, alignment=1024),
        elems_per_stage=STEP_ROWS * D,
        leading_byte_offset=16,
        stride_byte_offset=1024,
        layout=2,
        tma_loads_per_tile=DQ_SLICES,
        tma_granu_elems=32,
        tma_subtile_stride_elems=STEP_ROWS * 32,
    )


def _tensor_map(tensor, rows):
    return tmap.create_tensor_map_tiled_from_view(tensor, box_dims=(1, rows, 1, 64), stride_order=(3, 1, 2, 0), swizzle=tmap.TensorMapSwizzle.s128b)


def _tensor_map_f32(tensor):
    return tmap.create_tensor_map_tiled_from_view(tensor, box_dims=(1, STEP_ROWS, 1, 32), stride_order=(3, 1, 2, 0), swizzle=tmap.TensorMapSwizzle.s128b)


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
def _select_i32(keep, a, b):
    return cutlass.Int32(arith.select(keep.ir_value(), a.ir_value(), b.ir_value()))


@cute.jit
def _zeros(n: cutlass.Constexpr[int]):
    return cutlass.Vector.from_elements(tuple(cutlass.Float32(0.0) for _ in range(n)), cutlass.Float32)


def _kmajor_offset(k, rows):
    """Descriptor offset (16-byte units) of k16 step ``k`` of a ``rows``-row K-major operand: slab ``k // 4``, 32 bytes per step inside it."""
    return (k // 4) * (rows * 128 // 16) + (k % 4) * 2


@cute.jit
def _walk(kv_tile, sq, causal: cutlass.Constexpr[bool]):
    """The CTA's query steps: ``count`` 64-row tiles from ``start``; causal CTAs walk down from the sequence end to their
    diagonal, unmasked CTAs walk up from the first tile."""
    tiles = (sq + STEP_ROWS - 1) // STEP_ROWS
    diagonal = kv_tile * (OWNED_ROWS // STEP_ROWS)
    if cutlass.const_expr(causal):
        start = tiles - 1
        count = _select_i32(diagonal < tiles, tiles - diagonal, cutlass.Int32(0))
    else:
        start = cutlass.Int32(0)
        count = tiles
    return start, count, tiles


@cute.jit
def _q_tile(start, i, causal: cutlass.Constexpr[bool]):
    if cutlass.const_expr(causal):
        return start - i
    return start + i


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
def _mma_dq(ds_desc_mn, k_desc_mn, dtype: cutlass.Constexpr):
    """The warpgroup's 64 x D fp32 dQ partial: dS against its own 64 key rows of the resident key tile, both MN-major."""
    acc = _zeros(D // 2)
    prims.wgmma_fence_aligned()
    for k in cutlass.range_constexpr(64 // 16):
        acc = wgmma_m64nNk16_f32(
            acc, ds_desc_mn + K16_ROW_UNITS * k, k_desc_mn + K16_ROW_UNITS * k, dtype, transpose_a=True, transpose_b=True, accumulate=k != 0
        )
    prims.wgmma_commit_group_sync_aligned()
    return acc


@cute.jit
def _stage_dq(acc, slab, tid):
    """Store a warpgroup's 64 x D fp32 fragment into the SW128 dQ staging slices."""
    row_base = 16 * (tid // cute.arch.WARP_SIZE) + (tid % cute.arch.WARP_SIZE) // 4
    upper = (tid % 4) >= 2
    for i in cutlass.range_constexpr(D // 4):
        # lanes 2 and 3 of a quad store the pair 16 columns away so a half-warp covers eight distinct 16-byte chunks
        j = i ^ 4
        row = row_base + 8 * (i % 2)
        col = 2 * (tid % 4) + _select_i32(upper, cutlass.Int32(8 * (j // 2)), cutlass.Int32(8 * (i // 2)))
        c = col % 32
        offset = DQS + (col // 32) * SLICE_BYTES + row * 128 + (((c // 4) ^ (row % 8)) * 16) + (c % 4) * 4
        lo = cutlass.Float32(arith.select(upper.ir_value(), acc[2 * j].ir_value(), acc[2 * i].ir_value()))
        hi = cutlass.Float32(arith.select(upper.ir_value(), acc[2 * j + 1].ir_value(), acc[2 * i + 1].ir_value()))
        Pointer(slab.data_ptr() + offset, dtype=cutlass.Float32).store(cutlass.Vector.from_elements((lo, hi), cutlass.Float32), alignment=8)


@cute.jit
def _fold_dq(acc, slab, tid):
    """Add a warpgroup's 64 x D fp32 fragment onto the staged dQ partial (the same lane pairing as the store)."""
    row_base = 16 * (tid // cute.arch.WARP_SIZE) + (tid % cute.arch.WARP_SIZE) // 4
    upper = (tid % 4) >= 2
    for i in cutlass.range_constexpr(D // 4):
        j = i ^ 4
        row = row_base + 8 * (i % 2)
        col = 2 * (tid % 4) + _select_i32(upper, cutlass.Int32(8 * (j // 2)), cutlass.Int32(8 * (i // 2)))
        c = col % 32
        offset = DQS + (col // 32) * SLICE_BYTES + row * 128 + (((c // 4) ^ (row % 8)) * 16) + (c % 4) * 4
        lo = cutlass.Float32(arith.select(upper.ir_value(), acc[2 * j].ir_value(), acc[2 * i].ir_value()))
        hi = cutlass.Float32(arith.select(upper.ir_value(), acc[2 * j + 1].ir_value(), acc[2 * i + 1].ir_value()))
        ptr = Pointer(slab.data_ptr() + offset, dtype=cutlass.Float32)
        old = ptr.load(count=2, alignment=8)
        ptr.store(cutlass.Vector.from_elements((old[0] + lo, old[1] + hi), cutlass.Float32), alignment=8)


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
# the producer warp
# ----------------------------------------------------------------------------------------------------------------------


@cute.jit
def _producer(slab, bars, k_desc, v_desc, q_desc, do_desc, kv_tile, kv_head, batch, sq, dtype: cutlass.Constexpr, causal: cutlass.Constexpr[bool]):
    bars.a_full.arrive(n_bytes=RESIDENT_BYTES, pred=prims.elect_sync())
    tma_load_tile(_tile(slab, 0, dtype, OWNED_ROWS), GmemTileTma(k_desc)(0, kv_tile * OWNED_ROWS, kv_head, batch), bars.a_full.smem_ptr)
    bars.b_full.arrive(n_bytes=RESIDENT_BYTES, pred=prims.elect_sync())
    tma_load_tile(_tile(slab, RESIDENT_BYTES, dtype, OWNED_ROWS), GmemTileTma(v_desc)(0, kv_tile * OWNED_ROWS, kv_head, batch), bars.b_full.smem_ptr)
    start, count, tiles = _walk(kv_tile, sq, causal)
    state = PipelineState.start(phase=1)
    for outer in cutlass.range(0, count * RATIO, 1, unroll=1):
        i = outer // RATIO
        g = outer % RATIO
        q_tile = _q_tile(start, i, causal)
        head = kv_head * RATIO + g
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
def _stats_loader(slab, bars, lse, delta, kv_tile, kv_head, batch, sq, lane, causal: cutlass.Constexpr[bool]):
    """Stage each step's LSE (log2 units) and delta rows in the stage's stats slot; rows past the sequence read the last row."""
    start, count, tiles = _walk(kv_tile, sq, causal)
    state = PipelineState.start(phase=1)
    for outer in cutlass.range(0, count * RATIO, 1, unroll=1):
        i = outer // RATIO
        g = outer % RATIO
        q0 = _q_tile(start, i, causal) * STEP_ROWS
        head = kv_head * RATIO + g
        _wait(bars.x_empty[state.idx], state.phase)
        slot = STATS + state.idx * STATS_BYTES
        for c in cutlass.range_constexpr(STEP_ROWS // cute.arch.WARP_SIZE):
            column = c * cute.arch.WARP_SIZE + lane
            qc = cute.math.min(q0 + column, sq - 1)
            Pointer(slab.data_ptr() + slot + column * 4, dtype=cutlass.Float32).store(cutlass.Float32(lse[batch, head, qc]) * cutlass.Float32(LOG2E))
            Pointer(slab.data_ptr() + slot + STEP_ROWS * 4 + column * 4, dtype=cutlass.Float32).store(cutlass.Float32(delta[batch, head, qc]))
        prims.bar_warp_sync(cute.arch.FULL_MASK)
        if prims.elect_sync():
            bars.s_full[state.idx].arrive()
        state = advance(state, STAGES)


# ----------------------------------------------------------------------------------------------------------------------
# the relay warp: the fp32 dQ partials in turn order
# ----------------------------------------------------------------------------------------------------------------------


@cute.jit
def _spin(slot, target):
    """Acquire-poll the turn counter until it reads ``target``."""
    seen = prims.load_ext(slot, order=prims.MemOrder.ACQUIRE, scope=prims.MemScope.GPU)
    loop = cutlass.while_generate([seen], lambda value: value != target)
    with loop as [value]:
        value = prims.load_ext(slot, order=prims.MemOrder.ACQUIRE, scope=prims.MemScope.GPU)
        cutlass.yield_out([value])


@cute.jit
def _relay(slab, bars, dq_desc, turns, kv_tile, kv_head, batch, sq, lane, causal: cutlass.Constexpr[bool]):
    """Add every step's dQ partial into the accumulator once the query tile's turn counter reaches the key tile index."""
    start, count, tiles = _walk(kv_tile, sq, causal)
    phase = cutlass.Int32(0)
    tile = _dq_tile(slab)
    counters = turns.iterator.raw_ptr()
    for outer in cutlass.range(0, count * RATIO, 1, unroll=1):
        i = outer // RATIO
        g = outer % RATIO
        q_tile = _q_tile(start, i, causal)
        head = kv_head * RATIO + g
        slot = counters + (
            cutlass.Int64(batch) * cutlass.Int64(turns.stride[0])
            + cutlass.Int64(head) * cutlass.Int64(turns.stride[1])
            + cutlass.Int64(q_tile) * cutlass.Int64(turns.stride[2])
        )
        _wait(bars.dq_full, phase)
        if lane == 0:
            _spin(slot, kv_tile)
        prims.bar_warp_sync(cute.arch.FULL_MASK)
        for s in cutlass.range_constexpr(DQ_SLICES):
            tma_reduce_add_subtile(tile, GmemTileTma(dq_desc)(0, q_tile * STEP_ROWS, head, batch), s)
        tma_store_commit()
        prims.cp_async_bulk_wait_group(0, read=True)
        prims.bar_warp_sync(cute.arch.FULL_MASK)
        if prims.elect_sync():
            bars.dq_empty.arrive()
        prims.cp_async_bulk_wait_group(0)
        prims.bar_warp_sync(cute.arch.FULL_MASK)
        if lane == 0:
            prims.store_ext((kv_tile + cutlass.Int32(1)).ir_value(), slot, order=prims.MemOrder.RELEASE, scope=prims.MemScope.GPU)
        phase = phase ^ 1


# ----------------------------------------------------------------------------------------------------------------------
# the consumer warpgroups
# ----------------------------------------------------------------------------------------------------------------------


@cute.jit
def _step(
    slab,
    bars,
    state,
    buf,
    dq_phase,
    tid,
    w,
    kd,
    vd,
    kdq,
    lse,
    delta,
    batch,
    head,
    kv_row0,
    q0,
    sq,
    skv,
    scale,
    dv,
    dk,
    dtype: cutlass.Constexpr,
    causal: cutlass.Constexpr[bool],
):
    _wait(bars.x_full[state.idx], state.phase)
    _wait(bars.y_full[state.idx], state.phase)
    _wait(bars.s_full[state.idx], state.phase)
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
    # the thread's 16 column Stats and deltas from the stage's stats slot
    lse_slot = _view(slab, STATS + state.idx * STATS_BYTES, cutlass.Float32, STEP_ROWS)
    delta_slot = _view(slab, STATS + state.idx * STATS_BYTES + STEP_ROWS * 4, cutlass.Float32, STEP_ROWS)
    stats, deltas = [], []
    for c in cutlass.range_constexpr(16):
        column = 2 * (tid % 4) + (c % 2) + 8 * (c // 2)
        stats.append(lse_slot[column])
        deltas.append(delta_slot[column])
    prims.wgmma_wait_group_sync_aligned(0)
    p, ds = [], []
    for i in cutlass.range_constexpr(32):
        q = col_base + (i % 2) + 8 * (i // 4)
        kv = row_base + 8 * ((i // 2) % 2)
        c = (i % 2) + 2 * (i // 4)
        value = cute.math.exp2(cs[i] * scale_log2 - stats[c], fastmath=True)
        value = _select(q < sq, value)
        value = _select(kv < skv, value)
        if cutlass.const_expr(causal):
            value = _select(q >= kv, value)
        p.append(value)
        ds.append(value * (cdp[i] - deltas[c]) * scale)
    p_words = cutlass.Vector.from_elements(tuple(p), cutlass.Float32).to(dtype).bitcast(cutlass.Uint32)
    ds_words = cutlass.Vector.from_elements(tuple(ds), cutlass.Float32).to(dtype).bitcast(cutlass.Uint32)
    dv = _mma_rs(dv, p_words, dodt, dtype)
    # this warpgroup's 64 key rows of the dS^T tile, the A operand of its own dQ partial
    ds_region = _view(slab, DS + buf * DS_BYTES + w * HALF_DS_BYTES, dtype, 64 * STEP_ROWS, alignment=1024)
    store_fragment(ds_words, ds_region, tid)
    prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
    dk = _mma_rs(dk, ds_words, qdt, dtype)
    prims.wgmma_wait_group_sync_aligned(0)
    prims.bar_warp_sync(cute.arch.FULL_MASK)
    if tid < cute.arch.WARP_SIZE:
        if prims.elect_sync():
            bars.x_empty[state.idx].arrive()
            bars.y_empty[state.idx].arrive()
    prims.barrier_cta_sync(4 + w, thread_count=cute.arch.THREADS_PER_WARPGROUP)
    dq = _mma_dq(_desc(ds_region, 64, mn_major=True), kdq, dtype)
    prims.wgmma_wait_group_sync_aligned(0)
    # warpgroup 0 stages its partial; warpgroup 1 folds its own onto it and publishes the sum
    if w == 0:
        _wait(bars.dq_empty, dq_phase)
        _stage_dq(dq, slab, tid)
        prims.bar_warp_sync(cute.arch.FULL_MASK)
        if tid % cute.arch.WARP_SIZE == 0:
            bars.fold_full.arrive()
    else:
        _wait(bars.fold_full, dq_phase ^ 1)
        _fold_dq(dq, slab, tid)
        prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
        prims.bar_warp_sync(cute.arch.FULL_MASK)
        if tid % cute.arch.WARP_SIZE == 0:
            bars.dq_full.arrive()
    return dv, dk


@cute.jit
def _consumer(
    slab, bars, w, tid, lse, delta, dk_desc, dv_desc, kv_tile, kv_head, batch, sq, skv, scale, dtype: cutlass.Constexpr, causal: cutlass.Constexpr[bool]
):
    start, count, tiles = _walk(kv_tile, sq, causal)
    dv_acc = cutlass.Array(cutlass.Float32, D // 2, alignment=16)
    dk_acc = cutlass.Array(cutlass.Float32, D // 2, alignment=16)
    dv_acc.store(_zeros(D // 2))
    dk_acc.store(_zeros(D // 2))
    kv_row0 = kv_tile * OWNED_ROWS + w * 64
    if count > 0:
        _wait(bars.a_full, cutlass.Int32(0))
        _wait(bars.b_full, cutlass.Int32(0))
        kd = _desc(_view(slab, w * ROWS64_BYTES, dtype, OWNED_ROWS * D, alignment=1024), OWNED_ROWS)
        vd = _desc(_view(slab, RESIDENT_BYTES + w * ROWS64_BYTES, dtype, OWNED_ROWS * D, alignment=1024), OWNED_ROWS)
        # the warpgroup's own 64 key rows of the resident key tile, all head columns (a slab is OWNED_ROWS rows away)
        kdq = _desc(_view(slab, w * ROWS64_BYTES, dtype, OWNED_ROWS * D, alignment=1024), OWNED_ROWS, mn_major=True)
        state = PipelineState.start()
        dq_phase = cutlass.Int32(1)
        step = cutlass.Int32(0)
        for outer in cutlass.range(0, count * RATIO, 1, unroll=1):
            i = outer // RATIO
            g = outer % RATIO
            q_tile = _q_tile(start, i, causal)
            head = kv_head * RATIO + g
            buf = step % 2
            dv, dk = _step(
                slab,
                bars,
                state,
                buf,
                dq_phase,
                tid,
                w,
                kd,
                vd,
                kdq,
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
            dq_phase = dq_phase ^ 1
            step = step + 1
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
def bprop_sm90_kernel(
    q_desc: cutlass.GridConstant[tmap.TensorMap],
    k_desc: cutlass.GridConstant[tmap.TensorMap],
    v_desc: cutlass.GridConstant[tmap.TensorMap],
    do_desc: cutlass.GridConstant[tmap.TensorMap],
    dk_desc: cutlass.GridConstant[tmap.TensorMap],
    dv_desc: cutlass.GridConstant[tmap.TensorMap],
    dq_desc: cutlass.GridConstant[tmap.TensorMap],
    lse: cute.Tensor,
    delta: cute.Tensor,
    turns: cute.Tensor,
    sq: cutlass.Int32,
    skv: cutlass.Int32,
    heads_kv: cutlass.Int32,
    scale: cutlass.Float32,
    dtype: cutlass.Constexpr,
    causal: cutlass.Constexpr[bool],
):
    block, _, batch = cute.arch.block_idx()
    kv_head = block % heads_kv
    slab = cutlass.Array(cutlass.Uint8, SMEM_BYTES, alignment=1024, space=cutlass.AddressSpace.smem)
    bars = _bars(slab)
    ticket = _view(slab, TICKET, cutlass.Int32, 4)
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    if warp == 0:
        if prims.elect_sync():
            for i in cutlass.range_constexpr(len(bars)):
                for stage in cutlass.range_constexpr(bars[i].stages):
                    bars[i][stage].init()
            # the head's ticket hands out key tiles in start order: a CTA only waits for tiles of CTAs already running
            counter = turns.iterator.raw_ptr() + (
                cutlass.Int64(batch) * cutlass.Int64(turns.stride[0])
                + cutlass.Int64(kv_head) * cutlass.Int64(RATIO) * cutlass.Int64(turns.stride[1])
                + cutlass.Int64(turns.shape[2] - 1) * cutlass.Int64(turns.stride[2])
            )
            claimed = prims.atomicrmw(prims.AtomicOp.ADD, counter, cutlass.Int32(1))
            ticket[0] = cutlass.Int32(claimed)
    prims.fence_mbarrier_init()
    prims.barrier_cta_sync()
    kv_tile = ticket[0]
    wg = warp // cute.arch.WARPS_PER_WARPGROUP
    tid = cute.arch.thread_idx()[0] % cute.arch.THREADS_PER_WARPGROUP
    if wg == 0:
        prims.setmaxregister(PRODUCER_REGS, prims.SetMaxRegisterAction.DECREASE)
        if warp == 0:
            _producer(slab, bars, k_desc, v_desc, q_desc, do_desc, kv_tile, kv_head, batch, sq, dtype, causal)
        if warp == 1:
            _relay(slab, bars, dq_desc, turns, kv_tile, kv_head, batch, sq, tid % cute.arch.WARP_SIZE, causal)
        if warp == 2:
            _stats_loader(slab, bars, lse, delta, kv_tile, kv_head, batch, sq, tid % cute.arch.WARP_SIZE, causal)
    else:
        prims.setmaxregister(CONSUMER_REGS, prims.SetMaxRegisterAction.INCREASE)
        _consumer(slab, bars, wg - 1, tid, lse, delta, dk_desc, dv_desc, kv_tile, kv_head, batch, sq, skv, scale, dtype, causal)


bprop_sm90_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def bprop_host(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    do: cute.Tensor,
    lse: cute.Tensor,
    delta: cute.Tensor,
    dq_accum: cute.Tensor,
    turns: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    scale: cutlass.Float32,
    causal: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    if cutlass.const_expr(q.element_type not in (cutlass.BFloat16, cutlass.Float16)):
        raise ValueError("the Hopper backward kernel requires BF16 / FP16")
    if cutlass.const_expr(q.shape[3] != D or k.shape[3] != D or q.shape[2] != k.shape[2] * RATIO):
        raise ValueError("the Hopper backward kernel requires its configured head dimension and head ratio")
    nk = (k.shape[1] + OWNED_ROWS - 1) // OWNED_ROWS
    bprop_sm90_kernel(
        _tensor_map(q, STEP_ROWS),
        _tensor_map(k, OWNED_ROWS),
        _tensor_map(v, OWNED_ROWS),
        _tensor_map(do, STEP_ROWS),
        _tensor_map(dk, STEP_ROWS),
        _tensor_map(dv, STEP_ROWS),
        _tensor_map_f32(dq_accum),
        lse,
        delta,
        turns,
        q.shape[1],
        k.shape[1],
        k.shape[2],
        scale,
        q.element_type,
        causal,
    ).launch(grid=(nk * k.shape[2], 1, q.shape[0]), block=(THREADS, 1, 1), stream=stream, min_blocks_per_mp=1)
