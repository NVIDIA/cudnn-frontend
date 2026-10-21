# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Grouped d128 backward: one Q-head owner, native dK/dV partials.

The qualified MHA math, epilogue, store and scheduler roles are imported
unchanged. Only the loaders map Q heads to shared K/V heads. Separate fold
kernels consume the uniquely owned native partials after this main pass.
"""

import cutlass
import cutlass.cute as cute
from cutlass.experimental import primitives as nvvm
from cutlass.experimental.cuda import tensor_map as tmap

from cudnn.frost.tile_dsl.barrier import PipelineState, advance
from cudnn.frost.tile_dsl.handles import GmemTileTma, tma_slice_runtime_desc
from cudnn.frost.tile_dsl.scheduler import Sched
from cudnn.frost.tile_dsl.tma import tma_load_subtiles, tma_load_tile, tma_tensormap_acquire
from . import bprop_d128_accum_f16 as dense
from . import bprop_d128_accum_thd_f16 as packed
from .bprop_d128_accum_thd_f16 import MAP_Q, MAP_K, MAP_V, MAP_DO
from ...config_sm100_d128_accum import CFG


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
    nvvm.setmaxregister(CFG.SERVICE_REGS, nvvm.SetMaxRegisterAction.DECREASE)
    q_state = PipelineState.start(phase=1)
    do_state = PipelineState.start(phase=1)
    kv_state = PipelineState.start(phase=1)
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        dense._credit(bars.sched_empty[sched_state.idx])
        lo, hi = dense._bounds(block, sq, causal)
        if lo < hi:
            dense._wait(bars.k_empty, kv_state.phase)
            bars.k_full.arrive(n_bytes=CFG.OPERAND_BYTES, pred=nvvm.elect_sync())
            tma_load_tile(dense._operand(slab, CFG.K, dtype), GmemTileTma(k_desc)(0, block * CFG.M, head // ratio, batch), bars.k_full.smem_ptr)
            dense._wait(bars.v_empty, kv_state.phase)
            bars.v_full.arrive(n_bytes=CFG.OPERAND_BYTES, pred=nvvm.elect_sync())
            tma_load_tile(dense._operand(slab, CFG.V, dtype), GmemTileTma(v_desc)(0, block * CFG.M, head // ratio, batch), bars.v_full.smem_ptr)
            kv_state = advance(kv_state, 1)
        for qtile in cutlass.range(lo, hi, 1, unroll=1):
            dense._wait(bars.q_empty[q_state.idx], q_state.phase)
            bars.q_full[q_state.idx].arrive(n_bytes=CFG.OPERAND_BYTES, pred=nvvm.elect_sync())
            tma_load_tile(
                dense._operand(slab, CFG.Q + q_state.idx * CFG.OPERAND_BYTES, dtype),
                GmemTileTma(q_desc)(0, qtile * CFG.N, head, batch),
                bars.q_full[q_state.idx].smem_ptr,
            )
            dense._wait(bars.do_empty, do_state.phase)
            for half in cutlass.range_constexpr(2):
                bars.do_full[half].arrive(n_bytes=CFG.SLICE_BYTES, pred=nvvm.elect_sync())
            for half in cutlass.range_constexpr(2):
                tma_load_subtiles(
                    dense._operand(slab, CFG.DO, dtype), GmemTileTma(do_desc)(0, qtile * CFG.N, head, batch), bars.do_full[half].smem_ptr, half, 1
                )
            q_state = advance(q_state, CFG.Q_STAGES)
            do_state = advance(do_state, 1)
        block, head, batch, valid, sched_state = dense._next_tile(sched, bars, sched_state, nheads, nk, causal, swizzle, batch_heads)


@cute.jit
def _load_thd(
    slab, bars, sched, maps, block, head, batch, meta, n_batch, causal: cutlass.Constexpr[bool], dtype: cutlass.Constexpr, ratio: cutlass.Constexpr[int]
):
    nvvm.setmaxregister(CFG.SERVICE_REGS, nvvm.SetMaxRegisterAction.DECREASE)
    q_state = PipelineState.start(phase=1)
    do_state = PipelineState.start(phase=1)
    kv_state = PipelineState.start(phase=1)
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        packed._credit(bars.sched_empty[sched_state.idx])
        lo, hi, seq_q, seq_k, q_start = packed._bounds(block, meta, batch, n_batch, causal)
        q_desc = packed._map_ptr(maps, batch, n_batch, MAP_Q)
        k_desc = packed._map_ptr(maps, batch, n_batch, MAP_K)
        v_desc = packed._map_ptr(maps, batch, n_batch, MAP_V)
        do_desc = packed._map_ptr(maps, batch, n_batch, MAP_DO)
        if lo < hi:
            tma_tensormap_acquire(q_desc)
            tma_tensormap_acquire(k_desc)
            tma_tensormap_acquire(v_desc)
            tma_tensormap_acquire(do_desc)
            packed._wait(bars.k_empty, kv_state.phase)
            bars.k_full.arrive(n_bytes=CFG.OPERAND_BYTES, pred=nvvm.elect_sync())
            tma_load_tile(
                packed._operand(slab, CFG.K, dtype), tma_slice_runtime_desc(k_desc, 0, block * CFG.M, head // ratio, 0), bars.k_full.smem_ptr, acquire=False
            )
            packed._wait(bars.v_empty, kv_state.phase)
            bars.v_full.arrive(n_bytes=CFG.OPERAND_BYTES, pred=nvvm.elect_sync())
            tma_load_tile(
                packed._operand(slab, CFG.V, dtype), tma_slice_runtime_desc(v_desc, 0, block * CFG.M, head // ratio, 0), bars.v_full.smem_ptr, acquire=False
            )
            kv_state = advance(kv_state, 1)
        for qtile in cutlass.range(lo, hi, 1, unroll=1):
            packed._wait(bars.q_empty[q_state.idx], q_state.phase)
            bars.q_full[q_state.idx].arrive(n_bytes=CFG.OPERAND_BYTES, pred=nvvm.elect_sync())
            tma_load_tile(
                packed._operand(slab, CFG.Q + q_state.idx * CFG.OPERAND_BYTES, dtype),
                tma_slice_runtime_desc(q_desc, 0, qtile * CFG.N, head, 0),
                bars.q_full[q_state.idx].smem_ptr,
                acquire=False,
            )
            packed._wait(bars.do_empty, do_state.phase)
            for half in cutlass.range_constexpr(2):
                bars.do_full[half].arrive(n_bytes=CFG.SLICE_BYTES, pred=nvvm.elect_sync())
            for half in cutlass.range_constexpr(2):
                tma_load_subtiles(
                    packed._operand(slab, CFG.DO, dtype),
                    tma_slice_runtime_desc(do_desc, 0, qtile * CFG.N, head, 0),
                    bars.do_full[half].smem_ptr,
                    half,
                    1,
                    acquire=False,
                )
            q_state = advance(q_state, CFG.Q_STAGES)
            do_state = advance(do_state, 1)
        block, head, batch, valid, sched_state = packed._next_tile(sched, bars, sched_state)


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
):
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    x, y, z = cute.arch.block_idx()
    nk = skv // CFG.M
    if cutlass.const_expr(causal):
        block, head, batch = dense._decode_flat(x, nheads, nk, swizzle, batch_heads)
    else:
        block, head, batch = (x, y, z)
    slab = cutlass.Array(cutlass.Uint8, CFG.SMEM_BYTES, alignment=1024, space=cutlass.AddressSpace.smem)
    bars = dense._bars(slab)
    tmem_slot = dense._view(slab, CFG.TMEM_HANDLE, cutlass.Int32, 1, alignment=4)
    sched_words = dense._view(slab, CFG.SCHED_PAYLOAD, cutlass.Int32, 16, alignment=16)
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
        dense._softmax_loop(slab, bars, sched, tmem_base, lse, delta, block, head, batch, sq, nk, nheads, batch_heads, swizzle, scale, causal, dtype)
    elif warp < 12:
        dense._epilogue_loop(slab, bars, sched, tmem_base, block, head, batch, sq, nk, nheads, batch_heads, swizzle, causal, dtype)
    elif warp == 12:
        dense._mma_loop(slab, bars, sched, tmem_base, block, head, batch, sq, nk, nheads, batch_heads, swizzle, causal, dtype)
    elif warp == 13:
        _load_dense(slab, bars, sched, q_desc, k_desc, v_desc, do_desc, block, head, batch, sq, nk, nheads, batch_heads, swizzle, causal, dtype, ratio)
    elif warp == 14:
        dense._store_loop(slab, bars, sched, dq_desc, dk_desc, dv_desc, block, head, batch, sq, nk, nheads, batch_heads, swizzle, causal, dtype)
    else:
        dense._scheduler_loop(bars, sched)
    nvvm.barrier_cta_sync()
    if warp == 0:
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
        nvvm.tcgen05_dealloc(nvvm.make_tmem_ptr(tmem_slot.load(), cutlass.Int8), cutlass.Int32(CFG.TMEM_COLS), group=nvvm.CTAGroup.CTA_1)


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
):
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    x, y, z = cute.arch.block_idx()
    block, head, batch = (x, y, z)
    slab = cutlass.Array(cutlass.Uint8, CFG.SMEM_BYTES, alignment=1024, space=cutlass.AddressSpace.smem)
    bars = packed._bars(slab)
    tmem_slot = packed._view(slab, CFG.TMEM_HANDLE, cutlass.Int32, 1, alignment=4)
    sched_words = packed._view(slab, CFG.SCHED_PAYLOAD, cutlass.Int32, 16, alignment=16)
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
        packed._softmax_loop(slab, bars, sched, tmem_base, lse, delta, origins, stats_origin, block, head, batch, meta, n_batch, scale, causal, dtype)
    elif warp < 12:
        packed._epilogue_loop(slab, bars, sched, tmem_base, block, head, batch, meta, n_batch, causal, dtype)
    elif warp == 12:
        packed._mma_loop(slab, bars, sched, tmem_base, block, head, batch, meta, n_batch, causal, dtype)
    elif warp == 13:
        _load_thd(slab, bars, sched, maps, block, head, batch, meta, n_batch, causal, dtype, ratio)
    elif warp == 14:
        packed._store_loop(slab, bars, sched, maps, block, head, batch, meta, n_batch, causal, dtype)
    else:
        packed._scheduler_loop(bars, sched)
    nvvm.barrier_cta_sync()
    if warp == 0:
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
        nvvm.tcgen05_dealloc(nvvm.make_tmem_ptr(tmem_slot.load(), cutlass.Int8), cutlass.Int32(CFG.TMEM_COLS), group=nvvm.CTAGroup.CTA_1)


bprop_d128_accum_grouped_thd_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def main_dense_host(q, k, v, do, lse, delta, dq_accum, dk_part, dv_part, scale, causal: cutlass.Constexpr[bool], stream):
    """Qualified dense geometry, with the partial output head axis equal to Hq."""
    if cutlass.const_expr(q.element_type not in (cutlass.BFloat16, cutlass.Float16)):
        raise ValueError("grouped d128 accumulation requires BF16/FP16")
    if cutlass.const_expr(q.shape[3] != CFG.D or k.shape[3] != CFG.D or q.shape[2] <= k.shape[2] or q.shape[2] % k.shape[2]):
        raise ValueError("grouped d128 accumulation requires integral Hq/Hkv > 1 and d128")
    if cutlass.const_expr(q.shape[1] <= 0 or k.shape[1] <= 0 or q.shape[1] % CFG.N or k.shape[1] % CFG.M):
        raise ValueError("dense grouped d128 accumulation requires positive Sq/Skv multiples of 128")
    nk = k.shape[1] // CFG.M
    heads_per_l2 = (50 * 1024 * 1024) // (q.shape[1] * (CFG.D + CFG.D) * CFG.BPE)
    swizzle = 1 << (heads_per_l2.bit_length() - 1) if heads_per_l2 > 0 else 1
    grid = (nk * q.shape[2] * q.shape[0], 1, 1) if causal else (nk, q.shape[2], q.shape[0])
    bprop_d128_accum_grouped_dense_kernel(
        dense._tensor_map(q, CFG.HEAD_BOX),
        dense._tensor_map(k, CFG.HEAD_BOX),
        dense._tensor_map(v, CFG.HEAD_BOX),
        dense._tensor_map(do, CFG.HEAD_BOX),
        dense._tensor_map(dq_accum, CFG.DQ_BOX),
        dense._tensor_map(dk_part, CFG.HEAD_BOX),
        dense._tensor_map(dv_part, CFG.HEAD_BOX),
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
    ).launch(grid=grid, block=(CFG.THREADS, 1, 1), stream=stream, min_blocks_per_mp=1)


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
):
    """One owner per Q head and declared KV-envelope tile; no host length read."""
    nk = cute.math.max((max_skv + CFG.M - 1) // CFG.M, cutlass.Int32(1))
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
    ).launch(grid=(nk, nheads, n_batch), block=(CFG.THREADS, 1, 1), stream=stream, min_blocks_per_mp=1)
