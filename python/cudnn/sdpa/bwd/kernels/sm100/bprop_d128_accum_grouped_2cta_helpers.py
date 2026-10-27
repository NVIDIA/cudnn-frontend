# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Grouped THD d128 chain on the cta_group::2 datapath: nine per-sequence maps, the pair main kernel, the shared folds.

Five launches, as the grouped THD chain: setup (THIS module's nine-map variant) -> delta_zero -> MAIN (the 2-CTA pair body)
-> cast_dq -> the fp32 dK / dV fold.  Everything but the setup kernel and the main launch is imported from the shipping
modules unchanged.  FOUR launches under the op-level field ``dkv_accum = "fp32_reduce"`` (``dkv_fp32``; the s_tail body only): the
per-KV-head fp32 dK / dV accumulators take the MAP_DK / MAP_DV slots as 128-row fp32 [M x DQ_BOX]-box maps, delta_zero zero-fills them,
the main kernel reduce-adds, cast_dq converts them once at each output's own origin, and the fold is not launched.
"""

from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass.experimental import primitives as nvvm
from cutlass.experimental.cuda import tensor_map as tmap

from cudnn.frost.tile_dsl.thd import THD_SETUP_THREADS, emit_seq_descs, write_thd_meta, write_thd_port_origins, write_thd_prefix_warp
from ..bprop_thd_reduce import dkv_reduce_thd_origins_host
from .bprop_d128_accum_grouped_2cta_stail_f16 import MAP_DK, MAP_DO1, MAP_DO2, MAP_DQ, MAP_DV, MAP_K, MAP_Q1, MAP_Q2, MAP_V, main_thd_2cta_stail_host
from . import bprop_d128_accum_grouped_2cta_stail_f16 as _stail
from .bprop_d128_accum_thd_helpers import _origin_row, cast_dq_thd_d128_kernel, delta_zero_thd_d128_kernel
from .bprop_d128_helpers import ROWS_PER_CTA, THREADS
from ...config_sm100_d128_accum_2cta import CFG

# The pair body's map order is the setup kernel's (the slots below are written in the body's MAP_* order); its box rows are the base geometry's.
assert (MAP_Q1, MAP_K, MAP_V, MAP_DO1, MAP_DQ, MAP_DK, MAP_DV, MAP_Q2, MAP_DO2) == tuple(range(CFG.MAPS_PER_SEQUENCE))
assert _stail.CFG.MAP_BOX_ROWS == CFG.MAP_BOX_ROWS and _stail.CFG.MAPS_PER_SEQUENCE == CFG.MAPS_PER_SEQUENCE


def _tensor_map_rows(tensor, rows, box):
    """Logical TMA coordinates (d, sequence, head, batch) with a ``rows``-row sequence box (the 2-CTA path needs 64- and 128-row boxes)."""
    return tmap.create_tensor_map_tiled_from_view(tensor, box_dims=(1, rows, 1, box), stride_order=(3, 1, 2, 0), swizzle=tmap.TensorMapSwizzle.s128b)


@cute.kernel
def setup_thd_d128_2cta_kernel(
    q1_desc: cutlass.GridConstant[tmap.TensorMap],
    k_desc: cutlass.GridConstant[tmap.TensorMap],
    v_desc: cutlass.GridConstant[tmap.TensorMap],
    do1_desc: cutlass.GridConstant[tmap.TensorMap],
    dq_desc: cutlass.GridConstant[tmap.TensorMap],
    dk_desc: cutlass.GridConstant[tmap.TensorMap],
    dv_desc: cutlass.GridConstant[tmap.TensorMap],
    q2_desc: cutlass.GridConstant[tmap.TensorMap],
    do2_desc: cutlass.GridConstant[tmap.TensorMap],
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    do: cute.Tensor,
    dq_accum: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    meta: cute.Tensor,
    origins: cute.Tensor,
    maps: cute.Tensor,
    q_lens: cute.Tensor,
    kv_lens: cute.Tensor,
    ro_q: Optional[cute.Tensor],
    ro_k: Optional[cute.Tensor],
    ro_v: Optional[cute.Tensor],
    ro_o: Optional[cute.Tensor],
    ro_do: Optional[cute.Tensor],
    ro_dq: Optional[cute.Tensor],
    ro_dk: Optional[cute.Tensor],
    ro_dv: Optional[cute.Tensor],
    ro_stats: Optional[cute.Tensor],
    origin_specs: cutlass.Constexpr,
    lens_form: cutlass.Int32,
    n_batch: cutlass.Int32,
):
    """Publish prefixes / origins, then the NINE sequence-local map arrays of the 2-CTA body (header section 10 of the kernel).

    The form of ``setup_thd_d128_kernel`` (grouped partials: compact KV origins for dK / dV, every external origin row retained
    for the final fold); the per-sequence descriptor order is ``MAP_Q1, MAP_K, MAP_V, MAP_DO1, MAP_DQ, MAP_DK, MAP_DV, MAP_Q2,
    MAP_DO2`` with the box rows of ``CFG.MAP_BOX_ROWS``.  Every map writer releases to the tensor-map proxy; the consuming
    main kernel acquires before use.
    """
    tid = cutlass.Int32(cute.arch.thread_idx()[0])
    warp, lane = tid // 32, tid % 32
    data = cutlass.make_array_view(meta)
    if n_batch <= 1:
        if tid == 0:
            write_thd_meta(data, cutlass.make_array_view(q_lens), cutlass.make_array_view(kv_lens), lens_form, n_batch)
    else:
        if warp == 0:
            write_thd_prefix_warp(data, cutlass.make_array_view(q_lens), n_batch, n_batch, (lens_form & 1) != 0, lane, store_lengths=False)
        if warp == 1:
            write_thd_prefix_warp(data, cutlass.make_array_view(kv_lens), n_batch, 2 * n_batch + 1, (lens_form & 2) != 0, lane, store_lengths=True)
    offsets = (ro_q, ro_k, ro_v, ro_o, ro_do, ro_dq, ro_dk, ro_dv, ro_stats)
    for port in cutlass.range_constexpr(9):
        if cutlass.const_expr(origin_specs[port] is not None):
            spec = origin_specs[port]
            write_thd_port_origins(origins, offsets[port], spec[0], spec[1], spec[2], n_batch, tid, cutlass.Int32(THD_SETUP_THREADS))
    cute.arch.barrier()
    if nvvm.elect_sync():
        cu_k = 2 * n_batch + 1
        batch_step = cutlass.Int32(THD_SETUP_THREADS // 32)
        # (base map, slot, cu offset, tensor, origin row): the q ports clip at cu_q, the kv ports at cu_k; the partials keep compact origins.
        emit_seq_descs(
            q1_desc,
            maps,
            data,
            n_batch,
            q,
            n_batch,
            cutlass.Int64(q.stride[1]),
            1,
            slot_base=MAP_Q1 * n_batch,
            first_batch=warp,
            batch_step=batch_step,
            origins=origins,
            origin_row=_origin_row(origin_specs, 0),
        )
        emit_seq_descs(
            k_desc,
            maps,
            data,
            cu_k,
            k,
            n_batch,
            cutlass.Int64(k.stride[1]),
            1,
            slot_base=MAP_K * n_batch,
            first_batch=warp,
            batch_step=batch_step,
            origins=origins,
            origin_row=_origin_row(origin_specs, 1),
        )
        emit_seq_descs(
            v_desc,
            maps,
            data,
            cu_k,
            v,
            n_batch,
            cutlass.Int64(v.stride[1]),
            1,
            slot_base=MAP_V * n_batch,
            first_batch=warp,
            batch_step=batch_step,
            origins=origins,
            origin_row=_origin_row(origin_specs, 2),
        )
        emit_seq_descs(
            do1_desc,
            maps,
            data,
            n_batch,
            do,
            n_batch,
            cutlass.Int64(do.stride[1]),
            1,
            slot_base=MAP_DO1 * n_batch,
            first_batch=warp,
            batch_step=batch_step,
            origins=origins,
            origin_row=_origin_row(origin_specs, 4),
        )
        emit_seq_descs(
            dq_desc,
            maps,
            data,
            n_batch,
            dq_accum,
            n_batch,
            cutlass.Int64(dq_accum.stride[1]),
            1,
            slot_base=MAP_DQ * n_batch,
            first_batch=warp,
            batch_step=batch_step,
        )
        emit_seq_descs(
            dk_desc,
            maps,
            data,
            cu_k,
            dk,
            n_batch,
            cutlass.Int64(dk.stride[1]),
            1,
            slot_base=MAP_DK * n_batch,
            first_batch=warp,
            batch_step=batch_step,
            origins=origins,
            origin_row=-1,
        )
        emit_seq_descs(
            dv_desc,
            maps,
            data,
            cu_k,
            dv,
            n_batch,
            cutlass.Int64(dv.stride[1]),
            1,
            slot_base=MAP_DV * n_batch,
            first_batch=warp,
            batch_step=batch_step,
            origins=origins,
            origin_row=-1,
        )
        emit_seq_descs(
            q2_desc,
            maps,
            data,
            n_batch,
            q,
            n_batch,
            cutlass.Int64(q.stride[1]),
            1,
            slot_base=MAP_Q2 * n_batch,
            first_batch=warp,
            batch_step=batch_step,
            origins=origins,
            origin_row=_origin_row(origin_specs, 0),
        )
        emit_seq_descs(
            do2_desc,
            maps,
            data,
            n_batch,
            do,
            n_batch,
            cutlass.Int64(do.stride[1]),
            1,
            slot_base=MAP_DO2 * n_batch,
            first_batch=warp,
            batch_step=batch_step,
            origins=origins,
            origin_row=_origin_row(origin_specs, 4),
        )
        nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)


setup_thd_d128_2cta_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def host_thd_2cta(
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
    diag_2cta_code: cutlass.Constexpr[int],
    warp_roles_code: cutlass.Constexpr[int],
    mask_form_code: cutlass.Constexpr[int],
    dkv_fp32: cutlass.Constexpr[bool],
    dk_acc,
    dv_acc,
    debug_clk: cutlass.Constexpr[bool],
    dbg_clk,
    stream,
):
    """Five native launches: the nine-map setup, delta / dQ zero, the cta_group::2 pair MAIN, the dQ cast, the fp32 group fold (four under
    ``dkv_fp32``: the fold is gone and the cast launch converts dK / dV too).
    ``mask_form_code`` = the pair body's softmax mask form (its header section 11: 0 = fsel, the default; 1 = predicated = the interior-tile
    mask skip; 2 = diag_mask_off, OUTPUT WRONG BY DESIGN), ``diag_2cta_code`` = its DIAGNOSTIC probe (section 11; 0 = none), ``warp_roles_code``
    = its ROLE SHAPE (section 11: 0 = 16w, the default rendering; 1 = 16w_fa4, the softmax ships the dS exchange from its registers and warps
    8-11 are the reduce group), ``debug_clk`` / ``dbg_clk`` = its STAMPS lever (section 11) and the Int32 dump tensor the prepared host carved
    (None when off) -- all resolved by the prepared host from the adapter's typed fields and read by the MAIN launch only."""
    rows = CFG.MAP_BOX_ROWS
    if cutlass.const_expr(dkv_fp32):
        dk_map, dv_map = _tensor_map_rows(dk_acc, rows[MAP_DK], CFG.DQ_BOX), _tensor_map_rows(dv_acc, rows[MAP_DV], CFG.DQ_BOX)
        dk_ws, dv_ws = dk_acc, dv_acc
    else:
        dk_map, dv_map = _tensor_map_rows(dk_part, rows[MAP_DK], CFG.HEAD_BOX), _tensor_map_rows(dv_part, rows[MAP_DV], CFG.HEAD_BOX)
        dk_ws, dv_ws = dk_part, dv_part
    setup_thd_d128_2cta_kernel(
        _tensor_map_rows(q, rows[MAP_Q1], CFG.HEAD_BOX),
        _tensor_map_rows(k, rows[MAP_K], CFG.HEAD_BOX),
        _tensor_map_rows(v, rows[MAP_V], CFG.HEAD_BOX),
        _tensor_map_rows(do, rows[MAP_DO1], CFG.HEAD_BOX),
        _tensor_map_rows(dq_accum, rows[MAP_DQ], CFG.DQ_BOX),
        dk_map,
        dv_map,
        _tensor_map_rows(q, rows[MAP_Q2], CFG.HEAD_BOX),
        _tensor_map_rows(do, rows[MAP_DO2], CFG.HEAD_BOX),
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
    ).launch(grid=(1, 1, 1), block=(THD_SETUP_THREADS, 1, 1), stream=stream)
    q_grid = cute.math.max((max_sq + ROWS_PER_CTA - 1) // ROWS_PER_CTA, cutlass.Int32(1))
    if cutlass.const_expr(dkv_fp32):
        kv_grid = cute.math.max((max_skv + ROWS_PER_CTA - 1) // ROWS_PER_CTA, cutlass.Int32(1))
        rows_grid = cute.math.max(q_grid, kv_grid)
        delta_zero_thd_d128_kernel(o, do, delta, dq_accum, meta, origins, cutlass.Int32(batch), origin_specs, dk_acc, dv_acc).launch(
            grid=(rows_grid, batch, q.shape[2]), block=(THREADS, 1, 1), stream=stream
        )
    else:
        delta_zero_thd_d128_kernel(o, do, delta, dq_accum, meta, origins, cutlass.Int32(batch), origin_specs).launch(
            grid=(q_grid, batch, q.shape[2]), block=(THREADS, 1, 1), stream=stream
        )
    main_thd_2cta_stail_host(
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
        diag_2cta_code,
        warp_roles_code,
        mask_form_code,
        dkv_fp32,
        debug_clk,
        dbg_clk,
        stream,
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
            cutlass.Int32(batch),
            max_skv,
            _origin_row(origin_specs, 6),
            _origin_row(origin_specs, 7),
            CFG.D,
            q.shape[2] // k.shape[2],
            q.element_type,
            stream,
        )
