# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Grouped THD d128 chain on the cta_group::2 datapath: eleven per-sequence maps, the pair main kernel, the shared folds.

Five launches, as the grouped THD chain: setup (THIS module's eleven-map variant: the nine operand / output maps + the two stats rows)
-> delta_zero -> MAIN (the 2-CTA pair body) -> cast_dq -> the fp32 dK / dV fold.  Everything but the setup kernel, the delta launch's
``stats_form = "tma_ring"`` variant and the main launch is imported from the shipping modules unchanged.  FOUR launches under the op-level
field ``dkv_accum = "fp32_reduce"`` (``dkv_fp32``; the s_tail body only): the per-KV-head fp32 dK / dV accumulators take the MAP_DK / MAP_DV
slots as 128-row fp32 [M x DQ_BOX]-box maps, delta_zero zero-fills them, the main kernel reduce-adds, cast_dq converts them once at each
output's own origin, and the fold is not launched.
Under ``stats_form = "tma_ring"`` (the s_tail body's header section 11; FAv4's LSElog2 + bulk-copy form) the delta launch is THIS module's
``delta_zero_stats_thd_d128_2cta_kernel``: the same dot tree and zero fills, plus the ldg softmax's two stat multiplies performed once per q row
and stored PRE-SCALED into the chain's q-tile-PADDED ``stats2`` region (fp32 [heads][PAD_TOTAL / N][2][N]: -log2(e) * LSE with the -inf -> 0
guard, then -scale * delta; every pad row [S_b, ceil(S_b / N) * N) written as 0.0 = the ldg form's `stat = 0` for a row past the sequence), and the
setup kernel writes the padded row offsets ``rowoff[b]`` the main kernel's loader addresses its ONE 1 KiB bulk copy per q tile with; under ``ldg``
the shipping delta kernel runs, no row offsets are written and the rendering is byte-identical.  The nine tensor maps are the same under both forms.
"""

from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass.experimental import primitives as nvvm
from cutlass.experimental.cuda import tensor_map as tmap

from cudnn.frost.tile_dsl.thd import (
    THD_SETUP_THREADS,
    emit_seq_descs,
    thd_port_origin,
    write_thd_meta,
    write_thd_port_origins,
    write_thd_prefix_warp,
    write_thd_row_offsets,
)
from ..bprop_thd_reduce import dkv_reduce_thd_origins_host
from .bprop_d128_accum_grouped_2cta_stail_f16 import (
    LOG2E,
    MAP_DK,
    MAP_DO1,
    MAP_DO2,
    MAP_DQ,
    MAP_DV,
    MAP_K,
    MAP_Q1,
    MAP_Q2,
    MAP_V,
    STATS_FORM_LDG,
    STATS_FORM_TMA_RING,
    main_thd_2cta_stail_host,
)
from . import bprop_d128_accum_grouped_2cta_stail_f16 as _stail
from .bprop_d128_accum_thd_helpers import _origin_row, _zero_group_vector, cast_dq_thd_d128_kernel, delta_zero_thd_d128_kernel
from .bprop_d128_helpers import ELEMENTS_PER_LANE, LANES_PER_ROW, RN, ROWS_PER_CTA, THREADS
from ...config_sm100_d128_accum_2cta import CFG
from ...d128_grouped_2cta_contract import STATS_KINDS

# The pair body's map order is the setup kernel's (the slots below are written in the body's MAP_* order); its box rows are the base geometry's.
assert (MAP_Q1, MAP_K, MAP_V, MAP_DO1, MAP_DQ, MAP_DK, MAP_DV, MAP_Q2, MAP_DO2) == tuple(range(CFG.MAPS_PER_SEQUENCE))
assert _stail.CFG.MAP_BOX_ROWS == CFG.MAP_BOX_ROWS and _stail.CFG.MAPS_PER_SEQUENCE == CFG.MAPS_PER_SEQUENCE == 9
assert (STATS_FORM_LDG, STATS_FORM_TMA_RING) == (0, 1) and _stail.CFG.STATS_KINDS == STATS_KINDS == 2 and _stail.CFG.STATS_PAD_GRAN == CFG.N
STATS_PAD_GRAN = _stail.CFG.STATS_PAD_GRAN  # 128: every sequence's stat rows padded to a whole q tile (write_thd_row_offsets gran; config = truth)


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
    rowoff: Optional[cute.Tensor] = None,
):
    """Publish prefixes / origins, then the NINE sequence-local map arrays of the 2-CTA body (header section 10 of the kernel); under
    ``stats_form = "tma_ring"`` ALSO the padded row offsets ``rowoff[b] = sum_{i<b} ceil(S_i / N) * N`` (+ the total at [B]) of the chain's padded
    pre-scaled stats region (``None`` = dropped from the signature: the ldg rendering is byte-identical).

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
    if cutlass.const_expr(rowoff is not None):
        # stats_form tma_ring: the padded row offsets over the cu_q prefix (written above), ONE thread (a serial prefix over the batches; the main
        # kernel's loader reads rowoff[batch] once per item); gran = the q tile: a sequence's last tile stays inside its own padded block
        if tid == 0:
            write_thd_row_offsets(data, n_batch, cutlass.Int32(STATS_PAD_GRAN), dst=cutlass.make_array_view(rowoff), dst0=0)
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


@cute.kernel
def delta_zero_stats_thd_d128_2cta_kernel(
    o: cute.Tensor,
    do: cute.Tensor,
    stats2: cute.Tensor,
    rowoff: cute.Tensor,
    dq_accum: cute.Tensor,
    meta: cute.Tensor,
    origins: cute.Tensor,
    lse: cute.Tensor,
    scale: cutlass.Float32,
    n_batch: cutlass.Int32,
    origin_specs: cutlass.Constexpr,
    dk_acc: Optional[cute.Tensor] = None,
    dv_acc: Optional[cute.Tensor] = None,
):
    """The delta launch of ``stats_form = "tma_ring"`` (the s_tail body's header section 11; FAv4's LSElog2 preprocess form): ``delta_zero_thd_d128_kernel``'s
    eight rows per CTA, its fp32 dot tree and its zero fills (dq_accum; dk_acc / dv_acc under dkv_accum fp32_reduce), PLUS the two stat
    multiplies the ldg softmax's `_stats` performs per q row, done HERE once per row and stored PRE-SCALED into the q-tile-PADDED region ``stats2``
    (fp32 [heads][PAD_TOTAL / N][2][N]; the row's padded token = ``rowoff[batch] + row``): kind 0 = ``-log2(e) * LSE`` (0 when the LSE is -inf: a row
    with no keys; the Stats port's own token origin, either Stats layout), kind 1 = ``-scale * rowsum(dO * O)``.  The SAME fp32 operands and the SAME
    operations as `_stats` (the ldg form stores the rowsum raw and reads it back losslessly; `-x * y` lowers to the same negate + mul.rn.f32), so
    every stat word is BITWISE the ldg form's.  Every PAD row [S_b, ceil(S_b / N) * N) of the sequence is written 0.0 (= `_stats`' `stat = 0` for a
    row past the sequence); rows past the padded block are never written.  The grid covers ceil(max_sq / N) * N rows per (batch, head) so the
    longest sequence's pad rows are written too (``host_thd_2cta``: x = ceil(max_sq / N) * (N / ROWS_PER_CTA))."""
    block, batch, head = cute.arch.block_idx()
    tid = cute.arch.thread_idx()[0]
    row = cutlass.Int32(block) * ROWS_PER_CTA + cutlass.Int32(tid // LANES_PER_ROW)
    column = cutlass.Int64(tid % LANES_PER_ROW) * ELEMENTS_PER_LANE
    data = cutlass.make_array_view(meta)
    start = cutlass.Int32(data[n_batch + batch])
    length = cutlass.Int32(data[n_batch + batch + 1]) - start
    live = row < length
    in_pad = row < ((length + cutlass.Int32(STATS_PAD_GRAN - 1)) // cutlass.Int32(STATS_PAD_GRAN)) * cutlass.Int32(
        STATS_PAD_GRAN
    )  # the sequence's padded block
    do_values = cutlass.Vector.from_elements(tuple(do.element_type(0.0) for _ in range(ELEMENTS_PER_LANE)), do.element_type)
    o_values = cutlass.Vector.from_elements(tuple(o.element_type(0.0) for _ in range(ELEMENTS_PER_LANE)), o.element_type)
    if live:
        token = cutlass.Int64(start) + cutlass.Int64(row)
        dq_offset = token * cutlass.Int64(dq_accum.stride[1]) + cutlass.Int64(head) * cutlass.Int64(dq_accum.stride[2]) + column
        zero4 = cutlass.Vector.from_elements(tuple(cutlass.Float32(0.0) for _ in range(4)), cutlass.Float32)
        (dq_accum.iterator.raw_ptr() + dq_offset).store(zero4, alignment=16)
        (dq_accum.iterator.raw_ptr() + dq_offset + 4).store(zero4, alignment=16)
        o_token = thd_port_origin(origins, _origin_row(origin_specs, 3), batch, start) + cutlass.Int64(row)
        do_token = thd_port_origin(origins, _origin_row(origin_specs, 4), batch, start) + cutlass.Int64(row)
        o_offset = o_token * cutlass.Int64(o.stride[1]) + cutlass.Int64(head) * cutlass.Int64(o.stride[2]) + column
        do_offset = do_token * cutlass.Int64(do.stride[1]) + cutlass.Int64(head) * cutlass.Int64(do.stride[2]) + column
        do_values = (do.iterator.raw_ptr() + do_offset).load(count=ELEMENTS_PER_LANE, alignment=16)
        o_values = (o.iterator.raw_ptr() + o_offset).load(count=ELEMENTS_PER_LANE, alignment=16)
    pairs = []
    for j in cutlass.range_constexpr(ELEMENTS_PER_LANE // 2):
        high = cute.math.mul(do_values[2 * j + 1].to(cutlass.Float32), o_values[2 * j + 1].to(cutlass.Float32), rounding=RN, ftz=True)
        pairs.append(cute.math.fma(do_values[2 * j].to(cutlass.Float32), o_values[2 * j].to(cutlass.Float32), high, rounding=RN, ftz=True))
    dot = cute.math.add(pairs[0], pairs[1], rounding=RN, ftz=True)
    dot = cute.math.add(dot, pairs[2], rounding=RN, ftz=True)
    dot = cute.math.add(dot, pairs[3], rounding=RN, ftz=True)
    # XOR distances 8,4,2,1 never cross a 16-lane row. Inactive row lanes carry zeros but remain active for the full-warp shuffle mask.
    for step in cutlass.range_constexpr(4):
        peer = nvvm.shfl_sync(thread_mask=0xFFFFFFFF, val=dot, offset=1 << (3 - step), mask_and_clamp=0x1F, kind=nvvm.Shfl.BFLY)
        dot = cute.math.add(dot, peer, rounding=RN, ftz=True)
    dot = cute.math.mul(dot, cutlass.Float32(1.0), rounding=RN, ftz=True)
    if in_pad and tid % LANES_PER_ROW == 0:
        # RESOURCE-NOTE: prescale in the producer: -log2e*LSE (-inf->0), -scale*delta; 64 FMUL2/lane/tile saved
        stat_lse = cutlass.Float32(0.0)  # the pad row's words: today's `stat = 0` guard for a row past the sequence, bit for bit
        stat_delta = cutlass.Float32(0.0)
        if live:
            stat_delta = -scale * dot  # `_stats` group 1: -scale * delta, from the rowsum the ldg form stores raw and reads back
            lse_token = thd_port_origin(origins, _origin_row(origin_specs, 8), batch, start) + cutlass.Int64(row)  # the Stats port's own origin
            if cutlass.const_expr(len(lse.shape) == 2):
                val = cutlass.Float32(lse[lse_token, head])
            else:
                val = cutlass.Float32(lse[0, head, lse_token])
            stat_lse = -cutlass.Float32(LOG2E) * val  # `_stats` group 0: -log2(e) * LSE, 0 for an LSE of -inf
            if val == cutlass.Float32(float("-inf")):
                stat_lse = cutlass.Float32(0.0)
        padded = cutlass.Int32(cutlass.make_array_view(rowoff)[batch]) + row  # the row's padded token (the sequence's block is N-aligned)
        record = cutlass.Int64(head) * cutlass.Int64(stats2.stride[0]) + cutlass.Int64(padded // cutlass.Int32(STATS_PAD_GRAN)) * cutlass.Int64(
            stats2.stride[1]
        )
        word = record + cutlass.Int64(padded % cutlass.Int32(STATS_PAD_GRAN))
        (stats2.iterator.raw_ptr() + word).store(stat_lse)
        (stats2.iterator.raw_ptr() + word + cutlass.Int64(stats2.stride[2])).store(stat_delta)
    if cutlass.const_expr(dk_acc is not None):
        # dkv_accum = fp32_reduce: an exact-zero fp32 image for the main kernel's reduce-adds; kv rows past the sequence are never written.
        kv_length = cutlass.Int32(data[batch])
        kv_start = cutlass.Int32(data[2 * n_batch + 1 + batch])
        if head < dk_acc.shape[2] and row < kv_length:
            kv_token = cutlass.Int64(kv_start) + cutlass.Int64(row)
            _zero_group_vector(dk_acc, kv_token, head, column)
            _zero_group_vector(dv_acc, kv_token, head, column)


delta_zero_stats_thd_d128_2cta_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


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
    stats_form_code: cutlass.Constexpr[int],
    dkv_fp32: cutlass.Constexpr[bool],
    dk_acc,
    dv_acc,
    debug_clk: cutlass.Constexpr[bool],
    dbg_clk,
    stats2,
    rowoff,
    stream,
):
    """Five native launches: the nine-map setup, delta / dQ zero, the cta_group::2 pair MAIN, the dQ cast, the fp32 group fold (four under
    ``dkv_fp32``: the fold is gone and the cast launch converts dK / dV too).
    ``mask_form_code`` = the pair body's softmax mask form (its header section 11: 0 = fsel, the default; 1 = predicated = the interior-tile
    mask skip; 2 = diag_mask_off, OUTPUT WRONG BY DESIGN), ``diag_2cta_code`` = its DIAGNOSTIC probe (section 11; 0 = none), ``warp_roles_code``
    = its ROLE SHAPE (section 11: 0 = 16w, the default rendering; 1 = 16w_fa4, the softmax ships the dS exchange from its registers and warps
    8-11 are the reduce group), ``debug_clk`` / ``dbg_clk`` = its STAMPS lever (section 11) and the Int32 dump tensor the prepared host carved
    (None when off) -- all resolved by the prepared host from the adapter's typed fields and read by the MAIN launch only.
    ``stats_form_code`` = its STATS FORM (section 11: 0 = ldg, selectable; 1 = tma_ring, the default: the setup kernel writes the padded row offsets
    ``rowoff``, the delta launch is ``delta_zero_stats_thd_d128_2cta_kernel`` writing the padded pre-scaled stats region ``stats2`` (``delta`` is None:
    the compact delta region does not exist under the form), and the MAIN launch's loader bulk-copies one (head, q tile) record per tile)."""
    tma_ring = stats_form_code == STATS_FORM_TMA_RING
    if cutlass.const_expr(stats_form_code not in (STATS_FORM_LDG, STATS_FORM_TMA_RING)):
        raise ValueError(f"stats_form_code must be one of {(STATS_FORM_LDG, STATS_FORM_TMA_RING)}; got {stats_form_code!r}")
    if cutlass.const_expr(tma_ring != (stats2 is not None) or tma_ring != (rowoff is not None) or tma_ring == (delta is not None)):
        raise ValueError("stats_form tma_ring carries stats2 + rowoff and no compact delta; the ldg form carries delta and neither ring region")
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
        rowoff,
    ).launch(grid=(1, 1, 1), block=(THD_SETUP_THREADS, 1, 1), stream=stream)
    q_grid = cute.math.max((max_sq + ROWS_PER_CTA - 1) // ROWS_PER_CTA, cutlass.Int32(1))
    if cutlass.const_expr(tma_ring):
        # RESOURCE-NOTE: grid x = ceil(max_sq/128)*16 under tma_ring: pad rows [S_b, ceil128) written as 0.0
        q_grid = cute.math.max((max_sq + STATS_PAD_GRAN - 1) // STATS_PAD_GRAN, cutlass.Int32(1)) * (STATS_PAD_GRAN // ROWS_PER_CTA)
    if cutlass.const_expr(dkv_fp32):
        kv_grid = cute.math.max((max_skv + ROWS_PER_CTA - 1) // ROWS_PER_CTA, cutlass.Int32(1))
        rows_grid = cute.math.max(q_grid, kv_grid)
        if cutlass.const_expr(tma_ring):
            delta_zero_stats_thd_d128_2cta_kernel(
                o, do, stats2, rowoff, dq_accum, meta, origins, lse, scale, cutlass.Int32(batch), origin_specs, dk_acc, dv_acc
            ).launch(grid=(rows_grid, batch, q.shape[2]), block=(THREADS, 1, 1), stream=stream)
        else:
            delta_zero_thd_d128_kernel(o, do, delta, dq_accum, meta, origins, cutlass.Int32(batch), origin_specs, dk_acc, dv_acc).launch(
                grid=(rows_grid, batch, q.shape[2]), block=(THREADS, 1, 1), stream=stream
            )
    else:
        if cutlass.const_expr(tma_ring):
            delta_zero_stats_thd_d128_2cta_kernel(o, do, stats2, rowoff, dq_accum, meta, origins, lse, scale, cutlass.Int32(batch), origin_specs).launch(
                grid=(q_grid, batch, q.shape[2]), block=(THREADS, 1, 1), stream=stream
            )
        else:
            delta_zero_thd_d128_kernel(o, do, delta, dq_accum, meta, origins, cutlass.Int32(batch), origin_specs).launch(
                grid=(q_grid, batch, q.shape[2]), block=(THREADS, 1, 1), stream=stream
            )
    main_thd_2cta_stail_host(
        maps,
        rowoff if cutlass.const_expr(tma_ring) else lse,  # the stats slots by form (the body's header section 14): the padded row offsets | the LSE
        stats2 if cutlass.const_expr(tma_ring) else delta,  # the padded pre-scaled stats region | the compact delta
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
        stats_form_code,
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
