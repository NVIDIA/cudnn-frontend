# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pointer entry for packed grouped d128 on the cta_group::2 datapath (compact native partials, nine per-sequence maps)."""

from typing import Optional

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver

from cudnn.frost.compiled_cache import compile_cached
from ...d128_grouped_2cta_contract import ABI_VERSION, MAPS_PER_SEQUENCE, stats2_pad_total, stats2_shape_strides, thd_workspace_offsets
from ...d128_grouped_contract import dkv_accum_fp32, resolve_dkv_accum, validate_fp32_accum_fits
from ...d128_thd_contract import ORIGIN_PORTS
from ...config_sm100_d128_accum_2cta import diag_2cta_code, resolve_diag_2cta
from ...config_sm100_d128_accum_2cta_stail import CFG as STAIL_CFG
from ...config_sm100_d128_accum_2cta_stail import (
    DBG_CLK,
    DBG_CLK_RING,
    dbg_clk_region_offset,
    STATS_FORM_CODES,
    mask_form_code,
    resolve_debug_clk,
    resolve_mask_form,
    resolve_stats_form,
    resolve_warp_roles,
    stats_form_code,
    validate_fields_for_roles,
    warp_roles_code,
)
from .bprop_d128_accum_grouped_2cta_helpers import host_thd_2cta


@cute.jit
def _io(ptr, tokens, heads: cutlass.Constexpr[int]):
    return cute.make_tensor(ptr, cute.make_layout((1, tokens, heads, 128), stride=(tokens * heads * 128, heads * 128, 128, 1)))


@cute.jit
def _scratch(payload, offset, dtype: cutlass.Constexpr, shape, strides):
    ptr = cute.make_ptr(dtype, payload + offset, cute.AddressSpace.gmem, assumed_align=256)
    return cute.make_tensor(ptr, cute.make_layout(shape, stride=strides))


@cute.jit
def host(
    t_q: cutlass.Int64,
    t_kv: cutlass.Int64,
    t_q_phys: cutlass.Int64,
    t_kv_phys: cutlass.Int64,
    max_sq: cutlass.Int64,
    max_skv: cutlass.Int64,
    stats_head_stride: cutlass.Int64,
    q_ptr: cute.Pointer,
    k_ptr: cute.Pointer,
    v_ptr: cute.Pointer,
    o_ptr: cute.Pointer,
    do_ptr: cute.Pointer,
    stats_ptr: cute.Pointer,
    dq_ptr: cute.Pointer,
    dk_ptr: cute.Pointer,
    dv_ptr: cute.Pointer,
    q_lens_ptr: cute.Pointer,
    kv_lens_ptr: cute.Pointer,
    ro_q_ptr: Optional[cute.Pointer],
    ro_k_ptr: Optional[cute.Pointer],
    ro_v_ptr: Optional[cute.Pointer],
    ro_o_ptr: Optional[cute.Pointer],
    ro_do_ptr: Optional[cute.Pointer],
    ro_dq_ptr: Optional[cute.Pointer],
    ro_dk_ptr: Optional[cute.Pointer],
    ro_dv_ptr: Optional[cute.Pointer],
    ro_stats_ptr: Optional[cute.Pointer],
    workspace: cutlass.Int64,
    scale: cutlass.Float32,
    lens_form: cutlass.Int32,
    config: cutlass.Constexpr,
    origin_specs: cutlass.Constexpr,
    causal: cutlass.Constexpr[bool],
    diag_code: cutlass.Constexpr[int],
    roles_code: cutlass.Constexpr[int],
    mask_code: cutlass.Constexpr[int],
    stats_code: cutlass.Constexpr[int],
    dkv_fp32: cutlass.Constexpr[bool],
    debug_clk: cutlass.Constexpr[bool],
    stream: driver.CUstream,
):
    batch, heads, kv_heads, token_major = config
    q, k, v = _io(q_ptr, t_q_phys, heads), _io(k_ptr, t_kv_phys, kv_heads), _io(v_ptr, t_kv_phys, kv_heads)
    o, do = _io(o_ptr, t_q_phys, heads), _io(do_ptr, t_q_phys, heads)
    dq, dk, dv = _io(dq_ptr, t_q_phys, heads), _io(dk_ptr, t_kv_phys, kv_heads), _io(dv_ptr, t_kv_phys, kv_heads)
    if cutlass.const_expr(token_major):
        stats = cute.make_tensor(stats_ptr, cute.make_layout((t_q_phys, heads), stride=(heads, 1)))
    else:
        stats = cute.make_tensor(stats_ptr, cute.make_layout((1, heads, stats_head_stride), stride=(heads * stats_head_stride, stats_head_stride, 1)))
    payload = (workspace + cutlass.Int64(255)) & cutlass.Int64(-256)
    tma_ring = stats_code == STATS_FORM_CODES["tma_ring"]  # the s_tail body's stats form (header section 11): the padded stats region replaces delta
    rounded, rounded_k, delta_off, meta_off, origins_off, maps_off, dk_off, dv_off, regions_end, rowoff_off = thd_workspace_offsets(
        batch, heads, t_q, t_kv, tma_ring
    )
    dq_accum = _scratch(payload, cutlass.Int64(0), cutlass.Float32, (1, rounded, heads, 128), (rounded * heads * 128, heads * 128, 128, 1))
    if cutlass.const_expr(tma_ring):
        # stats_form = tma_ring: the q-tile-padded pre-scaled stats region [heads][PAD_TOTAL / N][2][N] fp32 the delta launch writes and the main
        # kernel's loader bulk-copies per q tile, plus the Int32 padded row offsets [batch + 1] the setup kernel writes (the contract's regions)
        stats_shape, stats_strides = stats2_shape_strides(heads, stats2_pad_total(batch, rounded))
        stats2 = _scratch(payload, delta_off, cutlass.Float32, stats_shape, stats_strides)
        rowoff = _scratch(payload, rowoff_off, cutlass.Int32, (batch + 1,), (1,))
        delta = None
    else:
        stats2, rowoff = None, None
        delta = _scratch(payload, delta_off, cutlass.Float32, (1, heads, rounded), (heads * rounded, rounded, 1))
    dk_part = _scratch(payload, dk_off, q.element_type, (1, rounded_k, heads, 128), (rounded_k * heads * 128, heads * 128, 128, 1))
    dv_part = _scratch(payload, dv_off, q.element_type, (1, rounded_k, heads, 128), (rounded_k * heads * 128, heads * 128, 128, 1))
    meta = _scratch(payload, meta_off, cutlass.Int32, (4 * batch + 4,), (1,))
    origins = _scratch(payload, origins_off, cutlass.Int64, (len(ORIGIN_PORTS), batch), (batch, 1))
    maps = _scratch(payload, maps_off, cutlass.Int64, (MAPS_PER_SEQUENCE * batch * 16,), (1,))
    if cutlass.const_expr(dkv_fp32):
        # dkv_accum = fp32_reduce: the per-KV-head fp32 accumulators [1, rounded_k, Hkv, 128] carved INSIDE the bf16 per-Q-head partial regions
        # (4 x Hkv <= 2 x Hq bytes per kv row on every grouped ratio: validate_fp32_accum_fits); the advertised workspace is unchanged
        dk_acc = _scratch(payload, dk_off, cutlass.Float32, (1, rounded_k, kv_heads, 128), (rounded_k * kv_heads * 128, kv_heads * 128, 128, 1))
        dv_acc = _scratch(payload, dv_off, cutlass.Float32, (1, rounded_k, kv_heads, 128), (rounded_k * kv_heads * 128, kv_heads * 128, 128, 1))
    else:
        dk_acc, dv_acc = None, None
    q_lens = cute.make_tensor(q_lens_ptr, cute.make_layout((batch + 1,), stride=(1,)))
    kv_lens = cute.make_tensor(kv_lens_ptr, cute.make_layout((batch + 1,), stride=(1,)))
    ro_ptrs = (ro_q_ptr, ro_k_ptr, ro_v_ptr, ro_o_ptr, ro_do_ptr, ro_dq_ptr, ro_dk_ptr, ro_dv_ptr, ro_stats_ptr)
    ro = tuple(cute.make_tensor(ptr, cute.make_layout((batch + 1,), stride=(1,))) if cutlass.const_expr(ptr is not None) else None for ptr in ro_ptrs)
    if cutlass.const_expr(debug_clk):
        # STAMPS (the s_tail body's header section 11): the Int32 record dump past the chain's regions -- one record per CTA of the MAIN grid
        # (config dbg_clk_records: ceil(max_skv / THD_UNIT_ROWS) pairs x CGA_M x heads x batch), at the same offset the adapter advertised
        # (config dbg_clk_region_offset of the regions' end); the kernel writes every record at exit, so the region needs no zeroing.
        n_pairs = cute.math.max((cutlass.Int32(max_skv) + STAIL_CFG.THD_UNIT_ROWS - 1) // STAIL_CFG.THD_UNIT_ROWS, cutlass.Int32(1))
        dbg_words = (DBG_CLK_RING if cutlass.const_expr(tma_ring) else DBG_CLK).WORDS  # the record by form: 96 (ldg) / 88 (tma_ring), config = truth
        n_words = n_pairs * (STAIL_CFG.CGA_M * heads * batch * dbg_words)
        dbg_clk = _scratch(payload, dbg_clk_region_offset(regions_end), cutlass.Int32, (n_words,), (1,))
    else:
        dbg_clk = None
    host_thd_2cta(
        q,
        k,
        v,
        o,
        do,
        stats,
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
        origin_specs,
        lens_form,
        batch,
        cutlass.Int32(max_sq),
        cutlass.Int32(max_skv),
        scale,
        causal,
        diag_code,
        roles_code,
        mask_code,
        stats_code,
        dkv_fp32,
        dk_acc,
        dv_acc,
        debug_clk,
        dbg_clk,
        stats2,
        rowoff,
        stream,
    )


def compile_host(
    config,
    origin_specs,
    origin_dtypes,
    dtype,
    causal,
    sm,
    diag_2cta=None,
    debug_clk=None,
    mask_form=None,
    dkv_accum=None,
    warp_roles=None,
    stats_form=None,
):
    """Capacities and bounds are runtime scalars, never artifact key values; the 2-CTA chain is keyed apart from the 1-CTA one and per typed
    field: ``diag_2cta`` (a ``CFG.DIAG_2CTA_NAMES`` name or ``None`` = no probe = the shipped body; output WRONG BY DESIGN when set, keyed apart
    so the probe never serves a real plan), ``mask_form`` (a ``CFG.MASK_FORMS`` name or ``None`` = "fsel" = the default rendering; "predicated"
    = the interior-tile mask skip; "diag_mask_off" = its DIAGNOSTIC ceiling, output wrong by design -- the kernel header's section 11),
    ``dkv_accum`` (a ``DKV_ACCUM_FORMS`` name or ``None`` = ``bf16_partials``; ``fp32_reduce`` keys the four-launch fp32 chain apart),
    ``warp_roles`` (a ``CFG.WARP_ROLES`` name or ``None`` = "16w" = the default rendering; "16w_fa4" = the alternate role shape of header
    sections 4 / 5 / 11, under which ``debug_clk`` and ``dkv_accum = "fp32_reduce"`` decline by name here, before any trace) and
    ``debug_clk`` (the STAMPS lever: a bool, ``None`` = False = the shipped rendering; the armed build carries the dump tensor and renders
    different PTX, never served to a plan that did not ask for it) and ``stats_form`` (a ``CFG.STATS_FORMS`` name or ``None`` = the role shape's
    form: "tma_ring" = the default = the loader-fed SMEM stats ring of header section 11 fed from the chain's padded pre-scaled stats region,
    bitwise the same stats; "ldg" = the previous default, the form "16w_fa4" resolves to; "tma_ring" NAMED declines by name under
    ``warp_roles = "16w_fa4"``; the ring's STAMPS record is the re-based spare).  Every threaded
    Constexpr is in the key by NAME."""
    form_mask = resolve_mask_form(mask_form)
    diag = resolve_diag_2cta(diag_2cta)
    clk = resolve_debug_clk(debug_clk)
    accum_form = resolve_dkv_accum(dkv_accum)
    shape = resolve_warp_roles(warp_roles)
    form_stats = resolve_stats_form(stats_form, shape)  # unset: the form the role shape renders (tma_ring under 16w, ldg under 16w_fa4)
    roles = validate_fields_for_roles(shape, diag, debug_clk=clk, dkv_accum=accum_form, stats_form=form_stats)
    fp32 = dkv_accum_fp32(accum_form)
    if fp32:
        validate_fp32_accum_fits(config[1], config[2])
    key = (
        "sdpa_bwd_sm100_d128_grouped_thd_2cta",
        ABI_VERSION,
        config,
        origin_specs,
        origin_dtypes,
        str(dtype),
        bool(causal),
        sm,
        diag,
        form_mask,
        accum_form,
        roles,
        clk,
        form_stats,
    )
    types = [dtype] * 5 + [cutlass.Float32] + [dtype] * 3 + [cutlass.Int32] * 2
    args = [cute.runtime.make_ptr(t, 16, cute.AddressSpace.gmem, assumed_align=4 if i == 5 or i >= 9 else 16) for i, t in enumerate(types)]
    ro_types = {"int32": (cutlass.Int32, 4), "int64": (cutlass.Int64, 8)}
    args += [
        None if name is None else cute.runtime.make_ptr(ro_types[name][0], 16, cute.AddressSpace.gmem, assumed_align=ro_types[name][1])
        for name in origin_dtypes
    ]
    return compile_cached(
        host,
        *([cutlass.Int64(0)] * 7),
        *args,
        cutlass.Int64(0),
        cutlass.Float32(1),
        cutlass.Int32(0),
        config,
        origin_specs,
        bool(causal),
        diag_2cta_code(diag),
        warp_roles_code(roles),
        mask_form_code(form_mask),
        stats_form_code(form_stats),
        fp32,
        clk,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options=f"--enable-tvm-ffi --gpu-arch sm_{sm}a",
        cache_key=repr(key),
        symbol="frost_sdpa_bwd_sm100_d128_grouped_thd_2cta_prepared",
    )
