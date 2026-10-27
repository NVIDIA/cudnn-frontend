# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pointer entry for packed grouped d128 on the cta_group::2 datapath (compact native partials, nine per-sequence maps)."""

from typing import Optional

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver

from cudnn.frost.compiled_cache import compile_cached
from ...d128_grouped_2cta_contract import ABI_VERSION, MAPS_PER_SEQUENCE, thd_workspace_offsets
from ...d128_grouped_contract import dkv_accum_fp32, resolve_dkv_accum, validate_fp32_accum_fits
from ...d128_thd_contract import ORIGIN_PORTS
from ...config_sm100_d128_accum_2cta import (
    diag_2cta_code,
    reg_split_code,
    resolve_reg_split,
    resolve_store_wait_form,
    resolve_tma_load_form,
    s_lookahead_after_dk,
    store_wait_spin,
    tma_load_plain,
)
from ...config_sm100_d128_accum_2cta_stail import CFG as STAIL_CFG
from ...config_sm100_d128_accum_2cta_stail import (
    DBG_CLK,
    dbg_clk_region_offset,
    dq_after_dv_for_map,
    dq_route_pair_code,
    mask_form_code,
    resolve_debug_clk_for_map,
    resolve_diag_2cta_for_map,
    resolve_dq_route_pair_for_map,
    resolve_mask_form_for_map,
    resolve_pair_mma_order_for_map,
    resolve_tmem_map,
    resolve_warp_roles_for_map,
    tmem_map_s_tail,
    validate_fields_for_roles,
    warp_roles_code,
)
from cudnn.frost.tile_dsl.requirements import red_global_v4_requirement_error
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
    lookahead_after_dk: cutlass.Constexpr[bool],
    after_dv: cutlass.Constexpr[bool],
    dq_route: cutlass.Constexpr[int],
    plain_tma: cutlass.Constexpr[bool],
    diag_code: cutlass.Constexpr[int],
    wait_spin: cutlass.Constexpr[bool],
    regs_code: cutlass.Constexpr[int],
    roles_code: cutlass.Constexpr[int],
    s_tail_map: cutlass.Constexpr[bool],
    mask_code: cutlass.Constexpr[int],
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
    rounded, rounded_k, delta_off, meta_off, origins_off, maps_off, dk_off, dv_off, regions_end = thd_workspace_offsets(batch, heads, t_q, t_kv)
    dq_accum = _scratch(payload, cutlass.Int64(0), cutlass.Float32, (1, rounded, heads, 128), (rounded * heads * 128, heads * 128, 128, 1))
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
        n_words = n_pairs * (STAIL_CFG.CGA_M * heads * batch * DBG_CLK.WORDS)
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
        lookahead_after_dk,
        after_dv,
        dq_route,
        plain_tma,
        diag_code,
        wait_spin,
        regs_code,
        roles_code,
        s_tail_map,
        mask_code,
        dkv_fp32,
        dk_acc,
        dv_acc,
        debug_clk,
        dbg_clk,
        stream,
    )


def compile_host(
    config,
    origin_specs,
    origin_dtypes,
    dtype,
    causal,
    sm,
    pair_mma_order=None,
    tma_load_form=None,
    diag_2cta=None,
    store_wait_form=None,
    reg_split=None,
    tmem_map=None,
    dq_route_pair=None,
    debug_clk=None,
    mask_form=None,
    dkv_accum=None,
    warp_roles=None,
):
    """Capacities and bounds are runtime scalars, never artifact key values; the 2-CTA chain is keyed apart from the 1-CTA one, per
    MMA issue order (``pair_mma_order``: a ``CFG.PAIR_MMA_ORDERS`` name, ``None`` = ``CFG.PAIR_MMA_ORDER_DEFAULT``; typed) and per
    cga2 TMA load form (``tma_load_form``: a ``CFG.TMA_LOAD_FORMS`` name, ``None`` = ``CFG.TMA_LOAD_FORM_DEFAULT``; typed) and per
    DIAGNOSTIC probe (``diag_2cta``: a ``CFG.DIAG_2CTA_NAMES`` name or ``None`` = no probe = the shipped body; output WRONG BY DESIGN
    when set, keyed apart so the probe never serves a real plan), and per perf-only knob: the store warp's wait form (``store_wait_form``:
    a ``CFG.STORE_WAIT_FORMS`` name, ``None`` = the default) and the register split (``reg_split``: a ``CFG.REG_SPLIT_NAMES`` name, ``None``
    = the default), and per pair BODY (``tmem_map``: a ``TMEM_MAPS`` name, ``None`` = "dp_tail" = the first body, byte-identical rendering;
    "s_tail" = ``bprop_d128_accum_grouped_2cta_stail_f16``, ARM (ii); the order and the probe domains are resolved PER MAP: ``fa4`` / ``fa4b`` only on
    s_tail, D7 declined there), and per dQ ROUTE (``dq_route_pair``: a ``CFG.DQ_ROUTES_PAIR`` name of the s_tail body or ``None`` = "tma" = the
    reduce-add path; "red_global" / "red_global_pair" = the register-direct red.global.add forms, s_tail only -- the s_tail kernel header's
    section 11; the part floor of the vector red is declined by name here, and D2-pair under a red route too: there is no reduce-add to skip).
    Every threaded Constexpr is in the key (RULE-GAPS 203): the order NAME keys ``fa4b`` apart, the route NAME keys the red forms apart, the
    mask-form NAME keys ``predicated`` / ``diag_mask_off`` apart (``mask_form``: a ``CFG.MASK_FORMS`` name of the s_tail body or ``None`` = "fsel"
    = today's rendering; "predicated" = the interior-tile mask skip of lever L5; "diag_mask_off" = its DIAGNOSTIC ceiling, output wrong by
    design, s_tail only -- the s_tail kernel header's section 11), the ``dkv_accum`` form NAME keys the fp32 chain apart (``dkv_accum``: a
    ``DKV_ACCUM_FORMS`` name or ``None`` = ``bf16_partials``; ``fp32_reduce`` is served by the s_tail body only -- declined BY NAME on dp_tail),
    the ROLE SHAPE name keys ``16w_fa4`` apart (``warp_roles``: a ``CFG.WARP_ROLES`` name of the s_tail body or ``None`` = "16w" = today's
    rendering; "16w_fa4" = X-FA4, the s_tail kernel header's sections 4 / 5 / 11, s_tail only -- under it a red dQ route, ``ds_gates_off``,
    ``debug_clk`` and ``dkv_accum = "fp32_reduce"`` decline by name here, before any trace).
    ``debug_clk`` (the s_tail body's STAMPS lever: a bool, ``None`` = False = the shipped rendering; True on s_tail only) is keyed too: the
    armed build carries the dump tensor and renders different PTX, never served to a plan that did not ask for it."""
    tmap = resolve_tmem_map(tmem_map)
    form_mask = resolve_mask_form_for_map(mask_form, tmap)
    order = resolve_pair_mma_order_for_map(pair_mma_order, tmap)
    form = resolve_tma_load_form(tma_load_form)
    diag = resolve_diag_2cta_for_map(diag_2cta, tmap)
    route = resolve_dq_route_pair_for_map(dq_route_pair, tmap)
    clk = resolve_debug_clk_for_map(debug_clk, tmap)
    accum_form = resolve_dkv_accum(dkv_accum)
    roles = validate_fields_for_roles(  # X-FA4: the role shape and the bundle it serves (the stack fold adds debug_clk / dkv_accum)
        resolve_warp_roles_for_map(warp_roles, tmap), route, diag, debug_clk=clk, dkv_accum=accum_form
    )
    if route != "tma":
        if diag == "dq_reduce_add_off":
            raise ValueError(
                f"diag_2cta 'dq_reduce_add_off' (D2-pair) is not defined under dq_route_pair {route!r}: the red route issues no reduce-add to skip"
            )
        if diag == "sdq_sts_off":
            raise ValueError(f"diag_2cta 'sdq_sts_off' (D9) is not defined under dq_route_pair {route!r}: the red route writes no sDQ slice to skip")
        message = red_global_v4_requirement_error((sm // 10, sm % 10))
        if message is not None:
            raise NotImplementedError(f"dq_route_pair {route!r}: {message}")
    wait_form = resolve_store_wait_form(store_wait_form)
    split = resolve_reg_split(reg_split)
    s_tail = tmem_map_s_tail(tmap)
    fp32 = dkv_accum_fp32(accum_form)
    if fp32 and not s_tail:
        raise ValueError(f"dkv_accum 'fp32_reduce' is served by the s_tail pair body only (tmem_map {tmap!r} has no fp32 dK / dV store path)")
    if fp32:
        validate_fp32_accum_fits(config[1], config[2])
    lookahead_after_dk = False if s_tail else s_lookahead_after_dk(order)  # the first body's order Constexpr; inert on the s_tail body
    after_dv = dq_after_dv_for_map(order, tmap)  # the s_tail body's order Constexpr (fa4b = dQ(t-1) issued after dV(t)); False under every first-body order
    key = (
        "sdpa_bwd_sm100_d128_grouped_thd_2cta",
        ABI_VERSION,
        config,
        origin_specs,
        origin_dtypes,
        str(dtype),
        bool(causal),
        sm,
        order,
        form,
        diag,
        wait_form,
        split,
        tmap,
        route,
        form_mask,
        accum_form,
        roles,
        clk,
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
        lookahead_after_dk,
        after_dv,
        dq_route_pair_code(route, tmap),
        tma_load_plain(form),
        diag_2cta_code(diag),
        store_wait_spin(wait_form),
        reg_split_code(split),
        warp_roles_code(roles, tmap),
        s_tail,
        mask_form_code(form_mask, tmap),
        fp32,
        clk,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options=f"--enable-tvm-ffi --gpu-arch sm_{sm}a",
        cache_key=repr(key),
        symbol="frost_sdpa_bwd_sm100_d128_grouped_thd_2cta_prepared",
    )
