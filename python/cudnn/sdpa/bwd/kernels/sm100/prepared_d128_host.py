# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The pointer hosts of the SM100 d128 (and d64 recompute) backward family: one ``compile_host_<route>`` per route of ``bprop_d128_f16.py``.

| route | entry | traced host | chain it calls (``bprop_d128_f16``) | symbol |
|---|---|---|---|---|
| recompute | ``compile_host_recompute`` | ``host_recompute`` | ``host_recompute`` (as ``tensor_host``; d64 by template) | ``frost_sdpa_bwd_sm100_d128_prepared`` |
| accumulation | ``compile_host_accum`` | ``host_accum`` | delta_zero -> ``host_accum`` (as ``accumulation_host``) -> cast_dq | ``frost_sdpa_bwd_sm100_d128_accumulation_prepared`` |
| grouped dense | ``compile_host_grouped_dense`` | ``host_grouped_dense`` | ``host_grouped_dense`` (as ``grouped_dense_chain_host``) | ``frost_sdpa_bwd_sm100_d128_grouped_dense_prepared`` |
| THD | ``compile_host_thd`` | ``host_thd`` | ``host_thd`` (as ``thd_chain_host``) | ``frost_sdpa_bwd_sm100_d128_thd_prepared`` |
| grouped THD | ``compile_host_grouped_thd`` | ``host_grouped_thd`` | ``host_grouped_thd`` (as ``grouped_thd_chain_host``) | ``frost_sdpa_bwd_sm100_d128_grouped_thd_prepared`` |
| grouped 2-CTA | ``compile_host_grouped_2cta`` | ``host_grouped_2cta`` | ``host_thd_2cta`` (as ``grouped_2cta_chain_host``) | ``frost_sdpa_bwd_sm100_d128_grouped_{thd,dense}_2cta_prepared`` |

Every ``cache_key`` string and tvm-ffi ``symbol`` is unchanged (the compiled-plan cache keys and symbols are ABI).  The four contracts'
``ABI_VERSION`` / ``MAPS_PER_SEQUENCE`` / ``thd_workspace_offsets`` carry different values per route and are imported under route-qualified
names.  The module is template-loaded for the d64 flavours (``FROST_TEMPLATE_PARAMS``), so every import is absolute.
"""

from pathlib import Path
from typing import Optional

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver

from cudnn.frost.compiled_cache import compile_cached
from cudnn.frost.template_loader import load_template
from cudnn.sdpa.bwd.config_sm100_d128_accum_2cta import diag_2cta_code, resolve_diag_2cta
from cudnn.sdpa.bwd.config_sm100_d128_accum_2cta_stail import CFG as STAIL_CFG
from cudnn.sdpa.bwd.config_sm100_d128_accum_2cta_stail import (
    SMEM_MAP_CODES,
    STATS_FORM_CODES,
    TMEM_MAP_CODES,
    dbg_clk_for_stats_form,
    dbg_clk_region_offset,
    mask_form_code,
    resolve_debug_clk,
    resolve_mask_form,
    resolve_smem_map,
    resolve_stats_form,
    resolve_tmem_map,
    resolve_warp_roles,
    smem_map_code,
    stats_form_code,
    tmem_map_code,
    validate_fields_for_roles,
    warp_roles_code,
)
from cudnn.sdpa.bwd.d128_contract import ABI_VERSION as DENSE_ABI_VERSION
from cudnn.sdpa.bwd.d128_grouped_2cta_contract import ABI_VERSION as GROUPED_2CTA_ABI_VERSION
from cudnn.sdpa.bwd.d128_grouped_2cta_contract import MAPS_PER_SEQUENCE as GROUPED_2CTA_MAPS_PER_SEQUENCE
from cudnn.sdpa.bwd.d128_grouped_2cta_contract import (
    DENSE_ORIGIN_DTYPES,
    DENSE_ORIGIN_SPECS,
    LAYOUT_FORM_CODES,
    layout_form_code,
    resolve_layout_form,
    stats2_pad_total,
    stats2_shape_strides,
)
from cudnn.sdpa.bwd.d128_grouped_2cta_contract import thd_workspace_offsets as grouped_2cta_thd_workspace_offsets
from cudnn.sdpa.bwd.d128_grouped_contract import ABI_VERSION as GROUPED_ABI_VERSION
from cudnn.sdpa.bwd.d128_grouped_contract import dkv_accum_fp32, resolve_dkv_accum, validate_fp32_accum_fits
from cudnn.sdpa.bwd.d128_grouped_contract import thd_workspace_offsets as grouped_thd_workspace_offsets
from cudnn.sdpa.bwd.d128_thd_contract import ABI_VERSION as THD_ABI_VERSION
from cudnn.sdpa.bwd.d128_thd_contract import MAPS_PER_SEQUENCE as THD_MAPS_PER_SEQUENCE
from cudnn.sdpa.bwd.d128_thd_contract import ORIGIN_PORTS, workspace_offsets
from cudnn.sdpa.bwd.kernels.sm100.bprop_d128_f16 import cast_dq_host, delta_zero_host
from cudnn.sdpa.bwd.kernels.sm100.bprop_d128_f16 import host_accum as accumulation_host
from cudnn.sdpa.bwd.kernels.sm100.bprop_d128_f16 import host_grouped_dense as grouped_dense_chain_host
from cudnn.sdpa.bwd.kernels.sm100.bprop_d128_f16 import host_grouped_thd as grouped_thd_chain_host
from cudnn.sdpa.bwd.kernels.sm100.bprop_d128_f16 import host_recompute as tensor_host
from cudnn.sdpa.bwd.kernels.sm100.bprop_d128_f16 import host_thd as thd_chain_host
from cudnn.sdpa.bwd.kernels.sm100.bprop_d128_f16 import host_thd_2cta as grouped_2cta_chain_host

# ======================================================================================================================
# route recompute (was kernels/sm100/prepared_d128_host.py)
# ======================================================================================================================
"""Pointer entry for the single-launch dense d128 backward kernel."""

HEAD_DIM = globals().get("FROST_TEMPLATE_PARAMS", 128)


@cute.jit
def host_recompute(
    q_ptr: cute.Pointer,
    k_ptr: cute.Pointer,
    v_ptr: cute.Pointer,
    o_ptr: cute.Pointer,
    do_ptr: cute.Pointer,
    lse_ptr: cute.Pointer,
    dq_ptr: cute.Pointer,
    dk_ptr: cute.Pointer,
    dv_ptr: cute.Pointer,
    workspace: cutlass.Int64,
    scale: cutlass.Float32,
    geometry: cutlass.Constexpr,
    causal: cutlass.Constexpr[bool],
    stream: driver.CUstream,
    window_left: cutlass.Constexpr[int] = 0,
):
    # The shared backward binder carries a workspace address even for a plan
    # requiring zero bytes. This unused scalar accepts a null address.
    q = cute.make_tensor(q_ptr, cute.make_layout(geometry[0][0], stride=geometry[0][1]))
    k = cute.make_tensor(k_ptr, cute.make_layout(geometry[1][0], stride=geometry[1][1]))
    v = cute.make_tensor(v_ptr, cute.make_layout(geometry[2][0], stride=geometry[2][1]))
    o = cute.make_tensor(o_ptr, cute.make_layout(geometry[3][0], stride=geometry[3][1]))
    do = cute.make_tensor(do_ptr, cute.make_layout(geometry[4][0], stride=geometry[4][1]))
    lse = cute.make_tensor(lse_ptr, cute.make_layout(geometry[5][0], stride=geometry[5][1]))
    dq = cute.make_tensor(dq_ptr, cute.make_layout(geometry[6][0], stride=geometry[6][1]))
    dk = cute.make_tensor(dk_ptr, cute.make_layout(geometry[7][0], stride=geometry[7][1]))
    dv = cute.make_tensor(dv_ptr, cute.make_layout(geometry[8][0], stride=geometry[8][1]))
    tensor_host(q, k, v, o, do, lse, dq, dk, dv, scale, causal, stream, window_left)


def compile_host_recompute(geometry, dtype, causal, sm, window_left=0):
    """Compile only plan-time geometry; bind storage and scale at execution."""
    global tensor_host
    head_dim = geometry[0][0][3]
    if head_dim == 64 and HEAD_DIM != 64:
        module = load_template(__file__, 64, tag="sdpa_bwd_d64_host")
        return module.compile_host_recompute(geometry, dtype, causal, sm, window_left)
    if head_dim != HEAD_DIM:
        raise ValueError("prepared backward head dimension does not match the kernel specialization")
    if HEAD_DIM == 64:
        # Load after this specialized module is initialized: load_template
        # holds a non-reentrant lock while executing module-level code.
        # Each host module is fixed to one width, so this assignment is invariant.
        tensor_host = load_template(str(Path(__file__).with_name("bprop_d128_f16.py")), 64, tag="sdpa_bwd_d64").host_recompute
    types = [dtype] * 5 + [cutlass.Float32] + [dtype] * 3
    args = [cute.runtime.make_ptr(t, 16, cute.AddressSpace.gmem, assumed_align=4 if i == 5 else 16) for i, t in enumerate(types)]
    return compile_cached(
        host_recompute,
        *args,
        cutlass.Int64(0),
        cutlass.Float32(1),
        geometry,
        bool(causal),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        int(window_left),
        options=f"--enable-tvm-ffi --gpu-arch sm_{sm}a",
        cache_key=repr(("sdpa_bwd_sm100_d128", geometry, str(dtype), bool(causal), sm, int(window_left), HEAD_DIM)),
        symbol="frost_sdpa_bwd_sm100_d128_prepared",
    )


# ======================================================================================================================
# route accum (was kernels/sm100/prepared_d128_accum_host.py)
# ======================================================================================================================
"""One pointer entry for dense d128 delta, accumulation and conversion."""


@cute.jit
def _view(ptr: cute.Pointer, geometry: cutlass.Constexpr):
    return cute.make_tensor(ptr, cute.make_layout(geometry[0], stride=geometry[1]))


@cute.jit
def _scratch_fp32_region(payload: cutlass.Int64, region: cutlass.Constexpr):
    offset, shape, strides = region
    pointer = cute.make_ptr(cutlass.Float32, payload + cutlass.Int64(offset), cute.AddressSpace.gmem, assumed_align=256)
    return _view(pointer, (shape, strides))


@cute.jit
def host_accum(
    q_ptr: cute.Pointer,
    k_ptr: cute.Pointer,
    v_ptr: cute.Pointer,
    o_ptr: cute.Pointer,
    do_ptr: cute.Pointer,
    lse_ptr: cute.Pointer,
    dq_ptr: cute.Pointer,
    dk_ptr: cute.Pointer,
    dv_ptr: cute.Pointer,
    workspace: cutlass.Int64,
    scale: cutlass.Float32,
    geometry: cutlass.Constexpr,
    regions: cutlass.Constexpr,
    causal: cutlass.Constexpr[bool],
    stream: driver.CUstream,
    window: cutlass.Constexpr[int] = 0,
):
    q = _view(q_ptr, geometry[0])
    k = _view(k_ptr, geometry[1])
    v = _view(v_ptr, geometry[2])
    o = _view(o_ptr, geometry[3])
    do = _view(do_ptr, geometry[4])
    lse = _view(lse_ptr, geometry[5])
    dq = _view(dq_ptr, geometry[6])
    dk = _view(dk_ptr, geometry[7])
    dv = _view(dv_ptr, geometry[8])
    payload = (workspace + cutlass.Int64(255)) & cutlass.Int64(-256)
    dq_accum = _scratch_fp32_region(payload, regions[0])
    delta = _scratch_fp32_region(payload, regions[1])
    delta_zero_host(o, do, delta, dq_accum, stream)
    accumulation_host(q, k, v, do, lse, delta, dq_accum, dk, dv, scale, causal, stream, window)
    cast_dq_host(dq_accum, dq, q.element_type, stream)


def compile_host_accum(geometry, regions, dtype, causal, sm, window=0):
    """Compile from declarations only; addresses and scale remain runtime slots."""
    global accumulation_host
    head_dim = geometry[0][0][3]
    if head_dim == 64 and HEAD_DIM != 64:
        module = load_template(__file__, 64, tag="sdpa_bwd_d64_accum_host")
        return module.compile_host_accum(geometry, regions, dtype, causal, sm, window)
    if head_dim != HEAD_DIM:
        raise ValueError("prepared accumulation head dimension does not match the kernel specialization")
    if HEAD_DIM == 64:
        accumulation_host = load_template(str(Path(__file__).with_name("bprop_d128_f16.py")), 64, tag="sdpa_bwd_d64").host_accum
    types = [dtype] * 5 + [cutlass.Float32] + [dtype] * 3
    args = [cute.runtime.make_ptr(t, 16, cute.AddressSpace.gmem, assumed_align=4 if i == 5 else 16) for i, t in enumerate(types)]
    return compile_cached(
        host_accum,
        *args,
        cutlass.Int64(0),
        cutlass.Float32(1),
        geometry,
        regions,
        bool(causal),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        int(window),
        options=f"--enable-tvm-ffi --gpu-arch sm_{sm}a",
        cache_key=repr(("sdpa_bwd_sm100_d128_accumulation", DENSE_ABI_VERSION, geometry, regions, str(dtype), bool(causal), sm, int(window), HEAD_DIM)),
        symbol="frost_sdpa_bwd_sm100_d128_accumulation_prepared",
    )


# ======================================================================================================================
# route grouped_dense (was kernels/sm100/prepared_d128_grouped_host.py)
# ======================================================================================================================
"""Pointer entry for dense grouped d128 with native dK/dV partials."""


@cute.jit
def _scratch_region(payload: cutlass.Int64, region: cutlass.Constexpr, dtype: cutlass.Constexpr):
    offset, shape, strides = region
    pointer = cute.make_ptr(dtype, payload + cutlass.Int64(offset), cute.AddressSpace.gmem, assumed_align=256)
    return _view(pointer, (shape, strides))


@cute.jit
def host_grouped_dense(
    q_ptr: cute.Pointer,
    k_ptr: cute.Pointer,
    v_ptr: cute.Pointer,
    o_ptr: cute.Pointer,
    do_ptr: cute.Pointer,
    lse_ptr: cute.Pointer,
    dq_ptr: cute.Pointer,
    dk_ptr: cute.Pointer,
    dv_ptr: cute.Pointer,
    workspace: cutlass.Int64,
    scale: cutlass.Float32,
    geometry: cutlass.Constexpr,
    regions: cutlass.Constexpr,
    causal: cutlass.Constexpr[bool],
    stream: driver.CUstream,
    diag_delete_dq_reduce_add: cutlass.Constexpr[bool] = False,
):
    q = _view(q_ptr, geometry[0])
    k = _view(k_ptr, geometry[1])
    v = _view(v_ptr, geometry[2])
    o = _view(o_ptr, geometry[3])
    do = _view(do_ptr, geometry[4])
    lse = _view(lse_ptr, geometry[5])
    dq = _view(dq_ptr, geometry[6])
    dk = _view(dk_ptr, geometry[7])
    dv = _view(dv_ptr, geometry[8])
    payload = (workspace + cutlass.Int64(255)) & cutlass.Int64(-256)
    dq_accum = _scratch_region(payload, regions[0], cutlass.Float32)
    delta = _scratch_region(payload, regions[1], cutlass.Float32)
    dk_part = _scratch_region(payload, regions[2], q.element_type)
    dv_part = _scratch_region(payload, regions[3], q.element_type)
    grouped_dense_chain_host(q, k, v, o, do, lse, dq, dk, dv, dq_accum, delta, dk_part, dv_part, scale, causal, stream, diag_delete_dq_reduce_add)


def compile_host_grouped_dense(geometry, regions, dtype, causal, sm, *, diag_delete_dq_reduce_add=False):
    """Compile from declarations only; addresses and scale remain runtime slots.

    ``diag_delete_dq_reduce_add`` (DIAGNOSTIC, default False) renders the shipped kernels under the unchanged cache key; the
    armed build (dQ stays zero by design) is keyed apart so it can never be served to a default plan."""
    key = ("sdpa_bwd_sm100_d128_grouped_dense", GROUPED_ABI_VERSION, geometry, regions, str(dtype), bool(causal), sm)
    if diag_delete_dq_reduce_add:
        key += ("diag_delete_dq_reduce_add",)
    types = [dtype] * 5 + [cutlass.Float32] + [dtype] * 3
    args = [cute.runtime.make_ptr(t, 16, cute.AddressSpace.gmem, assumed_align=4 if i == 5 else 16) for i, t in enumerate(types)]
    return compile_cached(
        host_grouped_dense,
        *args,
        cutlass.Int64(0),
        cutlass.Float32(1),
        geometry,
        regions,
        bool(causal),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        bool(diag_delete_dq_reduce_add),
        options=f"--enable-tvm-ffi --gpu-arch sm_{sm}a",
        cache_key=repr(key),
        symbol="frost_sdpa_bwd_sm100_d128_grouped_dense_prepared",
    )


# ======================================================================================================================
# route thd (was kernels/sm100/prepared_d128_thd_host.py)
# ======================================================================================================================
"""Pointer entry for native packed d128 backward with runtime capacities."""


@cute.jit
def _io(ptr, tokens, heads: cutlass.Constexpr[int]):
    return cute.make_tensor(ptr, cute.make_layout((1, tokens, heads, 128), stride=(tokens * heads * 128, heads * 128, 128, 1)))


@cute.jit
def _scratch(payload, offset, dtype: cutlass.Constexpr, shape, strides):
    ptr = cute.make_ptr(dtype, payload + offset, cute.AddressSpace.gmem, assumed_align=256)
    return cute.make_tensor(ptr, cute.make_layout(shape, stride=strides))


@cute.jit
def host_thd(
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
    stream: driver.CUstream,
    diag_delete_dq_reduce_add: cutlass.Constexpr[bool] = False,
):
    batch, heads, token_major = config
    q, k, v = _io(q_ptr, t_q_phys, heads), _io(k_ptr, t_kv_phys, heads), _io(v_ptr, t_kv_phys, heads)
    o, do = _io(o_ptr, t_q_phys, heads), _io(do_ptr, t_q_phys, heads)
    dq, dk, dv = _io(dq_ptr, t_q_phys, heads), _io(dk_ptr, t_kv_phys, heads), _io(dv_ptr, t_kv_phys, heads)
    if cutlass.const_expr(token_major):
        stats = cute.make_tensor(stats_ptr, cute.make_layout((t_q_phys, heads), stride=(heads, 1)))
    else:
        stats = cute.make_tensor(stats_ptr, cute.make_layout((1, heads, stats_head_stride), stride=(heads * stats_head_stride, stats_head_stride, 1)))
    payload = (workspace + cutlass.Int64(255)) & cutlass.Int64(-256)
    rounded, delta_off, meta_off, origins_off, maps_off, _ = workspace_offsets(batch, heads, t_q)
    dq_accum = _scratch(payload, cutlass.Int64(0), cutlass.Float32, (1, rounded, heads, 128), (rounded * heads * 128, heads * 128, 128, 1))
    delta = _scratch(payload, delta_off, cutlass.Float32, (1, heads, rounded), (heads * rounded, rounded, 1))
    meta = _scratch(payload, meta_off, cutlass.Int32, (4 * batch + 4,), (1,))
    origins = _scratch(payload, origins_off, cutlass.Int64, (len(ORIGIN_PORTS), batch), (batch, 1))
    maps = _scratch(payload, maps_off, cutlass.Int64, (THD_MAPS_PER_SEQUENCE * batch * 16,), (1,))
    q_lens = cute.make_tensor(q_lens_ptr, cute.make_layout((batch + 1,), stride=(1,)))
    kv_lens = cute.make_tensor(kv_lens_ptr, cute.make_layout((batch + 1,), stride=(1,)))
    ro_ptrs = (ro_q_ptr, ro_k_ptr, ro_v_ptr, ro_o_ptr, ro_do_ptr, ro_dq_ptr, ro_dk_ptr, ro_dv_ptr, ro_stats_ptr)
    ro = tuple(cute.make_tensor(ptr, cute.make_layout((batch + 1,), stride=(1,))) if cutlass.const_expr(ptr is not None) else None for ptr in ro_ptrs)
    thd_chain_host(
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
        stream,
        diag_delete_dq_reduce_add,
    )


def compile_host_thd(config, origin_specs, origin_dtypes, dtype, causal, sm, *, diag_delete_dq_reduce_add=False):
    """Capacities and bounds are runtime scalars, never artifact key values.

    ``diag_delete_dq_reduce_add`` (DIAGNOSTIC, default False) renders the shipped kernels under the unchanged cache key; the
    armed build (dQ stays zero by design) is keyed apart so it can never be served to a default plan."""
    key = ("sdpa_bwd_sm100_d128_thd", THD_ABI_VERSION, config, origin_specs, origin_dtypes, str(dtype), bool(causal), sm)
    if diag_delete_dq_reduce_add:
        key += ("diag_delete_dq_reduce_add",)
    types = [dtype] * 5 + [cutlass.Float32] + [dtype] * 3 + [cutlass.Int32] * 2
    args = [cute.runtime.make_ptr(t, 16, cute.AddressSpace.gmem, assumed_align=4 if i == 5 or i >= 9 else 16) for i, t in enumerate(types)]
    ro_types = {"int32": (cutlass.Int32, 4), "int64": (cutlass.Int64, 8)}
    args += [
        None if name is None else cute.runtime.make_ptr(ro_types[name][0], 16, cute.AddressSpace.gmem, assumed_align=ro_types[name][1])
        for name in origin_dtypes
    ]
    return compile_cached(
        host_thd,
        *([cutlass.Int64(0)] * 7),
        *args,
        cutlass.Int64(0),
        cutlass.Float32(1),
        cutlass.Int32(0),
        config,
        origin_specs,
        bool(causal),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        bool(diag_delete_dq_reduce_add),
        options=f"--enable-tvm-ffi --gpu-arch sm_{sm}a",
        cache_key=repr(key),
        symbol="frost_sdpa_bwd_sm100_d128_thd_prepared",
    )


# ======================================================================================================================
# route grouped_thd (was kernels/sm100/prepared_d128_grouped_thd_host.py)
# ======================================================================================================================
"""Pointer entry for packed grouped d128 with compact native partials."""


@cute.jit
def host_grouped_thd(
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
    stream: driver.CUstream,
    diag_delete_dq_reduce_add: cutlass.Constexpr[bool] = False,
    dkv_fp32: cutlass.Constexpr[bool] = False,
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
    rounded, rounded_k, delta_off, meta_off, origins_off, maps_off, dk_off, dv_off, _ = grouped_thd_workspace_offsets(batch, heads, t_q, t_kv)
    dq_accum = _scratch(payload, cutlass.Int64(0), cutlass.Float32, (1, rounded, heads, 128), (rounded * heads * 128, heads * 128, 128, 1))
    delta = _scratch(payload, delta_off, cutlass.Float32, (1, heads, rounded), (heads * rounded, rounded, 1))
    dk_part = _scratch(payload, dk_off, q.element_type, (1, rounded_k, heads, 128), (rounded_k * heads * 128, heads * 128, 128, 1))
    dv_part = _scratch(payload, dv_off, q.element_type, (1, rounded_k, heads, 128), (rounded_k * heads * 128, heads * 128, 128, 1))
    meta = _scratch(payload, meta_off, cutlass.Int32, (4 * batch + 4,), (1,))
    origins = _scratch(payload, origins_off, cutlass.Int64, (len(ORIGIN_PORTS), batch), (batch, 1))
    maps = _scratch(payload, maps_off, cutlass.Int64, (THD_MAPS_PER_SEQUENCE * batch * 16,), (1,))
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
    grouped_thd_chain_host(
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
        stream,
        diag_delete_dq_reduce_add,
        dkv_fp32,
        dk_acc,
        dv_acc,
    )


def compile_host_grouped_thd(config, origin_specs, origin_dtypes, dtype, causal, sm, *, diag_delete_dq_reduce_add=False, dkv_accum=None):
    """Capacities and bounds are runtime scalars, never artifact key values.

    ``diag_delete_dq_reduce_add`` (DIAGNOSTIC, default False) renders the shipped kernels under the unchanged cache key; the
    armed build (dQ stays zero by design) is keyed apart so it can never be served to a default plan.  ``dkv_accum`` (the op-level
    field: a ``DKV_ACCUM_FORMS`` name or ``None`` = ``bf16_partials``, the shipped chain under the unchanged key; typed, a ``ValueError``
    by name) keys the ``fp32_reduce`` chain apart (every threaded Constexpr is in the key)."""
    form = resolve_dkv_accum(dkv_accum)
    fp32 = dkv_accum_fp32(form)
    if fp32:
        validate_fp32_accum_fits(config[1], config[2])
    key = ("sdpa_bwd_sm100_d128_grouped_thd", GROUPED_ABI_VERSION, config, origin_specs, origin_dtypes, str(dtype), bool(causal), sm)
    if diag_delete_dq_reduce_add:
        key += ("diag_delete_dq_reduce_add",)
    if fp32:
        key += ("dkv_accum", form)
    types = [dtype] * 5 + [cutlass.Float32] + [dtype] * 3 + [cutlass.Int32] * 2
    args = [cute.runtime.make_ptr(t, 16, cute.AddressSpace.gmem, assumed_align=4 if i == 5 or i >= 9 else 16) for i, t in enumerate(types)]
    ro_types = {"int32": (cutlass.Int32, 4), "int64": (cutlass.Int64, 8)}
    args += [
        None if name is None else cute.runtime.make_ptr(ro_types[name][0], 16, cute.AddressSpace.gmem, assumed_align=ro_types[name][1])
        for name in origin_dtypes
    ]
    return compile_cached(
        host_grouped_thd,
        *([cutlass.Int64(0)] * 7),
        *args,
        cutlass.Int64(0),
        cutlass.Float32(1),
        cutlass.Int32(0),
        config,
        origin_specs,
        bool(causal),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        bool(diag_delete_dq_reduce_add),
        fp32,
        options=f"--enable-tvm-ffi --gpu-arch sm_{sm}a",
        cache_key=repr(key),
        symbol="frost_sdpa_bwd_sm100_d128_grouped_thd_prepared",
    )


# ======================================================================================================================
# route grouped_2cta (was kernels/sm100/prepared_d128_grouped_2cta_host.py)
# ======================================================================================================================
"""Pointer entry for grouped d128 on the cta_group::2 datapath (compact native partials, nine per-sequence maps): the packed THD chain and its
dense compact-BSHD twin (``layout`` = "thd" | "dense", by NAME in the key; the dense form binds no length / offset operands and the setup
kernel synthesizes the metadata from the declared extents -- ``d128_grouped_2cta_contract`` module docstring)."""

_SMEM_MAP_NAMES = {code: name for name, code in SMEM_MAP_CODES.items()}  # the body's smem_map_code -> the resolved map name


@cute.jit
def host_grouped_2cta(
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
    q_lens_ptr: Optional[cute.Pointer],
    kv_lens_ptr: Optional[cute.Pointer],
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
    layout_code: cutlass.Constexpr[int],
    tmem_code: cutlass.Constexpr[int],
    smem_code: cutlass.Constexpr[int],
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
    # the pair body's TMEM map (header section 9) and SMEM map (section 8), PART-KEYED: the workspace depends on neither, the STAMPS record on both
    dq576 = tmem_code == TMEM_MAP_CODES["dq_576"]
    # smem_code (SMEM_MAP_CODES, "rubin_b2" among them) does not enter the workspace: the SMEM map only changes the carveout (header section 8)
    rounded, rounded_k, delta_off, meta_off, origins_off, maps_off, dk_off, dv_off, regions_end, rowoff_off = grouped_2cta_thd_workspace_offsets(
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
    maps = _scratch(payload, maps_off, cutlass.Int64, (GROUPED_2CTA_MAPS_PER_SEQUENCE * batch * 16,), (1,))
    if cutlass.const_expr(dkv_fp32):
        # dkv_accum = fp32_reduce: the per-KV-head fp32 accumulators [1, rounded_k, Hkv, 128] carved INSIDE the bf16 per-Q-head partial regions
        # (4 x Hkv <= 2 x Hq bytes per kv row on every grouped ratio: validate_fp32_accum_fits); the advertised workspace is unchanged
        dk_acc = _scratch(payload, dk_off, cutlass.Float32, (1, rounded_k, kv_heads, 128), (rounded_k * kv_heads * 128, kv_heads * 128, 128, 1))
        dv_acc = _scratch(payload, dv_off, cutlass.Float32, (1, rounded_k, kv_heads, 128), (rounded_k * kv_heads * 128, kv_heads * 128, 128, 1))
    else:
        dk_acc, dv_acc = None, None
    dense = layout_code == LAYOUT_FORM_CODES["dense"]  # the layout form (trace-time): dense = no length / offset operands, the extents synthesize them
    if cutlass.const_expr(dense != (q_lens_ptr is None and kv_lens_ptr is None)):
        raise ValueError("the dense layout binds no length operands; the thd layout binds both")
    if cutlass.const_expr(dense):
        q_lens, kv_lens = None, None
        dense_sq, dense_skv = cutlass.Int32(max_sq), cutlass.Int32(max_skv)  # the declared S_q / S_kv: the uniform lengths of the dense twin
    else:
        q_lens = cute.make_tensor(q_lens_ptr, cute.make_layout((batch + 1,), stride=(1,)))
        kv_lens = cute.make_tensor(kv_lens_ptr, cute.make_layout((batch + 1,), stride=(1,)))
        dense_sq, dense_skv = None, None
    ro_ptrs = (ro_q_ptr, ro_k_ptr, ro_v_ptr, ro_o_ptr, ro_do_ptr, ro_dq_ptr, ro_dk_ptr, ro_dv_ptr, ro_stats_ptr)
    ro = tuple(cute.make_tensor(ptr, cute.make_layout((batch + 1,), stride=(1,))) if cutlass.const_expr(ptr is not None) else None for ptr in ro_ptrs)
    if cutlass.const_expr(debug_clk):
        # STAMPS (the s_tail body's header section 11): the Int32 record dump past the chain's regions -- one record per CTA of the MAIN grid
        # (config dbg_clk_records: ceil(max_skv / THD_UNIT_ROWS) pairs x CGA_M x heads x batch), at the same offset the adapter advertised
        # (config dbg_clk_region_offset of the regions' end); the kernel writes every record at exit, so the region needs no zeroing.
        n_pairs = cute.math.max((cutlass.Int32(max_skv) + STAIL_CFG.THD_UNIT_ROWS - 1) // STAIL_CFG.THD_UNIT_ROWS, cutlass.Int32(1))
        # the record by (form, map): 96 (ldg) / 88 (tma_ring) words on the s_tail map, 104 / 96 under dq_576 (the shorter bar block's spare), config = truth
        dbg_words = dbg_clk_for_stats_form(
            "tma_ring" if cutlass.const_expr(tma_ring) else "ldg",
            None,
            "dq_576" if cutlass.const_expr(dq576) else "s_tail",
            _SMEM_MAP_NAMES[smem_code],  # the STAMPS record per SMEM map (sm100_oversized = the sm100 records)
        ).WORDS
        n_words = n_pairs * (STAIL_CFG.CGA_M * heads * batch * dbg_words)
        dbg_clk = _scratch(payload, dbg_clk_region_offset(regions_end), cutlass.Int32, (n_words,), (1,))
    else:
        dbg_clk = None
    grouped_2cta_chain_host(
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
        tmem_code,
        smem_code,
        dkv_fp32,
        dk_acc,
        dv_acc,
        debug_clk,
        dbg_clk,
        stats2,
        rowoff,
        stream,
        layout_code,
        dense_sq,
        dense_skv,
    )


def compile_host_grouped_2cta(
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
    layout="thd",
    tmem_map=None,
    smem_map=None,
):
    """Capacities and bounds are runtime scalars, never artifact key values; the 2-CTA chain is keyed apart from the 1-CTA one and per typed
    field: ``layout`` (a ``LAYOUT_FORMS`` name: "thd" = the packed chain, the default; "dense" = the compact-BSHD twin: ``config[3]`` False,
    ``origin_specs`` == ``DENSE_ORIGIN_SPECS`` and ``origin_dtypes`` == ``DENSE_ORIGIN_DTYPES`` -- declined by name otherwise -- no length
    pointers in the signature, the setup kernel's dense arm; the THD rendering is byte-identical), ``diag_2cta`` (a ``CFG.DIAG_2CTA_NAMES`` name or ``None`` = no probe = the shipped body; output WRONG BY DESIGN when set, keyed apart
    so the probe never serves a real plan), ``mask_form`` (a ``CFG.MASK_FORMS`` name or ``None`` = "fsel" = the default rendering; "predicated"
    = the interior-tile mask skip; "diag_mask_off" = its DIAGNOSTIC ceiling, output wrong by design -- the kernel header's section 11),
    ``dkv_accum`` (a ``DKV_ACCUM_FORMS`` name or ``None`` = ``bf16_partials``; ``fp32_reduce`` keys the four-launch fp32 chain apart),
    ``warp_roles`` (a ``CFG.WARP_ROLES`` name or ``None`` = "16w" = the default rendering; "16w_fa4" = the alternate role shape of header
    sections 4 / 5 / 11, under which ``debug_clk`` and ``dkv_accum = "fp32_reduce"`` decline by name here, before any trace) and
    ``debug_clk`` (the STAMPS lever: a bool, ``None`` = False = the shipped rendering; the armed build carries the dump tensor and renders
    different PTX, never served to a plan that did not ask for it) and ``stats_form`` (a ``CFG.STATS_FORMS`` name or ``None`` = the role shape's
    form: "tma_ring" = the default = the loader-fed SMEM stats ring of header section 11 fed from the chain's padded pre-scaled stats region,
    bitwise the same stats; "ldg" = the previous default, the form "16w_fa4" resolves to; "tma_ring" NAMED declines by name under
    ``warp_roles = "16w_fa4"``; the ring's STAMPS record is the re-based spare) and ``tmem_map`` (a ``CFG.TMEM_MAPS`` name or ``None`` = the
    PART's default, PART-KEYED by ``sm`` = the device cc the artifact is keyed and compiled for: "s_tail" = the dQ image in S^T's dead tail, the
    every part's default; "dq_576" = the independent dQ image at [512, 576) of the 576-column carve, selectable by name on cc 10.7 under the
    default role shape (not a default: the body's header sections 11 / 13); "dq_576" NAMED on a 512-column part or under "16w_fa4" and the three S-tail
    WAIT probes under "dq_576" decline by name here, before any trace; the workspace does not depend on the map) and ``smem_map`` (a ``CFG.SMEM_MAPS``
    name or ``None`` = the PART's default = "sm100" everywhere: the opt-in slab chain; "rubin_b2" = the 327 KiB oversized carveout of cc 10.7 with the
    two-stage dS / exchange staging ring + the sQ1 / sDO1 rings and the dq_read_done row, selectable by name ON "dq_576" under the default role shape and
    stats form -- the body's header sections 5 / 8 / 11; NAMED on a 512-column part it is the engine's typed decline, NAMED under "s_tail", "16w_fa4"
    or "ldg" it declines by name here, before any trace; the workspace does not depend on it).  Every threaded Constexpr is in the key by NAME."""
    form_mask = resolve_mask_form(mask_form)
    diag = resolve_diag_2cta(diag_2cta)
    clk = resolve_debug_clk(debug_clk)
    accum_form = resolve_dkv_accum(dkv_accum)
    shape = resolve_warp_roles(warp_roles)
    form_stats = resolve_stats_form(stats_form, shape)  # unset: the form the role shape renders (tma_ring under 16w, ldg under 16w_fa4)
    cc = (sm // 10, sm % 10)  # the part the artifact is compiled for (--gpu-arch sm_{sm}a): the TMEM map is keyed by it
    form_map = resolve_tmem_map(tmem_map, cc, shape)  # unset: the part's map (s_tail everywhere; dq_576 selectable by name on cc 10.7)
    form_smem = resolve_smem_map(smem_map, cc, shape, form_stats, form_map)  # unset: sm100 everywhere; rubin_b2 selectable by name on cc 10.7 ON dq_576
    roles = validate_fields_for_roles(shape, diag, debug_clk=clk, dkv_accum=accum_form, stats_form=form_stats, tmem_map=form_map, smem_map=form_smem)
    fp32 = dkv_accum_fp32(accum_form)
    if fp32:
        validate_fp32_accum_fits(config[1], config[2])
    form_layout = resolve_layout_form(layout)
    dense = form_layout == "dense"
    if dense and (config[3] or tuple(origin_specs) != DENSE_ORIGIN_SPECS or tuple(origin_dtypes) != DENSE_ORIGIN_DTYPES):
        raise ValueError(
            f"layout 'dense' binds head-major Stats with the synthesized stats origin only (token_major False, origin_specs {DENSE_ORIGIN_SPECS}, "
            f"origin_dtypes {DENSE_ORIGIN_DTYPES}); got token_major {config[3]!r}, origin_specs {tuple(origin_specs)!r}, origin_dtypes {tuple(origin_dtypes)!r}"
        )
    key = (
        "sdpa_bwd_sm100_d128_grouped_thd_2cta",
        GROUPED_2CTA_ABI_VERSION,
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
        form_layout,
        form_map,
        form_smem,
    )
    types = [dtype] * 5 + [cutlass.Float32] + [dtype] * 3 + [cutlass.Int32] * 2
    args = [cute.runtime.make_ptr(t, 16, cute.AddressSpace.gmem, assumed_align=4 if i == 5 or i >= 9 else 16) for i, t in enumerate(types)]
    if dense:
        args[9] = args[10] = None  # no length operands: the setup kernel's dense arm synthesizes the prefixes from the declared extents
    ro_types = {"int32": (cutlass.Int32, 4), "int64": (cutlass.Int64, 8)}
    args += [
        None if name is None else cute.runtime.make_ptr(ro_types[name][0], 16, cute.AddressSpace.gmem, assumed_align=ro_types[name][1])
        for name in origin_dtypes
    ]
    return compile_cached(
        host_grouped_2cta,
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
        layout_form_code(form_layout),
        tmem_map_code(form_map, cc, roles),
        smem_map_code(form_smem, cc, roles, form_stats, form_map),
        fp32,
        clk,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options=f"--enable-tvm-ffi --gpu-arch sm_{sm}a",
        cache_key=repr(key),
        symbol=f"frost_sdpa_bwd_sm100_d128_grouped_{form_layout}_2cta_prepared",
    )
