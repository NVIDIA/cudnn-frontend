# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Prepared pointer entry for d128_f16_accum."""

from pathlib import Path
from typing import Optional
import cutlass
import cutlass.cute as cute
from cuda.bindings import driver
from cudnn.frost.compiled_cache import compile_cached
from cudnn.frost.template_loader import load_template
from cudnn.sdpa.bwd.d128_contract import ABI_VERSION as DENSE_ABI_VERSION
from cudnn.sdpa.bwd.d128_grouped_contract import ABI_VERSION as GROUPED_ABI_VERSION
from cudnn.sdpa.bwd.d128_grouped_contract import dkv_accum_fp32, resolve_dkv_accum, validate_fp32_accum_fits
from cudnn.sdpa.bwd.d128_grouped_contract import thd_workspace_offsets as grouped_thd_workspace_offsets
from cudnn.sdpa.bwd.d128_thd_contract import ABI_VERSION as THD_ABI_VERSION
from cudnn.sdpa.bwd.d128_thd_contract import MAPS_PER_SEQUENCE as THD_MAPS_PER_SEQUENCE
from cudnn.sdpa.bwd.d128_thd_contract import ORIGIN_PORTS, workspace_offsets
from cudnn.sdpa.bwd.kernels.sm100._d128_chain import cast_dq_host, delta_zero_host
from cudnn.sdpa.bwd.kernels.sm100._prepared_d128_common import _view, _scratch_fp32_region
from cudnn.sdpa.bwd.kernels.sm100.d128_f16_accum.kernel import host_accum as accumulation_host
from cudnn.sdpa.bwd.kernels.sm100.d128_f16_accum.kernel import host_grouped_dense as grouped_dense_chain_host
from cudnn.sdpa.bwd.kernels.sm100.d128_f16_accum.kernel import host_grouped_thd as grouped_thd_chain_host
from cudnn.sdpa.bwd.kernels.sm100.d128_f16_accum.kernel import host_thd as thd_chain_host
from cudnn.sdpa.bwd.kernels.sm100._prepared_d128_common import _scratch_region, _io, _scratch

# ======================================================================================================================
# Shared head-width specialization for accumulation and the deterministic pair
# ======================================================================================================================
"""Recompute has its own pointer entry in d128_f16_recompute/prepared_host.py."""

_TEMPLATE = globals().get("FROST_TEMPLATE_PARAMS", 128)
HEAD_DIM = int(_TEMPLATE) if isinstance(_TEMPLATE, int) else int(_TEMPLATE[0])


# ======================================================================================================================
# route accum (was kernels/sm100/prepared_d128_accum_host.py)
# ======================================================================================================================
"""One pointer entry for dense d128 delta, accumulation and conversion."""


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
        accumulation_host = load_template(str(Path(__file__).with_name("kernel.py")), 64, tag="sdpa_bwd_d64").host_accum
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
