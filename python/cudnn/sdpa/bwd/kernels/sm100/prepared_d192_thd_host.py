# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pointer entry for native packed d192 backward with runtime capacities."""

from typing import Optional

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver

from cudnn.frost.compiled_cache import compile_cached
from cudnn.sdpa.bwd.d192_thd_contract import ABI_VERSION, MAPS_PER_SEQUENCE, ORIGIN_PORTS, workspace_offsets
from .bprop_d192_accum_thd_helpers import host_thd


@cute.jit
def _io(ptr, tokens, heads: cutlass.Constexpr[int], dim: cutlass.Constexpr[int]):
    return cute.make_tensor(ptr, cute.make_layout((1, tokens, heads, dim), stride=(tokens * heads * dim, heads * dim, dim, 1)))


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
    stream: driver.CUstream,
):
    batch, heads, token_major = config
    q, k, v = _io(q_ptr, t_q_phys, heads, 192), _io(k_ptr, t_kv_phys, heads, 192), _io(v_ptr, t_kv_phys, heads, 128)
    o, do = _io(o_ptr, t_q_phys, heads, 128), _io(do_ptr, t_q_phys, heads, 128)
    dq, dk, dv = _io(dq_ptr, t_q_phys, heads, 192), _io(dk_ptr, t_kv_phys, heads, 192), _io(dv_ptr, t_kv_phys, heads, 128)
    if cutlass.const_expr(token_major):
        stats = cute.make_tensor(stats_ptr, cute.make_layout((t_q_phys, heads), stride=(heads, 1)))
    else:
        stats = cute.make_tensor(stats_ptr, cute.make_layout((1, heads, stats_head_stride), stride=(heads * stats_head_stride, stats_head_stride, 1)))
    payload = (workspace + cutlass.Int64(255)) & cutlass.Int64(-256)
    rounded, delta_off, meta_off, origins_off, maps_off, _ = workspace_offsets(batch, heads, t_q)
    dq_accum = _scratch(payload, cutlass.Int64(0), cutlass.Float32, (1, rounded, heads, 192), (rounded * heads * 192, heads * 192, 192, 1))
    delta = _scratch(payload, delta_off, cutlass.Float32, (1, heads, rounded), (heads * rounded, rounded, 1))
    meta = _scratch(payload, meta_off, cutlass.Int32, (4 * batch + 4,), (1,))
    origins = _scratch(payload, origins_off, cutlass.Int64, (len(ORIGIN_PORTS), batch), (batch, 1))
    maps = _scratch(payload, maps_off, cutlass.Int64, (MAPS_PER_SEQUENCE * batch * 16,), (1,))
    q_lens = cute.make_tensor(q_lens_ptr, cute.make_layout((batch + 1,), stride=(1,)))
    kv_lens = cute.make_tensor(kv_lens_ptr, cute.make_layout((batch + 1,), stride=(1,)))
    ro_ptrs = (ro_q_ptr, ro_k_ptr, ro_v_ptr, ro_o_ptr, ro_do_ptr, ro_dq_ptr, ro_dk_ptr, ro_dv_ptr, ro_stats_ptr)
    ro = tuple(cute.make_tensor(ptr, cute.make_layout((batch + 1,), stride=(1,))) if cutlass.const_expr(ptr is not None) else None for ptr in ro_ptrs)
    host_thd(
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
    )


def compile_host(config, origin_specs, origin_dtypes, dtype, causal, sm):
    """Capacities and bounds are runtime scalars, never artifact key values."""
    key = ("sdpa_bwd_sm100_d192_thd", ABI_VERSION, config, origin_specs, origin_dtypes, str(dtype), bool(causal), sm)
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
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options=f"--enable-tvm-ffi --gpu-arch sm_{sm}a",
        cache_key=repr(key),
        symbol="frost_sdpa_bwd_sm100_d192_thd_prepared",
    )
