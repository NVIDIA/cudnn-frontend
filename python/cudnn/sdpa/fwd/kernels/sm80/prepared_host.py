# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pointer-only dense and packed hosts shared by the SM80 forward flavors."""

from functools import lru_cache
from typing import Optional

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver

from cudnn.frost.compiled_cache import compile_cached, positional_entry, template_key
from cudnn.frost.tile_dsl.mask import MASK_CAUSAL, MASK_NONE, MASK_SWA
from .packed_init import FROST_SOURCE_DIGEST as _INIT_DIGEST, zero_outputs


@cute.jit
def _view(ptr: Optional[cute.Pointer], geometry: cutlass.Constexpr):
    if cutlass.const_expr(ptr is None):
        return None
    shape, strides = geometry
    # Keep native outer strides wide before device address multiplication.
    return cute.make_tensor(ptr, cute.make_layout(shape, stride=tuple(cutlass.Int64(st) if i < len(strides) - 1 else st for i, st in enumerate(strides))))


@cute.jit
def _host(
    q: cute.Pointer,
    k: cute.Pointer,
    v: cute.Pointer,
    o: cute.Pointer,
    stats: Optional[cute.Pointer],
    seq_kv: Optional[cute.Pointer],
    seq_q: Optional[cute.Pointer],
    sink: Optional[cute.Pointer],
    bias: Optional[cute.Pointer],
    scale_log2: cutlass.Float32,
    inv_scale: cutlass.Float32,
    module: cutlass.Constexpr,
    geometry: cutlass.Constexpr,
    dtype: cutlass.Constexpr,
    swa_window: cutlass.Constexpr[int],
    right_bound: cutlass.Constexpr[int],
    stream: driver.CUstream,
    paged_k: Optional[cute.Pointer] = None,
    paged_v: Optional[cute.Pointer] = None,
    paged_geometry: cutlass.Constexpr = None,
    max_skv: cutlass.Constexpr = 0,
):
    # Reuse the template configuration without a redundant dataclass argument;
    # compiled_cache can then export and reload the pointer-only call ABI.
    params = module.PARAMS
    sq, d = geometry[0][0][1], geometry[0][0][3]
    skv = max_skv if cutlass.const_expr(params.paged_page_size > 0) else geometry[1][0][1]
    mask = (MASK_CAUSAL if params.is_causal else MASK_NONE) | (MASK_SWA if params.has_swa else 0)
    module._sdpa_host(
        _view(q, geometry[0]),
        _view(k, geometry[1]),
        _view(v, geometry[2]),
        _view(o, geometry[3]),
        _view(stats, geometry[4]),
        _view(seq_kv, geometry[5]),
        _view(seq_q, geometry[6]),
        _view(sink, geometry[7]),
        _view(bias, geometry[8]),
        None,
        None,
        None,
        params.tile_m,
        params.num_warps,
        params.tile_n,
        params.d_qk,
        params.d_v,
        dtype,
        sq % params.tile_m == 0 and skv % params.tile_n == 0,
        d == params.d_qk,
        mask,
        swa_window,
        params.causal_bottom_right,
        params.has_seq_kv_lens,
        params.has_seq_q_lens,
        params.has_sink,
        params.has_bias,
        params.bias_is_fp32,
        False,
        params.has_rope,
        params.sched_policy,
        params.sched_l2_mib * 1024 * 1024,
        cutlass.Int32((skv + params.tile_n - 1) // params.tile_n),
        scale_log2,
        cutlass.Int32(sq),
        cutlass.Int32(skv),
        cutlass.Int32(d),
        cutlass.Int32(right_bound),
        inv_scale,
        cutlass.Int32(0),
        cutlass.Int32(1),
        stream,
        _view(paged_k, paged_geometry[0]) if cutlass.const_expr(paged_geometry is not None) else None,
        _view(paged_v, paged_geometry[1]) if cutlass.const_expr(paged_geometry is not None) else None,
    )


@cute.jit
def host(
    q: cute.Pointer,
    k: cute.Pointer,
    v: cute.Pointer,
    o: cute.Pointer,
    stats: Optional[cute.Pointer],
    seq_kv: Optional[cute.Pointer],
    seq_q: Optional[cute.Pointer],
    sink: Optional[cute.Pointer],
    bias: Optional[cute.Pointer],
    scale_log2: cutlass.Float32,
    inv_scale: cutlass.Float32,
    module: cutlass.Constexpr,
    geometry: cutlass.Constexpr,
    dtype: cutlass.Constexpr,
    swa_window: cutlass.Constexpr[int],
    right_bound: cutlass.Constexpr[int],
    stream: driver.CUstream,
):
    _host(q, k, v, o, stats, seq_kv, seq_q, sink, bias, scale_log2, inv_scale, module, geometry, dtype, swa_window, right_bound, stream)


@cute.jit
def paged_host(
    q: cute.Pointer,
    k: cute.Pointer,
    v: cute.Pointer,
    o: cute.Pointer,
    stats: Optional[cute.Pointer],
    seq_kv: Optional[cute.Pointer],
    seq_q: Optional[cute.Pointer],
    sink: Optional[cute.Pointer],
    bias: Optional[cute.Pointer],
    paged_k: cute.Pointer,
    paged_v: cute.Pointer,
    scale_log2: cutlass.Float32,
    inv_scale: cutlass.Float32,
    module: cutlass.Constexpr,
    geometry: cutlass.Constexpr,
    dtype: cutlass.Constexpr,
    swa_window: cutlass.Constexpr,
    right_bound: cutlass.Constexpr,
    max_skv: cutlass.Constexpr,
    stream: driver.CUstream,
):
    _host(
        q,
        k,
        v,
        o,
        stats,
        seq_kv,
        seq_q,
        sink,
        bias,
        scale_log2,
        inv_scale,
        module,
        geometry[:9],
        dtype,
        swa_window,
        right_bound,
        stream,
        paged_k,
        paged_v,
        geometry[9:],
        max_skv,
    )


def _compile_pointer_host(entry, module, params, geometry, swa_window, right_bound, cache_key, max_skv=None):
    dtype = cutlass.BFloat16 if params.io_bf16 else cutlass.Float16
    types = [dtype] * 4 + [cutlass.Float32, cutlass.Int32, cutlass.Int32, cutlass.Float32, cutlass.Float32 if params.bias_is_fp32 else dtype]
    if params.paged_page_size:
        types.extend((cutlass.Int32, cutlass.Int32))
    args = [
        (
            cute.runtime.make_ptr(t, 16, cute.AddressSpace.gmem, assumed_align=16 if i < 4 else (2 if i == 8 and not params.bias_is_fp32 else 4))
            if g is not None
            else None
        )
        for i, (t, g) in enumerate(zip(types, geometry))
    ]
    return compile_cached(
        entry,
        *args,
        cutlass.Float32(0),
        cutlass.Float32(0),
        module,
        geometry,
        dtype,
        swa_window,
        right_bound,
        *((max_skv,) if params.paged_page_size else ()),
        driver.CUstream(0),
        options="--enable-tvm-ffi",
        cache_key=cache_key,
        symbol="frost_sdpa_fwd",
    )


def compile_host(module, params, geometry, swa_window, right_bound, cache_key, max_skv=None):
    entry = paged_host if params.paged_page_size else host
    return _compile_pointer_host(entry, module, params, geometry, swa_window, right_bound, cache_key, max_skv)


@cute.jit
def _thd_host(
    q: cute.Pointer,
    k: cute.Pointer,
    v: cute.Pointer,
    o: cute.Pointer,
    stats: Optional[cute.Pointer],
    cu_q: cute.Pointer,
    cu_k: cute.Pointer,
    sink: Optional[cute.Pointer],
    t_q: cutlass.Int64,
    t_kv: cutlass.Int64,
    max_sq: cutlass.Int64,
    q_s: cutlass.Int64,
    q_h: cutlass.Int64,
    k_s: cutlass.Int64,
    k_h: cutlass.Int64,
    v_s: cutlass.Int64,
    v_h: cutlass.Int64,
    scale_log2: cutlass.Float32,
    inv_scale: cutlass.Float32,
    right_bound: cutlass.Int32,
    aux_strides: tuple,
    module: cutlass.Constexpr,
    h: cutlass.Constexpr,
    h_kv: cutlass.Constexpr,
    n_seq: cutlass.Constexpr,
    swa_window: cutlass.Constexpr,
    lse_stride: cutlass.Constexpr,
    initialize_outputs: cutlass.Constexpr,
    o_stride: cutlass.Constexpr,
    dimensions: cutlass.Constexpr,
    stream: driver.CUstream,
    paged_k: Optional[cute.Pointer] = None,
    paged_v: Optional[cute.Pointer] = None,
    paged_geometry: cutlass.Constexpr = None,
    max_skv: cutlass.Constexpr = 0,
    ragged_q: Optional[cute.Pointer] = None,
    ragged_k: Optional[cute.Pointer] = None,
    ragged_v: Optional[cute.Pointer] = None,
    ragged_o: Optional[cute.Pointer] = None,
    ragged_stats: Optional[cute.Pointer] = None,
    ragged_geometry: cutlass.Constexpr = (None, None, None, None, None),
    origin_batch_strides: cutlass.Constexpr = (None, None, None, None, None),
):
    p = module.PARAMS
    d_qk, d_v = dimensions
    dtype = cutlass.BFloat16 if p.io_bf16 else cutlass.Float16
    mask = (MASK_CAUSAL if p.is_causal else MASK_NONE) | (MASK_SWA if p.has_swa else 0)
    output_stride = (0, h * d_v, d_v, 1) if o_stride is None else (0, o_stride[0], o_stride[1], 1)
    # Wrapper outputs retain zeroed holes/tails outside the live prefixes.
    # Empty capacities do not launch initialization or step either pointer.
    if cutlass.const_expr(initialize_outputs):
        if t_q > 0:
            zero_outputs((_view(o, ((1, t_q, h, d_v), output_stride)), _view(stats, ((1, h, t_q), (0, t_q, 1)))), stream)
    if cutlass.const_expr(p.paged_page_size > 0):
        k_view = _view(k, paged_geometry[0])
        v_view = _view(v, paged_geometry[1])
        skv = cutlass.Int64(max_skv)
    else:
        k_view = _view(k, ((1, t_kv, h_kv, d_qk), (0, k_s, k_h, 1)))
        v_view = _view(v, ((1, t_kv, h_kv, d_v), (0, v_s, v_h, 1)))
        skv = t_kv
    # Packed capacity and Stats head pitch remain dynamic Int64 values. The
    # never-stepped batch stride is zero, so it cannot specialize on capacity.
    module._sdpa_host(
        _view(q, ((1, t_q, h, d_qk), (0, q_s, q_h, 1))),
        k_view,
        v_view,
        _view(o, ((1, t_q, h, d_v), output_stride)),
        _view(stats, ((1, h, t_q), (0, t_q, 1) if lse_stride is None else lse_stride)),
        None,
        None,
        _view(sink, ((h,), (aux_strides[2],))),
        None,
        _view(cu_q, ((n_seq + 1,), (aux_strides[0],))),
        _view(cu_k, ((n_seq + 1,), (aux_strides[1],))),
        None,
        p.tile_m,
        p.num_warps,
        p.tile_n,
        p.d_qk,
        p.d_v,
        dtype,
        False,
        d_qk == p.d_qk,
        mask,
        swa_window,
        p.causal_bottom_right,
        False,
        False,
        p.has_sink,
        False,
        False,
        True,
        False,
        p.sched_policy,
        p.sched_l2_mib * 1024 * 1024,
        cutlass.Int32((skv + p.tile_n - 1) // p.tile_n),
        scale_log2,
        cutlass.Int32(t_q),
        cutlass.Int32(skv),
        cutlass.Int32(d_qk),
        right_bound,
        inv_scale,
        cutlass.Int32((max_sq + p.tile_m - 1) // p.tile_m),
        cutlass.Int32(n_seq),
        stream,
        _view(paged_k, paged_geometry[2]) if cutlass.const_expr(p.paged_page_size > 0) else None,
        _view(paged_v, paged_geometry[3]) if cutlass.const_expr(p.paged_page_size > 0) else None,
        _view(ragged_q, ((n_seq + 1,), (ragged_geometry[0][1],))) if cutlass.const_expr(ragged_q is not None) else None,
        _view(ragged_k, ((n_seq + 1,), (ragged_geometry[1][1],))) if cutlass.const_expr(ragged_k is not None) else None,
        _view(ragged_v, ((n_seq + 1,), (ragged_geometry[2][1],))) if cutlass.const_expr(ragged_v is not None) else None,
        _view(ragged_o, ((n_seq + 1,), (ragged_geometry[3][1],))) if cutlass.const_expr(ragged_o is not None) else None,
        _view(ragged_stats, ((n_seq + 1,), (ragged_geometry[4][1],))) if cutlass.const_expr(ragged_stats is not None) else None,
        origin_batch_strides,
        tuple(g[2] if g is not None else 1 for g in ragged_geometry),
    )


@cute.jit
def thd_host(
    q: cute.Pointer,
    k: cute.Pointer,
    v: cute.Pointer,
    o: cute.Pointer,
    stats: Optional[cute.Pointer],
    cu_q: cute.Pointer,
    cu_k: cute.Pointer,
    sink: Optional[cute.Pointer],
    t_q: cutlass.Int64,
    t_kv: cutlass.Int64,
    max_sq: cutlass.Int64,
    q_s: cutlass.Int64,
    q_h: cutlass.Int64,
    k_s: cutlass.Int64,
    k_h: cutlass.Int64,
    v_s: cutlass.Int64,
    v_h: cutlass.Int64,
    scale_log2: cutlass.Float32,
    inv_scale: cutlass.Float32,
    right_bound: cutlass.Int32,
    aux_strides: tuple,
    module: cutlass.Constexpr,
    h: cutlass.Constexpr,
    h_kv: cutlass.Constexpr,
    n_seq: cutlass.Constexpr,
    swa_window: cutlass.Constexpr,
    lse_stride: cutlass.Constexpr,
    initialize_outputs: cutlass.Constexpr,
    o_stride: cutlass.Constexpr,
    dimensions: cutlass.Constexpr,
    stream: driver.CUstream,
):
    _thd_host(
        q,
        k,
        v,
        o,
        stats,
        cu_q,
        cu_k,
        sink,
        t_q,
        t_kv,
        max_sq,
        q_s,
        q_h,
        k_s,
        k_h,
        v_s,
        v_h,
        scale_log2,
        inv_scale,
        right_bound,
        aux_strides,
        module,
        h,
        h_kv,
        n_seq,
        swa_window,
        lse_stride,
        initialize_outputs,
        o_stride,
        dimensions,
        stream,
    )


@lru_cache(maxsize=128)
def compile_thd_host(
    module,
    h,
    h_kv,
    n_seq,
    swa_window,
    prefix_dtypes=("torch.int32", "torch.int32"),
    sink_dtype="torch.float32",
    lse_stride=None,
    initialize_outputs=True,
    o_stride=None,
    dimensions=None,
    paged_geometry=None,
    max_skv=0,
    ragged_geometry=None,
    origin_batch_strides=(None, None, None, None, None),
):
    """One pointer artifact per immutable flavor/head/mask contract."""
    p = module.PARAMS
    dimensions = (p.d_qk, p.d_v) if dimensions is None else tuple(dimensions)
    dtype = cutlass.BFloat16 if p.io_bf16 else cutlass.Float16
    aux_types = {
        "torch.int32": cutlass.Int32,
        "torch.int64": cutlass.Int64,
        "torch.float32": cutlass.Float32,
        "torch.float16": cutlass.Float16,
        "torch.bfloat16": cutlass.BFloat16,
    }
    types = [dtype] * 4 + [cutlass.Float32, *(aux_types[t] for t in prefix_dtypes), aux_types[sink_dtype]]
    pointers = [
        (
            cute.runtime.make_ptr(t, 16, cute.AddressSpace.gmem, assumed_align=16 if i < 4 else t.width // 8)
            if (i != 7 or p.has_sink) and (i != 4 or p.has_lse)
            else None
        )
        for i, t in enumerate(types)
    ]
    key = template_key(
        vars(module),
        dict(
            h=h,
            h_kv=h_kv,
            n_seq=n_seq,
            swa_window=swa_window,
            packed_init=_INIT_DIGEST,
            prefix_dtypes=prefix_dtypes,
            sink_dtype=sink_dtype,
            lse_stride=lse_stride,
            initialize_outputs=initialize_outputs,
            o_stride=o_stride,
            dimensions=dimensions,
            paged_geometry=paged_geometry,
            max_skv=max_skv,
            ragged_geometry=ragged_geometry,
            origin_batch_strides=origin_batch_strides,
        ),
        "prepared_thd",
    )
    extended = bool(p.paged_page_size) or ragged_geometry is not None or any(s is not None for s in origin_batch_strides)
    extra = ()
    if p.paged_page_size or extended:
        table_ptr = lambda: cute.runtime.make_ptr(cutlass.Int32, 16, cute.AddressSpace.gmem, assumed_align=4)
        extra = (table_ptr() if p.paged_page_size else None, table_ptr() if p.paged_page_size else None, paged_geometry, max_skv)
    if extended:
        ragged_geometry = (None,) * 5 if ragged_geometry is None else ragged_geometry
        extra += tuple(
            cute.runtime.make_ptr(aux_types[g[0]], 16, cute.AddressSpace.gmem, assumed_align=aux_types[g[0]].width // 8) if g is not None else None
            for g in ragged_geometry
        ) + (ragged_geometry, origin_batch_strides)
    artifact = compile_cached(
        _thd_host if p.paged_page_size or extended else thd_host,
        *pointers,
        *(cutlass.Int64(0) for _ in range(9)),
        cutlass.Float32(0),
        cutlass.Float32(0),
        cutlass.Int32(0),
        (cutlass.Int64(0), cutlass.Int64(0), cutlass.Int64(0)),
        module,
        h,
        h_kv,
        n_seq,
        swa_window,
        lse_stride,
        initialize_outputs,
        o_stride,
        dimensions,
        driver.CUstream(0),
        *extra,
        options="--enable-tvm-ffi --gpu-arch=sm_80",
        cache_key=key,
        symbol="frost_sdpa_fwd_thd",
    )
    fn = positional_entry(artifact)
    if fn is None:
        raise NotImplementedError("SM80 prepared THD forward requires a positional tvm-ffi entry")
    return artifact, fn
