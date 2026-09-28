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


@cute.jit
def _view(ptr: Optional[cute.Pointer], geometry: cutlass.Constexpr):
    if cutlass.const_expr(ptr is None):
        return None
    shape, strides = geometry
    # Materialize the fixed outer strides as Int64 kernel arguments. Keeping
    # them static changes device address code generation and regresses replay
    # on SM80; they are still constants in this compiled host, never per-call
    # Python metadata. The contiguous innermost dimension remains static.
    return cute.make_tensor(ptr, cute.make_layout(shape, stride=tuple(cutlass.Int64(st) if i < len(strides) - 1 else st for i, st in enumerate(strides))))


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
    # Reuse the template configuration without a redundant dataclass argument;
    # compiled_cache can then export and reload the pointer-only call ABI.
    params = module.PARAMS
    sq, skv, d = geometry[0][0][1], geometry[1][0][1], geometry[0][0][3]
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
        False,
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
    )


def compile_host(module, params, geometry, swa_window, right_bound, cache_key):
    dtype = cutlass.BFloat16 if params.io_bf16 else cutlass.Float16
    types = [dtype] * 4 + [cutlass.Float32, cutlass.Int32, cutlass.Int32, cutlass.Float32, cutlass.Float32 if params.bias_is_fp32 else dtype]
    args = [
        (
            cute.runtime.make_ptr(t, 16, cute.AddressSpace.gmem, assumed_align=16 if i < 4 else (4 if i < 8 or params.bias_is_fp32 else 2))
            if g is not None
            else None
        )
        for i, (t, g) in enumerate(zip(types, geometry))
    ]
    return compile_cached(
        host,
        *args,
        cutlass.Float32(0),
        cutlass.Float32(0),
        module,
        geometry,
        dtype,
        swa_window,
        right_bound,
        driver.CUstream(0),
        options="--enable-tvm-ffi",
        cache_key=cache_key,
        symbol="frost_sdpa_fwd",
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
    module: cutlass.Constexpr,
    h: cutlass.Constexpr,
    h_kv: cutlass.Constexpr,
    n_seq: cutlass.Constexpr,
    swa_window: cutlass.Constexpr,
    lse_stride: cutlass.Constexpr,
    stream: driver.CUstream,
):
    p = module.PARAMS
    dtype = cutlass.BFloat16 if p.io_bf16 else cutlass.Float16
    mask = (MASK_CAUSAL if p.is_causal else MASK_NONE) | (MASK_SWA if p.has_swa else 0)
    # Packed capacity and Stats head pitch remain dynamic Int64 values. The
    # never-stepped batch stride is zero, so it cannot specialize on capacity.
    module._sdpa_host(
        _view(q, ((1, t_q, h, p.d_qk), (0, q_s, q_h, 1))),
        _view(k, ((1, t_kv, h_kv, p.d_qk), (0, k_s, k_h, 1))),
        _view(v, ((1, t_kv, h_kv, p.d_v), (0, v_s, v_h, 1))),
        _view(o, ((1, t_q, h, p.d_v), (0, h * p.d_v, p.d_v, 1))),
        _view(stats, ((1, h, t_q), (0, t_q, 1) if lse_stride is None else lse_stride)),
        None,
        None,
        _view(sink, ((h,), (1,))),
        None,
        _view(cu_q, ((n_seq + 1,), (1,))),
        _view(cu_k, ((n_seq + 1,), (1,))),
        None,
        p.tile_m,
        p.num_warps,
        p.tile_n,
        p.d_qk,
        p.d_v,
        dtype,
        False,
        True,
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
        cutlass.Int32((t_kv + p.tile_n - 1) // p.tile_n),
        scale_log2,
        cutlass.Int32(t_q),
        cutlass.Int32(t_kv),
        cutlass.Int32(p.d_qk),
        right_bound,
        inv_scale,
        cutlass.Int32((max_sq + p.tile_m - 1) // p.tile_m),
        cutlass.Int32(n_seq),
        stream,
    )


@lru_cache(maxsize=128)
def compile_thd_host(module, h, h_kv, n_seq, swa_window, lse_stride=None):
    """One pointer artifact per immutable flavor/head/mask contract."""
    p = module.PARAMS
    dtype = cutlass.BFloat16 if p.io_bf16 else cutlass.Float16
    types = [dtype] * 4 + [cutlass.Float32, cutlass.Int32, cutlass.Int32, cutlass.Float32]
    pointers = [
        cute.runtime.make_ptr(t, 16, cute.AddressSpace.gmem, assumed_align=16 if i < 4 else 4) if (i != 7 or p.has_sink) and (i != 4 or p.has_lse) else None
        for i, t in enumerate(types)
    ]
    key = template_key(vars(module), dict(h=h, h_kv=h_kv, n_seq=n_seq, swa_window=swa_window, lse_stride=lse_stride), "prepared_thd")
    artifact = compile_cached(
        thd_host,
        *pointers,
        *(cutlass.Int64(0) for _ in range(9)),
        cutlass.Float32(0),
        cutlass.Float32(0),
        cutlass.Int32(0),
        module,
        h,
        h_kv,
        n_seq,
        swa_window,
        lse_stride,
        driver.CUstream(0),
        options="--enable-tvm-ffi",
        cache_key=key,
        symbol="frost_sdpa_fwd_thd",
    )
    fn = positional_entry(artifact)
    if fn is None:
        raise NotImplementedError("SM80 prepared THD forward requires a positional tvm-ffi entry")
    return artifact, fn
