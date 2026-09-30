# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One pointer entry for the complete SM120 backward launch chain."""

from typing import Optional

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver

from cudnn.frost.compiled_cache import compile_cached
from .bprop_chain_f16 import convert_dbias_host, convert_dq_host, dkv_reduce_host, dot_do_o_host, dsink_host


@cute.jit
def _view(ptr: Optional[cute.Pointer], geometry: cutlass.Constexpr):
    if cutlass.const_expr(ptr is None):
        return None
    shape, strides = geometry
    # Shape and layout are fixed plan facts; device addressing retains their
    # full Python-integer width and the chain's static layout specializations.
    return cute.make_tensor(ptr, cute.make_layout(shape, stride=strides))


@cute.jit
def _scratch(workspace: cute.Pointer, region: cutlass.Constexpr, dtype: cutlass.Constexpr):
    offset, shape, strides = region
    ptr = cute.make_ptr(dtype, (workspace + offset).toint(), cute.AddressSpace.gmem, assumed_align=16)
    return _view(ptr, (shape, strides))


@cute.kernel
def _zero_bias(dst: cute.Tensor, elements: cutlass.Constexpr[int]):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    i = cutlass.Int64(bid) * 256 + tid
    if i < elements:
        dst.iterator[i] = cutlass.Float32(0)


_zero_bias.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def host(
    q_ptr: cute.Pointer,
    k_ptr: cute.Pointer,
    v_ptr: cute.Pointer,
    o_ptr: cute.Pointer,
    do_ptr: cute.Pointer,
    lse_ptr: cute.Pointer,
    dq_ptr: cute.Pointer,
    dk_ptr: cute.Pointer,
    dv_ptr: cute.Pointer,
    seq_q_ptr: Optional[cute.Pointer],
    seq_kv_ptr: Optional[cute.Pointer],
    sink_ptr: Optional[cute.Pointer],
    dsink_ptr: Optional[cute.Pointer],
    bias_ptr: Optional[cute.Pointer],
    dbias_ptr: Optional[cute.Pointer],
    workspace: cute.Pointer,
    scale_log2: cutlass.Float32,
    scale: cutlass.Float32,
    kernel: cutlass.Constexpr,
    dq_gemm: cutlass.Constexpr,
    config: cutlass.Constexpr,
    geometry: cutlass.Constexpr,
    regions: cutlass.Constexpr,
    dtype: cutlass.Constexpr,
    group: cutlass.Constexpr[int],
    dbias_elements: cutlass.Constexpr[int],
    stream: driver.CUstream,
):
    det_2kernel, dbias_present, dbias_is_fp32, dsink_present = config
    q = _view(q_ptr, geometry[0])
    k = _view(k_ptr, geometry[1])
    v = _view(v_ptr, geometry[2])
    o = _view(o_ptr, geometry[3])
    do = _view(do_ptr, geometry[4])
    lse = _view(lse_ptr, geometry[5])
    dq = _view(dq_ptr, geometry[6])
    dk = _view(dk_ptr, geometry[7])
    dv = _view(dv_ptr, geometry[8])
    seq_q = _view(seq_q_ptr, geometry[9])
    seq_kv = _view(seq_kv_ptr, geometry[10])
    sink = _view(sink_ptr, geometry[11])
    dsink = _view(dsink_ptr, geometry[12])
    bias = _view(bias_ptr, geometry[13])
    dbias = _view(dbias_ptr, geometry[14])
    delta = _scratch(workspace, regions[0], cutlass.Float32)
    dq_accum = None
    dq_sem = None
    ds_ws = None
    if cutlass.const_expr(det_2kernel):
        ds_ws = _scratch(workspace, regions[1], dtype)
    else:
        dq_accum = _scratch(workspace, regions[1], cutlass.Float32)
        dq_sem = _scratch(workspace, regions[2], cutlass.Int32)
    dbias_dst = dbias
    if cutlass.const_expr(dbias_present):
        if cutlass.const_expr(not dbias_is_fp32):
            dbias_dst = _scratch(workspace, regions[3], cutlass.Float32)
        _zero_bias(dbias_dst, dbias_elements).launch(grid=((dbias_elements + 255) // 256, 1, 1), block=(256, 1, 1), stream=stream)
    dk_ws = dk
    dv_ws = dv
    if cutlass.const_expr(group != 1):
        dk_ws = _scratch(workspace, regions[4], dtype)
        dv_ws = _scratch(workspace, regions[5], dtype)

    dot_do_o_host(o, do, delta, dq_accum, dq_sem, kernel.q_tile, kernel.d_qk, kernel.d_v, kernel.chunk_elems, kernel.use_pdl, kernel.deterministic, stream)
    kernel(q, k, v, do, lse, delta, dq_accum, dq_sem, ds_ws, dk_ws, dv_ws, seq_q, seq_kv, bias, dbias_dst, scale_log2, scale, stream)
    if cutlass.const_expr(det_2kernel):
        dq_gemm(k, ds_ws, dq, scale, stream)
    if cutlass.const_expr(group != 1):
        dkv_reduce_host(dk_ws, dv_ws, dk, dv, kernel.d_qk, kernel.d_v, group, dtype, kernel.use_pdl, stream)
    if cutlass.const_expr(not det_2kernel):
        convert_dq_host(dq_accum, dq, kernel.q_tile, kernel.d_qk, kernel.chunk_elems, kernel.warps_m_dq, scale, dtype, kernel.use_pdl, stream)
    if cutlass.const_expr(dbias_present and not dbias_is_fp32):
        flat = cute.make_layout((dbias_elements,), stride=(1,))
        convert_dbias_host(cute.make_tensor(dbias_dst.iterator, flat), cute.make_tensor(dbias.iterator, flat), dtype, kernel.use_pdl, stream)
    if cutlass.const_expr(dsink_present):
        dsink_host(lse, delta, sink, dsink, seq_q, kernel.use_pdl, stream)


def compact(shape):
    strides = []
    stride = 1
    for extent in reversed(shape):
        strides.append(stride)
        stride *= extent
    return tuple(shape), tuple(reversed(strides))


def compile_host(kernel, dq_gemm, params, geometry, regions, dtype, group, cache_key):
    """Pointer-only compilation: tensor operands are constructed inside host IR."""

    def pointer(t, align=16):
        return cute.runtime.make_ptr(t, 16, cute.AddressSpace.gmem, assumed_align=align)

    types = [dtype] * 5 + [cutlass.Float32] + [dtype] * 3
    args = [pointer(t, 4 if i == 5 else 16) for i, t in enumerate(types)]
    args += [pointer(cutlass.Int32, 4) if enabled else None for enabled in (params.seq_q_lens_present, params.seq_kv_lens_present)]
    args += [pointer(cutlass.Float32, 4) if enabled else None for enabled in (params.sink_present, params.dsink_present)]
    args += [
        pointer(cutlass.Float32 if fp32 else dtype, 4 if fp32 else 2) if enabled else None
        for enabled, fp32 in ((params.bias_present, params.bias_is_fp32), (params.dbias_present, params.dbias_is_fp32))
    ]
    dbias_elements = 1
    for n in geometry[14][0]:
        dbias_elements *= n
    # The persistent artifact wrapper accepts primitive constexpr tuples; a
    # dataclass argument prevents export even when it is compile-time-only.
    return compile_cached(
        host,
        *args,
        pointer(cutlass.Uint8),
        cutlass.Float32(1),
        cutlass.Float32(1),
        kernel,
        dq_gemm,
        (params.det_2kernel, params.dbias_present, params.dbias_is_fp32, params.dsink_present),
        geometry,
        regions,
        dtype,
        group,
        dbias_elements,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options="--enable-tvm-ffi",
        cache_key=cache_key,
        symbol="frost_sdpa_bwd_prepared",
    )
