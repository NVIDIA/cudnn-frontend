# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plan-time pointer host for native dense SM80 backward chains."""

import math
from typing import Optional

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver

from cudnn.frost.compiled_cache import compile_cached
from cudnn.frost.tile_dsl.mask import MASK_CAUSAL, MASK_NONE, MASK_PADDED, MASK_SWA
from cudnn.frost.tile_dsl.tma import st_global_v4


def workspace_regions(api):
    """Return the one workspace layout used by sizing and compilation."""
    b, h, hk, sq, skv = api.batch_size, api.h_q, api.h_kv, api.s_q_max, api.s_k_max
    dq, dv = api.head_dim_qk, api.head_dim_v
    qtile = 64  # Both shipped native pipelines use a 64-row Q tile.
    shapes = (
        (b, h, sq, dq) if api._use_d64 else (b, sq, h, dq),
        (b, h, sq),
        (b, skv, h, dq) if h != hk else None,
        (b, skv, h, dv) if h != hk else None,
        (b * h * ((sq + qtile - 1) // qtile),) if api.deterministic else None,
        (api._bias_batch, h, sq, skv) if api._has_bias else None,
        (h,) if api.sink_desc is not None else None,
    )
    regions, offset = [], 0
    for index, shape in enumerate(shapes):
        if shape is None:
            regions.append(None)
            continue
        strides = tuple(math.prod(shape[i + 1 :]) for i in range(len(shape)))
        regions.append((offset, shape, strides))
        width = 2 if index in (2, 3) else 4
        offset += ((math.prod(shape) * width + 127) // 128) * 128
    return tuple(regions), offset


@cute.jit
def _view(ptr: Optional[cute.Pointer], geometry: cutlass.Constexpr):
    if cutlass.const_expr(ptr is None):
        return None
    shape, strides = geometry
    return cute.make_tensor(ptr, cute.make_layout(shape, stride=tuple(cutlass.Int64(st) if i < len(strides) - 1 else st for i, st in enumerate(strides))))


@cute.jit
def _scratch(workspace: cute.Pointer, region: cutlass.Constexpr, dtype: cutlass.Constexpr):
    if cutlass.const_expr(region is None):
        return None
    offset, shape, strides = region
    ptr = cute.make_ptr(dtype, (workspace + offset).toint(), cute.AddressSpace.gmem, assumed_align=16)
    return cute.make_tensor(ptr, cute.make_layout(shape, stride=strides))


@cute.kernel
def _zero_regions(workspace: cute.Pointer, regions: cutlass.Constexpr, max_words: cutlass.Constexpr[int]):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    blocks, _, _ = cute.arch.grid_dim()
    i = cutlass.Int64(bid) * 256 + tid
    while i * 4 < max_words:
        word = i * 4
        for region_index in cutlass.range_constexpr(4):
            offset, words = regions[region_index]
            if cutlass.const_expr(words > 0):
                address = (workspace + offset).toint()
                if word + 3 < words:
                    st_global_v4(address + word * 4, [cutlass.Int32(0)] * 4, cutlass.Int32)
                elif word < words:
                    ptr = cute.make_ptr(cutlass.Int32, address, cute.AddressSpace.gmem, assumed_align=16)
                    dst = cute.make_tensor(ptr, cute.make_layout((words,), stride=(1,)))
                    for col in cutlass.range_constexpr(4):
                        if word + col < words:
                            dst[word + col] = cutlass.Int32(0)
        i += cutlass.Int64(blocks) * 256


_zero_regions.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.kernel
def _copy_aux(
    bias_acc: Optional[cute.Tensor],
    bias_out: Optional[cute.Tensor],
    sink_acc: Optional[cute.Tensor],
    sink_out: Optional[cute.Tensor],
    bias_dtype: cutlass.Constexpr,
):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    i = cutlass.Int64(bid) * 256 + tid
    if cutlass.const_expr(bias_out is not None):
        if i < cute.size(bias_out):
            bias_out.iterator[i] = bias_acc.iterator[i].to(bias_dtype)
    if cutlass.const_expr(sink_out is not None):
        if i < cute.size(sink_out):
            sink_out.iterator[i] = sink_acc.iterator[i]


_copy_aux.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def host(
    q_ptr: cute.Pointer,
    k_ptr: cute.Pointer,
    v_ptr: cute.Pointer,
    o_ptr: cute.Pointer,
    do_ptr: cute.Pointer,
    stats_ptr: cute.Pointer,
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
    module: cutlass.Constexpr,
    d64_module: cutlass.Constexpr,
    geometry: cutlass.Constexpr,
    regions: cutlass.Constexpr,
    zero_regions: cutlass.Constexpr,
    zero_words: cutlass.Constexpr[int],
    dtype: cutlass.Constexpr,
    dbias_dtype: cutlass.Constexpr,
    swa_window: cutlass.Constexpr[int],
    right_bound: cutlass.Constexpr[int],
    stream: driver.CUstream,
):
    # The template module already owns the immutable configuration. Passing its
    # dataclass again prevents compiled_cache from exporting this pointer ABI.
    params = module.PARAMS
    q = _view(q_ptr, geometry[0])
    k = _view(k_ptr, geometry[1])
    v = _view(v_ptr, geometry[2])
    o = _view(o_ptr, geometry[3])
    do = _view(do_ptr, geometry[4])
    stats = _view(stats_ptr, geometry[5])
    dq = _view(dq_ptr, geometry[6])
    dk = _view(dk_ptr, geometry[7])
    dv = _view(dv_ptr, geometry[8])
    seq_q = _view(seq_q_ptr, geometry[9])
    seq_kv = _view(seq_kv_ptr, geometry[10])
    sink = _view(sink_ptr, geometry[11])
    dsink = _view(dsink_ptr, geometry[12])
    bias = _view(bias_ptr, geometry[13])
    dbias = _view(dbias_ptr, geometry[14])
    dq_acc = _scratch(workspace, regions[0], cutlass.Float32)
    dot = _scratch(workspace, regions[1], cutlass.Float32)
    dk_work = dk
    dv_work = dv
    if cutlass.const_expr(regions[2] is not None):
        dk_work = _scratch(workspace, regions[2], dtype)
        dv_work = _scratch(workspace, regions[3], dtype)
    sem = _scratch(workspace, regions[4], cutlass.Int32)
    dbias_acc = _scratch(workspace, regions[5], cutlass.Float32)
    dsink_acc = _scratch(workspace, regions[6], cutlass.Float32)
    b, sq, h, d = q.shape
    skv, hk, dim_v = k.shape[1], k.shape[2], v.shape[3]
    _zero_regions(workspace, zero_regions, zero_words).launch(grid=(min((zero_words + 1023) // 1024, 65535), 1, 1), block=(256, 1, 1), stream=stream)
    module._do_dot_host(o, do, dot, dim_v, dtype, cutlass.Int32(b * h * sq), stream)
    if cutlass.const_expr(params.has_sink):
        module._dsink_host(stats, dot, sink, dsink_acc, None, False, False, cutlass.Int32(b * h), stream)
    if cutlass.const_expr(d64_module is not None):
        n_q = cutlass.Int32((sq + d64_module.M_BLOCK - 1) // d64_module.M_BLOCK)
        d64_module._bprop_host(q, k, v, do, dq_acc, dk_work, dv_work, stats, dot, d, dtype, n_q, scale_log2, scale, stream)
        d64_module._unpermute_host(dq_acc, dq, d, dtype, n_q, stream)
    else:
        mask = (MASK_CAUSAL if params.is_causal else MASK_NONE) | (MASK_SWA if params.has_swa else 0) | (MASK_PADDED if params.has_seq_kv_lens else 0)
        module._bprop_host(
            q,
            k,
            v,
            do,
            dq_acc,
            dk_work,
            dv_work,
            stats,
            dot,
            seq_kv,
            bias,
            dbias_acc,
            None,
            None,
            None,
            seq_q,
            sem,
            params.d_qk,
            params.d_v,
            params.tile_kv,
            params.tile_q,
            params.warps_per_sg,
            1 if params.d_qk >= 256 else 2,
            params.d_qk <= 128,
            dtype,
            mask,
            swa_window,
            params.causal_bottom_right,
            params.has_seq_kv_lens,
            params.has_bias,
            params.bias_is_fp32,
            False,
            params.has_seq_q_lens,
            False,
            False,
            params.deterministic,
            params.sched_policy,
            params.sched_l2_mib * 1024 * 1024,
            cutlass.Int32((sq + params.tile_q - 1) // params.tile_q),
            scale_log2,
            scale,
            cutlass.Int32(right_bound),
            cutlass.Float32(1.0) / scale,
            cutlass.Int64(0 if params.bias_broadcast else h * sq * skv),
            cutlass.Int32((sq + params.tile_q - 1) // params.tile_q if params.deterministic else 0),
            cutlass.Int32(0),
            cutlass.Int32(0),
            stream,
        )
        module._cast_host(dq_acc, dq, dtype, cutlass.Int32(b * sq * h * d // 2), stream)
        if cutlass.const_expr(h != hk):
            module._dkv_reduce_host(dk_work, dk, d, h, hk, dtype, cutlass.Int32(b * skv * hk * d), stream)
            module._dkv_reduce_host(dv_work, dv, dim_v, h, hk, dtype, cutlass.Int32(b * skv * hk * dim_v), stream)
    if cutlass.const_expr(dbias is not None or dsink is not None):
        n_bias = 0 if cutlass.const_expr(dbias is None) else cute.size(dbias)
        n_sink = 0 if cutlass.const_expr(dsink is None) else cute.size(dsink)
        _copy_aux(dbias_acc, dbias, dsink_acc, dsink, dbias_dtype).launch(grid=((max(n_bias, n_sink) + 255) // 256, 1, 1), block=(256, 1, 1), stream=stream)


def compile_host(api, geometry, d64_module, cache_key):
    dtype = cutlass.BFloat16 if api._params.io_bf16 else cutlass.Float16
    dbias_dtype = cutlass.Float32 if api.dbias_desc is None or str(api.dbias_desc.dtype).endswith("float32") else dtype
    types = (
        [dtype] * 5
        + [cutlass.Float32]
        + [dtype] * 3
        + [cutlass.Int32] * 2
        + [cutlass.Float32] * 2
        + [cutlass.Float32 if api._bias_is_fp32 else dtype, dbias_dtype]
    )
    args = [
        (
            cute.runtime.make_ptr(t, 16, cute.AddressSpace.gmem, assumed_align=16 if i < 5 or 6 <= i < 9 else (2 if i >= 13 and t != cutlass.Float32 else 4))
            if g is not None
            else None
        )
        for i, (t, g) in enumerate(zip(types, geometry))
    ]
    regions, workspace_bytes = workspace_regions(api)
    zeros = tuple((regions[i][0], math.prod(regions[i][1])) if regions[i] is not None else (0, 0) for i in (0, 4, 5, 6))
    artifact = compile_cached(
        host,
        *args,
        cute.runtime.make_ptr(cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16),
        cutlass.Float32(0),
        cutlass.Float32(0),
        api._kmod,
        d64_module,
        geometry,
        regions,
        zeros,
        max(n for _, n in zeros),
        dtype,
        dbias_dtype,
        api.swa_window_runtime,
        api.right_bound_runtime,
        driver.CUstream(0),
        options="--enable-tvm-ffi",
        cache_key=cache_key,
        symbol="frost_sdpa_bwd",
    )
    return artifact, workspace_bytes
