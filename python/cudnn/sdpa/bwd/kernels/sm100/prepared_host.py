# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pointer host for the large-head SM100 backward chain."""

from dataclasses import dataclass
from typing import Optional

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver

from cudnn.frost.compiled_cache import compile_cached
from cudnn.frost.tile_dsl.tma import st_global_v4
from cudnn.sdpa.bwd.kernels.sm120.bprop_chain_f16 import dot_do_o_host, dkv_reduce_host
from cudnn.sdpa.bwd.kernels.sm120.prepared_host import _scratch, _view
from cudnn.sdpa.bwd.kernels.thd_helpers import thd_bwd_setup_host


@dataclass(frozen=True)
class Params:
    batch: int
    heads: int
    kv_heads: int
    dim: int
    q_max: int
    kv_max: int
    q_rows: int
    kv_rows: int
    chunk: int
    thd: bool
    zero_workspace: bool
    units: int
    granularity: int


@cute.kernel
def _zero_workspace(first: cute.Tensor, second: cute.Tensor):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    blocks, _, _ = cute.arch.grid_dim()
    i = cutlass.Int64(bid) * 256 + tid
    # S/dS are compact half-precision regions with tile-rounded dimensions and
    # 128-byte-aligned bases. One 128-bit store clears eight elements; scalar
    # stores launch excessive blocks and regress small packed-THD replay.
    first_addr, second_addr = first.iterator.toint(), second.iterator.toint()
    while i < cute.size(first) // 8:
        zeros = [cutlass.Int32(0)] * 4
        st_global_v4(first_addr + i * 16, zeros, cutlass.Int32)
        st_global_v4(second_addr + i * 16, zeros, cutlass.Int32)
        i += cutlass.Int64(blocks) * 256


_zero_workspace.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _heads(tensor: cute.Tensor, begin, count: cutlass.Constexpr, step: cutlass.Constexpr = 1):
    shape = (tensor.shape[0], tensor.shape[1], count, tensor.shape[3])
    strides = (tensor.stride[0], tensor.stride[1], tensor.stride[2] * step, tensor.stride[3])
    return cute.make_tensor(tensor.iterator + cutlass.Int64(begin) * tensor.stride[2], cute.make_layout(shape, stride=strides))


@cute.jit
def _workspace_heads(tensor: cute.Tensor, begin, count: cutlass.Constexpr, step: cutlass.Constexpr):
    shape = (tensor.shape[0], count, tensor.shape[2], tensor.shape[3])
    strides = (tensor.stride[0], tensor.stride[1] * step, tensor.stride[2], tensor.stride[3])
    return cute.make_tensor(tensor.iterator + cutlass.Int64(begin) * tensor.stride[1], cute.make_layout(shape, stride=strides))


@cute.jit
def _permuted(tensor: cute.Tensor, order: cutlass.Constexpr):
    return cute.make_tensor(tensor.iterator, cute.make_layout(tuple(tensor.shape[i] for i in order), stride=tuple(tensor.stride[i] for i in order)))


@cute.jit
def _matmul(entry: cutlass.Constexpr, a, b, output, heads: cutlass.Constexpr, batches: cutlass.Constexpr, grid_m: cutlass.Constexpr, meta, desc, stream):
    # Widen dimensions and strides before the stage-3 descriptor's byte products.
    problem = tuple(
        cutlass.Int64(x)
        for x in (grid_m, b.shape[0], a.shape[1], heads, batches, *a.stride, *b.stride, *output.stride, b.shape[1], a.shape[0], output.shape[0])
    )
    entry(problem, a, b, output, meta, desc, stream)


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
    q_lens_ptr: Optional[cute.Pointer],
    kv_lens_ptr: Optional[cute.Pointer],
    workspace: cute.Pointer,
    scale_log2: cutlass.Float32,
    scale: cutlass.Float32,
    lens_form: cutlass.Int32,
    stage2: cutlass.Constexpr,
    mm_lo: cutlass.Constexpr,
    mm_hi: cutlass.Constexpr,
    config: cutlass.Constexpr,
    geometry: cutlass.Constexpr,
    regions: cutlass.Constexpr,
    dtype: cutlass.Constexpr,
    stream: driver.CUstream,
):
    batch, heads, kv_heads, dim, q_max, kv_max, q_rows, kv_rows, chunk, thd, zero_workspace, units, granularity = config
    q = _view(q_ptr, geometry[0])
    k = _view(k_ptr, geometry[1])
    v = _view(v_ptr, geometry[2])
    o = _view(o_ptr, geometry[3])
    do = _view(do_ptr, geometry[4])
    stats = _view(stats_ptr, geometry[5])
    dq = _view(dq_ptr, geometry[6])
    dk = _view(dk_ptr, geometry[7])
    dv = _view(dv_ptr, geometry[8])
    delta = _scratch(workspace, regions[0], cutlass.Float32)
    s_full = _scratch(workspace, regions[1], dtype)
    ds_full = _scratch(workspace, regions[2], dtype)
    meta = _scratch(workspace, regions[3], cutlass.Int32)
    desc2 = _scratch(workspace, regions[4], cutlass.Int64)
    desc3 = desc2
    if cutlass.const_expr(thd):
        # All three GEMMs patch this descriptor scratch immediately before
        # launching; sequential launches on this stream serialize its reuse.
        desc3 = _scratch(workspace, regions[5], cutlass.Int64)
        # The setup kernel branches on lens_form before reading the prefix tail.
        q_lens = _view(q_lens_ptr, ((batch + 1,), (1,)))
        kv_lens = _view(kv_lens_ptr, ((batch + 1,), (1,)))
        thd_bwd_setup_host(meta, q_lens, kv_lens, lens_form, heads, batch, 128, granularity, units, stream)
    # Zero once outside the head-chunk loop: stage 2 leaves mask-skipped tiles
    # unwritten, and stage 3 can consume a wider tile. The skipped set is the
    # same for every chunk, including THD, whose stage-3 K range is untrimmed.
    if cutlass.const_expr(zero_workspace):
        _zero_workspace(s_full, ds_full).launch(grid=(min((cute.size(s_full) // 8 + 255) // 256, 4096), 1, 1), block=(256, 1, 1), stream=stream)
    padded_dim = (dim + 63) // 64 * 64
    dot_do_o_host(o, do, delta, None, None, 128, padded_dim, padded_dim, 64, False, False, stream)
    group = heads // kv_heads
    dk_target, dv_target = dk, dv
    if cutlass.const_expr(group > 1):
        dk_target = _scratch(workspace, regions[6], dtype)
        dv_target = _scratch(workspace, regions[7], dtype)
    s_view, ds_view = s_full, ds_full
    if cutlass.const_expr(not thd):
        extent = (batch, chunk, q_max, kv_max)
        s_view = cute.make_tensor(s_full.iterator, cute.make_layout(extent, stride=s_full.stride))
        ds_view = cute.make_tensor(ds_full.iterator, cute.make_layout(extent, stride=ds_full.stride))
    for chunk_id in range(heads // chunk):
        head_base = chunk_id * chunk
        problem = (batch, heads, q_rows, kv_rows, chunk, kv_heads, q_max, kv_max, units)
        stage2(q, k, v, do, s_full, ds_full, stats, delta, meta, desc2, problem, scale, scale_log2, scale, head_base, 0, stream)
        do_heads = _heads(do, head_base, chunk)
        q_heads = _heads(q, head_base, chunk)
        dv_heads = _heads(dv_target, head_base, chunk)
        dk_heads = _heads(dk_target, head_base, chunk)
        _matmul(
            mm_lo,
            _permuted(s_view, (3, 2, 1, 0)),
            _permuted(do_heads, (3, 1, 2, 0)),
            _permuted(dv_heads, (1, 3, 2, 0)),
            chunk,
            batch,
            kv_max,
            meta,
            desc3,
            stream,
        )
        _matmul(
            mm_lo,
            _permuted(ds_view, (3, 2, 1, 0)),
            _permuted(q_heads, (3, 1, 2, 0)),
            _permuted(dk_heads, (1, 3, 2, 0)),
            chunk,
            batch,
            kv_max,
            meta,
            desc3,
            stream,
        )
        kv_count = chunk // group
        k_heads = _heads(k, head_base // group, kv_count)
        # Each GQA member addresses every group-th Q head against the shared K
        # head; dK/dV instead write per-Q-head partials and reduce below.
        for member in range(group):
            a = _workspace_heads(ds_view, member, kv_count, group)
            output = _heads(dq, head_base + member, kv_count, group)
            _matmul(
                mm_hi,
                _permuted(a, (2, 3, 1, 0)),
                _permuted(k_heads, (3, 1, 2, 0)),
                _permuted(output, (1, 3, 2, 0)),
                kv_count,
                batch,
                q_max,
                meta,
                desc3,
                stream,
            )
    if cutlass.const_expr(group > 1):
        dkv_reduce_host(dk_target, dv_target, dk, dv, dim, dim, group, dtype, False, stream)


def compile_host(stage2, mm_lo, mm_hi, params, geometry, regions, dtype, sm, cache_key):
    # Source codegen domain; the complete engine remains qualified only on
    # SM100/SM103. Other targets are used for isolated lowering checks.
    if sm not in (100, 103, 107, 110):
        raise ValueError(f"SM100 SDPA bwd has codegen targets for SM100, SM103, SM107, SM110; got SM{sm}")

    def ptr(t, align=16):
        return cute.runtime.make_ptr(t, 16, cute.AddressSpace.gmem, assumed_align=align)

    args = [ptr(dtype) for _ in range(5)] + [ptr(cutlass.Float32, 4)] + [ptr(dtype) for _ in range(3)]
    args += [ptr(cutlass.Int32, 4) if params.thd else None for _ in range(2)]
    # The persistent artifact wrapper accepts primitive constexpr tuples; a
    # dataclass argument prevents export even when it is compile-time-only.
    return compile_cached(
        host,
        *args,
        ptr(cutlass.Uint8),
        cutlass.Float32(1),
        cutlass.Float32(1),
        cutlass.Int32(0),
        stage2,
        mm_lo,
        mm_hi,
        (
            params.batch,
            params.heads,
            params.kv_heads,
            params.dim,
            params.q_max,
            params.kv_max,
            params.q_rows,
            params.kv_rows,
            params.chunk,
            params.thd,
            params.zero_workspace,
            params.units,
            params.granularity,
        ),
        geometry,
        regions,
        dtype,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options=f"--enable-tvm-ffi --gpu-arch sm_{sm}a",
        cache_key=cache_key,
        symbol="frost_sdpa_bwd_sm100_prepared",
    )
