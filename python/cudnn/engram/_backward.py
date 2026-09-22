# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic streamed gradient apply, packed dKV, and partial dWeight."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils
from ._math import _add, _mul, _load, _store


@cute.jit
def _reduce_eight(values):
    for level in cutlass.range_constexpr(3):
        span: cutlass.Constexpr = 4 >> level
        for i in cutlass.range_constexpr(span):
            for j in cutlass.range_constexpr(4):
                values[i * 4 + j] = _add(values[i * 4 + j], values[(i + span) * 4 + j])
    result = cute.make_rmem_tensor(4, cutlass.Float32)
    for j in cutlass.range_constexpr(4):
        result[j] = values[j]
    return result


@cute.kernel
def engram_gate_saved_apply(
    x: cute.Tensor,
    key: cute.Tensor,
    weight: cute.Tensor,
    upstream: cute.Tensor,
    stats: cute.Tensor,
    dx: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    partial: cute.Tensor,
    d: cutlass.Constexpr,
    ks: cutlass.Constexpr,
    dks: cutlass.Constexpr,
    dv_stride: cutlass.Constexpr,
):
    tid, _, _ = cute.arch.thread_idx()
    split, column, _ = cute.arch.block_idx()
    lane, warp = tid % 32, tid // 32
    col = column * 128 + lane * 4
    cached_dv = cute.make_rmem_tensor(32, cutlass.Float32)
    cached_dv.fill(0)
    scratch = cutlass.utils.SmemAllocator().allocate_tensor(cutlass.Float32, cute.make_layout(8 * 128), byte_alignment=16)
    for head in cutlass.range(4, unroll=1):
        terms = cute.make_rmem_tensor(32, cutlass.Float32)
        w = _load(weight, head * d + col, 4)
        for step in cutlass.range_constexpr(8):
            token = split * 64 + step * 8 + warp
            row = cutlass.Int64(token) * 4 + head
            xk = _load(x, row * d + col, 4)
            kk = _load(key, cutlass.Int64(token) * ks + head * d + col, 4)
            gk = _load(upstream, row * d + col, 4)
            st = _load(stats, row * 4, 4)
            ox, ok = [cute.make_rmem_tensor(4, cutlass.BFloat16) for _ in range(2)]
            for j in cutlass.range_constexpr(4):
                xv, kv, gv = xk[j].to(cutlass.Float32), kk[j].to(cutlass.Float32), gk[j].to(cutlass.Float32)
                cw = _mul(st[1], w[j])
                ox[j] = _add(_add(gv, _mul(cw, kv)), -_mul(st[2], xv)).to(cutlass.BFloat16)
                ok[j] = _add(_mul(cw, xv), -_mul(st[3], kv)).to(cutlass.BFloat16)
                cached_dv[step * 4 + j] = _add(cached_dv[step * 4 + j], _mul(st[0], gv))
                terms[step * 4 + j] = _mul(_mul(st[1], xv), kv)
            _store(ox, dx, row * d + col, 4)
            _store(ok, dk, cutlass.Int64(token) * dks + head * d + col, 4)
        local_sum = _reduce_eight(terms)
        _store(local_sum, scratch, warp * 128 + lane * 4, 4)
        cute.arch.barrier()
        if warp == 0:
            cross_warp = cute.make_rmem_tensor(32, cutlass.Float32)
            for i in cutlass.range_constexpr(8):
                v = _load(scratch, i * 128 + lane * 4, 4)
                for j in cutlass.range_constexpr(4):
                    cross_warp[i * 4 + j] = v[j]
            total = _reduce_eight(cross_warp)
            _store(total, partial, (cutlass.Int64(split) * 4 + head) * d + col, 4)
        cute.arch.barrier()
    for step in cutlass.range_constexpr(8):
        token = split * 64 + step * 8 + warp
        result = cute.make_rmem_tensor(4, cutlass.BFloat16)
        for j in cutlass.range_constexpr(4):
            result[j] = cached_dv[step * 4 + j].to(cutlass.BFloat16)
        _store(result, dv, cutlass.Int64(token) * dv_stride + col, 4)


engram_gate_saved_apply.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def launch_apply(
    x: cute.Pointer,
    key: cute.Pointer,
    weight: cute.Pointer,
    upstream: cute.Pointer,
    stats: cute.Pointer,
    dx: cute.Pointer,
    dk: cute.Pointer,
    dv: cute.Pointer,
    partial: cute.Pointer,
    tokens: cutlass.Int32,
    stream: cuda.CUstream,
    d: cutlass.Constexpr,
    ks: cutlass.Constexpr,
    dks: cutlass.Constexpr,
    dv_stride: cutlass.Constexpr,
):
    layout = cute.make_layout((1 << 63) - 1)
    tensors = [cute.make_tensor(p, layout) for p in (x, key, weight, upstream, stats, dx, dk, dv, partial)]
    engram_gate_saved_apply(*tensors, d, ks, dks, dv_stride).launch(grid=(tokens // 64, d // 128, 1), block=(256, 1, 1), stream=stream)
