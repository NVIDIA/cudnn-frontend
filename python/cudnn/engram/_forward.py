# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-order FP32 gate forward with caller-owned saved state."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from ._math import _add, _mul, _source_sum, _copysign, _load, _store


@cute.kernel
def engram_gate_saved_forward(
    x: cute.Tensor,
    key: cute.Tensor,
    value: cute.Tensor,
    weight: cute.Tensor,
    mask: cute.Tensor,
    out: cute.Tensor,
    saved: cute.Tensor,
    d: cutlass.Constexpr,
    eps: cutlass.Constexpr,
    key_stride: cutlass.Constexpr,
    value_stride: cutlass.Constexpr,
):
    tid, _, _ = cute.arch.thread_idx()
    token, _, _ = cute.arch.block_idx()
    lane, head = tid % 32, tid // 32
    row = token * 4 + head
    base = cutlass.Int64(row) * d
    ax, ak, ad = [cute.make_rmem_tensor(4, cutlass.Float32) for _ in range(3)]
    for accumulator in (ax, ak, ad):
        accumulator.fill(0)
    # Contiguous FP32 Torch reduction: four independent vector accumulators,
    # 40 iterations at H5120, followed by their ordered left fold.
    for step in cutlass.range_constexpr(d // 128):
        col = (step * 32 + lane) * 4
        hx = _load(x, base + col, 4)
        hk = _load(key, cutlass.Int64(token) * key_stride + head * d + col, 4)
        hw = _load(weight, head * d + col, 4)
        for j in cutlass.range_constexpr(4):
            a, b = hx[j].to(cutlass.Float32), hk[j].to(cutlass.Float32)
            ax[j] = _add(ax[j], _mul(a, a))
            ak[j] = _add(ak[j], _mul(b, b))
            ad[j] = _add(ad[j], _mul(_mul(a, hw[j]), b))
    xx, kk, weighted = _source_sum(ax), _source_sum(ak), _source_sum(ad)
    rx = cute.math.rsqrt(_add(_mul(xx, 1.0 / d), eps), fastmath=True)
    rk = cute.math.rsqrt(_add(_mul(kk, 1.0 / d), eps), fastmath=True)
    dot = _mul(_mul(weighted, _mul(rx, rk)), d**-0.5)
    absolute = cute.arch.fmax(dot, -dot)
    root = cute.math.sqrt(cute.arch.fmax(absolute, 1e-6), fastmath=False)
    gate = cutlass.Float32(1) / _add(1, cute.math.exp(-_copysign(root, dot), fastmath=True))
    if mask[token] == 0:
        gate = cutlass.Float32(0)
    if lane == 0:
        saved[row * 4] = gate
        saved[row * 4 + 1] = dot
        saved[row * 4 + 2] = rx
        saved[row * 4 + 3] = rk
    # Reload X after the reduction to avoid retaining 160 elements per lane.
    for step in cutlass.range_constexpr(d // 128):
        col = (step * 32 + lane) * 4
        hx = _load(x, base + col, 4)
        hv = _load(value, cutlass.Int64(token) * value_stride + col, 4)
        result = cute.make_rmem_tensor(4, cutlass.BFloat16)
        for j in cutlass.range_constexpr(4):
            result[j] = _add(hx[j].to(cutlass.Float32), _mul(gate, hv[j].to(cutlass.Float32))).to(cutlass.BFloat16)
        _store(result, out, base + col, 4)


engram_gate_saved_forward.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def launch_forward(
    x: cute.Pointer,
    key: cute.Pointer,
    value: cute.Pointer,
    weight: cute.Pointer,
    mask: cute.Pointer,
    out: cute.Pointer,
    saved: cute.Pointer,
    tokens: cutlass.Int32,
    stream: cuda.CUstream,
    d: cutlass.Constexpr,
    eps: cutlass.Constexpr,
    key_stride: cutlass.Constexpr,
    value_stride: cutlass.Constexpr,
):
    layout = cute.make_layout((1 << 63) - 1)
    tensors = [cute.make_tensor(p, layout) for p in (x, key, value, weight, mask, out, saved)]
    engram_gate_saved_forward(*tensors, d, eps, key_stride, value_stride).launch(grid=(tokens, 1, 1), block=(128, 1, 1), stream=stream)
