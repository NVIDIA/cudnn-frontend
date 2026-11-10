# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""SM120 FP8 mma.sync backward with fixed query/key ownership.

One warp owns 16 gradient rows and 128 columns. It streams 32-row score
blocks; key owners fold query heads in a fixed order. Only the linear delta
buffer is global scratch. P/dS quantization and all input reads are native.
"""

from typing import Optional

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as cuda
from cutlass.base_dsl.typing import Pointer
from cutlass.experimental import primitives as nvvm

from cudnn.frost.tile_dsl.mma import mma_m16n8k32_f32
from cudnn.frost.tile_dsl.pointwise import fmax_f32, fp32_to_fp8x2
from cudnn.sdpa.bwd.kernels.sm100.bprop_d192_dkdv_fp8 import _fold_amax, _scalar

FP8 = cutlass.Float8E4M3FN
SPEC = globals().get("FROST_TEMPLATE_PARAMS", (128, 128, False, False, 0, "sm120"))
WINDOW = int(SPEC[4])
M, N, C = 16, 32, 128


@cute.jit
def _zero_word():
    # mma.sync accepts register operands only. A fully masked static tail
    # must not fold to an immediate 0 in the inline PTX instruction.
    return cute.arch.inline_ptx("mov.b32 $0, 0;", write_only_types=[cutlass.Int32], read_only_args=[])


@cute.jit
def _word(t, batch, row, head, col):
    bits = _zero_word()
    if row < t.shape[1] and col < t.shape[3]:
        off = cutlass.Int64(batch) * t.stride[0] + cutlass.Int64(row) * t.stride[1] + cutlass.Int64(head) * t.stride[2] + col
        values = (t.iterator.raw_ptr() + off).load(count=4, alignment=4)
        bits = values.bitcast(cutlass.Int32)[0]
    return bits


@cute.jit
def _column_word(t, batch, row, head, col):
    bits = _zero_word().bitcast(cutlass.Uint32)
    for i in cutlass.range_constexpr(4):
        byte = cutlass.Uint8(0)
        if row + i < t.shape[1] and col < t.shape[3]:
            off = cutlass.Int64(batch) * t.stride[0] + cutlass.Int64(row + i) * t.stride[1] + cutlass.Int64(head) * t.stride[2] + col
            byte = (t.iterator.raw_ptr() + off).load().bitcast(cutlass.Uint8)
        bits = bits | (cutlass.Uint32(byte) << (8 * i))
    return bits.bitcast(cutlass.Int32)


@cute.jit
def _score(a, b, batch, ahead, bhead, ar, br):
    lane = cute.arch.lane_idx()
    r, c = lane // 4, lane % 4 * 4
    accum = cutlass.Array(cutlass.Float32, 16, alignment=16)
    for i in cutlass.range_constexpr(16):
        accum[i] = cutlass.Float32(0.0)
    for kc in cutlass.range_constexpr((a.shape[3] + 31) // 32):
        av = [_word(a, batch, ar + r + (8 if i % 2 else 0), ahead, kc * 32 + c + (16 if i >= 2 else 0)) for i in range(4)]
        for nf in cutlass.range_constexpr(4):
            b0 = _word(b, batch, br + nf * 8 + r, bhead, kc * 32 + c)
            b1 = _word(b, batch, br + nf * 8 + r, bhead, kc * 32 + c + 16)
            vals = mma_m16n8k32_f32(*av, b0, b1, *[accum[nf * 4 + i] for i in range(4)], FP8)
            for i in cutlass.range_constexpr(4):
                accum[nf * 4 + i] = vals[i]
    return accum


@cute.jit
def _grad(accum, a, b, batch, head, row, chunk):
    lane = cute.arch.lane_idx()
    r, c = lane // 4, lane % 4 * 4
    av = [Pointer(a.subview((r + (8 if i % 2 else 0)) * N + c + (16 if i >= 2 else 0)).data_ptr(), dtype=cutlass.Int32).load() for i in range(4)]
    for nf in cutlass.range_constexpr(C // 8):
        col = chunk * C + nf * 8 + r
        b0 = _column_word(b, batch, row + c, head, col)
        b1 = _column_word(b, batch, row + c + 16, head, col)
        vals = mma_m16n8k32_f32(*av, b0, b1, *[accum[nf * 4 + i] for i in range(4)], FP8)
        for i in cutlass.range_constexpr(4):
            accum[nf * 4 + i] = vals[i]


@cute.jit
def _store(accum, out, batch, head, owner, chunk, factor, out_scale, amax: Optional[cute.Tensor]):
    lane = cute.arch.lane_idx()
    maximum = cutlass.Float32(0.0)
    for nf in cutlass.range_constexpr(C // 8):
        for i in cutlass.range_constexpr(4):
            row = owner * M + lane // 4 + (8 if i >= 2 else 0)
            col = chunk * C + nf * 8 + lane % 4 * 2 + i % 2
            if row < out.shape[1] and col < out.shape[3]:
                val = accum[nf * 4 + i] * factor
                maximum = fmax_f32(maximum, cute.math.abs(val))
                if cutlass.const_expr(out.element_type == FP8):
                    bits = cutlass.Uint8(fp32_to_fp8x2(val * out_scale, cutlass.Float32(0.0)))
                    off = cutlass.Int64(batch) * out.stride[0] + cutlass.Int64(row) * out.stride[1] + cutlass.Int64(head) * out.stride[2] + col
                    (out.iterator.raw_ptr() + off).store(bits.bitcast(FP8))
                else:
                    off = cutlass.Int64(batch) * out.stride[0] + cutlass.Int64(row) * out.stride[1] + cutlass.Int64(head) * out.stride[2] + col
                    (out.iterator.raw_ptr() + off).store(val.to(out.element_type))
    if cutlass.const_expr(amax is not None):
        _fold_amax(maximum, amax)


@cute.kernel
def kernel(
    q,
    k,
    v,
    do,
    lse,
    delta,
    dg,
    dv,
    scales: tuple,
    amax_g: Optional[cute.Tensor],
    amax_v: Optional[cute.Tensor],
    amax_p: Optional[cute.Tensor],
    scale: cutlass.Float32,
    causal: cutlass.Constexpr[bool],
    window: cutlass.Constexpr[int],
    dq: cutlass.Constexpr[bool],
):
    x, head, batch = cute.arch.block_idx()
    chunks = (q.shape[3] + C - 1) // C
    owner, chunk = x // chunks, x % chunks
    lane = cute.arch.lane_idx()
    sq, skv = q.shape[1], k.shape[1]
    ratio = q.shape[2] // k.shape[2]
    ds = cutlass.Array(cutlass.Uint8, M * N, alignment=16, space=cutlass.AddressSpace.smem)
    p = cutlass.Array(cutlass.Uint8, M * N, alignment=16, space=cutlass.AddressSpace.smem)
    grad = cutlass.Array(cutlass.Float32, C // 2, alignment=16)
    gradv = cutlass.Array(cutlass.Float32, C // 2, alignment=16)
    for i in cutlass.range_constexpr(C // 2):
        grad[i] = cutlass.Float32(0.0)
        gradv[i] = cutlass.Float32(0.0)
    dqscale, dkscale, dvscale, ds_descale, s_scale, do_descale, dp_descale, g_scale, v_scale, dp_scale = tuple(_scalar(t) for t in scales)
    score_scale = scale * dqscale * dkscale
    delta_scale = scale * ds_descale * dp_scale
    dp_factor = delta_scale * dvscale * do_descale
    maximum = cutlass.Float32(0.0)
    lo = cutlass.Int32(0)
    hi = (skv + N - 1) // N if cutlass.const_expr(dq) else (sq + N - 1) // N
    if cutlass.const_expr(dq):
        if cutlass.const_expr(window > 0):
            lo = cute.math.max(cutlass.Int32(0), (owner * M - window + 1) // N)
        if cutlass.const_expr(causal):
            hi = cute.math.min(hi, (owner * M + M + N - 1) // N)
    else:
        if cutlass.const_expr(causal):
            lo = owner * M // N
        if cutlass.const_expr(window > 0):
            hi = cute.math.min(hi, (owner * M + M + window - 1 + N - 1) // N)
    nheads = 1 if cutlass.const_expr(dq) else ratio
    for rh in cutlass.range(nheads, unroll=1):
        hq = head if cutlass.const_expr(dq) else head * ratio + rh
        hk = head // ratio if cutlass.const_expr(dq) else head
        for other in cutlass.range(lo, hi, unroll=1):
            if cutlass.const_expr(dq):
                scores = _score(q, k, batch, hq, hk, owner * M, other * N)
                deriv = _score(do, v, batch, hq, hk, owner * M, other * N)
            else:
                scores = _score(k, q, batch, hk, hq, owner * M, other * N)
                deriv = _score(v, do, batch, hk, hq, owner * M, other * N)
            for nf in cutlass.range_constexpr(4):
                for half in cutlass.range_constexpr(2):
                    ps, dss = [], []
                    r = lane // 4 + half * 8
                    col = nf * 8 + lane % 4 * 2
                    for j in cutlass.range_constexpr(2):
                        qi = owner * M + r if cutlass.const_expr(dq) else other * N + col + j
                        ki = other * N + col + j if cutlass.const_expr(dq) else owner * M + r
                        live = (qi < sq) & (ki < skv)
                        if cutlass.const_expr(causal):
                            live = live & (ki <= qi)
                        if cutlass.const_expr(window > 0):
                            live = live & (ki + window > qi)
                        prob, dsval = cutlass.Float32(0.0), cutlass.Float32(0.0)
                        if live:
                            idx = nf * 4 + half * 2 + j
                            prob = (
                                cute.math.exp2((scores[idx] * score_scale - lse[batch, hq, qi]) * cutlass.Float32(1.4426950408889634), fastmath=True) * s_scale
                            )
                            dsval = (deriv[idx] * dp_factor - delta[batch, hq, qi] * delta_scale) * prob
                        ps.append(prob)
                        dss.append(dsval)
                        maximum = fmax_f32(maximum, cute.math.abs(dsval))
                    Pointer(ds.subview(r * N + col).data_ptr(), dtype=cutlass.Uint16).store(fp32_to_fp8x2(*dss))
                    if cutlass.const_expr(not dq):
                        Pointer(p.subview(r * N + col).data_ptr(), dtype=cutlass.Uint16).store(fp32_to_fp8x2(*ps))
            nvvm.bar_warp_sync(cute.arch.FULL_MASK)
            if cutlass.const_expr(dq):
                _grad(grad, ds, k, batch, hk, other * N, chunk)
            else:
                _grad(grad, ds, q, batch, hq, other * N, chunk)
                _grad(gradv, p, do, batch, hq, other * N, chunk)
            nvvm.bar_warp_sync(cute.arch.FULL_MASK)
    factor = dp_descale * (dkscale if cutlass.const_expr(dq) else dqscale)
    _store(grad, dg, batch, head, owner, chunk, factor, g_scale, amax_g)
    if cutlass.const_expr(not dq):
        _store(gradv, dv, batch, head, owner, chunk, ds_descale * do_descale, v_scale, amax_v)
    if cutlass.const_expr(amax_p is not None):
        _fold_amax(maximum / dp_scale, amax_p)


kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def host(
    q,
    k,
    v,
    do,
    lse,
    delta,
    dg,
    dv,
    scales,
    amax_g,
    amax_v,
    amax_p,
    scale,
    causal: cutlass.Constexpr[bool],
    window: cutlass.Constexpr[int],
    dq: cutlass.Constexpr[bool],
    stream,
):
    rows = q.shape[1] if cutlass.const_expr(dq) else k.shape[1]
    heads = q.shape[2] if cutlass.const_expr(dq) else k.shape[2]
    chunks = (q.shape[3] + C - 1) // C
    kernel(q, k, v, do, lse, delta, dg, dv, scales, amax_g, amax_v, amax_p, scale, causal, window, dq).launch(
        grid=((rows + M - 1) // M * chunks, heads, q.shape[0]), block=(32, 1, 1), stream=stream
    )


@cute.kernel
def delta_kernel(o, do, delta, descale_o, descale_do, amaxes: tuple):
    block, head, batch = cute.arch.block_idx()
    tid = cute.arch.thread_idx()[0]
    row, lane = block * 8 + tid // 16, tid % 16
    if block == 0 and head == 0 and batch == 0 and tid == 0:
        for tensor in amaxes:
            if cutlass.const_expr(tensor is not None):
                tensor[0] = cutlass.Float32(0.0)
    val = cutlass.Float32(0.0)
    if row < o.shape[1]:
        for i in cutlass.range_constexpr((o.shape[3] + 15) // 16):
            col = i * 16 + lane
            if col < o.shape[3]:
                oo = cutlass.Int64(batch) * o.stride[0] + cutlass.Int64(row) * o.stride[1] + cutlass.Int64(head) * o.stride[2] + col
                od = cutlass.Int64(batch) * do.stride[0] + cutlass.Int64(row) * do.stride[1] + cutlass.Int64(head) * do.stride[2] + col
                val = val + cutlass.Float32((o.iterator.raw_ptr() + oo).load()) * cutlass.Float32((do.iterator.raw_ptr() + od).load())
    for step in cutlass.range_constexpr(4):
        val = val + nvvm.shfl_sync(thread_mask=0xFFFFFFFF, val=val, offset=1 << step, mask_and_clamp=0x1F, kind=nvvm.Shfl.BFLY)
    if lane == 0:
        delta[batch, head, row] = val * _scalar(descale_o) * _scalar(descale_do)


delta_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def delta_host(o, do, delta, descale_o, descale_do, amax_dq, amax_dk, amax_dv, amax_dp, stream):
    delta_kernel(o, do, delta, descale_o, descale_do, (amax_dq, amax_dk, amax_dv, amax_dp)).launch(
        grid=(delta.shape[2] // 8, o.shape[2], o.shape[0]), block=(128, 1, 1), stream=stream
    )


@cute.jit
def dkdv_host(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    do: cute.Tensor,
    lse: cute.Tensor,
    delta: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    descale_q: cute.Tensor,
    descale_k: cute.Tensor,
    descale_v: cute.Tensor,
    descale_s: cute.Tensor,
    scale_s: cute.Tensor,
    descale_do: cute.Tensor,
    descale_dp: cute.Tensor,
    scale_dk: cute.Tensor,
    scale_dv: cute.Tensor,
    scale_dp: cute.Tensor,
    amax_dk: Optional[cute.Tensor],
    amax_dv: Optional[cute.Tensor],
    amax_dp: Optional[cute.Tensor],
    scale: cutlass.Float32,
    causal: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
    ratio: cutlass.Constexpr[int] = 1,
):
    scales = (descale_q, descale_k, descale_v, descale_s, scale_s, descale_do, descale_dp, scale_dk, scale_dv, scale_dp)
    host(q, k, v, do, lse, delta, dk, dv, scales, amax_dk, amax_dv, amax_dp, scale, causal, WINDOW, False, stream)


@cute.jit
def dq_host(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    do: cute.Tensor,
    lse: cute.Tensor,
    delta: cute.Tensor,
    dq: cute.Tensor,
    descale_q: cute.Tensor,
    descale_k: cute.Tensor,
    descale_v: cute.Tensor,
    descale_s: cute.Tensor,
    scale_s: cute.Tensor,
    descale_do: cute.Tensor,
    descale_dp: cute.Tensor,
    scale_dq: cute.Tensor,
    scale_dp: cute.Tensor,
    amax_dq: Optional[cute.Tensor],
    amax_dp: Optional[cute.Tensor],
    scale: cutlass.Float32,
    causal: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
    ratio: cutlass.Constexpr[int] = 1,
):
    scales = (descale_q, descale_k, descale_v, descale_s, scale_s, descale_do, descale_dp, scale_dq, scale_dq, scale_dp)
    host(q, k, v, do, lse, delta, dq, dq, scales, amax_dq, None, amax_dp, scale, causal, WINDOW, True, stream)
