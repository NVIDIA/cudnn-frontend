# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""SM120 MXFP8 backward: native block-scaled MMA with canonical SF reads.

Each warp owns 16 gradient rows and 128 columns. dS is quantized per 32-key
block; each gradient output has a fixed owner and GQA reduction order.
"""

from typing import Optional
import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as cuda
from cutlass.base_dsl.typing import Pointer
from cutlass.experimental import primitives as nvvm
from cudnn.frost.tile_dsl.mma import mma_m16n8k32_mxf8_f32
from cudnn.frost.tile_dsl.pointwise import fmax_f32, fp32_to_fp8x2, amax_to_ue8m0_rp
from cudnn.frost.tile_dsl.sf_layout import sf_atom_byte, sf_atom_offset
from cudnn.sdpa.bwd.kernels.sm120.bprop_fp8 import _word, _column_word, _store

FP8 = cutlass.Float8E4M3FN
SPEC = globals().get("FROST_TEMPLATE_PARAMS", (128, 128, False, True, 0, "sm120"))
WINDOW = int(SPEC[4])
M, N, C = 16, 32, 128


@cute.jit
def _sf_row(sf, batch, head, row, block, live_rows):
    value = cutlass.Int32(127)
    if row < live_rows:
        offset = (
            cutlass.Int64(batch) * sf.stride[0]
            + cutlass.Int64(head) * sf.stride[1]
            + cutlass.Int64(row // 128) * sf.stride[2]
            + sf_atom_offset(row % 128, block)
        )
        value = cutlass.Int32((sf.iterator.raw_ptr() + offset).load().bitcast(cutlass.Uint8))
    return value


@cute.jit
def _sf_column(sf, batch, head, heads, col, row, live_cols):
    value = cutlass.Int32(127)
    if col < live_cols:
        offset = (
            cutlass.Int64(col // 128) * sf.stride[0]
            + (cutlass.Int64(batch) * heads + head) * sf.stride[1]
            + cutlass.Int64(row // 128) * sf.stride[2]
            + sf_atom_byte(col % 128, row // 32 % 4)
        )
        value = cutlass.Int32((sf.iterator.raw_ptr() + offset).load().bitcast(cutlass.Uint8))
    return value


@cute.jit
def _score(a, b, sfa, sfb, batch, ahead, bhead, ar, br):
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
            sa = _sf_row(sfa, batch, ahead, ar + r + 8 * (lane % 2), kc, a.shape[1])
            sb = _sf_row(sfb, batch, bhead, br + nf * 8 + r, kc, b.shape[1])
            vals = mma_m16n8k32_mxf8_f32(*av, b0, b1, *[accum[nf * 4 + i] for i in range(4)], sa, sb, 0, 0, FP8)
            for i in cutlass.range_constexpr(4):
                accum[nf * 4 + i] = vals[i]
    return accum


@cute.jit
def _grad(accum, a, b, sf_a, sf_b, batch, head, row, chunk, probability: cutlass.Constexpr[bool]):
    lane = cute.arch.lane_idx()
    r, c = lane // 4, lane % 4 * 4
    av = [Pointer(a.subview((r + (8 if i % 2 else 0)) * N + c + (16 if i >= 2 else 0)).data_ptr(), dtype=cutlass.Int32).load() for i in range(4)]
    sa = cutlass.Int32(119)
    if cutlass.const_expr(not probability):
        sa = cutlass.Int32(sf_a[r + 8 * (lane % 2)])
    for nf in cutlass.range_constexpr(C // 8):
        col = chunk * C + nf * 8 + r
        b0 = _column_word(b, batch, row + c, head, col)
        b1 = _column_word(b, batch, row + c + 16, head, col)
        sb = _sf_column(sf_b, batch, head, b.shape[2], col, row, b.shape[3])
        vals = mma_m16n8k32_mxf8_f32(*av, b0, b1, *[accum[nf * 4 + i] for i in range(4)], sa, sb, 0, 0, FP8)
        for i in cutlass.range_constexpr(4):
            accum[nf * 4 + i] = vals[i]


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
    qt,
    kt,
    dot,
    sf_q,
    sf_k,
    sf_v,
    sf_do,
    sf_qt,
    sf_kt,
    sf_dot,
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
    sf_ds = cutlass.Array(cutlass.Uint8, M, alignment=16, space=cutlass.AddressSpace.smem)
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
                scores = _score(q, k, sf_q, sf_k, batch, hq, hk, owner * M, other * N)
                deriv = _score(do, v, sf_do, sf_v, batch, hq, hk, owner * M, other * N)
            else:
                scores = _score(k, q, sf_k, sf_q, batch, hk, hq, owner * M, other * N)
                deriv = _score(v, do, sf_v, sf_do, batch, hk, hq, owner * M, other * N)
            ds_values = cutlass.Array(cutlass.Float32, 16, alignment=16)
            p_values = cutlass.Array(cutlass.Float32, 16, alignment=16)
            for nf in cutlass.range_constexpr(4):
                for half in cutlass.range_constexpr(2):
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
                        idx = nf * 4 + half * 2 + j
                        if live:
                            prob = cute.math.exp2(
                                (scores[idx] * scale - lse[batch, hq, qi]) * cutlass.Float32(1.4426950408889634), fastmath=True
                            ) * cutlass.Float32(256.0)
                            dsval = (deriv[idx] - delta[batch, hq, qi]) * (scale * cutlass.Float32(1.0 / 256.0)) * prob
                        ds_values[idx], p_values[idx] = dsval, prob
            for half in cutlass.range_constexpr(2):
                maximum = cutlass.Float32(0.0)
                for nf in cutlass.range_constexpr(4):
                    for j in cutlass.range_constexpr(2):
                        maximum = fmax_f32(maximum, cute.math.abs(ds_values[nf * 4 + half * 2 + j]))
                for step in cutlass.range_constexpr(2):
                    other_max = nvvm.shfl_sync(thread_mask=0xFFFFFFFF, val=maximum, offset=1 << step, mask_and_clamp=0x1F, kind=nvvm.Shfl.BFLY)
                    maximum = fmax_f32(maximum, other_max)
                exponent, inverse = amax_to_ue8m0_rp(maximum)
                r = lane // 4 + half * 8
                if lane % 4 == 0:
                    sf_ds[r] = cutlass.Uint8(exponent)
                for nf in cutlass.range_constexpr(4):
                    col = nf * 8 + lane % 4 * 2
                    i = nf * 4 + half * 2
                    Pointer(ds.subview(r * N + col).data_ptr(), dtype=cutlass.Uint16).store(fp32_to_fp8x2(ds_values[i] * inverse, ds_values[i + 1] * inverse))
                    if cutlass.const_expr(not dq):
                        Pointer(p.subview(r * N + col).data_ptr(), dtype=cutlass.Uint16).store(fp32_to_fp8x2(p_values[i], p_values[i + 1]))
            nvvm.bar_warp_sync(cute.arch.FULL_MASK)
            if cutlass.const_expr(dq):
                _grad(grad, ds, kt, sf_ds, sf_kt, batch, hk, other * N, chunk, False)
            else:
                _grad(grad, ds, qt, sf_ds, sf_qt, batch, hq, other * N, chunk, False)
                _grad(gradv, p, dot, sf_ds, sf_dot, batch, hq, other * N, chunk, True)
            nvvm.bar_warp_sync(cute.arch.FULL_MASK)
    _store(grad, dg, batch, head, owner, chunk, cutlass.Float32(1.0), cutlass.Float32(1.0), None)
    if cutlass.const_expr(not dq):
        _store(gradv, dv, batch, head, owner, chunk, cutlass.Float32(1.0), cutlass.Float32(1.0), None)


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
    qt,
    kt,
    dot,
    sf_q,
    sf_k,
    sf_v,
    sf_do,
    sf_qt,
    sf_kt,
    sf_dot,
    scale,
    causal: cutlass.Constexpr[bool],
    window: cutlass.Constexpr[int],
    dq: cutlass.Constexpr[bool],
    stream,
):
    rows = q.shape[1] if cutlass.const_expr(dq) else k.shape[1]
    heads = q.shape[2] if cutlass.const_expr(dq) else k.shape[2]
    chunks = (q.shape[3] + C - 1) // C
    kernel(q, k, v, do, lse, delta, dg, dv, qt, kt, dot, sf_q, sf_k, sf_v, sf_do, sf_qt, sf_kt, sf_dot, scale, causal, window, dq).launch(
        grid=((rows + M - 1) // M * chunks, heads, q.shape[0]), block=(32, 1, 1), stream=stream
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
    descale_q: Optional[cute.Tensor],
    descale_k: Optional[cute.Tensor],
    descale_v: Optional[cute.Tensor],
    descale_s: Optional[cute.Tensor],
    scale_s: Optional[cute.Tensor],
    descale_do: Optional[cute.Tensor],
    descale_dp: Optional[cute.Tensor],
    scale_dk: Optional[cute.Tensor],
    scale_dv: Optional[cute.Tensor],
    scale_dp: Optional[cute.Tensor],
    amax_dk: Optional[cute.Tensor],
    amax_dv: Optional[cute.Tensor],
    amax_dp: Optional[cute.Tensor],
    scale: cutlass.Float32,
    causal: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
    ratio: cutlass.Constexpr[int] = 1,
    qt: Optional[cute.Tensor] = None,
    dot: Optional[cute.Tensor] = None,
    sf_q: Optional[cute.Tensor] = None,
    sf_qt: Optional[cute.Tensor] = None,
    sf_k: Optional[cute.Tensor] = None,
    sf_v: Optional[cute.Tensor] = None,
    sf_do: Optional[cute.Tensor] = None,
    sf_dot: Optional[cute.Tensor] = None,
):
    host(q, k, v, do, lse, delta, dk, dv, qt, k, dot, sf_q, sf_k, sf_v, sf_do, sf_qt, sf_k, sf_dot, scale, causal, WINDOW, False, stream)


@cute.jit
def dq_host(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    do: cute.Tensor,
    lse: cute.Tensor,
    delta: cute.Tensor,
    dq: cute.Tensor,
    descale_q: Optional[cute.Tensor],
    descale_k: Optional[cute.Tensor],
    descale_v: Optional[cute.Tensor],
    descale_s: Optional[cute.Tensor],
    scale_s: Optional[cute.Tensor],
    descale_do: Optional[cute.Tensor],
    descale_dp: Optional[cute.Tensor],
    scale_dq: Optional[cute.Tensor],
    scale_dp: Optional[cute.Tensor],
    amax_dq: Optional[cute.Tensor],
    amax_dp: Optional[cute.Tensor],
    scale: cutlass.Float32,
    causal: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
    ratio: cutlass.Constexpr[int] = 1,
    kt: Optional[cute.Tensor] = None,
    sf_q: Optional[cute.Tensor] = None,
    sf_do: Optional[cute.Tensor] = None,
    sf_k: Optional[cute.Tensor] = None,
    sf_kt: Optional[cute.Tensor] = None,
    sf_v: Optional[cute.Tensor] = None,
):
    host(q, k, v, do, lse, delta, dq, dq, q, kt, do, sf_q, sf_k, sf_v, sf_do, sf_q, sf_kt, sf_do, scale, causal, WINDOW, True, stream)
