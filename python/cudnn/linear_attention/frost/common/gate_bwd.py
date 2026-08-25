# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Safe-gate backward helper: convert the main backward kernels' dGate
(gradient wrt the transformed log-decay) into the raw-logit gradient in
place, and produce the gate-parameter gradients dA_log / ddt_bias.

Two forms:
  - scalar (GDN):       g = -exp(A_log[h]) * softplus(g_raw + dt_bias[h])
  - per-channel (KDA):  g = lb * sigmoid(exp(a_log) * (g_raw + dt_bias))

The parameter gradients are cross-token sums, done deterministically: every
fp32 accumulator owns a statically assigned (head[, channel], token-slice)
and partials are combined through fixed-shape trees only — the bracketing is
a pure function of the shapes, never of scheduling, so results are bitwise
stable run to run.  Two stages with a launch boundary as the grid barrier:
a partial pass over token stripes (the same pass rewrites dGate -> dg_raw),
then a finisher that folds the stripe partials.

Scalar stripes scale with the token axis (:func:`scalar_gate_blocks`, capped)
so the partial pass fills the machine; channel stripes stay at
``GATE_BWD_BLOCKS`` (the partials carve is ``128 * HO * 128`` fp32).
"""

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.experimental.primitives as nvvm
from cutlass.cute.arch.nvvm_wrappers import inline_ptx
from cutlass.cute.runtime import from_dlpack

from cudnn.frost.buffers import data_ptr
from cudnn.frost.tile_dsl.pointwise import fadd2, ffma2, fmul2, lane_group_sum, sigmoid, sigmoid2, softplus

GATE_BWD_BLOCKS = 128  # channel-gate token stripes (partials carve = 128 * HO * 128 fp32)
SCALAR_BLOCK_CAP = 8192  # scalar-gate stripe ceiling
SCALAR_SLICE_TOKENS = 16  # scalar-gate target tokens per stripe
SCALAR_HEAD_TILE = 32  # scalar-gate heads per block: one warp, tiled over grid.y


def scalar_gate_blocks(n_tokens: int) -> int:
    """Scalar-gate stripe count: shape-only (so the summation bracketing is
    deterministic) and scaling with the token axis so the partial pass runs
    memory-bound instead of 128 * HO threads walking long serial slices."""
    return min(SCALAR_BLOCK_CAP, max(1, -(-n_tokens // SCALAR_SLICE_TOKENS)))


@cute.kernel
def scalar_gate_bwd_partial_kernel(
    mDg: cute.Tensor,
    mG: cute.Tensor,
    mALog: cute.Tensor,
    mDtBias: cute.Tensor,
    mPartA: cute.Tensor,
    mPartDt: cute.Tensor,
    n_tokens: cutlass.Int32,
    h_o: cutlass.Int32,
    slice_len: cutlass.Int32,
) -> None:
    """Grid (stripes, head tiles), block (SCALAR_HEAD_TILE,): thread h walks
    stripe g's token slice in order, rewriting dGate -> dg_raw in place and
    accumulating the (g, h) partials for dA_log (dGate * g_transformed) and
    ddt_bias (dg_raw).  Tiling heads over grid.y rather than one thread per
    head takes any HO and keeps a small HO down to a single warp; each
    (stripe, head) still has exactly one owner, so the partial layout and the
    finisher's summation order are unchanged."""
    bid = cute.arch.block_idx()
    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    g_blk = cutlass.Int32(bid[0])
    h = cutlass.Int32(bid[1]) * cutlass.Int32(SCALAR_HEAD_TILE) + tidx
    if h < h_o:
        neg_exp_a = -cute.math.exp(mALog[h], fastmath=True)
        bias = mDtBias[h]
        t = g_blk * slice_len
        t_end = t + slice_len
        if t_end > n_tokens:
            t_end = n_tokens
        acc_a = cutlass.Float32(0.0)
        acc_dt = cutlass.Float32(0.0)
        while t < t_end:
            d_gate = mDg[t, h]
            y = mG[t, h] + bias
            dg_raw = d_gate * neg_exp_a * sigmoid(y)
            acc_a += d_gate * (neg_exp_a * softplus(y))
            acc_dt += dg_raw
            mDg[t, h] = dg_raw
            t += 1
        mPartA[g_blk * h_o + h] = acc_a
        mPartDt[g_blk * h_o + h] = acc_dt


@cute.kernel
def scalar_gate_bwd_finish_kernel(
    mPartA: cute.Tensor,
    mPartDt: cute.Tensor,
    mDA: cute.Tensor,
    mDDt: cute.Tensor,
    h_o: cutlass.Int32,
    n_blocks: cutlass.Int32,
) -> None:
    """Grid (HO,), block (32,): head h's stripe partials fold as 8 fixed
    interleaved chains per lane (independent accumulators so the loads
    pipeline instead of serializing at memory latency), then the 8 chains
    pairwise and one butterfly tree across the 32 lanes — a fixed-shape
    bracketing regardless of the stripe count."""
    bid = cute.arch.block_idx()
    lane_idx = cutlass.Int32(cute.arch.thread_idx()[0])
    h = cutlass.Int32(bid[0])
    a8 = cutlass.Array(cutlass.Float32, 8)
    d8 = cutlass.Array(cutlass.Float32, 8)
    for j in cutlass.range_constexpr(8):
        a8[j] = cutlass.Float32(0.0)
        d8[j] = cutlass.Float32(0.0)
    s = lane_idx
    while s + cutlass.Int32(224) < n_blocks:
        for j in cutlass.range_constexpr(8):
            idx = (s + cutlass.Int32(32 * j)) * h_o + h
            a8[j], d8[j] = fadd2(a8[j], d8[j], mPartA[idx], mPartDt[idx])
        s += cutlass.Int32(256)
    while s < n_blocks:
        idx = s * h_o + h
        a8[0], d8[0] = fadd2(a8[0], d8[0], mPartA[idx], mPartDt[idx])
        s += cutlass.Int32(32)
    # the A and dt chains are independent, so each packed add folds both
    p0a, p0d = fadd2(a8[0], d8[0], a8[1], d8[1])
    p1a, p1d = fadd2(a8[2], d8[2], a8[3], d8[3])
    p2a, p2d = fadd2(a8[4], d8[4], a8[5], d8[5])
    p3a, p3d = fadd2(a8[6], d8[6], a8[7], d8[7])
    q0a, q0d = fadd2(p0a, p0d, p1a, p1d)
    q1a, q1d = fadd2(p2a, p2d, p3a, p3d)
    acc_a, acc_dt = fadd2(q0a, q0d, q1a, q1d)
    acc_a = lane_group_sum(acc_a, 32)
    acc_dt = lane_group_sum(acc_dt, 32)
    if lane_idx == 0:
        mDA[h] = acc_a
        mDDt[h] = acc_dt


@cute.kernel
def channel_gate_bwd_partial_kernel(
    mDg: cute.Tensor,
    mG: cute.Tensor,
    mALog: cute.Tensor,
    mDtBias: cute.Tensor,
    mPartA: cute.Tensor,
    mPartDt: cute.Tensor,
    n_tokens: cutlass.Int32,
    h_o: cutlass.Int32,
    slice_len: cutlass.Int32,
    lower_bound: cutlass.Float32,
) -> None:
    """Grid (GATE_BWD_BLOCKS, HO), block (128,) = 4 token-phased warps of 32
    lanes; lane owns channels [lane*4, lane*4 + 4) through 128-bit loads and
    stores.  Warp w walks tokens t0 + w, t0 + w + 4, ... of the stripe: with
    z = exp(a_log) * (g_raw + dt_bias) and w_ = dGate * lb * sig(z)(1-sig(z)),
    dg_raw = w_ * exp(a_log) (rewritten in place), the dA_log integrand is
    w_ * z, the ddt_bias one dg_raw.  The four warp partials per channel fold
    w = 0..3 through SMEM (fixed order) into the (g, h) partial slots.
    a_log is one scalar per head; dt_bias is per (head, channel)."""
    bid = cute.arch.block_idx()
    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    g_blk = cutlass.Int32(bid[0])
    h = cutlass.Int32(bid[1])
    wrp = tidx // cutlass.Int32(32)
    lane_idx = tidx % cutlass.Int32(32)
    d0 = lane_idx * cutlass.Int32(4)
    exp_a = cute.math.exp(mALog[h], fastmath=True)
    bias = cutlass.Array(cutlass.Float32, 4)
    for q in cutlass.range_constexpr(4):
        bias[q] = mDtBias[h, d0 + cutlass.Int32(q)]
    acc_a = cutlass.Array(cutlass.Float32, 4)
    acc_dt = cutlass.Array(cutlass.Float32, 4)
    out = cutlass.Array(cutlass.Float32, 4)
    for q in cutlass.range_constexpr(4):
        acc_a[q] = cutlass.Float32(0.0)
        acc_dt[q] = cutlass.Float32(0.0)
    dg_base = mDg.iterator.toint()
    g_base = mG.iterator.toint()
    dg_s0 = cutlass.Int64(mDg.stride[0])
    dg_s1 = cutlass.Int64(mDg.stride[1])
    g_s0 = cutlass.Int64(mG.stride[0])
    g_s1 = cutlass.Int64(mG.stride[1])
    t = g_blk * slice_len + wrp
    t_end = g_blk * slice_len + slice_len
    if t_end > n_tokens:
        t_end = n_tokens
    while t < t_end:
        dg_addr = dg_base + (cutlass.Int64(t) * dg_s0 + cutlass.Int64(h) * dg_s1 + cutlass.Int64(d0)) * cutlass.Int64(4)
        g_addr = g_base + (cutlass.Int64(t) * g_s0 + cutlass.Int64(h) * g_s1 + cutlass.Int64(d0)) * cutlass.Int64(4)
        dg0, dg1, dg2, dg3 = inline_ptx(
            "ld.global.v4.f32 {$0, $1, $2, $3}, [$4];",
            write_only_types=[cutlass.Float32, cutlass.Float32, cutlass.Float32, cutlass.Float32],
            read_only_args=[dg_addr],
        )
        gv0, gv1, gv2, gv3 = inline_ptx(
            "ld.global.v4.f32 {$0, $1, $2, $3}, [$4];",
            write_only_types=[cutlass.Float32, cutlass.Float32, cutlass.Float32, cutlass.Float32],
            read_only_args=[g_addr],
        )
        dgv = (dg0, dg1, dg2, dg3)
        gvv = (gv0, gv1, gv2, gv3)
        for p in cutlass.range_constexpr(2):
            i = 2 * p
            j = i + 1
            one = cutlass.Float32(1.0)
            y_lo, y_hi = fadd2(gvv[i], gvv[j], bias[i], bias[j])
            z_lo, z_hi = fmul2(exp_a, exp_a, y_lo, y_hi)
            sig_lo, sig_hi = sigmoid2(z_lo, z_hi)
            c_lo, c_hi = fadd2(one, one, -sig_lo, -sig_hi)
            k_lo, k_hi = fmul2(sig_lo, sig_hi, c_lo, c_hi)
            b_lo, b_hi = fmul2(dgv[i], dgv[j], lower_bound, lower_bound)
            w_lo, w_hi = fmul2(b_lo, b_hi, k_lo, k_hi)
            raw_lo, raw_hi = fmul2(w_lo, w_hi, exp_a, exp_a)
            acc_a[i], acc_a[j] = ffma2(w_lo, w_hi, z_lo, z_hi, acc_a[i], acc_a[j])
            acc_dt[i], acc_dt[j] = fadd2(acc_dt[i], acc_dt[j], raw_lo, raw_hi)
            out[i] = raw_lo
            out[j] = raw_hi
        inline_ptx(
            "st.global.v4.f32 [$0], {$1, $2, $3, $4};",
            read_only_args=[dg_addr, out[0], out[1], out[2], out[3]],
        )
        t += cutlass.Int32(4)
    sA = cutlass.Array(cutlass.Float32, 512, space=cutlass.AddressSpace.smem, alignment=16)
    sDt = cutlass.Array(cutlass.Float32, 512, space=cutlass.AddressSpace.smem, alignment=16)
    for q in cutlass.range_constexpr(4):
        sA[wrp * cutlass.Int32(128) + d0 + cutlass.Int32(q)] = acc_a[q]
        sDt[wrp * cutlass.Int32(128) + d0 + cutlass.Int32(q)] = acc_dt[q]
    nvvm.barrier_cta_sync()
    if wrp == 0:
        for q in cutlass.range_constexpr(4):
            d = d0 + cutlass.Int32(q)
            fa = ((sA[d] + sA[cutlass.Int32(128) + d]) + sA[cutlass.Int32(256) + d]) + sA[cutlass.Int32(384) + d]
            fdt = ((sDt[d] + sDt[cutlass.Int32(128) + d]) + sDt[cutlass.Int32(256) + d]) + sDt[cutlass.Int32(384) + d]
            base = (g_blk * h_o + h) * cutlass.Int32(128) + d
            mPartA[base] = fa
            mPartDt[base] = fdt


@cute.kernel
def channel_gate_bwd_finish_kernel(
    mPartA: cute.Tensor,
    mPartDt: cute.Tensor,
    mDA: cute.Tensor,
    mDDt: cute.Tensor,
    h_o: cutlass.Int32,
) -> None:
    """Grid (HO,), block (128,): thread d folds its channel column over the
    GATE_BWD_BLOCKS stripe partials as 8 fixed interleaved chains
    (independent accumulators so the loads pipeline) folded pairwise.
    ddt_bias is per (head, channel), so thread d stores its column outright;
    dA_log is per head, so the channel axis folds through a fixed tree (warp
    butterflies, then the 4 warp sums in index order)."""
    bid = cute.arch.block_idx()
    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    h = cutlass.Int32(bid[0])
    d = tidx
    wrp = tidx // cutlass.Int32(32)
    lane_idx = tidx % cutlass.Int32(32)
    a8 = cutlass.Array(cutlass.Float32, 8)
    d8 = cutlass.Array(cutlass.Float32, 8)
    for j in cutlass.range_constexpr(8):
        a8[j] = cutlass.Float32(0.0)
        d8[j] = cutlass.Float32(0.0)
    c = cutlass.Int32(0)
    while c < cutlass.Int32(GATE_BWD_BLOCKS // 8):
        for j in cutlass.range_constexpr(8):
            idx = ((c * cutlass.Int32(8) + cutlass.Int32(j)) * h_o + h) * cutlass.Int32(128) + d
            a8[j] = a8[j] + mPartA[idx]
            d8[j] = d8[j] + mPartDt[idx]
        c += 1
    col_a = ((a8[0] + a8[1]) + (a8[2] + a8[3])) + ((a8[4] + a8[5]) + (a8[6] + a8[7]))
    col_dt = ((d8[0] + d8[1]) + (d8[2] + d8[3])) + ((d8[4] + d8[5]) + (d8[6] + d8[7]))
    mDDt[h, d] = col_dt
    sWa = cutlass.Array(cutlass.Float32, 4, space=cutlass.AddressSpace.smem, alignment=16)
    va = lane_group_sum(col_a, 32)
    if lane_idx == 0:
        sWa[wrp] = va
    nvvm.barrier_cta_sync()
    if tidx == 0:
        mDA[h] = ((sWa[0] + sWa[1]) + sWa[2]) + sWa[3]


@cute.jit
def scalar_gate_bwd_launch(
    d_gate: cute.Tensor,
    g_raw: cute.Tensor,
    a_log: cute.Tensor,
    dt_bias: cute.Tensor,
    part_a: cute.Tensor,
    part_dt: cute.Tensor,
    d_a_log: cute.Tensor,
    d_dt_bias: cute.Tensor,
    n_tokens: cutlass.Int32,
    h_o: cutlass.Int32,
    slice_len: cutlass.Int32,
    n_blocks: cutlass.Int32,
    head_tiles: cutlass.Int32,
    stream: cuda.CUstream,
):
    scalar_gate_bwd_partial_kernel(d_gate, g_raw, a_log, dt_bias, part_a, part_dt, n_tokens, h_o, slice_len).launch(
        grid=(n_blocks, head_tiles, 1), block=(SCALAR_HEAD_TILE, 1, 1), stream=stream
    )
    scalar_gate_bwd_finish_kernel(part_a, part_dt, d_a_log, d_dt_bias, h_o, n_blocks).launch(grid=(h_o, 1, 1), block=(32, 1, 1), stream=stream)


@cute.jit
def channel_gate_bwd_launch(
    d_gate: cute.Tensor,
    g_raw: cute.Tensor,
    a_log: cute.Tensor,
    dt_bias: cute.Tensor,
    part_a: cute.Tensor,
    part_dt: cute.Tensor,
    d_a_log: cute.Tensor,
    d_dt_bias: cute.Tensor,
    n_tokens: cutlass.Int32,
    h_o: cutlass.Int32,
    slice_len: cutlass.Int32,
    lower_bound: cutlass.Float32,
    stream: cuda.CUstream,
):
    channel_gate_bwd_partial_kernel(d_gate, g_raw, a_log, dt_bias, part_a, part_dt, n_tokens, h_o, slice_len, lower_bound).launch(
        grid=(GATE_BWD_BLOCKS, h_o, 1), block=(128, 1, 1), stream=stream
    )
    channel_gate_bwd_finish_kernel(part_a, part_dt, d_a_log, d_dt_bias, h_o).launch(grid=(h_o, 1, 1), block=(128, 1, 1), stream=stream)


@functools.cache
def gate_bwd_cache(key):
    return {}


def scalar_gate_bwd(d_gate, g_raw, a_log, dt_bias, d_a_log, d_dt_bias, part_a, part_dt, *, stream):
    """Scalar-gate backward: rewrite d_gate [total, HO] fp32 in place from
    transformed-space to raw-logit space and fill d_a_log/d_dt_bias (HO,).
    part_a/part_dt are (scalar_gate_blocks(total) * HO,) fp32 workspace carves."""
    n_tokens, h_o = (int(s_) for s_ in d_gate.shape)
    n_blocks = scalar_gate_blocks(n_tokens)
    head_tiles = -(-h_o // SCALAR_HEAD_TILE)
    slice_len = (n_tokens + n_blocks - 1) // n_blocks
    for name, buf in (("part_a", part_a), ("part_dt", part_dt)):
        if int(buf.shape[0]) < n_blocks * h_o:
            raise ValueError(f"{name} must hold scalar_gate_blocks({n_tokens}) * {h_o} = {n_blocks * h_o} fp32; got {int(buf.shape[0])}")
    cache = gate_bwd_cache(("gdn",))
    cu_stream = cuda.CUstream(int(stream))
    tensors = (d_gate, g_raw, a_log, dt_bias, part_a, part_dt, d_a_log, d_dt_bias)
    if "compiled" not in cache:
        traced = [from_dlpack(t, assumed_align=4) for t in tensors[:2]]
        traced = [tr.mark_layout_dynamic(leading_dim=1) for tr in traced]
        traced += [from_dlpack(t, assumed_align=4).mark_layout_dynamic(leading_dim=0) for t in tensors[2:]]
        cache["compiled"] = cute.compile(
            scalar_gate_bwd_launch,
            *traced,
            cutlass.Int32(n_tokens),
            cutlass.Int32(h_o),
            cutlass.Int32(slice_len),
            cutlass.Int32(n_blocks),
            cutlass.Int32(head_tiles),
            cu_stream,
            options="--enable-tvm-ffi",
        )
    cache["compiled"](*tensors, n_tokens, h_o, slice_len, n_blocks, head_tiles, cu_stream)


def channel_gate_bwd(d_gate, g_raw, a_log, dt_bias, d_a_log, d_dt_bias, part_a, part_dt, gate_lower_bound, *, stream):
    """Per-channel-gate backward: rewrite d_gate [total, HO, 128] fp32 in
    place and fill d_a_log/d_dt_bias at their parameter shapes ((HO,) params
    get the channel axis folded in the finisher)."""
    n_tokens, h_o, d_k = (int(s_) for s_ in d_gate.shape)
    if d_k != 128:
        raise ValueError(f"per-channel gate backward requires 128 channels; got {d_k}")
    for name, t in (("d_gate", d_gate), ("g_raw", g_raw)):
        st = tuple(int(s_) for s_ in t.stride())
        if st[2] != 1 or st[0] % 4 != 0 or st[1] % 4 != 0 or data_ptr(t) % 16 != 0:
            raise ValueError(f"{name} needs a 16B-aligned base, unit channel stride, and outer strides in multiples of 4; got {st}")
    for name, t in (("a_log", a_log), ("d_a_log", d_a_log)):
        if tuple(int(s_) for s_ in t.shape) != (h_o,) or tuple(int(s_) for s_ in t.stride()) != (1,):
            raise ValueError(f"{name} must be a contiguous ({h_o},) per-head fp32 parameter; got shape {tuple(t.shape)}")
    for name, t in (("dt_bias", dt_bias), ("d_dt_bias", d_dt_bias)):
        if tuple(int(s_) for s_ in t.shape) != (h_o, d_k) or tuple(int(s_) for s_ in t.stride()) != (d_k, 1):
            raise ValueError(f"{name} must be a contiguous ({h_o}, {d_k}) per-channel fp32 parameter; got shape {tuple(t.shape)}")
    slice_len = (n_tokens + GATE_BWD_BLOCKS - 1) // GATE_BWD_BLOCKS
    cache = gate_bwd_cache(("channel",))
    cu_stream = cuda.CUstream(int(stream))
    tensors = (d_gate, g_raw, a_log, dt_bias, part_a, part_dt, d_a_log, d_dt_bias)
    args = (n_tokens, h_o, slice_len)
    if "compiled" not in cache:
        traced = [from_dlpack(t, assumed_align=16).mark_layout_dynamic(leading_dim=2) for t in tensors[:2]]
        traced += [from_dlpack(t, assumed_align=4).mark_layout_dynamic(leading_dim=len(t.shape) - 1) for t in tensors[2:]]
        cache["compiled"] = cute.compile(
            channel_gate_bwd_launch,
            *traced,
            *(cutlass.Int32(a) for a in args),
            cutlass.Float32(gate_lower_bound),
            cu_stream,
            options="--enable-tvm-ffi",
        )
    cache["compiled"](*tensors, *args, float(gate_lower_bound), cu_stream)
