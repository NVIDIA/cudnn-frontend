# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Safe-gate backward helper: convert the main backward kernels' dGate
(gradient wrt the transformed log-decay) into the raw-logit gradient in
place, and produce the gate-parameter gradients dA_log / ddt_bias.

Three forms:
  - scalar (GDN):       g = -exp(A_log[h]) * softplus(g_raw + dt_bias[h])
  - per-channel (KDA):  g = lb * sigmoid(exp(a_log) * (g_raw + dt_bias))
  - Mamba-2:            dt = softplus(g_raw + dt_bias[h]) is both the decay exponent (A[h] dt) and the write strength;
                        its raw-dt gradient comes from the bprop's row dots and exit dots per chunk slot, with the
                        dA / d_dt_bias / dD partials; the z-gate backward (dO silu(z), dZ) runs ahead of the bprop

a_log and dt_bias are each optional (an absent a_log means exp(a_log) = 1, an
absent dt_bias means bias 0); an absent parameter's gradient and partial carve
are None and not produced, while the in-place dGate rewrite always runs.

The parameter gradients are cross-token sums, done deterministically: every
fp32 accumulator owns a statically assigned (head[, channel], token-slice)
and partials are combined through fixed-shape trees only, the bracketing is
a pure function of the shapes, never of scheduling, so results are bitwise
stable run to run.  Two stages with a launch boundary as the grid barrier:
a partial pass over token stripes (the same pass rewrites dGate -> dg_raw),
then a sum kernel that folds the stripe partials.
"""

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.experimental.primitives as nvvm
from cutlass.cute.runtime import from_dlpack

from cudnn.frost.tile_dsl.pointwise import (
    f16x2_to_f32,
    fadd2,
    ffma2,
    fmul2,
    fp32_to_fp16,
    lane_group_sum,
    opaque_f32_zero,
    sigmoid,
    sigmoid2,
    softplus,
)
from cudnn.frost.tile_dsl.barrier import wait_on_dependent_grids
from cudnn.frost.tile_dsl.tma import ld_global_v2, ld_global_v4, st_global_v2, st_global_v4

from .host import get_dtype
from cudnn.frost.device import current_device

USE_PDL = True

GATE_BWD_BLOCKS = 128  # channel-gate token stripes (partials carve = GATE_BWD_BLOCKS * HO * d_k fp32)
CHANNEL_BLOCK = 128  # channel-gate partial-kernel CTA threads
SCALAR_BLOCK_CAP = 8192  # scalar-gate stripe ceiling
SCALAR_SLICE_TOKENS = 16  # scalar-gate target tokens per stripe
SCALAR_HEAD_TILE = 32  # scalar-gate heads per block: one warp, tiled over grid.y
MAMBA2_GATE_HEADS = 16  # heads per gate backward partial CTA
MAMBA2_GATE_THREADS = 512  # its threads: 128 tokens of a chunk by 4 heads per thread
MAMBA2_SKIP_SUM_GROUPS = 8  # row groups of the per-channel D-skip sum, each d_v threads
MAMBA2_ZGATE_THREADS = 256


def scalar_gate_blocks(n_tokens: int) -> int:
    """Scalar-gate stripe count: shape-only (so the summation bracketing is
    deterministic) and scaling with the token axis."""
    return min(SCALAR_BLOCK_CAP, max(1, -(-n_tokens // SCALAR_SLICE_TOKENS)))


@cute.kernel
def frost_scalar_gate_bwd_partial(
    mDg: cute.Tensor,
    mG: cute.Tensor,
    mALog: cute.Tensor | None,
    mDtBias: cute.Tensor | None,
    mPartA: cute.Tensor | None,
    mPartDt: cute.Tensor | None,
    n_tokens: cutlass.Int32,
    h_o: cutlass.Int32,
    slice_len: cutlass.Int32,
) -> None:
    """Grid (stripes, head tiles), block (SCALAR_HEAD_TILE,): thread h walks
    stripe g's token slice in order, rewriting dGate -> dg_raw in place and
    accumulating the (g, h) partials for dA_log (dGate * g_transformed) and
    ddt_bias (dg_raw).  mPartA is None iff mALog is, mPartDt iff mDtBias is."""
    if cutlass.const_expr(USE_PDL):
        wait_on_dependent_grids()
    bid = cute.arch.block_idx()
    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    stripe_idx = cutlass.Int32(bid[0])
    h = cutlass.Int32(bid[1]) * cutlass.Int32(SCALAR_HEAD_TILE) + tidx
    if h < h_o:
        if cutlass.const_expr(mALog is not None):
            neg_exp_a = -cute.math.exp(mALog[h].to(cutlass.Float32), fastmath=True)
        else:
            neg_exp_a = -(opaque_f32_zero() + cutlass.Float32(1.0))
        if cutlass.const_expr(mDtBias is not None):
            bias = mDtBias[h].to(cutlass.Float32)
        else:
            bias = opaque_f32_zero()
        t = stripe_idx * slice_len
        t_end = t + slice_len
        if t_end > n_tokens:
            t_end = n_tokens
        acc_a = cutlass.Float32(0.0)
        acc_dt = cutlass.Float32(0.0)
        dg_ptr = mDg.iterator.raw_ptr()
        g_ptr = mG.iterator.raw_ptr()
        dg_off = cutlass.Int64(t) * cutlass.Int64(mDg.stride[0]) + cutlass.Int64(h) * cutlass.Int64(mDg.stride[1])
        g_off = cutlass.Int64(t) * cutlass.Int64(mG.stride[0]) + cutlass.Int64(h) * cutlass.Int64(mG.stride[1])
        dg_step = cutlass.Int64(mDg.stride[0])
        g_step = cutlass.Int64(mG.stride[0])
        while t < t_end:
            d_gate = (dg_ptr + dg_off).load().to(cutlass.Float32)
            y = (g_ptr + g_off).load().to(cutlass.Float32) + bias
            dg_raw = d_gate * neg_exp_a * sigmoid(y)
            if cutlass.const_expr(mPartA is not None):
                acc_a += d_gate * (neg_exp_a * softplus(y))
            if cutlass.const_expr(mPartDt is not None):
                acc_dt += dg_raw
            (dg_ptr + dg_off).store(dg_raw.to(mDg.element_type))
            t += 1
            dg_off += dg_step
            g_off += g_step
        if cutlass.const_expr(mPartA is not None):
            mPartA[stripe_idx * h_o + h] = acc_a
        if cutlass.const_expr(mPartDt is not None):
            mPartDt[stripe_idx * h_o + h] = acc_dt


@cute.kernel
def frost_scalar_gate_bwd_sum(
    mPartA: cute.Tensor | None,
    mPartDt: cute.Tensor | None,
    mDA: cute.Tensor | None,
    mDDt: cute.Tensor | None,
    h_o: cutlass.Int32,
    n_blocks: cutlass.Int32,
) -> None:
    """Grid (HO,), block (32,): head h's stripe partials fold as 8 fixed
    interleaved chains per lane (independent accumulators, four rounds of
    loads in flight), then the 8 chains pairwise and one butterfly tree across
    the 32 lanes, a fixed-shape bracketing regardless of the stripe count.
    mDA is None iff mPartA is, mDDt iff mPartDt is; at least one pair is
    present."""
    if cutlass.const_expr(USE_PDL):
        wait_on_dependent_grids()
    bid = cute.arch.block_idx()
    lane_idx = cutlass.Int32(cute.arch.thread_idx()[0])
    h = cutlass.Int32(bid[0])
    a8 = cutlass.Array(cutlass.Float32, 8)
    d8 = cutlass.Array(cutlass.Float32, 8)
    for j in cutlass.range_constexpr(8):
        a8[j] = cutlass.Float32(0.0)
        d8[j] = cutlass.Float32(0.0)
    s = lane_idx
    row_step = cutlass.Int32(32) * h_o
    base = s * h_o + h
    a_values = cutlass.Array(cutlass.Float32, 32)
    d_values = cutlass.Array(cutlass.Float32, 32)
    # ---- four rounds of loads in flight before their adds; chain j keeps rows s + 32 j + 256 r in r order -------
    while s + cutlass.Int32(992) < n_blocks:
        for r in cutlass.range_constexpr(4):
            for j in cutlass.range_constexpr(8):
                idx = base + cutlass.Int32(8 * r + j) * row_step
                if cutlass.const_expr(mPartA is not None):
                    a_values[8 * r + j] = mPartA[idx]
                else:
                    a_values[8 * r + j] = cutlass.Float32(0.0)
                if cutlass.const_expr(mPartDt is not None):
                    d_values[8 * r + j] = mPartDt[idx]
                else:
                    d_values[8 * r + j] = cutlass.Float32(0.0)
        for r in cutlass.range_constexpr(4):
            for j in cutlass.range_constexpr(8):
                a8[j], d8[j] = fadd2(a8[j], d8[j], a_values[8 * r + j], d_values[8 * r + j])
        s += cutlass.Int32(1024)
        base += cutlass.Int32(32) * row_step
    while s + cutlass.Int32(224) < n_blocks:
        for j in cutlass.range_constexpr(8):
            idx = base + cutlass.Int32(j) * row_step
            if cutlass.const_expr(mPartA is not None):
                part_a_value = mPartA[idx]
            else:
                part_a_value = cutlass.Float32(0.0)
            if cutlass.const_expr(mPartDt is not None):
                part_dt_value = mPartDt[idx]
            else:
                part_dt_value = cutlass.Float32(0.0)
            a8[j], d8[j] = fadd2(a8[j], d8[j], part_a_value, part_dt_value)
        s += cutlass.Int32(256)
        base += cutlass.Int32(8) * row_step
    while s < n_blocks:
        if cutlass.const_expr(mPartA is not None):
            tail_a = mPartA[base]
        else:
            tail_a = cutlass.Float32(0.0)
        if cutlass.const_expr(mPartDt is not None):
            tail_dt = mPartDt[base]
        else:
            tail_dt = cutlass.Float32(0.0)
        a8[0], d8[0] = fadd2(a8[0], d8[0], tail_a, tail_dt)
        s += cutlass.Int32(32)
        base += row_step
    p0a, p0d = fadd2(a8[0], d8[0], a8[1], d8[1])
    p1a, p1d = fadd2(a8[2], d8[2], a8[3], d8[3])
    p2a, p2d = fadd2(a8[4], d8[4], a8[5], d8[5])
    p3a, p3d = fadd2(a8[6], d8[6], a8[7], d8[7])
    q0a, q0d = fadd2(p0a, p0d, p1a, p1d)
    q1a, q1d = fadd2(p2a, p2d, p3a, p3d)
    acc_a, acc_dt = fadd2(q0a, q0d, q1a, q1d)
    acc_a = lane_group_sum(acc_a, 32)
    acc_dt = lane_group_sum(acc_dt, 32)
    if cutlass.const_expr(mDA is not None):
        if lane_idx == 0:
            mDA[h] = acc_a.to(mDA.element_type)
    if cutlass.const_expr(mDDt is not None):
        if lane_idx == 0:
            mDDt[h] = acc_dt.to(mDDt.element_type)


@cute.kernel
def frost_channel_gate_bwd_partial(
    d_k: cutlass.Constexpr[int],
    mDg: cute.Tensor,
    mG: cute.Tensor,
    mALog: cute.Tensor | None,
    mDtBias: cute.Tensor | None,
    mPartA: cute.Tensor | None,
    mPartDt: cute.Tensor | None,
    n_tokens: cutlass.Int32,
    h_o: cutlass.Int32,
    slice_len: cutlass.Int32,
    lower_bound: cutlass.Float32,
) -> None:
    """Grid (GATE_BWD_BLOCKS, HO), block (CHANNEL_BLOCK,) = ``phases``
    token-phased row groups of ``d_k // 4`` lanes; a lane owns channels
    [d0, d0 + 4) through 128-bit loads and stores.  Group p walks tokens
    t0 + p, t0 + p + phases, ... of the stripe: with z = exp(a_log) *
    (g_raw + dt_bias) and w_ = dGate * lb * sig(z)(1-sig(z)), dg_raw =
    w_ * exp(a_log) (rewritten in place), the dA_log integrand is w_ * z,
    the ddt_bias one dg_raw.  The ``phases`` group partials per channel
    fold p = 0..phases-1 through SMEM (fixed order) into the (g, h)
    partial slots.  a_log is one scalar per head; dt_bias is per
    (head, channel).  mPartA is None iff mALog is, mPartDt iff mDtBias is."""
    lanes_per_row = d_k // 4
    phases = CHANNEL_BLOCK // lanes_per_row
    active = phases * lanes_per_row
    if cutlass.const_expr(USE_PDL):
        wait_on_dependent_grids()
    bid = cute.arch.block_idx()
    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    stripe_idx = cutlass.Int32(bid[0])
    h = cutlass.Int32(bid[1])
    phase_id = tidx // cutlass.Int32(lanes_per_row)
    d0 = (tidx % cutlass.Int32(lanes_per_row)) * cutlass.Int32(4)
    sA = cutlass.Array(cutlass.Float32, phases * d_k, space=cutlass.AddressSpace.smem, alignment=16)
    sDt = cutlass.Array(cutlass.Float32, phases * d_k, space=cutlass.AddressSpace.smem, alignment=16)
    if tidx < cutlass.Int32(active):
        if cutlass.const_expr(mALog is not None):
            exp_a = cute.math.exp(mALog[h].to(cutlass.Float32), fastmath=True)
        else:
            exp_a = opaque_f32_zero() + cutlass.Float32(1.0)
        bias = cutlass.Array(cutlass.Float32, 4)
        for q in cutlass.range_constexpr(4):
            if cutlass.const_expr(mDtBias is not None):
                bias[q] = mDtBias[h, d0 + cutlass.Int32(q)].to(cutlass.Float32)
            else:
                bias[q] = opaque_f32_zero()
        acc_a = cutlass.Array(cutlass.Float32, 4)
        acc_dt = cutlass.Array(cutlass.Float32, 4)
        out = cutlass.Array(cutlass.Float32, 4)
        for q in cutlass.range_constexpr(4):
            acc_a[q] = cutlass.Float32(0.0)
            acc_dt[q] = cutlass.Float32(0.0)
        dg_base = mDg.iterator.toint()
        g_base = mG.iterator.toint()
        gate_elem_bytes = cutlass.const_expr(mG.element_type.width // 8)
        dg_s0 = cutlass.Int64(mDg.stride[0])
        dg_s1 = cutlass.Int64(mDg.stride[1])
        g_s0 = cutlass.Int64(mG.stride[0])
        g_s1 = cutlass.Int64(mG.stride[1])
        t = stripe_idx * slice_len + phase_id
        t_end = stripe_idx * slice_len + slice_len
        if t_end > n_tokens:
            t_end = n_tokens
        dg_addr = dg_base + (cutlass.Int64(t) * dg_s0 + cutlass.Int64(h) * dg_s1 + cutlass.Int64(d0)) * cutlass.Int64(gate_elem_bytes)
        g_addr = g_base + (cutlass.Int64(t) * g_s0 + cutlass.Int64(h) * g_s1 + cutlass.Int64(d0)) * cutlass.Int64(gate_elem_bytes)
        dg_addr_step = dg_s0 * cutlass.Int64(phases * gate_elem_bytes)
        g_addr_step = g_s0 * cutlass.Int64(phases * gate_elem_bytes)
        while t < t_end:
            if cutlass.const_expr(mG.element_type == cutlass.Float32):
                dgv = ld_global_v4(dg_addr, cutlass.Float32)
                gvv = ld_global_v4(g_addr, cutlass.Float32)
            else:
                dw0, dw1 = ld_global_v2(dg_addr, cutlass.Int32)
                gw0, gw1 = ld_global_v2(g_addr, cutlass.Int32)
                dfrag = cutlass.Vector.from_elements((dw0, dw1), cutlass.Int32).bitcast(mG.element_type).to(cutlass.Float32)
                gfrag = cutlass.Vector.from_elements((gw0, gw1), cutlass.Int32).bitcast(mG.element_type).to(cutlass.Float32)
                dgv = (dfrag[0], dfrag[1], dfrag[2], dfrag[3])
                gvv = (gfrag[0], gfrag[1], gfrag[2], gfrag[3])
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
                if cutlass.const_expr(mPartA is not None):
                    acc_a[i], acc_a[j] = ffma2(w_lo, w_hi, z_lo, z_hi, acc_a[i], acc_a[j])
                if cutlass.const_expr(mPartDt is not None):
                    acc_dt[i], acc_dt[j] = fadd2(acc_dt[i], acc_dt[j], raw_lo, raw_hi)
                out[i] = raw_lo
                out[j] = raw_hi
            if cutlass.const_expr(mG.element_type == cutlass.Float32):
                st_global_v4(dg_addr, out, cutlass.Float32)
            else:
                words = cutlass.Vector.from_elements((out[0], out[1], out[2], out[3]), cutlass.Float32).to(mG.element_type).bitcast(cutlass.Int32)
                st_global_v2(dg_addr, words, cutlass.Int32)
            t += cutlass.Int32(phases)
            dg_addr += dg_addr_step
            g_addr += g_addr_step
        for q in cutlass.range_constexpr(4):
            if cutlass.const_expr(mPartA is not None):
                sA[phase_id * cutlass.Int32(d_k) + d0 + cutlass.Int32(q)] = acc_a[q]
            if cutlass.const_expr(mPartDt is not None):
                sDt[phase_id * cutlass.Int32(d_k) + d0 + cutlass.Int32(q)] = acc_dt[q]
    nvvm.barrier_cta_sync()
    if cutlass.const_expr(mPartA is not None or mPartDt is not None):
        if tidx < cutlass.Int32(lanes_per_row):
            for q in cutlass.range_constexpr(4):
                d = d0 + cutlass.Int32(q)
                base = (stripe_idx * h_o + h) * cutlass.Int32(d_k) + d
                if cutlass.const_expr(mPartA is not None):
                    fa = sA[d]
                    for p in cutlass.range_constexpr(1, phases):
                        fa = fa + sA[cutlass.Int32(p * d_k) + d]
                    mPartA[base] = fa
                if cutlass.const_expr(mPartDt is not None):
                    fdt = sDt[d]
                    for p in cutlass.range_constexpr(1, phases):
                        fdt = fdt + sDt[cutlass.Int32(p * d_k) + d]
                    mPartDt[base] = fdt


@cute.kernel
def frost_channel_gate_bwd_sum(
    d_k: cutlass.Constexpr[int],
    mPartA: cute.Tensor | None,
    mPartDt: cute.Tensor | None,
    mDA: cute.Tensor | None,
    mDDt: cute.Tensor | None,
    h_o: cutlass.Int32,
) -> None:
    """Grid (HO,), block (ceil(d_k/32)*32,): thread d folds its channel
    column over the GATE_BWD_BLOCKS stripe partials as 8 fixed interleaved
    chains (independent accumulators, 32 loads in flight) folded pairwise.
    ddt_bias is per (head, channel), so thread d stores its column outright;
    dA_log is per head, so the channel axis folds through a fixed tree (warp
    butterflies, then the warp sums in index order).  mDA is None iff mPartA
    is, mDDt iff mPartDt is; at least one pair is present."""
    n_warps = (d_k + 31) // 32
    if cutlass.const_expr(USE_PDL):
        wait_on_dependent_grids()
    bid = cute.arch.block_idx()
    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    h = cutlass.Int32(bid[0])
    d = tidx
    warp_id = tidx // cutlass.Int32(32)
    lane_idx = tidx % cutlass.Int32(32)
    a8 = cutlass.Array(cutlass.Float32, 8)
    d8 = cutlass.Array(cutlass.Float32, 8)
    for j in cutlass.range_constexpr(8):
        a8[j] = cutlass.Float32(0.0)
        d8[j] = cutlass.Float32(0.0)
    if d < cutlass.Int32(d_k):
        row_step = h_o * cutlass.Int32(d_k)
        base = h * cutlass.Int32(d_k) + d
        a_values = cutlass.Array(cutlass.Float32, 32)
        d_values = cutlass.Array(cutlass.Float32, 32)
        c = cutlass.Int32(0)
        while c < cutlass.Int32(GATE_BWD_BLOCKS // 32):
            for u in cutlass.range_constexpr(32):
                idx = base + cutlass.Int32(u) * row_step
                if cutlass.const_expr(mPartA is not None):
                    a_values[u] = mPartA[idx]
                if cutlass.const_expr(mPartDt is not None):
                    d_values[u] = mPartDt[idx]
            for u in cutlass.range_constexpr(32):
                if cutlass.const_expr(mPartA is not None):
                    a8[u % 8] = a8[u % 8] + a_values[u]
                if cutlass.const_expr(mPartDt is not None):
                    d8[u % 8] = d8[u % 8] + d_values[u]
            c += 1
            base += cutlass.Int32(32) * row_step
    if cutlass.const_expr(mDDt is not None):
        col_dt = ((d8[0] + d8[1]) + (d8[2] + d8[3])) + ((d8[4] + d8[5]) + (d8[6] + d8[7]))
        if d < cutlass.Int32(d_k):
            mDDt[h, d] = col_dt.to(mDDt.element_type)
    if cutlass.const_expr(mDA is not None):
        col_a = ((a8[0] + a8[1]) + (a8[2] + a8[3])) + ((a8[4] + a8[5]) + (a8[6] + a8[7]))
        sWa = cutlass.Array(cutlass.Float32, n_warps, space=cutlass.AddressSpace.smem, alignment=16)
        va = lane_group_sum(col_a, 32)
        if lane_idx == 0:
            sWa[warp_id] = va
        nvvm.barrier_cta_sync()
        if tidx == 0:
            col = sWa[0]
            for w in cutlass.range_constexpr(1, n_warps):
                col = col + sWa[w]
            mDA[h] = col.to(mDA.element_type)


@cute.jit
def scalar_gate_bwd_launch(
    d_gate: cute.Tensor,
    g_raw: cute.Tensor,
    a_log: cute.Tensor | None,
    dt_bias: cute.Tensor | None,
    part_a: cute.Tensor | None,
    part_dt: cute.Tensor | None,
    d_a_log: cute.Tensor | None,
    d_dt_bias: cute.Tensor | None,
    n_tokens: cutlass.Int32,
    h_o: cutlass.Int32,
    slice_len: cutlass.Int32,
    n_blocks: cutlass.Int32,
    head_tiles: cutlass.Int32,
    stream: cuda.CUstream,
):
    frost_scalar_gate_bwd_partial(d_gate, g_raw, a_log, dt_bias, part_a, part_dt, n_tokens, h_o, slice_len).launch(
        grid=(n_blocks, head_tiles, 1), block=(SCALAR_HEAD_TILE, 1, 1), stream=stream, use_pdl=USE_PDL
    )
    if cutlass.const_expr(d_a_log is not None or d_dt_bias is not None):
        frost_scalar_gate_bwd_sum(part_a, part_dt, d_a_log, d_dt_bias, h_o, n_blocks).launch(grid=(h_o, 1, 1), block=(32, 1, 1), stream=stream, use_pdl=USE_PDL)


@cute.jit
def channel_gate_bwd_launch(
    d_k: cutlass.Constexpr[int],
    d_gate: cute.Tensor,
    g_raw: cute.Tensor,
    a_log: cute.Tensor | None,
    dt_bias: cute.Tensor | None,
    part_a: cute.Tensor | None,
    part_dt: cute.Tensor | None,
    d_a_log: cute.Tensor | None,
    d_dt_bias: cute.Tensor | None,
    n_tokens: cutlass.Int32,
    h_o: cutlass.Int32,
    slice_len: cutlass.Int32,
    lower_bound: cutlass.Float32,
    stream: cuda.CUstream,
):
    frost_channel_gate_bwd_partial(d_k, d_gate, g_raw, a_log, dt_bias, part_a, part_dt, n_tokens, h_o, slice_len, lower_bound).launch(
        grid=(GATE_BWD_BLOCKS, h_o, 1), block=(CHANNEL_BLOCK, 1, 1), stream=stream, use_pdl=USE_PDL
    )
    if cutlass.const_expr(d_a_log is not None or d_dt_bias is not None):
        frost_channel_gate_bwd_sum(d_k, part_a, part_dt, d_a_log, d_dt_bias, h_o).launch(
            grid=(h_o, 1, 1), block=(((d_k + 31) // 32) * 32, 1, 1), stream=stream, use_pdl=USE_PDL
        )


@functools.cache
def gate_bwd_cache(key):
    return {}


def scalar_gate_bwd(d_gate, g_raw, a_log, dt_bias, d_a_log, d_dt_bias, part_a, part_dt, *, stream):
    """Scalar-gate backward: rewrite d_gate [total, HO] (gate dtype) in place from
    transformed-space to raw-logit space and fill d_a_log/d_dt_bias (HO,).
    part_a/part_dt are (scalar_gate_blocks(total) * HO,) fp32 workspace carves.
    a_log/dt_bias may be None (unit amplitude / zero bias); the matching
    d_*/part_* are then None and that gradient is not produced."""
    n_tokens, h_o = (int(dim) for dim in d_gate.shape)
    n_blocks = scalar_gate_blocks(n_tokens)
    head_tiles = -(-h_o // SCALAR_HEAD_TILE)
    slice_len = (n_tokens + n_blocks - 1) // n_blocks
    cache = gate_bwd_cache(
        (
            "gdn",
            str(g_raw.dtype),
            str(a_log.dtype) if a_log is not None else "none",
            str(dt_bias.dtype) if dt_bias is not None else "none",
            d_a_log is not None,
            d_dt_bias is not None,
            part_a is not None,
            part_dt is not None,
            current_device(),
        )
    )
    cu_stream = cuda.CUstream(int(stream))
    tensors = (d_gate, g_raw, a_log, dt_bias, part_a, part_dt, d_a_log, d_dt_bias)
    if "compiled" not in cache:
        traced = [from_dlpack(t, assumed_align=4) for t in tensors[:2]]
        traced = [tr.mark_layout_dynamic(leading_dim=1) for tr in traced]
        traced += [from_dlpack(t, assumed_align=4).mark_layout_dynamic(leading_dim=0) if t is not None else None for t in tensors[2:]]
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
    """Per-channel-gate backward: rewrite d_gate [total, HO, d_k] (gate dtype) in
    place and fill d_a_log/d_dt_bias at their parameter shapes ((HO,) params
    get the channel axis folded in the sum kernel).  a_log/dt_bias may be None
    (unit amplitude / zero bias); the matching d_*/part_* are then None and
    that gradient is not produced."""
    n_tokens, h_o, d_k = (int(dim) for dim in d_gate.shape)
    g_vector_bytes = 4 * (get_dtype(g_raw.dtype).width // 8)
    slice_len = (n_tokens + GATE_BWD_BLOCKS - 1) // GATE_BWD_BLOCKS
    cache = gate_bwd_cache(
        (
            "channel",
            str(g_raw.dtype),
            str(a_log.dtype) if a_log is not None else "none",
            str(dt_bias.dtype) if dt_bias is not None else "none",
            d_a_log is not None,
            d_dt_bias is not None,
            part_a is not None,
            part_dt is not None,
            d_k,
            current_device(),
        )
    )
    cu_stream = cuda.CUstream(int(stream))
    tensors = (d_gate, g_raw, a_log, dt_bias, part_a, part_dt, d_a_log, d_dt_bias)
    args = (n_tokens, h_o, slice_len)
    if "compiled" not in cache:
        traced = [from_dlpack(t, assumed_align=g_vector_bytes).mark_layout_dynamic(leading_dim=2) for t in (d_gate, g_raw)]
        traced += [from_dlpack(t, assumed_align=4).mark_layout_dynamic(leading_dim=len(t.shape) - 1) if t is not None else None for t in tensors[2:]]
        cache["compiled"] = cute.compile(
            channel_gate_bwd_launch,
            d_k,
            *traced,
            *(cutlass.Int32(a) for a in args),
            cutlass.Float32(gate_lower_bound),
            cu_stream,
            options="--enable-tvm-ffi",
        )
    cache["compiled"](*tensors, *args, float(gate_lower_bound), cu_stream)


@cute.jit
def exit_slot_table_body(b_t: cutlass.Constexpr[int], tidx, n_threads: cutlass.Constexpr[int], n_batch: cutlass.Int32, mCu, mSlots):
    """Every thread of one CTA fills the exit-slot table ``mSlots`` [slots, 2]: row ``slot(b, c)`` holds (first token
    of the chunk, ``cu[b + 1]``) and a spare slot past its sequence's chunks holds (-1, 0), so the gate backward partial
    reads one entry instead of searching ``cu``.  Chunk 0 of sequence b sits at slot cu[b] // b_t + 2 b."""
    n_slots = cutlass.Int32(mSlots.shape[0])
    row = cutlass.Int32(tidx)
    while row < n_slots:
        lo = cutlass.Int32(0)
        hi = n_batch - cutlass.Int32(1)
        while lo < hi:
            mid = (lo + hi + cutlass.Int32(1)) // cutlass.Int32(2)
            if cutlass.Int32(mCu[mid]) // cutlass.Int32(b_t) + cutlass.Int32(2) * mid <= row:
                lo = mid
            else:
                hi = mid - cutlass.Int32(1)
        start = cutlass.Int32(mCu[lo])
        end = cutlass.Int32(mCu[lo + cutlass.Int32(1)])
        c = row - (start // cutlass.Int32(b_t) + cutlass.Int32(2) * lo)
        n_chunks = (end - start + cutlass.Int32(b_t - 1)) // cutlass.Int32(b_t)
        filled = c < n_chunks
        mSlots[row, 0] = start + c * cutlass.Int32(b_t) if filled else cutlass.Int32(-1)
        mSlots[row, 1] = end if filled else cutlass.Int32(0)
        row = row + cutlass.Int32(n_threads)


@cute.kernel
def frost_mamba2_zgate_bwd(
    mDoOut: cute.Tensor,
    mZ: cute.Tensor,
    mYpre: cute.Tensor,
    mDoPre: cute.Tensor,
    mDz: cute.Tensor,
    num_vectors: cutlass.Int32,
    d_v: cutlass.Constexpr[int],
    io_dtype: cutlass.Constexpr,
) -> None:
    """CTA column ``blockIdx.y`` owns one head: its threads grid-stride over that head's ``num_vectors = total * d_v /
    vector_width`` 16-byte vectors of the ``[total, HO, d_v]`` tensors, every tensor through its own row and head strides
    (one byte address and one loop-invariant step per tensor), two vectors per step with their six loads in flight before
    the math: dO_pre = dO silu(z) and dz = dO y_pre sigmoid(z) (1 + z (1 - sigmoid(z))), the eight elements of a vector
    as four packed fp32 pairs."""
    if cutlass.const_expr(USE_PDL):
        wait_on_dependent_grids()
    element_bytes = io_dtype.width // 8
    vector_width = 16 // element_bytes
    pairs = vector_width // 2
    vectors_per_row = d_v // vector_width
    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    bidx = cutlass.Int32(cute.arch.block_idx()[0])
    head = cutlass.Int64(cute.arch.block_idx()[1])
    nblk = cutlass.Int32(cute.arch.grid_dim()[0])
    one = opaque_f32_zero() + cutlass.Float32(1.0)
    stride = nblk * cutlass.Int32(MAMBA2_ZGATE_THREADS)
    idx = bidx * cutlass.Int32(MAMBA2_ZGATE_THREADS) + tidx
    row = cutlass.Int64(idx // cutlass.Int32(vectors_per_row))
    col = cutlass.Int64((idx % cutlass.Int32(vectors_per_row)) * cutlass.Int32(vector_width))
    step_rows = cutlass.Int64(stride // cutlass.Int32(vectors_per_row)) * cutlass.Int64(element_bytes)
    do_addr = mDoOut.iterator.toint() + (row * cutlass.Int64(mDoOut.stride[0]) + head * cutlass.Int64(mDoOut.stride[1]) + col) * cutlass.Int64(element_bytes)
    z_addr = mZ.iterator.toint() + (row * cutlass.Int64(mZ.stride[0]) + head * cutlass.Int64(mZ.stride[1]) + col) * cutlass.Int64(element_bytes)
    y_addr = mYpre.iterator.toint() + (row * cutlass.Int64(mYpre.stride[0]) + head * cutlass.Int64(mYpre.stride[1]) + col) * cutlass.Int64(element_bytes)
    do_prime_addr = mDoPre.iterator.toint() + (row * cutlass.Int64(mDoPre.stride[0]) + head * cutlass.Int64(mDoPre.stride[1]) + col) * cutlass.Int64(
        element_bytes
    )
    dz_addr = mDz.iterator.toint() + (row * cutlass.Int64(mDz.stride[0]) + head * cutlass.Int64(mDz.stride[1]) + col) * cutlass.Int64(element_bytes)
    do_step = step_rows * cutlass.Int64(mDoOut.stride[0])
    z_step = step_rows * cutlass.Int64(mZ.stride[0])
    y_step = step_rows * cutlass.Int64(mYpre.stride[0])
    do_prime_step = step_rows * cutlass.Int64(mDoPre.stride[0])
    dz_step = step_rows * cutlass.Int64(mDz.stride[0])
    while idx + stride < num_vectors:
        do_words = [ld_global_v4(do_addr, cutlass.Int32), ld_global_v4(do_addr + do_step, cutlass.Int32)]
        z_words = [ld_global_v4(z_addr, cutlass.Int32), ld_global_v4(z_addr + z_step, cutlass.Int32)]
        y_words = [ld_global_v4(y_addr, cutlass.Int32), ld_global_v4(y_addr + y_step, cutlass.Int32)]
        for u in cutlass.range_constexpr(2):
            do_prime_values = []
            dz_values = []
            for m in cutlass.range_constexpr(pairs):
                do_lo, do_hi = f16x2_to_f32(do_words[u][m], dtype=io_dtype)
                z_lo, z_hi = f16x2_to_f32(z_words[u][m], dtype=io_dtype)
                y_lo, y_hi = f16x2_to_f32(y_words[u][m], dtype=io_dtype)
                sig_lo, sig_hi = sigmoid2(z_lo, z_hi)
                zsig_lo, zsig_hi = fmul2(z_lo, z_hi, sig_lo, sig_hi)
                do_prime_lo, do_prime_hi = fmul2(do_lo, do_hi, zsig_lo, zsig_hi)
                rest_lo, rest_hi = fadd2(one, one, -sig_lo, -sig_hi)
                inner_lo, inner_hi = ffma2(rest_lo, rest_hi, z_lo, z_hi, one, one)
                slope_lo, slope_hi = fmul2(sig_lo, sig_hi, inner_lo, inner_hi)
                doy_lo, doy_hi = fmul2(do_lo, do_hi, y_lo, y_hi)
                dz_lo, dz_hi = fmul2(doy_lo, doy_hi, slope_lo, slope_hi)
                do_prime_values += [do_prime_lo, do_prime_hi]
                dz_values += [dz_lo, dz_hi]
            do_prime_words = [fp32_to_fp16(do_prime_values[2 * m], do_prime_values[2 * m + 1], dtype=io_dtype) for m in range(pairs)]
            dz_words = [fp32_to_fp16(dz_values[2 * m], dz_values[2 * m + 1], dtype=io_dtype) for m in range(pairs)]
            st_global_v4(do_prime_addr + do_prime_step * cutlass.Int64(u), do_prime_words, cutlass.Int32)
            st_global_v4(dz_addr + dz_step * cutlass.Int64(u), dz_words, cutlass.Int32)
        idx = idx + stride * cutlass.Int32(2)
        do_addr = do_addr + do_step * cutlass.Int64(2)
        z_addr = z_addr + z_step * cutlass.Int64(2)
        y_addr = y_addr + y_step * cutlass.Int64(2)
        do_prime_addr = do_prime_addr + do_prime_step * cutlass.Int64(2)
        dz_addr = dz_addr + dz_step * cutlass.Int64(2)
    if idx < num_vectors:
        do_word = ld_global_v4(do_addr, cutlass.Int32)
        z_word = ld_global_v4(z_addr, cutlass.Int32)
        y_word = ld_global_v4(y_addr, cutlass.Int32)
        do_prime_values = []
        dz_values = []
        for m in cutlass.range_constexpr(pairs):
            do_lo, do_hi = f16x2_to_f32(do_word[m], dtype=io_dtype)
            z_lo, z_hi = f16x2_to_f32(z_word[m], dtype=io_dtype)
            y_lo, y_hi = f16x2_to_f32(y_word[m], dtype=io_dtype)
            sig_lo, sig_hi = sigmoid2(z_lo, z_hi)
            zsig_lo, zsig_hi = fmul2(z_lo, z_hi, sig_lo, sig_hi)
            do_prime_lo, do_prime_hi = fmul2(do_lo, do_hi, zsig_lo, zsig_hi)
            rest_lo, rest_hi = fadd2(one, one, -sig_lo, -sig_hi)
            inner_lo, inner_hi = ffma2(rest_lo, rest_hi, z_lo, z_hi, one, one)
            slope_lo, slope_hi = fmul2(sig_lo, sig_hi, inner_lo, inner_hi)
            doy_lo, doy_hi = fmul2(do_lo, do_hi, y_lo, y_hi)
            dz_lo, dz_hi = fmul2(doy_lo, doy_hi, slope_lo, slope_hi)
            do_prime_values += [do_prime_lo, do_prime_hi]
            dz_values += [dz_lo, dz_hi]
        do_prime_words = [fp32_to_fp16(do_prime_values[2 * m], do_prime_values[2 * m + 1], dtype=io_dtype) for m in range(pairs)]
        dz_words = [fp32_to_fp16(dz_values[2 * m], dz_values[2 * m + 1], dtype=io_dtype) for m in range(pairs)]
        st_global_v4(do_prime_addr, do_prime_words, cutlass.Int32)
        st_global_v4(dz_addr, dz_words, cutlass.Int32)


@cute.kernel
def frost_mamba2_gate_bwd_partial(
    mDotQ: cute.Tensor,
    mDotK: cute.Tensor,
    mExit: cute.Tensor,
    mG: cute.Tensor,
    mA: cute.Tensor,
    mDtBias: cute.Tensor | None,
    mSlots: cute.Tensor,
    mDg: cute.Tensor,
    mPartA: cute.Tensor,
    mPartDt: cute.Tensor | None,
    mSkipPart: cute.Tensor | None,
    h_o: cutlass.Int32,
    dt_min: cutlass.Float32,
    dt_max: cutlass.Float32,
    b_t: cutlass.Constexpr[int],
    dt_softplus: cutlass.Constexpr[bool],
    skip_channels: cutlass.Constexpr[int],
) -> None:
    """Grid (slots, ceil(HO / MAMBA2_GATE_HEADS)), block (MAMBA2_GATE_THREADS,): one chunk slot by MAMBA2_GATE_HEADS
    heads.  The slot table names the chunk (first token, sequence end; -1 at a spare slot past its sequence's chunks,
    where the heads' dA / d_dt_bias partials and the bprop's four dD partial rows are zeroed).  Loads run with lane =
    head (the heads of one token are contiguous in the head-major tensors) into padded SMEM tiles; the math runs with
    lane = token, four heads per thread: the raw-dt gradient of every token from the row dots and the exit term (the
    reverse cumulative sum within the chunk), stored through SMEM with lane = head again, plus the per-head dA /
    d_dt_bias partials over the chunk in a fixed bracketing."""
    HB = cutlass.const_expr(MAMBA2_GATE_HEADS)
    LD = cutlass.const_expr(HB + 1)
    heads_per_thread = cutlass.const_expr(4)
    tokens_per_warp = cutlass.const_expr(32 // HB)
    tokens_per_pass = cutlass.const_expr(MAMBA2_GATE_THREADS // HB)
    passes = cutlass.const_expr(b_t // tokens_per_pass)
    quarters = cutlass.const_expr(b_t // 32)
    if cutlass.const_expr(USE_PDL):
        wait_on_dependent_grids()
    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    slot = cutlass.Int32(cute.arch.block_idx()[0])
    h0 = cutlass.Int32(cute.arch.block_idx()[1]) * cutlass.Int32(HB)
    lane_idx = tidx % cutlass.Int32(32)
    warp = tidx // cutlass.Int32(32)
    first = cutlass.Int32(mSlots[slot, 0])
    sequence_end = cutlass.Int32(mSlots[slot, 1])
    sDotQ = cutlass.Array(cutlass.Float32, b_t * LD, space=cutlass.AddressSpace.smem, alignment=16)
    sDotK = cutlass.Array(cutlass.Float32, b_t * LD, space=cutlass.AddressSpace.smem, alignment=16)
    sG = cutlass.Array(cutlass.Float32, b_t * LD, space=cutlass.AddressSpace.smem, alignment=16)
    sDg = cutlass.Array(cutlass.Float32, b_t * LD, space=cutlass.AddressSpace.smem, alignment=16)
    sWarp = cutlass.Array(cutlass.Float32, HB * quarters, space=cutlass.AddressSpace.smem, alignment=16)
    sWarpA = cutlass.Array(cutlass.Float32, HB * quarters, space=cutlass.AddressSpace.smem, alignment=16)
    sWarpDt = cutlass.Array(cutlass.Float32, HB * quarters, space=cutlass.AddressSpace.smem, alignment=16)
    if first >= cutlass.Int32(0):
        # ---- loads with lane = head: the heads of one token are contiguous ----------------------------
        load_head = lane_idx % cutlass.Int32(HB)
        head = h0 + load_head
        head_ok = head < h_o
        head_c = head if head_ok else h_o - cutlass.Int32(1)
        for p in cutlass.range_constexpr(passes):
            t = cutlass.Int32(p * tokens_per_pass) + warp * cutlass.Int32(tokens_per_warp) + lane_idx // cutlass.Int32(HB)
            token = first + t
            valid = token < sequence_end
            token_c = token if valid else sequence_end - cutlass.Int32(1)
            take = valid and head_ok
            sDotQ[t * LD + load_head] = mDotQ[token_c, head_c] if take else cutlass.Float32(0.0)
            sDotK[t * LD + load_head] = mDotK[token_c, head_c] if take else cutlass.Float32(0.0)
            sG[t * LD + load_head] = mG[token_c, head_c].to(cutlass.Float32)
        nvvm.barrier_cta_sync()

        # ---- math with lane = token, four heads per thread -------------------------------------------
        quarter = warp % cutlass.Int32(quarters)
        head_group = warp // cutlass.Int32(quarters)
        t = quarter * cutlass.Int32(32) + lane_idx
        token = first + t
        valid = token < sequence_end
        a_values = []
        exit_values = []
        dot_k_values = []
        dt_values = []
        slope_values = []
        clamped_values = []
        d_pre_values = []
        incl_values = []
        for i in cutlass.range_constexpr(heads_per_thread):
            hl = head_group * cutlass.Int32(heads_per_thread) + cutlass.Int32(i)
            math_head = h0 + hl
            math_head_c = math_head if math_head < h_o else h_o - cutlass.Int32(1)
            a = mA[math_head_c].to(cutlass.Float32)
            if cutlass.const_expr(mDtBias is not None):
                bias = mDtBias[math_head_c].to(cutlass.Float32)
            else:
                bias = cutlass.Float32(0.0)
            y = sG[t * LD + hl] + bias
            dot_q = sDotQ[t * LD + hl]
            dot_k = sDotK[t * LD + hl]
            exit_term = mExit[slot + cutlass.Int32(1), math_head_c]
            if cutlass.const_expr(dt_softplus):
                dt = softplus(y)
                slope = sigmoid(y)
            else:
                dt = y
                slope = cutlass.Float32(1.0)
            clamped = dt < dt_min or dt > dt_max
            dt_c = cutlass.max(cutlass.min(dt, dt_max), dt_min)
            d_pre = dot_q - dot_k

            # ---- reverse inclusive cumulative sum within the chunk: total - exclusive prefix ---------
            incl = d_pre
            for offset in [1, 2, 4, 8, 16]:
                n = nvvm.shfl_sync(0xFFFFFFFF, incl, offset, 0, kind=nvvm.Shfl.UP)
                if lane_idx >= offset:
                    incl = incl + n
            if lane_idx == 31:
                sWarp[hl * cutlass.Int32(quarters) + quarter] = incl
            a_values.append(a)
            exit_values.append(exit_term)
            dot_k_values.append(dot_k)
            dt_values.append(dt_c)
            slope_values.append(slope)
            clamped_values.append(clamped)
            d_pre_values.append(d_pre)
            incl_values.append(incl)
        nvvm.barrier_cta_sync()
        for i in cutlass.range_constexpr(heads_per_thread):
            hl = head_group * cutlass.Int32(heads_per_thread) + cutlass.Int32(i)
            warp_prefix = cutlass.Float32(0.0)
            total = cutlass.Float32(0.0)
            for w in cutlass.range_constexpr(quarters):
                w_sum = sWarp[hl * cutlass.Int32(quarters) + cutlass.Int32(w)]
                total = total + w_sum
                if cutlass.Int32(w) < quarter:
                    warp_prefix = warp_prefix + w_sum
            suffix = total - (warp_prefix + incl_values[i] - d_pre_values[i])
            dl = suffix + exit_values[i]
            dt_c = dt_values[i]
            dbeta = dot_k_values[i] / dt_c if dt_c > cutlass.Float32(0.0) else cutlass.Float32(0.0)
            ddt = dl * a_values[i] + dbeta
            dg = cutlass.Float32(0.0) if clamped_values[i] else ddt * slope_values[i]
            sDg[t * LD + hl] = dg
            part_a = dl * dt_c if valid else cutlass.Float32(0.0)
            part_dt = dg if valid else cutlass.Float32(0.0)
            for offset in [16, 8, 4, 2, 1]:
                part_a = part_a + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, part_a, offset, 31, kind=nvvm.Shfl.BFLY))
                part_dt = part_dt + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, part_dt, offset, 31, kind=nvvm.Shfl.BFLY))
            if lane_idx == 0:
                sWarpA[hl * cutlass.Int32(quarters) + quarter] = part_a
                sWarpDt[hl * cutlass.Int32(quarters) + quarter] = part_dt
        nvvm.barrier_cta_sync()

        # ---- stores with lane = head, the partials by the first HB threads ---------------------------
        for p in cutlass.range_constexpr(passes):
            t = cutlass.Int32(p * tokens_per_pass) + warp * cutlass.Int32(tokens_per_warp) + lane_idx // cutlass.Int32(HB)
            token = first + t
            if (token < sequence_end) and head_ok:
                mDg[token, head] = sDg[t * LD + load_head].to(mDg.element_type)
        if tidx < cutlass.Int32(HB):
            out_head = h0 + tidx
            if out_head < h_o:
                part_a = sWarpA[tidx * cutlass.Int32(quarters)]
                part_dt = sWarpDt[tidx * cutlass.Int32(quarters)]
                for w in cutlass.range_constexpr(1, quarters):
                    part_a = part_a + sWarpA[tidx * cutlass.Int32(quarters) + cutlass.Int32(w)]
                    part_dt = part_dt + sWarpDt[tidx * cutlass.Int32(quarters) + cutlass.Int32(w)]
                mPartA[slot * h_o + out_head] = part_a
                if cutlass.const_expr(mPartDt is not None):
                    mPartDt[slot * h_o + out_head] = part_dt
    else:
        # ---- a spare slot past its sequence's chunks: zeros for the sums' fixed shape ----------------
        if cutlass.const_expr(mSkipPart is not None):
            for base in cutlass.range_constexpr(0, 4 * skip_channels * HB, MAMBA2_GATE_THREADS):
                idx = cutlass.Int32(base) + tidx
                if idx < cutlass.Int32(4 * skip_channels * HB):
                    hl = idx // cutlass.Int32(4 * skip_channels)
                    rem = idx % cutlass.Int32(4 * skip_channels)
                    head = h0 + hl
                    if head < h_o:
                        row = slot * cutlass.Int32(4) + rem // cutlass.Int32(skip_channels)
                        mSkipPart[(row * h_o + head) * cutlass.Int32(skip_channels) + rem % cutlass.Int32(skip_channels)] = cutlass.Float32(0.0)
        if tidx < cutlass.Int32(HB):
            out_head = h0 + tidx
            if out_head < h_o:
                mPartA[slot * h_o + out_head] = cutlass.Float32(0.0)
                if cutlass.const_expr(mPartDt is not None):
                    mPartDt[slot * h_o + out_head] = cutlass.Float32(0.0)


@cute.kernel
def frost_mamba2_skip_bwd_sum(
    mPart: cute.Tensor,
    mDD: cute.Tensor,
    h_o: cutlass.Int32,
    n_rows: cutlass.Int32,
    d_v: cutlass.Constexpr[int],
) -> None:
    """Grid (HO,), block (8 d_v,): group g of d_v threads folds rows g, g + 8, ... of head h for its channel p over eight
    interleaved chains; the eight group sums combine through shared memory in fixed order."""
    if cutlass.const_expr(USE_PDL):
        wait_on_dependent_grids()
    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    p = tidx % cutlass.Int32(d_v)
    g = tidx // cutlass.Int32(d_v)
    h = cutlass.Int32(cute.arch.block_idx()[0])
    sGroup = cutlass.Array(cutlass.Float32, MAMBA2_SKIP_SUM_GROUPS * d_v, space=cutlass.AddressSpace.smem, alignment=16)
    chains = cutlass.Array(cutlass.Float32, 8)
    for j in cutlass.range_constexpr(8):
        chains[j] = cutlass.Float32(0.0)
    r = g
    row_step = cutlass.Int32(MAMBA2_SKIP_SUM_GROUPS) * h_o * cutlass.Int32(d_v)
    base = (r * h_o + h) * cutlass.Int32(d_v) + p
    values = cutlass.Array(cutlass.Float32, 32)
    # ---- four rounds of loads in flight before their adds; chain j keeps rows g + 8 j + 64 d in d order ---------
    while r + cutlass.Int32(31 * MAMBA2_SKIP_SUM_GROUPS) < n_rows:
        for d in cutlass.range_constexpr(4):
            for j in cutlass.range_constexpr(8):
                values[8 * d + j] = mPart[base + cutlass.Int32(8 * d + j) * row_step]
        for d in cutlass.range_constexpr(4):
            for j in cutlass.range_constexpr(8):
                chains[j] = chains[j] + values[8 * d + j]
        r += cutlass.Int32(32 * MAMBA2_SKIP_SUM_GROUPS)
        base += cutlass.Int32(32) * row_step
    while r + cutlass.Int32(7 * MAMBA2_SKIP_SUM_GROUPS) < n_rows:
        for j in cutlass.range_constexpr(8):
            chains[j] = chains[j] + mPart[base + cutlass.Int32(j) * row_step]
        r += cutlass.Int32(8 * MAMBA2_SKIP_SUM_GROUPS)
        base += cutlass.Int32(8) * row_step
    while r < n_rows:
        chains[0] = chains[0] + mPart[base]
        r += cutlass.Int32(MAMBA2_SKIP_SUM_GROUPS)
        base += row_step
    sGroup[g * cutlass.Int32(d_v) + p] = ((chains[0] + chains[1]) + (chains[2] + chains[3])) + ((chains[4] + chains[5]) + (chains[6] + chains[7]))
    nvvm.barrier_cta_sync()
    if g == 0:
        total = ((sGroup[p] + sGroup[cutlass.Int32(d_v) + p]) + (sGroup[cutlass.Int32(2 * d_v) + p] + sGroup[cutlass.Int32(3 * d_v) + p])) + (
            (sGroup[cutlass.Int32(4 * d_v) + p] + sGroup[cutlass.Int32(5 * d_v) + p])
            + (sGroup[cutlass.Int32(6 * d_v) + p] + sGroup[cutlass.Int32(7 * d_v) + p])
        )
        mDD[h, p] = total.to(mDD.element_type)


@cute.jit
def mamba2_zgate_bwd_launch(
    mDoOut: cute.Tensor,
    mZ: cute.Tensor,
    mYpre: cute.Tensor,
    mDoPre: cute.Tensor,
    mDz: cute.Tensor,
    num_vectors: cutlass.Int32,
    d_v: cutlass.Constexpr[int],
    io_dtype: cutlass.Constexpr,
    h_o: cutlass.Int32,
    n_blocks: cutlass.Int32,
    stream: cuda.CUstream,
):
    frost_mamba2_zgate_bwd(mDoOut, mZ, mYpre, mDoPre, mDz, num_vectors, d_v, io_dtype).launch(
        grid=(n_blocks, h_o, 1), block=(MAMBA2_ZGATE_THREADS, 1, 1), stream=stream, use_pdl=USE_PDL
    )


@cute.jit
def mamba2_gate_bwd_launch(
    mDotQ: cute.Tensor,
    mDotK: cute.Tensor,
    mExit: cute.Tensor,
    mG: cute.Tensor,
    mA: cute.Tensor,
    mDtBias: cute.Tensor | None,
    mSlots: cute.Tensor,
    mDg: cute.Tensor,
    mPartA: cute.Tensor,
    mPartDt: cute.Tensor | None,
    mSkipPart: cute.Tensor | None,
    mDA: cute.Tensor,
    mDDt: cute.Tensor | None,
    mDD: cute.Tensor | None,
    h_o: cutlass.Int32,
    n_slots: cutlass.Int32,
    dt_min: cutlass.Float32,
    dt_max: cutlass.Float32,
    b_t: cutlass.Constexpr[int],
    dt_softplus: cutlass.Constexpr[bool],
    skip_channels: cutlass.Constexpr[int],
    stream: cuda.CUstream,
):
    frost_mamba2_gate_bwd_partial(
        mDotQ, mDotK, mExit, mG, mA, mDtBias, mSlots, mDg, mPartA, mPartDt, mSkipPart, h_o, dt_min, dt_max, b_t, dt_softplus, skip_channels
    ).launch(grid=(n_slots, cute.ceil_div(h_o, MAMBA2_GATE_HEADS), 1), block=(MAMBA2_GATE_THREADS, 1, 1), stream=stream, use_pdl=USE_PDL)
    frost_scalar_gate_bwd_sum(mPartA, mPartDt, mDA, mDDt, h_o, n_slots).launch(grid=(h_o, 1, 1), block=(32, 1, 1), stream=stream, use_pdl=USE_PDL)
    if cutlass.const_expr(mSkipPart is not None):
        if cutlass.const_expr(skip_channels == 1):
            frost_scalar_gate_bwd_sum(mSkipPart, None, mDD, None, h_o, n_slots * cutlass.Int32(4)).launch(
                grid=(h_o, 1, 1), block=(32, 1, 1), stream=stream, use_pdl=USE_PDL
            )
        else:
            frost_mamba2_skip_bwd_sum(mSkipPart, mDD, h_o, n_slots * cutlass.Int32(4), skip_channels).launch(
                grid=(h_o, 1, 1), block=(MAMBA2_SKIP_SUM_GROUPS * skip_channels, 1, 1), stream=stream, use_pdl=USE_PDL
            )


def mamba2_zgate_bwd(do_out, z, y_pre, do_pre, dz, *, stream) -> None:
    """z-gate backward over ``[total, HO, d_v]`` io-dtype tensors with a stride-1 innermost dim and free outer strides:
    writes ``do_pre`` (the bprop's dO) and ``dz``."""
    io_dtype = get_dtype(z.dtype)
    total, h_o, d_v = (int(dim) for dim in z.shape)
    vector_width = 16 // (io_dtype.width // 8)
    num_vectors = total * (d_v // vector_width)
    n_blocks = max(1, min(-(-num_vectors // MAMBA2_ZGATE_THREADS), max(1, 4096 // h_o)))
    cu_stream = cuda.CUstream(int(stream))
    tensors = (do_out, z, y_pre, do_pre, dz)
    cache = gate_bwd_cache(("mamba2_zgate", str(z.dtype), d_v, current_device()))
    if "compiled" not in cache:
        traced = [from_dlpack(t, assumed_align=16).mark_layout_dynamic(leading_dim=2) for t in tensors]
        cache["compiled"] = cute.compile(
            mamba2_zgate_bwd_launch,
            *traced,
            cutlass.Int32(num_vectors),
            d_v,
            io_dtype,
            cutlass.Int32(h_o),
            cutlass.Int32(n_blocks),
            cu_stream,
            options="--enable-tvm-ffi",
        )
    cache["compiled"](*tensors, num_vectors, h_o, n_blocks, cu_stream)


def mamba2_gate_bwd(
    dot_q,
    dot_k,
    exit_dots,
    g,
    a,
    dt_bias,
    slots,
    dg,
    da,
    d_dt_bias,
    part_a,
    part_dt,
    *,
    b_t,
    dt_softplus,
    dt_min,
    dt_max,
    stream,
    skip_bwd_part=None,
    dd=None,
) -> None:
    """dt-gate backward: ``dg`` [total, HO] (gate dtype) from the bprop's row dots and exit dots, ``dA`` (HO,) and,
    with ``dt_bias``, ``d_dt_bias`` (HO,); with ``skip_bwd_part`` (the bprop's four per-warp dD partial rows per slot, flat
    over ``(4 * slots, HO[, d_v])``), ``dd`` (HO,) or (HO, d_v) in D's dtype.  ``slots`` is the int32 [slots, 2]
    exit-slot table the backward's prologue wrote (:func:`exit_slot_table_body`); ``part_a`` / ``part_dt`` are
    ``(slots * HO,)`` fp32 workspace carves."""
    n_slots, h_o = (int(dim) for dim in exit_dots.shape)
    skip_channels = int(dd.shape[1]) if dd is not None and len(dd.shape) == 2 else 1
    cu_stream = cuda.CUstream(int(stream))
    tensors = (dot_q, dot_k, exit_dots, g, a, dt_bias, slots, dg, part_a, part_dt, skip_bwd_part, da, d_dt_bias, dd)
    cache = gate_bwd_cache(
        (
            "mamba2_gate",
            str(g.dtype),
            str(a.dtype),
            str(dt_bias.dtype) if dt_bias is not None else "none",
            d_dt_bias is not None,
            str(dd.dtype) if dd is not None else "none",
            skip_channels,
            int(b_t),
            bool(dt_softplus),
            current_device(),
        )
    )
    if "compiled" not in cache:
        traced = [from_dlpack(t, assumed_align=4).mark_layout_dynamic(leading_dim=1) for t in tensors[:4]]
        traced += [from_dlpack(t, assumed_align=4).mark_layout_dynamic(leading_dim=0) if t is not None else None for t in tensors[4:6]]
        traced.append(from_dlpack(slots, assumed_align=8).mark_layout_dynamic(leading_dim=1))
        traced.append(from_dlpack(dg, assumed_align=4).mark_layout_dynamic(leading_dim=1))
        traced += [from_dlpack(t, assumed_align=4).mark_layout_dynamic(leading_dim=0) if t is not None else None for t in tensors[8:13]]
        traced.append(from_dlpack(dd, assumed_align=4).mark_layout_dynamic(leading_dim=len(dd.shape) - 1) if dd is not None else None)
        cache["compiled"] = cute.compile(
            mamba2_gate_bwd_launch,
            *traced,
            cutlass.Int32(h_o),
            cutlass.Int32(n_slots),
            cutlass.Float32(float(dt_min)),
            cutlass.Float32(float(dt_max)),
            int(b_t),
            bool(dt_softplus),
            skip_channels,
            cu_stream,
            options="--enable-tvm-ffi",
        )
    cache["compiled"](*tensors, h_o, n_slots, float(dt_min), float(dt_max), cu_stream)


frost_scalar_gate_bwd_partial.set_name_prefix("cudnn", remove_cutlass_symbol=False)
frost_scalar_gate_bwd_sum.set_name_prefix("cudnn", remove_cutlass_symbol=False)
frost_channel_gate_bwd_partial.set_name_prefix("cudnn", remove_cutlass_symbol=False)
frost_channel_gate_bwd_sum.set_name_prefix("cudnn", remove_cutlass_symbol=False)
frost_mamba2_zgate_bwd.set_name_prefix("cudnn", remove_cutlass_symbol=False)
frost_mamba2_gate_bwd_partial.set_name_prefix("cudnn", remove_cutlass_symbol=False)
frost_mamba2_skip_bwd_sum.set_name_prefix("cudnn", remove_cutlass_symbol=False)
