# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Q/K row L2 normalization helpers for GDN (the main kernels stay unchanged
and consume normalized workspace copies through their usual descriptors).

Forward: normalize every 128-element q/k row into compact io-dtype workspace
buffers and stash the fp32 inverse norms.  Backward: project the main
kernel's dq_n/dk_n (gradients wrt the normalized rows) back through the
normalize Jacobian in place: ``dq = inv_norm * (dq_n - (dq_n . q_n) q_n)``.
"""

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.arch.nvvm_wrappers import inline_ptx
from cutlass.cute.runtime import from_dlpack

from cudnn.frost.buffers import data_ptr
from cudnn.frost.tile_dsl.pointwise import f16x2_to_f32, ffma2, fmul2, fp32_to_fp16, l2norm_inv, lane_group_sum

THREADS_PER_ROW = 16
ROWS_PER_CTA = 8
FWD_LANES = 4  # fwd lanes per row: 32 elements/lane (4 x 128-bit loads), 2-step butterfly
FWD_ROWS_PER_GROUP = 2  # fwd rows batched per lane group: 8 independent loads in flight per thread


@cute.kernel
def l2norm_qk_kernel(
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mQn: cute.Tensor,
    mKn: cute.Tensor,
    mInvQ: cute.Tensor,
    mInvK: cute.Tensor,
    n_q_rows: cutlass.Int32,
    n_rows: cutlass.Int32,
    h_q: cutlass.Int32,
    h_k: cutlass.Int32,
) -> None:
    """Grid over all q rows then all k rows, FWD_LANES lanes x 32 elements
    per row (4 x 128-bit loads per lane), FWD_ROWS_PER_GROUP consecutive rows
    batched per lane group.  All rows' loads issue before the reductions and
    the 4-lane butterfly is 2 steps, so memory latency pipelines (the
    16-lane single-row shape measured 75.7% of DRAM peak on GB200, this one
    88.3%; FLA's autotuned Triton tile is 91%).  fp32 sums of squares,
    rsqrt with the shared epsilon floor, normalized rows to the compact io
    workspace, fp32 inverse norms to their slots.  Tail rows clamp their
    loads and skip stores."""
    bid = cute.arch.block_idx()
    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    grp = tidx // cutlass.Int32(FWD_LANES)
    lane_idx = tidx % cutlass.Int32(FWD_LANES)
    row0 = (cutlass.Int32(bid[0]) * cutlass.Int32(128 // FWD_LANES) + grp) * cutlass.Int32(FWD_ROWS_PER_GROUP)
    v0 = lane_idx * cutlass.Int32(32)
    rows = []
    workspace_addrs = []
    nrm_addrs = []
    vals = []
    for r in cutlass.range_constexpr(FWD_ROWS_PER_GROUP):
        row = row0 + cutlass.Int32(r)
        row_r = row if row < n_rows else n_rows - cutlass.Int32(1)
        # q rows first, then k rows, head-fastest; sources honor their own
        # [T, H, 128] strides while workspace rows sit at row * 128.  Both
        # branches are traced, so the addresses have to exist beforehand.
        src_addr = cutlass.Int64(0)
        workspace_addr = cutlass.Int64(0)
        nrm_addr = cutlass.Int64(0)
        if row_r < n_q_rows:
            t = row_r // h_q
            h = row_r % h_q
            src_elements = t * cutlass.Int64(mQ.stride[0]) + h * cutlass.Int64(mQ.stride[1]) + cutlass.Int64(v0)
            src_addr = mQ.iterator.toint() + src_elements * cutlass.Int64(2)
            workspace_addr = mQn.iterator.toint() + (cutlass.Int64(row_r) * cutlass.Int64(128) + cutlass.Int64(v0)) * cutlass.Int64(2)
            nrm_addr = mInvQ.iterator.toint() + cutlass.Int64(row_r) * cutlass.Int64(4)
        else:
            k_row = row_r - n_q_rows
            t = k_row // h_k
            h = k_row % h_k
            src_elements = t * cutlass.Int64(mK.stride[0]) + h * cutlass.Int64(mK.stride[1]) + cutlass.Int64(v0)
            src_addr = mK.iterator.toint() + src_elements * cutlass.Int64(2)
            workspace_addr = mKn.iterator.toint() + (cutlass.Int64(k_row) * cutlass.Int64(128) + cutlass.Int64(v0)) * cutlass.Int64(2)
            nrm_addr = mInvK.iterator.toint() + cutlass.Int64(k_row) * cutlass.Int64(4)
        chunks = []
        for c in cutlass.range_constexpr(4):
            w0, w1, w2, w3 = inline_ptx(
                "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
                write_only_types=[cutlass.Int32, cutlass.Int32, cutlass.Int32, cutlass.Int32],
                read_only_args=[src_addr + cutlass.Int64(16 * c)],
            )
            f0, f1 = f16x2_to_f32(w0, dtype=mQ.element_type)
            f2, f3 = f16x2_to_f32(w1, dtype=mQ.element_type)
            f4, f5 = f16x2_to_f32(w2, dtype=mQ.element_type)
            f6, f7 = f16x2_to_f32(w3, dtype=mQ.element_type)
            chunks.append((f0, f1, f2, f3, f4, f5, f6, f7))
        rows.append(row)
        workspace_addrs.append(workspace_addr)
        nrm_addrs.append(nrm_addr)
        vals.append(chunks)
    for r in cutlass.range_constexpr(FWD_ROWS_PER_GROUP):
        acc = cutlass.Float32(0.0)
        for c in cutlass.range_constexpr(4):
            f = vals[r][c]
            acc = acc + ((f[0] * f[0] + f[1] * f[1]) + (f[2] * f[2] + f[3] * f[3])) + ((f[4] * f[4] + f[5] * f[5]) + (f[6] * f[6] + f[7] * f[7]))
        inv = l2norm_inv(lane_group_sum(acc, FWD_LANES))
        if rows[r] < n_rows:
            for c in cutlass.range_constexpr(4):
                f = vals[r][c]
                s0, s1 = fmul2(f[0], f[1], inv, inv)
                s2, s3 = fmul2(f[2], f[3], inv, inv)
                s4, s5 = fmul2(f[4], f[5], inv, inv)
                s6, s7 = fmul2(f[6], f[7], inv, inv)
                w0 = fp32_to_fp16(s0, s1, dtype=mQ.element_type)
                w1 = fp32_to_fp16(s2, s3, dtype=mQ.element_type)
                w2 = fp32_to_fp16(s4, s5, dtype=mQ.element_type)
                w3 = fp32_to_fp16(s6, s7, dtype=mQ.element_type)
                inline_ptx(
                    "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
                    read_only_args=[workspace_addrs[r] + cutlass.Int64(16 * c), w0, w1, w2, w3],
                )
            if lane_idx == cutlass.Int32(0):
                inline_ptx("st.global.f32 [$0], $1;", read_only_args=[nrm_addrs[r], inv])


@cute.kernel
def l2norm_qk_bwd_kernel(
    mDq: cute.Tensor,
    mDk: cute.Tensor,
    mQn: cute.Tensor,
    mKn: cute.Tensor,
    mInvQ: cute.Tensor,
    mInvK: cute.Tensor,
    n_q_rows: cutlass.Int32,
    n_rows: cutlass.Int32,
    h_q: cutlass.Int32,
    h_k: cutlass.Int32,
) -> None:
    """In-place normalize-Jacobian projection of dq/dk: per row, fp32 dot of
    the incoming gradient with the saved normalized row (butterfly reduce),
    then ``inv_norm * (grad - dot * row_n)`` back to the caller's buffer.
    Single row per lane group: the dot+projection already fills the memory
    latency (92.9% SOL measured; the forward's row batching REGRESSED this
    kernel to 86% on GB200, lyris job 2694953)."""
    bid = cute.arch.block_idx()
    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    row = cutlass.Int32(bid[0]) * cutlass.Int32(ROWS_PER_CTA) + tidx // cutlass.Int32(THREADS_PER_ROW)
    lane_idx = tidx % cutlass.Int32(THREADS_PER_ROW)
    if row < n_rows:
        v0 = lane_idx * cutlass.Int32(8)
        grad_addr = cutlass.Int64(0)
        workspace_addr = cutlass.Int64(0)
        nrm_addr = cutlass.Int64(0)
        if row < n_q_rows:
            t = row // h_q
            h = row % h_q
            grad_elements = t * cutlass.Int64(mDq.stride[0]) + h * cutlass.Int64(mDq.stride[1]) + cutlass.Int64(v0)
            grad_addr = mDq.iterator.toint() + grad_elements * cutlass.Int64(2)
            workspace_addr = mQn.iterator.toint() + (cutlass.Int64(row) * cutlass.Int64(128) + cutlass.Int64(v0)) * cutlass.Int64(2)
            nrm_addr = mInvQ.iterator.toint() + cutlass.Int64(row) * cutlass.Int64(4)
        else:
            k_row = row - n_q_rows
            t = k_row // h_k
            h = k_row % h_k
            grad_elements = t * cutlass.Int64(mDk.stride[0]) + h * cutlass.Int64(mDk.stride[1]) + cutlass.Int64(v0)
            grad_addr = mDk.iterator.toint() + grad_elements * cutlass.Int64(2)
            workspace_addr = mKn.iterator.toint() + (cutlass.Int64(k_row) * cutlass.Int64(128) + cutlass.Int64(v0)) * cutlass.Int64(2)
            nrm_addr = mInvK.iterator.toint() + cutlass.Int64(k_row) * cutlass.Int64(4)
        gw0, gw1, gw2, gw3 = inline_ptx(
            "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
            write_only_types=[cutlass.Int32, cutlass.Int32, cutlass.Int32, cutlass.Int32],
            read_only_args=[grad_addr],
        )
        d0, d1 = f16x2_to_f32(gw0, dtype=mDq.element_type)
        d2, d3 = f16x2_to_f32(gw1, dtype=mDq.element_type)
        d4, d5 = f16x2_to_f32(gw2, dtype=mDq.element_type)
        d6, d7 = f16x2_to_f32(gw3, dtype=mDq.element_type)
        nw0, nw1, nw2, nw3 = inline_ptx(
            "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
            write_only_types=[cutlass.Int32, cutlass.Int32, cutlass.Int32, cutlass.Int32],
            read_only_args=[workspace_addr],
        )
        n0, n1 = f16x2_to_f32(nw0, dtype=mQn.element_type)
        n2, n3 = f16x2_to_f32(nw1, dtype=mQn.element_type)
        n4, n5 = f16x2_to_f32(nw2, dtype=mQn.element_type)
        n6, n7 = f16x2_to_f32(nw3, dtype=mQn.element_type)
        dot_lo, dot_hi = fmul2(d0, d1, n0, n1)
        dot_lo, dot_hi = ffma2(d2, d3, n2, n3, dot_lo, dot_hi)
        dot_lo, dot_hi = ffma2(d4, d5, n4, n5, dot_lo, dot_hi)
        dot_lo, dot_hi = ffma2(d6, d7, n6, n7, dot_lo, dot_hi)
        dot = lane_group_sum(dot_lo + dot_hi, THREADS_PER_ROW)
        inv = inline_ptx(
            "ld.global.f32 $0, [$1];",
            write_only_types=[cutlass.Float32],
            read_only_args=[nrm_addr],
        )
        neg_dot = -dot
        p0, p1 = ffma2(neg_dot, neg_dot, n0, n1, d0, d1)
        p2, p3 = ffma2(neg_dot, neg_dot, n2, n3, d2, d3)
        p4, p5 = ffma2(neg_dot, neg_dot, n4, n5, d4, d5)
        p6, p7 = ffma2(neg_dot, neg_dot, n6, n7, d6, d7)
        q0, q1 = fmul2(p0, p1, inv, inv)
        q2, q3 = fmul2(p2, p3, inv, inv)
        q4, q5 = fmul2(p4, p5, inv, inv)
        q6, q7 = fmul2(p6, p7, inv, inv)
        w0 = fp32_to_fp16(q0, q1, dtype=mDq.element_type)
        w1 = fp32_to_fp16(q2, q3, dtype=mDq.element_type)
        w2 = fp32_to_fp16(q4, q5, dtype=mDq.element_type)
        w3 = fp32_to_fp16(q6, q7, dtype=mDq.element_type)
        inline_ptx(
            "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
            read_only_args=[grad_addr, w0, w1, w2, w3],
        )


@cute.jit
def l2norm_qk_launch(
    q: cute.Tensor,
    k: cute.Tensor,
    q_n: cute.Tensor,
    k_n: cute.Tensor,
    inv_q: cute.Tensor,
    inv_k: cute.Tensor,
    n_q_rows: cutlass.Int32,
    n_rows: cutlass.Int32,
    h_q: cutlass.Int32,
    h_k: cutlass.Int32,
    n_blocks: cutlass.Int32,
    stream: cuda.CUstream,
):
    l2norm_qk_kernel(q, k, q_n, k_n, inv_q, inv_k, n_q_rows, n_rows, h_q, h_k).launch(
        grid=(n_blocks, 1, 1), block=(THREADS_PER_ROW * ROWS_PER_CTA, 1, 1), stream=stream
    )


@cute.jit
def l2norm_qk_bwd_launch(
    dq: cute.Tensor,
    dk: cute.Tensor,
    q_n: cute.Tensor,
    k_n: cute.Tensor,
    inv_q: cute.Tensor,
    inv_k: cute.Tensor,
    n_q_rows: cutlass.Int32,
    n_rows: cutlass.Int32,
    h_q: cutlass.Int32,
    h_k: cutlass.Int32,
    n_blocks: cutlass.Int32,
    stream: cuda.CUstream,
):
    l2norm_qk_bwd_kernel(dq, dk, q_n, k_n, inv_q, inv_k, n_q_rows, n_rows, h_q, h_k).launch(
        grid=(n_blocks, 1, 1), block=(THREADS_PER_ROW * ROWS_PER_CTA, 1, 1), stream=stream
    )


@functools.cache
def l2norm_cache(key):
    return {}


def l2norm_qk(q, k, q_n, k_n, inv_q, inv_k, *, stream):
    """Normalize q/k rows into the compact io workspace copies and stash the
    fp32 inverse norms.  Sources are read through their own strides."""
    ROWS = (128 // FWD_LANES) * FWD_ROWS_PER_GROUP
    total, h_q, d = (int(s_) for s_ in q.shape)
    total_k, h_k, d_k = (int(s_) for s_ in k.shape)
    if d != 128 or d_k != 128 or total_k != total:
        raise ValueError(f"q/k must be [total, H, 128] with matching totals; got {tuple(q.shape)} / {tuple(k.shape)}")
    for name, t in (("q", q), ("k", k)):
        st = tuple(int(s_) for s_ in t.stride())
        if st[2] != 1 or st[0] % 8 != 0 or st[1] % 8 != 0 or data_ptr(t) % 16 != 0:
            raise ValueError(f"{name} needs a 16B-aligned base, unit channel stride, and outer strides in multiples of 8; got {st}")
    for name, buf, h in (("q_n", q_n, h_q), ("k_n", k_n, h_k)):
        if tuple(int(s_) for s_ in buf.stride()) != (h * 128, 128, 1):
            raise ValueError(f"{name} workspace must be compact [total, {h}, 128]")
    for name, buf, h in (("inv_q", inv_q, h_q), ("inv_k", inv_k, h_k)):
        if tuple(int(s_) for s_ in buf.stride()) != (h, 1):
            raise ValueError(f"{name} workspace must be compact [total, {h}] fp32")
    n_q_rows = total * h_q
    n_rows = n_q_rows + total * h_k
    args = (n_q_rows, n_rows, h_q, h_k, (n_rows + ROWS - 1) // ROWS)
    cache = l2norm_cache(("fwd", str(q.dtype)))
    cu_stream = cuda.CUstream(int(stream))
    if "compiled" not in cache:
        tensors = (q, k, q_n, k_n, inv_q, inv_k)
        cache["compiled"] = cute.compile(
            l2norm_qk_launch,
            *(from_dlpack(t, assumed_align=16).mark_layout_dynamic(leading_dim=lead) for t, lead in zip(tensors, (2, 2, 2, 2, 1, 1))),
            *(cutlass.Int32(a) for a in args),
            cu_stream,
            options="--enable-tvm-ffi",
        )
    cache["compiled"](q, k, q_n, k_n, inv_q, inv_k, *args, cu_stream)


def l2norm_qk_bwd(dq, dk, q_n, k_n, inv_q, inv_k, *, stream):
    """Project dq/dk in place through the normalize Jacobian using the saved
    normalized rows and inverse norms.  Run after any head-group fold so the
    gradients are back at the native q/k head counts."""
    ROWS = ROWS_PER_CTA
    total, h_q, d = (int(s_) for s_ in dq.shape)
    total_k, h_k, d_k = (int(s_) for s_ in dk.shape)
    if d != 128 or d_k != 128 or total_k != total:
        raise ValueError(f"dq/dk must be [total, H, 128] with matching totals; got {tuple(dq.shape)} / {tuple(dk.shape)}")
    for name, t in (("dq", dq), ("dk", dk)):
        st = tuple(int(s_) for s_ in t.stride())
        if st[2] != 1 or st[0] % 8 != 0 or st[1] % 8 != 0 or data_ptr(t) % 16 != 0:
            raise ValueError(f"{name} needs a 16B-aligned base, unit channel stride, and outer strides in multiples of 8; got {st}")
    for name, buf, h in (("q_n", q_n, h_q), ("k_n", k_n, h_k)):
        if tuple(int(s_) for s_ in buf.stride()) != (h * 128, 128, 1):
            raise ValueError(f"{name} workspace must be compact [total, {h}, 128]")
    for name, buf, h in (("inv_q", inv_q, h_q), ("inv_k", inv_k, h_k)):
        if tuple(int(s_) for s_ in buf.stride()) != (h, 1):
            raise ValueError(f"{name} workspace must be compact [total, {h}] fp32")
    n_q_rows = total * h_q
    n_rows = n_q_rows + total * h_k
    args = (n_q_rows, n_rows, h_q, h_k, (n_rows + ROWS - 1) // ROWS)
    cache = l2norm_cache(("bwd", str(dq.dtype)))
    cu_stream = cuda.CUstream(int(stream))
    if "compiled" not in cache:
        tensors = (dq, dk, q_n, k_n, inv_q, inv_k)
        cache["compiled"] = cute.compile(
            l2norm_qk_bwd_launch,
            *(from_dlpack(t, assumed_align=16).mark_layout_dynamic(leading_dim=lead) for t, lead in zip(tensors, (2, 2, 2, 2, 1, 1))),
            *(cutlass.Int32(a) for a in args),
            cu_stream,
            options="--enable-tvm-ffi",
        )
    cache["compiled"](dq, dk, q_n, k_n, inv_q, inv_k, *args, cu_stream)
