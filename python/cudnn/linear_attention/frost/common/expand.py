# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sub-token row expansion/gather for GDP.

GDP runs the GDN kernels on a timeline expanded by ``num_householder``:
q/g scatter into expanded workspace copies (readout sub-token ``n - 1``,
gate sub-token ``0``) and O/dQ/dG gather back to real-token rows."""

from typing import NamedTuple

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda

from cutlass.cute.arch.nvvm_wrappers import inline_ptx
from cutlass.cute.runtime import from_dlpack

BLOCK = 256


@cute.jit
def strided_row_addr(m, tok, h_idx, w_off, h_count):
    bpe = cutlass.const_expr(m.element_type.width // 8)
    elems = tok * cutlass.Int64(m.stride[0])
    if cutlass.const_expr(h_count > 1):
        elems = elems + h_idx * cutlass.Int64(m.stride[1])
    return m.iterator.toint() + elems * cutlass.Int64(bpe) + w_off * cutlass.Int64(4)


@cute.jit
def expand_rows(mSrc, mDst, chunk, h_count, inner_words, num_householder, phase, vec_words):
    """One chunk of zero-fill expansion: every expanded row is stored, only
    the ``phase`` sub-token loads the source."""
    row_chunks = cutlass.const_expr(inner_words // vec_words)
    seg = chunk // cutlass.Int64(row_chunks)
    w_off = (chunk - seg * cutlass.Int64(row_chunks)) * cutlass.Int64(vec_words)
    row_out = seg // cutlass.Int64(h_count)
    h_idx = seg - row_out * cutlass.Int64(h_count)
    tok = row_out // cutlass.Int64(num_householder)
    slot = row_out - tok * cutlass.Int64(num_householder)
    src_addr = strided_row_addr(mSrc, tok, h_idx, w_off, h_count)
    dst_addr = mDst.iterator.toint() + chunk * cutlass.Int64(vec_words * 4)
    if cutlass.const_expr(vec_words == 4):
        w0 = cutlass.Int32(0)
        w1 = cutlass.Int32(0)
        w2 = cutlass.Int32(0)
        w3 = cutlass.Int32(0)
        if slot == cutlass.Int64(phase):
            w0, w1, w2, w3 = inline_ptx(
                "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
                write_only_types=[cutlass.Int32, cutlass.Int32, cutlass.Int32, cutlass.Int32],
                read_only_args=[src_addr],
            )
        inline_ptx("st.global.v4.b32 [$0], {$1, $2, $3, $4};", read_only_args=[dst_addr, w0, w1, w2, w3])
    else:
        w0 = cutlass.Int32(0)
        if slot == cutlass.Int64(phase):
            w0 = inline_ptx("ld.global.b32 $0, [$1];", write_only_types=[cutlass.Int32], read_only_args=[src_addr])
        inline_ptx("st.global.b32 [$0], $1;", read_only_args=[dst_addr, w0])


@cute.jit
def scatter_rows(mSrc, mDst, chunk, h_count, inner_words, num_householder, phase):
    """One of phase-row scatter: real-token rows land on
    sub-token ``phase``; the other expanded rows are left untouched."""
    row_chunks = cutlass.const_expr(inner_words // 4)
    seg = chunk // cutlass.Int64(row_chunks)
    w_off = (chunk - seg * cutlass.Int64(row_chunks)) * cutlass.Int64(4)
    tok = seg // cutlass.Int64(h_count)
    h_idx = seg - tok * cutlass.Int64(h_count)
    src_addr = strided_row_addr(mSrc, tok, h_idx, w_off, h_count)
    dst_words = ((tok * cutlass.Int64(num_householder) + cutlass.Int64(phase)) * cutlass.Int64(h_count) + h_idx) * cutlass.Int64(inner_words) + w_off
    dst_addr = mDst.iterator.toint() + dst_words * cutlass.Int64(4)
    w0, w1, w2, w3 = inline_ptx(
        "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        write_only_types=[cutlass.Int32, cutlass.Int32, cutlass.Int32, cutlass.Int32],
        read_only_args=[src_addr],
    )
    inline_ptx("st.global.v4.b32 [$0], {$1, $2, $3, $4};", read_only_args=[dst_addr, w0, w1, w2, w3])


@cute.jit
def gather_rows(mSrc, mDst, chunk, h_count, inner_words, num_householder, phase, vec_words):
    """One chunk of phase-row gather: sub-token ``phase`` rows of the
    expanded source copy back to real-token rows."""
    row_chunks = cutlass.const_expr(inner_words // vec_words)
    seg = chunk // cutlass.Int64(row_chunks)
    w_off = (chunk - seg * cutlass.Int64(row_chunks)) * cutlass.Int64(vec_words)
    tok = seg // cutlass.Int64(h_count)
    h_idx = seg - tok * cutlass.Int64(h_count)
    src_words = ((tok * cutlass.Int64(num_householder) + cutlass.Int64(phase)) * cutlass.Int64(h_count) + h_idx) * cutlass.Int64(inner_words) + w_off
    src_addr = mSrc.iterator.toint() + src_words * cutlass.Int64(4)
    dst_addr = strided_row_addr(mDst, tok, h_idx, w_off, h_count)
    if cutlass.const_expr(vec_words == 4):
        w0, w1, w2, w3 = inline_ptx(
            "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
            write_only_types=[cutlass.Int32, cutlass.Int32, cutlass.Int32, cutlass.Int32],
            read_only_args=[src_addr],
        )
        inline_ptx("st.global.v4.b32 [$0], {$1, $2, $3, $4};", read_only_args=[dst_addr, w0, w1, w2, w3])
    else:
        w0 = inline_ptx("ld.global.b32 $0, [$1];", write_only_types=[cutlass.Int32], read_only_args=[src_addr])
        inline_ptx("st.global.b32 [$0], $1;", read_only_args=[dst_addr, w0])


@cute.kernel
def pack_fwd_kernel(
    with_q: cutlass.Constexpr[bool],
    num_householder: cutlass.Constexpr[int],
    g_inner: cutlass.Constexpr[int],
    g_vec: cutlass.Constexpr[int],
    q_heads: cutlass.Constexpr[int],
    q_inner: cutlass.Constexpr[int],
    mG: cute.Tensor,
    mGx: cute.Tensor,
    mQ: cute.Tensor,
    mQx: cute.Tensor,
) -> None:
    tidx, _, _ = cute.arch.thread_idx()
    bidx = cute.arch.block_idx()[0]
    chunk_idx = cutlass.Int64(cutlass.Int32(bidx)) * cutlass.Int64(BLOCK) + cutlass.Int64(cutlass.Int32(tidx))
    g_chunks = cutlass.Int64(mGx.shape[0]) * cutlass.Int64(cutlass.const_expr(g_inner // g_vec))
    if chunk_idx < g_chunks:
        expand_rows(mG, mGx, chunk_idx, 1, g_inner, num_householder, 0, g_vec)
    else:
        if cutlass.const_expr(with_q):
            chunk = chunk_idx - g_chunks
            if chunk < cutlass.Int64(mQ.shape[0]) * cutlass.Int64(cutlass.const_expr(q_heads * q_inner // 4)):
                scatter_rows(mQ, mQx, chunk, q_heads, q_inner, num_householder, num_householder - 1)


@cute.kernel
def pack_bwd_kernel(
    num_householder: cutlass.Constexpr[int],
    q_heads: cutlass.Constexpr[int],
    q_inner: cutlass.Constexpr[int],
    g_inner: cutlass.Constexpr[int],
    g_vec: cutlass.Constexpr[int],
    do_heads: cutlass.Constexpr[int],
    do_inner: cutlass.Constexpr[int],
    mQ: cute.Tensor,
    mQx: cute.Tensor,
    mG: cute.Tensor,
    mGx: cute.Tensor,
    mDo: cute.Tensor,
    mDox: cute.Tensor,
) -> None:
    tidx, _, _ = cute.arch.thread_idx()
    bidx = cute.arch.block_idx()[0]
    chunk_idx = cutlass.Int64(cutlass.Int32(bidx)) * cutlass.Int64(BLOCK) + cutlass.Int64(cutlass.Int32(tidx))
    q_chunks = cutlass.Int64(mQx.shape[0]) * cutlass.Int64(cutlass.const_expr(q_heads * q_inner // 4))
    g_chunks = cutlass.Int64(mGx.shape[0]) * cutlass.Int64(cutlass.const_expr(g_inner // g_vec))
    if chunk_idx < q_chunks:
        expand_rows(mQ, mQx, chunk_idx, q_heads, q_inner, num_householder, num_householder - 1, 4)
    else:
        chunk = chunk_idx - q_chunks
        if chunk < g_chunks:
            expand_rows(mG, mGx, chunk, 1, g_inner, num_householder, 0, g_vec)
        else:
            chunk = chunk - g_chunks
            if chunk < cutlass.Int64(mDox.shape[0]) * cutlass.Int64(cutlass.const_expr(do_heads * do_inner // 4)):
                expand_rows(mDo, mDox, chunk, do_heads, do_inner, num_householder, num_householder - 1, 4)


@cute.kernel
def gather_o_kernel(
    num_householder: cutlass.Constexpr[int],
    o_heads: cutlass.Constexpr[int],
    o_inner: cutlass.Constexpr[int],
    mOx: cute.Tensor,
    mO: cute.Tensor,
) -> None:
    tidx, _, _ = cute.arch.thread_idx()
    bidx = cute.arch.block_idx()[0]
    chunk_idx = cutlass.Int64(cutlass.Int32(bidx)) * cutlass.Int64(BLOCK) + cutlass.Int64(cutlass.Int32(tidx))
    if chunk_idx < cutlass.Int64(mO.shape[0]) * cutlass.Int64(cutlass.const_expr(o_heads * o_inner // 4)):
        gather_rows(mOx, mO, chunk_idx, o_heads, o_inner, num_householder, num_householder - 1, 4)


@cute.kernel
def gather_bwd_kernel(
    num_householder: cutlass.Constexpr[int],
    dq_heads: cutlass.Constexpr[int],
    dq_inner: cutlass.Constexpr[int],
    dg_inner: cutlass.Constexpr[int],
    dg_vec: cutlass.Constexpr[int],
    mDqx: cute.Tensor,
    mDq: cute.Tensor,
    mDgx: cute.Tensor,
    mDg: cute.Tensor,
) -> None:
    tidx, _, _ = cute.arch.thread_idx()
    bidx = cute.arch.block_idx()[0]
    chunk_idx = cutlass.Int64(cutlass.Int32(bidx)) * cutlass.Int64(BLOCK) + cutlass.Int64(cutlass.Int32(tidx))
    dq_chunks = cutlass.Int64(mDq.shape[0]) * cutlass.Int64(cutlass.const_expr(dq_heads * dq_inner // 4))
    if chunk_idx < dq_chunks:
        gather_rows(mDqx, mDq, chunk_idx, dq_heads, dq_inner, num_householder, num_householder - 1, 4)
    else:
        chunk = chunk_idx - dq_chunks
        if chunk < cutlass.Int64(mDg.shape[0]) * cutlass.Int64(cutlass.const_expr(dg_inner // dg_vec)):
            gather_rows(mDgx, mDg, chunk, 1, dg_inner, num_householder, 0, dg_vec)


@cute.jit
def pack_fwd_launch(
    with_q: cutlass.Constexpr[bool],
    num_householder: cutlass.Constexpr[int],
    g_inner: cutlass.Constexpr[int],
    g_vec: cutlass.Constexpr[int],
    q_heads: cutlass.Constexpr[int],
    q_inner: cutlass.Constexpr[int],
    mG: cute.Tensor,
    mGx: cute.Tensor,
    mQ: cute.Tensor,
    mQx: cute.Tensor,
    grid_x: cutlass.Int32,
    stream: cuda.CUstream,
) -> None:
    pack_fwd_kernel(with_q, num_householder, g_inner, g_vec, q_heads, q_inner, mG, mGx, mQ, mQx).launch(grid=(grid_x, 1, 1), block=(BLOCK, 1, 1), stream=stream)


@cute.jit
def pack_bwd_launch(
    num_householder: cutlass.Constexpr[int],
    q_heads: cutlass.Constexpr[int],
    q_inner: cutlass.Constexpr[int],
    g_inner: cutlass.Constexpr[int],
    g_vec: cutlass.Constexpr[int],
    do_heads: cutlass.Constexpr[int],
    do_inner: cutlass.Constexpr[int],
    mQ: cute.Tensor,
    mQx: cute.Tensor,
    mG: cute.Tensor,
    mGx: cute.Tensor,
    mDo: cute.Tensor,
    mDox: cute.Tensor,
    grid_x: cutlass.Int32,
    stream: cuda.CUstream,
) -> None:
    pack_bwd_kernel(num_householder, q_heads, q_inner, g_inner, g_vec, do_heads, do_inner, mQ, mQx, mG, mGx, mDo, mDox).launch(
        grid=(grid_x, 1, 1), block=(BLOCK, 1, 1), stream=stream
    )


@cute.jit
def gather_o_launch(
    num_householder: cutlass.Constexpr[int],
    o_heads: cutlass.Constexpr[int],
    o_inner: cutlass.Constexpr[int],
    mOx: cute.Tensor,
    mO: cute.Tensor,
    grid_x: cutlass.Int32,
    stream: cuda.CUstream,
) -> None:
    gather_o_kernel(num_householder, o_heads, o_inner, mOx, mO).launch(grid=(grid_x, 1, 1), block=(BLOCK, 1, 1), stream=stream)


@cute.jit
def gather_bwd_launch(
    num_householder: cutlass.Constexpr[int],
    dq_heads: cutlass.Constexpr[int],
    dq_inner: cutlass.Constexpr[int],
    dg_inner: cutlass.Constexpr[int],
    dg_vec: cutlass.Constexpr[int],
    mDqx: cute.Tensor,
    mDq: cute.Tensor,
    mDgx: cute.Tensor,
    mDg: cute.Tensor,
    grid_x: cutlass.Int32,
    stream: cuda.CUstream,
) -> None:
    gather_bwd_kernel(num_householder, dq_heads, dq_inner, dg_inner, dg_vec, mDqx, mDq, mDgx, mDg).launch(
        grid=(grid_x, 1, 1), block=(BLOCK, 1, 1), stream=stream
    )


compiled_cache = {}


class PackFwdRecipe(NamedTuple):
    """Build-time facts of one GDP forward pack launch (g expand + optional
    q scatter).  Produced by :func:`build_pack_fwd`."""

    compiled: object
    with_q: bool
    grid_x: int


def run_pack_fwd(r, g, g_x, q, q_x, stream) -> None:
    """The lowered forward pack launch: no validation, no key build."""
    r.compiled(g, g_x, q if r.with_q else g_x, q_x if r.with_q else g_x, r.grid_x, cuda.CUstream(int(stream)))


def build_pack_fwd(g, g_x, q, q_x, num_householder, stream) -> PackFwdRecipe:
    """Compile (cached), run once, and bake the forward pack: g expands onto
    sub-token 0 with zero fill; q (when not fused into the l2norm store)
    scatters onto sub-token ``n - 1``."""
    n = int(num_householder)
    with_q = q is not None
    g_inner = int(g.shape[1])
    g_vec = 4 if (g_inner % 4 == 0 and g.stride()[0] % 4 == 0) else 1
    q_heads = int(q.shape[1]) if with_q else 1
    q_inner = int(q.shape[2]) // 2 if with_q else 4
    total = int(g_x.shape[0]) * g_inner // g_vec + (int(q.shape[0]) * q_heads * q_inner // 4 if with_q else 0)
    grid_x = -(-total // BLOCK)
    cu_stream = cuda.CUstream(int(stream))
    key = ("pack_fwd", with_q, n, g_inner, g_vec, q_heads, q_inner, str(g.dtype), str(q.dtype) if with_q else None)
    if key not in compiled_cache:
        g_x_c = from_dlpack(g_x, assumed_align=4)
        g_x_c.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
        q_x_c = g_x_c
        if with_q:
            q_x_c = from_dlpack(q_x, assumed_align=4)
            q_x_c.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2), divisibility=1)
        compiled_cache[key] = cute.compile(
            pack_fwd_launch,
            with_q,
            n,
            g_inner,
            g_vec,
            q_heads,
            q_inner,
            from_dlpack(g, assumed_align=4).mark_layout_dynamic(leading_dim=1),
            g_x_c,
            from_dlpack(q, assumed_align=4).mark_layout_dynamic(leading_dim=2) if with_q else g_x_c,
            q_x_c,
            cutlass.Int32(grid_x),
            cu_stream,
            options="--enable-tvm-ffi",
        )
    r = PackFwdRecipe(compiled_cache[key], with_q, grid_x)
    run_pack_fwd(r, g, g_x, q, q_x, stream)
    return r


class PackBwdRecipe(NamedTuple):
    """Build-time facts of one GDP backward pack launch (q/g/dO zero-fill
    expands).  Produced by :func:`build_pack_bwd`."""

    compiled: object
    grid_x: int


def run_pack_bwd(r, q, q_x, g, g_x, do, do_x, stream) -> None:
    """The lowered backward pack launch: no validation, no key build."""
    r.compiled(q, q_x, g, g_x, do, do_x, r.grid_x, cuda.CUstream(int(stream)))


def build_pack_bwd(q, q_x, g, g_x, do, do_x, num_householder, stream) -> PackBwdRecipe:
    """Compile (cached), run once, and bake the backward pack: q and dO
    expand onto sub-token ``n - 1``, g onto sub-token 0, all zero-filled
    (off-phase backward rows are consumed by the kernels)."""
    n = int(num_householder)
    q_heads, q_inner = int(q.shape[1]), int(q.shape[2]) // 2
    g_inner = int(g.shape[1])
    g_vec = 4 if (g_inner % 4 == 0 and g.stride()[0] % 4 == 0) else 1
    do_heads, do_inner = int(do.shape[1]), int(do.shape[2]) // 2
    total = int(q_x.shape[0]) * q_heads * q_inner // 4 + int(g_x.shape[0]) * g_inner // g_vec + int(do_x.shape[0]) * do_heads * do_inner // 4
    grid_x = -(-total // BLOCK)
    cu_stream = cuda.CUstream(int(stream))
    key = ("pack_bwd", n, q_heads, q_inner, g_inner, g_vec, do_heads, do_inner, str(q.dtype), str(do.dtype))
    if key not in compiled_cache:
        q_x_c = from_dlpack(q_x, assumed_align=4)
        q_x_c.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2), divisibility=1)
        g_x_c = from_dlpack(g_x, assumed_align=4)
        g_x_c.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
        do_x_c = from_dlpack(do_x, assumed_align=4)
        do_x_c.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2), divisibility=1)
        compiled_cache[key] = cute.compile(
            pack_bwd_launch,
            n,
            q_heads,
            q_inner,
            g_inner,
            g_vec,
            do_heads,
            do_inner,
            from_dlpack(q, assumed_align=4).mark_layout_dynamic(leading_dim=2),
            q_x_c,
            from_dlpack(g, assumed_align=4).mark_layout_dynamic(leading_dim=1),
            g_x_c,
            from_dlpack(do, assumed_align=4).mark_layout_dynamic(leading_dim=2),
            do_x_c,
            cutlass.Int32(grid_x),
            cu_stream,
            options="--enable-tvm-ffi",
        )
    r = PackBwdRecipe(compiled_cache[key], grid_x)
    run_pack_bwd(r, q, q_x, g, g_x, do, do_x, stream)
    return r


class GatherORecipe(NamedTuple):
    """Build-time facts of one O gather launch.  Produced by
    :func:`build_gather_o`."""

    compiled: object
    grid_x: int


def run_gather_o(r, o_x, o, stream) -> None:
    """The lowered O gather launch: no validation, no key build."""
    r.compiled(o_x, o, r.grid_x, cuda.CUstream(int(stream)))


def build_gather_o(o_x, o, num_householder, stream) -> GatherORecipe:
    """Compile (cached), run once, and bake the O gather: sub-token ``n - 1``
    rows of the expanded output copy back to real-token rows."""
    n = int(num_householder)
    o_heads, o_inner = int(o.shape[1]), int(o.shape[2]) // 2
    grid_x = -(-(int(o.shape[0]) * o_heads * o_inner // 4) // BLOCK)
    cu_stream = cuda.CUstream(int(stream))
    key = ("gather_o", n, o_heads, o_inner, str(o.dtype))
    if key not in compiled_cache:
        o_x_c = from_dlpack(o_x, assumed_align=4)
        o_x_c.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2), divisibility=1)
        compiled_cache[key] = cute.compile(
            gather_o_launch,
            n,
            o_heads,
            o_inner,
            o_x_c,
            from_dlpack(o, assumed_align=4).mark_layout_dynamic(leading_dim=2),
            cutlass.Int32(grid_x),
            cu_stream,
            options="--enable-tvm-ffi",
        )
    r = GatherORecipe(compiled_cache[key], grid_x)
    run_gather_o(r, o_x, o, stream)
    return r


class GatherBwdRecipe(NamedTuple):
    """Build-time facts of one dQ/dG gather launch.  Produced by
    :func:`build_gather_bwd`."""

    compiled: object
    grid_x: int


def run_gather_bwd(r, dq_x, dq, dg_x, dg, stream) -> None:
    """The lowered dQ/dG gather launch: no validation, no key build."""
    r.compiled(dq_x, dq, dg_x, dg, r.grid_x, cuda.CUstream(int(stream)))


def build_gather_bwd(dq_x, dq, dg_x, dg, num_householder, stream) -> GatherBwdRecipe:
    """Compile (cached), run once, and bake the backward gathers: dQ from
    sub-token ``n - 1``, dG from sub-token 0."""
    n = int(num_householder)
    dq_heads, dq_inner = int(dq.shape[1]), int(dq.shape[2]) // 2
    dg_inner = int(dg.shape[1])
    dg_vec = 4 if (dg_inner % 4 == 0 and dg.stride()[0] % 4 == 0) else 1
    total = int(dq.shape[0]) * dq_heads * dq_inner // 4 + int(dg.shape[0]) * dg_inner // dg_vec
    grid_x = -(-total // BLOCK)
    cu_stream = cuda.CUstream(int(stream))
    key = ("gather_bwd", n, dq_heads, dq_inner, dg_inner, dg_vec, str(dq.dtype))
    if key not in compiled_cache:
        dq_x_c = from_dlpack(dq_x, assumed_align=4)
        dq_x_c.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2), divisibility=1)
        dg_x_c = from_dlpack(dg_x, assumed_align=4)
        dg_x_c.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
        compiled_cache[key] = cute.compile(
            gather_bwd_launch,
            n,
            dq_heads,
            dq_inner,
            dg_inner,
            dg_vec,
            dq_x_c,
            from_dlpack(dq, assumed_align=4).mark_layout_dynamic(leading_dim=2),
            dg_x_c,
            from_dlpack(dg, assumed_align=4).mark_layout_dynamic(leading_dim=1),
            cutlass.Int32(grid_x),
            cu_stream,
            options="--enable-tvm-ffi",
        )
    r = GatherBwdRecipe(compiled_cache[key], grid_x)
    run_gather_bwd(r, dq_x, dq, dg_x, dg, stream)
    return r
