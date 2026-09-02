# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sub-token row expansion/gather kernels for GDP."""

from typing import NamedTuple

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda

from cutlass.cute.runtime import from_dlpack

from cudnn.frost.tile_dsl.barrier import launch_dependent_grids, wait_on_dependent_grids
from cudnn.frost.tile_dsl.tma import ld_global_v4, st_global_v4

USE_PDL = True

BLOCK = 256


@cute.jit
def strided_row_addr(m, tok, h_idx, w_off, h_count):
    bpe = cutlass.const_expr(m.element_type.width // 8)
    elems = tok * cutlass.Int64(m.stride[0])
    if cutlass.const_expr(h_count > 1):
        elems = elems + h_idx * cutlass.Int64(m.stride[1])
    return m.iterator.toint() + elems * cutlass.Int64(bpe) + w_off * cutlass.Int64(4)


@cute.jit
def expand_rows(mSrc, mDst, chunk, h_count, inner_words, num_householder, phase):
    """One chunk of zero-fill expansion: every expanded row is stored, only
    the ``phase`` sub-token loads the source."""
    row_chunks = cutlass.const_expr(inner_words // 4)
    seg = chunk // cutlass.Int64(row_chunks)
    w_off = (chunk - seg * cutlass.Int64(row_chunks)) * cutlass.Int64(4)
    row_out = seg // cutlass.Int64(h_count)
    h_idx = seg - row_out * cutlass.Int64(h_count)
    tok = row_out // cutlass.Int64(num_householder)
    slot = row_out - tok * cutlass.Int64(num_householder)
    src_addr = strided_row_addr(mSrc, tok, h_idx, w_off, h_count)
    dst_addr = mDst.iterator.toint() + chunk * cutlass.Int64(16)
    w0 = cutlass.Int32(0)
    w1 = cutlass.Int32(0)
    w2 = cutlass.Int32(0)
    w3 = cutlass.Int32(0)
    if slot == cutlass.Int64(phase):
        w0, w1, w2, w3 = ld_global_v4(src_addr, cutlass.Int32)
    st_global_v4(dst_addr, (w0, w1, w2, w3), cutlass.Int32)


@cute.jit
def scatter_rows(mSrc, mDst, chunk, h_count, inner_words, num_householder, phase):
    """One chunk of phase-row scatter: real-token rows land on
    sub-token ``phase``; the other expanded rows are left untouched."""
    row_chunks = cutlass.const_expr(inner_words // 4)
    seg = chunk // cutlass.Int64(row_chunks)
    w_off = (chunk - seg * cutlass.Int64(row_chunks)) * cutlass.Int64(4)
    tok = seg // cutlass.Int64(h_count)
    h_idx = seg - tok * cutlass.Int64(h_count)
    src_addr = strided_row_addr(mSrc, tok, h_idx, w_off, h_count)
    dst_words = ((tok * cutlass.Int64(num_householder) + cutlass.Int64(phase)) * cutlass.Int64(h_count) + h_idx) * cutlass.Int64(inner_words) + w_off
    dst_addr = mDst.iterator.toint() + dst_words * cutlass.Int64(4)
    w0, w1, w2, w3 = ld_global_v4(src_addr, cutlass.Int32)
    st_global_v4(dst_addr, (w0, w1, w2, w3), cutlass.Int32)


@cute.jit
def gather_rows(mSrc, mDst, chunk, h_count, inner_words, num_householder, phase):
    """One chunk of phase-row gather: sub-token ``phase`` rows of the
    expanded source copy back to real-token rows."""
    row_chunks = cutlass.const_expr(inner_words // 4)
    seg = chunk // cutlass.Int64(row_chunks)
    w_off = (chunk - seg * cutlass.Int64(row_chunks)) * cutlass.Int64(4)
    tok = seg // cutlass.Int64(h_count)
    h_idx = seg - tok * cutlass.Int64(h_count)
    src_words = ((tok * cutlass.Int64(num_householder) + cutlass.Int64(phase)) * cutlass.Int64(h_count) + h_idx) * cutlass.Int64(inner_words) + w_off
    src_addr = mSrc.iterator.toint() + src_words * cutlass.Int64(4)
    dst_addr = strided_row_addr(mDst, tok, h_idx, w_off, h_count)
    w0, w1, w2, w3 = ld_global_v4(src_addr, cutlass.Int32)
    st_global_v4(dst_addr, (w0, w1, w2, w3), cutlass.Int32)


@cute.kernel
def pack_fwd_kernel(
    num_householder: cutlass.Constexpr[int],
    q_heads: cutlass.Constexpr[int],
    q_inner: cutlass.Constexpr[int],
    mQ: cute.Tensor,
    mQx: cute.Tensor,
) -> None:
    if cutlass.const_expr(USE_PDL):
        wait_on_dependent_grids()
    tidx, _, _ = cute.arch.thread_idx()
    bidx = cute.arch.block_idx()[0]
    chunk_idx = cutlass.Int64(cutlass.Int32(bidx)) * cutlass.Int64(BLOCK) + cutlass.Int64(cutlass.Int32(tidx))
    if chunk_idx < cutlass.Int64(mQ.shape[0]) * cutlass.Int64(cutlass.const_expr(q_heads * q_inner // 4)):
        scatter_rows(mQ, mQx, chunk_idx, q_heads, q_inner, num_householder, num_householder - 1)
    if cutlass.const_expr(USE_PDL):
        launch_dependent_grids()


@cute.kernel
def pack_bwd_kernel(
    with_q: cutlass.Constexpr[bool],
    num_householder: cutlass.Constexpr[int],
    q_heads: cutlass.Constexpr[int],
    q_inner: cutlass.Constexpr[int],
    do_heads: cutlass.Constexpr[int],
    do_inner: cutlass.Constexpr[int],
    mQ: cute.Tensor,
    mQx: cute.Tensor,
    mDo: cute.Tensor,
    mDox: cute.Tensor,
) -> None:
    if cutlass.const_expr(USE_PDL):
        wait_on_dependent_grids()
    tidx, _, _ = cute.arch.thread_idx()
    bidx = cute.arch.block_idx()[0]
    chunk_idx = cutlass.Int64(cutlass.Int32(bidx)) * cutlass.Int64(BLOCK) + cutlass.Int64(cutlass.Int32(tidx))
    q_chunks = cutlass.Int64(0)
    if cutlass.const_expr(with_q):
        q_chunks = cutlass.Int64(mQx.shape[0]) * cutlass.Int64(cutlass.const_expr(q_heads * q_inner // 4))
    if cutlass.const_expr(with_q) and chunk_idx < q_chunks:
        expand_rows(mQ, mQx, chunk_idx, q_heads, q_inner, num_householder, num_householder - 1)
    else:
        chunk = chunk_idx - q_chunks
        if chunk < cutlass.Int64(mDox.shape[0]) * cutlass.Int64(cutlass.const_expr(do_heads * do_inner // 4)):
            expand_rows(mDo, mDox, chunk, do_heads, do_inner, num_householder, num_householder - 1)
    if cutlass.const_expr(USE_PDL):
        launch_dependent_grids()


@cute.kernel
def gather_dq_kernel(
    num_householder: cutlass.Constexpr[int],
    dq_heads: cutlass.Constexpr[int],
    dq_inner: cutlass.Constexpr[int],
    mDqx: cute.Tensor,
    mDq: cute.Tensor,
) -> None:
    if cutlass.const_expr(USE_PDL):
        wait_on_dependent_grids()
    tidx, _, _ = cute.arch.thread_idx()
    bidx = cute.arch.block_idx()[0]
    chunk_idx = cutlass.Int64(cutlass.Int32(bidx)) * cutlass.Int64(BLOCK) + cutlass.Int64(cutlass.Int32(tidx))
    if chunk_idx < cutlass.Int64(mDq.shape[0]) * cutlass.Int64(cutlass.const_expr(dq_heads * dq_inner // 4)):
        gather_rows(mDqx, mDq, chunk_idx, dq_heads, dq_inner, num_householder, num_householder - 1)
    if cutlass.const_expr(USE_PDL):
        launch_dependent_grids()


@cute.jit
def pack_fwd_launch(
    num_householder: cutlass.Constexpr[int],
    q_heads: cutlass.Constexpr[int],
    q_inner: cutlass.Constexpr[int],
    mQ: cute.Tensor,
    mQx: cute.Tensor,
    grid_x: cutlass.Int32,
    stream: cuda.CUstream,
) -> None:
    pack_fwd_kernel(num_householder, q_heads, q_inner, mQ, mQx).launch(grid=(grid_x, 1, 1), block=(BLOCK, 1, 1), stream=stream, use_pdl=USE_PDL)


@cute.jit
def pack_bwd_launch(
    with_q: cutlass.Constexpr[bool],
    num_householder: cutlass.Constexpr[int],
    q_heads: cutlass.Constexpr[int],
    q_inner: cutlass.Constexpr[int],
    do_heads: cutlass.Constexpr[int],
    do_inner: cutlass.Constexpr[int],
    mQ: cute.Tensor,
    mQx: cute.Tensor,
    mDo: cute.Tensor,
    mDox: cute.Tensor,
    grid_x: cutlass.Int32,
    stream: cuda.CUstream,
) -> None:
    pack_bwd_kernel(with_q, num_householder, q_heads, q_inner, do_heads, do_inner, mQ, mQx, mDo, mDox).launch(
        grid=(grid_x, 1, 1), block=(BLOCK, 1, 1), stream=stream, use_pdl=USE_PDL
    )


@cute.jit
def gather_dq_launch(
    num_householder: cutlass.Constexpr[int],
    dq_heads: cutlass.Constexpr[int],
    dq_inner: cutlass.Constexpr[int],
    mDqx: cute.Tensor,
    mDq: cute.Tensor,
    grid_x: cutlass.Int32,
    stream: cuda.CUstream,
) -> None:
    gather_dq_kernel(num_householder, dq_heads, dq_inner, mDqx, mDq).launch(grid=(grid_x, 1, 1), block=(BLOCK, 1, 1), stream=stream, use_pdl=USE_PDL)


compiled_cache = {}


class PackFwdRecipe(NamedTuple):
    """Build-time facts of one GDP forward pack launch (the q scatter).
    Produced by :func:`build_pack_fwd`."""

    compiled: object
    grid_x: int


def run_pack_fwd(r, q, q_x, stream) -> None:
    """The lowered forward pack launch: no validation, no key build."""
    r.compiled(q, q_x, r.grid_x, cuda.CUstream(int(stream)))


def build_pack_fwd(q, q_x, num_householder, stream) -> PackFwdRecipe:
    """Compile (cached), run once, and bake the forward pack: q scatters onto
    sub-token ``n - 1``."""
    n = int(num_householder)
    q_heads = int(q.shape[1])
    q_inner = int(q.shape[2]) // 2
    grid_x = -(-(int(q.shape[0]) * q_heads * q_inner // 4) // BLOCK)
    cu_stream = cuda.CUstream(int(stream))
    key = ("pack_fwd", n, q_heads, q_inner, str(q.dtype))
    if key not in compiled_cache:
        q_x_c = from_dlpack(q_x, assumed_align=4)
        q_x_c.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2), divisibility=1)
        compiled_cache[key] = cute.compile(
            pack_fwd_launch,
            n,
            q_heads,
            q_inner,
            from_dlpack(q, assumed_align=4).mark_layout_dynamic(leading_dim=2),
            q_x_c,
            cutlass.Int32(grid_x),
            cu_stream,
            options="--enable-tvm-ffi",
        )
    r = PackFwdRecipe(compiled_cache[key], grid_x)
    run_pack_fwd(r, q, q_x, stream)
    return r


class PackBwdRecipe(NamedTuple):
    """Build-time facts of one GDP backward pack launch (q/dO zero-fill
    expands).  Produced by :func:`build_pack_bwd`."""

    compiled: object
    with_q: bool
    grid_x: int


def run_pack_bwd(r, q, q_x, do, do_x, stream) -> None:
    """The lowered backward pack launch: no validation, no key build."""
    r.compiled(
        q if r.with_q else do,
        q_x if r.with_q else do_x,
        do,
        do_x,
        r.grid_x,
        cuda.CUstream(int(stream)),
    )


def build_pack_bwd(q, q_x, do, do_x, num_householder, stream) -> PackBwdRecipe:
    """Compile (cached), run once, and bake the backward pack: q and dO
    expand onto sub-token ``n - 1``, zero-filled.  ``q=None`` drops the q leg."""
    n = int(num_householder)
    with_q = q is not None
    q_heads = int(q.shape[1]) if with_q else 1
    q_inner = int(q.shape[2]) // 2 if with_q else 4
    do_heads, do_inner = int(do.shape[1]), int(do.shape[2]) // 2
    q_chunks = int(q_x.shape[0]) * q_heads * q_inner // 4 if with_q else 0
    total = q_chunks + int(do_x.shape[0]) * do_heads * do_inner // 4
    grid_x = -(-total // BLOCK)
    cu_stream = cuda.CUstream(int(stream))
    key = ("pack_bwd", with_q, n, q_heads, q_inner, do_heads, do_inner, str(q.dtype) if with_q else None, str(do.dtype))
    if key not in compiled_cache:
        do_c = from_dlpack(do, assumed_align=4).mark_layout_dynamic(leading_dim=2)
        do_x_c = from_dlpack(do_x, assumed_align=4)
        do_x_c.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2), divisibility=1)
        q_c = do_c
        q_x_c = do_x_c
        if with_q:
            q_c = from_dlpack(q, assumed_align=4).mark_layout_dynamic(leading_dim=2)
            q_x_c = from_dlpack(q_x, assumed_align=4)
            q_x_c.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2), divisibility=1)
        compiled_cache[key] = cute.compile(
            pack_bwd_launch,
            with_q,
            n,
            q_heads,
            q_inner,
            do_heads,
            do_inner,
            q_c,
            q_x_c,
            do_c,
            do_x_c,
            cutlass.Int32(grid_x),
            cu_stream,
            options="--enable-tvm-ffi",
        )
    r = PackBwdRecipe(compiled_cache[key], with_q, grid_x)
    run_pack_bwd(r, q, q_x, do, do_x, stream)
    return r


class GatherDqRecipe(NamedTuple):
    """Build-time facts of one dQ gather launch.  Produced by
    :func:`build_gather_dq`."""

    compiled: object
    grid_x: int


def run_gather_dq(r, dq_x, dq, stream) -> None:
    """The lowered dQ gather launch: no validation, no key build."""
    r.compiled(dq_x, dq, r.grid_x, cuda.CUstream(int(stream)))


def build_gather_dq(dq_x, dq, num_householder, stream) -> GatherDqRecipe:
    """Compile (cached), run once, and bake the dQ gather from sub-token
    ``n - 1``."""
    n = int(num_householder)
    dq_heads, dq_inner = int(dq.shape[1]), int(dq.shape[2]) // 2
    grid_x = -(-(int(dq.shape[0]) * dq_heads * dq_inner // 4) // BLOCK)
    cu_stream = cuda.CUstream(int(stream))
    key = ("gather_dq", n, dq_heads, dq_inner, str(dq.dtype))
    if key not in compiled_cache:
        dq_x_c = from_dlpack(dq_x, assumed_align=4)
        dq_x_c.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2), divisibility=1)
        compiled_cache[key] = cute.compile(
            gather_dq_launch,
            n,
            dq_heads,
            dq_inner,
            dq_x_c,
            from_dlpack(dq, assumed_align=4).mark_layout_dynamic(leading_dim=2),
            cutlass.Int32(grid_x),
            cu_stream,
            options="--enable-tvm-ffi",
        )
    r = GatherDqRecipe(compiled_cache[key], grid_x)
    run_gather_dq(r, dq_x, dq, stream)
    return r
