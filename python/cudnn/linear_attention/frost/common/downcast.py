# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Initial-state staging for the FROST LA backward kernels: copy the caller's
``[N, HO, K, V]`` state (fp32 or io dtype, padded outer strides fine) into
the compact io-dtype buffer the per-(b,h) state descriptors read.

Alignment is the caller's contract, as everywhere TMA is involved: 16-byte
aligned bases, and outer strides that keep every 8-element V chunk address
16-byte aligned (compact buffers trivially qualify)."""

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.arch.nvvm_wrappers import inline_ptx
from cutlass.cute.runtime import from_dlpack

from cudnn.frost.tile_dsl.pointwise import f16x2_to_f32, fp32_to_fp16


@cute.kernel
def downcast_state_kernel(
    mState0: cute.Tensor,
    mOut: cute.Tensor,
    n_k: cutlass.Int32,
    threads_per_row: cutlass.Int32,
    rows_per_cta: cutlass.Int32,
) -> None:
    """Vectorized copy of the ``[N, HO, K, V]`` initial state into the
    io-dtype buffer the backward's static state descriptor reads: grid
    (K-tiles, HO, N), one 8-element V chunk per thread as 128-bit loads and
    one 128-bit store, source read through its (dynamic) strides so padded
    outer layouts stage zero-copy.  fp32 sources convert through packed
    ``cvt.rn.{f16,bf16}x2.f32``; same-dtype io sources copy words verbatim;
    a 16-bit cross convert unpacks to fp32 pairs and repacks."""
    bid = cute.arch.block_idx()
    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    k_idx = cutlass.Int32(bid[0]) * rows_per_cta + tidx // threads_per_row
    v0 = (tidx % threads_per_row) * cutlass.Int32(8)
    n_idx = cutlass.Int32(bid[2])
    h_idx = cutlass.Int32(bid[1])
    if k_idx < n_k:
        src_elems = (
            cutlass.Int64(n_idx) * cutlass.Int64(mState0.stride[0])
            + cutlass.Int64(h_idx) * cutlass.Int64(mState0.stride[1])
            + cutlass.Int64(k_idx) * cutlass.Int64(mState0.stride[2])
            + cutlass.Int64(v0)
        )
        dst_elems = (
            cutlass.Int64(n_idx) * cutlass.Int64(mOut.stride[0])
            + cutlass.Int64(h_idx) * cutlass.Int64(mOut.stride[1])
            + cutlass.Int64(k_idx) * cutlass.Int64(mOut.stride[2])
            + cutlass.Int64(v0)
        )
        dst_addr = mOut.iterator.toint() + dst_elems * cutlass.Int64(2)
        if cutlass.const_expr(mState0.element_type == cutlass.Float32):
            src_addr = mState0.iterator.toint() + src_elems * cutlass.Int64(4)
            f0, f1, f2, f3 = inline_ptx(
                "ld.global.v4.f32 {$0, $1, $2, $3}, [$4];",
                write_only_types=[cutlass.Float32, cutlass.Float32, cutlass.Float32, cutlass.Float32],
                read_only_args=[src_addr],
            )
            f4, f5, f6, f7 = inline_ptx(
                "ld.global.v4.f32 {$0, $1, $2, $3}, [$4];",
                write_only_types=[cutlass.Float32, cutlass.Float32, cutlass.Float32, cutlass.Float32],
                read_only_args=[src_addr + cutlass.Int64(16)],
            )
            w0 = fp32_to_fp16(f0, f1, dtype=mOut.element_type)
            w1 = fp32_to_fp16(f2, f3, dtype=mOut.element_type)
            w2 = fp32_to_fp16(f4, f5, dtype=mOut.element_type)
            w3 = fp32_to_fp16(f6, f7, dtype=mOut.element_type)
        else:
            src_addr = mState0.iterator.toint() + src_elems * cutlass.Int64(2)
            w0, w1, w2, w3 = inline_ptx(
                "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
                write_only_types=[cutlass.Int32, cutlass.Int32, cutlass.Int32, cutlass.Int32],
                read_only_args=[src_addr],
            )
            if cutlass.const_expr(mState0.element_type != mOut.element_type):
                lo0, hi0 = f16x2_to_f32(w0, dtype=mState0.element_type)
                lo1, hi1 = f16x2_to_f32(w1, dtype=mState0.element_type)
                lo2, hi2 = f16x2_to_f32(w2, dtype=mState0.element_type)
                lo3, hi3 = f16x2_to_f32(w3, dtype=mState0.element_type)
                w0 = fp32_to_fp16(lo0, hi0, dtype=mOut.element_type)
                w1 = fp32_to_fp16(lo1, hi1, dtype=mOut.element_type)
                w2 = fp32_to_fp16(lo2, hi2, dtype=mOut.element_type)
                w3 = fp32_to_fp16(lo3, hi3, dtype=mOut.element_type)
        inline_ptx(
            "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
            read_only_args=[dst_addr, w0, w1, w2, w3],
        )


@cute.jit
def downcast_state_launch(
    state0: cute.Tensor,
    out: cute.Tensor,
    n_k: cutlass.Int32,
    threads_per_row: cutlass.Int32,
    rows_per_cta: cutlass.Int32,
    n_blocks: cutlass.Int32,
    ho: cutlass.Int32,
    n_seq: cutlass.Int32,
    stream: cuda.CUstream,
):
    downcast_state_kernel(
        state0,
        out,
        n_k,
        threads_per_row,
        rows_per_cta,
    ).launch(grid=(n_blocks, ho, n_seq), block=(128, 1, 1), stream=stream)


@functools.cache
def downcast_state_cache(key):
    return {}


def downcast_state(initial_state, out, *, stream):
    """Copy the initial state ``[N, HO, K, V]`` (fp32 or io dtype, padded
    outer strides fine) into ``out`` (io dtype, same shape, compact) — the
    buffer the backward's per-(b,h) state descriptors read."""
    if tuple(int(s_) for s_ in initial_state.shape) != tuple(int(s_) for s_ in out.shape):
        raise ValueError(f"initial_state must match the io state buffer shape {tuple(out.shape)}; got {tuple(initial_state.shape)}")
    n_seq, ho, k, v = (int(s_) for s_ in out.shape)
    if v % 8 != 0:
        raise ValueError(f"state V dim must be a multiple of 8 (8-element staging chunks); got {v}")
    if v > 1024:
        raise ValueError(f"state V dim must be <= 1024 (one 128-thread block stages a full row); got {v}")
    threads_per_row = v // 8
    rows_per_cta = max(128 // threads_per_row, 1)
    n_blocks = (k + rows_per_cta - 1) // rows_per_cta
    key = (str(initial_state.dtype), str(out.dtype))
    cache = downcast_state_cache(key)
    cu_stream = cuda.CUstream(int(stream))
    if "compiled" not in cache:
        state0_c = from_dlpack(initial_state, assumed_align=16).mark_layout_dynamic(leading_dim=3)
        out_c = from_dlpack(out, assumed_align=16).mark_layout_dynamic(leading_dim=3)
        cache["compiled"] = cute.compile(
            downcast_state_launch,
            state0_c,
            out_c,
            cutlass.Int32(k),
            cutlass.Int32(threads_per_row),
            cutlass.Int32(rows_per_cta),
            cutlass.Int32(n_blocks),
            cutlass.Int32(ho),
            cutlass.Int32(n_seq),
            cu_stream,
            options="--enable-tvm-ffi",
        )
    cache["compiled"](initial_state, out, k, threads_per_row, rows_per_cta, n_blocks, ho, n_seq, cu_stream)
