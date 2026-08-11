# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Initial-state staging for the FROST LA backward kernels: copy the caller's
``[N, HO, K, V]`` state (fp32 or io dtype, padded outer strides fine) into
the compact io-dtype buffer the per-(b,h) state descriptors read."""

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def downcast_state_f16_kernel(
    mState0: cute.Tensor,
    mOut: cute.Tensor,
    n_k: cutlass.Int32,
    threads_per_row: cutlass.Int32,
    rows_per_cta: cutlass.Int32,
) -> None:
    """Row-chunk copy of the ``[N, HO, K, V]`` initial state into the io-dtype
    buffer the backward's static state descriptor reads: grid (K-tiles, HO, N),
    one 8-element V chunk per thread, source read through its (dynamic)
    strides so padded outer layouts stage zero-copy."""
    bid = cute.arch.block_idx()
    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    k_idx = cutlass.Int32(bid[0]) * rows_per_cta + tidx // threads_per_row
    v0 = (tidx % threads_per_row) * cutlass.Int32(8)
    n_idx = cutlass.Int32(bid[2])
    h_idx = cutlass.Int32(bid[1])
    if k_idx < n_k:
        for i in cutlass.range_constexpr(8):
            mOut[n_idx, h_idx, k_idx, v0 + i] = mState0[n_idx, h_idx, k_idx, v0 + i].to(mOut.element_type)


@cute.jit
def downcast_state_f16(
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
    downcast_state_f16_kernel(
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
    buffer the backward's per-(b,h) state descriptors read. Stride-aware:
    the source is read through its own layout, never reshaped or copied
    host-side."""
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
            downcast_state_f16,
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
