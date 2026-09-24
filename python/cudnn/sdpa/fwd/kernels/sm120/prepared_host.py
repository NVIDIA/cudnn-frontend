# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Pointer host shared by the SM120 dense half-precision templates."""

from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as cuda_driver

from cudnn.frost.compiled_cache import compile_cached
from cudnn.sdpa.fwd.kernels._common_blackwell import _bshd


@cute.jit
def host(
    q_ptr: cute.Pointer,
    k_ptr: cute.Pointer,
    v_ptr: cute.Pointer,
    o_ptr: cute.Pointer,
    lse_ptr: Optional[cute.Pointer],
    sinks_ptr: cute.Pointer,
    meta_ptr: cute.Pointer,
    o_desc_ptr: cute.Pointer,
    problem_size: Tuple[int, int, int, int, int, int],
    q_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    k_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    v_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    o_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    lse_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    scale_softmax_log2: cutlass.Float32,
    seq_q_lens_addr: cutlass.Int64,
    kernel: cutlass.Constexpr,
    qh: cutlass.Constexpr[int],
    kh: cutlass.Constexpr[int],
    d_qk: cutlass.Constexpr[int],
    d_v: cutlass.Constexpr[int],
    persistent_ctas: cutlass.Constexpr[int],
    stream: cuda_driver.CUstream,
):
    """The binder validates runtime geometry; head counts and dimensions are plan constants."""
    b, _, _, sq, skv, _ = problem_size
    q = _bshd(q_ptr, b, sq, qh, d_qk, q_strides, False)
    k = _bshd(k_ptr, b, skv, kh, d_qk, k_strides, False)
    v = _bshd(v_ptr, b, skv, kh, d_v, v_strides, False)
    o = _bshd(o_ptr, b, sq, qh, d_v, o_strides, False)
    lse = None
    if cutlass.const_expr(lse_ptr is not None):
        lse = cute.make_tensor(lse_ptr, cute.make_layout((b, qh, sq), stride=lse_strides))
    sinks = None
    if cutlass.const_expr(kernel.has_sink):
        sinks = cute.make_tensor(sinks_ptr, cute.make_layout((qh,), stride=(1,)))
    q_lens_ptr = cute.make_ptr(cutlass.Int32, seq_q_lens_addr, cute.AddressSpace.gmem, assumed_align=4)
    q_lens = cute.make_tensor(q_lens_ptr, cute.make_layout((b,), stride=(1,)))
    kv_lens = cute.make_tensor(meta_ptr, cute.make_layout((b,), stride=(1,)))
    kernel(
        q, k, v, o, lse, sinks, q_lens, kv_lens, scale_softmax_log2, cutlass.Int32(0), None, None, None, cutlass.Int32(persistent_ctas), stream, prepared=True
    )


def compile_host(kernel, dtype, qh, kh, d_qk, d_v, has_lse, persistent_ctas, cache_key):
    """Compile one layout-generic dense entry with Int64 stride leaves."""
    if kernel.thd_varlen or kernel.split_kv != 1:
        raise NotImplementedError("prepared SM120 serves dense unsplit half-precision launches")

    def ptr(t, align=16):
        return cute.runtime.make_ptr(t, 16, cute.AddressSpace.gmem, assumed_align=align)

    strides = (cutlass.Int64(0),) * 3
    return compile_cached(
        host,
        ptr(dtype),
        ptr(dtype),
        ptr(dtype),
        ptr(dtype),
        ptr(cutlass.Float32, 4) if has_lse else None,
        ptr(cutlass.Float32, 4),
        ptr(cutlass.Int32, 4),
        ptr(cutlass.Int64),
        (0, 0, 0, 0, 0, 0),
        strides,
        strides,
        strides,
        strides,
        strides,
        cutlass.Float32(0),
        cutlass.Int64(0),
        kernel,
        qh,
        kh,
        d_qk,
        d_v,
        persistent_ctas,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options="--enable-tvm-ffi",
        cache_key=cache_key,
        symbol="frost_sdpa_fwd_prepared",
    )
