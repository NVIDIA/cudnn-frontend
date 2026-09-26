# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Pointer host shared by SM120 dense, split and THD half-precision templates."""

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
    lse_ext: cutlass.Int32,
    scale_softmax_log2: cutlass.Float32,
    seq_q_lens_addr: cutlass.Int64,
    thd_q_lens_ptr: Optional[cute.Pointer],
    thd_kv_lens_ptr: Optional[cute.Pointer],
    thd_lens_form: Optional[cutlass.Int32],
    n_thd_units: cutlass.Int32,
    kernel: cutlass.Constexpr,
    qh: cutlass.Constexpr[int],
    kh: cutlass.Constexpr[int],
    d_qk: cutlass.Constexpr[int],
    d_v: cutlass.Constexpr[int],
    persistent_ctas: cutlass.Constexpr[int],
    thd_max_sq: cutlass.Constexpr[int],
    stream: cuda_driver.CUstream,
):
    """The binder validates runtime geometry; head counts and dimensions are plan constants."""
    b, _, _, sq, skv, _ = problem_size
    q = _bshd(q_ptr, b, sq, qh, d_qk, q_strides, kernel.thd_varlen)
    k = _bshd(k_ptr, b, skv, kh, d_qk, k_strides, kernel.thd_varlen)
    v = _bshd(v_ptr, b, skv, kh, d_v, v_strides, kernel.thd_varlen)
    o = _bshd(o_ptr, b * kernel.split_kv, sq, qh, d_v, o_strides, kernel.thd_varlen)
    lse = None
    if cutlass.const_expr(lse_ptr is not None):
        if cutlass.const_expr(kernel.thd_varlen):
            if cutlass.const_expr(kernel.thd_lse_padded):
                lse = cute.make_tensor(lse_ptr, cute.make_layout((b, qh, thd_max_sq), stride=lse_strides))
            elif cutlass.const_expr(kernel.thd_lse_head_major):
                lse = cute.make_tensor(lse_ptr, cute.make_layout((qh, lse_ext), stride=(cutlass.Int64(lse_ext), 1)))
            else:
                lse = cute.make_tensor(lse_ptr, cute.make_layout((sq, qh), stride=(qh, 1)))
        else:
            lse = cute.make_tensor(lse_ptr, cute.make_layout((b * kernel.split_kv, qh, sq), stride=lse_strides))
    sinks = None
    if cutlass.const_expr(kernel.has_sink):
        sinks = cute.make_tensor(sinks_ptr, cute.make_layout((qh,), stride=(1,)))
    q_lens_ptr = cute.make_ptr(cutlass.Int32, seq_q_lens_addr, cute.AddressSpace.gmem, assumed_align=4)
    q_lens = cute.make_tensor(q_lens_ptr, cute.make_layout((b,), stride=(1,)))
    kv_lens = cute.make_tensor(meta_ptr, cute.make_layout((4 * b + 4 if kernel.thd_varlen else b,), stride=(1,)))
    thd_q_lens = None
    thd_kv_lens = None
    n_ctas = cutlass.Int32(persistent_ctas)
    if cutlass.const_expr(kernel.thd_varlen):
        # The length form is runtime metadata, including mixed CU/per-batch forms.
        # The common binder validated the exact element count before this call.
        thd_q_lens = cute.make_tensor(thd_q_lens_ptr, cute.make_layout((b + (thd_lens_form & 1),), stride=(1,)))
        thd_kv_lens = cute.make_tensor(thd_kv_lens_ptr, cute.make_layout((b + ((thd_lens_form >> 1) & 1),), stride=(1,)))
        n_ctas = n_thd_units
    kernel(
        q,
        k,
        v,
        o,
        lse,
        sinks,
        q_lens,
        kv_lens,
        scale_softmax_log2,
        cutlass.Int32(thd_max_sq),
        thd_q_lens,
        thd_kv_lens,
        thd_lens_form,
        n_ctas,
        stream,
        prepared=True,
    )


def compile_host(kernel, dtype, qh, kh, d_qk, d_v, has_lse, persistent_ctas, cache_key, thd_max_sq=0):
    """Compile one pointer entry with Int64 stride leaves and a fixed head geometry."""
    if kernel.thd_varlen and kernel.split_kv != 1:
        raise NotImplementedError("SM120 THD does not support split-KV")
    if kernel.split_kv > 1 and not has_lse:
        raise ValueError("SM120 split-KV requires partial LSE")

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
        cutlass.Int32(0),
        cutlass.Float32(0),
        cutlass.Int64(0),
        ptr(cutlass.Int32, 4) if kernel.thd_varlen else None,
        ptr(cutlass.Int32, 4) if kernel.thd_varlen else None,
        cutlass.Int32(0) if kernel.thd_varlen else None,
        cutlass.Int32(0),
        kernel,
        qh,
        kh,
        d_qk,
        d_v,
        persistent_ctas,
        thd_max_sq,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options="--enable-tvm-ffi",
        cache_key=cache_key,
        symbol="frost_sdpa_fwd_prepared",
    )
