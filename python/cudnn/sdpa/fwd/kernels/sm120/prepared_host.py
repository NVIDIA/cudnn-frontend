# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Pointer host shared by SM120 half and per-tensor FP8 templates."""

from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as cuda_driver

from cudnn.frost.compiled_cache import compile_cached
from cudnn.sdpa.fwd.kernels._common_blackwell import _bshd
from cudnn.sdpa.fwd.kernels._quantized import _reset_amax_kernel, _unscale_amax_kernel


@cute.jit
def _operands(
    q_ptr: cute.Pointer,
    k_ptr: cute.Pointer,
    v_ptr: cute.Pointer,
    o_ptr: cute.Pointer,
    lse_ptr: Optional[cute.Pointer],
    sinks_ptr: cute.Pointer,
    meta_ptr: cute.Pointer,
    problem_size: Tuple[int, int, int, int, int, int],
    q_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    k_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    v_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    o_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    lse_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    lse_ext: cutlass.Int32,
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
):
    """The binder validates runtime geometry; head counts and dimensions are plan constants."""
    b, _, _, sq, skv, _ = problem_size
    q = _bshd(q_ptr, b, sq, qh, d_qk, q_strides, kernel.thd_varlen)
    k = _bshd(k_ptr, b, skv, kh, d_qk, k_strides, kernel.thd_varlen)
    v = _bshd(v_ptr, b, skv, kh, d_v, v_strides, kernel.thd_varlen)
    o = _bshd(o_ptr, b * kernel.split_kv, sq, qh, d_v // (2 if getattr(kernel, "o_block_scale", 0) == 16 else 1), o_strides, kernel.thd_varlen)
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
    return q, k, v, o, lse, sinks, q_lens, kv_lens, thd_q_lens, thd_kv_lens, n_ctas


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
    q, k, v, o, lse, sinks, q_lens, kv_lens, thd_q_lens, thd_kv_lens, n_ctas = _operands(
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        lse_ptr,
        sinks_ptr,
        meta_ptr,
        problem_size,
        q_strides,
        k_strides,
        v_strides,
        o_strides,
        lse_strides,
        lse_ext,
        seq_q_lens_addr,
        thd_q_lens_ptr,
        thd_kv_lens_ptr,
        thd_lens_form,
        n_thd_units,
        kernel,
        qh,
        kh,
        d_qk,
        d_v,
        persistent_ctas,
        thd_max_sq,
    )
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


@cute.jit
def fp8_host(
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
    descale_q_ptr: cute.Pointer,
    descale_k_ptr: cute.Pointer,
    descale_v_ptr: cute.Pointer,
    scale_o_ptr: Optional[cute.Pointer],
    amax_o_ptr: cute.Pointer,
    sf_o_ptr: Optional[cute.Pointer],
    has_amax: cutlass.Constexpr[bool],
    sfo_geometry: cutlass.Constexpr,
    kernel: cutlass.Constexpr,
    qh: cutlass.Constexpr[int],
    kh: cutlass.Constexpr[int],
    d_qk: cutlass.Constexpr[int],
    d_v: cutlass.Constexpr[int],
    persistent_ctas: cutlass.Constexpr[int],
    thd_max_sq: cutlass.Constexpr[int],
    stream: cuda_driver.CUstream,
):
    q, k, v, o, lse, sinks, q_lens, kv_lens, thd_q_lens, thd_kv_lens, n_ctas = _operands(
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        lse_ptr,
        sinks_ptr,
        meta_ptr,
        problem_size,
        q_strides,
        k_strides,
        v_strides,
        o_strides,
        lse_strides,
        lse_ext,
        seq_q_lens_addr,
        thd_q_lens_ptr,
        thd_kv_lens_ptr,
        thd_lens_form,
        n_thd_units,
        kernel,
        qh,
        kh,
        d_qk,
        d_v,
        persistent_ctas,
        thd_max_sq,
    )

    def scalar(ptr):
        return cute.make_tensor(ptr, cute.make_layout((1,), stride=(1,)))

    # The attention epilogue uses atomicMax on the nonnegative fp32 bit pattern.
    amax_i32 = cute.make_ptr(cutlass.Int32, amax_o_ptr.toint(), cute.AddressSpace.gmem, assumed_align=4)
    if cutlass.const_expr(kernel.split_kv == 1 and has_amax):
        _reset_amax_kernel(amax_o_ptr).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)
    kwargs = dict(prepared=True)
    if cutlass.const_expr(sfo_geometry is not None):
        kwargs.update(
            sf_o=scalar(sf_o_ptr),
            sfo_plane_stride=cutlass.Int64(sfo_geometry[0]),
            sfo_row_off_b=cutlass.Int64(sfo_geometry[1]),
            sfo_col_off_h=cutlass.Int64(sfo_geometry[2]),
            sfo_cols=cutlass.Int64(sfo_geometry[3]),
        )
    kernel(
        q,
        k,
        v,
        o,
        lse,
        sinks,
        q_lens,
        kv_lens,
        scalar(amax_i32) if cutlass.const_expr(has_amax) else None,
        scale_softmax_log2,
        cutlass.Float32(1.0),
        scalar(descale_q_ptr),
        scalar(descale_k_ptr),
        scalar(descale_v_ptr),
        None if cutlass.const_expr(scale_o_ptr is None) else scalar(scale_o_ptr),
        cutlass.Int32(thd_max_sq),
        thd_q_lens,
        thd_kv_lens,
        thd_lens_form,
        n_ctas,
        stream,
        **kwargs,
    )
    if cutlass.const_expr(has_amax and kernel.split_kv == 1):
        _unscale_amax_kernel(amax_o_ptr, scale_o_ptr).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)


def compile_host(
    kernel,
    dtype,
    qh,
    kh,
    d_qk,
    d_v,
    has_lse,
    persistent_ctas,
    cache_key,
    thd_max_sq=0,
    *,
    output_dtype=None,
    has_amax=False,
    scale_o_in_combine=False,
    sfo_geometry=None,
):
    """Compile one pointer entry with Int64 stride leaves and a fixed head geometry."""
    if kernel.thd_varlen and kernel.split_kv != 1:
        raise NotImplementedError("SM120 THD does not support split-KV")
    if kernel.split_kv > 1 and not has_lse:
        raise ValueError("SM120 split-KV requires partial LSE")

    def ptr(t, align=16):
        return cute.runtime.make_ptr(t, 16, cute.AddressSpace.gmem, assumed_align=align)

    fp8 = output_dtype is not None
    strides = (cutlass.Int64(0),) * 3
    return compile_cached(
        fp8_host if fp8 else host,
        ptr(dtype),
        ptr(dtype),
        ptr(dtype),
        ptr(output_dtype if fp8 else dtype),
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
        *(
            (ptr(cutlass.Float32, 4),) * 3
            + (
                None if scale_o_in_combine else ptr(cutlass.Float32, 4),
                ptr(cutlass.Float32, 4),
                ptr(cutlass.Int8) if sfo_geometry is not None else None,
                has_amax,
                sfo_geometry,
            )
            if fp8
            else ()
        ),
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
