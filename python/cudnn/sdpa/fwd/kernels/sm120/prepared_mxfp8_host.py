# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Pointer host for the SM120 block-scaled MXFP8 templates (``sm120/prefill_mxfp8.py``,
``sm120/prefill_d512_mxfp8.py``).

The scale slots (``sf_q_ptr`` / ``sf_k_ptr`` / ``sf_v_ptr``, ``sf_tiles``,
``amax_o_ptr``, and for a block-scaled O ``sf_o_ptr`` / ``scale_o_ptr``) use the
MXFP8 forward host vocabulary the native binder fills. Scale tensors are
``(B, H, tiles, tile_bytes)`` over opaque F8_128x4 bytes; THD binds batch-1
views over the packed tile count.
"""

from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as cuda_driver

from cudnn.frost.compiled_cache import compile_cached
from cudnn.sdpa.fwd.kernels._quantized import _reset_amax_kernel, _unscale_amax_kernel
from cudnn.sdpa.fwd.kernels.sm120.prepared_host import _operands


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
    lse_ext: cutlass.Int64,
    scale_softmax_log2: cutlass.Float32,
    seq_q_lens_addr: cutlass.Int64,
    thd_q_lens_ptr: Optional[cute.Pointer],
    thd_kv_lens_ptr: Optional[cute.Pointer],
    thd_lens_form: Optional[cutlass.Int32],
    n_thd_units: cutlass.Int32,
    sf_q_ptr: cute.Pointer,
    sf_k_ptr: cute.Pointer,
    sf_v_ptr: cute.Pointer,
    sf_tiles: Tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32],
    amax_o_ptr: cute.Pointer,
    sf_o_ptr: Optional[cute.Pointer],
    scale_o_ptr: Optional[cute.Pointer],
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
    """Bind the pointers into tensors and launch the MXFP8 kernel.

    ``sf_tiles`` holds the Q, K and V scale-tile counts per (batch, head), or
    the packed totals under THD. ``Amax_O`` is zeroed on the launch stream
    unless the split prologue initializes it. A block-scaled O (``sfo_geometry``)
    writes its scale factors to ``sf_o_ptr``; its optional global ``scale_o``
    multiplies O before the block quantization, and ``Amax_O`` is divided back
    to the unscaled attention output afterwards.
    """
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
    b = problem_size[0]

    def scale_tensor(ptr, heads, tiles, size):
        head_stride = cutlass.Int64(tiles) * size
        return cute.make_tensor(
            ptr,
            cute.make_layout(
                (1 if cutlass.const_expr(kernel.thd_varlen) else b, heads, tiles, size),
                stride=(cutlass.Int64(heads) * head_stride, head_stride, size, 1),
            ),
        )

    sf_q = scale_tensor(sf_q_ptr, qh, sf_tiles[0], kernel.qk_sf_tile_bytes)
    sf_k = scale_tensor(sf_k_ptr, kh, sf_tiles[1], kernel.qk_sf_tile_bytes)
    sf_v = scale_tensor(sf_v_ptr, kh, sf_tiles[2], kernel.v_sf_tile_bytes)
    amax_i32 = cute.make_ptr(cutlass.Int32, amax_o_ptr.toint(), cute.AddressSpace.gmem, assumed_align=4)
    if cutlass.const_expr(kernel.split_kv == 1 and has_amax):
        _reset_amax_kernel(amax_o_ptr).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)

    def scalar(ptr):
        return cute.make_tensor(ptr, cute.make_layout((1,), stride=(1,)))

    kwargs = dict(prepared=True)
    if cutlass.const_expr(sfo_geometry is not None):
        kwargs.update(
            scale_o_t=None if cutlass.const_expr(scale_o_ptr is None) else scalar(scale_o_ptr),
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
        cute.make_tensor(amax_i32, cute.make_layout((1,), stride=(1,))) if cutlass.const_expr(has_amax) else None,
        sf_q,
        sf_k,
        sf_v,
        scale_softmax_log2,
        cutlass.Int32(thd_max_sq),
        thd_q_lens,
        thd_kv_lens,
        thd_lens_form,
        n_ctas,
        stream,
        **kwargs,
    )
    if cutlass.const_expr(has_amax and sfo_geometry is not None and scale_o_ptr is not None):
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
    output_dtype,
    has_amax=False,
    sfo_geometry=None,
    has_scale_o=False,
):
    """Compile one pointer entry with Int64 stride leaves and a fixed head geometry."""
    if kernel.thd_varlen and kernel.split_kv != 1:
        raise NotImplementedError("SM120 THD does not support split-KV")
    if kernel.split_kv > 1 and not has_lse:
        raise ValueError("SM120 split-KV requires partial LSE")

    def ptr(t, align=16):
        return cute.runtime.make_ptr(t, 16, cute.AddressSpace.gmem, assumed_align=align)

    strides = (cutlass.Int64(0),) * 3
    i32 = cutlass.Int32(0)
    return compile_cached(
        host,
        ptr(dtype),
        ptr(dtype),
        ptr(dtype),
        ptr(output_dtype),
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
        cutlass.Int64(0),
        cutlass.Float32(0),
        cutlass.Int64(0),
        ptr(cutlass.Int32, 4) if kernel.thd_varlen else None,
        ptr(cutlass.Int32, 4) if kernel.thd_varlen else None,
        i32 if kernel.thd_varlen else None,
        i32,
        ptr(cutlass.Int8),
        ptr(cutlass.Int8),
        ptr(cutlass.Int8),
        (i32, i32, i32),
        ptr(cutlass.Float32, 4),
        ptr(cutlass.Int8) if sfo_geometry is not None else None,
        ptr(cutlass.Float32, 4) if has_scale_o else None,
        has_amax,
        sfo_geometry,
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
        symbol="frost_sdpa_fwd_prepared_mxfp8",
    )
