# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Pointer host for MXFP8 scalar-output forward launches."""

from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as _cuda_driver

from cudnn.frost.compiled_cache import compile_cached as _compile_cached
from cudnn.sdpa.fwd.kernels._common_blackwell import sdpa_operand_tensors
from cudnn.sdpa.fwd.kernels._quantized import _reset_amax_kernel

LSE_KINDS = ("dense", "token", "head", "padded")


@cute.jit
def _launch(
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
    n_thd_units: cutlass.Int32,
    seq_q_lens_addr: cutlass.Int64,
    thd_q_lens_ptr: Optional[cute.Pointer],
    thd_kv_lens_ptr: Optional[cute.Pointer],
    thd_lens_form: Optional[cutlass.Int32],
    o_partial_ptr: Optional[cute.Pointer],
    sf_q_ptr: cute.Pointer,
    sf_k_ptr: cute.Pointer,
    sf_v_ptr: cute.Pointer,
    sf_tiles: Tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32],
    amax_o_ptr: cute.Pointer,
    kernel_host: cutlass.Constexpr,
    cfg: cutlass.Constexpr,
    sf_smem_sizes: cutlass.Constexpr[tuple],
    d_qk: cutlass.Constexpr[int],
    d_v: cutlass.Constexpr[int],
    lse_kind: cutlass.Constexpr[str],
    partial_slot: cutlass.Constexpr[bool],
    thd_slots: cutlass.Constexpr[bool],
    has_amax: cutlass.Constexpr[bool],
    optional_amax: cutlass.Constexpr[bool],
    stream: _cuda_driver.CUstream = None,
) -> None:
    operands = sdpa_operand_tensors(
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        lse_ptr,
        sinks_ptr,
        meta_ptr,
        o_desc_ptr,
        problem_size,
        q_strides,
        k_strides,
        v_strides,
        o_strides,
        lse_strides,
        lse_ext,
        thd_q_lens_ptr,
        thd_kv_lens_ptr,
        thd_lens_form,
        o_partial_ptr,
        d_qk=d_qk,
        d_v=d_v,
        lse_kind=lse_kind,
        thd=cfg.THD_VARLEN,
        split_kv=cfg.SPLIT_KV,
        tensor_map_qwords=16,
    )
    b, qh, kh, _, _, _ = problem_size

    def scale_tensor(ptr, heads, tiles, size):
        # F8_128x4 bytes are opaque, dense storage. Widen before every stride
        # product; an Int64 cast after an Int32 multiplication is too late.
        head_stride = cutlass.Int64(tiles) * size
        batch_stride = cutlass.Int64(heads) * head_stride
        return cute.make_tensor(
            ptr,
            cute.make_layout(
                (1 if cutlass.const_expr(cfg.THD_VARLEN) else b, heads, tiles, size),
                stride=(batch_stride, head_stride, size, 1),
            ),
        )

    sf_q = scale_tensor(sf_q_ptr, qh, sf_tiles[0], sf_smem_sizes[0])
    sf_k = scale_tensor(sf_k_ptr, kh, sf_tiles[1], sf_smem_sizes[1])
    sf_v = scale_tensor(sf_v_ptr, kh, sf_tiles[2], sf_smem_sizes[2])
    amax = None if cutlass.const_expr(optional_amax and not has_amax) else cute.make_tensor(amax_o_ptr, cute.make_layout((1,), stride=(1,)))
    if cutlass.const_expr(cfg.SPLIT_KV == 1 and (has_amax or not optional_amax)):
        _reset_amax_kernel(amax_o_ptr).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)
    kwargs = dict()
    if cutlass.const_expr(thd_slots):
        kwargs.update(prepared=True)
    if cutlass.const_expr(partial_slot):
        kwargs.update(o_partial_f32=operands.o_partial)
    args = (
        operands.q,
        operands.k,
        operands.v,
        operands.o,
        sf_q,
        sf_k,
        sf_v,
        operands.lse,
        amax,
        operands.sinks,
        operands.meta,
        operands.o_desc,
        problem_size,
        scale_softmax_log2,
        n_thd_units,
        seq_q_lens_addr,
    )
    if cutlass.const_expr(thd_slots):
        args += (operands.thd_q_lens, operands.thd_kv_lens, thd_lens_form)
    kernel_host(*args, stream=stream, **kwargs)


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
    n_thd_units: cutlass.Int32,
    seq_q_lens_addr: cutlass.Int64,
    thd_q_lens_ptr: Optional[cute.Pointer],
    thd_kv_lens_ptr: Optional[cute.Pointer],
    thd_lens_form: Optional[cutlass.Int32],
    o_partial_ptr: Optional[cute.Pointer],
    sf_q_ptr: cute.Pointer,
    sf_k_ptr: cute.Pointer,
    sf_v_ptr: cute.Pointer,
    sf_tiles: Tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32],
    amax_o_ptr: cute.Pointer,
    kernel_host: cutlass.Constexpr,
    cfg: cutlass.Constexpr,
    sf_smem_sizes: cutlass.Constexpr[tuple],
    d_qk: cutlass.Constexpr[int],
    d_v: cutlass.Constexpr[int],
    lse_kind: cutlass.Constexpr[str],
    partial_slot: cutlass.Constexpr[bool],
    thd_slots: cutlass.Constexpr[bool],
    has_amax: cutlass.Constexpr[bool],
    optional_amax: cutlass.Constexpr[bool],
    static_lse_strides: cutlass.Constexpr,
    stream: _cuda_driver.CUstream = None,
) -> None:
    leading = (
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        lse_ptr,
        sinks_ptr,
        meta_ptr,
        o_desc_ptr,
        problem_size,
        q_strides,
        k_strides,
        v_strides,
        o_strides,
    )
    trailing = (
        lse_ext,
        scale_softmax_log2,
        n_thd_units,
        seq_q_lens_addr,
        thd_q_lens_ptr,
        thd_kv_lens_ptr,
        thd_lens_form,
        o_partial_ptr,
        sf_q_ptr,
        sf_k_ptr,
        sf_v_ptr,
        sf_tiles,
        amax_o_ptr,
        kernel_host,
        cfg,
        sf_smem_sizes,
        d_qk,
        d_v,
        lse_kind,
        partial_slot,
        thd_slots,
        has_amax,
        optional_amax,
    )
    if cutlass.const_expr(static_lse_strides is None):
        _launch(*leading, lse_strides, *trailing, stream=stream)
    else:
        # Preserve generic runtime layouts while specializing the declared
        # dense Stats layout. Both arms launch the same logical operation.
        if (lse_strides[0] == static_lse_strides[0]) & (lse_strides[1] == static_lse_strides[1]) & (lse_strides[2] == static_lse_strides[2]):
            _launch(*leading, static_lse_strides, *trailing, stream=stream)
        else:
            _launch(*leading, lse_strides, *trailing, stream=stream)


def compile_host(
    kernel_host,
    cfg,
    storage_dtype,
    output_dtype,
    sf_smem_sizes,
    cache_key,
    d_qk,
    d_v,
    has_lse,
    lse_kind,
    *,
    partial_slot=True,
    thd_slots=True,
    has_amax=True,
    optional_amax=False,
    static_lse_strides=None,
):
    if static_lse_strides is not None:
        if not has_lse or cfg.THD_VARLEN or cfg.SPLIT_KV != 1:
            raise ValueError("Stats stride specialization requires dense unsplit Stats")
        if len(static_lse_strides) != 3 or any(x < 0 for x in static_lse_strides):
            raise ValueError("Stats stride specialization requires three nonnegative strides")
    if cfg.SPLIT_KV > 1 and not has_lse:
        raise ValueError("prepared MXFP8 split-KV requires partial LSE")
    gmem = cute.AddressSpace.gmem

    def pointer(dtype, align=16):
        return cute.runtime.make_ptr(dtype, 16, gmem, assumed_align=align)

    i32 = cutlass.Int32(0)
    strides = (cutlass.Int64(0),) * 3
    thd = bool(cfg.THD_VARLEN)
    fp32_partial = cfg.SPLIT_KV > 1 and partial_slot
    return _compile_cached(
        host,
        pointer(storage_dtype),
        pointer(storage_dtype),
        pointer(storage_dtype),
        pointer(cutlass.Float32 if fp32_partial else output_dtype),
        pointer(cutlass.Float32, 4) if has_lse else None,
        pointer(cutlass.Float32),
        pointer(cutlass.Int32),
        pointer(cutlass.Int64),
        (0, 0, 0, 0, 0, 0),
        strides,
        strides,
        strides,
        strides,
        strides,
        i32,
        cutlass.Float32(0.0),
        i32,
        cutlass.Int64(0),
        pointer(cutlass.Int32, 4) if thd else None,
        pointer(cutlass.Int32, 4) if thd else None,
        i32 if thd else None,
        pointer(cutlass.Float32) if fp32_partial else None,
        pointer(cutlass.Int8),
        pointer(cutlass.Int8),
        pointer(cutlass.Int8),
        (i32, i32, i32),
        pointer(cutlass.Float32, 4),
        kernel_host,
        cfg,
        sf_smem_sizes,
        d_qk,
        d_v,
        lse_kind,
        partial_slot,
        thd_slots,
        has_amax,
        optional_amax,
        static_lse_strides,
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options="--enable-tvm-ffi",
        cache_key=cache_key,
        symbol="frost_sdpa_fwd_prepared_mxfp8",
    )
