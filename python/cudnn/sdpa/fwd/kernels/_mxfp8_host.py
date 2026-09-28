# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Pointer host for MXFP8 scalar-output forward launches."""

from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as _cuda_driver

from cudnn.frost.compiled_cache import compile_cached as _compile_cached
from cudnn.sdpa.fwd.kernels._common_blackwell import sdpa_operand_tensors
from cudnn.sdpa.fwd.kernels._quantized import _reset_amax_kernel, _unscale_amax_kernel

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
    block_table_ptr: Optional[cute.Pointer],
    block_table_v_ptr: Optional[cute.Pointer],
    table_strides: Tuple[cutlass.Int64, cutlass.Int64],
    table_v_strides: Optional[Tuple[cutlass.Int64, cutlass.Int64]],
    n_pages: cutlass.Int32,
    sf_q_ptr: cute.Pointer,
    sf_k_ptr: cute.Pointer,
    sf_v_ptr: Optional[cute.Pointer],
    sf_tiles: Tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32],
    amax_o_ptr: cute.Pointer,
    sf_o_ptr: Optional[cute.Pointer],
    scale_o_ptr: Optional[cute.Pointer],
    gate_ptr: Optional[cute.Pointer],
    gate_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    kernel_host: cutlass.Constexpr,
    config: cutlass.Constexpr,
    sf_smem_sizes: cutlass.Constexpr[tuple],
    d_qk: cutlass.Constexpr[int],
    d_v: cutlass.Constexpr[int],
    lse_kind: cutlass.Constexpr[str],
    partial_slot: cutlass.Constexpr[bool],
    thd_slots: cutlass.Constexpr[bool],
    has_amax: cutlass.Constexpr[bool],
    optional_amax: cutlass.Constexpr[bool],
    sfo_geometry: cutlass.Constexpr,
    stream: _cuda_driver.CUstream = None,
) -> None:
    thd, split_kv, o_block_scale, epilogue_gate, pv_bf16, paged, page_size = config
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
        thd=thd,
        split_kv=split_kv,
        tensor_map_qwords=16,
        paged=paged,
        page_size=page_size,
        block_table_ptr=block_table_ptr,
        block_table_v_ptr=block_table_v_ptr,
        table_strides=table_strides,
        n_pages=n_pages,
        o_pack=2 if o_block_scale == 16 else 1,
        table_v_strides=table_v_strides,
    )
    b, qh, kh, _, _, _ = problem_size

    def scale_tensor(ptr, batch, heads, tiles, size):
        # F8_128x4 bytes are opaque, dense storage. Widen before every stride
        # product; an Int64 cast after an Int32 multiplication is too late.
        head_stride = cutlass.Int64(tiles) * size
        batch_stride = cutlass.Int64(heads) * head_stride
        return cute.make_tensor(
            ptr,
            cute.make_layout(
                (1 if cutlass.const_expr(thd) else batch, heads, tiles, size),
                stride=(batch_stride, head_stride, size, 1),
            ),
        )

    # K/V SF pools page with K/V: one batch entry per page.
    kv_batch = n_pages if cutlass.const_expr(paged) else b
    sf_q = scale_tensor(sf_q_ptr, b, qh, sf_tiles[0], sf_smem_sizes[0])
    sf_k = scale_tensor(sf_k_ptr, kv_batch, kh, sf_tiles[1], sf_smem_sizes[1])
    sf_v = None if cutlass.const_expr(pv_bf16) else scale_tensor(sf_v_ptr, kv_batch, kh, sf_tiles[2], sf_smem_sizes[2])
    amax = None if cutlass.const_expr(optional_amax and not has_amax) else cute.make_tensor(amax_o_ptr, cute.make_layout((1,), stride=(1,)))
    if cutlass.const_expr(split_kv == 1 and (has_amax or not optional_amax)):
        _reset_amax_kernel(amax_o_ptr).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)
    kwargs = dict()
    if cutlass.const_expr(epilogue_gate):
        _, _, _, sq, _, _ = problem_size
        kwargs.update(gate_tensor=cute.make_tensor(gate_ptr, cute.make_layout((b, sq, qh, d_v), stride=(*gate_strides, 1))))
    if cutlass.const_expr(sfo_geometry is not None):
        kwargs.update(
            sf_o_tensor=cute.make_tensor(sf_o_ptr, cute.make_layout((1,), stride=(1,))),
            scale_o_t=None if cutlass.const_expr(scale_o_ptr is None) else cute.make_tensor(scale_o_ptr, cute.make_layout((1,), stride=(1,))),
            sfo_plane_stride=cutlass.Int64(sfo_geometry[0]),
            sfo_row_off_b=cutlass.Int64(sfo_geometry[1]),
            sfo_col_off_h=cutlass.Int64(sfo_geometry[2]),
            sfo_cols=cutlass.Int64(sfo_geometry[3]),
            prepared_sfo_geometry=sfo_geometry,
        )
    if cutlass.const_expr(thd_slots):
        kwargs.update(prepared=True)
    if cutlass.const_expr(partial_slot):
        kwargs.update(o_partial_f32=operands.o_partial)
    if cutlass.const_expr(paged):
        kwargs.update(block_table_tensor=operands.block_table, block_table_v_tensor=operands.block_table_v)
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
    if cutlass.const_expr(has_amax and sfo_geometry is not None and scale_o_ptr is not None):
        # Block-output epilogues reduce after the optional global scale, as
        # the tensor adapter did; Amax_O describes the unscaled attention.
        _unscale_amax_kernel(amax_o_ptr, scale_o_ptr).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)


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
    block_table_ptr: Optional[cute.Pointer],
    block_table_v_ptr: Optional[cute.Pointer],
    table_strides: Tuple[cutlass.Int64, cutlass.Int64],
    table_v_strides: Optional[Tuple[cutlass.Int64, cutlass.Int64]],
    n_pages: cutlass.Int32,
    sf_q_ptr: cute.Pointer,
    sf_k_ptr: cute.Pointer,
    sf_v_ptr: Optional[cute.Pointer],
    sf_tiles: Tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32],
    amax_o_ptr: cute.Pointer,
    sf_o_ptr: Optional[cute.Pointer],
    scale_o_ptr: Optional[cute.Pointer],
    gate_ptr: Optional[cute.Pointer],
    gate_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    kernel_host: cutlass.Constexpr,
    config: cutlass.Constexpr,
    sf_smem_sizes: cutlass.Constexpr[tuple],
    d_qk: cutlass.Constexpr[int],
    d_v: cutlass.Constexpr[int],
    lse_kind: cutlass.Constexpr[str],
    partial_slot: cutlass.Constexpr[bool],
    thd_slots: cutlass.Constexpr[bool],
    has_amax: cutlass.Constexpr[bool],
    optional_amax: cutlass.Constexpr[bool],
    sfo_geometry: cutlass.Constexpr,
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
        block_table_ptr,
        block_table_v_ptr,
        table_strides,
        table_v_strides,
        n_pages,
        sf_q_ptr,
        sf_k_ptr,
        sf_v_ptr,
        sf_tiles,
        amax_o_ptr,
        sf_o_ptr,
        scale_o_ptr,
        gate_ptr,
        gate_strides,
        kernel_host,
        config,
        sf_smem_sizes,
        d_qk,
        d_v,
        lse_kind,
        partial_slot,
        thd_slots,
        has_amax,
        optional_amax,
        sfo_geometry,
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
    sfo_geometry=None,
    has_scale_o=False,
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
    paged = bool(getattr(cfg, "PAGED_KV", False))
    fp32_partial = cfg.SPLIT_KV > 1 and partial_slot
    return _compile_cached(
        host,
        pointer(storage_dtype),
        pointer(storage_dtype),
        pointer(cutlass.BFloat16 if getattr(cfg, "PV_BF16", False) else storage_dtype),
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
        pointer(cutlass.Int32, 4) if paged else None,
        pointer(cutlass.Int32, 4) if paged else None,
        (cutlass.Int64(0), cutlass.Int64(0)),
        (cutlass.Int64(0), cutlass.Int64(0)) if paged else None,
        i32,
        pointer(cutlass.Int8),
        pointer(cutlass.Int8),
        None if getattr(cfg, "PV_BF16", False) else pointer(cutlass.Int8),
        (i32, i32, i32),
        pointer(cutlass.Float32, 4),
        pointer(cutlass.Int8) if sfo_geometry is not None else None,
        pointer(cutlass.Float32, 4) if has_scale_o else None,
        pointer(cutlass.BFloat16) if getattr(cfg, "EPILOGUE_GATE", False) else None,
        strides,
        kernel_host,
        (
            bool(cfg.THD_VARLEN),
            int(cfg.SPLIT_KV),
            int(getattr(cfg, "O_BLOCK_SCALE", 0)),
            bool(getattr(cfg, "EPILOGUE_GATE", False)),
            bool(getattr(cfg, "PV_BF16", False)),
            paged,
            int(getattr(cfg, "PAGE_SIZE", 0)),
        ),
        sf_smem_sizes,
        d_qk,
        d_v,
        lse_kind,
        partial_slot,
        thd_slots,
        has_amax,
        optional_amax,
        sfo_geometry,
        static_lse_strides,
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options="--enable-tvm-ffi",
        cache_key=cache_key,
        symbol="frost_sdpa_fwd_prepared_mxfp8",
    )
