# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Shared SM100/SM107 per-tensor FP8 prepared pointer host."""

from types import SimpleNamespace
from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as _cuda_driver

from cudnn.frost.compiled_cache import compile_cached as _compile_cached
from cudnn.sdpa.fwd.kernels._common_blackwell import sdpa_operand_tensors
from cudnn.sdpa.fwd.kernels._quantized import _reset_amax_kernel, _unscale_amax_kernel

LSE_KINDS = ("dense", "token", "head", "padded")


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
    n_pages: cutlass.Int32,
    descale_q_ptr: cute.Pointer,
    descale_k_ptr: cute.Pointer,
    descale_v_ptr: cute.Pointer,
    scale_o_ptr: Optional[cute.Pointer],
    amax_o_ptr: cute.Pointer,
    sf_o_ptr: Optional[cute.Pointer],
    gate_ptr: Optional[cute.Pointer],
    gate_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    has_amax: cutlass.Constexpr[bool],
    kernel_host: cutlass.Constexpr,
    config: cutlass.Constexpr,
    d256: cutlass.Constexpr[bool],
    d_qk: cutlass.Constexpr[int],
    d_v: cutlass.Constexpr[int],
    lse_kind: cutlass.Constexpr[str],
    paged_hnd: cutlass.Constexpr[bool],
    partial_slot: cutlass.Constexpr[bool],
    optional_amax: cutlass.Constexpr[bool],
    sfo_geometry: cutlass.Constexpr,
    table_v_strides: Optional[Tuple[cutlass.Int64, cutlass.Int64]],
    stream: _cuda_driver.CUstream = None,
) -> None:
    """Bind dense or THD pointer views and launch the selected FP8 host.

    Extents and Int64 element strides are runtime slots; the shared binder
    validates their layout and capacity. Scalar scales remain device pointers.
    The legacy kernel reduces amax after scale_o, so a requested amax is
    normalized on the same stream without constructing a framework tensor.
    """
    thd, split_kv, paged, page_size, o_block_scale, epilogue_gate = config
    (
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        lse_tensor,
        sinks_tensor,
        seq_kv_lens_tensor,
        o_desc_words,
        thd_q_lens_tensor,
        thd_kv_lens_tensor,
        o_partial_f32,
        block_table_tensor,
        block_table_v_tensor,
    ) = sdpa_operand_tensors(
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

    def scalar(ptr):
        return cute.make_tensor(ptr, cute.make_layout((1,), stride=(1,)))

    args = (q_tensor, k_tensor, v_tensor, o_tensor, lse_tensor, sinks_tensor, seq_kv_lens_tensor, o_desc_words, problem_size)
    if cutlass.const_expr(d256):
        # The diagonal-only fast case specializes equal lengths. Prepared
        # execution permits rectangular overrides, so retain the general mask.
        args += (False,)
    # Keep the scalar reset on the SM execution path. A captured driver
    # memset creates an extra engine dependency before the attention kernel.
    if cutlass.const_expr(split_kv == 1 and (has_amax or not optional_amax)):
        _reset_amax_kernel(amax_o_ptr).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)
    kernel_kwargs = dict(prepared=True)
    if cutlass.const_expr(epilogue_gate):
        b, qh, _, sq, _, _ = problem_size
        kernel_kwargs.update(gate_tensor=cute.make_tensor(gate_ptr, cute.make_layout((b, sq, qh, d_v), stride=(*gate_strides, 1))))
    if cutlass.const_expr(sfo_geometry is not None):
        kernel_kwargs.update(
            sf_o_tensor=scalar(sf_o_ptr),
            sfo_plane_stride=cutlass.Int64(sfo_geometry[0]),
            sfo_row_off_b=cutlass.Int64(sfo_geometry[1]),
            sfo_col_off_h=cutlass.Int64(sfo_geometry[2]),
            sfo_cols=cutlass.Int64(sfo_geometry[3]),
            prepared_sfo_geometry=sfo_geometry,
        )
    if cutlass.const_expr(partial_slot):
        kernel_kwargs.update(o_partial_f32=o_partial_f32)
    if cutlass.const_expr(paged):
        kernel_kwargs.update(block_table_tensor=block_table_tensor, block_table_v_tensor=block_table_v_tensor, paged_hnd_prepared=paged_hnd)
    kernel_host(
        *args,
        scale_softmax_log2,
        cutlass.Float32(1.0),
        n_thd_units,
        scalar(descale_q_ptr),
        scalar(descale_k_ptr),
        scalar(descale_v_ptr),
        None if cutlass.const_expr(scale_o_ptr is None) else scalar(scale_o_ptr),
        None if cutlass.const_expr(optional_amax and not has_amax) else scalar(amax_o_ptr),
        seq_q_lens_addr,
        thd_q_lens_tensor,
        thd_kv_lens_tensor,
        thd_lens_form,
        stream=stream,
        **kernel_kwargs,
    )
    if cutlass.const_expr(has_amax and split_kv == 1):
        _unscale_amax_kernel(amax_o_ptr, scale_o_ptr).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)


def compile_host(
    kernel_host,
    cfg,
    storage_dtype,
    output_dtype,
    d256,
    cache_key,
    d_qk,
    d_v,
    has_lse,
    lse_kind,
    has_amax,
    scale_o_in_combine=False,
    paged_hnd=False,
    *,
    partial_slot=True,
    optional_amax=False,
    sfo_geometry=None,
):
    if cfg.SPLIT_KV > 1 and not has_lse:
        raise ValueError("prepared FP8 split-KV requires partial LSE")
    gmem = cute.AddressSpace.gmem

    def P(dtype, align=16):
        return cute.runtime.make_ptr(dtype, 16, gmem, assumed_align=align)  # fake: type only

    i32 = cutlass.Int32(0)
    i64_3 = (cutlass.Int64(0),) * 3  # stride slots: Int64 leaves, see _host
    thd = bool(cfg.THD_VARLEN)
    # Only the primitive host facts cross the Constexpr boundary. A dataclass
    # argument prevents compiled_cache from exporting the positional artifact.
    return _compile_cached(
        host,
        P(storage_dtype),
        P(storage_dtype),
        P(storage_dtype),
        P(cutlass.Float32 if cfg.SPLIT_KV > 1 else output_dtype),
        P(cutlass.Float32, 4) if has_lse else None,
        P(cutlass.Float32),
        P(cutlass.Int32),
        P(cutlass.Int64),
        (0, 0, 0, 0, 0, 0),
        i64_3,
        i64_3,
        i64_3,
        i64_3,
        i64_3,
        i32,
        cutlass.Float32(0.0),
        i32,
        cutlass.Int64(0),
        P(cutlass.Int32, 4) if thd else None,
        P(cutlass.Int32, 4) if thd else None,
        i32 if thd else None,
        P(cutlass.Float32) if cfg.SPLIT_KV > 1 else None,
        P(cutlass.Int32, 4) if getattr(cfg, "PAGED_KV", False) else None,
        P(cutlass.Int32, 4) if getattr(cfg, "PAGED_KV", False) else None,
        (cutlass.Int64(0), cutlass.Int64(0)),
        i32,
        P(cutlass.Float32, 4),
        P(cutlass.Float32, 4),
        P(cutlass.Float32, 4),
        None if scale_o_in_combine else P(cutlass.Float32, 4),
        P(cutlass.Float32, 4),
        P(cutlass.Int8) if sfo_geometry is not None else None,
        P(cutlass.BFloat16) if getattr(cfg, "EPILOGUE_GATE", False) else None,
        i64_3,
        has_amax,
        kernel_host,
        (
            bool(cfg.THD_VARLEN),
            int(cfg.SPLIT_KV),
            bool(getattr(cfg, "PAGED_KV", False)),
            int(getattr(cfg, "PAGE_SIZE", 0)),
            int(getattr(cfg, "O_BLOCK_SCALE", 0)),
            bool(getattr(cfg, "EPILOGUE_GATE", False)),
        ),
        d256,
        d_qk,
        d_v,
        lse_kind,
        paged_hnd,
        partial_slot,
        optional_amax,
        sfo_geometry,
        (cutlass.Int64(0), cutlass.Int64(0)) if getattr(cfg, "PAGED_KV", False) else None,
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options="--enable-tvm-ffi",
        cache_key=cache_key,
        symbol="frost_sdpa_fwd_prepared",
    )
