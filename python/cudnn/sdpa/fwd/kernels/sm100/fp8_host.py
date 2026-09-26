# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Shared SM100 per-tensor FP8 host binding and compile-time auxiliary operands."""

from types import SimpleNamespace
from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as _cuda_driver

from cudnn.frost.compiled_cache import compile_cached as _compile_cached
from cudnn.sdpa.fwd.kernels._common_blackwell import sdpa_operand_tensors

LSE_KINDS = ("dense", "token", "head", "padded")


@cute.kernel
def _unscale_amax_kernel(amax: cute.Pointer, scale: cute.Pointer):
    amax.store(amax.load() / scale.load())


_unscale_amax_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


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
    scale_o_ptr: cute.Pointer,
    amax_o_ptr: cute.Pointer,
    has_amax: cutlass.Constexpr[bool],
    kernel_host: cutlass.Constexpr,
    cfg: cutlass.Constexpr,
    d256: cutlass.Constexpr[bool],
    d_qk: cutlass.Constexpr[int],
    d_v: cutlass.Constexpr[int],
    lse_kind: cutlass.Constexpr[str],
    paged_hnd: cutlass.Constexpr[bool],
    stream: _cuda_driver.CUstream = None,
) -> None:
    """Bind dense or THD pointer views and launch the selected FP8 host.

    Extents and Int64 element strides are runtime slots; the shared binder
    validates their layout and capacity. Scalar scales remain device pointers.
    The legacy kernel reduces amax after scale_o, so a requested amax is
    normalized on the same stream without constructing a framework tensor.
    """
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
        thd=cfg.THD_VARLEN,
        split_kv=1,
        tensor_map_qwords=16,
        paged=False,
        page_size=0,
        block_table_ptr=block_table_ptr,
        block_table_v_ptr=block_table_v_ptr,
        table_strides=table_strides,
        n_pages=n_pages,
    )

    def scalar(ptr):
        return cute.make_tensor(ptr, cute.make_layout((1,), stride=(1,)))

    args = (q_tensor, k_tensor, v_tensor, o_tensor, lse_tensor, sinks_tensor, seq_kv_lens_tensor, o_desc_words, problem_size)
    if cutlass.const_expr(d256):
        # The diagonal-only fast case specializes equal lengths. Prepared
        # execution permits rectangular overrides, so retain the general mask.
        args += (False,)
    kernel_host(
        *args,
        scale_softmax_log2,
        cutlass.Float32(1.0),
        n_thd_units,
        scalar(descale_q_ptr),
        scalar(descale_k_ptr),
        scalar(descale_v_ptr),
        scalar(scale_o_ptr),
        scalar(amax_o_ptr),
        seq_q_lens_addr,
        thd_q_lens_tensor,
        thd_kv_lens_tensor,
        thd_lens_form,
        stream=stream,
        prepared=True,
    )
    if cutlass.const_expr(has_amax):
        _unscale_amax_kernel(amax_o_ptr, scale_o_ptr).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)


def compile_host(kernel_host, cfg, storage_dtype, output_dtype, d256, cache_key, d_qk, d_v, has_lse, lse_kind, has_amax):
    gmem = cute.AddressSpace.gmem

    def P(dtype, align=16):
        return cute.runtime.make_ptr(dtype, 16, gmem, assumed_align=align)  # fake: type only

    i32 = cutlass.Int32(0)
    i64_3 = (cutlass.Int64(0),) * 3  # stride slots: Int64 leaves, see _host
    thd = bool(cfg.THD_VARLEN)
    return _compile_cached(
        host,
        P(storage_dtype),
        P(storage_dtype),
        P(storage_dtype),
        P(output_dtype),
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
        None,
        None,
        None,
        (cutlass.Int64(0), cutlass.Int64(0)),
        i32,
        P(cutlass.Float32, 4),
        P(cutlass.Float32, 4),
        P(cutlass.Float32, 4),
        P(cutlass.Float32, 4),
        P(cutlass.Float32, 4),
        has_amax,
        kernel_host,
        cfg,
        d256,
        d_qk,
        d_v,
        lse_kind,
        False,
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options="--enable-tvm-ffi",
        cache_key=cache_key,
        symbol="frost_sdpa_fwd_prepared",
    )


def make_fake_aux(b, qh, *, amax_align=16):
    """Dense tensor-entry auxiliaries for the remaining split/paged/conversion paths.

    Prepared launches never construct these tensor fakes. All SM100 per-tensor
    FP8 THD routes now use the pointer entry, so no packed metadata, descriptor
    arrays, dynamic head extents or length-array fakes remain here.
    """

    def vector(dtype, length, align):
        return cute.runtime.make_fake_compact_tensor(dtype, (length,), stride_order=(0,), assumed_align=align)

    return SimpleNamespace(
        sinks=vector(cutlass.Float32, qh, 16),
        kv_lens=vector(cutlass.Int32, b, 16),
        o_desc=vector(cutlass.Int64, 1, 16),
        seq_q_lens=cutlass.Int64(0),  # device address, never an Int32 length
        scales=tuple(vector(cutlass.Float32, 1, 4) for _ in range(4)),
        amax_o=vector(cutlass.Float32, 1, amax_align),
    )
