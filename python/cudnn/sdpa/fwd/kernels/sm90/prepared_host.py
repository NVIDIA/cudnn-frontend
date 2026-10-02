# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Pointer-only host for the SM90 D512 tile; operands come from the shared binder."""

from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as cuda_driver

from cudnn.frost.compiled_cache import compile_cached
from cudnn.frost.tile_dsl.thd import THD_MAPS_META_WORDS


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
    scale_softmax: cutlass.Float32,
    seq_q_lens_addr: cutlass.Int64,
    thd_q_lens_ptr: Optional[cute.Pointer],
    thd_kv_lens_ptr: Optional[cute.Pointer],
    thd_lens_form: Optional[cutlass.Int32],
    n_thd_units: cutlass.Int32,
    thd_max_sq: cutlass.Int32,
    kernel: cutlass.Constexpr,
    d_qk: cutlass.Constexpr[int],
    d_v: cutlass.Constexpr[int],
    lse_head_major: cutlass.Constexpr[bool],
    stream: cuda_driver.CUstream,
):
    """Construct DSL views from validated pointer facts and launch the fixed SM90 tile."""
    # Dense extents and the THD batch are fixed by this tile's scheduler. Packed
    # capacities, strides, addresses, lengths and the unit envelope remain per-call.
    _, _, _, sq, skv, _ = problem_size
    batch = 1 if kernel.thd_varlen else kernel.b

    def bhsd(ptr, heads, tokens, dim, strides):
        """Map binder BSH strides to the kernel's BHSD view without changing storage."""
        bs, ss, hs = strides
        return cute.make_tensor(ptr, cute.make_layout((batch, heads, tokens, dim), stride=(bs, hs, ss, 1)))

    q = bhsd(q_ptr, kernel.h_q, sq, d_qk, q_strides)
    k = bhsd(k_ptr, kernel.h_kv, skv, d_qk, k_strides)
    v = bhsd(v_ptr, kernel.h_kv, skv, d_v, v_strides)
    o = bhsd(o_ptr, kernel.h_q, sq, d_v, o_strides)
    lse = None
    if cutlass.const_expr(lse_ptr is not None):
        strides = lse_strides
        if cutlass.const_expr(kernel.thd_varlen):
            strides = (0, cutlass.Int64(kernel.thd_lse_head_stride), 1) if lse_head_major else (0, 1, kernel.h_q)
        lse = cute.make_tensor(lse_ptr, cute.make_layout((batch, kernel.h_q, sq), stride=strides))
    sinks = None
    if cutlass.const_expr(kernel.has_sink):
        sinks = cute.make_tensor(sinks_ptr, cute.make_layout((kernel.h_q,), stride=(1,)))
    q_lens_ptr = cute.make_ptr(cutlass.Int32, seq_q_lens_addr, cute.AddressSpace.gmem, assumed_align=4)
    q_lens = cute.make_tensor(q_lens_ptr, cute.make_layout((kernel.b,), stride=(1,)))
    kv_lens = cute.make_tensor(meta_ptr, cute.make_layout((THD_MAPS_META_WORDS(kernel.b) if kernel.thd_varlen else kernel.b,), stride=(1,)))
    thd_q_lens = thd_kv_lens = None
    if cutlass.const_expr(kernel.thd_varlen):
        thd_q_lens = cute.make_tensor(thd_q_lens_ptr, cute.make_layout((kernel.b + (thd_lens_form & 1),), stride=(1,)))
        thd_kv_lens = cute.make_tensor(thd_kv_lens_ptr, cute.make_layout((kernel.b + ((thd_lens_form >> 1) & 1),), stride=(1,)))
    kernel(q, k, v, o, lse, sinks, q_lens, kv_lens, scale_softmax, thd_max_sq, thd_q_lens, thd_kv_lens, thd_lens_form, stream)


def compile_host(kernel, dtype, d_qk, d_v, has_lse, lse_head_major, cache_key, target):
    """Compile/cache the pointer ABI with Int64 strides and an explicit CUDA stream."""

    def ptr(dtype, align=16):
        """Describe a pointer argument for tracing; no device allocation is made."""
        return cute.runtime.make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=align)

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
        cutlass.Int32(0),
        kernel,
        d_qk,
        d_v,
        lse_head_major,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options=f"--enable-tvm-ffi --gpu-arch={target}",
        cache_key=cache_key,
        symbol="frost_sdpa_fwd_prepared",
    )
