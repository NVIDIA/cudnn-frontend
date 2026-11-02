# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Pointer-only host for the SM90 D512 tile; operands come from the shared binder."""

from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as cuda_driver

from cudnn.frost.compiled_cache import compile_cached
from cudnn.sdpa.fwd.config_sm90 import thd_meta_words


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
    scale_softmax: cutlass.Float32,
    seq_q_lens_addr: cutlass.Int64,
    thd_q_lens_ptr: Optional[cute.Pointer],
    thd_kv_lens_ptr: Optional[cute.Pointer],
    thd_lens_form: Optional[cutlass.Int32],
    n_thd_units: cutlass.Int32,
    thd_max_sq: cutlass.Int32,
    thd_origin_ptrs,
    thd_origin_caps,
    kernel: cutlass.Constexpr,
    d_qk: cutlass.Constexpr[int],
    d_v: cutlass.Constexpr[int],
    lse_head_major: cutlass.Constexpr[bool],
    stream: cuda_driver.CUstream,
):
    """Construct DSL views from validated pointer facts and launch the fixed SM90 tile.

    ``thd_origin_ptrs`` is the shared binder's per-port (Q, K, V, O, Stats) physical-offset
    pointers of a THD plan with ``kernel.thd_origins`` (None elsewhere), and ``thd_origin_caps`` each
    port's own token capacity. Sequences sit at their origins, so one port's bound says nothing about
    another's, and the binder's packed tq / tkv are live-token bounds, not the physical spans of gapped
    buffers. The setup launch clamps every origin into its port's capacity, so a bad offset never
    addresses memory outside the bound buffers."""
    # Dense extents and the THD batch are fixed by this tile's scheduler. Packed
    # capacities, strides, addresses, lengths and the unit envelope remain per-call.
    _, _, _, sq, skv, _ = problem_size
    batch = 1 if kernel.thd_varlen else kernel.b
    caps = (sq, skv, skv, sq, sq)
    if cutlass.const_expr(kernel.thd_origins is not None):
        caps = tuple(cutlass.Int32(c) for c in thd_origin_caps)

    def bhsd(ptr, heads, tokens, dim, strides):
        """Map binder BSH strides to the kernel's BHSD view without changing storage."""
        bs, ss, hs = strides
        return cute.make_tensor(ptr, cute.make_layout((batch, heads, tokens, dim), stride=(bs, hs, ss, 1)))

    q = bhsd(q_ptr, kernel.h_q, caps[0], d_qk, q_strides)
    k = bhsd(k_ptr, kernel.h_kv, caps[1], d_qk, k_strides)
    v = bhsd(v_ptr, kernel.h_kv, caps[2], d_v, v_strides)
    o = bhsd(o_ptr, kernel.h_q, caps[3], d_v, o_strides)
    lse = None
    if cutlass.const_expr(lse_ptr is not None):
        strides = lse_strides
        if cutlass.const_expr(kernel.thd_varlen):
            strides = (0, cutlass.Int64(kernel.thd_lse_head_stride), 1) if lse_head_major else (0, 1, kernel.h_q)
        lse = cute.make_tensor(lse_ptr, cute.make_layout((batch, kernel.h_q, caps[4]), stride=strides))
    sinks = None
    if cutlass.const_expr(kernel.has_sink):
        sinks = cute.make_tensor(sinks_ptr, cute.make_layout((kernel.h_q,), stride=(1,)))
    q_lens_ptr = cute.make_ptr(cutlass.Int32, seq_q_lens_addr, cute.AddressSpace.gmem, assumed_align=4)
    q_lens = cute.make_tensor(q_lens_ptr, cute.make_layout((kernel.b,), stride=(1,)))
    kv_lens = cute.make_tensor(
        meta_ptr, cute.make_layout((thd_meta_words(kernel.b, kernel.thd_origins is not None) if kernel.thd_varlen else kernel.b,), stride=(1,))
    )
    thd_q_lens = thd_kv_lens = None
    if cutlass.const_expr(kernel.thd_varlen):
        thd_q_lens = cute.make_tensor(thd_q_lens_ptr, cute.make_layout((kernel.b + (thd_lens_form & 1),), stride=(1,)))
        thd_kv_lens = cute.make_tensor(thd_kv_lens_ptr, cute.make_layout((kernel.b + ((thd_lens_form >> 1) & 1),), stride=(1,)))
    offsets = [None] * 5
    if cutlass.const_expr(kernel.thd_origins is not None):
        for port in cutlass.range_constexpr(5):
            if cutlass.const_expr(kernel.thd_origins[port] is not None):
                offsets[port] = cute.make_tensor(thd_origin_ptrs[port], cute.make_layout((kernel.b + 1,), stride=(1,)))
    kernel(q, k, v, o, lse, sinks, q_lens, kv_lens, scale_softmax, thd_max_sq, thd_q_lens, thd_kv_lens, thd_lens_form, *offsets, stream)


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
        cutlass.Int64(0),
        cutlass.Float32(0),
        cutlass.Int64(0),
        ptr(cutlass.Int32, 4) if kernel.thd_varlen else None,
        ptr(cutlass.Int32, 4) if kernel.thd_varlen else None,
        cutlass.Int32(0) if kernel.thd_varlen else None,
        cutlass.Int32(0),
        cutlass.Int32(0),
        (
            tuple(ptr({32: cutlass.Int32, 64: cutlass.Int64}[spec[0]], spec[0] // 8) if spec is not None else None for spec in kernel.thd_origins)
            if kernel.thd_origins is not None
            else None
        ),
        (cutlass.Int64(0),) * 5 if kernel.thd_origins is not None else None,
        kernel,
        d_qk,
        d_v,
        lse_head_major,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options=f"--enable-tvm-ffi --gpu-arch={target}",
        cache_key=cache_key,
        symbol="frost_sdpa_fwd_prepared",
    )
