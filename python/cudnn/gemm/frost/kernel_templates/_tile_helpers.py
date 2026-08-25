# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Tile-level helpers shared by the rendered kernel templates.

A template is RENDERED (its `@@INJECT_*@@` blocks become module-level
constants) and then exec'd from the kernel cache under a synthetic module
name, so it cannot use relative imports and this module is never rendered.
Everything here therefore takes what it needs as ARGUMENTS -- a helper that
reads an injected constant (`num_mma_m`, `tile_swizzle_n`, `ab_dtype`, ...)
has to stay in the template, or be re-signed to receive it.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.experimental.primitives as nvvm


@cute.jit
def l2_swizzle_tile(raw_m, raw_n, nt_m, nt_n, swizzle_w, identity=False):
    """N-direction super-block rasterization of the (m, n) cgrp-tile coord, for
    L2 reuse. ``identity=True`` compiles out the general mapping when the caller
    knows that ``swizzle_w == 1``.
    """
    if cutlass.const_expr(identity):
        return raw_m, raw_n
    t = raw_n * nt_m + raw_m
    blk = nt_m * swizzle_w
    sb = t // blk
    off = t - sb * blk
    base_n = sb * swizzle_w
    cur_S = cutlass.min(cutlass.Int32(swizzle_w), nt_n - base_n)
    log_m = off // cur_S
    log_n = base_n + off - log_m * cur_S
    return log_m, log_n


def epi_subtile_spans(cols, epi_n=32):
    """Power-of-two column spans the epilogue drains a tile in (host-side).
    Starts at ``epi_n`` and halves to fit the remainder, so any 8-multiple N is
    covered whatever the widest span is."""
    spans = []
    off = 0
    while off < cols:
        w = epi_n
        while w > cols - off:
            w //= 2
        spans.append((off, w))
        off += w
    return spans


TENSOR_MAP_QWORDS = 16


def moe_swizzle_tile(t, nt_m, nt_n, swizzle_w):
    """Group-local linear tile index -> (m, n) under an N-super-block walk.
    ``swizzle_w == nt_n`` reproduces the plain n-fast split; ``1`` gives m-fast.
    """
    blk = cutlass.max(nt_m * swizzle_w, cutlass.Int32(1))
    sb = t // blk
    off = t - sb * blk
    base_n = sb * swizzle_w
    cur_S = cutlass.min(cutlass.Int32(swizzle_w), nt_n - base_n)
    tile_m = off // cur_S
    tile_n = base_n + off - tile_m * cur_S
    return tile_m, tile_n


@cute.jit
def replace_tensormap_global_dim_1(desc_ptr, new_dim) -> None:
    nvvm.tensormap_replace(
        nvvm.TensormapField.GLOBAL_DIM,
        desc_ptr,
        new_value=cutlass.Int32(new_dim),
        ord=1,
    )


@cute.jit
def replace_tensormap_global_dim_2(desc_ptr, new_dim) -> None:
    nvvm.tensormap_replace(
        nvvm.TensormapField.GLOBAL_DIM,
        desc_ptr,
        new_value=cutlass.Int32(new_dim),
        ord=2,
    )


@cute.jit
def replace_tensormap_global_address(desc_ptr, new_address) -> None:
    nvvm.tensormap_replace(
        nvvm.TensormapField.GLOBAL_ADDRESS,
        desc_ptr,
        new_value=cutlass.Int64(new_address),
    )


@cute.jit
def fence_tensormap_release() -> None:
    nvvm.fence_proxy_release(
        nvvm.MemScope.GPU,
        from_proxy=nvvm.Proxy.GENERIC,
        to_proxy=nvvm.Proxy.TENSORMAP,
    )


@cute.jit
def fence_tensormap_acquire(desc_ptr) -> None:
    nvvm.fence_proxy_acquire(
        nvvm.MemScope.GPU,
        desc_ptr,
        TENSOR_MAP_QWORDS * 8,
        from_proxy=nvvm.Proxy.GENERIC,
        to_proxy=nvvm.Proxy.TENSORMAP,
    )


@cute.jit
def moe_group_at(visit_idx, num_groups, num_experts):
    """Visitation index -> routed group index.

    ``num_groups == num_experts`` (or a non-multiple) walks groups in order. Batched MoE
    (``num_groups == B * num_experts``) walks expert-major -- the B groups sharing expert
    ``g % E`` become consecutive, so the expert weight is fetched once instead of B times.
    """
    per_expert = num_groups // cutlass.max(num_experts, cutlass.Int32(1))
    group = visit_idx
    if per_expert > 1 and per_expert * num_experts == num_groups:
        group = (visit_idx % per_expert) * num_experts + (visit_idx // per_expert)
    return group


@cute.jit
def copy_tensormap_to_workspace(src_desc_ptr, dst_i64_ptr) -> None:
    """Copy the 128-byte A tensormap into ``dst_i64_ptr`` (seeds the SMEM copy).

    The trip count is a compile-time constant, so this is a constexpr loop --
    the templates had drifted into two spellings of the same fully-unrolled
    copy (`range_constexpr` vs `range(..., unroll_full=True)`).
    """
    src_words = cute.make_ptr(cutlass.Int64, src_desc_ptr.toint(), mem_space=cute.AddressSpace.generic)
    for i in cutlass.range_constexpr(TENSOR_MAP_QWORDS):
        dst_i64_ptr.subview(i).store((src_words + i).load())


def tcgen05_alloc(tmem_ptr, num_cols, *, is_exclusive=False, group=None):
    if is_exclusive:
        nvvm.tcgen05_alloc(tmem_ptr, num_cols, is_exclusive=True, group=group)
    else:
        nvvm.tcgen05_alloc(tmem_ptr, num_cols, group=group)


def tcgen05_dealloc(tmem_ptr, num_cols, *, is_exclusive=False, group=None):
    if is_exclusive:
        nvvm.tcgen05_dealloc(tmem_ptr, num_cols, is_exclusive=True, group=group)
    else:
        nvvm.tcgen05_dealloc(tmem_ptr, num_cols, group=group)


def tcgen05_mma_block_scale(mma_kind, cta_group, d, a, b, idesc, *, enable_input_d, scale_a, scale_b, scale_vec_size, b_collector_op=None):
    if b_collector_op is None:
        nvvm.tcgen05_mma_block_scale(
            mma_kind,
            cta_group,
            d,
            a,
            b,
            idesc,
            enable_input_d=enable_input_d,
            scale_a=scale_a,
            scale_b=scale_b,
            scale_vec_size=scale_vec_size,
        )
    else:
        nvvm.tcgen05_mma_block_scale(
            mma_kind,
            cta_group,
            d,
            a,
            b,
            idesc,
            enable_input_d=enable_input_d,
            scale_a=scale_a,
            scale_b=scale_b,
            scale_vec_size=scale_vec_size,
            b_collector_op=b_collector_op,
        )
