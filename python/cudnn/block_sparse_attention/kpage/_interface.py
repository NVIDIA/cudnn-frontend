# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Compile-cached launchers for the KPage forward (M128 x N128 and M64 x N256
# routes on SM100/SM103, Q-paired blk64 on SM90) and the KV-owned paged backward,
# plus the forward/backward entry points called from api.py.

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import Tensor

import cutlass.cute as cute

from cudnn.block_sparse_attention._interface import _get_device_arch, _to_cute_tensor, torch2cute_dtype_map
from cudnn.block_sparse_attention.csrc.bwd.bsa_bwd_prepost import _bwd_postprocess_convert, _bwd_preprocess
from cudnn.block_sparse_attention.csrc.bwd.sm100_blk128.bsa_bwd_sm100_kpage import (
    BlockSparseAttnBackwardSm100KPage,
    PagedKVWalkTensors,
)
from cudnn.block_sparse_attention.csrc.fwd.sm100_blk128.bsa_fwd_sm100_kpage import BlockSparseAttnForwardSm100KPage
from cudnn.block_sparse_attention.csrc.fwd.sm100_blk128.bsa_fwd_sm100_kpage_q64 import BlockSparseAttnForwardSm100KPageQ64
from cudnn.block_sparse_attention.csrc.fwd.sm90_blk64.bsa_fwd_sm90_kpage import BlockSparseAttnForwardSm90KPage
from cudnn.block_sparse_attention.csrc.utils.cute_dsl_utils import get_broadcast_dims, to_cute_tensor as to_cute_tensor_static
from cudnn.block_sparse_attention.csrc.utils.kpage_utils import BLOCK, HEAD_DIM, PAGE
from ._build import Blk64PairForwardTables, PagedKVWalkTables, set_pad_q_stats, zero_pad_forward_state, zero_pad_rows
from ._compile import compile_key, current_stream, fake_stream, to_cute_tensor
from ._layout import KPageLayout


def _fwd_m128(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    lse: Tensor,
    page_table: Tensor,
    mask_table: Tensor,
    tile_offsets: Tensor,
    tile_counts: Tensor,
    full_tile_counts: Tensor,
    work_indices: Tensor,
    softmax_scale: float,
) -> None:
    head_dim = q.shape[-1]
    qhead_per_kvhead = q.shape[1] // k.shape[1]
    tail_page = k.shape[2] % PAGE != 0  # K/V end inside a page: row-extent descriptors
    nonpositive_scale = not softmax_scale > 0.0  # scale the scores before the masks (NaN/inf otherwise)
    tensors = (q, k, v, out, lse, page_table, mask_table, tile_offsets, tile_counts, full_tile_counts, work_indices)
    key = compile_key("kpage_fwd_m128", (torch2cute_dtype_map[q.dtype], head_dim, qhead_per_kvhead, tail_page, nonpositive_scale, _get_device_arch()), tensors)
    cache = _fwd_m128.compile_cache
    if key not in cache:
        q_t, k_t, v_t, o_t = [to_cute_tensor(t, (0, 1, 2)) for t in (q, k, v, out)]
        lse_t = to_cute_tensor(lse, (0, 1, 2), assumed_align=4)
        pt_t = to_cute_tensor(page_table, 0, assumed_align=32)
        mt_t = to_cute_tensor(mask_table, 0, assumed_align=16)
        to_t, tc_t, ft_t = [to_cute_tensor(t, (0, 1, 2), assumed_align=4) for t in (tile_offsets, tile_counts, full_tile_counts)]
        wi_t = to_cute_tensor(work_indices, 0, assumed_align=4)
        cache[key] = cute.compile(
            BlockSparseAttnForwardSm100KPage(head_dim, head_dim, qhead_per_kvhead=qhead_per_kvhead, tail_page=tail_page, nonpositive_scale=nonpositive_scale),
            q_t,
            k_t,
            v_t,
            o_t,
            lse_t,
            softmax_scale,
            pt_t,
            mt_t,
            to_t,
            ft_t,
            tc_t,
            wi_t,
            fake_stream(),
            options="--enable-tvm-ffi",
        )
    cache[key](
        q.detach(),
        k.detach(),
        v.detach(),
        out.detach(),
        lse,
        softmax_scale,
        page_table,
        mask_table,
        tile_offsets,
        full_tile_counts,
        tile_counts,
        work_indices,
        current_stream(),
    )


_fwd_m128.compile_cache = {}


def _fwd_m64(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    lse: Tensor,
    page_table: Tensor,
    mask_table: Tensor,
    tile_offsets: Tensor,
    iter_counts: Tensor,
    full_iter_counts: Tensor,
    work_indices: Tensor,
    softmax_scale: float,
) -> None:
    head_dim = q.shape[-1]
    qhead_per_kvhead = q.shape[1] // k.shape[1]
    tail_page = k.shape[2] % PAGE != 0
    nonpositive_scale = not softmax_scale > 0.0  # scale the scores before the masks (NaN/inf otherwise)
    tensors = (q, k, v, out, lse, page_table, mask_table, tile_offsets, iter_counts, full_iter_counts, work_indices)
    key = compile_key("kpage_fwd_m64", (torch2cute_dtype_map[q.dtype], head_dim, qhead_per_kvhead, tail_page, nonpositive_scale, _get_device_arch()), tensors)
    cache = _fwd_m64.compile_cache
    if key not in cache:
        q_t, k_t, v_t, o_t = [to_cute_tensor(t, (0, 1, 2)) for t in (q, k, v, out)]
        lse_t = to_cute_tensor(lse, (0, 1, 2), assumed_align=4)
        pt_t = to_cute_tensor(page_table, 0, assumed_align=32)
        mt_t = to_cute_tensor(mask_table, 0, assumed_align=16)
        to_t, ic_t, fi_t = [to_cute_tensor(t, (0, 1, 2), assumed_align=4) for t in (tile_offsets, iter_counts, full_iter_counts)]
        wi_t = to_cute_tensor(work_indices, 0, assumed_align=4)
        cache[key] = cute.compile(
            BlockSparseAttnForwardSm100KPageQ64(
                head_dim, head_dim, qhead_per_kvhead=qhead_per_kvhead, tail_page=tail_page, nonpositive_scale=nonpositive_scale
            ),
            q_t,
            k_t,
            v_t,
            o_t,
            lse_t,
            softmax_scale,
            pt_t,
            mt_t,
            to_t,
            fi_t,
            ic_t,
            wi_t,
            fake_stream(),
            options="--enable-tvm-ffi",
        )
    cache[key](
        q.detach(),
        k.detach(),
        v.detach(),
        out.detach(),
        lse,
        softmax_scale,
        page_table,
        mask_table,
        tile_offsets,
        full_iter_counts,
        iter_counts,
        work_indices,
        current_stream(),
    )


_fwd_m64.compile_cache = {}


def _fwd_sm90_pairs(q: Tensor, k: Tensor, v: Tensor, out: Tensor, lse: Tensor, tables: Blk64PairForwardTables, softmax_scale: float) -> None:
    """SM90 blk64 forward over Q-pair walks; q/k/v/out are [1, B * H, S, D] views, lse [1, B * H, S]."""
    head_dim = q.shape[-1]
    nonpositive_scale = not softmax_scale > 0.0
    # The kernel takes (S, D, H, B) Q/K/O, (D, S, H, B) V and (S, H, B) LSE views.
    views = (q.permute(2, 3, 1, 0), k.permute(2, 3, 1, 0), v.permute(3, 2, 1, 0), out.permute(2, 3, 1, 0), lse.permute(2, 1, 0))
    tensors = (*views, tables.page_table, tables.keep, tables.ranges)
    key = compile_key("kpage_fwd_sm90_pairs", (torch2cute_dtype_map[q.dtype], head_dim, nonpositive_scale, _get_device_arch()), tensors)
    cache = _fwd_sm90_pairs.compile_cache
    if key not in cache:
        leading = (1, 1, 0, 1, 0, 1, 2, 2)
        aligns = (128, 128, 128, 128, 4, 16, 16, 4)
        converted = [_to_cute_tensor(t, assumed_align=a, leading_dim=d) for t, d, a in zip(tensors, leading, aligns)]
        cache[key] = cute.compile(
            BlockSparseAttnForwardSm90KPage(head_dim, nonpositive_scale=nonpositive_scale),
            *converted,
            softmax_scale,
            fake_stream(),
            options="--enable-tvm-ffi",
        )
    cache[key](*(t.detach() for t in tensors), softmax_scale, current_stream())


_fwd_sm90_pairs.compile_cache = {}


def _bwd_paged_kv(
    dout: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    lse: Tensor,
    out: Tensor,
    tables: PagedKVWalkTables,
    q_sizes: Optional[Tensor],  # None when every Q block is dense
    softmax_scale: float,
    dq: Tensor,
    dk: Tensor,
    dv: Tensor,
) -> None:
    batch_size, num_heads, seqlen_q, head_dim = q.shape
    num_heads_kv, seqlen_k = k.shape[1], k.shape[2]
    q_bshd, k_bshd, v_bshd, out_bshd, dout_bshd = [t.transpose(1, 2) for t in (q, k, v, out, dout)]
    dq_bshd, dk_bshd, dv_bshd = [t.transpose(1, 2) for t in (dq, dk, dv)]

    # fp32 accumulators in the layout the stock BSA pre/postprocess kernels use, at the
    # 128-row-rounded lengths the page walks address (preprocess gives the rounded tail
    # LSE = +inf, dPsum = 0; postprocess converts the first seqlen rows)
    head_dim_rounded = (head_dim + 31) // 32 * 32
    seqlen_q_rounded = (seqlen_q + BLOCK - 1) // BLOCK * BLOCK
    seqlen_k_rounded = (seqlen_k + BLOCK - 1) // BLOCK * BLOCK
    dq_accum = torch.empty(batch_size, num_heads, seqlen_q_rounded * head_dim_rounded, dtype=torch.float32, device=q.device)
    dpsum = torch.empty(batch_size, num_heads, seqlen_q_rounded, dtype=torch.float32, device=q.device)
    lse_log2 = torch.empty_like(dpsum)
    dk_accum = torch.zeros(batch_size, num_heads_kv, seqlen_k_rounded * head_dim_rounded, dtype=torch.float32, device=q.device)
    dv_accum = torch.zeros_like(dk_accum)

    dtype = torch2cute_dtype_map[q.dtype]
    arch = _get_device_arch()
    _bwd_preprocess(out_bshd, dout_bshd, dpsum, lse, lse_log2, dq_accum, None, None, None, dtype, head_dim, head_dim, BLOCK)
    if q_sizes is not None:
        set_pad_q_stats(lse_log2, dpsum, q_sizes)

    key = (
        "kpage_bwd",
        arch,
        dtype,
        head_dim,
        tables.has_tail_pages,
        num_heads // num_heads_kv,
        get_broadcast_dims(q_bshd),
        get_broadcast_dims(k_bshd),
        get_broadcast_dims(v_bshd),
        get_broadcast_dims(dout_bshd),
    )
    cache = _bwd_paged_kv.compile_cache
    if tables.work_indices.shape[0] > 0:
        if key not in cache:
            q_t, k_t, v_t, do_t = [to_cute_tensor_static(t) for t in (q_bshd, k_bshd, v_bshd, dout_bshd)]
            dk_t, dv_t, dq_accum_t, dpsum_t, lse_log2_t = [to_cute_tensor_static(t) for t in (dk_accum, dv_accum, dq_accum, dpsum, lse_log2)]
            walk = PagedKVWalkTensors(
                to_cute_tensor_static(tables.tile_offsets, assumed_align=4),
                to_cute_tensor_static(tables.tile_counts, assumed_align=4),
                to_cute_tensor_static(tables.page_table, assumed_align=32),
                to_cute_tensor_static(tables.mask_table, assumed_align=8),
                to_cute_tensor_static(tables.kv_page_table, assumed_align=32),
                to_cute_tensor_static(tables.kv_valid_table, assumed_align=8),
                to_cute_tensor(tables.work_indices, 0, assumed_align=8),
            )
            cache[key] = cute.compile(
                BlockSparseAttnBackwardSm100KPage(head_dim, mask_rows=tables.has_tail_pages, qhead_per_kvhead=num_heads // num_heads_kv),
                q_t,
                k_t,
                v_t,
                do_t,
                lse_log2_t,
                dpsum_t,
                dq_accum_t,
                dk_t,
                dv_t,
                softmax_scale,
                walk,
                cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                options="--enable-tvm-ffi",
            )
        cache[key](
            q_bshd.detach(),
            k_bshd.detach(),
            v_bshd.detach(),
            dout_bshd,
            lse_log2,
            dpsum,
            dq_accum,
            dk_accum,
            dv_accum,
            softmax_scale,
            (
                tables.tile_offsets,
                tables.tile_counts,
                tables.page_table,
                tables.mask_table,
                tables.kv_page_table,
                tables.kv_valid_table,
                tables.work_indices,
            ),
        )

    _bwd_postprocess_convert(dq_accum, dq_bshd, softmax_scale, None, None, arch, dtype, head_dim, BLOCK, False)
    _bwd_postprocess_convert(dk_accum, dk_bshd, softmax_scale, None, None, arch, dtype, head_dim, BLOCK, False)
    _bwd_postprocess_convert(dv_accum, dv_bshd, 1.0, None, None, arch, dtype, head_dim, BLOCK, False)


_bwd_paged_kv.compile_cache = {}


def forward(q: Tensor, k: Tensor, v: Tensor, layout: KPageLayout, softmax_scale: Optional[float]) -> Tuple[Tensor, Tensor]:
    """Forward over a prepared layout; returns (o, lse) with zeroed padding rows (blk128; the blk64 path has none)."""
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(HEAD_DIM)
    softmax_scale = float(softmax_scale)
    out = torch.empty_like(q)  # keeps BHSD or BSHD strides
    lse = torch.empty(q.shape[:-1], dtype=torch.float32, device=q.device)
    t64 = layout._blk64_tables
    if layout._pair_tables is not None:
        _fwd_sm90_pairs(q, k, v, out, lse, layout._pair_tables, softmax_scale)
    elif t64 is not None and t64.work_indices.numel() > 0:
        _fwd_m64(q, k, v, out, lse, t64.page_table, t64.mask_table, t64.tile_offsets, t64.iter_counts, t64.full_iter_counts, t64.work_indices, softmax_scale)
    t128 = layout._tables  # None on the blk64 path: every Q block is one M64 tile
    if t128 is not None and t128.work_indices.numel() > 0:
        _fwd_m128(
            q, k, v, out, lse, t128.page_table, t128.mask_table, t128.tile_offsets, t128.tile_counts, t128.full_tile_counts, t128.work_indices, softmax_scale
        )
    if layout._has_partial_q is not False:
        zero_pad_forward_state(out, lse, layout._q_sizes)
    return out, lse


def _bwd_sm90_refined(
    dout: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    lse: Tensor,
    layout: KPageLayout,
    softmax_scale: float,
    dq: Optional[Tensor],
    dk: Optional[Tensor],
    dv: Optional[Tensor],
) -> Tuple[Tensor, Tensor, Tensor]:
    """SM90 blk128: the blk64 backward on the refined metadata ([1, B * H, S, D] views).

    Padding Q rows get LSE = +inf, so P and dS vanish on them and their q / do rows take no part, as in the SM100
    KPage backward; the caller zeroes the padding rows of the gradients.
    """
    from cudnn.block_sparse_attention._interface import bsa_attn_bwd

    refined = layout._refined
    if layout._has_partial_q is not False:
        lse = lse.masked_fill(layout._padding_q_rows(), float("inf"))
    # resolve_layout admits compact BHSD or BSHD (batch 1) views; run BSHD storage as BSHD instead of copying it.
    bshd = not q.is_contiguous()
    swap = (lambda t: None if t is None else t.transpose(1, 2)) if bshd else (lambda t: t)
    grads = bsa_attn_bwd(
        *(swap(t) for t in (dout, q, k, v, out)),
        lse,
        refined.block_index,
        refined.block_index.shape[-1],
        refined.block_sizes,
        q2k_block_nums=refined.block_count,
        softmax_scale=softmax_scale,
        dq=swap(dq),
        dk=swap(dk),
        dv=swap(dv),
        sparse_block_size=64,
        layout="bshd" if bshd else "bhsd",
    )
    result = []
    for given, got in zip((dq, dk, dv), grads):
        got = swap(got)
        if given is not None and given.data_ptr() != got.data_ptr():
            given.copy_(got)
            got = given
        result.append(got)
    return tuple(result)


def backward(
    dout: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    lse: Tensor,
    layout: KPageLayout,
    softmax_scale: Optional[float],
    dq: Optional[Tensor],
    dk: Optional[Tensor],
    dv: Optional[Tensor],
) -> Tuple[Tensor, Tensor, Tensor]:
    """Backward over a prepared layout; dq/dk/dv are exact on valid rows and zero on padding rows."""
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(HEAD_DIM)
    if layout._refined is not None:
        dq, dk, dv = _bwd_sm90_refined(dout, q, k, v, out, lse, layout, float(softmax_scale), dq, dk, dv)
        if layout._has_partial_q is not False or layout._has_partial_kv is not False:
            zero_pad_rows(dq, dk, dv, layout._q_sizes, layout._kv_sizes)
        return dq, dk, dv
    dq = torch.empty_like(q) if dq is None else dq
    dk = torch.empty_like(k) if dk is None else dk
    dv = torch.empty_like(v) if dv is None else dv
    q_sizes = layout._q_sizes if layout._has_partial_q is not False else None
    _bwd_paged_kv(dout, q, k, v, lse, out, layout._tables.resolve_bwd_tables(), q_sizes, float(softmax_scale), dq, dk, dv)
    if layout._has_partial_q is not False or layout._has_partial_kv is not False:
        zero_pad_rows(dq, dk, dv, layout._q_sizes, layout._kv_sizes)
    return dq, dk, dv
