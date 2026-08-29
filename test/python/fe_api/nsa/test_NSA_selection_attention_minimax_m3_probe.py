# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in public-API probe for MiniMax-M3 selected attention.

The metadata intentionally selects sparse, non-monotonic block sets and places
the current 128-token block at varying slots. This catches implementations that
implicitly sort the selected blocks or omit token-level causality inside the
current block. Run it explicitly with::

    pytest -m L4 fe_api/nsa/test_NSA_selection_attention_minimax_m3_probe.py
"""

import math

import pytest
import torch

from test_utils import torch_fork_set_rng

_SEQLEN = 2048
_Q_HEADS = 64
_KV_HEADS = 4
_HEAD_DIM = 128
_BLOCK_SIZE = 128
_TOPK_BLOCKS = 16
_REFERENCE_QUERY_CHUNK = 128


def _minimax_metadata(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    indices = torch.zeros((_SEQLEN, _KV_HEADS, _TOPK_BLOCKS), dtype=torch.int32)
    counts = torch.zeros((_SEQLEN, _KV_HEADS), dtype=torch.int32)

    for query in range(_SEQLEN):
        current_block = query // _BLOCK_SIZE
        for kv_head in range(_KV_HEADS):
            if current_block == _TOPK_BLOCKS - 1 and (query + kv_head) % 64 == 0:
                # Exercise the full MiniMax K=16 metadata capacity on a few rows.
                selected = list(range(_TOPK_BLOCKS))
            else:
                selected = []
                if current_block > 0:
                    first_old_block = (query // 19 + 2 * kv_head) % current_block
                    selected.append(first_old_block)
                if current_block >= 3:
                    second_old_block = (first_old_block + current_block // 2 + 1) % current_block
                    if second_old_block not in selected:
                        selected.append(second_old_block)
                selected.append(current_block)

            # Vary the order per query/head. In particular, the current block is
            # not assigned a distinguished metadata slot.
            shift = (query // 11 + kv_head) % len(selected)
            selected = selected[shift:] + selected[:shift]
            if (query + kv_head) % 2:
                selected.reverse()

            counts[query, kv_head] = len(selected)
            indices[query, kv_head, : len(selected)] = torch.tensor(selected, dtype=torch.int32)

    return indices.to(device), counts.to(device)


def _causal_sparse_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_indices: torch.Tensor,
    block_counts: torch.Tensor,
    scale: float,
):
    q_grouped = q.view(_SEQLEN, _KV_HEADS, _Q_HEADS // _KV_HEADS, _HEAD_DIM)
    out = torch.empty_like(q, dtype=torch.float32).view_as(q_grouped)
    row_sum = torch.empty((_SEQLEN, _KV_HEADS, _Q_HEADS // _KV_HEADS), device=q.device, dtype=torch.float32)
    row_max = torch.empty_like(row_sum)

    query_positions = torch.arange(_SEQLEN, device=q.device)
    block_offsets = torch.arange(_BLOCK_SIZE, device=q.device)
    metadata_slots = torch.arange(_TOPK_BLOCKS, device=q.device)

    for kv_head in range(_KV_HEADS):
        key_positions = block_indices[:, kv_head, :, None].to(torch.int64) * _BLOCK_SIZE + block_offsets
        valid_slots = metadata_slots[None, :] < block_counts[:, kv_head, None]
        valid_tokens = valid_slots[:, :, None] & (key_positions < _SEQLEN) & (key_positions <= query_positions[:, None, None])

        selected_tokens = torch.zeros((_SEQLEN, _SEQLEN), device=q.device, dtype=torch.bool)
        query_rows = query_positions[:, None, None].expand_as(key_positions)
        selected_tokens[query_rows[valid_tokens], key_positions[valid_tokens]] = True

        for query_start in range(0, _SEQLEN, _REFERENCE_QUERY_CHUNK):
            query_end = min(query_start + _REFERENCE_QUERY_CHUNK, _SEQLEN)
            scores = torch.einsum("tgd,sd->tgs", q_grouped[query_start:query_end, kv_head].float(), k[:, kv_head].float()) * scale
            scores.masked_fill_(~selected_tokens[query_start:query_end, None, :], -torch.inf)
            max_for_chunk = scores.amax(dim=-1)
            probabilities = torch.softmax(scores, dim=-1)
            out[query_start:query_end, kv_head] = torch.einsum("tgs,sd->tgd", probabilities, v[:, kv_head].float())
            row_max[query_start:query_end, kv_head] = max_for_chunk
            row_sum[query_start:query_end, kv_head] = torch.exp(scores - max_for_chunk[..., None]).sum(dim=-1)

    return out.view_as(q), row_sum.view(_SEQLEN, _Q_HEADS, 1), row_max.view(_SEQLEN, _Q_HEADS, 1)


@pytest.mark.L4
@torch_fork_set_rng(seed=2029)
def test_minimax_m3_selection_attention_block128_causal_probe():
    if not torch.cuda.is_available():
        pytest.skip("MiniMax-M3 selection probe requires CUDA")
    major, minor = torch.cuda.get_device_capability()
    if major != 10:
        pytest.skip(f"MiniMax-M3 selection probe targets SM10x, found SM{major}{minor}")

    try:
        from cuda.bindings import driver as cuda
        from cudnn import NSA
    except ImportError:
        pytest.skip("cuDNN Frontend CuTe DSL dependencies are not installed")

    device = torch.device("cuda")
    dtype = torch.bfloat16
    scale = 1.0 / math.sqrt(_HEAD_DIM)
    q = torch.randn((_SEQLEN, _Q_HEADS, _HEAD_DIM), device=device, dtype=dtype)
    k = torch.randn((_SEQLEN, _KV_HEADS, _HEAD_DIM), device=device, dtype=dtype)
    v = torch.randn_like(k)
    o = torch.empty_like(q)
    row_sum = torch.empty((_SEQLEN, _Q_HEADS, 1), device=device, dtype=torch.float32)
    row_max = torch.empty_like(row_sum)
    block_indices, block_counts = _minimax_metadata(device)
    cu_seqlens = torch.tensor([0, _SEQLEN], device=device, dtype=torch.int32)

    operation = NSA.SelectionAttention(
        sample_q=q,
        sample_k=k,
        sample_v=v,
        sample_o=o,
        sample_l=row_sum,
        sample_m=row_max,
        sample_block_indices=block_indices,
        sample_block_counts=block_counts,
        sample_cum_seqlen_q=cu_seqlens,
        sample_cum_seqlen_k=cu_seqlens,
        max_s_q=_SEQLEN,
        max_s_k=_SEQLEN,
        block_size=_BLOCK_SIZE,
        is_causal=True,
        scale_softmax=scale,
    )
    assert operation.check_support()
    operation.compile()
    operation.execute(
        q_tensor=q,
        k_tensor=k,
        v_tensor=v,
        o_tensor=o,
        l_tensor=row_sum,
        m_tensor=row_max,
        block_indices_tensor=block_indices,
        block_counts_tensor=block_counts,
        cum_seqlen_q_tensor=cu_seqlens,
        cum_seqlen_k_tensor=cu_seqlens,
        current_stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )

    o_ref, row_sum_ref, row_max_ref = _causal_sparse_reference(q, k, v, block_indices, block_counts, scale)
    torch.testing.assert_close(o.float(), o_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(row_sum, row_sum_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(row_max, row_max_ref, atol=3e-2, rtol=3e-2)
