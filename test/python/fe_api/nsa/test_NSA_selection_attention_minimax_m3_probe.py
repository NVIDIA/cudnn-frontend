# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in functional probe for MiniMax-M3's exact selected-attention shape.

This test deliberately does not widen ``SelectionAttention.check_support()``.
It exercises the private kernel only, with the token-level causal mask required
when the current 128-token block is selected. Run it explicitly with::

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


class _MiniMaxM3CausalSelectionAttentionFwd:
    """Late-binding shim so importing this test does not eagerly import CuTe DSL."""

    def __new__(cls, *args, **kwargs):
        from cudnn.native_sparse_attention.selection.NSA_select_attn_fwd_hmma import HopperSelectAttentionFwd

        return HopperSelectAttentionFwd(*args, **kwargs, causal_within_selected_blocks=True)


def _minimax_metadata(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    query_positions = torch.arange(_SEQLEN, device=device, dtype=torch.int32)
    block_ids = torch.arange(_TOPK_BLOCKS, device=device, dtype=torch.int32)
    indices = block_ids.view(1, 1, -1).expand(_SEQLEN, _KV_HEADS, -1).contiguous()
    counts = (query_positions // _BLOCK_SIZE + 1).clamp(max=_TOPK_BLOCKS)
    return indices, counts.view(-1, 1).expand(-1, _KV_HEADS).contiguous()


def _causal_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float):
    q_grouped = q.view(_SEQLEN, _KV_HEADS, _Q_HEADS // _KV_HEADS, _HEAD_DIM)
    out = torch.empty_like(q, dtype=torch.float32).view_as(q_grouped)
    row_sum = torch.empty((_SEQLEN, _KV_HEADS, _Q_HEADS // _KV_HEADS), device=q.device, dtype=torch.float32)
    row_max = torch.empty_like(row_sum)
    causal = torch.ones((_SEQLEN, _SEQLEN), device=q.device, dtype=torch.bool).tril_()

    for kv_head in range(_KV_HEADS):
        scores = torch.einsum("tgd,sd->tgs", q_grouped[:, kv_head].float(), k[:, kv_head].float()) * scale
        scores.masked_fill_(~causal[:, None, :], -torch.inf)
        max_for_head = scores.amax(dim=-1)
        probabilities = torch.softmax(scores, dim=-1)
        out[:, kv_head] = torch.einsum("tgs,sd->tgd", probabilities, v[:, kv_head].float())
        row_max[:, kv_head] = max_for_head
        row_sum[:, kv_head] = torch.exp(scores - max_for_head[..., None]).sum(dim=-1)

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
        scale_softmax=scale,
    )

    # Exact-shape private-kernel probe: preserve the public block-size gate until
    # this path has target-GPU evidence and a production MiniMax adapter.
    operation.input_layout = "T,H,D"
    operation.dtype = dtype
    operation.h_q = _Q_HEADS
    operation.h_kv = _KV_HEADS
    operation.gqa_group_size = _Q_HEADS // _KV_HEADS
    operation.head_dim = _HEAD_DIM
    operation.value_dim = _HEAD_DIM
    operation.l_desc = operation._unpad_tensor_to_ndim(operation.l_desc, 2, "sample_l")
    operation.m_desc = operation._unpad_tensor_to_ndim(operation.m_desc, 2, "sample_m")
    operation._kernel = _MiniMaxM3CausalSelectionAttentionFwd
    operation._is_supported = True

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

    o_ref, row_sum_ref, row_max_ref = _causal_reference(q, k, v, scale)
    torch.testing.assert_close(o.float(), o_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(row_sum, row_sum_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(row_max, row_max_ref, atol=3e-2, rtol=3e-2)
