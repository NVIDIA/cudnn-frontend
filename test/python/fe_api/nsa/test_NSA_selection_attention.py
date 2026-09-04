# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

import pytest
from test_utils import torch_fork_set_rng

from fe_api.nsa.nsa_utils import (
    with_nsa_selection_attention_params,
    nsa_init,
    generate_block_indices,
    allocate_input_tensors,
    allocate_output_tensors,
)
from fe_api.nsa.nsa_reference import check_ref_nsa_selection_attention

"""
SelectionAttention API with explicitset_params, compile, and execute paths. 
Use this method when running one static configuration for each FmhaCute object.
"""


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_nsa_selection_attention_params
def test_nsa_selection_compile_execute(
    layout,
    dtype,
    acc_dtype,
    topk_size,
    block_size,
    request,
):
    try:
        from cudnn import NSA
        from cuda.bindings import driver as cuda
    except ImportError as e:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")
    if layout != "thd":
        pytest.skip("Only THD layout supported for selection attention, bshd layout not yet implemented")

    cfg = nsa_init(
        request=request,
        layout=layout,
        dtype=dtype,
        acc_dtype=acc_dtype,
        topk_size=topk_size,
        block_size=block_size,
    )

    Q, K, V, _, actual_s_q, _, cum_seqlen_q, cum_seqlen_kv, max_s_q, max_s_kv = allocate_input_tensors(cfg)
    block_counts, block_indices = generate_block_indices(
        cfg["actual_s_q"],
        cfg["h_k"],
        cfg["topk_sizes"],
        cfg["block_size"],
    )
    O, L, M, _, _ = allocate_output_tensors(cfg)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    selection_attention = NSA.SelectionAttention(
        sample_q=Q,
        sample_k=K,
        sample_v=V,
        sample_o=O,
        sample_l=L,
        sample_m=M,
        sample_block_indices=block_indices,
        sample_block_counts=block_counts,
        sample_cum_seqlen_q=cum_seqlen_q,
        sample_cum_seqlen_k=cum_seqlen_kv,
        max_s_q=max_s_q,
        max_s_k=max_s_kv,
        acc_dtype=cfg["acc_dtype"],
        block_size=cfg["block_size"],
        scale_softmax=cfg["scale_softmax"],
    )
    assert selection_attention.check_support()
    selection_attention.compile()
    selection_attention.execute(
        q_tensor=Q,
        k_tensor=K,
        v_tensor=V,
        o_tensor=O,
        l_tensor=L,
        m_tensor=M,
        block_indices_tensor=block_indices,
        block_counts_tensor=block_counts,
        cum_seqlen_q_tensor=cum_seqlen_q,
        cum_seqlen_k_tensor=cum_seqlen_kv,
        scale_softmax=cfg["scale_softmax"],
        current_stream=stream,
    )
    check_ref_nsa_selection_attention(
        Q,
        K,
        V,
        O,
        L,
        M,
        block_indices,
        block_counts,
        cfg,
    )


"""
SelectionAttention API with selection_attention_wrapper:
Use the wrapper to directly call SelectionAttention without explicit setup and compilation.
"""


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_nsa_selection_attention_params
def test_nsa_selection_wrapper(
    layout,
    dtype,
    acc_dtype,
    topk_size,
    block_size,
    request,
):
    try:
        from cudnn import NSA
        from cuda.bindings import driver as cuda
    except ImportError as e:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    if layout != "thd":
        pytest.skip("Only THD layout supported for selection attention, bshd layout not yet implemented")

    cfg = nsa_init(
        request=request,
        layout=layout,
        dtype=dtype,
        acc_dtype=acc_dtype,
        topk_size=topk_size,
        block_size=block_size,
    )

    Q, K, V, _, actual_s_q, _, cum_seqlen_q, cum_seqlen_kv, max_s_q, max_s_kv = allocate_input_tensors(cfg)
    block_counts, block_indices = generate_block_indices(
        cfg["actual_s_q"],
        cfg["h_k"],
        cfg["topk_sizes"],
        cfg["block_size"],
    )

    O, L, M = NSA.selection_attention_wrapper(
        q_tensor=Q,
        k_tensor=K,
        v_tensor=V,
        block_indices_tensor=block_indices,
        block_counts_tensor=block_counts,
        cum_seqlen_q_tensor=cum_seqlen_q,
        cum_seqlen_k_tensor=cum_seqlen_kv,
        block_size=cfg["block_size"],
        scale_softmax=cfg["scale_softmax"],
        o_dtype=cfg["dtype"],
        acc_dtype=cfg["acc_dtype"],
    )

    check_ref_nsa_selection_attention(
        Q,
        K,
        V,
        O,
        L,
        M,
        block_indices,
        block_counts,
        cfg,
    )


@pytest.mark.L0
def test_selection_attention_wrapper_cache_separates_is_causal(monkeypatch):
    """The wrapper must not reuse a non-causal compiled plan for a causal call."""
    try:
        from cudnn.native_sparse_attention.selection import api as selection_api
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    plans = []

    class FakeSelectionAttention:
        def __init__(self, **kwargs):
            self.is_causal = kwargs["is_causal"]
            self.compile_count = 0
            self.execute_count = 0
            plans.append(self)

        def check_support(self):
            return True

        def compile(self):
            self.compile_count += 1

        def execute(self, **kwargs):
            self.execute_count += 1

    monkeypatch.setattr(selection_api, "SelectionAttention", FakeSelectionAttention)
    monkeypatch.setattr(selection_api, "_cache_of_SelectionAttentionObjects", {})

    device = torch.device("cuda")
    q = torch.empty((4, 2, 8), device=device, dtype=torch.bfloat16)
    k = torch.empty((4, 1, 8), device=device, dtype=torch.bfloat16)
    v = torch.empty_like(k)
    block_indices = torch.zeros((4, 1, 1), device=device, dtype=torch.int32)
    block_counts = torch.ones((4, 1), device=device, dtype=torch.int32)
    cu_seqlens = torch.tensor([0, 4], device=device, dtype=torch.int32)

    def call(is_causal):
        """Invoke the wrapper with only the causal specialization changed."""
        selection_api.selection_attention_wrapper(
            q_tensor=q,
            k_tensor=k,
            v_tensor=v,
            block_indices_tensor=block_indices,
            block_counts_tensor=block_counts,
            cum_seqlen_q_tensor=cu_seqlens,
            cum_seqlen_k_tensor=cu_seqlens,
            max_s_q=4,
            max_s_k=4,
            is_causal=is_causal,
        )

    call(False)
    call(True)
    call(False)

    assert len(selection_api._cache_of_SelectionAttentionObjects) == 2
    assert [plan.is_causal for plan in plans] == [False, True]
    assert [plan.compile_count for plan in plans] == [1, 1]
    assert [plan.execute_count for plan in plans] == [2, 1]
