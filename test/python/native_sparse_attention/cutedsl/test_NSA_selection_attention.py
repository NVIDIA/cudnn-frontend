# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

import pytest
from test_utils import torch_fork_set_rng

from native_sparse_attention.cutedsl.nsa_utils import (
    with_nsa_selection_attention_params,
    nsa_init,
    generate_block_indices,
    allocate_input_tensors,
    allocate_output_tensors,
)
from native_sparse_attention.cutedsl.nsa_reference import check_ref_nsa_selection_attention, run_ref_nsa_selection_attention

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


def _select_own_blocks_rotated(seq_lens, num_kv_heads, block_size, device, past_end_block=False):
    """Every query selects all blocks of its own sequence, rotated by its position.

    The rotation puts the sequence's last block in every metadata slot; the kernel
    visits the slots in reverse order, so that block is processed first, in the
    middle and last. With past_end_block, the block after the sequence's last one
    (it starts at or past the end of the sequence) is appended in the last slot,
    so the kernel processes it first.
    """
    extra = 1 if past_end_block else 0
    max_blocks = max((seq_len + block_size - 1) // block_size for seq_len in seq_lens) + extra
    block_counts = torch.zeros(sum(seq_lens), num_kv_heads, dtype=torch.int32)
    block_indices = torch.zeros(sum(seq_lens), num_kv_heads, max_blocks, dtype=torch.int32)
    offset = 0
    for seq_len in seq_lens:
        num_blocks = (seq_len + block_size - 1) // block_size
        for t in range(seq_len):
            shift = t % num_blocks
            blocks = list(range(shift, num_blocks)) + list(range(shift)) + list(range(num_blocks, num_blocks + extra))
            block_indices[offset + t, :, : len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
            block_counts[offset + t, :] = len(blocks)
        offset += seq_len
    return block_counts.to(device), block_indices.to(device)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize("block_size, past_end_block", [(32, False), (64, False), (128, False), (64, True)])
def test_nsa_selection_thd_partial_last_block(block_size, past_end_block, request):
    """Ragged THD lengths that are not multiples of block_size, non-causal.

    Every query selects the partial last block of its sequence. The rest of that
    block's tile holds the next sequence's keys, or rows past the end of K/V for
    the last sequence; they must not enter the softmax. A selected block that
    starts past the end of its sequence has no keys and contributes nothing, also
    when the kernel processes it first. The reference clamps every block to its
    sequence, and the row max (M) and row sum (L) must match it too.
    """
    try:
        from cudnn import NSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    seq_lens = [70, 130, 100]
    cfg = nsa_init(
        request=request,
        layout="thd",
        dtype=torch.float16,
        acc_dtype=torch.float32,
        block_size=block_size,
    )
    cfg["b"] = len(seq_lens)
    cfg["actual_s_q"] = torch.tensor(seq_lens, dtype=torch.int32).cuda()
    cfg["actual_s_kv"] = cfg["actual_s_q"].clone()

    Q, K, V, _, _, _, cum_seqlen_q, cum_seqlen_kv, _, _ = allocate_input_tensors(cfg)
    block_counts, block_indices = _select_own_blocks_rotated(seq_lens, cfg["h_k"], block_size, Q.device, past_end_block)

    O, L, M = NSA.selection_attention_wrapper(
        q_tensor=Q,
        k_tensor=K,
        v_tensor=V,
        block_indices_tensor=block_indices,
        block_counts_tensor=block_counts,
        cum_seqlen_q_tensor=cum_seqlen_q,
        cum_seqlen_k_tensor=cum_seqlen_kv,
        block_size=block_size,
        scale_softmax=cfg["scale_softmax"],
        o_dtype=cfg["dtype"],
        acc_dtype=cfg["acc_dtype"],
        is_causal=False,
    )

    # Reference logits in fp64, so the M and L bounds do not depend on the reference's own rounding.
    O_ref, L_ref, M_ref = run_ref_nsa_selection_attention(
        Q,
        K,
        V,
        torch.zeros_like(O, dtype=torch.float32),
        torch.zeros_like(L),
        torch.zeros_like(M),
        seq_lens,
        block_indices,
        block_counts,
        block_size,
        cfg["scale_softmax"],
        dtype=torch.float64,
    )
    torch.testing.assert_close(O.float(), O_ref.float(), atol=0.01, rtol=1e-05)
    torch.testing.assert_close(M, M_ref, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(L, L_ref, atol=1e-3, rtol=1e-3)


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
