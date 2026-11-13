# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM90 teacher-target output matches the attention reference and preserves gradients."""

import math

import pytest
import torch
from deepseek_sparse_attention.cutedsl.dsa_reference import ref_sparse_attention_forward
from test_utils import torch_fork_set_rng


@pytest.mark.L0
@pytest.mark.parametrize("real_heads", [32, 64])
@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("with_sink", [False, True])
@torch_fork_set_rng(seed=17)
def test_cli_target_reference_and_gradient_parity(real_heads, deterministic, with_sink):
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("CLI target output is SM90-only")
    from cudnn import DSA

    rows, kv_rows, topk, dim = 4, 256, 256, 576
    scale = 1 / math.sqrt(dim)
    q = torch.randn(rows, 64, dim, device="cuda", dtype=torch.bfloat16)
    q[:, real_heads:] = 0
    kv = torch.randn(kv_rows, dim, device="cuda", dtype=q.dtype)
    sink = torch.full((64,), float("-inf"), device="cuda")
    if with_sink:
        sink[:real_heads] = torch.randn(real_heads, device="cuda")
    indices = torch.arange(topk, device="cuda", dtype=torch.int32).expand(rows, -1).clone()
    lengths = torch.tensor([topk, 131, 1, 0], dtype=torch.int32, device="cuda")
    indices.masked_fill_(torch.arange(topk, device="cuda") >= lengths[:, None], -1)
    out, lse = ref_sparse_attention_forward(q, kv, sink, indices, topk_length=lengths, softmax_scale=scale)
    out = out[..., :512].contiguous()
    lse[:, real_heads:] = float("inf")
    grad = torch.randn_like(out)
    grad[:, real_heads:] = 0
    inputs = (q, kv, out, grad, lse, sink, indices)
    options = dict(softmax_scale=scale, topk_length=lengths, deterministic=deterministic)
    baseline = DSA.sparse_attention_backward_wrapper(*inputs, **options)
    result = DSA.sparse_attention_backward_wrapper(
        *inputs,
        **options,
        return_cli_target=True,
        cli_target_head_count=real_heads,
    )
    safe_indices = indices.clamp_min(0).long()
    scores = torch.einsum("qhd,qkd->qhk", q[:, :real_heads].float(), kv[safe_indices].float()) * scale
    normalizer = torch.logaddexp(lse[:, :real_heads], sink[None, :real_heads])
    probabilities = torch.exp(scores - normalizer[..., None])
    probabilities.masked_fill_(indices[:, None] < 0, 0)
    expected = probabilities.mean(1)
    torch.testing.assert_close(result["cli_target"], expected, atol=2e-4, rtol=2e-2)
    for name in ("dq", "dkv", "d_sink"):
        torch.testing.assert_close(result[name], baseline[name], atol=3e-2, rtol=3e-2)
    assert result["cli_target"].dtype == torch.float32
    assert torch.count_nonzero(result["cli_target"][-1]) == 0
    # Repeated execution overwrites all target slots, including empty rows.
    repeated = DSA.sparse_attention_backward_wrapper(
        *inputs,
        **options,
        return_cli_target=True,
        cli_target_head_count=real_heads,
    )
    assert torch.equal(result["cli_target"].view(torch.int32), repeated["cli_target"].view(torch.int32))
    empty_lengths = torch.zeros_like(lengths)
    empty = DSA.sparse_attention_backward_wrapper(
        *inputs,
        softmax_scale=scale,
        topk_length=empty_lengths,
        deterministic=deterministic,
        return_cli_target=True,
        cli_target_head_count=real_heads,
    )
    assert torch.count_nonzero(empty["cli_target"]) == 0


@pytest.mark.L0
def test_cli_plan_rejects_missing_output_and_head_contract():
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("CLI target output is SM90-only")
    from cudnn import DSA

    q = torch.zeros(2, 64, 512, device="cuda", dtype=torch.bfloat16)
    kv = torch.zeros(256, 512, device="cuda", dtype=q.dtype)
    out, grad = torch.zeros_like(q), torch.zeros_like(q)
    lse = torch.zeros(2, 64, device="cuda")
    sink = torch.full((64,), float("-inf"), device="cuda")
    indices = torch.arange(256, device="cuda", dtype=torch.int32).expand(2, -1).contiguous()
    plan = DSA.SparseAttentionBackward(
        q,
        kv,
        out,
        grad,
        lse,
        sink,
        indices,
        return_cli_target=True,
        cli_target_head_count=32,
    )
    assert plan.check_support()
    with pytest.raises(ValueError, match="presence must match"):
        plan.execute(
            q,
            kv,
            out,
            grad,
            lse,
            sink,
            indices,
            dq=torch.empty_like(q),
            dkv=torch.empty_like(kv),
            d_sink=torch.empty_like(sink),
        )
    bad = DSA.SparseAttentionBackward(
        q,
        kv,
        out,
        grad,
        lse,
        sink,
        indices,
        return_cli_target=True,
        cli_target_head_count=65,
    )
    with pytest.raises((ValueError, RuntimeError), match="cli_target_head_count"):
        bad.check_support()
