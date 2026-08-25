# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import pytest
import torch

from cudnn.flex_attention import create_mask_plan, flex_attn_func
from cudnn.flex_attention.runtime.arch import SUPPORTED_ARCHES


def _current_arch() -> int:
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group(name="gpu_exclusive")
@pytest.mark.L1
def test_fixed_causal_forward_backward_matches_fp32_reference():
    if _current_arch() not in SUPPORTED_ARCHES:
        pytest.skip("Flex Attention requires SM90, SM100, or SM103")

    torch.manual_seed(2026)
    batch, seqlen, heads, head_dim = 1, 128, 2, 64
    shape = (batch, seqlen, heads, head_dim)
    q = (0.5 * torch.randn(shape, device="cuda", dtype=torch.bfloat16)).requires_grad_()
    k = (0.5 * torch.randn(shape, device="cuda", dtype=torch.bfloat16)).requires_grad_()
    v = (0.5 * torch.randn(shape, device="cuda", dtype=torch.bfloat16)).requires_grad_()
    dout = 0.5 * torch.randn_like(v)
    endpoints = torch.arange(1, seqlen + 1, device="cuda", dtype=torch.int32).view(1, 1, seqlen)

    plan = create_mask_plan(endpoints, q, k, v, build_backward=True)
    out, lse = flex_attn_func(q, k, v, mask_plan=plan, return_lse=True)
    out.backward(dout)

    q_ref = q.detach().float().requires_grad_()
    k_ref = k.detach().float().requires_grad_()
    v_ref = v.detach().float().requires_grad_()
    scores = torch.einsum("bqhd,bkhd->bhqk", q_ref, k_ref) / math.sqrt(head_dim)
    causal = torch.ones((seqlen, seqlen), dtype=torch.bool, device="cuda").tril()
    scores = scores.masked_fill(~causal, float("-inf"))
    lse_ref = torch.logsumexp(scores, dim=-1)
    probabilities = torch.softmax(scores, dim=-1)
    out_ref = torch.einsum("bhqk,bkhd->bqhd", probabilities, v_ref)
    out_ref.backward(dout.float())

    torch.testing.assert_close(out.float(), out_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(lse, lse_ref, atol=3e-2, rtol=3e-2)
    for actual, reference in ((q.grad, q_ref.grad), (k.grad, k_ref.grad), (v.grad, v_ref.grad)):
        torch.testing.assert_close(actual.float(), reference, atol=5e-2, rtol=5e-2)


@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group(name="gpu_exclusive")
@pytest.mark.L2
def test_varlen_gqa_forward_backward_matches_fp32_reference():
    if _current_arch() not in SUPPORTED_ARCHES:
        pytest.skip("Flex Attention requires SM90, SM100, or SM103")

    torch.manual_seed(2027)
    q_lengths = (96, 64)
    k_lengths = (80, 48)
    q_heads, kv_heads, head_dim = 4, 2, 64
    q = (
        0.5
        * torch.randn(
            (sum(q_lengths), q_heads, head_dim),
            device="cuda",
            dtype=torch.bfloat16,
        )
    ).requires_grad_()
    k = (
        0.5
        * torch.randn(
            (sum(k_lengths), kv_heads, head_dim),
            device="cuda",
            dtype=torch.bfloat16,
        )
    ).requires_grad_()
    v = (
        0.5
        * torch.randn(
            (sum(k_lengths), kv_heads, head_dim),
            device="cuda",
            dtype=torch.bfloat16,
        )
    ).requires_grad_()
    dout = 0.5 * torch.randn_like(q)
    cu_q = torch.tensor((0, q_lengths[0], sum(q_lengths)), device="cuda", dtype=torch.int32)
    cu_k = torch.tensor((0, k_lengths[0], sum(k_lengths)), device="cuda", dtype=torch.int32)
    endpoints = torch.cat(
        [torch.arange(1, q_length + 1, device="cuda", dtype=torch.int32).clamp(max=k_length) for q_length, k_length in zip(q_lengths, k_lengths)]
    ).view(1, 1, -1)

    plan = create_mask_plan(
        endpoints,
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max(q_lengths),
        max_seqlen_k=max(k_lengths),
        build_backward=True,
    )
    cu_q.add_(1)
    cu_k.add_(1)
    out, lse = flex_attn_func(q, k, v, mask_plan=plan, return_lse=True)
    out.backward(dout)

    q_ref = q.detach().float().requires_grad_()
    k_ref = k.detach().float().requires_grad_()
    v_ref = v.detach().float().requires_grad_()
    out_parts = []
    lse_parts = []
    q_offset = 0
    k_offset = 0
    group_size = q_heads // kv_heads
    for q_length, k_length in zip(q_lengths, k_lengths):
        q_sample = q_ref[q_offset : q_offset + q_length]
        k_sample = k_ref[k_offset : k_offset + k_length].repeat_interleave(group_size, dim=1)
        v_sample = v_ref[k_offset : k_offset + k_length].repeat_interleave(group_size, dim=1)
        scores = torch.einsum("qhd,khd->hqk", q_sample, k_sample) / math.sqrt(head_dim)
        causal = torch.arange(k_length, device="cuda")[None, :] <= torch.arange(q_length, device="cuda")[:, None]
        scores = scores.masked_fill(~causal, float("-inf"))
        lse_parts.append(torch.logsumexp(scores, dim=-1))
        out_parts.append(torch.einsum("hqk,khd->qhd", torch.softmax(scores, dim=-1), v_sample))
        q_offset += q_length
        k_offset += k_length
    out_ref = torch.cat(out_parts)
    lse_ref = torch.cat(lse_parts, dim=1)
    out_ref.backward(dout.float())

    torch.testing.assert_close(out.float(), out_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(lse, lse_ref, atol=3e-2, rtol=3e-2)
    for actual, reference in ((q.grad, q_ref.grad), (k.grad, k_ref.grad), (v.grad, v_ref.grad)):
        torch.testing.assert_close(actual.float(), reference, atol=5e-2, rtol=5e-2)
