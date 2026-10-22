# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import pytest
import torch

from cudnn.flex_attention import FlexAttentionBwd, FlexAttentionFwd, create_mask_plan, flex_attn_func
from cudnn.flex_attention.runtime.arch import SUPPORTED_ARCHES


def _current_arch() -> int:
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group(name="gpu_exclusive")
@pytest.mark.L1
def test_explicit_forward_backward_reuses_plan_executors_and_workspaces():
    if _current_arch() not in SUPPORTED_ARCHES:
        pytest.skip("Flex Attention requires SM90, SM100, or SM103")

    torch.manual_seed(2028)
    batch, seqlen, heads, head_dim = 1, 64, 2, 64
    shape = (batch, seqlen, heads, head_dim)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    endpoints = torch.arange(1, seqlen + 1, device="cuda", dtype=torch.int32).view(1, 1, seqlen)
    plan = create_mask_plan(endpoints, q, k, v, build_backward=True)

    out = torch.empty_like(q)
    lse = torch.empty((batch, heads, seqlen), dtype=torch.float32, device="cuda")
    api = FlexAttentionFwd(q, k, v, out, plan, lse)
    assert api.check_support()
    api.compile()
    workspace = torch.empty((api.workspace_size,), dtype=torch.uint8, device="cuda")
    api.execute(q, k, v, out, plan, lse, workspace=workspace)

    q_next = torch.randn_like(q)
    out_next = torch.empty_like(out)
    lse_next = torch.empty_like(lse)
    api.execute(q_next, k, v, out_next, plan, lse_next, workspace=workspace)

    q_ref = q_next.float().requires_grad_()
    k_ref = k.float().requires_grad_()
    v_ref = v.float().requires_grad_()
    scores = torch.einsum("bqhd,bkhd->bhqk", q_ref, k_ref) / math.sqrt(head_dim)
    causal = torch.ones((seqlen, seqlen), dtype=torch.bool, device="cuda").tril()
    scores = scores.masked_fill(~causal, float("-inf"))
    out_ref = torch.einsum("bhqk,bkhd->bqhd", torch.softmax(scores, dim=-1), v_ref)
    torch.testing.assert_close(out_next.float(), out_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(lse_next, torch.logsumexp(scores, dim=-1), atol=3e-2, rtol=3e-2)

    do = torch.randn_like(out_next)
    dq, dk, dv = torch.empty_like(q_next), torch.empty_like(k), torch.empty_like(v)
    bwd = FlexAttentionBwd(q_next, k, v, out_next, do, lse_next, dq, dk, dv, plan)
    assert bwd.check_support()
    bwd.compile()
    bwd_workspace = torch.empty((bwd.workspace_size,), dtype=torch.uint8, device="cuda")
    bwd.execute(q_next, k, v, out_next, do, lse_next, dq, dk, dv, plan, workspace=bwd_workspace)
    first_grads = tuple(grad.clone() for grad in (dq, dk, dv))

    for grad in (dq, dk, dv):
        grad.fill_(float("nan"))
    bwd.execute(q_next, k, v, out_next, do, lse_next, dq, dk, dv, plan, workspace=bwd_workspace)
    out_ref.backward(do.float())
    for actual, first, reference in zip((dq, dk, dv), first_grads, (q_ref.grad, k_ref.grad, v_ref.grad)):
        torch.testing.assert_close(actual, first, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(actual.float(), reference, atol=5e-2, rtol=5e-2)


@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group(name="gpu_exclusive")
@pytest.mark.L1
@pytest.mark.parametrize("head_dim,seqlen,deterministic", [(64, 128, False), (256, 79, False), (256, 80, False), (256, 81, False), (256, 129, True)])
def test_fixed_causal_forward_backward_matches_fp32_reference(head_dim, seqlen, deterministic):
    if _current_arch() not in SUPPORTED_ARCHES:
        pytest.skip("Flex Attention requires SM90, SM100, or SM103")
    if head_dim == 256 and _current_arch() != 90:
        pytest.skip("D256 tile-boundary regression targets SM90")

    torch.manual_seed(2026)
    batch, heads = 1, 2
    shape = (batch, seqlen, heads, head_dim)
    q = (0.5 * torch.randn(shape, device="cuda", dtype=torch.bfloat16)).requires_grad_()
    k = (0.5 * torch.randn(shape, device="cuda", dtype=torch.bfloat16)).requires_grad_()
    v = (0.5 * torch.randn(shape, device="cuda", dtype=torch.bfloat16)).requires_grad_()
    dout = 0.5 * torch.randn_like(v)
    endpoints = torch.arange(1, seqlen + 1, device="cuda", dtype=torch.int32).view(1, 1, seqlen)

    plan = create_mask_plan(endpoints, q, k, v, build_backward=True)
    out, lse = flex_attn_func(q, k, v, mask_plan=plan, return_lse=True, deterministic=deterministic)
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
@pytest.mark.parametrize("head_dim,deterministic", [(64, False), (256, False), (256, True)])
def test_varlen_gqa_forward_backward_matches_fp32_reference(head_dim, deterministic):
    if _current_arch() not in SUPPORTED_ARCHES:
        pytest.skip("Flex Attention requires SM90, SM100, or SM103")
    if head_dim == 256 and _current_arch() != 90:
        pytest.skip("D256 backward regression targets SM90")

    torch.manual_seed(2027)
    q_lengths = (96, 64)
    k_lengths = (80, 48)
    q_heads, kv_heads = 4, 2
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
    # The plan owns cloned prefixes; later caller mutations must not affect it.
    cu_q.zero_()
    cu_k.zero_()
    out, lse = flex_attn_func(q, k, v, mask_plan=plan, return_lse=True, deterministic=deterministic)
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


@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group(name="gpu_exclusive")
@pytest.mark.L2
@pytest.mark.parametrize("kv_heads", [2, 4])
def test_sm90_d256_plan_reuses_both_backward_modes_on_stream_and_graph(kv_heads):
    if _current_arch() != 90:
        pytest.skip("Dual backward payloads target SM90 D256")

    torch.manual_seed(2032)
    q = torch.randn((1, 129, 4, 256), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn((1, 81, kv_heads, 256), device="cuda", dtype=q.dtype, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    do = torch.randn_like(q)
    endpoints = torch.arange(1, 130, device="cuda", dtype=torch.int32).clamp(max=81).view(1, 1, -1)
    plan = create_mask_plan(endpoints, q, k, v, build_backward=True)
    out, lse = flex_attn_func(q, k, v, mask_plan=plan, return_lse=True)

    qr, kr, vr = [t.detach().float().requires_grad_() for t in (q, k, v)]
    group_size = 4 // kv_heads
    scores = torch.einsum("bqhd,bkhd->bhqk", qr, kr.repeat_interleave(group_size, dim=2)) / 16
    mask = torch.arange(81, device="cuda")[None, :] <= torch.arange(129, device="cuda")[:, None]
    scores = scores.masked_fill(~mask, float("-inf"))
    reference = torch.einsum("bhqk,bkhd->bqhd", scores.softmax(-1), vr.repeat_interleave(group_size, dim=2))
    reference_grads = torch.autograd.grad(reference, (qr, kr, vr), do.float())

    executors = []
    for deterministic in (False, True):
        grads = tuple(torch.empty_like(t) for t in (q, k, v))
        api = FlexAttentionBwd(q, k, v, out, do, lse, *grads, plan, deterministic=deterministic)
        assert api.check_support()
        api.compile()
        workspace = torch.empty(api.workspace_size, dtype=torch.uint8, device="cuda")
        executors.append((api, grads, workspace))

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    deterministic_first = None
    for _ in range(2):
        for api, grads, workspace in executors:
            api.execute(q, k, v, out, do, lse, *grads, plan, workspace=workspace, current_stream=stream)
            stream.synchronize()
            for actual, expected in zip(grads, reference_grads):
                torch.testing.assert_close(actual.float(), expected, atol=5e-2, rtol=5e-2)
            if api.deterministic:
                if deterministic_first is not None:
                    for actual, first in zip(grads, deterministic_first):
                        torch.testing.assert_close(actual, first, atol=0, rtol=0)
                deterministic_first = tuple(t.clone() for t in grads)

    # Replays must clear reduction workspaces and semaphores every time.
    graph = torch.cuda.CUDAGraph()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.graph(graph, stream=stream):
        for api, grads, workspace in executors:
            api.execute(q, k, v, out, do, lse, *grads, plan, workspace=workspace)
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        for api, grads, _ in executors:
            for actual, expected in zip(grads, reference_grads):
                torch.testing.assert_close(actual.float(), expected, atol=5e-2, rtol=5e-2)
            if api.deterministic:
                for actual, first in zip(grads, deterministic_first):
                    torch.testing.assert_close(actual, first, atol=0, rtol=0)

    do.mul_(0.5)
    graph.replay()
    torch.cuda.synchronize()
    for _, grads, _ in executors:
        for actual, expected in zip(grads, reference_grads):
            torch.testing.assert_close(actual.float(), expected * 0.5, atol=5e-2, rtol=5e-2)
    do.mul_(2)

    # Exercise allocating-wrapper caches in alternating mode order as well.
    for deterministic in (True, False, True, False):
        value = flex_attn_func(q, k, v, mask_plan=plan, deterministic=deterministic)
        grads = torch.autograd.grad(value, (q, k, v), do)
        for actual, expected in zip(grads, reference_grads):
            torch.testing.assert_close(actual.float(), expected, atol=5e-2, rtol=5e-2)

    # Compiled executors do not own runtime plans. Releasing a plan after an
    # asynchronous launch must keep the selected payload alive on its stream.
    for api, grads, workspace in executors:
        ephemeral = create_mask_plan(endpoints, q, k, v, build_backward=True)
        stream.wait_stream(torch.cuda.current_stream())
        api.execute(q, k, v, out, do, lse, *grads, ephemeral, workspace=workspace, current_stream=stream)
        del ephemeral
        replacement = create_mask_plan(torch.full_like(endpoints, 81), q, k, v, build_backward=True)
        stream.synchronize()
        for actual, expected in zip(grads, reference_grads):
            torch.testing.assert_close(actual.float(), expected, atol=5e-2, rtol=5e-2)
        del replacement
