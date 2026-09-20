# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import pytest
import torch

from cudnn.flex_attention import FlexAttentionBwd, FlexAttentionFwd, create_mask_plan, flex_attn_func
from cudnn.flex_attention.runtime.arch import SUPPORTED_ARCHES


@pytest.fixture(autouse=True)
def _fp32_reference_matmul():
    allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32


def _precision_input(shape, profile):
    # Scale in FP32 before quantizing the actual kernel inputs.
    scale = 0.5 if profile == "outliers" else profile
    tensor = scale * torch.randn(shape, device="cuda", dtype=torch.float32)
    if profile == "outliers":
        # Sparse large and small finite values amid nontrivial random inputs.
        tensor.flatten()[:5] = torch.tensor((8.0, -8.0, 2**-10, -(2**-10), 0.0), device="cuda")
    return tensor.to(torch.bfloat16)


def _attention_reference(q, k, v, visible, *, upcast):
    # Match FA4's FP32 reference and reordered, same-dtype PyTorch baseline.
    dtype = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    group_size = q.shape[2] // k.shape[2]
    k = k.repeat_interleave(group_size, dim=2)
    v = v.repeat_interleave(group_size, dim=2)
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.einsum("bqhd,bkhd->bhqk", q * scale if upcast else q, k if upcast else k * scale)
    scores = scores.masked_fill(~visible, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    empty = ~visible.any(-1)
    probabilities = torch.softmax(scores.masked_fill(empty[None, None, :, None], 0), dim=-1).to(v.dtype)
    probabilities = probabilities.masked_fill(empty[None, None, :, None], 0)
    return torch.einsum("bhqk,bkhd->bqhd", probabilities, v).to(dtype), lse


def _assert_fa_accuracy(actual, reference, pytorch):
    # FA4 tests/cute/test_flash_attn.py at 301c551: no-softcap error budget.
    for tensor in (actual, reference, pytorch):
        assert torch.isfinite(tensor).all()
    rounding_atol = 2 * (reference + 0.3 - 0.3 - reference).abs().max().item()
    reference_error = (pytorch - reference).abs().max().item()
    actual_error = (actual - reference).abs().max().item()
    assert actual_error <= 2 * reference_error + rounding_atol, f"max error {actual_error} exceeds FA4 budget: 2 * {reference_error} + {rounding_atol}"


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

    causal = torch.ones((seqlen, seqlen), dtype=torch.bool, device="cuda").tril()
    reference_inputs = tuple(t.detach().requires_grad_() for t in (q_next, k, v))
    out_ref, lse_ref = _attention_reference(*reference_inputs, causal, upcast=True)
    out_pt, _ = _attention_reference(*reference_inputs, causal, upcast=False)
    _assert_fa_accuracy(out_next, out_ref, out_pt)
    torch.testing.assert_close(lse_next, lse_ref, atol=3e-2, rtol=3e-2)

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
    grads_ref = torch.autograd.grad(out_ref, reference_inputs, do)
    grads_pt = torch.autograd.grad(out_pt, reference_inputs, do)
    for actual, first, reference, pytorch in zip((dq, dk, dv), first_grads, grads_ref, grads_pt):
        torch.testing.assert_close(actual, first, atol=1e-2, rtol=1e-2)
        _assert_fa_accuracy(actual, reference, pytorch)


def _assert_deterministic_gradients(out, inputs, dout):
    for _ in range(10):
        repeated = torch.autograd.grad(out, inputs, dout, retain_graph=True)
        for tensor, again in zip(inputs, repeated):
            # Integer byte views also distinguish positive and negative zero.
            torch.testing.assert_close(tensor.grad.contiguous().view(torch.uint8), again.contiguous().view(torch.uint8), atol=0, rtol=0)


@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group(name="gpu_exclusive")
@pytest.mark.parametrize("input_profile", [0.0, 0.125, 0.5, 2.0, "outliers"], ids=["zero", "small", "nominal", "large", "outliers"])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k,heads,kv_heads,head_dim,pattern,deterministic",
    [
        pytest.param(128, 128, 2, 2, 64, "causal", False, marks=pytest.mark.L1, id="causal-original"),
        pytest.param(384, 768, 8, 8, 128, "full", True, marks=pytest.mark.L2, id="full-mha"),
        pytest.param(257, 513, 64, 8, 128, "full", True, marks=pytest.mark.L2, id="full-gqa-tail"),
        pytest.param(384, 768, 64, 8, 128, "causal", True, marks=pytest.mark.L2, id="causal-gqa"),
        pytest.param(1024, 2048, 8, 8, 128, "document", True, marks=pytest.mark.L2, id="document-mha"),
        pytest.param(257, 513, 64, 8, 128, "mixed_empty", True, marks=pytest.mark.L2, id="empty-gqa-tail"),
        pytest.param(1024, 2048, 64, 8, 128, "alternating", True, marks=pytest.mark.L2, id="noncontiguous-full-gqa"),
        pytest.param(257, 513, 8, 2, 16, "alternating", True, marks=pytest.mark.L2, id="partial-d16"),
        pytest.param(257, 513, 8, 2, 64, "alternating", True, marks=pytest.mark.L2, id="partial-d64"),
        pytest.param(257, 513, 8, 2, 192, "alternating", True, marks=pytest.mark.L2, id="partial-d192"),
        pytest.param(384, 768, 8, 2, 16, "full", True, marks=pytest.mark.L2, id="full-d16"),
        pytest.param(384, 768, 8, 2, 64, "full", True, marks=pytest.mark.L2, id="full-d64"),
        pytest.param(384, 768, 8, 2, 192, "full", True, marks=pytest.mark.L2, id="full-d192"),
        pytest.param(384, 768, 8, 2, 192, "causal", True, marks=pytest.mark.L2, id="causal-d192"),
    ],
)
def test_fixed_forward_backward_matches_fp32_reference(seqlen_q, seqlen_k, heads, kv_heads, head_dim, pattern, deterministic, input_profile):
    if _current_arch() not in SUPPORTED_ARCHES:
        pytest.skip("Flex Attention requires SM90, SM100, or SM103")
    if deterministic and _current_arch() not in (100, 103):
        pytest.skip("These deterministic pipeline cases target SM100/SM103")

    torch.manual_seed(2026)
    head_dim_v = min(head_dim, 128)
    q, k, v = [
        _precision_input((1, length, h, dim), input_profile).requires_grad_()
        for length, h, dim in ((seqlen_q, heads, head_dim), (seqlen_k, kv_heads, head_dim), (seqlen_k, kv_heads, head_dim_v))
    ]
    dout = _precision_input((1, seqlen_q, heads, head_dim_v), input_profile)
    positions = torch.arange(seqlen_q, device="cuda", dtype=torch.int32)
    lower = torch.zeros_like(positions)
    upper = torch.full_like(positions, seqlen_k)
    if pattern == "causal":
        upper = positions * seqlen_k // seqlen_q + 1
    if pattern in ("document", "mixed_empty"):
        lower[positions >= seqlen_q // 2] = seqlen_k // 2
        upper[positions < seqlen_q // 2] = seqlen_k // 2
    if pattern == "mixed_empty":
        upper = torch.minimum(upper, positions * seqlen_k // seqlen_q + 1)
        lower[:32] = 0
        upper[:32] = 0
    if pattern == "alternating":
        # Noncontiguous full Q blocks and partial boundaries in the same KV row.
        odd_block = (positions // 128) % 2 == 1
        lower = torch.where(odd_block, seqlen_k // 2 + 17, 0).to(torch.int32)
        upper = torch.where(odd_block, seqlen_k, seqlen_k * 3 // 4).to(torch.int32)
    endpoints = upper.view(1, 1, -1) if pattern == "causal" else torch.stack((torch.zeros_like(lower), lower, upper))[None].contiguous()

    plan = create_mask_plan(endpoints, q, k, v, build_backward=True)
    out, lse = flex_attn_func(q, k, v, mask_plan=plan, return_lse=True, deterministic=deterministic)
    out.backward(dout, retain_graph=deterministic)
    if deterministic:
        _assert_deterministic_gradients(out, (q, k, v), dout)

    visible = (torch.arange(seqlen_k, device="cuda")[None, :] >= lower[:, None]) & (torch.arange(seqlen_k, device="cuda")[None, :] < upper[:, None])
    empty = ~visible.any(-1)
    reference_inputs = tuple(t.detach().requires_grad_() for t in (q, k, v))
    out_ref, lse_ref = _attention_reference(*reference_inputs, visible, upcast=True)
    out_pt, _ = _attention_reference(*reference_inputs, visible, upcast=False)
    grads_ref = torch.autograd.grad(out_ref, reference_inputs, dout)
    grads_pt = torch.autograd.grad(out_pt, reference_inputs, dout)

    for tensor in (out, q.grad, k.grad, v.grad):
        assert torch.isfinite(tensor).all()
    _assert_fa_accuracy(out, out_ref, out_pt)
    assert torch.isfinite(lse[..., ~empty]).all()
    assert torch.isneginf(lse[..., empty]).all()
    torch.testing.assert_close(lse, lse_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(out[:, empty], torch.zeros_like(out[:, empty]), atol=0, rtol=0)
    torch.testing.assert_close(q.grad[:, empty], torch.zeros_like(q.grad[:, empty]), atol=0, rtol=0)
    for actual, reference, pytorch in zip((q.grad, k.grad, v.grad), grads_ref, grads_pt):
        _assert_fa_accuracy(actual, reference, pytorch)


@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group(name="gpu_exclusive")
@pytest.mark.L2
@pytest.mark.parametrize("input_profile", [0.0, 0.125, 0.5, 2.0, "outliers"], ids=["zero", "small", "nominal", "large", "outliers"])
@pytest.mark.parametrize(
    "q_lengths,k_lengths,head_dim,pattern,deterministic",
    [
        pytest.param((96, 64), (80, 48), 64, "causal", False, id="causal-original"),
        pytest.param((512, 257), (1024, 513), 128, "full", True, id="full-d128-tail"),
        pytest.param((512, 257), (1024, 513), 128, "causal", True, id="causal-d128-tail"),
    ],
)
def test_varlen_gqa_forward_backward_matches_fp32_reference(q_lengths, k_lengths, head_dim, pattern, deterministic, input_profile):
    if _current_arch() not in SUPPORTED_ARCHES:
        pytest.skip("Flex Attention requires SM90, SM100, or SM103")

    torch.manual_seed(2027)
    if deterministic and _current_arch() not in (100, 103):
        pytest.skip("These deterministic pipeline cases target SM100/SM103")
    q_heads, kv_heads = 4, 2
    q = _precision_input((sum(q_lengths), q_heads, head_dim), input_profile).requires_grad_()
    k = _precision_input((sum(k_lengths), kv_heads, head_dim), input_profile).requires_grad_()
    v = _precision_input((sum(k_lengths), kv_heads, head_dim), input_profile).requires_grad_()
    dout = _precision_input(q.shape, input_profile)
    cu_q = torch.tensor((0, q_lengths[0], sum(q_lengths)), device="cuda", dtype=torch.int32)
    cu_k = torch.tensor((0, k_lengths[0], sum(k_lengths)), device="cuda", dtype=torch.int32)
    endpoints = torch.cat(
        [
            (
                torch.arange(1, q_length + 1, device="cuda", dtype=torch.int32).clamp(max=k_length)
                if pattern == "causal"
                else torch.full((q_length,), k_length, device="cuda", dtype=torch.int32)
            )
            for q_length, k_length in zip(q_lengths, k_lengths)
        ]
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
    out, lse = flex_attn_func(q, k, v, mask_plan=plan, return_lse=True, deterministic=deterministic)
    out.backward(dout, retain_graph=deterministic)
    if deterministic:
        _assert_deterministic_gradients(out, (q, k, v), dout)

    reference_inputs = tuple(t.detach().requires_grad_() for t in (q, k, v))
    reference_outputs = []
    reference_gradients = []
    for upcast in (True, False):
        out_parts, lse_parts = [], []
        q_offset, k_offset = 0, 0
        for q_length, k_length in zip(q_lengths, k_lengths):
            q_sample = reference_inputs[0][None, q_offset : q_offset + q_length]
            k_sample = reference_inputs[1][None, k_offset : k_offset + k_length]
            v_sample = reference_inputs[2][None, k_offset : k_offset + k_length]
            visible = torch.ones((q_length, k_length), device="cuda", dtype=torch.bool)
            if pattern == "causal":
                visible = torch.arange(k_length, device="cuda")[None, :] <= torch.arange(q_length, device="cuda")[:, None]
            sample_out, sample_lse = _attention_reference(q_sample, k_sample, v_sample, visible, upcast=upcast)
            out_parts.append(sample_out.squeeze(0))
            lse_parts.append(sample_lse.squeeze(0))
            q_offset += q_length
            k_offset += k_length
        reference_out = torch.cat(out_parts)
        reference_outputs.append(reference_out)
        reference_gradients.append(torch.autograd.grad(reference_out, reference_inputs, dout))
        if upcast:
            lse_ref = torch.cat(lse_parts, dim=1)
    out_ref, out_pt = reference_outputs
    grads_ref, grads_pt = reference_gradients

    for tensor in (out, q.grad, k.grad, v.grad):
        assert torch.isfinite(tensor).all()
    _assert_fa_accuracy(out, out_ref, out_pt)
    assert torch.isfinite(lse).all()
    torch.testing.assert_close(lse, lse_ref, atol=3e-2, rtol=3e-2)
    for actual, reference, pytorch in zip((q.grad, k.grad, v.grad), grads_ref, grads_pt):
        _assert_fa_accuracy(actual, reference, pytorch)
