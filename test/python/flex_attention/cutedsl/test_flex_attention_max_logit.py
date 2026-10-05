# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

from cudnn.flex_attention import FlexAttentionFwd, create_mask_plan, flex_attn_func
from cudnn.flex_attention.runtime.arch import SUPPORTED_ARCHES

pytestmark = [pytest.mark.L1, pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]


@pytest.fixture(autouse=True)
def supported_arch():
    major, minor = torch.cuda.get_device_capability()
    if major * 10 + minor not in SUPPORTED_ARCHES:
        pytest.skip("Flex Attention requires SM90, SM100, or SM103")


def _case(*, varlen=False, head_dim=64, dtype=torch.bfloat16, pack_gqa=True, variant=None, requires_grad=False):
    torch.manual_seed(1041)
    q_lengths = (137, 71) if varlen else (137, 137)
    k_lengths = (271, 163) if varlen else (271, 271)
    hq, hkv = 4, 2
    dv = 128 if head_dim == 192 else head_dim
    q_shape = (sum(q_lengths), hq, head_dim) if varlen else (2, q_lengths[0], hq, head_dim)
    k_shape = (sum(k_lengths), hkv, head_dim) if varlen else (2, k_lengths[0], hkv, head_dim)
    q = torch.randn(q_shape, device="cuda", dtype=dtype, requires_grad=requires_grad)
    k = torch.randn(k_shape, device="cuda", dtype=dtype, requires_grad=requires_grad)
    v = torch.randn((*k_shape[:-1], dv), device="cuda", dtype=dtype, requires_grad=requires_grad)
    # Per-head masks: include empty rows, a wholly empty head, and multiple KV tiles.
    ends = torch.cat([torch.arange(n, device="cuda", dtype=torch.int32).mul(3).clamp(max=nk) for n, nk in zip(q_lengths, k_lengths)])
    endpoints = ends.view(1, 1, -1).repeat(1 if pack_gqa else hq, 1, 1)
    if not pack_gqa:
        endpoints[0].zero_()
        endpoints[1, :, ::3] = 0
    kwargs = {}
    if varlen:
        kwargs = dict(
            cu_seqlens_q=torch.tensor((0, q_lengths[0], sum(q_lengths)), device="cuda", dtype=torch.int32),
            cu_seqlens_k=torch.tensor((0, k_lengths[0], sum(k_lengths)), device="cuda", dtype=torch.int32),
            max_seqlen_q=max(q_lengths),
            max_seqlen_k=max(k_lengths),
        )
    plan = create_mask_plan(endpoints, q, k, v, pack_gqa=pack_gqa, build_backward=requires_grad, _fwd_variant=variant, **kwargs)
    return q, k, v, plan, endpoints, q_lengths, k_lengths


def _reference(q, k, endpoints, q_lengths, k_lengths, scale):
    q_flat = q.detach().reshape(-1, q.shape[-2], q.shape[-1]).double()
    k_flat = k.detach().reshape(-1, k.shape[-2], k.shape[-1]).double()
    result = torch.full((q.shape[-2],), -torch.inf, dtype=torch.float64, device=q.device)
    qo = ko = 0
    for nq, nk in zip(q_lengths, k_lengths):
        keys = k_flat[ko : ko + nk].repeat_interleave(q.shape[-2] // k.shape[-2], dim=1)
        scores = torch.einsum("qhd,khd->hqk", q_flat[qo : qo + nq], keys) * scale
        visible = torch.arange(nk, device=q.device)[None, None, :] < endpoints[:, 0, qo : qo + nq, None]
        scores.masked_fill_(~visible, -torch.inf)
        result = torch.maximum(result, scores.amax(dim=(1, 2)))
        qo += nq
        ko += nk
    return result.float()


@pytest.mark.parametrize("varlen", [False, True])
@pytest.mark.parametrize("pack_gqa", [False, True])
@pytest.mark.parametrize(
    "head_dim,variant,dtype",
    [
        (64, "qstage1_1cta", torch.float16),
        (128, "qstage2_1cta", torch.bfloat16),
        (192, "qstage1_2cta", torch.bfloat16),
        (256, "qstage1_1cta", torch.float16),
        (256, "qstage1_2cta", torch.bfloat16),
    ],
)
def test_max_logit_matches_masked_reference(varlen, pack_gqa, head_dim, variant, dtype):
    if torch.cuda.get_device_capability()[0] == 9:
        if variant != "qstage1_1cta":
            pytest.skip("SM100-specific forward variant")
        variant = None
    if head_dim == 256 and pack_gqa and torch.cuda.get_device_capability()[0] == 10:
        pytest.skip("SM100 hd256 requires unpacked GQA")
    case = _case(varlen=varlen, head_dim=head_dim, dtype=dtype, pack_gqa=pack_gqa, variant=variant)
    q, k, v, plan, endpoints, q_lengths, k_lengths = case
    for scale in (1 / math.sqrt(head_dim), 0.0):
        reference = _reference(q, k, endpoints, q_lengths, k_lengths, scale)
        for return_lse in (False, True):
            result = flex_attn_func(q, k, v, mask_plan=plan, softmax_scale=scale, return_lse=return_lse, return_max_logit=True)
            max_logit = result[-1]
            assert max_logit.shape == (q.shape[-2],)
            assert max_logit.dtype == torch.float32
            assert not max_logit.requires_grad
            torch.testing.assert_close(max_logit, reference, atol=1e-4, rtol=1e-4)
            if scale != 0:
                baseline = flex_attn_func(q, k, v, mask_plan=plan, softmax_scale=scale)
                torch.testing.assert_close(result[0], baseline, atol=5e-3, rtol=5e-3)


def test_max_logit_preallocated_stream_capture_and_reuse(monkeypatch):
    q, k, v, plan, endpoints, q_lengths, k_lengths = _case()
    out = torch.empty_like(q)
    maximum = torch.empty((q.shape[-2],), dtype=torch.float32, device=q.device)
    api = FlexAttentionFwd(q, k, v, out, plan, sample_max_logit=maximum)
    api.check_support()
    api.compile()
    workspace = torch.empty((api.workspace_size,), dtype=torch.uint8, device=q.device)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    def execute():
        api.execute(q, k, v, out, plan, workspace=workspace, current_stream=stream, max_logit_tensor=maximum)

    execute()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        execute()
    # A high stale value must not survive the reduction on a reused output buffer.
    maximum.fill_(float("inf"))
    q.mul_(0.25)
    graph.replay()
    torch.cuda.synchronize()
    reference = _reference(q, k, endpoints, q_lengths, k_lengths, 1 / math.sqrt(q.shape[-1]))
    torch.testing.assert_close(maximum, reference, atol=1e-4, rtol=1e-4)

    def unexpected(*args, **kwargs):
        raise AssertionError("execute must not allocate or compile")

    import cudnn.flex_attention.execution as execution

    with monkeypatch.context() as patch:
        for name in ("empty", "zeros", "empty_like", "zeros_like"):
            original = getattr(torch, name)

            def checked_allocation(*args, _original=original, **kwargs):
                result = _original(*args, **kwargs)
                if result.is_cuda:
                    unexpected()
                return result

            patch.setattr(torch, name, checked_allocation)
        patch.setattr(execution, "_compile_flex_attn_fwd", unexpected)
        torch.cuda.set_sync_debug_mode("error")
        try:
            execute()
        finally:
            torch.cuda.set_sync_debug_mode("default")
    stream.synchronize()
    torch.testing.assert_close(maximum, reference, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("bad_output", ["shape", "dtype", "stride"])
def test_max_logit_descriptor_validation(bad_output):
    q, k, v, plan, *_ = _case()
    maximum = torch.empty((4,), device=q.device, dtype=torch.float32)
    if bad_output == "shape":
        maximum = maximum.view(2, 2)
    elif bad_output == "dtype":
        maximum = maximum.half()
    else:
        maximum = torch.empty((8,), device=q.device, dtype=torch.float32)[::2]
    api = FlexAttentionFwd(q, k, v, torch.empty_like(q), plan, sample_max_logit=maximum)
    with pytest.raises(ValueError, match="max_logit"):
        api.check_support()


def test_max_logit_runtime_presence_and_scale_validation():
    q, k, v, plan, *_ = _case()
    out = torch.empty_like(q)
    maximum = torch.empty((4,), device=q.device, dtype=torch.float32)
    for enabled in (False, True):
        api = FlexAttentionFwd(q, k, v, out, plan, sample_max_logit=maximum if enabled else None)
        api.compile()
        workspace = torch.empty((api.workspace_size,), dtype=torch.uint8, device=q.device)
        with pytest.raises(ValueError, match="max_logit presence"):
            api.execute(q, k, v, out, plan, workspace=workspace, max_logit_tensor=None if enabled else maximum)
        if enabled:
            with pytest.raises(ValueError, match="non-negative"):
                api.execute(q, k, v, out, plan, workspace=workspace, max_logit_tensor=maximum, softmax_scale=-0.5)


@pytest.mark.parametrize("varlen", [False, True])
def test_max_logit_does_not_change_backward(varlen):
    q, k, v, plan, *_ = _case(varlen=varlen, requires_grad=True, pack_gqa=False)
    out, lse, maximum = flex_attn_func(q, k, v, mask_plan=plan, return_lse=True, return_max_logit=True)
    assert maximum.grad_fn is None and not maximum.requires_grad
    dout, dlse = torch.randn_like(out), torch.randn_like(lse)
    actual = torch.autograd.grad((out, lse), (q, k, v), (dout, dlse))
    out, lse = flex_attn_func(q, k, v, mask_plan=plan, return_lse=True)
    expected = torch.autograd.grad((out, lse), (q, k, v), (dout, dlse))
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("empty", [False, True])
def test_max_logit_signed_scores_and_fully_masked_attention(empty):
    q = torch.ones((2, 65, 4, 64), dtype=torch.bfloat16, device="cuda")
    k = torch.full((2, 257, 2, 64), -2.0, dtype=q.dtype, device=q.device)
    v = torch.ones_like(k)
    # Mask out a large positive score: the signed maximum must stay negative.
    k[:, -1].fill_(100.0)
    endpoints = torch.full((1, 1, 130), 0 if empty else 256, dtype=torch.int32, device=q.device)
    plan = create_mask_plan(endpoints, q, k, v, build_backward=False)
    for scale in (0.25, 0.0):
        out, maximum = flex_attn_func(q, k, v, mask_plan=plan, return_max_logit=True, softmax_scale=scale)
        expected = torch.full_like(maximum, -torch.inf if empty else -128.0 * scale)
        torch.testing.assert_close(maximum, expected, atol=0, rtol=0)
        if empty:
            torch.testing.assert_close(out, torch.zeros_like(out), atol=0, rtol=0)


@pytest.mark.parametrize("return_max_logit", [False, True])
def test_qstage2_empty_tile_clears_nan_tmem(return_max_logit):
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("Q-stage 2 is an SM100/SM103 pipeline")
    q = torch.ones((2, 137, 4, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.ones((2, 271, 2, 128), device=q.device, dtype=q.dtype)
    v = torch.ones_like(k)
    ends = torch.full((1, 1, 274), 271, device=q.device, dtype=torch.int32)
    options = dict(pack_gqa=False, build_backward=False, _fwd_variant="qstage2_1cta")
    full = create_mask_plan(ends, q, k, v, **options)
    empty = create_mask_plan(torch.zeros_like(ends), q, k, v, **options)
    kwargs = dict(return_max_logit=True) if return_max_logit else {}
    # TMEM survives CTA reuse. Multiplying its old NaNs by zero cannot clear it.
    v.fill_(float("nan"))
    flex_attn_func(q, k, v, mask_plan=full, **kwargs)
    v.fill_(1)
    result = flex_attn_func(q, k, v, mask_plan=empty, **kwargs)
    out = result[0] if return_max_logit else result
    torch.testing.assert_close(out, torch.zeros_like(out), atol=0, rtol=0)
    if return_max_logit:
        assert torch.isneginf(result[1]).all()
