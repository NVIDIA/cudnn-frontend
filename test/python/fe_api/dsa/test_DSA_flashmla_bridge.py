# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract and B200 integration tests for the optional FlashMLA bridge."""

from __future__ import annotations

import math
import types

import pytest
import torch

from cudnn.deepseek_sparse_attention import flashmla_bridge as bridge
from fe_api.dsa.dsa_reference import (
    check_ref_dsa_sparse_attention_backward,
    check_ref_sparse_score_recompute,
    ref_sparse_attention_forward,
)

pytestmark = pytest.mark.L0


@pytest.mark.parametrize(
    "heads,head_dim,topk,launch_heads,launch_topk,tile",
    [
        (16, 576, 65, 64, 128, 64),
        (32, 512, 64, 64, 64, 64),
        (64, 576, 129, 64, 192, 64),
        (128, 512, 129, 128, 192, 64),
        (128, 512, 1281, 128, 1408, 128),
        (128, 576, 129, 128, 256, 128),
    ],
)
def test_flashmla_bridge_plan(heads, head_dim, topk, launch_heads, launch_topk, tile):
    plan = bridge.plan_flashmla_sparse_forward(heads, head_dim, topk)
    assert plan.launch_num_heads == launch_heads
    assert plan.launch_topk == launch_topk
    assert plan.topk_tile == tile
    assert plan.value_dim == 512


@pytest.mark.parametrize(
    "heads,head_dim,topk,error",
    [
        (8, 512, 64, "num_heads"),
        (64, 256, 64, "head_dim"),
        (64, 512, 0, "topk"),
        (True, 512, 64, "num_heads"),
    ],
)
def test_flashmla_bridge_plan_rejects_unsupported_contract(heads, head_dim, topk, error):
    with pytest.raises((TypeError, ValueError), match=error):
        bridge.plan_flashmla_sparse_forward(heads, head_dim, topk)


def test_flashmla_bridge_zero_copy_views_when_aligned():
    q = torch.randn(2, 64, 512, dtype=torch.bfloat16)
    kv = torch.randn(7, 512, dtype=torch.bfloat16)
    indices = torch.zeros(2, 64, dtype=torch.int32)
    sink = torch.randn(64, dtype=torch.float32)
    plan = bridge.plan_flashmla_sparse_forward(64, 512, 64)

    launch = bridge._prepare_flashmla_launch_inputs(q, kv, indices, sink, None, plan)

    assert launch.q is q
    assert launch.attn_sink is sink
    assert launch.kv.untyped_storage().data_ptr() == kv.untyped_storage().data_ptr()
    assert launch.indices.untyped_storage().data_ptr() == indices.untyped_storage().data_ptr()
    assert launch.kv.shape == (7, 1, 512)
    assert launch.indices.shape == (2, 1, 64)


def test_flashmla_bridge_pads_heads_and_only_the_topk_tail():
    q = torch.randn(2, 32, 576, dtype=torch.bfloat16)
    kv = torch.randn(7, 576, dtype=torch.bfloat16)
    indices = torch.arange(65, dtype=torch.int32).expand(2, -1).contiguous()
    sink = torch.randn(32, dtype=torch.float32)
    lengths = torch.tensor([65, 7], dtype=torch.int32)
    plan = bridge.plan_flashmla_sparse_forward(32, 576, 65)

    launch = bridge._prepare_flashmla_launch_inputs(q, kv, indices, sink, lengths, plan)

    assert launch.q.shape == (2, 64, 576)
    assert torch.equal(launch.q[:, :32], q)
    assert torch.count_nonzero(launch.q[:, 32:]) == 0
    assert launch.kv.untyped_storage().data_ptr() == kv.untyped_storage().data_ptr()
    assert launch.indices.shape == (2, 1, 128)
    assert torch.equal(launch.indices[:, 0, :65], indices)
    assert bool((launch.indices[:, 0, 65:] == -1).all())
    assert torch.equal(launch.attn_sink[:32], sink)
    assert torch.count_nonzero(launch.attn_sink[32:]) == 0
    assert launch.topk_length is lengths


def test_flashmla_dependency_is_lazy_and_missing_dependency_fails_closed(monkeypatch):
    calls = []

    def unavailable(name):
        calls.append(name)
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(bridge, "import_module", unavailable)
    bridge.plan_flashmla_sparse_forward(64, 512, 64)
    assert calls == []
    with pytest.raises(bridge.FlashMLABridgeUnavailableError, match="official deepseek-ai/FlashMLA"):
        bridge._resolve_flashmla_sparse_fwd()
    assert calls == ["flash_mla"]


def test_flashmla_dependency_without_sparse_entrypoint_fails_closed(monkeypatch):
    monkeypatch.setattr(bridge, "import_module", lambda _name: types.SimpleNamespace())
    with pytest.raises(bridge.FlashMLABridgeUnavailableError, match="flash_mla_sparse_fwd"):
        bridge._resolve_flashmla_sparse_fwd()


def test_flashmla_training_bridge_requires_sink_before_dependency_resolution(
    monkeypatch,
):
    monkeypatch.setattr(
        bridge,
        "_resolve_flashmla_sparse_fwd",
        lambda: pytest.fail("dependency must not be loaded"),
    )
    with pytest.raises(ValueError, match="attn_sink is required"):
        bridge.flashmla_cudnn_sparse_attention_wrapper(None, None, None, None)


def test_flashmla_training_metadata_normalizes_all_invalid_sentinels_without_lengths():
    indices = torch.tensor([[0, -7, 4, 5, 99]], dtype=torch.int32)
    safe_indices, safe_length = bridge._normalize_cudnn_sparse_metadata(indices, None, s_kv=5)
    assert torch.equal(safe_indices, torch.tensor([[0, -1, 4, -1, -1]], dtype=torch.int32))
    assert safe_length is None


def test_flashmla_training_metadata_masks_compacts_and_bounds_lengths():
    indices = torch.tensor(
        [
            [0, 99, -7, 4, 3],
            [4, 3, 2, 1, 0],
            [4, 3, 2, 1, 0],
        ],
        dtype=torch.int32,
    )
    lengths = torch.tensor([9, 2, -3], dtype=torch.int32)

    def compactify(normalized):
        packed = torch.full_like(normalized, -1)
        packed_lengths = torch.empty(normalized.shape[0], dtype=torch.int32)
        for row, values in enumerate(normalized):
            valid = values[values >= 0]
            packed[row, : valid.numel()] = valid
            packed_lengths[row] = valid.numel()
        return {"indices": packed, "topk_length": packed_lengths}

    safe_indices, safe_length = bridge._normalize_cudnn_sparse_metadata(
        indices,
        lengths,
        s_kv=5,
        _compactify=compactify,
    )
    assert torch.equal(
        safe_indices,
        torch.tensor(
            [[0, 4, 3, -1, -1], [4, 3, -1, -1, -1], [-1, -1, -1, -1, -1]],
            dtype=torch.int32,
        ),
    )
    assert torch.equal(safe_length, torch.tensor([3, 2, 0], dtype=torch.int32))


def _require_b200_flashmla():
    if not torch.cuda.is_available():
        pytest.skip("exact NVIDIA B200 and official FlashMLA required")
    if torch.cuda.get_device_capability() != (10, 0) or torch.cuda.get_device_name() != "NVIDIA B200":
        pytest.skip("exact NVIDIA B200 required")
    try:
        return bridge._resolve_flashmla_sparse_fwd()
    except bridge.FlashMLABridgeUnavailableError as exc:
        pytest.skip(str(exc))


@pytest.mark.parametrize(
    "heads,head_dim,topk",
    [
        pytest.param(64, 512, 64, id="native-h64-d512"),
        pytest.param(32, 576, 65, id="padded-h32-d576-k65"),
    ],
)
def test_flashmla_bridge_forward_matches_reference(heads, head_dim, topk):
    _require_b200_flashmla()
    torch.manual_seed(410)
    device = torch.device("cuda")
    s_q, s_kv = 4, 96
    scale = 1.0 / math.sqrt(head_dim)
    q = torch.randn(s_q, heads, head_dim, dtype=torch.bfloat16, device=device)
    kv = torch.randn(s_kv, head_dim, dtype=torch.bfloat16, device=device)
    sink = torch.randn(heads, dtype=torch.float32, device=device)
    indices = torch.stack([torch.randperm(s_kv, device=device)[:topk] for _ in range(s_q)]).to(torch.int32)
    indices[0, 0] = s_kv + 17
    indices[1, 1] = -9
    lengths = torch.tensor([topk, topk - 1, topk // 2, 1], dtype=torch.int32, device=device)

    actual = bridge.flashmla_sparse_forward_wrapper(
        q,
        kv,
        indices,
        softmax_scale=scale,
        attn_sink=sink,
        topk_length=lengths,
    )
    out_ref, lse_ref = ref_sparse_attention_forward(q, kv, sink, indices, topk_length=lengths, softmax_scale=scale)

    assert actual["output"].shape == (s_q, heads, 512)
    assert actual["max_logits"].shape == actual["lse"].shape == (s_q, heads)
    torch.testing.assert_close(actual["output"], out_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(actual["lse"], lse_ref, atol=2e-4, rtol=2e-4)


def test_flashmla_cudnn_training_and_score_recompute_match_references():
    _require_b200_flashmla()
    torch.manual_seed(411)
    device = torch.device("cuda")
    s_q, s_kv, heads, head_dim, topk = 4, 96, 32, 576, 65
    scale = 1.0 / math.sqrt(head_dim)
    q = torch.randn(s_q, heads, head_dim, dtype=torch.bfloat16, device=device, requires_grad=True)
    kv = torch.randn(s_kv, head_dim, dtype=torch.bfloat16, device=device, requires_grad=True)
    sink = torch.randn(heads, dtype=torch.float32, device=device, requires_grad=True)
    indices = torch.stack([torch.randperm(s_kv, device=device)[:topk] for _ in range(s_q)]).to(torch.int32)
    # Both forms are valid FlashMLA sentinels.  The bridge must prevent the
    # wider official contract from reaching cuDNN backward as an OOB index.
    indices[0, 0] = s_kv + 17
    indices[1, 1] = -9
    lengths = torch.tensor([topk + 11, topk - 1, topk // 2, 1], dtype=torch.int32, device=device)

    result = bridge.flashmla_cudnn_sparse_attention_wrapper(
        q,
        kv,
        indices,
        sink,
        softmax_scale=scale,
        topk_length=lengths,
    )
    dout = torch.randn_like(result["output"])
    result["output"].backward(dout)

    check_ref_dsa_sparse_attention_backward(
        q,
        kv,
        sink,
        indices,
        result["output"],
        dout,
        result["lse"],
        q.grad,
        kv.grad,
        sink.grad,
        softmax_scale=scale,
        topk_length=lengths,
        atol=5e-2,
        rtol=5e-2,
    )

    score = bridge.flashmla_sparse_score_recompute_wrapper(
        q.detach(),
        kv.detach(),
        result["lse"],
        indices,
        softmax_scale=scale,
        topk_length=lengths,
    )
    target, safe_indices = score["target"], score["indices"]
    bounded_lengths = lengths.clamp(min=0, max=topk)
    positions = torch.arange(topk, device=device).unsqueeze(0)
    invalid = (indices < 0) | (indices >= s_kv) | (positions >= bounded_lengths.unsqueeze(1))
    assert target.shape == indices.shape
    assert target.is_contiguous()
    assert bool((target[invalid] == 0).all())
    check_ref_sparse_score_recompute(
        "attention",
        q.detach().unsqueeze(0),
        result["lse"].unsqueeze(0),
        safe_indices.unsqueeze(0),
        target.unsqueeze(0),
        aux=kv.detach().unsqueeze(0),
        softmax_scale=scale,
        topk_length=None,
        atol=1e-3,
        rtol=1e-3,
    )


def test_flashmla_cudnn_deepseek_v32_h128_d576_k2048_contract():
    """Exercise the production DeepSeek V3.2 H/D/Top-K contract on B200."""

    _require_b200_flashmla()
    torch.manual_seed(412)
    device = torch.device("cuda")
    s_q, s_kv, heads, head_dim, topk = 1, 2304, 128, 576, 2048
    scale = 1.0 / math.sqrt(head_dim)
    q = torch.randn(s_q, heads, head_dim, dtype=torch.bfloat16, device=device, requires_grad=True)
    kv = torch.randn(s_kv, head_dim, dtype=torch.bfloat16, device=device, requires_grad=True)
    sink = torch.randn(heads, dtype=torch.float32, device=device, requires_grad=True)
    indices = torch.randperm(s_kv, device=device)[:topk].reshape(s_q, topk).to(torch.int32)
    lengths = torch.full((s_q,), topk, dtype=torch.int32, device=device)

    result = bridge.flashmla_cudnn_sparse_attention_wrapper(
        q,
        kv,
        indices,
        sink,
        softmax_scale=scale,
        topk_length=lengths,
    )
    dout = torch.randn_like(result["output"])
    result["output"].backward(dout)

    assert result["output"].shape == (s_q, heads, 512)
    assert result["lse"].shape == (s_q, heads)
    check_ref_dsa_sparse_attention_backward(
        q,
        kv,
        sink,
        indices,
        result["output"],
        dout,
        result["lse"],
        q.grad,
        kv.grad,
        sink.grad,
        softmax_scale=scale,
        topk_length=lengths,
        atol=5e-2,
        rtol=5e-2,
    )
