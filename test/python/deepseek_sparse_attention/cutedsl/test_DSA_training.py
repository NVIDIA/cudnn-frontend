# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Numerical and training-contract coverage for native DSA composition."""

import pytest
import torch
from torch.utils.checkpoint import checkpoint
from cudnn import DSA

pytestmark = pytest.mark.L0


@pytest.fixture(autouse=True)
def supported():
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("native DSA training requires SM100")
    from cudnn.frost.buffers import cutedsl_requirement_error

    error = cutedsl_requirement_error("DSA training test")
    if error:
        pytest.skip(error)


def inputs(heads=16, dim=512, sink=True, lengths=True):
    torch.manual_seed(125)
    q = (torch.randn(4, heads, dim, device="cuda", dtype=torch.bfloat16) * 0.25).requires_grad_()
    kv = (torch.randn(19, dim, device="cuda", dtype=torch.bfloat16) * 0.25).requires_grad_()
    a = torch.randn(heads, device="cuda", requires_grad=True) if sink else None
    idx = torch.tensor([[1, -9, 4, 22, 1, 8, 3], [5, 6, 2, 1, 8, 3, 4], [-1] * 7, [2, 1, 0, 9, 8, 10, 12]], dtype=torch.int32, device="cuda")
    lens = torch.tensor([5, 2, -1, 90], dtype=torch.int32, device="cuda") if lengths else None
    return q, kv, idx, a, lens


def reference(q, kv, idx, sink, lengths):
    valid = (idx >= 0) & (idx < kv.shape[0])
    if lengths is not None:
        valid &= torch.arange(idx.shape[1], device=idx.device)[None] < lengths.clamp(0, idx.shape[1])[:, None]
    selected = kv[idx.clamp(0, kv.shape[0] - 1).long()].float()
    scores = torch.einsum("shd,skd->shk", q.float(), selected) * q.shape[-1] ** -0.5
    scores = scores.masked_fill(~valid[:, None], -float("inf"))
    sink_logits = q.new_full(q.shape[:2], -float("inf"), dtype=torch.float32) if sink is None else sink[None].expand(q.shape[:2])
    if sink is None:
        sink_logits = torch.where(valid.any(-1)[:, None], sink_logits, 0.0)
    probs = torch.softmax(torch.cat([scores, sink_logits[..., None]], dim=-1), dim=-1)[..., :-1]
    out = torch.einsum("shk,skd->shd", probs, selected[..., :512])
    lse = torch.logsumexp(scores, -1)
    lse = torch.where(valid.any(-1)[:, None], lse, float("inf"))
    target = torch.exp(scores - lse[..., None]).sum(1)
    target = target / target.sum(-1, keepdim=True).clamp_min(1e-12)
    return out, lse, target, valid


def assert_numerics(actual, expected, tol=0.035):
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), expected.float(), atol=tol * max(1.0, expected.detach().abs().max().item()), rtol=tol)


@pytest.mark.parametrize(
    "heads,dim,sink,lengths",
    [
        (16, 512, True, True),
        (16, 576, True, True),
        (32, 576, False, True),
        (32, 512, False, True),
        (64, 512, True, False),
        (64, 576, True, True),
        (128, 512, True, True),
    ],
)
def test_native_training_and_original_score_slots(heads, dim, sink, lengths):
    q, kv, idx, a, lens = inputs(heads, dim, sink, lengths)
    qr, kr = q.detach().float().requires_grad_(), kv.detach().float().requires_grad_()
    ar = a.detach().clone().requires_grad_() if a is not None else None
    out_ref, lse_ref, target_ref, valid = reference(qr, kr, idx, ar, lens)
    result = DSA.sparse_attention(q, kv, idx, a, topk_length=lens)
    assert not result["lse"].requires_grad and not result["max_logits"].requires_grad
    assert_numerics(result["out"], out_ref, 0.01)
    torch.testing.assert_close(result["lse"], lse_ref, atol=3e-4, rtol=3e-4)
    grad = torch.randn_like(result["out"])
    result["out"].backward(grad)
    out_ref.backward(grad.float())
    for actual, expected in ((q.grad, qr.grad), (kv.grad, kr.grad)):
        assert_numerics(actual, expected)
    if a is not None:
        assert_numerics(a.grad, ar.grad)
    score = DSA.sparse_attention_score_recompute(q.detach(), kv.detach(), result["lse"], idx, topk_length=lens)
    torch.testing.assert_close(score["target"], target_ref.detach(), atol=2e-3, rtol=2e-3)
    assert torch.equal(score["indices"], torch.where(valid, idx, -1))
    assert torch.equal(score["target"][~valid], torch.zeros_like(score["target"][~valid]))


@pytest.mark.parametrize("reentrant", [False, True])
def test_checkpoint_and_no_device_sync(reentrant):
    q, kv, idx, a, lens = inputs()

    def run(q, kv, a):
        return DSA.sparse_attention(q, kv, idx, a, topk_length=lens)["out"]

    # Prime JIT before testing the warmed execution path.
    run(q, kv, a).sum().backward()
    expected = [t.grad.clone() for t in (q, kv, a)]
    for t in (q, kv, a):
        t.grad = None
    torch.cuda.set_sync_debug_mode("error")
    try:
        checkpoint(run, q, kv, a, use_reentrant=reentrant).sum().backward()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    for tensor, ref in zip((q, kv, a), expected):
        assert_numerics(tensor.grad, ref, 0.01)


def test_inactive_nan_suffix_and_trusted_metadata():
    q, kv, idx, a, lens = inputs()
    with torch.no_grad():
        kv[18].fill_(float("nan"))
    idx[:, -1] = 18
    lens = torch.full((4,), 5, device="cuda", dtype=torch.int32)
    result = DSA.sparse_attention(q, kv, idx, a, topk_length=lens)
    assert torch.isfinite(result["out"]).all()
    from cudnn.deepseek_sparse_attention.training import _normalize_cudnn_sparse_metadata

    safe, count = _normalize_cudnn_sparse_metadata(idx, lens, kv.shape[0])
    fast = DSA.sparse_attention(q, kv, safe, a, topk_length=count, trusted_compact_metadata=True)
    torch.testing.assert_close(fast["out"], result["out"], atol=0, rtol=0)


def test_stream_and_changed_input_graph_replay():
    q, kv, idx, a, lens = inputs()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.no_grad():
        for _ in range(3):
            result = DSA.sparse_attention(q, kv, idx, a, topk_length=lens)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = DSA.sparse_attention(q, kv, idx, a, topk_length=lens)["out"]
        q.mul_(2)
        kv.mul_(0.5)
        graph.replay()
        expected = DSA.sparse_attention(q, kv, idx, a, topk_length=lens)["out"]
        torch.testing.assert_close(captured, expected)
    torch.cuda.current_stream().wait_stream(stream)


def test_unsupported_shape_and_device_scale_fail_before_launch():
    q, kv, idx, a, lens = inputs(128, 576)
    with pytest.raises(NotImplementedError, match="H128 D512"):
        DSA.sparse_attention(q, kv, idx, a)
    q, kv, idx, a, lens = inputs()
    with pytest.raises(TypeError, match="host real"):
        DSA.sparse_attention(q, kv, idx, a, softmax_scale=torch.tensor(1.0, device="cuda"))


def test_old_dsl_declines_before_any_kernel_launch(monkeypatch):
    from cudnn.frost import buffers

    q, kv, idx, a, lens = inputs()
    monkeypatch.setattr(buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "4.6.2")))
    with pytest.raises(RuntimeError, match="requires nvidia-cutlass-dsl >= 4.7.0; found 4.6.2"):
        DSA.sparse_attention(q, kv, idx, a)


def test_training_graph_replay_uses_changed_inputs_and_metadata():
    q, kv, idx, a, lens = inputs()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):

        def step():
            for tensor in (q, kv, a):
                if tensor.grad is not None:
                    tensor.grad.zero_()
            output = DSA.sparse_attention(q, kv, idx, a, topk_length=lens)["out"]
            output.sum().backward()
            return output

        for _ in range(3):
            step()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = step()
        with torch.no_grad():
            q.mul_(1.5)
            kv.mul_(0.7)
            idx[0, 0] = 3
            lens[1] = 1
        graph.replay()
        actual = [tensor.grad.clone() for tensor in (q, kv, a)]
        actual_output = captured.clone()
        expected_output = step()
        assert_numerics(actual_output, expected_output, 0.01)
        for grad, tensor in zip(actual, (q, kv, a)):
            assert_numerics(grad, tensor.grad, 0.01)
    torch.cuda.current_stream().wait_stream(stream)
