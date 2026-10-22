# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Relax-style token groups, packed isolation, caller workspace and replay."""

import subprocess
import sys

import pytest
import torch

from cudnn.ops import qsa_indexed_forward, qsa_indexed_backward
from cudnn.deepseek_sparse_attention.qsa_training.api import _plan
from QSA.cutedsl.test_qsa import make_inputs, reference, assert_close, supported_gpu

pytestmark = pytest.mark.L0


@pytest.fixture(autouse=True)
def indexed_supported_device():
    # The shared bitmap-QSA fixture also admits SM107; this indexed API does not.
    if torch.cuda.is_available() and torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("Indexed QSA requires SM100 or SM103")


def indices_for(inputs):
    total, blocks = inputs["sel_blocks"].shape
    tokens = inputs["block_first"][None, :, None] + torch.arange(4, device="cuda", dtype=torch.int32)
    valid = (
        (inputs["sel_blocks"][:, :, None] != 0)
        & (tokens <= inputs["block_last"][None, :, None])
        & (tokens <= inputs["hi"][:, None, None])
        & (tokens >= inputs["lo"][:, None, None])
    )
    groups = torch.where(valid, tokens, -1)
    order = torch.rand((total, blocks), device="cuda").argsort(dim=1)
    return groups.gather(1, order[:, :, None].expand(-1, -1, 4)).reshape(total, 4 * blocks).contiguous()


def check_case(inputs, indices):
    out = qsa_indexed_forward(inputs["q"], inputs["k"], inputs["v"], indices, inputs["scale"])
    grad = inputs["q"].clone()
    grads = qsa_indexed_backward(grad, inputs["q"], inputs["k"], inputs["v"], out["out_fp32"], out["lse"], indices, inputs["scale"])
    for actual, expected in zip(out, reference(inputs)):
        assert_close(actual, expected)
    for actual, expected in zip(grads, reference(inputs, grad, out["out_fp32"], out["lse"])):
        assert_close(actual, expected)
    return out, grads


@pytest.mark.parametrize("lengths", [(1,), (3, 5, 7), (129, 133), (2049,)])
def test_indexed_packed_reference(supported_gpu, lengths):
    inputs = make_inputs(lengths, 12, 1, empty=True, all_selected=True)
    check_case(inputs, indices_for(inputs))


def test_indexed_changed_values_replay(supported_gpu):
    inputs = make_inputs((3, 65), 12, 1, all_selected=True)
    indices = indices_for(inputs)
    clean_indices = indices.clone()
    out, grads = check_case(inputs, indices)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=stream):
            captured = qsa_indexed_forward(inputs["q"], inputs["k"], inputs["v"], indices)
            captured_grad = qsa_indexed_backward(inputs["q"], inputs["q"], inputs["k"], inputs["v"], captured["out_fp32"], captured["lse"], indices)
        indices.fill_(-1)
        for tensor in (*captured, *captured_grad):
            tensor.fill_(torch.nan)
        graph.replay()
        torch.cuda.synchronize()
        for tensor in (captured["out"], captured["out_fp32"], *captured_grad):
            assert torch.count_nonzero(tensor) == 0
        assert torch.isneginf(captured["lse"]).all()
        indices.copy_(clean_indices)
        inputs["k"][3:] = torch.nan
        inputs["v"][3:] = torch.inf
        for tensor in (*captured, *captured_grad):
            tensor.fill_(torch.nan)
        graph.replay()
        torch.cuda.synchronize()
        for name in ("dq", "dk", "dv"):
            torch.testing.assert_close(captured_grad[name][:3], grads[name][:3], atol=0.04, rtol=0.04)
        torch.testing.assert_close(captured["out"][:3], out["out"][:3], atol=0.04, rtol=0.04)
    finally:
        graph.reset()


def test_indexed_execute_uses_only_caller_storage(supported_gpu, monkeypatch):
    inputs = make_inputs((3, 9), 12, 1, all_selected=True)
    q, k, v = (inputs[name] for name in ("q", "k", "v"))
    indices = indices_for(inputs)
    out, expected = check_case(inputs, indices)
    plan = _plan(q.shape[0], indices.shape[1], q.device, "bprop")
    outputs = tuple(torch.empty_like(expected[name]) for name in ("dq", "dk", "dv"))
    workspace = torch.empty(plan.scratch_workspace_bytes(), dtype=torch.uint8, device=q.device)

    def forbidden(*args, **kwargs):
        raise AssertionError("execute allocated, converted, or synchronized")

    with monkeypatch.context() as patch:
        for name in ("empty", "empty_like", "zeros", "zeros_like", "arange"):
            patch.setattr(torch, name, forbidden)
        patch.setattr(torch.cuda, "synchronize", forbidden)
        for name in ("cpu", "to", "item", "tolist", "numpy"):
            patch.setattr(torch.Tensor, name, forbidden)
        plan.execute(q, k, v, indices, outputs, 1 / 16, workspace, q, out["out_fp32"], out["lse"])
    for actual, ref in zip(outputs, expected):
        assert_close(actual, ref)
    with monkeypatch.context() as patch:
        patch.setattr(plan, "_metadata", forbidden)
        with pytest.raises(ValueError, match="workspace"):
            plan.execute(q, k, v, indices, outputs, 1 / 16, workspace[:0], q, out["out_fp32"], out["lse"])
        with pytest.raises(ValueError, match="overlap"):
            plan.execute(q, k, v, indices, (q, *outputs[1:]), 1 / 16, workspace, q, out["out_fp32"], out["lse"])
        with pytest.raises(ValueError, match="aligned"):
            bad = torch.empty(q.numel() + 1, device=q.device, dtype=q.dtype)[1:].view(q.shape)
            plan.execute(bad, k, v, indices, outputs, 1 / 16, workspace, q, out["out_fp32"], out["lse"])


def test_indexed_contract_rejection(supported_gpu):
    inputs = make_inputs((5,), 12, 1)
    q, k, v = (inputs[name] for name in ("q", "k", "v"))
    indices = indices_for(inputs)
    for invalid in (indices[:, :3].contiguous(), torch.empty((5, 2056), dtype=torch.int32, device=q.device)):
        with pytest.raises(NotImplementedError, match="width"):
            qsa_indexed_forward(q, k, v, invalid)
    with pytest.raises(ValueError, match="dtype"):
        qsa_indexed_forward(q, k, v, indices.long())
    with pytest.raises(ValueError, match="finite"):
        qsa_indexed_forward(q, k, v, indices, float("nan"))
    with pytest.raises(NotImplementedError, match="12Q"):
        qsa_indexed_forward(q[:, :3].contiguous(), k, v, indices)


def test_indexed_exports_are_lazy():
    code = """import sys
import cudnn
import cudnn.ops
assert "torch" not in sys.modules and "cutlass" not in sys.modules
from cudnn.ops import qsa_indexed_forward, qsa_indexed_backward
assert callable(qsa_indexed_forward) and callable(qsa_indexed_backward)
assert cudnn.DSA.qsa_indexed_backward is qsa_indexed_backward
"""
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("version", [None, ("nvidia-cutlass-dsl", "4.6.2")])
def test_indexed_dsl_gate_precedes_kernel_import(supported_gpu, monkeypatch, version):
    from cudnn.deepseek_sparse_attention.qsa_training.api import _QsaIndexedPlan
    from cudnn.frost import buffers

    plan = _QsaIndexedPlan(7, 4, torch.device("cuda", torch.cuda.current_device()), "bprop")
    monkeypatch.setattr(buffers, "cutedsl_state", lambda: (version is not None, version))
    with pytest.raises(ImportError, match="nvidia-cutlass-dsl.*4.7.0"):
        plan.compile()
