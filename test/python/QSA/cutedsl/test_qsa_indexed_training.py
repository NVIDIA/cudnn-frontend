# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Relax-style token groups, packed isolation, caller workspace and replay."""

import subprocess
import sys
import inspect

import pytest
import torch

from cudnn.ops import qsa_indexed_forward, qsa_indexed_backward
from cudnn.QSA.indexed.api import _plan
from QSA.cutedsl.test_qsa import make_inputs, reference, assert_close, supported_gpu

pytestmark = pytest.mark.L0


@pytest.fixture(autouse=True)
def indexed_supported_device():
    if torch.cuda.is_available() and torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("Indexed QSA requires SM100, SM103 or SM107")


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


@pytest.mark.parametrize("lengths", [(3, 5, 7), (129, 133)])
def test_indexed_six_query_heads(supported_gpu, lengths):
    """6Q/1KV shares the kernels; its backward batches four queries per compact-ID row (12Q/1KV batches two)."""
    inputs = make_inputs(lengths, 6, 1, empty=True, all_selected=True)
    indices = indices_for(inputs)
    check_case(inputs, indices)
    plan = _plan(inputs["q"].shape[0], indices.shape[1], inputs["q"].device, "bprop", 6)
    assert plan.paired and plan.nq == 4 and "flags" in {name for name, _, _ in plan.layouts}
    assert dict((name, shape) for name, shape, _ in plan.layouts)["ids"] == ((inputs["q"].shape[0] + 3) // 4, indices.shape[1])
    paired = _plan(inputs["q"].shape[0], indices.shape[1], inputs["q"].device, "bprop")
    assert paired.paired and paired.nq == 2
    out = qsa_indexed_forward(inputs["q"], inputs["k"], inputs["v"], indices)
    with pytest.raises(NotImplementedError, match="causal_topk requires 12Q/1KV"):
        qsa_indexed_backward(inputs["q"], inputs["q"], inputs["k"], inputs["v"], out["out_fp32"], out["lse"], indices, selection_mode="causal_topk")


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
assert not hasattr(cudnn.DSA, "qsa_indexed_backward")
"""
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.fixture
def canonical_inputs(supported_gpu):
    from cudnn.QSA.two_pass.api import _dsl_cuda_version
    from QSA.cutedsl.two_pass_reference import make_inputs as canonical

    if torch.cuda.get_device_capability() == (10, 7):
        pytest.skip("The canonical two-pass backward is currently qualified on SM100/SM103")
    if _dsl_cuda_version()[0] < 13:
        pytest.skip("12Q/1KV sparse two-pass requires the CUDA 13 DSL compiler")
    return canonical


def _canonical_call(inputs, **kwargs):
    return qsa_indexed_backward(
        inputs["dout"],
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["out"],
        inputs["lse"],
        inputs["topk_idxs"],
        selection_mode="causal_topk",
        cu_seqlens=inputs.get("cu_seqlens"),
        **kwargs,
    )


@pytest.mark.parametrize("total, bounds", [(4099, None), (32769, None), (32769, [0, 3, 3, 32769])])
def test_public_causal_topk_uses_two_pass_and_native_saved_state(canonical_inputs, monkeypatch, total, bounds):
    from QSA.cutedsl.two_pass_reference import reference_backward, assert_grads_close
    from cudnn.QSA.two_pass import api as two_pass

    inputs = canonical_inputs(total, 71, h_kv=1, cu_seqlens=bounds, width=2052, saved_output_dtype=torch.float32)
    # Use saved state from the actual public forward, not a private test-only path.
    out = qsa_indexed_forward(inputs["q"], inputs["k"], inputs["v"], inputs["topk_idxs"])
    inputs["out"], inputs["lse"] = out["out_fp32"], out["lse"]
    calls = []
    original = two_pass._backward

    def record(*args, **kwargs):
        calls.append((args[3].data_ptr(), args[5].data_ptr()))
        return original(*args, **kwargs)

    monkeypatch.setattr(two_pass, "_backward", record)
    actual = _canonical_call(inputs)
    expected = reference_backward(**{k: inputs[k] for k in ("q", "k", "v", "out", "dout", "lse", "topk_idxs")})
    for name, value, ref in zip(("dq", "dk", "dv"), actual, expected):
        assert_grads_close(name, value, ref)
    assert calls == [(inputs["out"].data_ptr(), inputs["lse"].data_ptr())]
    assert tuple(actual.keys()) == ("dq", "dk", "dv")


def test_public_causal_topk_graph_refreshes_packed_bounds(canonical_inputs):
    from QSA.cutedsl.two_pass_reference import reference_backward, assert_grads_close

    inputs = canonical_inputs(32773, 73, h_kv=1, cu_seqlens=[0, 3, 32773], width=2052, saved_output_dtype=torch.float32)
    _canonical_call(inputs)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=stream):
            outputs = _canonical_call(inputs)
        # First remove all sparse rows, then restore them with an empty document.
        for bounds in ([0, 16385, 32773], [0, 0, 32773]):
            fresh = canonical_inputs(32773, 79, h_kv=1, cu_seqlens=bounds, width=2052, saved_output_dtype=torch.float32)
            for name, tensor in inputs.items():
                if isinstance(tensor, torch.Tensor):
                    tensor.copy_(fresh[name])
            for tensor in outputs:
                tensor.fill_(torch.nan)
            graph.replay()
            torch.cuda.synchronize()
            expected = reference_backward(**{k: inputs[k] for k in ("q", "k", "v", "out", "dout", "lse", "topk_idxs")})
            for name, value, ref in zip(("dq", "dk", "dv"), outputs, expected):
                assert_grads_close(name, value, ref)
    finally:
        graph.reset()


def test_general_selection_never_dispatches_to_two_pass(supported_gpu, monkeypatch):
    from cudnn.QSA.two_pass import api as two_pass

    def forbidden(*args, **kwargs):
        raise AssertionError("general/empty selections reached the canonical backend")

    monkeypatch.setattr(two_pass, "_backward", forbidden)
    inputs = make_inputs((3, 9), 12, 1, empty=True)
    small = indices_for(inputs)
    indices = torch.full((12, 2052), -1, dtype=torch.int32, device="cuda")
    indices[:, : small.shape[1]] = small
    check_case(inputs, indices)


def test_causal_topk_options_are_explicit_and_append_only(supported_gpu):
    signature = inspect.signature(qsa_indexed_backward)
    assert list(signature.parameters)[:8] == ["grad_out", "q", "k", "v", "out_fp32", "lse", "indices", "scale"]
    assert all(p.kind == inspect.Parameter.KEYWORD_ONLY for p in list(signature.parameters.values())[8:])
    inputs = make_inputs((5,), 12, 1, all_selected=True)
    q, k, v = (inputs[name] for name in ("q", "k", "v"))
    indices = indices_for(inputs)
    out = qsa_indexed_forward(q, k, v, indices)
    args = (q, q, k, v, out["out_fp32"], out["lse"], indices)
    for kwargs, message in [
        ({"selection_mode": "two_pass"}, "selection_mode"),
        ({"selection_mode": None}, "selection_mode"),
        ({"cu_seqlens": torch.tensor([0, 5], device="cuda", dtype=torch.int32)}, "require selection_mode"),
        ({"max_seq_len": 5}, "require selection_mode"),
    ]:
        with pytest.raises(ValueError, match=message):
            qsa_indexed_backward(*args, **kwargs)
    with pytest.raises(NotImplementedError, match="width 2052"):
        qsa_indexed_backward(*args, selection_mode="causal_topk")
    padded = torch.full((5, 2052), -1, dtype=torch.int32, device="cuda")
    padded[:, : indices.shape[1]] = indices
    args = (*args[:-1], padded)
    for value in (True, 3.5, 0, 6):
        with pytest.raises(ValueError, match="host integer"):
            qsa_indexed_backward(*args, selection_mode="causal_topk", max_seq_len=value)
    with pytest.raises(ValueError, match="single sequence"):
        qsa_indexed_backward(*args, selection_mode="causal_topk", max_seq_len=3)
    with pytest.raises(NotImplementedError, match="scale"):
        qsa_indexed_backward(*args, scale=0.5, selection_mode="causal_topk")
    with pytest.raises(ValueError, match="dtype"):
        qsa_indexed_backward(*args, selection_mode="causal_topk", cu_seqlens=torch.tensor([0, 5], device="cuda", dtype=torch.int64))


@pytest.mark.parametrize("version", [None, ("nvidia-cutlass-dsl", "4.6.2")])
def test_indexed_dsl_gate_precedes_kernel_import(supported_gpu, monkeypatch, version):
    from cudnn.QSA.indexed.api import _QsaIndexedPlan
    from cudnn.frost import buffers

    plan = _QsaIndexedPlan(7, 4, torch.device("cuda", torch.cuda.current_device()), "bprop")
    monkeypatch.setattr(buffers, "cutedsl_state", lambda: (version is not None, version))
    with pytest.raises(ImportError, match="nvidia-cutlass-dsl.*4.7.0"):
        plan.compile()
