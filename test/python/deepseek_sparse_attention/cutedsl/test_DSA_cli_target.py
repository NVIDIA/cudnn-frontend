# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM90 teacher-target output matches the attention reference and preserves gradients."""

import math
from types import SimpleNamespace

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


def _cli_direct_case(dtype=torch.bfloat16, dim=512, deterministic=False):
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("CLI target output is SM90-only")
    from cudnn import DSA

    rows, kv_rows, topk, real_heads = 4, 256, 256, 32
    scale = 1 / math.sqrt(dim)
    q = (torch.randn(rows, 64, dim, device="cuda") / 5).to(dtype)
    q[:, real_heads:] = 0
    kv = (torch.randn(kv_rows, dim, device="cuda") / 5).to(dtype)
    sink = torch.full((64,), float("-inf"), device="cuda")
    indices = torch.arange(topk, device="cuda", dtype=torch.int32).expand(rows, -1).clone()
    lengths = torch.tensor([topk, 131, 1, 0], device="cuda", dtype=torch.int32)
    indices.masked_fill_(torch.arange(topk, device="cuda") >= lengths[:, None], -1)
    out, lse = ref_sparse_attention_forward(q, kv, sink, indices, topk_length=lengths, softmax_scale=scale)
    out = out[..., :512].contiguous()
    lse[:, real_heads:] = float("inf")
    dout = torch.randn_like(out)
    dout[:, real_heads:] = 0
    inputs = (q, kv, out, dout, lse, sink, indices)
    plan = DSA.SparseAttentionBackward(
        *inputs,
        sample_topk_length=lengths,
        softmax_scale=scale,
        deterministic=deterministic,
        return_cli_target=True,
        cli_target_head_count=real_heads,
    )
    assert plan.check_support()
    plan.compile()
    return SimpleNamespace(
        plan=plan,
        inputs=inputs,
        lengths=lengths,
        scale=scale,
        real_heads=real_heads,
        outputs=(torch.empty_like(q), torch.empty_like(kv), torch.empty_like(sink)),
        target=torch.full(indices.shape, 123.0, device="cuda", dtype=torch.float32),
        workspace=torch.empty(plan.scratch_workspace_bytes(), device="cuda", dtype=torch.uint8),
    )


def _execute_cli_direct(case, workspace):
    return case.plan.execute(
        *case.inputs,
        case.outputs[0],
        case.outputs[1],
        topk_length=case.lengths,
        workspace=workspace,
        d_sink=case.outputs[2],
        cli_target=case.target,
    )


def _cli_reference_target(case):
    q, kv, _, _, lse, sink, indices = case.inputs
    scores = torch.einsum("qhd,qkd->qhk", q[:, : case.real_heads].float(), kv[indices.clamp_min(0).long()].float()) * case.scale
    normalizer = torch.logaddexp(lse[:, : case.real_heads], sink[None, : case.real_heads])
    probabilities = torch.exp(scores - normalizer[..., None])
    valid = (indices >= 0) & (torch.arange(indices.size(1), device=indices.device) < case.lengths[:, None])
    probabilities.masked_fill_(~valid[:, None], 0)
    return probabilities.mean(1)


@pytest.mark.L0
@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("bad_workspace", ["missing", "short", "cpu", "strided"])
@torch_fork_set_rng(seed=101)
def test_cli_rejected_workspace_preserves_outputs(deterministic, bad_workspace):
    case = _cli_direct_case(deterministic=deterministic)
    nbytes = case.workspace.numel()
    invalid = {
        "missing": lambda: None,
        "short": lambda: case.workspace[:-1],
        "cpu": lambda: torch.empty(nbytes, dtype=torch.uint8),
        "strided": lambda: torch.empty(2 * nbytes, device="cuda", dtype=torch.uint8)[::2],
    }[bad_workspace]()
    buffers = (*case.outputs, case.target)
    for tensor in buffers:
        tensor.fill_(123)
    saved = tuple(tensor.clone() for tensor in buffers)
    with pytest.raises(ValueError, match="workspace"):
        _execute_cli_direct(case, invalid)
    torch.cuda.synchronize()
    for actual, expected in zip(buffers, saved):
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)), "Rejected workspace call modified a caller output"


@pytest.mark.L0
@torch_fork_set_rng(seed=102)
def test_cli_rejected_indexing_preserves_outputs(monkeypatch):
    from cudnn.deepseek_sparse_attention.sparse_attention_backward import _interface_sm90

    case = _cli_direct_case(deterministic=True)
    monkeypatch.setattr(_interface_sm90, "sm90_deterministic_relay_indexing_error", lambda *_: "forced CLI indexing rejection")
    buffers = (*case.outputs, case.target)
    for tensor in buffers:
        tensor.fill_(123)
    saved = tuple(tensor.clone() for tensor in buffers)
    with pytest.raises(ValueError, match="forced CLI indexing rejection"):
        _execute_cli_direct(case, case.workspace)
    torch.cuda.synchronize()
    for actual, expected in zip(buffers, saved):
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)), "Rejected indexing call modified a caller output"


@pytest.mark.L0
@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group(name="gpu_exclusive")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("dim", [512, 576], ids=["d512", "d576"])
@pytest.mark.parametrize("deterministic", [False, True])
@torch_fork_set_rng(seed=103)
def test_cli_direct_buffer_reuse_and_changed_input_graph(dtype, dim, deterministic):
    from cudnn import DSA

    case = _cli_direct_case(dtype=dtype, dim=dim, deterministic=deterministic)
    launch_stream = torch.cuda.Stream()
    launch_stream.wait_stream(torch.cuda.current_stream())
    buffers = (*case.outputs, case.target)
    addresses = tuple(tensor.data_ptr() for tensor in buffers)

    def snapshot():
        return tuple(tensor.clone() for tensor in buffers)

    def poison():
        case.workspace.fill_(0xA5)
        for tensor in buffers:
            tensor.fill_(123)

    def check(expected):
        assert tuple(tensor.data_ptr() for tensor in buffers) == addresses
        for index, (actual, reference) in enumerate(zip(buffers, expected)):
            if deterministic or index == 3:
                assert torch.equal(actual.view(torch.uint8), reference.view(torch.uint8)), f"Output {index} differs on replay"
            else:
                torch.testing.assert_close(actual, reference, atol=3e-2, rtol=3e-2)
        torch.testing.assert_close(case.target, _cli_reference_target(case), atol=2e-4, rtol=2e-2)

    with torch.cuda.stream(launch_stream):
        baseline = DSA.sparse_attention_backward_wrapper(*case.inputs, softmax_scale=case.scale, topk_length=case.lengths, deterministic=deterministic)
        poison()
        _execute_cli_direct(case, case.workspace)
        for name, actual in zip(("dq", "dkv", "d_sink"), case.outputs):
            torch.testing.assert_close(actual, baseline[name], atol=3e-2, rtol=3e-2)
        torch.testing.assert_close(case.target, _cli_reference_target(case), atol=2e-4, rtol=2e-2)
        first_target = case.target.clone()
        # Reuse the SAME buffers and change dO: teacher probabilities are dO-independent.
        case.inputs[3].mul_(2)
        poison()
        _execute_cli_direct(case, case.workspace)
        assert torch.equal(case.target.view(torch.uint8), first_target.view(torch.uint8))
        eager = snapshot()
    launch_stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=launch_stream):
            _execute_cli_direct(case, case.workspace)
        for _ in range(2):
            with torch.cuda.stream(launch_stream):
                poison()
                graph.replay()
                check(eager)
            launch_stream.synchronize()

        with torch.cuda.stream(launch_stream):
            case.inputs[4].add_(0.5)  # Change LSE in place, preserving captured addresses.
            poison()
            _execute_cli_direct(case, case.workspace)
            changed = snapshot()
            assert not torch.equal(case.target.view(torch.uint8), first_target.view(torch.uint8))
            poison()
            graph.replay()
            check(changed)
        launch_stream.synchronize()

        with torch.cuda.stream(launch_stream):
            case.lengths.zero_()  # Changed-to-empty replay must overwrite prior targets.
            poison()
            _execute_cli_direct(case, case.workspace)
            empty = snapshot()
            poison()
            graph.replay()
            check(empty)
            assert torch.count_nonzero(case.target) == 0
        launch_stream.synchronize()
    finally:
        launch_stream.synchronize()
        graph.reset()
