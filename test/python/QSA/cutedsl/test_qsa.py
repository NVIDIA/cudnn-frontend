# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Packed QSA API, numerics, and changed-input replay."""

import inspect
import subprocess
import sys

import pytest
import torch

from cudnn.ops import qsa_backward, qsa_forward


@pytest.fixture
def supported_gpu():
    from cudnn.frost.buffers import cutedsl_arch_requirement_error, cutedsl_requirement_error, cutedsl_state, cutedsl_too_old

    cc = torch.cuda.get_device_capability()
    if cc not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("QSA requires SM100, SM103 or SM107")
    installed, version = cutedsl_state()
    if not installed or cutedsl_too_old(version):
        pytest.skip(cutedsl_requirement_error("QSA") or "QSA requires nvidia-cutlass-dsl")
    reason = cutedsl_arch_requirement_error(cc)
    if reason:
        pytest.skip(reason)


def make_inputs(lengths, hq, hkv, *, empty=False, all_selected=False):
    rng = torch.Generator().manual_seed(123)
    total = sum(lengths)
    first, last, bases, origins = [], [], [], []
    offset, base = 0, 0
    for length in lengths:
        starts = list(range(offset, offset + length, 4))
        first.extend(starts)
        last.extend(min(i + 3, offset + length - 1) for i in starts)
        bases.extend([base] * length)
        origins.extend([offset] * length)
        offset += length
        base += len(starts)
    first, last, bases, origins = [torch.tensor(x, dtype=torch.int32) for x in (first, last, bases, origins)]
    hi = torch.arange(total, dtype=torch.int32)
    valid = (last[None] >= origins[:, None]) & (first[None] <= hi[:, None])
    sel = valid if all_selected else valid & (torch.rand((total, base), generator=rng) > 0.4)
    if empty:
        sel[::7] = False

    def normal(heads):
        return torch.randn((total, heads, 256), generator=rng).to(torch.bfloat16).cuda()

    return dict(
        q=normal(hq),
        k=normal(hkv),
        v=normal(hkv),
        sel_blocks=sel.byte().cuda(),
        lo=origins.cuda(),
        hi=hi.cuda(),
        blk_base=bases.cuda(),
        tok_base=origins.clone().cuda(),
        block_first=first.cuda(),
        block_last=last.cuda(),
        scale=1 / 16,
    )


@torch.no_grad()
def reference(inputs, grad_out=None, saved_out=None, saved_lse=None):
    q, k, v = (inputs[name].double() for name in ("q", "k", "v"))
    total, hq, _ = q.shape
    hkv = k.shape[1]
    keys = torch.arange(total, device=q.device)
    blocks = inputs["blk_base"][:, None] + (keys[None] - inputs["tok_base"][:, None]) // 4
    in_bounds = (blocks >= 0) & (blocks < inputs["sel_blocks"].shape[1])
    selected = torch.gather(inputs["sel_blocks"], 1, blocks.clamp(0, inputs["sel_blocks"].shape[1] - 1).long()) != 0
    valid = in_bounds & selected & (keys[None] >= inputs["lo"][:, None]) & (keys[None] <= inputs["hi"][:, None])
    out, lse = torch.empty_like(q), torch.empty((hq, total), dtype=torch.float64, device=q.device)
    dq, dk, dv = torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
    for head in range(hq):
        kv_head = head // (hq // hkv)
        scores = (q[:, head] @ k[:, kv_head].T) * inputs["scale"]
        scores.masked_fill_(~valid, -torch.inf)
        normalizer = torch.logsumexp(scores, dim=-1)
        safe_lse = torch.where(normalizer.isfinite(), normalizer, 0)
        p = torch.exp(scores - safe_lse[:, None])
        out[:, head] = p @ v[:, kv_head]
        lse[head] = normalizer
        if grad_out is not None:
            do = grad_out[:, head].double()
            stats = saved_lse[head].double()
            p = torch.exp(scores - torch.where(stats.isfinite(), stats, 0)[:, None])
            p = torch.where(valid & stats.isfinite()[:, None], p, 0)
            delta = (do * saved_out[:, head].double()).sum(-1)
            ds = p * (do @ v[:, kv_head].T - delta[:, None])
            dq[:, head] = (ds @ k[:, kv_head]) * inputs["scale"]
            dk[:, kv_head] += (ds.T @ q[:, head]) * inputs["scale"]
            dv[:, kv_head] += p.T @ do
    if grad_out is not None:
        return dq, dk, dv
    return out.to(inputs["q"].dtype), out, lse


def assert_close(actual, expected):
    assert torch.equal(torch.isneginf(actual), torch.isneginf(expected))
    finite = expected.isfinite()
    torch.testing.assert_close(actual[finite].double(), expected[finite].double(), atol=0.04, rtol=0.04)
    diff = (actual[finite].double() - expected[finite].double()).norm()
    # Absolute error governs analytically zero gradients.
    bound = max(0.025 * expected[finite].double().norm().item(), 1e-3)
    assert diff.item() <= bound


@pytest.mark.L0
def test_signature():
    tail = ["sel_blocks", "lo", "hi", "blk_base", "tok_base", "block_first", "block_last", "scale", "block_size"]
    for function, prefix in ((qsa_forward, ["q", "k", "v"]), (qsa_backward, ["grad_out", "q", "k", "v", "out_fp32", "lse"])):
        signature = inspect.signature(function)
        assert list(signature.parameters) == prefix + tail
        assert all(p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD for p in signature.parameters.values())
        assert signature.parameters["block_size"].default == 4
        assert all(p.default is inspect.Parameter.empty for name, p in signature.parameters.items() if name != "block_size")


@pytest.mark.L0
@pytest.mark.parametrize("first", ["cudnn.QSA.api", "cudnn.ops"])
def test_exports_in_fresh_process(first):
    probe = f"""
import importlib
import sys
import cudnn
import cudnn.QSA
import cudnn.ops
assert 'torch' not in sys.modules
assert 'cutlass' not in sys.modules
importlib.import_module({first!r})
from cudnn.QSA.api import qsa_forward, qsa_backward
assert cudnn.ops.qsa_forward is qsa_forward
assert cudnn.ops.qsa_backward is qsa_backward
assert callable(qsa_forward) and callable(qsa_backward)
"""
    run = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr


@pytest.mark.L0
@pytest.mark.parametrize("state,match", [((True, ("nvidia-cutlass-dsl", "4.6.2")), "4.6.2"), ((False, None), "QSA requires nvidia-cutlass-dsl")])
def test_dsl_gate_before_kernel_import(supported_gpu, monkeypatch, state, match):
    from cudnn.QSA import api
    from cudnn.frost import buffers

    inputs = make_inputs((1,), 12, 1)
    monkeypatch.setattr(buffers, "cutedsl_state", lambda: state)

    def forbidden(*args, **kwargs):
        pytest.fail("Unsupported DSL reached a kernel import")

    monkeypatch.setattr(api.importlib, "import_module", forbidden)
    with pytest.raises(ImportError, match=match):
        qsa_forward(**inputs)


def check_case(inputs):
    expected = reference(inputs)
    result = qsa_forward(**inputs)
    assert list(result.keys()) == ["out", "out_fp32", "lse"]
    for actual, ref in zip(result, expected):
        assert_close(actual, ref)
    assert result["out"].dtype == inputs["q"].dtype
    assert result["out_fp32"].dtype == result["lse"].dtype == torch.float32
    grad_out = inputs["q"].flip(0).contiguous()
    expected_grads = reference(inputs, grad_out, result["out_fp32"], result["lse"])
    grads = qsa_backward(grad_out, **inputs, out_fp32=result["out_fp32"], lse=result["lse"])
    assert list(grads.keys()) == ["dq", "dk", "dv"]
    for actual, ref in zip(grads, expected_grads):
        assert actual.dtype == torch.bfloat16
        assert_close(actual, ref)


@pytest.mark.L0
@pytest.mark.parametrize("heads", [(12, 1), (24, 2)])
@pytest.mark.parametrize("lengths", [(1,), (3, 5, 9), (61, 68)])
def test_packed_forward_backward(supported_gpu, heads, lengths):
    check_case(make_inputs(lengths, *heads, empty=True))


@pytest.mark.L1
def test_more_than_512_blocks(supported_gpu):
    check_case(make_inputs((2056,), 12, 1, all_selected=True))


@pytest.mark.L0
def test_single_token_and_reversed_bounds(supported_gpu):
    inputs = make_inputs((1,), 12, 1, all_selected=True)
    check_case(inputs)
    inputs = make_inputs((3, 8), 24, 2, all_selected=True)
    inputs["lo"].copy_(inputs["hi"] + 1)
    check_case(inputs)


@pytest.mark.L0
@pytest.mark.parametrize("scale", [0.0, -0.125, 0.1])
def test_runtime_scale(supported_gpu, scale):
    inputs = make_inputs((5, 12), 12, 1, empty=True)
    inputs["scale"] = scale
    check_case(inputs)


@pytest.mark.L0
@pytest.mark.parametrize("heads", [(12, 1), (24, 2)])
@pytest.mark.parametrize("poison", [(torch.nan, torch.inf), (torch.inf, torch.nan), (1e20, -1e20)], ids=["nan-inf", "inf-nan", "finite"])
def test_masked_kv_poison(supported_gpu, heads, poison):
    inputs = make_inputs((3, 5), *heads, all_selected=True)
    inputs["sel_blocks"][3:] = 0
    inputs["hi"][:3] = 0
    clean = qsa_forward(**inputs)
    grad_out = inputs["q"].clone()
    clean_grads = qsa_backward(grad_out, **inputs, out_fp32=clean["out_fp32"], lse=clean["lse"])
    saved_k, saved_v = inputs["k"].clone(), inputs["v"].clone()
    inputs["k"][1:] = poison[0]
    inputs["v"][1:] = poison[1]
    poisoned = qsa_forward(**inputs)
    grads = qsa_backward(grad_out, **inputs, out_fp32=poisoned["out_fp32"], lse=poisoned["lse"])
    for actual, expected in zip((*poisoned, *grads), (*clean, *clean_grads)):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for grad in (grads["dk"], grads["dv"]):
        assert torch.count_nonzero(grad[1:]).item() == 0

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=stream):
            captured = qsa_forward(**inputs)
            captured_grads = qsa_backward(grad_out, **inputs, out_fp32=captured["out_fp32"], lse=captured["lse"])
        for restore in (False, True):
            if restore:
                inputs["k"].copy_(saved_k)
                inputs["v"].copy_(saved_v)
            for tensor in (*captured, *captured_grads):
                tensor.fill_(torch.nan)
            graph.replay()
            torch.cuda.synchronize()
            for actual, expected in zip((*captured, *captured_grads), (*clean, *clean_grads)):
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    finally:
        graph.reset()


@pytest.mark.L0
def test_changed_selection_replay(supported_gpu):
    inputs = make_inputs((7, 22), 12, 1, empty=True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.stream(stream):
            for _ in range(2):
                warm = qsa_forward(**inputs)
                qsa_backward(inputs["q"], **inputs, out_fp32=warm["out_fp32"], lse=warm["lse"])
        stream.synchronize()
        with torch.cuda.graph(graph, stream=stream):
            result = qsa_forward(**inputs)
            grads = qsa_backward(inputs["q"], **inputs, out_fp32=result["out_fp32"], lse=result["lse"])
        inputs["sel_blocks"].zero_()
        graph.replay()
        torch.cuda.synchronize()
        for tensor in (result["out"], result["out_fp32"], *grads):
            assert torch.count_nonzero(tensor).item() == 0
        assert torch.isneginf(result["lse"]).all()
        inputs["sel_blocks"].fill_(1)
        graph.replay()
        torch.cuda.synchronize()
        for actual, ref in zip(result, reference(inputs)):
            assert_close(actual, ref)
        for actual, ref in zip(grads, reference(inputs, inputs["q"], result["out_fp32"], result["lse"])):
            assert_close(actual, ref)
    finally:
        graph.reset()


@pytest.mark.L0
def test_dynamic_extents_reuse_compile(supported_gpu, monkeypatch):
    first = make_inputs((17,), 12, 1)
    check_case(first)
    import cutlass.cute as cute

    def forbidden(*args, **kwargs):
        pytest.fail("Changed token/block extents triggered compilation")

    monkeypatch.setattr(cute, "compile", forbidden)
    check_case(make_inputs((13, 21), 12, 1, empty=True))


@pytest.mark.L0
def test_reject_before_launch(supported_gpu, monkeypatch):
    from cudnn.QSA import api

    inputs = make_inputs((17,), 12, 1)

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid bindings reached kernel selection")

    monkeypatch.setattr(api, "_implementation", forbidden)
    with pytest.raises(ValueError, match="sel_blocks"):
        qsa_forward(**dict(inputs, sel_blocks=inputs["sel_blocks"].int()))
    with pytest.raises(NotImplementedError, match="block_size"):
        qsa_forward(**inputs, block_size=8)
    with pytest.raises(ValueError, match="lse"):
        qsa_backward(inputs["q"], **inputs, out_fp32=inputs["q"].float(), lse=torch.empty((17, 12), device="cuda"))
