# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the public IEEE FP32 Kimi context-parallel merge."""

import importlib
import inspect

import pytest
import torch

from cudnn.frost.buffers import cutedsl_requirement_error
from cudnn.linear_attention.merge import KimiCPMerge, kimi_cp_merge

requirement_error = cutedsl_requirement_error("Kimi CP merge tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

pytestmark = pytest.mark.L0


@pytest.fixture(autouse=True)
def supported_device():
    if not torch.cuda.is_available():
        pytest.skip("Kimi CP merge needs CUDA")
    from cudnn.api_base import TensorDesc

    sample = TensorDesc(torch.float32, (1, 1, 64, 64), (4096, 4096, 64, 1), (3, 2, 1, 0), torch.device("cuda", torch.cuda.current_device()))
    try:
        KimiCPMerge(sample, sample, sample, sample).check_support()
    except NotImplementedError as exc:
        pytest.skip(str(exc))


def inputs(p=8, h=2, v=64, k=64):
    # CPU draws make the corpus independent of device SM counts.
    rng = torch.Generator().manual_seed(918)
    hm = torch.randn(p, h, v, k, generator=rng) * 0.1
    mm = torch.randn(p, h, k, k, generator=rng) * 0.008 + torch.eye(k) * 0.94
    seed = torch.randn(1, h, v, k, generator=rng) * 0.1
    return tuple(x.cuda() for x in (hm, mm, seed))


def oracle(h, m, seed, reverse=False):
    # FP64 is independent of ambient Torch TF32 configuration.
    state = seed[0].double()
    product = torch.eye(m.shape[-1], device=m.device, dtype=torch.float64).expand(m.shape[1], -1, -1).clone()
    states = torch.empty_like(h, dtype=torch.float64)
    order = range(len(h) - 1, -1, -1) if reverse else range(len(h))
    for j in order:
        states[j] = state
        matrix = m[j].double().transpose(-1, -2) if reverse else m[j].double()
        state = state @ matrix + h[j].double()
        product = product @ matrix
    return states, state[None], product[None]


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("v,k", [(64, 64), (128, 64), (256, 64), (64, 128), (128, 128), (256, 128)])
def test_merge_matches_fp64_reference(reverse, v, k):
    h, m, seed = inputs(v=v, k=k)
    expected = oracle(h, m, seed, reverse)
    actual = kimi_cp_merge(h, m, seed, reverse=reverse, output_transition=True)
    for got, want in zip(actual, expected):
        error = (got.double() - want).norm() / want.norm().clamp_min(1e-12)
        assert error < 2e-6, error.item()
    first = len(h) - 1 if reverse else 0
    assert torch.equal(actual["initial_states"][first], seed[0])


def test_cp64_fp32_merge_stays_close_to_fp64():
    h, m, seed = inputs(p=64, h=1, v=128, k=128)
    expected = oracle(h, m, seed)[1]
    actual = kimi_cp_merge(h, m, seed)["final_state"]
    error = (actual.double() - expected).norm() / expected.norm().clamp_min(1e-12)
    assert error < 2e-6, error.item()


def test_public_api_exposes_only_fp32_merge():
    assert "precision" not in inspect.signature(KimiCPMerge).parameters
    assert "precision" not in inspect.signature(kimi_cp_merge).parameters
    h, m, seed = inputs(p=1)
    m.zero_()
    actual = kimi_cp_merge(h, m)
    assert torch.count_nonzero(actual["initial_states"]) == 0
    assert torch.equal(actual["final_state"][0], h[0])


def test_absent_seed_means_zero():
    h, m, _ = inputs(p=4)
    zero = torch.zeros((1, h.shape[1], h.shape[2], h.shape[3]), dtype=torch.float32, device=h.device)
    actual = kimi_cp_merge(h, m)
    expected = oracle(h, m, zero)
    torch.testing.assert_close(actual["initial_states"].double(), expected[0], rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(actual["final_state"].double(), expected[1], rtol=2e-6, atol=2e-6)


def test_prepared_merge_rebind_capture_no_alloc_or_sync():
    h, m, seed = inputs()
    x, tail = torch.empty_like(h), torch.empty_like(seed)
    plan = KimiCPMerge(h, m, x, tail, seed).compile()
    assert plan.scratch_workspace_bytes() == 0
    plan.execute(h, m, x, tail, seed)
    torch.cuda.synchronize()
    previous = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        count = torch.cuda.memory_stats()["allocation.all.allocated"]
        for _ in range(3):
            plan.execute(h, m, x, tail, seed)
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == count
    finally:
        torch.cuda.set_sync_debug_mode(previous)

    h2, m2, seed2 = h.clone(), m.clone(), seed.clone()
    x2, tail2 = torch.empty_like(x), torch.empty_like(tail)
    ambient = torch.cuda.current_stream()
    stream = torch.cuda.Stream()
    stream.wait_stream(ambient)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=stream):
            plan.execute(h2, m2, x2, tail2, seed2, current_stream=stream)
        ambient.wait_stream(stream)
        h2.mul_(1.5)
        graph.replay()
        torch.cuda.synchronize()
        expected = kimi_cp_merge(h2, m2, seed2)
        torch.testing.assert_close(tail2, expected["final_state"], rtol=0, atol=0)
        torch.testing.assert_close(x2, expected["initial_states"], rtol=0, atol=0)
    finally:
        graph.reset()


def test_merge_rejects_invalid_bindings_before_launch(monkeypatch):
    h, m, seed = inputs()
    x, tail = torch.empty_like(h), torch.empty_like(seed)
    plan = KimiCPMerge(h, m, x, tail, seed).compile()
    module = importlib.import_module("cudnn.linear_attention.frost.common.piece_chain")

    monkeypatch.setattr(module, "run_state_chain", lambda *a, **kw: pytest.fail("invalid binding reached launch"))
    for kwargs in (dict(h=h.half()), dict(m=m[:1]), dict(initial_states=h), dict(initial_state=None), dict(transition=tail)):
        tensors = dict(h=h, m=m, initial_states=x, final_state=tail, initial_state=seed)
        tensors.update(kwargs)
        with pytest.raises(ValueError):
            plan.execute(**tensors)


def test_prepared_merge_rejects_autograd_declaration_before_compile(monkeypatch):
    h, m, seed = inputs()
    h.requires_grad_(True)
    plan = KimiCPMerge(h, m, torch.empty_like(h), torch.empty_like(seed), seed)
    module = importlib.import_module("cudnn.linear_attention.frost.common.piece_chain")

    monkeypatch.setattr(module, "build_state_chain", lambda **kwargs: pytest.fail("autograd declaration reached compiler"))
    with pytest.raises(ValueError, match="no autograd"):
        plan.compile()


def test_requires_grad_input_is_allowed_with_grad_disabled():
    h, m, seed = inputs()
    h.requires_grad_(True)
    with torch.no_grad():
        actual = kimi_cp_merge(h, m, seed)
        expected = kimi_cp_merge(h.detach(), m, seed)
    torch.testing.assert_close(actual["initial_states"], expected["initial_states"], rtol=0, atol=0)
    torch.testing.assert_close(actual["final_state"], expected["final_state"], rtol=0, atol=0)


@pytest.mark.parametrize("bad", ["dtype", "stride", "shape", "autograd", "reverse", "alias"])
def test_wrapper_rejects_before_compilation(monkeypatch, bad):
    module = importlib.import_module(kimi_cp_merge.__module__)

    h, m, seed = inputs()
    kwargs = dict(initial_state=seed)
    if bad == "dtype":
        h = h.half()
    elif bad == "stride":
        m = m.transpose(-1, -2)
    elif bad == "shape":
        m = m[:1]
    elif bad == "autograd":
        h.requires_grad_(True)
    elif bad == "alias":
        m = h
    else:
        kwargs["reverse"] = "yes"
    monkeypatch.setattr(module, "_plan", lambda *a, **kw: pytest.fail("invalid input reached compiler"))
    with pytest.raises(ValueError):
        module.kimi_cp_merge(h, m, **kwargs)


def test_old_dsl_declines_through_real_version_checker(monkeypatch):
    buffers = importlib.import_module("cudnn.frost.buffers")

    h, m, seed = inputs()
    monkeypatch.setattr(buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "4.6.2")))
    with pytest.raises(NotImplementedError, match=r"requires nvidia-cutlass-dsl >= 4\.7\.0; found 4\.6\.2"):
        KimiCPMerge(h, m, torch.empty_like(h), torch.empty_like(seed), seed).check_support()


def test_sm107_without_sm107a_target_declines(monkeypatch):
    buffers = importlib.import_module("cudnn.frost.buffers")
    merge_module = importlib.import_module(KimiCPMerge.__module__)

    h, m, seed = inputs()
    monkeypatch.setattr(merge_module, "get_compute_capability", lambda *_: (10, 7))
    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: False)
    with pytest.raises(NotImplementedError, match="sm_107a"):
        KimiCPMerge(h, m, torch.empty_like(h), torch.empty_like(seed), seed).check_support()
