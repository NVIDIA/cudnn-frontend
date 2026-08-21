# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Correctness and contract tests for ``cudnn.gemm.ops.gelu_mlp``."""

import importlib
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from cudnn.gemm.ops import gelu_mlp


def _cc():
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


_SM100 = pytest.mark.skipif(
    not (torch.cuda.is_available() and _cc() == 100),
    reason="cuDNN GELU-MLP requires SM100",
)
_TOL = 2e-2


def _ref(x, w1, b1, w2, b2):
    hidden = F.gelu(F.linear(x, w1, b1), approximate="tanh")
    return F.linear(hidden, w2, b2)


def _rel_l2(actual, expected):
    return (actual.float() - expected.float()).norm().item() / max(expected.float().norm().item(), 1e-9)


def _inputs(*, requires=(False, False, False, False, False)):
    torch.manual_seed(0)
    M, H, intermediate, O = 128, 256, 512, 192
    base = (
        torch.randn(2, M, H, device="cuda", dtype=torch.bfloat16),
        torch.randn(intermediate, H, device="cuda", dtype=torch.bfloat16) * 0.02,
        torch.randn(intermediate, device="cuda", dtype=torch.bfloat16) * 0.02,
        torch.randn(O, intermediate, device="cuda", dtype=torch.bfloat16) * 0.02,
        torch.randn(O, device="cuda", dtype=torch.bfloat16) * 0.02,
    )
    return tuple(t.detach().requires_grad_(need) for t, need in zip(base, requires))


@pytest.mark.L0
@_SM100
def test_gelu_mlp_full_autograd_parity():
    required = (True,) * 5
    args = _inputs(requires=required)
    refs = tuple(t.detach().clone().requires_grad_(True) for t in args)
    dout = torch.randn(2, 128, 192, device="cuda", dtype=torch.bfloat16)

    output = gelu_mlp(*args)
    reference = _ref(*refs)
    output.backward(dout)
    reference.backward(dout)

    assert _rel_l2(output, reference) < _TOL
    for name, actual, expected in zip(("dx", "dw1", "db1", "dw2", "db2"), args, refs):
        assert _rel_l2(actual.grad, expected.grad) < _TOL, name


@pytest.mark.L0
@_SM100
def test_gelu_mlp_sum_loss_backward_parity():
    """Exercise the expanded zero-stride gradient produced by a sum loss."""
    args = _inputs(requires=(True,) * 5)
    refs = tuple(t.detach().clone().requires_grad_(True) for t in args)

    gelu_mlp(*args).sum().backward()
    _ref(*refs).sum().backward()

    for name, actual, expected in zip(("dx", "dw1", "db1", "dw2", "db2"), args, refs):
        assert _rel_l2(actual.grad, expected.grad) < _TOL, name


@pytest.mark.L0
@_SM100
def test_gelu_mlp_double_backward_fails_closed():
    args = _inputs(requires=(True,) * 5)
    output = gelu_mlp(*args)

    with pytest.raises(NotImplementedError, match="double backward is not supported"):
        torch.autograd.grad(output.sum(), args, create_graph=True)


@pytest.mark.L0
@_SM100
@pytest.mark.parametrize("tokens", [512, 4096], ids=["text-512", "image-4096"])
def test_gelu_mlp_qwen_image_sequence_length_parity(tokens):
    """Cover both token counts used by the Qwen-Image image/text FFNs."""
    torch.manual_seed(11)
    H, intermediate = 128, 256
    x = torch.randn(1, tokens, H, device="cuda", dtype=torch.bfloat16)
    w1 = torch.randn(intermediate, H, device="cuda", dtype=torch.bfloat16) * 0.02
    b1 = torch.randn(intermediate, device="cuda", dtype=torch.bfloat16) * 0.02
    w2 = torch.randn(H, intermediate, device="cuda", dtype=torch.bfloat16) * 0.02
    b2 = torch.randn(H, device="cuda", dtype=torch.bfloat16) * 0.02

    with torch.no_grad():
        output = gelu_mlp(x, w1, b1, w2, b2)
        reference = _ref(x, w1, b1, w2, b2)

    assert _rel_l2(output, reference) < _TOL


@pytest.mark.L0
@_SM100
@pytest.mark.parametrize(
    "required,expected_saved",
    [
        ((True, False, False, False, False), ("w1", "w2", "pre_activation")),
        ((False, True, False, False, False), ("x2", "w2", "pre_activation")),
        ((False, False, True, False, False), ("w2", "pre_activation")),
        ((False, False, False, True, False), ("hidden",)),
        ((False, False, False, False, True), ()),
        ((False, True, True, True, True), ("x2", "w2", "pre_activation", "hidden")),
    ],
    ids=["x-only", "w1-only", "b1-only", "w2-only", "b2-only", "weights-and-biases"],
)
def test_gelu_mlp_partial_grad(required, expected_saved):
    args = _inputs(requires=required)
    refs = tuple(t.detach().clone().requires_grad_(need) for t, need in zip(args, required))
    dout = torch.randn(2, 128, 192, device="cuda", dtype=torch.bfloat16)

    output = gelu_mlp(*args)
    reference = _ref(*refs)
    assert tuple(output.grad_fn.saved_names) == expected_saved
    output.backward(dout)
    reference.backward(dout)

    assert _rel_l2(output, reference) < _TOL
    for name, actual, expected, need in zip(("dx", "dw1", "db1", "dw2", "db2"), args, refs, required):
        if need:
            assert _rel_l2(actual.grad, expected.grad) < _TOL, name
        else:
            assert actual.grad is None, name


@pytest.mark.L0
@_SM100
@pytest.mark.parametrize(
    "grad_mode",
    [torch.no_grad, torch.inference_mode],
    ids=["no-grad", "inference-mode"],
)
def test_gelu_mlp_inference_omits_pre_activation(monkeypatch, grad_mode):
    module = importlib.import_module("cudnn.gemm.ops._gelu_mlp")
    original = module._linear_bias
    observed = []

    def record(*args, **kwargs):
        observed.append((kwargs["gelu"], kwargs.get("save_pre_activation", False)))
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_linear_bias", record)
    args = _inputs(requires=(True,) * 5)
    with grad_mode():
        output = module.gelu_mlp(*args)
        reference = _ref(*args)

    assert not output.requires_grad
    assert observed == [(True, False), (False, False)]
    assert _rel_l2(output, reference) < _TOL


@pytest.mark.L0
@_SM100
@pytest.mark.parametrize("use_reentrant", [True, False], ids=["reentrant", "non-reentrant"])
def test_gelu_mlp_checkpoint_parity(use_reentrant):
    args = _inputs(requires=(True,) * 5)
    refs = tuple(t.detach().clone().requires_grad_(True) for t in args)
    dout = torch.randn(2, 128, 192, device="cuda", dtype=torch.bfloat16)

    output = torch.utils.checkpoint.checkpoint(gelu_mlp, *args, use_reentrant=use_reentrant)
    reference = _ref(*refs)
    output.backward(dout)
    reference.backward(dout)

    assert _rel_l2(output, reference) < _TOL
    for actual, expected in zip(args, refs):
        assert _rel_l2(actual.grad, expected.grad) < _TOL


@pytest.mark.L0
@_SM100
def test_gelu_mlp_cache_is_stream_local():
    module = importlib.import_module("cudnn.gemm.ops._gelu_mlp")
    module._HANDLES.clear()
    module._LINEAR_CACHE.clear()
    module._MM_CACHE.clear()
    module._DGELU_CACHE.clear()
    args = _inputs()

    gelu_mlp(*args)
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        gelu_mlp(*args)
    torch.cuda.current_stream().wait_stream(side_stream)

    streams = {key[-1] for key in module._LINEAR_CACHE}
    assert len(module._HANDLES) == 2
    assert len(streams) == 2
    assert len(module._LINEAR_CACHE) == 4  # two stages on each stream


@pytest.mark.L0
@_SM100
@pytest.mark.parametrize("save_pre_activation", [False, True], ids=["inference", "training"])
def test_gelu_mlp_fc1_is_one_fused_kernel(save_pre_activation):
    """The first Linear, bias, bf16 boundary, and tanh-GELU stay one launch."""
    module = importlib.import_module("cudnn.gemm.ops._gelu_mlp")
    x, w1, b1, _, _ = _inputs()
    x2 = x.reshape(-1, x.shape[-1])
    module._linear_bias(
        x2,
        w1,
        b1,
        gelu=True,
        save_pre_activation=save_pre_activation,
    )
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as profiler:
        module._linear_bias(
            x2,
            w1,
            b1,
            gelu=True,
            save_pre_activation=save_pre_activation,
        )
        torch.cuda.synchronize()

    launches = sum(event.count for event in profiler.key_averages() if event.self_device_time_total > 0)
    stream = torch.cuda.current_stream().cuda_stream
    matching = [
        entry
        for key, entry in module._LINEAR_CACHE.items()
        if key[-1] == stream and key[0] is True and key[1] is save_pre_activation and key[2] == (1, x2.shape[0], x2.shape[1])
    ]
    assert len(matching) == 1
    graph, *_, best, _workspace = matching[0]
    plan_name = graph.get_plan_name_at_index(best)
    assert launches == 1, f"expected one FC1 fusion launch from {plan_name}, saw {launches}"


@pytest.mark.L0
@_SM100
def test_gelu_mlp_inference_forward_is_exactly_two_kernels():
    """The complete forward is fused FC1+bias+GELU plus FC2+bias."""
    args = _inputs()
    with torch.no_grad():
        gelu_mlp(*args)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as profiler:
        with torch.no_grad():
            gelu_mlp(*args)
        torch.cuda.synchronize()

    launches = sum(event.count for event in profiler.key_averages() if event.self_device_time_total > 0)
    assert launches == 2, f"expected exactly two GELU-MLP forward launches, saw {launches}"


@pytest.mark.L0
@_SM100
def test_gelu_mlp_dhidden_dgelu_is_one_fused_kernel():
    """Backward must not materialize ``dhidden`` between GEMM and dGELU."""
    module = importlib.import_module("cudnn.gemm.ops._gelu_mlp")
    x, w1, b1, w2, _ = _inputs()
    x2 = x.reshape(-1, x.shape[-1])
    _, pre_activation = module._linear_bias(
        x2,
        w1,
        b1,
        gelu=True,
        save_pre_activation=True,
    )
    dout2 = torch.randn(x2.shape[0], w2.shape[0], device="cuda", dtype=torch.bfloat16)
    module._linear_dgelu(dout2, w2, pre_activation)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as profiler:
        module._linear_dgelu(dout2, w2, pre_activation)
        torch.cuda.synchronize()

    launches = sum(event.count for event in profiler.key_averages() if event.self_device_time_total > 0)
    stream = torch.cuda.current_stream().cuda_stream
    matching = [entry for key, entry in module._DGELU_CACHE.items() if key[-1] == stream and key[0] == (1, x2.shape[0], w2.shape[0])]
    assert len(matching) == 1
    graph, *_, best, _workspace = matching[0]
    plan_name = graph.get_plan_name_at_index(best)
    assert launches == 1, f"expected one dLinear+dGELU launch from {plan_name}, saw {launches}"


@pytest.mark.L0
@_SM100
def test_gelu_mlp_backward_runtime_failure_is_not_hidden(monkeypatch):
    module = importlib.import_module("cudnn.gemm.ops._gelu_mlp")

    def fail(*_args, **_kwargs):
        raise RuntimeError("synthetic fused backward launch failure")

    monkeypatch.setattr(module, "_linear_dgelu", fail)
    args = _inputs(requires=(True, False, False, False, False))
    output = module.gelu_mlp(*args)
    with pytest.raises(RuntimeError, match="synthetic fused backward launch failure"):
        output.sum().backward()


@pytest.mark.L0
def test_gelu_mlp_is_lazy_public_export():
    assert callable(gelu_mlp)
    import cudnn.gemm

    assert cudnn.gemm.gelu_mlp is gelu_mlp


@pytest.mark.L0
def test_gelu_mlp_public_export_survives_internal_module_first_import():
    code = """
import importlib
import cudnn.gemm.ops as ops
importlib.import_module('cudnn.gemm.ops._gelu_mlp')
assert callable(ops.gelu_mlp)
import cudnn.gemm
assert cudnn.gemm.gelu_mlp is ops.gelu_mlp
"""
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize(
    "mutate,error,match",
    [
        (lambda xs: xs.__setitem__(0, xs[0].float()), TypeError, "x must be bfloat16"),
    ],
    ids=["dtype"],
)
@pytest.mark.L0
def test_gelu_mlp_cpu_validation(mutate, error, match):
    tensors = [
        torch.empty(2, 3, 4, dtype=torch.bfloat16),
        torch.empty(8, 4, dtype=torch.bfloat16),
        torch.empty(8, dtype=torch.bfloat16),
        torch.empty(6, 8, dtype=torch.bfloat16),
        torch.empty(6, dtype=torch.bfloat16),
    ]
    mutate(tensors)
    with pytest.raises(error, match=match):
        gelu_mlp(*tensors)


@pytest.mark.L0
def test_gelu_mlp_rejects_cpu_operands():
    tensors = [
        torch.empty(2, 3, 4, dtype=torch.bfloat16),
        torch.empty(8, 4, dtype=torch.bfloat16),
        torch.empty(8, dtype=torch.bfloat16),
        torch.empty(6, 8, dtype=torch.bfloat16),
        torch.empty(6, dtype=torch.bfloat16),
    ]
    with pytest.raises(ValueError, match="x must be a CUDA tensor"):
        gelu_mlp(*tensors)


@pytest.mark.L0
@_SM100
def test_gelu_mlp_rejects_wrong_rank():
    x, w1, b1, w2, b2 = _inputs()
    with pytest.raises(ValueError, match=r"expected x\[\.\.\.,H\]"):
        gelu_mlp(x, w1, b1.reshape(1, -1), w2, b2)


@pytest.mark.L0
@_SM100
def test_gelu_mlp_rejects_noncontiguous_nn_linear_weight():
    x, w1, b1, w2, b2 = _inputs()
    square = torch.empty(w1.shape[0], w1.shape[0], device="cuda", dtype=torch.bfloat16)
    w1_bad = square.t()
    x_bad = torch.empty(*x.shape[:-1], w1_bad.shape[1], device="cuda", dtype=torch.bfloat16)
    w2_bad_shape = torch.empty(w2.shape[0], w1_bad.shape[0], device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="noncontiguous: w1"):
        gelu_mlp(x_bad, w1_bad, b1, w2_bad_shape, b2)
