# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""B200 parity/fallback coverage for the production FLA ``GatedMLP`` shim."""

from __future__ import annotations

from importlib import metadata

import pytest
import torch
import torch.utils.checkpoint

fla_mlp = pytest.importorskip("fla.modules.mlp")

try:
    _FLA_VERSION = metadata.version("flash-linear-attention")
except metadata.PackageNotFoundError:
    _FLA_VERSION = None

from cudnn.fla import accelerate_fla, is_accelerated, mlp_last_path, restore_fla
from cudnn.fla.gated_mlp import make_gated_mlp_forward

pytestmark = [
    pytest.mark.L0,
    pytest.mark.skipif(
        not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10),
        reason="cuDNN SwiGLU-MLP fusion requires SM100",
    ),
    pytest.mark.skipif(
        _FLA_VERSION != "0.5.2",
        reason="the production GatedMLP shim intentionally supports FLA 0.5.2 exactly",
    ),
]

_REAL_FORWARD = fla_mlp.GatedMLP.forward
_SHIM_FORWARD = make_gated_mlp_forward(_REAL_FORWARD, fla_mlp.GatedMLP, fla_mlp.SwiGLULinear)
_TOL = 2e-2


def _rel_l2(actual, expected):
    return (actual.float() - expected.float()).norm().item() / max(expected.float().norm().item(), 1e-9)


def _module():
    return fla_mlp.GatedMLP(hidden_size=256, intermediate_size=512, hidden_act="swish", fuse_swiglu=True).cuda().to(torch.bfloat16)


def _clone_pair():
    stock = _module()
    native = _module()
    native.load_state_dict(stock.state_dict())
    return stock, native


def _assert_grads(native, stock):
    for (native_name, native_parameter), (stock_name, stock_parameter) in zip(native.named_parameters(), stock.named_parameters()):
        assert native_name == stock_name
        assert native_parameter.grad is not None and stock_parameter.grad is not None
        assert _rel_l2(native_parameter.grad, stock_parameter.grad) < _TOL, native_name


def test_gated_mlp_forward_backward_parity_and_kwargs():
    torch.manual_seed(10)
    stock, native = _clone_pair()
    x_stock = torch.randn(2, 128, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    x_native = x_stock.detach().clone().requires_grad_(True)
    dout = torch.randn_like(x_stock)

    out_stock = _REAL_FORWARD(stock, x_stock, ignored_by_fla=object())
    out_native = _SHIM_FORWARD(native, x_native, ignored_by_fla=object())
    assert mlp_last_path() == "native"
    out_stock.backward(dout)
    out_native.backward(dout)

    assert _rel_l2(out_native, out_stock) < _TOL
    assert _rel_l2(x_native.grad, x_stock.grad) < _TOL
    _assert_grads(native, stock)


@pytest.mark.parametrize("use_reentrant", [True, False], ids=["reentrant", "non-reentrant"])
def test_gated_mlp_checkpoint_parity(use_reentrant):
    torch.manual_seed(11)
    stock, native = _clone_pair()
    x_stock = torch.randn(1, 128, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    x_native = x_stock.detach().clone().requires_grad_(True)
    dout = torch.randn_like(x_stock)

    out_stock = torch.utils.checkpoint.checkpoint(lambda value: _REAL_FORWARD(stock, value), x_stock, use_reentrant=use_reentrant)
    out_native = torch.utils.checkpoint.checkpoint(lambda value: _SHIM_FORWARD(native, value), x_native, use_reentrant=use_reentrant)
    out_stock.backward(dout)
    out_native.backward(dout)

    assert mlp_last_path() == "native"
    assert _rel_l2(out_native, out_stock) < _TOL
    assert _rel_l2(x_native.grad, x_stock.grad) < _TOL
    _assert_grads(native, stock)


@pytest.mark.parametrize(
    "autocast_dtype,expected_path",
    [(torch.bfloat16, "native"), (torch.float16, "fallback:autocast")],
    ids=["bf16-native", "fp16-fallback"],
)
def test_gated_mlp_autocast_preserves_fla_semantics(autocast_dtype, expected_path):
    module = _module()
    x = torch.randn(1, 128, 256, device="cuda", dtype=torch.bfloat16)
    calls = []

    def recording_original(self, value, **kwargs):
        calls.append(kwargs)
        return _REAL_FORWARD(self, value, **kwargs)

    shim = make_gated_mlp_forward(recording_original, fla_mlp.GatedMLP, fla_mlp.SwiGLULinear)
    with torch.autocast("cuda", dtype=autocast_dtype):
        out = shim(module, x, opaque=object())

    assert out.shape == x.shape
    assert out.dtype is autocast_dtype
    assert mlp_last_path() == expected_path
    assert len(calls) == (1 if autocast_dtype is torch.float16 else 0)


@pytest.mark.parametrize("variant", ["noncontiguous", "fp16", "hook", "custom-linear"], ids=str)
def test_gated_mlp_unsupported_variant_calls_original(variant):
    module = _module()
    x = torch.randn(2, 128, 256, device="cuda", dtype=torch.bfloat16)
    calls = []

    def recording_original(self, value, **kwargs):
        calls.append(kwargs)
        return _REAL_FORWARD(self, value, **kwargs)

    shim = make_gated_mlp_forward(recording_original, fla_mlp.GatedMLP, fla_mlp.SwiGLULinear)
    hook_calls = []
    if variant == "noncontiguous":
        x = torch.randn(2, 256, 128, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    elif variant == "fp16":
        module = module.half()
        x = x.half()
    elif variant == "hook":
        module.gate_proj.register_forward_hook(lambda *args: hook_calls.append(True))
    elif variant == "custom-linear":

        class CustomLinear(torch.nn.Linear):
            pass

        replacement = CustomLinear(256, 512, bias=False, device="cuda", dtype=torch.bfloat16)
        replacement.weight.data.copy_(module.gate_proj.weight)
        module.gate_proj = replacement
    else:
        raise AssertionError(variant)

    out = shim(module, x, opaque=object())

    assert out.shape == x.shape
    assert len(calls) == 1 and set(calls[0]) == {"opaque"}
    assert mlp_last_path().startswith("fallback:")
    assert hook_calls == ([True] if variant == "hook" else [])


def test_gated_mlp_api_is_opt_in_and_covers_existing_instances():
    original = fla_mlp.GatedMLP.forward
    existing = _module()
    try:
        accelerate_fla(verbose=False)
        assert fla_mlp.GatedMLP.forward is original
        assert not is_accelerated("gated_mlp")

        accelerate_fla(verbose=False, targets="gated_mlp")
        replacement = fla_mlp.GatedMLP.forward
        assert replacement is not original
        assert existing.forward.__func__ is replacement
        assert _module().forward.__func__ is replacement
        assert is_accelerated("mlp")

        accelerate_fla(verbose=False, targets="mlp")
        assert fla_mlp.GatedMLP.forward is replacement

        x = torch.randn(1, 128, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        out = existing(x)
        assert mlp_last_path() == "native"
        out.float().sum().backward()
        assert x.grad is not None
        assert all(parameter.grad is not None for parameter in existing.parameters())
    finally:
        restore_fla()
    assert fla_mlp.GatedMLP.forward is original
