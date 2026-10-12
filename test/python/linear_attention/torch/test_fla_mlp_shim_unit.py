# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU/mock contract tests for the opt-in FLA ``GatedMLP`` shim."""

from __future__ import annotations

import functools
import sys
import types

import pytest
import torch
import torch.nn.functional as F

import cudnn.fla as fla_api
import cudnn.fla.gated_mlp as gated_mlp

pytestmark = pytest.mark.L0


class _FakeSwiGLULinear(torch.nn.Module):
    def forward(self, gate, up, weight, bias):
        return F.linear(F.silu(gate) * up, weight, bias)


class _FakeGatedMLP(torch.nn.Module):
    def __init__(self, *, hidden=16, intermediate=32, dtype=torch.bfloat16):
        super().__init__()
        self.hidden_size = hidden
        self.hidden_ratio = 4
        self.intermediate_size = intermediate
        self.hidden_act = "swish"
        self.fuse_swiglu = True
        self.powglu_power = 3.0
        self.gate_proj = torch.nn.Linear(hidden, intermediate, bias=False, dtype=dtype)
        self.up_proj = torch.nn.Linear(hidden, intermediate, bias=False, dtype=dtype)
        self.down_proj = torch.nn.Linear(intermediate, hidden, bias=False, dtype=dtype)
        self.swiglu_linear = _FakeSwiGLULinear()
        self.fallback_calls = 0
        self.fallback_kwargs = None

    def forward(self, x, **kwargs):
        self.fallback_calls += 1
        self.fallback_kwargs = kwargs
        gate, up = self.gate_proj(x), self.up_proj(x)
        return self.swiglu_linear(gate, up, self.down_proj.weight, self.down_proj.bias)


@pytest.fixture
def mock_sm100(monkeypatch):
    monkeypatch.setattr(gated_mlp, "_installed_fla_version", lambda: "0.5.2")
    monkeypatch.setattr(gated_mlp, "_is_cuda_tensor", lambda tensor: True)
    monkeypatch.setattr(gated_mlp, "_device_capability", lambda device: (10, 0))
    monkeypatch.setattr(gated_mlp, "_has_global_module_hooks", lambda: False)


def _wrapped():
    return gated_mlp.make_gated_mlp_forward(_FakeGatedMLP.forward, _FakeGatedMLP, _FakeSwiGLULinear)


def _input(dtype=torch.bfloat16):
    return torch.randn(2, 3, 16, dtype=dtype)


def test_native_path_uses_exact_weights_and_preserves_ignored_kwargs(mock_sm100, monkeypatch):
    module = _FakeGatedMLP()
    x = _input()
    calls = []

    def native(input_, gate, up, down):
        calls.append((input_, gate, up, down))
        return input_.clone()

    monkeypatch.setattr(gated_mlp, "_call_native", native)
    out = _wrapped()(module, x, arbitrary_object=object())

    assert torch.equal(out, x)
    assert calls == [(x, module.gate_proj.weight, module.up_proj.weight, module.down_proj.weight)]
    assert module.fallback_calls == 0
    assert gated_mlp.last_path() == "native"


@pytest.mark.parametrize(
    "case,reason",
    [
        ("fla-version", "fla-version"),
        ("compile", "compile"),
        ("fp16-autocast", "autocast"),
        ("cpu", "non-cuda"),
        ("pre-sm100", "non-sm100"),
        ("non-bf16", "non-bf16"),
        ("noncontiguous-input", "shape-or-layout"),
        ("dtensor-like-input", "tensor-subclass"),
        ("tensor-parallel-module", "custom-gated-mlp"),
        ("instance-forward", "custom-gated-mlp"),
        ("bias", "bias"),
        ("lora-linear", "custom-module"),
        ("quantized-linear", "custom-module"),
        ("parametrized", "parametrized"),
        ("extra-parameter", "custom-module"),
        ("noncontiguous-weight", "noncontiguous"),
        ("parameter-hook", "hooks"),
    ],
)
def test_unsupported_variants_call_original_fla(mock_sm100, monkeypatch, case, reason):
    module = _FakeGatedMLP()
    x = _input()

    if case == "fla-version":
        monkeypatch.setattr(gated_mlp, "_installed_fla_version", lambda: "0.5.3")
    elif case == "compile":
        monkeypatch.setattr(gated_mlp, "_is_compiling", lambda: True)
    elif case == "fp16-autocast":
        monkeypatch.setattr(gated_mlp, "_cuda_autocast_dtype", lambda: torch.float16)
    elif case == "cpu":
        monkeypatch.setattr(gated_mlp, "_is_cuda_tensor", lambda tensor: False)
    elif case == "pre-sm100":
        monkeypatch.setattr(gated_mlp, "_device_capability", lambda device: (9, 0))
    elif case == "non-bf16":
        module = module.float()
        x = x.float()
    elif case == "noncontiguous-input":
        x = torch.randn(2, 16, 3, dtype=torch.bfloat16).transpose(1, 2)
    elif case == "dtensor-like-input":

        class DTensorLike(torch.Tensor):
            pass

        x = x.as_subclass(DTensorLike)
    elif case == "tensor-parallel-module":

        class TensorParallelGatedMLP(_FakeGatedMLP):
            pass

        module = TensorParallelGatedMLP()
    elif case == "instance-forward":
        module.forward = types.MethodType(_FakeGatedMLP.forward, module)
    elif case == "bias":
        module.gate_proj = torch.nn.Linear(16, 32, bias=True, dtype=torch.bfloat16)
    elif case == "lora-linear":

        class LoraLinear(torch.nn.Linear):
            pass

        module.gate_proj = LoraLinear(16, 32, bias=False, dtype=torch.bfloat16)
    elif case == "quantized-linear":

        class QuantizedLinear(torch.nn.Linear):
            pass

        module.gate_proj = QuantizedLinear(16, 32, bias=False, dtype=torch.bfloat16)
    elif case == "parametrized":

        class Identity(torch.nn.Module):
            def forward(self, weight):
                return weight

        torch.nn.utils.parametrize.register_parametrization(module.gate_proj, "weight", Identity())
    elif case == "extra-parameter":
        module.gate_proj.register_parameter("lora_A", torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16)))
    elif case == "noncontiguous-weight":
        module.gate_proj.weight = torch.nn.Parameter(torch.randn(16, 32, dtype=torch.bfloat16).t())
    elif case == "parameter-hook":
        module.gate_proj.weight.register_hook(lambda grad: grad)
    else:
        raise AssertionError(case)

    def must_not_run(*args):
        raise AssertionError("native path should have declined")

    monkeypatch.setattr(gated_mlp, "_call_native", must_not_run)
    kwargs = {"opaque": object()}
    out = _wrapped()(module, x, **kwargs)

    assert out.shape == x.shape
    assert module.fallback_calls == 1
    assert module.fallback_kwargs == kwargs
    assert gated_mlp.last_path() == f"fallback:{reason}"


def test_bf16_autocast_remains_on_native_path(mock_sm100, monkeypatch):
    module = _FakeGatedMLP()
    x = _input()
    monkeypatch.setattr(gated_mlp, "_cuda_autocast_dtype", lambda: torch.bfloat16)
    monkeypatch.setattr(gated_mlp, "_call_native", lambda *args: x.clone())

    out = _wrapped()(module, x)

    assert torch.equal(out, x)
    assert module.fallback_calls == 0
    assert gated_mlp.last_path() == "native"


def test_projection_hook_forces_fallback_and_still_runs(mock_sm100, monkeypatch):
    module = _FakeGatedMLP()
    seen = []
    module.gate_proj.register_forward_hook(lambda *args: seen.append(True))
    monkeypatch.setattr(gated_mlp, "_call_native", lambda *args: pytest.fail("native path should have declined"))

    _wrapped()(module, _input())

    assert module.fallback_calls == 1
    assert seen == [True]
    assert gated_mlp.last_path() == "fallback:hooks"


@pytest.mark.parametrize(
    "error",
    [NotImplementedError, gated_mlp.cudnn.cudnnGraphNotSupportedError, ImportError],
    ids=["not-implemented", "graph-not-supported", "optional-dependency"],
)
def test_typed_native_decline_falls_back(mock_sm100, monkeypatch, error):
    module = _FakeGatedMLP()

    def decline(*args):
        raise error("decline")

    monkeypatch.setattr(gated_mlp, "_call_native", decline)
    _wrapped()(module, _input())

    assert module.fallback_calls == 1
    assert gated_mlp.last_path() == f"fallback:{error.__name__}"


def test_unexpected_native_error_propagates(mock_sm100, monkeypatch):
    module = _FakeGatedMLP()

    def fail(*args):
        raise RuntimeError("synthetic launch failure")

    monkeypatch.setattr(gated_mlp, "_call_native", fail)
    with pytest.raises(RuntimeError, match="synthetic launch failure"):
        _wrapped()(module, _input())
    assert module.fallback_calls == 0
    assert gated_mlp.last_path() == "error:RuntimeError"


def test_native_control_flow_exception_propagates_without_overwriting_route(mock_sm100, monkeypatch):
    module = _FakeGatedMLP()

    class ControlFlow(Exception):
        pass

    def stop_recomputation(*args):
        raise ControlFlow("synthetic checkpoint control flow")

    monkeypatch.setattr(gated_mlp, "_call_native", stop_recomputation)
    with pytest.raises(ControlFlow, match="synthetic checkpoint control flow"):
        _wrapped()(module, _input())
    assert module.fallback_calls == 0
    assert gated_mlp.last_path() == "native"


def test_target_registry_is_incremental_idempotent_and_restorable(monkeypatch):
    function_module = types.ModuleType("_cudnn_fla_test_function_target")

    def original_function():
        return "original-function"

    function_module.test_function_target = original_function

    method_module = types.ModuleType("_cudnn_fla_test_method_target")

    class Owner:
        def forward(self):
            return "original-method"

    method_module.Owner = Owner
    method_module.test_method_target = object()
    original_method = Owner.forward

    def make_function(module, owner, original):
        del module, owner

        def replacement():
            return "patched-" + original()

        return replacement

    def make_method(module, owner, original):
        del module, owner

        def replacement(self):
            return "patched-" + original(self)

        return replacement

    targets = {
        "default": fla_api._PatchSpec(function_module.__name__, "test_function_target", make_function),
        "gated_mlp": fla_api._PatchSpec(method_module.__name__, "forward", make_method, owner_attribute="Owner", default=False),
    }
    monkeypatch.setitem(sys.modules, function_module.__name__, function_module)
    monkeypatch.setitem(sys.modules, method_module.__name__, method_module)
    monkeypatch.setattr(fla_api, "_TARGETS", targets)
    monkeypatch.setattr(fla_api, "_ALIASES", {"mlp": "gated_mlp"})
    monkeypatch.setattr(fla_api, "_DEFAULT_TARGETS", ("default",))
    monkeypatch.setattr(fla_api, "_ORIGINALS", {})

    existing = Owner()
    fla_api.accelerate_fla(verbose=False)
    assert function_module.test_function_target() == "patched-original-function"
    assert existing.forward() == "original-method"
    assert fla_api.is_accelerated("default")
    assert not fla_api.is_accelerated("mlp")

    fla_api.accelerate_fla(verbose=False, targets="mlp")
    replacement = Owner.forward
    assert existing.forward() == "patched-original-method"
    assert Owner().forward() == "patched-original-method"
    assert fla_api.is_accelerated("gated_mlp")

    fla_api.accelerate_fla(verbose=False, targets=("gated_mlp", "default"))
    assert Owner.forward is replacement

    fla_api.restore_fla(targets="mlp")
    assert Owner.forward is original_method
    assert existing.forward() == "original-method"
    assert fla_api.is_accelerated("default")

    fla_api.accelerate_fla(verbose=False, targets="mlp")

    def third_party(self):
        return "third-party-method"

    Owner.forward = third_party
    assert not fla_api.is_accelerated("mlp")
    fla_api.restore_fla(targets="mlp")
    assert Owner.forward is third_party

    # A generic registry target can be claimed again after displacement; its
    # eventual restore belongs to the callable that was live at re-activation.
    fla_api.accelerate_fla(verbose=False, targets="mlp")
    assert fla_api.is_accelerated("mlp")
    assert Owner().forward() == "patched-third-party-method"
    fla_api.restore_fla(targets="mlp")
    assert Owner.forward is third_party

    fla_api.restore_fla()
    assert function_module.test_function_target is original_function
    assert not fla_api.is_accelerated()


def test_displaced_unavailable_default_target_does_not_report_stale_success(monkeypatch):
    method_module = types.ModuleType("_cudnn_fla_test_displaced_target")

    class Owner:
        def forward(self):
            return "original"

    method_module.Owner = Owner

    def make_method(module, owner, original):
        del module, owner

        def replacement(self):
            return "patched-" + original(self)

        return replacement

    target = fla_api._PatchSpec(method_module.__name__, "forward", make_method, owner_attribute="Owner")
    monkeypatch.setitem(sys.modules, method_module.__name__, method_module)
    monkeypatch.setattr(fla_api, "_TARGETS", {"default": target})
    monkeypatch.setattr(fla_api, "_ALIASES", {})
    monkeypatch.setattr(fla_api, "_DEFAULT_TARGETS", ("default",))
    monkeypatch.setattr(fla_api, "_ORIGINALS", {})

    fla_api.accelerate_fla(verbose=False)
    Owner.forward = lambda self: "third-party"
    assert not fla_api.is_accelerated("default")
    monkeypatch.delitem(sys.modules, method_module.__name__)

    with pytest.raises(ImportError, match="could not find supported FLA target"):
        fla_api.accelerate_fla(verbose=False)
    assert not fla_api.is_accelerated()
    assert Owner().forward() == "third-party"


def _fake_fla_mlp_module():
    module = types.ModuleType("fla.modules.mlp")

    class SwiGLULinear:
        pass

    class GatedMLP:
        def forward(self, x, **kwargs):
            del kwargs
            return x

    SwiGLULinear.__module__ = module.__name__
    GatedMLP.__module__ = module.__name__
    GatedMLP.forward.__module__ = module.__name__
    GatedMLP.forward.__qualname__ = "GatedMLP.forward"
    module.SwiGLULinear = SwiGLULinear
    module.GatedMLP = GatedMLP
    return module


def test_public_mlp_activation_rejects_unsupported_fla_version(monkeypatch):
    module = _fake_fla_mlp_module()
    original = module.GatedMLP.forward
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(fla_api, "_supports_installed_fla", lambda: False)
    monkeypatch.setattr(fla_api, "_ORIGINALS", {})

    with pytest.raises(ImportError, match=r"requires flash-linear-attention==0\.5\.2"):
        fla_api.accelerate_fla(verbose=False, targets="gated_mlp")

    assert module.GatedMLP.forward is original
    assert not fla_api.is_accelerated("gated_mlp")


def test_public_mlp_activation_rejects_prewrapped_class_method(monkeypatch):
    module = _fake_fla_mlp_module()
    stock = module.GatedMLP.forward

    @functools.wraps(stock)
    def third_party(self, x, **kwargs):
        return stock(self, x, **kwargs)

    module.GatedMLP.forward = third_party
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(fla_api, "_supports_installed_fla", lambda: True)
    monkeypatch.setattr(fla_api, "_ORIGINALS", {})

    with pytest.raises(ImportError, match="was replaced before cuDNN acceleration"):
        fla_api.accelerate_fla(verbose=False, targets="gated_mlp")

    assert module.GatedMLP.forward is third_party
    assert not fla_api.is_accelerated("gated_mlp")


def test_unknown_target_is_rejected():
    with pytest.raises(ValueError, match="unknown FLA acceleration target"):
        fla_api.accelerate_fla(verbose=False, targets="does-not-exist")
