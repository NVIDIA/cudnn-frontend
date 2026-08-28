# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU/mock contract tests for the opt-in FLA short-convolution shim."""

from __future__ import annotations

import functools
import importlib
import sys
import types

import pytest
import torch

import cudnn.fla as fla_api

short_conv = importlib.import_module("cudnn.fla.short_conv")

pytestmark = pytest.mark.L0


@pytest.fixture
def mock_sm100(monkeypatch):
    monkeypatch.setattr(short_conv, "_is_cuda_tensor", lambda tensor: True)
    monkeypatch.setattr(short_conv, "_device_capability", lambda device: (10, 0))
    monkeypatch.setattr(short_conv, "_is_compiling", lambda: False)


def _inputs(shape=(3, 8)):
    x = torch.randn(shape, dtype=torch.bfloat16)
    n_rows, n_channels = shape if len(shape) == 2 else (shape[0], shape[-1])
    if len(shape) == 3 and shape[0] == 1:
        n_rows = shape[1]
    weight = torch.randn(n_channels, 4, dtype=torch.bfloat16)
    cache = torch.randn(n_rows, n_channels, 4, dtype=torch.bfloat16)
    return x, weight, cache


def _original_spy(calls):
    def original(x, cache, residual=None, weight=None, bias=None, activation=None):
        calls.append((x, cache, residual, weight, bias, activation))
        return x.clone(), cache

    return original


@pytest.mark.parametrize("shape", [(3, 8), (3, 1, 8), (1, 3, 8)])
def test_native_layouts_are_zero_copy_and_preserve_fla_shape_and_cache_identity(mock_sm100, monkeypatch, shape):
    x, weight, cache = _inputs(shape)
    fallback_calls = []
    native_calls = []

    def native(native_x, native_weight, native_cache):
        native_calls.append((native_x, native_weight, native_cache))
        assert native_x.data_ptr() == x.data_ptr()
        return native_x.clone()

    monkeypatch.setattr(short_conv, "_call_native", native)
    wrapped = short_conv.make_causal_conv1d_update(_original_spy(fallback_calls))

    output, returned_cache = wrapped(x, cache, weight=weight, activation="silu")

    assert output.shape == x.shape
    assert returned_cache is cache
    assert len(native_calls) == 1
    assert native_calls[0][0].data_ptr() == x.data_ptr()
    assert native_calls[0][0].shape == (cache.shape[0], weight.shape[0])
    assert native_calls[0][1] is weight
    assert native_calls[0][2] is cache
    assert fallback_calls == []
    assert short_conv.last_path() == "native"


def test_sm110_calls_original_once_without_native_and_preserves_cache_identity(monkeypatch):
    x, weight, cache = _inputs()
    fallback_calls = []
    native_calls = []
    monkeypatch.setattr(short_conv, "_is_cuda_tensor", lambda tensor: True)
    monkeypatch.setattr(short_conv, "_device_capability", lambda device: (11, 0))
    monkeypatch.setattr(short_conv, "_is_compiling", lambda: False)

    def native(*args):
        native_calls.append(args)
        pytest.fail("SM110 must not enter the SM100 native route")

    monkeypatch.setattr(short_conv, "_call_native", native)
    wrapped = short_conv.make_causal_conv1d_update(_original_spy(fallback_calls))

    output, returned_cache = wrapped(x, cache, weight=weight, activation="silu")

    assert output.shape == x.shape
    assert returned_cache is cache
    assert len(fallback_calls) == 1
    assert fallback_calls[0][1] is cache
    assert native_calls == []
    assert short_conv.last_path() == "fallback:non-sm100"


@pytest.mark.parametrize(
    "mutation,reason",
    [
        (lambda x, weight, cache: {"residual": torch.zeros_like(x)}, "residual"),
        (lambda x, weight, cache: {"weight": None}, "weight"),
        (lambda x, weight, cache: {"bias": torch.zeros(weight.shape[0], dtype=weight.dtype)}, "bias"),
        (lambda x, weight, cache: {"activation": None}, "activation"),
        (lambda x, weight, cache: {"activation": "relu"}, "activation"),
        (lambda x, weight, cache: {"x": x.float()}, "non-bf16"),
        (lambda x, weight, cache: {"x": x.requires_grad_()}, "autograd"),
        (lambda x, weight, cache: {"x": x[:, ::2]}, "noncontiguous"),
        (lambda x, weight, cache: {"cache": cache[:2]}, "shape"),
        (lambda x, weight, cache: {"weight": weight[:, :3].contiguous()}, "shape"),
    ],
)
def test_unsupported_variants_call_original_unchanged(mock_sm100, monkeypatch, mutation, reason):
    x, weight, cache = _inputs()
    args = {"x": x, "cache": cache, "residual": None, "weight": weight, "bias": None, "activation": "swish"}
    args.update(mutation(x, weight, cache))
    fallback_calls = []
    monkeypatch.setattr(short_conv, "_call_native", lambda *unused: pytest.fail("native path should have declined"))
    wrapped = short_conv.make_causal_conv1d_update(_original_spy(fallback_calls))

    output, returned_cache = wrapped(**args)

    assert output.shape == args["x"].shape
    assert returned_cache is args["cache"]
    assert fallback_calls == [
        (
            args["x"],
            args["cache"],
            args["residual"],
            args["weight"],
            args["bias"],
            args["activation"],
        )
    ]
    assert short_conv.last_path() == f"fallback:{reason}"


@pytest.mark.parametrize("error", [NotImplementedError, short_conv.cudnn.cudnnGraphNotSupportedError, ImportError])
def test_typed_native_decline_falls_back(mock_sm100, monkeypatch, error):
    x, weight, cache = _inputs()
    fallback_calls = []
    monkeypatch.setattr(short_conv, "_call_native", lambda *unused: (_ for _ in ()).throw(error("declined")))

    output, returned_cache = short_conv.make_causal_conv1d_update(_original_spy(fallback_calls))(
        x,
        cache,
        weight=weight,
        activation="silu",
    )

    assert output.shape == x.shape
    assert returned_cache is cache
    assert len(fallback_calls) == 1
    assert short_conv.last_path() == f"fallback:{error.__name__}"


def test_unexpected_native_error_is_not_swallowed(mock_sm100, monkeypatch):
    x, weight, cache = _inputs()
    fallback_calls = []
    monkeypatch.setattr(short_conv, "_call_native", lambda *unused: (_ for _ in ()).throw(RuntimeError("launch failed")))

    with pytest.raises(RuntimeError, match="launch failed"):
        short_conv.make_causal_conv1d_update(_original_spy(fallback_calls))(x, cache, weight=weight, activation="silu")

    assert fallback_calls == []
    assert short_conv.last_path() == "error:RuntimeError"


def _fake_fla_short_conv_module():
    module = types.ModuleType("fla.modules.conv.triton.ops")

    def causal_conv1d_update(x, cache, residual=None, weight=None, bias=None, activation=None):
        del residual, weight, bias, activation
        return x, cache

    causal_conv1d_update.__module__ = module.__name__
    causal_conv1d_update.__qualname__ = "causal_conv1d_update"
    module.causal_conv1d_update = causal_conv1d_update
    return module


def test_public_alias_rebind_and_restore(monkeypatch):
    module = _fake_fla_short_conv_module()
    original = module.causal_conv1d_update
    captured_import = types.ModuleType("_cudnn_fla_short_conv_consumer")
    captured_import.causal_conv1d_update = original
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setitem(sys.modules, captured_import.__name__, captured_import)
    monkeypatch.setattr(fla_api, "_supports_short_conv_fla", lambda: True)
    monkeypatch.setattr(fla_api, "_matches_short_conv_target", lambda target_module, target: True)
    monkeypatch.setattr(fla_api, "_ORIGINALS", {})

    fla_api.accelerate_fla(verbose=False, targets="shortconv")

    replacement = module.causal_conv1d_update
    assert replacement is not original
    assert captured_import.causal_conv1d_update is replacement
    assert fla_api.is_accelerated("short_conv")

    fla_api.restore_fla(targets="short_conv")

    assert module.causal_conv1d_update is original
    assert captured_import.causal_conv1d_update is original
    assert not fla_api.is_accelerated("shortconv")


def test_public_activation_rejects_other_fla_version(monkeypatch):
    module = _fake_fla_short_conv_module()
    original = module.causal_conv1d_update
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(fla_api, "_supports_short_conv_fla", lambda: False)
    monkeypatch.setattr(fla_api, "_ORIGINALS", {})

    with pytest.raises(ImportError, match=r"requires flash-linear-attention==0\.5\.2"):
        fla_api.accelerate_fla(verbose=False, targets="short_conv")

    assert module.causal_conv1d_update is original
    assert not fla_api.is_accelerated("shortconv")


def test_public_activation_rejects_replaced_target(monkeypatch):
    module = _fake_fla_short_conv_module()
    stock = module.causal_conv1d_update

    @functools.wraps(stock)
    def third_party(*args, **kwargs):
        return args, kwargs

    module.causal_conv1d_update = third_party
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(fla_api, "_supports_short_conv_fla", lambda: True)
    monkeypatch.setattr(fla_api, "_ORIGINALS", {})

    with pytest.raises(ImportError, match="does not match the expected owning module"):
        fla_api.accelerate_fla(verbose=False, targets="short_conv")

    assert module.causal_conv1d_update is third_party
    assert not fla_api.is_accelerated("shortconv")
