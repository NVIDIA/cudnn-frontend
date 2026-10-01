# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU/mock contract tests for the opt-in FLA short-convolution shim."""

from __future__ import annotations

import functools
import importlib
import sys
import types

import cudnn
import cudnn.fla as fla_api
import cudnn.ops as cudnn_ops
import pytest
import torch

short_conv = importlib.import_module("cudnn.fla.short_conv")

pytestmark = pytest.mark.L0

_SUPPORTED_COMPUTE_CAPABILITIES = (
    (8, 0),
    (8, 6),
    (8, 7),
    (8, 9),
    (9, 0),
    (10, 0),
    (10, 3),
    (11, 0),
    (12, 0),
    (12, 1),
)
_FALLBACK_VALUE = -17.0


@pytest.fixture(autouse=True)
def _restore_short_conv_patch():
    fla_api.restore_fla(targets="short_conv")
    yield
    fla_api.restore_fla(targets="short_conv")


@pytest.fixture
def supported_tensor_environment(monkeypatch):
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


def _fused_projection_view(layout, *, n_rows=3, n_channels=8):
    projection = torch.randn(n_rows, 3 * n_channels, dtype=torch.bfloat16)
    x = projection[:, n_channels : 2 * n_channels]
    if layout == "ND":
        return x
    if layout == "N1D":
        return x.unsqueeze(1)
    if layout == "1ND":
        return x.unsqueeze(0)
    raise AssertionError(layout)


def _install_fake_fla_module(monkeypatch, fallback_calls):
    packages = {}
    for name in ("fla", "fla.modules", "fla.modules.conv", "fla.modules.conv.triton"):
        package = types.ModuleType(name)
        package.__path__ = []
        packages[name] = package
        monkeypatch.setitem(sys.modules, name, package)

    module = types.ModuleType("fla.modules.conv.triton.ops")

    def original_callable(x, cache, residual=None, weight=None, bias=None, activation=None):
        del residual, weight, bias, activation
        return torch.full_like(x, _FALLBACK_VALUE), cache

    def causal_conv1d_update(x, cache, residual=None, weight=None, bias=None, activation=None):
        fallback_calls.append(1)
        return original_callable(x, cache, residual=residual, weight=weight, bias=bias, activation=activation)

    causal_conv1d_update.__module__ = module.__name__
    causal_conv1d_update.__name__ = "causal_conv1d_update"
    causal_conv1d_update.__qualname__ = "causal_conv1d_update"
    causal_conv1d_update.__wrapped__ = original_callable
    causal_conv1d_update._torchdynamo_disable = True
    causal_conv1d_update._torchdynamo_orig_callable = original_callable
    module.causal_conv1d_update = causal_conv1d_update
    monkeypatch.setitem(sys.modules, module.__name__, module)

    packages["fla"].modules = packages["fla.modules"]
    packages["fla.modules"].conv = packages["fla.modules.conv"]
    packages["fla.modules.conv"].triton = packages["fla.modules.conv.triton"]
    packages["fla.modules.conv.triton"].ops = module
    return module, causal_conv1d_update


def _activate_fake_fla(monkeypatch, fallback_calls):
    monkeypatch.setattr(short_conv.metadata, "version", lambda distribution: "0.5.2")
    module, original = _install_fake_fla_module(monkeypatch, fallback_calls)
    fla_api.accelerate_fla(verbose=False, targets="short_conv")
    assert fla_api.is_accelerated("short_conv")
    return module, original


def _install_native_spy(monkeypatch, native_calls):
    def native_update(x, *args, **kwargs):
        del args, kwargs
        native_calls.append(1)
        return x + 1

    monkeypatch.setattr(cudnn_ops, "causal_conv1d_update", native_update)


@pytest.mark.parametrize("shape", [(3, 8), (3, 1, 8), (1, 3, 8)])
def test_public_shim_preserves_fla_shape_and_cache_identity(supported_tensor_environment, monkeypatch, shape):
    x, weight, cache = _inputs(shape)
    fallback_calls = []
    native_calls = []
    module, _ = _activate_fake_fla(monkeypatch, fallback_calls)
    _install_native_spy(monkeypatch, native_calls)

    output, returned_cache = module.causal_conv1d_update(x, cache, weight=weight, activation="silu")

    assert output.shape == x.shape
    torch.testing.assert_close(output, x + 1)
    assert returned_cache is cache
    assert native_calls == [1]
    assert fallback_calls == []
    assert fla_api.short_conv_last_path() == "native"


@pytest.mark.parametrize("layout", ["ND", "N1D", "1ND"])
def test_fused_projection_views_route_through_public_shim(supported_tensor_environment, monkeypatch, layout):
    x = _fused_projection_view(layout)
    n_rows, n_channels = 3, 8
    weight = torch.randn(n_channels, 4, dtype=torch.bfloat16)
    cache = torch.randn(n_rows, n_channels, 4, dtype=torch.bfloat16)
    fallback_calls = []
    native_calls = []
    module, _ = _activate_fake_fla(monkeypatch, fallback_calls)
    _install_native_spy(monkeypatch, native_calls)

    output, returned_cache = module.causal_conv1d_update(x, cache, weight=weight, activation="silu")

    assert output.shape == x.shape
    torch.testing.assert_close(output, x + 1)
    assert returned_cache is cache
    assert native_calls == [1]
    assert fallback_calls == []
    assert fla_api.short_conv_last_path() == "native"


@pytest.mark.parametrize("n_channels", [7, 10], ids=["d7", "d10"])
def test_compact_rows_accept_channel_counts_not_divisible_by_eight(supported_tensor_environment, monkeypatch, n_channels):
    x, weight, cache = _inputs((3, n_channels))
    fallback_calls = []
    native_calls = []
    module, _ = _activate_fake_fla(monkeypatch, fallback_calls)
    _install_native_spy(monkeypatch, native_calls)

    output, returned_cache = module.causal_conv1d_update(x, cache, weight=weight, activation="silu")

    torch.testing.assert_close(output, x + 1)
    assert returned_cache is cache
    assert native_calls == [1]
    assert fallback_calls == []
    assert fla_api.short_conv_last_path() == "native"


@pytest.mark.parametrize(
    "x,reason",
    [
        (torch.empty_strided((3, 8), (7, 1), dtype=torch.bfloat16), "noncontiguous"),
        (torch.empty_strided((3, 10), (12, 1), dtype=torch.bfloat16), "alignment"),
    ],
    ids=["overlapping-rows", "misaligned-padded-rows"],
)
def test_unsupported_row_strides_fall_back(supported_tensor_environment, monkeypatch, x, reason):
    n_rows, n_channels = x.shape
    weight = torch.randn(n_channels, 4, dtype=torch.bfloat16)
    cache = torch.randn(n_rows, n_channels, 4, dtype=torch.bfloat16)
    fallback_calls = []
    native_calls = []
    module, _ = _activate_fake_fla(monkeypatch, fallback_calls)
    _install_native_spy(monkeypatch, native_calls)

    output, returned_cache = module.causal_conv1d_update(x, cache, weight=weight, activation="silu")

    torch.testing.assert_close(output, torch.full_like(x, _FALLBACK_VALUE))
    assert returned_cache is cache
    assert fallback_calls == [1]
    assert native_calls == []
    assert fla_api.short_conv_last_path() == f"fallback:{reason}"


@pytest.mark.parametrize("capability", _SUPPORTED_COMPUTE_CAPABILITIES)
def test_supported_architectures_enter_native_route(monkeypatch, capability):
    monkeypatch.setattr(short_conv, "_is_cuda_tensor", lambda tensor: True)
    monkeypatch.setattr(short_conv, "_device_capability", lambda device: capability)
    monkeypatch.setattr(short_conv, "_is_compiling", lambda: False)
    x, weight, cache = _inputs()
    fallback_calls = []
    native_calls = []
    module, _ = _activate_fake_fla(monkeypatch, fallback_calls)
    _install_native_spy(monkeypatch, native_calls)

    output, returned_cache = module.causal_conv1d_update(x, cache, weight=weight, activation="silu")

    torch.testing.assert_close(output, x + 1)
    assert returned_cache is cache
    assert native_calls == [1]
    assert fallback_calls == []
    assert fla_api.short_conv_last_path() == "native"


def test_unlisted_arch_calls_original_once_and_preserves_cache_identity(monkeypatch):
    monkeypatch.setattr(short_conv, "_is_cuda_tensor", lambda tensor: True)
    monkeypatch.setattr(short_conv, "_device_capability", lambda device: (11, 1))
    monkeypatch.setattr(short_conv, "_is_compiling", lambda: False)
    x, weight, cache = _inputs()
    fallback_calls = []
    native_calls = []
    module, _ = _activate_fake_fla(monkeypatch, fallback_calls)
    _install_native_spy(monkeypatch, native_calls)

    output, returned_cache = module.causal_conv1d_update(x, cache, weight=weight, activation="silu")

    torch.testing.assert_close(output, torch.full_like(x, _FALLBACK_VALUE))
    assert returned_cache is cache
    assert fallback_calls == [1]
    assert native_calls == []
    assert fla_api.short_conv_last_path() == "fallback:unsupported-arch"


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
def test_unsupported_variants_call_original(supported_tensor_environment, monkeypatch, mutation, reason):
    x, weight, cache = _inputs()
    arguments = {"x": x, "cache": cache, "residual": None, "weight": weight, "bias": None, "activation": "swish"}
    arguments.update(mutation(x, weight, cache))
    fallback_calls = []
    native_calls = []
    module, _ = _activate_fake_fla(monkeypatch, fallback_calls)
    _install_native_spy(monkeypatch, native_calls)

    output, returned_cache = module.causal_conv1d_update(**arguments)

    torch.testing.assert_close(output, torch.full_like(arguments["x"], _FALLBACK_VALUE))
    assert returned_cache is arguments["cache"]
    assert fallback_calls == [1]
    assert native_calls == []
    assert fla_api.short_conv_last_path() == f"fallback:{reason}"


@pytest.mark.parametrize("error", [NotImplementedError, cudnn.cudnnGraphNotSupportedError, ImportError])
def test_typed_native_decline_falls_back(supported_tensor_environment, monkeypatch, error):
    x, weight, cache = _inputs()
    fallback_calls = []
    module, _ = _activate_fake_fla(monkeypatch, fallback_calls)

    def decline(*args, **kwargs):
        del args, kwargs
        raise error("declined")

    monkeypatch.setattr(cudnn_ops, "causal_conv1d_update", decline)

    output, returned_cache = module.causal_conv1d_update(x, cache, weight=weight, activation="silu")

    torch.testing.assert_close(output, torch.full_like(x, _FALLBACK_VALUE))
    assert returned_cache is cache
    assert fallback_calls == [1]
    assert fla_api.short_conv_last_path() == f"fallback:{error.__name__}"


def test_unexpected_native_error_is_not_swallowed(supported_tensor_environment, monkeypatch):
    x, weight, cache = _inputs()
    fallback_calls = []
    module, _ = _activate_fake_fla(monkeypatch, fallback_calls)

    def fail(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("launch failed")

    monkeypatch.setattr(cudnn_ops, "causal_conv1d_update", fail)

    with pytest.raises(RuntimeError, match="launch failed"):
        module.causal_conv1d_update(x, cache, weight=weight, activation="silu")

    assert fallback_calls == []
    assert fla_api.short_conv_last_path() == "error:RuntimeError"


def test_public_alias_rebind_and_restore(monkeypatch):
    monkeypatch.setattr(short_conv.metadata, "version", lambda distribution: "0.5.2")
    fallback_calls = []
    module, original = _install_fake_fla_module(monkeypatch, fallback_calls)
    captured_import = types.ModuleType("_cudnn_fla_short_conv_consumer")
    captured_import.causal_conv1d_update = original
    monkeypatch.setitem(sys.modules, captured_import.__name__, captured_import)

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
    monkeypatch.setattr(short_conv.metadata, "version", lambda distribution: "9.9.9")
    fallback_calls = []
    module, original = _install_fake_fla_module(monkeypatch, fallback_calls)

    with pytest.raises(ImportError, match=r"requires flash-linear-attention==0\.5\.2"):
        fla_api.accelerate_fla(verbose=False, targets="short_conv")

    assert module.causal_conv1d_update is original
    assert not fla_api.is_accelerated("shortconv")


def test_public_activation_rejects_replaced_target(monkeypatch):
    monkeypatch.setattr(short_conv.metadata, "version", lambda distribution: "0.5.2")
    fallback_calls = []
    module, stock = _install_fake_fla_module(monkeypatch, fallback_calls)

    @functools.wraps(stock)
    def third_party(*args, **kwargs):
        return args, kwargs

    module.causal_conv1d_update = third_party

    with pytest.raises(ImportError, match="does not match the expected owning module"):
        fla_api.accelerate_fla(verbose=False, targets="short_conv")

    assert module.causal_conv1d_update is third_party
    assert not fla_api.is_accelerated("shortconv")
