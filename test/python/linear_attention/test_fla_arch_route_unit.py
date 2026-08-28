# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU/mock architecture-routing tests for the FLA linear-attention shims."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

gated_delta_rule = importlib.import_module("cudnn.fla.gated_delta_rule")
kda = importlib.import_module("cudnn.fla.kda")

pytestmark = pytest.mark.L0


@pytest.mark.parametrize(
    "shim_module,factory",
    [
        pytest.param(gated_delta_rule, gated_delta_rule.make_chunk_gated_delta_rule, id="gdn"),
        pytest.param(kda, kda.make_chunk_kda, id="kda"),
    ],
)
def test_sm110_calls_saved_fla_once_without_trying_native(monkeypatch, shim_module, factory):
    q = SimpleNamespace(is_cuda=True, device=object())
    k, v, g, beta = object(), object(), object(), object()
    expected = object()
    original_calls = []
    native_calls = []

    def original(*args, **kwargs):
        original_calls.append((args, kwargs))
        return expected

    def native(*args, **kwargs):
        native_calls.append((args, kwargs))
        pytest.fail("SM110 must not enter the native/cuTile route")

    monkeypatch.setattr(shim_module.torch.cuda, "get_device_capability", lambda device: (11, 0))
    monkeypatch.setattr(shim_module, "_to_native", native)

    result = factory(original)(q, k, v, g, beta)

    assert result is expected
    assert len(original_calls) == 1
    assert original_calls[0][0] == (q, k, v, g, beta)
    assert native_calls == []
    assert shim_module.last_path() == "fallback:sm11"


@pytest.mark.parametrize(
    "shim_module,factory",
    [
        pytest.param(gated_delta_rule, gated_delta_rule.make_chunk_gated_delta_rule, id="gdn"),
        pytest.param(kda, kda.make_chunk_kda, id="kda"),
    ],
)
@pytest.mark.parametrize("capability", [(10, 0), (10, 3), (10, 7)], ids=["sm100", "sm103", "sm107"])
def test_validated_sm10_variants_keep_native_route(monkeypatch, shim_module, factory, capability):
    q = SimpleNamespace(is_cuda=True, device=object())
    k, v, g, beta = object(), object(), object(), object()
    expected = object()
    original_calls = []
    native_calls = []

    def original(*args, **kwargs):
        original_calls.append((args, kwargs))
        pytest.fail("validated SM10x must keep the native route")

    def native(*args, **kwargs):
        native_calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(shim_module.torch.cuda, "get_device_capability", lambda device: capability)
    monkeypatch.setattr(shim_module, "_to_native", native)

    result = factory(original)(q, k, v, g, beta)

    assert result is expected
    assert original_calls == []
    assert len(native_calls) == 1
    assert shim_module.last_path() == "native"
