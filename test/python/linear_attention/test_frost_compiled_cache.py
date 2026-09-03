# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observable device contract for FROST linear-attention plans."""

import inspect
from types import SimpleNamespace

import pytest

from cudnn.engines.base import ExecutionContext
from cudnn.linear_attention.frost import engine as frost_engine

pytestmark = pytest.mark.L0


def setup_module():
    if "is current; build one plan per device" not in inspect.getsource(frost_engine.FrostLaPlan.execute):
        pytest.fail("the loaded FrostLaPlan has no device guard — a stale installed cudnn package is likely shadowing the source tree")


class _Compiled:
    def __init__(self, device):
        self.device = device
        self.plan_name = "test FROST LA plan"
        self.workspace_size = 0
        self.calls = []

    def workspace_bytes(self):
        return 0

    def bind(self, names):
        self.names = names

    def run(self, views, workspace, stream):
        self.calls.append((views, workspace, stream))


class _VariantPack:
    def __init__(self, device):
        self._device = device

    @property
    def device(self):
        return self._device

    def all_dense_layout(self):
        return True, -1

    def operands(self, indices):
        assert indices == []
        return ()


def _execution_context(device, stream=17):
    handle = SimpleNamespace(device=SimpleNamespace(ordinal=device))
    return ExecutionContext(handle=handle, stream=stream)


def _prepare_plan(monkeypatch, device):
    slots = SimpleNamespace(inputs={}, outputs={})
    monkeypatch.setattr(frost_engine, "bind_ports", lambda graph, variant_pack: {object(): slots})
    monkeypatch.setattr(frost_engine.Workspace, "over", lambda variant_pack, size, name: "workspace")
    compiled = _Compiled(device)
    return frost_engine.FrostLaPlan(compiled), compiled


def test_current_device_authorizes_matching_plan_even_when_the_handle_differs(monkeypatch):
    """The pack's device is the context actually bound for the launch; a
    disagreeing handle does not override it (frost gemm semantics)."""
    plan, compiled = _prepare_plan(monkeypatch, device=3)
    pack = _VariantPack(device=3)

    plan.execute(object(), pack, _execution_context(device=9))

    assert compiled.calls == [((), "workspace", 17)]


def test_rejects_plan_compiled_for_another_device_despite_a_matching_handle(monkeypatch):
    plan, compiled = _prepare_plan(monkeypatch, device=3)
    pack = _VariantPack(device=4)

    with pytest.raises(ValueError, match=r"plan was built for cuda:3, but cuda:4 is current"):
        plan.execute(object(), pack, _execution_context(device=3))

    assert compiled.calls == []


def test_handleless_execution_uses_the_variant_pack_device(monkeypatch):
    plan, compiled = _prepare_plan(monkeypatch, device=3)
    pack = _VariantPack(device=3)

    plan.execute(object(), pack, ExecutionContext(stream=23))

    assert compiled.calls == [((), "workspace", 23)]
