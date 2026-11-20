# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise JAX trace placement with real tracers and modeled GPU properties."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

jax = pytest.importorskip("jax", minversion="0.9.1")
pytest.importorskip("cudnn._jax", exc_type=ImportError)

from cudnn.frost import device
from cudnn.linear_attention import _jax_execution

pytestmark = pytest.mark.L0


@dataclass(frozen=True)
class _Gpu:
    local_hardware_id: int
    platform = "gpu"


@dataclass(frozen=True)
class _PlacedArray:
    device: _Gpu

    def devices(self):
        return {self.device}


@pytest.mark.parametrize("capture_every_operand", [True, False], ids=["fully-captured", "partially-traced"])
def test_choose_trace_device_independently_of_captured_arrays(monkeypatch, capture_every_operand):
    sm100, sm103, sm90 = (_Gpu(i) for i in range(3))
    capabilities = {0: (10, 0), 1: (10, 3), 2: (9, 0)}
    config = SimpleNamespace(jax_default_device=None)
    runtime = SimpleNamespace(Array=_PlacedArray, core=jax.core, config=config, local_devices=lambda: (sm100, sm103, sm90))
    monkeypatch.setattr(_jax_execution, "jax", runtime)
    monkeypatch.setattr(device, "compute_capability", capabilities.__getitem__)
    captured = _PlacedArray(sm100)
    assert _jax_execution._device((captured,)) == (0, "sm_100a")

    selected = []

    @jax.make_jaxpr
    def trace(*args):
        selected.append(_jax_execution._device((captured, *args)))
        return args

    args = () if capture_every_operand else (jax.ShapeDtypeStruct((1,), "float32"),)
    with pytest.raises(ValueError, match="cannot choose between SM100 and SM103"):
        trace(*args)

    config.jax_default_device = sm103
    trace(*args)
    assert selected == [(1, "sm_103a")]
    assert _jax_execution._device((captured,)) == (0, "sm_100a")


def test_tracing_ignores_an_unrelated_gpu_when_the_default_is_supported(monkeypatch):
    sm103, sm90 = _Gpu(1), _Gpu(2)
    capabilities = {1: (10, 3), 2: (9, 0)}
    runtime = SimpleNamespace(Array=jax.Array, core=jax.core, config=SimpleNamespace(jax_default_device=None), local_devices=lambda: (sm103, sm90))
    monkeypatch.setattr(_jax_execution, "jax", runtime)
    monkeypatch.setattr(device, "compute_capability", capabilities.__getitem__)

    @jax.make_jaxpr
    def trace(x):
        assert _jax_execution._device((x,)) == (1, "sm_103a")
        return x

    trace(jax.ShapeDtypeStruct((1,), "float32"))
