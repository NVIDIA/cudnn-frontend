# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-only boundaries for both dense d128 declaration entry points."""

from types import SimpleNamespace

import pytest
import torch

import cudnn
from cudnn.sdpa.bwd import engines
from cudnn.sdpa.bwd.heuristics import recommend
from cudnn.sdpa.graph_analyzer import SdpaGraphFacts

pytestmark = pytest.mark.L0


@pytest.fixture(autouse=True)
def compiler_and_device(monkeypatch):
    from cudnn import _pybind_module
    from cudnn.frost import buffers, device

    state = lambda: (True, ("nvidia-cutlass-dsl", "4.8.0"))
    monkeypatch.setattr(engines, "cutedsl_state", state)
    monkeypatch.setattr(buffers, "cutedsl_state", state)
    monkeypatch.setattr(device, "compute_capability", lambda _: (10, 0))
    monkeypatch.setattr(_pybind_module, "_SdpaBwdBinder", SimpleNamespace(supports_zero_workspace=True))


def _declarations(*, b=1, h=1, sq=128, sk=128, causal=False, scale=0.125):
    from cudnn.api_base import TensorDesc
    from cudnn.sdpa.bwd.api_dsl_sm100_d128 import SdpaBwdDslSm100D128

    samples = {}
    ports = []
    for role in ("q", "k", "v", "o", "dO", "stats", "dQ", "dK", "dV"):
        s = sk if role in ("k", "v", "dK", "dV") else sq
        shape = (b, h, s, 1 if role == "stats" else 128)
        stride = (h * s, s, 1, 1) if role == "stats" else (s * h * 128, 128, h * 128, 1)
        samples["sample_" + role.lower()] = TensorDesc(
            dtype=torch.float32 if role == "stats" else torch.bfloat16,
            shape=shape,
            stride=stride,
            stride_order=TensorDesc._compute_stride_order(shape, stride),
            device=torch.device("cuda", 0),
            name=role,
        )
        if role != "stats":
            ports.append((role, shape, stride))
    facts = SdpaGraphFacts(
        is_backward=True,
        b=b,
        h_q=h,
        h_kv=h,
        s_q=sq,
        s_kv=sk,
        d_qk=128,
        d_v=128,
        dtype=cudnn.data_type.BFLOAT16,
        device_cc=(10, 0),
        scale=scale,
        causal=causal,
        port_layouts=tuple(ports),
    )
    return facts, SdpaBwdDslSm100D128(**samples, is_causal=causal, scale_softmax=scale)


def _reason(facts):
    spec = next(s for s in engines.ENGINE_SPECS if s.name == "sdpa_bwd_sm100_d128")
    return engines.mismatch(spec.capabilities, facts)


@pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("entry", ["graph", "standalone"])
@pytest.mark.parametrize(
    "geometry,token",
    [
        *[({side: value}, "tile rounding") for side in ("sq", "sk") for value in (2**31 - 127, 2**31 - 1, 2**31)],
        ({"b": 2**30, "sq": 1}, "grid"),
        ({"b": 2**24}, "Stats index"),
        ({"h": 2**25, "sq": 1}, "tensor-map stride"),
    ],
)
def test_dense_declaration_rejects_unrepresentable_geometry(geometry, token, causal, entry):
    facts, api = _declarations(**geometry, causal=causal)
    if entry == "graph":
        reason = _reason(facts)
        assert reason is not None and token in reason
        assert recommend("A", facts, {"sdpa_bwd_sm100_d128": 20609}) == []
    else:
        with pytest.raises(NotImplementedError, match=token):
            api.check_support()


@pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize(
    "geometry,implementation",
    [
        ({}, "accumulation"),
        ({"sq": 129, "sk": 257}, "recompute"),
        ({"sq": 2**31 - 128}, "accumulation"),
        ({"sk": 2**31 - 128}, "recompute"),
        ({"b": 2**30 - 1, "sq": 1}, "recompute"),
        ({"b": 2**24 - 1}, "recompute"),
        ({"h": 2**25 - 1, "sq": 1}, "recompute"),
        ({"scale": -0.125}, "recompute"),
    ],
)
def test_dense_declaration_accepts_safe_boundaries(geometry, implementation, causal):
    facts, api = _declarations(**geometry, causal=causal)
    assert _reason(facts) is None
    assert [(p.engine_id, p.knobs) for p in recommend("A", facts, {"sdpa_bwd_sm100_d128": 20609})] == [(20609, None)]
    assert api.check_support()
    assert api._implementation == implementation
