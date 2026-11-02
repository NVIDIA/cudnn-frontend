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


def _declarations(*, b=1, h=1, sq=128, sk=128, causal=False, scale=0.125, d=128, window_left=None):
    from cudnn.api_base import TensorDesc
    from cudnn.sdpa.bwd.api_dsl_sm100_d128 import SdpaBwdDslSm100D128

    samples = {}
    ports = []
    for role in ("q", "k", "v", "o", "dO", "stats", "dQ", "dK", "dV"):
        s = sk if role in ("k", "v", "dK", "dV") else sq
        shape = (b, h, s, 1 if role == "stats" else d)
        stride = (h * s, s, 1, 1) if role == "stats" else (s * h * d, d, h * d, 1)
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
        d_qk=d,
        d_v=d,
        window_left=window_left,
        dtype=cudnn.data_type.BFLOAT16,
        device_cc=(10, 0),
        scale=scale,
        causal=causal,
        port_layouts=tuple(ports),
    )
    return facts, SdpaBwdDslSm100D128(**samples, is_causal=causal, scale_softmax=scale, window_size_left=window_left)


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
        ({"sq": 129, "sk": 257}, "accumulation"),
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


@pytest.mark.parametrize("entry", ["graph", "standalone"])
@pytest.mark.parametrize("window_left", [-1, 2**31 - 257])
def test_dense_window_declines_invalid_radius(entry, window_left):
    facts, api = _declarations(d=64, window_left=window_left)
    if entry == "graph":
        reason = _reason(facts)
        assert reason is not None and "window" in reason
    else:
        with pytest.raises(NotImplementedError, match="window"):
            api.check_support()


@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("window_left", [0, 1, 127, 128, 2**31 - 258])
def test_dense_window_accepts_safe_radius(d, window_left):
    facts, api = _declarations(d=d, window_left=window_left)
    assert _reason(facts) is None
    assert api.check_support()
    accumulation = d == 64 and window_left + 1 + 128 + 256 <= 2**31 - 1
    assert api._implementation == ("accumulation" if accumulation else "recompute")


@pytest.mark.parametrize("window_left", [True, 1.5])
def test_dense_window_graph_declines_noninteger_radius(window_left):
    # Standalone follows the shared adapter's integer normalization; graph
    # facts are already normalized and must carry an integer radius.
    facts, _ = _declarations(d=64, window_left=window_left)
    reason = _reason(facts)
    assert reason is not None and "window" in reason


@pytest.mark.parametrize(
    "d,sk,implementation", [(64, 4_194_176, "accumulation"), (64, 4_194_304, "recompute"), (64, 4_194_432, "recompute"), (128, 4_194_432, "accumulation")]
)
def test_accumulation_swizzle_uses_declared_head_dimension(d, sk, implementation):
    # Metadata-only: D64's host swizzle is twice D128's for Sq=2.
    # At Sk=2**22 its divisor reaches 2**31, despite a small launch grid.
    _, api = _declarations(d=d, sq=2, sk=sk, causal=True)
    assert api.check_support()
    assert api._implementation == implementation


@pytest.mark.parametrize("d,dv,last_safe", [(64, 64, 2_097_024), (128, 128, 4_194_176), (192, 128, 4_194_176)])
@pytest.mark.parametrize("overflow_rows", [0, 1, 128])
@pytest.mark.parametrize("long_side", ["kv", "q"])
def test_fp8_accumulation_swizzle_accounts_for_byte_payload(d, dv, last_safe, overflow_rows, long_side):
    from cudnn.api_base import TensorDesc
    from cudnn.sdpa.bwd.api_dsl_sm100_d128_fp8 import SdpaBwdDslSm100D128Fp8
    from cudnn.sdpa.bwd.api_dsl_sm100_d192_fp8 import SdpaBwdDslSm100D192Fp8

    # The paired schedulers use opposite operands for their L2 swizzle.
    # One row beyond the safe boundary already rounds to an unsafe tile.
    sq, sk = 2, last_safe + overflow_rows
    if long_side == "q":
        sq, sk = sk, sq
    samples = {}
    for role in ("q", "k", "v", "o", "do", "stats", "dq", "dk", "dv"):
        length = sk if role in ("k", "v", "dk", "dv") else sq
        width = 1 if role == "stats" else dv if role in ("v", "o", "do", "dv") else d
        shape, stride = (1, 1, length, width), (length * width, width, width, 1)
        dtype = torch.float32 if role == "stats" else torch.bfloat16 if role in ("dq", "dk", "dv") else torch.float8_e4m3fn
        samples["sample_" + role] = TensorDesc(
            dtype=dtype, shape=shape, stride=stride, stride_order=TensorDesc._compute_stride_order(shape, stride), device=torch.device("cuda", 0), name=role
        )
    api_type = SdpaBwdDslSm100D192Fp8 if d == 192 else SdpaBwdDslSm100D128Fp8
    api = api_type(**samples, is_causal=True)
    if overflow_rows:
        with pytest.raises(NotImplementedError, match="outside the accumulation envelope"):
            api.check_support()
    else:
        assert api.check_support()
