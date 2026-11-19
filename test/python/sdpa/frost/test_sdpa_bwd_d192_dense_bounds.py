# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Metadata-only D192 graph/standalone parity; no device allocation or launch."""

from types import SimpleNamespace

import pytest
import torch

import cudnn
from cudnn.api_base import TensorDesc
from cudnn.sdpa.bwd import engines
from cudnn.sdpa.bwd.heuristics import recommend
from cudnn.sdpa.graph_analyzer import SdpaGraphFacts

pytestmark = pytest.mark.L0

_DTYPES = {torch.float16: cudnn.data_type.HALF, torch.bfloat16: cudnn.data_type.BFLOAT16, torch.float8_e4m3fn: cudnn.data_type.FP8_E4M3}
_ROUTES = [pytest.param(cc, dtype, dtype, id=f"sm{cc[0]}{cc[1]}-{dtype}") for cc in ((10, 0), (10, 3)) for dtype in (torch.float16, torch.bfloat16)] + [
    pytest.param(cc, torch.float8_e4m3fn, grad, id=f"sm{cc[0]}{cc[1]}-fp8-{grad}")
    for cc in ((10, 0), (10, 3), (10, 7))
    for grad in (torch.float16, torch.bfloat16, torch.float8_e4m3fn)
]


def _declarations(monkeypatch, cc, dtype, grad, *, b=1, h=1, sq=2, sk=3, scale=None, causal=False, deterministic=False):
    from cudnn.frost import buffers, device
    from cudnn.sdpa.bwd.api_dsl_sm100_d192 import SdpaBwdDslSm100D192
    from cudnn.sdpa.bwd.api_dsl_sm100_d192_fp8 import SdpaBwdDslSm100D192Fp8

    state = lambda: (True, ("nvidia-cutlass-dsl", "4.8.1"))
    for module in (engines, buffers):
        monkeypatch.setattr(module, "cutedsl_state", state)
        monkeypatch.setattr(module, "cutedsl_arch_requirement_error", lambda cc: None)
    monkeypatch.setattr(device, "compute_capability", lambda _: cc)
    monkeypatch.setattr(cudnn._pybind_module, "_SdpaBwdBinder", SimpleNamespace(supports_zero_workspace=True))

    samples, ports = {}, []
    for role in ("q", "k", "v", "o", "dO", "stats", "dQ", "dK", "dV"):
        length = sk if role in ("k", "v", "dK", "dV") else sq
        width = 1 if role == "stats" else 128 if role in ("v", "o", "dO", "dV") else 192
        shape = (b, h, length, width)
        stride = (h * length, length, 1, 1) if role == "stats" else (length * h * width, width, h * width, 1)
        samples["sample_" + role.lower()] = TensorDesc(
            dtype=torch.float32 if role == "stats" else grad if role in ("dQ", "dK", "dV") else dtype,
            shape=shape,
            stride=stride,
            stride_order=TensorDesc._compute_stride_order(shape, stride),
            device=torch.device("cuda", 0),
            name=role,
        )
        if role != "stats":
            ports.append((role, shape, stride))
    fp8 = dtype == torch.float8_e4m3fn
    facts = SdpaGraphFacts(
        is_backward=True,
        b=b,
        h_q=h,
        h_kv=h,
        s_q=sq,
        s_kv=sk,
        d_qk=192,
        d_v=128,
        dtype=_DTYPES[dtype],
        dtype_o=_DTYPES[grad],
        is_fp8=fp8,
        device_cc=cc,
        scale=scale,
        causal=causal,
        deterministic=deterministic,
        port_layouts=tuple(ports),
    )
    api_type = SdpaBwdDslSm100D192Fp8 if fp8 else SdpaBwdDslSm100D192
    api = api_type(**samples, is_causal=causal, deterministic=deterministic, scale_softmax=scale)
    name, engine_id = ("sdpa_bwd_sm100_d192_fp8", 20613) if fp8 else ("sdpa_bwd_sm100_d192", 20612)
    spec = next(s for s in engines.ENGINE_SPECS if s.name == name)
    return facts, api, spec, {name: engine_id}


def _check_entry(facts, api, spec, offered, entry, expected):
    if entry == "standalone":
        if expected is None:
            assert api.check_support()
        else:
            with pytest.raises(NotImplementedError, match=expected):
                api.check_support()
    else:
        reason = engines.mismatch(spec.capabilities, facts)
        plans = recommend("A", facts, offered)
        if expected is None:
            assert reason is None, reason
            assert [p.engine_id for p in plans] == list(offered.values())
        else:
            assert reason is not None and expected in reason, reason
            assert plans == []


@pytest.mark.parametrize("cc,dtype,grad", _ROUTES)
@pytest.mark.parametrize("entry", ["graph", "standalone"])
@pytest.mark.parametrize(
    "geometry,expected",
    [
        ({"b": 65535}, None),
        ({"b": 65536}, "accumulation envelope"),
        ({"h": 65535}, None),
        ({"h": 65536}, "accumulation envelope"),
        ({"sq": 2**31 - 127}, "tile rounding"),
        ({"sk": 2**31 - 127}, "tile rounding"),
        ({"b": 65535, "h": 65535}, "work-id range"),
        ({"b": 4096, "h": 4096, "sq": 128}, "Stats index"),
        ({"b": 128, "h": 128, "sq": 130944}, None),
        ({"b": 128, "h": 128, "sq": 130945}, "accumulation envelope"),
        ({"h": 32768, "sk": 131072}, "tensor-map stride"),
        ({"scale": 1e-50}, "accumulation envelope"),
        ({"scale": 3e38}, "accumulation envelope"),
        ({"scale": 0.125, "sq": 129, "sk": 257, "causal": True}, None),
    ],
)
def test_dense_bounds_match_standalone(monkeypatch, cc, dtype, grad, entry, geometry, expected):
    _check_entry(*_declarations(monkeypatch, cc, dtype, grad, **geometry), entry, expected)


@pytest.mark.parametrize("cc,dtype,grad", _ROUTES)
@pytest.mark.parametrize("entry", ["graph", "standalone"])
@pytest.mark.parametrize("long_side", ["sq", "sk"])
@pytest.mark.parametrize("overflow", [False, True])
def test_l2_divisor_boundary_uses_payload_width_and_both_fp8_schedulers(monkeypatch, cc, dtype, grad, entry, long_side, overflow):
    # FP8's byte payload doubles the swizzle. Its dQ scheduler also checks
    # K/V's footprint, which the half accumulation implementation does not use.
    fp8 = dtype == torch.float8_e4m3fn
    last_safe = 4_194_176 if fp8 else 8_388_480
    geometry = {"sq": 2, "sk": 2, long_side: last_safe + int(overflow), "causal": True, "deterministic": fp8}
    expected = "accumulation envelope" if overflow and (fp8 or long_side == "sk") else None
    _check_entry(*_declarations(monkeypatch, cc, dtype, grad, **geometry), entry, expected)
