# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Declaration-only d128 packed eligibility and standalone parity."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

import cudnn
from cudnn.sdpa.bwd import engines
from cudnn.sdpa.bwd.heuristics import recommend
from cudnn.sdpa.graph_analyzer import SdpaGraphFacts

pytestmark = pytest.mark.L0


@pytest.fixture(autouse=True)
def compiler_version(monkeypatch):
    from cudnn.frost import buffers

    state = lambda: (True, ("nvidia-cutlass-dsl", "4.8.0"))
    monkeypatch.setattr(engines, "cutedsl_state", state)
    monkeypatch.setattr(buffers, "cutedsl_state", state)


def facts(*, packing="TH1", **overrides):
    values = dict(
        is_backward=True,
        b=2,
        h_q=2,
        h_kv=2,
        s_q=129,
        s_kv=257,
        d_qk=128,
        d_v=128,
        dtype=cudnn.data_type.BFLOAT16,
        device_cc=(10, 0),
        scale=128**-0.5,
        thd=True,
        padded=True,
        max_total_seq_len_q=512,
        max_total_seq_len_kv=768,
    )
    values.update(overrides)
    b, h, sq, sk = values["b"], values["h_q"], values["s_q"], values["s_kv"]
    ports = []
    for role in ("q", "k", "v", "o", "dO", "dQ", "dK", "dV"):
        n = sk if role in ("k", "v", "dK", "dV") else sq
        ports.append((role, (b, h, n, 128), (7, 128, h * 128, 1)))
    pitch = max(int(values["max_total_seq_len_q"] or 1), 1) + 16
    stride = (pitch * h, 1, h, 1) if packing == "TH1" else (pitch * h, pitch, 1, 1)
    stats = SimpleNamespace(ragged_offset=object(), get_dim=lambda: (b, h, sq, 1), get_stride=lambda: stride, get_data_type=lambda: cudnn.data_type.FLOAT)
    return SdpaGraphFacts(**values, port_layouts=tuple(ports), stats_t=stats)


def reason(value):
    spec = next(s for s in engines.ENGINE_SPECS if s.name == "sdpa_bwd_sm100_d128")
    return engines.mismatch(spec.capabilities, value)


@pytest.mark.parametrize(
    "packing,dtype,causal",
    [
        ("TH1", cudnn.data_type.BFLOAT16, False),
        ("HN", cudnn.data_type.HALF, True),
    ],
)
def test_thd_admission_and_ranking(packing, dtype, causal):
    value = facts(packing=packing, dtype=dtype, causal=causal)
    assert reason(value) is None
    plans = recommend("A", value, {"sdpa_bwd_sm100_d128": 20609})
    assert [(plan.engine_id, plan.knobs) for plan in plans] == [(20609, None)]


@pytest.mark.parametrize(
    "overrides,token",
    [
        ({"device_cc": (10, 3)}, "cc 10.0"),
        ({"scale": float("inf")}, "positive"),
        ({"scale": float("nan")}, "positive"),
        ({"scale": 0.0}, "attn_scale = 0"),
        ({"max_total_seq_len_q": None}, "positive"),
        ({"max_total_seq_len_kv": 0}, "positive"),
        ({"s_q": 2**31 - 127}, "tile rounding"),
        ({"b": 65536}, "grid"),
        ({"max_total_seq_len_q": 2**32}, "dimension"),
        ({"max_total_seq_len_q": 2**31}, "byte stride"),
        ({"bottom_right": True, "causal": True}, "bottom-right"),
        ({"deterministic": True}, "deterministic"),
    ],
)
def test_thd_rejects_before_ranking(overrides, token):
    value = facts(**overrides)
    assert token.lower() in reason(value).lower()
    assert recommend("A", value, {"sdpa_bwd_sm100_d128": 20609}) == []


@pytest.mark.parametrize("version", [("nvidia-cutlass-dsl", "4.7.4"), ("cutlass", "4.8.0"), ("nvidia-cutlass-dsl", "unknown")])
def test_thd_requires_public_dsl_48(monkeypatch, version):
    monkeypatch.setattr(engines, "cutedsl_state", lambda: (True, version))
    assert "nvidia-cutlass-dsl >= 4.8.0" in reason(facts())


@pytest.mark.parametrize("cc", [(10, 0), (10, 3)])
def test_dense_keeps_older_compiler_and_negative_scale(monkeypatch, cc):
    monkeypatch.setattr(engines, "cutedsl_state", lambda: (True, ("nvidia-cutlass-dsl", "4.7.4")))
    value = SdpaGraphFacts(is_backward=True, b=2, h_q=2, h_kv=2, s_q=129, s_kv=257, d_qk=128, d_v=128, dtype=cudnn.data_type.HALF, device_cc=cc, scale=-0.1)
    assert reason(value) is None


def test_packed_batch_pitch_is_ignored_but_token_pitch_is_not():
    value = facts(s_q=1)
    assert reason(value) is None
    ports = list(value.port_layouts)
    name, dim, stride = ports[0]
    ports[0] = (name, dim, (stride[0], stride[1], stride[2] + 128, stride[3]))
    assert "compact BSHD" in reason(replace(value, port_layouts=tuple(ports)))


def test_head_major_stats_covers_physical_capacity_with_gaps():
    value = facts(packing="HN")
    value.stats_t.get_stride = lambda: (2 * 258, 258, 1, 1)
    assert "physical Q-token capacity" in reason(value)


@pytest.mark.parametrize("side", ["s_q", "s_kv"])
def test_tile_rounding_boundary(side):
    assert reason(facts(**{side: 2**31 - 128})) is None
    assert "tile rounding" in reason(facts(**{side: 2**31 - 127}))


@pytest.mark.parametrize("scale,accepted", [(None, True), (0.1, True), (-0.1, False), (1e-100, False), (3e38, False)])
def test_thd_scale_gate_matches_standalone(monkeypatch, scale, accepted):
    import torch
    from cudnn import _pybind_module
    from cudnn.api_base import TensorDesc
    from cudnn.frost import device
    from cudnn.sdpa.bwd.api_dsl_sm100_d128 import SdpaBwdDslSm100D128

    monkeypatch.setattr(device, "compute_capability", lambda _: (10, 0))
    monkeypatch.setattr(_pybind_module, "_SdpaBwdBinder", SimpleNamespace(supports_zero_workspace=True))
    value = facts(scale=scale)
    geometry = {name.lower(): (dim, stride) for name, dim, stride in value.port_layouts}
    geometry["stats"] = (value.stats_t.get_dim(), value.stats_t.get_stride())
    samples = {
        "sample_"
        + name: TensorDesc(
            dtype=torch.float32 if name == "stats" else torch.bfloat16,
            shape=dim,
            stride=stride,
            stride_order=TensorDesc._compute_stride_order(dim, stride),
            device=torch.device("cuda", 0),
            name=name,
        )
        for name, (dim, stride) in geometry.items()
    }
    api = SdpaBwdDslSm100D128(
        **samples,
        thd=True,
        scale_softmax=scale,
        max_total_seq_len_q=512,
        max_total_seq_len_kv=768,
        thd_stats_token_major=True,
        thd_ragged_offsets={name: (torch.int32, 1) for name in geometry},
    )
    assert (reason(value) is None) == accepted
    if accepted:
        assert api.check_support()
    else:
        with pytest.raises(NotImplementedError, match="positive"):
            api.check_support()


@pytest.mark.parametrize(
    "shape,stride,token",
    [
        ((258,), (1,), "stats must be"),
        ((258, 2), (2, 1), "stats must be"),
        ((2, 2, 129, 1), (258, 1), "rank 4"),
        ((2, 2, 129, 1), (258, 1, 2, 1), None),
    ],
)
def test_thd_stats_malformed_descriptor_declines(shape, stride, token):
    value = facts()
    value.stats_t.get_dim = lambda: shape
    value.stats_t.get_stride = lambda: stride
    observed = reason(value)
    if token is None:
        assert observed is None
    else:
        assert token in observed
    plans = recommend("A", value, {"sdpa_bwd_sm100_d128": 20609})
    assert (len(plans) == 1 and plans[0].engine_id == 20609) == (token is None)


@pytest.mark.parametrize(
    "cc,scale,missing_totals,token",
    [((10, 0), 0.125, False, None), ((10, 3), 0.125, False, "cc 10.0"), ((10, 0), -0.125, False, "positive"), ((10, 0), 0.125, True, "positive")],
)
def test_real_thd_graph_admission(monkeypatch, cc, scale, missing_totals, token):
    from cudnn.sdpa import graph_analyzer
    from test_sdpa_bwd_dsl_sm107 import _half_bwd_graph

    monkeypatch.setattr(graph_analyzer, "_device_cc", lambda: cc)
    if missing_totals:
        original = cudnn.pygraph.sdpa_backward

        def without_totals(graph, *args, **kwargs):
            kwargs.pop("max_total_seq_len_q", None)
            kwargs.pop("max_total_seq_len_kv", None)
            return original(graph, *args, **kwargs)

        monkeypatch.setattr(cudnn.pygraph, "sdpa_backward", without_totals)
    graph, _, _ = _half_bwd_graph(d=128, thd=True, scale=scale)
    value = graph_analyzer.analyze(graph)
    assert value is not None and value.thd
    observed = reason(value)
    if token is None:
        assert observed is None
    else:
        assert token in observed
    plans = recommend("A", value, {"sdpa_bwd_sm100_d128": 20609})
    assert (len(plans) == 1 and plans[0].engine_id == 20609) == (token is None)
