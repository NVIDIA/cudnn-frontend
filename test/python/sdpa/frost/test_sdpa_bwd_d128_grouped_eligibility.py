# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Grouped d128 declaration gates through graph analysis and standalone API."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

import cudnn
from cudnn.sdpa import graph_analyzer as ga
from cudnn.sdpa.bwd import engines
from cudnn.sdpa.bwd.heuristics import recommend
from test_sdpa_bwd_d128_thd_eligibility import compiler_version, reason

pytestmark = pytest.mark.L0


def grouped_facts(*, thd, packing="TH1", **overrides):
    values = dict(
        is_backward=True,
        b=2,
        h_q=4,
        h_kv=2,
        s_q=129 if thd else 128,
        s_kv=257 if thd else 256,
        d_qk=128,
        d_v=128,
        dtype=cudnn.data_type.BFLOAT16,
        device_cc=(10, 0),
        scale=128**-0.5,
        thd=thd,
        padded=thd,
        max_total_seq_len_q=512 if thd else None,
        max_total_seq_len_kv=768 if thd else None,
    )
    values.update(overrides)
    b, h, hk, sq, sk = (values[key] for key in ("b", "h_q", "h_kv", "s_q", "s_kv"))
    ports = []
    for role in ("q", "k", "v", "o", "dO", "dQ", "dK", "dV"):
        n, heads = (sk, hk) if role in ("k", "v", "dK", "dV") else (sq, h)
        ports.append((role, (b, heads, n, 128), (7 if thd else n * heads * 128, 128, heads * 128, 1)))
    pitch = max(int(values["max_total_seq_len_q"] or 1), 1) + 16 if thd else sq
    stride = (pitch * h, 1, h, 1) if thd and packing == "TH1" else (pitch * h, pitch, 1, 1)
    stats = SimpleNamespace(
        ragged_offset=object() if thd else None, get_dim=lambda: (b, h, sq, 1), get_stride=lambda: stride, get_data_type=lambda: cudnn.data_type.FLOAT
    )
    return ga.SdpaGraphFacts(**values, port_layouts=tuple(ports), stats_t=stats)


@pytest.mark.parametrize(
    "thd,overrides,token",
    [
        (False, {"b": 65536}, "grid"),
        (True, {"b": 65536}, "grid"),
        (False, {"s_q": 2**31 - 127}, "tile rounding"),
        (True, {"s_q": 2**31 - 127}, "tile rounding"),
        (False, {"scale": 3e38}, "positive"),
        (True, {"scale": 3e38}, "positive"),
        (False, {"window_left": 16}, "sliding window"),
        (True, {"window_left": 16}, "sliding window"),
    ],
)
def test_grouped_declaration_declines_before_ranking(thd, overrides, token):
    value = grouped_facts(thd=thd, **overrides)
    assert token.lower() in reason(value).lower()
    assert recommend("A", value, {"sdpa_bwd_sm100_d128": 20609}) == []


@pytest.mark.parametrize("side", ["s_q", "s_kv"])
def test_grouped_dense_requires_complete_tiles(side):
    value = grouped_facts(thd=False, **{side: 129})
    assert "multiples of 128" in reason(value)
    assert reason(grouped_facts(thd=True, **{side: 129})) is None


def test_grouped_range_gates_and_widened_query_helpers():
    # The shared dense reducer forms an Int32 vector index before widening.
    max_sk = 2**31 // 16 - 128
    assert reason(grouped_facts(thd=False, b=1, h_q=2, h_kv=1, s_kv=max_sk)) is None
    assert "vector index" in reason(grouped_facts(thd=False, b=1, h_q=2, h_kv=1, s_kv=max_sk + 128))
    # External KV uses Hkv, but native partial maps use Hq and compact KV capacity.
    value = grouped_facts(thd=True, b=1, h_q=32768, h_kv=1, s_q=128, s_kv=131072, max_total_seq_len_q=128, max_total_seq_len_kv=131072)
    assert "native KV partial byte stride" in reason(value)
    assert "dimension" in reason(grouped_facts(thd=True, max_total_seq_len_kv=2**32))
    assert "positive" in reason(grouped_facts(thd=True, max_total_seq_len_q=None))


@pytest.mark.parametrize("packing", ["TH1", "HN"])
def test_grouped_packed_layout_uses_each_ports_head_count(packing):
    value = grouped_facts(thd=True, packing=packing, h_q=6, h_kv=2)
    assert reason(value) is None
    ports = list(value.port_layouts)
    role, dim, stride = ports[1]
    ports[1] = role, dim, (stride[0], stride[1], value.h_q * 128, stride[3])
    assert "compact BSHD" in reason(replace(value, port_layouts=tuple(ports)))
    if packing == "HN":
        value.stats_t.get_stride = lambda: (6 * 258, 258, 1, 1)
        assert "physical Q-token capacity" in reason(value)


@pytest.mark.parametrize(
    "thd,cc,scale,version,accepted",
    [
        (True, (10, 0), None, "4.8.0", True),
        (False, (10, 0), 0.125, "4.8.0", True),
        (False, (10, 3), 0.125, "4.8.0", False),
        (True, (10, 0), -0.125, "4.8.0", False),
        (False, (10, 0), 1e-100, "4.8.0", False),
        (True, (10, 0), 0.125, "4.7.4", False),
    ],
)
def test_grouped_graph_gate_matches_standalone(monkeypatch, thd, cc, scale, version, accepted):
    value = grouped_facts(thd=thd, packing="HN", device_cc=cc, scale=scale)
    _assert_grouped_graph_and_standalone_gate(monkeypatch, value, version, accepted)


@pytest.mark.parametrize("batch,accepted", [(32767, True), (32768, False), (32769, False)])
def test_grouped_dense_stats_int32_boundary(monkeypatch, batch, accepted):
    value = grouped_facts(thd=False, b=batch, h_q=2, h_kv=1, s_q=32768, s_kv=128)
    if not accepted:
        assert "Stats/Delta span" in reason(value)
        assert recommend("A", value, {"sdpa_bwd_sm100_d128": 20609}) == []
    _assert_grouped_graph_and_standalone_gate(monkeypatch, value, "4.8.0", accepted)


def _assert_grouped_graph_and_standalone_gate(monkeypatch, value, version, accepted):
    from cudnn import _pybind_module
    from cudnn.api_base import TensorDesc
    from cudnn.frost import buffers, device
    from cudnn.sdpa.bwd.api_dsl_sm100_d128 import SdpaBwdDslSm100D128

    state = lambda: (True, ("nvidia-cutlass-dsl", version))
    monkeypatch.setattr(engines, "cutedsl_state", state)
    monkeypatch.setattr(buffers, "cutedsl_state", state)
    monkeypatch.setattr(device, "compute_capability", lambda _: value.device_cc)
    monkeypatch.setattr(_pybind_module, "_SdpaBwdBinder", SimpleNamespace(supports_zero_workspace=True))
    thd = value.thd
    geometry = {name.lower(): (dim, stride) for name, dim, stride in value.port_layouts}
    geometry["stats"] = value.stats_t.get_dim(), value.stats_t.get_stride()
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
        thd=thd,
        scale_softmax=value.scale,
        max_total_seq_len_q=value.max_total_seq_len_q,
        max_total_seq_len_kv=value.max_total_seq_len_kv,
        thd_stats_head_stride=value.stats_t.get_stride()[1] if thd else None,
        thd_ragged_offsets={name: (torch.int32, 1) for name in geometry} if thd else None,
    )
    assert (reason(value) is None) == accepted
    if accepted:
        assert api.check_support()
        assert api._implementation == ("accumulation_grouped_thd" if thd else "accumulation_grouped_dense")
    else:
        with pytest.raises(NotImplementedError):
            api.check_support()


@pytest.mark.parametrize(
    "thd,heads,dtype,causal",
    [
        (False, (4, 2), torch.bfloat16, False),
        (False, (6, 2), torch.bfloat16, True),
        (True, (3, 1), torch.bfloat16, False),
        (True, (6, 2), torch.bfloat16, True),
    ],
)
def test_real_grouped_graph_accepts(monkeypatch, thd, heads, dtype, causal):
    from test_sdpa_bwd_dsl_sm107 import _half_bwd_graph

    monkeypatch.setattr(ga, "_device_cc", lambda: (10, 0))
    graph, _, _ = _half_bwd_graph(d=128, hq=heads[0], hkv=heads[1], thd=thd, dt=dtype, sq=129 if thd else 128, skv=257 if thd else 256, use_causal_mask=causal)
    value = ga.analyze(graph)
    assert value is not None and value.h_q != value.h_kv and value.thd == thd
    assert reason(value) is None
    assert [plan.engine_id for plan in recommend("A", value, {"sdpa_bwd_sm100_d128": 20609})] == [20609]


@pytest.mark.parametrize(
    "thd,cc,kwargs,token",
    [
        (False, (10, 3), {}, "cc 10.0"),
        (True, (10, 3), {}, "cc 10.0"),
        (False, (10, 0), {"scale": -0.125}, "positive"),
        (True, (10, 0), {"scale": -0.125}, "positive"),
        (False, (10, 0), {"sq": 129}, "multiples of 128"),
        (False, (10, 0), {"hq": 3, "hkv": 2}, "divisible"),
        (True, (10, 0), {"use_deterministic_algorithm": True}, "deterministic"),
        (True, (10, 0), {"use_causal_mask_bottom_right": True}, "bottom-right"),
    ],
)
def test_real_grouped_graph_declines(monkeypatch, thd, cc, kwargs, token):
    from test_sdpa_bwd_dsl_sm107 import _half_bwd_graph

    monkeypatch.setattr(ga, "_device_cc", lambda: cc)
    graph, _, _ = _half_bwd_graph(**{**dict(d=128, hq=4, hkv=2, thd=thd), **kwargs})
    value = ga.analyze(graph)
    assert value is not None
    assert token.lower() in reason(value).lower()
    assert recommend("A", value, {"sdpa_bwd_sm100_d128": 20609}) == []


@pytest.mark.parametrize(
    "shape,stride,token",
    [
        ((258,), (1,), "stats must be"),
        ((258, 4), (4, 1), "stats must be"),
        ((2, 4, 129, 1), (516, 1), "rank 4"),
        ((2, 4, 129, 1), (516, 1, 4, 1), None),
    ],
)
def test_grouped_thd_stats_descriptor_declines_safely(shape, stride, token):
    value = grouped_facts(thd=True)
    value.stats_t.get_dim = lambda: shape
    value.stats_t.get_stride = lambda: stride
    observed = reason(value)
    assert observed is None if token is None else token in observed


@pytest.mark.parametrize("head_pitch,missing_totals,token", [(528, False, None), (258, False, "physical Q-token capacity"), (528, True, "positive")])
def test_real_grouped_thd_declared_capacity_and_head_major_stats(monkeypatch, head_pitch, missing_totals, token):
    from test_sdpa_bwd_dsl_sm107 import _half_bwd_graph

    monkeypatch.setattr(ga, "_device_cc", lambda: (10, 0))
    original = cudnn.pygraph.sdpa_backward

    def declared_totals(graph, *args, **kwargs):
        if missing_totals:
            kwargs.pop("max_total_seq_len_q", None)
            kwargs.pop("max_total_seq_len_kv", None)
        else:
            kwargs.update(max_total_seq_len_q=512, max_total_seq_len_kv=768)
        return original(graph, *args, **kwargs)

    monkeypatch.setattr(cudnn.pygraph, "sdpa_backward", declared_totals)
    graph, tensors, _ = _half_bwd_graph(d=128, hq=6, hkv=2, thd=True, sq=129, skv=257)
    tensors["stats"].set_stride([6 * head_pitch, head_pitch, 1, 1])
    value = ga.analyze(graph)
    assert value is not None and value.thd
    observed = reason(value)
    assert observed is None if token is None else token in observed
    plans = recommend("A", value, {"sdpa_bwd_sm100_d128": 20609})
    assert (len(plans) == 1 and plans[0].engine_id == 20609) == (token is None)


@pytest.mark.parametrize("thd", [False, True])
@pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("heads", [(4, 2), (3, 1), (2, 2)], ids=["gqa", "mqa", "mha-preserved"])
def test_fp16_grouped_declines_while_mha_remains_supported(monkeypatch, thd, causal, heads):
    from cudnn import _pybind_module
    from cudnn.api_base import TensorDesc
    from cudnn.frost import device
    from cudnn.sdpa.bwd.api_dsl_sm100_d128 import SdpaBwdDslSm100D128
    from test_sdpa_bwd_dsl_sm107 import _half_bwd_graph

    monkeypatch.setattr(ga, "_device_cc", lambda: (10, 0))
    monkeypatch.setattr(device, "compute_capability", lambda _: (10, 0))
    monkeypatch.setattr(_pybind_module, "_SdpaBwdBinder", SimpleNamespace(supports_zero_workspace=True))
    graph, _, _ = _half_bwd_graph(
        d=128, hq=heads[0], hkv=heads[1], thd=thd, dt=torch.float16, sq=129 if thd else 128, skv=257 if thd else 256, use_causal_mask=causal
    )
    value = ga.analyze(graph)
    assert value is not None
    accepted = heads[0] == heads[1]
    observed = reason(value)
    assert observed is None if accepted else "BF16 grouped attention only" in observed
    plans = recommend("A", value, {"sdpa_bwd_sm100_d128": 20609})
    assert (len(plans) == 1 and plans[0].engine_id == 20609) == accepted
    geometry = {name.lower(): (dim, stride) for name, dim, stride in value.port_layouts}
    geometry["stats"] = value.stats_t.get_dim(), value.stats_t.get_stride()
    samples = {
        "sample_"
        + name: TensorDesc(
            dtype=torch.float32 if name == "stats" else torch.float16,
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
        thd=thd,
        is_causal=causal,
        max_total_seq_len_q=value.max_total_seq_len_q,
        max_total_seq_len_kv=value.max_total_seq_len_kv,
        thd_stats_token_major=thd,
        thd_ragged_offsets={name: (torch.int32, 1) for name in geometry} if thd else None,
    )
    if accepted:
        assert api.check_support()
    else:
        with pytest.raises(NotImplementedError, match="BF16 grouped attention only"):
            api.check_support()
