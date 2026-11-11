# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Packed Stats signed-span admission, without allocating large carriers."""

from types import SimpleNamespace

import pytest
import torch

from cudnn.sdpa import graph_analyzer as ga
from cudnn.sdpa.bwd.d128_thd_contract import validate_stats_span
from cudnn.sdpa.bwd.heuristics import recommend
from test_sdpa_bwd_d128_thd_eligibility import compiler_version, reason

pytestmark = pytest.mark.L0


@pytest.mark.parametrize("heads", [(2, 2), (4, 2)])
@pytest.mark.parametrize("boundary", ["wide_int64", "last_accepted", "first_rejected"])
def test_hn_stats_span_graph_and_standalone_boundary(monkeypatch, heads, boundary):
    from cudnn import _pybind_module
    from cudnn.api_base import TensorDesc
    from cudnn.frost import device
    from cudnn.sdpa.bwd.api_dsl_sm100_d128 import SdpaBwdDslSm100D128
    from test_sdpa_bwd_dsl_sm107 import _half_bwd_graph

    hq, hkv = heads
    maximum = (2**63 - 1) // (hq * 4)
    pitch = 2**33 if boundary == "wide_int64" else maximum + (boundary == "first_rejected")
    accepted = boundary != "first_rejected"
    if accepted:
        validate_stats_span(hq, pitch)
    else:
        with pytest.raises(ValueError, match="Stats span.*signed pointer"):
            validate_stats_span(hq, pitch)
    monkeypatch.setattr(ga, "_device_cc", lambda: (10, 0))
    monkeypatch.setattr(device, "compute_capability", lambda _: (10, 0))
    monkeypatch.setattr(_pybind_module, "_SdpaBwdBinder", SimpleNamespace(supports_zero_workspace=True))
    graph, tensors, _ = _half_bwd_graph(b=2, hq=hq, hkv=hkv, sq=129, skv=257, d=128, thd=True)
    tensors["stats"].set_stride([1, pitch, 1, 1])
    value = ga.analyze(graph)
    assert value is not None and value.thd
    observed = reason(value)
    assert observed is None if accepted else "Stats span" in observed and "signed pointer" in observed
    plans = recommend("A", value, {"sdpa_bwd_sm100_d128": 20609})
    assert (len(plans) == 1 and plans[0].engine_id == 20609) == accepted
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
        thd=True,
        max_total_seq_len_q=value.max_total_seq_len_q,
        max_total_seq_len_kv=value.max_total_seq_len_kv,
        thd_stats_head_stride=pitch,
        thd_ragged_offsets={name: (torch.int64, 1) for name in geometry},
    )
    if accepted:
        assert api.check_support()
    else:
        with pytest.raises(NotImplementedError, match="Stats span.*signed pointer"):
            api.check_support()
