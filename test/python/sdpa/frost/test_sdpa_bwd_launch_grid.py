# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backward admission follows the actual grid, including auxiliary launches (#1573)."""

from types import SimpleNamespace

import pytest
import torch

import cudnn
from cudnn.sdpa.bwd import engines
from cudnn.sdpa.bwd.heuristics import recommend
from cudnn.sdpa.graph_analyzer import SdpaGraphFacts

pytestmark = pytest.mark.L0

# The flat LPT SM80 causal body is a positive control: a blanket B/H cap
# would hide the launch bug by also removing a valid implementation.
CASES = [
    ("sm80_d64", (8, 0), "sdpa_bwd_sm80_f16_d64", 64, False, False, False, False),
    ("sm80_natural", (8, 0), "sdpa_bwd_sm80_f16_two_subgroup", 128, False, False, False, False),
    ("sm80_flat", (8, 0), "sdpa_bwd_sm80_f16_two_subgroup", 128, True, False, False, True),
    ("sm80_deterministic", (8, 0), "sdpa_bwd_sm80_f16_two_subgroup", 128, True, True, False, False),
    ("sm80_thd", (8, 0), "sdpa_bwd_sm80_f16_two_subgroup", 128, True, False, True, False),
    ("sm120_dense", (12, 0), "sdpa_bwd_sm120_f16_fused", 128, False, False, False, False),
    ("sm121_dense", (12, 1), "sdpa_bwd_sm120_f16_fused", 128, True, False, False, False),
]


@pytest.fixture(autouse=True)
def compiler_version(monkeypatch):
    from cudnn.frost import buffers

    state = lambda: (True, ("nvidia-cutlass-dsl", "4.8.0"))
    monkeypatch.setattr(engines, "cutedsl_state", state)
    monkeypatch.setattr(buffers, "cutedsl_state", state)


def _facts(case, axis, extent, dtype):
    _, cc, _, d, causal, deterministic, thd, _ = case
    b, h = (extent, 1) if axis == "batch" else (1, extent)
    sq, sk = 64, 128
    ports = []
    for role in ("q", "k", "v", "o", "dO", "dQ", "dK", "dV"):
        n = sk if role in ("k", "v", "dK", "dV") else sq
        ports.append((role, (b, h, n, d), (n * h * d, d, h * d, 1)))
    stats_stride = (sq * h, 1, h, 1) if thd else (sq * h, sq, 1, 1)
    stats = SimpleNamespace(
        ragged_offset=object() if thd else None,
        get_dim=lambda: (b, h, sq, 1),
        get_stride=lambda: stats_stride,
        get_data_type=lambda: cudnn.data_type.FLOAT,
    )
    return SdpaGraphFacts(
        is_backward=True,
        b=b,
        h_q=h,
        h_kv=h,
        s_q=sq,
        s_kv=sk,
        d_qk=d,
        d_v=d,
        dtype=dtype,
        device_cc=cc,
        scale=d**-0.5,
        causal=causal,
        deterministic=deterministic,
        thd=thd,
        padded=thd,
        max_total_seq_len_q=64 if thd else None,
        max_total_seq_len_kv=128 if thd else None,
        port_layouts=tuple(ports),
        stats_t=stats,
    )


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("axis", ["batch", "heads"])
@pytest.mark.parametrize("extent", [65535, 65536])
@pytest.mark.parametrize("dtype", [cudnn.data_type.HALF, cudnn.data_type.BFLOAT16])
def test_graph_admission_and_recommendation(case, axis, extent, dtype):
    facts = _facts(case, axis, extent, dtype)
    name = case[2]
    caps = next(s.capabilities for s in engines.ENGINE_SPECS if s.name == name)
    reason = engines.mismatch(caps, facts)
    accepted = extent <= 65535 or case[-1]
    if accepted:
        assert reason is None, reason
    else:
        assert reason is not None and "65535" in reason and "grid" in reason, reason
        for mode in ("A", "FALLBACK"):
            assert recommend(mode, facts, {name: 20600}) == []


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("axis", ["batch", "heads"])
@pytest.mark.parametrize("extent", [65535, 65536])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_standalone_admission(case, axis, extent, dtype, monkeypatch):
    from cudnn.api_base import TensorDesc
    from cudnn.sdpa.bwd.api_dsl import SdpaBwdDslSm80, SdpaBwdDslSm120

    facts = _facts(case, axis, extent, cudnn.data_type.HALF)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: case[1])
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *a: SimpleNamespace(total_memory=80 * 2**30))
    descriptors = {}
    for role, shape, stride in facts.port_layouts:
        descriptors["sample_" + role.lower()] = TensorDesc(dtype, shape, stride, (3, 1, 2, 0), "cuda:0")
    descriptors["sample_stats"] = TensorDesc(
        torch.float32, facts.stats_t.get_dim(), facts.stats_t.get_stride(), (3, 1, 2, 0) if facts.thd else (3, 2, 1, 0), "cuda:0"
    )
    api_type = SdpaBwdDslSm80 if case[1] == (8, 0) else SdpaBwdDslSm120
    api = api_type(
        **descriptors,
        is_causal=facts.causal,
        deterministic=facts.deterministic,
        thd=facts.thd,
        max_total_seq_len_q=facts.max_total_seq_len_q,
        max_total_seq_len_kv=facts.max_total_seq_len_kv,
        thd_stats_token_major=facts.thd,
    )
    if extent <= 65535 or case[-1]:
        assert api.check_support()
    else:
        with pytest.raises(NotImplementedError, match="grid.*65535"):
            api.check_support()
