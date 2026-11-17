# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Host admission boundaries for the SM120 empty-query backend defect."""

from dataclasses import replace
import cudnn
import pytest
from cudnn.sdpa.fwd.backend_guard import backend_engine_guard
from cudnn.sdpa.graph_analyzer import SdpaGraphFacts

pytestmark = pytest.mark.L0


def _facts(**changes):
    return replace(
        SdpaGraphFacts(b=3, h_q=4, h_kv=2, s_q=129, s_kv=257, d_qk=64, d_v=64, dtype=cudnn.data_type.BFLOAT16, padded=True, thd=True, device_cc=(12, 0)),
        **changes,
    )


@pytest.mark.parametrize("version,excluded", [(92300, True), (92302, True), (92400, True), (92500, True), (92501, True), (92600, False), (92700, False)])
def test_sm120_empty_q_version_boundary(monkeypatch, version, excluded):
    monkeypatch.setattr(cudnn, "backend_version", lambda: version)
    actual = backend_engine_guard(None, _facts())
    assert set(actual) == ({11} if excluded else set())
    if excluded:
        assert "zero-length Q" in actual[11]


@pytest.mark.parametrize("features", [{}, {"causal": True}, {"window_left": 96}, {"has_sink": True}, {"has_paged_kv": True}, {"has_cu_seq_len": True}])
@pytest.mark.parametrize("dtype", [cudnn.data_type.HALF, cudnn.data_type.BFLOAT16])
def test_sm120_empty_q_features_do_not_hide_runtime_lengths(monkeypatch, features, dtype):
    monkeypatch.setattr(cudnn, "backend_version", lambda: 92400)
    assert set(backend_engine_guard(None, _facts(dtype=dtype, **features))) == {11}


@pytest.mark.parametrize(
    "changes",
    [
        {"device_cc": (9, 0)},
        {"device_cc": (10, 0)},
        {"device_cc": (10, 3)},
        {"device_cc": (10, 7)},
        {"thd": False},
        {"padded": False},
        {"is_backward": True},
        {"dtype": cudnn.data_type.FP8_E4M3},
        {"invalid": "invalid graph"},
    ],
)
def test_sm120_empty_q_preserves_other_domains(monkeypatch, changes):
    monkeypatch.setattr(cudnn, "backend_version", lambda: 92400)
    expected = {9} if changes.get("device_cc") == (9, 0) else set()
    assert set(backend_engine_guard(None, _facts(**changes))) == expected


def test_sm120_empty_q_preserves_wide_v_guard(monkeypatch):
    monkeypatch.setattr(cudnn, "backend_version", lambda: 92400)
    assert set(backend_engine_guard(None, _facts(device_cc=(10, 0), d_v=256))) == {10, 18}
    assert backend_engine_guard(None, None) == {}
