# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Host contract for the versioned Hopper empty-KV engine exclusion."""

from dataclasses import replace

import cudnn
import pytest

from cudnn.sdpa.fwd.backend_guard import backend_engine_guard
from cudnn.sdpa.graph_analyzer import SdpaGraphFacts

pytestmark = pytest.mark.L0


def _facts(**kwargs):
    return replace(
        SdpaGraphFacts(b=3, h_q=4, h_kv=2, s_q=129, s_kv=257, d_qk=128, d_v=128, dtype=cudnn.data_type.BFLOAT16, padded=True, device_cc=(9, 0)), **kwargs
    )


@pytest.mark.parametrize("version,excluded", [(92301, False), (92400, True), (92401, True), (92500, False), (92501, False), (93000, False)])
def test_zero_kv_guard_version_boundary(monkeypatch, version, excluded):
    monkeypatch.setattr(cudnn, "backend_version", lambda: version)
    actual = backend_engine_guard(None, _facts())
    assert set(actual) == ({9} if excluded else set())
    if excluded:
        assert "zero-length KV" in actual[9]


@pytest.mark.parametrize("layout", [{}, {"thd": True}, {"has_paged_kv": True}, {"has_cu_seq_len": True}])
@pytest.mark.parametrize("dtype", [cudnn.data_type.HALF, cudnn.data_type.BFLOAT16])
def test_zero_kv_guard_applies_to_declared_kv_lengths(monkeypatch, layout, dtype):
    monkeypatch.setattr(cudnn, "backend_version", lambda: 92400)
    assert set(backend_engine_guard(None, _facts(dtype=dtype, **layout))) == {9}


@pytest.mark.parametrize(
    "overrides",
    [
        {"device_cc": (8, 0)},
        {"device_cc": (8, 9)},
        {"device_cc": (10, 0)},
        {"device_cc": (10, 3)},
        {"device_cc": (10, 7)},
        {"device_cc": (12, 0)},
        {"padded": False},
        {"is_backward": True},
        {"dtype": cudnn.data_type.FP8_E4M3},
        {"invalid": "invalid input"},
    ],
)
def test_zero_kv_guard_preserves_other_domains(monkeypatch, overrides):
    monkeypatch.setattr(cudnn, "backend_version", lambda: 92400)
    assert backend_engine_guard(None, _facts(**overrides)) == {}


def test_zero_kv_guard_keeps_wide_v_exclusion(monkeypatch):
    monkeypatch.setattr(cudnn, "backend_version", lambda: 92400)
    assert set(backend_engine_guard(None, _facts(device_cc=(10, 0), d_v=256))) == {10, 18}
    assert backend_engine_guard(None, None) == {}
