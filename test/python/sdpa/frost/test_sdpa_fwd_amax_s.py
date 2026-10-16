# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A declared Amax_S output keeps the FROST FP8 rows' decline and logs the deprecation once per process."""

import logging
from types import SimpleNamespace

import pytest

import cudnn
from cudnn.sdpa.fwd import engines
from cudnn.sdpa.graph_analyzer import SdpaGraphFacts

pytestmark = pytest.mark.L0

AMAX_S_REASON = "graph requests the Amax_S output, which the FROST engines do not produce"


def _fp8_facts(**kw):
    base = dict(
        b=1,
        h_q=16,
        h_kv=2,
        s_q=4096,
        s_kv=4096,
        d_qk=256,
        d_v=256,
        dtype=cudnn.data_type.FP8_E4M3,
        dtype_o=cudnn.data_type.BFLOAT16,
        is_fp8=True,
        causal=True,
        bottom_right=True,
        device_cc=(10, 0),
        device_sm_count=148,
    )
    base.update(kw)
    return SdpaGraphFacts(**base)


@pytest.mark.parametrize("row, cc", [("sdpa_fwd_prefill_sm100_fp8", (10, 0)), ("sdpa_fwd_prefill_sm107_fp8", (10, 7))])
def test_declared_amax_s_declines_and_warns_once(row, cc, caplog, monkeypatch):
    monkeypatch.setattr(engines, "_amax_s_warned", False)
    spec = next(s for s in engines.ENGINE_SPECS if s.name == row)
    facts = _fp8_facts(device_cc=cc, amax_s_t=SimpleNamespace())
    with caplog.at_level(logging.WARNING, logger=engines.__name__):
        assert engines.mismatch(spec.capabilities, facts) == AMAX_S_REASON
        assert engines.mismatch(spec.capabilities, facts) == AMAX_S_REASON
    warned = [r for r in caplog.records if "deprecated Amax_S" in r.getMessage()]
    assert len(warned) == 1


def test_virtual_amax_s_does_not_warn(caplog, monkeypatch):
    monkeypatch.setattr(engines, "_amax_s_warned", False)
    spec = next(s for s in engines.ENGINE_SPECS if s.name == "sdpa_fwd_prefill_sm100_fp8")
    with caplog.at_level(logging.WARNING, logger=engines.__name__):
        assert engines.mismatch(spec.capabilities, _fp8_facts()) != AMAX_S_REASON
    assert not [r for r in caplog.records if "Amax_S" in r.getMessage()]


def test_amax_s_on_a_device_outside_the_row_does_not_warn(caplog, monkeypatch):
    # The arch check precedes the Amax_S rule, so a device no FROST FP8 row serves stays silent.
    monkeypatch.setattr(engines, "_amax_s_warned", False)
    spec = next(s for s in engines.ENGINE_SPECS if s.name == "sdpa_fwd_prefill_sm100_fp8")
    with caplog.at_level(logging.WARNING, logger=engines.__name__):
        assert engines.mismatch(spec.capabilities, _fp8_facts(device_cc=(9, 0), amax_s_t=SimpleNamespace())) != AMAX_S_REASON
    assert not [r for r in caplog.records if "Amax_S" in r.getMessage()]
