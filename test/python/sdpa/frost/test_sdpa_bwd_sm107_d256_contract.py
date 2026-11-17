# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed D256 records preserve the native default and the diagnostic boundary."""

import pytest

import cudnn
from frost_test_utils import requires_dsl

pytestmark = [pytest.mark.L0, requires_dsl]


@pytest.fixture(autouse=True)
def host_target(monkeypatch):
    from cudnn.frost import buffers
    from cudnn.sdpa.bwd import engines

    monkeypatch.setattr(engines, "cutedsl_state", lambda: (True, ("nvidia-cutlass-dsl", "4.8.0")))
    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)


@pytest.mark.parametrize("enabled,legacy_profile", [(False, 2), (True, 1), (True, 2), (True, 99)])
def test_fixed_hosts_ignore_the_standalone_body_controls(monkeypatch, enabled, legacy_profile):
    from cudnn.sdpa.bwd import api_dsl_sm107
    from cudnn.sdpa.bwd.kernels.sm100.d256_f16_2x2.host import SdpaBwdDslSm107D256TwoByTwo
    from cudnn.sdpa.bwd.kernels.sm107.d256_f16_4x1.host import SdpaBwdDslSm107D256FourByOne
    from test_sdpa_bwd_dsl_sm107 import _adapter

    monkeypatch.setattr(api_dsl_sm107, "BWD_D256_2X2", enabled)
    monkeypatch.setattr(api_dsl_sm107, "BWD_D256_2X2_PROFILE", legacy_profile)
    for cls, path, profile in (
        (SdpaBwdDslSm107D256FourByOne, "sm107/d256_f16_4x1/kernel.py", 0),
        (SdpaBwdDslSm107D256TwoByTwo, "sm100/d256_f16_2x2/kernel.py", 2),
    ):
        api = _adapter(cls)
        assert api._kernel_file() == path
        assert api._datapath_2x2_profile() == api._template_params().datapath_2x2_profile == profile
        assert api._BINDING_NAME == "sdpa_bwd_sm107"


@pytest.mark.parametrize("thd", [False, True], ids=["dense", "packed"])
@pytest.mark.parametrize("dtype", [cudnn.data_type.HALF, cudnn.data_type.BFLOAT16], ids=["fp16", "bf16"])
def test_legacy_record_delegates_and_only_native_serves_packed(thd, dtype):
    from cudnn.sdpa.bwd.engines import ENGINE_SPECS, mismatch
    from test_sdpa_bwd_heuristics import _facts

    facts = _facts(d_qk=256, d_v=256, dtype=dtype, device_cc=(10, 7), thd=thd, max_total_seq_len_q=512, max_total_seq_len_kv=512)
    by_name = {s.name: s for s in ENGINE_SPECS}
    assert "owned by the explicit implementation" in mismatch(by_name["sdpa_bwd_sm107"].capabilities, facts)
    assert mismatch(by_name["sdpa_bwd_sm107_d256_f16_4x1"].capabilities, facts) is None
    assert (mismatch(by_name["sdpa_bwd_sm107_d256_f16_2x2"].capabilities, facts) is None) == (not thd)


def test_fixed_ids_and_default_placement(monkeypatch):
    from cudnn.engines.heuristics import is_backend_block, is_last_resort
    from cudnn.engines.manifest import MANIFEST
    from cudnn.sdpa.bwd import heuristics
    from test_sdpa_bwd_heuristics import _facts

    family = next(f for f in MANIFEST if f.name == "frost_sdpa_bwd")
    assert family.slots["sdpa_bwd_sm107"].slot == 4
    assert family.slots["sdpa_bwd_sm107_d256_f16_4x1"].slot == 32
    assert not family.slots["sdpa_bwd_sm107_d256_f16_4x1"].opt_in
    assert family.slots["sdpa_bwd_sm107_d256_f16_2x2"].slot == 33
    assert family.slots["sdpa_bwd_sm107_d256_f16_2x2"].opt_in
    offered = {name: family.engine_id + slot.slot for name, slot in family.slots.items()}
    facts = _facts(d_qk=256, d_v=256, dtype=cudnn.data_type.BFLOAT16, device_cc=(10, 7))
    monkeypatch.delenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", raising=False)
    ordinary = {name: eid for name, eid in offered.items() if not family.slots[name].opt_in}
    plans = heuristics.propose("A", facts, ordinary)
    assert [p.engine_id for p in plans if not is_backend_block(p) and not is_last_resort(p)] == [20632]
    assert is_backend_block(plans[0])
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    assert [p.engine_id for p in heuristics.recommend("A", facts, offered)] == [20632, 20633]
