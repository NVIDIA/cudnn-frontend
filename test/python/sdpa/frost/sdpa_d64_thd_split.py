# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Nonpaged D64 THD split preserves live lengths and independent origins."""

import pytest


@pytest.mark.parametrize("cc", [(10, 0), (10, 3), (10, 7)])
@pytest.mark.parametrize("packed", [False, True])
def test_explicit_contract_and_template(monkeypatch, cc, packed):
    from dataclasses import replace
    import cudnn
    from cudnn.frost import buffers
    from cudnn.sdpa.fwd import engines
    from cudnn.sdpa.fwd.api_dsl import _load_sm100_kernel_module
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams
    from cudnn.sdpa.graph_analyzer import SdpaGraphFacts

    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)
    facts = SdpaGraphFacts(b=2, h_q=8, h_kv=2, s_q=129, s_kv=2048, d_qk=64, d_v=64, dtype=cudnn.data_type.BFLOAT16, thd=True, device_cc=cc, device_sm_count=148)
    name = "sdpa_fwd_prefill_sm107" if cc == (10, 7) else "sdpa_fwd_prefill_sm100"
    caps = next(s.capabilities for s in engines.ENGINE_SPECS if s.name == name)
    knobs = engines.SdpaFwdKnobs(cga=1, split_kv=2, pack_gqa=packed, tile_n=128, sched_policy=0)
    assert engines.mismatch(caps, facts, knobs) is None
    assert engines.mismatch(caps, facts, replace(knobs, cga=2)) is not None
    assert engines.mismatch(caps, facts, replace(knobs, tile_n=64)) is not None
    params = TemplateParams(
        dtype_qkv=2, dtype_o=2, d_flavor=64, cta_mma=1, split_kv=2, thd_varlen=True, decode_tile=True, pack_gqa=packed, qh_per_kh=4, seq_kv_lens_present=True
    )
    module = _load_sm100_kernel_module((64, 64), params, rubin=cc == (10, 7), part=cc)
    assert module.__file__.endswith("sm100/decode_d128_f16.py")
    assert module.CFG.TILE_K == module.CFG.TILE_O == 64
    assert module.CFG.TILES_Q == module.CFG.CTA_MMA == module.CFG.THD_VARLEN == 1
    assert module.SPLIT_KV == 2 and module._FP32_PARTIALS
    binder = cudnn._pybind_module._SdpaThdBinder

    class PreviousExtension:
        def __getattr__(self, attr):
            if attr == "supports_nonpaged_d64_packed_split":
                return False
            return getattr(binder, attr)

    monkeypatch.setattr(cudnn._pybind_module, "_SdpaThdBinder", PreviousExtension())
    assert "matching native" in engines.mismatch(caps, facts, knobs)


@pytest.mark.parametrize("dtype_name", ["bfloat16", "float16"])
@pytest.mark.parametrize("stats_layout", ["token", "head"])
@pytest.mark.parametrize("packed,splits,policy", [(False, 2, 0), (True, 2, 1), (True, 9, 2)])
@pytest.mark.parametrize("compact", [False, True])
def test_origins_and_replay(dtype_name, stats_layout, packed, splits, policy, compact, monkeypatch):
    import cudnn
    import torch
    from test_sdpa_fwd_thd_origins import test_independent_thd_origins_rebind_and_capture as run

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("SM100/SM103/SM107 nonpaged D64 split")
    name = "sdpa_fwd_prefill_sm107" if torch.cuda.get_device_capability() == (10, 7) else "sdpa_fwd_prefill_sm100"
    original = cudnn.pygraph.create_execution_plan
    effective_policy = min(policy, 1) if name.endswith("sm107") else policy

    def select_schedule(graph, engine_id, knobs=None):
        return original(graph, engine_id, {**(knobs or {}), cudnn.knob_type.SCHED_POLICY: effective_policy})

    monkeypatch.setattr(cudnn.pygraph, "create_execution_plan", select_schedule)
    run(64, dtype_name, stats_layout, True, packed, splits, engine_name=name, cga=1, compact=compact)


def test_proper_divisor_pack_replay():
    import torch
    from test_sdpa_fwd_thd_origins import test_independent_thd_origins_rebind_and_capture as run

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("native SM107 proper-divisor PackGQA")
    run(64, "bfloat16", "head", True, True, 9, engine_name="sdpa_fwd_prefill_sm107", cga=1, heads_override=12)


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize("packed,splits", [(False, 2), (True, 9)])
def test_standalone_compact_replay(dtype_name, packed, splits):
    import torch
    from native_d64_sm107_test_utils import test_native_d64_standalone_compact_thd as run

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("SM100/SM103/SM107 nonpaged D64 split")
    run(getattr(torch, dtype_name), 1, packed, heads=8, splits=splits)
