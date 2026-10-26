# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit D128 narrow tiles use two score slots without changing the default."""

import dataclasses

import pytest

from frost_test_utils import requires_dsl

pytestmark = [pytest.mark.L1, requires_dsl]


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize("thd,cga", [(False, 2), (True, 1), (True, 2)])
@pytest.mark.parametrize("tile_n", [None, 64, 128])
def test_native_narrow_adapter_record(dtype_name, thd, cga, tile_n, monkeypatch):
    import torch

    from test_sdpa_fwd_dsl_sm107 import _fake_cc, _gate_api

    from cudnn.frost import buffers

    _fake_cc(monkeypatch, (10, 7))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)
    api = _gate_api(d=128, h=16, h_kv=4, dtype=getattr(torch, dtype_name), with_gate=False, thd=thd, cga=cga, pack_gqa=True, tile_n=tile_n)
    assert api.check_support()
    assert api._effective_tile_n() == (128 if tile_n is None else tile_n)
    assert api._s_slots() == (2 if tile_n == 64 else 1)


@pytest.mark.parametrize("thd,cga,split", [(False, 2, 1), (False, 2, 2), (True, 1, 1), (True, 2, 1)])
@pytest.mark.parametrize("packed", [False, True])
def test_native_narrow_graph_admission(thd, cga, split, packed, monkeypatch):
    from test_sdpa_fwd_dsl_sm107 import _caps, _f16_facts

    from cudnn.frost import buffers
    from cudnn.sdpa.fwd import engines

    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)
    caps = _caps("sdpa_fwd_prefill_sm107")
    facts = _f16_facts(h_q=16, h_kv=4, thd=thd)
    knobs = engines.SdpaFwdKnobs(cga=cga, split_kv=split, pack_gqa=packed, tile_m=128, tile_n=64, sched_policy=0)
    assert engines.effective_tile_ns(caps, facts, knobs) == frozenset({64, 128})
    assert engines.mismatch(caps, facts, knobs) is None
    assert engines.mismatch(caps, facts, dataclasses.replace(knobs, tile_n=128)) is None


@pytest.mark.parametrize("change", ["dense_cga1", "ragged_split", "paged", "bias", "kv_mask", "sm100"])
def test_native_narrow_keeps_other_body_domains(change, monkeypatch):
    from test_sdpa_fwd_dsl_sm107 import _caps, _f16_facts

    from cudnn.frost import buffers
    from cudnn.sdpa.fwd import engines

    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)
    facts = _f16_facts()
    knobs = engines.SdpaFwdKnobs(cga=2, split_kv=1, tile_n=64)
    caps = _caps("sdpa_fwd_prefill_sm107")
    if change == "dense_cga1":
        knobs = dataclasses.replace(knobs, cga=1)
    elif change == "ragged_split":
        facts = dataclasses.replace(facts, thd=True)
        knobs = dataclasses.replace(knobs, cga=1, split_kv=2)
    elif change == "paged":
        facts = dataclasses.replace(facts, thd=True, has_paged_kv=True)
    elif change == "bias":
        facts = dataclasses.replace(facts, has_bias=True)
    elif change == "kv_mask":
        facts = dataclasses.replace(facts, has_kv_mask=True)
    else:
        facts = dataclasses.replace(facts, device_cc=(10, 0))
        caps = _caps("sdpa_fwd_prefill_sm100")
    assert 64 not in engines.effective_tile_ns(caps, facts, knobs)
    assert engines.mismatch(caps, facts, knobs) is not None


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize("with_stats", [False, True])
@pytest.mark.parametrize("geometry", ["mha", "split", "keyless", "window", "partial12", "partial_envelope", "partial_asymmetric"])
def test_native_narrow_dense_replay(dtype_name, with_stats, geometry):
    from test_sdpa_native_pack_sm107 import _check_native_d128_packing_dense_replay

    _check_native_d128_packing_dense_replay(dtype_name, 0, with_stats, geometry, tile_n=64)


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize("cga", [1, 2])
@pytest.mark.parametrize(
    "layout,log2,packed,causal,br,window",
    [("token", False, True, True, False, None), ("head", True, False, False, False, None), ("padded", True, True, True, True, 17)],
)
def test_native_narrow_ragged_rebind(dtype_name, cga, layout, log2, packed, causal, br, window):
    import torch

    from test_sdpa_prepared_thd import _check_rubin_ragged_pack_rebind

    _check_rubin_ragged_pack_rebind(
        getattr(torch, dtype_name),
        layout,
        log2,
        "partial12",
        cga=cga,
        packed=packed,
        causal=causal,
        bottom_right=br,
        window_left=window,
        tile_n=64,
    )
