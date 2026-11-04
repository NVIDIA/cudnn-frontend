# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Paged THD D128 narrow-tile checks exposed by the SM107 CI entry."""

import dataclasses

import pytest


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize("cga", [1, 2])
@pytest.mark.parametrize("tile_n", [None, 64, 128])
def test_admission_and_record(dtype_name, cga, tile_n, monkeypatch):
    import cudnn
    import torch
    from cudnn.frost import buffers
    from cudnn.sdpa.fwd import engines
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100
    from test_sdpa_fwd_dsl_sm107 import _caps, _desc, _f16_facts, _fake_cc

    _fake_cc(monkeypatch, (10, 7))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)
    dtype = getattr(torch, dtype_name)
    pool = _desc((64, 4, 16, 128), dtype, "k")
    api = SdpaFwdDslSm100(
        _desc((2, 16, 320, 128), dtype, "q"),
        pool,
        dataclasses.replace(pool, name="v"),
        _desc((2, 16, 320, 128), dtype, "o"),
        None,
        is_causal=True,
        causal_bottom_right=True,
        thd=True,
        max_total_seq_len_q=640,
        seq_kv_lens_present=True,
        paged_page_size=16,
        paged_max_seq_len_kv=512,
        cga=cga,
        tile_n=tile_n,
        pack_gqa=True,
        split_kv=1,
    )
    assert api.check_support()
    params = api.template_params()
    assert params.paged_kv and params.thd_varlen and params.split_kv == 1
    assert params.tile_n == (128 if tile_n is None else tile_n)
    assert params.s_slots == (2 if tile_n == 64 else 1)
    caps = _caps("sdpa_fwd_prefill_sm107")
    facts = dataclasses.replace(
        _f16_facts(h_q=16, h_kv=4, thd=True),
        has_paged_kv=True,
        padded=True,
        page_size=16,
        dtype=cudnn.data_type.HALF if dtype_name == "float16" else cudnn.data_type.BFLOAT16,
    )
    knobs = engines.SdpaFwdKnobs(cga=cga, split_kv=1, pack_gqa=True, tile_m=128, tile_n=tile_n, sched_policy=0)
    assert engines.effective_tile_ns(caps, facts, knobs) == frozenset({64, 128})
    assert engines.mismatch(caps, facts, knobs) is None


@pytest.mark.parametrize(
    "dim,dtype_name,pool_layout,stats_layout,cga,packed,page,window,sink,policy",
    [
        (128, "bfloat16", "hnd", "token", 2, True, 16, None, None, 0),
        (128, "float16", "nhd", "head", 1, False, 64, None, None, 1),
        (128, "bfloat16", "mixed", "head", 2, False, 8, None, None, 0),
        (128, "float16", "mixed", "token", 1, True, 256, None, None, 0),
        (128, "bfloat16", "mixed", "head", 2, True, 16, 32, -120.0, 0),
        (128, "float16", "hnd", "token", 1, False, 64, 73, 3.0, 1),
        (128, "bfloat16", "nhd", "token", 2, True, 32, None, None, 1),
        (128, "float16", "mixed", "head", 1, False, 128, None, None, 0),
    ],
)
def test_origins_rebind_and_replay(dim, dtype_name, pool_layout, stats_layout, cga, packed, page, window, sink, policy):
    import torch
    from test_sdpa_fwd_paged_thd_origins import test_paged_thd_origins_rebind_and_capture

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("native SM107 paged THD N64")
    test_paged_thd_origins_rebind_and_capture(
        dim,
        dtype_name,
        pool_layout,
        stats_layout,
        cga,
        1,
        packed,
        page_size=page,
        window_left=window,
        sink=sink,
        tile_n=64,
        sched_policy=policy,
        cpu_reference=True,
    )
