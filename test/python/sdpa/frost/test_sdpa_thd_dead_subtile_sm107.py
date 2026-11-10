# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""THD dead-subtile selection uses runtime lengths, not descriptor envelopes."""

import functools
from itertools import accumulate

import pytest
import torch

from frost_test_utils import requires_dsl, requires_rubin

pytestmark = [pytest.mark.L0, requires_dsl]


@pytest.mark.parametrize("cc,dim,packed,expected", [((10, 7), 64, False, 1), ((10, 7), 64, True, 0), ((10, 7), 128, False, 0), ((10, 0), 64, False, 0)])
@pytest.mark.parametrize("envelope", [128, 256, 65536])
def test_thd_quiet_uses_runtime_lengths(cc, dim, packed, expected, envelope, monkeypatch):
    from test_sdpa_fwd_dsl_sm107 import _fake_cc, _gate_api

    from cudnn.frost import buffers

    _fake_cc(monkeypatch, cc)
    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)
    api = _gate_api(d=dim, s=envelope, h=8, h_kv=2, with_gate=False, thd=True, cga=1, pack_gqa=packed)
    assert api.check_support()
    assert api.template_params().dead_subtile_quiet == expected


@requires_rubin
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("cga,tile_n", [(1, 64), (1, 128), (2, 64), (2, 128)])
def test_thd_quiet_replay_is_bitwise_full_body(dtype, cga, tile_n, monkeypatch):
    from test_sdpa_prepared_thd import _buffers, _compact_thd_api, _execute_compact_thd

    from cudnn.sdpa.fwd import api_dsl

    b, sq, sk, hq, hk, d = 3, 513, 1025, 4, 2, 64
    inputs = _buffers(b, sq, sk, hq, hk, d, seed=731, dtype=dtype)
    outputs = [{**inputs, "o": torch.empty_like(inputs["o"]), "lse": torch.empty_like(inputs["lse"])} for _ in range(2)]
    plans, workspaces = [], []
    constructor = api_dsl.SdpaFwdDslSm100
    for form, bufs in enumerate(outputs):
        with monkeypatch.context() as patch:
            patch.setattr(api_dsl, "SdpaFwdDslSm100", functools.partial(constructor, dead_subtile_quiet=form, tile_n=tile_n))
            api = _compact_thd_api(bufs, b, sq, sk, hq, hk, d, cga=cga)
        assert api._k_mod.CFG.DEAD_SUBTILE_QUIET == form
        plans.append(api)
        workspaces.append(torch.empty(max(api.scratch_workspace_bytes(), 1), device="cuda", dtype=torch.uint8))

    def execute():
        for api, bufs, ws in zip(plans, outputs, workspaces):
            _execute_compact_thd(api, bufs, ws)

    execute()
    capture = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(capture):
            execute()
        for qs, ks in [([129, 0, 257], [513, 0, 1025]), ([31, 7, 0], [40, 19, 0]), ([0, 128, 128], [0, 257, 513])]:
            inputs["cu_q"].copy_(torch.tensor([0, *accumulate(qs)], device="cuda", dtype=torch.int32))
            inputs["cu_kv"].copy_(torch.tensor([0, *accumulate(ks)], device="cuda", dtype=torch.int32))
            for bufs, ws in zip(outputs, workspaces):
                bufs["o"].fill_(19)
                bufs["lse"].fill_(19)
                ws.fill_(255)
            capture.replay()
            torch.cuda.synchronize()
            for name in ("o", "lse"):
                assert torch.equal(outputs[0][name], outputs[1][name]), (qs, ks, name)
                assert torch.all(outputs[1][name][sum(qs) :] == 19)
    finally:
        capture.reset()
