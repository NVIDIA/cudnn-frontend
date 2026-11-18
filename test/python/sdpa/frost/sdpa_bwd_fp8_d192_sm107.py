# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""SM107 per-tensor fp8 backward at D192/V128: the stage-3 GEMMs on the (256, 192) pair row."""

import pytest
import torch

import test_sdpa_bwd_fp8_sm107 as fp8_suite
from frost_test_utils import requires_dsl, requires_rubin

ds_knob = fp8_suite.ds_knob

pytestmark = [pytest.mark.L0, requires_dsl]

_CASES = {
    "dense": dict(),
    "causal": dict(causal=True),
    "gqa-causal": dict(hq=8, hkv=2, causal=True),
    "ragged": dict(sq=768, skv=1280),
    "bottom-right": dict(causal=True, bottom_right=True, sq=768, skv=1280),
    "window": dict(causal=True, left=256, sq=1024, skv=1024),
    "bf16-grads": dict(grad_dtype=torch.bfloat16),
}


def test_stage3_tile_rule_fp8_d192():
    from cudnn.sdpa.bwd.api_dsl_sm107 import SdpaBwdDslSm107Fp8

    for dims, tile in (((192, 128), (256, 192)), ((256, 256), (256, 256))):
        api = SdpaBwdDslSm107Fp8.__new__(SdpaBwdDslSm107Fp8)
        api.head_dim_qk, api.head_dim_v = dims
        assert api._stage3_tile_mn(107) == tile


def _run_d192(monkeypatch, **kw):
    monkeypatch.setattr(fp8_suite, "_D", 192)
    return fp8_suite._run_fp8(d_v=128, **kw)


@requires_rubin
@pytest.mark.parametrize("case", list(_CASES), ids=list(_CASES))
def test_d192_native_stage3_matches_the_oracle_and_the_d256_row(monkeypatch, ds_knob, case):  # noqa: F811
    from cudnn.sdpa.bwd.api_dsl_sm107 import SdpaBwdDslSm107Fp8

    tiles = []
    original = SdpaBwdDslSm107Fp8._compile_plan

    def record(api, main, dk, dq):
        tiles.append((dk.PARAMS.cgrp_tile_mn, dq.PARAMS.cgrp_tile_mn, dq.cta_group))
        return original(api, main, dk, dq)

    monkeypatch.setattr(SdpaBwdDslSm107Fp8, "_compile_plan", record)
    native = _run_d192(monkeypatch, runs=2, **_CASES[case]).check()
    assert tiles[-1] == ((256, 192), (256, 192), 2)
    for name in ("dQ", "dK", "dV"):
        assert torch.equal(native.outs[0][name].view(torch.int8), native.outs[1][name].view(torch.int8)), f"{name}: repeated run"
    monkeypatch.setattr(SdpaBwdDslSm107Fp8, "_native_fp8_d192", lambda self: False)
    wide = _run_d192(monkeypatch, **_CASES[case])
    assert tiles[-1][:2] == ((256, 256), (256, 256))
    for name in ("dQ", "dK", "dV"):
        assert torch.equal(native.outs[0][name].view(torch.int8), wide.outs[0][name].view(torch.int8)), f"{name}: native vs (256, 256) row"
    for name in native.amax[0]:
        assert torch.equal(native.amax[0][name], wide.amax[0][name]), f"amax_{name}: native vs (256, 256) row"
