# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Native Rubin half D192: real engine admission, geometry and retained replay."""

import pytest
import torch

import cudnn
import test_sdpa_bwd_thd_d256_sm100 as thd
from frost_test_utils import requires_dsl, requires_rubin
from cudnn.sdpa.bwd.api_dsl_sm107 import SdpaBwdDslSm107
from test_sdpa_bwd_d192_sm100 import check_backward_replay_and_rebind
from test_sdpa_bwd_dsl_sm107 import _decline_reason

pytestmark = [pytest.mark.L0, requires_dsl]
_ENGINE = "sdpa_bwd_sm107_d192"


@pytest.mark.parametrize("d,dv", [(192, 128), (128, 128), (192, 192), (256, 128), (256, 256)])
@pytest.mark.parametrize("thd", [False, True])
def test_exact_pair_admission(monkeypatch, d, dv, thd):
    from cudnn.frost import buffers

    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)
    reason = _decline_reason(monkeypatch, engine=_ENGINE, d=d, d_v=dv, thd=thd, use_deterministic_algorithm=True)
    assert (reason is None) == ((d, dv) == (192, 128)), reason


@pytest.mark.parametrize("cc", [(10, 0), (10, 3), (12, 0)])
def test_other_architectures_decline(monkeypatch, cc):
    assert _decline_reason(monkeypatch, engine=_ENGINE, cc=cc, d=192, d_v=128) is not None


@pytest.mark.parametrize("dt", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("thd", [False, True], ids=["dense", "thd"])
def test_decode_declines_before_lowering(monkeypatch, dt, thd):
    from cudnn.frost import buffers

    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)
    reason = _decline_reason(monkeypatch, engine=_ENGINE, d=192, d_v=128, sq=1, dt=dt, thd=thd)
    assert reason is not None and "decode" in reason, reason


def test_stable_opt_in_slot():
    from cudnn.engines.manifest import MANIFEST

    slot = next(f for f in MANIFEST if f.name == "frost_sdpa_bwd").slots[_ENGINE]
    assert slot.opt_in and slot.slot == 10


@requires_rubin
@pytest.mark.parametrize("dt", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("deterministic", [False, True], ids=["ordinary", "deterministic"])
@pytest.mark.parametrize(
    "sq,sk,hk,causal,br,left,budget",
    [
        (256, 512, 4, False, False, None, None),
        (257, 385, 2, True, True, None, None),
        (257, 513, 1, True, False, 128, None),
        (257, 385, 2, False, False, 128, None),
        (512, 512, 2, True, False, None, 1 << 20),
    ],
    ids=["mha", "gqa-tail-br", "mqa-window", "noncausal-window", "chunked"],
)
def test_dense_rebind_and_replay(monkeypatch, dt, deterministic, sq, sk, hk, causal, br, left, budget):
    check_backward_replay_and_rebind(
        monkeypatch,
        dt,
        sq,
        sk,
        hk,
        causal,
        br,
        left,
        budget,
        deterministic=deterministic,
        api_type=SdpaBwdDslSm107,
        engine=_ENGINE,
    )


@requires_rubin
@pytest.mark.parametrize("dt", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("contract", ["zero-scale", "batch-pitch"])
def test_dense_declared_contracts(monkeypatch, dt, contract):
    check_backward_replay_and_rebind(
        monkeypatch,
        dt,
        257,
        385,
        2,
        True,
        True,
        None,
        None,
        deterministic=True,
        api_type=SdpaBwdDslSm107,
        engine=_ENGINE,
        scale=0.0 if contract == "zero-scale" else None,
        batch_pitch=contract == "batch-pitch",
    )


@requires_rubin
@pytest.mark.parametrize("dt,token_major", [(torch.bfloat16, True), (torch.float16, False)], ids=["bf16-token-stats", "fp16-head-stats"])
@pytest.mark.parametrize("deterministic", [False, True], ids=["ordinary", "deterministic"])
@pytest.mark.parametrize(
    "surface,physical_origins", [("standalone", False), ("standalone", True), ("graph", True)], ids=["compact", "origins-api", "origins-graph"]
)
def test_thd_rebind_and_replay(monkeypatch, cudnn_handle, dt, token_major, deterministic, surface, physical_origins):
    from cudnn.sdpa.bwd.kernels.sm107 import prepared_host

    compiled = []
    original = prepared_host.compile_host_f16_thd

    def record(main, dk, dq, *args, **kwargs):
        assert dk is None, "native D192 must fuse dK into main"
        compiled.append(main)
        return original(main, dk, dq, *args, **kwargs)

    original_plan = SdpaBwdDslSm107._compile_plan

    def record_plan(api, main, dk, dq):
        assert (main.CFG.TILE_K, main.CFG.TILE_O) == (192, 128) and main.CFG.FUSE_DK
        assert dq.PARAMS.cgrp_tile_mn == (128, 192) and dq.cta_group == 1
        assert next(shape for name, shape, _ in api._scratch_shapes() if name == "desc_words") == ((2 * api.batch_size + 1) * 16,)
        return original_plan(api, main, dk, dq)

    monkeypatch.setattr(SdpaBwdDslSm107, "_compile_plan", record_plan)
    monkeypatch.setattr(prepared_host, "compile_host_f16_thd", record)
    saved_stream = cudnn.get_stream(cudnn_handle)
    try:
        thd.check_origins_rebind_and_replay(
            monkeypatch,
            dt,
            token_major,
            surface,
            d=192,
            d_v=128,
            physical_origins=physical_origins,
            cudnn_handle=cudnn_handle,
            deterministic=deterministic,
            api_type=SdpaBwdDslSm107,
            engine=_ENGINE,
        )
        assert compiled, "the native D192 body must compile"
    finally:
        cudnn.set_stream(cudnn_handle, saved_stream)
