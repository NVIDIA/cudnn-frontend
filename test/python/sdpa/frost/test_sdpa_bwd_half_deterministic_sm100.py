# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic native half backward across dense and packed storage."""

import pytest
import torch

import cudnn
import test_sdpa_bwd_d192_sm100 as dense
import test_sdpa_bwd_thd_d256_sm100 as thd
from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell
from test_sdpa_bwd_d256_sm100 import _decline_reason

pytestmark = [pytest.mark.L0, requires_dsl]


@pytest.mark.parametrize("cc", [(10, 0), (10, 3)], ids=["sm100", "sm103"])
@pytest.mark.parametrize("d,dv", [(192, 128), (256, 256)], ids=["d192", "d256"])
@pytest.mark.parametrize("packed", [False, True], ids=["dense", "thd"])
def test_deterministic_declaration(monkeypatch, cc, d, dv, packed):
    """The graph accepts the actual deterministic request on both native widths."""
    reason = _decline_reason(monkeypatch, cc=cc, d=d, d_v=dv, thd=packed, use_deterministic_algorithm=True)
    assert reason is None, reason


@requires_pre_rubin_blackwell
@pytest.mark.parametrize("d,dv", [(192, 128), (256, 256)], ids=["d192", "d256"])
@pytest.mark.parametrize("dt", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize(
    "sq,sk,hk,causal,br,left,budget",
    [
        (256, 512, 4, False, False, None, None),
        (257, 385, 2, True, True, None, None),
        (257, 513, 1, True, False, 128, None),
        (512, 512, 1, False, False, None, 1 << 19),
    ],
    ids=["mha", "gqa-br-tail", "mqa-window", "head-subgroups"],
)
def test_deterministic_dense_rebind_replay(monkeypatch, d, dv, dt, sq, sk, hk, causal, br, left, budget):
    """Repeated execution, fresh pointers and retained replay must match raw bits."""
    dense.check_backward_replay_and_rebind(monkeypatch, dt, sq, sk, hk, causal, br, left, budget, d=d, dv=dv, deterministic=True)


@requires_pre_rubin_blackwell
@pytest.mark.parametrize("d,dv", [(192, 128), (256, 256)], ids=["d192", "d256"])
@pytest.mark.parametrize("dt,token_major", [(torch.bfloat16, True), (torch.float16, False)], ids=["bf16-token", "fp16-head"])
@pytest.mark.parametrize(
    "surface,physical_origins", [("standalone", False), ("standalone", True), ("graph", True)], ids=["compact", "origins-api", "origins-graph"]
)
def test_deterministic_packed_rebind_replay(monkeypatch, cudnn_handle, d, dv, dt, token_major, surface, physical_origins):
    """Changing lengths/origins and empty sequences retain exact gradients."""
    saved_stream = cudnn.get_stream(cudnn_handle)
    try:
        thd.check_origins_rebind_and_replay(
            monkeypatch,
            dt,
            token_major,
            surface,
            d=d,
            d_v=dv,
            physical_origins=physical_origins,
            cudnn_handle=cudnn_handle,
            deterministic=True,
        )
    finally:
        cudnn.set_stream(cudnn_handle, saved_stream)
