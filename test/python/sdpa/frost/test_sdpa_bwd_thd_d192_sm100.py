# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native rectangular THD backward, including independent per-port origins."""

import pytest
import torch

import cudnn
import test_sdpa_bwd_thd_d256_sm100 as thd
from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell

pytestmark = [pytest.mark.L0, requires_dsl, requires_pre_rubin_blackwell]


@pytest.mark.parametrize("dt,token_major", [(torch.bfloat16, True), (torch.float16, False)], ids=["bf16-token-stats", "fp16-head-stats"])
@pytest.mark.parametrize(
    "surface,physical_origins", [("standalone", False), ("standalone", True), ("graph", True)], ids=["compact", "origins-api", "origins-graph"]
)
def test_rectangular_thd_rebind_and_replay(monkeypatch, cudnn_handle, dt, token_major, surface, physical_origins):
    from cudnn.sdpa.bwd.kernels.sm107 import prepared_host

    compiled_hosts = []
    original_host = prepared_host.compile_host_f16_thd

    def record_host(main, mm_dk, mm_dq, *args, **kwargs):
        assert mm_dk is None, "THD must omit the separate dK GEMM"
        compiled_hosts.append(main)
        return original_host(main, mm_dk, mm_dq, *args, **kwargs)

    monkeypatch.setattr(prepared_host, "compile_host_f16_thd", record_host)
    geometry = []
    original = thd.SdpaBwdDslSm100D256._compile_plan

    def record(api, main, mm_dk, mm_dq):
        geometry.append((main.CFG.TILE_K, main.CFG.TILE_O))
        assert geometry[-1] == (192, 128)
        assert main.CFG.FUSE_DK == 1 and main.CFG.XFER_STAGES == 2
        assert next(shape for name, shape, _ in api._scratch_shapes() if name == "desc_words") == ((2 * api.batch_size + 1) * 16,)
        for stage3 in (mm_dk, mm_dq):
            assert stage3.PARAMS.cgrp_tile_mn == (128, 192) and stage3.cta_group == 1
        return original(api, main, mm_dk, mm_dq)

    monkeypatch.setattr(thd.SdpaBwdDslSm100D256, "_compile_plan", record)
    saved_stream = cudnn.get_stream(cudnn_handle)
    try:
        thd.check_origins_rebind_and_replay(monkeypatch, dt, token_major, surface, d=192, d_v=128, physical_origins=physical_origins, cudnn_handle=cudnn_handle)
        assert geometry, "the test must compile the native rectangular body"
        assert compiled_hosts
    finally:
        cudnn.set_stream(cudnn_handle, saved_stream)


@pytest.mark.parametrize("hq,hkv", [(2, 2), (8, 2)], ids=["mha", "gqa"])
@pytest.mark.parametrize(
    "causal,bottom_right,window_left", [(False, False, None), (True, False, None), (True, True, 129)], ids=["unmasked", "top-left", "window"]
)
def test_rectangular_thd_masks(cudnn_handle, hq, hkv, causal, bottom_right, window_left):
    p = thd.make_payload(
        torch.bfloat16,
        True,
        (257, 93, 128),
        (383, 129, 267),
        0,
        False,
        hq=hq,
        hkv=hkv,
        causal=causal,
        bottom_right=bottom_right,
        window_left=window_left,
        d=192,
        d_v=128,
    )
    saved_stream = cudnn.get_stream(cudnn_handle)
    try:
        graph, execute = thd.make_graph(p, True, cudnn_handle)
        p["ws"] = torch.empty(graph.get_workspace_size(), dtype=torch.uint8, device="cuda")
        p["snapshots"] = thd.snapshots(p)
        thd.poison(p)
        execute(p)
        torch.cuda.synchronize()
        thd.check(p)
    finally:
        cudnn.set_stream(cudnn_handle, saved_stream)
