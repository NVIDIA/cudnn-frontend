# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM100 large-head THD backward (``sdpa_bwd_sm100``, D in (256, 512]) with independent physical origins.

Every port is bound at its own per-sequence offsets with poisoned gaps between sequences (NaN inputs, sentinel
outputs).  A chain that addressed a port at the compact length prefixes would read the poison or write the gap.
"""

import pytest
import torch

import cudnn.sdpa.bwd.api_dsl as api_dsl
import test_sdpa_bwd_thd_d256_sm100 as thd
from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell
from test_sdpa_bwd_thd_sm100 import stage2_datapath  # noqa: F401  (fixture: pins and checks the stage-2 body)

pytestmark = [requires_dsl, requires_pre_rubin_blackwell]
_ENGINE = "sdpa_bwd_sm100"
_LQ, _LK = (257, 93, 128), (383, 129, 267)


_MASKS = {"unmasked": (False, False, None), "top-left": (True, False, None), "bottom-right": (True, True, None), "window": (True, True, 129)}
_DTYPES = {"bf16-token-stats": (torch.bfloat16, True), "fp16-head-stats": (torch.float16, False)}
_HEADS = {"mha": (2, 2), "gqa": (8, 2)}


def _level(d, dtype_id, heads_id, mask_id):
    """D512 BF16 over every head layout and the unmasked / bottom-right masks stays in L0; the sweep is L1."""
    return pytest.mark.L0 if d == 512 and dtype_id == "bf16-token-stats" and mask_id in ("unmasked", "bottom-right") else pytest.mark.L1


@pytest.mark.parametrize(
    "d,dt,token_major,hq,hkv,causal,bottom_right,window_left",
    [
        pytest.param(d, *_DTYPES[t], *_HEADS[h], *_MASKS[m], id=f"{m}-{h}-{t}-{d}", marks=_level(d, t, h, m))
        for d in (512, 384, 264)
        for t in _DTYPES
        for h in _HEADS
        for m in _MASKS
    ],
)
def test_graph_independent_origins(d, dt, token_major, hq, hkv, causal, bottom_right, window_left):
    p = thd.make_payload(dt, token_major, _LQ, _LK, 0, False, hq=hq, hkv=hkv, causal=causal, bottom_right=bottom_right, window_left=window_left, d=d, d_v=d)
    g, execute = thd.make_graph(p, token_major, engine=_ENGINE)
    p["ws"] = torch.empty(g.get_workspace_size(), dtype=torch.uint8, device="cuda")
    p["snapshots"] = thd.snapshots(p)
    thd.poison(p)
    execute(p)
    torch.cuda.synchronize()
    thd.check(p)


@pytest.mark.parametrize("stage2_datapath", [True], ids=["2x2"], indirect=True)
@pytest.mark.parametrize(
    "d,dt,token_major,hq,hkv,causal,bottom_right,window_left",
    [
        pytest.param(d, *_DTYPES[t], *_HEADS[h], *_MASKS[m], id=f"{m}-{h}-{t}-{d}", marks=_level(d, t, h, m))
        for d, t in ((512, "bf16-token-stats"), (512, "fp16-head-stats"), (264, "bf16-token-stats"))
        for h in _HEADS
        for m in ("unmasked", "bottom-right", "window")
    ],
)
def test_graph_independent_origins_2x2(stage2_datapath, d, dt, token_major, hq, hkv, causal, bottom_right, window_left):
    """The same gapped graphs on the 2x2 stage-2 twin (``STAGE2_2X2``), which takes the sequence-local input maps too."""
    test_graph_independent_origins(d, dt, token_major, hq, hkv, causal, bottom_right, window_left)


@pytest.mark.L0
def test_graph_deterministic_origins_repeat_bitwise():
    p = thd.make_payload(torch.bfloat16, True, _LQ, _LK, 0, False, d=512, d_v=512)
    g, execute = thd.make_graph(p, True, deterministic=True, engine=_ENGINE)
    p["ws"] = torch.empty(g.get_workspace_size(), dtype=torch.uint8, device="cuda")
    p["snapshots"] = thd.snapshots(p)
    runs = []
    for _ in range(2):
        thd.poison(p)
        execute(p)
        torch.cuda.synchronize()
        thd.check(p)
        runs.append([p["data"][port].clone() for port in ("dq", "dk", "dv")])
    for first, second in zip(*runs):
        assert torch.equal(first.view(torch.int16), second.view(torch.int16))


@pytest.mark.L0
@pytest.mark.parametrize("dt,token_major", [(torch.bfloat16, True), (torch.float16, False)], ids=["bf16-token-stats", "fp16-head-stats"])
@pytest.mark.parametrize("surface", ["standalone", "graph"])
def test_origins_rebind_and_replay(monkeypatch, dt, token_major, surface):
    # The shared harness forces two GQA groups through reused scratch; this chain's chunk rule is a module function.
    monkeypatch.setattr(api_dsl.SdpaBwdDslSm100, "_ws_budget_bytes", None, raising=False)
    monkeypatch.setattr(api_dsl, "_sm100_head_chunk_thd", lambda *args, **kwargs: 4)
    thd.check_origins_rebind_and_replay(monkeypatch, dt, token_major, surface, d=512, d_v=512, api_type=api_dsl.SdpaBwdDslSm100, engine=_ENGINE)


@pytest.mark.L0
def test_origin_ports_are_optional_per_port():
    """A port bound without offsets keeps its compact prefix while the others take their own origins."""
    p = thd.make_payload(torch.bfloat16, True, _LQ, _LK, 0, False, hq=2, hkv=2, d=512, d_v=512)
    compact = thd.make_payload(torch.bfloat16, True, _LQ, _LK, 0, False, hq=2, hkv=2, d=512, d_v=512, physical_origins=False)
    # dQ compact, every other port at its physical origins.
    for key in ("data", "starts", "live"):
        p[key]["dq"] = compact[key]["dq"]
    del p["offsets"]["dq"], p["spec"]["dq"]
    api = thd.make_adapter(p, True, api_type=api_dsl.SdpaBwdDslSm100)
    p["ws"] = torch.empty(api.scratch_workspace_bytes(), dtype=torch.uint8, device="cuda")
    p["snapshots"] = thd.snapshots(p)
    thd.poison(p)
    thd.run(api, p)
    torch.cuda.synchronize()
    thd.check(p)


@pytest.mark.L0
def test_compact_standalone_plan_is_unchanged_by_the_origin_contract():
    """No declared offsets: the packed-prefix artifact, its frame and its workspace carve are the pre-existing ones."""
    p = thd.make_payload(torch.bfloat16, True, _LQ, _LK, 0, False, hq=2, hkv=2, d=512, d_v=512, physical_origins=False)
    api = thd.make_adapter(p, True, api_type=api_dsl.SdpaBwdDslSm100)
    assert api._thd_origins is None and len(api._prepared.operands) == 11
    p["ws"] = torch.empty(api.scratch_workspace_bytes(), dtype=torch.uint8, device="cuda")
    p["snapshots"] = thd.snapshots(p)
    thd.poison(p)
    thd.run(api, p)
    torch.cuda.synchronize()
    thd.check(p)


@pytest.mark.parametrize("surface", ["standalone", "graph"])
@pytest.mark.parametrize("token_major", [True, False], ids=["token-stats", "head-stats"])
@pytest.mark.parametrize(
    "at_capacity,initialize_guard",
    [
        pytest.param(True, False, id="end-uninitialized", marks=pytest.mark.L0),
        pytest.param(True, True, id="end-initialized-control", marks=pytest.mark.L1),
        pytest.param(False, False, id="interior-control", marks=pytest.mark.L1),
    ],
)
def test_empty_final_origin_stats_guard(stage2_datapath, surface, token_major, at_capacity, initialize_guard):
    """Run under Compute Sanitizer initcheck: numerics alone cannot see a dead-unit Stats read.

    Retain uninitialized storage past the logical Stats view, so the unfixed kernel's read is observable without
    leaving an allocated block. The initialized-guard and interior-origin cases discriminate the boundary read
    from unrelated uninitialized accesses. Both stage-2 variants and the public graph binding are exercised.
    """
    p = thd.make_payload(torch.bfloat16, token_major, (257, 93, 0), (383, 129, 0), 0, False, hq=2, hkv=2, causal=False, bottom_right=False, d=512, d_v=512)
    if at_capacity:
        for port, tensor in p["data"].items():
            cap = 1500 if port in ("k", "v", "dk", "dv") else 1000
            stride = (2 if token_major else 1) if port == "stats" else tensor.stride(1)
            multiplier = p["spec"][port][1]
            p["starts"][port][-1] = cap
            p["offsets"][port][-2:] = cap * stride // multiplier

    stats = p["data"]["stats"]
    backing = torch.empty(stats.numel() + 256 * 2, dtype=stats.dtype, device=stats.device)
    if initialize_guard:
        backing.zero_()
    # Keep the original strides for both token-major and head-major Stats.
    p["data"]["stats"] = backing[: stats.numel()].view(stats.shape)
    p["data"]["stats"].copy_(stats)
    p["stats_guard_backing"] = backing

    if surface == "graph":
        graph, execute = thd.make_graph(p, token_major, engine=_ENGINE)
        workspace_size = graph.get_workspace_size()
    else:
        api = thd.make_adapter(p, token_major, api_type=api_dsl.SdpaBwdDslSm100)
        workspace_size = api.scratch_workspace_bytes()

        def execute(payload):
            thd.run(api, payload)

    p["ws"] = torch.empty(workspace_size, dtype=torch.uint8, device="cuda")
    p["snapshots"] = thd.snapshots(p)
    thd.poison(p)
    execute(p)
    torch.cuda.synchronize()
    thd.check(p)
