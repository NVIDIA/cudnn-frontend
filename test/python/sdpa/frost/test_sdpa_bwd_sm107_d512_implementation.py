# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""The existing D512 record keeps its body through rebuild, rebind and replay."""

import pytest

from frost_test_utils import requires_dsl, requires_rubin

pytestmark = pytest.mark.L0


@requires_dsl
@requires_rubin
@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"], ids=["fp16", "bf16"])
@pytest.mark.parametrize("d,hkv,causal", [(384, 2, False), (512, 4, True), (512, 2, True)], ids=["envelope-gqa", "native-mha", "native-gqa"])
def test_public_record_rebuild_rebind_and_capture(monkeypatch, dtype_name, d, hkv, causal):
    import torch

    import test_sdpa_bwd_dsl_sm100 as helper
    from cudnn.sdpa.bwd import api_dsl
    from cudnn.sdpa.bwd.kernels.sm107.d512_f16_2x2.contract import KERNEL_FILE, STAGE2_TAG

    torch.manual_seed(34607)
    dtype = getattr(torch, dtype_name)
    b, h, sq, skv = 1, 4, 257, 257
    tensors = {name: helper._bshd(b, length, heads, d, dt=dtype) for name, length, heads in (("q", sq, h), ("k", skv, hkv), ("v", skv, hkv), ("do", sq, h))}
    tensors["o"] = helper._bshd(b, sq, h, d, dt=dtype, fill=False)
    tensors["stats"] = torch.empty((b, h, sq, 1), dtype=torch.float32, device="cuda")
    for dst, src in (("dq", "q"), ("dk", "k"), ("dv", "v")):
        tensors[dst] = torch.empty_like(tensors[src]).fill_(float("nan"))
    keep = helper._causal_keep(sq, skv) if causal else None

    def reference(values):
        o, stats, _, dq, dk, dv = helper._reference(*(values[name] for name in ("q", "k", "v", "do")), keep, h // hkv)
        values["o"].copy_(o)
        values["stats"].copy_(stats.unsqueeze(-1))
        return dq, dk, dv

    def check(values, expected):
        from types import SimpleNamespace

        helper._check_prepared(SimpleNamespace(tensors=values, expected=expected))

    seen, original = [], api_dsl.load_template

    def spy(path, params, tag="template"):
        if tag == STAGE2_TAG:
            seen.append(path.rsplit("/kernels/", 1)[-1])
        return original(path, params, tag)

    monkeypatch.setattr(api_dsl, "load_template", spy)

    def build(record, values):
        g, refs, outputs = helper._build_graph(b, h, hkv, sq, skv, d, d**-0.5, dt=dtype, record=record, use_causal_mask=causal)
        g.select_plan(0)
        g.check_support()
        g.build_plans()
        refs.update(zip(("dq", "dk", "dv"), outputs))
        workspace = torch.empty(g.get_workspace_size(), device="cuda", dtype=torch.uint8)
        return g, {refs[name]: value for name, value in values.items()}, workspace

    expected = reference(tensors)
    monkeypatch.setattr(api_dsl, "STAGE2_2X2", False)
    graph, pack, workspace = build((20608, {}), tensors)
    graph.execute(pack, workspace)
    check(tensors, expected)
    snapshots = [tensors[name].clone() for name in ("dq", "dk", "dv")]
    record = graph.get_engine_and_knobs_at_index(0)
    assert record == (20608, {})

    rebound = {name: value.clone() for name, value in tensors.items()}
    assert all(rebound[name].data_ptr() != value.data_ptr() for name, value in tensors.items())
    monkeypatch.setattr(api_dsl, "STAGE2_2X2", True)
    rebuilt, pack, workspace = build(record, rebound)
    for name in ("dq", "dk", "dv"):
        rebound[name].fill_(float("nan"))
    workspace.fill_(0xBD)
    rebuilt.execute(pack, workspace)
    check(rebound, expected)
    for name, want in zip(("dq", "dk", "dv"), snapshots):
        assert torch.equal(rebound[name].view(torch.int16), want.view(torch.int16)), name
    assert seen == [KERNEL_FILE, KERNEL_FILE]

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        rebuilt.execute(pack, workspace)
    stream.synchronize()
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture, stream=stream):
        rebuilt.execute(pack, workspace)
    rebound["q"].mul_(0.75)
    rebound["do"].mul_(1.25)
    expected = reference(rebound)
    for name in ("dq", "dk", "dv"):
        rebound[name].fill_(float("nan"))
    workspace.fill_(0xBD)
    capture.replay()
    check(rebound, expected)
    snapshots = [rebound[name].clone() for name in ("dq", "dk", "dv")]
    for _ in range(3):
        for name in ("dq", "dk", "dv"):
            rebound[name].fill_(float("nan"))
        workspace.fill_(0xBD)
        capture.replay()
        for name, want in zip(("dq", "dk", "dv"), snapshots):
            assert torch.equal(rebound[name].view(torch.int16), want.view(torch.int16)), name
