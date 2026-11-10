# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Native SM120 per-tensor FP8 backward graph checks."""

import pytest
import torch
import test_sdpa_bwd_fp8_sm107 as fp8
from frost_test_utils import requires_dsl

pytestmark = [pytest.mark.L0, requires_dsl]


@pytest.fixture(autouse=True)
def native(monkeypatch):
    if torch.cuda.get_device_capability() not in ((12, 0), (12, 1)):
        pytest.skip("requires native SM120/SM121")
    monkeypatch.setattr(fp8, "_ENGINE", "sdpa_bwd_sm120_fp8")


@pytest.mark.parametrize("dims", [(64, 64), (128, 128), (192, 128), (256, 256), (512, 512), (504, 248), (40, 8)])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mask", ["full", "causal", "swa"])
def test_native_graph(monkeypatch, dims, dtype, mask):
    monkeypatch.setattr(fp8, "_D", dims[0])
    run = fp8._run_fp8(
        b=2, hq=4, hkv=2, sq=19, skv=37, d_v=dims[1], causal=mask != "full", left=7 if mask == "swa" else None, grad_dtype=dtype, runs=2, ws_poison=255
    ).check()
    for outputs in (run.outs, run.amax):
        for name in outputs[0]:
            assert torch.equal(outputs[0][name].view(torch.uint8), outputs[1][name].view(torch.uint8)), name


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float16, torch.bfloat16], ids=["e4m3", "fp16", "bf16"])
@pytest.mark.parametrize("width", [192, 512])
def test_replay_batch_and_golden(monkeypatch, dtype, width):
    monkeypatch.setattr(fp8, "_D", width)
    d512 = fp8
    import cudnn
    from determinism_utils import assert_bitwise, assert_golden, assert_replays, snapshot
    from frost_test_utils import select_engine

    run = d512._run_fp8(b=3, hq=2, hkv=1, sq=17, skv=33, causal=True, left=9, grad_dtype=dtype, seed=91457).check()
    sources = {name: value.clone() for name, value in run.inputs.items()}
    scalars = {name: run.pack[run.ts[name]].clone() for name in d512._SCALARS}
    before = snapshot({**sources, **scalars})
    plans = {}

    def prepare(order):
        batch = len(order)
        if batch not in plans:
            dt = {torch.float8_e4m3fn: cudnn.data_type.FP8_E4M3, torch.float16: cudnn.data_type.HALF, torch.bfloat16: cudnn.data_type.BFLOAT16}[dtype]
            graph, ports = d512._graph_and_ports(
                batch, 2, 17, width, hkv=1, skv=33, causal=True, left_bound=9, deterministic=True, grad_dtypes=(dt,) * 3, scale=run.scale
            )
            graph.validate()
            graph.build_operation_graph()
            graph.create_execution_plans([cudnn.heur_mode.A])
            select_engine(graph, "sdpa_bwd_sm120_fp8")
            graph.check_support()
            graph.build_plans()
            assert graph.get_workspace_size() < 4 * batch * 2 * 128 + 256
            plans[batch] = graph, ports, torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
        graph, ports, workspace = plans[batch]
        inputs = {name: torch.cat([value[i : i + 1] for i in order], 0) for name, value in sources.items()}
        out = {name: torch.empty((batch, *run.outs_t[name].shape[1:]), device="cuda", dtype=dtype) for name in run.outs_t}
        am = {name: torch.empty_like(value) for name, value in run.amax_t.items()}
        pack = {ports[name]: value if name == "stats" else d512._view_bhsd(value) for name, value in inputs.items()}
        pack.update({ports[name]: value for name, value in scalars.items()})
        pack.update({ports[name]: d512._view_bhsd(value) for name, value in out.items()})
        pack.update({ports["amax_" + name]: value for name, value in am.items()})

        def execute():
            for value in (*out.values(), *am.values()):
                value.fill_(float("nan"))
            workspace.fill_(255)
            graph.execute(pack, workspace)
            return {**out, **{"amax_" + name: value for name, value in am.items()}}

        return graph, pack, workspace, inputs, out, am, execute

    single = [snapshot(prepare([i])[-1]()) for i in range(3)]
    for i, result in enumerate(single):
        assert_bitwise({name: result[name] for name in run.outs_t}, {name: value[i : i + 1] for name, value in run.outs[0].items()}, "singleton")
    baseline = None
    for order in ([0, 1, 2], [2, 0, 1], [1, 0, 1, 2], [0, 2, 1] * 43):
        graph, pack, workspace, inputs, out, am, execute = prepare(order)
        result = assert_replays(execute, repeats=3)
        expected = {name: torch.cat([single[i][name] for i in order], 0) for name in out}
        expected.update({"amax_" + name: torch.stack([single[i]["amax_" + name] for i in order]).amax(0) for name in am})
        assert_bitwise(result, expected, "reordered/duplicated batch")
        if len(order) == 3 and order[0] == 0:
            baseline = result
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured):
            graph.execute(pack, workspace)
        inputs["dO"].view(torch.uint8).bitwise_xor_(128)
        eager_changed = snapshot(execute())
        for name in out:
            torch.testing.assert_close(eager_changed[name].float(), -result[name].float(), atol=0, rtol=0)
        for value in (*out.values(), *am.values()):
            value.fill_(float("nan"))
        workspace.fill_(255)
        captured.replay()
        assert_bitwise(snapshot({**out, **{"amax_" + name: value for name, value in am.items()}}), eager_changed, "changed-input CUDA Graph")
        torch.cuda.set_sync_debug_mode("error")
        try:
            graph.execute(pack, workspace)
        finally:
            torch.cuda.set_sync_debug_mode("default")
    assert_bitwise(snapshot({**sources, **scalars}), before, "inputs preserved")
    assert_golden("sdpa_sm120_fp8_bwd_d" + str(width) + "_" + str(dtype).split(".")[-1], before, baseline)


@pytest.mark.parametrize("shape", [(65, 33, 4, 4, False, None), (129, 257, 8, 1, True, 127), (97, 19, 4, 2, False, 7)])
@pytest.mark.parametrize("width", [64, 192, 512])
def test_longer_and_keyless_graph(monkeypatch, shape, width):
    monkeypatch.setattr(fp8, "_D", width)
    sq, skv, hq, hkv, causal, left = shape
    fp8._run_fp8(sq=sq, skv=skv, hq=hq, hkv=hkv, causal=causal, left=left, grad_dtype=torch.float16, runs=2, ws_poison=255).check()


def test_omitted_amax_capture_and_rebinding(monkeypatch):
    from determinism_utils import snapshot, assert_bitwise

    monkeypatch.setattr(fp8, "_D", 192)
    run = fp8._run_fp8(b=2, hq=4, hkv=1, sq=33, skv=65, d_v=128, causal=True, left=17, grad_dtype=torch.bfloat16, request_amax=()).check()
    graph, pack, ws = run.graph, run.pack, run.workspace
    before = snapshot(run.outs_t)
    rebound = {key: value.clone() for key, value in pack.items()}
    fresh_ws = torch.empty_like(ws)
    graph.execute(rebound, fresh_ws)
    out = {name: rebound[run.ts[name]].permute(0, 2, 1, 3) for name in run.outs_t}
    assert_bitwise(snapshot(out), before, "fresh pointers")
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture):
        graph.execute(rebound, fresh_ws)
    rebound[run.ts["descale_dP"]].mul_(2.0)
    graph.execute(rebound, fresh_ws)
    changed = snapshot(out)
    expected = {name: value * (1 if name == "dV" else 2) for name, value in before.items()}
    assert_bitwise(changed, expected, "nonreciprocal device descale doubles dQ/dK only")
    for value in out.values():
        value.fill_(float("nan"))
    fresh_ws.fill_(255)
    capture.replay()
    assert_bitwise(snapshot(out), changed, "device scale change under capture")
    torch.cuda.set_sync_debug_mode("error")
    try:
        graph.execute(rebound, fresh_ws)
    finally:
        torch.cuda.set_sync_debug_mode("default")
