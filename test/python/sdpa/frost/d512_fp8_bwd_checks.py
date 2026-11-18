# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""D512 per-tensor FP8 backward through the graph contract."""

import pytest
import torch


@pytest.mark.parametrize(
    "kwargs", [dict(d=256), dict(d_v=256), dict(bottom_right=True), dict(right_bound=4), dict(padded=True), dict(sink=True), dict(bhsd=True)]
)
def test_unsupported_graphs_decline(d512, monkeypatch, kwargs):
    kwargs = dict(kwargs)
    if kwargs.pop("bhsd", False):
        kwargs["stride_fn"] = d512._bhsd
    reason = d512._decline_reason(monkeypatch, engine="sdpa_bwd_d512_fp8", cc=torch.cuda.get_device_capability(), **{"d": 512, **kwargs})
    assert reason is not None


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float16, torch.bfloat16], ids=["e4m3", "fp16", "bf16"])
@pytest.mark.parametrize(
    "shape",
    [(129, 257, 2, 1, True, 127), (257, 129, 4, 2, False, None), (17, 256, 1, 1, True, 1)],
    ids=["gqa-swa-tail", "rect-gqa-full", "self-window-empty-k"],
)
def test_graph_numerics(d512, dtype, shape):
    sq, sk, hq, hk, causal, left = shape
    run = d512._run_fp8(sq=sq, skv=sk, hq=hq, hkv=hk, causal=causal, left=left, grad_dtype=dtype, runs=2, ws_poison=255).check()
    for name in run.outs[0]:
        assert torch.equal(run.outs[0][name].view(torch.uint8), run.outs[1][name].view(torch.uint8)), name
    for name in run.amax[0]:
        assert torch.equal(run.amax[0][name].view(torch.uint8), run.amax[1][name].view(torch.uint8)), name


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float16, torch.bfloat16], ids=["e4m3", "fp16", "bf16"])
def test_replay_batch_and_golden(d512, dtype):
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
                batch, 2, 17, 512, hkv=1, skv=33, causal=True, left_bound=9, deterministic=True, grad_dtypes=(dt,) * 3, scale=run.scale
            )
            graph.validate()
            graph.build_operation_graph()
            graph.create_execution_plans([cudnn.heur_mode.A])
            select_engine(graph, "sdpa_bwd_d512_fp8")
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
    assert_golden("sdpa_d512_fp8_bwd_" + str(dtype).split(".")[-1], before, baseline)
