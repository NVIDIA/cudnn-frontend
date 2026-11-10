# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Native SM120 canonical block-scale backward numerical checks."""

import pytest
import torch
import test_sdpa_bwd_mxfp8_sm100 as mx
from frost_test_utils import requires_dsl

pytestmark = [pytest.mark.L0, requires_dsl]


@pytest.fixture(autouse=True)
def native(monkeypatch):
    if torch.cuda.get_device_capability() not in ((12, 0), (12, 1)):
        pytest.skip("requires native SM120/SM121")
    monkeypatch.setattr(mx, "_ENGINE", "sdpa_bwd_sm120_mxfp8")
    original = mx._plan_index
    monkeypatch.setattr(mx, "_plan_index", lambda graph: original(graph, "sdpa_bwd_sm120_mxfp8"))
    with torch.cuda.stream(torch.cuda.Stream()):
        yield


@pytest.mark.parametrize("dims", [(64, 64), (128, 128), (192, 128), (256, 256), (512, 512), (504, 248), (40, 8)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mask", ["full", "causal", "swa"])
def test_native_graph(dims, dtype, mask):
    kw = {"left_bound": 9} if mask == "swa" else {}
    _check_gradients(mx._run(b=2, hq=4, hkv=2, sq=19, skv=37, d=dims[0], dv=dims[1], out_dt=dtype, causal=mask != "full", repeat_outputs=3, **kw))


def _check_gradients(run):
    # Cosine alone cannot detect an erroneous common scale. Bound magnitude
    # error separately for each gradient against the independent MXFP8 oracle.
    for name, actual, reference in zip(("dQ", "dK", "dV"), run.expected, run.reference):
        assert torch.isfinite(actual).all(), name
        relative_l2 = (actual.float() - reference.float()).norm() / reference.float().norm().clamp_min(1e-20)
        assert relative_l2 < 0.05, f"{name} relative L2 error: {relative_l2.item()}"
    return run


def _independent_run(mx, monkeypatch, width, **kwargs):
    original = torch.randn
    generator = torch.Generator(device="cpu").manual_seed(118037)

    def randn(*shape, **options):
        device = options.pop("device", None)
        return original(*shape, **options, generator=generator, device="cpu").to(device)

    with monkeypatch.context() as patch, torch.cuda.stream(torch.cuda.Stream()):
        patch.setattr(torch, "randn", randn)
        return _check_gradients(mx._run(d=width, **kwargs))


@pytest.mark.parametrize("width", [128, 512])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_replay_batch_and_golden(monkeypatch, dtype, width):
    import cudnn
    from determinism_utils import assert_bitwise, assert_golden, assert_replays, snapshot
    from frost_test_utils import select_engine
    from cudnn.sdpa.bwd.kernels.sm100.prepared_d192_fp8_host import sf_geometry, sf_geometry_t

    run = _independent_run(mx, monkeypatch, width, b=3, hq=2, hkv=1, sq=17, skv=33, out_dt=dtype, causal=True, left_bound=9)
    sources = {n: run.pack[t].clone() for n, t in run.refs.items() if n not in ("dq", "dk", "dv")}
    before = snapshot(sources)
    plans = {}

    def reorder(name, value, order):
        if name.startswith("sf_"):
            kv = name in ("sf_k", "sf_k_T", "sf_v")
            h, s = (1, 33) if kv else (2, 17)
            if name.endswith("_T"):
                shape, _ = sf_geometry_t(3, h, s, width)
                data = value.view(torch.uint8).reshape(shape).reshape((width + 127) // 128, 3, h, *shape[2:])
                return torch.cat([data[:, i : i + 1] for i in order], 1).contiguous().reshape(-1)
            shape, _ = sf_geometry(3, h, s, width)
            data = value.view(torch.uint8).reshape(shape)
            return torch.cat([data[i : i + 1] for i in order], 0).contiguous().reshape(-1)
        data = torch.cat([value[i : i + 1] for i in order], 0)
        return data.contiguous() if name == "stats" else mx._to_bshd(data)

    def prepare(order):
        b = len(order)
        if b not in plans:
            graph, ports, outputs = mx._build_graph(
                b, 2, 1, 17, 33, width, out_dt=dtype, scale=run.scale, use_causal_mask=True, left_bound=9, use_deterministic_algorithm=True
            )
            graph.create_execution_plans([cudnn.heur_mode.A])
            select_engine(graph, "sdpa_bwd_sm120_mxfp8")
            graph.check_support()
            graph.build_plans()
            assert graph.get_workspace_size() < 4 * b * 2 * 128 + 256
            plans[b] = graph, ports, outputs, torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
        graph, ports, outputs, workspace = plans[b]
        inputs = {name: reorder(name, value, order) for name, value in sources.items()}
        grads = {name: mx._to_bshd(torch.empty((b, *x.shape[1:]), device="cuda", dtype=dtype)) for name, x in zip(("dQ", "dK", "dV"), run.expected)}
        pack = {ports[name]: value for name, value in inputs.items()}
        pack.update(zip(outputs[:3], grads.values()))

        def execute():
            for x in grads.values():
                x.fill_(float("nan"))
            workspace.fill_(255)
            graph.execute(pack, workspace)
            return grads

        return graph, pack, workspace, inputs, grads, execute

    singles = [snapshot(prepare([i])[-1]()) for i in range(3)]
    for i in range(3):
        assert_bitwise(singles[i], {n: v[i : i + 1] for n, v in zip(("dQ", "dK", "dV"), run.expected)}, "singleton")
    baseline = None
    for order in ([0, 1, 2], [2, 0, 1], [1, 0, 1, 2], [0, 2, 1] * 43):
        graph, pack, workspace, inputs, grads, execute = prepare(order)
        got = assert_replays(execute, repeats=3)
        expected = {n: torch.cat([singles[i][n] for i in order], 0) for n in grads}
        assert_bitwise(got, expected, "batch reorder and duplicate")
        if order == [0, 1, 2]:
            baseline = got
        capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture):
            graph.execute(pack, workspace)
        inputs["dO"].view(torch.uint8).bitwise_xor_(128)
        inputs["dO_T"].view(torch.uint8).bitwise_xor_(128)
        inputs["dO_f16"].neg_()
        changed = snapshot(execute())
        for n in grads:
            torch.testing.assert_close(changed[n], -got[n], rtol=0, atol=0)
        # Changing a scale atom after capture must be read by the captured path.
        inputs["sf_dO"].view(torch.uint8).add_(1)
        changed_scale = snapshot(execute())
        assert any(not torch.equal(changed_scale[n].view(torch.uint8), changed[n].view(torch.uint8)) for n in grads)
        for x in grads.values():
            x.fill_(float("nan"))
        workspace.fill_(255)
        capture.replay()
        assert_bitwise(snapshot(grads), changed_scale, "captured payload and canonical SF mutation")
        torch.cuda.set_sync_debug_mode("error")
        try:
            graph.execute(pack, workspace)
        finally:
            torch.cuda.set_sync_debug_mode("default")
    assert_bitwise(snapshot(sources), before, "inputs preserved")
    assert_golden("sdpa_sm120_mxfp8_bwd_d" + str(width) + "_" + str(dtype).split(".")[-1], before, baseline)


@pytest.mark.parametrize("dims", [(64, 64), (192, 128), (512, 512)])
@pytest.mark.parametrize("shape", [(65, 33, 4, 4, False, None), (129, 257, 8, 1, True, 127), (97, 19, 4, 2, False, 7)])
def test_longer_and_keyless(monkeypatch, dims, shape):
    sq, sk, h, hk, causal, left = shape
    kw = {} if left is None else {"left_bound": left}
    _independent_run(mx, monkeypatch, dims[0], dv=dims[1], b=2, hq=h, hkv=hk, sq=sq, skv=sk, out_dt=torch.float16, causal=causal, repeat_outputs=3, **kw)
