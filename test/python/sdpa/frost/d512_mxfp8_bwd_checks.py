# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Shared D512 MXFP8 checks; architecture entries supply the helper fixture."""

import pytest
import torch


def _independent_run(mx, monkeypatch, **kwargs):
    original = torch.randn
    generator = torch.Generator(device="cpu").manual_seed(118037)

    def randn(*shape, **options):
        device = options.pop("device", None)
        return original(*shape, **options, generator=generator, device="cpu").to(device)

    with monkeypatch.context() as patch, torch.cuda.stream(torch.cuda.Stream()):
        patch.setattr(torch, "randn", randn)
        return mx._run(d=512, **kwargs)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize(
    "shape", [(129, 257, 2, 1, True, 127), (257, 129, 4, 2, False, None), (17, 256, 1, 1, True, 1)], ids=["gqa-swa-tail", "rect-gqa-full", "short-empty-k"]
)
def test_graph_numerics(d512_mx, monkeypatch, dtype, shape):
    sq, sk, h, hk, causal, left = shape
    kwargs = {} if left is None else {"left_bound": left}
    _independent_run(d512_mx, monkeypatch, sq=sq, skv=sk, hq=h, hkv=hk, out_dt=dtype, causal=causal, repeat_outputs=4, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(d=256),
        dict(dv=256),
        dict(with_sink=True),
        dict(declare_amax=True),
        dict(use_causal_mask_bottom_right=True),
        dict(right_bound=4),
        dict(use_padding_mask=True, seq_len_dims=(1, 1, 1, 1)),
    ],
)
def test_declines_unsupported(d512_mx, kwargs):
    assert d512_mx._decline_reason(**{"d": 512, **kwargs}) is not None


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_replay_batch_and_golden(d512_mx, monkeypatch, dtype):
    import cudnn
    from determinism_utils import assert_bitwise, assert_golden, assert_replays, snapshot
    from frost_test_utils import select_engine
    from cudnn.sdpa.bwd.kernels.sm100.prepared_d192_fp8_host import sf_geometry, sf_geometry_t

    mx = d512_mx
    run = _independent_run(mx, monkeypatch, b=3, hq=2, hkv=1, sq=17, skv=33, out_dt=dtype, causal=True, left_bound=9)
    sources = {n: run.pack[t].clone() for n, t in run.refs.items() if n not in ("dq", "dk", "dv")}
    before = snapshot(sources)
    plans = {}

    def reorder(name, value, order):
        if name.startswith("sf_"):
            kv = name in ("sf_k", "sf_k_T", "sf_v")
            h, s = (1, 33) if kv else (2, 17)
            if name.endswith("_T"):
                shape, _ = sf_geometry_t(3, h, s, 512)
                data = value.view(torch.uint8).reshape(shape).reshape(4, 3, h, *shape[2:])
                return torch.cat([data[:, i : i + 1] for i in order], 1).contiguous().reshape(-1)
            shape, _ = sf_geometry(3, h, s, 512)
            data = value.view(torch.uint8).reshape(shape)
            return torch.cat([data[i : i + 1] for i in order], 0).contiguous().reshape(-1)
        data = torch.cat([value[i : i + 1] for i in order], 0)
        return data.contiguous() if name == "stats" else mx._to_bshd(data)

    def prepare(order):
        b = len(order)
        if b not in plans:
            graph, ports, outputs = mx._build_graph(
                b, 2, 1, 17, 33, 512, out_dt=dtype, scale=run.scale, use_causal_mask=True, left_bound=9, use_deterministic_algorithm=True
            )
            graph.create_execution_plans([cudnn.heur_mode.A])
            select_engine(graph, "sdpa_bwd_d512_mxfp8")
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
    assert_golden("sdpa_d512_mxfp8_bwd_" + str(dtype).split(".")[-1], before, baseline)
