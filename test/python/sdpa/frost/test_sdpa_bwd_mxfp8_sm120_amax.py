# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Optional MXFP8 gradient amax outputs through the common graph contract."""

import pytest
import torch
import cudnn
import test_sdpa_bwd_mxfp8_sm100 as mx
from test_sdpa_bwd_mxfp8_sm120 import native, _independent_run
from frost_test_utils import requires_dsl, select_engine
from determinism_utils import assert_bitwise, assert_golden, assert_replays, snapshot

pytestmark = [pytest.mark.L0, requires_dsl]
_NAMES = ("amax_dQ", "amax_dK", "amax_dV")


def _prepare(run, requested, order=None):
    from cudnn.sdpa.bwd.kernels.sm100.prepared_d192_fp8_host import sf_geometry, sf_geometry_t

    source = {n: run.pack[t] for n, t in run.refs.items() if n not in ("dq", "dk", "dv")}
    b0, h, sq, d = source["q"].shape
    _, hk, sk, dv = source["v"].shape
    order = list(range(b0)) if order is None else order
    b = len(order)
    graph, ports, outputs = mx._build_graph(
        b,
        h,
        hk,
        sq,
        sk,
        d,
        dv=dv,
        out_dt=run.expected[0].dtype,
        scale=run.scale,
        use_causal_mask=True,
        left_bound=9,
        use_deterministic_algorithm=True,
        declare_amax=requested,
    )
    graph.create_execution_plans([cudnn.heur_mode.A])
    select_engine(graph, "sdpa_bwd_sm120_mxfp8")
    graph.check_support()
    graph.build_plans()
    assert graph.get_workspace_size() < 4 * b * h * ((sq + 127) // 128 * 128) + 256

    def reorder(name, value):
        if name.startswith("sf_"):
            heads, seq = (hk, sk) if name in ("sf_k", "sf_k_T", "sf_v") else (h, sq)
            width = dv if name in ("sf_v", "sf_dO", "sf_dO_T") else d
            if name.endswith("_T"):
                shape, _ = sf_geometry_t(b0, heads, seq, width)
                data = value.view(torch.uint8).reshape(shape).reshape((width + 127) // 128, b0, heads, *shape[2:])
                return torch.cat([data[:, i : i + 1] for i in order], 1).contiguous().reshape(-1)
            shape, _ = sf_geometry(b0, heads, seq, width)
            data = value.view(torch.uint8).reshape(shape)
            return torch.cat([data[i : i + 1] for i in order], 0).contiguous().reshape(-1)
        data = torch.cat([value[i : i + 1] for i in order], 0)
        return data.contiguous() if name == "stats" else mx._to_bshd(data)

    inputs = {n: reorder(n, v) for n, v in source.items()}
    result = {name: mx._to_bshd(torch.empty((b, *ref.shape[1:]), device="cuda", dtype=ref.dtype)) for name, ref in zip(("dQ", "dK", "dV"), run.expected)}
    result.update({name: torch.empty((1, 1, 1, 1), device="cuda", dtype=torch.float32) for name in requested})
    pack = {ports[n]: v for n, v in inputs.items()}
    pack.update(zip(outputs[:3], (result[n] for n in ("dQ", "dK", "dV"))))
    pack.update({tensor: result[name] for name, tensor in zip(_NAMES, outputs[3:]) if name in requested})
    ws = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

    def execute():
        for value in result.values():
            value.fill_(float("nan"))
        ws.fill_(255)
        graph.execute(pack, ws)
        return result

    return graph, inputs, result, pack, ws, execute


def _check_amax(result):
    for grad, name in zip(("dQ", "dK", "dV"), _NAMES):
        if name not in result:
            continue
        value = result[name]
        assert torch.isfinite(value).all()
        # amax is the FP32 accumulator magnitude before the half store. Bound
        # only that final rounding, independently of the gradient oracle.
        expected = result[grad].float().abs().max().reshape_as(value)
        rtol = 0.004 if result[grad].dtype == torch.bfloat16 else 0.0005
        torch.testing.assert_close(value, expected, rtol=rtol, atol=1e-6)


@pytest.mark.parametrize("bits", range(8))
def test_optional_outputs(monkeypatch, bits):
    requested = tuple(name for i, name in enumerate(_NAMES) if bits & (1 << i))
    run = _independent_run(mx, monkeypatch, 192, dv=128, b=2, hq=4, hkv=2, sq=19, skv=37, causal=True, left_bound=9)
    graph, inputs, result, pack, ws, execute = _prepare(run, requested)
    before = snapshot(inputs)
    got = assert_replays(execute, repeats=3)
    assert_bitwise({n: got[n] for n in ("dQ", "dK", "dV")}, dict(zip(("dQ", "dK", "dV"), run.expected)), "amax leaves gradient bytes unchanged")
    _check_amax(got)
    capture = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(capture):
            graph.execute(pack, ws)
        for value in result.values():
            value.fill_(float("inf"))
        ws.fill_(255)
        capture.replay()
        assert_bitwise(snapshot(result), got, "capture overwrites poisoned amax and gradients")
        assert_bitwise(snapshot(inputs), before, "input preservation")
    finally:
        capture.reset()


@pytest.mark.parametrize("requested", [(), _NAMES], ids=["none", "all"])
def test_standalone_binding(monkeypatch, requested):
    from cudnn.sdpa.bwd.api_dsl_sm120_mxfp8 import SdpaBwdDslSm120Mxfp8

    run = _independent_run(mx, monkeypatch, 192, dv=128, b=2, hq=4, hkv=2, sq=19, skv=37, causal=True, left_bound=9)
    _, inputs, result, _, ws, execute = _prepare(run, requested)
    expected = snapshot(execute())
    ports = dict(q="q", k="k", v="v", o="o_f16", do="dO", stats="stats", q_T="q_T", k_T="k_T", do_T="dO_T", do_f16="dO_f16")
    tensors = {name: inputs[source] for name, source in ports.items()}
    tensors.update(zip(("dq", "dk", "dv"), (result[name] for name in ("dQ", "dK", "dV"))))
    scales = {
        name: inputs[source]
        for name, source in dict(sf_q="sf_q", sf_q_T="sf_q_T", sf_k="sf_k", sf_k_T="sf_k_T", sf_v="sf_v", sf_do="sf_dO", sf_do_T="sf_dO_T").items()
    }
    api = SdpaBwdDslSm120Mxfp8(
        **{"sample_" + name: value for name, value in {**tensors, **scales}.items()},
        is_causal=True,
        window_size_left=8,  # graph left_bound=9 includes the diagonal
        deterministic=True,
        scale_softmax=run.scale,
        amax_requested=requested,
    )
    api.check_support()
    api.compile()
    args = {name + "_tensor": value for name, value in tensors.items()}
    args.update(scales, workspace=ws)
    args.update({name: result[name] for name in requested})
    for value in result.values():
        value.fill_(float("nan"))
    api.execute(**args)
    assert_bitwise(snapshot(result), expected, "standalone matches graph")
    bad = dict(args)
    if requested:
        bad.pop("amax_dQ")
    else:
        bad["amax_dQ"] = torch.empty(1, device="cuda")
    for value in result.values():
        value.fill_(23)
    ws.fill_(117)
    before = snapshot(dict(result, workspace=ws))
    with pytest.raises(ValueError, match="amax_dQ must be bound"):
        api.execute(**bad)
    assert_bitwise(snapshot(dict(result, workspace=ws)), before, "invalid binding must not launch")
    api.execute(**args)
    assert_bitwise(snapshot(result), expected, "recovery after invalid binding")


@pytest.mark.parametrize("width", [128, 512])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_batch_replay_and_golden(monkeypatch, width, dtype):
    run = _independent_run(mx, monkeypatch, width, b=3, hq=2, hkv=1, sq=17, skv=33, out_dt=dtype, causal=True, left_bound=9)
    singles = [snapshot(_prepare(run, _NAMES, [i])[-1]()) for i in range(3)]
    baseline = None
    for order in ([0, 1, 2], [2, 0, 1], [1, 0, 1, 2], [0, 2, 1] * 43):
        graph, inputs, result, pack, ws, execute = _prepare(run, _NAMES, order)
        got = assert_replays(execute, repeats=3)
        expected = {n: torch.cat([singles[i][n] for i in order], 0) for n in ("dQ", "dK", "dV")}
        expected.update({n: torch.stack([singles[i][n] for i in order]).amax(0) for n in _NAMES})
        assert_bitwise(got, expected, "per-sequence gradients and global amax composition")
        _check_amax(got)
        capture = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(capture):
                graph.execute(pack, ws)
            for name in ("dO", "dO_T"):
                inputs[name].view(torch.uint8).bitwise_xor_(128)
            inputs["dO_f16"].neg_()
            changed = snapshot(execute())
            for n in ("dQ", "dK", "dV"):
                torch.testing.assert_close(changed[n], -got[n], rtol=0, atol=0)
            assert_bitwise({n: changed[n] for n in _NAMES}, {n: got[n] for n in _NAMES}, "negated payload preserves amax")
            for value in result.values():
                value.fill_(float("nan"))
            capture.replay()
            assert_bitwise(snapshot(result), changed, "changed-payload capture")
        finally:
            capture.reset()
        if len(order) == 3 and order == [0, 1, 2]:
            baseline = got
    source = {n: run.pack[t] for n, t in run.refs.items() if n not in ("dq", "dk", "dv")}
    assert_golden(f"sdpa_sm120_mxfp8_amax_d{width}_{str(dtype).split('.')[-1]}", snapshot(source), baseline)
