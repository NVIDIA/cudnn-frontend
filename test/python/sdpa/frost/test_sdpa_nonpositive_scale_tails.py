# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Nonpositive backward scales across real sequence tails and legal controls."""

import cudnn
import pytest
import torch

from cudnn.engines.engine_ids import is_python_engine
from test_sdpa_nonpositive_scale_backend import _check_outputs, _execute, _graph, _inputs, native_only  # noqa: F401

pytestmark = pytest.mark.L0

GEOMETRIES = [
    pytest.param((2, 129, 129, 128, 2, 2), id="both-tails"),
    pytest.param((2, 128, 129, 128, 2, 2), id="kv-tail"),
    pytest.param((2, 129, 128, 128, 2, 2), id="q-tail-control"),
    pytest.param((2, 128, 128, 128, 2, 2), id="aligned-control"),
    pytest.param((2, 129, 129, 128, 4, 1), id="gqa-kv-tail"),
    pytest.param((2, 1, 64, 128, 2, 2), id="decode"),
    pytest.param((2, 129, 192, 128, 2, 2), id="kv192-control"),
    pytest.param((2, 129, 129, 64, 2, 2), id="d64-kv-tail"),
    pytest.param((2, 129, 129, 256, 2, 2), id="d256-kv-tail"),
]


def _backward_reference(data, ref, stats, scale, causal):
    """FP64 arithmetic using the exact O/Stats bound to the backward graph.

    O is an independent forward oracle rounded to the input dtype. Backward
    consumes that rounded O in delta = rowsum(O * dO), not its FP64 precursor.
    This matters for D256 without changing the comparison tolerances.
    """
    q, k, v, do = [x.double() for x in data]
    groups = q.shape[1] // k.shape[1]
    kr, vr = [x.repeat_interleave(groups, 1) for x in (k, v)]
    logits = (q @ kr.transpose(-1, -2)) * scale
    if causal:
        logits = logits.masked_fill(torch.ones(q.shape[2], k.shape[2], device=q.device, dtype=torch.bool).triu(1), -float("inf"))
    p = torch.exp(logits - stats.double())
    delta = (ref.to(data[0].dtype).double() * do).sum(-1, keepdim=True)
    ds = p * (do @ vr.transpose(-1, -2) - delta) * scale
    dq = ds @ kr
    dk = (ds.transpose(-1, -2) @ q).reshape(k.shape[0], k.shape[1], groups, k.shape[2], k.shape[3]).sum(2)
    dv = (p.transpose(-1, -2) @ do).reshape(v.shape[0], v.shape[1], groups, v.shape[2], v.shape[3]).sum(2)
    return tuple(x.float() for x in (dq, dk, dv))


@pytest.mark.parametrize("geometry", GEOMETRIES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("scale", [0.0, -0.125, 0.125], ids=["zero", "negative", "positive"])
@pytest.mark.parametrize("causal", [False, True], ids=["unmasked", "causal-control"])
@pytest.mark.parametrize("deterministic", [False, True], ids=["default", "deterministic"])
@pytest.mark.parametrize("fill", [0.0, float("nan")], ids=["zero-init", "nan-init"])
def test_backward_tail_plans_match_independent_oracle(geometry, dtype, scale, causal, deterministic, fill):
    if geometry[3] > 128 and torch.cuda.get_device_capability()[0] not in (9, 10):
        pytest.skip("native SM8X/SM12X backward supports head dimensions up to 128")
    if geometry[3] == 256 and torch.cuda.get_device_capability()[0] == 10 and cudnn.backend_version() < 92300:
        pytest.skip("native Blackwell D256 backward requires cuDNN 9.23 or newer")
    if deterministic and torch.cuda.get_device_capability()[0] == 10 and cudnn.backend_version() < 91800:
        pytest.skip("Blackwell deterministic backward requires cuDNN 9.18 or newer")
    data, ref, stats, _ = _inputs(dtype, scale, causal, geometry=geometry, oracle_dtype=torch.float64)
    grads = _backward_reference(data, ref, stats, scale, causal)

    def make():
        return _graph(data, ref, stats, scale, causal, True, True, deterministic=deterministic)

    g, _, _ = make()
    try:
        g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    except cudnn.cudnnGraphNotSupportedError:
        records = []
    else:
        records = [g.get_engine_and_knobs_at_index(i) for i, plan in enumerate(g.plans) if not is_python_engine(plan.engine_id)]
    if not records:
        # A diagnosed nonpositive-scale domain may have no correct native
        # fallback. Verify the explicit-record refusal too, rather than skip.
        blackwell_d256 = torch.cuda.get_device_capability()[0] == 10 and geometry[3] == 256
        assert scale <= 0 and (not causal or blackwell_d256)
        excluded = g._backend_engine_exclusions()
        assert (5 if blackwell_d256 else 7) in excluded
        for engine in excluded:
            selected, _, _ = make()
            with pytest.raises(cudnn.cudnnGraphNotSupportedError, match="#1428"):
                selected.create_execution_plan(engine, {cudnn.knob_type.KERNEL_CFG: 2, cudnn.knob_type.TILE_N: 1})
        return
    for engine, knobs in records:
        selected, pack, outputs = make()
        selected.create_execution_plan(engine, knobs)
        for output in outputs:
            output.fill_(fill)
        _execute(selected, pack)
        _check_outputs(outputs, grads, dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("scale", [0.0, -0.125, 0.125])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("wants_stats", [False, True])
def test_forward_partial_tile_control(dtype, scale, causal, wants_stats):
    data, ref, stats, _ = _inputs(dtype, scale, causal, geometry=(2, 129, 129, 128, 2, 2), oracle_dtype=torch.float64)
    make = lambda: _graph(data, ref, stats, scale, causal, wants_stats, False)
    g, _, _ = make()
    g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    records = [g.get_engine_and_knobs_at_index(i) for i, plan in enumerate(g.plans) if not is_python_engine(plan.engine_id)]
    assert records
    for engine, knobs in records:
        selected, pack, outputs = make()
        selected.create_execution_plan(engine, knobs)
        _execute(selected, pack)
        _check_outputs(outputs, [ref, stats] if wants_stats else [ref], dtype)


@pytest.mark.parametrize("cc", [(8, 0), (8, 9), (9, 0), (10, 0), (10, 3)])
@pytest.mark.parametrize("version", [91600, 92400, 92800])
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("deterministic", [False, True])
def test_partial_or_variable_kv_guard_contract(monkeypatch, cc, version, dynamic, deterministic):
    from dataclasses import replace
    from cudnn.sdpa import nonpositive_scale_guard as guard
    from cudnn.sdpa.graph_analyzer import SdpaGraphFacts

    monkeypatch.setattr(cudnn, "backend_version", lambda: version)
    facts = SdpaGraphFacts(
        b=2,
        h_q=2,
        h_kv=2,
        s_q=129,
        s_kv=129,
        d_qk=128,
        d_v=128,
        dtype=cudnn.data_type.HALF,
        is_backward=True,
        device_cc=cc,
        scale=0.125 if dynamic else 0.0,
        dynamic_scale=dynamic,
        deterministic=deterministic,
    )
    ordinary = {5} if cc[0] == 10 else {6} if cc == (9, 0) and version >= 92600 else set()
    tails = {6, 7} if cc == (9, 0) else {7}
    assert set(guard.backend_engine_guard(facts)) == ordinary | tails
    # Q alone can have a tail. Explicit padding selects a separate correct
    # path, even when device lengths introduce tails under an aligned maximum.
    assert set(guard.backend_engine_guard(replace(facts, s_kv=192))) == ordinary
    for change in (dict(padded=True), dict(thd=True, padded=True)):
        assert set(guard.backend_engine_guard(replace(facts, s_kv=192, **change))) == ({5} if cc[0] == 10 else set())
    assert set(guard.backend_engine_guard(replace(facts, causal=True))) == ({5} if cc[0] == 10 else set())
    assert not guard.backend_engine_guard(replace(facts, scale=0.125, dynamic_scale=False))
    monkeypatch.setattr(guard, "BACKWARD_KV_TAIL_FIXED_IN", dict.fromkeys((6, 7), version))
    assert set(guard.backend_engine_guard(facts)) == ordinary


@pytest.mark.parametrize("scale", [0.0, -0.125, "tensor"])
@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("cfg", [2, 26], ids=["current", "legacy"])
def test_partial_kv_public_record_declines_before_lowering(monkeypatch, scale, deterministic, cfg):
    from cudnn.sdpa import graph_analyzer

    monkeypatch.setattr(graph_analyzer, "_device_cc", lambda: (10, 0))
    value = 0.125 if scale == "tensor" else scale
    data, ref, stats, _ = _inputs(torch.float16, value, False, geometry=(2, 129, 129, 128, 2, 2))
    tensor = torch.tensor([value], device="cuda", dtype=torch.float32).reshape(1, 1, 1, 1) if scale == "tensor" else None
    g, _, _ = _graph(data, ref, stats, value, False, True, True, tensor, deterministic=deterministic)
    monkeypatch.setattr(g, "_lower_backend_graph", lambda: pytest.fail("unsafe tail record reached native lowering"))
    with pytest.raises(cudnn.cudnnGraphNotSupportedError, match="#1428"):
        g.create_execution_plan(7, {cudnn.knob_type.KERNEL_CFG: cfg, cudnn.knob_type.TILE_N: 1})


def test_partial_kv_backend_guard_currency(monkeypatch):
    """Re-measure the tail failure separately from the aligned atomic bug."""
    from cudnn.sdpa import nonpositive_scale_guard as guard
    from frost_routing import measured

    cc = torch.cuda.get_device_capability()
    if cc not in ((8, 0), (8, 9), (9, 0), (10, 0), (10, 3)):
        pytest.skip("no measured partial-KV native exclusion on this architecture")
    targets = (6, 7) if cc == (9, 0) else (7,)
    bounds = dict(guard.BACKWARD_KV_TAIL_FIXED_IN)
    monkeypatch.setattr(guard, "BACKWARD_KV_TAIL_FIXED_IN", dict.fromkeys(bounds, 0))
    monkeypatch.setattr(guard, "NONDETERMINISTIC_BACKWARD_FIXED_IN", dict.fromkeys((6, 7), 0))
    outcomes = {engine: [] for engine in targets}
    for scale in (0.0, -0.125, 0.125):
        data, ref, stats, grads = _inputs(torch.float16, scale, False, geometry=(2, 129, 129, 128, 2, 2), oracle_dtype=torch.float64)
        make = lambda: _graph(data, ref, stats, scale, False, True, True)
        g, _, _ = make()
        g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        records = [g.get_engine_and_knobs_at_index(i) for i, p in enumerate(g.plans) if p.engine_id in targets]
        for engine, knobs in records:
            selected, pack, outputs = make()
            selected.create_execution_plan(engine, knobs)
            for output in outputs:
                output.zero_()
            _execute(selected, pack)
            good = True
            try:
                _check_outputs(outputs, grads, torch.float16)
            except AssertionError:
                good = False
            if scale > 0:
                assert good, (engine, knobs)
            outcomes[engine].append((scale, good))
    for engine, results in outcomes.items():
        assert results, f"tail currency detector did not exercise engine {engine}"
        wrong = any(scale <= 0 and not good for scale, good in results)
        armed = bounds[engine] is None or cudnn.backend_version() < bounds[engine]
        measured(f"nonpositive-KV-tail-{cc}-{engine}", f"backend={cudnn.backend_version()} wrong={wrong}, {results}")
        assert wrong == armed, (engine, results, bounds[engine])
