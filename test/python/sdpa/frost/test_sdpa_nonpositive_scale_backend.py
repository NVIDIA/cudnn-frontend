# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native SDPA scale-sign correctness, including plans an autotuner can select.

Backward receives independently computed O/Stats: a broken forward must not be
mistaken for a separate backward defect. CPU-generated inputs are identical on
different GPU architectures; the oracle uses FP32 and the actual rounded inputs.
"""

import pytest
import torch

import cudnn
from cudnn.engines.engine_ids import is_python_engine

pytestmark = pytest.mark.L0


@pytest.fixture(autouse=True)
def native_only(monkeypatch):
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "0")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("native half SDPA requires an Ampere or newer GPU")
    if cudnn.backend_version() < 91600:
        pytest.skip("the nonpositive-scale regression is qualified from cuDNN 9.16")


def _inputs(dtype, scale, causal):
    gen = torch.Generator().manual_seed(1412)
    q, k, v, do = [torch.randn(1, 64, 2, 64, generator=gen).to(device="cuda", dtype=dtype).transpose(1, 2) for _ in range(4)]
    qr, kr, vr = [x.float().detach().requires_grad_(True) for x in (q, k, v)]
    logits = (qr @ kr.transpose(-1, -2)) * scale
    if causal:
        logits = logits.masked_fill(torch.ones(64, 64, device="cuda", dtype=torch.bool).triu(1), -float("inf"))
    o = torch.softmax(logits, -1) @ vr
    stats = torch.logsumexp(logits.detach(), -1, keepdim=True)
    o.backward(do.float())
    return (q, k, v, do), o.detach(), stats, tuple(x.grad for x in (qr, kr, vr))


def _graph(data, ref, stats, scale, causal, wants_stats, backward, scale_tensor=None, handle=None, retained_inputs=None, deterministic=False):
    q, k, v, do = data
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.HALF if q.dtype == torch.float16 else cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=handle,
    )
    Q, K, V = [g.tensor_like(t) for t in (q, k, v)]
    pack = {Q: q, K: k, V: v}
    if scale_tensor is not None:
        scale = g.tensor_like(scale_tensor)
        pack[scale] = scale_tensor
    if backward:
        o = ref.to(q.dtype).transpose(1, 2).contiguous().transpose(1, 2)
        O, DO, ST = [g.tensor_like(t) for t in (o, do, stats)]
        pack.update({O: o, DO: do, ST: stats})
        if retained_inputs is not None:
            retained_inputs.update(o=o, stats=stats)
        outputs = g.sdpa_backward(q=Q, k=K, v=V, o=O, dO=DO, stats=ST, attn_scale=scale, use_causal_mask=causal, use_deterministic_algorithm=deterministic)
        carriers = [torch.full_like(t, float("nan")) for t in (q, k, v)]
    else:
        O, ST = g.sdpa(q=Q, k=K, v=V, attn_scale=scale, use_causal_mask=causal, generate_stats=wants_stats)
        outputs = [O, ST] if wants_stats else [O]
        carriers = [torch.full_like(q, float("nan"))]
        if wants_stats:
            carriers.append(torch.full_like(stats, float("nan")))
    for desc, carrier in zip(outputs, carriers):
        desc.set_output(True).set_dim(carrier.shape).set_stride(carrier.stride())
        if carrier.dtype == torch.float32:
            desc.set_data_type(cudnn.data_type.FLOAT)
        pack[desc] = carrier
    g.validate()
    g.build_operation_graph()
    return g, pack, carriers


def _execute(g, pack):
    g.select_plan(0)
    g.check_support()
    g.build_plans()
    workspace = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    g.execute(pack, workspace)
    torch.cuda.synchronize()


def _check_outputs(actual, expected, dtype):
    for result, reference in zip(actual, expected):
        # Check values as well as finiteness: SM100 negative-scale unmasked O
        # can be finite and wrong while its Stats are -inf (#1428).
        torch.testing.assert_close(result.float(), reference, atol=0.02 if dtype == torch.bfloat16 else 0.003, rtol=0.02)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("scale", [0.0, -0.125, 0.125], ids=["zero", "negative", "positive"])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("mode", ["inference", "training", "backward", "deterministic-backward"])
def test_native_frost_sdpa_scale_sign_all_offered_plans(dtype, scale, causal, mode):
    data, ref, stats, grads = _inputs(dtype, scale, causal)
    backward, wants_stats = mode.endswith("backward"), mode == "training"
    deterministic = mode == "deterministic-backward"
    if deterministic and torch.cuda.get_device_capability()[0] == 10 and cudnn.backend_version() < 91800:
        pytest.skip("Blackwell deterministic backward requires cuDNN 9.18 or newer")
    make = lambda: _graph(data, ref, stats, scale, causal, wants_stats, backward, deterministic=deterministic)  # noqa: E731
    g, _, _ = make()
    g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    records = [g.get_engine_and_knobs_at_index(i) for i, plan in enumerate(g.plans) if not is_python_engine(plan.engine_id)]
    assert records, "the native fallback must remain available for nonpositive scales"
    expected = grads if backward else ([ref, stats] if wants_stats else [ref])
    for engine, knobs in records:
        # Replay the public autotune record on a fresh graph. This catches a
        # guard that hides heuristics entries but leaves explicit pins unsafe.
        selected, pack, outputs = make()
        selected.create_execution_plan(engine, knobs)
        _execute(selected, pack)
        _check_outputs(outputs, expected, dtype)


@pytest.mark.parametrize("backward", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
def test_native_frost_tensor_scale_changes_after_build_and_capture(cudnn_handle, backward, dtype, causal):
    """The runtime scalar changes sign without rebuilding or synchronizing it to the host."""
    data, ref, stats, grads = _inputs(dtype, 0.125, causal)
    scale_tensor = torch.tensor([0.125], device="cuda", dtype=torch.float32).reshape(1, 1, 1, 1)
    retained_inputs = {}
    make = lambda: _graph(data, ref, stats, 0.125, causal, True, backward, scale_tensor, cudnn_handle, retained_inputs)  # noqa: E731
    g, _, _ = make()
    g.create_execution_plans([cudnn.heur_mode.OPENSOURCE, cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    from cudnn.engines.engine_ids import BACKEND_HEURISTIC_ENGINE_ID

    records = [
        g.get_engine_and_knobs_at_index(i) for i, p in enumerate(g.plans) if not is_python_engine(p.engine_id) and p.engine_id != BACKEND_HEURISTIC_ENGINE_ID
    ]
    opaque = any(p.engine_id == BACKEND_HEURISTIC_ENGINE_ID for p in g.plans)
    assert records
    # An opaque delegate cannot bypass the engine exclusions.
    if g._backend_engine_exclusions() or g._backend_plan_guard() is not None:
        assert not opaque
    if opaque:
        records.append((None, None))
    previous = cudnn.get_stream(cudnn_handle)
    stream = torch.cuda.Stream()
    try:
        cudnn.set_stream(cudnn_handle, stream.cuda_stream)
        for engine, knobs in records:
            selected, pack, outputs = make()
            if engine is None:
                selected.create_execution_plans([cudnn.heur_mode.OPENSOURCE, cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
            else:
                selected.create_execution_plan(engine, knobs)
            selected.select_plan(0)
            selected.check_support()
            selected.build_plans()
            workspace = torch.empty(max(selected.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                selected.execute(pack, workspace, handle=cudnn_handle)
            torch.cuda.current_stream().wait_stream(stream)
            captured = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(captured, stream=stream):
                    selected.execute(pack, workspace, handle=cudnn_handle)
                for value in (0.125, -0.125, 0.0):
                    _, new_ref, new_stats, new_grads = _inputs(dtype, value, causal)
                    scale_tensor.fill_(value)
                    if backward:
                        # Update the retained O/Stats inputs from the independent
                        # forward oracle, never from an affected native forward.
                        retained_inputs["o"].copy_(new_ref)
                        retained_inputs["stats"].copy_(new_stats)
                    for result in outputs:
                        result.fill_(float("nan"))
                    captured.replay()
                    torch.cuda.synchronize()
                    _check_outputs(outputs, new_grads if backward else [new_ref, new_stats], dtype)
            finally:
                captured.reset()
    finally:
        cudnn.set_stream(cudnn_handle, previous)


def test_native_frost_scale_guard_currency(monkeypatch):
    """Re-measure every offered affected engine; a passing matrix retires its guard."""
    from cudnn.sdpa import nonpositive_scale_guard as guard
    from frost_routing import measured

    cc = torch.cuda.get_device_capability()
    targets = {
        (8, 0): {True: (7,)},
        (8, 9): {True: (7,)},
        (9, 0): {False: (2, 9), True: (6, 7)},
        (10, 0): {False: (3, 10), True: (5, 7)},
        (10, 3): {False: (3, 10), True: (5, 7)},
    }
    if cc not in targets:
        pytest.skip("no nonpositive-scale exclusion on this architecture")
    bounds = {False: dict(guard.FORWARD_FIXED_IN), True: {**guard.BACKWARD_FIXED_IN, **guard.NONDETERMINISTIC_BACKWARD_FIXED_IN}}
    starts = dict(guard.NONDETERMINISTIC_BACKWARD_BROKEN_SINCE)
    monkeypatch.setattr(guard, "FORWARD_FIXED_IN", dict.fromkeys(guard.FORWARD_FIXED_IN, 0))
    monkeypatch.setattr(guard, "BACKWARD_FIXED_IN", dict.fromkeys(guard.BACKWARD_FIXED_IN, 0))
    monkeypatch.setattr(guard, "NONDETERMINISTIC_BACKWARD_FIXED_IN", dict.fromkeys(guard.NONDETERMINISTIC_BACKWARD_FIXED_IN, 0))
    outcomes = {}
    import itertools

    for backward, engines in targets[cc].items():
        for dtype, scale, causal, wants_stats in itertools.product(
            (torch.float16, torch.bfloat16), (0.0, -0.125, 0.125), (False, True), (False, True) if not backward else (True,)
        ):
            data, ref, stats, grads = _inputs(dtype, scale, causal)
            g, pack, outputs = _graph(data, ref, stats, scale, causal, wants_stats, backward)
            g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
            records = [g.get_engine_and_knobs_at_index(i) for i, p in enumerate(g.plans) if p.engine_id in engines]
            for engine, knobs in records:
                selected, pack, outputs = _graph(data, ref, stats, scale, causal, wants_stats, backward)
                selected.create_execution_plan(engine, knobs)
                _execute(selected, pack)
                good = True
                try:
                    _check_outputs(outputs, grads if backward else ([ref, stats] if wants_stats else [ref]), dtype)
                except AssertionError:
                    good = False
                outcomes.setdefault((backward, engine), []).append((scale, good))
                if scale > 0:
                    assert good, f"the positive-scale control failed on engine {engine}"
    if not outcomes:
        pytest.skip("installed backend offers none of the measured engines")
    for (backward, engine), results in outcomes.items():
        wrong = any(not good for scale, good in results if scale <= 0)
        bound = bounds[backward][engine]
        since = starts.get(engine, 0) if backward else 0
        armed = cudnn.backend_version() >= since and (bound is None or cudnn.backend_version() < bound)
        measured(f"nonpositive-scale-{cc}-{backward}-{engine}", f"backend={cudnn.backend_version()} wrong={wrong}, {results}")
        assert wrong == armed, (
            f"engine {engine}, backward={backward}, cuDNN {cudnn.backend_version()}: "
            f"wrong={wrong}, recorded affected interval=[{since}, {bound}); requalify and update the per-engine bounds"
        )


@pytest.mark.parametrize(
    "cc,backward,engines", [((9, 0), False, (2, 9)), ((10, 0), False, (3, 10)), ((10, 3), False, (3, 10)), ((10, 0), True, (5,)), ((10, 3), True, (5,))]
)
@pytest.mark.parametrize("scale", [0.0, -0.125, "tensor"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("version", [91600, 92800])
def test_native_frost_unsafe_explicit_record_declines_before_lowering(monkeypatch, cc, backward, engines, scale, dtype, version):
    from cudnn.sdpa import graph_analyzer

    monkeypatch.setattr(graph_analyzer, "_device_cc", lambda: cc)
    monkeypatch.setattr(cudnn, "backend_version", lambda: version)
    value = 0.125 if scale == "tensor" else scale
    data, ref, stats, _ = _inputs(dtype, value, False)
    tensor = torch.tensor([value], device="cuda", dtype=torch.float32).reshape(1, 1, 1, 1) if scale == "tensor" else None
    g, _, _ = _graph(data, ref, stats, value, False, True, backward, tensor)
    facts = graph_analyzer.analyze(g)
    assert facts.invalid is None and facts.dynamic_scale == (scale == "tensor")
    assert facts.is_backward == backward
    monkeypatch.setattr(g, "_lower_backend_graph", lambda: pytest.fail("unsafe explicit record reached native lowering"))
    for engine in engines:
        with pytest.raises(cudnn.cudnnGraphNotSupportedError, match="#1428"):
            g.create_execution_plan(engine, {})


def test_native_frost_scale_guard_preserves_other_domains_and_bounds(monkeypatch):
    from dataclasses import replace
    from cudnn.sdpa.graph_analyzer import SdpaGraphFacts
    from cudnn.sdpa import nonpositive_scale_guard as guard

    monkeypatch.setattr(cudnn, "backend_version", lambda: 92800)
    base = SdpaGraphFacts(dtype=cudnn.data_type.HALF, scale=0.0, device_cc=(10, 0))
    assert set(guard.backend_engine_guard(base)) == {3, 10}
    for change in (
        dict(scale=0.125),
        dict(scale=None),
        dict(dtype=cudnn.data_type.FLOAT),
        dict(is_fp8=True),
        dict(is_mxfp8=True),
        dict(device_cc=(8, 0)),
        dict(device_cc=(8, 9)),
        dict(device_cc=(12, 0)),
        dict(device_cc=None),
        dict(invalid="malformed"),
        dict(device_cc=(9, 0), is_backward=True, deterministic=True),
    ):
        assert not guard.backend_engine_guard(replace(base, **change)), change
    assert not guard.backend_engine_guard(None)
    monkeypatch.setattr(guard, "FORWARD_FIXED_IN", {**guard.FORWARD_FIXED_IN, 3: 92800, 10: 92801})
    assert set(guard.backend_engine_guard(base)) == {10}
    monkeypatch.setattr(cudnn, "backend_version", lambda: 92801)
    assert not guard.backend_engine_guard(base)
    # No lower-version speculation: the older legacy engines are unsafe too.
    monkeypatch.setattr(cudnn, "backend_version", lambda: 91000)
    assert set(guard.backend_engine_guard(base)) == {3, 10}


def test_native_frost_scale_guard_composes_with_existing_exclusions(monkeypatch):
    from cudnn.sdpa.graph_analyzer import SdpaGraphFacts
    from cudnn.sdpa.fwd.backend_guard import backend_engine_guard as forward_guard
    from cudnn.sdpa.bwd.backend_guard import backend_engine_guard as backward_guard

    monkeypatch.setattr(cudnn, "backend_version", lambda: 92400)
    wide_v = SdpaGraphFacts(dtype=cudnn.data_type.HALF, scale=0.0, device_cc=(10, 0), d_qk=64, d_v=256)
    excluded = forward_guard(None, wide_v)
    assert set(excluded) == {3, 10, 18}
    assert "#2401" in excluded[10] and "#1428" in excluded[3]
    ragged = SdpaGraphFacts(dtype=cudnn.data_type.BFLOAT16, scale=0.0, device_cc=(9, 0), d_qk=256, d_v=128, thd=True, is_backward=True)
    assert set(backward_guard(None, ragged)) == {6}


@pytest.mark.parametrize("version", [91600, 91801, 92400, 92500, 92501, 92600, 92700, 92800])
@pytest.mark.parametrize("dynamic", [False, True])
def test_native_frost_hopper_backward_scale_regression_boundary(monkeypatch, version, dynamic):
    from dataclasses import replace
    from cudnn.sdpa.graph_analyzer import SdpaGraphFacts
    from cudnn.sdpa import nonpositive_scale_guard as guard

    monkeypatch.setattr(cudnn, "backend_version", lambda: version)
    facts = SdpaGraphFacts(dtype=cudnn.data_type.HALF, scale=0.0, dynamic_scale=dynamic, device_cc=(9, 0), is_backward=True)
    assert set(guard.backend_engine_guard(facts)) == ({6} if version >= 92600 else set())
    assert not guard.backend_engine_guard(replace(facts, causal=True))
    assert not guard.backend_engine_guard(replace(facts, deterministic=True))
    # The independent engine-7 bug remains guarded on both sides of engine 6's
    # introduction boundary, while its correct configurations stay available.
    check = guard.backend_plan_guard(None, facts)
    assert check(7, {cudnn.knob_type.KERNEL_CFG: 28}) is not None
    assert check(7, {cudnn.knob_type.KERNEL_CFG: 26}) is None
    monkeypatch.setattr(guard, "NONDETERMINISTIC_BACKWARD_FIXED_IN", {6: version, 7: None})
    assert not guard.backend_engine_guard(facts)


@pytest.mark.parametrize("scale", [0.0, -0.125, "tensor"])
@pytest.mark.parametrize(
    "knobs",
    [{}, {cudnn.knob_type.KERNEL_CFG: 28}, {cudnn.knob_type.KERNEL_CFG: 2}, {cudnn.knob_type.KERNEL_CFG: 2, cudnn.knob_type.TILE_N: 2}],
)
def test_native_frost_unsafe_config_replay_declines_before_lowering(monkeypatch, scale, knobs):
    monkeypatch.setattr(cudnn, "backend_version", lambda: 92800)
    value = 0.125 if scale == "tensor" else scale
    data, ref, stats, _ = _inputs(torch.float16, value, False)
    tensor = torch.tensor([value], device="cuda", dtype=torch.float32).reshape(1, 1, 1, 1) if scale == "tensor" else None
    g, _, _ = _graph(data, ref, stats, value, False, True, True, tensor)
    monkeypatch.setattr(g, "_lower_backend_graph", lambda: pytest.fail("unsafe config reached native lowering"))
    with pytest.raises(cudnn.cudnnGraphNotSupportedError, match="#1428"):
        g.create_execution_plan(7, knobs)


def test_native_frost_config_guard_preserves_safe_siblings_and_fixed_version(monkeypatch):
    from dataclasses import replace
    from cudnn.sdpa.graph_analyzer import SdpaGraphFacts
    from cudnn.sdpa import nonpositive_scale_guard as guard

    monkeypatch.setattr(cudnn, "backend_version", lambda: 92800)
    facts = SdpaGraphFacts(dtype=cudnn.data_type.BFLOAT16, scale=0.0, device_cc=(8, 0), is_backward=True)
    check = guard.backend_plan_guard(None, facts)
    assert check(7, {cudnn.knob_type.KERNEL_CFG: 28}) is not None
    assert check(7, {cudnn.knob_type.KERNEL_CFG: 26}) is None
    assert check(7, {cudnn.knob_type.KERNEL_CFG: 2, cudnn.knob_type.TILE_N: 1}) is None
    assert check(5, {}) is None  # the separate whole-engine guard owns this exclusion
    for change in (dict(scale=0.125), dict(causal=True), dict(deterministic=True), dict(is_backward=False), dict(is_fp8=True)):
        assert guard.backend_plan_guard(None, replace(facts, **change)) is None
    monkeypatch.setattr(guard, "NONDETERMINISTIC_BACKWARD_FIXED_IN", {6: None, 7: 92801})
    assert guard.backend_plan_guard(None, facts) is not None
    monkeypatch.setattr(cudnn, "backend_version", lambda: 92801)
    assert guard.backend_plan_guard(None, facts) is None
