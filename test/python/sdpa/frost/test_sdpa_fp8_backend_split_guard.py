# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Native FP8 split correctness: exclude configurations, retain their siblings."""

from dataclasses import replace
import json
import math

import cudnn
import pytest

from cudnn.engines import manifest
from cudnn.engines.base import PlanConfig
from cudnn.engines.engine_ids import BACKEND_HEURISTIC_ENGINE_ID
from cudnn.sdpa import graph_analyzer
from cudnn.sdpa.fwd import backend_guard as guard
from cudnn.sdpa.graph_analyzer import SdpaGraphFacts

pytestmark = pytest.mark.L0


def _facts(**changes):
    return replace(
        SdpaGraphFacts(
            b=1,
            h_q=4,
            h_kv=2,
            s_q=128,
            s_kv=8192,
            d_qk=128,
            d_v=128,
            dtype=cudnn.data_type.FP8_E4M3,
            dtype_o=cudnn.data_type.BFLOAT16,
            is_fp8=True,
            device_cc=(10, 0),
        ),
        **changes,
    )


@pytest.mark.parametrize("version,blocked", [(92302, False), (92400, True), (92401, True), (92500, True), (92501, True), (92600, False), (93000, False)])
@pytest.mark.parametrize("engine", [3, 10])
def test_native_fp8_split_version_boundary(monkeypatch, version, blocked, engine):
    monkeypatch.setattr(cudnn, "backend_version", lambda: version)
    reason = guard.backend_plan_guard(None, _facts(), engine, {cudnn.knob_type.STREAM_K: 1})
    assert (reason is not None) == blocked
    if reason:
        assert "STREAM_K=1" in reason and "STREAM_K=0" in reason
    assert guard.backend_plan_guard(None, _facts(), engine, {cudnn.knob_type.STREAM_K: 0}) is None


@pytest.mark.parametrize(
    "changes",
    [
        {"is_backward": True},
        {"is_fp8": False},
        {"invalid": "invalid graph"},
        {"device_cc": (9, 0)},
        {"device_cc": (10, 7)},
        {"device_cc": (12, 0)},
        {"s_q": 1},
        {"dtype_o": cudnn.data_type.FLOAT},
    ],
)
def test_native_fp8_split_other_domains(monkeypatch, changes):
    monkeypatch.setattr(cudnn, "backend_version", lambda: 92400)
    facts = _facts(**changes)
    for engine in (3, 10, BACKEND_HEURISTIC_ENGINE_ID):
        assert guard.backend_plan_guard(None, facts, engine, {cudnn.knob_type.STREAM_K: 1}) is None


@pytest.mark.parametrize("cc", [(10, 0), (10, 3)])
@pytest.mark.parametrize("out_dtype", [cudnn.data_type.HALF, cudnn.data_type.BFLOAT16, cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E5M2])
def test_native_fp8_split_preserves_unrelated_engines(monkeypatch, cc, out_dtype):
    monkeypatch.setattr(cudnn, "backend_version", lambda: 92400)
    facts = _facts(device_cc=cc, dtype_o=out_dtype)
    assert guard.backend_plan_guard(None, facts, 8, {cudnn.knob_type.STREAM_K: 1}) is None
    assert guard.backend_plan_guard(None, facts, 20527, {cudnn.knob_type.STREAM_K: 1}) is None
    assert guard.backend_plan_guard(None, facts, BACKEND_HEURISTIC_ENGINE_ID, None) is not None
    assert guard.backend_plan_guard(None, None, 3, {}) is None


class _Backend:
    """Configuration-only stand-in; no numerical or GPU behavior."""

    def __init__(self, records=()):
        self.records = list(records)

    def get_execution_plan_count(self):
        return len(self.records)

    def get_engine_and_knobs_at_index(self, index):
        return self.records[index]

    def get_engine_count(self):
        return 12

    def create_execution_plan(self, engine, knobs):
        self.records.append((engine, {cudnn.knob_type.STREAM_K: 1, **knobs}))


def _fake_planning(monkeypatch, records):
    graph = cudnn.pygraph()
    graph._lowered_graph = _Backend(records)
    monkeypatch.setattr(graph, "_lower_backend_graph", lambda: None)
    monkeypatch.setattr(graph, "_create_backend_plans", lambda: None)
    monkeypatch.setattr(graph, "_forward_note_filters", lambda: None)
    monkeypatch.setattr(graph, "_backend_plan_guard", lambda: lambda engine, knobs: guard.backend_plan_guard(graph, _facts(), engine, knobs))
    monkeypatch.setattr(cudnn, "backend_version", lambda: 92400)
    return graph


@pytest.mark.parametrize("mode", [cudnn.heur_mode.A, cudnn.heur_mode.OPENSOURCE])
def test_plan_filter_keeps_same_engine_sibling_and_backend_order(monkeypatch, mode):
    records = [(3, {cudnn.knob_type.STREAM_K: 1}), (3, {cudnn.knob_type.STREAM_K: 0}), (8, {})]
    graph = _fake_planning(monkeypatch, records)
    graph._backend_heuristics = [mode]
    entries = graph.backend_plan_entries()
    assert [(entry.engine_id, entry.cpp_index) for entry in entries] == [(3, 1), (8, 2)]
    assert all(entry.engine_id != BACKEND_HEURISTIC_ENGINE_ID for entry in entries)


@pytest.mark.parametrize("records", [[], [(3, {cudnn.knob_type.STREAM_K: 1})]])
def test_opaque_delegate_does_not_restore_excluded_plan(monkeypatch, records):
    graph = _fake_planning(monkeypatch, records)
    graph._backend_heuristics = [cudnn.heur_mode.OPENSOURCE]
    assert graph.backend_plan_entries() == []
    assert "opaque backend fallback" in str(graph._backend_declined)


@pytest.mark.parametrize("knobs", [{}, {cudnn.knob_type.TILE_M: 3}, {cudnn.knob_type.STREAM_K: 1}])
def test_explicit_and_resolved_default_knobs_are_guarded(monkeypatch, knobs):
    graph = _fake_planning(monkeypatch, [])
    graph._is_validated = True
    with pytest.raises(cudnn.cudnnGraphNotSupportedError, match="STREAM_K=1"):
        graph.create_execution_plan(3, knobs)
    assert graph.plans == []
    # A failed pin may leave a rejected concrete C++ config. The next accepted
    # record must address its own index and must not revive that config.
    expected_index = graph._lowered_graph.get_execution_plan_count()
    graph.create_execution_plan(3, {cudnn.knob_type.STREAM_K: 0})
    assert graph.plans[-1].cpp_index == expected_index
    assert graph.plans[-1].knobs == {cudnn.knob_type.STREAM_K: 0}


@pytest.mark.parametrize(
    "cfg",
    [
        PlanConfig(3, {cudnn.knob_type.STREAM_K: 1}),
        PlanConfig(3, {cudnn.knob_type.STREAM_K: 0}, cpp_index=0),
        PlanConfig(BACKEND_HEURISTIC_ENGINE_ID),
    ],
)
def test_custom_plan_cannot_restore_a_guarded_configuration(monkeypatch, cfg):
    graph = _fake_planning(monkeypatch, [(3, {cudnn.knob_type.STREAM_K: 1})])
    graph._plans = [cfg]
    graph._planning_done = True
    monkeypatch.setattr(graph, "_engine_for", lambda cfg: None)
    with pytest.raises(cudnn.cudnnGraphNotSupportedError, match="STREAM_K=1"):
        graph._build_plan_at(0)


def test_explicit_plan_without_family_guard_needs_no_config_query(monkeypatch):
    graph = _fake_planning(monkeypatch, [])
    monkeypatch.setattr(graph, "_backend_plan_guard", lambda: None)

    def forbidden(*args):
        pytest.fail("a family without a plan guard must not query resolved knobs")

    monkeypatch.setattr(graph._lowered_graph, "get_engine_and_knobs_at_index", forbidden)
    assert graph._append_backend_plan(8, {}) == 0


def _graph(*, raw=False, d=128, d_v=None, sq=128, skv=8192, input_dtype=None, output_dtype=None):
    d_v = d if d_v is None else d_v
    input_dtype = input_dtype or cudnn.data_type.FP8_E4M3
    output_dtype = output_dtype or cudnn.data_type.BFLOAT16
    graph = (cudnn._pybind_module.backend_graph if raw else cudnn.pygraph)(
        io_data_type=input_dtype, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT
    )
    make = graph._make_tensor if raw else graph.tensor
    tensors = {}
    for uid, name, heads, seq in ((1, "q", 4, sq), (2, "k", 2, skv), (3, "v", 2, skv)):
        width = d_v if name == "v" else d
        tensors[name] = make(dim=[1, heads, seq, width], stride=[seq * heads * width, width, heads * width, 1], data_type=input_dtype, name=name, uid=uid)
    scales = {}
    for uid, name in enumerate(("descale_q", "descale_k", "descale_v", "scale_o", "scale_s", "descale_s"), 4):
        tensors[name] = scales[name] = make(dim=[1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.FLOAT, name=name, uid=uid)
    o, stats, amax_s, amax_o = graph.sdpa_fp8(q=tensors["q"], k=tensors["k"], v=tensors["v"], attn_scale=d**-0.5, generate_stats=True, **scales)
    amax_s.set_dim([1, 1, 1, 1]).set_stride([1, 1, 1, 1]).set_data_type(cudnn.data_type.FLOAT)
    for uid, name, tensor, dims, strides, dtype in (
        (20, "o", o, [1, 4, sq, d_v], [sq * 4 * d_v, d_v, 4 * d_v, 1], output_dtype),
        (21, "stats", stats, [1, 4, sq, 1], [4 * sq, sq, 1, 1], cudnn.data_type.FLOAT),
        (22, "amax_o", amax_o, [1, 1, 1, 1], [1, 1, 1, 1], cudnn.data_type.FLOAT),
    ):
        tensor.set_output(True).set_dim(dims).set_stride(strides).set_data_type(dtype).set_uid(uid)
        tensors[name] = tensor
    return graph, tensors


def test_native_fp8_plan_guard_is_registered_and_reads_graph_facts(monkeypatch):
    monkeypatch.setattr(cudnn, "backend_version", lambda: 92400)
    monkeypatch.setattr(graph_analyzer, "_device_cc", lambda: (10, 0))
    graph, _ = _graph()
    family = manifest.family_for(graph)
    assert manifest.resolve_backend_plan_guard(family) is guard.backend_plan_guard
    assert graph._backend_plan_guard()(3, {cudnn.knob_type.STREAM_K: 1}) is not None
    assert graph._backend_plan_guard()(3, {cudnn.knob_type.STREAM_K: 0}) is None


def _native_device():
    import torch

    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("native SM100/SM103 required")
    return torch


def _buffers(torch, tensors):
    dtype_map = {
        cudnn.data_type.FP8_E4M3: torch.float8_e4m3fn,
        cudnn.data_type.FP8_E5M2: torch.float8_e5m2,
        cudnn.data_type.HALF: torch.float16,
        cudnn.data_type.BFLOAT16: torch.bfloat16,
        cudnn.data_type.FLOAT: torch.float32,
    }
    buffers = {
        name: torch.empty_strided(tuple(t.get_dim()), tuple(t.get_stride()), dtype=dtype_map[t.get_data_type()], device="cuda") for name, t in tensors.items()
    }
    for name, buf in buffers.items():
        buf.fill_(0 if name in ("q", "k") else 1)
    buffers["scale_s"].fill_(256)
    buffers["descale_s"].fill_(1 / 256)
    return buffers


def _records(graph):
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A])
    return [graph.get_engine_and_knobs_at_index(i) for i in range(graph.get_execution_plan_count())]


def _run(torch, graph, tensors, record, *, raw=False):
    # Old raw C++ FP8 lowering transposes K metadata in place. Preserve the
    # logical length for the independent Stats oracle before lowering.
    skv = tensors["k"].get_dim()[2]
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plan(*record)
    index = graph.get_execution_plan_count() - 1
    graph.build_plan_at_index(index)
    workspace = torch.empty(max(1, graph.get_workspace_size_plan_at_index(index)), dtype=torch.uint8, device="cuda")
    workspace.fill_(0xA5)
    buffers = _buffers(torch, tensors)
    handle = cudnn.create_handle()
    captured = None
    try:
        pack = {t.get_uid(): buffers[name].data_ptr() for name, t in tensors.items()} if raw else {t: buffers[name] for name, t in tensors.items()}

        def execute():
            cudnn.set_stream(handle, torch.cuda.current_stream().cuda_stream)
            if raw:
                graph._execute_plan_at_index(pack, workspace.data_ptr(), index, handle.backend_handle)
            else:
                graph.execute_plan_at_index(pack, workspace, index=index, handle=handle)

        execute()
        torch.cuda.synchronize()
        eager = {name: buffers[name].clone() for name in ("o", "stats", "amax_o")}
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured):
            execute()
        for name in eager:
            buffers[name].fill_(float("nan"))
        captured.replay()
        torch.cuda.synchronize()
        for name in eager:
            assert torch.equal(buffers[name].view(torch.uint8), eager[name].view(torch.uint8)), name
        # Exact flat-row probabilities at S256/K8192 are representable in both
        # FP8 formats. Stats alone would pass the known wrong-output kernel.
        errors = {"o": (buffers["o"].float() - 1).abs().max().item(), "amax_o": abs(buffers["amax_o"].item() - 1)}
        torch.testing.assert_close(buffers["stats"], torch.full_like(buffers["stats"], math.log(skv)), atol=1e-4, rtol=0)
        buffers["v"].fill_(2)
        for name in eager:
            buffers[name].fill_(float("nan"))
        captured.replay()
        torch.cuda.synchronize()
        errors["changed_o"] = (buffers["o"].float() - 2).abs().max().item()
        errors["changed_amax_o"] = abs(buffers["amax_o"].item() - 2)
        assert torch.equal(buffers["stats"].view(torch.uint8), eager["stats"].view(torch.uint8))
        return errors
    finally:
        if captured is not None:
            captured.reset()
        cudnn.destroy_handle(handle)


@pytest.mark.parametrize("d,d_v", [(64, 64), (80, 80), (128, 128), (192, 128), (256, 256)], ids=["D64", "D80", "D128", "D192V128", "D256"])
@pytest.mark.parametrize("skv", [128, 8192])
@pytest.mark.parametrize("input_dtype", [cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E5M2])
@pytest.mark.parametrize("output_dtype", [cudnn.data_type.HALF, cudnn.data_type.BFLOAT16, cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E5M2])
def test_native_fp8_split_guard_numerical_detector(monkeypatch, record_property, d, d_v, skv, input_dtype, output_dtype):
    torch = _native_device()
    if d == d_v == 256 and cudnn.backend_version() < 92600:
        pytest.skip("the native FP8 graph validator admits D256 only on cuDNN 9.26+")
    monkeypatch.delenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", raising=False)
    params = dict(d=d, d_v=d_v, skv=skv, input_dtype=input_dtype, output_dtype=output_dtype)
    raw_graph, _ = _graph(raw=True, **params)
    records = [r for r in _records(raw_graph) if r[0] in (3, 10)]
    record_property("backend_version", cudnn.backend_version())
    record_property("native_records", json.dumps([(engine, {int(k): v for k, v in knobs.items()}) for engine, knobs in records]))
    split = next((r for r in records if r[1].get(cudnn.knob_type.STREAM_K) == 1), None)
    unsplit = next((r for r in records if r[1].get(cudnn.knob_type.STREAM_K) == 0), None)
    if unsplit is None:
        pytest.skip(f"backend does not offer native STREAM_K=0 for D{d}/V{d_v}")
    # This literal pins the verified release boundary independently of the
    # implementation, so the numerical detector also runs on the unfixed FE.
    affected = cudnn.backend_version() < 92600
    if split is not None:
        # Raw C++ is an intentional diagnostic bypass: it re-measures whether
        # the installed backend needs this guard without spoofing its version.
        graph, tensors = _graph(raw=True, **params)
        errors = _run(torch, graph, tensors, split, raw=True)
        record_property("raw_split_errors", json.dumps(errors))
        if affected:
            assert errors["o"] > 1, f"native split no longer fails: {cudnn.backend_version()}, {split}, {errors}"
        else:
            assert max(errors.values()) < 1e-3, errors
    else:
        record_property("split_coverage", "backend offered no split configuration; unsplit native control only")
    graph, tensors = _graph(**params)
    offered = _records(graph)
    assert unsplit in offered
    if affected:
        assert not any(engine in (3, 10) and knobs.get(cudnn.knob_type.STREAM_K) == 1 for engine, knobs in offered)
        if split is not None:
            with pytest.raises(cudnn.cudnnGraphNotSupportedError, match="STREAM_K=1"):
                graph.create_execution_plan(*split)
    elif split is not None:
        assert split in offered
    graph, tensors = _graph(**params)
    errors = _run(torch, graph, tensors, unsplit)
    record_property("retained_unsplit_errors", json.dumps(errors))
    assert max(errors.values()) < 1e-3, errors
