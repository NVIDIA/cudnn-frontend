# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RL numerical-note discovery, selection and replay without kernel compilation."""

from types import SimpleNamespace

import pytest

import cudnn
from cudnn._pygraph import pygraph
from cudnn.engines import manifest
from cudnn.engines.engine_ids import BACKEND_HEURISTIC_ENGINE_ID

pytestmark = pytest.mark.L0


def test_rl_gemm_manifest_slot():
    family = next(row for row in manifest.MANIFEST if row.name == "frost_gemm")
    assert family.slots["rl_gemm"].slot == 1


@pytest.mark.parametrize("late", [False, True])
def test_rl_note_unlocks_gemm_and_preserves_the_contract_on_replay(monkeypatch, late):
    """Exercise discovery and serialization on a CPU host. Only the device probe
    and kernel compilation are substituted; selection uses the shipped manifest."""
    from cudnn.engines.base import CompiledPlan, PlanConfig
    from cudnn.gemm.rl import engine

    monkeypatch.delenv(manifest._ENABLE_ENV, raising=False)
    monkeypatch.setattr(engine.devices, "current_device", lambda: 0)
    monkeypatch.setattr(engine.devices, "compute_capability", lambda device: (10, 0))
    monkeypatch.setattr(engine, "cutedsl_state", lambda: (True, None))
    monkeypatch.setattr(engine, "cutedsl_requirement_error", lambda *args: None)
    monkeypatch.setattr(engine.RlGemmEngine, "build_plan", lambda self, graph, plan, ctx: CompiledPlan())
    monkeypatch.setattr(pygraph, "_resolve_stream", lambda self, handle, device=None: 0)

    graph = pygraph(io_data_type=cudnn.data_type.BFLOAT16, compute_data_type=cudnn.data_type.FLOAT)
    graph.matmul(graph.tensor([3, 32]), graph.tensor([32, 8])).set_output(True)
    if late:
        # An opaque backend candidate has no C++ index on which a classic note
        # filter could act. A late request must bar it in the Python walk too.
        graph._lowered_graph = SimpleNamespace()
        graph._cpp_bog_done = graph._cpp_plans_created = True
        graph._backend_entries = [PlanConfig(BACKEND_HEURISTIC_ENGINE_ID)]
        graph.create_execution_plans()
        assert graph.selected_engine is None
    graph.select_numeric_notes([cudnn.numerical_note.RL])
    if not late:
        graph.create_execution_plans()
    graph.check_support()
    graph.build_plans()
    assert graph.selected_engine.name == "rl_gemm"
    if late:
        with pytest.raises(ValueError, match="excluded"):
            graph.build_plan_at_index(0)
    with pytest.raises(cudnn.cudnnGraphNotSupportedError, match="RL"):
        graph.create_execution_plan(0)

    restored = pygraph()
    restored.deserialize(graph.serialize(as_bytes=True, include_kernels=False))
    assert restored.selected_engine.name == "rl_gemm"
    with pytest.raises(cudnn.cudnnGraphNotSupportedError, match="RL"):
        restored.create_execution_plan(0)


def test_rl_note_declines_an_unsupported_matmul(monkeypatch):
    monkeypatch.delenv(manifest._ENABLE_ENV, raising=False)
    graph = pygraph(io_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    graph.matmul(graph.tensor([3, 32]), graph.tensor([32, 8])).set_output(True)
    graph.select_numeric_notes([cudnn.numerical_note.RL])
    with pytest.raises(cudnn.cudnnGraphNotSupportedError, match="rl_gemm.*BF16"):
        graph.create_execution_plans()


def test_rl_note_rejects_native_graph():
    graph = cudnn._pybind_module.backend_graph(handle=0)
    with pytest.raises(ValueError, match="RL"):
        graph.select_numeric_notes([cudnn.numerical_note.RL])


def test_rl_note_distinguishes_the_graph_cache_key():
    graph = pygraph()
    graph._lowered_graph = SimpleNamespace(key=lambda: 721075, get_execution_plan_count=lambda: 0)
    ordinary = graph.key()
    graph.select_numeric_notes([cudnn.numerical_note.RL])
    assert 0 <= graph.key() < 2**64 and graph.key() != ordinary
