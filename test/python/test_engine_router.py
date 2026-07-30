# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the backend Router + BaseEngine contract.

These run without a GPU or cuDNN: they exercise pygraph -> Router -> ranked
plan list -> engine-id dispatch using the pure-PyTorch ReferenceMatmulEngine.
This is the CI-safe proof that the unification contract works end to end.
"""

import pytest

torch = pytest.importorskip("torch")

from cudnn._pygraph import pygraph
from cudnn.engines import BaseEngine, Router, ReferenceMatmulEngine, PYTHON_ENGINE_ID_BASE, is_python_engine
from cudnn.engines.engine_ids import BACKEND_HEURISTIC_ENGINE_ID

pytestmark = pytest.mark.L0


def test_router_plan_list_includes_supporting_engines_then_cudnn():
    """Plan list = supporting python engines (by id) + a trailing backend entry."""

    class Declines(BaseEngine):
        name = "declines"
        engine_id = PYTHON_ENGINE_ID_BASE + 50

        def check_support(self, graph):
            raise NotImplementedError("nope")

        def execute(self, graph, tensor_data, ctx=None):
            raise AssertionError("should not run")

    class Accepts(BaseEngine):
        name = "accepts"
        engine_id = PYTHON_ENGINE_ID_BASE + 10

        def execute(self, graph, tensor_data, ctx=None):
            pass

    g = pygraph()
    a = g.tensor(dim=[4, 8], name="A")
    b = g.tensor(dim=[8, 4], name="B")
    g.matmul(a, b, name="mm")
    g.register_backend(Accepts()).register_backend(Declines())

    plans = Router().plan(g, g.backends)
    ids = [p.engine_id for p in plans]
    # Only the supporting python engine is included, then the backend entry last.
    assert ids == [PYTHON_ENGINE_ID_BASE + 10, BACKEND_HEURISTIC_ENGINE_ID]
    assert is_python_engine(ids[0]) and not is_python_engine(ids[-1])


def test_reference_matmul_execute_cpu():
    """ReferenceMatmulEngine runs a matmul on CPU and writes the output buffer."""
    g = pygraph()
    g.register_backend(ReferenceMatmulEngine())

    a = torch.randn(2, 3, 4)
    b = torch.randn(2, 4, 5)
    C = g.matmul(a, b, name="mm")
    c = torch.empty(2, 3, 5)

    g.execute({C: c})

    assert g.selected_engine is not None
    assert g.selected_engine.name == "reference_matmul"
    torch.testing.assert_close(c, torch.matmul(a, b))


def test_reference_matmul_bias_relu_fusion_cpu():
    """A small matmul + add + relu chain routes to the reference and matches."""
    g = pygraph()
    g.register_backend(ReferenceMatmulEngine())

    a = torch.randn(3, 4)
    b = torch.randn(4, 5)
    bias = torch.randn(3, 5)
    mm = g.matmul(a, b, name="mm")
    biased = g.add(mm, g._ensure_tensor(bias, name="bias"), name="bias_add")
    out = g.relu(biased, name="act")
    c = torch.empty(3, 5)

    g.execute({out: c})

    ref = torch.relu(torch.matmul(a, b) + bias)
    torch.testing.assert_close(c, ref)


def test_select_plan_survives_build_and_execute():
    """Regression (review item 2): select_plan(i) must not be reset by the
    implicit build() inside execute()."""

    class EngA(BaseEngine):
        name = "a"
        engine_id = PYTHON_ENGINE_ID_BASE + 10
        ran = 0

        def execute(self, graph, tensor_data, ctx=None):
            type(self).ran += 1

    class EngB(BaseEngine):
        name = "b"
        engine_id = PYTHON_ENGINE_ID_BASE + 11
        ran = 0

        def execute(self, graph, tensor_data, ctx=None):
            type(self).ran += 1

    g = pygraph()
    g.register_backend(EngA()).register_backend(EngB())
    a = torch.randn(2, 2)
    C = g.matmul(a, torch.randn(2, 2))
    g.create_execution_plans()
    assert g.selected_engine.name == "a"
    g.select_plan(1)  # pin engine B
    assert g.selected_engine.name == "b"
    g.execute({C: torch.empty(2, 2)})  # implicit build() must preserve the pin
    assert EngB.ran == 1 and EngA.ran == 0


def test_register_backend_validation():
    """Regression (review item 6): duplicate/invalid ids rejected at
    registration; registration after planning rejected."""

    class NoId(BaseEngine):
        name = "noid"  # forgets to declare engine_id (base default is None)

        def execute(self, graph, tensor_data, ctx=None):
            pass

    class E1(BaseEngine):
        name = "e1"
        engine_id = PYTHON_ENGINE_ID_BASE + 20

        def execute(self, graph, tensor_data, ctx=None):
            pass

    g = pygraph()
    with pytest.raises(ValueError, match="engine_id"):
        g.register_backend(NoId())
    g.register_backend(E1())
    with pytest.raises(ValueError, match="already registered"):
        g.register_backend(E1())
    a = g.tensor(dim=[2, 2], name="A")
    g.matmul(a, g.tensor(dim=[2, 2], name="B"))
    g.create_execution_plans()
    with pytest.raises(RuntimeError, match="after create_execution_plans"):

        class E2(E1):
            engine_id = PYTHON_ENGINE_ID_BASE + 21

        g.register_backend(E2())


def test_unexpected_engine_exception_propagates():
    """Regression (review item 6): only NotImplementedError /
    cudnnGraphNotSupportedError decline; other exceptions are engine bugs."""

    class Buggy(BaseEngine):
        name = "buggy"
        engine_id = PYTHON_ENGINE_ID_BASE + 30

        def check_support(self, graph):
            raise RuntimeError("driver exploded")

        def execute(self, graph, tensor_data, ctx=None):
            pass

    g = pygraph()
    a = g.tensor(dim=[2, 2], name="A")
    g.matmul(a, g.tensor(dim=[2, 2], name="B"))
    g.register_backend(Buggy())
    with pytest.raises(RuntimeError, match="driver exploded"):
        g.create_execution_plans()


def test_no_backend_plan_list_is_cudnn_only():
    """With no python engine, the plan list is just the backend entry (selected=None)."""
    g = pygraph()
    a = g.tensor(dim=[4, 8], name="A")
    b = g.tensor(dim=[8, 4], name="B")
    g.matmul(a, b, name="mm")
    g.validate()

    from cudnn.engines.router import default_router

    plans = default_router.plan(g, g.backends)
    assert [p.engine_id for p in plans] == [BACKEND_HEURISTIC_ENGINE_ID]
    g._plans = plans
    assert g.selected_engine is None  # backend path


def test_compiled_plan_lifecycle_knobs_and_reuse():
    """Review item 1 acceptance: multiple knob proposals from one engine; the
    selected plan's knobs reach build_plan; compilation runs once per plan and
    the artifact is reused; caller workspace + stream context reach execute."""
    from cudnn.engines import CompiledPlan, ExecutionContext, PlanConfig

    compiled_log = []

    class TunablePlan(CompiledPlan):
        def __init__(self, knobs):
            self.knobs = knobs
            self.executed = []

        def get_workspace_size(self):
            return 4096

        def execute(self, graph, tensor_data, ctx):
            self.executed.append((self.knobs, ctx.workspace))

    class Tunable(BaseEngine):
        name = "tunable"
        engine_id = PYTHON_ENGINE_ID_BASE + 40

        def propose_plans(self, graph):
            return [PlanConfig(self.engine_id, {"tile": 128}), PlanConfig(self.engine_id, {"tile": 256})]

        def build_plan(self, graph, plan, ctx=None):
            compiled_log.append(plan.knobs)
            return TunablePlan(plan.knobs)

    g = pygraph()
    g.register_backend(Tunable())
    C = g.matmul(torch.randn(2, 2), torch.randn(2, 2))
    g.create_execution_plans()
    assert [p.knobs for p in g.plans[:2]] == [{"tile": 128}, {"tile": 256}]

    ws = torch.empty(4096, dtype=torch.uint8)
    out = torch.empty(2, 2)
    g.select_plan(1)  # the tile=256 plan
    assert len(g.plans) == 3  # two knob proposals + the backend delegating entry
    g.build_plans()
    assert compiled_log == [{"tile": 256}]  # compiled once, correct knobs
    assert g.get_workspace_size() == 4096  # plan-specific workspace
    g.execute({C: out}, workspace=ws)
    g.execute({C: out}, workspace=ws)
    assert compiled_log == [{"tile": 256}]  # reused, no recompilation
    plan = g._compiled_plans[g._plan_index]
    assert plan.executed[0] == ({"tile": 256}, ws)  # knobs + caller workspace observed

    # same engine instance on a second graph: no state collision
    g2 = pygraph()
    g2.register_backend(Tunable())
    C2 = g2.matmul(torch.randn(2, 2), torch.randn(2, 2))
    g2.execute({C2: torch.empty(2, 2)})
    assert compiled_log == [{"tile": 256}, {"tile": 128}]  # g2 compiled its own plan


def _mk_engine(id_off, knobs=None, log=None):
    from cudnn.engines import CompiledPlan, PlanConfig

    class _Plan(CompiledPlan):
        def __init__(self, k):
            self.knobs = k

        def execute(self, graph, tensor_data, ctx):
            (log if log is not None else []).append(self.knobs)

    class _E(BaseEngine):
        name = f"e{id_off}"
        engine_id = PYTHON_ENGINE_ID_BASE + id_off
        default_knobs = knobs

        def build_plan(self, graph, plan, ctx=None):
            return _Plan(plan.knobs)

        def execute(self, graph, tensor_data, ctx=None):
            pass

    return _E()


def test_planning_is_one_shot():
    """Classic conformance: re-planning was never a supported call pattern (the
    C++ graph appends plans by accident on a second call; nobody re-plans).
    A second create_execution_plans() raises — a stale compiled artifact can
    therefore never execute. Plan differently => build a new graph."""
    log = []
    eng = _mk_engine(60, knobs="old", log=log)
    g = pygraph()
    g.register_backend(eng)
    C = g.matmul(torch.randn(2, 2), torch.randn(2, 2))
    g.create_execution_plans()
    g.build_plans()
    with pytest.raises(RuntimeError, match="one-shot"):
        g.create_execution_plans()
    g.execute({C: torch.empty(2, 2)})
    assert log[-1] == "old"  # the planned artifact, unchanged


def test_mixed_router_ordering_dispatch():
    """Follow-up item 2: dispatch honors arbitrary Router ordering (backend-first,
    interleaved), never a python-prefix assumption."""
    from cudnn.engines import PlanConfig, Router
    from cudnn.engines.engine_ids import BACKEND_HEURISTIC_ENGINE_ID

    ran = []
    ea, eb = _mk_engine(61, "A", ran), _mk_engine(62, "B", ran)

    class Interleaved(Router):
        def plan(self, graph, backends):
            return [
                PlanConfig(ea.engine_id, "A"),
                PlanConfig(BACKEND_HEURISTIC_ENGINE_ID),
                PlanConfig(eb.engine_id, "B"),
            ]

    g = pygraph(router=Interleaved())
    g.register_backend(ea).register_backend(eb)
    C = g.matmul(torch.randn(2, 2), torch.randn(2, 2))
    g.create_execution_plans()
    # slot 0 = python A, slot 1 = cuDNN, slot 2 = python B
    assert g.selected_engine.name == "e61"
    g.select_plan(1)  # the backend delegating entry is selectable in place
    assert g.selected_engine is None  # None == the backend path
    g.select_plan(2)
    assert g.selected_engine.name == "e62"
    g.execute({C: torch.empty(2, 2)})
    assert ran[-1] == "B"
    # the middle routed entry is the backend delegating one, and routed indices
    # are STABLE: python-B stays at index 2 regardless of lowering. (Real
    # execution THROUGH the backend slot of a mixed router is the GPU test
    # test_mixed_router_backend_slot_executes in test_native_backend_lowering.py.)
    assert g.plans[1].engine_id == BACKEND_HEURISTIC_ENGINE_ID
    assert g.selected_engine.name == "e62"


def test_empty_router_output_rejected():
    """A Router returning [] is an error — there is no legal empty planning
    state (it would defeat the one-shot flag and every needs-planning check)."""

    class Empty(Router):
        def plan(self, graph, backends):
            return []

    g = pygraph(router=Empty())
    g.matmul(torch.randn(2, 2), torch.randn(2, 2))
    with pytest.raises(ValueError, match="empty plan list"):
        g.create_execution_plans()
    # the failed call did NOT consume the one-shot: fixing the router by
    # rebuilding the graph is the documented path, but the graph must not be
    # left half-planned either
    assert not g._planning_done


def test_set_router_frozen_after_planning():
    """set_router() after planning raises (it could not affect the already
    planned list; accepting it silently would lie)."""
    g = pygraph()
    g.register_backend(_mk_engine(70))
    g.matmul(torch.randn(2, 2), torch.randn(2, 2))
    g.create_execution_plans()
    with pytest.raises(RuntimeError, match="one-shot"):
        g.set_router(Router())


def test_backend_count_is_a_separate_space():
    """get_execution_plan_count() is the classic backend-count passthrough,
    never the routed-list length; the routed list is graph.plans/select_plan.
    A python-only routed graph has no backend plans and says so."""

    class PythonOnly(Router):
        def plan(self, graph, backends):
            from cudnn.engines import PlanConfig

            return [PlanConfig(backends[0].engine_id)]

    g = pygraph(router=PythonOnly())
    g.register_backend(_mk_engine(71))
    C = g.matmul(torch.randn(2, 2), torch.randn(2, 2))
    g.create_execution_plans()
    assert len(g.plans) == 1
    with pytest.raises(RuntimeError, match="graph.plans"):
        g.get_execution_plan_count()  # no backend entry -> no backend plans
    g.execute({C: torch.empty(2, 2)})  # the routed python plan still runs


def test_constructor_backends_validated_and_proposals_checked():
    """Follow-up item 6: constructor path uses registration validation; foreign
    engine ids in proposals are rejected."""
    from cudnn.engines import PlanConfig

    class NoId(BaseEngine):
        def execute(self, graph, tensor_data, ctx=None):
            pass

    with pytest.raises(ValueError, match="engine_id"):
        pygraph(backends=[NoId()])

    class Impostor(BaseEngine):
        name = "impostor"
        engine_id = PYTHON_ENGINE_ID_BASE + 63

        def propose_plans(self, graph):
            return [PlanConfig(PYTHON_ENGINE_ID_BASE + 99, None)]  # foreign id

        def execute(self, graph, tensor_data, ctx=None):
            pass

    g = pygraph(backends=[Impostor()])
    g.matmul(torch.randn(2, 2), torch.randn(2, 2))
    with pytest.raises(ValueError, match="foreign engine_id"):
        g.create_execution_plans()


def test_python_plan_rejects_dynamic_workspace_overrides():
    g = pygraph()
    g.register_backend(ReferenceMatmulEngine())
    C = g.matmul(torch.randn(2, 3), torch.randn(3, 2))
    g.build()
    with pytest.raises(NotImplementedError, match="overrides"):
        g.get_workspace_size(1234)
    g.execute({C: torch.empty(2, 2)})  # normal path unaffected


def test_failed_stream_query_on_supplied_handle_raises(monkeypatch):
    """Follow-up item 3: a supplied handle whose stream cannot be queried is a
    correctness error — never a silent stream-0 fallback."""
    import cudnn as _cudnn

    g = pygraph()
    g.register_backend(ReferenceMatmulEngine())
    C = g.matmul(torch.randn(2, 3), torch.randn(3, 2))
    g.build()

    def boom(handle):
        raise RuntimeError("stream query failed")

    monkeypatch.setattr(_cudnn, "get_stream", boom)
    with pytest.raises(RuntimeError, match="stream query failed"):
        g.execute({C: torch.empty(2, 2)}, handle=42)
