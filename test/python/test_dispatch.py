# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the unified dispatch: ranking + the BaseEngine contract.

These run without a GPU: they exercise pygraph -> heuristics -> ONE ranked plan
list -> engine-id dispatch, using throwaway engines defined in this file. This
is the CI-safe proof that the unification contract works end to end.

The stand-in engines here do NO arithmetic. What a dispatch test can prove is
that the walk reached the engine, handed it the graph and the caller's buffers,
and let it write the output; checking the result against ``torch.matmul``
proves torch, not dispatch, and puts a torch reference implementation of matmul
inside a dispatch test.

There is ONE plan list: the python engines that claim the graph (injected here
through the same ``engines/manifest.py`` production uses) and the backend's own
ranked entries, ordered by ``engines.heuristics.rank``. Which of those
participate depends on the machine, so every assertion below addresses a plan
by NAME or engine id and pins it with ``select_plan()`` — never by a hard-coded
absolute index.
"""

import pytest

torch = pytest.importorskip("torch")

from cudnn._pygraph import pygraph
from cudnn.engines import BaseEngine, is_backend_engine, is_python_engine
from cudnn.engines.engine_ids import FAMILY_BLOCK, PYTHON_ENGINE_ID_BASE
from cudnn.engines.base import resolve_node_buffers
from cudnn.engines.engine_ids import BACKEND_HEURISTIC_ENGINE_ID
from cudnn.graph_types import NodeType

pytestmark = pytest.mark.L0

# A family block no shipped family uses, for the fakes below. Test engines are
# offered the same way real ones are -- through the manifest -- because that is
# now the only way a python engine exists.
_FAKE = PYTHON_ENGINE_ID_BASE + 9_000


# ---------------------------------------------------------------------------
# The stand-in engine: claims a matmul (+ a few pointwise) graph and records
# what dispatch handed it. No arithmetic -- see the module docstring.
# ---------------------------------------------------------------------------
MARK = 42.0  # a value nothing but StubEngine writes


class StubEngine(BaseEngine):
    """Accepts MATMUL plus a few POINTWISE modes, wherever the buffers live.

    ``execute`` marks every output buffer it was given and records the ports it
    resolved, so a test can assert that the walk reached this engine and that
    ``resolve_node_buffers`` handed it the caller's storage — the two things
    dispatch is responsible for. No GPU, no JIT.
    """

    name = "stub"
    engine_id = _FAKE + 0

    _POINTWISE = frozenset(("relu", "add", "bias", "mul"))

    def __init__(self):
        self.seen = []  # one (node name, {port: buffer or None}) per node executed

    def check_support(self, graph):
        for node in graph.nodes:
            if node.node_type == NodeType.MATMUL:
                continue
            if node.node_type == NodeType.POINTWISE and node.params.get("mode") in self._POINTWISE:
                if all(k == "mode" for k in node.params):
                    continue  # scalar attributes (clips, slopes) not modelled
            raise NotImplementedError(f"stub: unsupported node {node.node_type.name}")

    def execute(self, graph, uid_to_data, ctx=None):
        buffers = resolve_node_buffers(graph, uid_to_data)
        for node in graph.nodes:
            nb = buffers[node]
            self.seen.append((node.name, {port: nb.inputs.get(port) for port in node.inputs}))
            for port in node.outputs:
                dst = nb.outputs.get(port)
                if dst is not None:  # a virtual intermediate has no storage
                    dst.fill_(MARK)


def _marked(t):
    """Whether ``t`` holds what StubEngine.execute writes, and only that."""
    return bool(torch.equal(t, torch.full_like(t, MARK)))


def _plan_names(g):
    return [g.get_plan_name_at_index(i) for i in range(len(g.plans))]


def _python_indices(g):
    return [i for i, p in enumerate(g.plans) if is_python_engine(p.engine_id)]


def _index_of(g, name):
    """Index of the ranked entry named ``name`` (``name[knobs]`` also matches)."""
    for i, n in enumerate(_plan_names(g)):
        if n == name or n.startswith(f"{name}["):
            return i
    raise AssertionError(f"no plan named {name!r} in {_plan_names(g)}")


def _pin(g, name):
    """Plan (if needed) and pin the entry named ``name``.

    The ranked list also carries the backend's own plans wherever cuDNN accepts
    the graph, and those rank first today — a CPU test must say which entry it
    means instead of relying on position.
    """
    if not g._planning_done:
        g.create_execution_plans()
    g.select_plan(_index_of(g, name))
    return g


def test_the_ranked_list_carries_claiming_engines_and_the_backend(monkeypatch):
    """The ranked list carries the python engines that CLAIM the graph, merged
    with the backend's own entries; a declining engine is absent."""

    class Declines(BaseEngine):
        name = "declines"
        engine_id = _FAKE + 50

        def check_support(self, graph):
            raise NotImplementedError("nope")

        def execute(self, graph, tensor_data, ctx=None):
            raise AssertionError("should not run")

    class Accepts(BaseEngine):
        name = "accepts"
        engine_id = _FAKE + 10

        def execute(self, graph, tensor_data, ctx=None):
            pass

    g = pygraph()
    a = g.tensor(dim=[4, 8], name="A")
    b = g.tensor(dim=[8, 4], name="B")
    g.matmul(a, b, name="mm")
    _offer(monkeypatch, Accepts(), Declines())

    from cudnn.engines import heuristics

    plans = heuristics.rank(g, g._candidate_engines(), g.backend_plan_entries())
    ids = [p.engine_id for p in plans]
    assert _FAKE + 10 in ids  # the supporting python engine
    assert _FAKE + 50 not in ids  # the declining one never joins
    # one flat id space: every entry is either a python or a backend engine, and
    # today's placeholder ranking puts the backend's entries first.
    assert all(is_python_engine(i) or is_backend_engine(i) for i in ids)
    assert ids[-1] == _FAKE + 10


def test_a_pinned_python_plan_runs_and_writes_the_callers_buffer(monkeypatch):
    """The end of the walk: pin -> build -> execute reaches the python engine,
    which writes the tensor the caller passed in the variant pack."""
    g = pygraph()
    eng = StubEngine()
    _offer(monkeypatch, eng)

    C = g.matmul(torch.randn(2, 3, 4), torch.randn(2, 4, 5), name="mm")
    c = torch.zeros(2, 3, 5)

    _pin(g, "stub")
    g.execute({C: c})

    assert g.selected_engine is not None
    assert g.selected_engine.name == "stub"
    assert _marked(c)


def test_every_node_reaches_the_engine_with_its_buffers_resolved(monkeypatch):
    """A fused matmul + bias + relu graph arrives WHOLE: every node in build
    order, each input port resolved to the caller's storage, and the virtual
    intermediates left without any — that last one is the distinction an engine
    needs to allocate its own scratch."""
    g = pygraph()
    eng = StubEngine()
    _offer(monkeypatch, eng)

    a, b, bias = torch.randn(3, 4), torch.randn(4, 5), torch.randn(3, 5)
    mm = g.matmul(a, b, name="mm")
    biased = g.add(mm, g._ensure_tensor(bias, name="bias"), name="bias_add")
    out = g.relu(biased, name="act")
    c = torch.zeros(3, 5)

    _pin(g, "stub")
    g.execute({out: c})

    assert [name for name, _ in eng.seen] == ["mm", "bias_add", "act"]
    ports = dict(eng.seen)
    # detached views of the caller's tensors, so compare storage, not identity
    assert ports["mm"]["A"].data_ptr() == a.data_ptr()
    assert ports["mm"]["B"].data_ptr() == b.data_ptr()
    assert ports["bias_add"]["IN_1"].data_ptr() == bias.data_ptr()
    assert ports["bias_add"]["IN_0"] is None, "the matmul output is virtual — no storage of its own"
    assert _marked(c)


def test_select_plan_survives_build_and_execute(monkeypatch):
    """Regression (review item 2): select_plan(i) must not be reset by the
    implicit build() inside execute()."""

    class EngA(BaseEngine):
        name = "a"
        engine_id = _FAKE + 10
        ran = 0

        def execute(self, graph, tensor_data, ctx=None):
            type(self).ran += 1

    class EngB(BaseEngine):
        name = "b"
        engine_id = _FAKE + 11
        ran = 0

        def execute(self, graph, tensor_data, ctx=None):
            type(self).ran += 1

    g = pygraph()
    _offer(monkeypatch, EngA(), EngB())
    a = torch.randn(2, 2)
    C = g.matmul(a, torch.randn(2, 2))
    g.create_execution_plans()
    ia, ib = _index_of(g, "a"), _index_of(g, "b")
    assert ia < ib  # candidate order is preserved inside the python block
    g.select_plan(ia)
    assert g.selected_engine.name == "a"
    g.select_plan(ib)  # pin engine B
    assert g.selected_engine.name == "b"
    g.execute({C: torch.empty(2, 2)})  # implicit build() must preserve the pin
    assert EngB.ran == 1 and EngA.ran == 0


def test_unexpected_engine_exception_propagates(monkeypatch):
    """Regression (review item 6): only NotImplementedError /
    cudnnGraphNotSupportedError decline; other exceptions are engine bugs."""

    class Buggy(BaseEngine):
        name = "buggy"
        engine_id = _FAKE + 30

        def check_support(self, graph):
            raise RuntimeError("driver exploded")

        def execute(self, graph, tensor_data, ctx=None):
            pass

    g = pygraph()
    a = g.tensor(dim=[2, 2], name="A")
    g.matmul(a, g.tensor(dim=[2, 2], name="B"))
    _offer(monkeypatch, Buggy())
    with pytest.raises(RuntimeError, match="driver exploded"):
        g.create_execution_plans()


def test_no_python_engine_plan_list_is_backend_only():
    """With no python engine claiming it, every ranked entry is a backend one
    and the selected plan reports no python engine."""
    g = pygraph()
    a = g.tensor(dim=[4, 8], name="A")
    b = g.tensor(dim=[8, 4], name="B")
    g.matmul(a, b, name="mm")

    from cudnn.engines import heuristics

    plans = heuristics.rank(g, [], g.backend_plan_entries())
    assert all(is_backend_engine(p.engine_id) for p in plans)
    g._plans = plans
    assert g.selected_engine is None  # backend path


def test_compiled_plan_lifecycle_knobs_and_reuse(monkeypatch):
    """Review item 1 acceptance: several knob candidates for one engine; the
    selected plan's knobs reach build_plan; compilation runs once per plan and
    the artifact is reused; caller workspace + stream context reach execute."""
    from cudnn.engines import CompiledPlan, PlanConfig

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
        engine_id = _FAKE + 40

        def build_plan(self, graph, plan, ctx=None):
            compiled_log.append(plan.knobs)
            return TunablePlan(plan.knobs)

    tunable = Tunable()
    _ranking(
        monkeypatch,
        lambda graph, engines, backend_plans, modes=None: [PlanConfig(tunable.engine_id, {"tile": 128}), PlanConfig(tunable.engine_id, {"tile": 256})],
    )

    g = pygraph()
    _offer(monkeypatch, tunable)
    C = g.matmul(torch.randn(2, 2), torch.randn(2, 2))
    g.create_execution_plans()
    tuned = _python_indices(g)
    assert [g.plans[i].knobs for i in tuned] == [{"tile": 128}, {"tile": 256}]

    ws = torch.empty(4096, dtype=torch.uint8)
    out = torch.empty(2, 2)
    g.select_plan(tuned[1])  # the tile=256 plan
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
    _offer(monkeypatch, tunable)
    C2 = g2.matmul(torch.randn(2, 2), torch.randn(2, 2))
    g2.create_execution_plans()
    g2.select_plan(_python_indices(g2)[0])
    g2.execute({C2: torch.empty(2, 2)})
    assert compiled_log == [{"tile": 256}, {"tile": 128}]  # g2 compiled its own plan


def _ranking(monkeypatch, fn):
    """Replace the ranking policy for one graph's planning.

    What a Router subclass used to do. Ranking has one home now, so a test
    that wants a specific order says so here rather than by subclassing.
    """
    from cudnn.engines import heuristics

    monkeypatch.setattr(heuristics, "rank", fn)


def _offer(monkeypatch, *engines, node_type="MATMUL", analyzer=None, heuristics=None, name="fake_family"):
    """Make ``engines`` the manifest's answer for graphs anchored at ``node_type``.

    What register_backend() used to do, through the path production uses: the
    manifest is the only way a python engine exists, so a test engine is offered
    by putting it in one. The fakes share a block at _FAKE so their ids stay out
    of every shipped family's.
    """
    from cudnn.engines import manifest

    family = manifest.EngineFamily(
        _FAKE,
        name,
        __name__,
        "unused_factory",
        slots={e.name: manifest.EngineSlot(e.engine_id - _FAKE) for e in engines},
        analyzer=analyzer,
        heuristics=heuristics,
    )
    monkeypatch.setattr(manifest, "MANIFEST", (family,))
    monkeypatch.setattr(manifest, "_ANCHOR_NODE_TO_FAMILY", {node_type: name})
    monkeypatch.setattr(manifest, "instantiate", lambda fam, ids: [e for e in engines if e.name in ids])
    return engines[0] if len(engines) == 1 else engines


def _mk_engine(id_off, knobs=None, log=None):
    from cudnn.engines import CompiledPlan

    class _Plan(CompiledPlan):
        def __init__(self, k):
            self.knobs = k

        def execute(self, graph, tensor_data, ctx):
            (log if log is not None else []).append(self.knobs)

    class _E(BaseEngine):
        name = f"e{id_off}"
        engine_id = _FAKE + id_off

        def build_plan(self, graph, plan, ctx=None):
            # The ranking names the knobs now; this fake keeps its own marker
            # for the entries a test did not give any.
            return _Plan(plan.knobs if plan.knobs is not None else knobs)

        def execute(self, graph, tensor_data, ctx=None):
            pass

    return _E()


def test_planning_is_one_shot(monkeypatch):
    """Classic conformance: re-planning was never a supported call pattern (the
    C++ graph appends plans by accident on a second call; nobody re-plans).
    A second create_execution_plans() raises — a stale compiled artifact can
    therefore never execute. Plan differently => build a new graph."""
    log = []
    eng = _mk_engine(60, knobs="old", log=log)
    g = pygraph()
    _offer(monkeypatch, eng)
    C = g.matmul(torch.randn(2, 2), torch.randn(2, 2))
    _pin(g, "e60")
    g.build_plans()
    with pytest.raises(RuntimeError, match="one-shot"):
        g.create_execution_plans()
    g.execute({C: torch.empty(2, 2)})
    assert log[-1] == "old"  # the planned artifact, unchanged


def test_mixed_ranking_dispatch(monkeypatch):
    """Follow-up item 2: dispatch honors arbitrary ranking (a backend entry in
    the MIDDLE), never a python-prefix assumption. Not hypothetical any more --
    the sdpa-fwd heuristics place backend entries between their own.

    The backend's own entries are stubbed so the ordering is the same with or
    without a cuDNN that accepts this toy graph; real execution THROUGH the
    backend slot of a mixed router is the GPU test
    test_mixed_ranking_backend_slot_executes in test_native_backend_lowering.py.
    """
    from cudnn.engines import PlanConfig

    ran = []
    ea, eb = _mk_engine(61, "A", ran), _mk_engine(62, "B", ran)
    stub = PlanConfig(0, {}, cpp_index=0)
    monkeypatch.setattr(pygraph, "backend_plan_entries", lambda self: [stub])

    _ranking(
        monkeypatch, lambda graph, engines, backend_plans, modes=None: [PlanConfig(ea.engine_id, "A")] + list(backend_plans) + [PlanConfig(eb.engine_id, "B")]
    )

    g = pygraph()
    _offer(monkeypatch, ea, eb)
    C = g.matmul(torch.randn(2, 2), torch.randn(2, 2))
    g.create_execution_plans()
    # slot 0 = python A, slot 1 = the backend's entry, slot 2 = python B
    assert g.selected_engine.name == "e61"
    g.select_plan(1)  # the backend entry is selectable in place
    assert g.selected_engine is None  # None == the backend path
    g.select_plan(2)
    assert g.selected_engine.name == "e62"
    g.execute({C: torch.empty(2, 2)})
    assert ran[-1] == "B"
    # the middle entry is the backend's, and routed indices are STABLE: python-B
    # stays at index 2 regardless of lowering.
    assert is_backend_engine(g.plans[1].engine_id)
    assert g.selected_engine.name == "e62"


def test_pinned_plan_that_declines_raises(monkeypatch):
    """A select_plan() pin is STRICT: the walk starts there and a decline raises
    instead of quietly running a different plan. Without a pin the same decline
    only advances the walk."""

    class Declines(BaseEngine):
        name = "declines_at_build"
        engine_id = _FAKE + 80

        def build_plan(self, graph, plan, ctx=None):
            raise NotImplementedError("cannot compile this graph")

        def execute(self, graph, tensor_data, ctx=None):
            raise AssertionError("should never run")

    g = pygraph()
    _offer(monkeypatch, Declines())
    g.matmul(torch.randn(2, 2), torch.randn(2, 2))
    _pin(g, "declines_at_build")
    with pytest.raises(NotImplementedError, match="cannot compile this graph"):
        g.build_plans()


def test_unpinned_plan_that_declines_advances_the_walk(monkeypatch, caplog):
    """The other half of the pin rule: WITHOUT a pin the same decline is logged
    and the walk moves to the next entry, so the graph still builds and no
    exception reaches the user."""
    import logging

    from cudnn.engines import PlanConfig

    class Declines(BaseEngine):
        name = "declines_at_build"
        engine_id = _FAKE + 81

        def build_plan(self, graph, plan, ctx=None):
            raise NotImplementedError("cannot compile this graph")

        def execute(self, graph, tensor_data, ctx=None):
            raise AssertionError("should never run")

    declines, works = Declines(), _mk_engine(82)
    _offer(monkeypatch, declines, works)
    _ranking(monkeypatch, lambda graph, engines, backend_plans, modes=None: [PlanConfig(declines.engine_id), PlanConfig(works.engine_id)])

    g = pygraph()
    g.matmul(torch.randn(2, 2), torch.randn(2, 2))
    g.create_execution_plans()
    assert _plan_names(g) == ["declines_at_build", works.name]
    with caplog.at_level(logging.INFO, logger="cudnn.pygraph"):
        g.build_plans()
    assert any("declined at build time" in r.getMessage() for r in caplog.records)
    assert g.selected_engine is works
    assert _plan_names(g)[g._plan_index] == works.name


def test_empty_ranking_output_rejected(monkeypatch):
    """Ranking that returns [] is an error — there is no legal empty planning
    state (it would defeat the one-shot flag and every needs-planning check)."""
    import cudnn

    _ranking(monkeypatch, lambda graph, engines, backend_plans, modes=None: [])

    g = pygraph()
    g.matmul(torch.randn(2, 2), torch.randn(2, 2))
    with pytest.raises(cudnn.cudnnGraphNotSupportedError, match="no engine"):
        g.create_execution_plans()
    # the failed call did NOT consume the one-shot: fixing the ranking by
    # rebuilding the graph is the documented path, but the graph must not be
    # left half-planned either
    assert not g._planning_done


def test_plan_count_is_the_whole_ranked_list(monkeypatch):
    """get_execution_plan_count() counts ONE list — python engines and backend
    engines alike (``graph.plans``), so a python-only graph reports its python
    plans instead of raising."""

    eng = _mk_engine(71)

    from cudnn.engines import PlanConfig

    _ranking(monkeypatch, lambda graph, engines, backend_plans, modes=None: [PlanConfig(eng.engine_id)])

    g = pygraph()
    _offer(monkeypatch, eng)
    C = g.matmul(torch.randn(2, 2), torch.randn(2, 2))
    g.create_execution_plans()
    assert len(g.plans) == 1
    assert g.get_execution_plan_count() == 1  # the routed list, not a backend count
    assert g.get_plan_name_at_index(0) == "e71"
    g.execute({C: torch.empty(2, 2)})  # the routed python plan still runs


def test_python_plan_accepts_the_classic_workspace_overloads(monkeypatch):
    """``get_workspace_size(handle)`` and the at-index override-shape form are
    what the native dynamic-shape API calls (test_override_shape_frost). A
    compiled python plan's workspace does not depend on either descriptor, so
    the answer is the plan's own size — rejecting the overload took a shipped
    L0 path out."""
    g = pygraph()
    _offer(monkeypatch, StubEngine())
    C = g.matmul(torch.randn(2, 3), torch.randn(3, 2))
    _pin(g, "stub")
    idx = _index_of(g, "stub")
    g.build_plans()
    plain = g.get_workspace_size()
    assert g.get_workspace_size(1234) == plain
    assert g.get_workspace_size_plan_at_index(idx, 1234, [1, 2, 3], [[2, 3]], [[3, 1]]) == plain
    g.execute({C: torch.empty(2, 2)})  # normal path unaffected


def test_failed_stream_query_on_supplied_handle_raises(monkeypatch):
    """Follow-up item 3: a supplied handle whose stream cannot be queried is a
    correctness error — never a silent stream-0 fallback."""
    import cudnn as _cudnn

    g = pygraph()
    _offer(monkeypatch, StubEngine())
    C = g.matmul(torch.randn(2, 3), torch.randn(3, 2))
    _pin(g, "stub")
    g.build_plans()

    def boom(handle):
        raise RuntimeError("stream query failed")

    monkeypatch.setattr(_cudnn, "get_stream", boom)
    with pytest.raises(RuntimeError, match="stream query failed"):
        g.execute({C: torch.empty(2, 2)}, handle=_cudnn.Handle(backend_handle=42))


# ---------------------------------------------------------------------------
# The opt-in gate: a manifest family may be withheld until its engines mature.
# ---------------------------------------------------------------------------


def _family(name):
    from cudnn.engines.manifest import MANIFEST

    return next(f for f in MANIFEST if f.name == name)


def test_note_filters_reach_python_plans(monkeypatch):
    """The four classic note filters used to fall through to C++, so they
    filtered backend plans and silently skipped every python one. A python
    plan's notes are declared by its engine."""
    import cudnn

    class Jitted(StubEngine):
        name = "jitted"
        engine_id = _FAKE + 80
        behavior_notes = (cudnn.behavior_note.RUNTIME_COMPILATION,)

    def _fresh():
        g = pygraph()
        _offer(monkeypatch, Jitted())
        g.matmul(torch.randn(2, 3), torch.randn(3, 2))
        g.create_execution_plans()
        return g, _index_of(g, "jitted")

    g, idx = _fresh()
    assert g.get_behavior_notes_for_plan_at_index(idx) == [cudnn.behavior_note.RUNTIME_COMPILATION]

    g, idx = _fresh()  # deselect: the plan carrying the note is barred
    g.deselect_behavior_notes([cudnn.behavior_note.RUNTIME_COMPILATION])
    assert idx in g._barred_indices()

    g, idx = _fresh()  # select: the plan carrying it is kept
    g.select_behavior_notes([cudnn.behavior_note.RUNTIME_COMPILATION])
    assert idx not in g._barred_indices()

    g, idx = _fresh()  # select a note it does NOT carry: barred
    g.select_behavior_notes([cudnn.behavior_note.CUBLASLT_DEPENDENCY])
    assert idx in g._barred_indices()

    g, idx = _fresh()  # numerical notes travel the same path
    g.deselect_numeric_notes([cudnn.numerical_note.NONDETERMINISTIC])
    assert idx not in g._barred_indices(), "engine declares no numerical notes"


def test_a_barred_note_advances_the_walk(monkeypatch):
    """Barring by note must change which plan RUNS, not just _barred_indices()."""
    import cudnn

    from cudnn.engines.base import PlanConfig

    class Jitted(StubEngine):
        name = "jitted"
        engine_id = _FAKE + 81
        behavior_notes = (cudnn.behavior_note.RUNTIME_COMPILATION,)

    _ranking(monkeypatch, lambda graph, engines, backend_plans, modes=None: [PlanConfig(e.engine_id, None) for e in engines] + [PlanConfig(0, {}, cpp_index=0)])

    def _fresh():
        g = pygraph()
        _offer(monkeypatch, Jitted())
        C = g.matmul(torch.randn(2, 3), torch.randn(3, 2))
        g._lowered_graph = _FakeBackend()
        g._cpp_plans_created = g._cpp_bog_done = True
        g.create_execution_plans()
        return g, C

    g, C = _fresh()
    g.build_plans()
    assert g.selected_engine is not None and g.selected_engine.name == "jitted"

    g, C = _fresh()
    g.deselect_behavior_notes([cudnn.behavior_note.RUNTIME_COMPILATION])
    g.build_plans()
    assert g.selected_engine is None, "the note-barred python plan should have been skipped"


def test_every_at_index_path_applies_the_exclusions(monkeypatch):
    """The ranked list is never filtered, so indices stay stable — which means
    each at-index entry point has to apply the exclusions itself.
    build_plan_at_index() did; execute_plan_at_index() and
    get_workspace_size_plan_at_index() compiled and ran the excluded plan."""
    import cudnn

    class Jitted(StubEngine):
        name = "jitted"
        engine_id = _FAKE + 82
        behavior_notes = (cudnn.behavior_note.RUNTIME_COMPILATION,)

    def _fresh():
        g = pygraph()
        _offer(monkeypatch, Jitted())
        C = g.matmul(torch.randn(2, 3), torch.randn(3, 2))
        g.create_execution_plans()
        i = _index_of(g, "jitted")
        g.deselect_behavior_notes([cudnn.behavior_note.RUNTIME_COMPILATION])
        return g, C, i

    g, C, i = _fresh()
    with pytest.raises(ValueError, match="excluded"):
        g.build_plan_at_index(i)

    g, C, i = _fresh()
    with pytest.raises(ValueError, match="excluded"):
        g.execute_plan_at_index({C: torch.empty(2, 2)}, None, i)

    g, C, i = _fresh()
    with pytest.raises(ValueError, match="excluded"):
        g.get_workspace_size_plan_at_index(i)


def test_a_note_filter_set_before_planning_reaches_the_backend(monkeypatch):
    """The backend's note filters mark indices into engine_configs
    (plans.h::filter_behavior_notes), so one forwarded while that list is empty
    marks nothing and is silently lost — measured: a plan deselected before
    create_execution_plans() still built, the same filter set after was
    refused. It has to be replayed once the plans exist."""
    import cudnn

    g = pygraph()
    _offer(monkeypatch, StubEngine())
    g.matmul(torch.randn(2, 3), torch.randn(3, 2))
    fake = _FakeBackend(planned=False)  # nothing for a filter to mark yet
    g._lowered_graph = fake
    g._cpp_bog_done = True

    g.deselect_behavior_notes([cudnn.behavior_note.RUNTIME_COMPILATION])
    assert fake.calls == [], "nothing to filter yet — the backend has no plans"
    g._create_backend_plans()
    # One create_execution_plans per heuristic mode (A, FALLBACK): that is how
    # the backend's entries get tagged with the mode that produced them.
    assert fake.calls == ["create_execution_plans", "create_execution_plans", "deselect_behavior_notes"], fake.calls


def test_cuda_graph_capture_declines_a_python_plan(monkeypatch):
    """populate/update_cuda_graph record the BACKEND's plan. On a python plan
    they used to reach __getattr__ and report "graph not lowered yet", which
    points at the wrong thing."""
    import cudnn

    g, C = _backend_first(monkeypatch)
    _pin(g, "stub")
    g.build_plans()
    for name in ("populate_cuda_graph", "update_cuda_graph"):
        with pytest.raises(cudnn.cudnnGraphNotSupportedError, match="stub"):
            getattr(g, name)(None, {}, None, None)


def test_key_says_which_op_has_no_backend_lowering():
    """key() is the backend's cache key, so it needs a lowered graph. For a
    python_only op that will never happen — say so instead of raising
    AttributeError about a lowering that is not pending."""
    import cudnn

    g = pygraph()
    T, N, H, K = 8, 1, 2, 4
    g.gdn(
        q=g.tensor([T, H, K], name="q"),
        k=g.tensor([T, H, K], name="k"),
        v=g.tensor([T, H, K], name="v"),
        g=g.tensor([T, H], name="g"),
        beta=g.tensor([T, H], name="beta"),
        cu_seqlens=g.tensor([N + 1], data_type=cudnn.data_type.INT32, name="cu_seqlens"),
        name="gdn",
    )
    with pytest.raises(cudnn.cudnnGraphNotSupportedError, match="GDN"):
        g.key()


def test_opt_in_families_are_withheld_by_default(monkeypatch):
    """An ``opt_in=True`` family is offered only with the env flag set.

    The flag is read live (not cached at import) so a process can flip it, and
    it gates NOTHING but candidacy: a withheld family is simply absent from the
    plan list, never a different ranking or a silent fallback."""
    from cudnn.engines import manifest

    family = _family("frost_gemm")
    assert all(s.opt_in for s in family.slots.values()), "frost_gemm is expected to still be maturing"
    monkeypatch.delenv(manifest._ENABLE_ENV, raising=False)
    assert family.offered_ids() == {}

    monkeypatch.setenv(manifest._ENABLE_ENV, "1")
    assert family.offered_ids() == {"frost_gemm": family.engine_id}


def test_sole_implementation_families_are_never_gated(monkeypatch):
    """An engine that is the ONLY implementation of its op must not be gated —
    the backend has no lowering for GDN/KDA/GDN2 nodes, so withholding those
    families would delete the operation rather than defer an optimization."""
    from cudnn.engines import manifest

    monkeypatch.delenv(manifest._ENABLE_ENV, raising=False)
    for name in ("gdn", "kda", "gdn2"):
        family = _family(name)
        assert not any(s.opt_in for s in family.slots.values()), f"{name} is the only implementation of its op and must not be gated"
        assert family.offered_ids(), f"{name} must be offered without the opt-in flag"


@pytest.mark.parametrize("value,offered", [("1", True), ("true", True), ("on", True), ("0", False), ("", False), ("no", False)])
def test_opt_in_flag_spellings(monkeypatch, value, offered):
    from cudnn.engines import manifest

    monkeypatch.setenv(manifest._ENABLE_ENV, value)
    assert manifest.opt_in_engines_enabled() is offered


def test_a_missing_optional_dependency_declines_rather_than_raising(monkeypatch):
    """Lowering imports resolve at build time now, so a missing extra surfaces
    from build_plan() rather than making the family vanish at import. The walk
    must treat that as a decline and move on -- otherwise a host without the
    cutedsl extra loses graphs the backend could have served."""

    class NeedsMissingExtra(StubEngine):
        name = "needs_extra"
        engine_id = _FAKE + 60

        def build_plan(self, graph, plan, ctx=None):
            raise ImportError("No module named 'not_installed_extra'")

    _offer(monkeypatch, NeedsMissingExtra(), StubEngine())

    g = pygraph()
    a, b = torch.randn(2, 3), torch.randn(3, 2)
    C = g.matmul(a, b)
    g.create_execution_plans()
    assert "needs_extra" in _plan_names(g), "it claims the graph; only lowering fails"

    g.build_plans()
    assert g.selected_engine is not None and g.selected_engine.name == "stub"

    c = torch.zeros(2, 2)
    g.execute({C: c})
    assert _marked(c)


def test_classification_is_a_partition():
    """Every node type names exactly ONE family, so "two families claimed this
    graph" is not a case that can arise — it is a lookup, not N competing
    claims that have to be proven disjoint."""
    from cudnn.engines.manifest import MANIFEST, _ANCHOR_NODE_TO_FAMILY

    known = {f.name for f in MANIFEST}
    for node_type, family in _ANCHOR_NODE_TO_FAMILY.items():
        assert family in known, f"{node_type} names {family!r}, which is not a family"


def test_a_graph_spanning_two_families_belongs_to_neither(monkeypatch):
    """A matmul and an sdpa in one graph is not a gemm graph and not an sdpa
    graph. No in-tree family serves that shape; the backend is the only
    candidate, and saying so beats letting one family half-claim it."""
    from cudnn.engines import manifest

    monkeypatch.setenv(manifest._ENABLE_ENV, "1")

    class Fake:
        def __init__(self, names):
            self.nodes = [type("N", (), {"node_type": type("T", (), {"name": n})})() for n in names]

    assert manifest.family_for(Fake(["MATMUL"])).name == "frost_gemm"
    assert manifest.family_for(Fake(["MATMUL", "POINTWISE"])).name == "frost_gemm"
    assert manifest.family_for(Fake(["MATMUL", "SDPA"])) is None
    assert manifest.family_for(Fake(["POINTWISE"])) is None


def test_classification_does_not_depend_on_the_machine(monkeypatch):
    """What kind of graph this is cannot depend on which machine is asking.

    Availability — arch range, maturity gate — is a separate question
    (offered_ids). Conflating them made "not that kind of graph"
    indistinguishable from "no engine for it here"."""
    from cudnn.engines import manifest

    class Fake:
        nodes = [type("N", (), {"node_type": type("T", (), {"name": "MATMUL"})})()]

    monkeypatch.delenv(manifest._ENABLE_ENV, raising=False)
    gated = manifest.family_for(Fake())
    monkeypatch.setenv(manifest._ENABLE_ENV, "1")
    assert manifest.family_for(Fake()) is gated, "the opt-in flag must not change the classification"

    family = _family("frost_gemm")
    assert manifest.family_for(Fake()) is family, "the opt-in state does not change what the graph IS"


def test_family_id_blocks_are_disjoint():
    """A family IS its id block: two families sharing one id would make the
    engine an autotune result names ambiguous."""
    from cudnn.engines.manifest import MANIFEST

    blocks = sorted((f.engine_id, f.id_end, f.name) for f in MANIFEST)
    for (_, prev_end, prev_name), (lo, _, name) in zip(blocks, blocks[1:]):
        assert lo >= prev_end, f"{prev_name} and {name} overlap in the id space"


# ---------------------------------------------------------------------------
# Facts: family-scoped, attached to the graph, read once.
# ---------------------------------------------------------------------------


def test_every_engine_spec_has_a_manifest_slot():
    """The manifest assigns ids, so an engine added without a slot would simply
    never be built -- silently. Catch that here rather than at runtime.

    Only families whose engines are enumerable without a device are checked;
    the point is that the two lists cannot drift apart unnoticed."""
    from cudnn.engines import manifest

    for family, spec_names in (
        (_family("frost_sdpa_fwd"), _spec_names("cudnn.sdpa.fwd.engines")),
        (_family("frost_sdpa_bwd"), _spec_names("cudnn.sdpa.bwd.engines")),
    ):
        missing = spec_names - set(family.slots)
        assert not missing, f"{family.name}: specs with no manifest slot: {sorted(missing)}"
        stale = set(family.slots) - spec_names
        assert not stale, f"{family.name}: slots naming no spec: {sorted(stale)} (never reuse a slot; leave it retired)"

    slots = [s.slot for f in manifest.MANIFEST for s in f.slots.values()]
    for family in manifest.MANIFEST:
        taken = [s.slot for s in family.slots.values()]
        assert len(taken) == len(set(taken)), f"{family.name}: two engines share a slot"
        assert all(0 <= s < manifest.FAMILY_BLOCK for s in taken), f"{family.name}: slot outside its block"
    assert slots  # the table is not empty


def _spec_names(module: str) -> set:
    import importlib

    return {s.name for s in importlib.import_module(module).ENGINE_SPECS}


def test_declared_analyzers_are_importable():
    """``analyzer`` is a pair of strings so matching stays import-free — which
    means a typo in it survives until something tries to rank that family."""
    from cudnn.engines import manifest

    for family in manifest.MANIFEST:
        analyzer = manifest.resolve_analyzer(family)
        if family.analyzer is None:
            assert analyzer is None
        else:
            assert callable(analyzer), family.name


_PROBE_CALLS = []


def _probe_analyzer(graph):
    """Stands in for a family's analyzer; named at module scope so a manifest
    entry can reference it the way a real family does (module, callable)."""
    _PROBE_CALLS.append(graph)
    return {"nodes": len(graph.nodes)}


def test_planning_attaches_facts_without_anyone_asking(monkeypatch):
    """Facts are not a call the user makes. Planning finds the graph's families
    through the manifest, runs each declared analyzer once, and hangs the
    records off the graph — a graph no family claims carries none.

    After _freeze(), not at validate(): a snapshot of a graph that can still
    change has to chase every mutation point, and missing one leaves facts
    describing a graph that is no longer there."""
    import contextlib

    import cudnn

    _PROBE_CALLS.clear()
    _offer(monkeypatch, StubEngine(), analyzer=(__name__, "_probe_analyzer"), name="probe_family")

    g = pygraph()
    g.matmul(torch.randn(2, 3), torch.randn(3, 2))
    g.validate()
    assert _PROBE_CALLS == [], "validate() must not snapshot a still-mutable graph"

    g.create_execution_plans()
    assert len(_PROBE_CALLS) == 1, "planning runs the family's analyzer itself"
    assert g._facts_for(_probe_analyzer) == {"nodes": 1}
    assert len(_PROBE_CALLS) == 1, "reading the payload must not re-parse"

    _offer(monkeypatch, StubEngine())

    # No MATMUL, so no family -- and with no family there is no python
    # candidate either, which on a box where the backend also declines a bare
    # relu leaves nothing to plan. The claim under test is about the payload,
    # not about the graph being servable, and _attach_facts runs either way.
    unclaimed = pygraph()
    unclaimed.relu(unclaimed.tensor(dim=[2, 2], name="X"))
    with contextlib.suppress(cudnn.cudnnGraphNotSupportedError):
        unclaimed.create_execution_plans()
    assert unclaimed._frozen and unclaimed._facts == {}


def test_ranking_and_engine_read_the_same_record(monkeypatch):
    """The contract is that the ranking and the engine cannot disagree about
    what the graph says — so exercise BOTH sides, not two calls to one
    accessor. The ranking resolves the analyzer from EngineFamily.analyzer;
    the engine passes the callable it already imports; keying on the analyzer
    itself is what makes those the same object."""
    seen = {}

    from cudnn.engines import heuristics

    real_rank = heuristics.rank

    def recording_rank(graph, engines, backend_plans, modes=None):
        # BEFORE reading it: planning must already have run the analyzer the
        # FAMILY declares. Without this the test passes even when the family
        # carries no analyzer at all, since the line below would resolve it.
        seen["parses_before_ranking"] = len(_PROBE_CALLS)
        seen["ranking"] = graph._facts_for(_probe_analyzer)
        return real_rank(graph, engines, backend_plans, modes)

    class Reader(StubEngine):
        name = "reader"
        engine_id = _FAKE + 0

        def check_support(self, graph):
            seen["engine"] = graph._facts_for(_probe_analyzer)

    _PROBE_CALLS.clear()
    monkeypatch.setattr(heuristics, "rank", recording_rank)
    # The analyzer has to reach the graph through EngineFamily.analyzer, which
    # is the path under test -- declaring a family by hand and then calling
    # _offer() would leave _offer's family the one that survives, with no
    # analyzer, and the assertions below would still pass because both sides
    # call _facts_for(_probe_analyzer) directly.
    _offer(monkeypatch, Reader(), analyzer=(__name__, "_probe_analyzer"), name="probe_family")
    g = pygraph()
    g.matmul(torch.randn(2, 3), torch.randn(3, 2))
    g.create_execution_plans()

    assert seen["parses_before_ranking"] == 1, "planning did not resolve the analyzer from EngineFamily.analyzer"
    assert seen["ranking"] is seen["engine"], "ranking and engine saw different records"
    assert len(_PROBE_CALLS) == 1, "one graph, one parse"


def test_facts_are_recomputed_when_the_graph_grows(monkeypatch):
    """Facts describe the graph AS READ; a graph that gained a node since is a
    different graph."""

    def analyzer(graph):
        return len(graph.nodes)

    g = pygraph()
    mm = g.matmul(torch.randn(2, 3), torch.randn(3, 2))
    assert g._facts_for(analyzer) == 1
    g.relu(mm)
    assert g._facts_for(analyzer) == 2


# ---------------------------------------------------------------------------
# Regressions from the !2280 review. Each of these ran the wrong way round
# before the fix, so they are the reason the fix is there.
# ---------------------------------------------------------------------------


class _FakeBackend:
    """Stands in for the lowered C++ graph: the walk's backend side, scriptable."""

    def __init__(self, check=None, build=None, planned=True):
        self._check, self._build = check, build
        self.calls = []
        self._planned = planned  # a fresh C++ graph reports 0 plans until create_execution_plans()

    def check_support(self):
        if self._check:
            raise self._check

    def build_plan_at_index(self, index):
        if self._build:
            raise self._build

    def get_plan_name_at_index(self, index):
        return f"fake_backend_plan_{index}"

    def get_execution_plan_count(self):
        return 1 if self._planned else 0

    def create_execution_plans(self, heur):
        self.calls.append("create_execution_plans")
        self._planned = True

    def get_engine_and_knobs_at_index(self, index):
        return (0, {})

    def deselect_engines(self, names):
        return self

    def deselect_behavior_notes(self, notes):
        self.calls.append("deselect_behavior_notes")
        return self

    def select_behavior_notes(self, notes):
        return self

    def deselect_numeric_notes(self, notes):
        return self

    def select_numeric_notes(self, notes):
        return self


def _backend_first(monkeypatch, *, check=None, build=None):
    """A graph whose ranked list is [backend, python], with a scripted backend."""
    from cudnn.engines.base import PlanConfig

    _ranking(monkeypatch, lambda graph, engines, backend_plans, modes=None: [PlanConfig(0, {}, cpp_index=0)] + [PlanConfig(e.engine_id, None) for e in engines])

    g = pygraph()
    _offer(monkeypatch, StubEngine())
    C = g.matmul(torch.randn(2, 3), torch.randn(3, 2))
    g._lowered_graph = _FakeBackend(check=check, build=build)
    g._cpp_plans_created = g._cpp_bog_done = True
    return g, C


def test_backend_check_support_decline_does_not_abort_the_walk(monkeypatch):
    """An aggregate backend check_support() answers for the BACKEND, not for one
    plan. Letting it raise from build() aborted the walk before it reached a
    python entry that can serve the graph."""
    import cudnn

    g, C = _backend_first(
        monkeypatch,
        check=cudnn.cudnnGraphNotSupportedError("backend cannot serve this graph"),
        build=cudnn.cudnnGraphNotSupportedError("backend cannot serve this graph"),
    )
    g.build()
    assert g.selected_engine is not None and g.selected_engine.name == "stub"
    g.execute({C: torch.empty(2, 2)})


def test_backend_check_support_decline_still_raises_when_it_is_the_only_plan():
    """...but with nothing else in the list, the decline is the answer."""
    import cudnn

    g = pygraph()
    g.matmul(torch.randn(2, 3), torch.randn(3, 2))
    g._lowered_graph = _FakeBackend(check=cudnn.cudnnGraphNotSupportedError("nope"))
    g._cpp_plans_created = g._cpp_bog_done = True
    g.create_execution_plans()
    with pytest.raises(cudnn.cudnnGraphNotSupportedError, match="nope"):
        g.check_support()


def test_execute_time_handle_reaches_a_lazily_built_python_plan(monkeypatch):
    """A JIT engine must compile for the device/stream it will run on. When the
    backend entry declines and the walk falls through to python DURING
    execute(handle=...), that handle has to reach build_plan()."""
    import cudnn

    monkeypatch.setattr(cudnn, "get_stream", lambda handle: None)
    seen = []

    class Recording(StubEngine):
        def build_plan(self, graph, plan, ctx=None):
            seen.append(ctx.handle if ctx else None)
            return super().build_plan(graph, plan, ctx)

    from cudnn.engines.base import PlanConfig

    _ranking(monkeypatch, lambda graph, engines, backend_plans, modes=None: [PlanConfig(0, {}, cpp_index=0)] + [PlanConfig(e.engine_id, None) for e in engines])

    g = pygraph()
    _offer(monkeypatch, Recording())
    C = g.matmul(torch.randn(2, 3), torch.randn(3, 2))
    g._lowered_graph = _FakeBackend(build=cudnn.cudnnGraphNotSupportedError("backend build declined"))
    g._cpp_plans_created = g._cpp_bog_done = True

    h = cudnn.Handle(backend_handle=42)
    g.execute({C: torch.empty(2, 2)}, None, h)
    assert seen == [h], f"build_plan saw {seen}, execute was given a cudnn.Handle"


def test_backend_runtime_error_is_not_a_decline(monkeypatch):
    """A CUDA/handle/backend-API failure arrives as RuntimeError. Reading it as
    'the backend declines' would silently run a python engine on a failing
    device and call it a routing decision."""
    g = pygraph()
    _offer(monkeypatch, StubEngine())
    g.matmul(torch.randn(2, 3), torch.randn(3, 2))

    def boom():
        raise RuntimeError("CUDA driver error: invalid device ordinal")

    # Lowering succeeds (the backend CAN represent the graph); the failure comes
    # from the heuristics call, where RuntimeError is not a routing answer.
    g._lower_backend_graph = lambda: setattr(g, "_lowered_graph", _FakeBackend())
    g._create_backend_plans = boom
    with pytest.raises(RuntimeError, match="invalid device ordinal"):
        g.backend_plan_entries()


def test_pinned_plan_that_was_deselected_raises(monkeypatch):
    """select_plan() and deselect_engines() contradicting each other is a caller
    error, not a licence to run a third plan."""
    g, C = _backend_first(monkeypatch)
    g.create_execution_plans()
    idx = _index_of(g, "stub")
    g.select_plan(idx)
    g.deselect_engines(["stub"])
    with pytest.raises(ValueError, match="pinned by select_plan"):
        g.build_plans()


def test_the_arch_probe_survives_a_missing_cuda_python():
    """An arch-gated family is only filtered when the arch is KNOWN. A missing
    cuda-python in the image must not read as 'wrong architecture' — that cost
    every arch-gated engine its place in the plan list once already."""
    import builtins

    from cudnn.frost import buffers

    if buffers.current_sm() is None:
        pytest.skip("no CUDA device to probe")
    real = builtins.__import__

    def blocked(name, *a, **k):
        if name == "cuda" or name.startswith("cuda.bindings"):
            raise ImportError("cuda-python is not installed")
        return real(name, *a, **k)

    builtins.__import__ = blocked
    try:
        assert buffers.current_sm() is not None, "the probe has no fallback once cuda-python is gone"
    finally:
        builtins.__import__ = real


def test_replayed_backend_entry_addresses_the_plan_it_replayed(monkeypatch):
    """``create_execution_plan`` APPENDS in C++ and ``build_plans`` short-circuits
    once a candidate exists, so a replayed entry must be addressed by the index it
    landed at — the plain calls would run whichever plan the backend already had."""

    class FakeAppendingBackend(_FakeBackend):
        def __init__(self):
            super().__init__()
            self.configs, self.candidate, self.ws = [0, 1], -1, {0: 111, 1: 222, 2: 333}

        def get_execution_plan_count(self):
            return len(self.configs)

        def create_execution_plan(self, engine_id, knobs):
            self.configs.append(engine_id)

        def build_plans(self, *a, **k):
            if self.candidate == -1:
                self.candidate = 0

        def build_plan_at_index(self, i):
            self.candidate = i

        def get_workspace_size(self, *a, **k):
            return self.ws[self.candidate]

        def get_workspace_size_plan_at_index(self, i, *a, **k):
            return self.ws[i]

    from cudnn.engines.base import PlanConfig

    _ranking(monkeypatch, lambda graph, engines, backend_plans, modes=None: [PlanConfig(0, {}, cpp_index=0), PlanConfig(7, {})])  # the 2nd is a replay

    g = pygraph()
    g.matmul(torch.randn(2, 3), torch.randn(3, 2))
    be = FakeAppendingBackend()
    g._lowered_graph = be
    g._cpp_bog_done = g._cpp_plans_created = True
    g.create_execution_plans()
    g.select_plan(1)
    g.build_plans()
    assert be.candidate == 2, f"the replayed plan landed at C++ index 2, backend built {be.candidate}"
    assert g.get_workspace_size() == 333


def test_create_execution_plan_freezes_the_graph(monkeypatch):
    """A plan refers to the graph as it is now. Replaying one on a fresh graph
    used to leave it mutable while _is_built said the plan was ready, so a node
    added afterwards silently diverged from the compiled plan."""
    g = pygraph()
    _offer(monkeypatch, StubEngine())
    g.matmul(torch.randn(2, 3), torch.randn(3, 2))
    g.create_execution_plan(StubEngine.engine_id, None)
    assert g._frozen
    with pytest.raises(RuntimeError, match="frozen"):
        g.matmul(torch.randn(2, 3), torch.randn(3, 2))


def test_a_frontend_bug_during_lowering_is_not_a_decline(monkeypatch):
    """'The backend cannot represent this graph' is a routing answer; an
    AssertionError from our own translator is a bug, and hiding it behind a
    python engine that happens to serve the graph is how it would stay hidden."""
    g = pygraph()
    _offer(monkeypatch, StubEngine())
    g.matmul(torch.randn(2, 3), torch.randn(3, 2))

    def boom():
        raise AssertionError("translator invariant failed")

    g._lower_backend_graph = boom
    with pytest.raises(AssertionError, match="translator invariant failed"):
        g.backend_plan_entries()


def test_backend_lowering_decline_hands_the_graph_to_a_python_engine(monkeypatch):
    """The dtype case: cuDNN has no descriptor for the graph, so it declines at
    LOWERING rather than from check_support(). The walk must still reach a
    python engine and produce the right answer, and the half-lowered state must
    be rolled back — a later build_operation_graph() would otherwise walk back
    into the descriptor that just failed."""
    import cudnn

    g = pygraph()
    _offer(monkeypatch, StubEngine())
    A, B = torch.randn(2, 3), torch.randn(3, 2)
    C = g.matmul(A, B)

    def declines():
        raise cudnn.cudnnGraphNotSupportedError("no descriptor for this data type")

    g._lower_backend_graph = declines
    g.create_execution_plans()

    assert all(is_python_engine(p.engine_id) for p in g.plans), _plan_names(g)
    assert g._lowered_graph is None and not g._cpp_bog_done and not g._cpp_plans_created
    g.build()
    assert g.selected_engine is not None and g.selected_engine.name == "stub"
    out = torch.zeros(2, 2)
    g.execute({C: out})
    assert _marked(out)


def test_the_ranking_output_is_the_plan_list_position_for_position(monkeypatch):
    """Nothing rewrites what the ranking returned. The delegating entry that
    backend_plan_entries() appends under heur_mode.OPENSOURCE used to share its
    id with a 'the backend goes here' placeholder the frontend expanded, so
    planning spliced the backend's whole list in a second time — wrong count,
    wrong ranking, wrong autotune result."""
    from cudnn.engines import PlanConfig

    made = []

    def entries(self):
        made.append(1)
        return [PlanConfig(0, {}, cpp_index=0), PlanConfig(BACKEND_HEURISTIC_ENGINE_ID)]

    monkeypatch.setattr(pygraph, "backend_plan_entries", entries)
    g = pygraph()
    g.matmul(torch.randn(2, 3), torch.randn(3, 2))
    g.create_execution_plans()

    assert [(p.engine_id, p.cpp_index) for p in g.plans] == [(0, 0), (BACKEND_HEURISTIC_ENGINE_ID, None)]
    assert len(made) == 1, "the backend's ranked list was queried twice in one planning pass"


def test_the_opensource_delegating_entry_outranks_the_concrete_plans(monkeypatch):
    """``Graph::build_plans`` builds the OSS engine BEFORE engine_configs and
    returns as soon as it succeeds. Ranking the delegating entry after the
    concrete plans made the walk build a native kernel first, so a caller who
    asked for heur_mode.OPENSOURCE silently got the thing they opted out of."""
    import cudnn

    class Fake:
        def get_execution_plan_count(self):
            return 2

        def get_engine_and_knobs_at_index(self, i):
            return (i, {})

        def get_engine_count(self):
            return 2

    g = pygraph()
    g.matmul(torch.randn(2, 3), torch.randn(3, 2))
    g._lowered_graph = Fake()
    g._cpp_plans_created = g._cpp_bog_done = True
    g._backend_heuristics = [cudnn.heur_mode.A, cudnn.heur_mode.OPENSOURCE]

    ids = [cfg.engine_id for cfg in g.backend_plan_entries()]
    assert ids[0] == BACKEND_HEURISTIC_ENGINE_ID, f"the OSS entry must rank first, got {ids}"


def test_backend_entries_are_queried_once_per_graph(monkeypatch):
    """A second C++ create_execution_plans() APPENDS to the same plan list, so
    asking the backend twice reports every plan twice. The ranking and the marker
    expansion both ask."""
    g = pygraph()
    _offer(monkeypatch, StubEngine())
    g.matmul(torch.randn(2, 3), torch.randn(3, 2))
    first = g.backend_plan_entries()
    assert g.backend_plan_entries() is first


def test_at_index_build_honours_the_exclusions(monkeypatch):
    """deselect_engines() / deselect_workspace_greater_than() are properties of
    the plan, not of the walk: the list is never filtered (indices stay stable),
    so build_plan_at_index() has to apply them too or it compiles and selects a
    plan the caller excluded."""
    g, C = _backend_first(monkeypatch)
    g.create_execution_plans()
    idx = _index_of(g, "stub")
    g.deselect_engines(["stub"])
    with pytest.raises(ValueError, match="deselect_engines"):
        g.build_plan_at_index(idx)


def test_at_index_build_honours_the_workspace_cap(monkeypatch):
    import cudnn

    class Hungry(StubEngine):
        name = "hungry"
        engine_id = _FAKE + 70

        def build_plan(self, graph, plan, ctx=None):
            plan_obj = super().build_plan(graph, plan, ctx)
            plan_obj.get_workspace_size = lambda: 4096
            return plan_obj

    g = pygraph()
    _offer(monkeypatch, Hungry())
    g.matmul(torch.randn(2, 3), torch.randn(3, 2))
    g.create_execution_plans()
    idx = _index_of(g, "hungry")
    g.deselect_workspace_greater_than(16)
    with pytest.raises(cudnn.cudnnGraphNotSupportedError, match="over the 16 limit"):
        g.build_plan_at_index(idx)


def test_behaviour_notes_reject_a_negative_index(monkeypatch):
    """Python indexing would quietly answer for the LAST plan; every other
    at-index API rejects it."""
    g, C = _backend_first(monkeypatch)
    g.create_execution_plans()
    with pytest.raises(IndexError, match="out of range"):
        g.get_behavior_notes_for_plan_at_index(-1)


def test_a_python_only_op_never_asks_the_backend():
    """GDN/KDA have no cuDNN node at all, which is knowable from the IR. Asking
    anyway cost a lowering that must fail on every such graph and logged an
    expected routing outcome at WARNING — 2 lines per KDA test in CI."""
    import cudnn

    reached = []
    T, N, H, K, V = 8, 1, 2, 4, 4

    g = pygraph()
    g.gdn(
        q=g.tensor([T, H, K], name="q"),
        k=g.tensor([T, H, K], name="k"),
        v=g.tensor([T, H, V], name="v"),
        g=g.tensor([T, H], name="g"),
        beta=g.tensor([T, H], name="beta"),
        cu_seqlens=g.tensor([N + 1], data_type=cudnn.data_type.INT32, name="cu_seqlens"),
        name="gdn",
    )
    g._lower_backend_graph = lambda: reached.append("lowered")

    assert g.backend_plan_entries() == []
    assert reached == [], "the backend was asked to lower a graph it has no node for"
    assert "GDN" in str(g._backend_declined)


def test_frost_opt_in_does_not_leak_out_of_the_frost_suites():
    """The frost conftests opt in per test, not at collection: a module-level
    env write would turn every later test in the same process into an opt-in
    run without saying so."""
    from cudnn.engines import manifest

    assert not manifest.opt_in_engines_enabled(), "CUDNN_FRONTEND_ENABLE_FROST_ENGINES leaked into the default-path tests"


def test_at_index_queries_answer_from_the_unified_list(monkeypatch):
    """get_engine_and_knobs_at_index() must describe the plan the caller just
    saw at that index — forwarding a unified index to C++ reported the backend's
    entry for a python plan's slot."""
    g, C = _backend_first(monkeypatch)
    g.create_execution_plans()
    idx = _index_of(g, "stub")
    eid, knobs = g.get_engine_and_knobs_at_index(idx)
    assert eid == StubEngine.engine_id
    assert (eid, knobs) == (g.plans[idx].engine_id, g.plans[idx].knobs)
    assert g.get_behavior_notes_for_plan_at_index(idx) == []  # engine declares none


def test_create_execution_plan_appends_a_python_plan(monkeypatch):
    """The deterministic-replay idiom: record (engine_id, knobs), rebuild it
    later, address it with count-1. It has to work for a python engine id too,
    or 'one id space' is only true for the backend."""
    g, C = _backend_first(monkeypatch)
    g.create_execution_plans()
    before = g.get_execution_plan_count()
    g.create_execution_plan(StubEngine.engine_id, None)
    assert g.get_execution_plan_count() == before + 1
    last = g.get_execution_plan_count() - 1
    assert g.get_engine_and_knobs_at_index(last)[0] == StubEngine.engine_id
    g.select_plan(last)
    g.build_plans()
    g.execute({C: torch.empty(2, 2)})
