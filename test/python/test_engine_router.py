"""CPU tests for the backend Router + BaseEngine contract.

These run without a GPU or cuDNN: they exercise NativeGraph -> Router -> ranked
plan list -> engine-id dispatch using the pure-PyTorch ReferenceMatmulEngine.
This is the CI-safe proof that the unification contract works end to end.
"""

import pytest

torch = pytest.importorskip("torch")

from cudnn.pygraph import NativeGraph
from cudnn.engines import BaseEngine, Router, ReferenceMatmulEngine, PYTHON_ENGINE_ID_BASE, is_python_engine
from cudnn.engines.engine_ids import CUDNN_HEURISTIC_ENGINE_ID

pytestmark = pytest.mark.L0


def test_router_plan_list_includes_supporting_engines_then_cudnn():
    """Plan list = supporting python engines (by id) + a trailing cuDNN entry."""

    class Declines(BaseEngine):
        name = "declines"
        engine_id = PYTHON_ENGINE_ID_BASE + 50

        def check_support(self, graph):
            raise NotImplementedError("nope")

        def execute(self, graph, tensor_data):
            raise AssertionError("should not run")

    class Accepts(BaseEngine):
        name = "accepts"
        engine_id = PYTHON_ENGINE_ID_BASE + 10

        def execute(self, graph, tensor_data):
            pass

    g = NativeGraph()
    a = g.tensor(dim=[4, 8], name="A")
    b = g.tensor(dim=[8, 4], name="B")
    g.matmul(a, b, name="mm")
    g.register_backend(Accepts()).register_backend(Declines())

    plans = Router().plan(g, g.backends)
    ids = [p.engine_id for p in plans]
    # Only the supporting python engine is included, then the cuDNN entry last.
    assert ids == [PYTHON_ENGINE_ID_BASE + 10, CUDNN_HEURISTIC_ENGINE_ID]
    assert is_python_engine(ids[0]) and not is_python_engine(ids[-1])


def test_reference_matmul_execute_cpu():
    """ReferenceMatmulEngine runs a matmul on CPU and writes the output buffer."""
    g = NativeGraph()
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
    g = NativeGraph()
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

        def execute(self, graph, tensor_data):
            type(self).ran += 1

    class EngB(BaseEngine):
        name = "b"
        engine_id = PYTHON_ENGINE_ID_BASE + 11
        ran = 0

        def execute(self, graph, tensor_data):
            type(self).ran += 1

    g = NativeGraph()
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

        def execute(self, graph, tensor_data):
            pass

    class E1(BaseEngine):
        name = "e1"
        engine_id = PYTHON_ENGINE_ID_BASE + 20

        def execute(self, graph, tensor_data):
            pass

    g = NativeGraph()
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

        def execute(self, graph, tensor_data):
            pass

    g = NativeGraph()
    a = g.tensor(dim=[2, 2], name="A")
    g.matmul(a, g.tensor(dim=[2, 2], name="B"))
    g.register_backend(Buggy())
    with pytest.raises(RuntimeError, match="driver exploded"):
        g.create_execution_plans()


def test_no_backend_plan_list_is_cudnn_only():
    """With no python engine, the plan list is just the cuDNN entry (selected=None)."""
    g = NativeGraph()
    a = g.tensor(dim=[4, 8], name="A")
    b = g.tensor(dim=[8, 4], name="B")
    g.matmul(a, b, name="mm")
    g.validate()

    from cudnn.engines.router import default_router

    plans = default_router.plan(g, g.backends)
    assert [p.engine_id for p in plans] == [CUDNN_HEURISTIC_ENGINE_ID]
    g._plans = plans
    assert g.selected_engine is None  # cuDNN path
