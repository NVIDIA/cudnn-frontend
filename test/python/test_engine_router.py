"""CPU tests for the backend Router + BaseEngine contract.

These run without a GPU or cuDNN: they exercise NativeGraph -> Router ->
selected backend using the pure-PyTorch ReferenceMatmulEngine, plus routing
priority / fallback semantics. This is the CI-safe proof that the unification
contract works end to end.
"""

import pytest

torch = pytest.importorskip("torch")

from cudnn.graph_native import NativeGraph
from cudnn.engines import BaseEngine, Router, ReferenceMatmulEngine

pytestmark = pytest.mark.L0


def test_router_selects_by_priority_and_support():
    """First-supporting, by ascending priority; unsupported declines."""

    class Declines(BaseEngine):
        name = "declines"
        priority = 1

        def check_support(self, graph):
            raise NotImplementedError("nope")

        def execute(self, graph, tensor_data):
            raise AssertionError("should not run")

    class Accepts(BaseEngine):
        name = "accepts"
        priority = 10
        ran = False

        def execute(self, graph, tensor_data):
            type(self).ran = True

    g = NativeGraph()
    a = g.tensor(dim=[4, 8], name="A")
    b = g.tensor(dim=[8, 4], name="B")
    g.matmul(a, b, name="mm")
    g.register_backend(Accepts()).register_backend(Declines())

    selected = Router().select(g, g.backends)
    assert selected is not None and selected.name == "accepts"


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
    import cudnn

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


def test_no_backend_falls_back_to_cudnn_path():
    """With no registered backend, routing selects the cuDNN path (selected=None)."""
    g = NativeGraph()
    a = g.tensor(dim=[4, 8], name="A")
    b = g.tensor(dim=[8, 4], name="B")
    g.matmul(a, b, name="mm")
    g.validate()

    from cudnn.engines.router import default_router

    assert default_router.select(g, g.backends) is None
