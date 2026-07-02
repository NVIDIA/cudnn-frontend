"""CPU test: cudnn.pygraph transparently routes represented graphs to a python engine.

Proves the in-place augmentation front-door: users build with the classic
``cudnn.pygraph`` API and, when a python engine is registered and supports the
whole (represented) graph, execution routes to it — no API change. Graphs with
an unrepresented op fall back to the classic cuDNN path.
"""

import pytest

torch = pytest.importorskip("torch")

import cudnn
from cudnn import pygraph_engines
from cudnn.engines import ReferenceMatmulEngine

# __init__ installs this on real builds; call again (idempotent) so the test is
# robust when run against a package whose __init__ predates the augmentation.
pygraph_engines.install(cudnn.pygraph)

pytestmark = pytest.mark.L0

M, K, N = 32, 16, 24


def test_pygraph_matmul_bias_relu_routes_to_reference_engine():
    a, b, bias = torch.randn(M, K), torch.randn(K, N), torch.randn(M, N)
    c = torch.empty(M, N)

    g = cudnn.pygraph(io_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    A = g.tensor(dim=[M, K], stride=[K, 1], data_type=cudnn.data_type.FLOAT)
    B = g.tensor(dim=[K, N], stride=[N, 1], data_type=cudnn.data_type.FLOAT)
    Bi = g.tensor(dim=[M, N], stride=[N, 1], data_type=cudnn.data_type.FLOAT)
    mm = g.matmul(A, B)
    bs = g.bias(input=mm, bias=Bi)
    Y = g.relu(input=bs)
    Y.set_output(True)

    g.register_backend(ReferenceMatmulEngine())
    g.execute({A: a, B: b, Bi: bias, Y: c})

    torch.testing.assert_close(c, torch.relu(a @ b + bias), atol=1e-4, rtol=1e-4)


def test_pygraph_without_engine_is_untouched():
    """No registered engine => the graph is not routed (classic behavior)."""
    g = cudnn.pygraph(io_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    A = g.tensor(dim=[M, K], stride=[K, 1], data_type=cudnn.data_type.FLOAT)
    B = g.tensor(dim=[K, N], stride=[N, 1], data_type=cudnn.data_type.FLOAT)
    C = g.matmul(A, B)
    C.set_output(True)

    st = pygraph_engines._STATE[g]
    assert st["selected"] is None
    assert pygraph_engines._route(g) is False  # no backend -> classic path
