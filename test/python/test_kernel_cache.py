import cudnn
import pytest
import torch
import itertools
from looseversion import LooseVersion

from collections import namedtuple

problem_defintion = namedtuple("problem_defintion", ["b", "m", "n", "k"])

shapes = [
    problem_defintion(b=16, m=32, n=32, k=128),
    problem_defintion(b=16, m=64, n=64, k=128),
    problem_defintion(b=16, m=80, n=80, k=128),
    problem_defintion(b=32, m=128, n=128, k=256),
    problem_defintion(b=32, m=64, n=64, k=256),
]


def build_cudnn_graph(handle, cache, shape):
    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.HALF,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=handle,
        kernel_cache=cache,
    )

    A = graph.tensor(
        name="A",
        dim=[shape.b, shape.m, shape.k],
        stride=[shape.m * shape.k, shape.k, 1],
    )
    B = graph.tensor(
        name="B",
        dim=[shape.b, shape.k, shape.n],
        stride=[shape.n * shape.k, shape.n, 1],
    )

    C = graph.matmul(name="matmul", A=A, B=B)
    C.set_output(True).set_uid(2)

    A.set_uid(0)
    B.set_uid(1)

    graph.build([cudnn.heur_mode.A])

    return graph


@pytest.mark.skipif(
    LooseVersion(cudnn.backend_version_string()) < "9.5",
    reason="requires cudnn 9.5 or higher",
)
def test_kernel_cache(cudnn_handle):

    cache = cudnn.create_kernel_cache()

    for shape in shapes:
        graph = build_cudnn_graph(cudnn_handle, cache, shape)

        A = torch.randn(
            shape.b,
            shape.m,
            shape.k,
            requires_grad=False,
            device="cuda",
            dtype=torch.bfloat16,
        )
        B = torch.randn(
            shape.b,
            shape.k,
            shape.n,
            requires_grad=False,
            device="cuda",
            dtype=torch.bfloat16,
        )
        C = torch.randn(
            shape.b,
            shape.m,
            shape.n,
            requires_grad=False,
            device="cuda",
            dtype=torch.bfloat16,
        )

        workspace = torch.empty(
            graph.get_workspace_size(), device="cuda", dtype=torch.uint8
        )

        print("Executing", shape)
        graph.execute({0: A, 1: B, 2: C}, workspace, handle=cudnn_handle)
