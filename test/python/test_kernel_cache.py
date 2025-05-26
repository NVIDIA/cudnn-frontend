import cudnn
import pytest
import torch
import itertools
import time
from looseversion import LooseVersion

from collections import namedtuple

"""
Unless overridden by a specific layer, the tolerance for each data type combination
follows this default definition.
"""
global_assert_opts_defaults = {
    # "hsh": dict(atol=5e-3, rtol=5e-3),
    # "hhh": dict(atol=1e-2, rtol=1e-2),
    # "sss": dict(atol=5e-3, rtol=5e-3),
    # "bbb": dict(atol=0, rtol=2e-2),
    "default": dict(atol=5e-2, rtol=5e-2),
}

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
@pytest.mark.L0
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


@pytest.mark.skipif(
    LooseVersion(cudnn.backend_version_string()) < "9.10",
    reason="requires cudnn 9.10 or higher",
)
def test_kernel_cache_persistence(cudnn_handle):
    kernel_cache = cudnn.create_kernel_cache()

    def create_my_graph(kernel_cache):
        g = cudnn.pygraph(
            io_data_type=cudnn.data_type.HALF,
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
            handle=cudnn_handle,
            kernel_cache=kernel_cache,
        )
        A = g.tensor(
            name="A",
            dim=[1, 8, 8],
            stride=[64, 8, 1],
        )
        B = g.tensor(
            name="B",
            dim=[1, 8, 8],
            stride=[64, 8, 1],
        )
        C = g.matmul(name="matmul", A=A, B=B)
        C.set_output(True).set_uid(2)
        A.set_uid(0)
        B.set_uid(1)
        return g

    graph = create_my_graph(kernel_cache)
    graph.build([cudnn.heur_mode.FALLBACK])
    str_json = kernel_cache.serialize()

    # Destroy the graph to ensure the kernel cache is independent of the graph
    del graph, kernel_cache

    kernel_cache = cudnn.create_kernel_cache()
    kernel_cache.deserialize(str_json)

    # Test that deserializing onto an existing kernel cache should be prohibited
    with pytest.raises(Exception):
        kernel_cache.deserialize(str_json)

    # Verify we can still use the deserialized kernel cache with a new graph
    graph = create_my_graph(kernel_cache)

    EXECUTION_TIME_LIMIT_MS = 10.0
    start_time = time.time()
    graph.build([cudnn.heur_mode.FALLBACK])
    build_time_ms = (time.time() - start_time) * 1000
    assert (
        build_time_ms <= EXECUTION_TIME_LIMIT_MS
    ), f"Graph build time {build_time_ms:.2f}ms exceeded limit of {EXECUTION_TIME_LIMIT_MS}ms"


@pytest.mark.skipif(
    LooseVersion(cudnn.backend_version_string()) < "9.10",
    reason="requires cudnn 9.10 or higher",
)
def test_serialize_both_graph_and_kernel_cache(cudnn_handle):
    kernel_cache = cudnn.create_kernel_cache()

    def create_my_graph(kernel_cache, m, n, k):
        g = cudnn.pygraph(
            io_data_type=cudnn.data_type.HALF,
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
            handle=cudnn_handle,
            kernel_cache=kernel_cache,
        )
        A = g.tensor(
            name="A",
            dim=[1, m, k],
            stride=[m * k, k, 1],
        )
        B = g.tensor(
            name="B",
            dim=[1, k, n],
            stride=[k * n, n, 1],
        )
        C = g.matmul(name="matmul", A=A, B=B)
        C.set_output(True).set_uid(2)
        A.set_uid(0)
        B.set_uid(1)
        return g

    graph = create_my_graph(kernel_cache, 8, 64, 128)
    graph.build([cudnn.heur_mode.FALLBACK])
    bytes_graph = graph.serialize()
    json_cache = kernel_cache.serialize()

    # Destroy the graph to ensure the kernel cache is independent of the graph
    del graph, kernel_cache

    # Verify the deserialized kernel cache and graph

    kernel_cache = cudnn.create_kernel_cache()
    kernel_cache.deserialize(json_cache)
    graph = cudnn.pygraph(
        # this is actually ignored as graph.deserialize() already skips graph.build()
        kernel_cache=kernel_cache,
    )
    graph.deserialize(bytes_graph)

    def create_tensors(m, n, k):
        A_gpu = torch.randn(
            1,
            m,
            k,
            device="cuda",
            dtype=torch.float16,
        )
        B_gpu = torch.randn(
            1,
            k,
            n,
            device="cuda",
            dtype=torch.float16,
        )
        C_expected = torch.matmul(A_gpu, B_gpu).to(torch.float16)
        C_actual = torch.empty_like(C_expected, device="cuda", dtype=torch.float16)
        return A_gpu, B_gpu, C_expected, C_actual

    A_gpu, B_gpu, C_expected, C_actual = create_tensors(8, 64, 128)
    workspace = torch.empty(
        graph.get_workspace_size(), device="cuda", dtype=torch.uint8
    )
    graph.execute({0: A_gpu, 1: B_gpu, 2: C_actual}, workspace, handle=cudnn_handle)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        C_actual, C_expected, **global_assert_opts_defaults["default"]
    )

    # try making a new one with the same kernel cache
    del graph
    C_actual = torch.empty_like(C_expected, device="cuda", dtype=torch.float16)
    graph = create_my_graph(kernel_cache, 16, 64, 256)
    EXECUTION_TIME_LIMIT_MS = 10.0
    start_time = time.time()
    graph.build([cudnn.heur_mode.FALLBACK])
    build_time_ms = (time.time() - start_time) * 1000
    assert (
        build_time_ms <= EXECUTION_TIME_LIMIT_MS
    ), f"Graph build time {build_time_ms:.2f}ms exceeded limit of {EXECUTION_TIME_LIMIT_MS}ms"
    A_gpu, B_gpu, C_expected, C_actual = create_tensors(8, 64, 128)
    workspace = torch.empty(
        graph.get_workspace_size(), device="cuda", dtype=torch.uint8
    )
    graph.execute({0: A_gpu, 1: B_gpu, 2: C_actual}, workspace, handle=cudnn_handle)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        C_actual, C_expected, **global_assert_opts_defaults["default"]
    )
