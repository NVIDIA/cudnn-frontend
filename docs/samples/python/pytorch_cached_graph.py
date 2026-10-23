# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cache a cudnn.Graph by the PyTorch tensor metadata that defines its contract."""

import cudnn
import torch

TensorKey = tuple[
    str,
    torch.dtype,
    tuple[int, ...],
    tuple[int, ...],
]
GraphKey = tuple[TensorKey, TensorKey]
GRAPH_CACHE: dict[GraphKey, cudnn.Graph] = {}


def tensor_key(tensor: torch.Tensor) -> TensorKey:
    return str(tensor.device), tensor.dtype, tuple(tensor.shape), tuple(tensor.stride())


def graph_key(a: torch.Tensor, b: torch.Tensor) -> GraphKey:
    return tensor_key(a), tensor_key(b)


def build_graph(a: torch.Tensor, b: torch.Tensor) -> cudnn.Graph:
    with cudnn.Graph(
        io_data_type=a.dtype,
        compute_data_type=torch.float32,
        inputs=["matmul::A", "matmul::B"],
        outputs=["C"],
        handle="auto",
    ) as graph:
        output = graph.matmul(name="matmul", A=a, B=b)
        output.set_name("C").set_output(True).set_data_type(a.dtype)
    return graph


def cached_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    key = graph_key(a, b)
    graph = GRAPH_CACHE.get(key)
    if graph is None:
        graph = build_graph(a, b)
        GRAPH_CACHE[key] = graph
    return graph(a, b)


def main() -> None:
    torch.manual_seed(7)
    for _ in range(2):
        a = torch.randn(2, 4, 8, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(1, 8, 6, device="cuda", dtype=torch.bfloat16)
        actual = cached_matmul(a, b)
        expected = torch.matmul(a.float(), b.float()).to(torch.bfloat16)
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

    assert graph_key(a, b) != graph_key(a, b.float())
    assert len(GRAPH_CACHE) == 1
    print("cached cudnn.Graph ok: builds=1 executions=2")


if __name__ == "__main__":
    main()
