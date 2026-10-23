# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Register a cached cuDNN Graph as a PyTorch custom operator."""

from __future__ import annotations

import torch

import cudnn

TensorKey = tuple[
    str,
    torch.dtype,
    tuple[int, ...],
    tuple[int, ...],
]
GraphKey = tuple[TensorKey, TensorKey]
GRAPH_CACHE: dict[GraphKey, cudnn.Graph] = {}


# --8<-- [start:cache]
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


# --8<-- [end:cache]


# --8<-- [start:registration]
@torch.library.custom_op(
    "cudnn_docs::matmul",
    mutates_args=(),
    device_types="cuda",
)
def cudnn_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return cached_matmul(a, b)


@cudnn_matmul.register_fake
def cudnn_matmul_fake(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.matmul(a, b)


# --8<-- [end:registration]


# --8<-- [start:autograd]
def setup_context(ctx, inputs, output) -> None:
    del output
    ctx.save_for_backward(*inputs)


def backward(ctx, grad_output: torch.Tensor):
    a, b = ctx.saved_tensors
    grad_a = (grad_output @ b.transpose(-2, -1)).sum_to_size(a.shape)
    grad_b = (a.transpose(-2, -1) @ grad_output).sum_to_size(b.shape)
    return grad_a, grad_b


torch.library.register_autograd(
    "cudnn_docs::matmul",
    backward,
    setup_context=setup_context,
)
# --8<-- [end:autograd]


def main() -> None:
    torch.manual_seed(11)
    a = torch.randn(1, 32, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    b = torch.randn(2, 64, 48, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    actual = cudnn_matmul(a, b)
    expected = a @ b
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    grad = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, (a, b), grad, retain_graph=True)
    expected_grads = torch.autograd.grad(expected, (a, b), grad)
    for got, want in zip(actual_grads, expected_grads):
        torch.testing.assert_close(got, want, rtol=3e-2, atol=3e-2)

    compiled = torch.compile(cudnn_matmul, fullgraph=True)
    compiled_result = compiled(a.detach(), b.detach())
    torch.testing.assert_close(compiled_result, expected.detach(), rtol=2e-2, atol=2e-2)
    assert len(GRAPH_CACHE) == 1
    print("custom operator ok: eager, autograd, fake registration, compile")


if __name__ == "__main__":
    main()
