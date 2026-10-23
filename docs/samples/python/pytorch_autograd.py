# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wrap cached cuDNN ReLU graphs in a PyTorch autograd.Function."""

import cudnn
import torch

TensorMetadata = tuple[str, torch.dtype, tuple[int, ...], tuple[int, ...]]
FORWARD_CACHE: dict[TensorMetadata, cudnn.Graph] = {}
BACKWARD_CACHE: dict[tuple[TensorMetadata, TensorMetadata], cudnn.Graph] = {}


def metadata(tensor: torch.Tensor) -> TensorMetadata:
    return str(tensor.device), tensor.dtype, tuple(tensor.shape), tuple(tensor.stride())


def build_forward_graph(x: torch.Tensor) -> cudnn.Graph:
    with cudnn.Graph(
        io_data_type=x.dtype,
        intermediate_data_type=torch.float32,
        compute_data_type=torch.float32,
        inputs=["X"],
        outputs=["Y"],
        handle="auto",
    ) as graph:
        X = graph.tensor(name="X", dim=x.shape, stride=x.stride())
        Y = graph.relu(input=X, lower_clip=0.0, name="relu")
        Y.set_output(True).set_name("Y").set_data_type(x.dtype)
    return graph


def build_backward_graph(dy: torch.Tensor, x: torch.Tensor) -> cudnn.Graph:
    with cudnn.Graph(
        io_data_type=x.dtype,
        intermediate_data_type=torch.float32,
        compute_data_type=torch.float32,
        inputs=["dY", "X"],
        outputs=["dX"],
        handle="auto",
    ) as graph:
        dY = graph.tensor(name="dY", dim=dy.shape, stride=dy.stride())
        X = graph.tensor(name="X", dim=x.shape, stride=x.stride())
        dX = graph.relu_backward(loss=dY, input=X, lower_clip=0.0, name="relu_backward")
        dX.set_output(True).set_name("dX").set_data_type(x.dtype)
    return graph


class CudnnReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        key = metadata(x)
        graph = FORWARD_CACHE.get(key)
        if graph is None:
            graph = build_forward_graph(x)
            FORWARD_CACHE[key] = graph
        ctx.save_for_backward(x)
        return graph(x)

    @staticmethod
    def backward(ctx, dy: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        key = metadata(dy), metadata(x)
        graph = BACKWARD_CACHE.get(key)
        if graph is None:
            graph = build_backward_graph(dy, x)
            BACKWARD_CACHE[key] = graph
        return graph(dy, x)


def sample() -> None:
    torch.manual_seed(7)
    x = torch.randn((2, 3, 4), device="cuda", dtype=torch.float32, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_(True)
    upstream = torch.randn_like(x)

    actual = CudnnReLU.apply(x)
    expected = torch.relu(reference_x)
    torch.testing.assert_close(actual, expected)

    actual.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(x.grad, reference_x.grad)

    assert len(FORWARD_CACHE) == 1
    assert len(BACKWARD_CACHE) == 1
    print("PyTorch autograd integration ok: forward=1 backward=1")


if __name__ == "__main__":
    sample()
