# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from cudnn.gnn import CscGraph, mha_gat_v2

from gnn._mha_test_utils import MhaTestSuite, _require_gnn_mha, graph_data

__all__ = ["graph_data"]


class TestMhaGatV2(MhaTestSuite):
    variant = "gat_v2"
    op = staticmethod(mha_gat_v2)
    weight_size = 8


@pytest.mark.L1
@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize(
    ("input_dtype", "grad_dtype"),
    [
        (torch.float16, torch.float16),
        (torch.float16, torch.float32),
        (torch.bfloat16, torch.bfloat16),
        (torch.bfloat16, torch.float32),
    ],
)
def test_gnn_mha_gat_v2_gradient_dtype(deterministic, input_dtype, grad_dtype):
    _require_gnn_mha("gat_v2")
    graph = CscGraph(
        torch.tensor([0, 2, 5], device="cuda", dtype=torch.int32),
        torch.tensor([0, 2, 1, 2, 3], device="cuda", dtype=torch.int32),
        num_src_nodes=4,
    ).with_reverse_csc()
    src_features = torch.randn((4, 8), device="cuda", dtype=input_dtype, requires_grad=True)
    dst_features = torch.randn((2, 8), device="cuda", dtype=input_dtype, requires_grad=True)
    edge_features = torch.randn((5, 8), device="cuda", dtype=input_dtype, requires_grad=True)
    attn_weights = torch.randn((8,), device="cuda", dtype=input_dtype, requires_grad=True)
    for tensor in (src_features, dst_features, edge_features, attn_weights):
        tensor.grad_dtype = grad_dtype

    output = mha_gat_v2(
        graph,
        (src_features, dst_features),
        attn_weights,
        edge_features=edge_features,
        num_heads=2,
        deterministic=deterministic,
        grad_dtype=grad_dtype,
    )
    output.sum().backward()

    for tensor in (src_features, dst_features, edge_features, attn_weights):
        assert tensor.grad.dtype == grad_dtype
        assert torch.isfinite(tensor.grad).all()


@pytest.mark.L0
def test_gnn_mha_gat_v2_rejects_invalid_gradient_dtype():
    graph = CscGraph(
        torch.tensor([0, 1], device="cuda", dtype=torch.int32),
        torch.tensor([0], device="cuda", dtype=torch.int32),
        num_src_nodes=1,
    )
    node = torch.randn((1, 8), device="cuda", dtype=torch.float16)
    weights = torch.randn((8,), device="cuda", dtype=torch.float16)

    with pytest.raises(ValueError, match="grad_dtype must be torch.float16 or torch.float32"):
        mha_gat_v2(graph, node, weights, grad_dtype=torch.float64)
    with pytest.raises(TypeError, match="grad_dtype must be a torch.dtype or None"):
        mha_gat_v2(graph, node, weights, grad_dtype="float32")
