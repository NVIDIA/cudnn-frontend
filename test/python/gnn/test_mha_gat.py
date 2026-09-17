# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from cudnn.gnn import CscGraph, mha_gat

from gnn._mha_test_utils import MhaTestSuite, _require_gnn_mha, graph_data

__all__ = ["graph_data"]


class TestMhaGat(MhaTestSuite):
    variant = "gat"
    op = staticmethod(mha_gat)
    weight_size = 16


@pytest.mark.L1
@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize(
    ("input_dtype", "feature_grad_dtype", "weight_grad_dtype"),
    [
        (torch.float16, torch.float16, torch.float16),
        (torch.float16, torch.float16, torch.float32),
        (torch.float16, torch.float32, torch.float32),
        (torch.bfloat16, torch.bfloat16, torch.bfloat16),
        (torch.bfloat16, torch.bfloat16, torch.float32),
        (torch.bfloat16, torch.float32, torch.float32),
    ],
)
def test_gnn_mha_gat_gradient_dtypes(deterministic, input_dtype, feature_grad_dtype, weight_grad_dtype):
    _require_gnn_mha("gat")
    graph = CscGraph(
        torch.tensor([0, 2, 5], device="cuda", dtype=torch.int32),
        torch.tensor([0, 2, 1, 2, 3], device="cuda", dtype=torch.int32),
        num_src_nodes=4,
    ).with_reverse_csc()
    src_features = torch.randn((4, 8), device="cuda", dtype=input_dtype, requires_grad=True)
    dst_features = torch.randn((2, 8), device="cuda", dtype=input_dtype, requires_grad=True)
    edge_features = torch.randn((5, 4), device="cuda", dtype=input_dtype, requires_grad=True)
    attn_weights = torch.randn((20,), device="cuda", dtype=input_dtype, requires_grad=True)
    src_features.grad_dtype = feature_grad_dtype
    dst_features.grad_dtype = feature_grad_dtype
    edge_features.grad_dtype = feature_grad_dtype
    attn_weights.grad_dtype = weight_grad_dtype

    output = mha_gat(
        graph,
        (src_features, dst_features),
        attn_weights,
        edge_features=edge_features,
        num_heads=2,
        deterministic=deterministic,
        feature_grad_dtype=feature_grad_dtype,
        weight_grad_dtype=weight_grad_dtype,
    )
    output.sum().backward()

    for feature in (src_features, dst_features, edge_features):
        assert feature.grad.dtype == feature_grad_dtype
        assert torch.isfinite(feature.grad).all()
    assert attn_weights.grad.dtype == weight_grad_dtype
    assert torch.isfinite(attn_weights.grad).all()


@pytest.mark.L0
def test_gnn_mha_gat_rejects_invalid_gradient_dtypes():
    graph = CscGraph(
        torch.tensor([0, 1], device="cuda", dtype=torch.int32),
        torch.tensor([0], device="cuda", dtype=torch.int32),
        num_src_nodes=1,
    )
    node = torch.randn((1, 8), device="cuda", dtype=torch.float16)
    weights = torch.randn((16,), device="cuda", dtype=torch.float16)

    with pytest.raises(ValueError, match="feature_grad_dtype=torch.float32 requires weight_grad_dtype=torch.float32"):
        mha_gat(graph, node, weights, feature_grad_dtype=torch.float32, weight_grad_dtype=torch.float16)
    with pytest.raises(ValueError, match="feature_grad_dtype must be torch.float16 or torch.float32"):
        mha_gat(graph, node, weights, feature_grad_dtype=torch.float64)
    with pytest.raises(TypeError, match="weight_grad_dtype must be a torch.dtype or None"):
        mha_gat(graph, node, weights, weight_grad_dtype="float32")
