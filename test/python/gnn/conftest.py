# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from cudnn.gnn import CscGraph


@pytest.fixture(params=[torch.int32, torch.int64], ids=["int32", "int64"])
def index_dtype(request):
    return request.param


@pytest.fixture
def small_csc_graph(index_dtype):
    offsets = torch.tensor([0, 2, 5], device="cuda", dtype=index_dtype)
    indices = torch.tensor([0, 2, 1, 2, 3], device="cuda", dtype=index_dtype)
    edge_map = torch.tensor([3, 0, 4, 1, 2], device="cuda", dtype=index_dtype)
    return CscGraph(offsets, indices, num_src_nodes=4, map_csc_to_coo=edge_map)


@pytest.fixture
def small_reverse_csc_graph(small_csc_graph):
    return small_csc_graph.with_reverse_csc()


@pytest.fixture
def medium_csc_graph(index_dtype):
    num_src_nodes = 257
    num_dst_nodes = 193
    num_edges = 4096

    ranks = torch.arange(1, num_dst_nodes + 1, dtype=torch.float64)
    weights = ranks.reciprocal()
    extra_degrees = torch.floor(weights / weights.sum() * (num_edges - num_dst_nodes)).to(torch.int64)
    extra_degrees[: num_edges - num_dst_nodes - extra_degrees.sum().item()] += 1
    degrees = extra_degrees + 1
    offsets = torch.cat((torch.zeros(1, dtype=torch.int64), degrees.cumsum(0)))

    generator = torch.Generator().manual_seed(2026)
    indices = torch.randint(num_src_nodes, (num_edges,), generator=generator, dtype=torch.int64)
    edge_map = torch.randperm(num_edges, generator=generator, dtype=torch.int64)
    return CscGraph(
        offsets.to(device="cuda", dtype=index_dtype),
        indices.to(device="cuda", dtype=index_dtype),
        num_src_nodes=num_src_nodes,
        map_csc_to_coo=edge_map.to(device="cuda", dtype=index_dtype),
    )


@pytest.fixture
def medium_reverse_csc_graph(medium_csc_graph):
    return medium_csc_graph.with_reverse_csc()
