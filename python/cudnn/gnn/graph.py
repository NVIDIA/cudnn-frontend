# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, replace
from typing import Optional

import torch
from torch import Tensor


@dataclass(frozen=True)
class CscGraph:
    """Tensor-backed CSC graph descriptor for cuDNN GNN operations.

    ``offsets`` has shape ``(num_dst_nodes + 1,)`` and ``indices`` has shape
    ``(num_edges,)``. Both tensors must use the same CUDA ``int32`` or
    ``int64`` dtype. ``map_csc_to_coo`` is optional and remaps each CSC edge
    position to the corresponding row in an edge-feature tensor.
    ``csc_rev_offsets`` and ``map_rev_to_coo`` optionally describe the graph
    in source-major reverse-CSC order for deterministic backward operations.
    They must be provided together.
    """

    offsets: Tensor
    indices: Tensor
    num_src_nodes: int
    map_csc_to_coo: Optional[Tensor] = None
    csc_rev_offsets: Optional[Tensor] = None
    map_rev_to_coo: Optional[Tensor] = None

    def __post_init__(self) -> None:
        if not isinstance(self.num_src_nodes, int):
            raise TypeError(f"num_src_nodes must be an int, got {type(self.num_src_nodes).__name__}")
        if self.num_src_nodes <= 0:
            raise ValueError(f"num_src_nodes must be positive, got {self.num_src_nodes}")
        if (self.csc_rev_offsets is None) != (self.map_rev_to_coo is None):
            raise ValueError("csc_rev_offsets and map_rev_to_coo must be provided together")

    @property
    def has_reverse_csc(self) -> bool:
        return self.csc_rev_offsets is not None

    def with_reverse_csc(self) -> "CscGraph":
        """Return this graph with reverse-CSC metadata for deterministic backward.

        Existing reverse metadata is preserved. Otherwise, the tensors are
        constructed on the graph device using the current PyTorch stream.
        """
        if self.has_reverse_csc:
            return self

        from ._utils import validate_csc_graph

        _, num_edges = validate_csc_graph(self.offsets, self.indices, self.map_csc_to_coo, self.num_src_nodes)
        counts = torch.zeros((self.num_src_nodes,), device=self.indices.device, dtype=self.indices.dtype)
        counts.scatter_add_(0, self.indices.long(), torch.ones_like(self.indices))
        csc_rev_offsets = torch.cat((torch.zeros((1,), device=self.indices.device, dtype=self.indices.dtype), counts.cumsum(dim=0, dtype=self.indices.dtype)))
        reverse_order = torch.argsort(self.indices, stable=True)
        edge_ids = torch.arange(num_edges, device=self.indices.device, dtype=self.indices.dtype) if self.map_csc_to_coo is None else self.map_csc_to_coo
        map_rev_to_coo = edge_ids[reverse_order]
        return replace(self, csc_rev_offsets=csc_rev_offsets, map_rev_to_coo=map_rev_to_coo)

    @property
    def num_dst_nodes(self) -> int:
        if self.offsets.ndim != 1:
            raise ValueError(f"offsets must be rank 1, got shape {tuple(self.offsets.shape)}")
        if self.offsets.numel() == 0:
            raise ValueError("offsets must contain at least one element")
        return self.offsets.numel() - 1

    @property
    def num_edges(self) -> int:
        if self.indices.ndim != 1:
            raise ValueError(f"indices must be rank 1, got shape {tuple(self.indices.shape)}")
        return self.indices.numel()

    @property
    def num_indices(self) -> int:
        return self.num_edges
