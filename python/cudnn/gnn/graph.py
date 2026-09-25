# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional

from torch import Tensor


@dataclass(frozen=True)
class CscGraph:
    """Tensor-backed CSC graph descriptor for cuDNN GNN operations.

    ``offsets`` has shape ``(num_dst_nodes + 1,)`` and ``indices`` has shape
    ``(num_edges,)``. Both tensors must use the same CUDA ``int32`` or
    ``int64`` dtype. ``map_csc_to_coo`` is optional and remaps each CSC edge
    position to the corresponding row in an edge-feature tensor.
    """

    offsets: Tensor
    indices: Tensor
    num_src_nodes: int
    map_csc_to_coo: Optional[Tensor] = None

    def __post_init__(self) -> None:
        if not isinstance(self.num_src_nodes, int):
            raise TypeError(f"num_src_nodes must be an int, got {type(self.num_src_nodes).__name__}")
        if self.num_src_nodes <= 0:
            raise ValueError(f"num_src_nodes must be positive, got {self.num_src_nodes}")

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
