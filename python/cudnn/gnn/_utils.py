# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared validation and binding helpers for cuDNN GNN operations."""

from typing import Optional, Tuple

from torch import Tensor

from ._dtypes import TORCH_INDEX_DTYPE_TO_CUDNN


def validate_csc_graph(
    offsets: Tensor,
    indices: Tensor,
    map_csc_to_coo: Optional[Tensor],
    num_src_nodes: int,
) -> Tuple[int, int]:
    """Validate shared CSC tensor invariants and return destination and edge counts.

    Operator-specific restrictions, such as whether an operation supports a
    nonempty destination set with zero edges, remain the caller's responsibility.
    """

    if offsets.ndim != 1:
        raise ValueError(f"offsets must be rank 1, got shape {tuple(offsets.shape)}")
    if indices.ndim != 1:
        raise ValueError(f"indices must be rank 1, got shape {tuple(indices.shape)}")
    if offsets.numel() == 0:
        raise ValueError("offsets must contain at least one element")
    if offsets.dtype not in TORCH_INDEX_DTYPE_TO_CUDNN:
        raise TypeError(f"offsets must have dtype int32 or int64, got {offsets.dtype}")
    if indices.dtype != offsets.dtype:
        raise TypeError(f"indices dtype {indices.dtype} must match offsets dtype {offsets.dtype}")
    if not offsets.is_cuda or not indices.is_cuda:
        raise ValueError(f"offsets and indices must be CUDA tensors, got {offsets.device} and {indices.device}")
    if offsets.device != indices.device:
        raise ValueError(f"offsets and indices must be on the same device, got {offsets.device} and {indices.device}")
    if not isinstance(num_src_nodes, int):
        raise TypeError(f"num_src_nodes must be an int, got {type(num_src_nodes).__name__}")
    if num_src_nodes <= 0:
        raise ValueError(f"num_src_nodes must be positive, got {num_src_nodes}")

    num_dst_nodes = offsets.numel() - 1
    num_edges = indices.numel()
    if num_dst_nodes == 0 and num_edges != 0:
        raise ValueError("a graph with zero destination nodes cannot contain edges")

    if map_csc_to_coo is not None:
        if map_csc_to_coo.ndim != 1 or map_csc_to_coo.numel() != num_edges:
            raise ValueError(f"map_csc_to_coo must have shape ({num_edges},), got {tuple(map_csc_to_coo.shape)}")
        if map_csc_to_coo.dtype != offsets.dtype:
            raise TypeError(f"map_csc_to_coo dtype {map_csc_to_coo.dtype} must match offsets dtype {offsets.dtype}")
        if map_csc_to_coo.device != offsets.device:
            raise ValueError(f"map_csc_to_coo must be on {offsets.device}, got {map_csc_to_coo.device}")

    return num_dst_nodes, num_edges


def tensor_pointer(tensor: Optional[Tensor]) -> int:
    """Return a tensor's device pointer, or a null pointer for ``None``."""

    return 0 if tensor is None else tensor.data_ptr()


def require_backend_symbols(*symbol_names: str) -> None:
    """Raise ``RuntimeError`` when required optional backend bindings are absent."""

    import cudnn

    missing = [symbol_name for symbol_name in symbol_names if not hasattr(cudnn, symbol_name)]
    if missing:
        raise RuntimeError(
            f"Required cuDNN backend bindings are unavailable: {', '.join(missing)}. "
            "Build cudnn-frontend against headers containing the required GNN APIs on a supported platform."
        )
