# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from cudnn._deps import torch_dep

if TYPE_CHECKING:
    import torch

from typing import Tuple, Optional



def make_tensor_strided_like(
    q_tensor: torch.Tensor,
    o_shape: Tuple[int, ...],
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
):
    """
    Create an empty tensor with the given shape that mimics the layout/strides of
    the provided `q_tensor` as closely as possible.
    """
    torch = torch_dep.require("cudnn.native_sparse_attention.utils.make_tensor_strided_like")
    q_strides = q_tensor.stride()
    rank_out = len(o_shape)
    order = tuple(sorted(range(min(len(q_strides), rank_out)), key=lambda i: q_strides[i]))

    strides = [0] * rank_out
    current = 1
    for dim in order:
        strides[dim] = current
        current *= o_shape[dim]

    return torch.empty_strided(
        o_shape,
        tuple(strides),
        dtype=dtype if dtype is not None else q_tensor.dtype,
        device=device if device is not None else q_tensor.device,
    )
