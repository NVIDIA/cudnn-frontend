# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple, Union

from torch import Tensor

from ._mha import _mha
from .graph import CscGraph


def mha_gat(
    graph: CscGraph,
    features: Union[Tensor, Tuple[Tensor, Tensor]],
    attn_weights: Tensor,
    *,
    edge_features: Optional[Tensor] = None,
    dropout_mask: Optional[Tensor] = None,
    num_heads: int = 1,
    concat_heads: bool = True,
    activation: str = "leaky_relu",
    activation_alpha: float = 0.2,
    return_attention_weights: bool = False,
    deterministic: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Apply GAT multi-head attention to a homogeneous or bipartite CSC graph."""
    return _mha(
        "gat",
        graph,
        features,
        attn_weights,
        edge_features=edge_features,
        dropout_mask=dropout_mask,
        num_heads=num_heads,
        concat_heads=concat_heads,
        activation=activation,
        activation_alpha=activation_alpha,
        return_attention_weights=return_attention_weights,
        deterministic=deterministic,
    )
