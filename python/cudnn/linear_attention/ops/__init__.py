# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend-independent linear-attention operation contracts (torch custom-op wrappers)."""

from .gdn import gated_delta_net, gated_delta_net_summary_bwd, gated_delta_net_summary
from .gdn2 import gated_delta_net_v2, gated_delta_net_v2_summary_bwd, gated_delta_net_v2_summary
from .gdp import gated_delta_product, gated_delta_product_summary_bwd, gated_delta_product_summary
from .kda import kimi_delta_attention, kimi_delta_attention_summary_bwd, kimi_delta_attention_summary

__all__ = [
    "gated_delta_net",
    "kimi_delta_attention",
    "gated_delta_net_v2",
    "gated_delta_product",
    "gated_delta_net_summary",
    "kimi_delta_attention_summary",
    "gated_delta_net_v2_summary",
    "gated_delta_product_summary",
    "gated_delta_net_summary_bwd",
    "kimi_delta_attention_summary_bwd",
    "gated_delta_net_v2_summary_bwd",
    "gated_delta_product_summary_bwd",
]
