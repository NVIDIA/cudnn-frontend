# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend-independent linear-attention operation contracts (torch custom-op wrappers)."""

from .gdn import gated_delta_net
from .gdn2 import gated_delta_net_v2
from .kda import kimi_delta_attention

__all__ = ["gated_delta_net", "kimi_delta_attention", "gated_delta_net_v2"]
