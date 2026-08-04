# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cudnn.linear_attention.frost: the FROST linear-attention engines —
Gated DeltaNet, Kimi Delta Attention, and Gated DeltaNet v2 on the SM100
chunked kernels built on Cutlass primitives. ``GdnFrostEngine`` is the
default GDN engine on SM100/SM103; ``KdaFrostEngine`` and ``Gdn2FrostEngine``
are forward-only (their backward kernels are stubs — KDA gradients run on
``KdaCuTileEngine``)."""

from .gdn_engine import GdnFrostEngine
from .gdn2_engine import Gdn2FrostEngine
from .kda_engine import KdaFrostEngine

__all__ = ["GdnFrostEngine", "KdaFrostEngine", "Gdn2FrostEngine"]
