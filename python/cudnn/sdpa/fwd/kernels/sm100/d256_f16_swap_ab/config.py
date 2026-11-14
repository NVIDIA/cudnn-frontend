# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""SM100/SM103 query-tile domain for the half D256 swap-AB body."""

from cudnn.sdpa.fwd.config_common import TemplateParams
from cudnn.sdpa.fwd.config_swap_ab import make_cfg_d256_decode as _make_cfg

from .contract import QUERY_TILES


def make_cfg_d256_decode(params: TemplateParams):
    """Resolve the chosen width; the caller, never this factory, chooses it."""
    return _make_cfg(params, q_tiles=QUERY_TILES)
