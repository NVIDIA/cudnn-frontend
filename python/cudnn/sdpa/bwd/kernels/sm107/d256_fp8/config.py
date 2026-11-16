# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Implementation configuration using the shared immutable geometry helpers."""

from cudnn.sdpa.bwd.config_sm107 import (
    FAMILY_FP8,
    TMEM_IS_EXCLUSIVE,
    TemplateParams,
    buffer_elems,
    desc_version,
    make_cfg_d256_bwd,
    q_write_tiles,
    tmem_layout,
)

PROFILE = 0

__all__ = ["FAMILY_FP8", "TMEM_IS_EXCLUSIVE", "TemplateParams", "buffer_elems", "desc_version", "make_cfg_d256_bwd", "q_write_tiles", "tmem_layout", "PROFILE"]
