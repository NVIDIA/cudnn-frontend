# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Implementation configuration using the shared immutable geometry helpers."""

from cudnn.sdpa.bwd.config_d256_2x2 import (
    FAMILY_F16,
    PROFILE_SM100,
    PROFILE_SM107_INTERLEAVED,
    TemplateParams,
    TemplateParams2x2,
    buffer_elems_2x2,
    desc_version_2x2,
    make_cfg_d256_2x2,
    q_write_tiles_2x2,
    tmem_layout_2x2,
    validate_head_chunk,
)

__all__ = [
    "FAMILY_F16",
    "PROFILE_SM100",
    "PROFILE_SM107_INTERLEAVED",
    "TemplateParams",
    "TemplateParams2x2",
    "buffer_elems_2x2",
    "desc_version_2x2",
    "make_cfg_d256_2x2",
    "q_write_tiles_2x2",
    "tmem_layout_2x2",
    "validate_head_chunk",
]
