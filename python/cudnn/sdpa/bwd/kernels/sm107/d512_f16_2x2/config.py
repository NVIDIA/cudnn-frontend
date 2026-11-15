# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Implementation-local ring profile over the shared typed geometry factory."""

from types import MappingProxyType

from cudnn.sdpa.bwd.config_sm100 import SM107_USABLE_DYN_SMEM_2X2, TemplateParams2x2

RUBIN_ARM = MappingProxyType(dict(stages_kv=8, cast_stages=2, smem_cap_bytes=SM107_USABLE_DYN_SMEM_2X2))


def make_record(stage2_fields: dict) -> TemplateParams2x2:
    """Preserve the fixed eight-stage ring and the shared configuration checks."""
    return TemplateParams2x2(**stage2_fields, **RUBIN_ARM)


__all__ = ["RUBIN_ARM", "make_record"]
