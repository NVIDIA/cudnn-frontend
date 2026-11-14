# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Immutable geometry parameters shared by the SM100/SM107 FP8 swap-AB body."""

from dataclasses import dataclass

from .config_common import TemplateParams


@dataclass(frozen=True)
class TemplateParamsDecode(TemplateParams):
    """Extra geometry for the new swap-AB route, without changing legacy records' cache keys."""

    decode_head_dim: int = 128
    decode_live_columns: int = 0
