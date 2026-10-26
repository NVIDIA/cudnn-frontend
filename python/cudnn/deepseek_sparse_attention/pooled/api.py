# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pooled DSA public functions."""

from cudnn.frost.buffers import cutedsl_requirement_error, cutedsl_state, cutedsl_too_old

if cutedsl_too_old(cutedsl_state()[1]):
    raise RuntimeError(cutedsl_requirement_error("Pooled DSA"))

from .cache import pool_cache_update_wrapper as pooled_cache_update_forward
from .forward import indexer_forward_wrapper as pooled_indexer_forward
from .indices import expand_indices_wrapper as pooled_expand_forward
from .selection import pooled_select_forward, pooled_select_workspace_size

__all__ = ["pooled_cache_update_forward", "pooled_indexer_forward", "pooled_expand_forward", "pooled_select_forward", "pooled_select_workspace_size"]
