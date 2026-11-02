# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared public constants and validation for the SM90 deterministic relay."""

import operator

SM90_DETERMINISTIC_RELAY_CLUSTER_SIZES = (4, 8, 16, 32, 64)
SM90_DEFAULT_DETERMINISTIC_RELAY_CLUSTER_SIZE = 16


def normalize_sm90_relay_cluster_size(value) -> int:
    """Return an exact integer relay width and reject lossy coercions."""
    try:
        cluster_size = operator.index(value)
    except TypeError as error:
        raise ValueError("SM90 deterministic_relay_cluster_size must be an integer " f"in {SM90_DETERMINISTIC_RELAY_CLUSTER_SIZES}, got {value!r}") from error
    if cluster_size not in SM90_DETERMINISTIC_RELAY_CLUSTER_SIZES:
        raise ValueError("SM90 deterministic_relay_cluster_size must be 4, 8, 16, 32, or 64 " f"(32, 16, 8, 4, or 2 FP32 planes), got {cluster_size}")
    return cluster_size
