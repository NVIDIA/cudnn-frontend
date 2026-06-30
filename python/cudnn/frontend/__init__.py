# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Internal support catalog for Torch-first frontend-only operations."""

from ._legacy_gaps import (
    PREEXISTING_JAX_GAP_ANCHORS,
    PREEXISTING_KERNEL_OWNERSHIP_GAPS,
    PREEXISTING_REGISTERED_JAX_GAP_IDS,
)
from ._registry import FRONTEND_OPERATION_REGISTRY, FrontendTarget
from .rmsnorm_rht_amax import (
    _rmsnorm_rht_amax_sm100_contract as _rmsnorm_rht_amax_sm100_contract,
)


def registered_operations():
    """Return registered Torch-canonical operation specifications."""

    return FRONTEND_OPERATION_REGISTRY.operations()


def known_jax_gaps():
    """Return the checked-in migration baseline of Torch-only API anchors."""

    return tuple(sorted(PREEXISTING_JAX_GAP_ANCHORS))


def known_kernel_ownership_gaps():
    """Return physical CuTe kernels not yet mapped to semantic operations."""

    return tuple(sorted(PREEXISTING_KERNEL_OWNERSHIP_GAPS))


def known_registered_jax_gap_ids():
    """Return operation-level JAX gaps grandfathered into the catalog."""

    return tuple(sorted(PREEXISTING_REGISTERED_JAX_GAP_IDS))


__all__ = [
    "FrontendTarget",
    "known_jax_gaps",
    "known_kernel_ownership_gaps",
    "known_registered_jax_gap_ids",
    "registered_operations",
]
