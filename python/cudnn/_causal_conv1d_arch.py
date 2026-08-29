# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal architecture policy for the causal-convolution decode update."""

SUPPORTED_CAUSAL_CONV1D_UPDATE_COMPUTE_CAPABILITIES = frozenset(
    {
        (8, 0),
        (8, 6),
        (8, 7),
        (8, 9),
        (9, 0),
        (10, 0),
        (10, 3),
        (11, 0),
        (12, 0),
        (12, 1),
    }
)
SM100_COMPUTE_CAPABILITY = (10, 0)


def is_supported_causal_conv1d_update_compute_capability(compute_capability: tuple[int, int]) -> bool:
    """Return whether the operation admits this functional target."""

    return compute_capability in SUPPORTED_CAUSAL_CONV1D_UPDATE_COMPUTE_CAPABILITIES


def uses_sm100_causal_conv1d_update_schedule(compute_capability: tuple[int, int]) -> bool:
    """Return whether the measured SM100-only row-batch schedule is allowed."""

    return compute_capability == SM100_COMPUTE_CAPABILITY


def supported_causal_conv1d_update_compute_capabilities_text() -> str:
    """Return the functional allowlist in a stable diagnostic format."""

    return ", ".join(f"{major}.{minor}" for major, minor in sorted(SUPPORTED_CAUSAL_CONV1D_UPDATE_COMPUTE_CAPABILITIES))
