# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal functional and schedule policy for the causal-convolution family."""

FUNCTIONAL_COMPUTE_CAPABILITIES = frozenset(
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
F32X2_COMPUTE_CAPABILITIES = frozenset(
    {
        (10, 0),
        (10, 3),
        (11, 0),
        (12, 0),
        (12, 1),
    }
)

# The update route currently admits the same functional targets as the bulk
# route. Keep this semantic alias so diagnostics remain operation-specific even
# if the two implementations diverge later.
SUPPORTED_CAUSAL_CONV1D_UPDATE_COMPUTE_CAPABILITIES = FUNCTIONAL_COMPUTE_CAPABILITIES


def is_functional_arch(compute_capability: tuple[int, int]) -> bool:
    return compute_capability in FUNCTIONAL_COMPUTE_CAPABILITIES


def uses_vec8_schedule(compute_capability: tuple[int, int], n_channels: int) -> bool:
    return compute_capability in F32X2_COMPUTE_CAPABILITIES and n_channels % 8 == 0


def is_supported_causal_conv1d_update_compute_capability(
    compute_capability: tuple[int, int],
) -> bool:
    """Return whether the operation admits this functional target."""

    return is_functional_arch(compute_capability)


def supported_causal_conv1d_update_compute_capabilities_text() -> str:
    """Return the functional allowlist in a stable diagnostic format."""

    return ", ".join(f"{major}.{minor}" for major, minor in sorted(SUPPORTED_CAUSAL_CONV1D_UPDATE_COMPUTE_CAPABILITIES))
