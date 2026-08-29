# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Architecture policy for the experimental bulk causal-conv1d API."""

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


def is_functional_arch(compute_capability: tuple[int, int]) -> bool:
    return compute_capability in FUNCTIONAL_COMPUTE_CAPABILITIES


def uses_vec8_schedule(compute_capability: tuple[int, int], n_channels: int) -> bool:
    return compute_capability in F32X2_COMPUTE_CAPABILITIES and n_channels % 8 == 0
