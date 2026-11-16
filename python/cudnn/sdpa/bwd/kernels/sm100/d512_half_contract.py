# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Main implementations of the SM100 large-head half backward chain."""

LEGACY_ROW = "sdpa_bwd_sm100"
ROLE_SPLIT = "sdpa_bwd_sm100_d512_role_split"
TWO_BY_TWO = "sdpa_bwd_sm100_d512_2x2"


def select_implementation(facts):
    """Both stage-2 implementations serve the row's domain; role split stays first."""
    return ROLE_SPLIT, TWO_BY_TWO
