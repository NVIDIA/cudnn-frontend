# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Main implementations of the SM100 large-head half backward chain."""

LEGACY_ROW = "sdpa_bwd_sm100"
ROLE_SPLIT = "sdpa_bwd_sm100_d512_role_split"
TWO_BY_TWO = "sdpa_bwd_sm100_d512_2x2"


def launch_error(batch: int):
    """All half chains put batch in the final GEMM's grid.z, including THD."""
    if batch > 65535:
        return f"D512 half backward GEMM grid.z requires batch <= 65535; got {batch}"
    return None


def grid_head_chunk(batch: int, heads: int, workspace_chunk: int) -> int:
    """Retain the budget's divisor unless its GEMM batch/head product overflows."""
    if batch * workspace_chunk <= 65535:
        return workspace_chunk
    # An impossible batch is rejected by check_support; keep initialization
    # metadata-only and preserve its nonzero chunk until that diagnostic.
    limit = max(1, min(workspace_chunk, 65535 // batch))
    return next(chunk for chunk in range(limit, 0, -1) if heads % chunk == 0)


def select_implementation(facts):
    """Both stage-2 implementations serve the row's domain; role split stays first."""
    return ROLE_SPLIT, TWO_BY_TWO
