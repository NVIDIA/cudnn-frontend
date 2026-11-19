# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Launch bounds of the shared D256 backward chain's batched gradient GEMMs."""

GRID_Z_MAX = 65535


def launch_error(batch, group, *, thd, batch_chunking, allow_subgroups=False):
    """Reject only when the smallest legal batch/head chunk cannot launch."""
    batches = batch if thd or not batch_chunking else 1
    heads = 1 if allow_subgroups and not thd else group
    if batches * heads > GRID_Z_MAX:
        return f"backward GEMM grid.z requires batch chunk * head chunk <= 65535; minimum is {batches} * {heads}"
    return None
