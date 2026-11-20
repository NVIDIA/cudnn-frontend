# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-only bounds for backward bodies that launch heads/batch on grid.y/z."""


def yz_grid_mismatch(batch, heads):
    if max(batch, heads) > 65535:
        return f"backward grid.y/z requires heads={heads} and batch={batch} each at most 65535"
    return None
