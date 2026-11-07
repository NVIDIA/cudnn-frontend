# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Device lowering of row partitions, shared by grouped copy pipelines."""

import cutlass
import cutlass.cute as cute


@cute.jit
def group_row_bounds(group, capacity, offsets, rows_per_group: cutlass.Constexpr[int]):
    """Return a bounded packed-row interval; zero selects runtime offsets.

    The existing grouped ABI starts group zero at row zero. Device offsets
    are untrusted: malformed ranges become empty or truncated, never OOB.
    """
    begin = cutlass.Int64(0)
    end = cutlass.Int64(0)
    if cutlass.const_expr(rows_per_group > 0):
        begin = cutlass.Int64(group) * rows_per_group
        end = begin + rows_per_group
    else:
        if group != 0:
            begin = cutlass.Int64(offsets[group])
        end = cutlass.Int64(offsets[group + 1])
    end = cutlass.min(cutlass.max(end, cutlass.Int64(0)), cutlass.Int64(capacity))
    begin = cutlass.min(cutlass.max(begin, cutlass.Int64(0)), end)
    return cutlass.Int32(begin), cutlass.Int32(end)


@cute.jit
def group_used_rows(capacity, num_groups, offsets, rows_per_group: cutlass.Constexpr[int]):
    """Row count for a grouped AVG, with a finite divisor for empty routing."""
    used = cutlass.Int64(capacity)
    if cutlass.const_expr(rows_per_group == 0):
        used = cutlass.min(cutlass.max(cutlass.Int64(offsets[num_groups]), cutlass.Int64(0)), used)
    return cutlass.max(used, cutlass.Int64(1))
