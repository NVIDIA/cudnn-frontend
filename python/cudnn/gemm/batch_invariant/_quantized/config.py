# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

"""Persistent grid and instruction selection for a scaled matmul graph."""

from .._schedule import select_schedule


def schedule_for(op, device):
    return select_schedule(op.m, op.n, op.k, op.b_major, device, in_fmt=op.in_fmt, batch=op.batch)
