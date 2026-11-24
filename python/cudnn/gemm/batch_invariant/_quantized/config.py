# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
#
# Chunk and instruction selection are adapted from FlashInfer's
# flashinfer/gemm/batch_invariant/schedule.py (e2d163fb3d46) for cuDNN graph plans.

"""Instruction and launch geometry for batch-invariant quantized GEMM. The graph probe needs no CUDA compiler."""

from dataclasses import dataclass


def _chunks(n, k, fmt):
    if fmt == "e4m3_bs" or (n + 31) // 32 >= 256:
        return 1
    # Chunk boundaries include whole MMA steps and, for block-scaled operands, whole columns of the 128x4 scale layout.
    unit = 32 if fmt in ("e4m3", "e5m2") else 64 if fmt == "nvfp4" else 128
    chunks = 1
    while 2 * chunks <= min(k // (16 * n), 128) and k % (2 * chunks * unit) == 0 and k // (2 * chunks) >= 256:
        chunks *= 2
    if fmt in ("mxfp4w8a", "mxfp4w8a_k64") and n <= 512 and k >= 16384 and k % (8 * unit) == 0:
        chunks = max(chunks, 8)
    return chunks


def _instruction_format(fmt, n, k, cc):
    if fmt == "e4m3_block128":
        return "e4m3_bs"
    if cc == (10, 7):
        if fmt == "nvfp4" and _chunks(n, k, "nvfp4_k128") == 1:
            return "nvfp4_k128"
        return {"mxfp8": "mxfp8_k64", "mxfp4": "mxfp4_k128", "mxfp4w8a": "mxfp4w8a_k64"}.get(fmt, fmt)
    return fmt


@dataclass(frozen=True)
class Geometry:
    fmt: str
    chunks: int
    tile: tuple[int, int]
    cluster: tuple[int, int]
    total_clusters: int
    resident_clusters: int


def schedule_for(op, cc, sm_count):
    fmt = _instruction_format(op.in_fmt, op.n, op.k, cc)
    chunks = _chunks(op.n, op.k, fmt)
    if fmt in ("e4m3", "e5m2") and op.m <= 64:
        tile, cluster = (64, 32), (1, 4)
    elif op.m >= 256 and op.n >= 256:
        tile, cluster = (256, 128), (2, 2)
    else:
        tile, cluster = (128, 64), (1, 1)
    slots = cluster[0] * cluster[1]
    # One M tile per MMA group (one or two blocks). Neighbouring groups along N multicast the A operand.
    planes = chunks if chunks > 1 else op.batch
    total = ((op.m - 1) // tile[0] + 1) * ((op.n - 1) // (tile[1] * cluster[1]) + 1) * planes
    resident = min(total, max(1, sm_count // slots))
    # The static scheduler computes the total number of blocks and advances its int32 cursor even after the last tile.
    if max(total * slots, total - 1 + resident) >= 2**31:
        raise NotImplementedError("batch_invariant_gemm quantized matrix geometry exceeds the persistent scheduler's int32 counters")
    return Geometry(fmt, chunks, tile, cluster, total, resident)
