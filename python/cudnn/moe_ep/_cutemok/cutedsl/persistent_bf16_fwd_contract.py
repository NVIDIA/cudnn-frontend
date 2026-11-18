# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae);
# ported to CuTe DSL and modified by NVIDIA.

"""Static structure of the BF16 forward megakernel, and the host's default geometry.

E/H/I/topk/macro/mini below are defaults of the host's plan only: device code must
not import them.  The runtime geometry is admitted by the host
(``persistent_bf16_fwd.PersistentBf16Plan.validate``), which also checks the signed
Int32 task / index bounds.

Only the cluster size, warp roles, pipeline depths, swizzle group and tile
dimensions are static device ABI.  The communication tile shape (``CommTileShape``,
one of COMM_TILE_SHAPES, chosen by the host per kernel variant) is the one
compile-time parameter of the communication CTAs: their shared-memory layout and
every communication task and readiness count derive from it; communication column
tails use actual byte counts within the fixed-size shared-memory slots.
"""

import dataclasses
import enum

NUM_LOCAL_EXPERTS = 64
HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 1024
TOPK = 10
MACROBATCH_ROWS = 32768
MINIBATCH_ROWS = 4096
CLUSTER_SIZE = 2
THREADS_PER_CTA = 256
WARPS_PER_CTA = THREADS_PER_CTA // 32
# Warp roles of the compute CTAs: warps 0-3 epilogue, 4 MMA,
# 5 CLC scheduler, 6 idle in BF16, 7 A/B TMA loads.
EPILOGUE_WARPS = (0, 1, 2, 3)
MMA_WARP = 4
CLC_WARP = 5
BF16_IDLE_WARP = 6
AB_LOAD_WARP = 7
MLP_TILE_ROWS = 256
MLP_TILE_COLUMNS = 256
MLP_K_TILE = 64
MLP_SUPERGROUP_SIZE = 8
# FC1 collective operand mapping: CTA_M rows per CTA, logical N per CTA,
# packed N across the pair; A/B load stages, accumulator stages, D store tiles.
FC1_CTA_M = 128
FC1_LOGICAL_N = 128
FC1_PACKED_N = 256
FC1_EPILOGUE_N = 32
AB_STAGES = 6
ACC_STAGES = 2
D_STAGES = 3
SWIGLU_TILE_ROWS = 128
SWIGLU_TILE_COLUMNS = 128
CLC_PIPE_DEPTH = 1
CLC_RESPONSE_BYTES = 16


@dataclasses.dataclass(frozen=True)
class CommTileShape:
    """Rows x BF16 columns of one communication tile of the BF16 task stream (``CuteMoKConfig.comm_tile_shape``):
    the compile-time parameter of the BF16 megakernels' communication CTAs, one kernel variant per shape.  A tile fills
    a communication warp's 32 KiB buffer of the shared stream (``_comm_stream``); lane ``r < rows`` moves row ``r`` of
    a column block with one bulk copy of ``row_chunk_bytes`` (1 KiB at 32 x 512, 2 KiB at 16 x 1024).  One arrival per
    tile.  Everything else is derived here (host and CPU model) or, over runtime sizes, by the stream modules: the
    tile rows of a minibatch (the stream's tasks), the column blocks of a row and the arrivals the compute side waits
    for."""

    rows: int
    columns: int

    @property
    def row_chunk_bytes(self) -> int:
        """Bytes of one row of a slot: the largest bulk copy."""
        return self.columns * 2

    @property
    def slot_bytes(self) -> int:
        return self.rows * self.row_chunk_bytes

    def row_tiles(self, rows: int) -> int:
        """Tile rows (stream tasks per direction) of ``rows`` routed rows, a multiple of 256."""
        return rows // self.rows

    def column_blocks(self, hidden_size: int) -> int:
        """Column blocks of one row; the last may be short."""
        return -(-hidden_size // self.columns)

    def tiles(self, rows: int, hidden_size: int) -> int:
        """Tiles of ``rows`` routed rows: the arrivals of a readiness counter over them."""
        return self.row_tiles(rows) * self.column_blocks(hidden_size)


# The shapes a kernel can be compiled for; the first is the default.  All fill the same slot.
COMM_TILE_SHAPES = (CommTileShape(32, 512), CommTileShape(16, 1024))
DEFAULT_COMM_TILE_SHAPE = COMM_TILE_SHAPES[0]
COMM_SLOT_BYTES = DEFAULT_COMM_TILE_SHAPE.slot_bytes  # 32 KiB
assert all(shape.slot_bytes == COMM_SLOT_BYTES and 256 % shape.rows == 0 for shape in COMM_TILE_SHAPES)


# Task->expert decode of the routed FC1 / Down tasks.  The method is a compile-time
# attribute of the kernel (no runtime branch between methods), split by the local expert
# count so that E <= EXPERT_TABLE_CAP runs exactly the per-expert shared-memory table
# decode with no group-walk code, and larger E gets the grouped table.  The method is part
# of the executor cache key.  The expert count, the group stride derived from it, the row
# counts and every tensor extent stay runtime values; the only static capacity is the
# EXPERT_TABLE_CAP shared-memory slots.  The host picks SMEM_BINARY for
# E <= EXPERT_TABLE_CAP and SMEM_GROUP above it (select_task_decode); LINEAR_SCAN is the
# table-free reference decode, reachable only by overriding that internal selector.
EXPERT_TABLE_CAP = 256


class TaskDecode(enum.IntEnum):
    LINEAR_SCAN = 0  # O(E) scan of tokens_per_expert per task
    SMEM_BINARY = 1  # shared inclusive row-block prefix per expert (stride 1, E <= EXPERT_TABLE_CAP):
    #                  upper_bound only, no global read in the decode
    SMEM_GROUP = 2  # the same prefix over groups of ceil(E / EXPERT_TABLE_CAP) experts (runtime
    #                  stride): upper_bound over the groups, then at most stride - 1 global reads
