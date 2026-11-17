# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae);
# ported to CuTe DSL and modified by NVIDIA.

"""BF16 forward megakernel: communication, FC1 (Gate/Up + SwiGLU) and Down in one persistent launch.

Layout of the launch (cluster size 2).  The first ``num_comm_sms / 2`` clusters are communication CTAs running
the dynamic task stream (``_fwd_dynamic_task_stream``): they pull the routed token rows from the peer ranks into
the X ring and push the finished Y-ring rows into the peers' combine buffers.  Every later cluster is a compute
cluster that takes one tile per task from the CLC scheduler (depth 1, group size 1): shared FC1, shared Down,
routed FC1 or routed Down.  The grid is sized for the schedule capacity; clusters past the true task count do no
work and are reclaimed by an explicit CLC drain.

Compute CTA.  The CLC warp decodes each task once and publishes the record (TASK_INFO_FIELDS) through a
one-stage shared-memory pipeline; the load, MMA, epilogue and idle warps read the record instead of decoding.
The shared-expert FC1 is one 2-CTA instruction in which CTA 0 supplies Gate and CTA 1 supplies Up.  The routed
Gate/Up and Down tiles run swap-AB (weights as operand A, tokens as operand B, N = align16(valid rows of the
block); see ``_swapab_fwd``).  Every FC1 rounds Gate and Up to BF16 and applies the activation in its epilogue;
no packed or copied Gate/Up weight is ever materialized.

Order and readiness.  Communication takes the macrobatches from the last to the first; with several
macrobatches the compute decode takes the minibatches of a macrobatch from the last to the first as well.  The
edges are global counters a consumer waits on before it loads: ``x_routed_ready`` (pulls of a minibatch ->
its routed FC1), ``hidden_row_block_ready`` (FC1 epilogue -> Down), ``y_routed_ready`` (routed Down of a
minibatch -> its pushes and the pulls that reuse its X-ring rows) and ``y_routed_done`` (pushes of macrobatch
m + 1 -> the routed Down of macrobatch m that overwrites the same Y-ring rows).

Runtime values: token counts, both hidden sizes, the intermediate size, the expert count, macrobatch and
minibatch size, the number of communication SMs, top-k and the clamp limit.  Compile time: tile shapes, the clamp
mode, the task decode method (TaskDecode) and the variant flags of ``_PersistentBf16FwdMega``.

The ``shared_fc1_only`` variant launches the shared FC1 tasks alone -- no communication cluster, no shared Down and
no routed minibatch in the grid -- for ``persistent_bf16_fwd.shared_fc1_bf16``; the kernel is the same code.

Private device module; the public entries are ``persistent_bf16_fwd.forward_bf16`` and ``shared_fc1_bf16``.
"""

from ._version_gate import require_cutedsl

require_cutedsl()  # CuTe DSL >= 4.8 (cu13), before CUTLASS is imported

import inspect

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass import BFloat16, Boolean, Float32, Int32, Int64, Uint32
from cutlass._mlir import ir
from cutlass.cute.nvgpu import cpasync

from . import routed_descriptors as routed_descriptors_pkg
from cutlass.cute.runtime import from_dlpack
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from ._mxfp8_bwd_math import glu_forward
from ._compat.gemm_sm100 import NamedBarrierGemm
from ._compat.pipeline import PipelineTmaUmma, PipelineUmmaAsync

from ._persistent_bf16_fwd_gemm import (
    AB_STAGES as _GATE_AB_STAGES,
    ACC_STAGES as _GATE_ACC_STAGES,
    A_BYTES as GEMM_A_BYTES,
    B_BYTES as GEMM_B_BYTES,
    D_STAGES as _GATE_D_STAGES,
    D_BYTES as GEMM_D_BYTES,
    EPILOGUE_N as _GATE_EPILOGUE_N,
    K_TILE as _GATE_TILE_K,
    LOGICAL_N as _GATE_LOGICAL_N,
    MMA_WARP as _GATE_MMA_WARP,
    PACKED_N as _GATE_TILE_N,
    TMA_WARP as _GATE_AB_LOAD_WARP,
    _Fc1Collective,
    _make_tma_block_load_fn,
)
from .persistent_bf16_fwd_contract import (
    BF16_IDLE_WARP,
    CLC_PIPE_DEPTH,
    CLC_RESPONSE_BYTES,
    CLC_WARP,
    CLUSTER_SIZE,
    COMM_TILE_SHAPES,
    DEFAULT_COMM_TILE_SHAPE,
    CommTileShape,
    EPILOGUE_WARPS,
    EXPERT_TABLE_CAP,
    FC1_CTA_M,
    MLP_TILE_COLUMNS,
    MLP_TILE_ROWS,
    SWIGLU_TILE_COLUMNS,
    SWIGLU_TILE_ROWS,
    THREADS_PER_CTA,
    TaskDecode,
    WARPS_PER_CTA,
)
from . import _dynamic_task_stream, _fwd_dynamic_task_stream, _swapab_fwd
from ._arch import COMPILE_OPTIONS
from ._host_common import peer_address_table, row_major_view
from ._bwd_device_helpers import _mbar_backoff_wait, named_barrier_sync_unaligned
from ._fwd_device_helpers import (
    _peer_address,
    _counter_arrive_gpu,
    _swizzled_mlp_tile_coord_i32,
    _counter_wait_gpu,
    _ceil_div_i32,
)

CLUSTER_SHAPE = (CLUSTER_SIZE, 1, 1)
FUSED_GATE_UP_TASK_GROUP_SIZE = 1
FUSED_DOWN_TASK_GROUP_SIZE = 1
CLC_DRAIN_WARPS = WARPS_PER_CTA
# The CLC response is read by the CLC warp of each CTA only (one arrival per CTA on rank 0's
# empty barrier).  That warp decodes the task once per CTA and publishes the decoded record to
# the other warps through the task-info pipeline (one scheduler warp, a TaskInfo ring with
# full/empty barriers; every other role reads the record instead of decoding).
CLC_COMPLETION_WARPS = CLUSTER_SIZE
# Task-info record: Int32 words written by the CLC warp, read by the load, MMA, four epilogue
# and idle warps (TASK_INFO_CONSUMER_WARPS, one elected arrival each).  "active" is the loop
# condition of every consumer (0 = stop); "needs_drain" is the CLC capacity-tail state of the
# final record.  One stage: the CLC warp requests the next task only after every consumer has
# taken the current record (CLC depth 1: the cluster holds at most the task it
# runs plus one in flight), so a second stage could never be filled before the first is released;
# a deeper ring is a CLC2-like scheduling change, not a transport option.
TASK_INFO_STAGES = 1
TASK_INFO_FIELDS = (
    "active",
    "needs_drain",
    "kind",
    "valid_tile",
    "expert_index",
    "tile_coord_m",
    "tile_coord_n",
    "input_ready_index",
    "input_ready_required",
    "output_ready_index",
    "k_tile_count",
    "expert_row_offset",
    "macrobatch",
    "tile_valid_rows",
)
TASK_INFO_WORDS = 16  # 64 B per stage, 16-byte aligned
TASK_INFO_CONSUMER_WARPS = WARPS_PER_CTA - 1
assert len(TASK_INFO_FIELDS) == len(set(TASK_INFO_FIELDS)) <= TASK_INFO_WORDS
_TI = {name: index for index, name in enumerate(TASK_INFO_FIELDS)}
# Communication CTAs run the dynamic task stream prefix (_fwd_dynamic_task_stream); its slot pool aliases the
# compute CTAs' GEMM arena.
GEMM_ARENA_BYTES = GEMM_A_BYTES + GEMM_B_BYTES + GEMM_D_BYTES
REUSABLE_ARENA_BYTES = max(_fwd_dynamic_task_stream.ARENA_BYTES, GEMM_ARENA_BYTES)
assert REUSABLE_ARENA_BYTES == 229376
# The task->expert decode method (TaskDecode) is a compile-time attribute of the
# kernel; EXPERT_TABLE_CAP (contract) is the shared-memory prefix capacity of
# SMEM_BINARY (one entry per expert) and SMEM_GROUP (one entry per expert group).
# The expert count and the group stride stay runtime values.

# This device path supports CLC depth 1 and group size 1 only; a deeper CLC
# ring or task grouping needs its own response ring and task loop.
assert CLC_PIPE_DEPTH == 1
assert FUSED_GATE_UP_TASK_GROUP_SIZE == 1
assert FUSED_DOWN_TASK_GROUP_SIZE == 1

_GATE_CTA_M = FC1_CTA_M
_GATE_EPILOGUE_SUBTILES = _GATE_TILE_N // _GATE_EPILOGUE_N
_GATE_CLC_WARP = CLC_WARP
_GATE_BF16_IDLE_WARP = BF16_IDLE_WARP
# Every warp but the CLC warp consumes each task record: epilogue warps 0 .. MMA_WARP - 1, the MMA,
# load and idle warps.  A role added outside that set must be counted here or the pipeline hangs.
assert len({_GATE_MMA_WARP, _GATE_AB_LOAD_WARP, _GATE_BF16_IDLE_WARP, _GATE_CLC_WARP}) == 4
assert TASK_INFO_CONSUMER_WARPS + 1 == WARPS_PER_CTA
assert min(_GATE_AB_LOAD_WARP, _GATE_BF16_IDLE_WARP, _GATE_CLC_WARP) > _GATE_MMA_WARP
assert TASK_INFO_CONSUMER_WARPS == _GATE_MMA_WARP + 3
# Routed swap-AB tiles: the epilogue stores the accumulator transposed, one 32-token subtile (the
# epilogue N) at a time; each CTA's 128 A rows are its 128 Down features, or [Gate 64 | Up 64] of its
# 64 Gate/Up features.  The transposed staging tiles live in the D stages.
_SWAP_EPI_ROWS = _GATE_EPILOGUE_N
_SWAP_HALF = _GATE_CTA_M // 2
assert _SWAP_EPI_ROWS * _GATE_CTA_M * 2 * _GATE_D_STAGES == GEMM_D_BYTES
# epilogue_combine: the routed Down epilogue stores every finished row into the source rank's combine buffer.  A
# 32-token subtile is staged transposed into padded rows (token row = this CTA's 128 features = 256 B + 16 B pad:
# the 8 rows of an stmatrix 8x8 matrix hit 8 distinct 16 B bank groups), _GATE_D_STAGES of them after the A/B rings,
# then the tile's 256 row pointers (lo / hi Int32 pairs); 16 lanes store one row, 16 B each.  All inside the arena
# (the GEMM arena uses 221184 B).
_EPI_ROW_BYTES = _GATE_CTA_M * 2
_EPI_ROW_STRIDE = _EPI_ROW_BYTES + 16
_EPI_STAGE_BYTES = _SWAP_EPI_ROWS * _EPI_ROW_STRIDE
_EPI_STAGE_OFFSET = GEMM_A_BYTES + GEMM_B_BYTES
_EPI_PTR_OFFSET = _EPI_STAGE_OFFSET + _GATE_D_STAGES * _EPI_STAGE_BYTES
_EPI_CHUNKS_PER_ROW = _EPI_ROW_BYTES // 16
_EPI_ROWS_PER_PASS = (len(EPILOGUE_WARPS) * 32) // _EPI_CHUNKS_PER_ROW  # 8 rows per pass of the 128 epilogue threads
assert _SWAP_EPI_ROWS % _EPI_ROWS_PER_PASS == 0
assert _EPI_PTR_OFFSET + MLP_TILE_ROWS * 8 <= REUSABLE_ARENA_BYTES, (_EPI_PTR_OFFSET, REUSABLE_ARENA_BYTES)


@cute.struct
class _PersistentSharedStorage:
    # One depth-1 CLC full/empty pair plus one 16-byte response.
    clc_mbarriers: cute.struct.MemRange[cutlass.Int64, 2 * CLC_PIPE_DEPTH]
    clc_response: cute.struct.Align[
        cute.struct.MemRange[cutlass.Int32, CLC_PIPE_DEPTH * CLC_RESPONSE_BYTES // 4],
        16,
    ]
    # Once a live compute cluster receives a successful capacity-only CLC
    # response, one independent drain slot per warp consumes the remaining
    # no-op responses until hardware reports failure.  Each slot has a full
    # and empty mbarrier plus one 16-byte response.
    clc_drain_mbarriers: cute.struct.MemRange[cutlass.Int64, 2 * CLC_DRAIN_WARPS]
    clc_drain_responses: cute.struct.Align[
        cute.struct.MemRange[cutlass.Int32, CLC_DRAIN_WARPS * CLC_RESPONSE_BYTES // 4],
        16,
    ]
    # Decoded-task records (TASK_INFO_FIELDS) with their full/empty barrier pair per stage.
    task_info_mbarriers: cute.struct.MemRange[cutlass.Int64, 2 * TASK_INFO_STAGES]
    task_info: cute.struct.Align[
        cute.struct.MemRange[cutlass.Int32, TASK_INFO_STAGES * TASK_INFO_WORDS],
        16,
    ]
    # One word per warp: the sum of the record words a consumer warp loaded, stored before it releases
    # the record (_load_task_info).  Written only.
    task_info_taken: cute.struct.MemRange[cutlass.Int32, WARPS_PER_CTA]

    # A full/empty barrier pair per A/B stage and per accumulator stage.  The eight N=32 epilogue
    # subtiles are not an eight-stage output ring: they reuse three 128x32 BF16 D tiles.
    gemm_ab_mbarriers: cute.struct.MemRange[cutlass.Int64, 2 * _GATE_AB_STAGES]
    gemm_acc_mbarriers: cute.struct.MemRange[cutlass.Int64, 2 * _GATE_ACC_STAGES]
    tmem_dealloc_mbarrier: cutlass.Int64
    tmem_holding_buffer: cutlass.Int32

    # Communication CTAs: one transaction barrier per pool slot and the pool's scratch words
    # (slot bitmap, per-slot phases).  The slots alias the arena.
    comm_mbarriers: cute.struct.MemRange[cutlass.Int64, _fwd_dynamic_task_stream.MBARRIERS]
    comm_scratch: cute.struct.MemRange[cutlass.Int32, _fwd_dynamic_task_stream.SCRATCH_WORDS]
    # expert_cu_end[g] = routed row blocks (MLP_TILE_ROWS each) of local experts
    # 0 .. min((g + 1) * stride, E) - 1: inclusive prefix over groups of stride experts
    # (group 0 starts at 0, so no leading slot), g < expert_groups = ceil(E / stride)
    # <= EXPERT_TABLE_CAP.  SMEM_BINARY: stride 1 (one entry per expert, E <= cap);
    # SMEM_GROUP: stride = ceil(E / cap).  Unused by LINEAR_SCAN.  Every index is < expert_groups.
    expert_cu_end: cute.struct.MemRange[cutlass.Int32, EXPERT_TABLE_CAP]
    arena: cute.struct.Align[
        cute.struct.MemRange[cutlass.Uint8, REUSABLE_ARENA_BYTES],
        1024,
    ]


# Barriers and the expert_cu_end table fit in the first two kilobytes; the arena starts
# at the 2048 B boundary (231424 B total, under the 232448 B SM103 opt-in maximum).
DYNAMIC_SHARED_MEMORY_BYTES = 2048 + REUSABLE_ARENA_BYTES
assert _PersistentSharedStorage.size_in_bytes() == DYNAMIC_SHARED_MEMORY_BYTES
assert DYNAMIC_SHARED_MEMORY_BYTES <= 232448  # SM103 opt-in dynamic shared-memory maximum


def _storage_member_bytes(member) -> int:
    """Byte size of one _PersistentSharedStorage annotation (MemRange, Align[...] or scalar)."""
    if hasattr(member, "size_in_bytes") and not hasattr(member, "width"):
        return int(member.size_in_bytes)  # struct.MemRange
    if hasattr(member, "_dtype") and getattr(member, "_align", None) is not None:
        return _storage_member_bytes(member._dtype)  # struct.Align[inner, n]
    return int(member.width) // 8  # DSL scalar


def shared_storage_layout() -> tuple[tuple[str, int, int], ...]:
    """(name, byte offset, byte size) of every _PersistentSharedStorage member, in order."""
    offsets = _PersistentSharedStorage._offsets
    annotations = _PersistentSharedStorage._annotations
    return tuple((name, int(offsets[name]), _storage_member_bytes(annotations[name])) for name in annotations)


# The expert table sits after every barrier and before the 1024-aligned arena: a table
# resized by mistake cannot overlap the mbarriers, and the slack between the table end and
# the arena is padding.  (An out-of-bounds table write would land in that padding and no
# precision test would notice it; hence the explicit expert_groups bound in the kernel.)
_STORAGE_LAYOUT = {name: (offset, size) for name, offset, size in shared_storage_layout()}
_TABLE_OFFSET, _TABLE_BYTES = _STORAGE_LAYOUT["expert_cu_end"]
_ARENA_OFFSET = _STORAGE_LAYOUT["arena"][0]
_TASK_INFO_OFFSET, _TASK_INFO_BYTES = _STORAGE_LAYOUT["task_info"]
assert _TABLE_BYTES == EXPERT_TABLE_CAP * 4
assert _TASK_INFO_BYTES == TASK_INFO_STAGES * TASK_INFO_WORDS * 4 and _TASK_INFO_OFFSET % 16 == 0
assert _TASK_INFO_OFFSET + _TASK_INFO_BYTES <= _TABLE_OFFSET
for _name, (_offset, _size) in _STORAGE_LAYOUT.items():
    if _name.endswith("mbarrier") or _name.endswith("mbarriers"):
        assert _offset + _size <= _TABLE_OFFSET, (_name, _offset, _size, _TABLE_OFFSET)
assert _TABLE_OFFSET + _TABLE_BYTES <= _ARENA_OFFSET, (_TABLE_OFFSET, _TABLE_BYTES, _ARENA_OFFSET)
assert _ARENA_OFFSET % 1024 == 0 and _ARENA_OFFSET + REUSABLE_ARENA_BYTES == DYNAMIC_SHARED_MEMORY_BYTES


@cute.jit
def _row_blocks_i32(rows: Int32) -> Int32:
    """Row blocks of a scheduler row count: the counts are non-negative multiples of
    MLP_TILE_ROWS, so the division is done unsigned (floor = truncation, no sign fix-up;
    a power-of-two tile lowers to one shift)."""
    return Int32(Uint32(rows) // Uint32(MLP_TILE_ROWS))


@cute.jit
def _expert_tile_from_owner(
    owner: Int32,
    offset: Int32,
    expert_rows: Int32,
    first_row_block: Int32,
    mini_end_row: Int32,
    column_blocks: Int32,
    task_index: Int32,
) -> tuple[Int32, Int32, Int32, Int32, Int32]:
    """Tile of ``task_index`` inside the owner expert's block grid in this minibatch: the
    expert owns row blocks [max(first, offset), min(mini_end, offset + expert_rows)) and
    `rows * column_blocks` consecutive tasks, swizzled inside that grid.  Returns (valid, expert,
    global row block, column, the expert's rows before that block)."""
    first = cutlass.max(first_row_block, offset)
    rows = cutlass.min(mini_end_row, offset + expert_rows) - first
    local_task = task_index - (first - first_row_block) * column_blocks
    local_row, column = _swizzled_mlp_tile_coord_i32(rows, column_blocks, local_task)
    row_block = first + local_row
    return Int32(1), owner, row_block, column, (row_block - offset) * Int32(MLP_TILE_ROWS)


@cute.jit
def _decode_linear_scan(
    tokens_per_expert: cute.Tensor,
    expert_cu_end: cute.Tensor,
    expert_stride: Int32,
    expert_groups: Int32,
    first_row_block: Int32,
    minibatch_row_blocks: Int32,
    column_blocks: Int32,
    task_index: Int32,
) -> tuple[Int32, Int32, Int32, Int32, Int32]:
    """TaskDecode.LINEAR_SCAN: walk tokens_per_expert from expert 0 until the expert whose
    task range inside the minibatch holds ``task_index`` (only the expert, its row-block offset,
    the remaining task count and a found flag are loop-carried).  The host selector
    (select_task_decode) never picks it; it is the table-free reference decode, reached only by
    overriding that selector.  The shared table is not used."""
    num_experts = Int32(tokens_per_expert.shape[0])
    mini_end_row = first_row_block + minibatch_row_blocks
    valid = Int32(0)
    owner = Int32(-1)
    row = Int32(0)
    column = Int32(0)
    row_offset = Int32(0)
    expert = Int32(0)
    offset = Int32(0)
    remaining = task_index
    found = Int32(0)
    while expert < num_experts and found == Int32(0):
        expert_rows = _row_blocks_i32(tokens_per_expert[expert])
        first = cutlass.max(first_row_block, offset)
        rows = cutlass.max(Int32(0), cutlass.min(mini_end_row, offset + expert_rows) - first)
        tasks = rows * column_blocks
        if remaining < tasks:
            found = Int32(1)
        else:
            remaining = remaining - tasks
            offset = offset + expert_rows
            expert = expert + Int32(1)
    if found != Int32(0):
        expert_rows = _row_blocks_i32(tokens_per_expert[expert])
        first = cutlass.max(first_row_block, offset)
        rows = cutlass.min(mini_end_row, offset + expert_rows) - first
        local_row, column = _swizzled_mlp_tile_coord_i32(rows, column_blocks, remaining)
        valid = Int32(1)
        owner = expert
        row = first + local_row
        row_offset = (row - offset) * Int32(MLP_TILE_ROWS)
    return valid, owner, row, column, row_offset


@cute.jit
def _cu_end_upper_bound(expert_cu_end: cute.Tensor, entries: Int32, row_block: Int32) -> Int32:
    """First entry g in [0, entries) with ``cu_end[g] > row_block`` (the caller guarantees
    ``row_block < cu_end[entries - 1]``, so the result is < entries)."""
    lo = Int32(0)
    hi = entries
    while lo < hi:
        mid = (lo + hi) // Int32(2)
        if expert_cu_end[mid] > row_block:
            hi = mid
        else:
            lo = mid + Int32(1)
    return lo


@cute.jit
def _decode_smem_binary(
    tokens_per_expert: cute.Tensor,
    expert_cu_end: cute.Tensor,
    expert_stride: Int32,
    expert_groups: Int32,
    first_row_block: Int32,
    minibatch_row_blocks: Int32,
    column_blocks: Int32,
    task_index: Int32,
) -> tuple[Int32, Int32, Int32, Int32, Int32]:
    """TaskDecode.SMEM_BINARY: the owner is the first expert e with ``cu_end[e] >
    row_block`` over the shared inclusive prefix of all E experts (stride 1, expert_groups =
    E <= EXPERT_TABLE_CAP); its start is cu_end[e - 1] (0 for e = 0).  No global read.
    Selected by the host whenever E <= EXPERT_TABLE_CAP."""
    mini_end_row = first_row_block + minibatch_row_blocks
    valid = Int32(0)
    owner = Int32(-1)
    row = Int32(0)
    column = Int32(0)
    row_offset = Int32(0)
    row_block = first_row_block + task_index // column_blocks
    if row_block < mini_end_row:
        if row_block < expert_cu_end[expert_groups - Int32(1)]:
            expert = _cu_end_upper_bound(expert_cu_end, expert_groups, row_block)
            offset = Int32(0)
            if expert > Int32(0):
                offset = expert_cu_end[expert - Int32(1)]
            expert_rows = expert_cu_end[expert] - offset
            valid, owner, row, column, row_offset = _expert_tile_from_owner(
                expert, offset, expert_rows, first_row_block, mini_end_row, column_blocks, task_index
            )
    return valid, owner, row, column, row_offset


@cute.jit
def _decode_smem_group(
    tokens_per_expert: cute.Tensor,
    expert_cu_end: cute.Tensor,
    expert_stride: Int32,
    expert_groups: Int32,
    first_row_block: Int32,
    minibatch_row_blocks: Int32,
    column_blocks: Int32,
    task_index: Int32,
) -> tuple[Int32, Int32, Int32, Int32, Int32]:
    """TaskDecode.SMEM_GROUP: ``cu_end`` holds the inclusive prefix over groups of
    expert_stride = ceil(E / EXPERT_TABLE_CAP) experts (expert_groups = ceil(E / stride)
    entries; group g covers experts [g * stride, min((g + 1) * stride, E)), so the last
    group may be short).  The owner group is the first g with ``cu_end[g] > row_block``;
    the owner expert is found by walking the group's experts in global tokens_per_expert,
    at most stride - 1 reads, because the group's last expert ends at cu_end[g].
    Selected by the host when E > EXPERT_TABLE_CAP."""
    num_experts = Int32(tokens_per_expert.shape[0])
    mini_end_row = first_row_block + minibatch_row_blocks
    valid = Int32(0)
    owner = Int32(-1)
    row = Int32(0)
    column = Int32(0)
    row_offset = Int32(0)
    row_block = first_row_block + task_index // column_blocks
    if row_block < mini_end_row:
        if row_block < expert_cu_end[expert_groups - Int32(1)]:
            group = _cu_end_upper_bound(expert_cu_end, expert_groups, row_block)
            offset = Int32(0)
            if group > Int32(0):
                offset = expert_cu_end[group - Int32(1)]
            group_end = expert_cu_end[group]
            expert = group * expert_stride
            last_expert = cutlass.min(expert + expert_stride, num_experts) - Int32(1)
            expert_rows = Int32(0)
            found = Int32(0)
            while expert < last_expert and found == Int32(0):
                expert_rows = _row_blocks_i32(tokens_per_expert[expert])
                if row_block < offset + expert_rows:
                    found = Int32(1)
                else:
                    offset = offset + expert_rows
                    expert = expert + Int32(1)
            if found == Int32(0):
                expert_rows = group_end - offset  # the group's last expert (row_block < group_end)
            valid, owner, row, column, row_offset = _expert_tile_from_owner(
                expert, offset, expert_rows, first_row_block, mini_end_row, column_blocks, task_index
            )
    return valid, owner, row, column, row_offset


@cute.jit
def _routed_expert_tile_i32(
    tokens_per_expert: cute.Tensor,
    expert_cu_end: cute.Tensor,
    expert_stride: Int32,
    expert_groups: Int32,
    task_decode: cutlass.Constexpr,
    first_row_block: Int32,
    minibatch_row_blocks: Int32,
    column_blocks: Int32,
    task_index: Int32,
) -> tuple[Int32, Int32, Int32, Int32, Int32]:
    """Map a minibatch-local task index to (valid, expert, global row block, column, the expert's
    rows before that block) with the compile-time decode method (TaskDecode): exactly one of the
    three helpers is compiled into a kernel variant.  Tasks are expert-major inside the minibatch;
    see the helpers."""
    if cutlass.const_expr(task_decode == TaskDecode.LINEAR_SCAN):
        return _decode_linear_scan(
            tokens_per_expert,
            expert_cu_end,
            expert_stride,
            expert_groups,
            first_row_block,
            minibatch_row_blocks,
            column_blocks,
            task_index,
        )
    elif cutlass.const_expr(task_decode == TaskDecode.SMEM_BINARY):
        return _decode_smem_binary(
            tokens_per_expert,
            expert_cu_end,
            expert_stride,
            expert_groups,
            first_row_block,
            minibatch_row_blocks,
            column_blocks,
            task_index,
        )
    else:
        return _decode_smem_group(
            tokens_per_expert,
            expert_cu_end,
            expert_stride,
            expert_groups,
            first_row_block,
            minibatch_row_blocks,
            column_blocks,
            task_index,
        )


@cute.jit
def _build_expert_cu_end(
    tokens_per_expert: cute.Tensor,
    expert_cu_end: cute.Tensor,
    task_decode: cutlass.Constexpr,
    thread: Int32,
) -> tuple[Int32, Int32]:
    """Build the per-CTA shared ``cu_end`` table and return (expert_stride, expert_groups).
    SMEM_BINARY: stride 1, one entry per expert (the host admits E <= EXPERT_TABLE_CAP);
    SMEM_GROUP: stride = ceil(E / EXPERT_TABLE_CAP), expert_groups = ceil(E / stride) <=
    EXPERT_TABLE_CAP, group g summing experts [g * stride, min((g + 1) * stride, E)) only;
    LINEAR_SCAN: no table (returns (1, E)).  Every thread sums its groups from global
    tokens_per_expert (indices < expert_groups), then thread 0 turns the counts into the
    inclusive prefix: at most EXPERT_TABLE_CAP dependent shared-memory adds per CTA per
    launch.  Ends with a CTA barrier, so the table is visible to every role."""
    num_experts = Int32(tokens_per_expert.shape[0])
    expert_stride = Int32(1)
    expert_groups = num_experts
    if cutlass.const_expr(task_decode == TaskDecode.SMEM_GROUP):
        expert_stride = _ceil_div_i32(num_experts, Int32(EXPERT_TABLE_CAP))
        expert_groups = _ceil_div_i32(num_experts, expert_stride)
    if cutlass.const_expr(task_decode != TaskDecode.LINEAR_SCAN):
        group = thread
        while group < expert_groups:
            expert = group * expert_stride
            group_end = cutlass.min(expert + expert_stride, num_experts)
            blocks = Int32(0)
            while expert < group_end:
                blocks = blocks + _row_blocks_i32(tokens_per_expert[expert])
                expert = expert + Int32(1)
            expert_cu_end[group] = blocks
            group = group + Int32(THREADS_PER_CTA)
        cute.arch.sync_threads()
        if thread == Int32(0):
            index = Int32(1)
            while index < expert_groups:
                expert_cu_end[index] = expert_cu_end[index] + expert_cu_end[index - Int32(1)]
                index = index + Int32(1)
    cute.arch.sync_threads()
    return expert_stride, expert_groups


@cute.jit
def _routed_gate_up_tile_i32(
    tokens_per_expert: cute.Tensor,
    expert_cu_end: cute.Tensor,
    expert_stride: Int32,
    expert_groups: Int32,
    task_decode: cutlass.Constexpr,
    num_tokens: Int32,
    macrobatch_index: Int32,
    minibatch_index: Int32,
    task_index: Int32,
    shared_gate_up_tasks: Int32,
    macrobatch_size: Int32,
    minibatch_size: Int32,
    comm_tile_shape: cutlass.Constexpr,
    routed_comm_tile_columns: Int32,
    intermediate_size: Int32,
) -> tuple[Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32]:
    """Decode a routed Gate/Up expert tile and its readiness edge.

    The returned row coordinate is relative to the macrobatch ring buffer.
    ``valid`` is zero when this fixed-capacity task has no expert rows in the
    runtime minibatch.  The last value is the expert's rows before this block
    (the swap-AB tiles subtract it from the expert's valid count).
    """

    global_minibatch = macrobatch_index * Int32(macrobatch_size // minibatch_size) + minibatch_index
    minibatch_row_blocks = Int32(minibatch_size // MLP_TILE_ROWS)
    first_row_block = global_minibatch * minibatch_row_blocks
    column_blocks = intermediate_size // Int32(_GATE_LOGICAL_N)
    valid, expert_index, global_row_block, column_block, expert_row_offset = _routed_expert_tile_i32(
        tokens_per_expert,
        expert_cu_end,
        expert_stride,
        expert_groups,
        task_decode,
        first_row_block,
        minibatch_row_blocks,
        column_blocks,
        task_index,
    )

    macrobatch_row_block = global_row_block - (macrobatch_index * Int32(macrobatch_size // MLP_TILE_ROWS))
    # The fused epilogue publishes Hidden directly.  The Gate/Up-tile counter
    # stays in the launch arguments but is not consumed on this edge.
    shared_row_blocks = Int32(shared_gate_up_tasks) // column_blocks
    ready_index = shared_row_blocks + global_row_block
    minibatch_first_row = global_minibatch * Int32(minibatch_size)
    minibatch_rows = cutlass.min(
        Int32(minibatch_size),
        num_tokens - minibatch_first_row,
    )
    # x_routed_ready: one arrival per communication tile of the minibatch.
    dispatch_required = _ceil_div_i32(minibatch_rows, Int32(comm_tile_shape.rows)) * routed_comm_tile_columns
    return (
        valid,
        expert_index,
        macrobatch_row_block,
        column_block,
        ready_index,
        global_minibatch,
        dispatch_required,
        expert_row_offset,
    )


@cute.jit
def _routed_down_tile_i32(
    tokens_per_expert: cute.Tensor,
    expert_cu_end: cute.Tensor,
    expert_stride: Int32,
    expert_groups: Int32,
    task_decode: cutlass.Constexpr,
    num_tokens: Int32,
    macrobatch_index: Int32,
    minibatch_index: Int32,
    task_index: Int32,
    shared_row_blocks: Int32,
    macrobatch_size: Int32,
    minibatch_size: Int32,
    routed_column_blocks: Int32,
    intermediate_size: Int32,
) -> tuple[Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32]:
    """Decode a routed Down tile and its two readiness edges (last value: the expert's rows
    before this block, as in _routed_gate_up_tile_i32)."""

    global_minibatch = macrobatch_index * Int32(macrobatch_size // minibatch_size) + minibatch_index
    minibatch_row_blocks = Int32(minibatch_size // MLP_TILE_ROWS)
    first_row_block = global_minibatch * minibatch_row_blocks
    column_blocks = routed_column_blocks
    valid, expert_index, global_row_block, column_block, expert_row_offset = _routed_expert_tile_i32(
        tokens_per_expert,
        expert_cu_end,
        expert_stride,
        expert_groups,
        task_decode,
        first_row_block,
        minibatch_row_blocks,
        column_blocks,
        task_index,
    )

    macrobatch_row_block = global_row_block - (macrobatch_index * Int32(macrobatch_size // MLP_TILE_ROWS))
    hidden_ready_index = Int32(shared_row_blocks) + global_row_block
    hidden_ready_required = Int32((MLP_TILE_ROWS // SWIGLU_TILE_ROWS) * (intermediate_size // SWIGLU_TILE_COLUMNS))
    return (
        valid,
        expert_index,
        macrobatch_row_block,
        column_block,
        hidden_ready_index,
        hidden_ready_required,
        global_minibatch,
        expert_row_offset,
    )


@cute.jit
def _store_task_info(
    s_task_info: cute.Tensor,
    stage: Int32,
    active: Int32,
    needs_drain: Int32,
    kind: Int32,
    valid_tile: Int32,
    expert_index: Int32,
    tile_coord_m: Int32,
    tile_coord_n: Int32,
    input_ready_index: Int32,
    input_ready_required: Int32,
    output_ready_index: Int32,
    k_tile_count: Int32,
    expert_row_offset: Int32,
    macrobatch: Int32,
    tile_valid_rows: Int32,
) -> None:
    """Write one decoded-task record (TASK_INFO_FIELDS order) into stage ``stage``; one thread."""
    base = stage * Int32(TASK_INFO_WORDS)
    s_task_info[base + _TI["active"]] = active
    s_task_info[base + _TI["needs_drain"]] = needs_drain
    s_task_info[base + _TI["kind"]] = kind
    s_task_info[base + _TI["valid_tile"]] = valid_tile
    s_task_info[base + _TI["expert_index"]] = expert_index
    s_task_info[base + _TI["tile_coord_m"]] = tile_coord_m
    s_task_info[base + _TI["tile_coord_n"]] = tile_coord_n
    s_task_info[base + _TI["input_ready_index"]] = input_ready_index
    s_task_info[base + _TI["input_ready_required"]] = input_ready_required
    s_task_info[base + _TI["output_ready_index"]] = output_ready_index
    s_task_info[base + _TI["k_tile_count"]] = k_tile_count
    s_task_info[base + _TI["expert_row_offset"]] = expert_row_offset
    s_task_info[base + _TI["macrobatch"]] = macrobatch
    s_task_info[base + _TI["tile_valid_rows"]] = tile_valid_rows


@cute.jit
def _load_task_info(
    s_task_info: cute.Tensor, stage: Int32, s_task_taken: cute.Tensor, warp: Int32
) -> tuple[Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32]:
    """Read the record of stage ``stage`` (TASK_INFO_FIELDS order); whole warp.  Every word is
    marked warp-uniform: the values drive MMA/TMA descriptors and loop bounds, and a plain
    shared-memory load is not provably uniform to the compiler.

    The caller releases the stage right after this call and uses the values after the release; the
    CLC warp then rewrites the record.  A shared-memory load whose result no instruction has read
    when the release executes can return the next record's word, and the compiler moves the uses
    (the compares, the warp-uniform shuffles) behind a release that does not depend on them.  So
    the consumption is made explicit here: the sum of every loaded word is stored to this warp's
    word of ``s_task_taken``.  The store needs every load, and a shared-memory store stays in front
    of the mbarrier arrive that follows it (the order every producer relies on).  Nothing reads the
    stored word."""
    base = stage * Int32(TASK_INFO_WORDS)
    words = [s_task_info[base + index] for index in range(len(TASK_INFO_FIELDS))]
    s_task_taken[warp] = sum(words[1:], words[0])
    return tuple([cute.arch.make_warp_uniform(word) for word in words])


@cute.jit
def _aligned_bf16_matrix(tensor: cute.Tensor) -> cute.Tensor:
    """Preserve dynamic dimensions/64-bit strides, asserting validated TMA alignment."""
    if cutlass.const_expr(cute.rank(tensor) == 3):
        strides = (cute.assume(tensor.stride[0], divby=8), 1, cute.assume(tensor.stride[2], divby=8))
    else:
        strides = (cute.assume(tensor.stride[0], divby=8), 1)
    return cute.make_tensor(tensor.iterator, cute.make_layout(tensor.shape, stride=strides))


class _PersistentBf16FwdMega:
    """One-launch BF16 communication + fused-FC1 + Down megakernel.  The constructor arguments are
    compile-time variant flags: each selects code, and a kernel argument used by one variant only is
    ``None`` (not a kernel parameter) in the others."""

    def __init__(
        self,
        is_clamped: bool,
        task_decode: TaskDecode,
        epilogue_combine: bool = False,
        split_weights: bool = False,
        is_situ: bool = False,
        has_shared: bool = True,
        shared_fc1_only: bool = False,
        has_tail: bool = False,
        comm_tile_shape: CommTileShape = DEFAULT_COMM_TILE_SHAPE,
    ) -> None:
        # The communication tile shape (contract CommTileShape): the one compile-time parameter of the
        # communication CTAs, like the compute tile sizes.  It lays out their slots and selects the lanes that copy
        # and the bytes of one copy; every count derived from it (tile rows per minibatch, column blocks per row,
        # the arrivals the compute CTAs wait for) is computed from the runtime sizes.
        if comm_tile_shape not in COMM_TILE_SHAPES:
            raise ValueError(f"comm_tile_shape must be one of {COMM_TILE_SHAPES}")
        self.comm_tile_shape = comm_tile_shape
        # The clamp mode is compile-time; the clamp value stays a runtime Float32.
        self.is_clamped = bool(is_clamped)
        # Where the combine is pushed (CuteMoKConfig.epilogue_combine): False = the communication CTAs read the Y
        # ring and push it; True = the routed Down epilogue stores each finished row into the source rank's combine
        # buffer and the communication CTAs only pull.
        self.epilogue_combine = bool(epilogue_combine)
        # SiTU-GLU instead of SwiGLU, betas at runtime; never with a clamp.
        self.is_situ = bool(is_situ)
        if self.is_situ and self.is_clamped:
            raise ValueError("SiTU cannot be combined with swiglu_limit")
        # Task->expert decode method (contract TaskDecode).
        self.task_decode = TaskDecode(task_decode)
        # The cluster-2 FC1 collective: every CTA owns different M128 activation rows; CTA 0
        # supplies Gate N128 and CTA 1 supplies Up N128 to one 2-CTA N256 instruction.
        self.collective = _Fc1Collective()
        # The routed experts are independent per-expert allocations instead of one grouped
        # parameter.  The allocations agree on dtype, shape and stride, so one descriptor schema
        # fits them all and only the base address varies: the atoms are built over expert 0 and
        # routed_descriptors retargets a copy per expert.  Compile-time, because it changes the
        # operands' rank.
        self.split_weights = bool(split_weights)
        # False for a routed-only layer (no shared expert).  Its shared tensors are placeholders
        # the kernel builds atoms over and never touches -- there are no shared tasks.
        self.has_shared = bool(has_shared)
        # The launch holds the shared FC1 tasks alone (Gate | Up and the activation: gate_shared, up_shared,
        # hidden_shared): no communication cluster, no shared Down and no routed minibatch.  It recomputes the
        # shared activations of a forward whose context was not kept (persistent_bf16_fwd.shared_fc1_bf16); the
        # task code is the forward's, so are the results.  Every routed tensor, the Down weights and y_shared
        # are placeholders the kernel builds atoms over and no task touches; the schedule is one expert without
        # rows.
        self.shared_fc1_only = bool(shared_fc1_only)
        if self.shared_fc1_only and not self.has_shared:
            raise ValueError("shared_fc1_only needs a shared expert")
        # H has a 128-wide tail (H % 256 != 0): one more Down tile per row block.
        # Compile time because H is a model parameter and the flag only selects whether the tail code is
        # generated: an aligned model runs loops without predicates or tail branches.  It doubles this kernel's
        # executors (one for every aligned shape, one for every shape with a tail) as one more entry of the
        # executor key, which a model keeps for its whole run.  The value of H stays a runtime argument: nothing
        # is specialised on a size.
        self.has_tail = bool(has_tail)

    @cute.jit
    def __call__(
        self,
        x_peer_table: cute.Tensor,
        combine_peer_table: cute.Tensor,
        schedule_peer_rank: cute.Tensor,
        schedule_peer_token_idx: cute.Tensor,
        num_tokens: cute.Tensor,
        tokens_per_expert: cute.Tensor,
        valid_tokens_per_expert: cute.Tensor,
        x_shared: cute.Tensor,
        x_routed: cute.Tensor,
        shared_gate_weights: cute.Tensor,
        routed_gate_weights: cute.Tensor,
        shared_up_weights: cute.Tensor,
        routed_up_weights: cute.Tensor,
        shared_down_weights: cute.Tensor,
        routed_down_weights: cute.Tensor,
        gate_shared: cute.Tensor,
        gate_routed: cute.Tensor,
        up_shared: cute.Tensor,
        up_routed: cute.Tensor,
        hidden_shared: cute.Tensor,
        hidden_routed: cute.Tensor,
        y_shared: cute.Tensor,
        y_routed: cute.Tensor,
        gate_up_tile_ready: cute.Tensor,
        hidden_row_block_ready: cute.Tensor,
        x_routed_ready: cute.Tensor,
        y_routed_ready: cute.Tensor,
        y_routed_done: cute.Tensor,
        # (3, num_local_experts) int64 base addresses of the Gate / Up / Down allocations, and the
        # 3 * num_local_experts * 128 bytes their descriptors live in.  Both are one-element
        # placeholders unless split_weights is set.
        routed_storage_tables: cute.Tensor,
        routed_descriptors: cute.Tensor,
        # Nonzero when routed_descriptors has to be built by this launch: once per set of address
        # tables (persistent_bf16_fwd.RoutedDescriptors).
        build_descriptors: Int32,
        # The Up half's first row when Gate and Up are one combined FC1 parameter, 0 otherwise.
        # Gate and Up then name the same full tensor (and, split, the same per-expert
        # allocations), and Up is that offset down from Gate.
        up_row_offset: Int32,
        # SiTU-GLU's two betas; unread unless is_situ.
        situ_beta1: Float32,
        situ_beta2: Float32,
        num_local_tokens: Int32,
        schedule_capacity: Int32,
        macrobatch_size: Int32,
        minibatch_size: Int32,
        num_comm_sms: Int32,
        topk: Int32,
        swiglu_limit: Float32,
        stream: cuda.CUstream,
    ):
        # Task counts at CLC depth 1 and group size 1.  A CLC item is one fused M256xN128 FC1
        # tile or one M256xN256 Down tile.
        # The shared expert works in H_s (x_shared, y_shared) and the routed experts
        # in H_r (x_routed ring, y_routed, dispatch/combine rows); the two chains never meet inside
        # the kernel.  Both are runtime tensor extents.  Not to be confused with the SwiGLU
        # activation tensors hidden_shared / hidden_routed (width I).
        shared_hidden_size = Int32(x_shared.shape[1])
        routed_hidden_size = Int32(x_routed.shape[1])
        # I is read off the routed FC1 (minus the Up offset when Gate and Up are combined), which a
        # routed-only layer has too.
        intermediate_size = Int32(routed_gate_weights.shape[0]) - up_row_offset
        if cutlass.const_expr(self.shared_fc1_only):
            # The routed weights are placeholders: I is the shared Gate weight's rows.
            intermediate_size = Int32(shared_gate_weights.shape[0])
        shared_row_blocks = Int32(0)
        if cutlass.const_expr(self.has_shared):
            shared_row_blocks = num_local_tokens // MLP_TILE_ROWS
        minibatch_row_blocks = minibatch_size // MLP_TILE_ROWS
        shared_gate_up_raw_tasks = shared_row_blocks * (intermediate_size // _GATE_LOGICAL_N)
        # H may have a 128-wide tail (H % 256 != 0): it is one more Down tile, whose out-of-range
        # half the TMA loads zero-fill and the TMA stores clip.
        shared_down_raw_tasks = shared_row_blocks * _ceil_div_i32(shared_hidden_size, Int32(MLP_TILE_COLUMNS))
        minibatch_gate_up_raw_tasks = minibatch_row_blocks * (intermediate_size // _GATE_LOGICAL_N)
        minibatch_down_raw_tasks = minibatch_row_blocks * _ceil_div_i32(routed_hidden_size, Int32(MLP_TILE_COLUMNS))
        shared_gate_up_tasks = _ceil_div_i32(shared_gate_up_raw_tasks, Int32(FUSED_GATE_UP_TASK_GROUP_SIZE))
        shared_down_tasks = _ceil_div_i32(shared_down_raw_tasks, Int32(FUSED_DOWN_TASK_GROUP_SIZE))
        minibatch_gate_up_tasks = _ceil_div_i32(minibatch_gate_up_raw_tasks, Int32(FUSED_GATE_UP_TASK_GROUP_SIZE))
        minibatch_down_tasks = _ceil_div_i32(minibatch_down_raw_tasks, Int32(FUSED_DOWN_TASK_GROUP_SIZE))
        shared_tasks = shared_gate_up_tasks + shared_down_tasks
        minibatch_tasks = minibatch_gate_up_tasks + minibatch_down_tasks
        # Per-chain tile geometry, derived once per launch from the two runtime hidden extents (the
        # task decode then divides by no hidden size).
        shared_k_tiles = shared_hidden_size // Int32(_GATE_TILE_K)
        shared_column_blocks = _ceil_div_i32(shared_hidden_size, Int32(MLP_TILE_COLUMNS))
        routed_k_tiles = routed_hidden_size // Int32(_GATE_TILE_K)
        routed_column_blocks = _ceil_div_i32(routed_hidden_size, Int32(MLP_TILE_COLUMNS))
        routed_comm_tile_columns = _dynamic_task_stream.column_blocks(self.comm_tile_shape, routed_hidden_size)
        comm_clusters = num_comm_sms // CLUSTER_SIZE
        capacity_minibatches = _ceil_div_i32(schedule_capacity, minibatch_size)
        capacity_clusters = comm_clusters + shared_tasks + capacity_minibatches * minibatch_tasks
        if cutlass.const_expr(self.shared_fc1_only):
            # The task table ends after the shared FC1 tasks and the grid is exactly those clusters: no
            # communication cluster, no shared Down, no routed minibatch.  The host passes a routed token count
            # of zero, so the kernel's true cluster count is the grid: every cluster decodes a shared FC1 task
            # and the CLC scheduler runs dry without a capacity tail to drain.
            comm_clusters = Int32(0)
            shared_down_tasks = Int32(0)
            shared_tasks = shared_gate_up_tasks
            capacity_clusters = shared_tasks
        # TMA requires outer byte strides divisible by 16.  The host validates
        # contiguous BF16 matrices; no extent is made static.
        x_shared = _aligned_bf16_matrix(x_shared)
        x_routed = _aligned_bf16_matrix(x_routed)
        shared_gate_weights = _aligned_bf16_matrix(shared_gate_weights)
        routed_gate_weights = _aligned_bf16_matrix(routed_gate_weights)
        shared_up_weights = _aligned_bf16_matrix(shared_up_weights)
        routed_up_weights = _aligned_bf16_matrix(routed_up_weights)
        shared_down_weights = _aligned_bf16_matrix(shared_down_weights)
        routed_down_weights = _aligned_bf16_matrix(routed_down_weights)
        gate_shared = _aligned_bf16_matrix(gate_shared)
        gate_routed = _aligned_bf16_matrix(gate_routed)
        up_shared = _aligned_bf16_matrix(up_shared)
        up_routed = _aligned_bf16_matrix(up_routed)
        hidden_shared = _aligned_bf16_matrix(hidden_shared)
        hidden_routed = _aligned_bf16_matrix(hidden_routed)
        y_shared = _aligned_bf16_matrix(y_shared)
        y_routed = _aligned_bf16_matrix(y_routed)
        collective = self.collective
        # The collective is configured from one expert's matrix: the whole point of the split
        # layout is that every expert has the same shape and stride, and the grouped layout slices
        # the same thing out of the L mode.
        routed_gate_weights_0 = routed_gate_weights
        if cutlass.const_expr(not self.split_weights):
            routed_gate_weights_0 = routed_gate_weights[None, None, 0]
        collective.configure(x_routed, routed_gate_weights_0, gate_routed)
        a_smem_layout = cute.slice_(
            collective.a_smem_layout_staged,
            (None, None, None, 0),
        )
        b_smem_layout = cute.slice_(
            collective.b_smem_layout_staged,
            (None, None, None, 0),
        )
        a_op = cpasync.CopyBulkTensorTileG2SOp(collective.cta_group)
        b_op = cpasync.CopyBulkTensorTileG2SOp(collective.cta_group)
        (
            tma_atom_a_shared_gate_up,
            tma_tensor_a_shared_gate_up,
        ) = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            x_shared,
            a_smem_layout,
            collective.mma_tiler,
            collective.tiled_mma,
            collective.cluster_layout_vmnk.shape,
        )
        (
            tma_atom_b_shared_gate,
            tma_tensor_b_shared_gate,
        ) = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            shared_gate_weights,
            b_smem_layout,
            collective.mma_tiler,
            collective.tiled_mma,
            collective.cluster_layout_vmnk.shape,
        )
        (
            tma_atom_b_shared_up,
            tma_tensor_b_shared_up,
        ) = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            shared_up_weights,
            b_smem_layout,
            collective.mma_tiler,
            collective.tiled_mma,
            collective.cluster_layout_vmnk.shape,
        )
        (
            tma_atom_a_shared_down,
            tma_tensor_a_shared_down,
        ) = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            hidden_shared,
            a_smem_layout,
            collective.mma_tiler,
            collective.tiled_mma,
            collective.cluster_layout_vmnk.shape,
        )
        (
            tma_atom_b_shared_down,
            tma_tensor_b_shared_down,
        ) = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            shared_down_weights,
            b_smem_layout,
            collective.mma_tiler,
            collective.tiled_mma,
            collective.cluster_layout_vmnk.shape,
        )
        # Routed Gate/Up (swap-AB): operand A of each CTA is [Gate 64 rows | Up 64 rows] of its 64
        # features, loaded as two 64-row boxes from the two weight tensors into the halves of the A
        # stage (the stage's own 128 B swizzle; the Gate/Up interleave exists in shared memory only, so
        # the two tensors may lie anywhere); operand B is the token rows of x.
        #
        # Split allocations have no common L stride, so there is no E mode to slice: the host
        # passes the representative expert alone, two-dimensional, and the kernel names the expert
        # through its own descriptor.  Every expert has the same shape and stride, so the tile
        # coordinates are unchanged either way -- and the rank is what tells the two layouts apart,
        # so handing one to the other variant fails to compile instead of reading expert 0 twice.
        swap_half_layout = cute.make_composed_layout(a_smem_layout.inner, 0, cute.make_layout((_SWAP_HALF, _GATE_TILE_K), stride=(_GATE_TILE_K, 1)))
        gate_atom_source = routed_gate_weights
        up_atom_source = routed_up_weights
        down_atom_source = routed_down_weights
        tma_atom_routed_gate, tma_tensor_routed_gate = cpasync.make_tiled_tma_atom(a_op, gate_atom_source, swap_half_layout, (_SWAP_HALF, _GATE_TILE_K))
        tma_atom_routed_up, tma_tensor_routed_up = cpasync.make_tiled_tma_atom(a_op, up_atom_source, swap_half_layout, (_SWAP_HALF, _GATE_TILE_K))
        (
            tma_atom_b_routed_gate_up,
            tma_tensor_b_routed_gate_up,
        ) = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            x_routed,
            b_smem_layout,
            collective.mma_tiler,
            collective.tiled_mma,
            collective.cluster_layout_vmnk.shape,
        )
        # Routed Down (swap-AB): operand A is the expert's Down weights (M = H features), operand B
        # the token rows of Hidden.
        (
            tma_atom_a_routed_down,
            tma_tensor_a_routed_down,
        ) = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            down_atom_source,
            a_smem_layout,
            collective.mma_tiler,
            collective.tiled_mma,
            collective.cluster_layout_vmnk.shape,
        )
        (
            tma_atom_b_routed_down,
            tma_tensor_b_routed_down,
        ) = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            hidden_routed,
            b_smem_layout,
            collective.mma_tiler,
            collective.tiled_mma,
            collective.cluster_layout_vmnk.shape,
        )
        # Routed epilogue stores, transposed (tokens as rows): 32 tokens x 64 features of Hidden and of
        # the Gate / Up context; y as the M-major D of the swap-AB Down tile, (H features, rows) with the
        # features contiguous, 128 features x 32 tokens per store.
        swap_context_layout = cute.make_layout((_SWAP_EPI_ROWS, _SWAP_HALF), stride=(_SWAP_HALF, 1))
        y_routed_features_major = cute.make_tensor(y_routed.iterator, cute.select(y_routed.layout, mode=[1, 0]))
        tma_atom_d_routed_gate, tma_tensor_d_routed_gate = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), gate_routed, swap_context_layout, (_SWAP_EPI_ROWS, _SWAP_HALF)
        )
        tma_atom_d_routed_up, tma_tensor_d_routed_up = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), up_routed, swap_context_layout, (_SWAP_EPI_ROWS, _SWAP_HALF)
        )
        tma_atom_d_routed_hidden, tma_tensor_d_routed_hidden = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), hidden_routed, swap_context_layout, (_SWAP_EPI_ROWS, _SWAP_HALF)
        )
        tma_atom_d_routed_down, tma_tensor_d_routed_down = collective._make_tma_epi_atoms_and_tensors(
            y_routed_features_major,
            _swapab_fwd.d_smem_layout(collective.epi_tile, _GATE_D_STAGES),
            collective.epi_tile,
            op_type="store",
        )
        tma_atom_d_shared_gate, tma_tensor_d_shared_gate = collective._make_tma_epi_atoms_and_tensors(
            gate_shared,
            collective.epi_smem_layout_staged,
            collective.epi_tile,
            op_type="store",
        )
        tma_atom_d_shared_up, tma_tensor_d_shared_up = collective._make_tma_epi_atoms_and_tensors(
            up_shared,
            collective.epi_smem_layout_staged,
            collective.epi_tile,
            op_type="store",
        )
        tma_atom_d_shared_hidden, tma_tensor_d_shared_hidden = collective._make_tma_epi_atoms_and_tensors(
            hidden_shared,
            collective.epi_smem_layout_staged,
            collective.epi_tile,
            op_type="store",
        )
        tma_atom_d_shared_down, tma_tensor_d_shared_down = collective._make_tma_epi_atoms_and_tensors(
            y_shared,
            collective.epi_smem_layout_staged,
            collective.epi_tile,
            op_type="store",
        )
        collective.num_tma_load_bytes = (cute.size_in_bytes(BFloat16, a_smem_layout) + cute.size_in_bytes(BFloat16, b_smem_layout)) * cute.size(
            collective.tiled_mma.thr_id.shape
        )
        scheduler_params = utils.ClcDynamicPersistentTileSchedulerParams(
            (capacity_clusters * CLUSTER_SIZE, 1, 1),
            CLUSTER_SHAPE,
        )
        # One descriptor array per routed weight, filled by its own launch before the mega kernel:
        # that one is persistent, so every CTA would otherwise race to write the same descriptors.
        # The kernel takes the split and SiTU arguments only in those variants: None is a compile-time constant,
        # not a kernel parameter, so every other variant's kernel is the one without the feature.
        descriptors = None
        kernel_situ_beta1 = None
        kernel_situ_beta2 = None
        if cutlass.const_expr(self.is_situ):
            kernel_situ_beta1 = situ_beta1
            kernel_situ_beta2 = situ_beta2
        if cutlass.const_expr(self.split_weights):
            descriptors = cute.make_ptr(
                cutlass.Uint8, routed_descriptors.iterator.toint(), cute.AddressSpace.gmem, assumed_align=routed_descriptors_pkg.BYTES_PER_DESCRIPTOR
            )
            if build_descriptors != Int32(0):
                routed_descriptors_pkg.build_routed_descriptors(
                    (tma_atom_routed_gate, tma_atom_routed_up, tma_atom_a_routed_down),
                    routed_storage_tables,
                    descriptors,
                    Int32(cute.size(routed_storage_tables.shape[1])),
                    stream,
                )
        self.kernel(
            collective.tiled_mma,
            tma_atom_a_shared_gate_up,
            tma_tensor_a_shared_gate_up,
            tma_atom_b_shared_gate,
            tma_tensor_b_shared_gate,
            tma_atom_b_shared_up,
            tma_tensor_b_shared_up,
            tma_atom_a_shared_down,
            tma_tensor_a_shared_down,
            tma_atom_b_shared_down,
            tma_tensor_b_shared_down,
            tma_atom_d_shared_gate,
            tma_tensor_d_shared_gate,
            tma_atom_d_shared_up,
            tma_tensor_d_shared_up,
            tma_atom_d_shared_hidden,
            tma_tensor_d_shared_hidden,
            tma_atom_d_shared_down,
            tma_tensor_d_shared_down,
            tma_atom_routed_gate,
            tma_tensor_routed_gate,
            tma_atom_routed_up,
            tma_tensor_routed_up,
            tma_atom_b_routed_gate_up,
            tma_tensor_b_routed_gate_up,
            tma_atom_a_routed_down,
            tma_tensor_a_routed_down,
            tma_atom_b_routed_down,
            tma_tensor_b_routed_down,
            tma_atom_d_routed_gate,
            tma_tensor_d_routed_gate,
            tma_atom_d_routed_up,
            tma_tensor_d_routed_up,
            tma_atom_d_routed_hidden,
            tma_tensor_d_routed_hidden,
            tma_atom_d_routed_down,
            tma_tensor_d_routed_down,
            collective.cluster_layout_vmnk,
            collective.a_smem_layout_staged,
            collective.b_smem_layout_staged,
            collective.epi_smem_layout_staged,
            collective.epi_tile,
            x_peer_table,
            combine_peer_table,
            schedule_peer_rank,
            schedule_peer_token_idx,
            num_tokens,
            tokens_per_expert,
            valid_tokens_per_expert,
            x_routed,
            gate_up_tile_ready,
            hidden_row_block_ready,
            y_routed,
            x_routed_ready,
            y_routed_ready,
            y_routed_done,
            scheduler_params,
            comm_clusters,
            shared_gate_up_tasks,
            shared_down_tasks,
            shared_tasks,
            shared_row_blocks,
            minibatch_gate_up_tasks,
            minibatch_down_tasks,
            minibatch_tasks,
            num_local_tokens,
            macrobatch_size,
            minibatch_size,
            num_comm_sms,
            topk,
            shared_k_tiles,
            shared_column_blocks,
            routed_k_tiles,
            routed_column_blocks,
            routed_comm_tile_columns,
            intermediate_size,
            swiglu_limit,
            up_row_offset,
            descriptors,
            kernel_situ_beta1,
            kernel_situ_beta2,
        ).launch(
            grid=(capacity_clusters * CLUSTER_SIZE, 1, 1),
            block=(THREADS_PER_CTA, 1, 1),
            cluster=CLUSTER_SHAPE,
            smem=_PersistentSharedStorage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.jit
    def _decode_compute_task(
        self,
        compute_cluster: Int32,
        num_tokens: Int32,
        shared_gate_up_tasks: Int32,
        shared_down_tasks: Int32,
        shared_tasks: Int32,
        minibatch_gate_up_tasks: Int32,
        minibatch_down_tasks: Int32,
        minibatch_tasks: Int32,
        macrobatch_size: Int32,
        minibatch_size: Int32,
    ) -> tuple[Int32, Int32, Int32, Int32]:
        """Map a compute cluster index to (kind, task, macrobatch, minibatch) with one staged return.

        Kinds are ``0=shared FC1``, ``1=shared Down``, ``2=routed FC1`` and
        ``3=routed Down``.  SwiGLU is an FC1 epilogue operation, not a task.
        Shared tasks come first, then the routed minibatches in the order of the
        communication task stream (below); ``task`` is the index inside the kind.
        """

        kind = Int32(0)
        task = compute_cluster
        macrobatch = Int32(-1)
        minibatch = Int32(-1)
        if compute_cluster >= Int32(shared_gate_up_tasks):
            kind = Int32(1)
            task = compute_cluster - Int32(shared_gate_up_tasks)
            if compute_cluster >= Int32(shared_tasks):
                routed = compute_cluster - Int32(shared_tasks)
                ordered_minibatch = routed // Int32(minibatch_tasks)
                minibatch_task = routed - ordered_minibatch * Int32(minibatch_tasks)
                num_macrobatches = _ceil_div_i32(num_tokens, Int32(macrobatch_size))
                true_minibatches = _ceil_div_i32(num_tokens, Int32(minibatch_size))
                minibatches_per_macro = Int32(macrobatch_size // minibatch_size)
                last_macro_minibatches = true_minibatches - (num_macrobatches - Int32(1)) * minibatches_per_macro
                # With several macrobatches the minibatches of a macrobatch go from the last to the first: the
                # communication stream pairs the pushes of (m, j) with the pulls of (m - 1, j) on the same ring
                # position, and the last minibatch's ring position is the first one the Down frees
                # (_fwd_dynamic_task_stream).  One macrobatch keeps the natural order.
                if ordered_minibatch < last_macro_minibatches:
                    macrobatch = num_macrobatches - Int32(1)
                    minibatch = ordered_minibatch
                    if num_macrobatches > Int32(1):
                        minibatch = last_macro_minibatches - Int32(1) - ordered_minibatch
                else:
                    index = ordered_minibatch - last_macro_minibatches
                    macrobatch = num_macrobatches - Int32(2) - index // minibatches_per_macro
                    minibatch = minibatches_per_macro - Int32(1) - index % minibatches_per_macro

                kind = Int32(2)
                task = minibatch_task
                if task >= Int32(minibatch_gate_up_tasks):
                    task = task - Int32(minibatch_gate_up_tasks)
                    kind = Int32(3)
        return kind, task, macrobatch, minibatch

    @cute.jit
    def _decode_routed_endpoint_gemm_tile(
        self,
        kind: Int32,
        task: Int32,
        macrobatch: Int32,
        minibatch: Int32,
        tokens_per_expert: cute.Tensor,
        expert_cu_end: cute.Tensor,
        expert_stride: Int32,
        expert_groups: Int32,
        task_decode: cutlass.Constexpr,
        num_tokens: Int32,
        shared_gate_up_tasks: Int32,
        shared_row_blocks: Int32,
        macrobatch_size: Int32,
        minibatch_size: Int32,
        shared_k_tiles: Int32,
        shared_column_blocks: Int32,
        routed_k_tiles: Int32,
        routed_column_blocks: Int32,
        routed_comm_tile_columns: Int32,
        intermediate_size: Int32,
    ) -> tuple[
        Int32,
        Int32,
        Int32,
        Int32,
        Int32,
        Int32,
        Int32,
        Int32,
        Int32,
    ]:
        """Decode a shared/routed fused-FC1 or Down tile with one return.  The last value is the
        owner expert's rows before the tile's row block (routed tiles; 0 for the shared expert)."""

        valid_tile = Int32(0)
        expert_index = Int32(-1)
        tile_coord_m = Int32(0)
        tile_coord_n = Int32(0)
        input_ready_index = Int32(0)
        input_ready_required = Int32(0)
        output_ready_index = Int32(0)
        k_tile_count = Int32(0)
        expert_row_offset = Int32(0)
        if kind == Int32(0):
            tile_coord_m, tile_coord_n = _swizzled_mlp_tile_coord_i32(
                Int32(shared_row_blocks),
                Int32(intermediate_size // _GATE_LOGICAL_N),
                task,
            )
            valid_tile = Int32(1)
            expert_index = Int32(0)
            output_ready_index = tile_coord_m
            k_tile_count = shared_k_tiles
        elif kind == Int32(1):
            tile_coord_m, tile_coord_n = _swizzled_mlp_tile_coord_i32(
                Int32(shared_row_blocks),
                shared_column_blocks,
                task,
            )
            valid_tile = Int32(1)
            expert_index = Int32(0)
            input_ready_index = tile_coord_m
            input_ready_required = Int32((MLP_TILE_ROWS // SWIGLU_TILE_ROWS) * (intermediate_size // SWIGLU_TILE_COLUMNS))
            k_tile_count = intermediate_size // Int32(_GATE_TILE_K)
        elif kind == Int32(2):
            (
                valid_tile,
                expert_index,
                tile_coord_m,
                tile_coord_n,
                output_ready_index,
                input_ready_index,
                input_ready_required,
                expert_row_offset,
            ) = _routed_gate_up_tile_i32(
                tokens_per_expert,
                expert_cu_end,
                expert_stride,
                expert_groups,
                task_decode,
                num_tokens,
                macrobatch,
                minibatch,
                task,
                shared_gate_up_tasks,
                macrobatch_size,
                minibatch_size,
                self.comm_tile_shape,
                routed_comm_tile_columns,
                intermediate_size,
            )
            k_tile_count = routed_k_tiles
        elif kind == Int32(3):
            (
                valid_tile,
                expert_index,
                tile_coord_m,
                tile_coord_n,
                input_ready_index,
                input_ready_required,
                output_ready_index,
                expert_row_offset,
            ) = _routed_down_tile_i32(
                tokens_per_expert,
                expert_cu_end,
                expert_stride,
                expert_groups,
                task_decode,
                num_tokens,
                macrobatch,
                minibatch,
                task,
                shared_row_blocks,
                macrobatch_size,
                minibatch_size,
                routed_column_blocks,
                intermediate_size,
            )
            k_tile_count = intermediate_size // Int32(_GATE_TILE_K)
        return (
            valid_tile,
            expert_index,
            tile_coord_m,
            tile_coord_n,
            input_ready_index,
            input_ready_required,
            output_ready_index,
            k_tile_count,
            expert_row_offset,
        )

    @cute.jit
    def _decode_task_record(
        self,
        work_valid: Int32,
        logical_cluster: Int32,
        true_clusters: Int32,
        comm_clusters: Int32,
        tokens_per_expert: cute.Tensor,
        expert_cu_end: cute.Tensor,
        expert_stride: Int32,
        expert_groups: Int32,
        valid_tokens_per_expert: cute.Tensor,
        num_tokens: Int32,
        shared_gate_up_tasks: Int32,
        shared_down_tasks: Int32,
        shared_tasks: Int32,
        shared_row_blocks: Int32,
        minibatch_gate_up_tasks: Int32,
        minibatch_down_tasks: Int32,
        minibatch_tasks: Int32,
        macrobatch_size: Int32,
        minibatch_size: Int32,
        shared_k_tiles: Int32,
        shared_column_blocks: Int32,
        routed_k_tiles: Int32,
        routed_column_blocks: Int32,
        routed_comm_tile_columns: Int32,
        intermediate_size: Int32,
    ) -> tuple[Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32]:
        """Decode one CLC work item into the task-info record (TASK_INFO_FIELDS order).

        ``active`` is 1 for a valid response naming a true compute cluster; a valid response past
        true_clusters is the capacity tail (``needs_drain`` = 1, the roles stop and drain) and an
        invalid response stops the roles without a drain.  Only an active record is decoded; a stop
        record has valid_tile = 0 (which skips every role body), expert_index = macrobatch = -1 and
        the other fields 0.  The block's valid
        rows (swap-AB N) are read from valid_tokens_per_expert here, once per CTA per task.
        """

        active = Int32(0)
        needs_drain = Int32(0)
        kind = Int32(0)
        valid_tile = Int32(0)
        expert_index = Int32(-1)
        tile_coord_m = Int32(0)
        tile_coord_n = Int32(0)
        input_ready_index = Int32(0)
        input_ready_required = Int32(0)
        output_ready_index = Int32(0)
        k_tile_count = Int32(0)
        expert_row_offset = Int32(0)
        macrobatch = Int32(-1)
        tile_valid_rows = Int32(0)
        if work_valid != Int32(0):
            if logical_cluster < true_clusters:
                active = Int32(1)
            else:
                needs_drain = Int32(1)
        if active != Int32(0):
            (
                kind,
                task,
                macrobatch,
                minibatch,
            ) = self._decode_compute_task(
                logical_cluster - Int32(comm_clusters),
                num_tokens,
                shared_gate_up_tasks,
                shared_down_tasks,
                shared_tasks,
                minibatch_gate_up_tasks,
                minibatch_down_tasks,
                minibatch_tasks,
                macrobatch_size,
                minibatch_size,
            )
            (
                valid_tile,
                expert_index,
                tile_coord_m,
                tile_coord_n,
                input_ready_index,
                input_ready_required,
                output_ready_index,
                k_tile_count,
                expert_row_offset,
            ) = self._decode_routed_endpoint_gemm_tile(
                kind,
                task,
                macrobatch,
                minibatch,
                tokens_per_expert,
                expert_cu_end,
                expert_stride,
                expert_groups,
                self.task_decode,
                num_tokens,
                shared_gate_up_tasks,
                shared_row_blocks,
                macrobatch_size,
                minibatch_size,
                shared_k_tiles,
                shared_column_blocks,
                routed_k_tiles,
                routed_column_blocks,
                routed_comm_tile_columns,
                intermediate_size,
            )
            if valid_tile != Int32(0):
                tile_valid_rows = _swapab_fwd.tile_valid_rows(valid_tokens_per_expert[expert_index], expert_row_offset)
        return (
            active,
            needs_drain,
            kind,
            valid_tile,
            expert_index,
            tile_coord_m,
            tile_coord_n,
            input_ready_index,
            input_ready_required,
            output_ready_index,
            k_tile_count,
            expert_row_offset,
            macrobatch,
            tile_valid_rows,
        )

    @cute.jit
    def _routed_down_push_epilogue(
        self,
        thread: Int32,
        cta_rank: Int32,
        is_tma_warp: Boolean,
        macrobatch: Int32,
        macrobatch_size: Int32,
        tile_coord_m: Int32,
        tile_coord_n: Int32,
        output_ready_index: Int32,
        tile_n: Int32,
        epi_tile_layout,
        swap_copy_t2r,
        swap_copy_r2s,
        swap_tTR_tAcc: cute.Tensor,
        swap_tTR_rAcc: cute.Tensor,
        swap_tRS_rD: cute.Tensor,
        swap_rD_bf16: cute.Tensor,
        epi_tRS_sD: cute.Tensor,
        epi_stage_base,
        epi_row_ptrs: cute.Tensor,
        routed_row_bytes: Int64,
        acc_pipeline,
        acc_consumer_state,
        schedule_peer_rank: cute.Tensor,
        schedule_peer_token_idx: cute.Tensor,
        combine_peer_table: cute.Tensor,
        y_routed_ready: cute.Tensor,
    ) -> None:
        """Routed Down epilogue that pushes the combine (``epilogue_combine``), swap-AB.  TMEM lanes are this CTA's 128
        features of the Down N tile, the columns the block's 256 tokens.  The tile's 256 row pointers -- the source
        rank's combine row at the route index, this CTA's 128-feature column -- go to shared memory first (0 for a
        padding row: not sent); every 32-token subtile below the tile's N is staged transposed (token rows of 128
        contiguous features, an stmatrix copy into padded rows) and each staged row is stored to the peer by 16 lanes
        x 16 B ``st.global.cg``.  These are the unweighted rows the communication CTAs would push, so the output is
        unchanged.  A CTA whose features lie past H (the second CTA of a 128-wide tail tile, ``has_tail``) sends
        nothing.  No Y ring store, no y_routed_done wait.  y_routed_ready (the X-ring reuse gate of the next
        macrobatch's pull) is arrived on entry, which is correct only because the caller has already passed
        ``acc_pipeline.consumer_wait`` for this tile (before its branch on the task kind): the accumulator is complete,
        so the tile's MMAs have read every Hidden row, and the FC1 that wrote them read the X-ring rows.  A caller that
        enters before that wait would release the X ring early.  The kernel-end barrier orders the peer stores before
        the combine is read."""
        if is_tma_warp:
            with cute.arch.elect_one():
                _counter_arrive_gpu(y_routed_ready, output_ready_index, Int32(1))
        # The staging rows alias the D stages: the TMA warp's outstanding bulk stores of the previous tile (FC1 Gate /
        # Up / Hidden, shared Down) must have read them first.
        if is_tma_warp:
            cute.arch.cp_async_bulk_wait_group(0, read=True)
        # the previous tile's threads may still read the pointer table / staging rows
        named_barrier_sync_unaligned(self.collective.epilogue_barrier)
        for half in cutlass.range_constexpr(2):  # thread t resolves rows t and t + 128
            ptr_row = thread + Int32(half * (len(EPILOGUE_WARPS) * 32))
            ptr_schedule_row = macrobatch * macrobatch_size + tile_coord_m * Int32(MLP_TILE_ROWS) + ptr_row
            ptr_peer_rank = schedule_peer_rank[ptr_schedule_row]
            ptr_destination = Int64(0)
            column_bytes = Int64(tile_coord_n * Int32(MLP_TILE_COLUMNS) + cta_rank * Int32(_GATE_CTA_M)) * Int64(2)
            if ptr_peer_rank >= Int32(0):
                ptr_destination = (
                    _peer_address(combine_peer_table, ptr_peer_rank) + Int64(schedule_peer_token_idx[ptr_schedule_row]) * routed_row_bytes + column_bytes
                )
            if cutlass.const_expr(self.has_tail):
                # The tail tile (H % 256 == 128) has valid features in CTA 0 only.  CTA 1's 128 features lie past H:
                # the TMA store of the default variant clips them, but a plain global store here would land in the
                # first 128 columns of the next combine row (past the buffer for the last row), racing with that
                # row's own stores.  Such a CTA sends nothing.
                if column_bytes >= routed_row_bytes:
                    ptr_destination = Int64(0)
            epi_row_ptrs[2 * ptr_row] = Int32(ptr_destination & Int64(0xFFFFFFFF))
            epi_row_ptrs[2 * ptr_row + Int32(1)] = Int32(ptr_destination >> Int64(32))
        named_barrier_sync_unaligned(self.collective.epilogue_barrier)
        chunk = thread % Int32(_EPI_CHUNKS_PER_ROW)
        for epi_idx in cutlass.range_constexpr(_GATE_EPILOGUE_SUBTILES):
            epi_coord = epi_tile_layout.get_hier_coord(epi_idx)
            self.collective.epi_load_acc_subtile(
                swap_copy_t2r,
                swap_copy_r2s,
                swap_tTR_tAcc,
                swap_tTR_rAcc,
                swap_tRS_rD,
                epi_coord,
                acc_pipeline,
                acc_consumer_state,
                _GATE_EPILOGUE_SUBTILES - 1,
            )
            if Int32(epi_idx * _SWAP_EPI_ROWS) < tile_n:
                epi_buffer = epi_idx % _GATE_D_STAGES
                swap_rD_bf16.store(swap_tRS_rD.load().to(BFloat16))
                named_barrier_sync_unaligned(self.collective.epilogue_barrier)
                cute.copy(swap_copy_r2s, swap_rD_bf16, epi_tRS_sD[None, None, None, epi_buffer])
                named_barrier_sync_unaligned(self.collective.epilogue_barrier)
                epi_stage = epi_stage_base + Int32(epi_buffer * _EPI_STAGE_BYTES)
                for row_pass in cutlass.range_constexpr(_SWAP_EPI_ROWS // _EPI_ROWS_PER_PASS):
                    local_row = thread // Int32(_EPI_CHUNKS_PER_ROW) + Int32(row_pass * _EPI_ROWS_PER_PASS)
                    send_row = Int32(epi_idx * _SWAP_EPI_ROWS) + local_row
                    send_destination = (Int64(epi_row_ptrs[2 * send_row + Int32(1)]) << Int64(32)) | (Int64(epi_row_ptrs[2 * send_row]) & Int64(0xFFFFFFFF))
                    if send_destination != Int64(0):
                        value = cute.arch.load(
                            cute.recast_ptr((epi_stage + local_row * Int32(_EPI_ROW_STRIDE) + chunk * Int32(16)).align(min_align=16), dtype=Uint32),
                            ir.VectorType.get([4], Uint32.mlir_type),
                            ss="cta",
                        )
                        cute.arch.store(
                            cute.make_ptr(Uint32, send_destination + Int64(chunk) * Int64(16), cute.AddressSpace.gmem, assumed_align=16), value, cop="cg"
                        )

    @cute.jit
    def _drain_clc_capacity_tail(
        self,
        needs_drain: Int32,
        clc_drain_pipeline,
        drain_responses,
        warp: Int32,
        cta_rank: Int32,
    ) -> None:
        """Drain capacity-only CLC responses after every role finishes.

        Lane 0 of each warp works on its own stage: each CTA arms its own full
        barrier, rank 0 issues the CLC query, both CTAs wait for the multicast
        response and test only its success bit, then release rank 0's empty
        barrier, which rank 0 waits before re-issuing.  The waits poll with a
        short nanosleep backoff.  The barrier is armed before the query is
        issued; the arming arrival is pending until it lands, so the phase
        cannot complete on the response bytes alone.
        """

        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        if needs_drain != Int32(0):
            if cute.arch.lane_idx() == Int32(0):
                full_bar = clc_drain_pipeline.sync_object_full.get_barrier(warp)
                empty_bar = clc_drain_pipeline.sync_object_empty.get_barrier(warp)
                response = (drain_responses + warp * Int32(CLC_RESPONSE_BYTES // 4)).align(min_align=CLC_RESPONSE_BYTES)
                full_phase = Int32(0)
                empty_phase = Int32(0)
                cute.arch.mbarrier_arrive_and_expect_tx(full_bar, Int32(CLC_RESPONSE_BYTES))
                if cta_rank == Int32(0):
                    cute.arch.issue_clc_query(full_bar, response)
                active = Int32(1)
                while active != Int32(0):
                    _mbar_backoff_wait(full_bar, full_phase)
                    full_phase = full_phase ^ Int32(1)
                    _, _, _, drained_valid = cute.arch.clc_response(response)
                    # The generic-proxy read of the response must be ordered before the
                    # next async-proxy CLC write into it.
                    cute.arch.fence_proxy("async.shared", space="cta")
                    clc_drain_pipeline.sync_object_empty.arrive(warp, clc_drain_pipeline.consumer_mask)
                    if cta_rank == Int32(0):
                        _mbar_backoff_wait(empty_bar, empty_phase)
                        empty_phase = empty_phase ^ Int32(1)
                    active = Int32(drained_valid)
                    if active != Int32(0):
                        cute.arch.mbarrier_arrive_and_expect_tx(full_bar, Int32(CLC_RESPONSE_BYTES))
                        if cta_rank == Int32(0):
                            cute.arch.issue_clc_query(full_bar, response)
        cute.arch.sync_warp()

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a_shared_gate_up: cute.CopyAtom,
        x_shared_tma: cute.Tensor,
        tma_atom_b_shared_gate: cute.CopyAtom,
        shared_gate_weights: cute.Tensor,
        tma_atom_b_shared_up: cute.CopyAtom,
        shared_up_weights: cute.Tensor,
        tma_atom_a_shared_down: cute.CopyAtom,
        hidden_shared_tma: cute.Tensor,
        tma_atom_b_shared_down: cute.CopyAtom,
        shared_down_weights: cute.Tensor,
        tma_atom_d_shared_gate: cute.CopyAtom,
        gate_shared: cute.Tensor,
        tma_atom_d_shared_up: cute.CopyAtom,
        up_shared: cute.Tensor,
        tma_atom_d_shared_hidden: cute.CopyAtom,
        hidden_shared: cute.Tensor,
        tma_atom_d_shared_down: cute.CopyAtom,
        y_shared_tma: cute.Tensor,
        tma_atom_routed_gate: cute.CopyAtom,
        routed_gate_weights: cute.Tensor,
        tma_atom_routed_up: cute.CopyAtom,
        routed_up_weights: cute.Tensor,
        tma_atom_b_routed_gate_up: cute.CopyAtom,
        x_routed_tma: cute.Tensor,
        tma_atom_a_routed_down: cute.CopyAtom,
        routed_down_weights: cute.Tensor,
        tma_atom_b_routed_down: cute.CopyAtom,
        hidden_routed_tma: cute.Tensor,
        tma_atom_d_routed_gate: cute.CopyAtom,
        gate_routed: cute.Tensor,
        tma_atom_d_routed_up: cute.CopyAtom,
        up_routed: cute.Tensor,
        tma_atom_d_routed_hidden: cute.CopyAtom,
        hidden_routed: cute.Tensor,
        tma_atom_d_routed_down: cute.CopyAtom,
        y_routed_tma: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
        epi_smem_layout: cute.ComposedLayout,
        epi_tile: cute.Tile,
        x_peer_table: cute.Tensor,
        combine_peer_table: cute.Tensor,
        schedule_peer_rank: cute.Tensor,
        schedule_peer_token_idx: cute.Tensor,
        num_tokens_tensor: cute.Tensor,
        tokens_per_expert: cute.Tensor,
        valid_tokens_per_expert: cute.Tensor,
        x_routed: cute.Tensor,
        gate_up_tile_ready: cute.Tensor,
        hidden_row_block_ready: cute.Tensor,
        y_routed: cute.Tensor,
        x_routed_ready: cute.Tensor,
        y_routed_ready: cute.Tensor,
        y_routed_done: cute.Tensor,
        scheduler_params: utils.ClcDynamicPersistentTileSchedulerParams,
        comm_clusters: Int32,
        shared_gate_up_tasks: Int32,
        shared_down_tasks: Int32,
        shared_tasks: Int32,
        shared_row_blocks: Int32,
        minibatch_gate_up_tasks: Int32,
        minibatch_down_tasks: Int32,
        minibatch_tasks: Int32,
        num_local_tokens: Int32,
        macrobatch_size: Int32,
        minibatch_size: Int32,
        num_comm_sms: Int32,
        topk: Int32,
        shared_k_tiles: Int32,
        shared_column_blocks: Int32,
        routed_k_tiles: Int32,
        routed_column_blocks: Int32,
        routed_comm_tile_columns: Int32,
        intermediate_size: Int32,
        swiglu_limit: Float32,
        # Where Up starts in the tensor passed as the routed Up weights: the rows of Gate when Gate and Up are
        # one combined FC1 parameter, 0 when they are separate weights.
        up_row_offset: Int32,
        # Variant-only arguments, None in every other variant -- hence no annotation: the DSL casts an
        # annotated argument, and None is what keeps them out of the kernel's parameters.
        # routed_descriptors (split_weights): the per-expert TMA descriptors of the routed weights, one
        # array per operand in the Gate / Up / Down order of the address table.
        # situ_beta1 / situ_beta2 (is_situ): Float32.
        routed_descriptors,
        situ_beta1,
        situ_beta2,
    ):
        thread = cute.arch.thread_idx()[0]
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        cluster_index = cute.arch.make_warp_uniform(cute.arch.cluster_idx()[0])
        # Every warp prefetches the TMA descriptors here, before the role dispatch.  The descriptor
        # addresses are then formed once, ahead of any branch, and stay in uniform registers for every TMA
        # the roles issue; prefetched from inside the load role, CuTe DSL 4.8 kept them in vector registers
        # and rebuilt each with R2UR in front of every TMA of the k loop.  A prefetch is a hint: issuing it
        # from every warp costs one instruction per descriptor per warp, once per CTA.
        cpasync.prefetch_descriptor(tma_atom_a_shared_gate_up)
        cpasync.prefetch_descriptor(tma_atom_b_shared_gate)
        cpasync.prefetch_descriptor(tma_atom_b_shared_up)
        cpasync.prefetch_descriptor(tma_atom_a_shared_down)
        cpasync.prefetch_descriptor(tma_atom_b_shared_down)
        cpasync.prefetch_descriptor(tma_atom_d_shared_gate)
        cpasync.prefetch_descriptor(tma_atom_d_shared_up)
        cpasync.prefetch_descriptor(tma_atom_d_shared_hidden)
        cpasync.prefetch_descriptor(tma_atom_d_shared_down)
        cpasync.prefetch_descriptor(tma_atom_routed_gate)
        cpasync.prefetch_descriptor(tma_atom_routed_up)
        cpasync.prefetch_descriptor(tma_atom_b_routed_gate_up)
        cpasync.prefetch_descriptor(tma_atom_a_routed_down)
        cpasync.prefetch_descriptor(tma_atom_b_routed_down)
        cpasync.prefetch_descriptor(tma_atom_d_routed_gate)
        cpasync.prefetch_descriptor(tma_atom_d_routed_up)
        cpasync.prefetch_descriptor(tma_atom_d_routed_hidden)
        cpasync.prefetch_descriptor(tma_atom_d_routed_down)
        num_tokens = num_tokens_tensor[0]
        # CuTe DSL staged conditionals cannot contain early returns, so capacity
        # clusters are masked below and reclaimed by the explicit CLC drain.
        true_minibatches = _ceil_div_i32(num_tokens, Int32(minibatch_size))
        true_clusters = Int32(comm_clusters) + Int32(shared_tasks) + true_minibatches * Int32(minibatch_tasks)

        smem = utils.SmemAllocator()
        storage = smem.allocate(_PersistentSharedStorage)
        # CuTe DSL 4.5.2 cannot carry a user-defined shared-storage object
        # through a dynamic region.  Materialize only the three communication
        # leaves used by that region before entering the role predicate.
        comm_mbarriers = storage.comm_mbarriers.data_ptr()
        comm_scratch = cute.make_tensor(
            storage.comm_scratch.data_ptr(),
            cute.make_layout((_fwd_dynamic_task_stream.SCRATCH_WORDS,)),
        )
        comm_arena = storage.arena.data_ptr()
        clc_response_ptr = storage.clc_response.data_ptr()
        # Per-CTA shared cu_end table for the compile-time decode method (no-op for LINEAR_SCAN).
        s_expert_cu_end = cute.make_tensor(storage.expert_cu_end.data_ptr(), cute.make_layout(EXPERT_TABLE_CAP))
        expert_stride, expert_groups = _build_expert_cu_end(tokens_per_expert, s_expert_cu_end, self.task_decode, thread)
        if cluster_index < Int32(comm_clusters):
            comm_args = _fwd_dynamic_task_stream.FwdCommArgs(
                x_peer_table=x_peer_table,
                combine_peer_table=combine_peer_table,
                schedule_peer_rank=schedule_peer_rank,
                schedule_peer_token_idx=schedule_peer_token_idx,
                x_routed=x_routed,
                y_routed=y_routed,
                x_routed_ready=x_routed_ready,
                y_routed_ready=y_routed_ready,
                y_routed_done=y_routed_done,
                num_tokens=num_tokens,
                macrobatch_size=Int32(macrobatch_size),
                minibatch_size=Int32(minibatch_size),
                hidden_size=Int32(x_routed.shape[1]),
                topk=Int32(topk),
                has_tail=self.has_tail,
            )
            slot_pool = _dynamic_task_stream.SlotPool(mbarriers=comm_mbarriers, scratch=comm_scratch, arena=comm_arena, shape=self.comm_tile_shape)
            _fwd_dynamic_task_stream.run_comm_prefix(
                comm_args,
                slot_pool,
                cluster_index * Int32(CLUSTER_SIZE) + cta_rank,
                Int32(num_comm_sms),
                self.epilogue_combine,
            )

        # No staged early return: the sentinel role WARPS_PER_CTA matches none of the
        # compute role predicates below, so only true compute clusters enter their
        # persistent role loops (communication and capacity-tail clusters skip them).
        role_warp = Int32(WARPS_PER_CTA)
        if cluster_index >= Int32(comm_clusters):
            if cluster_index < true_clusters:
                role_warp = warp

        is_leader_cta = cta_rank == Int32(0)
        is_two_cta = cute.size(tiled_mma.thr_id.shape) == 2
        arena = storage.arena.data_ptr().align(min_align=1024)
        sA = cute.make_tensor(
            cute.recast_ptr(arena, a_smem_layout.inner, dtype=BFloat16),
            a_smem_layout.outer,
        )
        sB = cute.make_tensor(
            cute.recast_ptr(
                arena + GEMM_A_BYTES,
                b_smem_layout.inner,
                dtype=BFloat16,
            ),
            b_smem_layout.outer,
        )
        sD = cute.make_tensor(
            cute.recast_ptr(
                arena + GEMM_A_BYTES + GEMM_B_BYTES,
                epi_smem_layout.inner,
                dtype=BFloat16,
            ),
            epi_smem_layout.outer,
        )
        # Routed swap-AB views of the same arena.  Gate / Up: the two 64-row halves of every A stage
        # (the swizzled inner layout of sA).  Raw MMA descriptors: the A and B ring base addresses.
        # Epilogue staging in the D stages: for Down, three 128-feature x 32-token M-major tiles (one
        # per D stage, _swapab_fwd.d_smem_layout); for Gate/Up, six 32-token x 64-feature tiles, two
        # alternating sets of (Gate, Up, Hidden).
        sA_gate = cute.make_tensor(
            cute.recast_ptr(arena, a_smem_layout.inner, dtype=BFloat16),
            cute.make_layout((_SWAP_HALF, _GATE_TILE_K, _GATE_AB_STAGES), stride=(_GATE_TILE_K, 1, _GATE_CTA_M * _GATE_TILE_K)),
        )
        sA_up = cute.make_tensor(
            cute.recast_ptr(arena + _SWAP_HALF * _GATE_TILE_K * 2, a_smem_layout.inner, dtype=BFloat16),
            cute.make_layout((_SWAP_HALF, _GATE_TILE_K, _GATE_AB_STAGES), stride=(_GATE_TILE_K, 1, _GATE_CTA_M * _GATE_TILE_K)),
        )
        a_ring_base = Int32(arena.toint())
        b_ring_base = a_ring_base + Int32(GEMM_A_BYTES)
        swap_d_layout = _swapab_fwd.d_smem_layout(epi_tile, _GATE_D_STAGES)
        assert cute.cosize(swap_d_layout) * 2 == GEMM_D_BYTES
        sDT = cute.make_tensor(
            cute.recast_ptr(arena + GEMM_A_BYTES + GEMM_B_BYTES, swap_d_layout.inner, dtype=BFloat16),
            swap_d_layout.outer,
        )
        sCT = cute.make_tensor(
            cute.recast_ptr(arena + GEMM_A_BYTES + GEMM_B_BYTES, dtype=BFloat16),
            cute.make_layout((_SWAP_EPI_ROWS, _SWAP_HALF, 2 * _GATE_D_STAGES), stride=(_SWAP_HALF, 1, _SWAP_EPI_ROWS * _SWAP_HALF)),
        )
        ab_pipeline = PipelineTmaUmma.create(
            num_stages=_GATE_AB_STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.collective.num_mcast_ctas_a + self.collective.num_mcast_ctas_b - 1,
            ),
            tx_count=self.collective.num_tma_load_bytes,
            barrier_storage=storage.gemm_ab_mbarriers.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        acc_pipeline = PipelineUmmaAsync.create(
            num_stages=_GATE_ACC_STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                len(self.collective.epilog_warp_id) * CLUSTER_SIZE,
            ),
            barrier_storage=storage.gemm_acc_mbarriers.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        # QuACK 0.5.0's SM100 epilogue performs the final accumulator release
        # inside ``with cute.arch.elect_one()``; 0.6.x moved that election into
        # optional pipeline-factory arguments which do not exist on 0.5.0.
        clc_cluster_layout_vmnk = cute.make_layout((1, CLUSTER_SIZE, 1, 1))
        clc_pipeline = pipeline.PipelineClcFetchAsync.create(
            barrier_storage=storage.clc_mbarriers.data_ptr(),
            num_stages=CLC_PIPE_DEPTH,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                CLC_COMPLETION_WARPS,
            ),
            tx_count=CLC_RESPONSE_BYTES,
            cta_layout_vmnk=clc_cluster_layout_vmnk,
            defer_sync=True,
        )
        clc_drain_pipeline = pipeline.PipelineClcFetchAsync.create(
            barrier_storage=storage.clc_drain_mbarriers.data_ptr(),
            num_stages=CLC_DRAIN_WARPS,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                CLUSTER_SIZE,
            ),
            tx_count=CLC_RESPONSE_BYTES,
            cta_layout_vmnk=clc_cluster_layout_vmnk,
            defer_sync=True,
        )
        # Decoded-task records: producer = the CLC warp (one elected thread writes and commits),
        # consumers = the other seven warps (one elected arrival each).  Per CTA, not cluster-wide:
        # each CTA's CLC warp receives the multicast CLC response and decodes the same work itself
        # (nothing in the record depends on the CTA rank).
        task_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.task_info_mbarriers.data_ptr(),
            num_stages=TASK_INFO_STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, TASK_INFO_CONSUMER_WARPS),
            defer_sync=True,
        )
        s_task_info = cute.make_tensor(storage.task_info.data_ptr(), cute.make_layout(TASK_INFO_STAGES * TASK_INFO_WORDS))
        s_task_taken = cute.make_tensor(storage.task_info_taken.data_ptr(), cute.make_layout(WARPS_PER_CTA))
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierGemm.TmemPtr),
            num_threads=((len(self.collective.epilog_warp_id) + 1) * cute.arch.WARP_SIZE),
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buffer.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.collective.epilog_warp_id[0],
            is_two_cta=is_two_cta,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbarrier.ptr,
        )
        # 16-byte aligned array of CLC_DRAIN_WARPS responses (four Int32 each);
        # each drain warp loads its own response atomically.
        drain_responses = storage.clc_drain_responses.data_ptr()
        pipeline_init_arrive(cluster_shape_mn=CLUSTER_SHAPE, is_relaxed=True)
        pipeline_init_wait(cluster_shape_mn=CLUSTER_SHAPE)

        # Each resident warp role owns its pipeline state for the entire CLC
        # lifetime.  The load, MMA, epilogue and idle warps take every record of
        # the task-info pipeline (wait, read, release) and run their body for a
        # valid tile; valid fused-FC1/Down tasks advance AB, ACC, and store state
        # in the same order in every producer/consumer role.  The final record
        # (active = 0) carries the CLC capacity-tail state for the drain.
        # The roles are one if / elif chain: a warp has one role, and no role's exit runs into the next
        # role's test.  With five separate ifs the epilogue test was also the continuation of the load and
        # MMA roles, whose exits follow lane-elected branches; the compiler then no longer treated the
        # epilogue warps as converged there and rebuilt their warp-uniform values in front of every use.
        if role_warp == Int32(_GATE_AB_LOAD_WARP):
            if cutlass.const_expr(self.split_weights):
                # Once per CTA: the routed
                # descriptors are built before this kernel is launched and are immutable while it
                # runs, so the mainloop only has to name one.
                routed_experts = Int32(cute.size(tokens_per_expert.shape))
                for operand in cutlass.range_constexpr(3):
                    routed_descriptors_pkg.acquire_descriptors(
                        routed_descriptors_pkg.operand_descriptors(routed_descriptors, routed_experts, operand), routed_experts
                    )
            task_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                TASK_INFO_STAGES,
            )
            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                _GATE_AB_STAGES,
            )
            thr_mma = tiled_mma.get_slice(cta_rank)
            # Gate and Up are independent N128 tensors.  Both use the rank-0 B
            # partition; CTA 1 issuing the Up descriptor places that fragment in
            # its own shared memory.
            b_rank0 = tiled_mma.get_slice(Int32(0))
            active = Int32(1)
            needs_drain = Int32(0)
            while active != Int32(0):
                task_pipeline.consumer_wait(task_consumer_state)
                (
                    active,
                    needs_drain,
                    kind,
                    valid_tile,
                    expert_index,
                    tile_coord_m,
                    tile_coord_n,
                    input_ready_index,
                    input_ready_required,
                    _,
                    k_tile_count,
                    expert_row_offset,
                    _,
                    tile_valid,
                ) = _load_task_info(s_task_info, task_consumer_state.index, s_task_taken, warp)
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    task_pipeline.consumer_release(task_consumer_state)
                task_consumer_state.advance()
                if valid_tile != Int32(0):
                    with cute.arch.elect_one():
                        if kind == Int32(1):
                            _counter_wait_gpu(
                                hidden_row_block_ready,
                                input_ready_index,
                                input_ready_required,
                            )
                        elif kind == Int32(2):
                            _counter_wait_gpu(
                                x_routed_ready,
                                input_ready_index,
                                input_ready_required,
                            )
                        elif kind == Int32(3):
                            _counter_wait_gpu(
                                hidden_row_block_ready,
                                input_ready_index,
                                input_ready_required,
                            )
                    cute.arch.sync_warp()
                    if kind == Int32(0):
                        gA_gate_up = cute.local_tile(
                            x_shared_tma,
                            cute.select(self.collective.mma_tiler, [0, 2]),
                            (tile_coord_m, None),
                        )
                        gate_tile_base = cute.domain_offset(
                            (
                                tile_coord_n * Int32(_GATE_LOGICAL_N),
                                Int32(0),
                            ),
                            shared_gate_weights,
                        )
                        up_tile_base = cute.domain_offset(
                            (
                                tile_coord_n * Int32(_GATE_LOGICAL_N),
                                Int32(0),
                            ),
                            shared_up_weights,
                        )
                        gB_gate = cute.local_tile(
                            gate_tile_base,
                            cute.select(self.collective.mma_tiler, [1, 2]),
                            (Int32(0), None),
                        )
                        gB_up = cute.local_tile(
                            up_tile_base,
                            cute.select(self.collective.mma_tiler, [1, 2]),
                            (Int32(0), None),
                        )
                        tCgA_gate_up = thr_mma.partition_A(gA_gate_up)
                        tCgB_gate = b_rank0.partition_B(gB_gate)
                        tCgB_up = b_rank0.partition_B(gB_up)
                        copy_A_gate_up = _make_tma_block_load_fn(
                            tma_atom_a_shared_gate_up,
                            src_tensor=tCgA_gate_up,
                            dst_tensor=sA,
                        )
                        copy_B_gate = _make_tma_block_load_fn(
                            tma_atom_b_shared_gate,
                            src_tensor=tCgB_gate,
                            dst_tensor=sB,
                        )
                        copy_B_up = _make_tma_block_load_fn(
                            tma_atom_b_shared_up,
                            src_tensor=tCgB_up,
                            dst_tensor=sB,
                        )
                        if cta_rank == Int32(0):
                            ab_producer_state = self.collective.load_tma(
                                ab_pipeline,
                                ab_producer_state,
                                [copy_A_gate_up, copy_B_gate],
                                k_tile_count,
                            )
                        else:
                            ab_producer_state = self.collective.load_tma(
                                ab_pipeline,
                                ab_producer_state,
                                [copy_A_gate_up, copy_B_up],
                                k_tile_count,
                            )
                    elif kind == Int32(1):
                        gA_down = cute.local_tile(
                            hidden_shared_tma,
                            cute.select(self.collective.mma_tiler, [0, 2]),
                            (tile_coord_m, None),
                        )
                        gB_down = cute.local_tile(
                            shared_down_weights,
                            cute.select(self.collective.mma_tiler, [1, 2]),
                            (tile_coord_n, None),
                        )
                        tCgA_down = thr_mma.partition_A(gA_down)
                        tCgB_down = thr_mma.partition_B(gB_down)
                        copy_A_down = _make_tma_block_load_fn(
                            tma_atom_a_shared_down,
                            src_tensor=tCgA_down,
                            dst_tensor=sA,
                        )
                        copy_B_down = _make_tma_block_load_fn(
                            tma_atom_b_shared_down,
                            src_tensor=tCgB_down,
                            dst_tensor=sB,
                        )
                        ab_producer_state = self.collective.load_tma(
                            ab_pipeline,
                            ab_producer_state,
                            [copy_A_down, copy_B_down],
                            k_tile_count,
                        )
                    else:
                        # Routed tiles, swap-AB: B holds the block's token rows, N = align16(valid rows);
                        # cta_group::2 splits N at N / 2, so the non-leader CTA's B half starts at token
                        # N / 2 instead of 128 (the box still has 128 rows, all inside the 256-row block; the
                        # rows past N are loaded but not read by the MMA).
                        tile_n = _swapab_fwd.mma_n(tile_valid)
                        b_shift = Int32(0)
                        if cta_rank != Int32(0):
                            b_shift = _swapab_fwd.nonleader_b_shift(tile_n)
                        if kind == Int32(2):
                            # A = [Gate rows | Up rows] of this CTA's 64 features (64-row tile
                            # tile_coord_n * 2 + cta_rank of each weight tensor), B = x.
                            feature_tile = tile_coord_n * Int32(CLUSTER_SIZE) + cta_rank
                            # Split allocations: the atom covers expert 0, so the operand is the
                            # whole (two-dimensional) tensor and the expert is named by the
                            # descriptor instead.  The tile coordinates are the same either way,
                            # every expert having the same shape and stride.
                            gate_source = routed_gate_weights
                            up_source = routed_up_weights
                            gate_desc = None
                            up_desc = None
                            if cutlass.const_expr(self.split_weights):
                                routed_experts = Int32(cute.size(tokens_per_expert.shape))
                                gate_desc = routed_descriptors_pkg.descriptor_argument(
                                    routed_descriptors_pkg.operand_descriptors(routed_descriptors, routed_experts, 0), expert_index
                                )
                                up_desc = routed_descriptors_pkg.descriptor_argument(
                                    routed_descriptors_pkg.operand_descriptors(routed_descriptors, routed_experts, 1), expert_index
                                )
                            else:
                                gate_source = routed_gate_weights[None, None, expert_index]
                                up_source = routed_up_weights[None, None, expert_index]
                            # A combined FC1: Up is up_row_offset rows into the parameter Gate
                            # starts at (0 for separate weights) -- a coordinate, so a retargeted
                            # descriptor keeps it.
                            up_source = cute.domain_offset((up_row_offset, Int32(0)), up_source)
                            gA_gate = cute.local_tile(
                                gate_source,
                                (_SWAP_HALF, _GATE_TILE_K),
                                (feature_tile, None),
                            )
                            gA_up = cute.local_tile(
                                up_source,
                                (_SWAP_HALF, _GATE_TILE_K),
                                (feature_tile, None),
                            )
                            gB_x = cute.local_tile(
                                cute.domain_offset((b_shift, Int32(0)), x_routed_tma),
                                cute.select(self.collective.mma_tiler, [1, 2]),
                                (tile_coord_m, None),
                            )
                            copy_A_gate = _make_tma_block_load_fn(tma_atom_routed_gate, src_tensor=gA_gate, dst_tensor=sA_gate, tma_desc_ptr=gate_desc)
                            copy_A_up = _make_tma_block_load_fn(tma_atom_routed_up, src_tensor=gA_up, dst_tensor=sA_up, tma_desc_ptr=up_desc)
                            copy_B_x = _make_tma_block_load_fn(tma_atom_b_routed_gate_up, src_tensor=thr_mma.partition_B(gB_x), dst_tensor=sB)
                            ab_producer_state = self.collective.load_tma(
                                ab_pipeline,
                                ab_producer_state,
                                [copy_A_gate, copy_A_up, copy_B_x],
                                k_tile_count,
                            )
                        else:
                            # A = the expert's Down weights (this CTA's 128 of the 256 H features), B = Hidden.
                            down_source = routed_down_weights
                            down_desc = None
                            if cutlass.const_expr(self.split_weights):
                                down_desc = routed_descriptors_pkg.descriptor_argument(
                                    routed_descriptors_pkg.operand_descriptors(routed_descriptors, Int32(cute.size(tokens_per_expert.shape)), 2), expert_index
                                )
                            else:
                                down_source = routed_down_weights[None, None, expert_index]
                            gA_down = cute.local_tile(
                                down_source,
                                cute.select(self.collective.mma_tiler, [0, 2]),
                                (tile_coord_n, None),
                            )
                            gB_down = cute.local_tile(
                                cute.domain_offset((b_shift, Int32(0)), hidden_routed_tma),
                                cute.select(self.collective.mma_tiler, [1, 2]),
                                (tile_coord_m, None),
                            )
                            copy_A_down = _make_tma_block_load_fn(
                                tma_atom_a_routed_down, src_tensor=thr_mma.partition_A(gA_down), dst_tensor=sA, tma_desc_ptr=down_desc
                            )
                            copy_B_down = _make_tma_block_load_fn(tma_atom_b_routed_down, src_tensor=thr_mma.partition_B(gB_down), dst_tensor=sB)
                            ab_producer_state = self.collective.load_tma(
                                ab_pipeline,
                                ab_producer_state,
                                [copy_A_down, copy_B_down],
                                k_tile_count,
                            )

            ab_pipeline.producer_tail(ab_producer_state)
            self._drain_clc_capacity_tail(
                needs_drain,
                clc_drain_pipeline,
                drain_responses,
                warp,
                cta_rank,
            )

        elif role_warp == Int32(_GATE_MMA_WARP):
            task_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                TASK_INFO_STAGES,
            )
            # The TMEM barrier counts this warp and the four epilogue warps, each from its own role branch:
            # unaligned, on the barrier itself (tmem.wait_for_alloc() is the aligned bar.sync).
            named_barrier_sync_unaligned(tmem_alloc_barrier)
            acc_tmem_ptr = tmem.retrieve_ptr(self.collective.acc_dtype)
            tCrA = tiled_mma.make_fragment_A(sA)
            tCrB = tiled_mma.make_fragment_B(sB)
            acc_shape = tiled_mma.partition_shape_C(self.collective.mma_tiler[:2])
            tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, _GATE_ACC_STAGES))
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)
            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                _GATE_AB_STAGES,
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                _GATE_ACC_STAGES,
            )
            mma_op = tiled_mma
            active = Int32(1)
            needs_drain = Int32(0)
            while active != Int32(0):
                task_pipeline.consumer_wait(task_consumer_state)
                (
                    active,
                    needs_drain,
                    kind,
                    valid_tile,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    k_tile_count,
                    _,
                    _,
                    tile_valid,
                ) = _load_task_info(s_task_info, task_consumer_state.index, s_task_taken, warp)
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    task_pipeline.consumer_release(task_consumer_state)
                task_consumer_state.advance()
                if valid_tile != Int32(0):
                    tCtAcc = tCtAcc_base[
                        None,
                        None,
                        None,
                        acc_producer_state.index,
                    ]
                    if kind == Int32(0) or kind == Int32(1):
                        (
                            ab_consumer_state,
                            acc_producer_state,
                            mma_op,
                        ) = self.collective.mma(
                            ab_pipeline,
                            acc_pipeline,
                            ab_consumer_state,
                            acc_producer_state,
                            mma_op,
                            tCrA,
                            tCrB,
                            tCtAcc,
                            k_tile_count,
                            is_leader_cta,
                            cta_rank,
                        )
                    else:
                        # Routed tiles, swap-AB: raw tcgen05 with N = align16(valid rows of the block).
                        tile_n = _swapab_fwd.mma_n(tile_valid)
                        ab_consumer_state, acc_producer_state = _swapab_fwd.mma_dynamic_n(
                            ab_pipeline,
                            acc_pipeline,
                            ab_consumer_state,
                            acc_producer_state,
                            a_ring_base,
                            b_ring_base,
                            GEMM_A_BYTES // _GATE_AB_STAGES,
                            Int32(tCtAcc.iterator.toint()),
                            k_tile_count,
                            tile_n,
                            is_leader_cta,
                        )

            # no more MMA: the epilogue warps may free TMEM (unaligned, as every use of this barrier)
            tmem_alloc_barrier.arrive_unaligned()
            acc_pipeline.producer_tail(acc_producer_state)
            self._drain_clc_capacity_tail(
                needs_drain,
                clc_drain_pipeline,
                drain_responses,
                warp,
                cta_rank,
            )

        elif role_warp < Int32(_GATE_MMA_WARP):
            task_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                TASK_INFO_STAGES,
            )
            tmem.allocate(self.collective.num_tmem_alloc_cols)
            named_barrier_sync_unaligned(tmem_alloc_barrier)  # the allocation is visible to the MMA warp
            acc_tmem_ptr = tmem.retrieve_ptr(self.collective.acc_dtype)
            acc_shape = tiled_mma.partition_shape_C(self.collective.mma_tiler[:2])
            tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, _GATE_ACC_STAGES))
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)
            tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = self.collective.epilog_tmem_copy_and_partition(
                thread,
                tCtAcc_base,
                epi_tile,
                is_two_cta,
            )
            tTR_rD = cute.make_rmem_tensor(
                tTR_rAcc.shape,
                self.collective.acc_dtype,
            )
            # Routed swap-AB epilogues: (lane m, column n) of every t2r register element (QuACK's
            # partition of the CTA tile's identity tensor) for the transposed staging, and the BF16 copy
            # of the fragment.
            swap_tTR_cAcc = tiled_copy_t2r.get_slice(thread).partition_D(cute.flat_divide(cute.make_identity_tensor((_GATE_CTA_M, _GATE_TILE_N)), epi_tile))
            swap_rT = cute.make_rmem_tensor_like(tTR_rAcc, BFloat16)
            tiled_copy_r2s, tRS_rD, tRS_sD = self.collective.epilog_smem_store_and_partition(
                tiled_copy_t2r,
                self.collective.d_layout,
                self.collective.d_dtype,
                tTR_rD,
                sD,
                thread,
            )
            # Routed Down, swap-AB: TMEM -> registers (16x256b) -> shared memory (stmatrix .trans).
            (
                swap_copy_t2r,
                swap_tTR_tAcc_base,
                swap_tTR_rAcc,
                swap_copy_r2s,
                swap_tRS_rD,
                swap_tRS_sD,
            ) = _swapab_fwd.epilogue_copies(self.collective.cta_tile_shape_mnk, tCtAcc_base, epi_tile, is_two_cta, sDT, thread)
            swap_rD_bf16 = cute.make_rmem_tensor_like(swap_tRS_rD, BFloat16)
            # epilogue_combine only (the default variant builds none of it): the same transposed stmatrix store into
            # padded staging rows (features contiguous), and the tile's 256 row pointers after them; both alias the D
            # region of the arena.
            epi_stage_base = epi_tRS_sD = epi_row_ptrs = routed_row_bytes = None
            if cutlass.const_expr(self.epilogue_combine):
                epi_stage_base = arena + _EPI_STAGE_OFFSET
                epi_staging = cute.make_tensor(
                    cute.recast_ptr(epi_stage_base, dtype=BFloat16),
                    cute.make_layout((_GATE_CTA_M, _SWAP_EPI_ROWS, _GATE_D_STAGES), stride=(1, _EPI_ROW_STRIDE // 2, _EPI_STAGE_BYTES // 2)),
                )
                _, _, _, _, _, epi_tRS_sD = _swapab_fwd.epilogue_copies(
                    self.collective.cta_tile_shape_mnk, tCtAcc_base, epi_tile, is_two_cta, epi_staging, thread
                )
                epi_row_ptrs = cute.make_tensor(cute.recast_ptr(arena + _EPI_PTR_OFFSET, dtype=Int32), cute.make_layout((2 * MLP_TILE_ROWS,)))
                routed_row_bytes = Int64(x_routed.shape[1]) * Int64(2)
            r_gate_bf16 = cute.make_rmem_tensor_like(tRS_rD, BFloat16)
            r_up_bf16 = cute.make_rmem_tensor_like(tRS_rD, BFloat16)
            rD_bf16 = cute.make_rmem_tensor_like(tRS_rD, BFloat16)
            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                _GATE_ACC_STAGES,
            )
            epi_store_pipeline = self.collective.make_epi_store_pipeline()
            is_tma_warp = Boolean(warp == Int32(0))
            epi_tile_shape = cute.zipped_divide(
                cute.make_layout(self.collective.cta_tile_shape_mnk[:2]),
                epi_tile,
            ).shape[1]
            epi_tile_layout = cute.make_ordered_layout(
                epi_tile_shape,
                order=(0, 1),
            )
            active = Int32(1)
            needs_drain = Int32(0)
            while active != Int32(0):
                task_pipeline.consumer_wait(task_consumer_state)
                (
                    active,
                    needs_drain,
                    kind,
                    valid_tile,
                    _,
                    tile_coord_m,
                    tile_coord_n,
                    _,
                    _,
                    output_ready_index,
                    _,
                    _,
                    macrobatch,
                    tile_valid,
                ) = _load_task_info(s_task_info, task_consumer_state.index, s_task_taken, warp)
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    task_pipeline.consumer_release(task_consumer_state)
                task_consumer_state.advance()
                if valid_tile != Int32(0):
                    fc1_tile_coord = (
                        tile_coord_m * Int32(CLUSTER_SIZE) + cta_rank,
                        tile_coord_n,
                        Int32(0),
                        Int32(0),
                    )
                    down_tile_coord = fc1_tile_coord
                    copy_D_shared_gate, _, _ = self.collective.epilog_gmem_copy_and_partition(
                        tma_atom_d_shared_gate,
                        gate_shared,
                        (_GATE_CTA_M, _GATE_LOGICAL_N),
                        epi_tile,
                        sD,
                        fc1_tile_coord,
                    )
                    copy_D_shared_up, _, _ = self.collective.epilog_gmem_copy_and_partition(
                        tma_atom_d_shared_up,
                        up_shared,
                        (_GATE_CTA_M, _GATE_LOGICAL_N),
                        epi_tile,
                        sD,
                        fc1_tile_coord,
                    )
                    copy_D_shared_hidden, _, _ = self.collective.epilog_gmem_copy_and_partition(
                        tma_atom_d_shared_hidden,
                        hidden_shared,
                        (_GATE_CTA_M, _GATE_LOGICAL_N),
                        epi_tile,
                        sD,
                        fc1_tile_coord,
                    )
                    copy_D_shared_down, _, _ = self.collective.epilog_gmem_copy_and_partition(
                        tma_atom_d_shared_down,
                        y_shared_tma,
                        self.collective.cta_tile_shape_mnk[:2],
                        epi_tile,
                        sD,
                        down_tile_coord,
                    )
                    tTR_tAcc = tTR_tAcc_base[
                        None,
                        None,
                        None,
                        None,
                        None,
                        acc_consumer_state.index,
                    ]
                    tTR_tAcc = cute.group_modes(
                        tTR_tAcc,
                        3,
                        cute.rank(tTR_tAcc),
                    )
                    swap_tTR_tAcc = swap_tTR_tAcc_base[None, None, None, None, None, acc_consumer_state.index]
                    swap_tTR_tAcc = cute.group_modes(swap_tTR_tAcc, 3, cute.rank(swap_tTR_tAcc))
                    # The routed tiles' N from the record's valid rows (shared tiles do not use it).
                    tile_n = _swapab_fwd.mma_n(tile_valid)
                    acc_pipeline.consumer_wait(acc_consumer_state)
                    if kind == Int32(0):
                        # Shared Gate/Up: TMEM columns 0..127 hold the Gate tile and 128..255 the Up
                        # tile (epilogue subtiles 0..3 and 4..7).  The paired load order keeps TMEM
                        # alive until Up subtile 7, where epi_load_acc_subtile releases it.
                        # The shared expert always stores its Gate / Up context.
                        for pair in cutlass.range_constexpr(_GATE_LOGICAL_N // _GATE_EPILOGUE_N):
                            gate_coord = epi_tile_layout.get_hier_coord(pair)
                            up_coord = epi_tile_layout.get_hier_coord(pair + _GATE_LOGICAL_N // _GATE_EPILOGUE_N)
                            self.collective.epi_load_acc_subtile(
                                tiled_copy_t2r,
                                tiled_copy_r2s,
                                tTR_tAcc,
                                tTR_rAcc,
                                tRS_rD,
                                gate_coord,
                                acc_pipeline,
                                acc_consumer_state,
                                _GATE_EPILOGUE_SUBTILES - 1,
                            )
                            r_gate_bf16.store(tRS_rD.load().to(BFloat16))
                            if is_tma_warp:
                                epi_store_pipeline.producer_acquire()
                            named_barrier_sync_unaligned(self.collective.epilogue_barrier)
                            cute.copy(
                                tiled_copy_r2s,
                                r_gate_bf16,
                                tRS_sD[None, None, None, 0],
                            )
                            cute.arch.fence_view_async_shared()
                            named_barrier_sync_unaligned(self.collective.epilogue_barrier)
                            if is_tma_warp:
                                copy_D_shared_gate(src_idx=0, dst_idx=gate_coord)
                                epi_store_pipeline.producer_commit()

                            self.collective.epi_load_acc_subtile(
                                tiled_copy_t2r,
                                tiled_copy_r2s,
                                tTR_tAcc,
                                tTR_rAcc,
                                tRS_rD,
                                up_coord,
                                acc_pipeline,
                                acc_consumer_state,
                                _GATE_EPILOGUE_SUBTILES - 1,
                            )
                            r_up_bf16.store(tRS_rD.load().to(BFloat16))
                            if is_tma_warp:
                                epi_store_pipeline.producer_acquire()
                            named_barrier_sync_unaligned(self.collective.epilogue_barrier)
                            cute.copy(
                                tiled_copy_r2s,
                                r_up_bf16,
                                tRS_sD[None, None, None, 1],
                            )
                            cute.arch.fence_view_async_shared()
                            named_barrier_sync_unaligned(self.collective.epilogue_barrier)
                            if is_tma_warp:
                                copy_D_shared_up(src_idx=1, dst_idx=gate_coord)
                                epi_store_pipeline.producer_commit()

                            r_hidden = cute.make_rmem_tensor(r_up_bf16.shape, BFloat16)
                            for value in cutlass.range_constexpr(cute.size(r_hidden)):
                                gate_value = Float32(r_gate_bf16[value])
                                up_value = Float32(r_up_bf16[value])
                                if cutlass.const_expr(self.is_clamped):
                                    limit = Float32(swiglu_limit)
                                    gate_value = cutlass.min(gate_value, limit)
                                    up_value = cutlass.min(
                                        cutlass.max(up_value, -limit),
                                        limit,
                                    )
                                r_hidden[value] = glu_forward(gate_value, up_value, self.is_situ, situ_beta1, situ_beta2).to(BFloat16)
                            if is_tma_warp:
                                epi_store_pipeline.producer_acquire()
                            named_barrier_sync_unaligned(self.collective.epilogue_barrier)
                            cute.copy(
                                tiled_copy_r2s,
                                r_hidden,
                                tRS_sD[None, None, None, 2],
                            )
                            cute.arch.fence_view_async_shared()
                            named_barrier_sync_unaligned(self.collective.epilogue_barrier)
                            if is_tma_warp:
                                copy_D_shared_hidden(src_idx=2, dst_idx=gate_coord)
                                epi_store_pipeline.producer_commit()
                        acc_consumer_state.advance()
                        if is_tma_warp:
                            cute.arch.cp_async_bulk_wait_group(0, read=False)
                            with cute.arch.elect_one():
                                _counter_arrive_gpu(
                                    hidden_row_block_ready,
                                    output_ready_index,
                                    Int32(1),
                                )
                    elif kind == Int32(2):
                        # Routed Gate/Up, swap-AB.  TMEM lane m holds Gate (m < 64) or Up (m >= 64) of
                        # feature tile_coord_n * 128 + cta_rank * 64 + m % 64; the columns are the
                        # block's tokens.  Per 32-token subtile: the Gate warps (0, 1) and the Up warps
                        # (2, 3) stage their BF16 values transposed into the (Gate, Up) staging tiles;
                        # one barrier; every thread takes one token and 16 features, reads Gate and Up
                        # back as 128-bit vectors, computes SwiGLU and stages Hidden; one barrier;
                        # thread 0 issues the TMA stores (Hidden, plus Gate and Up when this tile saves
                        # the context).  Two staging sets alternate, so a subtile's stores only have to
                        # finish before the next-but-one subtile stages into the same set.
                        # Subtiles at or past the tile's N hold padding rows only: the Down GEMM never
                        # reads their Hidden, so they just load TMEM (the last one releases the
                        # accumulator), except in a context-saving tile, where their Gate / Up / Hidden
                        # rows are stored as zeros; padding rows inside a stored subtile are zero as
                        # well.  That is what zero x padding rows give, and a backward reads the
                        # macrobatch-0 context of every padded row (SwiGLU-backward multiplies it by
                        # router weight 0, so it must be finite; zero keeps it exact).
                        save_context = Int32(macrobatch == Int32(0))
                        feature_tile = tile_coord_n * Int32(CLUSTER_SIZE) + cta_rank
                        token_of_thread = thread // Int32(4)
                        chunk_of_thread = thread % Int32(4)
                        # The staging tiles alias the D stages: the previous tile's stores must have
                        # finished reading them before the first subtile is staged.  Only warp 0 waits
                        # for them at the end of a tile, so the other warps need this barrier too.
                        if thread == Int32(0):
                            cute.arch.cp_async_bulk_wait_group(0, read=True)
                        named_barrier_sync_unaligned(self.collective.epilogue_barrier)
                        for sub in cutlass.range_constexpr(_GATE_EPILOGUE_SUBTILES):
                            sub_coord = epi_tile_layout.get_hier_coord(sub)
                            self.collective.epi_load_acc_subtile(
                                tiled_copy_t2r,
                                tiled_copy_r2s,
                                tTR_tAcc,
                                tTR_rAcc,
                                tRS_rD,
                                sub_coord,
                                acc_pipeline,
                                acc_consumer_state,
                                _GATE_EPILOGUE_SUBTILES - 1,
                            )
                            live = Int32(Int32(sub * _SWAP_EPI_ROWS) < tile_n)
                            if (live | save_context) != Int32(0):
                                staging = (sub % 2) * 3
                                if live != Int32(0):
                                    swap_rT.store(tTR_rAcc.load().to(BFloat16))
                                    sub_coords = swap_tTR_cAcc[None, None, None, sub_coord[0], sub_coord[1]]
                                    lane_m, _ = sub_coords[0]
                                    is_up = Int32(lane_m >= Int32(_SWAP_HALF))
                                    feature = lane_m - is_up * Int32(_SWAP_HALF)
                                    for value in cutlass.range_constexpr(cute.size(swap_rT)):
                                        _, column = sub_coords[value]
                                        sCT[Int32(column) - Int32(sub * _SWAP_EPI_ROWS), feature, Int32(staging) + is_up] = swap_rT[value]
                                named_barrier_sync_unaligned(self.collective.epilogue_barrier)
                                gate_vec = cute.local_tile(sCT[token_of_thread, None, staging], (16,), (chunk_of_thread,))
                                up_vec = cute.local_tile(sCT[token_of_thread, None, staging + 1], (16,), (chunk_of_thread,))
                                hidden_vec = cute.local_tile(sCT[token_of_thread, None, staging + 2], (16,), (chunk_of_thread,))
                                r_gate = cute.make_rmem_tensor_like(gate_vec)
                                r_up = cute.make_rmem_tensor_like(up_vec)
                                r_hidden = cute.make_rmem_tensor_like(hidden_vec)
                                if Int32(sub * _SWAP_EPI_ROWS) + token_of_thread < tile_valid:
                                    cute.autovec_copy(gate_vec, r_gate)
                                    cute.autovec_copy(up_vec, r_up)
                                else:
                                    zero = Float32(0.0).to(BFloat16)
                                    for value in cutlass.range_constexpr(16):
                                        r_gate[value] = zero
                                        r_up[value] = zero
                                    cute.autovec_copy(r_gate, gate_vec)
                                    cute.autovec_copy(r_up, up_vec)
                                for value in cutlass.range_constexpr(16):
                                    gate_value = Float32(r_gate[value])
                                    up_value = Float32(r_up[value])
                                    if cutlass.const_expr(self.is_clamped):
                                        limit = Float32(swiglu_limit)
                                        gate_value = cutlass.min(gate_value, limit)
                                        up_value = cutlass.min(
                                            cutlass.max(up_value, -limit),
                                            limit,
                                        )
                                    r_hidden[value] = glu_forward(gate_value, up_value, self.is_situ, situ_beta1, situ_beta2).to(BFloat16)
                                cute.autovec_copy(r_hidden, hidden_vec)
                                cute.arch.fence_view_async_shared()
                                if thread == Int32(0):
                                    # the previous subtile's stores read the other set, which the next subtile stages
                                    cute.arch.cp_async_bulk_wait_group(0, read=True)
                                named_barrier_sync_unaligned(self.collective.epilogue_barrier)
                                if thread == Int32(0):
                                    row_tile = tile_coord_m * Int32(_GATE_EPILOGUE_SUBTILES) + Int32(sub)
                                    if save_context != Int32(0):
                                        _swapab_fwd.store_tile(
                                            tma_atom_d_routed_gate,
                                            gate_routed,
                                            (_SWAP_EPI_ROWS, _SWAP_HALF),
                                            (row_tile, feature_tile),
                                            sCT[None, None, staging],
                                        )
                                        _swapab_fwd.store_tile(
                                            tma_atom_d_routed_up,
                                            up_routed,
                                            (_SWAP_EPI_ROWS, _SWAP_HALF),
                                            (row_tile, feature_tile),
                                            sCT[None, None, staging + 1],
                                        )
                                    # Hidden too in a context-saving tile (zeros past N): a backward reads
                                    # the macrobatch-0 Hidden of every padded row.
                                    _swapab_fwd.store_tile(
                                        tma_atom_d_routed_hidden,
                                        hidden_routed,
                                        (_SWAP_EPI_ROWS, _SWAP_HALF),
                                        (row_tile, feature_tile),
                                        sCT[None, None, staging + 2],
                                    )
                                    cute.arch.cp_async_bulk_commit_group()
                        acc_consumer_state.advance()
                        if is_tma_warp:
                            cute.arch.cp_async_bulk_wait_group(0, read=False)
                            with cute.arch.elect_one():
                                _counter_arrive_gpu(
                                    hidden_row_block_ready,
                                    output_ready_index,
                                    Int32(1),
                                )
                    elif kind == Int32(1):
                        # Shared Down keeps the M128xN256 BF16 epilogue.
                        for epi_idx in cutlass.range_constexpr(_GATE_EPILOGUE_SUBTILES):
                            epi_coord = epi_tile_layout.get_hier_coord(epi_idx)
                            self.collective.epi_load_acc_subtile(
                                tiled_copy_t2r,
                                tiled_copy_r2s,
                                tTR_tAcc,
                                tTR_rAcc,
                                tRS_rD,
                                epi_coord,
                                acc_pipeline,
                                acc_consumer_state,
                                _GATE_EPILOGUE_SUBTILES - 1,
                            )
                            rD_bf16.store(tRS_rD.load().to(BFloat16))
                            if is_tma_warp:
                                epi_store_pipeline.producer_acquire()
                            named_barrier_sync_unaligned(self.collective.epilogue_barrier)
                            epi_buffer = epi_idx % _GATE_D_STAGES
                            cute.copy(
                                tiled_copy_r2s,
                                rD_bf16,
                                tRS_sD[None, None, None, epi_buffer],
                            )
                            cute.arch.fence_view_async_shared()
                            named_barrier_sync_unaligned(self.collective.epilogue_barrier)
                            if is_tma_warp:
                                copy_D_shared_down(
                                    src_idx=epi_buffer,
                                    dst_idx=epi_coord,
                                )
                                epi_store_pipeline.producer_commit()
                        acc_consumer_state.advance()
                        if is_tma_warp:
                            cute.arch.cp_async_bulk_wait_group(0, read=False)
                    elif cutlass.const_expr(self.epilogue_combine):
                        # Routed Down with the combine in the epilogue (_routed_down_push_epilogue).
                        self._routed_down_push_epilogue(
                            thread,
                            cta_rank,
                            is_tma_warp,
                            macrobatch,
                            macrobatch_size,
                            tile_coord_m,
                            tile_coord_n,
                            output_ready_index,
                            tile_n,
                            epi_tile_layout,
                            swap_copy_t2r,
                            swap_copy_r2s,
                            swap_tTR_tAcc,
                            swap_tTR_rAcc,
                            swap_tRS_rD,
                            swap_rD_bf16,
                            epi_tRS_sD,
                            epi_stage_base,
                            epi_row_ptrs,
                            routed_row_bytes,
                            acc_pipeline,
                            acc_consumer_state,
                            schedule_peer_rank,
                            schedule_peer_token_idx,
                            combine_peer_table,
                            y_routed_ready,
                        )
                        acc_consumer_state.advance()
                    else:
                        # Routed Down, swap-AB: TMEM lanes are this CTA's 128 H features, the columns
                        # the block's 256 tokens.  Each CTA therefore overwrites all 256 ring rows of the
                        # block (its 128 features) and waits for both 128-row halves that macrobatch + 1
                        # used last (y_routed_done counts per 128 rows).  Every 32-token subtile below
                        # the tile's N goes M-major (features contiguous) into a D stage with stmatrix
                        # .trans and is stored by TMA; subtiles at or past N hold padding tokens the
                        # combine never pushes, so they just load TMEM (the last one releases the
                        # accumulator).
                        num_macrobatches = _ceil_div_i32(num_tokens, Int32(macrobatch_size))
                        if macrobatch + Int32(1) < num_macrobatches:
                            prior_offset = (macrobatch + Int32(1)) * Int32(macrobatch_size)
                            prior_rows = cutlass.min(
                                Int32(macrobatch_size),
                                num_tokens - prior_offset,
                            )
                            for half in cutlass.range_constexpr(CLUSTER_SIZE):
                                half_row = tile_coord_m * Int32(MLP_TILE_ROWS) + Int32(half * _GATE_CTA_M)
                                if half_row < prior_rows:
                                    if is_tma_warp:
                                        with cute.arch.elect_one():
                                            _counter_wait_gpu(
                                                y_routed_done,
                                                (prior_offset + half_row) // Int32(_GATE_CTA_M),
                                                Int32(self.comm_tile_shape.row_tiles(MLP_TILE_ROWS // CLUSTER_SIZE)) * routed_comm_tile_columns,
                                            )
                            named_barrier_sync_unaligned(self.collective.epilogue_barrier)
                        for epi_idx in cutlass.range_constexpr(_GATE_EPILOGUE_SUBTILES):
                            epi_coord = epi_tile_layout.get_hier_coord(epi_idx)
                            self.collective.epi_load_acc_subtile(
                                swap_copy_t2r,
                                swap_copy_r2s,
                                swap_tTR_tAcc,
                                swap_tTR_rAcc,
                                swap_tRS_rD,
                                epi_coord,
                                acc_pipeline,
                                acc_consumer_state,
                                _GATE_EPILOGUE_SUBTILES - 1,
                            )
                            if Int32(epi_idx * _SWAP_EPI_ROWS) < tile_n:
                                swap_rD_bf16.store(swap_tRS_rD.load().to(BFloat16))
                                if is_tma_warp:
                                    epi_store_pipeline.producer_acquire()
                                named_barrier_sync_unaligned(self.collective.epilogue_barrier)
                                epi_buffer = epi_idx % _GATE_D_STAGES
                                cute.copy(swap_copy_r2s, swap_rD_bf16, swap_tRS_sD[None, None, None, epi_buffer])
                                cute.arch.fence_view_async_shared()
                                named_barrier_sync_unaligned(self.collective.epilogue_barrier)
                                if is_tma_warp:
                                    if thread == Int32(0):
                                        _swapab_fwd.store_tile(
                                            tma_atom_d_routed_down,
                                            y_routed_tma,
                                            (_GATE_CTA_M, _SWAP_EPI_ROWS),
                                            (tile_coord_n * Int32(CLUSTER_SIZE) + cta_rank, tile_coord_m * Int32(_GATE_EPILOGUE_SUBTILES) + Int32(epi_idx)),
                                            sDT[None, None, epi_buffer],
                                        )
                                    epi_store_pipeline.producer_commit()
                        acc_consumer_state.advance()
                        if is_tma_warp:
                            cute.arch.cp_async_bulk_wait_group(0, read=False)
                            with cute.arch.elect_one():
                                _counter_arrive_gpu(
                                    y_routed_ready,
                                    output_ready_index,
                                    Int32(1),
                                )

            if is_tma_warp:
                epi_store_pipeline.producer_tail()
            tmem.relinquish_alloc_permit()
            named_barrier_sync_unaligned(tmem_alloc_barrier)  # the MMA warp has issued its last MMA
            tmem.free(acc_tmem_ptr)
            self._drain_clc_capacity_tail(
                needs_drain,
                clc_drain_pipeline,
                drain_responses,
                warp,
                cta_rank,
            )

        elif role_warp == Int32(_GATE_CLC_WARP):
            # Scheduler warp of this CTA: decode each work item once (the whole warp runs the
            # decode uniformly) and publish the record; the CLC query
            # for the next item is issued only after every consumer has taken the current record
            # and both CTAs have released the response (CLC depth 1).  The response that
            # produced a record is released after that acquire, so rank 1's consumers gate rank
            # 0's query the same way rank 0's own do.
            scheduler = utils.ClcDynamicPersistentTileScheduler.create(
                scheduler_params,
                cute.arch.block_idx(),
                cute.arch.grid_dim(),
                clc_response_ptr,
            )
            work = scheduler.initial_work_tile_info()
            clc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                CLC_PIPE_DEPTH,
            )
            clc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.ProducerConsumer,
                CLC_PIPE_DEPTH,
            )
            task_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                TASK_INFO_STAGES,
            )
            logical_cluster = work.tile_idx[0] // Int32(CLUSTER_SIZE)
            work_valid = Int32(work.is_valid_tile)
            response_pending = Int32(0)
            active = Int32(1)
            needs_drain = Int32(0)
            while active != Int32(0):
                (
                    active,
                    needs_drain,
                    kind,
                    valid_tile,
                    expert_index,
                    tile_coord_m,
                    tile_coord_n,
                    input_ready_index,
                    input_ready_required,
                    output_ready_index,
                    k_tile_count,
                    expert_row_offset,
                    macrobatch,
                    tile_valid,
                ) = self._decode_task_record(
                    work_valid,
                    logical_cluster,
                    true_clusters,
                    comm_clusters,
                    tokens_per_expert,
                    s_expert_cu_end,
                    expert_stride,
                    expert_groups,
                    valid_tokens_per_expert,
                    num_tokens,
                    shared_gate_up_tasks,
                    shared_down_tasks,
                    shared_tasks,
                    shared_row_blocks,
                    minibatch_gate_up_tasks,
                    minibatch_down_tasks,
                    minibatch_tasks,
                    macrobatch_size,
                    minibatch_size,
                    shared_k_tiles,
                    shared_column_blocks,
                    routed_k_tiles,
                    routed_column_blocks,
                    routed_comm_tile_columns,
                    intermediate_size,
                )
                task_pipeline.producer_acquire(task_producer_state)
                with cute.arch.elect_one():
                    _store_task_info(
                        s_task_info,
                        task_producer_state.index,
                        active,
                        needs_drain,
                        kind,
                        valid_tile,
                        expert_index,
                        tile_coord_m,
                        tile_coord_n,
                        input_ready_index,
                        input_ready_required,
                        output_ready_index,
                        k_tile_count,
                        expert_row_offset,
                        macrobatch,
                        tile_valid,
                    )
                    task_pipeline.producer_commit(task_producer_state)
                task_producer_state.advance()
                # Every consumer has taken this record (one stage: the phase the commit opened);
                # for the final record this is also the pipeline's tail.
                task_pipeline.producer_acquire(task_producer_state)
                if response_pending != Int32(0):
                    with cute.arch.elect_one():
                        clc_pipeline.consumer_release(clc_consumer_state)
                    clc_consumer_state.advance()
                if active != Int32(0):
                    if cta_rank == Int32(0):
                        clc_pipeline.producer_acquire(clc_producer_state)
                        scheduler.advance_to_next_work(clc_pipeline.producer_get_barrier(clc_producer_state))
                        clc_producer_state.advance()
                    clc_pipeline.consumer_wait(clc_consumer_state)
                    next_work = scheduler.get_current_work()
                    cute.arch.sync_warp()
                    logical_cluster = next_work.tile_idx[0] // Int32(CLUSTER_SIZE)
                    work_valid = Int32(next_work.is_valid_tile)
                    response_pending = Int32(1)

            if cta_rank == Int32(0):
                clc_pipeline.producer_tail(clc_producer_state)
            self._drain_clc_capacity_tail(
                needs_drain,
                clc_drain_pipeline,
                drain_responses,
                warp,
                cta_rank,
            )

        elif role_warp == Int32(_GATE_BF16_IDLE_WARP):
            # The idle warp takes every record too: it needs the final record's drain state.
            task_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                TASK_INFO_STAGES,
            )
            active = Int32(1)
            needs_drain = Int32(0)
            while active != Int32(0):
                task_pipeline.consumer_wait(task_consumer_state)
                (
                    active,
                    needs_drain,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                ) = _load_task_info(s_task_info, task_consumer_state.index, s_task_taken, warp)
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    task_pipeline.consumer_release(task_consumer_state)
                task_consumer_state.advance()

            self._drain_clc_capacity_tail(
                needs_drain,
                clc_drain_pipeline,
                drain_responses,
                warp,
                cta_rank,
            )


def routed_weight_view(tensor):
    """A routed weight in the kernel's operand order.

    Grouped: the ``[E, N, K]`` parameter as the strided ``[N, K, E]`` view whose L mode the
    kernel slices per expert.  Split: the representative expert alone, ``[N, K]`` -- independent
    allocations have no common L stride, so the expert is named by a TMA descriptor instead.
    Never a packed or copied mirror either way.
    """
    view = row_major_view(tensor)
    return view if view.ndim == 2 else view.permute(1, 2, 0)


def _routed_tables_or_placeholder(tables, device):
    """The per-expert base addresses, or a (1, 1) stand-in when the weights are grouped."""
    import torch

    if tables is not None:
        return tables
    return torch.zeros(1, 1, dtype=torch.int64, device=device)


def _routed_descriptors_or_placeholder(descriptors, device):
    """The bytes the per-expert descriptors live in, or a stand-in when the weights are grouped."""
    import torch

    if descriptors is not None:
        return descriptors
    return torch.zeros(routed_descriptors_pkg.BYTES_PER_DESCRIPTOR, dtype=torch.uint8, device=device)


def make_mega_args(
    x_peer_addresses,
    combine_peer_addresses,
    schedule_peer_rank,
    schedule_peer_token_idx,
    num_tokens,
    tokens_per_expert,
    valid_tokens_per_expert,
    x_shared,
    shared_gate_weights,
    routed_gate_weights,
    shared_up_weights,
    routed_up_weights,
    shared_down_weights,
    routed_down_weights,
    state,
    *,
    num_local_tokens: int,
    schedule_capacity: int,
    macrobatch_size: int = 32768,
    minibatch_size: int = 4096,
    num_comm_sms: int = 40,
    topk: int = 10,
    swiglu_limit: float = 0.0,
    stream=None,
    launch: bool = False,
    peer_tables: dict | None = None,
    routed_storage_tables=None,
    routed_descriptors=None,
    build_descriptors: int = 1,
    up_row_offset: int = 0,
    situ_beta=(0.0, 0.0),
):
    """Build the CuTe launch keyword arguments after the host gate succeeds.

    ``state`` is duck-typed to the nine-output/five-counter object owned by
    :mod:`persistent_bf16_fwd`; keeping that import outside this module prevents
    the CPU-only host validator from importing CUTLASS.  Routed weights stay
    as strided ``[N,K,E]`` views, never a packed or copied mirror.

    ``launch=False`` builds the CuTe tensors and DSL scalars that ``cute.compile``
    traces.  ``launch=True`` builds the arguments of a call to the compiled TVM FFI
    function: the same torch views and plain Python scalars, which the FFI entry
    converts natively (no ``from_dlpack`` per tensor and call).
    ``peer_tables`` is the workspace-owned cache of peer address tables (``None``: build them).
    """

    import torch

    tables = {} if peer_tables is None else peer_tables
    if stream is None:
        stream = torch.cuda.current_stream(x_shared.device)
    cuda_stream = stream if isinstance(stream, cuda.CUstream) else cuda.CUstream(stream.cuda_stream)

    routed_gate_nke = routed_weight_view(routed_gate_weights)
    routed_up_nke = routed_weight_view(routed_up_weights)
    routed_down_nke = routed_weight_view(routed_down_weights)

    def dynamic_tensor(tensor, leading_dim):
        # Keep only the contiguous dimension's unit stride static. Extents,
        # including E, and all other strides are runtime descriptor metadata.
        if launch:
            return tensor
        return from_dlpack(tensor, assumed_align=16, enable_tvm_ffi=True).mark_layout_dynamic(
            leading_dim=leading_dim,
        )

    if launch:
        Int32, Float32 = int, float  # noqa: N806 -- plain scalars at the FFI call
    else:
        from cutlass import Float32, Int32  # noqa: N806

    # Keyword arguments named after the kernel's __call__ parameters: the launch
    # ABI is readable at the call site and cannot be reordered silently.
    return dict(
        x_peer_table=dynamic_tensor(peer_address_table(x_peer_addresses, x_shared.device, tables), 0),
        combine_peer_table=dynamic_tensor(peer_address_table(combine_peer_addresses, x_shared.device, tables), 0),
        schedule_peer_rank=dynamic_tensor(row_major_view(schedule_peer_rank), 0),
        schedule_peer_token_idx=dynamic_tensor(row_major_view(schedule_peer_token_idx), 0),
        num_tokens=dynamic_tensor(row_major_view(num_tokens), 0),
        tokens_per_expert=dynamic_tensor(row_major_view(tokens_per_expert), 0),
        valid_tokens_per_expert=dynamic_tensor(row_major_view(valid_tokens_per_expert), 0),
        x_shared=dynamic_tensor(row_major_view(x_shared), 1),
        x_routed=dynamic_tensor(row_major_view(state.x_routed), 1),
        shared_gate_weights=dynamic_tensor(row_major_view(shared_gate_weights), 1),
        routed_gate_weights=dynamic_tensor(routed_gate_nke, 1),
        shared_up_weights=dynamic_tensor(row_major_view(shared_up_weights), 1),
        routed_up_weights=dynamic_tensor(routed_up_nke, 1),
        shared_down_weights=dynamic_tensor(row_major_view(shared_down_weights), 1),
        routed_down_weights=dynamic_tensor(routed_down_nke, 1),
        gate_shared=dynamic_tensor(row_major_view(state.gate_shared), 1),
        gate_routed=dynamic_tensor(row_major_view(state.gate_routed), 1),
        up_shared=dynamic_tensor(row_major_view(state.up_shared), 1),
        up_routed=dynamic_tensor(row_major_view(state.up_routed), 1),
        hidden_shared=dynamic_tensor(row_major_view(state.hidden_shared), 1),
        hidden_routed=dynamic_tensor(row_major_view(state.hidden_routed), 1),
        y_shared=dynamic_tensor(row_major_view(state.y_shared), 1),
        y_routed=dynamic_tensor(row_major_view(state.y_routed), 1),
        gate_up_tile_ready=dynamic_tensor(row_major_view(state.gate_up_tile_ready), 0),
        hidden_row_block_ready=dynamic_tensor(row_major_view(state.hidden_row_block_ready), 0),
        x_routed_ready=dynamic_tensor(row_major_view(state.x_routed_ready), 0),
        y_routed_ready=dynamic_tensor(row_major_view(state.y_routed_ready), 0),
        y_routed_done=dynamic_tensor(row_major_view(state.y_routed_done), 0),
        # One-element placeholders when the routed weights are grouped: the kernel's split path is
        # compiled out, but the ABI is one shape so the executor stays positional.
        # (3, E): the experts are the contiguous mode, so that is the static unit stride.
        routed_storage_tables=dynamic_tensor(_routed_tables_or_placeholder(routed_storage_tables, state.y_routed_done.device), 1),
        routed_descriptors=dynamic_tensor(_routed_descriptors_or_placeholder(routed_descriptors, state.y_routed_done.device), 0),
        build_descriptors=Int32(build_descriptors),
        up_row_offset=Int32(up_row_offset),
        situ_beta1=Float32(situ_beta[0]),
        situ_beta2=Float32(situ_beta[1]),
        num_local_tokens=Int32(num_local_tokens),
        schedule_capacity=Int32(schedule_capacity),
        macrobatch_size=Int32(macrobatch_size),
        minibatch_size=Int32(minibatch_size),
        num_comm_sms=Int32(num_comm_sms),
        topk=Int32(topk),
        swiglu_limit=Float32(swiglu_limit),
        stream=cuda_stream,
    )


def fill_mega_launch_slots(args, x_shared, schedule, weights, outputs, counters, stream, build=0):
    """Fill the per-call slots of a launch-argument list built by make_mega_args(launch=True): the
    caller-owned tensors (x_shared: the caller's shared input or the workspace x buffer; schedule =
    peer_rank, peer_token_idx, num_tokens, tokens_per_expert, valid_tokens_per_expert; weights =
    shared/routed gate, up, down), the nine fresh
    outputs, the five fresh counters (row-major already) and the stream.  The kept template holds only
    the peer tables and scalars, plus -- when the routed weights are split -- the derived (3, E)
    address table and descriptor bytes, which are not per-call (see MEGA_PER_CALL_SLOTS).
    Positions follow make_mega_args."""

    rmv = row_major_view
    args[2] = rmv(schedule[0])
    args[3] = rmv(schedule[1])
    args[4] = rmv(schedule[2])
    args[5] = rmv(schedule[3])
    args[6] = rmv(schedule[4])
    args[7] = rmv(x_shared)
    args[8] = outputs[0]
    args[9] = rmv(weights[0])
    args[10] = routed_weight_view(weights[1])
    args[11] = rmv(weights[2])
    args[12] = routed_weight_view(weights[3])
    args[13] = rmv(weights[4])
    args[14] = routed_weight_view(weights[5])
    args[15:23] = outputs[1:]
    args[23:28] = counters
    args[_BUILD_DESCRIPTORS_SLOT] = build
    args[-1] = stream
    return args


# Slots fill_mega_launch_slots fills on every call (the rest are peer tables and scalars).
# The routed storage table and the descriptor bytes (28, 29) are not among them: they are derived
# from the caller's per-expert address tables, which are part of the launch plan's key, so a
# weight that moves gives a new plan rather than a kept table that has gone stale.
# 30 is build_descriptors, which is per call: whether this launch builds the routed descriptors.
_BUILD_DESCRIPTORS_SLOT = 30
MEGA_PER_CALL_SLOTS = tuple(range(2, 28)) + (_BUILD_DESCRIPTORS_SLOT, -1)


def prepare_mega_bf16(
    *args,
    is_clamped: bool,
    task_decode: TaskDecode,
    epilogue_combine: bool = False,
    split_weights: bool = False,
    is_situ: bool = False,
    has_shared: bool = True,
    shared_fc1_only: bool = False,
    has_tail: bool = False,
    comm_tile_shape: CommTileShape = DEFAULT_COMM_TILE_SHAPE,
    **kwargs,
):
    """Compile one kernel variant (clamp mode, task decode, combine placement, weight layout, activation,
    shared expert, shared FC1 alone, tail tiles, communication tile shape) and return the executor; every
    argument stays in the launch ABI."""

    cute_kwargs = make_mega_args(*args, **kwargs)
    # The TVM FFI executor is positional: the launch passes cute_kwargs.values() in
    # this order, so the keys must be exactly the __call__ parameters in order.
    parameters = tuple(inspect.signature(_PersistentBf16FwdMega.__call__).parameters)[1:]
    if tuple(cute_kwargs) != parameters:
        raise RuntimeError("make_mega_args keys must match __call__ parameters in order")
    return cute.compile(
        _PersistentBf16FwdMega(
            is_clamped=is_clamped,
            task_decode=task_decode,
            epilogue_combine=epilogue_combine,
            split_weights=split_weights,
            is_situ=is_situ,
            has_shared=has_shared,
            shared_fc1_only=shared_fc1_only,
            has_tail=has_tail,
            comm_tile_shape=comm_tile_shape,
        ),
        **cute_kwargs,
        options=COMPILE_OPTIONS,
    )


__all__ = [
    "MEGA_PER_CALL_SLOTS",
    "fill_mega_launch_slots",
    "make_mega_args",
    "prepare_mega_bf16",
    "routed_weight_view",
]
