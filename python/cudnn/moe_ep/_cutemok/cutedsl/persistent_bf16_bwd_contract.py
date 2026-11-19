# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae);
# ported to CuTe DSL and modified by NVIDIA.

"""Runtime geometry contract for the ordinary BF16 backward (any EP size).

Tile and pipeline modes are static; geometry and decoder metadata are per-call values.  The compatibility
constants describe one fixed reference geometry only.  ``BwdGeometry`` is the host allocation geometry;
``BwdConfig`` adds actual values for the pure-Python model of the task counts, segments and task stream.
"""

from collections import Counter
from dataclasses import dataclass
from enum import Enum
from functools import cache, lru_cache
from typing import Optional

# The task->expert decode method (TaskDecode) is the forward contract's.  The backward table has its own
# capacity: one Int32 token end per expert, so 512 entries (2 KiB) keep up to 512 local experts in the table
# (SMEM_BINARY, no global read in the owner decode or the Wgrad K range) instead of the grouped lookup the
# forward's 256 would force above 256 local experts.
from .persistent_bf16_fwd_contract import (  # noqa: F401  (re-exported)
    COMM_TILE_SHAPES,
    DEFAULT_COMM_TILE_SHAPE,
    CommTileShape,
    EXPERT_TABLE_CAP,
    TaskDecode,
)

BWD_EXPERT_TABLE_CAP = 512


NUM_LOCAL_EXPERTS = 64
NUM_LOCAL_TOKENS = 102_400
HIDDEN_SIZE = 4_096
INTERMEDIATE_SIZE = 1_024
TOPK = 10
ROW_ALIGNMENT = 256

SCHEDULE_CAPACITY = 1_048_576
MACROBATCH_SIZE = 32_768
MINIBATCH_SIZE = 16_384
NUM_COMM_SMS = 42
FIXED_RANK0_ROUTED_ROWS = 1_032_192
FIXED_EXPERT_ROWS = tuple(16_128 for _ in range(NUM_LOCAL_EXPERTS))
FIXED_CONFIG = (SCHEDULE_CAPACITY, MACROBATCH_SIZE, MINIBATCH_SIZE, NUM_COMM_SMS)
FIXED_CONFIGS = (
    (32_768, 16_384, 42),
    (8_192, 4_096, 52),
    (16_384, 4_096, 44),
    (32_768, 4_096, 40),
)
FIXED_COMPLETED_SEGMENTS = 95

CLUSTER_SIZE = 2
THREADS_PER_CTA = 256
WARPS_PER_CTA = 8
EPILOGUE_WARPS = (0, 1, 2, 3)
UMMA_WARP = 4
CLC_WARP = 5
BF16_IDLE_WARP = 6
AB_LOAD_WARP = 7

MLP_TILE_M = 256
MLP_TILE_N = 256
MLP_TILE_K = 64
MLP_SUPERGROUP_SIZE = 8
MLP_LOAD_STAGES = 6
MLP_ACC_STAGES = 2
MLP_EPILOGUE_PIECES = 8
MLP_BF16_D_TILES = 3
SWIGLU_TILE_M = 128
SWIGLU_TILE_N = 128
# The communication task stream (``_bwd_dynamic_task_stream`` on ``_comm_stream``) moves the tiles of the forward
# contract's CommTileShape (``BwdGeometry.comm_tile_shape``; compile time).  Every communication readiness count
# (d_y_routed_ready, replayed_x_routed_ready, reverse_dispatch_done) is one arrival per tile.
CLC_DEPTH = 1
CLC_DRAIN_DEPTH = 8
REUSABLE_ARENA_BYTES = 229_376
GEMM_ARENA_BYTES = 221_184


def ceil_div(value: int, divisor: int) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("value must be a non-negative integer")
    if type(divisor) is not int or divisor <= 0:
        raise ValueError("divisor must be a positive integer")
    return (value + divisor - 1) // divisor


def selected_peer(peer_rank: int, ep_size: int) -> Optional[int]:
    """Return a valid peer in ``[0, ep_size)`` or ``None`` for padding/invalid metadata."""

    if type(peer_rank) is not int or type(ep_size) is not int or ep_size <= 0:
        raise TypeError("peer_rank and ep_size must be integers, ep_size positive")
    return peer_rank if 0 <= peer_rank < ep_size else None


def reads_peer_token_index(peer_rank: int, ep_size: int) -> bool:
    """Padding (notably -1) must never evaluate ``peer_token_idx``."""

    return selected_peer(peer_rank, ep_size) is not None


def _validate_expert_rows(rows: tuple[int, ...], num_tokens: int, experts: int) -> None:
    if not isinstance(rows, tuple) or len(rows) != experts:
        raise ValueError("tokens_per_expert must contain E entries")
    if any(type(value) is not int or value < 0 or value % ROW_ALIGNMENT for value in rows):
        raise ValueError("expert rows must be nonnegative 256-row multiples")
    if sum(rows) != num_tokens:
        raise ValueError("num_tokens must equal the padded tokens_per_expert sum")


@dataclass(frozen=True)
class BwdGeometry:
    """Host allocation geometry only; never contains device schedule values."""

    schedule_capacity: int = SCHEDULE_CAPACITY
    num_local_tokens: int = NUM_LOCAL_TOKENS
    macrobatch_size: int = MACROBATCH_SIZE
    minibatch_size: int = MINIBATCH_SIZE
    num_comm_sms: int = NUM_COMM_SMS
    num_local_experts: int = NUM_LOCAL_EXPERTS
    hidden_size: int = HIDDEN_SIZE  # shared chain (x / d_y_shared / d_x_shared)
    intermediate_size: int = INTERMEDIATE_SIZE
    topk: int = TOPK
    swiglu_limit: Optional[float] = None
    ep_size: int = 8  # peer count of the workspace: runtime table length, not a device constant
    routed_hidden_size: Optional[int] = None  # routed chain (rings, communication, routed weights); None = hidden_size
    # The forward context holds no routed activations: the first macrobatch is replayed like the later ones.
    recompute_forward_context: bool = False
    # The communication tile shape; selects the kernel variant.
    comm_tile_shape: CommTileShape = DEFAULT_COMM_TILE_SHAPE

    def __post_init__(self) -> None:
        if self.routed_hidden_size is None:
            object.__setattr__(self, "routed_hidden_size", self.hidden_size)
        if self.comm_tile_shape not in COMM_TILE_SHAPES:
            raise ValueError(f"comm_tile_shape must be one of {COMM_TILE_SHAPES}; got {self.comm_tile_shape!r}")
        for name, value in (
            ("ep_size", self.ep_size),
            ("schedule_capacity", self.schedule_capacity),
            ("num_local_tokens", self.num_local_tokens),
            ("macrobatch_size", self.macrobatch_size),
            ("minibatch_size", self.minibatch_size),
            ("num_comm_sms", self.num_comm_sms),
            ("num_local_experts", self.num_local_experts),
            ("hidden_size", self.hidden_size),
            ("intermediate_size", self.intermediate_size),
            ("topk", self.topk),
            ("routed_hidden_size", self.routed_hidden_size),  # last: it defaults to hidden_size
        ):
            if type(value) is not int or not 0 < value <= 0x7FFFFFFF:
                raise ValueError(f"{name} must be a positive CUDA int32 value")
        for name in ("num_local_tokens", "minibatch_size", "macrobatch_size", "schedule_capacity"):
            if getattr(self, name) % ROW_ALIGNMENT:
                raise ValueError(f"{name} must be 256-aligned for the CUDA tile ABI; got {getattr(self, name)}")
        # The hidden extent may have a 128-wide tail, like the intermediate one below: its last tile is
        # counted (ceil) and its out-of-range half zero-filled / clipped by the TMA.  A routed hidden
        # that differs from the shared one (latent layers) stays 256-aligned.
        if self.hidden_size < 256 or self.hidden_size % 128:
            raise ValueError(f"hidden_size must be at least 256 and a multiple of 128; got {self.hidden_size}")
        if self.routed_hidden_size != self.hidden_size and self.routed_hidden_size % ROW_ALIGNMENT:
            raise ValueError(f"routed_hidden_size must be 256-aligned; got {self.routed_hidden_size}")
        # The intermediate extent is tiled at MLP_TILE_N=256 by the dgrad/wgrad GEMMs, but a
        # 128-multiple is admissible: the last I tile is a predicated tail.
        if self.intermediate_size < 256 or self.intermediate_size % 128:
            raise ValueError("intermediate_size must be at least 256 and a multiple of 128; got " f"{self.intermediate_size}")
        if self.macrobatch_size % self.minibatch_size:
            raise ValueError("macrobatch must contain whole minibatches; got " f"{self.macrobatch_size}/{self.minibatch_size}")
        # The forward's optional clamp domain: zero is clamped; only None disables clamping.
        if self.swiglu_limit is not None and (type(self.swiglu_limit) not in (int, float) or self.swiglu_limit < 0):
            raise ValueError("swiglu_limit must be None or a non-negative number")
        if self.num_local_tokens * self.topk > 0x7FFFFFFF:
            raise ValueError("flattened route indices exceed CUDA int32 ABI")
        if self.num_comm_sms % CLUSTER_SIZE:
            raise ValueError(f"num_comm_sms must be even for cluster size two; got {self.num_comm_sms}")
        plan = TaskPlan(self)
        if plan.cta_grid > 0x7FFFFFFF:
            raise ValueError("capacity task grid exceeds CUDA int32 ABI")
        # Python integer products are wide before checking the device int32 ABI.
        lengths = [self.segment_capacity, plan.task_owner_bucket_count, self.capacity_minibatches + self.num_local_experts - 1]
        lengths.extend(item.shape[0] for item in counter_descriptors(plan))
        if any(length > 0x7FFFFFFF for length in lengths):
            raise ValueError("metadata/counter capacity exceeds CUDA int32 ABI")

    @property
    def capacity_minibatches(self) -> int:
        return ceil_div(self.schedule_capacity, self.minibatch_size)

    @property
    def build_o1_decode(self) -> bool:
        # From the capacity, NOT the device's actual routed row count.  recompute_forward_context always takes the
        # per-generation task stream, which reads the decoder tables.
        return self.recompute_forward_context or ceil_div(self.schedule_capacity, self.macrobatch_size) > 1

    @property
    def minibatches_per_macrobatch(self) -> int:
        return self.macrobatch_size // self.minibatch_size

    @property
    def saved_minibatches(self) -> int:
        """Minibatches whose routed X / Gate / Up come from the forward context (not replayed)."""
        return 0 if self.recompute_forward_context else self.minibatches_per_macrobatch

    @property
    def segment_capacity(self) -> int:
        # Maximum number of completed expert/macrobatch segments; no E64 ceiling.
        return ceil_div(self.schedule_capacity, self.macrobatch_size) + self.num_local_experts - 1

    @property
    def task_owner_bucket_width(self) -> int:
        return 1 << (task_counts(self).bwd_per_minibatch.bit_length() - 1)

    @property
    def task_owner_bucket_shift(self) -> int:
        return self.task_owner_bucket_width.bit_length() - 1 if self.build_o1_decode else 0

    @property
    def d_y_ready_required(self) -> int:
        return self.comm_tile_shape.tiles(self.minibatch_size, self.routed_hidden_size)

    @property
    def d_x_ready_required(self) -> int:
        return (self.minibatch_size // MLP_TILE_M) * ceil_div(self.routed_hidden_size, MLP_TILE_N) * CLUSTER_SIZE

    @property
    def routed_buffers_done_required(self) -> int:
        return CLUSTER_SIZE * task_counts(self).bwd_per_minibatch

    @property
    def reverse_dispatch_done_required(self) -> int:
        return self.d_y_ready_required  # one arrive per pushed dX tile of a full minibatch

    @property
    def comm_clusters(self) -> int:
        return self.num_comm_sms // CLUSTER_SIZE


@dataclass(frozen=True)
class BwdConfig(BwdGeometry):
    """CPU oracle with actual values; the public device path uses BwdGeometry."""

    num_tokens: int = FIXED_RANK0_ROUTED_ROWS
    tokens_per_expert: tuple[int, ...] = FIXED_EXPERT_ROWS

    def __post_init__(self) -> None:
        super().__post_init__()
        if type(self.num_tokens) is not int or self.num_tokens < 0:
            raise ValueError("num_tokens must be nonnegative")
        if self.num_tokens > self.schedule_capacity:
            raise ValueError("num_tokens exceeds schedule capacity")
        if self.num_tokens % ROW_ALIGNMENT:
            raise ValueError("num_tokens must be 256-row aligned")
        _validate_expert_rows(self.tokens_per_expert, self.num_tokens, self.num_local_experts)

    @property
    def schedule_rows(self) -> int:
        return self.num_tokens

    @property
    def num_minibatches(self) -> int:
        return ceil_div(self.num_tokens, self.minibatch_size)

    @property
    def num_macrobatches(self) -> int:
        return ceil_div(self.num_tokens, self.macrobatch_size)

    @property
    def use_minibatch_pipeline(self) -> bool:
        # recompute_forward_context: every generation has replay tasks, so any routed row takes the per-generation
        # stream.
        return self.num_tokens > 0 if self.recompute_forward_context else self.num_macrobatches > 1

    def minibatch_rows(self, generation: int) -> int:
        if type(generation) is not int or not 0 <= generation < self.num_minibatches:
            raise ValueError("generation is outside the true routed prefix")
        return min(
            self.minibatch_size,
            self.num_tokens - generation * self.minibatch_size,
        )

    def macrobatch_rows(self, macrobatch: int) -> int:
        if type(macrobatch) is not int or not 0 <= macrobatch < self.num_macrobatches:
            raise ValueError("macrobatch is outside the true routed prefix")
        return min(
            self.macrobatch_size,
            self.num_tokens - macrobatch * self.macrobatch_size,
        )

    def ring_row(self, global_row: int) -> int:
        if type(global_row) is not int or not 0 <= global_row < self.num_tokens:
            raise ValueError("global row is outside the true routed prefix")
        return global_row % self.macrobatch_size

    def minibatch_ring_begin(self, generation: int) -> int:
        self.minibatch_rows(generation)
        return (generation % self.minibatches_per_macrobatch) * self.minibatch_size


@dataclass(frozen=True)
class ExpertMacroSegment:
    ordinal: int
    expert_index: int
    macrobatch_index: int
    global_begin: int
    global_end: int
    completion_generation: int
    sole_contribution: bool
    segment_index: int = 0  # k-th macrobatch of the expert (0 = holds its first row; 0 if empty)

    @property
    def rows(self) -> int:
        return self.global_end - self.global_begin

    @property
    def store_add(self) -> bool:
        """Segment 0 stores dW (also the all-zero store of an empty expert); later
        segments reduce-add after segment k-1's store."""
        return self.segment_index > 0


@lru_cache(maxsize=32)
def expert_offsets(config: BwdConfig) -> tuple[int, ...]:
    offsets = [0]
    for rows in config.tokens_per_expert:
        offsets.append(offsets[-1] + rows)
    return tuple(offsets)


@lru_cache(maxsize=32)
def expert_macro_segments(config: BwdConfig) -> tuple[ExpertMacroSegment, ...]:
    """Return the expert/macrobatch intersections in completion order."""

    offsets = expert_offsets(config)
    raw: list[tuple[int, int, int, int, int]] = []
    # Experts without rows on this rank are registered once, in macrobatch 0 /
    # generation 0 (by the metadata builder's minibatch-0 pass, hence only when a
    # minibatch exists), with an empty range: their Wgrad tasks store zero tiles
    # (the kernel writes every routed dW itself; there is no external zero kernel).
    # With no routed rows at all the kernel uses the dense per-expert Wgrad grid.
    if config.num_minibatches >= 1:
        for expert, (expert_begin, expert_end) in enumerate(zip(offsets, offsets[1:])):
            if expert_begin == expert_end:
                raw.append((0, expert, 0, expert_begin, expert_end, 0))
    for macrobatch in range(config.num_macrobatches):
        macro_begin = macrobatch * config.macrobatch_size
        macro_end = min(macro_begin + config.macrobatch_size, config.num_tokens)
        for expert, (expert_begin, expert_end) in enumerate(zip(offsets, offsets[1:])):
            begin = max(expert_begin, macro_begin)
            end = min(expert_end, macro_end)
            if begin < end:
                raw.append(
                    (
                        (end - 1) // config.minibatch_size,
                        expert,
                        macrobatch,
                        begin,
                        end,
                        macrobatch - expert_begin // config.macrobatch_size,
                    )
                )
    raw.sort(key=lambda item: (item[0], item[2], item[1]))
    contributions = Counter(expert for _, expert, _, _, _, _ in raw)
    return tuple(
        ExpertMacroSegment(
            ordinal=ordinal,
            completion_generation=generation,
            expert_index=expert,
            macrobatch_index=macrobatch,
            global_begin=begin,
            global_end=end,
            sole_contribution=contributions[expert] == 1,
            segment_index=segment_index,
        )
        for ordinal, (generation, expert, macrobatch, begin, end, segment_index) in enumerate(raw)
    )


@lru_cache(maxsize=32)
def completed_segment_offsets(config: BwdConfig) -> tuple[int, ...]:
    counts = Counter(segment.completion_generation for segment in expert_macro_segments(config))
    result = [0]
    for generation in range(config.capacity_minibatches):
        result.append(result[-1] + counts[generation])
    return tuple(result)


@lru_cache(maxsize=32)
def completed_segment_experts(config: BwdConfig) -> tuple[int, ...]:
    return tuple(segment.expert_index for segment in expert_macro_segments(config))


@lru_cache(maxsize=8192)
def active_experts_for_minibatch(config: BwdConfig, generation: int) -> tuple[int, ...]:
    rows = config.minibatch_rows(generation)
    begin = generation * config.minibatch_size
    end = begin + rows
    offsets = expert_offsets(config)
    return tuple(expert for expert, (expert_begin, expert_end) in enumerate(zip(offsets, offsets[1:])) if max(begin, expert_begin) < min(end, expert_end))


@dataclass(frozen=True)
class TaskCounts:
    shared_down_dgrad: int
    shared_gate_up_dgrad: int
    shared_wgrad_down: int
    shared_wgrad_gate: int
    shared_wgrad_up: int
    routed_down_dgrad: int
    routed_gate_up_dgrad: int
    replay_gate_up: int
    wgrad_tile_tasks: int
    num_local_experts: int

    @property
    def shared_total(self) -> int:
        return self.shared_down_dgrad + self.shared_gate_up_dgrad + self.shared_wgrad_down + self.shared_wgrad_gate + self.shared_wgrad_up

    @property
    def bwd_per_minibatch(self) -> int:
        return self.routed_down_dgrad + self.routed_gate_up_dgrad

    @property
    def routed_wgrad_per_matrix(self) -> int:
        return self.num_local_experts * self.wgrad_tile_tasks

    @property
    def routed_wgrad_total(self) -> int:
        return 3 * self.routed_wgrad_per_matrix

    @property
    def replay_per_minibatch(self) -> int:
        # Replay Gate+Up only: the routed Down dgrad epilogue writes w*Hidden itself.
        return self.replay_gate_up

    @property
    def wgrad_per_segment(self) -> int:
        return 3 * self.wgrad_tile_tasks


def task_counts(config: BwdGeometry) -> TaskCounts:
    shared_rows = config.num_local_tokens // MLP_TILE_M
    mini_rows = config.minibatch_size // MLP_TILE_M
    i_cols = ceil_div(config.intermediate_size, MLP_TILE_N)
    shared_h_cols = ceil_div(config.hidden_size, MLP_TILE_N)
    routed_h_cols = ceil_div(config.routed_hidden_size, MLP_TILE_N)
    shared_wgrad_tiles = i_cols * shared_h_cols
    routed_wgrad_tiles = i_cols * routed_h_cols
    return TaskCounts(
        shared_down_dgrad=shared_rows * i_cols,
        shared_gate_up_dgrad=shared_rows * shared_h_cols,
        shared_wgrad_down=shared_wgrad_tiles,
        shared_wgrad_gate=shared_wgrad_tiles,
        shared_wgrad_up=shared_wgrad_tiles,
        routed_down_dgrad=mini_rows * i_cols,
        routed_gate_up_dgrad=mini_rows * routed_h_cols,
        replay_gate_up=(1 if config.build_o1_decode else 2) * mini_rows * i_cols,
        wgrad_tile_tasks=routed_wgrad_tiles,
        num_local_experts=config.num_local_experts,
    )


class TaskFamily(str, Enum):
    # SwiGLU-backward has no task family: it is fused into the Down dgrad epilogues (the device
    # kind IDs 1 and 8 are unused).
    SHARED_DOWN_DGRAD = "shared_down_dgrad"
    SHARED_GATE_UP_DGRAD = "shared_gate_up_dgrad"
    SHARED_WGRAD_DOWN = "shared_wgrad_down"
    SHARED_WGRAD_GATE = "shared_wgrad_gate"
    SHARED_WGRAD_UP = "shared_wgrad_up"
    REPLAY_GATE_UP = "replay_gate_up"
    ROUTED_DOWN_DGRAD = "routed_down_dgrad"
    ROUTED_GATE_UP_DGRAD = "routed_gate_up_dgrad"
    ROUTED_WGRAD_DOWN = "routed_wgrad_down"
    ROUTED_WGRAD_GATE = "routed_wgrad_gate"
    ROUTED_WGRAD_UP = "routed_wgrad_up"


TASK_FAMILY_ORDER = tuple(TaskFamily)


@dataclass(frozen=True)
class TaskClaim:
    global_index: int
    family: TaskFamily
    task_index: int
    generation: Optional[int] = None
    row_tile: Optional[int] = None
    column_tile: Optional[int] = None
    expert_index: Optional[int] = None
    macrobatch_index: Optional[int] = None
    segment_ordinal: Optional[int] = None
    k_start: Optional[int] = None
    k_end: Optional[int] = None
    valid: bool = True
    wgrad_store_add: bool = False


def swizzled_tile_coord(
    row_blocks: int,
    column_blocks: int,
    task_index: int,
) -> tuple[int, int]:
    """Grouped-GEMM row swizzle with an eight-column supergroup."""

    if type(row_blocks) is not int or row_blocks <= 0:
        raise ValueError("row_blocks must be positive")
    if type(column_blocks) is not int or column_blocks <= 0:
        raise ValueError("column_blocks must be positive")
    if type(task_index) is not int or not 0 <= task_index < row_blocks * column_blocks:
        raise ValueError("task_index is outside the tile grid")
    supergroup_numel = row_blocks * MLP_SUPERGROUP_SIZE
    supergroup = task_index // supergroup_numel
    full_columns = (column_blocks // MLP_SUPERGROUP_SIZE) * MLP_SUPERGROUP_SIZE
    full_numel = row_blocks * full_columns
    if task_index < full_numel:
        row = (task_index % supergroup_numel) // MLP_SUPERGROUP_SIZE
        column = supergroup * MLP_SUPERGROUP_SIZE + task_index % MLP_SUPERGROUP_SIZE
    else:
        remaining_columns = column_blocks - full_columns
        remainder = task_index - full_numel
        row = remainder // remaining_columns
        column = full_columns + remainder % remaining_columns
    if supergroup % 2:
        row = row_blocks - row - 1
    return row, column


def _routed_gemm_coord(
    config: BwdConfig,
    generation: int,
    task_index: int,
    column_blocks: int,
) -> tuple[bool, Optional[int], Optional[int], Optional[int]]:
    mini_begin = generation * config.minibatch_size
    mini_end = min(mini_begin + config.minibatch_size, config.num_tokens)
    remaining = task_index
    offsets = expert_offsets(config)
    for expert, (expert_begin, expert_end) in enumerate(zip(offsets, offsets[1:])):
        first = max(expert_begin, mini_begin) // MLP_TILE_M
        end = min(expert_end, mini_end) // MLP_TILE_M
        row_blocks = max(0, end - first)
        expert_tasks = row_blocks * column_blocks
        if remaining < expert_tasks:
            row, column = swizzled_tile_coord(row_blocks, column_blocks, remaining)
            return True, expert, first + row, column
        remaining -= expert_tasks
    return False, None, None, None


def _wgrad_coord(config: BwdConfig, family: TaskFamily, tile: int) -> tuple[int, int]:
    shared = family in (TaskFamily.SHARED_WGRAD_DOWN, TaskFamily.SHARED_WGRAD_GATE, TaskFamily.SHARED_WGRAD_UP)
    hidden = config.hidden_size if shared else config.routed_hidden_size
    if family in (TaskFamily.SHARED_WGRAD_DOWN, TaskFamily.ROUTED_WGRAD_DOWN):
        return swizzled_tile_coord(
            ceil_div(hidden, MLP_TILE_N),
            ceil_div(config.intermediate_size, MLP_TILE_N),
            tile,
        )
    return swizzled_tile_coord(
        ceil_div(config.intermediate_size, MLP_TILE_N),
        ceil_div(hidden, MLP_TILE_N),
        tile,
    )


class TaskPlan:
    """Geometry supplies capacities; actual task/oracle methods need BwdConfig."""

    def __init__(self, config: BwdGeometry):
        self.config = config
        self.counts = task_counts(config)

    @property
    def compute_tasks(self) -> int:
        if not self.config.use_minibatch_pipeline:
            # Dense per-expert Wgrad grid, also with no routed rows (every expert is
            # then empty and its Wgrad tasks write the zero dW tiles).
            return (
                self.counts.shared_total
                + self.config.num_minibatches * self.counts.bwd_per_minibatch
                + max(1, self.config.num_macrobatches) * self.counts.routed_wgrad_total
            )
        return (
            self.counts.shared_total
            + self.config.num_minibatches * self.counts.bwd_per_minibatch
            + max(0, self.config.num_minibatches - self.config.saved_minibatches) * self.counts.replay_per_minibatch
            + len(expert_macro_segments(self.config)) * self.counts.wgrad_per_segment
        )

    @property
    def capacity_compute_tasks(self) -> int:
        return (
            self.counts.shared_total
            + self.config.capacity_minibatches * self.counts.bwd_per_minibatch
            + max(0, self.config.capacity_minibatches - self.config.saved_minibatches) * self.counts.replay_per_minibatch
            + self.config.segment_capacity * self.counts.wgrad_per_segment
        )

    @property
    def capacity_tail_tasks(self) -> int:
        return self.capacity_compute_tasks - self.compute_tasks

    @property
    def logical_clusters(self) -> int:
        return self.config.comm_clusters + self.compute_tasks

    @property
    def capacity_logical_clusters(self) -> int:
        return self.config.comm_clusters + self.capacity_compute_tasks

    @property
    def cta_grid(self) -> int:
        return self.config.num_comm_sms + CLUSTER_SIZE * self.capacity_compute_tasks

    @property
    def true_cta_grid(self) -> int:
        return self.config.num_comm_sms + CLUSTER_SIZE * self.compute_tasks

    def generation_prefix(self, generation: int) -> int:
        """Routed-task prefix before one global minibatch, excluding shared."""

        if type(generation) is not int or not 0 <= generation <= self.config.num_minibatches:
            raise ValueError("generation is outside the routed task stream")
        segment_prefix = completed_segment_offsets(self.config)[generation]
        replay_minibatches = max(0, generation - self.config.saved_minibatches)
        return (
            generation * self.counts.bwd_per_minibatch + replay_minibatches * self.counts.replay_per_minibatch + segment_prefix * self.counts.wgrad_per_segment
        )

    def generation_task_count(self, generation: int) -> int:
        if type(generation) is not int or not 0 <= generation < self.config.num_minibatches:
            raise ValueError("generation is outside the routed task stream")
        segment_offsets = completed_segment_offsets(self.config)
        return (
            self.counts.bwd_per_minibatch
            + (self.counts.replay_per_minibatch if generation >= self.config.saved_minibatches else 0)
            + (segment_offsets[generation + 1] - segment_offsets[generation]) * self.counts.wgrad_per_segment
        )

    def capacity_generation_prefix(self, generation: int) -> int:
        if type(generation) is not int or not 0 <= generation <= self.config.capacity_minibatches:
            raise ValueError("generation is outside the capacity task stream")
        segment_prefix = completed_segment_offsets(self.config)[generation]
        return (
            generation * self.counts.bwd_per_minibatch
            + max(0, generation - self.config.saved_minibatches) * self.counts.replay_per_minibatch
            + segment_prefix * self.counts.wgrad_per_segment
        )

    @property
    def task_owner_bucket_count(self) -> int:
        if not self.config.build_o1_decode:
            return 1  # the num_tokens alias, not an allocated owner table
        return max(
            1,
            ceil_div(
                self.capacity_compute_tasks - self.counts.shared_total,
                self.config.task_owner_bucket_width,
            ),
        )

    def task_owner_buckets(self) -> tuple[int, ...]:
        if not self.config.build_o1_decode:
            return (self.config.num_tokens,)
        # Only actual owners, then a constant capacity tail.
        n = self.config.num_minibatches
        routed_tasks = self.generation_prefix(n)
        owners = []
        owner = 0
        for task in range(0, routed_tasks, self.config.task_owner_bucket_width):
            while owner + 1 < n and self.generation_prefix(owner + 1) <= task:
                owner += 1
            owners.append(owner)
        owners.extend([max(0, n - 1)] * (self.task_owner_bucket_count - len(owners)))
        return tuple(owners)

    def _claim(
        self,
        global_index: int,
        family: TaskFamily,
        task_index: int,
        generation: Optional[int] = None,
        *,
        expert_index: Optional[int] = None,
        segment: Optional[ExpertMacroSegment] = None,
        k_start: Optional[int] = None,
        k_end: Optional[int] = None,
    ) -> TaskClaim:
        row = column = None
        valid = True
        if family == TaskFamily.SHARED_DOWN_DGRAD:
            row, column = swizzled_tile_coord(
                self.config.num_local_tokens // MLP_TILE_M,
                ceil_div(self.config.intermediate_size, MLP_TILE_N),
                task_index,
            )
            expert_index = 0
        elif family == TaskFamily.SHARED_GATE_UP_DGRAD:
            row, column = swizzled_tile_coord(
                self.config.num_local_tokens // MLP_TILE_M,
                ceil_div(self.config.hidden_size, MLP_TILE_N),
                task_index,
            )
            expert_index = 0
        elif family in (
            TaskFamily.SHARED_WGRAD_DOWN,
            TaskFamily.SHARED_WGRAD_GATE,
            TaskFamily.SHARED_WGRAD_UP,
        ):
            row, column = _wgrad_coord(self.config, family, task_index)
            expert_index = 0
            k_start, k_end = 0, self.config.num_local_tokens
        elif family in (
            TaskFamily.ROUTED_DOWN_DGRAD,
            TaskFamily.ROUTED_GATE_UP_DGRAD,
        ):
            assert generation is not None
            columns = (
                ceil_div(self.config.routed_hidden_size, MLP_TILE_N)
                if family == TaskFamily.ROUTED_GATE_UP_DGRAD
                else ceil_div(self.config.intermediate_size, MLP_TILE_N)
            )
            valid, expert_index, row, column = _routed_gemm_coord(
                self.config,
                generation,
                task_index,
                columns,
            )
        elif family in (
            TaskFamily.ROUTED_WGRAD_DOWN,
            TaskFamily.ROUTED_WGRAD_GATE,
            TaskFamily.ROUTED_WGRAD_UP,
        ):
            row, column = _wgrad_coord(self.config, family, task_index)
            assert segment is not None
            expert_index = segment.expert_index
            k_start, k_end = segment.global_begin, segment.global_end
            valid = k_start < k_end

        return TaskClaim(
            global_index=global_index,
            family=family,
            task_index=task_index,
            generation=generation,
            row_tile=row,
            column_tile=column,
            expert_index=expert_index,
            macrobatch_index=(None if segment is None else segment.macrobatch_index),
            segment_ordinal=None if segment is None else segment.ordinal,
            k_start=k_start,
            k_end=k_end,
            valid=valid,
            wgrad_store_add=False if segment is None else segment.store_add,
        )

    @staticmethod
    def _routed_wgrad_family(local: int, matrix_tasks: int) -> tuple[TaskFamily, int]:
        """Routed Wgrad order of one expert set (``matrix_tasks`` dW tiles per matrix): the Down
        tiles first, then Gate and Up tile t of an expert next to each other (they read the same
        X slab): pair p -> Gate if p is even, Up if odd, tile index p // 2."""
        if local < matrix_tasks:
            return TaskFamily.ROUTED_WGRAD_DOWN, local
        pair = local - matrix_tasks
        if pair >= 2 * matrix_tasks:
            raise AssertionError("routed Wgrad decoder exhausted no family")
        return (TaskFamily.ROUTED_WGRAD_GATE, TaskFamily.ROUTED_WGRAD_UP)[pair & 1], pair >> 1

    @cache
    def decode(self, compute_index: int) -> TaskClaim:
        """Python model of the kernel's task stream (``_decode_compute_task``)."""
        if type(compute_index) is not int or not 0 <= compute_index < self.compute_tasks:
            raise ValueError("compute_index is outside the true task stream")
        counts = self.counts
        remaining = compute_index
        shared = (
            (TaskFamily.SHARED_DOWN_DGRAD, counts.shared_down_dgrad),
            (TaskFamily.SHARED_GATE_UP_DGRAD, counts.shared_gate_up_dgrad),
            (TaskFamily.SHARED_WGRAD_DOWN, counts.shared_wgrad_down),
            (TaskFamily.SHARED_WGRAD_GATE, counts.shared_wgrad_gate),
            (TaskFamily.SHARED_WGRAD_UP, counts.shared_wgrad_up),
        )
        for family, count in shared:
            if remaining < count:
                return self._claim(compute_index, family, remaining)
            remaining -= count

        if not self.config.use_minibatch_pipeline:
            minis = self.config.num_minibatches
            down = counts.routed_down_dgrad
            bwd_tasks = minis * counts.bwd_per_minibatch
            if remaining < bwd_tasks:
                # Down dgrad of minibatch s + 1 ahead of Gate/Up dgrad of s: Down(0), then per s
                # [Down(s + 1), Gate/Up(s)]; the last block has Gate/Up only (same task count).
                if remaining < down:
                    return self._claim(compute_index, TaskFamily.ROUTED_DOWN_DGRAD, remaining, 0)
                generation, local = divmod(remaining - down, counts.bwd_per_minibatch)
                if generation + 1 < minis:
                    if local < down:
                        return self._claim(compute_index, TaskFamily.ROUTED_DOWN_DGRAD, local, generation + 1)
                    local -= down
                return self._claim(compute_index, TaskFamily.ROUTED_GATE_UP_DGRAD, local, generation)
            # Dense per-expert Wgrad grid: Down tiles of every expert, then the Gate/Up pairs.
            family, local = self._routed_wgrad_family(remaining - bwd_tasks, counts.routed_wgrad_per_matrix)
            expert, tile = divmod(local, counts.wgrad_tile_tasks)
            offsets = expert_offsets(self.config)
            segment = ExpertMacroSegment(expert, expert, 0, offsets[expert], offsets[expert + 1], 0, True)
            return self._claim(compute_index, family, tile, 0, segment=segment)

        generation = next(generation for generation in range(self.config.num_minibatches) if remaining < self.generation_prefix(generation + 1))
        local = remaining - self.generation_prefix(generation)
        if generation >= self.config.saved_minibatches:
            for family, count in ((TaskFamily.REPLAY_GATE_UP, counts.replay_gate_up),):
                if local < count:
                    return self._claim(compute_index, family, local, generation)
                local -= count
        for family, count in (
            (TaskFamily.ROUTED_DOWN_DGRAD, counts.routed_down_dgrad),
            (TaskFamily.ROUTED_GATE_UP_DGRAD, counts.routed_gate_up_dgrad),
        ):
            if local < count:
                return self._claim(compute_index, family, local, generation)
            local -= count

        # Wgrad of the segments completed by this generation: Down tiles of every segment, then
        # the Gate/Up pairs (segment lane = tile index // tiles per matrix).
        segment_offsets = completed_segment_offsets(self.config)
        first_segment = segment_offsets[generation]
        segment_count = segment_offsets[generation + 1] - first_segment
        segments = expert_macro_segments(self.config)
        family, local = self._routed_wgrad_family(local, segment_count * counts.wgrad_tile_tasks)
        segment_lane, tile = divmod(local, counts.wgrad_tile_tasks)
        return self._claim(
            compute_index,
            family,
            tile,
            generation,
            segment=segments[first_segment + segment_lane],
        )


# The backward's per-call tensors: seven scratch tensors (0-6) and the six weight gradients (7-12).
# There is no d_hidden_shared / d_hidden_routed: the SwiGLU backward runs in the Down dgrad
# epilogue and dH stays in registers.
RAW13_NAMES = (
    "d_x_shared",
    "d_x_routed",
    "d_gate_shared",
    "d_gate_routed",
    "d_up_shared",
    "d_up_routed",
    "d_y_routed",
    "d_w_shared_gate",
    "d_w_routed_gate",
    "d_w_shared_up",
    "d_w_routed_up",
    "d_w_shared_down",
    "d_w_routed_down",
)

COUNTER_NAMES = (
    "d_y_routed_ready",
    "d_gate_up_ready",
    "d_x_routed_ready",
    "replayed_x_routed_ready",
    "replayed_gate_up_ready",
    "routed_buffers_done",
    "wgrad_read_consumed",
    "wgrad_segment_done",
    "reverse_dispatch_done",
)


@dataclass(frozen=True)
class TensorDescriptor:
    name: str
    shape: tuple[int, ...]
    dtype: str


def raw13_descriptors(config: BwdGeometry) -> tuple[TensorDescriptor, ...]:
    t, b, e, h, r, i = (
        config.num_local_tokens,
        config.macrobatch_size,
        config.num_local_experts,
        config.hidden_size,  # shared chain
        config.routed_hidden_size,  # routed chain
        config.intermediate_size,
    )
    shapes = (
        (t, h),
        (b, r),
        (t, i),
        (b, i),
        (t, i),
        (b, i),
        (b, r),
        (i, h),
        (e, i, r),
        (i, h),
        (e, i, r),
        (h, i),
        (e, r, i),
    )
    return tuple(TensorDescriptor(name, shape, "bfloat16") for name, shape in zip(RAW13_NAMES, shapes))


METADATA_NAMES = (
    "minibatch_expert_offsets",
    "completed_segment_offsets",
    "completed_segment_experts",
    "task_owner_buckets",
)


def metadata_descriptors(plan: TaskPlan) -> tuple[TensorDescriptor, ...]:
    return (
        TensorDescriptor(
            "minibatch_expert_offsets",
            (plan.config.capacity_minibatches + 1,),
            "int32",
        ),
        TensorDescriptor(
            "completed_segment_offsets",
            (plan.config.capacity_minibatches + 1,),
            "int32",
        ),
        TensorDescriptor(
            "completed_segment_experts",
            (max(1, plan.config.segment_capacity) if plan.config.build_o1_decode else 1,),
            "int32",
        ),
        TensorDescriptor(
            "task_owner_buckets",
            (plan.task_owner_bucket_count,),
            "int32",
        ),
    )


def counter_descriptors(plan: TaskPlan) -> tuple[TensorDescriptor, ...]:
    config = plan.config
    shared_blocks = config.num_local_tokens // MLP_TILE_M
    capacity_blocks = config.schedule_capacity // MLP_TILE_M
    i_cols = ceil_div(config.intermediate_size, MLP_TILE_N)
    lengths = (
        config.capacity_minibatches,
        shared_blocks + capacity_blocks,
        config.capacity_minibatches,
        config.capacity_minibatches,
        capacity_blocks * i_cols,
        config.capacity_minibatches,
        config.capacity_minibatches,
        3 * config.num_local_experts * task_counts(config).wgrad_tile_tasks,
        config.capacity_minibatches + 1,  # + the communication task stream's head (_bwd_dynamic_task_stream)
    )
    return tuple(TensorDescriptor(name, (length,), "int32") for name, length in zip(COUNTER_NAMES, lengths))


__all__ = [
    "AB_LOAD_WARP",
    "BF16_IDLE_WARP",
    "BwdConfig",
    "BwdGeometry",
    "CLC_DEPTH",
    "CLC_DRAIN_DEPTH",
    "CLC_WARP",
    "CLUSTER_SIZE",
    "COMM_TILE_SHAPES",
    "CommTileShape",
    "COUNTER_NAMES",
    "ExpertMacroSegment",
    "EPILOGUE_WARPS",
    "BWD_EXPERT_TABLE_CAP",
    "EXPERT_TABLE_CAP",
    "FIXED_CONFIG",
    "FIXED_CONFIGS",
    "FIXED_EXPERT_ROWS",
    "FIXED_COMPLETED_SEGMENTS",
    "FIXED_RANK0_ROUTED_ROWS",
    "GEMM_ARENA_BYTES",
    "MACROBATCH_SIZE",
    "METADATA_NAMES",
    "MINIBATCH_SIZE",
    "MLP_ACC_STAGES",
    "MLP_BF16_D_TILES",
    "MLP_EPILOGUE_PIECES",
    "MLP_LOAD_STAGES",
    "NUM_COMM_SMS",
    "NUM_LOCAL_TOKENS",
    "RAW13_NAMES",
    "REUSABLE_ARENA_BYTES",
    "SCHEDULE_CAPACITY",
    "TASK_FAMILY_ORDER",
    "THREADS_PER_CTA",
    "TaskClaim",
    "TaskCounts",
    "TaskDecode",
    "TaskFamily",
    "TaskPlan",
    "TensorDescriptor",
    "UMMA_WARP",
    "WARPS_PER_CTA",
    "completed_segment_experts",
    "completed_segment_offsets",
    "counter_descriptors",
    "expert_macro_segments",
    "expert_offsets",
    "metadata_descriptors",
    "raw13_descriptors",
    "reads_peer_token_index",
    "selected_peer",
    "swizzled_tile_coord",
    "task_counts",
]
