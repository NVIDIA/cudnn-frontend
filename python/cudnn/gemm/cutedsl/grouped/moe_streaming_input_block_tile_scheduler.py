# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileContributor: Wentao Guo, Phuong Nguyen, Zhongbo Zhu, Subhadeep Bhattacharya, Fei Wu, Jimmy Zhang
# SPDX-License-Identifier: Apache-2.0

"""Streaming input-block tile scheduler for the MoE grouped GEMMs (2Dx3D).

A producer kernel, such as an expert-parallel dispatch, fills A while the GEMM runs.

``signal[]``: int32, one entry per cluster M tile of A.
- Entry ``t`` covers A rows ``[t * cluster tile M, (t + 1) * cluster tile M)``; experts are padded to whole M tiles.
- The rows per entry (slot rows) are ``min(cluster tile M, rows_alignment)``, from the kernel's expert row alignment.
  Here ``rows_alignment`` must be a multiple of the cluster tile M (no tail tile), so a slot is always one M tile.
- ``t`` is in ``[0, padded_offsets[E - 1] / cluster tile M)``, at most ``streaming_signal_capacity``.
- Reset to -1 before launch (never 0: any value >= 0 reads as ready).
- The producer stores ``signal[t] = t`` with a gpu-scope release once M tile ``t`` is in memory.

The GEMM runs ready M tiles in expert-consecutive batches as soon as they are ready.

Producer contract: docs/operations/moe/streaming_signal_contract.md.

Terms, with ``Q_N = ceil(N / cluster tile N)`` N tiles per M tile:
- Ready M tile: ``signal[t] >= 0``.
- ``claim_idx``: one atomic add on the counter, one (M tile, N tile) of work.
  From ``M tiles * Q_N`` on, a claim gets END.
- Expert-consecutive M batch: up to ``EXPERT_BATCH_MAX_M_TILES`` ready M tiles of one expert in a row.
  It takes the next ``m_tile_count * Q_N`` claims short side first, as the stock dynamic scheduler orders an expert: N tile
  by N tile (each N tile of B reused from L2), or, when the batch has more M tiles than N tiles, M tile by M tile (the
  M tile's A rows shared, and its C / D rows read and written whole, by the clusters running its N tiles at once).
- ``linear_tile_idx = m_tile * Q_N + n_tile``, broadcast to the cluster. END is ``total_m_tiles * Q_N``.

Example, ``Q_N = 2``, expert e0 with M tiles m0 and m1, e1 with m2; ``signal[1]`` is still -1::

    M tile t            m0 (0, e0)    m1 (1, e0)    m2 (2, e1)
    signal[t]           0             -1            2
    linear_tile_idx     0, 1          2, 3          4, 5          END = 3 * 2 = 6
    claim_idx           0, 1          4, 5          2, 3          >= 6: END

Claim 0 opens e0 {m0} and claim 2 opens e1 {m2}.
Claim 4 waits until ``signal[1] = 1``, then opens e0 {m1}.

Opening a batch: one warp, serialized across claims, scans the experts 32 at a time, one lane each.
- A lane counts its expert's ready run: one round of ``EXPERT_BATCH_MAX_M_TILES`` independent flag loads from the
  expert's first unassigned M tile, clamped to the expert's last one, and counted only below its remaining M tiles.
- A lane whose expert has no M tile left (empty, all assigned, or past the last expert) issues no loads, and a round
  where no lane has one skips them warp-wide. Its count is 0 with or without the loads, so batches are unchanged.
- The warp then picks one expert (the one opened last if its run ends it or fills the cap, else the longest run)
  and, with no ready M tile anywhere, scans again.
- The same opening also publishes, after that batch and in lane order, one batch per other lane whose best expert has
  a complete ready run (it ends the expert or fills the cap), and one release hands them all to the claimers.  With no
  other complete run, it publishes the chosen batch alone.  Openings are serialized, so one batch per opening would
  starve the SMs with many small experts (128 experts of 3 M tiles each, every tile ready: 12% to 43% slower than the
  stock dynamic scheduler, against 1% to 3% this way).

Assumes A rows <= (2^31 - 1) * cluster tile M.
"""

from typing import List, Optional, Tuple
import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import (
    Boolean,
    Int32,
    extract_mlir_values,
    new_from_mlir_values,
    const_expr,
)
from cutlass.cute.typing import AddressSpace
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, nvvm
import cutlass.pipeline as pipeline
from cutlass.pipeline import Agent, CooperativeGroup, PipelineUserType, make_pipeline_state
from .moe_kernel_helpers import (
    check_dynamic_sched_cluster_shape,
    streaming_scheduler_state_offset,
    streaming_signal_ptr,
)
from .moe_utils import ceil_div, reject_pdl_behind_helper
from cudnn.frost.tile_dsl.consume import ReleaseZero, consume_then_arrive_on_peer
from .moe_persistent_scheduler import (
    atomic_add_i32,
    MoEPersistentTileScheduler,
    MoESchedulerParams,
    MoEWorkTileInfo,
    _DynamicSchedState,
)

#: Most M-tiles in one expert-consecutive-M batch: its A rows must stay in L2 beside the N-tile
#: of weights they multiply.
EXPERT_BATCH_MAX_M_TILES = 16

#: Named barrier id of the batch opener's publish (its one warp).  Every kernel on this scheduler keeps its own named
#: barriers in ids 1-8 (the SM90 kernels use all of them), so 9 collides with none.
OPENER_PUBLISH_BARRIER_ID = 9


def _state_region_offsets(streaming_signal_capacity: int, local_expert_count: int) -> dict:
    """Int32 offset of each region of the scheduler state, and the total.  The reset is all zero; a batch record
    needs none, being written before ``assigned_claim_count`` exposes it."""
    # A word per expert, then assigned_claim_count; a batch record is 3 words per M-tile; then, per expert,
    # 1 + the first claim index of its last batch (0: none).
    expert_last_batch = local_expert_count + 1 + 3 * streaming_signal_capacity
    return {
        "expert_assigned_m_tiles": 0,
        "assigned_claim_count": local_expert_count,
        "expert_consecutive_m_batch_record": local_expert_count + 1,
        "expert_last_batch": expert_last_batch,
        "total": expert_last_batch + local_expert_count,
    }


def check_linear_tile_idx_range(intermediate, cluster_tile_n: int, streaming_signal_capacity) -> None:
    """Build-time bounds on a linear tile index (M tile * Q_N + N tile): N > 0, and every claim index, END included,
    below 2^31.  The batch opener scores an expert ``EXPERT_BATCH_MAX_M_TILES + 1 + claim index``, so the bound leaves
    that much room too."""
    assert isinstance(intermediate, int), f"the signal scheduler needs a build-time N (dense B), not {intermediate}"
    assert intermediate > 0, f"the signal scheduler needs N > 0 (a linear tile index // Q_N), got N = {intermediate}"
    assert isinstance(streaming_signal_capacity, int), f"streaming_signal_capacity must be a build-time int, not {streaming_signal_capacity}"
    n_tiles_per_m_tile = -(-intermediate // cluster_tile_n)  # host-side: moe_utils.ceil_div is @cute.jit
    assert (
        streaming_signal_capacity * n_tiles_per_m_tile < 2**31 - EXPERT_BATCH_MAX_M_TILES - 1
    ), f"streaming_signal_capacity {streaming_signal_capacity} x Q_N {n_tiles_per_m_tile} claim indices overflow the batch opener's int32 score"


def streaming_signal_workspace_bytes_after_counter(streaming_signal_capacity: int, local_expert_count: int) -> int:
    """Workspace bytes past the 4-byte claim counter: ``signal[]``, the padding to a 128-byte
    line, then the scheduler state (see ``_state_region_offsets``)."""
    return streaming_scheduler_state_offset(streaming_signal_capacity) - 4 + 4 * _state_region_offsets(streaming_signal_capacity, local_expert_count)["total"]


@cute.jit
def _spin_until_at_least(ptr, floor: Int32) -> Int32:
    """Block, without bound, until ``*ptr >= floor`` and return it, acquiring the writer's earlier stores.
    One load point in the loop: a load ahead of it would make the compiler duplicate the loop."""
    value = Int32(floor - 1)
    while value < floor:
        value = cute.arch.load(ptr.llvm_ptr, Int32, sem="acquire", scope="gpu")
    return value


class MoEInputBlockSignalSchedulerParams:
    """Parameters for MoE tile scheduler, used on host (grid shape) and on device: expert_shape is
    (local_expert_count, intermediate, hidden), a GEMM (M=expert_i_row_count, N=intermediate, K=hidden) per expert."""

    def __init__(
        self,
        expert_shape: Tuple[int | Int32, int | Int32, int | Int32],
        cta_tile_shape_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
    ):
        # Whatever the caller passed, keep as-is: a Python int stays trace-time and is
        # never serialized, so expressions over it fold instead of costing a register.
        self.local_expert_count, self.intermediate, self.hidden = expert_shape
        self.cta_tile_shape_mnk = cta_tile_shape_mnk
        self.cluster_shape_mn = cluster_shape_mn

    @property
    def cluster_tile_m(self) -> int:
        """Cluster tile size along M = cta_tile_m * cluster_m."""
        return const_expr(self.cta_tile_shape_mnk[0] * self.cluster_shape_mn[0])

    @property
    def cluster_tile_n(self) -> int:
        """Cluster tile size along N = cta_tile_n * cluster_n."""
        return const_expr(self.cta_tile_shape_mnk[1] * self.cluster_shape_mn[1])

    def __extract_mlir_values__(self) -> List[ir.Value]:
        """Only serialize runtime values, not codegen-time constants."""
        values = []
        for field in (self.local_expert_count, self.intermediate, self.hidden):
            values.extend(extract_mlir_values(field))
        return values

    def __new_from_mlir_values__(self, values: List[ir.Value]) -> "MoEInputBlockSignalSchedulerParams":
        # A static field consumes no values and rebuilds as itself.
        value_iterator = iter(values)

        def take(field):
            value_count = len(extract_mlir_values(field))
            return new_from_mlir_values(field, [next(value_iterator) for _ in range(value_count)]) if value_count else field

        rebuilt = tuple(take(field) for field in (self.local_expert_count, self.intermediate, self.hidden))
        return MoEInputBlockSignalSchedulerParams(
            expert_shape=rebuilt,
            cta_tile_shape_mnk=self.cta_tile_shape_mnk,
            cluster_shape_mn=self.cluster_shape_mn,
        )


class MoEStreamingInputBlockSignalTileScheduler:
    """Persistent tile scheduler for the MoE grouped GEMM: tile iteration only (claim a linear tile index,
    broadcast it to the cluster).  MoESchedExtension converts domains and picks TMA descriptors.
    Cluster (2, 1) with the 2-CTA MMA only: a (1, 1) cluster is not supported (``__init__`` asserts two CTAs)."""

    def internal_init(self):
        """Create the internal cluster broadcast pipeline (dynamic mode only)."""
        cluster_size = const_expr(self.params.cluster_shape_mn[0] * self.params.cluster_shape_mn[1])
        self.cluster_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=self._sched_storage.cluster_mbar.data_ptr(),
            num_stages=1,
            producer_group=CooperativeGroup(Agent.Thread, 1),
            consumer_group=CooperativeGroup(Agent.Thread, 32 * cluster_size),
            defer_sync=True,
        )
        cute.arch.mbarrier_init_fence()

    def internal_finalize(self):
        """Cluster-wide barrier that must run before any CTA of the cluster exits.  Unconditional:
        ``__init__`` asserts a cluster of at least two CTAs."""
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()

    def __init__(
        self,
        params: MoEInputBlockSignalSchedulerParams,
        expert_freq_offsets: cute.Tensor,
        cta_id_in_cluster: cute.Coord,
        dynamic_state: _DynamicSchedState,
        sched_storage,
        cluster_pipeline,
        streaming_signal_capacity: int,
        rows_alignment: int,
        device_layout: bool,
        release_zero: ReleaseZero,
    ):
        self.params = params
        self.expert_freq_offsets = expert_freq_offsets
        self.cta_id_in_cluster = cta_id_in_cluster
        self._dynamic_state = dynamic_state
        self._sched_storage = sched_storage
        self.cluster_pipeline = cluster_pipeline
        # Never serialized: a build-time int that only places the scheduler state.
        self.streaming_signal_capacity = streaming_signal_capacity
        self.device_layout = device_layout
        # The kernel's release-zero (consume-then-arrive slot release), wrapped once at kernel entry; never serialized.
        self.release_zero = release_zero
        # Rows per signal[] slot, derived from the expert row alignment and the tile (never serialized).  Experts are
        # padded to whole M tiles here (there is no tail tile under the cluster tile M), so a slot is one M tile.
        assert isinstance(rows_alignment, int), f"rows_alignment must be a build-time int, not {rows_alignment}"
        self.rows_alignment = rows_alignment
        self.slot_rows = min(params.cluster_tile_m, rows_alignment)
        assert rows_alignment >= params.cluster_tile_m and rows_alignment % params.cluster_tile_m == 0, (
            f"rows_alignment {rows_alignment} must be a multiple of the cluster tile M {params.cluster_tile_m}: "
            "the signal scheduler runs one slot per M tile"
        )
        check_linear_tile_idx_range(params.intermediate, params.cluster_tile_n, streaming_signal_capacity)
        # mapa.shared::cluster at one CTA is an illegal peer store.  check_dynamic_sched_cluster_shape admits (1, 1)
        # with the 1-CTA MMA, so this assert is what rejects it for the signal scheduler.
        assert const_expr(
            params.cluster_shape_mn[0] * params.cluster_shape_mn[1] > 1
        ), "the signal scheduler broadcasts with mapa.shared::cluster, which needs >= 2 CTAs"

    def __extract_mlir_values__(self) -> List[ir.Value]:
        values = []
        values.extend(extract_mlir_values(self.params))
        values.extend(extract_mlir_values(self.expert_freq_offsets))
        values.extend(extract_mlir_values(self.cta_id_in_cluster))
        values.extend(extract_mlir_values(self._dynamic_state))
        return values

    def __new_from_mlir_values__(self, values: List[ir.Value]) -> "MoEStreamingInputBlockSignalTileScheduler":

        def take(field):
            """Consume this field's own share off the front; `values` shrinks as it goes."""
            nonlocal values
            value_count = len(extract_mlir_values(field))
            field_values, values = (values[:value_count], values[value_count:])
            return new_from_mlir_values(field, field_values)

        return MoEStreamingInputBlockSignalTileScheduler(
            params=take(self.params),
            expert_freq_offsets=take(self.expert_freq_offsets),
            cta_id_in_cluster=take(self.cta_id_in_cluster),
            dynamic_state=take(self._dynamic_state),
            sched_storage=self._sched_storage,
            cluster_pipeline=self.cluster_pipeline,
            streaming_signal_capacity=self.streaming_signal_capacity,
            rows_alignment=self.rows_alignment,
            device_layout=const_expr(self.device_layout),
            release_zero=self.release_zero,
        )

    @staticmethod
    def create(
        params: MoEInputBlockSignalSchedulerParams,
        expert_freq_offsets: cute.Tensor,
        counter_ptr: cute.Pointer,
        sched_storage,
        *,
        streaming_signal_capacity: int,
        rows_alignment: int,
        release_zero: ReleaseZero,
        device_layout: bool = False,
    ) -> "MoEStreamingInputBlockSignalTileScheduler":
        """Create the scheduler from the claim counter ``counter_ptr``; ``signal[]`` and the scheduler state follow it.
        ``rows_alignment``: the kernel's expert row alignment (every ``padded_offsets`` entry is a multiple of it); the
        scheduler derives the rows per signal slot from it and the cluster tile M.
        ``device_layout``: wait for the dispatch-written layout before the first claim (see the module docstring).
        ``release_zero``: ``release_zero(<the kernel's release-zero argument>)``, which orders each slot release after the
        read of the slot (consume-then-arrive slot release)."""
        # %cluster_ctarank, hinted warp-uniform so it lands in a uniform register; CTAs are x-fastest in the cluster.
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        cta_id_in_cluster = (
            cta_rank_in_cluster % const_expr(params.cluster_shape_mn[0]),
            cta_rank_in_cluster // const_expr(params.cluster_shape_mn[0]),
            0,
        )
        # Rank 0, not "even rank": this gates the atomicAdd that claims a tile, and a
        # cluster with two claimers would take two claim indices for one cluster's work.
        is_leader_cta = cta_rank_in_cluster == 0
        broadcast_ptr = sched_storage.cluster_broadcast.data_ptr()
        dynamic_state = _DynamicSchedState(
            counter_ptr=counter_ptr,
            broadcast_ptr=broadcast_ptr,
            is_leader_cta=is_leader_cta,
            producer_state=make_pipeline_state(PipelineUserType.Producer, 1),
            consumer_state=make_pipeline_state(PipelineUserType.Consumer, 1),
        )
        return MoEStreamingInputBlockSignalTileScheduler(
            params=params,
            expert_freq_offsets=expert_freq_offsets,
            cta_id_in_cluster=cta_id_in_cluster,
            dynamic_state=dynamic_state,
            sched_storage=sched_storage,
            cluster_pipeline=None,
            streaming_signal_capacity=streaming_signal_capacity,
            rows_alignment=rows_alignment,
            device_layout=device_layout,
            release_zero=release_zero,
        )

    @cute.jit
    def advance_to_next_work(self) -> MoEWorkTileInfo:
        """Claim the cluster's next linear tile index and return its work tile, invalid past the last."""
        claimed_linear_tile_idx = self._broadcast_claimed_tile()
        work_tile_info = self._get_work_tile_for_linear_idx(claimed_linear_tile_idx)
        # The broadcast value is consumed: a full warp per CTA arrives on the leader's empty barrier, each lane at an
        # address computed from the slot word it read, so the read completes before the leader may refill the slot (consume-then-arrive slot release).
        consume_then_arrive_on_peer(
            self.cluster_pipeline.sync_object_empty.get_barrier(self._dynamic_state.consumer_state.index),
            0,
            claimed_linear_tile_idx,
            self.release_zero,
        )
        self._dynamic_state.consumer_state.advance()
        return work_tile_info

    @cute.jit
    def initial_work_tile_info(self) -> MoEWorkTileInfo:
        """The first work tile, claimed like every later one (``device_layout``: once the dispatch has written the
        layout)."""
        if const_expr(self.device_layout):
            self._wait_for_producer_layout()
        return self.advance_to_next_work()

    @cute.jit
    def _wait_for_producer_layout(self) -> None:
        """Each lane waits for its layout_entries[0:E+1] (entry e on lane e % 32) to leave the reset value -1, in every CTA
        (each SM drops stale L1); extra lanes re-wait the last entry (a branch makes ptxas emulate warp collectives)."""
        lane_idx = cute.arch.lane_idx()
        layout_entry_count = const_expr(self.params.local_expert_count + 1)
        # layout_entries[0] (expert 0's offset) sits one entry before expert_freq_offsets = layout_entries[1:E+1].
        layout_entries = self.expert_freq_offsets.iterator + (-1)
        for j in cutlass.range_constexpr(ceil_div(layout_entry_count, cute.arch.WARP_SIZE)):
            layout_entry_idx = cutlass.min(lane_idx + j * cute.arch.WARP_SIZE, layout_entry_count - 1)
            _spin_until_at_least(layout_entries + layout_entry_idx, 0)
        cute.arch.sync_warp()

    def _state_region_ptr(self, region: cutlass.Constexpr):
        """Where ``region`` of the scheduler state starts (see ``_state_region_offsets``): bytes past the claim counter, a
        trace-time number with an int capacity."""
        return cute.make_ptr(
            Int32,
            self._dynamic_state.counter_ptr.toint()
            + (
                streaming_scheduler_state_offset(self.streaming_signal_capacity)
                + 4 * _state_region_offsets(self.streaming_signal_capacity, self.params.local_expert_count)[region]
            ),
            AddressSpace.gmem,
            assumed_align=4,
        )

    def _expert_consecutive_m_batch_record_ptr(self, m_tile_order: Int32):
        """Batch record of the k-th M-tile put in a batch (read by claims [k * Q_N, (k + 1) * Q_N)): first claim index,
        first M tile, M-tile count."""
        return self._state_region_ptr("expert_consecutive_m_batch_record") + 3 * m_tile_order

    def _set_expert_consecutive_m_batch_record(self, m_tile_order: Int32, record: Tuple[Int32, Int32, Int32]) -> None:
        """Store ``record`` (first claim index, first M tile, M-tile count)."""
        record_ptr = self._expert_consecutive_m_batch_record_ptr(m_tile_order)
        # Scheduler state is read and written at gpu scope (relaxed: L1 bypassed, so other SMs' writes are seen);
        # the release on ``assigned_claim_count`` publishes these stores.
        for record_entry_idx, value in enumerate(record):
            cute.arch.store((record_ptr + record_entry_idx).llvm_ptr, value, sem="relaxed", scope="gpu")

    def _get_expert_consecutive_m_batch_record(self, m_tile_order: Int32) -> Tuple[Int32, Int32, Int32]:
        """The record (first claim index, first M tile, M-tile count) at ``m_tile_order``."""
        record_ptr = self._expert_consecutive_m_batch_record_ptr(m_tile_order)
        return tuple(cute.arch.load((record_ptr + record_entry_idx).llvm_ptr, Int32, sem="relaxed", scope="gpu") for record_entry_idx in range(3))

    @cute.jit
    def _expert_end_m_tile(self, expert: Int32) -> Int32:
        """One past the last M tile of ``expert``: the offsets are padded to whole M tiles and each one is where its
        expert ENDS, so an expert starts where the one before it ends.  A slot is one M tile (see ``__init__``)."""
        return self.expert_freq_offsets[expert] // const_expr(self.slot_rows)

    @cute.jit
    def _expert_first_m_tile(self, expert: Int32) -> Int32:
        """The first M tile of ``expert`` (see ``_expert_end_m_tile``)."""
        first_m_tile = Int32(0)
        if expert > 0:
            first_m_tile = self._expert_end_m_tile(expert - 1)
        return first_m_tile

    @cute.jit
    def _total_m_tiles(self) -> Int32:
        """M tiles over every expert: the padded row total over the cluster tile M."""
        return self._expert_end_m_tile(const_expr(self.params.local_expert_count - 1))

    @cute.jit
    def _expert_ready_m_tiles(self, expert: Int32) -> Int32:
        """How many M-tiles of ``expert`` are READY in a row from its first unassigned M-tile on, at most
        ``EXPERT_BATCH_MAX_M_TILES``; none past the last expert.  One round of independent flag loads, no loop; a lane
        with no M-tile left to count skips them, and so does the whole warp when no lane has one.  A converged warp
        only (full-mask vote)."""
        local_expert_count = const_expr(self.params.local_expert_count)
        # A lane past the experts reads the last one's offsets, in bounds, and counts nothing.
        clamped_expert = cutlass.min(expert, local_expert_count - 1)
        expert_end_m_tile = self._expert_end_m_tile(clamped_expert)
        assigned_m_tiles_ptr = self._state_region_ptr("expert_assigned_m_tiles") + clamped_expert
        assigned_m_tiles = cute.arch.load(assigned_m_tiles_ptr.llvm_ptr, Int32, sem="relaxed", scope="gpu")
        first_unassigned_m_tile = self._expert_first_m_tile(clamped_expert) + assigned_m_tiles
        run_limit = cutlass.min(expert_end_m_tile - first_unassigned_m_tile, EXPERT_BATCH_MAX_M_TILES)
        if expert >= local_expert_count:
            run_limit = Int32(0)
        signal = streaming_signal_ptr(self._dynamic_state.counter_ptr)
        ready_m_tile_count = Int32(0)
        # An idle lane counts 0 either way (``m_tile_offset < run_limit`` never holds), so skipping its loads changes
        # no lane's count; with many small experts most lanes are idle late in the GEMM.
        if cute.arch.vote_any_sync(run_limit > 0):
            if run_limit > 0:
                # A flag past the expert's end reads its last M tile instead, in bounds and not counted.
                last_m_tile = cutlass.max(expert_end_m_tile - 1, 0)
                flags = [
                    cute.arch.load((signal + cutlass.min(first_unassigned_m_tile + m_tile_offset, last_m_tile)).llvm_ptr, Int32, sem="relaxed", scope="gpu")
                    for m_tile_offset in range(EXPERT_BATCH_MAX_M_TILES)
                ]
                for m_tile_offset in cutlass.range_constexpr(EXPERT_BATCH_MAX_M_TILES):
                    if ready_m_tile_count == m_tile_offset and m_tile_offset < run_limit:
                        if flags[m_tile_offset] >= 0:
                            ready_m_tile_count = ready_m_tile_count + 1
        return ready_m_tile_count

    @cute.jit
    def _open_expert_consecutive_m_batch(self, claim_idx: Int32, n_tiles_per_m_tile: Int32, lane_idx: Int32) -> None:
        """Open and publish the batch at ``claim_idx`` (only the ``assigned_claim_count`` warp: no atomics): prefer
        the expert opened last whose ready run ends it or fills the cap (weights in L2), else the most ready M-tiles."""
        local_expert_count = const_expr(self.params.local_expert_count)
        expert_last_batch_ptr = self._state_region_ptr("expert_last_batch")

        m_tile_count = Int32(0)
        expert = Int32(0)
        # This lane's best expert of the last scan (read after the loop to open more batches).
        lane_ready_m_tile_count = Int32(0)
        lane_expert = Int32(0)
        lane_run_complete = Boolean(False)
        # No ready M-tile: look again.  ``claim_idx`` exists, so one is still unassigned, and
        # the producer does not wait on this consumer, so it does get published.
        while m_tile_count == 0:
            # Score: 0 with no ready M-tile; an expert that had a batch scores past any ready count, by how recently
            # that batch opened; any other its ready count.
            lane_score = Int32(0)
            lane_ready_m_tile_count = Int32(0)
            lane_expert = Int32(0)
            lane_run_complete = Boolean(False)
            for round_first_expert in cutlass.range_constexpr(0, local_expert_count, cute.arch.WARP_SIZE):
                scanned_expert = round_first_expert + lane_idx
                ready_m_tile_count = self._expert_ready_m_tiles(scanned_expert)
                score = ready_m_tile_count
                run_complete = Boolean(False)
                if ready_m_tile_count > 0:
                    last_batch = cute.arch.load((expert_last_batch_ptr + scanned_expert).llvm_ptr, Int32, sem="relaxed", scope="gpu")
                    # Only a run that ends the expert or fills the batch cap: a shorter one (its rows still arriving)
                    # would open a small batch and cannot spare the expert another pass over its weights.
                    assigned_m_tiles_ptr = self._state_region_ptr("expert_assigned_m_tiles") + scanned_expert
                    assigned_m_tiles = cute.arch.load(assigned_m_tiles_ptr.llvm_ptr, Int32, sem="relaxed", scope="gpu")
                    run_end_m_tile = self._expert_first_m_tile(scanned_expert) + assigned_m_tiles + ready_m_tile_count
                    ready_run_ends_expert_or_fills_batch = ready_m_tile_count == const_expr(EXPERT_BATCH_MAX_M_TILES) or (
                        run_end_m_tile >= self._expert_end_m_tile(scanned_expert)
                    )
                    run_complete = ready_run_ends_expert_or_fills_batch
                    if last_batch != 0 and ready_run_ends_expert_or_fills_batch:
                        score = const_expr(EXPERT_BATCH_MAX_M_TILES) + last_batch
                if score > lane_score:
                    lane_score = score
                    lane_ready_m_tile_count = ready_m_tile_count
                    lane_expert = scanned_expert
                    lane_run_complete = run_complete
            top_expert_score = cute.arch.warp_redux_sync(lane_score, "max")
            # Only a lane that holds the best score names its expert; the others
            # answer with the expert count, which is past all of them.
            expert_candidate = Int32(local_expert_count)
            if lane_score == top_expert_score:
                expert_candidate = lane_expert
            expert = cute.arch.warp_redux_sync(expert_candidate, "min")
            # And the lane that named it, its ready count (0 when no expert has one: look again).
            ready_m_tile_count_candidate = Int32(0)
            if lane_expert == expert:
                ready_m_tile_count_candidate = lane_ready_m_tile_count
            m_tile_count = cute.arch.warp_redux_sync(ready_m_tile_count_candidate, "max")

        self._publish_batches(claim_idx, n_tiles_per_m_tile, lane_idx, expert, m_tile_count, lane_expert, lane_ready_m_tile_count, lane_run_complete)

    @cute.jit
    def _publish_batches(
        self,
        claim_idx: Int32,
        n_tiles_per_m_tile: Int32,
        lane_idx: Int32,
        expert: Int32,
        m_tile_count: Int32,
        lane_expert: Int32,
        lane_ready_m_tile_count: Int32,
        lane_run_complete: Boolean,
    ) -> None:
        """Publish the chosen expert's batch first, then, in lane order, every other lane's best expert whose ready run
        is complete (ends the expert or fills the cap), and release them all at once."""
        assigned_claim_count_ptr = self._state_region_ptr("assigned_claim_count")
        expert_last_batch_ptr = self._state_region_ptr("expert_last_batch")
        top_lane_candidate = Int32(cute.arch.WARP_SIZE)
        if lane_expert == expert:
            if lane_ready_m_tile_count > 0:
                top_lane_candidate = Int32(lane_idx)
        top_lane = cute.arch.warp_redux_sync(top_lane_candidate, "min")
        lane_m_tiles = Int32(0)
        if lane_idx == top_lane:
            lane_m_tiles = m_tile_count
        elif lane_run_complete:
            if lane_expert != expert:
                lane_m_tiles = lane_ready_m_tile_count
        # Exclusive prefix of the other lanes' M tiles in lane order, after the chosen batch.
        other_m_tiles = lane_m_tiles
        if lane_idx == top_lane:
            other_m_tiles = Int32(0)
        prefix = other_m_tiles
        for shift in cutlass.range_constexpr(5):
            # mask_and_clamp=0: lanes below ``1 << shift`` read their own value back (and drop it below), not lane 0's.
            up = cute.arch.shuffle_sync_up(prefix, 1 << shift, mask_and_clamp=0)
            if lane_idx >= (1 << shift):
                prefix = prefix + up
        total_m_tiles = m_tile_count + cute.arch.warp_redux_sync(other_m_tiles, "add")
        first_m_tile_order_in_opening = m_tile_count + prefix - other_m_tiles
        if lane_idx == top_lane:
            first_m_tile_order_in_opening = Int32(0)
        if lane_m_tiles > 0:
            batch_claim_idx = claim_idx + first_m_tile_order_in_opening * n_tiles_per_m_tile
            expert_assigned_m_tiles_ptr = self._state_region_ptr("expert_assigned_m_tiles") + lane_expert
            already_assigned_m_tiles = cute.arch.load(expert_assigned_m_tiles_ptr.llvm_ptr, Int32, sem="relaxed", scope="gpu")
            first_m_tile = self._expert_first_m_tile(lane_expert) + already_assigned_m_tiles
            cute.arch.store(expert_assigned_m_tiles_ptr.llvm_ptr, already_assigned_m_tiles + lane_m_tiles, sem="relaxed", scope="gpu")
            cute.arch.store((expert_last_batch_ptr + lane_expert).llvm_ptr, batch_claim_idx + 1, sem="relaxed", scope="gpu")
            # The batch takes the next ``lane_m_tiles`` M-tile orders: the same record in each.
            first_m_tile_order = batch_claim_idx // n_tiles_per_m_tile
            for m_tile_offset in cutlass.range_constexpr(EXPERT_BATCH_MAX_M_TILES):
                if m_tile_offset < lane_m_tiles:
                    self._set_expert_consecutive_m_batch_record(first_m_tile_order + m_tile_offset, (batch_claim_idx, first_m_tile, lane_m_tiles))
        # A 32-thread named barrier (bar.sync, a synchronizing operation of the PTX memory model, unlike bar.warp.sync)
        # orders every lane's stores above before the elected lane's release below.
        cute.arch.barrier(barrier_id=OPENER_PUBLISH_BARRIER_ID, number_of_threads=cute.arch.WARP_SIZE)
        with cute.arch.elect_one():
            # Published LAST, and that order is load-bearing: the release is what hands the
            # records to the claimers and ``expert_assigned_m_tiles[]`` to the next opener.
            cute.arch.store(assigned_claim_count_ptr.llvm_ptr, claim_idx + total_m_tiles * n_tiles_per_m_tile, sem="release", scope="gpu")
        # ``assigned_claim_count`` is past ``claim_idx`` now, so this returns at once.  It is
        # here for its acquire: other lanes wrote the records, and every lane is about to read them.
        _spin_until_at_least(assigned_claim_count_ptr, claim_idx + 1)

    @cute.jit
    def _claim_linear_tile_idx(self, n_tiles_per_m_tile: Int32, lane_idx: Int32) -> Int32:
        """Claim a number (one atomic add), wait for every earlier one to have a batch, open one if it has none, and
        return its linear tile index, END past the last."""
        dynamic_state = self._dynamic_state
        claim_idx = 0
        if lane_idx == 0:
            claim_idx = atomic_add_i32(dynamic_state.counter_ptr.llvm_ptr, Int32(1))
        # The other lanes hold 0, so the OR hands every lane lane 0's number.
        claim_idx = cute.arch.warp_redux_sync(claim_idx, "or")
        # One claim per (M tile, N tile); from there on a claim broadcasts END, the first index past every tile.
        claim_count = self._total_m_tiles() * n_tiles_per_m_tile
        linear_tile_idx = claim_count
        if claim_idx < claim_count:
            # Every lane lands on the same side of this test: once ``assigned_claim_count``
            # is ``claim_idx`` it stays there until this very warp moves it.
            if _spin_until_at_least(self._state_region_ptr("assigned_claim_count"), claim_idx) == claim_idx:
                self._open_expert_consecutive_m_batch(claim_idx, n_tiles_per_m_tile, lane_idx)
            # ``assigned_claim_count`` is past ``claim_idx`` and every lane has acquired
            # it, so the record at this claim index's M-tile order is written and visible.
            first_claim_idx, first_m_tile, m_tile_count = self._get_expert_consecutive_m_batch_record(claim_idx // n_tiles_per_m_tile)
            claim_idx_in_batch = claim_idx - first_claim_idx
            # Short side first (the stock dynamic scheduler's order within an expert): N tiles fastest when the batch has
            # more M tiles than N tiles, else M tiles fastest.  Any order is safe: a batch holds only ready M tiles.
            n_tile = claim_idx_in_batch // m_tile_count
            m_tile = first_m_tile + claim_idx_in_batch % m_tile_count
            if n_tiles_per_m_tile < m_tile_count:
                n_tile = claim_idx_in_batch % n_tiles_per_m_tile
                m_tile = first_m_tile + claim_idx_in_batch // n_tiles_per_m_tile
            # A batch holds only ready M-tiles, so this returns at once.  It is here for its
            # acquire, which pairs the producer's release and makes the M-tile's rows visible.
            _spin_until_at_least(streaming_signal_ptr(dynamic_state.counter_ptr) + m_tile, 0)
            linear_tile_idx = m_tile * n_tiles_per_m_tile + n_tile
        return linear_tile_idx

    @cute.jit
    def _broadcast_claimed_tile(self) -> Int32:
        """Claim a number with one atomic add, wait until every earlier one has a batch, open
        one if it has none, and broadcast its linear tile index (END past the last) to the cluster."""
        dynamic_state = self._dynamic_state
        broadcast_tensor = cute.make_tensor(dynamic_state.broadcast_ptr, cute.make_layout((1,)))
        if dynamic_state.is_leader_cta:
            self.cluster_pipeline.producer_acquire(dynamic_state.producer_state)
            full_barrier_ptr = self.cluster_pipeline.sync_object_full.get_barrier(dynamic_state.producer_state.index)
            lane_idx = cute.arch.lane_idx()
            cluster_size = const_expr(self.params.cluster_shape_mn[0] * self.params.cluster_shape_mn[1])
            # Q_N: every M-tile is this many N-tiles, and so this many claim indices.
            n_tiles_per_m_tile = ceil_div(self.params.intermediate, const_expr(self.params.cluster_tile_n))
            linear_tile_idx = self._claim_linear_tile_idx(n_tiles_per_m_tile, lane_idx)
            if lane_idx < cluster_size:
                cute.arch.store_async_dsmem(dynamic_state.broadcast_ptr, linear_tile_idx, full_barrier_ptr, lane_idx)
            # Fan out one arrive per CTA to the full barrier.
            with cute.arch.elect_one():
                # fence.release.cluster: orders the slot's st.async and earlier stores before the peers' arrives.
                llvm.fence(llvm.AtomicOrdering.release, syncscope="cluster")
                for peer in cutlass.range_constexpr(cluster_size):
                    # Arrive + expect_tx on each peer's full barrier, .relaxed.cluster (the fence above releases).
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        full_barrier_ptr, Int32(4), peer_cta_rank_in_cluster=peer, relaxed=True, scope=nvvm.MemScopeKind.CLUSTER
                    )
        dynamic_state.producer_state.advance()
        consumer_full_barrier_ptr = self.cluster_pipeline.sync_object_full.get_barrier(dynamic_state.consumer_state.index)
        broadcast_arrived = Boolean(False)
        while not broadcast_arrived:
            # try_wait.parity.acquire.cluster: the default .acquire.cta does not synchronize with a peer's arrive.
            broadcast_arrived = Boolean(
                nvvm.mbarrier_wait_parity(
                    consumer_full_barrier_ptr.to_llvm_ptr(),
                    Int32(dynamic_state.consumer_state.phase).ir_value(),
                    nvvm.MBarrierWaitKind.TRY,
                    scope=nvvm.MemScopeKind.CLUSTER,
                    order=nvvm.MemOrderKind.ACQUIRE,
                )
            )
        return broadcast_tensor[0]

    @cute.jit
    def _get_work_tile_for_linear_idx(self, linear_tile_idx: Int32) -> MoEWorkTileInfo:
        """The work tile of ``linear_tile_idx = m_tile * Q_N + n_tile``; END, past every expert's last M tile, gives an
        invalid tile."""
        local_expert_count = const_expr(self.params.local_expert_count)
        n_tiles_per_m_tile = ceil_div(self.params.intermediate, const_expr(self.params.cluster_tile_n))
        m_tile = linear_tile_idx // n_tiles_per_m_tile
        n_tile = linear_tile_idx % n_tiles_per_m_tile
        # The warp finds the first expert whose end M tile is past ``m_tile`` in one round of loads (lane l reads experts
        # l, l + 32, ...) and a min-reduction; a walk from expert 0 costs one dependent L2 load per expert per claim.
        lane_idx = cute.arch.lane_idx()
        expert = Int32(local_expert_count)
        for first_expert in cutlass.range_constexpr(0, local_expert_count, cute.arch.WARP_SIZE):
            scanned_expert = first_expert + lane_idx
            end_m_tile = self._expert_end_m_tile(cutlass.min(scanned_expert, local_expert_count - 1))
            expert_candidate = Int32(local_expert_count)
            if scanned_expert < local_expert_count and m_tile < end_m_tile:
                expert_candidate = scanned_expert
            expert = cutlass.min(expert, cute.arch.warp_redux_sync(expert_candidate, "min"))
        work_tile_info = MoEWorkTileInfo(expert_idx=Int32(-1), tile_m_idx=Int32(0), tile_n_idx=Int32(0), k_tile_cnt=Int32(0))
        if expert < local_expert_count:
            m_tile_in_expert = m_tile - self._expert_first_m_tile(expert)
            cta_tile_m_idx = m_tile_in_expert * const_expr(self.params.cluster_shape_mn[0]) + self.cta_id_in_cluster[0]
            cta_tile_n_idx = n_tile * const_expr(self.params.cluster_shape_mn[1]) + self.cta_id_in_cluster[1]
            k_tile_cnt = self._compute_k_tile_cnt()
            work_tile_info = MoEWorkTileInfo(
                expert_idx=expert,
                tile_m_idx=cta_tile_m_idx,
                tile_n_idx=cta_tile_n_idx,
                k_tile_cnt=k_tile_cnt,
            )
        return work_tile_info

    @cute.jit
    def _compute_k_tile_cnt(self) -> Int32:
        """K tiles per work tile: ``ceil(hidden / cta_tile_k)``, folded when ``hidden`` is static.
        The Int32 keeps the work tile's field type the same on both arms of the caller's branch."""
        return Int32(ceil_div(self.params.hidden, const_expr(self.params.cta_tile_shape_mnk[2])))


class StreamingInputSignalMixin:
    """The streaming input signal wiring of a MoE grouped GEMM kernel whose A (and SFA) a producer fills while it runs: the
    options, the workspace, the scheduler, entry and exit.  Contract and options:
    docs/operations/moe/streaming_signal_contract.md.

    The kernel calls ``init_streaming_input_signal`` in ``__init__``; with ``streaming_signal_capacity`` None every method
    below gives the stock dynamic scheduler's code."""

    def init_streaming_input_signal(
        self,
        streaming_signal_capacity: Optional[int],
        *,
        use_dynamic_sched: bool,
        use_2cta_instrs: bool,
        cluster_shape_mn: Tuple[int, int],
        unmet_requirement: Optional[str] = None,
        device_layout: bool = False,
        use_pdl: bool = False,
        launch_pdl_dependents_at_entry: bool = False,
        cluster_barrier_at_exit: bool = False,
        launches_helper: bool = False,
    ) -> None:
        """Check and keep the options.  ``unmet_requirement``: what this kernel configuration lacks for a producer (for
        example "dense weights"), or None.  ``cluster_barrier_at_exit``: the kernel has no 2-CTA MMA and every CTA passes
        a cluster barrier before it exits (the SM90 kernels), so a peer's last arrive always lands; a producer then needs
        cluster (2, 1).  ``launches_helper``: the host launch puts the kernel's helper_kernel right before it
        (``reject_pdl_behind_helper``)."""
        self.streaming_signal_capacity = streaming_signal_capacity
        self.use_streaming_signal_sched = streaming_signal_capacity is not None
        if self.use_streaming_signal_sched and not use_dynamic_sched:
            raise ValueError("streaming_signal_capacity requires use_dynamic_sched=True")
        if self.use_streaming_signal_sched and unmet_requirement is not None:
            raise ValueError(f"streaming_signal_capacity requires {unmet_requirement}")
        if device_layout and not self.use_streaming_signal_sched:
            raise ValueError("device_layout requires streaming_signal_capacity")
        reject_pdl_behind_helper(use_pdl, launches_helper, "streaming_signal_capacity (dense weights)")
        if self.use_streaming_signal_sched and cluster_barrier_at_exit:
            if tuple(cluster_shape_mn) != (2, 1):
                raise ValueError(f"streaming_signal_capacity requires cluster_shape_mn=(2, 1) on this kernel, got {tuple(cluster_shape_mn)}")
        elif self.use_streaming_signal_sched:
            check_dynamic_sched_cluster_shape(use_2cta_instrs, cluster_shape_mn)
        self.device_layout = device_layout
        self.use_pdl = use_pdl
        self.launch_pdl_dependents_at_entry = launch_pdl_dependents_at_entry

    def streaming_input_workspace_bytes(self) -> int:
        """Workspace bytes after the claim counter: ``signal[]`` and the scheduler state (0 without a producer)."""
        if not self.use_streaming_signal_sched:
            return 0
        return streaming_signal_workspace_bytes_after_counter(self.streaming_signal_capacity, self.expert_cnt)

    def check_streaming_input_scheduler_counter(self, scheduler_counter) -> None:
        """In ``__call__``: ``signal[]`` and the scheduler state follow the kernel's own claim counter in the workspace."""
        if self.use_streaming_signal_sched and scheduler_counter is not None:
            raise ValueError("streaming_signal_capacity owns the claim counter: pass no scheduler_counter")

    def make_streaming_input_sched_params(self, expert_shape, cta_tile_shape_mnk, cluster_shape_mn, use_dynamic_sched: bool):
        """The scheduler parameters: the signal scheduler's, or the stock 2Dx3D ones."""
        if self.use_streaming_signal_sched:
            return MoEInputBlockSignalSchedulerParams(expert_shape=expert_shape, cta_tile_shape_mnk=cta_tile_shape_mnk, cluster_shape_mn=cluster_shape_mn)
        return MoESchedulerParams(
            scenario="2Dx3D",
            expert_shape=expert_shape,
            cta_tile_shape_mnk=cta_tile_shape_mnk,
            cluster_shape_mn=cluster_shape_mn,
            use_dynamic_sched=use_dynamic_sched,
        )

    def create_streaming_input_scheduler(self, sched_params, padded_offsets, counter_ptr, sched_storage, zero):
        """In the kernel: the signal scheduler, or the stock persistent one.  ``zero``: ``release_zero(<the kernel's
        release-zero argument>)``, which the host passes with the signal scheduler (consume-then-arrive slot release).
        The kernel's ``FIX_PAD_SIZE`` is its expert row alignment."""
        if self.use_streaming_signal_sched:
            return MoEStreamingInputBlockSignalTileScheduler.create(
                sched_params,
                padded_offsets,
                counter_ptr=counter_ptr,
                sched_storage=sched_storage,
                streaming_signal_capacity=self.streaming_signal_capacity,
                rows_alignment=self.FIX_PAD_SIZE,
                release_zero=zero,
                device_layout=self.device_layout,
            )
        return MoEPersistentTileScheduler.create(
            sched_params,
            padded_offsets,
            cute.arch.block_idx(),
            cute.arch.grid_dim(),
            counter_ptr=counter_ptr,
            sched_storage=sched_storage,
        )

    @cute.jit
    def streaming_input_entry(self) -> None:
        """At kernel entry: ``launch_pdl_dependents_at_entry`` lets this kernel's own dependent launch."""
        if const_expr(self.launch_pdl_dependents_at_entry):
            cute.arch.griddepcontrol_launch_dependents()

    @cute.jit
    def streaming_input_tile_fence(self) -> None:
        """In the TMA warp, per tile: the producer wrote A (and SFA) through the generic proxy, so order its stores (acquired
        by the claim) before this tile's TMA loads."""
        if const_expr(self.use_streaming_signal_sched):
            cute.arch.fence_proxy("async.global")

    @cute.jit
    def streaming_input_exit(self, scheduler) -> None:
        """Every thread, last: the signal scheduler's last cluster handoff writes into a peer's shared memory, so the
        cluster waits for it; with ``use_pdl`` the grid this launch overlapped must end before this one does."""
        if const_expr(self.use_streaming_signal_sched):
            scheduler.internal_finalize()
        if const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()
