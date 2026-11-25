# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileContributor: Wentao Guo, Phuong Nguyen, Zhongbo Zhu, Subhadeep Bhattacharya, Fei Wu, Jimmy Zhang
# SPDX-License-Identifier: Apache-2.0

"""Streaming output-chunk tile scheduler and notifier for the MoE grouped GEMMs (2Dx3D).

With ``StreamingOutputNotifyMode.N_CHUNK`` the GEMM runs its tiles one N chunk of ``C = n_chunk_size`` columns
at a time, so a consumer kernel, such as an expert-parallel combine, can send each chunk while the GEMM still runs.

``signal[]``: int32, one entry per N chunk.
- Chunk ``u`` covers D columns ``[u * C, min((u + 1) * C, N))``.
- ``u`` is in ``[0, ceil(N / C))``, at most ``n_chunk_capacity`` (see ``streaming_output_notify_capacity``).
- Reset to -1; stays negative (``-1 - tiles counted``) while incomplete.
- The GEMM stores ``signal[u] = u`` with a gpu-scope release once every row of chunk ``u`` is in global memory.

Consumer contract: docs/operations/moe/streaming_signal_contract.md.

Terms:
- Tiles counted: CTA tiles, two per cluster M tile with the 2-CTA MMA.
- N-chunk order: ``cluster_linear_idx`` runs chunk by chunk, then M tile, then the chunk's N tiles.
  Odd chunks walk M backward to reuse the A rows still in L2.
- Extra clusters (rank >= ``clusters_beside_consumer``) exit once their claim reaches chunk 0's last wave,
  leaving their SMs to the consumer.

Example, cluster (1, 1), 2 M tiles x 4 N tiles, C = 2 N tiles; each tile shows its ``cluster_linear_idx``::

             |- u0 -|- u1 -|
              n0 n1  n2 n3
        m0     0  1   6  7
        m1     2  3   4  5        u1 starts on m1, whose A rows u0 just used

    signal[0], 4 CTA tiles:  -1  ->  -3 (a CTA counts 2)  ->  -5 (another counts 2: 4 of 4)  ->  0 (release store)

Assumes N <= (2^31 - 1) * n_chunk_size.
"""

import copy
from typing import Dict, List, Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Int32, const_expr, dsl_user_op, extract_mlir_values, new_from_mlir_values
from cutlass.cute.typing import AddressSpace
from cutlass._mlir import ir

from .moe_persistent_scheduler import MoEPersistentTileScheduler, MoEWorkTileInfo, atomic_add_i32
from .moe_utils import ceil_div, compute_expert_token_range


class MoEStreamingOutputChunkSignalTileScheduler(MoEPersistentTileScheduler):
    """Runs a stock scheduler N chunk -> every M tile -> the chunk's N tiles so a chunk finishes early; odd chunks walk
    M backward to reuse the A rows still in L2.  Experts are padded to whole cluster M tiles."""

    def __init__(
        self,
        stock: MoEPersistentTileScheduler,
        n_chunk_size: int,
        clusters_beside_consumer: int,
        local_expert_count: int,
    ):
        assert (
            n_chunk_size > 0 and n_chunk_size % stock.cluster_tile_n == 0
        ), f"N chunk {n_chunk_size} must be a positive multiple of N_TILE_SIZE {stock.cluster_tile_n}"
        assert stock.use_dynamic_sched, "the extra clusters stop through the dynamic scheduler's claims"
        self.__dict__.update(stock.__dict__)
        self.n_chunk_size = n_chunk_size
        self.clusters_beside_consumer = clusters_beside_consumer
        # The stock ``expert_cnt`` is a kernel argument; this is the same count at trace time.
        self.local_expert_count = local_expert_count
        self.expert_freq_offsets = stock.offs
        # The current expert's M-tile range.  The stock walk state, which counts tiles
        # in the stock order, carries it across dynamic regions.
        self.current_expert_m_tile_start = stock.expert_tile_start
        self.current_expert_m_tile_end = stock.expert_tile_end

    def __extract_mlir_values__(self) -> List[ir.Value]:
        self.expert_tile_start, self.expert_tile_end = self.current_expert_m_tile_start, self.current_expert_m_tile_end
        return super().__extract_mlir_values__()

    def __new_from_mlir_values__(self, values: List[ir.Value]) -> "MoEStreamingOutputChunkSignalTileScheduler":
        return MoEStreamingOutputChunkSignalTileScheduler(
            super().__new_from_mlir_values__(values),
            self.n_chunk_size,
            self.clusters_beside_consumer,
            self.local_expert_count,
        )

    @cute.jit
    def _total_m_tiles(self) -> Int32:
        """Cluster M tiles over every expert."""
        return self.expert_freq_offsets[const_expr(self.local_expert_count - 1)] // const_expr(self.cluster_tile_m)

    @cute.jit
    def _expert_m_tiles(self, expert_idx: Int32) -> Int32:
        """Cluster M tiles of one expert."""
        _, expert_i_row_count = compute_expert_token_range(self.expert_freq_offsets, expert_idx)
        return expert_i_row_count // const_expr(self.cluster_tile_m)

    @dsl_user_op
    @cute.jit
    def _exits_early(self, *, loc=None, ip=None) -> Int32:
        """1 when this is an extra cluster (rank >= ``clusters_beside_consumer``) at or past N chunk 0's last wave, else 0."""
        # The grid is (cluster shape, clusters), so bidz is this cluster's rank among them.
        _, _, cluster_rank = cute.arch.block_idx(loc=loc, ip=ip)
        # N chunk 0's tiles, as ``_get_work_tile_for_linear_idx`` orders them: every M tile times its N tiles.
        N_TILE_SIZE = const_expr(self.cluster_tile_n)
        total_M_tiles = self._total_m_tiles()
        first_n_chunk_tiles = total_M_tiles * cutlass.min(const_expr(self.n_chunk_size // N_TILE_SIZE), ceil_div(self.intermediate, N_TILE_SIZE))
        # Never below 0: every cluster claims its first tile.
        early_exit_linear_idx = cutlass.max(first_n_chunk_tiles - self.num_persistent_clusters, 0)
        exits_early = Int32(0)
        if cluster_rank >= const_expr(self.clusters_beside_consumer):
            if self._current_work_linear_idx >= early_exit_linear_idx:
                exits_early = Int32(1)
        return exits_early

    @dsl_user_op
    @cute.jit
    def _fetch_one_cta_cluster_idx(self, *, loc=None, ip=None) -> Int32:
        """``_fetch_next_cluster_idx`` at one CTA: the stock elected claim, with no broadcast slot."""
        exits_early = self._exits_early(loc=loc, ip=ip)
        claim = cute.make_rmem_tensor((1,), Int32)
        claim[0] = Int32(0)
        with cute.arch.elect_one(loc=loc, ip=ip):
            claim[0] = atomic_add_i32(self._dynamic_state.counter_ptr.llvm_ptr, 1 - exits_early, loc=loc, ip=ip)
        # The full-mask shuffle needs a converged warp, so elect.sync picked lane 0.
        cluster_idx = cute.arch.shuffle_sync(claim[0], offset=0, mask=0xFFFFFFFF, mask_and_clamp=31)
        if exits_early == 1:
            cluster_idx = Int32(2**31 - 1)  # past every tile: invalid, so the CTA exits
        return cluster_idx

    @dsl_user_op
    @cute.jit
    def _fetch_next_cluster_idx(self, *, loc=None, ip=None) -> Int32:
        """The stock fetch, but an extra cluster (rank >= ``clusters_beside_consumer``) whose last claim is at or past
        N chunk 0's last wave adds 0 and broadcasts an index past every tile; the leader decides for the cluster."""
        if const_expr(self.cluster_shape_mn[0] * self.cluster_shape_mn[1] == 1):
            # No broadcast: the stock ``_release_cluster_idx`` releases no slot at one CTA.
            return self._fetch_one_cta_cluster_idx(loc=loc, ip=ip)
        dynamic_state = self._dynamic_state
        broadcast_tensor = cute.make_tensor(dynamic_state.broadcast_ptr, cute.make_layout((1,)))
        if dynamic_state.is_leader_cta:
            self.cluster_pipeline.producer_acquire(dynamic_state.producer_state)
            full_barrier_ptr = self.cluster_pipeline.sync_object_full.get_barrier(dynamic_state.producer_state.index, loc=loc, ip=ip)
            lane_idx = cute.arch.lane_idx()
            cluster_size = const_expr(self.cluster_shape_mn[0] * self.cluster_shape_mn[1])
            exits_early = self._exits_early(loc=loc, ip=ip)
            atomic_idx = Int32(0)
            if lane_idx == 0:
                atomic_idx = atomic_add_i32(dynamic_state.counter_ptr.llvm_ptr, 1 - exits_early, loc=loc, ip=ip)
            atomic_idx = cute.arch.shuffle_sync(atomic_idx, offset=0, mask=0xFFFFFFFF, mask_and_clamp=31)
            if exits_early == 1:
                # SM100/107: cluster (2,1) 2-CTA or (1,1) only; this exit has no cluster barrier (check_dynamic_sched_cluster_shape).
                atomic_idx = Int32(2**31 - 1)  # past every tile: invalid, so the cluster exits
            if lane_idx < cluster_size:
                cute.arch.store_async_dsmem(dynamic_state.broadcast_ptr, atomic_idx, full_barrier_ptr, lane_idx, loc=loc, ip=ip)
            self._cluster_producer_commit(loc=loc, ip=ip)
        dynamic_state.producer_state.advance()
        self.cluster_pipeline.consumer_wait(dynamic_state.consumer_state)
        return broadcast_tensor[0]

    @dsl_user_op
    @cute.jit
    def _get_work_tile_for_linear_idx(self, cluster_linear_idx: Int32, *, loc=None, ip=None) -> MoEWorkTileInfo:
        """The work tile at an index of the N-chunk order; invalid (expert -1) past the last tile."""
        N_TILE_SIZE = const_expr(self.cluster_tile_n)
        last_expert_idx = const_expr(self.local_expert_count - 1)
        N_tiles_per_M_tile = ceil_div(self.intermediate, N_TILE_SIZE)
        total_M_tiles = self._total_m_tiles()
        work_tile_info = MoEWorkTileInfo(expert_idx=Int32(-1), tile_m_idx=Int32(0), tile_n_idx=Int32(0), k_tile_cnt=Int32(0))
        if cluster_linear_idx < total_M_tiles * N_tiles_per_M_tile:
            # Every M-tile runs the N-tiles of one N chunk before the next chunk starts; the last
            # chunk can be partial.
            N_tiles_per_N_chunk = const_expr(self.n_chunk_size // N_TILE_SIZE)
            N_chunk_idx = cluster_linear_idx // (total_M_tiles * N_tiles_per_N_chunk)
            idx_in_N_chunk = cluster_linear_idx - N_chunk_idx * (total_M_tiles * N_tiles_per_N_chunk)
            N_tiles_in_N_chunk = cutlass.min(N_tiles_per_N_chunk, N_tiles_per_M_tile - N_chunk_idx * N_tiles_per_N_chunk)
            M_tile_idx = idx_in_N_chunk // N_tiles_in_N_chunk
            N_tile_idx = N_chunk_idx * N_tiles_per_N_chunk + idx_in_N_chunk - M_tile_idx * N_tiles_in_N_chunk
            if N_chunk_idx % 2 == 1:
                M_tile_idx = total_M_tiles - 1 - M_tile_idx
            # Walk the expert range to M_tile_idx, back or forward.
            if self.current_expert_m_tile_end == 0:
                self.current_expert_idx = Int32(0)
                self.current_expert_m_tile_start = Int32(0)
                self.current_expert_m_tile_end = self._expert_m_tiles(Int32(0))
            while M_tile_idx < self.current_expert_m_tile_start and self.current_expert_idx > 0:
                self.current_expert_idx = self.current_expert_idx - 1
                self.current_expert_m_tile_end = self.current_expert_m_tile_start
                self.current_expert_m_tile_start = self.current_expert_m_tile_start - self._expert_m_tiles(self.current_expert_idx)
            while M_tile_idx >= self.current_expert_m_tile_end and self.current_expert_idx < last_expert_idx:
                self.current_expert_idx = self.current_expert_idx + 1
                self.current_expert_m_tile_start = self.current_expert_m_tile_end
                self.current_expert_m_tile_end = self.current_expert_m_tile_end + self._expert_m_tiles(self.current_expert_idx)
            work_tile_info = MoEWorkTileInfo(
                expert_idx=self.current_expert_idx,
                tile_m_idx=(M_tile_idx - self.current_expert_m_tile_start) * const_expr(self.cluster_shape_mn[0]) + self.cta_id_in_cluster[0],
                tile_n_idx=N_tile_idx * const_expr(self.cluster_shape_mn[1]) + self.cta_id_in_cluster[1],
                k_tile_cnt=self._compute_k_tile_cnt(self.current_expert_idx, loc=loc, ip=ip),
            )
        return work_tile_info


#: Threads of the one-block reset of ``signal[]``.
RESET_BLOCK_THREADS = 256


class StreamingOutputNotifyMode:
    NONE = 0
    N_CHUNK = 1


def streaming_output_notify_capacity(n_chunk_size: int, output_cols: int) -> int:
    """The ``n_chunk_capacity`` for ``output_cols`` D columns: the N chunks that cover them."""
    return -(-output_cols // n_chunk_size)  # host-side: moe_utils.ceil_div is @cute.jit


class _StreamingOutputNotifyState:
    """The N chunk the D-store warp is in, and its tiles of that chunk not yet counted in ``signal[]``.
    Empty when notification is off."""

    def __init__(self, current_n_chunk: Optional[Int32] = None, pending_tile_count: Optional[Int32] = None):
        self.current_n_chunk, self.pending_tile_count = current_n_chunk, pending_tile_count

    def __extract_mlir_values__(self) -> List[ir.Value]:
        if self.current_n_chunk is None:
            return []
        return extract_mlir_values(self.current_n_chunk) + extract_mlir_values(self.pending_tile_count)

    def __new_from_mlir_values__(self, values: List[ir.Value]) -> "_StreamingOutputNotifyState":
        if self.current_n_chunk is None:
            return _StreamingOutputNotifyState()
        return _StreamingOutputNotifyState(new_from_mlir_values(self.current_n_chunk, values[:1]), new_from_mlir_values(self.pending_tile_count, values[1:]))


class StreamingOutputNotifyConfig:
    """Trace-time notification settings; the default notifies nothing and costs nothing.  ``caller_reset``: the
    caller runs ``reset`` ahead of each GEMM.  N_CHUNK: grid clusters from ``clusters_beside_consumer`` on are extras."""

    def __init__(
        self,
        mode: int = StreamingOutputNotifyMode.NONE,
        n_chunk_size: int = 0,
        n_chunk_capacity: int = 0,
        caller_reset: bool = False,
        clusters_beside_consumer: int = 0,
    ):
        assert (
            mode == StreamingOutputNotifyMode.NONE or clusters_beside_consumer >= 1
        ), f"N_CHUNK needs clusters_beside_consumer >= 1, got {clusters_beside_consumer}"
        assert mode == StreamingOutputNotifyMode.NONE or n_chunk_size > 0, f"N_CHUNK needs n_chunk_size > 0, got {n_chunk_size}"
        self.mode, self.n_chunk_size, self.n_chunk_capacity = mode, n_chunk_size, n_chunk_capacity
        self.caller_reset = caller_reset
        self.clusters_beside_consumer = clusters_beside_consumer
        self.enabled = mode != StreamingOutputNotifyMode.NONE
        # ``signal[n_chunk_capacity]``, int32, at the end of the kernel's workspace.
        self.workspace_bytes = 4 * n_chunk_capacity if self.enabled else 0

    def bind(
        self,
        cta_tile_mn: Tuple[int, int],
        local_expert_count: int,
        use_streaming_signal_sched: bool,
    ) -> "StreamingOutputNotifyConfig":
        """A copy tied to one kernel's CTA tile and expert count."""
        assert not (use_streaming_signal_sched and self.enabled), "output notification is an FC2 feature; the signal scheduler is FC1's"
        assert not self.enabled or self.n_chunk_size % cta_tile_mn[1] == 0, f"N chunk {self.n_chunk_size} must be a multiple of CTA tile N {cta_tile_mn[1]}"
        resolved_config = copy.copy(self)
        resolved_config.cta_tile_m, resolved_config.cta_tile_n = cta_tile_mn
        resolved_config.expert_cnt = local_expert_count
        return resolved_config

    def check_output_cols(self, output_cols) -> None:
        """Host side, before launch: ``signal[]`` must hold every N chunk of the GEMM's ``output_cols`` D columns, else
        the notifier writes past it; N must be build-time to check."""
        if self.enabled:
            assert isinstance(output_cols, int), f"N_CHUNK needs a build-time N (dense B), not {output_cols}"
            required_capacity = streaming_output_notify_capacity(self.n_chunk_size, output_cols)
            assert self.n_chunk_capacity >= required_capacity, (
                f"n_chunk_capacity {self.n_chunk_capacity} < {required_capacity} N chunks of {self.n_chunk_size} in "
                f"N = {output_cols}: size it with streaming_output_notify_capacity"
            )

    def tile_order(self, scheduler):
        """The scheduler the kernel runs: the stock one as created when off, else in N-chunk order."""
        if not self.enabled:
            return scheduler
        return MoEStreamingOutputChunkSignalTileScheduler(scheduler, self.n_chunk_size, self.clusters_beside_consumer, self.expert_cnt)

    def _n_chunk_signal_ptr(self, workspace_ptr: cute.Pointer, kernel_workspace_bytes: int) -> cute.Pointer:
        """``signal[]``, which ends the kernel's workspace."""
        byte_offset = kernel_workspace_bytes - self.workspace_bytes
        return cute.make_ptr(Int32, workspace_ptr.toint() + byte_offset, AddressSpace.gmem, assumed_align=4)

    @cute.kernel
    def reset_kernel(self, workspace_ptr: cute.Pointer, kernel_workspace_bytes: cutlass.Constexpr):
        """``signal[] = -1``, no tile counted: one block, one N chunk per thread and step."""
        thread_idx, _, _ = cute.arch.thread_idx()
        n_chunk_capacity = const_expr(self.n_chunk_capacity)
        n_chunk_signal = cute.make_tensor(self._n_chunk_signal_ptr(workspace_ptr, kernel_workspace_bytes), cute.make_layout(n_chunk_capacity))
        for step in cutlass.range_constexpr(ceil_div(n_chunk_capacity, RESET_BLOCK_THREADS)):
            n_chunk = step * RESET_BLOCK_THREADS + thread_idx
            if n_chunk < n_chunk_capacity:
                n_chunk_signal[n_chunk] = -1

    reset_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)

    @cute.jit
    def reset(self, workspace_ptr: cute.Pointer, kernel_workspace_bytes: cutlass.Constexpr, stream: cuda.CUstream) -> None:
        """Enqueue the reset on ``stream``; ``kernel_workspace_bytes`` is ``get_workspace_bytes()``.
        A ``caller_reset`` caller compiles this once and runs it ahead of each GEMM."""
        self.reset_kernel(workspace_ptr, kernel_workspace_bytes).launch(grid=(1, 1, 1), block=(RESET_BLOCK_THREADS, 1, 1), stream=stream)

    def launch_reset(self, workspace_ptr: cute.Pointer, kernel_workspace_bytes: int, stream) -> None:
        """The GEMM call's own reset, ahead of it on its stream; none when off or caller-owned."""
        if const_expr(self.enabled and not self.caller_reset):
            self.reset(workspace_ptr, kernel_workspace_bytes, stream)

    def make_notifier(self, workspace_ptr: cute.Pointer, kernel_workspace_bytes: int, output_cols: Int32) -> "StreamingOutputStoreNotifier":
        """The device notifier over this kernel's ``signal[]``; ``output_cols`` is the GEMM N (D columns)."""
        if const_expr(not self.enabled):
            return StreamingOutputStoreNotifier(self)
        return StreamingOutputStoreNotifier(self, self._n_chunk_signal_ptr(workspace_ptr, kernel_workspace_bytes), output_cols)


class StreamingOutputStoreNotifier:
    """The D-store warp's hooks, no-ops when notification is off: ``begin_tile`` before a tile's
    D stores, ``end_tile`` after them, ``finalize`` before ``d_pipeline.producer_tail()``."""

    def __init__(self, config: StreamingOutputNotifyConfig, n_chunk_signal_ptr: Optional[cute.Pointer] = None, output_cols: Optional[Int32] = None):
        self.config = config
        self.n_chunk_signal_ptr, self.output_cols = n_chunk_signal_ptr, output_cols

    def _device_fields(self) -> list:
        return [self.n_chunk_signal_ptr, self.output_cols] if self.config.enabled else []

    def __extract_mlir_values__(self) -> List[ir.Value]:
        return [value for field in self._device_fields() for value in extract_mlir_values(field)]

    def __new_from_mlir_values__(self, values: List[ir.Value]) -> "StreamingOutputStoreNotifier":
        rebuilt_fields = []
        for field in self._device_fields():
            field_value_count = len(extract_mlir_values(field))
            rebuilt_fields.append(new_from_mlir_values(field, values[:field_value_count]))
            values = values[field_value_count:]
        return StreamingOutputStoreNotifier(self.config, *rebuilt_fields)

    def make_state(self) -> _StreamingOutputNotifyState:
        if const_expr(not self.config.enabled):
            return _StreamingOutputNotifyState()
        return _StreamingOutputNotifyState(Int32(-1), Int32(0))

    @cute.jit
    def begin_tile(self, state: _StreamingOutputNotifyState, work_tile: MoEWorkTileInfo, padded_offsets: cute.Tensor) -> _StreamingOutputNotifyState:
        """Before a tile's D stores: when it opens a new N chunk, count the previous chunk's pending tiles."""
        config = self.config
        if const_expr(config.enabled):
            if work_tile.tile_n_idx * const_expr(config.cta_tile_n) < self.output_cols:
                n_chunk = work_tile.tile_n_idx // const_expr(config.n_chunk_size // config.cta_tile_n)
                if n_chunk != state.current_n_chunk:
                    self.count_pending_tiles_and_publish_n_chunk(state, padded_offsets)
                    state = _StreamingOutputNotifyState(n_chunk, Int32(0))
            else:
                # A tile wholly past N (cluster N rounds the N tiles up) stores nothing: cancel
                # the count ``end_tile`` adds for it.
                state = _StreamingOutputNotifyState(state.current_n_chunk, state.pending_tile_count - 1)
        return state

    def end_tile(self, state: _StreamingOutputNotifyState) -> _StreamingOutputNotifyState:
        """After a tile's last D store is issued: one more pending tile."""
        if const_expr(not self.config.enabled):
            return state
        return _StreamingOutputNotifyState(state.current_n_chunk, state.pending_tile_count + 1)

    @cute.jit
    def finalize(self, state: _StreamingOutputNotifyState, padded_offsets: cute.Tensor) -> None:
        """Count the last N chunk's pending tiles; before ``d_pipeline.producer_tail()``."""
        if const_expr(self.config.enabled):
            self.count_pending_tiles_and_publish_n_chunk(state, padded_offsets)

    @cute.jit
    def count_pending_tiles_and_publish_n_chunk(self, state: _StreamingOutputNotifyState, padded_offsets: cute.Tensor) -> None:
        """Count the pending tiles in ``signal[n_chunk]`` once their D stores are in global memory, and
        publish the chunk if that completes it.  The acq_rel add acquires every earlier add's stores."""
        config = self.config
        n_chunk, pending_tile_count = state.current_n_chunk, state.pending_tile_count
        if pending_tile_count > 0:
            # Not ``PipelineTmaStore``'s ``.read`` wait, which only frees the source SMEM.
            cute.arch.cp_async_bulk_wait_group(0, read=False)
            # The bulk-store issuer need not be lane 0: order every lane's completed stores before lane 0's add.
            cute.arch.sync_warp()
            lane_idx = cute.arch.lane_idx()
            # An incomplete chunk's word is -1 - (tiles counted): never the ``n_chunk`` the consumer waits for.
            signal_before = Int32(0)
            if lane_idx == 0:
                signal_before = cute.arch.atomic_add(self.n_chunk_signal_ptr + n_chunk, -pending_tile_count, sem="acq_rel", scope="gpu")
            # The other lanes hold 0, so the OR hands every lane lane 0's word.
            counted_tile_count = pending_tile_count - 1 - cute.arch.warp_redux_sync(signal_before, "or")
            # CTA tiles of the chunk: every M tile times the N tiles of this possibly partial chunk.
            n_chunk_cols = cutlass.min(const_expr(config.n_chunk_size), self.output_cols - n_chunk * const_expr(config.n_chunk_size))
            total_rows = padded_offsets[const_expr(config.expert_cnt - 1)]
            n_chunk_tile_count = total_rows // const_expr(config.cta_tile_m) * ceil_div(n_chunk_cols, const_expr(config.cta_tile_n))
            if counted_tile_count == n_chunk_tile_count:
                # Lane 0 holds the acquire; elect.sync's leader is not specified to be lane 0.
                if lane_idx == 0:
                    cute.arch.store(self.n_chunk_signal_ptr + n_chunk, n_chunk, sem="release", scope="gpu")


def streaming_output_notify_offsets(kernel) -> Dict[str, int]:
    """The byte offset of ``signal[]`` inside ``kernel``'s workspace, and its capacity."""
    config = kernel.streaming_output_notify
    return {"signal": kernel.get_workspace_bytes() - config.workspace_bytes, "capacity": config.n_chunk_capacity, "total_bytes": kernel.get_workspace_bytes()}
