# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The compact-SFA SF warp of the block-scaled MXFP8 grouped GEMM kernels (SM100, cluster (2, 1) with the 2-CTA MMA).

An EP dispatch delivers each token's E8M0 scales as a compact row ``[K / 32]``, not as SF atoms. With
``compact_sfa_sf_warp`` a dedicated SF warp reads each tile's 128 rows per CTA and stores the stage's SF atom itself, and
the TMA warp loads no SFA. ``CompactSfaSfWarpMixin`` holds that warp's loop, its checks and the MMA warp's acquire for
any kernel whose A/B pipeline, sSFA atoms and tile_info slots follow the GLU kernel's layout.

The handoff to the leader's MMA (both CTAs' atoms, read by ``tcgen05.cp``) is the PTX ``sync_restrict`` mbarrier pattern
for shared memory, per K tile:

- SF warp, each lane: ``st.shared::cta`` of its rows; ``fence.proxy.async.shared::cta``; ``bar.warp.sync``; the elected
  lane: ``fence.release.sync_restrict::shared::cta.cluster``, then ``mbarrier.arrive.relaxed.cluster`` on the leader's
  full barrier.
- MMA leader: ``mbarrier.try_wait.relaxed.cluster`` on that barrier, then
  ``fence.acquire.sync_restrict::shared::cluster.cluster``, then ``tcgen05.cp``.

The release covers the warp's ``.shared::cta`` stores (the other lanes' through ``bar.warp.sync``), the acquire the
``.shared::cluster`` reads after it, and each lane's proxy fence hands the generic stores to the async proxy.  See
docs/operations/moe/streaming_signal_contract.md (section 4a).
"""

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass._mlir import ir
from cutlass._mlir.dialects import nvvm, vector

from cudnn.frost.tile_dsl.consume import consume_then_arrive

from .moe_kernel_helpers import (
    fence_acquire_sync_restrict_shared_cluster_cluster,
    fence_release_sync_restrict_shared_cta_cluster,
    mbarrier_try_wait_parity_relaxed_cluster,
)
from .moe_utils import compute_expert_token_range


def check_compact_sfa_options(compact_sfa_sf_warp: bool, sf_vec_size: int, use_2cta_instrs: bool, cluster_shape_mn) -> None:
    """Raise unless the construction options admit ``compact_sfa_sf_warp``."""
    if compact_sfa_sf_warp and (sf_vec_size != 32 or not use_2cta_instrs or tuple(cluster_shape_mn) != (2, 1)):
        raise ValueError("compact_sfa_sf_warp requires MXFP8 (sf_vec_size 32), cluster (2, 1) with the 2-CTA MMA")


class CompactSfaSfWarpMixin:
    """The SF warp.  The kernel sets ``compact_sfa_sf_warp``, ``sf_vec_size``, ``num_ab_stage``, ``num_tile_stage`` and
    ``cta_tile_shape_mnk``, counts 2 more producer arrivals per stage on its ab_pipeline (each CTA's SF warp), and calls
    ``compact_sfa_acquire`` in its MMA warp after each A/B full wait."""

    #: K tiles of compact SFA words the SF warp loads at once: one 16-byte L2-only load per row (K % 512 == 0).
    SF_WARP_K_TILES_PER_LOAD = 4

    def check_compact_sfa_attributes(self, a) -> None:
        """In ``__call__`` after ``_setup_attributes``: whole 16-byte loads of the compact rows, the 4 stages the SF warp
        holds between its empty waits and its arrives, and one SF atom (4 scales per row) per K tile."""
        if not self.compact_sfa_sf_warp:
            return
        if cute.is_static(a.shape[1]) and cute.size(a, mode=[1]) % (self.SF_WARP_K_TILES_PER_LOAD * 4 * self.sf_vec_size) != 0:
            raise ValueError(f"compact_sfa_sf_warp requires K % 512 == 0, got K = {cute.size(a, mode=[1])}")
        if self.num_ab_stage < self.SF_WARP_K_TILES_PER_LOAD:
            raise ValueError(f"compact_sfa_sf_warp needs >= {self.SF_WARP_K_TILES_PER_LOAD} A/B stages, got {self.num_ab_stage}")
        if self.cta_tile_shape_mnk[2] != 4 * self.sf_vec_size:
            raise ValueError(f"compact_sfa_sf_warp needs one SF atom (4 scales per row) per K tile, got a K tile of {self.cta_tile_shape_mnk[2]}")

    @cute.jit
    def compact_sfa_acquire(self, ab_pipeline, ab_consumer_state) -> None:
        """MMA warp, after the stage's full wait: acquire both SF warps' atom stores (the module docstring's pattern)."""
        if cutlass.const_expr(self.compact_sfa_sf_warp):
            full_barrier = ab_pipeline.sync_object_full.get_barrier(ab_consumer_state.index)
            sf_acquired = cutlass.Boolean(False)
            while not sf_acquired:
                sf_acquired = mbarrier_try_wait_parity_relaxed_cluster(full_barrier, ab_consumer_state.phase)
            fence_acquire_sync_restrict_shared_cluster_cluster()

    @cute.jit
    def compact_sfa_sf_warp_loop(self, compact_sfa_words, sSFA, mA_mkl, padded_offsets, sInfo, tile_info_pipeline, ab_pipeline, leader_cta, lane_idx, zero):
        """Per tile and K tile: wait for the stage, store this CTA's SF atom from the compact rows, release it to the
        leader's full barrier.  Lane l owns rows l + 32 r (r < 4) of the CTA's 128 rows; a row's word of a K tile is its
        4 scales."""
        rows_total, k = mA_mkl.shape[0], mA_mkl.shape[1]
        words_per_row = k // (4 * self.sf_vec_size)
        # (row, row_group, K tile) words of the compact rows; (lane, row_group, stage) words of the SF atoms in sSFA.
        words = cute.make_tensor(compact_sfa_words, cute.make_layout((rows_total, 4, words_per_row), stride=(words_per_row, 32 * words_per_row, 1)))
        lines = cute.make_tensor(cute.recast_ptr(sSFA.iterator, dtype=cutlass.Int32), cute.make_layout((32, 4, self.num_ab_stage), stride=(4, 1, 128)))
        group = self.SF_WARP_K_TILES_PER_LOAD
        # Two groups of K tiles in flight: a group's loads are issued right after its slots were stored, 2 groups ahead.
        queue = [cute.make_rmem_tensor(cute.make_layout(4), cutlass.Int32) for _ in range(2 * group)]
        sf_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_ab_stage)
        tile_info_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_tile_stage)
        tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
        is_valid_tile = self._sf_warp_next_tile(sInfo, tile_info, tile_info_pipeline, tile_info_consumer_state, zero)
        tile_info_consumer_state.advance()
        while is_valid_tile:
            k_tile_cnt = tile_info[3]
            token_offset, _ = compute_expert_token_range(padded_offsets, tile_info[0])
            # A 64-bit row: row x words per row passes 2^31 at large M.
            row = cutlass.Int64(token_offset) + cutlass.Int64(tile_info[1]) * self.cta_tile_shape_mnk[0] + cutlass.Int64(lane_idx)
            rows = words[(row, None, None)]
            self._load_compact_sfa_group(queue[:group], rows, 0)
            if group < k_tile_cnt:
                self._load_compact_sfa_group(queue[group:], rows, group)
            sf_producer_state.reset_count()
            # k_tile_cnt % 4 == 0 (K % 512 == 0): whole groups of K tiles.
            for k_tile_base in cutlass.range(0, k_tile_cnt, 2 * group, unroll=1):
                for half in cutlass.range_constexpr(2):
                    if k_tile_base + half * group < k_tile_cnt:
                        for j in cutlass.range_constexpr(group):
                            self._store_compact_sfa_stage(lines, queue[half * group + j], ab_pipeline, sf_producer_state, leader_cta, lane_idx)
                            sf_producer_state.advance()
                        # This group's slots are stored: load the group two ahead into them.
                        if k_tile_base + half * group + 2 * group < k_tile_cnt:
                            self._load_compact_sfa_group(queue[half * group : (half + 1) * group], rows, k_tile_base + half * group + 2 * group)
            is_valid_tile = self._sf_warp_next_tile(sInfo, tile_info, tile_info_pipeline, tile_info_consumer_state, zero)
            tile_info_consumer_state.advance()

    @cute.jit
    def _store_compact_sfa_stage(self, lines, slot, ab_pipeline, state, leader_cta, lane_idx):
        """One K tile: wait for the stage, store this lane's 4 words of the CTA's atom, release them to the leader."""
        ab_pipeline.sync_object_empty.wait(state.index, state.phase)
        # The MMA's tcgen05.cp read this stage through the async proxy: order that before the stores.
        cute.arch.fence_proxy("async.shared", space="cta")
        for row_group in cutlass.range_constexpr(4):
            lines[(lane_idx, row_group, state.index)] = slot[row_group]
        # tcgen05.cp reads the atoms through the async proxy (the peer's from the leader's MMA).
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_warp()
        with cute.arch.elect_one():
            fence_release_sync_restrict_shared_cta_cluster()
            cute.arch.mbarrier_arrive(ab_pipeline.producer_get_barrier(state), leader_cta, relaxed=True, scope=nvvm.MemScopeKind.CLUSTER)

    @cute.jit
    def _load_compact_sfa_group(self, queue, rows, first_k_tile):
        """K tiles ``first_k_tile`` .. + 3 of this lane's 4 rows, one 16-byte L2-only load per row: the producer wrote them
        while this kernel ran, so no L1 line may serve them."""
        for row_group in cutlass.range_constexpr(4):
            vec = cute.arch.load(
                (rows.iterator + cute.crd2idx((row_group, first_k_tile), rows.layout)).llvm_ptr,
                ir.VectorType.get([self.SF_WARP_K_TILES_PER_LOAD], cutlass.Int32.mlir_type),
                cop="cg",
            )
            for j in cutlass.range_constexpr(self.SF_WARP_K_TILES_PER_LOAD):
                queue[j][row_group] = cutlass.Int32(vector.extract(vec, [], [j]))

    @cute.jit
    def _sf_warp_next_tile(self, sInfo, tile_info, tile_info_pipeline, tile_info_consumer_state, zero):
        """Read the next tile_info slot and release it with an arrive that depends on the values read (consume-then-arrive slot release)."""
        tile_info_pipeline.consumer_wait(tile_info_consumer_state)
        for idx in cutlass.range(4, unroll_full=True):
            tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
        consume_then_arrive(tile_info_pipeline.sync_object_empty.get_barrier(tile_info_consumer_state.index), [tile_info[idx] for idx in range(4)], zero)
        return tile_info[0] >= cutlass.Int32(0)
