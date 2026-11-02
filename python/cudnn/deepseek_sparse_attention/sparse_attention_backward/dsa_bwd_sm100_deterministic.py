# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded-wave deterministic policy for the SM100 M64 DSA backward kernel."""

import math
from typing import Optional, Tuple, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass._mlir.dialects import nvvm
from cutlass.cute.typing import Float32, Int32, Int64, Uint32
from cutlass.cutlass_dsl import dsl_user_op

from .dsa_bwd_sm100 import FlashAttentionDSABackwardSm100

_SUPPORTED_DKV_SHARDS = (32, 64, 128)
_LARGE_CONTEXT_WORKSPACE_BYTES = 16 << 30
_RELAY_WORKSPACE_BUDGET_BYTES = 32 << 30
_RELAY_WAVE_CTAS = 128
_CLUSTER4_MIN_Q = 8192


@dsl_user_op
def _partial_cta_sync(barrier_id: int, thread_count: int, *, loc=None, ip=None):
    """Synchronize a named-barrier subset without an aligned-CTA promise."""
    nvvm.barrier_cta_sync(
        Int32(barrier_id).ir_value(loc=loc, ip=ip),
        thread_count=Int32(thread_count).ir_value(loc=loc, ip=ip),
        aligned=False,
        loc=loc,
        ip=ip,
    )


@cute.jit
def _wait_lower_rank_mask(mask_ptr, lower_rank_mask: Uint32):
    """Wait until every lower cluster rank has published this KV row."""
    current = cute.arch.load(mask_ptr, Uint32, sem="relaxed", scope="gpu")
    loop = cutlass.while_generate([current], lambda value: (value & lower_rank_mask) != Uint32(0))
    with loop as [value]:
        value = cute.arch.load(mask_ptr, Uint32, sem="relaxed", scope="gpu")
        cutlass.yield_out([value])
    # Keep the acquire observation control-dependent: an unused final load is
    # dead-code eliminated before PTX lowering and would lose the release/
    # acquire visibility edge. The relaxed loop remains the common polling
    # path; this second loop normally executes only its initial acquire load.
    acquired = cute.arch.load(mask_ptr, Uint32, sem="acquire", scope="gpu")
    acquire_loop = cutlass.while_generate([acquired], lambda value: (value & lower_rank_mask) != Uint32(0))
    with acquire_loop as [value]:
        value = cute.arch.load(mask_ptr, Uint32, sem="acquire", scope="gpu")
        cutlass.yield_out([value])


def deterministic_dkv_shards(total_seqlen_q: int) -> int:
    """Select 32, 64, or 128 shards from the declared query extent.

    Thirty-two shards are the floor: smaller grids leave most SM100-family
    multiprocessors idle while increasing the number of kernel boundaries.
    The policy targets at most four waves through 512 queries and retains the
    established 128-shard memory bound for larger batches.
    """
    if total_seqlen_q <= 0:
        raise ValueError(f"deterministic DSA backward requires at least one query row, got {total_seqlen_q}")
    required = (int(total_seqlen_q) + 3) // 4
    for num_shards in _SUPPORTED_DKV_SHARDS:
        if required <= num_shards:
            return num_shards
    return _SUPPORTED_DKV_SHARDS[-1]


def use_cluster_dkv_relay(total_seqlen_q: int, total_seqlen_kv: int, head_dim: int) -> bool:
    """Return whether the selected bounded-wave layout exceeds the cap."""
    num_shards = deterministic_dkv_shards(total_seqlen_q)
    if total_seqlen_kv <= 0 or head_dim <= 0:
        raise ValueError("deterministic DSA backward requires positive KV and head extents, " f"got S_kv={total_seqlen_kv}, D={head_dim}")
    k = (int(total_seqlen_kv) + 7) // 8 * 8
    d = (int(head_dim) + 7) // 8 * 8
    return num_shards * k * d * 4 > _LARGE_CONTEXT_WORKSPACE_BYTES


def _relay_workspace_bytes(total_seqlen_q: int, total_seqlen_kv: int, head_dim: int, num_heads: int, cluster_size: int) -> int:
    """Return the complete FP32-plane and packed-mask relay allocation."""
    k = (int(total_seqlen_kv) + 7) // 8 * 8
    d = (int(head_dim) + 7) // 8 * 8
    num_planes = _RELAY_WAVE_CTAS // cluster_size
    mask_rows_per_word = 32 // cluster_size
    num_waves = (int(total_seqlen_q) + _RELAY_WAVE_CTAS - 1) // _RELAY_WAVE_CTAS
    num_head_blocks = (int(num_heads) + 63) // 64
    mask_words = num_waves * num_head_blocks * ((k + mask_rows_per_word - 1) // mask_rows_per_word)
    mask_rows = (mask_words + d - 1) // d
    return num_planes * (k + mask_rows) * d * 4


def deterministic_dkv_relay_cluster_size(total_seqlen_q: int, total_seqlen_kv: int, head_dim: int, num_heads: int = 64) -> int:
    """Select the preferred relay or its minimum-workspace fallback.

    Query batches with at least 8K rows prefer four-CTA relays for reducer
    parallelism; shorter batches prefer eight. Cluster-16 is the
    minimum-workspace fallback when the
    preferred layout would exceed the memory-first budget. The long-query path
    deliberately skips cluster-8: measured GB200 sweeps found it slower than
    cluster-16 while retaining twice as many FP32 planes.
    """
    if total_seqlen_q <= 0:
        raise ValueError(f"deterministic DSA backward requires at least one query row, got {total_seqlen_q}")
    if total_seqlen_kv <= 0 or head_dim <= 0 or num_heads <= 0:
        raise ValueError(
            "deterministic DSA backward requires positive KV, head-dimension, and head-count extents, "
            f"got S_kv={total_seqlen_kv}, D={head_dim}, H={num_heads}"
        )

    preferred_cluster_size = 4 if total_seqlen_q >= _CLUSTER4_MIN_Q else 8
    preferred_bytes = _relay_workspace_bytes(total_seqlen_q, total_seqlen_kv, head_dim, num_heads, preferred_cluster_size)
    return preferred_cluster_size if preferred_bytes <= _RELAY_WORKSPACE_BUDGET_BYTES else 16


class _DeterministicReductionPolicy:
    """Reusable fixed-owner dKV/dSink reduction policy.

    Each launch contains ``num_dkv_shards`` query CTAs. CTA ``i`` is the only
    writer of dKV shard ``i`` in that wave, and same-stream launch ordering
    serializes reuse by later waves. No semaphore or cooperative launch is
    used.
    """

    num_dkv_shards = 128
    q_wave_ctas = num_dkv_shards
    dkv_fold_group_size = 8
    num_dkv_fold_groups = num_dkv_shards // dkv_fold_group_size
    num_dkv_final_terms = num_dkv_fold_groups
    dkv_final_stride = dkv_fold_group_size
    run_dkv_prefold = True
    serialize_head_blocks = True
    dkv_relay_cluster_size = 1

    @classmethod
    def _get_workspace_size_dKV(
        cls,
        k: int,
        d: int,
        b: int,
        acc_dtype: Type[cutlass.Numeric],
    ):
        """Return byte-shaped storage for this specialization's FP32 shards."""
        d = (d + 7) // 8 * 8
        k = (k + 7) // 8 * 8
        workspace_bytes = d * acc_dtype.width // 8
        return (b, cls.num_dkv_shards, k, workspace_bytes)

    @cute.jit
    def make_dKV_accumulator(
        self,
        workspace_dKV: cute.Tensor,
        total_seqlen_KV: Int32,
        head_dim: Int32,
    ) -> cute.Tensor:
        """View deterministic scratch as contiguous FP32 shards."""
        dkv_iter = cute.recast_ptr(workspace_dKV.iterator, dtype=self.acc_dtype)
        return cute.make_tensor(
            dkv_iter,
            cute.make_layout(
                (head_dim, total_seqlen_KV * self.num_dkv_shards, (1, 1)),
                stride=(1, Int64(head_dim), (0, 0)),
            ),
        )

    @cute.jit
    def select_dKV_accumulator(
        self,
        mdKV_acc: cute.Tensor,
        max_seqlen_kv: Int32,
        token_idx: Int32,
    ) -> cute.Tensor:
        """Return the CTA lane's shard using 64-bit pointer arithmetic."""
        head_dim = cute.round_up(self.head_dim, 8)
        shard_idx = token_idx % self.num_dkv_shards
        shard_offset = Int64(shard_idx) * Int64(max_seqlen_kv) * Int64(head_dim)
        return cute.make_tensor(
            mdKV_acc.iterator + shard_offset,
            cute.make_layout(
                (head_dim, max_seqlen_kv, (1, 1)),
                stride=(1, Int64(head_dim), (0, 0)),
            ),
        )

    @cute.kernel
    def fold_dKV_shards(self, mdKV_acc: cute.Tensor, seqlen: Int32):
        """Fold groups of eight shards in a fixed FP32 order, in place."""
        seq_id, group_idx, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        group_begin = group_idx * self.dkv_fold_group_size
        if seq_id < seqlen:
            for dim_idx in cutlass.range(tidx, self.head_dim, 256):
                acc = Float32(0.0)
                for local_shard in cutlass.range_constexpr(self.dkv_fold_group_size):
                    shard_idx = group_begin + local_shard
                    acc += mdKV_acc[dim_idx, seq_id + shard_idx * seqlen, (0, 0)]
                mdKV_acc[dim_idx, seq_id + group_begin * seqlen, (0, 0)] = acc

    @cute.jit
    def finalize_dKV(
        self,
        mdKV_acc: cute.Tensor,
        mdKV: cute.Tensor,
        seqlen: Int32,
        stream: cuda.CUstream,
    ):
        """Launch the optional first fold followed by ordered conversion."""
        if cutlass.const_expr(self.run_dkv_prefold):
            self.fold_dKV_shards(mdKV_acc, seqlen).launch(
                grid=[seqlen, self.num_dkv_fold_groups, 1],
                block=[256, 1, 1],
                stream=stream,
            )
        convert_grid_x = (seqlen + self.block_seq - 1) // self.block_seq
        self.convert(mdKV_acc, mdKV, seqlen).launch(
            grid=[convert_grid_x, 1, 1],
            block=[self.num_threads_D_convert, self.num_threads_seq, 1],
            stream=stream,
        )

    @cute.kernel
    def convert(
        self,
        mdKV_acc: cute.Tensor,
        mdKV: cute.Tensor,
        seqlen: Int32,
    ):
        """Fold the specialization's final terms and store the output dtype."""
        tidx, tidy, _ = cute.arch.thread_idx()
        seq_block_idx, _, batch_idx = cute.arch.block_idx()
        seq_id = self.block_seq * seq_block_idx + tidy
        if seq_id < seqlen:
            cur_mdKV_row = mdKV[None, seq_id, (0, batch_idx)]
            num_128_tiles = self.head_dim_main // 64
            for i in cutlass.range(num_128_tiles, unroll_full=True):
                for j in cutlass.range(2, unroll_full=True):
                    acc = Float32(0.0)
                    for group_idx in cutlass.range_constexpr(self.num_dkv_final_terms):
                        shard_idx = group_idx * self.dkv_final_stride
                        row = mdKV_acc[None, seq_id + shard_idx * seqlen, (0, batch_idx)]
                        tile = cute.flat_divide(cute.flat_divide(row, (64,)), (32,))
                        acc += tile[tidx, j, i]
                    dim_idx = tidx // 4 + tidx % 4 * 8 + j * 32 + i * 64
                    cur_mdKV_row[dim_idx] = self.element_dtype(acc)
            if cutlass.const_expr(not self.same_hdim_kv):
                for j in cutlass.range(2, unroll_full=True):
                    acc = Float32(0.0)
                    for group_idx in cutlass.range_constexpr(self.num_dkv_final_terms):
                        shard_idx = group_idx * self.dkv_final_stride
                        row = mdKV_acc[None, seq_id + shard_idx * seqlen, (0, batch_idx)]
                        tile = cute.flat_divide(cute.flat_divide(row, (64,)), (32,))
                        acc += tile[tidx, j, num_128_tiles]
                    k = tidx // 2 + j * 16
                    dim_idx = self.head_dim_main + (k // 8) * 16 + k % 8 + (tidx % 2) * 8
                    cur_mdKV_row[dim_idx] = self.element_dtype(acc)

    def dSink_grid_q(self, problem_shape):
        """Use one query-reduction block so each head has one dSink writer."""
        return 1

    @cute.kernel
    def sum_dSink(
        self,
        sum_OdO: cute.Tensor,
        scaled_lse: cute.Tensor,
        attn_sink: cute.Tensor,
        dSink: cute.Tensor,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Tuple[Int32, Int32], Int32]],
    ):
        """Reduce each head's dSink across queries in a fixed thread order."""
        _, head_idx, batch_idx = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        q_idx = tidx
        log2_e = Float32(math.log2(math.e))
        sink_log2 = attn_sink[head_idx, (0, batch_idx)] * log2_e
        acc = Float32(0.0)
        while q_idx < problem_shape[0]:
            p_sink = Float32(0.0)
            if sink_log2 == Float32(float("inf")):
                p_sink = Float32(1.0)
            elif sink_log2 == Float32(float("-inf")):
                p_sink = Float32(0.0)
            else:
                p_sink = cute.math.exp2(sink_log2 + scaled_lse[head_idx, (q_idx, batch_idx)])
            acc += p_sink * sum_OdO[head_idx, (q_idx, batch_idx)]
            q_idx += self.dSink_num_threads
        acc = cute.arch.warp_reduction_sum(acc, threads_in_group=self.dSink_num_threads)
        if tidx == 0:
            dSink[head_idx, (0, batch_idx)] = acc


class FlashAttentionDSABackwardSm100Deterministic(_DeterministicReductionPolicy, FlashAttentionDSABackwardSm100):
    """Deterministic reduction layered on the ordinary M64 kernel."""

    # The reduction policy deliberately owns these attributes; spell out the
    # override contract instead of relying on base order alone.
    num_dkv_shards = _DeterministicReductionPolicy.num_dkv_shards
    q_wave_ctas = _DeterministicReductionPolicy.q_wave_ctas
    serialize_head_blocks = _DeterministicReductionPolicy.serialize_head_blocks
    make_dKV_accumulator = _DeterministicReductionPolicy.make_dKV_accumulator
    select_dKV_accumulator = _DeterministicReductionPolicy.select_dKV_accumulator
    finalize_dKV = _DeterministicReductionPolicy.finalize_dKV
    convert = _DeterministicReductionPolicy.convert
    dSink_grid_q = _DeterministicReductionPolicy.dSink_grid_q
    sum_dSink = _DeterministicReductionPolicy.sum_dSink

    @classmethod
    def _get_workspace_size_dKV(cls, k: int, d: int, b: int, acc_dtype: Type[cutlass.Numeric]):
        return _DeterministicReductionPolicy._get_workspace_size_dKV.__func__(cls, k, d, b, acc_dtype)


class FlashAttentionDSABackwardSm100Deterministic32(FlashAttentionDSABackwardSm100Deterministic):
    """Thirty-two-shard specialization for short query batches."""

    num_dkv_shards = 32
    q_wave_ctas = num_dkv_shards
    num_dkv_fold_groups = num_dkv_shards // FlashAttentionDSABackwardSm100Deterministic.dkv_fold_group_size
    num_dkv_final_terms = num_dkv_shards
    dkv_final_stride = 1
    run_dkv_prefold = False


class FlashAttentionDSABackwardSm100Deterministic64(FlashAttentionDSABackwardSm100Deterministic):
    """Sixty-four-shard specialization for medium query batches."""

    num_dkv_shards = 64
    q_wave_ctas = num_dkv_shards
    num_dkv_fold_groups = num_dkv_shards // FlashAttentionDSABackwardSm100Deterministic.dkv_fold_group_size
    num_dkv_final_terms = num_dkv_fold_groups


class _FlashAttentionDSABackwardSm100DeterministicRelay(FlashAttentionDSABackwardSm100Deterministic):
    """Large-context policy with row-aware clustered relays.

    Each launched cluster owns one FP32 plane. A packed per-wave rank mask
    orders only CTAs that update the same KV row; disjoint sparse rows retain
    parallel reducer progress. GPU-scope release/acquire publication fixes
    the accumulation order without waiting on non-resident CTAs.
    """

    q_wave_ctas = _RELAY_WAVE_CTAS
    dkv_fold_group_size = 8
    dkv_final_stride = dkv_fold_group_size
    run_dkv_prefold = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dkv_relay_barrier = pipeline.NamedBarrier(
            barrier_id=11,
            num_threads=self.num_reduce_warps * self.threads_per_warp,
        )

    @classmethod
    def _get_workspace_size_dKV(
        cls,
        k: int,
        d: int,
        b: int,
        acc_dtype: Type[cutlass.Numeric],
    ):
        """Return the data-only fallback shape used without a query extent."""
        d = (d + 7) // 8 * 8
        k = (k + 7) // 8 * 8
        workspace_bytes = d * acc_dtype.width // 8
        return (b, cls.num_dkv_shards, k, workspace_bytes)

    @classmethod
    def _get_workspace_size_dKV_for_q(
        cls,
        q: int,
        k: int,
        d: int,
        h: int,
        b: int,
        acc_dtype: Type[cutlass.Numeric],
    ):
        """Add packed per-wave row-rank masks after each plane's data."""
        d = (d + 7) // 8 * 8
        k = (k + 7) // 8 * 8
        num_waves = (q + cls.q_wave_ctas - 1) // cls.q_wave_ctas
        num_head_blocks = (h + 63) // 64
        mask_words = num_waves * num_head_blocks * ((k + cls.dkv_mask_rows_per_word - 1) // cls.dkv_mask_rows_per_word)
        mask_rows = (mask_words + d - 1) // d
        workspace_bytes = d * acc_dtype.width // 8
        return (b, cls.num_dkv_shards, k + mask_rows, workspace_bytes)

    @cute.jit
    def make_dKV_accumulator(
        self,
        workspace_dKV: cute.Tensor,
        total_seqlen_KV: Int32,
        head_dim: Int32,
    ) -> cute.Tensor:
        """Expose data planes and their trailing packed masks as one view."""
        dkv_iter = cute.recast_ptr(workspace_dKV.iterator, dtype=self.acc_dtype)
        physical_rows = workspace_dKV.shape[2] * self.num_dkv_shards
        return cute.make_tensor(
            dkv_iter,
            cute.make_layout(
                (head_dim, physical_rows, (1, 1)),
                stride=(1, Int64(head_dim), (0, 0)),
            ),
        )

    @cute.jit
    def initialize_dKV_workspace(
        self,
        mdKV_acc: cute.Tensor,
        seqlen: Int32,
        stream: cuda.CUstream,
    ):
        """Clear all data planes and their trailing packed masks."""
        rows_per_cta = 64
        num_rows = mdKV_acc.shape[1]
        self.clear_dKV_workspace(mdKV_acc, num_rows).launch(
            grid=[cute.ceil_div(num_rows, rows_per_cta), 1, 1],
            block=[32, 8, 1],
            stream=stream,
        )

    @cute.jit
    def _physical_rows_per_plane(self, mdKV_acc: cute.Tensor) -> Int32:
        return mdKV_acc.shape[1] // self.num_dkv_shards

    @cute.jit
    def _relay_plane(self, token_idx: Int32) -> Int32:
        return (token_idx % self.q_wave_ctas) // self.dkv_relay_cluster_size

    @cute.jit
    def _mask_word_ptr(
        self,
        mdKV_acc: cute.Tensor,
        max_seqlen_kv: Int32,
        token_idx: Int32,
        head_block_idx: Int32,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Int32, Int32]],
        kv_row: Int32,
    ):
        physical_rows = self._physical_rows_per_plane(mdKV_acc)
        padded_k = cute.round_up(max_seqlen_kv, 8)
        data_rows = padded_k
        words_per_wave = padded_k // self.dkv_mask_rows_per_word
        num_waves = cute.ceil_div(problem_shape[0], self.q_wave_ctas)
        plane_idx = self._relay_plane(token_idx)
        wave_idx = token_idx // self.q_wave_ctas
        phase_idx = head_block_idx * num_waves + wave_idx
        plane_base = Int64(plane_idx) * Int64(physical_rows) * Int64(self.head_dim)
        mask_base = plane_base + Int64(data_rows) * Int64(self.head_dim)
        word_offset = Int64(phase_idx) * Int64(words_per_wave) + Int64(kv_row // self.dkv_mask_rows_per_word)
        return cute.recast_ptr(mdKV_acc.iterator, dtype=Uint32) + mask_base + word_offset

    @cute.jit
    def select_dKV_accumulator(
        self,
        mdKV_acc: cute.Tensor,
        max_seqlen_kv: Int32,
        token_idx: Int32,
    ) -> cute.Tensor:
        """Map all CTAs in one cluster to their shared data plane."""
        physical_rows = self._physical_rows_per_plane(mdKV_acc)
        plane_idx = self._relay_plane(token_idx)
        plane_offset = Int64(plane_idx) * Int64(physical_rows) * Int64(self.head_dim)
        return cute.make_tensor(
            mdKV_acc.iterator + plane_offset,
            cute.make_layout(
                (self.head_dim, max_seqlen_kv, (1, 1)),
                stride=(1, Int64(self.head_dim), (0, 0)),
            ),
        )

    @cute.kernel
    def build_dKV_relay_masks(
        self,
        mdKV_acc: cute.Tensor,
        mTopkIdxs: cute.Tensor,
        mTopkLength: Optional[cute.Tensor],
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Int32, Int32]],
    ):
        """Build packed rank masks for every wave, head block, and KV row."""
        token_idx, _, batch_idx = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        topk = mTopkIdxs.shape[0]
        if cutlass.const_expr(mTopkLength is not None):
            topk = mTopkLength[token_idx]
        topk = min(max(topk, Int32(0)), mTopkIdxs.shape[0])
        rank = token_idx % self.dkv_relay_cluster_size
        bit_in_row = Uint32(1) << Uint32(rank)
        num_head_blocks = cute.ceil_div(problem_shape[3][0], self.block_tile)
        for slot in cutlass.range(tidx, topk, 256):
            kv_row = mTopkIdxs[slot, (token_idx, batch_idx)]
            if kv_row >= 0 and kv_row < problem_shape[1]:
                row_shift = Uint32((kv_row % self.dkv_mask_rows_per_word) * self.dkv_relay_cluster_size)
                bit = bit_in_row << row_shift
                for head_block_idx in cutlass.range(num_head_blocks, unroll=1):
                    mask_ptr = self._mask_word_ptr(mdKV_acc, problem_shape[1], token_idx, head_block_idx, problem_shape, kv_row)
                    cute.arch.atomic_or(mask_ptr.llvm_ptr, bit, sem="relaxed", scope="gpu")

    @cute.jit
    def prepare_dKV_relay(
        self,
        mdKV_acc: cute.Tensor,
        mTopkIdxs: cute.Tensor,
        mTopkLength: Optional[cute.Tensor],
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Int32, Int32]],
        stream: cuda.CUstream,
    ):
        self.build_dKV_relay_masks(mdKV_acc, mTopkIdxs, mTopkLength, problem_shape).launch(
            grid=[problem_shape[0], 1, problem_shape[3][1]],
            block=[256, 1, 1],
            stream=stream,
        )

    @cute.jit
    def acquire_dKV_tile_relay(
        self,
        mdKV_acc: cute.Tensor,
        mTopkIdxs: cute.Tensor,
        max_seqlen_kv: Int32,
        tile_index: Int32,
        topk: Int32,
        token_idx: Int32,
        head_block_idx: Int32,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Int32, Int32]],
    ):
        """Import predecessor releases before reducers update one dKV tile."""
        tidx, _, _ = cute.arch.thread_idx()
        _, _, batch_idx = cute.arch.block_idx()
        tidx_in_wg = tidx - self.reduce_warp_id[0] * self.threads_per_warp
        rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        # Rank zero owns the first update and therefore has no predecessor
        # release to import. Avoid a CTA barrier and GPU fence on every tile.
        lower_rank_bits = (Uint32(1) << Uint32(rank)) - Uint32(1)
        if tidx_in_wg < self.block_tile and rank > 0:
            tile_begin = tile_index * self.block_tile
            slot = tile_begin + tidx_in_wg
            if slot < topk:
                kv_row = mTopkIdxs[slot, (token_idx, batch_idx)]
                if kv_row >= 0 and kv_row < max_seqlen_kv:
                    row_shift = Uint32((kv_row % self.dkv_mask_rows_per_word) * self.dkv_relay_cluster_size)
                    lower_rank_mask = lower_rank_bits << row_shift
                    mask_ptr = self._mask_word_ptr(mdKV_acc, max_seqlen_kv, token_idx, head_block_idx, problem_shape, kv_row)
                    _wait_lower_rank_mask(mask_ptr, lower_rank_mask)
        # The slot lanes import predecessor releases. The CTA barrier carries
        # that acquire causality to every reducer thread before its following
        # GMEM updates; an additional GPU-wide fence is redundant.
        if rank > 0:
            _partial_cta_sync(self.dkv_relay_barrier.barrier_id, self.dkv_relay_barrier.num_threads)

    @cute.jit
    def release_dKV_tile_relay(
        self,
        mdKV_acc: cute.Tensor,
        mTopkIdxs: cute.Tensor,
        max_seqlen_kv: Int32,
        tile_index: Int32,
        topk: Int32,
        token_idx: Int32,
        head_block_idx: Int32,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Int32, Int32]],
    ):
        """Publish one dKV tile before later cluster ranks acquire it."""
        tidx, _, _ = cute.arch.thread_idx()
        _, _, batch_idx = cute.arch.block_idx()
        tidx_in_wg = tidx - self.reduce_warp_id[0] * self.threads_per_warp
        rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        # No later rank can observe the last rank's ownership bit.
        # Publish every reducer thread's relaxed FP32 atomics through the CTA
        # barrier before a slot lane release-clears this rank's ownership bit.
        # CTA-barrier causality is transitive into the release operation, so
        # no GPU-wide fence is required here.
        if rank < self.dkv_relay_cluster_size - 1:
            _partial_cta_sync(self.dkv_relay_barrier.barrier_id, self.dkv_relay_barrier.num_threads)
        if tidx_in_wg < self.block_tile and rank < self.dkv_relay_cluster_size - 1:
            tile_begin = tile_index * self.block_tile
            slot = tile_begin + tidx_in_wg
            if slot < topk:
                kv_row = mTopkIdxs[slot, (token_idx, batch_idx)]
                if kv_row >= 0 and kv_row < max_seqlen_kv:
                    row_shift = Uint32((kv_row % self.dkv_mask_rows_per_word) * self.dkv_relay_cluster_size)
                    bit_shift = row_shift + Uint32(rank)
                    mask_ptr = self._mask_word_ptr(mdKV_acc, max_seqlen_kv, token_idx, head_block_idx, problem_shape, kv_row)
                    word = cute.arch.load(mask_ptr, Uint32, sem="relaxed", scope="gpu")
                    row_mask = (word >> row_shift) & Uint32(self.dkv_row_rank_mask)
                    current_bit = Uint32(1) << Uint32(rank)
                    if (row_mask & ~current_bit) != Uint32(0):
                        keep_mask = ~(Uint32(1) << bit_shift)
                        cute.arch.atomic_and(mask_ptr.llvm_ptr, keep_mask, sem="release", scope="gpu")
        # The next participating relay barrier is the rendezvous after the
        # slot lanes publish their release; kernel completion covers the final
        # tile. A second barrier here only stalls the non-slot reducer warps.

    @cute.kernel
    def fold_dKV_shards(self, mdKV_acc: cute.Tensor, seqlen: Int32):
        """Fold plane groups in fixed order while skipping trailing mask rows."""
        seq_id, group_idx, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        group_begin = group_idx * self.dkv_fold_group_size
        physical_rows = self._physical_rows_per_plane(mdKV_acc)
        if seq_id < seqlen:
            for dim_idx in cutlass.range(tidx, self.head_dim, 256):
                acc = Float32(0.0)
                for local_shard in cutlass.range_constexpr(self.dkv_fold_group_size):
                    shard_idx = group_begin + local_shard
                    row_idx = seq_id + shard_idx * physical_rows
                    acc += mdKV_acc[dim_idx, row_idx, (0, 0)]
                mdKV_acc[dim_idx, seq_id + group_begin * physical_rows, (0, 0)] = acc

    @cute.kernel
    def convert(
        self,
        mdKV_acc: cute.Tensor,
        mdKV: cute.Tensor,
        seqlen: Int32,
    ):
        """Fold the relay planes in fixed order and convert to output dtype."""
        tidx, tidy, _ = cute.arch.thread_idx()
        seq_block_idx, _, batch_idx = cute.arch.block_idx()
        seq_id = self.block_seq * seq_block_idx + tidy
        physical_rows = self._physical_rows_per_plane(mdKV_acc)
        if seq_id < seqlen:
            cur_mdKV_row = mdKV[None, seq_id, (0, batch_idx)]
            num_128_tiles = self.head_dim_main // 64
            for i in cutlass.range(num_128_tiles, unroll_full=True):
                for j in cutlass.range(2, unroll_full=True):
                    acc = Float32(0.0)
                    for group_idx in cutlass.range_constexpr(self.num_dkv_final_terms):
                        shard_idx = group_idx * self.dkv_final_stride
                        row = mdKV_acc[None, seq_id + shard_idx * physical_rows, (0, batch_idx)]
                        tile = cute.flat_divide(cute.flat_divide(row, (64,)), (32,))
                        acc += tile[tidx, j, i]
                    dim_idx = tidx // 4 + tidx % 4 * 8 + j * 32 + i * 64
                    cur_mdKV_row[dim_idx] = self.element_dtype(acc)
            if cutlass.const_expr(not self.same_hdim_kv):
                for j in cutlass.range(2, unroll_full=True):
                    acc = Float32(0.0)
                    for group_idx in cutlass.range_constexpr(self.num_dkv_final_terms):
                        shard_idx = group_idx * self.dkv_final_stride
                        row = mdKV_acc[None, seq_id + shard_idx * physical_rows, (0, batch_idx)]
                        tile = cute.flat_divide(cute.flat_divide(row, (64,)), (32,))
                        acc += tile[tidx, j, num_128_tiles]
                    k = tidx // 2 + j * 16
                    dim_idx = self.head_dim_main + (k // 8) * 16 + k % 8 + (tidx % 2) * 8
                    cur_mdKV_row[dim_idx] = self.element_dtype(acc)


class FlashAttentionDSABackwardSm100DeterministicCluster8(_FlashAttentionDSABackwardSm100DeterministicRelay):
    """Sixteen-plane specialization using eight-CTA relays."""

    num_dkv_shards = 16
    dkv_relay_cluster_size = 8
    dkv_mask_rows_per_word = 32 // dkv_relay_cluster_size
    dkv_row_rank_mask = (1 << dkv_relay_cluster_size) - 1
    dkv_cluster_shape = (dkv_relay_cluster_size, 1, 1)
    num_dkv_fold_groups = num_dkv_shards // _FlashAttentionDSABackwardSm100DeterministicRelay.dkv_fold_group_size
    run_dkv_prefold = False
    num_dkv_final_terms = num_dkv_shards
    dkv_final_stride = 1


class FlashAttentionDSABackwardSm100DeterministicCluster16(_FlashAttentionDSABackwardSm100DeterministicRelay):
    """Eight-plane memory-first specialization using sixteen-CTA relays."""

    num_dkv_shards = 8
    dkv_relay_cluster_size = 16
    dkv_mask_rows_per_word = 32 // dkv_relay_cluster_size
    dkv_row_rank_mask = (1 << dkv_relay_cluster_size) - 1
    dkv_cluster_shape = (dkv_relay_cluster_size, 1, 1)
    num_dkv_fold_groups = num_dkv_shards // _FlashAttentionDSABackwardSm100DeterministicRelay.dkv_fold_group_size
    # Eight planes fit comfortably in the conversion kernel's ordered loop.
    # Sum them there instead of writing and rereading an FP32 prefold plane.
    run_dkv_prefold = False
    num_dkv_final_terms = num_dkv_shards
    dkv_final_stride = 1


class FlashAttentionDSABackwardSm100DeterministicCluster4(_FlashAttentionDSABackwardSm100DeterministicRelay):
    """Thirty-two-plane specialization using four-CTA relays."""

    num_dkv_shards = 32
    dkv_relay_cluster_size = 4
    dkv_mask_rows_per_word = 32 // dkv_relay_cluster_size
    dkv_row_rank_mask = (1 << dkv_relay_cluster_size) - 1
    dkv_cluster_shape = (dkv_relay_cluster_size, 1, 1)
    num_dkv_fold_groups = num_dkv_shards // _FlashAttentionDSABackwardSm100DeterministicRelay.dkv_fold_group_size
    run_dkv_prefold = False
    num_dkv_final_terms = num_dkv_shards
    dkv_final_stride = 1


_DETERMINISTIC_KERNELS = {
    32: FlashAttentionDSABackwardSm100Deterministic32,
    64: FlashAttentionDSABackwardSm100Deterministic64,
    128: FlashAttentionDSABackwardSm100Deterministic,
}

_DETERMINISTIC_RELAY_KERNELS = {
    4: FlashAttentionDSABackwardSm100DeterministicCluster4,
    8: FlashAttentionDSABackwardSm100DeterministicCluster8,
    16: FlashAttentionDSABackwardSm100DeterministicCluster16,
}


def deterministic_kernel_class(
    total_seqlen_q: int,
    total_seqlen_kv: int | None = None,
    head_dim: int | None = None,
    num_heads: int = 64,
):
    """Return the compile-time deterministic specialization for one plan."""
    if total_seqlen_kv is not None or head_dim is not None:
        if total_seqlen_kv is None or head_dim is None:
            raise ValueError("total_seqlen_kv and head_dim must be provided together")
        if use_cluster_dkv_relay(total_seqlen_q, total_seqlen_kv, head_dim):
            relay_width = deterministic_dkv_relay_cluster_size(total_seqlen_q, total_seqlen_kv, head_dim, num_heads)
            return _DETERMINISTIC_RELAY_KERNELS[relay_width]
    return _DETERMINISTIC_KERNELS[deterministic_dkv_shards(total_seqlen_q)]


def _deterministic_kernel_class_for_shards(num_shards: int):
    """Return the deterministic specialization for an already selected shard count."""
    try:
        return _DETERMINISTIC_KERNELS[num_shards]
    except KeyError as exc:
        raise ValueError(f"unsupported deterministic dKV shard count {num_shards}") from exc


def _deterministic_kernel_class_for_config(num_shards: int, relay_cluster_size: int):
    """Reconstruct the plan-time specialization from its cache-key fields."""
    if relay_cluster_size == 1:
        return _deterministic_kernel_class_for_shards(num_shards)
    try:
        kernel_cls = _DETERMINISTIC_RELAY_KERNELS[relay_cluster_size]
    except KeyError as exc:
        raise ValueError(f"unsupported deterministic dKV relay cluster size {relay_cluster_size}") from exc
    if kernel_cls.num_dkv_shards != num_shards:
        raise ValueError("inconsistent deterministic dKV relay configuration: " f"{num_shards} planes with cluster size {relay_cluster_size}")
    return kernel_cls
