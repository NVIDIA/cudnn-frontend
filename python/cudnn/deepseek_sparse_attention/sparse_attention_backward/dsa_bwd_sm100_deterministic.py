# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded-wave deterministic policy for the SM100 M64 DSA backward kernel."""

import math
from typing import Tuple, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Float32, Int32, Int64

from .dsa_bwd_sm100 import FlashAttentionDSABackwardSm100


class FlashAttentionDSABackwardSm100Deterministic(FlashAttentionDSABackwardSm100):
    """Deterministic dKV/dSink reduction layered on the ordinary M64 kernel.

    Each launch contains 128 query CTAs. CTA ``i`` is the only writer of dKV
    shard ``i`` in that wave, and same-stream launch ordering serializes reuse
    of the shard by later waves. No semaphore or cooperative launch is used.
    """

    num_dkv_shards = 128
    q_wave_ctas = num_dkv_shards
    dkv_fold_group_size = 8
    num_dkv_fold_groups = num_dkv_shards // dkv_fold_group_size
    serialize_head_blocks = True

    @staticmethod
    def _get_workspace_size_dKV(
        k: int,
        d: int,
        b: int,
        acc_dtype: Type[cutlass.Numeric],
    ):
        """Return byte-shaped storage for all 128 padded FP32 shards."""
        d = (d + 7) // 8 * 8
        k = (k + 7) // 8 * 8
        workspace_bytes = d * acc_dtype.width // 8
        return (b, FlashAttentionDSABackwardSm100Deterministic.num_dkv_shards, k, workspace_bytes)

    @cute.jit
    def make_dKV_accumulator(
        self,
        workspace_dKV: cute.Tensor,
        total_seqlen_KV: Int32,
        head_dim: Int32,
    ) -> cute.Tensor:
        """View deterministic scratch as 128 contiguous FP32 shards."""
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
        """Launch the fixed first fold followed by ordered conversion."""
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
        """Fold the 16 group results in order and store the output dtype."""
        tidx, tidy, _ = cute.arch.thread_idx()
        seq_block_idx, _, batch_idx = cute.arch.block_idx()
        seq_id = self.block_seq * seq_block_idx + tidy
        if seq_id < seqlen:
            cur_mdKV_row = mdKV[None, seq_id, (0, batch_idx)]
            num_128_tiles = self.head_dim_main // 64
            for i in cutlass.range(num_128_tiles, unroll_full=True):
                for j in cutlass.range(2, unroll_full=True):
                    acc = Float32(0.0)
                    for group_idx in cutlass.range_constexpr(self.num_dkv_fold_groups):
                        shard_idx = group_idx * self.dkv_fold_group_size
                        row = mdKV_acc[None, seq_id + shard_idx * seqlen, (0, batch_idx)]
                        tile = cute.flat_divide(cute.flat_divide(row, (64,)), (32,))
                        acc += tile[tidx, j, i]
                    dim_idx = tidx // 4 + tidx % 4 * 8 + j * 32 + i * 64
                    cur_mdKV_row[dim_idx] = self.element_dtype(acc)
            if cutlass.const_expr(not self.same_hdim_kv):
                for j in cutlass.range(2, unroll_full=True):
                    acc = Float32(0.0)
                    for group_idx in cutlass.range_constexpr(self.num_dkv_fold_groups):
                        shard_idx = group_idx * self.dkv_fold_group_size
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
            p_sink = cute.math.exp2(sink_log2 + scaled_lse[head_idx, (q_idx, batch_idx)])
            acc += p_sink * sum_OdO[head_idx, (q_idx, batch_idx)]
            q_idx += self.dSink_num_threads
        acc = cute.arch.warp_reduction_sum(acc, threads_in_group=self.dSink_num_threads)
        if tidx == 0:
            dSink[head_idx, (0, batch_idx)] = acc
