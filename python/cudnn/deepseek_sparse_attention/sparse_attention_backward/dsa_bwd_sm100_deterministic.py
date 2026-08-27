# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded-wave deterministic policy for the SM100 H64 DSA backward kernel."""

import math
from typing import Tuple, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Float32, Int32, Int64

from .dsa_bwd_sm100 import FlashAttentionDSABackwardSm100


class FlashAttentionDSABackwardSm100Deterministic(FlashAttentionDSABackwardSm100):
    """Deterministic dKV/dSink reduction layered on the ordinary H64 kernel.

    Each launch contains 128 query CTAs. CTA ``i`` is the only writer of dKV
    shard ``i`` in that wave, and same-stream launch ordering serializes reuse
    of the shard by later waves. No semaphore or cooperative launch is used.
    """

    num_dkv_shards = 128
    dkv_fold_group_size = 8
    num_dkv_fold_groups = num_dkv_shards // dkv_fold_group_size

    def __init__(
        self,
        element_dtype: Type[cutlass.Numeric],
        head_dim: int,
        head_dim_v: int,
        block_tile: int,
        max_topk: int = 0,
    ):
        super().__init__(
            element_dtype=element_dtype,
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            block_tile=block_tile,
            max_topk=max_topk,
        )
        assert block_tile == 64, "bounded-wave deterministic DSA backward requires the H64 M64 kernel"
        self.q_wave_ctas = self.num_dkv_shards

    @staticmethod
    def _get_workspace_size_dKV(
        k: int,
        d: int,
        b: int,
        acc_dtype: Type[cutlass.Numeric],
    ):
        d = (d + 7) // 8 * 8
        k = (k + 7) // 8 * 8
        workspace_bytes = d * acc_dtype.width // 8
        return (b, FlashAttentionDSABackwardSm100Deterministic.num_dkv_shards, k, workspace_bytes)

    def get_workspace_tensor(
        self,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Int32, Int32]],
        workspace_LSE_OdO: cute.Tensor,
        workspace_dKV: cute.Tensor,
        total_seqlen_Q: Int32,
        total_seqlen_KV: Int32,
        acc_dtype: Type[cutlass.Numeric],
    ) -> Tuple[cute.Tensor, cute.Tensor, cute.Tensor]:
        sum_OdO, scaled_lse, _ = super().get_workspace_tensor(
            problem_shape,
            workspace_LSE_OdO,
            workspace_dKV,
            total_seqlen_Q,
            total_seqlen_KV,
            acc_dtype,
        )
        head_dim = cute.round_up(problem_shape[2], 8)
        dkv_iter = cute.recast_ptr(workspace_dKV.iterator, dtype=self.acc_dtype)
        dkv_acc = cute.make_tensor(
            dkv_iter,
            cute.make_layout(
                (head_dim, total_seqlen_KV * self.num_dkv_shards, (1, 1)),
                stride=(1, Int64(head_dim), (0, 0)),
            ),
        )
        return sum_OdO, scaled_lse, dkv_acc

    @cute.jit
    def prepare_dKV_workspace(self, mdKV_acc: cute.Tensor, mdKV: cute.Tensor):
        return mdKV_acc

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
