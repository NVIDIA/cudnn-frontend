# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded scratch and ordered reductions for deterministic D512 two-CTA DSA.

Wave-local queries own separate scratch shards, with additional shards for
some warp-group writers. Cluster ranks own disjoint columns within each shard.
The main kernel orders duplicate-index writers, and fixed-order folds combine
the FP32 shards before BF16 output conversion.
"""

import math
from typing import Type
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.cute.typing import Float32, Int32, Int64


@dsl_user_op
def _reduce_dkv_quad(target: cute.Pointer, value_0: Float32, value_1: Float32, value_2: Float32, value_3: Float32, *, loc=None, ip=None) -> None:
    """Atomically add four FP32 values without returning the old values."""
    target_i64 = target.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [
            target_i64,
            Float32(value_0).ir_value(loc=loc, ip=ip),
            Float32(value_1).ir_value(loc=loc, ip=ip),
            Float32(value_2).ir_value(loc=loc, ip=ip),
            Float32(value_3).ir_value(loc=loc, ip=ip),
        ],
        "red.global.v4.f32.add [$0], {$1, $2, $3, $4};",
        "l,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def ordered_dkv_quad(target: cute.Tensor, values: cute.Tensor):
    """Add four FP32 values when the caller has established one writer.

    Warp/CTA barriers order different duplicate-slot owners. Same-thread,
    same-element atomics retain program order under PTX SC-per-location.
    Cluster ranks must own disjoint columns or shards. This uses the original
    two-CTA FP32 atomic operation, including its global-memory RN/FTZ behavior.
    """
    _reduce_dkv_quad(target.iterator, values[0], values[1], values[2], values[3])


class TwoCTADeterministicReduction:
    """Scratch layout and dSink reductions for deterministic D512 two-CTA DSA."""

    deterministic = True
    # The public workspace reserves 128 planes. Subclasses may use fewer
    # active planes while keeping that caller-visible capacity unchanged.
    num_dkv_shards = 128
    max_query_pairs_per_wave = 74
    split_wg_queries_per_wave = num_dkv_shards - max_query_pairs_per_wave
    dkv_fold_group_size = 8

    @staticmethod
    def _get_workspace_size_dKV(k: int, d: int, b: int, acc_dtype: Type[cutlass.Numeric]):
        return (b, 128, (k + 7) // 8 * 8, (d + 7) // 8 * 8 * (acc_dtype.width // 8))

    @cute.jit
    def make_dKV_accumulator(self, workspace_dKV: cute.Tensor, total_seqlen_KV: Int32):
        pitch = cute.round_up(total_seqlen_KV, 8)
        return cute.make_tensor(
            cute.recast_ptr(workspace_dKV.iterator, dtype=Float32),
            cute.make_layout((self.head_dim, pitch * self.num_dkv_shards, (1, 1)), stride=(1, Int64(self.head_dim), (0, 0))),
        )

    @cute.jit
    def select_dKV_accumulator(self, mdKV_acc: cute.Tensor, total_seqlen_KV: Int32, token_idx: Int32, rank: Int32 = 0):
        # The wave-local query ordinal selects its plane; the first queries
        # reserve an adjacent second plane for a separate warp-group writer.
        query_shard = token_idx
        split_wg_shard = query_shard < Int32(self.split_wg_queries_per_wave)
        shard = query_shard + min(query_shard, Int32(self.split_wg_queries_per_wave))
        pitch = cute.round_up(total_seqlen_KV, 8)
        offset = Int64(shard) * Int64(pitch) * Int64(self.head_dim)
        accumulator = cute.make_tensor(
            mdKV_acc.iterator + offset, cute.make_layout((self.head_dim, total_seqlen_KV, (1, 1)), stride=(1, Int64(self.head_dim), (0, 0)))
        )
        return (accumulator, split_wg_shard)

    @cute.jit
    def _sum_dSink_partial_head_deterministic(
        self,
        sum_odo: cute.Tensor,
        scaled_lse: cute.Tensor,
        attn_sink: cute.Tensor,
        problem_shape,
        q_block_idx: Int32,
        head: Int32,
        chunk_size: cutlass.Constexpr[int],
    ):
        """Reduce one fixed query chunk for one head into consumed storage."""
        sink_log2 = Float32(attn_sink[head, (0, 0)]) * Float32(math.log2(math.e))
        acc_0 = Float32(0.0)
        acc_1 = Float32(0.0)
        acc_2 = Float32(0.0)
        acc_3 = Float32(0.0)
        q_begin = q_block_idx * Int32(chunk_size)
        q_end = min(problem_shape[0], q_begin + Int32(chunk_size))
        q_idx = q_begin
        while q_idx + 3 < q_end:
            p_0 = cute.math.exp2(sink_log2 + scaled_lse[head, (q_idx, 0)])
            p_1 = cute.math.exp2(sink_log2 + scaled_lse[head, (q_idx + 1, 0)])
            p_2 = cute.math.exp2(sink_log2 + scaled_lse[head, (q_idx + 2, 0)])
            p_3 = cute.math.exp2(sink_log2 + scaled_lse[head, (q_idx + 3, 0)])
            if sink_log2 == Float32(float("inf")):
                p_0 = p_1 = p_2 = p_3 = Float32(1.0)
            elif sink_log2 == Float32(float("-inf")):
                p_0 = p_1 = p_2 = p_3 = Float32(0.0)
            acc_0 += p_0 * sum_odo[head, (q_idx, 0)]
            acc_1 += p_1 * sum_odo[head, (q_idx + 1, 0)]
            acc_2 += p_2 * sum_odo[head, (q_idx + 2, 0)]
            acc_3 += p_3 * sum_odo[head, (q_idx + 3, 0)]
            q_idx += 4
        while q_idx < q_end:
            p = cute.math.exp2(sink_log2 + scaled_lse[head, (q_idx, 0)])
            if sink_log2 == Float32(float("inf")):
                p = Float32(1.0)
            elif sink_log2 == Float32(float("-inf")):
                p = Float32(0.0)
            acc_0 += p * sum_odo[head, (q_idx, 0)]
            q_idx += 1
        # Main-kernel waves have consumed these O*dO entries before this launch.
        # Their storage can now hold one partial for the terminal dSink fold.
        sum_odo[head, (q_begin, 0)] = acc_0 + acc_1 + (acc_2 + acc_3)

    @cute.kernel
    def sum_dSink_partials_deterministic(
        self, sum_odo: cute.Tensor, scaled_lse: cute.Tensor, attn_sink: cute.Tensor, problem_shape, chunk_size: cutlass.Constexpr[int]
    ):
        """Reduce one fixed query chunk per CTA into consumed sum_odo storage."""
        q_block_idx, _, _ = cute.arch.block_idx()
        head, _, _ = cute.arch.thread_idx()
        self._sum_dSink_partial_head_deterministic(sum_odo, scaled_lse, attn_sink, problem_shape, q_block_idx, head, chunk_size)

    @cute.jit
    def _finalize_dSink_head_deterministic(self, sum_odo: cute.Tensor, d_sink: cute.Tensor, problem_shape, head: Int32, chunk_size: cutlass.Constexpr[int]):
        """Fold one head's query-chunk partials in the established fixed order."""
        chunk_count = cute.ceil_div(problem_shape[0], chunk_size)
        acc_0 = Float32(0.0)
        acc_1 = Float32(0.0)
        acc_2 = Float32(0.0)
        acc_3 = Float32(0.0)
        chunk_idx = Int32(0)
        while chunk_idx + 3 < chunk_count:
            acc_0 += sum_odo[head, (chunk_idx * Int32(chunk_size), 0)]
            acc_1 += sum_odo[head, ((chunk_idx + 1) * Int32(chunk_size), 0)]
            acc_2 += sum_odo[head, ((chunk_idx + 2) * Int32(chunk_size), 0)]
            acc_3 += sum_odo[head, ((chunk_idx + 3) * Int32(chunk_size), 0)]
            chunk_idx += 4
        while chunk_idx < chunk_count:
            acc_0 += sum_odo[head, (chunk_idx * Int32(chunk_size), 0)]
            chunk_idx += 1
        d_sink[head, (0, 0)] = acc_0 + acc_1 + (acc_2 + acc_3)
