# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.cute.typing import Float32, Int32

from .dsa_bwd_sm100 import FlashAttentionDSABackwardSm100
from .dsa_bwd_sm100_deterministic import FlashAttentionDSABackwardSm100Deterministic, _DeterministicReductionPolicy
from .dsa_bwd_sm100_h32 import FlashAttentionDSABackwardSm100H32


@dsl_user_op
def _h96_load_f32x4(source: cute.Pointer, *, loc=None, ip=None) -> Tuple[Float32, Float32, Float32, Float32]:
    """Load four aligned FP32 shard values with one instruction."""
    source_i64 = source.toint(loc=loc, ip=ip).ir_value()
    result_type = llvm.StructType.get_literal([T.f32()] * 4)
    values = llvm.inline_asm(
        result_type,
        [source_i64],
        "ld.global.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return tuple(Float32(llvm.extractvalue(T.f32(), values, [index], loc=loc, ip=ip)) for index in range(4))


@dsl_user_op
def _h96_store_f32x4(
    destination: cute.Pointer,
    value_0: Float32,
    value_1: Float32,
    value_2: Float32,
    value_3: Float32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Store four aligned FP32 shard values with one instruction."""
    destination_i64 = destination.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [
            destination_i64,
            Float32(value_0).ir_value(loc=loc, ip=ip),
            Float32(value_1).ir_value(loc=loc, ip=ip),
            Float32(value_2).ir_value(loc=loc, ip=ip),
            Float32(value_3).ir_value(loc=loc, ip=ip),
        ],
        "st.global.v4.f32 [$0], {$1, $2, $3, $4};",
        "l,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


class FlashAttentionDSABackwardSm100H96H64(FlashAttentionDSABackwardSm100):
    """H64 body of H96, including the shared auxiliary kernels."""

    finalize_dkv = False
    bwd_num_heads = 64


class FlashAttentionDSABackwardSm100H96H32(FlashAttentionDSABackwardSm100H32):
    """H32 tail of H96, consuming shared auxiliaries and finalizing dKV."""

    initialize_dkv = False
    run_sum_odo = False
    run_dsink = False
    head_offset = 64
    workspace_num_heads = 96
    workspace_head_offset = 64


class FlashAttentionDSABackwardSm100H96H64Deterministic(FlashAttentionDSABackwardSm100Deterministic):
    """Deterministic H64 body of H96, leaving dKV shards for the tail."""

    finalize_dkv = False
    bwd_num_heads = 64


class FlashAttentionDSABackwardSm100H96H32Deterministic(_DeterministicReductionPolicy, FlashAttentionDSABackwardSm100H32):
    """Deterministic H32 tail sharing the H64 body's fixed-owner shards."""

    # The reduction policy deliberately owns these attributes in the composite
    # hierarchy; keep the overrides explicit instead of relying on base order.
    num_dkv_shards = _DeterministicReductionPolicy.num_dkv_shards
    q_wave_ctas = _DeterministicReductionPolicy.q_wave_ctas
    serialize_head_blocks = _DeterministicReductionPolicy.serialize_head_blocks
    make_dKV_accumulator = _DeterministicReductionPolicy.make_dKV_accumulator
    select_dKV_accumulator = _DeterministicReductionPolicy.select_dKV_accumulator
    convert = _DeterministicReductionPolicy.convert
    dSink_grid_q = _DeterministicReductionPolicy.dSink_grid_q
    sum_dSink = _DeterministicReductionPolicy.sum_dSink

    initialize_dkv = False
    run_sum_odo = False
    run_dsink = False
    head_offset = 64
    workspace_num_heads = 96
    workspace_head_offset = 64

    @classmethod
    def _get_workspace_size_dKV(cls, k: int, d: int, b: int, acc_dtype: Type[cutlass.Numeric]):
        return _DeterministicReductionPolicy._get_workspace_size_dKV.__func__(cls, k, d, b, acc_dtype)

    @cute.kernel
    def fold_dKV_shards(self, mdKV_acc: cute.Tensor, seqlen: Int32):
        """Fold ordered shard groups four adjacent FP32 dimensions at a time."""
        seq_id, group_idx, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        group_begin = group_idx * self.dkv_fold_group_size
        dim_base = tidx * 4
        if seq_id < seqlen:
            acc_0 = Float32(0.0)
            acc_1 = Float32(0.0)
            acc_2 = Float32(0.0)
            acc_3 = Float32(0.0)
            for local_shard in cutlass.range_constexpr(self.dkv_fold_group_size):
                shard_idx = group_begin + local_shard
                row = mdKV_acc[None, seq_id + shard_idx * seqlen, (0, 0)]
                values = _h96_load_f32x4(row.iterator + dim_base)
                acc_0 += values[0]
                acc_1 += values[1]
                acc_2 += values[2]
                acc_3 += values[3]
            out_row = mdKV_acc[None, seq_id + group_begin * seqlen, (0, 0)]
            _h96_store_f32x4(out_row.iterator + dim_base, acc_0, acc_1, acc_2, acc_3)

    @cute.jit
    def finalize_dKV(
        self,
        mdKV_acc: cute.Tensor,
        mdKV: cute.Tensor,
        seqlen: Int32,
        stream: cuda.CUstream,
    ):
        """Run the vector fold before the inherited ordered conversion."""
        self.fold_dKV_shards(mdKV_acc, seqlen).launch(
            grid=[seqlen, self.num_dkv_fold_groups, 1],
            block=[self.head_dim // 4, 1, 1],
            stream=stream,
        )
        convert_grid_x = (seqlen + self.block_seq - 1) // self.block_seq
        self.convert(mdKV_acc, mdKV, seqlen).launch(
            grid=[convert_grid_x, 1, 1],
            block=[self.num_threads_D_convert, self.num_threads_seq, 1],
            stream=stream,
        )


class FlashAttentionDSABackwardSm100H96:
    """Compose the tuned H64 body and H32 tail into one launch sequence."""

    def __init__(
        self,
        element_dtype: Type[cutlass.Numeric],
        head_dim: int,
        head_dim_v: int,
        block_tile: int,
        max_topk: int = 0,
    ):
        if head_dim != 576 or head_dim_v != 512 or block_tile != 64:
            raise ValueError("H96 composition requires head_dim=576, head_dim_v=512, and block_tile=64")
        common = (element_dtype, head_dim, head_dim_v, block_tile, max_topk)
        self.h64 = FlashAttentionDSABackwardSm100H96H64(*common)
        self.h32 = FlashAttentionDSABackwardSm100H96H32(*common)

    @cute.jit
    def __call__(
        self,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Int32, Int32]],
        mQ: cute.Tensor,
        mKV: cute.Tensor,
        mOut: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mAttnSink: cute.Tensor,
        mTopkIdxs: cute.Tensor,
        mTopkLength: Optional[cute.Tensor],
        mdQ: cute.Tensor,
        mdKV: cute.Tensor,
        mdSink: cute.Tensor,
        workspace_LSE_OdO: cute.Tensor,
        workspace_dKV: cute.Tensor,
        softmax_scale: Float32 | float,
        stream: cuda.CUstream,
    ):
        self.h64(
            problem_shape,
            mQ,
            mKV,
            mOut,
            mdO,
            mLSE,
            mAttnSink,
            mTopkIdxs,
            mTopkLength,
            mdQ,
            mdKV,
            mdSink,
            workspace_LSE_OdO,
            workspace_dKV,
            softmax_scale,
            stream,
        )
        h32_problem_shape = (problem_shape[0], problem_shape[1], problem_shape[2], (32, problem_shape[3][1]))
        self.h32(
            h32_problem_shape,
            mQ,
            mKV,
            mOut,
            mdO,
            mLSE,
            mAttnSink,
            mTopkIdxs,
            mTopkLength,
            mdQ,
            mdKV,
            mdSink,
            workspace_LSE_OdO,
            workspace_dKV,
            softmax_scale,
            stream,
        )


class FlashAttentionDSABackwardSm100H96Deterministic(_DeterministicReductionPolicy):
    """Compose deterministic H64 and tuned H32 stages over shared shards."""

    def __init__(
        self,
        element_dtype: Type[cutlass.Numeric],
        head_dim: int,
        head_dim_v: int,
        block_tile: int,
        max_topk: int = 0,
    ):
        if head_dim != 576 or head_dim_v != 512 or block_tile != 64:
            raise ValueError("deterministic H96 composition requires head_dim=576, head_dim_v=512, and block_tile=64")
        common = (element_dtype, head_dim, head_dim_v, block_tile, max_topk)
        self.h64 = FlashAttentionDSABackwardSm100H96H64Deterministic(*common)
        self.h32 = FlashAttentionDSABackwardSm100H96H32Deterministic(*common)

    @cute.jit
    def __call__(
        self,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Int32, Int32]],
        mQ: cute.Tensor,
        mKV: cute.Tensor,
        mOut: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mAttnSink: cute.Tensor,
        mTopkIdxs: cute.Tensor,
        mTopkLength: Optional[cute.Tensor],
        mdQ: cute.Tensor,
        mdKV: cute.Tensor,
        mdSink: cute.Tensor,
        workspace_LSE_OdO: cute.Tensor,
        workspace_dKV: cute.Tensor,
        softmax_scale: Float32 | float,
        stream: cuda.CUstream,
    ):
        self.h64(
            problem_shape,
            mQ,
            mKV,
            mOut,
            mdO,
            mLSE,
            mAttnSink,
            mTopkIdxs,
            mTopkLength,
            mdQ,
            mdKV,
            mdSink,
            workspace_LSE_OdO,
            workspace_dKV,
            softmax_scale,
            stream,
        )
        h32_problem_shape = (problem_shape[0], problem_shape[1], problem_shape[2], (32, problem_shape[3][1]))
        self.h32(
            h32_problem_shape,
            mQ,
            mKV,
            mOut,
            mdO,
            mLSE,
            mAttnSink,
            mTopkIdxs,
            mTopkLength,
            mdQ,
            mdKV,
            mdSink,
            workspace_LSE_OdO,
            workspace_dKV,
            softmax_scale,
            stream,
        )
