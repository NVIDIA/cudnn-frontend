# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Float32, Int32

from .dsa_bwd_sm100 import FlashAttentionDSABackwardSm100
from .dsa_bwd_sm100_h32 import FlashAttentionDSABackwardSm100H32


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
