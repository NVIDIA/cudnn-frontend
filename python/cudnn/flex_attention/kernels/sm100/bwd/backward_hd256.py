# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025, Siyu Wang, Shengbin Di, Yuxi Chi, Johnsonms, Linfeng Zheng, Haoyan Huang, Lanbo Li, Yun Zhong, Man Yuan, Minmin Sun, Yong Li, Wei Lin.


"""Fused multi-head attention (FMHA) backward for the SM100 architecture using CUTE DSL.

Constraints:
* Supported head dimensions: 256 only
* mma_tiler_mn must be 64,64
* Batch size must be the same for Q, K, and V tensors
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Int32

import cuda.bindings.driver as cuda

from cudnn.flex_attention.kernels.sm100.bwd.backward_dq_hd256 import (
    BlackwellFusedMultiHeadAttentionBackwardDQKernel,
)
from cudnn.flex_attention.kernels.sm100.bwd.backward_dkdv_hd256 import (
    BlackwellFusedMultiHeadAttentionBackwardDKDVKernel,
)
from cudnn.flex_attention.plan.kernels import BlockSparseTensors
from cudnn.flex_attention.runtime.dsl_utils import as_bshkrd_tensor, assume_tensor_aligned


def _as_shhb_tensor(
    tensor: cute.Tensor,
    h_k: Int32,
    h_r: Int32,
    b: Int32,
    varlen: bool,
) -> cute.Tensor:
    """Normalize (B,H,S)/(H,S) tensors to (S, ((H_r, H_k), B)) view."""
    if cutlass.const_expr(cute.rank(tensor.layout) == 3):
        return cute.make_tensor(
            tensor.iterator,
            cute.make_layout(
                (tensor.shape[2], ((h_r, h_k), tensor.shape[0])),
                stride=(
                    tensor.stride[2],
                    ((tensor.stride[1], tensor.stride[1] * h_r), tensor.stride[0]),
                ),
            ),
        )
    assert cutlass.const_expr(cute.rank(tensor.layout) == 2), "Expected rank-2 varlen tensor"
    assert cutlass.const_expr(varlen), "Rank-2 input is only valid for varlen backward"
    return cute.make_tensor(
        tensor.iterator,
        cute.make_layout(
            (tensor.shape[1], ((h_r, h_k), b)),
            stride=(
                tensor.stride[1],
                ((tensor.stride[0], tensor.stride[0] * h_r), 0),
            ),
        ),
    )


class BlackwellFusedMultiHeadAttentionBackward:
    """FMHA backward class for executing CuTeDSL kernel."""

    def __init__(
        self,
        head_dim: int,
        head_dim_v: int | None = None,
        qhead_per_kvhead: cutlass.Constexpr[int] = 1,
        mask_payload_valid_words_dq: int = 4,
        mask_payload_padded_words_dq: int = 4,
        mask_payload_valid_words_dkdv: int = 1,
        mask_payload_padded_words_dkdv: int = 1,
        subtile_factor: cutlass.Constexpr[int] = 1,
        tile_m_dq: int = 128,
        tile_n_dq: int = 128,
        tile_m_dkdv: int = 128,
        tile_n_dkdv: int = 64,
    ):
        """Initialization."""
        head_dim_v = head_dim if head_dim_v is None else head_dim_v
        assert head_dim == 256 and head_dim_v == 256, "SM100 dedicated backward kernel only supports (head_dim, head_dim_v) = (256, 256)"
        assert tile_m_dq == 128 and tile_n_dq == 128, "SM100 dedicated backward kernel only supports tile_m_dq=128 and tile_n_dq=128"
        assert tile_m_dkdv == 128 and tile_n_dkdv == 64, "SM100 dedicated backward kernel only supports tile_m_dkdv=128 and tile_n_dkdv=64"
        assert qhead_per_kvhead >= 1
        assert subtile_factor == 2, "SM100 backward with head_dim=256 arbitrary requires Q256 K2Q entries"
        assert mask_payload_valid_words_dq == 4 and mask_payload_padded_words_dq == 4, "SM100 hd256 arbitrary dQ payload requires four valid/padded words"
        assert mask_payload_valid_words_dkdv == 1 and mask_payload_padded_words_dkdv == 1, "SM100 hd256 arbitrary dKdV payload requires one valid/padded word"
        self.subtile_factor = subtile_factor

        self.acc_dtype = cutlass.Float32
        self.qhead_per_kvhead = qhead_per_kvhead
        self.tile_m_dq = tile_m_dq
        self.tile_n_dq = tile_n_dq
        self.tile_m_dkdv = tile_m_dkdv
        self.tile_n_dkdv = tile_n_dkdv

        self.dq_kernel = BlackwellFusedMultiHeadAttentionBackwardDQKernel(
            self.acc_dtype,
            (self.tile_m_dq, self.tile_n_dq, 256),
            qhead_per_kvhead=self.qhead_per_kvhead,
            mask_payload_valid_words=mask_payload_valid_words_dq,
            mask_payload_padded_words=mask_payload_padded_words_dq,
        )
        self.dkdv_kernel = BlackwellFusedMultiHeadAttentionBackwardDKDVKernel(
            self.acc_dtype,
            (self.tile_m_dkdv, self.tile_n_dkdv, 256),
            qhead_per_kvhead=self.qhead_per_kvhead,
            mask_payload_valid_words=mask_payload_valid_words_dkdv,
            mask_payload_padded_words=mask_payload_padded_words_dkdv,
            subtile_factor=self.subtile_factor,
        )

    @cute.jit
    def __call__(
        self,
        Q: cute.Tensor,
        K: cute.Tensor,
        V: cute.Tensor,
        dO: cute.Tensor,
        lse_log2: cute.Tensor,
        dpsum: cute.Tensor,
        dQ_accum: cute.Tensor | None,
        dK: cute.Tensor,
        dV: cute.Tensor,
        scale_softmax: cutlass.Float32,
        cumulative_s_q: cute.Tensor | None,
        cumulative_s_k: cute.Tensor | None,
        block_sparse_tensors_dq: BlockSparseTensors = None,
        block_sparse_tensors: BlockSparseTensors = None,
        max_seqlen_q_runtime: Int32 = Int32(0),
        max_seqlen_k_runtime: Int32 = Int32(0),
        stream: cuda.CUstream = None,
    ):
        """Host function to launch CuTeDSL kernel."""
        varlen = cumulative_s_q is not None or cumulative_s_k is not None
        if cutlass.const_expr(block_sparse_tensors_dq is not None):
            assert block_sparse_tensors_dq.mask_block_offset is not None, "SM100 backward with head_dim=256 dQ only supports linear CSR block sparse tensors"
        if cutlass.const_expr(block_sparse_tensors is not None):
            assert block_sparse_tensors.mask_block_offset is not None, "SM100 backward with head_dim=256 only supports linear CSR block sparse tensors"
        assert (cumulative_s_q is None) == (cumulative_s_k is None), "SM100 hd256 arbitrary varlen backward requires both cu_seqlens tensors"
        assert block_sparse_tensors_dq is not None and block_sparse_tensors is not None, (
            "SM100 backward with head_dim=256 arbitrary requires independent " "dQ Q2K and dKdV K2Q plans"
        )
        assert block_sparse_tensors_dq.mask_block_masks is not None, "SM100 backward with head_dim=256 arbitrary dQ requires mask_block_masks"
        assert block_sparse_tensors.mask_block_masks is not None, "SM100 backward with head_dim=256 arbitrary dKdV requires mask_block_masks"
        assert dQ_accum is not None, "SM100 backward with head_dim=256 expects dQ tensor at dQ_accum slot"
        dQ = dQ_accum
        q_rank = cute.rank(Q.layout)
        k_rank = cute.rank(K.layout)
        if cutlass.const_expr(q_rank == 5):
            h_q = Q.shape[2] * Q.shape[3]
        elif cutlass.const_expr(q_rank == 4):
            h_q = Q.shape[2]
        else:
            h_q = Q.shape[1]
        if cutlass.const_expr(k_rank == 5):
            h_k = K.shape[2]
        elif cutlass.const_expr(k_rank == 4):
            h_k = K.shape[2]
        else:
            h_k = K.shape[1]
        h_r = h_q // h_k
        if cutlass.const_expr(cumulative_s_q is not None):
            b = cumulative_s_q.shape[0] - 1
        elif cutlass.const_expr(cumulative_s_k is not None):
            b = cumulative_s_k.shape[0] - 1
        else:
            b = Q.shape[0]

        Q, K, V, dQ, dK, dV, dO = [assume_tensor_aligned(t) for t in (Q, K, V, dQ, dK, dV, dO)]

        Q = as_bshkrd_tensor(Q, h_k, h_r, varlen)
        K = as_bshkrd_tensor(K, h_k, 1, varlen)
        V = as_bshkrd_tensor(V, h_k, 1, varlen)
        dQ = as_bshkrd_tensor(dQ, h_k, h_r, varlen)
        dK = as_bshkrd_tensor(dK, h_k, 1, varlen)
        dV = as_bshkrd_tensor(dV, h_k, 1, varlen)
        dO = as_bshkrd_tensor(dO, h_k, h_r, varlen)
        scaled_LSE = _as_shhb_tensor(lse_log2, h_k, h_r, b, varlen)
        sum_OdO = _as_shhb_tensor(dpsum, h_k, h_r, b, varlen)

        # Keep original order: dQ first, then dKdV.
        self.dq_kernel(
            Q,
            K,
            V,
            dQ,
            dO,
            scaled_LSE,
            sum_OdO,
            cumulative_s_q,
            cumulative_s_k,
            scale_softmax,
            block_sparse_tensors_dq,
            max_seqlen_q_runtime,
            stream,
        )
        self.dkdv_kernel(
            Q,
            K,
            V,
            dK,
            dV,
            dO,
            scaled_LSE,
            sum_OdO,
            cumulative_s_q,
            cumulative_s_k,
            scale_softmax,
            block_sparse_tensors,
            max_seqlen_k_runtime,
            stream,
        )
