# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Checked-in baseline of FE-OSS API anchors not yet migrated to JAX.

This list is intentionally explicit. API discovery tests reject a newly added
public ``APIBase`` class or wrapper unless it is owned by a registered semantic
operation with an explicit JAX status or added here as migration debt. The
baseline should shrink and should not be treated as the normal path for new
operations.

Kernel ownership is tracked separately because one semantic operation may use
several main/helper kernels. A new physical ``@cute.kernel`` must be attached
to a registered operation; the second baseline only records migration debt
that existed when target-parity enforcement was introduced.
"""

PREEXISTING_JAX_GAP_ANCHORS = frozenset(
    {
        "cudnn.deepseek_sparse_attention.indexer_backward.api:DenseIndexerBackward",
        "cudnn.deepseek_sparse_attention.indexer_backward.api:IndexerBackward",
        "cudnn.deepseek_sparse_attention.indexer_backward.api:dense_indexer_backward_wrapper",
        "cudnn.deepseek_sparse_attention.indexer_backward.api:indexer_backward_wrapper",
        "cudnn.deepseek_sparse_attention.indexer_forward.api:IndexerForward",
        "cudnn.deepseek_sparse_attention.indexer_forward.api:indexer_forward_wrapper",
        "cudnn.deepseek_sparse_attention.indexer_top_k.api:IndexerTopK",
        "cudnn.deepseek_sparse_attention.indexer_top_k.api:compactify_wrapper",
        "cudnn.deepseek_sparse_attention.indexer_top_k.api:indexer_top_k_wrapper",
        "cudnn.deepseek_sparse_attention.indexer_top_k.api:local_to_global_wrapper",
        "cudnn.deepseek_sparse_attention.score_recompute.api:DenseAttnScoreRecompute",
        "cudnn.deepseek_sparse_attention.score_recompute.api:DenseIndexerScoreRecompute",
        "cudnn.deepseek_sparse_attention.score_recompute.api:SparseAttnScoreRecompute",
        "cudnn.deepseek_sparse_attention.score_recompute.api:SparseIndexerScoreRecompute",
        "cudnn.deepseek_sparse_attention.score_recompute.api:dense_attn_score_recompute_wrapper",
        "cudnn.deepseek_sparse_attention.score_recompute.api:dense_indexer_score_recompute_wrapper",
        "cudnn.deepseek_sparse_attention.score_recompute.api:sparse_attn_score_recompute_wrapper",
        "cudnn.deepseek_sparse_attention.score_recompute.api:sparse_indexer_score_recompute_wrapper",
        "cudnn.deepseek_sparse_attention.sparse_attention_backward.api:SparseAttentionBackward",
        "cudnn.deepseek_sparse_attention.sparse_attention_backward.api:sparse_attention_backward_wrapper",
        "cudnn.discrete_grouped_gemm.discrete_grouped_gemm_dswiglu.api:DiscreteGroupedGemmDswigluSm100",
        "cudnn.discrete_grouped_gemm.discrete_grouped_gemm_dswiglu.api:discrete_grouped_gemm_dswiglu_wrapper_sm100",
        "cudnn.discrete_grouped_gemm.discrete_grouped_gemm_swiglu.api:DiscreteGroupedGemmSwigluSm100",
        "cudnn.discrete_grouped_gemm.discrete_grouped_gemm_swiglu.api:discrete_grouped_gemm_swiglu_wrapper_sm100",
        "cudnn.gemm_amax.api:GemmAmaxSm100",
        "cudnn.gemm_amax.api:gemm_amax_wrapper_sm100",
        "cudnn.gemm_dsrelu.api:GemmDsreluSm100",
        "cudnn.gemm_dsrelu.api:gemm_dsrelu_wrapper_sm100",
        "cudnn.gemm_srelu.api:GemmSreluSm100",
        "cudnn.gemm_srelu.api:gemm_srelu_wrapper_sm100",
        "cudnn.gemm_swiglu.api:GemmSwigluSm100",
        "cudnn.gemm_swiglu.api:gemm_swiglu_wrapper_sm100",
        "cudnn.grouped_gemm.grouped_gemm_dglu.api:GroupedGemmDgluSm100",
        "cudnn.grouped_gemm.grouped_gemm_dglu.api:grouped_gemm_dglu_wrapper_sm100",
        "cudnn.grouped_gemm.grouped_gemm_dsrelu.api:GroupedGemmDsreluSm100",
        "cudnn.grouped_gemm.grouped_gemm_dsrelu.api:grouped_gemm_dsrelu_wrapper_sm100",
        "cudnn.grouped_gemm.grouped_gemm_dswiglu.api:GroupedGemmDswigluSm100",
        "cudnn.grouped_gemm.grouped_gemm_dswiglu.api:grouped_gemm_dswiglu_wrapper_sm100",
        "cudnn.grouped_gemm.grouped_gemm_glu.api:GroupedGemmGluSm100",
        "cudnn.grouped_gemm.grouped_gemm_glu.api:grouped_gemm_glu_wrapper_sm100",
        "cudnn.grouped_gemm.grouped_gemm_glu_hadamard.api:GroupedGemmGluHadamardSm100",
        "cudnn.grouped_gemm.grouped_gemm_glu_hadamard.api:grouped_gemm_glu_hadamard_wrapper_sm100",
        "cudnn.grouped_gemm.grouped_gemm_quant.api:GroupedGemmQuantSm100",
        "cudnn.grouped_gemm.grouped_gemm_quant.api:grouped_gemm_quant_wrapper_sm100",
        "cudnn.grouped_gemm.grouped_gemm_srelu.api:GroupedGemmSreluSm100",
        "cudnn.grouped_gemm.grouped_gemm_srelu.api:grouped_gemm_srelu_wrapper_sm100",
        "cudnn.grouped_gemm.grouped_gemm_swiglu.api:GroupedGemmSwigluSm100",
        "cudnn.grouped_gemm.grouped_gemm_swiglu.api:grouped_gemm_swiglu_wrapper_sm100",
        "cudnn.grouped_gemm.grouped_gemm_wgrad.api:GroupedGemmWgradSm100",
        "cudnn.grouped_gemm.grouped_gemm_wgrad.api:grouped_gemm_wgrad_wrapper_sm100",
        "cudnn.native_sparse_attention.compression.api:CompressionAttention",
        "cudnn.native_sparse_attention.compression.api:compression_attention_wrapper",
        "cudnn.native_sparse_attention.selection.api:SelectionAttention",
        "cudnn.native_sparse_attention.selection.api:selection_attention_wrapper",
        "cudnn.native_sparse_attention.sliding_window_attention.api:SlidingWindowAttention",
        "cudnn.native_sparse_attention.sliding_window_attention.api:sliding_window_attention_wrapper",
        "cudnn.native_sparse_attention.top_k.api:TopKReduction",
        "cudnn.native_sparse_attention.top_k.api:topk_reduction_wrapper",
        "cudnn.sdpa.bwd.api:SdpabwdSm100D256",
        "cudnn.sdpa.bwd.api:sdpa_bwd_wrapper_sm100_d256",
        "cudnn.sdpa.fwd.api:SdpafwdSm100D256",
        "cudnn.sdpa.fwd.api:sdpa_fwd_wrapper_sm100_d256",
    }
)


# No semantic operation rows had been migrated with a declared JAX gap when
# this catalog was introduced. CI must compare this set with the merge base so
# it can shrink but cannot be used to admit a new Torch-only operation.
PREEXISTING_REGISTERED_JAX_GAP_IDS = frozenset()


PREEXISTING_KERNEL_OWNERSHIP_GAPS = frozenset(
    {
        "cudnn.deepseek_sparse_attention.indexer_backward.dense_indexer_backward_sm100:DenseIndexerBackward2QGemmSm100.kernel_gemm_dense_2q",
        "cudnn.deepseek_sparse_attention.indexer_backward.dense_indexer_backward_sm100:ScoreGradDense.kernel_score_grad",
        "cudnn.deepseek_sparse_attention.indexer_backward.dense_indexer_backward_sm90:ScoreGradDenseSm90.kernel_score_grad",
        "cudnn.deepseek_sparse_attention.indexer_backward.indexer_backward_sm100:IndexerBackwardSm100.kernel_gemm",
        "cudnn.deepseek_sparse_attention.indexer_backward.indexer_backward_sm100:ScoreGradSm100.kernel_score_grad",
        "cudnn.deepseek_sparse_attention.indexer_backward.indexer_backward_sm90:IndexerBackwardSm90.kernel",
        "cudnn.deepseek_sparse_attention.indexer_backward.indexer_backward_sm90:ScoreGradSm90.kernel_score_grad",
        "cudnn.deepseek_sparse_attention.indexer_forward.indexer_fwd_sm100:IndexerForwardSm100.kernel",
        "cudnn.deepseek_sparse_attention.indexer_forward.indexer_fwd_sm90:IndexerForwardSm90.kernel",
        "cudnn.deepseek_sparse_attention.indexer_top_k.block_scan:block_prefix_sum",
        "cudnn.deepseek_sparse_attention.indexer_top_k.compactify:CompactifyKernel.kernel",
        "cudnn.deepseek_sparse_attention.indexer_top_k.indexer_top_k_decode_varlen:ComputeDynamicCTAOffsets.compute_offsets_kernel",
        "cudnn.deepseek_sparse_attention.indexer_top_k.indexer_top_k_decode_varlen:IndexerTopKKernelVarlenDecode.indexer_topk_kernel",
        "cudnn.deepseek_sparse_attention.indexer_top_k.local_to_global_dsl:LocalToGlobalTopK.kernel",
        "cudnn.deepseek_sparse_attention.score_recompute.dense_score_recompute_sm100:DenseScoreRecomputeSm100.kernel",
        "cudnn.deepseek_sparse_attention.score_recompute.dense_score_recompute_sm90:DenseScoreRecomputeSm90.kernel",
        "cudnn.deepseek_sparse_attention.score_recompute.sparse_score_recompute_sm100:SparseScoreRecomputeSm100.kernel",
        "cudnn.deepseek_sparse_attention.score_recompute.sparse_score_recompute_sm90:SparseScoreRecomputeSm90.kernel",
        "cudnn.deepseek_sparse_attention.sparse_attention_backward.dsa_bwd_sm100:FlashAttentionDSABackwardSm100.bwd",
        "cudnn.deepseek_sparse_attention.sparse_attention_backward.dsa_bwd_sm100:FlashAttentionDSABackwardSm100.convert",
        "cudnn.deepseek_sparse_attention.sparse_attention_backward.dsa_bwd_sm100:FlashAttentionDSABackwardSm100.sum_OdO",
        "cudnn.deepseek_sparse_attention.sparse_attention_backward.dsa_bwd_sm100:FlashAttentionDSABackwardSm100.sum_dSink",
        "cudnn.deepseek_sparse_attention.sparse_attention_backward.dsa_bwd_sm90:FlashAttentionDSABackwardSm90.kernel",
        "cudnn.deepseek_sparse_attention.sparse_attention_backward.dsa_bwd_sm90:_FlashAttentionDSABackwardPostprocessSm90.kernel",
        "cudnn.deepseek_sparse_attention.sparse_attention_backward.dsa_bwd_sm90:_FlashAttentionDSABackwardPreprocessSm90.kernel",
        "cudnn.discrete_grouped_gemm.discrete_grouped_gemm_dswiglu.discrete_B_blockscaled_grouped_gemm_dglu_dbias:BlockScaledDiscreteWeightDgluDbiasGroupedGemmKernel.desc_init_kernel_device_ptrs",
        "cudnn.discrete_grouped_gemm.discrete_grouped_gemm_dswiglu.discrete_B_blockscaled_grouped_gemm_dglu_dbias:BlockScaledDiscreteWeightDgluDbiasGroupedGemmKernel.kernel",
        "cudnn.discrete_grouped_gemm.discrete_grouped_gemm_swiglu.discrete_B_blockscaled_grouped_gemm_glu_bias:BlockScaledDiscreteWeightGroupedGemmBiasKernel.desc_init_kernel_device_ptrs",
        "cudnn.discrete_grouped_gemm.discrete_grouped_gemm_swiglu.discrete_B_blockscaled_grouped_gemm_glu_bias:BlockScaledDiscreteWeightGroupedGemmBiasKernel.kernel",
        "cudnn.gemm_amax.dense_blockscaled_gemm_persistent_amax:Sm100BlockScaledPersistentDenseGemmKernel.kernel",
        "cudnn.gemm_dsrelu.dense_blockscaled_gemm_persistent_dsrelu_quant:Sm100BlockScaledPersistentDenseGemmKernel.kernel",
        "cudnn.gemm_srelu.dense_blockscaled_gemm_persistent_srelu_quant:Sm100BlockScaledPersistentDenseGemmKernel.kernel",
        "cudnn.gemm_swiglu.dense_blockscaled_gemm_persistent_swiglu_interleaved_quant:Sm100BlockScaledPersistentDenseGemmKernel.kernel",
        "cudnn.gemm_swiglu.dense_gemm_persistent_swiglu:PersistentDenseGemmKernel.kernel",
        "cudnn.grouped_gemm.grouped_gemm_dglu.moe_blockscaled_grouped_gemm_dglu_dbias:BlockScaledMoEGroupedGemmDgluDbiasKernel.helper_kernel",
        "cudnn.grouped_gemm.grouped_gemm_dglu.moe_blockscaled_grouped_gemm_dglu_dbias:BlockScaledMoEGroupedGemmDgluDbiasKernel.kernel",
        "cudnn.grouped_gemm.grouped_gemm_dsrelu.moe_blockscaled_grouped_gemm_dsrelu_quant:BlockScaledMoEGroupedGemmQuantBwdKernel.helper_kernel",
        "cudnn.grouped_gemm.grouped_gemm_dsrelu.moe_blockscaled_grouped_gemm_dsrelu_quant:BlockScaledMoEGroupedGemmQuantBwdKernel.kernel",
        "cudnn.grouped_gemm.grouped_gemm_dswiglu.grouped_gemm_dswiglu_quant:BlockScaledContiguousGroupedGemmKernel.kernel",
        "cudnn.grouped_gemm.grouped_gemm_glu.moe_blockscaled_grouped_gemm_glu_bias:BlockScaledMoEGroupedGemmGluBiasKernel.helper_kernel",
        "cudnn.grouped_gemm.grouped_gemm_glu.moe_blockscaled_grouped_gemm_glu_bias:BlockScaledMoEGroupedGemmGluBiasKernel.kernel",
        "cudnn.grouped_gemm.grouped_gemm_glu_hadamard.moe_blockscaled_grouped_gemm_glu_hadamard:BlockScaledMoEGroupedGemmGluHadamardKernel.helper_kernel",
        "cudnn.grouped_gemm.grouped_gemm_glu_hadamard.moe_blockscaled_grouped_gemm_glu_hadamard:BlockScaledMoEGroupedGemmGluHadamardKernel.kernel",
        "cudnn.grouped_gemm.grouped_gemm_quant.grouped_gemm_quant:BlockScaledMoEGroupedGemmQuantKernel.helper_kernel",
        "cudnn.grouped_gemm.grouped_gemm_quant.grouped_gemm_quant:BlockScaledMoEGroupedGemmQuantKernel.kernel",
        "cudnn.grouped_gemm.grouped_gemm_srelu.moe_blockscaled_grouped_gemm_srelu_quant:BlockScaledMoEGroupedGemmQuantKernel.helper_kernel",
        "cudnn.grouped_gemm.grouped_gemm_srelu.moe_blockscaled_grouped_gemm_srelu_quant:BlockScaledMoEGroupedGemmQuantKernel.kernel",
        "cudnn.grouped_gemm.grouped_gemm_swiglu.grouped_gemm_swiglu_quant:BlockScaledContiguousGroupedGemmKernel.kernel",
        "cudnn.grouped_gemm.grouped_gemm_wgrad.moe_blockscaled_grouped_gemm_wgrad:BlockScaledMoEGroupedGemmWgradKernel.helper_kernel",
        "cudnn.grouped_gemm.grouped_gemm_wgrad.moe_blockscaled_grouped_gemm_wgrad:BlockScaledMoEGroupedGemmWgradKernel.kernel",
        "cudnn.native_sparse_attention.compression.fmha:BlackwellFusedMultiHeadAttentionForward.kernel",
        "cudnn.native_sparse_attention.selection.NSA_select_attn_fwd_hmma:HopperSelectAttentionFwd.kernel",
        "cudnn.native_sparse_attention.top_k.nsa_top_k_reduction_fwd:FineGrainedReductionQK.kernel",
        "cudnn.sdpa.bwd.fmha_backward_sm100_2kernel:BlackwellFusedMultiHeadAttentionBackward.sum_OdO",
        "cudnn.sdpa.bwd.fmha_dkdv_d256_sm100:BlackwellFusedAttentionDKDVKernel.dkdv_bwd",
        "cudnn.sdpa.bwd.fmha_dq_d256_sm100:BlackwellFusedAttentionDQKernel.kernel",
        "cudnn.sdpa.fwd.fmha_forward_sm100_d256:BlackwellFusedMultiHeadAttentionForward.kernel",
    }
)


__all__ = [
    "PREEXISTING_JAX_GAP_ANCHORS",
    "PREEXISTING_KERNEL_OWNERSHIP_GAPS",
    "PREEXISTING_REGISTERED_JAX_GAP_IDS",
]
