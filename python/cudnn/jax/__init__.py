# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""JAX facade for co-located frontend-only operation APIs.

Importing :mod:`cudnn.jax` is the explicit dependency boundary for JAX and
CuTe DSL. Dependencies are validated once here, public wrapper modules are then
imported, and architecture-specific kernel modules remain deferred until an
operation is traced.
"""

from importlib import import_module
from types import SimpleNamespace

_INSTALL_HINT = "pip install 'nvidia-cudnn-frontend[jax]'"


def _require_dependencies(module_names: list[str]) -> None:
    for module_name in module_names:
        try:
            import_module(module_name)
        except ModuleNotFoundError as error:
            missing_name = error.name
            if missing_name is None or not (module_name == missing_name or module_name.startswith(f"{missing_name}.")):
                raise
            raise ImportError(f"cudnn.jax requires the {module_name!r} module. Install the " f"JAX integration with `{_INSTALL_HINT}`.") from error


_require_dependencies(["jax", "cutlass"])

import cutlass.jax
import jax

if not cutlass.jax.is_available():
    minimum_version_info = tuple(cutlass.jax.CUTE_DSL_MIN_SUPPORTED_JAX_VERSION)
    minimum_version = ".".join(str(part) for part in minimum_version_info)
    installed_version = getattr(jax, "__version__", "unknown")
    reason = f"CUTLASS JAX support is unavailable with JAX {installed_version}; " f"the minimum supported JAX version is {minimum_version}."
    raise ImportError(f"cudnn.jax cannot be imported because {reason} Install the JAX " f"integration with `{_INSTALL_HINT}`.")

from .._jax.api_base import ApiBaseJax, JaxTensorDesc, TupleDict

from ..deepseek_sparse_attention.indexer_forward.jax import (
    IndexerForward,
    indexer_forward_wrapper,
)
from ..deepseek_sparse_attention.indexer_backward.jax import (
    IndexerBackward,
    indexer_backward_wrapper,
)
from ..deepseek_sparse_attention.indexer_top_k.jax import (
    IndexerTopK,
    compactify_wrapper,
    indexer_top_k_wrapper,
    local_to_global_wrapper,
)
from ..deepseek_sparse_attention.score_recompute.jax import (
    DenseAttnScoreRecompute,
    DenseIndexerScoreRecompute,
    SparseAttnScoreRecompute,
    SparseIndexerScoreRecompute,
    dense_attn_score_recompute_wrapper,
    dense_indexer_score_recompute_wrapper,
    sparse_attn_score_recompute_wrapper,
    sparse_indexer_score_recompute_wrapper,
)
from ..deepseek_sparse_attention.sparse_attention_backward.jax import (
    SparseAttentionBackward,
    sparse_attention_backward_wrapper,
)
from ..gemm_amax.jax import GemmAmaxSm100, gemm_amax_wrapper_sm100
from ..gemm_dsrelu.jax import (
    GemmDsreluSm100,
    gemm_dsrelu_wrapper_sm100,
)
from ..gemm_srelu.jax import GemmSreluSm100, gemm_srelu_wrapper_sm100
from ..gemm_swiglu.jax import (
    GemmSwigluSm100,
    gemm_swiglu_wrapper_sm100,
)
from ..grouped_gemm.grouped_gemm_dswiglu.jax import (
    GroupedGemmDswigluSm100,
    grouped_gemm_dswiglu_wrapper_sm100,
)
from ..grouped_gemm.grouped_gemm_dglu.jax import (
    GroupedGemmDgluSm100,
    grouped_gemm_dglu_wrapper_sm100,
)
from ..grouped_gemm.grouped_gemm_dsrelu.jax import (
    GroupedGemmDsreluSm100,
    grouped_gemm_dsrelu_wrapper_sm100,
)
from ..grouped_gemm.grouped_gemm_glu.jax import (
    GroupedGemmGluSm100,
    grouped_gemm_glu_wrapper_sm100,
)
from ..grouped_gemm.grouped_gemm_glu_hadamard.jax import (
    GroupedGemmGluHadamardSm100,
    grouped_gemm_glu_hadamard_wrapper_sm100,
)
from ..grouped_gemm.grouped_gemm_quant.jax import (
    GroupedGemmQuantSm100,
    grouped_gemm_quant_wrapper_sm100,
)
from ..grouped_gemm.grouped_gemm_srelu.jax import (
    GroupedGemmSreluSm100,
    grouped_gemm_srelu_wrapper_sm100,
)
from ..grouped_gemm.grouped_gemm_swiglu.jax import (
    GroupedGemmSwigluSm100,
    grouped_gemm_swiglu_wrapper_sm100,
)
from ..grouped_gemm.grouped_gemm_wgrad.jax import (
    GroupedGemmWgradSm100,
    grouped_gemm_wgrad_wrapper_sm100,
)
from ..native_sparse_attention.compression.jax import (
    CompressionAttention,
    compression_attention_wrapper,
)
from ..native_sparse_attention.selection.jax import (
    SelectionAttention,
    selection_attention_wrapper,
)
from ..native_sparse_attention.top_k.jax import (
    TopKReduction,
    topk_reduction_wrapper,
)
from ..rmsnorm_rht_amax.jax import (
    RmsNormRhtAmaxSm100,
    rmsnorm_rht_amax_sm100,
)
from ..sdpa.bwd.jax import SdpabwdSm100D256, sdpa_bwd_wrapper_sm100_d256
from ..sdpa.fwd.jax import SdpafwdSm100D256, sdpa_fwd_wrapper_sm100_d256

DSA = SimpleNamespace(
    IndexerBackward=IndexerBackward,
    IndexerForward=IndexerForward,
    IndexerTopK=IndexerTopK,
    SparseIndexerScoreRecompute=SparseIndexerScoreRecompute,
    SparseAttnScoreRecompute=SparseAttnScoreRecompute,
    DenseIndexerScoreRecompute=DenseIndexerScoreRecompute,
    DenseAttnScoreRecompute=DenseAttnScoreRecompute,
    SparseAttentionBackward=SparseAttentionBackward,
    indexer_backward_wrapper=indexer_backward_wrapper,
    indexer_forward_wrapper=indexer_forward_wrapper,
    indexer_top_k_wrapper=indexer_top_k_wrapper,
    local_to_global_wrapper=local_to_global_wrapper,
    compactify_wrapper=compactify_wrapper,
    sparse_indexer_score_recompute_wrapper=sparse_indexer_score_recompute_wrapper,
    sparse_attn_score_recompute_wrapper=sparse_attn_score_recompute_wrapper,
    dense_indexer_score_recompute_wrapper=dense_indexer_score_recompute_wrapper,
    dense_attn_score_recompute_wrapper=dense_attn_score_recompute_wrapper,
    sparse_attention_backward_wrapper=sparse_attention_backward_wrapper,
)

NSA = SimpleNamespace(
    CompressionAttention=CompressionAttention,
    SelectionAttention=SelectionAttention,
    TopKReduction=TopKReduction,
    compression_attention_wrapper=compression_attention_wrapper,
    selection_attention_wrapper=selection_attention_wrapper,
    topk_reduction_wrapper=topk_reduction_wrapper,
)

__all__ = [
    "ApiBaseJax",
    "DSA",
    "NSA",
    "CompressionAttention",
    "DenseAttnScoreRecompute",
    "DenseIndexerScoreRecompute",
    "GemmAmaxSm100",
    "GemmDsreluSm100",
    "GemmSreluSm100",
    "GemmSwigluSm100",
    "GroupedGemmDswigluSm100",
    "GroupedGemmDgluSm100",
    "GroupedGemmDsreluSm100",
    "GroupedGemmGluHadamardSm100",
    "GroupedGemmGluSm100",
    "GroupedGemmQuantSm100",
    "GroupedGemmSreluSm100",
    "GroupedGemmSwigluSm100",
    "GroupedGemmWgradSm100",
    "IndexerBackward",
    "IndexerForward",
    "IndexerTopK",
    "JaxTensorDesc",
    "RmsNormRhtAmaxSm100",
    "SdpabwdSm100D256",
    "SdpafwdSm100D256",
    "SelectionAttention",
    "SparseAttentionBackward",
    "SparseAttnScoreRecompute",
    "SparseIndexerScoreRecompute",
    "TopKReduction",
    "TupleDict",
    "compactify_wrapper",
    "compression_attention_wrapper",
    "dense_attn_score_recompute_wrapper",
    "dense_indexer_score_recompute_wrapper",
    "indexer_forward_wrapper",
    "indexer_backward_wrapper",
    "indexer_top_k_wrapper",
    "gemm_amax_wrapper_sm100",
    "gemm_dsrelu_wrapper_sm100",
    "gemm_srelu_wrapper_sm100",
    "gemm_swiglu_wrapper_sm100",
    "grouped_gemm_dswiglu_wrapper_sm100",
    "grouped_gemm_dglu_wrapper_sm100",
    "grouped_gemm_dsrelu_wrapper_sm100",
    "grouped_gemm_glu_hadamard_wrapper_sm100",
    "grouped_gemm_glu_wrapper_sm100",
    "grouped_gemm_quant_wrapper_sm100",
    "grouped_gemm_srelu_wrapper_sm100",
    "grouped_gemm_swiglu_wrapper_sm100",
    "grouped_gemm_wgrad_wrapper_sm100",
    "local_to_global_wrapper",
    "rmsnorm_rht_amax_sm100",
    "sdpa_bwd_wrapper_sm100_d256",
    "sdpa_fwd_wrapper_sm100_d256",
    "selection_attention_wrapper",
    "sparse_attention_backward_wrapper",
    "sparse_attn_score_recompute_wrapper",
    "sparse_indexer_score_recompute_wrapper",
    "topk_reduction_wrapper",
]
