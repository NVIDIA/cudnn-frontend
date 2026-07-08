# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Public optional-dependency boundary for JAX operation APIs.

JAX and CUTLASS are validated eagerly at this boundary. Individual operation
adapters remain lazy so importing :mod:`cudnn.jax` does not import every CuTe
kernel or any Torch API module.
"""

from importlib import import_module
from typing import Any

_INSTALL_HINT = "pip install 'nvidia-cudnn-frontend[jax]'"
_OPTIONAL_DEPENDENCY_PREFIXES = ("cuda", "cutlass", "jax")

try:
    import cutlass.jax as _cutlass_jax
    import jax as _jax
except ModuleNotFoundError as error:
    missing_name = error.name
    if missing_name is None or not any(
        missing_name == prefix or missing_name.startswith(f"{prefix}.")
        for prefix in _OPTIONAL_DEPENDENCY_PREFIXES
    ):
        raise
    raise ImportError(
        f"cuDNN JAX APIs require the optional JAX dependencies. Install them with `{_INSTALL_HINT}`."
    ) from error

if not _cutlass_jax.is_available():
    minimum_version = ".".join(
        str(part) for part in _cutlass_jax.CUTE_DSL_MIN_SUPPORTED_JAX_VERSION
    )
    installed_version = getattr(_jax, "__version__", "unknown")
    raise ImportError(
        f"CUTLASS JAX support is unavailable with JAX {installed_version}; "
        f"the minimum supported JAX version is {minimum_version}. "
        f"Install the optional JAX dependencies with `{_INSTALL_HINT}`."
    )

from .._jax import JaxApiBase, JaxTensorDesc, TupleDict  # noqa: E402

_OPERATION_EXPORTS = {
    "RmsNormRhtAmaxSm100": ("..rmsnorm_rht_amax.jax", "RmsNormRhtAmaxSm100"),
    "rmsnorm_rht_amax_sm100": ("..rmsnorm_rht_amax.jax", "rmsnorm_rht_amax_sm100"),
    "GemmSwigluSm100": ("..gemm_swiglu.jax", "GemmSwigluSm100"),
    "gemm_swiglu_wrapper_sm100": ("..gemm_swiglu.jax", "gemm_swiglu_wrapper_sm100"),
    "GemmSreluSm100": ("..gemm_srelu.jax", "GemmSreluSm100"),
    "gemm_srelu_wrapper_sm100": ("..gemm_srelu.jax", "gemm_srelu_wrapper_sm100"),
    "GemmDsreluSm100": ("..gemm_dsrelu.jax", "GemmDsreluSm100"),
    "gemm_dsrelu_wrapper_sm100": ("..gemm_dsrelu.jax", "gemm_dsrelu_wrapper_sm100"),
    "GemmAmaxSm100": ("..gemm_amax.jax", "GemmAmaxSm100"),
    "gemm_amax_wrapper_sm100": ("..gemm_amax.jax", "gemm_amax_wrapper_sm100"),
    "SparseAttentionBackward": (
        "..deepseek_sparse_attention.sparse_attention_backward.jax",
        "SparseAttentionBackward",
    ),
    "sparse_attention_backward_wrapper": (
        "..deepseek_sparse_attention.sparse_attention_backward.jax",
        "sparse_attention_backward_wrapper",
    ),
    "IndexerForward": (
        "..deepseek_sparse_attention.indexer_forward.jax",
        "IndexerForward",
    ),
    "indexer_forward_wrapper": (
        "..deepseek_sparse_attention.indexer_forward.jax",
        "indexer_forward_wrapper",
    ),
    "IndexerTopK": ("..deepseek_sparse_attention.indexer_top_k.jax", "IndexerTopK"),
    "indexer_top_k_wrapper": (
        "..deepseek_sparse_attention.indexer_top_k.jax",
        "indexer_top_k_wrapper",
    ),
    "local_to_global_wrapper": (
        "..deepseek_sparse_attention.indexer_top_k.jax",
        "local_to_global_wrapper",
    ),
    "compactify_wrapper": (
        "..deepseek_sparse_attention.indexer_top_k.jax",
        "compactify_wrapper",
    ),
    "SparseIndexerScoreRecompute": (
        "..deepseek_sparse_attention.score_recompute.jax",
        "SparseIndexerScoreRecompute",
    ),
    "sparse_indexer_score_recompute_wrapper": (
        "..deepseek_sparse_attention.score_recompute.jax",
        "sparse_indexer_score_recompute_wrapper",
    ),
    "SparseAttnScoreRecompute": (
        "..deepseek_sparse_attention.score_recompute.jax",
        "SparseAttnScoreRecompute",
    ),
    "sparse_attn_score_recompute_wrapper": (
        "..deepseek_sparse_attention.score_recompute.jax",
        "sparse_attn_score_recompute_wrapper",
    ),
    "DenseIndexerScoreRecompute": (
        "..deepseek_sparse_attention.score_recompute.jax",
        "DenseIndexerScoreRecompute",
    ),
    "dense_indexer_score_recompute_wrapper": (
        "..deepseek_sparse_attention.score_recompute.jax",
        "dense_indexer_score_recompute_wrapper",
    ),
    "DenseAttnScoreRecompute": (
        "..deepseek_sparse_attention.score_recompute.jax",
        "DenseAttnScoreRecompute",
    ),
    "dense_attn_score_recompute_wrapper": (
        "..deepseek_sparse_attention.score_recompute.jax",
        "dense_attn_score_recompute_wrapper",
    ),
    "IndexerBackward": (
        "..deepseek_sparse_attention.indexer_backward.jax",
        "IndexerBackward",
    ),
    "indexer_backward_wrapper": (
        "..deepseek_sparse_attention.indexer_backward.jax",
        "indexer_backward_wrapper",
    ),
    "DenseIndexerBackward": (
        "..deepseek_sparse_attention.indexer_backward.jax",
        "DenseIndexerBackward",
    ),
    "dense_indexer_backward_wrapper": (
        "..deepseek_sparse_attention.indexer_backward.jax",
        "dense_indexer_backward_wrapper",
    ),
    "GroupedGemmSwigluSm100": (
        "..grouped_gemm.grouped_gemm_swiglu.jax",
        "GroupedGemmSwigluSm100",
    ),
    "grouped_gemm_swiglu_wrapper_sm100": (
        "..grouped_gemm.grouped_gemm_swiglu.jax",
        "grouped_gemm_swiglu_wrapper_sm100",
    ),
    "GroupedGemmDswigluSm100": (
        "..grouped_gemm.grouped_gemm_dswiglu.jax",
        "GroupedGemmDswigluSm100",
    ),
    "grouped_gemm_dswiglu_wrapper_sm100": (
        "..grouped_gemm.grouped_gemm_dswiglu.jax",
        "grouped_gemm_dswiglu_wrapper_sm100",
    ),
    "GroupedGemmQuantSm100": (
        "..grouped_gemm.grouped_gemm_quant.jax",
        "GroupedGemmQuantSm100",
    ),
    "grouped_gemm_quant_wrapper_sm100": (
        "..grouped_gemm.grouped_gemm_quant.jax",
        "grouped_gemm_quant_wrapper_sm100",
    ),
    "GroupedGemmSreluSm100": (
        "..grouped_gemm.grouped_gemm_srelu.jax",
        "GroupedGemmSreluSm100",
    ),
    "grouped_gemm_srelu_wrapper_sm100": (
        "..grouped_gemm.grouped_gemm_srelu.jax",
        "grouped_gemm_srelu_wrapper_sm100",
    ),
    "GroupedGemmDsreluSm100": (
        "..grouped_gemm.grouped_gemm_dsrelu.jax",
        "GroupedGemmDsreluSm100",
    ),
    "grouped_gemm_dsrelu_wrapper_sm100": (
        "..grouped_gemm.grouped_gemm_dsrelu.jax",
        "grouped_gemm_dsrelu_wrapper_sm100",
    ),
    "GroupedGemmGluSm100": (
        "..grouped_gemm.grouped_gemm_glu.jax",
        "GroupedGemmGluSm100",
    ),
    "grouped_gemm_glu_wrapper_sm100": (
        "..grouped_gemm.grouped_gemm_glu.jax",
        "grouped_gemm_glu_wrapper_sm100",
    ),
    "GroupedGemmGluHadamardSm100": (
        "..grouped_gemm.grouped_gemm_glu_hadamard.jax",
        "GroupedGemmGluHadamardSm100",
    ),
    "grouped_gemm_glu_hadamard_wrapper_sm100": (
        "..grouped_gemm.grouped_gemm_glu_hadamard.jax",
        "grouped_gemm_glu_hadamard_wrapper_sm100",
    ),
    "GroupedGemmDgluSm100": (
        "..grouped_gemm.grouped_gemm_dglu.jax",
        "GroupedGemmDgluSm100",
    ),
    "grouped_gemm_dglu_wrapper_sm100": (
        "..grouped_gemm.grouped_gemm_dglu.jax",
        "grouped_gemm_dglu_wrapper_sm100",
    ),
    "GroupedGemmWgradSm100": (
        "..grouped_gemm.grouped_gemm_wgrad.jax",
        "GroupedGemmWgradSm100",
    ),
    "grouped_gemm_wgrad_wrapper_sm100": (
        "..grouped_gemm.grouped_gemm_wgrad.jax",
        "grouped_gemm_wgrad_wrapper_sm100",
    ),
    "DiscreteGroupedGemmSwigluSm100": (
        "..discrete_grouped_gemm.discrete_grouped_gemm_swiglu.jax",
        "DiscreteGroupedGemmSwigluSm100",
    ),
    "discrete_grouped_gemm_swiglu_wrapper_sm100": (
        "..discrete_grouped_gemm.discrete_grouped_gemm_swiglu.jax",
        "discrete_grouped_gemm_swiglu_wrapper_sm100",
    ),
    "DiscreteGroupedGemmDswigluSm100": (
        "..discrete_grouped_gemm.discrete_grouped_gemm_dswiglu.jax",
        "DiscreteGroupedGemmDswigluSm100",
    ),
    "discrete_grouped_gemm_dswiglu_wrapper_sm100": (
        "..discrete_grouped_gemm.discrete_grouped_gemm_dswiglu.jax",
        "discrete_grouped_gemm_dswiglu_wrapper_sm100",
    ),
    "SelectionAttention": (
        "..native_sparse_attention.selection.jax",
        "SelectionAttention",
    ),
    "selection_attention_wrapper": (
        "..native_sparse_attention.selection.jax",
        "selection_attention_wrapper",
    ),
    "CompressionAttention": (
        "..native_sparse_attention.compression.jax",
        "CompressionAttention",
    ),
    "compression_attention_wrapper": (
        "..native_sparse_attention.compression.jax",
        "compression_attention_wrapper",
    ),
    "SlidingWindowAttention": (
        "..native_sparse_attention.sliding_window_attention.jax",
        "SlidingWindowAttention",
    ),
    "sliding_window_attention_wrapper": (
        "..native_sparse_attention.sliding_window_attention.jax",
        "sliding_window_attention_wrapper",
    ),
    "TopKReduction": ("..native_sparse_attention.top_k.jax", "TopKReduction"),
    "topk_reduction_wrapper": (
        "..native_sparse_attention.top_k.jax",
        "topk_reduction_wrapper",
    ),
    "SdpafwdSm100D256": ("..sdpa.fwd.jax", "SdpafwdSm100D256"),
    "sdpa_fwd_wrapper_sm100_d256": ("..sdpa.fwd.jax", "sdpa_fwd_wrapper_sm100_d256"),
    "SdpabwdSm100D256": ("..sdpa.bwd.jax", "SdpabwdSm100D256"),
    "sdpa_bwd_wrapper_sm100_d256": ("..sdpa.bwd.jax", "sdpa_bwd_wrapper_sm100_d256"),
    "BlockSparseAttentionForward": (
        "..block_sparse_attention.jax",
        "BlockSparseAttentionForward",
    ),
    "block_sparse_attention_forward": (
        "..block_sparse_attention.jax",
        "block_sparse_attention_forward",
    ),
    "BlockSparseAttentionBackward": (
        "..block_sparse_attention.jax",
        "BlockSparseAttentionBackward",
    ),
    "block_sparse_attention_backward": (
        "..block_sparse_attention.jax",
        "block_sparse_attention_backward",
    ),
}

_DSA_EXPORTS = frozenset(
    name
    for name, (module_name, _) in _OPERATION_EXPORTS.items()
    if module_name.startswith("..deepseek_sparse_attention.")
)
_NSA_EXPORTS = frozenset(
    name
    for name, (module_name, _) in _OPERATION_EXPORTS.items()
    if module_name.startswith("..native_sparse_attention.")
)
_BSA_EXPORTS = frozenset(
    name
    for name, (module_name, _) in _OPERATION_EXPORTS.items()
    if module_name.startswith("..block_sparse_attention.")
)


def _load_operation(name: str) -> Any:
    module_name, symbol_name = _OPERATION_EXPORTS[name]
    value = getattr(import_module(module_name, __name__), symbol_name)
    globals()[name] = value
    return value


class _OperationNamespace:
    """Lazy view of one JAX operation family."""

    def __init__(self, qualified_name: str, exports: frozenset[str]) -> None:
        self._qualified_name = qualified_name
        self._exports = exports

    def __getattr__(self, name: str) -> Any:
        if name not in self._exports:
            raise AttributeError(f"{self._qualified_name} has no attribute {name!r}")
        value = _load_operation(name)
        setattr(self, name, value)
        return value

    def __dir__(self) -> list[str]:
        return sorted((*vars(self), *self._exports))


DSA = _OperationNamespace("cudnn.jax.DSA", _DSA_EXPORTS)
NSA = _OperationNamespace("cudnn.jax.NSA", _NSA_EXPORTS)
BSA = _OperationNamespace("cudnn.jax.BSA", _BSA_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _OPERATION_EXPORTS:
        return _load_operation(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted((*globals(), *_OPERATION_EXPORTS))


__all__ = [
    "BSA",
    "DSA",
    "JaxApiBase",
    "JaxTensorDesc",
    "NSA",
    "TupleDict",
    *_OPERATION_EXPORTS,
]
