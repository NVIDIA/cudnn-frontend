"""Lazy API exports for DSA sparse-attention backward."""

from ..._operation_api import make_operation_api

_API_EXPORTS = (
    "SparseAttentionBackward",
    "sparse_attention_backward_wrapper",
)

__all__, __getattr__ = make_operation_api(
    globals(),
    exports=_API_EXPORTS,
)
