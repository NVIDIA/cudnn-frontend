"""
Utilities for DSA (DeepSeek Sparse Attention) tests.

Parameterization decorators and init helpers. Mirrors the NSA test utilities
pattern (see test/python/fe_api/nsa/nsa_utils.py).
"""

from typing import Optional, Tuple

import pytest
import torch

# Parameterization marks shared by every DSA test
DSA_PARAM_MARKS = [
    pytest.mark.parametrize("dtype", [torch.bfloat16]),
    pytest.mark.parametrize("acc_dtype", [torch.float32]),
]

DSA_SPARSE_ATTENTION_BACKWARD_PARAM_MARKS = [
    pytest.mark.parametrize("head_dim", [512]),
    pytest.mark.parametrize("head_dim_v", [512]),
    pytest.mark.parametrize("num_heads", [64]),
    pytest.mark.parametrize("topk", [512]),
    pytest.mark.parametrize("has_topk_length", [False, True]),
]

DSA_INDEXER_FORWARD_PARAM_MARKS = [
    pytest.mark.parametrize("head_dim", [128]),
    pytest.mark.parametrize("qhead_per_kv_head", [32, 64]),
    pytest.mark.parametrize("ratio", [4]),
]

DSA_INDEXER_TOP_K_PARAM_MARKS = [
    pytest.mark.parametrize("top_k", [512]),
    pytest.mark.parametrize("next_n", [1]),
    pytest.mark.parametrize("return_val", [True, False]),
]

DSA_SCORE_RECOMPUTE_PARAM_MARKS = [
    pytest.mark.parametrize("head_dim", [128]),
    pytest.mark.parametrize("qhead_per_kv_head", [32]),
    pytest.mark.parametrize("score_type", ["indexer", "attention"]),
]

DSA_INDEXER_BACKWARD_PARAM_MARKS = [
    pytest.mark.parametrize("head_dim", [128]),
    # IndexerBackward requires heads >= 64 (warp-specialized GEMM layout).
    pytest.mark.parametrize("qhead_per_kv_head", [64]),
    pytest.mark.parametrize("block_I", [128]),
]

DSA_DENSE_INDEXER_BACKWARD_PARAM_MARKS = [
    pytest.mark.parametrize("head_dim", [128]),
    pytest.mark.parametrize("qhead_per_kv_head", [64]),
    pytest.mark.parametrize("block_I", [128]),
    pytest.mark.parametrize("ratio", [1]),
]


def _apply(marks, func):
    for mark in reversed(marks):
        func = mark(func)
    return func


def with_dsa_sparse_attention_backward_params(func):
    return _apply(DSA_PARAM_MARKS + DSA_SPARSE_ATTENTION_BACKWARD_PARAM_MARKS, func)


def with_dsa_indexer_forward_params(func):
    return _apply(DSA_PARAM_MARKS + DSA_INDEXER_FORWARD_PARAM_MARKS, func)


def with_dsa_indexer_top_k_params(func):
    return _apply(DSA_PARAM_MARKS + DSA_INDEXER_TOP_K_PARAM_MARKS, func)


def with_dsa_score_recompute_params(func):
    return _apply(DSA_PARAM_MARKS + DSA_SCORE_RECOMPUTE_PARAM_MARKS, func)


def with_dsa_indexer_backward_params(func):
    return _apply(DSA_PARAM_MARKS + DSA_INDEXER_BACKWARD_PARAM_MARKS, func)


def with_dsa_dense_indexer_backward_params(func):
    return _apply(DSA_PARAM_MARKS + DSA_DENSE_INDEXER_BACKWARD_PARAM_MARKS, func)


def dsa_init(
    request: pytest.FixtureRequest,
    dtype: Optional[torch.dtype] = None,
    acc_dtype: Optional[torch.dtype] = None,
    head_dim: Optional[int] = None,
    head_dim_v: Optional[int] = None,
    num_heads: Optional[int] = None,
    qhead_per_kv_head: Optional[int] = None,
    topk: Optional[int] = None,
    ratio: Optional[int] = None,
    score_type: Optional[str] = None,
    has_topk_length: Optional[bool] = None,
    next_n: Optional[int] = None,
    return_val: Optional[bool] = None,
    block_I: Optional[int] = None,
    top_k: Optional[int] = None,
    min_compute_capability: int = 100,
    b_default: int = 1,
    s_q_default: int = 1024,
    s_kv_default: int = 1024,
) -> dict:
    """Resolve DSA test configuration, applying CLI overrides from --dsa-* options.

    Also performs the requested compute-capability gate and skips if unmet.
    """
    major, minor = torch.cuda.get_device_capability()
    compute_capability = major * 10 + minor
    if compute_capability < min_compute_capability:
        required_major = min_compute_capability // 10
        required_minor = min_compute_capability % 10
        pytest.skip(f"DSA requires compute capability >= {required_major}.{required_minor}, " f"found SM{major}{minor}")

    def opt(name: str, default):
        value = request.config.getoption(name, default=None)
        return value if value is not None else default

    b = opt("--dsa-b", b_default)
    s_q = opt("--dsa-s_q", s_q_default)
    s_kv = opt("--dsa-s_kv", s_kv_default)
    h_q_cli = opt("--dsa-h_q", None)
    h_kv_cli = opt("--dsa-h_kv", None)
    d_qk_cli = opt("--dsa-d_qk", None)
    d_v_cli = opt("--dsa-d_v", None)
    topk_cli = opt("--dsa-topk", None)
    ratio_cli = opt("--dsa-ratio", None)

    cfg = {
        "dtype": dtype if dtype is not None else torch.bfloat16,
        "acc_dtype": acc_dtype if acc_dtype is not None else torch.float32,
        "b": b,
        "s_q": s_q,
        "s_kv": s_kv,
        "h_q": h_q_cli if h_q_cli is not None else (num_heads if num_heads is not None else 64),
        "h_kv": h_kv_cli if h_kv_cli is not None else 1,
        "head_dim": d_qk_cli if d_qk_cli is not None else (head_dim if head_dim is not None else 128),
        "head_dim_v": d_v_cli if d_v_cli is not None else (head_dim_v if head_dim_v is not None else 128),
        "qhead_per_kv_head": qhead_per_kv_head,
        "topk": topk_cli if topk_cli is not None else (topk if topk is not None else (top_k if top_k is not None else 512)),
        "ratio": ratio_cli if ratio_cli is not None else (ratio if ratio is not None else 4),
        "score_type": score_type,
        "has_topk_length": has_topk_length,
        "next_n": next_n,
        "return_val": return_val,
        "block_I": block_I,
        "skip_ref": bool(request.config.getoption("--skip-ref", default=False)),
    }
    return cfg
