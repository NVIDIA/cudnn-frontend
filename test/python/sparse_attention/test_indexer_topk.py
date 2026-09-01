# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract + oracle tests for cudnn.sparse_attention.indexer_topk.

No device kernel is registered yet (kernels arrive via the Kernel Factory
campaigns and register into the dispatch table): the wrapper raises
NotImplementedError for every valid configuration, the validation matrix
raises the documented error classes, and the oracle
(``indexer_topk_reference.py``) is property-tested and cross-checked against
torch.topk so kernels land against a trusted target.
"""

import pytest
import torch

from test_utils import torch_fork_set_rng

from sparse_attention.indexer_topk_reference import reference_indexer_topk
from sparse_attention.sparse_attention_reference import reference_sparse_attention_forward

pytestmark = pytest.mark.L0


def _wrapper():
    from cudnn.sparse_attention.indexer_topk import indexer_topk_wrapper

    return indexer_topk_wrapper


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required")


def _mk(t_q=48, t_e=48, h_i=8, h_ik=1, d_i=32, device="cuda"):
    q = torch.randn(t_q, h_i, d_i, dtype=torch.bfloat16, device=device) / 4
    k = torch.randn(t_e, h_ik, d_i, dtype=torch.bfloat16, device=device) / 4
    w = torch.rand(t_q, h_i, dtype=torch.float32, device=device) + 0.5
    cu_q = torch.tensor([0, t_q], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, t_e], dtype=torch.int32, device=device)
    return q, k, w, cu_q, cu_k


# ---------------------------------------------------------------------------
# Oracle property tests + torch.topk cross-check
# ---------------------------------------------------------------------------
@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize("activation", ["relu", "none"])
@pytest.mark.parametrize("ratio,score_pool", [(1, 1), (4, 1), (1, 4)])
def test_oracle_matches_dense_topk(activation, ratio, score_pool):
    """No-force case: oracle == exact topk over the dense masked pooled scores."""
    _require_cuda()
    q, k, w, cu_q, cu_k = _mk()
    top_k = 6
    idx, length, logit = reference_indexer_topk(
        q, k, top_k, weights=w, activation=activation, ratio=ratio, score_pool=score_pool, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k
    )
    t_q, h_i, _ = q.shape
    g_ent = ratio * score_pool
    for t in range(0, t_q, 7):
        n_valid = min((t + 1) // ratio, k.shape[0])
        n_pooled = n_valid // score_pool
        # dense recompute
        s = q[t].float() @ k[:, 0, :].float().t()
        if activation == "relu":
            s = torch.relu(s)
        score = (s * w[t].unsqueeze(-1)).sum(0)[:n_valid]
        if n_pooled == 0:
            assert length[t] == 0
            continue
        pooled = score[: n_pooled * score_pool].reshape(n_pooled, score_pool).amax(-1)
        k_eff = min(top_k, n_pooled)
        expect = torch.topk(pooled, k_eff).indices.sort().values
        got = idx[t, : int(length[t])].long()
        assert int(length[t]) == k_eff
        assert torch.equal(got, expect), f"row {t}: {got} vs {expect}"
        assert torch.equal(got, got.sort().values)  # ascending
        torch.testing.assert_close(logit[t, :k_eff], pooled[got], atol=1e-5, rtol=1e-5)
        assert (idx[t, int(length[t]) :] == -1).all()


@torch_fork_set_rng(seed=1)
def test_oracle_forced_includes_and_groups():
    _require_cuda()
    q, k, w, cu_q, cu_k = _mk(h_i=8, h_ik=4)
    idx, length, _ = reference_indexer_topk(
        q, k, 6, weights=None, activation="none", head_groups=4, force_first=1, force_last=2, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k
    )
    assert idx.shape == (q.shape[0], 4, 6) and length.shape == (q.shape[0], 4)
    for t in (10, 33, 47):
        n_pooled = min(t + 1, k.shape[0])
        for g in range(4):
            got = set(idx[t, g, : int(length[t, g])].tolist())
            assert 0 in got  # force_first
            assert n_pooled - 1 in got and n_pooled - 2 in got  # force_last=2
            assert len(got) == min(6, n_pooled)


@torch_fork_set_rng(seed=2)
def test_oracle_chunked_prefill_offsets_and_determinism():
    _require_cuda()
    q, k, w, cu_q, _ = _mk(t_q=16, t_e=64)
    cu_k = torch.tensor([0, 64], dtype=torch.int32, device="cuda")
    offs = torch.tensor([32], dtype=torch.int32, device="cuda")  # queries are tokens 32..47
    r1 = reference_indexer_topk(q, k, 8, weights=w, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, q_causal_offsets=offs)
    r2 = reference_indexer_topk(q, k, 8, weights=w, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, q_causal_offsets=offs)
    idx, length, _ = r1
    # query row 0 sits at global position 32 -> 33 valid entries
    assert int(length[0]) == 8 and int(idx[0].max()) <= 32
    for a, b in zip(r1, r2):
        assert torch.equal(a, b)


@torch_fork_set_rng(seed=3)
def test_oracle_indices_feed_forward_oracle():
    """Pipeline: indexer_topk oracle -> sparse_attention forward oracle."""
    _require_cuda()
    t = 64
    qi = torch.randn(t, 4, 32, dtype=torch.bfloat16, device="cuda") / 4
    ki = torch.randn(t // 4, 1, 32, dtype=torch.bfloat16, device="cuda") / 4
    cu_q = torch.tensor([0, t], dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0, t // 4], dtype=torch.int32, device="cuda")
    idx, length, _ = reference_indexer_topk(qi, ki, 8, ratio=4, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k)

    q = torch.randn(t, 8, 64, dtype=torch.bfloat16, device="cuda") / 4
    k = torch.randn(t, 2, 64, dtype=torch.bfloat16, device="cuda") / 4
    v = torch.randn(t, 2, 64, dtype=torch.bfloat16, device="cuda") / 4
    out, lse = reference_sparse_attention_forward(q, k, v, idx, topk_length=length, index_granularity=4)
    assert out.isfinite().all()
    live = length > 0
    assert lse[live].isfinite().all() and torch.isneginf(lse[~live]).all()


# ---------------------------------------------------------------------------
# Contract: rejection matrix (no kernel registered -> valid configs also raise)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "mutate, err",
    [
        (lambda a: dict(), NotImplementedError),  # valid config, no registered kernel yet
        (lambda a: dict(activation="softmax"), ValueError),
        (lambda a: dict(head_groups=3), ValueError),  # does not divide H_i=8
        (lambda a: dict(force_first=5, force_last=5), ValueError),  # exceeds top_k
        (lambda a: dict(ratio=0), ValueError),
        (lambda a: dict(cu_seqlens_q=None), ValueError),  # THD without cu_seqlens
        (lambda a: dict(weights=a[2][:, :4]), ValueError),  # wrong weights shape
        (lambda a: dict(k_index=a[1].expand(-1, 3, -1).contiguous()), ValueError),  # H_ik not in {1, G}
        (lambda a: dict(q_index=a[0].float()), ValueError),  # fp32 q
    ],
)
def test_rejection_matrix(mutate, err):
    _require_cuda()
    q, k, w, cu_q, cu_k = _mk()
    kwargs = dict(weights=w, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k)
    kwargs.update(mutate((q, k, w, cu_q, cu_k)))
    q_arg = kwargs.pop("q_index", q)
    k_arg = kwargs.pop("k_index", k)
    with pytest.raises(err):
        _wrapper()(q_arg, k_arg, 8, **kwargs)


# ---------------------------------------------------------------------------
# Framework neutrality
# ---------------------------------------------------------------------------
def test_import_without_torch():
    import os
    import subprocess
    import sys

    script = (
        "import sys\n"
        "class B:\n"
        "    def find_module(self, name, path=None):\n"
        "        if name == 'torch' or name.startswith('torch.'):\n"
        "            raise ImportError('torch blocked')\n"
        "sys.meta_path.insert(0, B())\n"
        "import cudnn.sparse_attention\n"
        "from cudnn.sparse_attention.indexer_topk import api\n"
        "assert 'torch' not in sys.modules\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env={**os.environ})
    assert result.returncode == 0, f"torch-free import failed:\n{result.stderr}"


def test_non_torch_inputs_reach_validation_cleanly():
    import numpy as np

    _require_cuda()
    q = np.zeros((8, 4, 32), dtype=np.float16)
    k = np.zeros((8, 1, 32), dtype=np.float16)
    cu = np.array([0, 8], dtype=np.int32)
    with pytest.raises(ValueError, match="CUDA"):
        _wrapper()(q, k, 4, cu_seqlens_q=cu, cu_seqlens_k=cu)
