# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Shared helpers for the linear-attention test suite: the Q/K/V input factory
(the gate tensors are drawn by ``test_la.gen_gates``).

Also overlays the source ``python/cudnn`` dir onto the installed ``cudnn``
package's ``__path__``: the built wheel may lack the engine subtrees the
suite pins (``cudnn.linear_attention.frost``). Unnecessary once the engines
ship in the built frontend package.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

try:
    import cudnn

    _SRC_CUDNN = Path(__file__).resolve().parents[3] / "python" / "cudnn"
    if _SRC_CUDNN.is_dir():
        source_path = str(_SRC_CUDNN)
        if source_path in cudnn.__path__:
            cudnn.__path__.remove(source_path)
        cudnn.__path__.insert(0, source_path)
except ImportError:
    pass  # test_la.py skips via importorskip


def multidist_randu(num_dists, dim, *, device, mean_std=0.05, lower=-0.25, upper=0.25):
    """Rows drawn from per-row uniform distributions with normally-distributed
    means: mildly heterogeneous data that still keeps the recurrence stable."""
    means = torch.randn(num_dists, 1, device=device, dtype=torch.float32) * mean_std
    return means + torch.rand(num_dists, dim, device=device, dtype=torch.float32) * (upper - lower) + lower


def gen_qkv(B, T, H, HV, K, V, dtype, device="cuda"):
    """Dense [B, T, heads, dim] Q/K/V. K is l2-normalized along the feature
    dim so the delta-rule update (I - Beta K K^T) stays contractive for
    Beta in (0, 1]."""
    q = multidist_randu(B * T * H, K, device=device).reshape(B, T, H, K)
    k = multidist_randu(B * T * H, K, device=device).reshape(B, T, H, K)
    k = F.normalize(k, p=2.0, dim=-1)
    v = multidist_randu(B * T * HV, V, device=device).reshape(B, T, HV, V)
    return (
        q.to(dtype).contiguous(),
        k.to(dtype).contiguous(),
        v.to(dtype).contiguous(),
    )
