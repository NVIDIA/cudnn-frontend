# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Fixtures for the GDN test suite: seeded input factories for q/k/v and the
gate tensors (g = log decay, beta = write strength).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F


def multidist_randu(num_dists, dim, *, device, mean_std=0.05, lower=-0.25, upper=0.25):
    """Rows drawn from per-row uniform distributions with normally-distributed
    means: mildly heterogeneous data that still keeps the recurrence stable."""
    means = torch.randn(num_dists, 1, device=device, dtype=torch.float32) * mean_std
    return means + torch.rand(num_dists, dim, device=device, dtype=torch.float32) * (upper - lower) + lower


def gen_qkv(B, T, H, HV, K, V, dtype, device="cuda"):
    """Dense [B, T, heads, dim] q/k/v. k is l2-normalized along the feature
    dim so the delta-rule update (I - beta k k^T) stays contractive for
    beta in (0, 1]."""
    q = multidist_randu(B * T * H, K, device=device).reshape(B, T, H, K)
    k = multidist_randu(B * T * H, K, device=device).reshape(B, T, H, K)
    k = F.normalize(k, p=2.0, dim=-1)
    v = multidist_randu(B * T * HV, V, device=device).reshape(B, T, HV, V)
    return (
        q.to(dtype).contiguous(),
        k.to(dtype).contiguous(),
        v.to(dtype).contiguous(),
    )


def gen_gates(B, T, HV, dtype, device="cuda", alpha=True, beta=True):
    """Gate tensors [B, T, HV]. alpha off -> g = 0 (decay factor exactly 1);
    beta off -> beta = 1 (plain delta rule)."""
    if alpha:
        a = torch.empty(B, T, HV, device=device, dtype=torch.float32).uniform_(0.1, 1.0)
        g = a.log()
    else:
        g = torch.zeros(B, T, HV, device=device, dtype=torch.float32)
    if beta:
        b = torch.rand(B, T, HV, device=device, dtype=torch.float32)
    else:
        b = torch.ones(B, T, HV, device=device, dtype=torch.float32)
    return g.to(dtype).contiguous(), b.to(dtype).contiguous()


def gen_kda_gates(B, T, HV, K, dtype, device="cuda", alpha=True, beta=True, lo=0.9):
    """KDA gate tensors: g [B, T, HV, K] fp32 per-key-channel log decay, beta
    [B, T, HV] scalar write strength. ``g`` stays fp32 (the per-channel decay
    kernel keeps it fp32); the per-channel cumulative product over a 64-token
    chunk stays well within bf16 range for ``alpha >= lo``. alpha off -> g = 0
    (no decay); beta off -> beta = 1 (plain delta rule)."""
    if alpha:
        a = torch.empty(B, T, HV, K, device=device, dtype=torch.float32).uniform_(lo, 1.0)
        g = a.log()
    else:
        g = torch.zeros(B, T, HV, K, device=device, dtype=torch.float32)
    if beta:
        b = torch.rand(B, T, HV, device=device, dtype=torch.float32)
    else:
        b = torch.ones(B, T, HV, device=device, dtype=torch.float32)
    return g.contiguous(), b.to(dtype).contiguous()


def gen_gdn2_gates(B, T, HO, K, V, dtype, device="cuda", alpha=True, beta=True, w=True, lo=0.5):
    """GDN-2 gate tensors: g [B, T, HO, K] fp32 per-key-channel log decay,
    beta [B, T, HO, K] per-key erase gate, w [B, T, HO, V] per-value write
    gate. beta/w are io-dtype (rounded before both kernel and reference see
    them). alpha off -> g = 0; beta/w off -> ones."""
    if alpha:
        a = torch.empty(B, T, HO, K, device=device, dtype=torch.float32).uniform_(lo, 1.0)
        g = a.log()
    else:
        g = torch.zeros(B, T, HO, K, device=device, dtype=torch.float32)
    if beta:
        b = (torch.rand(B, T, HO, K, device=device, dtype=torch.float32).sigmoid() * 2.0).to(dtype)
    else:
        b = torch.ones(B, T, HO, K, device=device, dtype=dtype)
    if w:
        wt = torch.rand(B, T, HO, V, device=device, dtype=torch.float32).sigmoid().to(dtype)
    else:
        wt = torch.ones(B, T, HO, V, device=device, dtype=dtype)
    return g.contiguous(), b.contiguous(), wt.contiguous()


@pytest.fixture()
def qkv_factory():
    return gen_qkv


@pytest.fixture()
def gate_factory():
    return gen_gates
