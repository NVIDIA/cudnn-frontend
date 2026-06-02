"""
Tests for the KDA PyTorch custom operator
(``cudnn.experimental.ops.kimi_delta_attention``).

KDA replaces GDN's scalar decay with a per-channel decay; ``g`` is
``[B, T, H, K]``. The same per-token recurrent reference acts as the
ground truth.
"""

from __future__ import annotations

import math

import pytest
import torch

import cudnn  # noqa: F401
from cudnn.experimental.ops import kimi_delta_attention
from cudnn.experimental.ops.linear_attention.kda import (
    _kda_forward_recurrent,
    _kda_forward_chunked,
)
from cudnn.experimental.ops.linear_attention._common import (
    bthd_to_bhtd,
    bhtd_to_bthd,
    bth_to_bht,
)


def _rand_inputs(B, T, H, K, V, dtype=torch.float64, device="cpu", seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    q = (torch.rand(B, T, H, K, dtype=dtype, device=device, generator=g) + 0.1) * 0.5
    k = (torch.rand(B, T, H, K, dtype=dtype, device=device, generator=g) + 0.1) * 0.5
    v = torch.randn(B, T, H, V, dtype=dtype, device=device, generator=g)
    alpha = 0.85 + 0.15 * torch.rand(B, T, H, K, dtype=dtype, device=device, generator=g)
    g_decay = torch.log(alpha)
    beta = torch.sigmoid(torch.randn(B, T, H, dtype=dtype, device=device, generator=g)) * (0.8 / K)
    return q, k, v, g_decay, beta


def _reference_recurrent(q, k, v, g_decay, beta, scale):
    qf = bthd_to_bhtd(q).to(torch.float64) * scale
    kf = bthd_to_bhtd(k).to(torch.float64)
    vf = bthd_to_bhtd(v).to(torch.float64)
    alphaf = bthd_to_bhtd(g_decay).exp().to(torch.float64)
    betaf = bth_to_bht(beta).to(torch.float64)
    S0 = torch.zeros(q.shape[0], q.shape[2], q.shape[3], v.shape[-1], dtype=torch.float64, device=q.device)
    out, final = _kda_forward_recurrent(qf, kf, vf, alphaf, betaf, S0)
    return bhtd_to_bthd(out), final


class TestKdaForward:
    @pytest.mark.parametrize("B,T,H,K,V,chunk_size", [
        (1, 8, 1, 16, 24, 4),
        (2, 64, 4, 32, 32, 8),
        (1, 10, 2, 8, 8, 4),
        (1, 32, 1, 16, 16, 32),
        (2, 100, 2, 8, 8, 8),
    ])
    def test_forward_matches_recurrent(self, B, T, H, K, V, chunk_size):
        q, k, v, g, beta = _rand_inputs(B, T, H, K, V, dtype=torch.float64)
        scale = 1.0 / math.sqrt(K)
        o, final = kimi_delta_attention(q, k, v, g, beta, scale=scale,
                                        chunk_size=chunk_size, output_final_state=True)
        o_ref, final_ref = _reference_recurrent(q, k, v, g, beta, scale)
        torch.testing.assert_close(o.double(), o_ref, atol=1e-9, rtol=1e-9)
        torch.testing.assert_close(final.double(), final_ref, atol=1e-9, rtol=1e-9)

    def test_default_scale(self):
        B, T, H, K, V = 1, 16, 1, 8, 8
        q, k, v, g, beta = _rand_inputs(B, T, H, K, V, dtype=torch.float64)
        o_default, _ = kimi_delta_attention(q, k, v, g, beta)
        o_explicit, _ = kimi_delta_attention(q, k, v, g, beta, scale=1.0 / math.sqrt(K))
        torch.testing.assert_close(o_default, o_explicit)

    def test_initial_state(self):
        B, T, H, K, V = 1, 16, 1, 8, 8
        q, k, v, g, beta = _rand_inputs(B, T, H, K, V, dtype=torch.float64)
        S0 = torch.randn(B, H, K, V, dtype=torch.float64) * 0.01
        o, final = kimi_delta_attention(q, k, v, g, beta, initial_state=S0,
                                        output_final_state=True, chunk_size=4)
        qf = bthd_to_bhtd(q).to(torch.float64) * (1.0 / math.sqrt(K))
        kf = bthd_to_bhtd(k).to(torch.float64)
        vf = bthd_to_bhtd(v).to(torch.float64)
        alphaf = bthd_to_bhtd(g).exp().to(torch.float64)
        betaf = bth_to_bht(beta).to(torch.float64)
        out_ref, final_ref = _kda_forward_recurrent(qf, kf, vf, alphaf, betaf, S0.clone())
        torch.testing.assert_close(o.double(), bhtd_to_bthd(out_ref), atol=1e-9, rtol=1e-9)
        torch.testing.assert_close(final.double(), final_ref, atol=1e-9, rtol=1e-9)


class TestKdaBackward:
    @pytest.mark.parametrize("B,T,H,K,V,chunk_size", [
        (1, 8, 1, 8, 8, 4),
        (2, 16, 2, 8, 8, 8),
        (1, 10, 1, 8, 8, 4),
    ])
    def test_grads_match_autograd_recurrent(self, B, T, H, K, V, chunk_size):
        q, k, v, g, beta = _rand_inputs(B, T, H, K, V, dtype=torch.float64)
        q.requires_grad_(True); k.requires_grad_(True); v.requires_grad_(True)
        g.requires_grad_(True); beta.requires_grad_(True)
        scale = 1.0 / math.sqrt(K)

        o, _ = kimi_delta_attention(q, k, v, g, beta, scale=scale, chunk_size=chunk_size)
        dO = torch.randn_like(o)
        o.backward(dO)

        qr = q.detach().clone().requires_grad_(True)
        kr = k.detach().clone().requires_grad_(True)
        vr = v.detach().clone().requires_grad_(True)
        gr = g.detach().clone().requires_grad_(True)
        br = beta.detach().clone().requires_grad_(True)
        qf = bthd_to_bhtd(qr) * scale
        kf = bthd_to_bhtd(kr)
        vf = bthd_to_bhtd(vr)
        alphaf = bthd_to_bhtd(gr).exp()
        betaf = bth_to_bht(br)
        S0 = torch.zeros(B, H, K, V, dtype=q.dtype, device=q.device)
        out_ref, _ = _kda_forward_recurrent(qf, kf, vf, alphaf, betaf, S0)
        bhtd_to_bthd(out_ref).backward(dO)

        torch.testing.assert_close(q.grad, qr.grad, atol=1e-8, rtol=1e-8)
        torch.testing.assert_close(k.grad, kr.grad, atol=1e-8, rtol=1e-8)
        torch.testing.assert_close(v.grad, vr.grad, atol=1e-8, rtol=1e-8)
        torch.testing.assert_close(g.grad, gr.grad, atol=1e-8, rtol=1e-8)
        torch.testing.assert_close(beta.grad, br.grad, atol=1e-8, rtol=1e-8)


class TestKdaCompileAndCuda:
    def test_torch_compile_forward(self):
        B, T, H, K, V = 1, 16, 1, 8, 8
        q, k, v, g, beta = _rand_inputs(B, T, H, K, V, dtype=torch.float32)
        compiled = torch.compile(kimi_delta_attention, fullgraph=True)
        o_eager, _ = kimi_delta_attention(q, k, v, g, beta)
        o_comp, _ = compiled(q, k, v, g, beta)
        torch.testing.assert_close(o_eager, o_comp)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_cuda_smoke(self):
        B, T, H, K, V = 1, 16, 1, 8, 8
        q, k, v, g, beta = _rand_inputs(B, T, H, K, V, dtype=torch.float32, device="cuda")
        q.requires_grad_(True); k.requires_grad_(True); v.requires_grad_(True)
        o, _ = kimi_delta_attention(q, k, v, g, beta)
        o.sum().backward()
        assert q.grad is not None and torch.isfinite(q.grad).all()
