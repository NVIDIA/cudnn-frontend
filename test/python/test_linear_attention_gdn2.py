"""
Tests for the GDN-2 PyTorch custom operator
(``cudnn.experimental.ops.gated_delta_net_v2``).

GDN-2 replaces GDN's scalar ``beta`` with two channel-wise gates: an
erase gate ``b`` (key axis, shape ``[B,T,H,K]``) and a write gate ``w``
(value axis, shape ``[B,T,H,V]``), and uses per-channel decay
``g: [B,T,H,K]``. We sanity-check three things:

1. The op output matches the per-token reference recurrence.
2. ``b = beta * 1_K``, ``w = 1_V``, ``g`` replicated across K reduces to GDN.
3. ``b = beta * 1_K``, ``w = 1_V``, full per-channel ``g`` reduces to KDA.
4. Backward gradients agree with autograd through the recurrent reference.
"""

from __future__ import annotations

import math

import pytest
import torch

import cudnn  # noqa: F401
from cudnn.experimental.ops import (
    gated_delta_net,
    gated_delta_net_v2,
    kimi_delta_attention,
)
from cudnn.experimental.ops.linear_attention.gdn2 import _gdn2_forward_recurrent
from cudnn.experimental.ops.linear_attention._common import (
    bthd_to_bhtd,
    bhtd_to_bthd,
)


def _rand_inputs(B, T, H, K, V, dtype=torch.float64, device="cpu", seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    q = (torch.rand(B, T, H, K, dtype=dtype, device=device, generator=g) + 0.1) * 0.5
    k = (torch.rand(B, T, H, K, dtype=dtype, device=device, generator=g) + 0.1) * 0.5
    v = torch.randn(B, T, H, V, dtype=dtype, device=device, generator=g)
    alpha = 0.85 + 0.15 * torch.rand(B, T, H, K, dtype=dtype, device=device, generator=g)
    g_decay = torch.log(alpha)
    # Erase / write gates as sigmoid logits (so they live in (0, 1)).
    b_gate = torch.sigmoid(torch.randn(B, T, H, K, dtype=dtype, device=device, generator=g)) * (0.8 / K)
    w_gate = torch.sigmoid(torch.randn(B, T, H, V, dtype=dtype, device=device, generator=g))
    return q, k, v, g_decay, b_gate, w_gate


def _reference(q, k, v, g, b, w, scale):
    qf = bthd_to_bhtd(q).to(torch.float64) * scale
    kf = bthd_to_bhtd(k).to(torch.float64)
    vf = bthd_to_bhtd(v).to(torch.float64)
    alphaf = bthd_to_bhtd(g).exp().to(torch.float64)
    bf = bthd_to_bhtd(b).to(torch.float64)
    wf = bthd_to_bhtd(w).to(torch.float64)
    S0 = torch.zeros(q.shape[0], q.shape[2], q.shape[3], v.shape[-1], dtype=torch.float64, device=q.device)
    out, final = _gdn2_forward_recurrent(qf, kf, vf, alphaf, bf, wf, S0)
    return bhtd_to_bthd(out), final


class TestGdn2Forward:
    @pytest.mark.parametrize("B,T,H,K,V", [
        (1, 8, 1, 8, 8),
        (2, 16, 2, 8, 8),
        (1, 32, 1, 16, 16),
    ])
    def test_forward_matches_recurrent(self, B, T, H, K, V):
        q, k, v, g, b, w = _rand_inputs(B, T, H, K, V, dtype=torch.float64)
        scale = 1.0 / math.sqrt(K)
        o, final = gated_delta_net_v2(q, k, v, g, b, w, scale=scale, output_final_state=True)
        o_ref, final_ref = _reference(q, k, v, g, b, w, scale)
        torch.testing.assert_close(o.double(), o_ref, atol=1e-9, rtol=1e-9)
        torch.testing.assert_close(final.double(), final_ref, atol=1e-9, rtol=1e-9)

    def test_reduces_to_gdn(self):
        """``w = 1_V``, ``b = beta * 1_K``, ``g = g_scalar`` replicated across K -> GDN."""
        B, T, H, K, V = 1, 16, 1, 8, 8
        torch.manual_seed(0)
        q = (torch.rand(B, T, H, K, dtype=torch.float64) + 0.1) * 0.5
        k = (torch.rand(B, T, H, K, dtype=torch.float64) + 0.1) * 0.5
        v = torch.randn(B, T, H, V, dtype=torch.float64)
        alpha = 0.85 + 0.15 * torch.rand(B, T, H, dtype=torch.float64)
        g_scalar = torch.log(alpha)                                                 # [B,T,H]
        beta = torch.sigmoid(torch.randn(B, T, H, dtype=torch.float64)) * (0.8 / K)  # [B,T,H]
        scale = 1.0 / math.sqrt(K)

        # GDN
        o_gdn, _ = gated_delta_net(q, k, v, g_scalar, beta, scale=scale, chunk_size=4)

        # GDN-2 with replicated gates / decay. GDN's single beta scales both
        # erase AND write, so b = w = beta * 1 in the corresponding axes.
        g_v2 = g_scalar.unsqueeze(-1).expand(B, T, H, K).contiguous()
        b_v2 = beta.unsqueeze(-1).expand(B, T, H, K).contiguous()
        w_v2 = beta.unsqueeze(-1).expand(B, T, H, V).contiguous()
        o_v2, _ = gated_delta_net_v2(q, k, v, g_v2, b_v2, w_v2, scale=scale)

        torch.testing.assert_close(o_v2, o_gdn, atol=1e-7, rtol=1e-7)

    def test_reduces_to_kda(self):
        """``w = 1_V``, ``b = beta * 1_K``, full per-channel ``g`` -> KDA."""
        B, T, H, K, V = 1, 16, 1, 8, 8
        torch.manual_seed(1)
        q = (torch.rand(B, T, H, K, dtype=torch.float64) + 0.1) * 0.5
        k = (torch.rand(B, T, H, K, dtype=torch.float64) + 0.1) * 0.5
        v = torch.randn(B, T, H, V, dtype=torch.float64)
        alpha = 0.85 + 0.15 * torch.rand(B, T, H, K, dtype=torch.float64)
        g_pc = torch.log(alpha)                                                  # [B,T,H,K]
        beta = torch.sigmoid(torch.randn(B, T, H, dtype=torch.float64)) * (0.8 / K)
        scale = 1.0 / math.sqrt(K)

        o_kda, _ = kimi_delta_attention(q, k, v, g_pc, beta, scale=scale, chunk_size=4)

        b_v2 = beta.unsqueeze(-1).expand(B, T, H, K).contiguous()
        w_v2 = beta.unsqueeze(-1).expand(B, T, H, V).contiguous()
        o_v2, _ = gated_delta_net_v2(q, k, v, g_pc, b_v2, w_v2, scale=scale)

        torch.testing.assert_close(o_v2, o_kda, atol=1e-7, rtol=1e-7)


class TestGdn2Backward:
    @pytest.mark.parametrize("B,T,H,K,V", [
        (1, 8, 1, 8, 8),
        (2, 16, 2, 8, 8),
    ])
    def test_grads_match_autograd_recurrent(self, B, T, H, K, V):
        q, k, v, g, b, w = _rand_inputs(B, T, H, K, V, dtype=torch.float64)
        for t in (q, k, v, g, b, w):
            t.requires_grad_(True)
        scale = 1.0 / math.sqrt(K)

        o, _ = gated_delta_net_v2(q, k, v, g, b, w, scale=scale)
        dO = torch.randn_like(o)
        o.backward(dO)

        # Reference: autograd through the per-token recurrence.
        qr = q.detach().clone().requires_grad_(True)
        kr = k.detach().clone().requires_grad_(True)
        vr = v.detach().clone().requires_grad_(True)
        gr = g.detach().clone().requires_grad_(True)
        br = b.detach().clone().requires_grad_(True)
        wr = w.detach().clone().requires_grad_(True)
        qf = bthd_to_bhtd(qr) * scale
        kf = bthd_to_bhtd(kr)
        vf = bthd_to_bhtd(vr)
        alphaf = bthd_to_bhtd(gr).exp()
        bf = bthd_to_bhtd(br)
        wf = bthd_to_bhtd(wr)
        S0 = torch.zeros(B, H, K, V, dtype=q.dtype, device=q.device)
        out_ref, _ = _gdn2_forward_recurrent(qf, kf, vf, alphaf, bf, wf, S0)
        bhtd_to_bthd(out_ref).backward(dO)

        torch.testing.assert_close(q.grad, qr.grad, atol=1e-8, rtol=1e-8)
        torch.testing.assert_close(k.grad, kr.grad, atol=1e-8, rtol=1e-8)
        torch.testing.assert_close(v.grad, vr.grad, atol=1e-8, rtol=1e-8)
        torch.testing.assert_close(g.grad, gr.grad, atol=1e-8, rtol=1e-8)
        torch.testing.assert_close(b.grad, br.grad, atol=1e-8, rtol=1e-8)
        torch.testing.assert_close(w.grad, wr.grad, atol=1e-8, rtol=1e-8)


class TestGdn2CompileAndCuda:
    def test_torch_compile_forward(self):
        B, T, H, K, V = 1, 16, 1, 8, 8
        q, k, v, g, b, w = _rand_inputs(B, T, H, K, V, dtype=torch.float32)
        compiled = torch.compile(gated_delta_net_v2, fullgraph=True)
        o_eager, _ = gated_delta_net_v2(q, k, v, g, b, w)
        o_comp, _ = compiled(q, k, v, g, b, w)
        torch.testing.assert_close(o_eager, o_comp)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_cuda_smoke(self):
        B, T, H, K, V = 1, 16, 1, 8, 8
        q, k, v, g, b, w = _rand_inputs(B, T, H, K, V, dtype=torch.float32, device="cuda")
        for t in (q, k, v, g, b, w):
            t.requires_grad_(True)
        o, _ = gated_delta_net_v2(q, k, v, g, b, w)
        o.sum().backward()
        assert q.grad is not None and torch.isfinite(q.grad).all()
