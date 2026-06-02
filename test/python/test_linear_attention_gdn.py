"""
Tests for the GDN PyTorch custom operator
(``cudnn.experimental.ops.gated_delta_net``).

The op is a pure-PyTorch reference; tests run on CPU in fp32/fp64 and
verify (1) the chunked forward against a per-token recurrent baseline
and (2) the registered autograd backward against autograd through the
recurrent reference. CUDA execution is exercised in a small smoke test
that is skipped when no GPU is present.
"""

from __future__ import annotations

import math

import pytest
import torch

import cudnn  # noqa: F401  (ensures ``cudnn.experimental.ops`` is registered)
from cudnn.experimental.ops import gated_delta_net
from cudnn.experimental.ops.linear_attention.gdn import (
    _gdn_forward_recurrent,
    _gdn_forward_chunked,
)
from cudnn.experimental.ops.linear_attention._common import (
    bthd_to_bhtd,
    bhtd_to_bthd,
    bth_to_bht,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rand_inputs(B, T, H, K, V, dtype=torch.float64, device="cpu", seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    q = (torch.rand(B, T, H, K, dtype=dtype, device=device, generator=g) + 0.1) * 0.5
    k = (torch.rand(B, T, H, K, dtype=dtype, device=device, generator=g) + 0.1) * 0.5
    v = torch.randn(B, T, H, V, dtype=dtype, device=device, generator=g)
    # alpha in (0.85, 1.0) -> g = log(alpha) is small negative.
    alpha = 0.85 + 0.15 * torch.rand(B, T, H, dtype=dtype, device=device, generator=g)
    g_decay = torch.log(alpha)
    beta = torch.sigmoid(torch.randn(B, T, H, dtype=dtype, device=device, generator=g)) * (0.8 / K)
    return q, k, v, g_decay, beta


def _reference_recurrent(q, k, v, g_decay, beta, scale):
    """[B,T,H,*] -> chunked-style reference output (fp64)."""
    qf = bthd_to_bhtd(q).to(torch.float64) * scale
    kf = bthd_to_bhtd(k).to(torch.float64)
    vf = bthd_to_bhtd(v).to(torch.float64)
    alphaf = bth_to_bht(g_decay).exp().to(torch.float64)
    betaf = bth_to_bht(beta).to(torch.float64)
    S0 = torch.zeros(q.shape[0], q.shape[2], q.shape[3], v.shape[-1], dtype=torch.float64, device=q.device)
    out, final = _gdn_forward_recurrent(qf, kf, vf, alphaf, betaf, S0)
    return bhtd_to_bthd(out), final


# ---------------------------------------------------------------------------
# Forward parity (chunked op vs. per-token recurrent)
# ---------------------------------------------------------------------------


class TestGdnForward:
    @pytest.mark.parametrize("B,T,H,K,V,chunk_size", [
        (1, 8, 1, 16, 24, 4),
        (2, 64, 4, 32, 32, 8),
        (1, 10, 2, 8, 8, 4),       # T not a multiple of chunk_size
        (1, 32, 1, 16, 16, 32),    # single chunk
        (2, 100, 2, 8, 8, 8),
    ])
    def test_forward_matches_recurrent(self, B, T, H, K, V, chunk_size):
        q, k, v, g, beta = _rand_inputs(B, T, H, K, V, dtype=torch.float64)
        scale = 1.0 / math.sqrt(K)
        o, final = gated_delta_net(q, k, v, g, beta, scale=scale,
                                   chunk_size=chunk_size, output_final_state=True)
        o_ref, final_ref = _reference_recurrent(q, k, v, g, beta, scale)
        torch.testing.assert_close(o.double(), o_ref, atol=1e-9, rtol=1e-9)
        torch.testing.assert_close(final.double(), final_ref, atol=1e-9, rtol=1e-9)

    def test_default_scale(self):
        B, T, H, K, V = 1, 16, 1, 8, 8
        q, k, v, g, beta = _rand_inputs(B, T, H, K, V, dtype=torch.float64)
        o_default, _ = gated_delta_net(q, k, v, g, beta)
        o_explicit, _ = gated_delta_net(q, k, v, g, beta, scale=1.0 / math.sqrt(K))
        torch.testing.assert_close(o_default, o_explicit)

    def test_no_final_state_returns_empty(self):
        B, T, H, K, V = 1, 8, 1, 8, 8
        q, k, v, g, beta = _rand_inputs(B, T, H, K, V, dtype=torch.float64)
        _o, final = gated_delta_net(q, k, v, g, beta, output_final_state=False)
        assert final.numel() == 0

    def test_initial_state(self):
        B, T, H, K, V = 1, 16, 1, 8, 8
        q, k, v, g, beta = _rand_inputs(B, T, H, K, V, dtype=torch.float64)
        S0 = torch.randn(B, H, K, V, dtype=torch.float64) * 0.01

        o, final = gated_delta_net(q, k, v, g, beta, initial_state=S0,
                                   output_final_state=True, chunk_size=4)

        # Recurrent reference with the same initial state.
        qf = bthd_to_bhtd(q).to(torch.float64) * (1.0 / math.sqrt(K))
        kf = bthd_to_bhtd(k).to(torch.float64)
        vf = bthd_to_bhtd(v).to(torch.float64)
        alphaf = bth_to_bht(g).exp().to(torch.float64)
        betaf = bth_to_bht(beta).to(torch.float64)
        out_ref, final_ref = _gdn_forward_recurrent(qf, kf, vf, alphaf, betaf, S0.clone())
        torch.testing.assert_close(o.double(), bhtd_to_bthd(out_ref), atol=1e-9, rtol=1e-9)
        torch.testing.assert_close(final.double(), final_ref, atol=1e-9, rtol=1e-9)


# ---------------------------------------------------------------------------
# Backward parity vs. autograd-through-recurrent
# ---------------------------------------------------------------------------


class TestGdnBackward:
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

        o, _ = gated_delta_net(q, k, v, g, beta, scale=scale, chunk_size=chunk_size)
        dO = torch.randn_like(o)
        o.backward(dO)
        dq_op, dk_op, dv_op = q.grad.clone(), k.grad.clone(), v.grad.clone()
        dg_op, dbeta_op = g.grad.clone(), beta.grad.clone()

        # Reference: autograd through the per-token recurrent form.
        qr = q.detach().clone().requires_grad_(True)
        kr = k.detach().clone().requires_grad_(True)
        vr = v.detach().clone().requires_grad_(True)
        gr = g.detach().clone().requires_grad_(True)
        br = beta.detach().clone().requires_grad_(True)
        qf = bthd_to_bhtd(qr) * scale
        kf = bthd_to_bhtd(kr)
        vf = bthd_to_bhtd(vr)
        alphaf = bth_to_bht(gr).exp()
        betaf = bth_to_bht(br)
        S0 = torch.zeros(B, H, K, V, dtype=q.dtype, device=q.device)
        out_ref, _ = _gdn_forward_recurrent(qf, kf, vf, alphaf, betaf, S0)
        out_ref_bthd = bhtd_to_bthd(out_ref)
        out_ref_bthd.backward(dO)

        torch.testing.assert_close(dq_op, qr.grad, atol=1e-8, rtol=1e-8)
        torch.testing.assert_close(dk_op, kr.grad, atol=1e-8, rtol=1e-8)
        torch.testing.assert_close(dv_op, vr.grad, atol=1e-8, rtol=1e-8)
        torch.testing.assert_close(dg_op, gr.grad, atol=1e-8, rtol=1e-8)
        torch.testing.assert_close(dbeta_op, br.grad, atol=1e-8, rtol=1e-8)


# ---------------------------------------------------------------------------
# torch.compile + CUDA smoke
# ---------------------------------------------------------------------------


class TestGdnCompileAndCuda:
    def test_torch_compile_forward(self):
        B, T, H, K, V = 1, 16, 1, 8, 8
        q, k, v, g, beta = _rand_inputs(B, T, H, K, V, dtype=torch.float32)
        compiled = torch.compile(gated_delta_net, fullgraph=True)
        o_eager, _ = gated_delta_net(q, k, v, g, beta)
        o_comp, _ = compiled(q, k, v, g, beta)
        torch.testing.assert_close(o_eager, o_comp)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_cuda_smoke(self):
        B, T, H, K, V = 1, 16, 1, 8, 8
        q, k, v, g, beta = _rand_inputs(B, T, H, K, V, dtype=torch.float32, device="cuda")
        q.requires_grad_(True); k.requires_grad_(True); v.requires_grad_(True)
        o, _ = gated_delta_net(q, k, v, g, beta)
        o.sum().backward()
        assert q.grad is not None and torch.isfinite(q.grad).all()
