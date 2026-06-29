"""
Tests for the GDN custom operator (``cudnn.experimental.ops.gated_delta_net``).

The op dispatches to the vendored cuTile chunked kernel; tests run on CUDA and
validate forward/backward against a per-token recurrent reference (computed in
fp64 on CPU). Skipped when no GPU or the ``cuda.tile`` runtime is unavailable.
"""

from __future__ import annotations

import math

import pytest
import torch

import cudnn  # noqa: F401  (registers ``cudnn.experimental.ops``)
from cudnn.experimental.ops import gated_delta_net
from cudnn.experimental.ops.linear_attention._common import (
    bthd_to_bhtd,
    bhtd_to_bthd,
    bth_to_bht,
)

_HAS_CUDA = torch.cuda.is_available()

try:
    from cudnn.experimental.ops.linear_attention._gdn_chunk_cutile import (  # noqa: F401
        chunk_gated_delta_rule,
    )
    _HAS_CUTILE = True
except ImportError:
    _HAS_CUTILE = False

pytestmark = [
    pytest.mark.L0,
    pytest.mark.skipif(not _HAS_CUDA, reason="needs CUDA"),
    pytest.mark.skipif(not _HAS_CUTILE, reason="needs the cuda.tile runtime"),
]


# ---------------------------------------------------------------------------
# Reference recurrence (fp64 oracle, ``[B, H, T, *]`` layout)
# ---------------------------------------------------------------------------


def _gdn_forward_recurrent(q, k, v, alpha, beta, initial_state):
    T = q.shape[2]
    S = initial_state
    out_steps = []
    for t in range(T):
        kt = k[:, :, t, :]
        vt = v[:, :, t, :]
        at = alpha[:, :, t][..., None, None]
        at_s = alpha[:, :, t][..., None]
        bt = beta[:, :, t][..., None, None]
        kt_S = (kt.unsqueeze(-2) @ S).squeeze(-2)
        residual = vt - at_s * kt_S
        S = at * S + bt * (kt.unsqueeze(-1) @ residual.unsqueeze(-2))
        qt = q[:, :, t, :].unsqueeze(-2)
        out_steps.append((qt @ S).squeeze(-2))
    return torch.stack(out_steps, dim=2), S


def _rand_inputs(B, T, H, K, V, dtype=torch.bfloat16, device="cuda", seed=0):
    # GDN runs in bf16 (q/k/v/g/beta); the tf32 matmul guard in the kernel
    # makes fp32 inputs an unsupported path.
    g = torch.Generator(device=device).manual_seed(seed)
    q = (torch.rand(B, T, H, K, dtype=torch.float32, device=device, generator=g) + 0.1) * 0.5
    k = (torch.rand(B, T, H, K, dtype=torch.float32, device=device, generator=g) + 0.1) * 0.5
    v = torch.randn(B, T, H, V, dtype=torch.float32, device=device, generator=g)
    alpha = 0.85 + 0.15 * torch.rand(B, T, H, dtype=torch.float32, device=device, generator=g)
    g_decay = torch.log(alpha)
    beta = torch.sigmoid(torch.randn(B, T, H, dtype=torch.float32, device=device, generator=g)) * (0.8 / K)
    return (x.to(dtype) for x in (q, k, v, g_decay, beta))


def _rms_ratio(out, ref):
    """RMS(out - ref) / RMS(ref), the bf16-appropriate parity metric ocean uses."""
    out = out.cpu().double()
    ref = ref.cpu().double()
    return (out - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt().clamp_min(1e-12)


def _reference_recurrent(q, k, v, g_decay, beta, scale):
    """fp64 CPU oracle; inputs in ``[B, T, H, *]`` on any device."""
    qf = bthd_to_bhtd(q).cpu().double() * scale
    kf = bthd_to_bhtd(k).cpu().double()
    vf = bthd_to_bhtd(v).cpu().double()
    alphaf = bth_to_bht(g_decay).cpu().double().exp()
    betaf = bth_to_bht(beta).cpu().double()
    S0 = torch.zeros(q.shape[0], q.shape[2], q.shape[3], v.shape[-1], dtype=torch.float64)
    out, final = _gdn_forward_recurrent(qf, kf, vf, alphaf, betaf, S0)
    return bhtd_to_bthd(out), final


# ---------------------------------------------------------------------------
# Forward / backward parity + smoke tests
# ---------------------------------------------------------------------------


class TestGdn:
    @pytest.mark.parametrize("B,T,H,K,V", [
        (1, 128, 1, 64, 64),
        (2, 256, 4, 64, 64),
        (1, 128, 2, 64, 128),  # K != V
    ])
    def test_forward_parity_vs_reference(self, B, T, H, K, V):
        q, k, v, g, beta = _rand_inputs(B, T, H, K, V)
        scale = 1.0 / math.sqrt(K)
        o, _ = gated_delta_net(q, k, v, g, beta, scale=scale, output_final_state=False)
        o_ref, _ = _reference_recurrent(q, k, v, g, beta, scale)
        assert _rms_ratio(o, o_ref) < 2e-2

    @pytest.mark.parametrize("B,T,H,K,V", [
        (1, 128, 1, 64, 64),
        (2, 128, 2, 64, 64),
    ])
    def test_backward_runs(self, B, T, H, K, V):
        q, k, v, g, beta = _rand_inputs(B, T, H, K, V)
        q.requires_grad_(True); k.requires_grad_(True); v.requires_grad_(True)
        g.requires_grad_(True); beta.requires_grad_(True)
        o, _ = gated_delta_net(q, k, v, g, beta)
        o.sum().backward()
        for name, t in [("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta)]:
            assert t.grad is not None, f"no grad for {name}"
            assert torch.isfinite(t.grad).all(), f"non-finite grad for {name}"

    def test_default_scale(self):
        q, k, v, g, beta = _rand_inputs(1, 128, 1, 64, 64)
        o_default, _ = gated_delta_net(q, k, v, g, beta)
        o_explicit, _ = gated_delta_net(q, k, v, g, beta, scale=1.0 / math.sqrt(64))
        torch.testing.assert_close(o_default, o_explicit)

    def test_no_final_state_returns_empty(self):
        q, k, v, g, beta = _rand_inputs(1, 128, 1, 64, 64)
        _o, final = gated_delta_net(q, k, v, g, beta, output_final_state=False)
        assert final.numel() == 0

    def test_initial_state(self):
        B, T, H, K, V = 1, 128, 1, 64, 64
        q, k, v, g, beta = _rand_inputs(B, T, H, K, V)
        S0 = torch.randn(B, H, K, V, dtype=torch.float32, device="cuda") * 0.01
        o, final = gated_delta_net(q, k, v, g, beta, initial_state=S0, output_final_state=True)
        assert o.shape == (B, T, H, V)
        assert final.shape == (B, H, K, V)
        assert torch.isfinite(o).all() and torch.isfinite(final).all()

    def test_torch_compile_forward(self):
        q, k, v, g, beta = _rand_inputs(1, 128, 1, 64, 64)
        compiled = torch.compile(gated_delta_net, fullgraph=True)
        o_eager, _ = gated_delta_net(q, k, v, g, beta)
        o_comp, _ = compiled(q, k, v, g, beta)
        torch.testing.assert_close(o_eager, o_comp)
