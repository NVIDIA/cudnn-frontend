# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GDN custom-op tests (``cudnn.linear_attention.gated_delta_net``): THD
layout, autograd, and torch.compile against the fp64 recurrent reference."""

from __future__ import annotations

import importlib.util
import math

import pytest
import torch

from cudnn.linear_attention.ops import gated_delta_net

from ..common import FWD_TOL, GDN_MARKS, STATE_TOL
from ..conftest import gen_gates, gen_qkv
from ..reference_gdn import gdn_reference, rms_ratio

pytestmark = GDN_MARKS


def _frost_gdn_available() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability()
    if major != 10 or importlib.util.find_spec("cutlass") is None:
        return False
    try:
        import cutlass.experimental.primitives  # noqa: F401 — the engine's own availability gate
    except Exception:  # noqa: BLE001
        return False
    return True


requires_frost = pytest.mark.skipif(not _frost_gdn_available(), reason="GDN with HV < HQ (GQA-style v broadcast) is served by the FROST engine only")


def _inputs(B, T, H, HV, K, V, dtype=torch.bfloat16, seed=0):
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    q, k, v = gen_qkv(B, T, H, HV, K, V, dtype)
    g, beta = gen_gates(B, T, HV, torch.float32)  # the op requires kernel-native fp32 gates
    return q, k, v, g, beta


def _thd(x):
    return x.reshape(-1, *x.shape[2:])


def _cu(B, T):
    return torch.arange(0, B + 1, dtype=torch.int32, device="cuda") * T


class TestGdnOp:
    @pytest.mark.parametrize(
        "B,T,H,K,V",
        [
            (1, 128, 1, 64, 64),
            (2, 256, 4, 64, 64),
            (1, 128, 2, 64, 128),  # K != V
        ],
    )
    def test_forward_parity(self, B, T, H, K, V):
        q, k, v, g, beta = _inputs(B, T, H, H, K, V)
        scale = 1.0 / math.sqrt(K)
        o, _ = gated_delta_net(_thd(q), _thd(k), _thd(v), _thd(g), _thd(beta), _cu(B, T), scale=scale)
        with torch.no_grad():
            o_ref, _ = gdn_reference(q, k, v, g, beta, scale=scale)
        assert rms_ratio(o.view_as(o_ref), o_ref) < FWD_TOL[torch.bfloat16]

    def test_forward_parity_gva(self, B=1, T=192, H=2, HV=4, K=64, V=64):
        q, k, v, g, beta = _inputs(B, T, H, HV, K, V)
        o, fs = gated_delta_net(_thd(q), _thd(k), _thd(v), _thd(g), _thd(beta), _cu(B, T), output_final_state=True)
        with torch.no_grad():
            o_ref, fs_ref = gdn_reference(q, k, v, g, beta)
        assert rms_ratio(o.view_as(o_ref), o_ref) < FWD_TOL[torch.bfloat16]
        assert rms_ratio(fs, fs_ref) < STATE_TOL[torch.bfloat16]

    @pytest.mark.parametrize("B,T,H,K,V", [(1, 128, 1, 64, 64), (2, 128, 2, 64, 64)])
    def test_backward_runs(self, B, T, H, K, V):
        q, k, v, g, beta = (_thd(x).detach().requires_grad_(True) for x in _inputs(B, T, H, H, K, V))
        o, _ = gated_delta_net(q, k, v, g, beta, _cu(B, T))
        o.sum().backward()
        for name, t in [("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta)]:
            assert t.grad is not None, f"no grad for {name}"
            assert torch.isfinite(t.grad).all(), f"non-finite grad for {name}"

    def test_default_scale(self):
        q, k, v, g, beta = (_thd(x) for x in _inputs(1, 128, 1, 1, 64, 64))
        cu = _cu(1, 128)
        o_default, _ = gated_delta_net(q, k, v, g, beta, cu)
        o_explicit, _ = gated_delta_net(q, k, v, g, beta, cu, scale=1.0 / math.sqrt(64))
        torch.testing.assert_close(o_default, o_explicit)

    def test_no_final_state_returns_empty(self):
        q, k, v, g, beta = (_thd(x) for x in _inputs(1, 128, 1, 1, 64, 64))
        _o, final = gated_delta_net(q, k, v, g, beta, _cu(1, 128), output_final_state=False)
        assert final.numel() == 0

    def test_initial_state(self):
        B, T, H, K, V = 1, 128, 2, 64, 64
        q, k, v, g, beta = _inputs(B, T, H, H, K, V)
        S0 = torch.randn(B, H, K, V, dtype=torch.float32, device="cuda") * 0.05
        o, fs = gated_delta_net(_thd(q), _thd(k), _thd(v), _thd(g), _thd(beta), _cu(B, T), initial_state=S0, output_final_state=True)
        with torch.no_grad():
            o_ref, fs_ref = gdn_reference(q, k, v, g, beta, initial_state=S0)
        assert rms_ratio(o.view_as(o_ref), o_ref) < FWD_TOL[torch.bfloat16]
        assert rms_ratio(fs, fs_ref) < STATE_TOL[torch.bfloat16]

    def test_packed_matches_per_sequence(self):
        B, T, H, K, V = 2, 128, 2, 64, 64
        q, k, v, g, beta = _inputs(B, T, H, H, K, V)
        o, fs = gated_delta_net(_thd(q), _thd(k), _thd(v), _thd(g), _thd(beta), _cu(B, T), output_final_state=True)
        for b in range(B):
            o_b, fs_b = gated_delta_net(q[b], k[b], v[b], g[b], beta[b], _cu(1, T), output_final_state=True)
            torch.testing.assert_close(o[b * T : (b + 1) * T], o_b)
            torch.testing.assert_close(fs[b], fs_b[0])

    def test_thd_ragged_parity_and_backward(self):
        seq_lens = [64, 192]
        H, K, V = 2, 64, 64
        total = sum(seq_lens)
        q, k, v, g, beta = (x.squeeze(0).detach().requires_grad_(True) for x in _inputs(1, total, H, H, K, V))
        cu = torch.tensor([0, 64, 256], dtype=torch.int32, device="cuda")

        o, _ = gated_delta_net(q, k, v, g, beta, cu_seqlens=cu)
        with torch.no_grad():
            o_ref, _ = gdn_reference(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), g.unsqueeze(0), beta.unsqueeze(0), cu_seqlens=cu)
        assert rms_ratio(o, o_ref.squeeze(0)) < FWD_TOL[torch.bfloat16]

        o.sum().backward()
        for name, t in [("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta)]:
            assert t.grad is not None and torch.isfinite(t.grad).all(), f"bad grad for {name}"

    @requires_frost
    def test_forward_parity_gqa(self, B=1, T=192, H=4, HV=1, K=128, V=128):
        """GQA (q heads group over v heads): gates/o/state at HO = H; the
        reference expands v over the head group (each output head runs its
        own q/k against the shared v)."""
        torch.random.manual_seed(0)
        torch.cuda.manual_seed(0)
        q, k, v = gen_qkv(B, T, H, HV, K, V, torch.bfloat16)
        g, beta = gen_gates(B, T, H, torch.float32)  # the op requires kernel-native fp32 gates
        o, fs = gated_delta_net(_thd(q), _thd(k), _thd(v), _thd(g), _thd(beta), _cu(B, T), output_final_state=True)
        v_exp = v.repeat_interleave(H // HV, dim=2)
        with torch.no_grad():
            o_ref, fs_ref = gdn_reference(q, k, v_exp, g, beta)
        assert rms_ratio(o.view_as(o_ref), o_ref) < FWD_TOL[torch.bfloat16]
        assert rms_ratio(fs, fs_ref) < STATE_TOL[torch.bfloat16]

    @requires_frost
    def test_backward_parity_gqa(self, B=1, T=128, H=4, HV=2, K=128, V=128):
        """GQA gradients vs the op's own (validated) MHA path on the expanded
        v: dQ/dK/dG/dBeta match directly, dV matches the head-group sum."""
        torch.random.manual_seed(0)
        torch.cuda.manual_seed(0)
        q, k, v = gen_qkv(B, T, H, HV, K, V, torch.bfloat16)
        g, beta = gen_gates(B, T, H, torch.float32)  # the op requires kernel-native fp32 gates
        r = H // HV
        cu = _cu(B, T)

        gqa = {n: _thd(x).detach().requires_grad_(True) for n, x in (("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta))}
        o, _ = gated_delta_net(gqa["q"], gqa["k"], gqa["v"], gqa["g"], gqa["beta"], cu)
        w = torch.randn_like(o.float())
        (o.float() * w).sum().backward()

        mha = {n: _thd(x).detach().requires_grad_(True) for n, x in (("q", q), ("k", k), ("v", v.repeat_interleave(r, dim=2)), ("g", g), ("beta", beta))}
        o_ref, _ = gated_delta_net(mha["q"], mha["k"], mha["v"], mha["g"], mha["beta"], cu)
        (o_ref.float() * w).sum().backward()

        torch.testing.assert_close(o, o_ref, atol=1e-3, rtol=1e-3)
        for name in ("q", "k", "g", "beta"):
            torch.testing.assert_close(gqa[name].grad, mha[name].grad, atol=1e-3, rtol=1e-3, msg=f"d{name} mismatch")
        dv_ref = mha["v"].grad.view(B * T, HV, r, V).sum(2)
        torch.testing.assert_close(gqa["v"].grad.float(), dv_ref.float(), atol=1e-2, rtol=1e-2, msg="dV mismatch")

    def test_forward_parity_qk_l2norm(self, B=1, T=192, H=2, K=64, V=64):
        import torch.nn.functional as F

        q, k, v, g, beta = _inputs(B, T, H, H, K, V)
        scale = 1.0 / math.sqrt(K)
        o, fs = gated_delta_net(_thd(q), _thd(k), _thd(v), _thd(g), _thd(beta), _cu(B, T), scale=scale, output_final_state=True, use_qk_l2norm_in_kernel=True)
        with torch.no_grad():
            o_ref, fs_ref = gdn_reference(F.normalize(q.float(), dim=-1), F.normalize(k.float(), dim=-1), v, g, beta, scale=scale)
        assert rms_ratio(o.view_as(o_ref), o_ref) < FWD_TOL[torch.bfloat16]
        assert rms_ratio(fs, fs_ref) < STATE_TOL[torch.bfloat16]

    @pytest.mark.parametrize("H,HV", [(2, 2), (2, 4)])
    def test_backward_runs_qk_l2norm(self, H, HV, B=1, T=128, K=64, V=64):
        q, k, v, g, beta = (_thd(x).detach().requires_grad_(True) for x in _inputs(B, T, H, HV, K, V))
        o, _ = gated_delta_net(q, k, v, g, beta, _cu(B, T), use_qk_l2norm_in_kernel=True)
        o.sum().backward()
        for name, t in [("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta)]:
            assert t.grad is not None, f"no grad for {name}"
            assert torch.isfinite(t.grad).all(), f"non-finite grad for {name}"

    def test_torch_compile_forward(self):
        q, k, v, g, beta = (_thd(x) for x in _inputs(1, 128, 1, 1, 64, 64))
        cu = _cu(1, 128)
        compiled = torch.compile(gated_delta_net, fullgraph=True)
        o_eager, _ = gated_delta_net(q, k, v, g, beta, cu)
        o_comp, _ = compiled(q, k, v, g, beta, cu)
        torch.testing.assert_close(o_eager, o_comp)
