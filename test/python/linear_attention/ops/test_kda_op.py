# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KDA custom-op tests (``cudnn.linear_attention.ops.kimi_delta_attention``):
THD layout, autograd, and torch.compile against the fp64 recurrent
reference."""

from __future__ import annotations

import importlib.util
import math

import pytest
import torch

from cudnn.linear_attention.ops import kimi_delta_attention

from ..common import FWD_TOL, KDA_MARKS, STATE_TOL
from ..conftest import gen_kda_gates, gen_qkv
from ..reference_kda import kda_reference, rms_ratio

pytestmark = KDA_MARKS


def _frost_kda_available() -> bool:
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


def _inputs(B, T, H, HV, K, V, dtype=torch.bfloat16, seed=0):
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    q, k, v = gen_qkv(B, T, H, HV, K, V, dtype)
    g, beta = gen_kda_gates(B, T, HV, K, torch.float32)  # the op requires kernel-native fp32 gates
    return q, k, v, g, beta


def _thd(x):
    return x.reshape(-1, *x.shape[2:])


def _cu(B, T):
    return torch.arange(0, B + 1, dtype=torch.int32, device="cuda") * T


class TestKdaOp:
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
        o, _ = kimi_delta_attention(_thd(q), _thd(k), _thd(v), _thd(g), _thd(beta), _cu(B, T), scale=scale)
        with torch.no_grad():
            o_ref, _ = kda_reference(q, k, v, g, beta, scale=scale)
        assert rms_ratio(o.view_as(o_ref), o_ref) < FWD_TOL[torch.bfloat16]

    def test_forward_parity_gva(self, B=1, T=192, H=2, HV=4, K=64, V=64):
        q, k, v, g, beta = _inputs(B, T, H, HV, K, V)
        o, fs = kimi_delta_attention(_thd(q), _thd(k), _thd(v), _thd(g), _thd(beta), _cu(B, T), output_final_state=True)
        with torch.no_grad():
            o_ref, fs_ref = kda_reference(q, k, v, g, beta)
        assert rms_ratio(o.view_as(o_ref), o_ref) < FWD_TOL[torch.bfloat16]
        assert rms_ratio(fs, fs_ref) < STATE_TOL[torch.bfloat16]

    @pytest.mark.parametrize("B,T,H,K,V", [(1, 128, 1, 64, 64), (2, 128, 2, 64, 64)])
    def test_backward_runs(self, B, T, H, K, V):
        q, k, v, g, beta = (_thd(x).detach().requires_grad_(True) for x in _inputs(B, T, H, H, K, V))
        o, _ = kimi_delta_attention(q, k, v, g, beta, _cu(B, T))
        o.sum().backward()
        for name, t in [("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta)]:
            assert t.grad is not None, f"no grad for {name}"
            assert torch.isfinite(t.grad).all(), f"non-finite grad for {name}"

    def test_default_scale(self):
        q, k, v, g, beta = (_thd(x) for x in _inputs(1, 128, 1, 1, 64, 64))
        cu = _cu(1, 128)
        o_default, _ = kimi_delta_attention(q, k, v, g, beta, cu)
        o_explicit, _ = kimi_delta_attention(q, k, v, g, beta, cu, scale=1.0 / math.sqrt(64))
        torch.testing.assert_close(o_default, o_explicit)

    def test_no_final_state_returns_empty(self):
        q, k, v, g, beta = (_thd(x) for x in _inputs(1, 128, 1, 1, 64, 64))
        _o, final = kimi_delta_attention(q, k, v, g, beta, _cu(1, 128), output_final_state=False)
        assert final.numel() == 0

    def test_initial_state(self):
        B, T, H, K, V = 1, 128, 2, 64, 64
        q, k, v, g, beta = _inputs(B, T, H, H, K, V)
        S0 = torch.randn(B, H, K, V, dtype=torch.float32, device="cuda") * 0.05
        o, fs = kimi_delta_attention(_thd(q), _thd(k), _thd(v), _thd(g), _thd(beta), _cu(B, T), initial_state=S0, output_final_state=True)
        with torch.no_grad():
            o_ref, fs_ref = kda_reference(q, k, v, g, beta, initial_state=S0)
        assert rms_ratio(o.view_as(o_ref), o_ref) < FWD_TOL[torch.bfloat16]
        assert rms_ratio(fs, fs_ref) < STATE_TOL[torch.bfloat16]

    def test_packed_matches_per_sequence(self):
        B, T, H, K, V = 2, 128, 2, 64, 64
        q, k, v, g, beta = _inputs(B, T, H, H, K, V)
        o, fs = kimi_delta_attention(_thd(q), _thd(k), _thd(v), _thd(g), _thd(beta), _cu(B, T), output_final_state=True)
        for b in range(B):
            o_b, fs_b = kimi_delta_attention(q[b], k[b], v[b], g[b], beta[b], _cu(1, T), output_final_state=True)
            torch.testing.assert_close(o[b * T : (b + 1) * T], o_b)
            torch.testing.assert_close(fs[b], fs_b[0])

    def test_thd_ragged_parity_and_backward(self):
        seq_lens = [64, 192]
        H, K, V = 2, 64, 64
        total = sum(seq_lens)
        q, k, v, g, beta = (x.squeeze(0).detach().requires_grad_(True) for x in _inputs(1, total, H, H, K, V))
        cu = torch.tensor([0, 64, 256], dtype=torch.int32, device="cuda")

        o, _ = kimi_delta_attention(q, k, v, g, beta, cu_seqlens=cu)
        with torch.no_grad():
            o_ref, _ = kda_reference(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), g.unsqueeze(0), beta.unsqueeze(0), cu_seqlens=cu)
        assert rms_ratio(o, o_ref.squeeze(0)) < FWD_TOL[torch.bfloat16]

        o.sum().backward()
        for name, t in [("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta)]:
            assert t.grad is not None and torch.isfinite(t.grad).all(), f"bad grad for {name}"

    def test_torch_compile_forward(self):
        q, k, v, g, beta = (_thd(x) for x in _inputs(1, 128, 1, 1, 64, 64))
        cu = _cu(1, 128)
        compiled = torch.compile(kimi_delta_attention, fullgraph=True)
        o_eager, _ = kimi_delta_attention(q, k, v, g, beta, cu)
        o_comp, _ = compiled(q, k, v, g, beta, cu)
        torch.testing.assert_close(o_eager, o_comp)

    def test_forward_parity_qk_l2norm(self, B=1, T=256, H=2, K=128, V=128):
        """D=128 + use_qk_l2norm: routes to KdaFrostEngine on SM100/SM103 (the
        cuTile engine serves it elsewhere — both honor the in-kernel norm)."""
        import torch.nn.functional as F

        q, k, v, g, beta = _inputs(B, T, H, H, K, V)
        # the FROST BT=16 kernel's Neumann inverse wants stronger decay and a
        # post-sigmoid beta (see its kernel suite)
        g = torch.empty(B, T, H, K, device="cuda").uniform_(0.5, 1.0).log()
        beta = beta.float().sigmoid().to(beta.dtype)
        scale = 1.0 / math.sqrt(K)
        o, fs = kimi_delta_attention(
            _thd(q), _thd(k), _thd(v), _thd(g), _thd(beta), _cu(B, T), scale=scale, output_final_state=True, use_qk_l2norm_in_kernel=True
        )
        with torch.no_grad():
            o_ref, fs_ref = kda_reference(F.normalize(q.float(), dim=-1), F.normalize(k.float(), dim=-1), v, g, beta, scale=scale)
        torch.testing.assert_close(o.view_as(o_ref).float(), o_ref.float(), atol=1e-1, rtol=1e-1)
        assert rms_ratio(fs, fs_ref) < 5e-2

    def test_forward_parity_no_l2norm_frost(self, B=1, T=256, H=2, K=128, V=128):
        """D=128 without the in-kernel norm routes to KdaFrostEngine on
        SM100/SM103, so q/k are pre-normalized here."""
        import torch.nn.functional as F

        q, k, v, g, beta = _inputs(B, T, H, H, K, V)
        g = torch.empty(B, T, H, K, device="cuda").uniform_(0.5, 1.0).log()
        beta = beta.float().sigmoid().to(beta.dtype)
        q = F.normalize(q.float(), dim=-1).to(q.dtype)
        k = F.normalize(k.float(), dim=-1).to(k.dtype)
        scale = 1.0 / math.sqrt(K)
        o, fs = kimi_delta_attention(_thd(q), _thd(k), _thd(v), _thd(g), _thd(beta), _cu(B, T), scale=scale, output_final_state=True)
        with torch.no_grad():
            o_ref, fs_ref = kda_reference(q.float(), k.float(), v, g, beta, scale=scale)
        torch.testing.assert_close(o.view_as(o_ref).float(), o_ref.float(), atol=1e-1, rtol=1e-1)
        assert rms_ratio(fs, fs_ref) < 5e-2

    def test_backward_runs_qk_l2norm(self, B=1, T=128, H=2, K=128, V=128):
        """D=128 + use_qk_l2norm: forward may run on KdaFrostEngine, backward
        always falls back to the cuTile engine (FROST KDA_BWD is a stub)."""
        q, k, v, g, beta = (_thd(x).detach().requires_grad_(True) for x in _inputs(B, T, H, H, K, V))
        o, _ = kimi_delta_attention(q, k, v, g, beta, _cu(B, T), use_qk_l2norm_in_kernel=True)
        o.sum().backward()
        for name, t in [("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta)]:
            assert t.grad is not None, f"no grad for {name}"
            assert torch.isfinite(t.grad).all(), f"non-finite grad for {name}"

    def test_thd_ragged_parity_frost(self):
        """D=128 ragged varlen: the FROST path on SM100/SM103 (cuTile elsewhere)."""
        import torch.nn.functional as F

        seq_lens = [64, 192]
        total, H, K, V = sum(seq_lens), 2, 128, 128
        q, k, v, g, beta = _inputs(1, total, H, H, K, V)
        g = torch.empty(1, total, H, K, device="cuda").uniform_(0.5, 1.0).log()
        beta = beta.float().sigmoid().to(beta.dtype)
        cu = torch.tensor([0, 64, 256], dtype=torch.int32, device="cuda")
        o, fs = kimi_delta_attention(_thd(q), _thd(k), _thd(v), _thd(g), _thd(beta), cu, output_final_state=True, use_qk_l2norm_in_kernel=True)
        with torch.no_grad():
            o_ref, fs_ref = kda_reference(F.normalize(q.float(), dim=-1), F.normalize(k.float(), dim=-1), v, g, beta, cu_seqlens=cu)
        torch.testing.assert_close(o.view_as(o_ref.squeeze(0)).float(), o_ref.squeeze(0).float(), atol=1e-1, rtol=1e-1)
        assert rms_ratio(fs, fs_ref) < 5e-2

    def test_initial_state_frost(self, B=1, T=256, H=2, K=128, V=128):
        """D=128 + initial state on the FROST path."""
        import torch.nn.functional as F

        q, k, v, g, beta = _inputs(B, T, H, H, K, V)
        g = torch.empty(B, T, H, K, device="cuda").uniform_(0.5, 1.0).log()
        beta = beta.float().sigmoid().to(beta.dtype)
        S0 = torch.randn(B, H, K, V, dtype=torch.float32, device="cuda") * 0.05
        o, fs = kimi_delta_attention(
            _thd(q), _thd(k), _thd(v), _thd(g), _thd(beta), _cu(B, T), initial_state=S0, output_final_state=True, use_qk_l2norm_in_kernel=True
        )
        with torch.no_grad():
            o_ref, fs_ref = kda_reference(F.normalize(q.float(), dim=-1), F.normalize(k.float(), dim=-1), v, g, beta, initial_state=S0)
        torch.testing.assert_close(o.view_as(o_ref).float(), o_ref.float(), atol=1e-1, rtol=1e-1)
        assert rms_ratio(fs, fs_ref) < 5e-2

    def test_frost_engine_selected(self):
        """On SM100/SM103 with the DSL runtime, D=128 + l2norm graphs must
        actually lower to KdaFrostEngine (guards the router ranking)."""
        if not _frost_kda_available():
            pytest.skip("needs an SM100-class GPU and the Cutlass DSL KDA prefill kernel runtime")
        from cudnn.linear_attention.frost.kda_engine import KdaFrostEngine
        from cudnn.linear_attention.ops import kda as kda_ops

        kda_ops._fwd_graph_cache.clear()
        q, k, v, g, beta = _inputs(1, 128, 2, 2, 128, 128)
        g = torch.empty(1, 128, 2, 128, device="cuda").uniform_(0.5, 1.0).log()
        beta = beta.float().sigmoid().to(beta.dtype)
        kimi_delta_attention(_thd(q), _thd(k), _thd(v), _thd(g), _thd(beta), _cu(1, 128), use_qk_l2norm_in_kernel=True)
        assert any(isinstance(graph.selected_engine, KdaFrostEngine) for graph, _t in kda_ops._fwd_graph_cache.values())
