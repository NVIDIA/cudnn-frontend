# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GDN-2 custom-op tests (``cudnn.linear_attention.ops.gated_delta_net_v2``):
THD layout and torch.compile against the fp64 recurrent reference. GDN-2 runs
on the FROST SM100 engine only (forward), so these tests require an
SM100-class GPU with the Cutlass DSL runtime."""

from __future__ import annotations

import importlib.util
import math

import pytest
import torch
import torch.nn.functional as F

from cudnn.linear_attention.ops import gated_delta_net_v2

from ..conftest import gen_gdn2_gates, gen_qkv
from ..reference_gdn2 import gdn2_reference, rms_ratio

pytestmark = pytest.mark.L0


def _frost_gdn2_available() -> bool:
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


requires_runtime = pytest.mark.skipif(not _frost_gdn2_available(), reason="needs an SM100-class GPU and the Cutlass DSL GDN-2 prefill kernel runtime")


def _inputs(B, T, H, K=128, V=128, dtype=torch.bfloat16, seed=0):
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    q, k, v = gen_qkv(B, T, H, H, K, V, dtype)
    g, beta, w = gen_gdn2_gates(B, T, H, K, V, dtype)
    return q, k, v, g, beta, w


def _thd(x):
    return x.reshape(-1, *x.shape[2:])


def _cu(B, T):
    return torch.arange(0, B + 1, dtype=torch.int32, device="cuda") * T


@requires_runtime
class TestGdn2Op:
    def test_forward_parity(self, B=1, T=256, H=2):
        q, k, v, g, beta, w = _inputs(B, T, H)
        scale = 1.0 / math.sqrt(128)
        o, _ = gated_delta_net_v2(_thd(q), _thd(k), _thd(v), _thd(g), _thd(beta), _thd(w), _cu(B, T), scale=scale, use_qk_l2norm_in_kernel=True)
        with torch.no_grad():
            o_ref, _ = gdn2_reference(F.normalize(q.float(), dim=-1), F.normalize(k.float(), dim=-1), v, g, beta, w, scale=scale)
        torch.testing.assert_close(o.view_as(o_ref).float(), o_ref.float(), atol=1e-1, rtol=1e-1)

    def test_forward_parity_no_l2norm(self, B=1, T=256, H=2):
        q, k, v, g, beta, w = _inputs(B, T, H)
        q = F.normalize(q.float(), dim=-1).to(q.dtype)
        k = F.normalize(k.float(), dim=-1).to(k.dtype)
        scale = 1.0 / math.sqrt(128)
        o, _ = gated_delta_net_v2(_thd(q), _thd(k), _thd(v), _thd(g), _thd(beta), _thd(w), _cu(B, T), scale=scale, use_qk_l2norm_in_kernel=False)
        with torch.no_grad():
            o_ref, _ = gdn2_reference(q.float(), k.float(), v, g, beta, w, scale=scale)
        torch.testing.assert_close(o.view_as(o_ref).float(), o_ref.float(), atol=1e-1, rtol=1e-1)

    def test_forward_parity_thd(self):
        seq_lens = [64, 192]
        total = sum(seq_lens)
        q, k, v, g, beta, w = (x.squeeze(0) for x in _inputs(1, total, 2))
        cu = torch.tensor([0, 64, 256], dtype=torch.int32, device="cuda")
        o, _ = gated_delta_net_v2(q, k, v, g, beta, w, cu, use_qk_l2norm_in_kernel=True)
        with torch.no_grad():
            o_ref, _ = gdn2_reference(
                F.normalize(q.float(), dim=-1).unsqueeze(0),
                F.normalize(k.float(), dim=-1).unsqueeze(0),
                v.unsqueeze(0),
                g.unsqueeze(0),
                beta.unsqueeze(0),
                w.unsqueeze(0),
                cu_seqlens=cu,
            )
        torch.testing.assert_close(o.float(), o_ref.squeeze(0).float(), atol=1e-1, rtol=1e-1)

    def test_initial_state(self, B=1, T=256, H=2):
        q, k, v, g, beta, w = _inputs(B, T, H)
        S0 = torch.randn(B, H, 128, 128, dtype=torch.float32, device="cuda") * 0.05
        scale = 1.0 / math.sqrt(128)
        o, fs = gated_delta_net_v2(
            _thd(q),
            _thd(k),
            _thd(v),
            _thd(g),
            _thd(beta),
            _thd(w),
            _cu(B, T),
            scale=scale,
            initial_state=S0,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        with torch.no_grad():
            o_ref, fs_ref = gdn2_reference(F.normalize(q.float(), dim=-1), F.normalize(k.float(), dim=-1), v, g, beta, w, scale=scale, initial_state=S0)
        torch.testing.assert_close(o.view_as(o_ref).float(), o_ref.float(), atol=1e-1, rtol=1e-1)
        assert rms_ratio(fs, fs_ref) < 5e-2

    def test_default_scale(self):
        q, k, v, g, beta, w = (_thd(x) for x in _inputs(1, 128, 1))
        cu = _cu(1, 128)
        o_default, _ = gated_delta_net_v2(q, k, v, g, beta, w, cu, use_qk_l2norm_in_kernel=True)
        o_explicit, _ = gated_delta_net_v2(q, k, v, g, beta, w, cu, scale=1.0 / math.sqrt(128), use_qk_l2norm_in_kernel=True)
        torch.testing.assert_close(o_default, o_explicit)

    def test_no_final_state_returns_empty(self):
        q, k, v, g, beta, w = (_thd(x) for x in _inputs(1, 128, 1))
        _o, final = gated_delta_net_v2(q, k, v, g, beta, w, _cu(1, 128), output_final_state=False, use_qk_l2norm_in_kernel=True)
        assert final.numel() == 0

    def test_torch_compile_forward(self):
        q, k, v, g, beta, w = (_thd(x) for x in _inputs(1, 128, 1))
        cu = _cu(1, 128)
        compiled = torch.compile(gated_delta_net_v2, fullgraph=True)
        o_eager, _ = gated_delta_net_v2(q, k, v, g, beta, w, cu, use_qk_l2norm_in_kernel=True)
        o_comp, _ = compiled(q, k, v, g, beta, w, cu, use_qk_l2norm_in_kernel=True)
        torch.testing.assert_close(o_eager, o_comp)
