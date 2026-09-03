# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``sdpa_bwd_sm100`` THD / varlen: the packed backward, end to end.

Drives ``SdpaBwdDslSm100`` directly rather than through a graph, because the
engine row does not admit THD yet -- the point here is the numerics of the
three-stage chain over PACKED input and a BLOCKED S/dS workspace, one layer
below the plan machinery.

The reference is per-sequence dense attention: unpack, run each sequence on its
own, and compare gradient by gradient.  A THD bug that leaks across a sequence
boundary shows up as one sequence's gradient contaminated by its neighbour's,
which a whole-tensor cosine would happily average away -- so every assertion is
per sequence.
"""

from __future__ import annotations

import math

import pytest
import torch

from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell

pytestmark = [pytest.mark.L0, requires_pre_rubin_blackwell, requires_dsl]

_D = 512
_TOL_COS = 0.999


def _ref_bwd(q, k, v, do, scale):
    """fp64 reference backward for ONE sequence."""
    q, k, v, do = (t.detach().double().requires_grad_(t is not do) for t in (q, k, v, do))
    s = (q @ k.transpose(-1, -2)) * scale
    p = torch.softmax(s, dim=-1)
    o = p @ v
    o.backward(do)
    return q.grad, k.grad, v.grad


def _cos(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    if a.norm() == 0 and b.norm() == 0:
        return 1.0
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def _run(lens_q, lens_kv, h=2, d=_D, dtype=torch.bfloat16, token_major_stats=False):
    from cudnn.sdpa.bwd.api_dsl import SdpaBwdDslSm100

    dev, b = "cuda", len(lens_q)
    t_q, t_kv = sum(lens_q), sum(lens_kv)
    cu_q, cu_k = [0], [0]
    for a, c in zip(lens_q, lens_kv):
        cu_q.append(cu_q[-1] + a)
        cu_k.append(cu_k[-1] + c)
    scale = 1.0 / math.sqrt(d)
    g = torch.Generator(device=dev).manual_seed(7)
    rnd = lambda t: torch.randn(1, t, h, d, generator=g, device=dev, dtype=dtype) * 0.3

    # Packed [1, T, H, D] storage, handed over as logical [1, H, T, D] views --
    # the same orientation the dense path takes.
    q_p, k_p, v_p, do_p = rnd(t_q), rnd(t_kv), rnd(t_kv), rnd(t_q)
    o_p = torch.empty_like(q_p)
    lse_p = torch.empty(1, h, t_q, device=dev, dtype=torch.float32)

    # Forward reference, per sequence, filling O and the packed LSE.
    for i in range(b):
        qs, ks, vs = (
            x[0, cu[i] : cu[i] + L].transpose(0, 1).double() for x, cu, L in ((q_p, cu_q, lens_q[i]), (k_p, cu_k, lens_kv[i]), (v_p, cu_k, lens_kv[i]))
        )
        s = (qs @ ks.transpose(-1, -2)) * scale
        lse_p[0, :, cu_q[i] : cu_q[i] + lens_q[i]] = torch.logsumexp(s, dim=-1).float()
        o_p[0, cu_q[i] : cu_q[i] + lens_q[i]] = (torch.softmax(s, dim=-1) @ vs).transpose(0, 1).to(dtype)

    view = lambda t: t.permute(0, 2, 1, 3)  # [1,T,H,D] -> logical [1,H,T,D]
    dq, dk, dv = torch.zeros_like(q_p), torch.zeros_like(k_p), torch.zeros_like(v_p)
    stats = lse_p.reshape(t_q, h) if token_major_stats else lse_p

    api = SdpaBwdDslSm100(
        sample_q=view(q_p),
        sample_k=view(k_p),
        sample_v=view(v_p),
        sample_o=view(o_p),
        sample_do=view(do_p),
        sample_stats=stats,
        sample_dq=view(dq),
        sample_dk=view(dk),
        sample_dv=view(dv),
        scale_softmax=scale,
        thd=True,
        max_total_seq_len_q=t_q,
        max_total_seq_len_kv=t_kv,
        thd_stats_token_major=token_major_stats,
    )
    assert api.check_support()
    ws = torch.empty(api.scratch_workspace_bytes(), dtype=torch.uint8, device=dev)
    api.execute(
        view(q_p),
        view(k_p),
        view(v_p),
        view(o_p),
        view(do_p),
        stats,
        view(dq),
        view(dk),
        view(dv),
        workspace=ws,
        seq_q_lens=torch.tensor(lens_q, dtype=torch.int32, device=dev),
        seq_kv_lens=torch.tensor(lens_kv, dtype=torch.int32, device=dev),
    )
    torch.cuda.synchronize()

    for i in range(b):
        sl_q, sl_k = slice(cu_q[i], cu_q[i] + lens_q[i]), slice(cu_k[i], cu_k[i] + lens_kv[i])
        rq, rk, rv = _ref_bwd(
            q_p[0, sl_q].transpose(0, 1),
            k_p[0, sl_k].transpose(0, 1),
            v_p[0, sl_k].transpose(0, 1),
            do_p[0, sl_q].transpose(0, 1),
            scale,
        )
        for name, got, want in (
            ("dQ", dq[0, sl_q].transpose(0, 1), rq),
            ("dK", dk[0, sl_k].transpose(0, 1), rk),
            ("dV", dv[0, sl_k].transpose(0, 1), rv),
        ):
            c = _cos(got, want)
            assert c > _TOL_COS, f"seq {i} {name}: cos {c:.6f} (lens q={lens_q[i]} kv={lens_kv[i]})"


@pytest.mark.parametrize("dt", (torch.bfloat16, torch.float16), ids=("bf16", "fp16"))
def test_thd_self_attention(dt):
    """Three sequences of unequal length, none a tile multiple."""
    _run((300, 128, 200), (300, 128, 200), dtype=dt)


def test_thd_cross_attention():
    """Unequal Q and KV lengths, and unequal packed totals with them."""
    _run((256, 100), (180, 300))


def test_thd_single_sequence_matches_dense_shape():
    """B == 1 is the degenerate packing: it must agree with the dense answer."""
    _run((512,), (512,))


def test_thd_stats_token_major():
    """The other packed Stats layout the forward can emit."""
    _run((300, 128), (300, 128), token_major_stats=True)
