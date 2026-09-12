# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example 07: GDP (Gated DeltaProduct) prefill (pure cuDNN frontend API).

GDP applies ``n = num_householder`` beta-gated delta-rule (Householder)
updates per token. The scalar decay ``alpha_t = exp(g_t)`` acts once, before
the token's updates, and the readout follows the last one::

    S'  = alpha_t S_{t-1}
    S'  = (I - beta_{t,j} k_{t,j}^T k_{t,j}) S' + beta_{t,j} k_{t,j}^T v_{t,j}    j = 0 .. n-1
    S_t = S'
    o_t = q_t S_t

THD layout on two timelines: ``q`` and ``g`` live at the real tokens
``[total, H, .]``, while ``k``, ``v`` and ``beta`` live on the expanded
timeline ``[total * n, H, .]`` with sub-token ``j`` of token ``t`` at row
``t * n + j``; ``cu_seqlens`` counts real tokens. ``num_householder=1`` is
exactly GDN. The final state comes back V-major ``[N, H, V, K]``.
"""

from __future__ import annotations

import math

import cudnn
import torch


def _build_plans(g) -> None:
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    names = [g.get_plan_name_at_index(i) for i in range(len(g.plans))]
    g.select_plan(names.index("gdp_frost"))
    g.check_support()
    g.build_plans()


def _rms_ratio(out, ref):
    out, ref = out.detach().double(), ref.detach().double()
    return ((out - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt().clamp_min(1e-12)).item()


def _reference(q, k, v, g, beta, cu, n, scale):
    """fp64 per-token recurrence over the packed batch, ``n`` updates per token. Returns (o, final_state)."""
    total, H, D = q.shape
    V = v.shape[2]
    q, k, v, g, beta = (x.double() for x in (q, k, v, g, beta))
    o = torch.zeros(total, H, V, dtype=torch.float64, device=q.device)
    fs = torch.zeros(cu.numel() - 1, H, D, V, dtype=torch.float64, device=q.device)
    for i in range(cu.numel() - 1):
        S = torch.zeros(H, D, V, dtype=torch.float64, device=q.device)
        for t in range(int(cu[i]), int(cu[i + 1])):
            S = g[t].exp()[:, None, None] * S  # the decay acts once per token, before its updates
            for r in range(t * n, (t + 1) * n):
                residual = v[r] - torch.einsum("hd,hdv->hv", k[r], S)
                S = S + beta[r][:, None, None] * torch.einsum("hd,hv->hdv", k[r], residual)
            o[t] = torch.einsum("hd,hdv->hv", q[t] * scale, S)
        fs[i] = S.transpose(-2, -1)
    return o, fs


def main(seq_lens=(192, 320), H: int = 2, D: int = 128, n: int = 2) -> None:
    torch.manual_seed(0)
    device = "cuda"
    total, num_seqs = sum(seq_lens), len(seq_lens)
    scale = 1.0 / math.sqrt(D)

    q = torch.randn(total, H, D, device=device).bfloat16()
    k = torch.nn.functional.normalize(torch.randn(total * n, H, D, device=device), dim=-1).bfloat16()
    v = torch.randn(total * n, H, D, device=device).bfloat16()
    gate = torch.empty(total, H, device=device).uniform_(0.1, 1.0).log().contiguous()
    beta = torch.rand(total * n, H, device=device).contiguous()
    cu = torch.tensor([0, *torch.tensor(seq_lens).cumsum(0).tolist()], dtype=torch.int32, device=device)

    g = cudnn.pygraph()
    q_t = g.tensor([total, H, D], data_type=cudnn.data_type.BFLOAT16, name="q")
    k_t = g.tensor([total * n, H, D], data_type=cudnn.data_type.BFLOAT16, name="k")
    v_t = g.tensor([total * n, H, D], data_type=cudnn.data_type.BFLOAT16, name="v")
    g_t = g.tensor([total, H], data_type=cudnn.data_type.FLOAT, name="g")
    beta_t = g.tensor([total * n, H], data_type=cudnn.data_type.FLOAT, name="beta")
    cu_t = g.tensor([num_seqs + 1], data_type=cudnn.data_type.INT32, name="cu_seqlens")
    O_t, fs_t, _h_t = g.gdp(
        q=q_t,
        k=k_t,
        v=v_t,
        g=g_t,
        beta=beta_t,
        cu_seqlens=cu_t,
        num_householder=n,
        scale=scale,
        output_final_state=True,
        name="gdp",
    )
    O_t.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    fs_t.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    _build_plans(g)

    o = torch.empty(total, H, D, dtype=torch.bfloat16, device=device)
    fs = torch.empty(num_seqs, H, D, D, dtype=torch.float32, device=device)
    pack = {q_t: q, k_t: k, v_t: v, g_t: gate, beta_t: beta, cu_t: cu, O_t: o, fs_t: fs}
    g.execute(pack, torch.empty(max(g.get_workspace_size(), 1), dtype=torch.uint8, device=device))
    torch.cuda.synchronize()

    o_ref, fs_ref = _reference(q, k, v, gate, beta, cu, n, scale)
    r_o, r_s = _rms_ratio(o, o_ref), _rms_ratio(fs, fs_ref)
    assert r_o < 2e-2, f"o rms ratio {r_o:.4g}"
    assert r_s < 2e-2, f"final_state rms ratio {r_s:.4g}"
    print(f"[07] PASS  gdp prefill                 seq_lens={list(seq_lens)} H={H} D={D} n={n} (o rms {r_o:.2e}, fs rms {r_s:.2e})")


if __name__ == "__main__":
    main()
