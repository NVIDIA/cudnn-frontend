# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example 02: GDN (Gated DeltaNet) backward (pure cuDNN frontend API).

The GDN_BWD node takes the forward inputs plus ``dO`` and returns
``(dQ, dK, dV, dG, dBeta, dS0)``. Without the optional per-chunk ``h`` input
the engine recomputes the forward state pass internally. Gradients are
checked against fp64 autograd through the per-token recurrence.
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
    g.select_plan(names.index("gdn_frost"))
    g.check_support()
    g.build_plans()


def _rms_ratio(out, ref):
    out, ref = out.detach().double(), ref.detach().double()
    return ((out - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt().clamp_min(1e-12)).item()


def _reference_o(q, k, v, g, beta, cu, scale):
    """Differentiable fp64 per-token recurrence; returns o."""
    total, H, _D = q.shape
    V = v.shape[2]
    outs = []
    for n in range(cu.numel() - 1):
        S = torch.zeros(H, q.shape[2], V, dtype=torch.float64, device=q.device)
        for t in range(int(cu[n]), int(cu[n + 1])):
            a, b = g[t].exp(), beta[t]
            residual = v[t] - a[:, None] * torch.einsum("hd,hdv->hv", k[t], S)
            S = a[:, None, None] * S + b[:, None, None] * torch.einsum("hd,hv->hdv", k[t], residual)
            outs.append(torch.einsum("hd,hdv->hv", q[t] * scale, S))
    return torch.stack(outs, dim=0)


def main(seq_lens=(192, 320), H: int = 2, D: int = 128) -> None:
    torch.manual_seed(0)
    device = "cuda"
    total, num_seqs = sum(seq_lens), len(seq_lens)
    scale = 1.0 / math.sqrt(D)

    q = torch.randn(total, H, D, device=device).bfloat16()
    k = torch.nn.functional.normalize(torch.randn(total, H, D, device=device), dim=-1).bfloat16()
    v = torch.randn(total, H, D, device=device).bfloat16()
    gate = torch.empty(total, H, device=device).uniform_(0.1, 1.0).log().contiguous()
    beta = torch.rand(total, H, device=device).contiguous()
    do = torch.randn(total, H, D, device=device).bfloat16()
    cu = torch.tensor([0, *torch.tensor(seq_lens).cumsum(0).tolist()], dtype=torch.int32, device=device)

    g = cudnn.pygraph()
    q_t = g.tensor([total, H, D], data_type=cudnn.data_type.BFLOAT16, name="q")
    k_t = g.tensor([total, H, D], data_type=cudnn.data_type.BFLOAT16, name="k")
    v_t = g.tensor([total, H, D], data_type=cudnn.data_type.BFLOAT16, name="v")
    g_t = g.tensor([total, H], data_type=cudnn.data_type.FLOAT, name="g")
    beta_t = g.tensor([total, H], data_type=cudnn.data_type.FLOAT, name="beta")
    cu_t = g.tensor([num_seqs + 1], data_type=cudnn.data_type.INT32, name="cu_seqlens")
    do_t = g.tensor([total, H, D], data_type=cudnn.data_type.BFLOAT16, name="dO")
    dQ_t, dK_t, dV_t, dG_t, dBeta_t, _dS0_t, _dA_t, _dDt_t = g.gdn_bwd(
        q=q_t,
        k=k_t,
        v=v_t,
        g=g_t,
        beta=beta_t,
        cu_seqlens=cu_t,
        dO=do_t,
        scale=scale,
        name="gdn_bwd",
    )
    for t_, dt in (
        (dQ_t, cudnn.data_type.BFLOAT16),
        (dK_t, cudnn.data_type.BFLOAT16),
        (dV_t, cudnn.data_type.BFLOAT16),
        (dG_t, cudnn.data_type.FLOAT),
        (dBeta_t, cudnn.data_type.FLOAT),
    ):
        t_.set_output(True).set_data_type(dt)
    _build_plans(g)

    dq = torch.empty(total, H, D, dtype=torch.bfloat16, device=device)
    dk = torch.empty(total, H, D, dtype=torch.bfloat16, device=device)
    dv = torch.empty(total, H, D, dtype=torch.bfloat16, device=device)
    dg = torch.empty(total, H, dtype=torch.float32, device=device)
    db = torch.empty(total, H, dtype=torch.float32, device=device)
    pack = {q_t: q, k_t: k, v_t: v, g_t: gate, beta_t: beta, cu_t: cu, do_t: do, dQ_t: dq, dK_t: dk, dV_t: dv, dG_t: dg, dBeta_t: db}
    g.execute(pack, torch.empty(max(g.get_workspace_size(), 1), dtype=torch.uint8, device=device))
    torch.cuda.synchronize()

    leaves = [x.double().requires_grad_(True) for x in (q, k, v, gate, beta)]
    o_ref = _reference_o(*leaves, cu, scale)
    grads = torch.autograd.grad((o_ref * do.double()).sum(), leaves)
    for name, out, ref, tol in (
        ("dQ", dq, grads[0], 5e-2),
        ("dK", dk, grads[1], 5e-2),
        ("dV", dv, grads[2], 5e-2),
        ("dG", dg, grads[3], 5e-2),
        ("dBeta", db, grads[4], 5e-2),
    ):
        r = _rms_ratio(out, ref)
        assert r < tol, f"{name} rms ratio {r:.4g}"
    print(f"[02] PASS  gdn backward (recompute)    seq_lens={list(seq_lens)} H={H} D={D}")


if __name__ == "__main__":
    main()
