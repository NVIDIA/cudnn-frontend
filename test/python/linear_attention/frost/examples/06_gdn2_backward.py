# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example 06: GDN-2 (Gated DeltaNet v2) backward (pure cuDNN frontend API).

The GDN2_BWD node takes the forward inputs plus ``dO`` and returns
``(dQ, dK, dV, dG, dBeta, dW)`` (per-key-channel ``dG``/``dBeta``, per-value
``dW``; ``dBeta``/``dW`` in io dtype). Without the optional per-chunk ``h``
input the engine recomputes the forward state pass internally. This example
runs the in-kernel q/k L2 norm, the bounded safe gate, and the erase-side
beta safeguard (``beta_guard``). The decay input is a raw pre-activation;
with ``safe_gate=True`` the kernel computes the bounded gate

    g = -5 * sigmoid(exp(A_log) * (a + dt_bias))

here with ``A_log = 0`` and ``dt_bias = 0``, ``dG`` is the raw-logit
gradient, and ``d_a_log`` / ``d_dt_bias`` are also produced. The guard is
straight-through, so the reference applies the projection detached (identity
gradient to ``beta``, none to the gate). Gradients are checked against fp64
autograd through the per-token recurrence.
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
    g.select_plan(names.index("gdn2_frost"))
    g.check_support()
    g.build_plans()


def _rms_ratio(out, ref):
    out, ref = out.detach().double(), ref.detach().double()
    return ((out - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt().clamp_min(1e-12)).item()


def _beta_guard(kn, beta, alpha, io_dtype):
    """fp64 mirror of the kernel beta guard: kn l2-normalized, alpha = exp(g)."""
    w = kn * kn
    n = w.sum(-1)
    a = (beta * w).sum(-1)
    nu = (beta * beta * w).sum(-1)
    r2 = (n * nu - a * a).clamp_min(0.0)
    inv_c2 = alpha.amax(-1).pow(2)
    c2 = 1.0 / inv_c2
    r2_crit = ((c2 - 1.0) * (1.0 - (1.0 - a).pow(2) * inv_c2)).clamp_min(0.0)
    unsafe = (n > 1.0e-20) & (r2 > r2_crit)
    mu = a / n.clamp_min(1.0e-20)
    eta = ((1.0 - 1.0 / 32) * r2_crit / r2.clamp_min(1.0e-30)).sqrt().clamp(0.0, 1.0)
    cand = torch.where(unsafe[..., None], mu[..., None] + eta[..., None] * (beta - mu[..., None]), beta).to(io_dtype).double()
    a_q = (cand * w).sum(-1)
    nu_q = (cand * cand * w).sum(-1)
    r2_q = (n * nu_q - a_q * a_q).clamp_min(0.0)
    r2_crit_q = ((c2 - 1.0) * (1.0 - (1.0 - a_q).pow(2) * inv_c2)).clamp_min(0.0)
    tol = 4.0 * torch.finfo(io_dtype).eps * (n * nu_q + a_q * a_q)
    fallback = unsafe & (r2_q > r2_crit_q + tol)
    mu_q = (a_q / n.clamp_min(1.0e-20)).to(io_dtype).double()
    return torch.where(fallback[..., None], mu_q[..., None], cand)


def _reference_o(q, k, v, g, beta, w, cu, scale):
    """Differentiable fp64 per-token recurrence with the in-kernel L2 norm and
    the straight-through beta guard; returns o."""
    total, H, D = q.shape
    V = v.shape[2]
    q = torch.nn.functional.normalize(q, dim=-1)
    k = torch.nn.functional.normalize(k, dim=-1)
    beta = beta + (_beta_guard(k.detach(), beta.detach(), g.exp().detach(), torch.bfloat16) - beta.detach())
    outs = []
    for n in range(cu.numel() - 1):
        S = torch.zeros(H, D, V, dtype=torch.float64, device=q.device)
        for t in range(int(cu[n]), int(cu[n + 1])):
            S = g[t].exp()[..., None] * S
            erase = torch.einsum("hd,hdv->hv", beta[t] * k[t], S)
            v_new = w[t] * v[t] - erase
            S = S + torch.einsum("hd,hv->hdv", k[t], v_new)
            outs.append(torch.einsum("hd,hdv->hv", q[t] * scale, S))
    return torch.stack(outs, dim=0)


def main(seq_lens=(192, 320), H: int = 2, D: int = 128) -> None:
    torch.manual_seed(0)
    device = "cuda"
    total, num_seqs = sum(seq_lens), len(seq_lens)
    scale = 1.0 / math.sqrt(D)

    q = torch.randn(total, H, D, device=device).bfloat16()
    k = torch.randn(total, H, D, device=device).bfloat16()
    v = torch.randn(total, H, D, device=device).bfloat16()
    gate = -2.5 + 0.7 * torch.randn(total, H, D, device=device)
    gate[1::2, :, :4] = -8.0
    gate = gate.contiguous()
    a_log = torch.zeros(H, device=device, dtype=torch.float32)
    dt_bias = torch.zeros(H, D, device=device, dtype=torch.float32)
    beta = (torch.rand(total, H, D, device=device).sigmoid() * 2.0).bfloat16().contiguous()
    w = torch.rand(total, H, D, device=device).sigmoid().bfloat16().contiguous()
    do = torch.randn(total, H, D, device=device).bfloat16()
    cu = torch.tensor([0, *torch.tensor(seq_lens).cumsum(0).tolist()], dtype=torch.int32, device=device)

    g = cudnn.pygraph()
    q_t = g.tensor([total, H, D], data_type=cudnn.data_type.BFLOAT16, name="q")
    k_t = g.tensor([total, H, D], data_type=cudnn.data_type.BFLOAT16, name="k")
    v_t = g.tensor([total, H, D], data_type=cudnn.data_type.BFLOAT16, name="v")
    g_t = g.tensor([total, H, D], data_type=cudnn.data_type.FLOAT, name="g")
    beta_t = g.tensor([total, H, D], data_type=cudnn.data_type.BFLOAT16, name="beta")
    w_t = g.tensor([total, H, D], data_type=cudnn.data_type.BFLOAT16, name="w")
    cu_t = g.tensor([num_seqs + 1], data_type=cudnn.data_type.INT32, name="cu_seqlens")
    do_t = g.tensor([total, H, D], data_type=cudnn.data_type.BFLOAT16, name="dO")
    a_log_t = g.tensor([H], data_type=cudnn.data_type.FLOAT, name="a_log")
    dt_bias_t = g.tensor([H, D], data_type=cudnn.data_type.FLOAT, name="dt_bias")
    dq_t, dk_t, dv_t, dg_t, db_t, dw_t, _dstate0_t, da_log_t, ddt_bias_t = g.gdn2_bwd(
        q=q_t,
        k=k_t,
        v=v_t,
        g=g_t,
        beta=beta_t,
        w=w_t,
        cu_seqlens=cu_t,
        dO=do_t,
        a_log=a_log_t,
        dt_bias=dt_bias_t,
        scale=scale,
        use_qk_l2norm=True,
        safe_gate=True,
        beta_guard=True,
        name="gdn2_bwd",
    )
    io_dt, f32_dt = cudnn.data_type.BFLOAT16, cudnn.data_type.FLOAT
    grads_t = [
        out.set_output(True).set_data_type(dt)
        for out, dt in ((dq_t, io_dt), (dk_t, io_dt), (dv_t, io_dt), (dg_t, f32_dt), (db_t, io_dt), (dw_t, io_dt), (da_log_t, f32_dt), (ddt_bias_t, f32_dt))
    ]
    _build_plans(g)

    dq = torch.empty(total, H, D, dtype=torch.bfloat16, device=device)
    dk = torch.empty(total, H, D, dtype=torch.bfloat16, device=device)
    dv = torch.empty(total, H, D, dtype=torch.bfloat16, device=device)
    dg = torch.empty(total, H, D, dtype=torch.float32, device=device)
    db = torch.empty(total, H, D, dtype=torch.bfloat16, device=device)
    dw = torch.empty(total, H, D, dtype=torch.bfloat16, device=device)
    d_a_log = torch.empty(H, dtype=torch.float32, device=device)
    d_dt_bias = torch.empty(H, D, dtype=torch.float32, device=device)
    pack = {q_t: q, k_t: k, v_t: v, g_t: gate, beta_t: beta, w_t: w, cu_t: cu, do_t: do, a_log_t: a_log, dt_bias_t: dt_bias}
    pack.update(dict(zip(grads_t, (dq, dk, dv, dg, db, dw, d_a_log, d_dt_bias))))
    g.execute(pack, torch.empty(max(g.get_workspace_size(), 1), dtype=torch.uint8, device=device))
    torch.cuda.synchronize()

    leaves = [x.double().requires_grad_(True) for x in (q, k, v, gate, beta, w)]
    a_leaf = a_log.double().requires_grad_(True)
    dt_leaf = dt_bias.double().requires_grad_(True)
    gact = -5.0 * torch.sigmoid(a_leaf.exp()[None, :, None] * (leaves[3] + dt_leaf[None]))
    o_ref = _reference_o(leaves[0], leaves[1], leaves[2], gact, leaves[4], leaves[5], cu, scale)
    grads = torch.autograd.grad((o_ref * do.double()).sum(), leaves + [a_leaf, dt_leaf])
    for name, out, ref in (
        ("dQ", dq, grads[0]),
        ("dK", dk, grads[1]),
        ("dV", dv, grads[2]),
        ("dG", dg, grads[3]),
        ("dBeta", db, grads[4]),
        ("dW", dw, grads[5]),
        ("d_a_log", d_a_log, grads[6]),
        ("d_dt_bias", d_dt_bias, grads[7]),
    ):
        r = _rms_ratio(out, ref)
        assert r < 5e-2, f"{name} rms ratio {r:.4g}"
    print(f"[06] PASS  gdn2 backward (safe gate + beta guard)  seq_lens={list(seq_lens)} H={H} D={D}")


if __name__ == "__main__":
    main()
