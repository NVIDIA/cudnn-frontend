# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example 18: Mamba-2 SSD backward (pure cuDNN frontend API).

The MAMBA2_BWD node takes the forward inputs plus ``dO`` and returns ``(dQ, dK, dV, dG, dA,
d_initial_state, d_dt_bias, dD, dZ)``: ``dG`` is the gradient of the raw ``dt`` and ``dA``
that of ``log(-A)``.  Without the optional per-chunk ``state_checkpoints`` input the engine
regenerates the state series internally.  The tensor gradients are checked against fp64 autograd
through the per-token recurrence; ``d_dt_bias`` against its identity ``sum_t dG_t`` (the per-head
parameter sums cancel heavily across tokens, so the fp64 comparison of ``dA`` is loose).
"""

from __future__ import annotations

import cudnn
import torch


def _build_plans(g) -> None:
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    names = [g.get_plan_name_at_index(i) for i in range(len(g.plans))]
    g.select_plan(names.index("mamba2_frost"))
    g.check_support()
    g.build_plans()


def _rms_ratio(out, ref):
    out, ref = out.detach().double(), ref.detach().double()
    return ((out - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt().clamp_min(1e-12)).item()


def _reference_o(q, k, v, dt_raw, a, dt_bias, cu):
    """Differentiable fp64 per-token recurrence; returns o."""
    H, P = v.shape[1], v.shape[2]
    N_state = q.shape[2]
    dt = torch.nn.functional.softplus(dt_raw + dt_bias)
    alpha = torch.exp(a * dt)
    outs = []
    for i in range(cu.numel() - 1):
        S = torch.zeros(H, N_state, P, dtype=torch.float64, device=q.device)
        for t in range(int(cu[i]), int(cu[i + 1])):
            S = alpha[t][:, None, None] * S + dt[t][:, None, None] * torch.einsum("hn,hp->hnp", k[t], v[t])
            outs.append(torch.einsum("hn,hnp->hp", q[t], S))
    return torch.stack(outs, dim=0)


def main(seq_lens=(192, 320), H: int = 2, N_state: int = 128, P: int = 64) -> None:
    torch.manual_seed(0)
    device = "cuda"
    total, num_seqs = sum(seq_lens), len(seq_lens)

    q = (torch.randn(total, H, N_state, device=device) * 0.1).bfloat16()
    k = torch.nn.functional.normalize(torch.randn(total, H, N_state, device=device), dim=-1).bfloat16()
    v = torch.randn(total, H, P, device=device).bfloat16()
    A = -(torch.rand(H, device=device) * 7.0 + 1.0)
    dt_bias = torch.rand(H, device=device) * 0.5
    dt_raw = (torch.rand(total, H, device=device) * 4.0 - 4.0).contiguous()
    do = torch.randn(total, H, P, device=device).bfloat16()
    cu = torch.tensor([0, *torch.tensor(seq_lens).cumsum(0).tolist()], dtype=torch.int32, device=device)

    g = cudnn.pygraph()
    q_t = g.tensor([total, H, N_state], data_type=cudnn.data_type.BFLOAT16, name="q")
    k_t = g.tensor([total, H, N_state], data_type=cudnn.data_type.BFLOAT16, name="k")
    v_t = g.tensor([total, H, P], data_type=cudnn.data_type.BFLOAT16, name="v")
    g_t = g.tensor([total, H], data_type=cudnn.data_type.FLOAT, name="g")
    a_t = g.tensor([H], data_type=cudnn.data_type.FLOAT, name="a")
    dt_bias_t = g.tensor([H], data_type=cudnn.data_type.FLOAT, name="dt_bias")
    cu_t = g.tensor([num_seqs + 1], data_type=cudnn.data_type.INT32, name="cu_seqlens")
    do_t = g.tensor([total, H, P], data_type=cudnn.data_type.BFLOAT16, name="dO")
    dQ_t, dK_t, dV_t, dG_t, dA_t, _dS0_t, dDt_t, _dD_t, _dZ_t = g.mamba2_bwd(
        q=q_t,
        k=k_t,
        v=v_t,
        g=g_t,
        a=a_t,
        cu_seqlens=cu_t,
        dO=do_t,
        dt_bias=dt_bias_t,
        name="mamba2_bwd",
    )
    for t_, dt in (
        (dQ_t, cudnn.data_type.BFLOAT16),
        (dK_t, cudnn.data_type.BFLOAT16),
        (dV_t, cudnn.data_type.BFLOAT16),
        (dG_t, cudnn.data_type.FLOAT),
        (dA_t, cudnn.data_type.FLOAT),
        (dDt_t, cudnn.data_type.FLOAT),
    ):
        t_.set_output(True).set_data_type(dt)
    _build_plans(g)

    dq = torch.empty(total, H, N_state, dtype=torch.bfloat16, device=device)
    dk = torch.empty(total, H, N_state, dtype=torch.bfloat16, device=device)
    dv = torch.empty(total, H, P, dtype=torch.bfloat16, device=device)
    dg = torch.empty(total, H, dtype=torch.float32, device=device)
    dA = torch.empty(H, dtype=torch.float32, device=device)
    d_dt_bias = torch.empty(H, dtype=torch.float32, device=device)
    pack = {
        q_t: q,
        k_t: k,
        v_t: v,
        g_t: dt_raw,
        a_t: A,
        dt_bias_t: dt_bias,
        cu_t: cu,
        do_t: do,
        dQ_t: dq,
        dK_t: dk,
        dV_t: dv,
        dG_t: dg,
        dA_t: dA,
        dDt_t: d_dt_bias,
    }
    g.execute(pack, torch.empty(max(g.get_workspace_size(), 1), dtype=torch.uint8, device=device))
    torch.cuda.synchronize()

    leaves = [x.double().requires_grad_(True) for x in (q, k, v, dt_raw, A, dt_bias)]
    o_ref = _reference_o(*leaves, cu)
    grads = torch.autograd.grad((o_ref * do.double()).sum(), leaves)
    for name, out, ref, tol in (
        ("dQ", dq, grads[0], 5e-2),
        ("dK", dk, grads[1], 5e-2),
        ("dV", dv, grads[2], 5e-2),
        ("dG", dg, grads[3], 5e-2),
    ):
        r = _rms_ratio(out, ref)
        assert r < tol, f"{name} rms ratio {r:.4g}"
    ident = dg.double().sum(0)
    assert (d_dt_bias.double() - ident).abs().max().item() / max(ident.abs().max().item(), 1e-6) < 1e-3, "d_dt_bias identity"
    r_a = (dA.double() - grads[4]).abs().max().item() / max(grads[4].abs().max().item(), 1e-6)
    print(f"[18] PASS  mamba2 backward (recompute) seq_lens={list(seq_lens)} H={H} N={N_state} P={P} (dA max-rel {r_a:.2e})")


if __name__ == "__main__":
    main()
