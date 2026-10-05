# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example 14: GDN-2 (Gated DeltaNet v2) backward summary (pure cuDNN frontend API).

The GDN2_SUMMARY_BWD node is the reverse state-gradient pass behind context
parallelism. For every packed sequence it returns ``d_initial_state``
``[N, H, V, K]``, the gradient of ``sum(o * dO)`` with respect to the span's
incoming state from a zero outgoing state gradient (the optional
``d_final_state`` seeds it instead), and with ``output_transition=True`` the
span ``transition`` ``[N, H, K, K]`` in the backward's orientation, so that
state gradients compose in reverse as::

    dX_j = dX_{j+1} @ transition_j + d_initial_state_j

The state gradient flows through the decay and the erase only, so the node
takes no write gate ``w``. A summary must read its gates exactly as the main
op does, so this example runs the same in-kernel q/k L2 norm, bounded safe
gate and erase-side beta safeguard (``beta_guard``) as example 06: with
``safe_gate=True`` the kernel computes

    g = -5 * sigmoid(exp(A_log) * (a + dt_bias))

here with ``A_log = 0`` and ``dt_bias = 0``, and the fp64 reference applies
the same activation and projection. Here one logical sequence is cut into two
spans that are packed as two sequences. Each span's ``d_initial_state`` is
checked against fp64 autograd through the recurrence, its ``transition``
against the forward recurrence run from the identity with ``v = 0``, and the
reverse host-side composition against the fp64 gradient of the uncut sequence
with respect to its initial state. Pre-activations far below zero keep the
decays near one and the transition well away from zero; odd tokens carry
planted near-zero-decay channels, so the guard fires on them and stays quiet
on the even tokens.
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
    g.select_plan(names.index("gdn2_summary_frost"))
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


def _reference(q, k, v, g, beta, w, a_log, dt_bias, cu, scale, s0):
    """fp64 per-token recurrence over the packed batch from the V-major seed ``s0``, differentiable in ``s0``,
    with the safe-gate activation, in-kernel L2 norm, and beta guard applied. Returns (o, final_state)."""
    q, k, v, g, beta, w = (x.double() for x in (q, k, v, g, beta, w))
    g = -5.0 * torch.sigmoid(a_log.double().exp()[None, :, None] * (g + dt_bias.double()[None]))
    q = torch.nn.functional.normalize(q, dim=-1)
    k = torch.nn.functional.normalize(k, dim=-1)
    beta = _beta_guard(k, beta, g.exp(), torch.bfloat16)
    outs, states = [], []
    for i in range(cu.numel() - 1):
        S = s0[i].transpose(-2, -1)
        for t in range(int(cu[i]), int(cu[i + 1])):
            S = g[t].exp()[..., None] * S
            erase = torch.einsum("hd,hdv->hv", beta[t] * k[t], S)
            v_new = w[t] * v[t] - erase
            S = S + torch.einsum("hd,hv->hdv", k[t], v_new)
            outs.append(torch.einsum("hd,hdv->hv", q[t] * scale, S))
        states.append(S.transpose(-2, -1))
    return torch.stack(outs, dim=0), torch.stack(states, dim=0)


def main(spans=(128, 192), H: int = 2, D: int = 128) -> None:
    torch.manual_seed(0)
    device = "cuda"
    total, num_spans = sum(spans), len(spans)
    scale = 1.0 / math.sqrt(D)

    q = torch.randn(total, H, D, device=device).bfloat16()
    k = torch.randn(total, H, D, device=device).bfloat16()
    v = torch.randn(total, H, D, device=device).bfloat16()
    gate = -6.5 + 0.7 * torch.randn(total, H, D, device=device)
    gate[1::2, :, :4] = -12.0
    gate = gate.contiguous()
    a_log = torch.zeros(H, device=device, dtype=torch.float32)
    dt_bias = torch.zeros(H, D, device=device, dtype=torch.float32)
    beta = (torch.rand(total, H, D, device=device).sigmoid() * 2.0).bfloat16().contiguous()
    w = torch.rand(total, H, D, device=device).sigmoid().bfloat16().contiguous()
    do = torch.randn(total, H, D, device=device).bfloat16()
    cu = torch.tensor([0, *torch.tensor(spans).cumsum(0).tolist()], dtype=torch.int32, device=device)

    g = cudnn.pygraph()
    q_t = g.tensor([total, H, D], data_type=cudnn.data_type.BFLOAT16, name="q")
    k_t = g.tensor([total, H, D], data_type=cudnn.data_type.BFLOAT16, name="k")
    g_t = g.tensor([total, H, D], data_type=cudnn.data_type.FLOAT, name="g")
    beta_t = g.tensor([total, H, D], data_type=cudnn.data_type.BFLOAT16, name="beta")
    cu_t = g.tensor([num_spans + 1], data_type=cudnn.data_type.INT32, name="cu_seqlens")
    do_t = g.tensor([total, H, D], data_type=cudnn.data_type.BFLOAT16, name="dO")
    a_log_t = g.tensor([H], data_type=cudnn.data_type.FLOAT, name="a_log")
    dt_bias_t = g.tensor([H, D], data_type=cudnn.data_type.FLOAT, name="dt_bias")
    dS0_t, transition_t = g.gdn2_summary_bwd(
        q=q_t,
        k=k_t,
        g=g_t,
        beta=beta_t,
        cu_seqlens=cu_t,
        dO=do_t,
        a_log=a_log_t,
        dt_bias=dt_bias_t,
        scale=scale,
        output_transition=True,
        use_qk_l2norm=True,
        safe_gate=True,
        beta_guard=True,
        name="gdn2_summary_bwd",
    )
    dS0_t.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    transition_t.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    _build_plans(g)

    d_initial_state = torch.empty(num_spans, H, D, D, dtype=torch.float32, device=device)
    transition = torch.empty(num_spans, H, D, D, dtype=torch.float32, device=device)
    pack = {q_t: q, k_t: k, g_t: gate, beta_t: beta, cu_t: cu, do_t: do, a_log_t: a_log, dt_bias_t: dt_bias, dS0_t: d_initial_state, transition_t: transition}
    g.execute(pack, torch.empty(max(g.get_workspace_size(), 1), dtype=torch.uint8, device=device))
    torch.cuda.synchronize()

    s0 = torch.zeros(num_spans, H, D, D, dtype=torch.float64, device=device, requires_grad=True)
    o_ref, _ = _reference(q, k, v, gate, beta, w, a_log, dt_bias, cu, scale, s0)
    (dS0_ref,) = torch.autograd.grad((o_ref * do.double()).sum(), [s0])
    eye = torch.eye(D, dtype=torch.float64, device=device).expand(num_spans, H, D, D)
    transition_ref = _reference(q, k, torch.zeros_like(v), gate, beta, w, a_log, dt_bias, cu, scale, eye)[1].transpose(-2, -1)
    cu_uncut = torch.tensor([0, total], dtype=torch.int32, device=device)
    s0_uncut = torch.zeros(1, H, D, D, dtype=torch.float64, device=device, requires_grad=True)
    o_uncut, _ = _reference(q, k, v, gate, beta, w, a_log, dt_bias, cu_uncut, scale, s0_uncut)
    (composed_ref,) = torch.autograd.grad((o_uncut * do.double()).sum(), [s0_uncut])
    composed = d_initial_state[1].double() @ transition[0].double() + d_initial_state[0].double()
    r_g, r_m, r_x = _rms_ratio(d_initial_state, dS0_ref), _rms_ratio(transition, transition_ref), _rms_ratio(composed, composed_ref[0])
    assert r_g < 5e-2, f"d_initial_state rms ratio {r_g:.4g}"
    assert r_m < 2e-2, f"transition rms ratio {r_m:.4g}"
    assert r_x < 5e-2, f"composed d_initial_state rms ratio {r_x:.4g}"
    print(
        f"[14] PASS  gdn2 summary backward (safe gate + beta guard)  spans={list(spans)} H={H} D={D} (dS0 rms {r_g:.2e}, transition rms {r_m:.2e}, composed rms {r_x:.2e})"
    )


if __name__ == "__main__":
    main()
