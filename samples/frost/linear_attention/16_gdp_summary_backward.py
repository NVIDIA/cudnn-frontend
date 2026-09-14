# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example 16: GDP (Gated DeltaProduct) backward summary (pure cuDNN frontend API).

The GDP_SUMMARY_BWD node is the reverse state-gradient pass behind context
parallelism, with GDP's ``n = num_householder`` delta-rule updates per token.
It reads ``q``, ``g`` and ``dO`` at the real tokens and ``k``, ``beta`` on the
expanded timeline (``total * n`` rows); ``cu_seqlens`` counts real tokens. For
every packed sequence it returns ``d_initial_state`` ``[N, H, V, K]``, the
gradient of ``sum(o * dO)`` with respect to the span's incoming state from a
zero outgoing state gradient (the optional ``d_final_state`` seeds it
instead), and with ``output_transition=True`` the span ``transition``
``[N, H, K, K]`` in the backward's orientation, so that state gradients
compose in reverse as::

    dX_j = dX_{j+1} @ transition_j + d_initial_state_j

Here one logical sequence is cut into two spans that are packed as two
sequences. Each span's ``d_initial_state`` is checked against fp64 autograd
through the recurrence, its ``transition`` against the forward recurrence run
from the identity with ``v = 0``, and the reverse host-side composition
against the fp64 gradient of the uncut sequence with respect to its initial
state.
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
    g.select_plan(names.index("gdp_summary_frost"))
    g.check_support()
    g.build_plans()


def _rms_ratio(out, ref):
    out, ref = out.detach().double(), ref.detach().double()
    return ((out - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt().clamp_min(1e-12)).item()


def _reference(q, k, v, g, beta, cu, n, scale, s0):
    """Differentiable fp64 per-token recurrence over the packed batch from the V-major seed ``s0``,
    ``n`` updates per token. Returns (o, final_state)."""
    outs, states = [], []
    for i in range(cu.numel() - 1):
        S = s0[i].transpose(-2, -1)
        for t in range(int(cu[i]), int(cu[i + 1])):
            S = g[t].exp()[:, None, None] * S
            for r in range(t * n, (t + 1) * n):
                residual = v[r] - torch.einsum("hd,hdv->hv", k[r], S)
                S = S + beta[r][:, None, None] * torch.einsum("hd,hv->hdv", k[r], residual)
            outs.append(torch.einsum("hd,hdv->hv", q[t] * scale, S))
        states.append(S.transpose(-2, -1))
    return torch.stack(outs, dim=0), torch.stack(states, dim=0)


def main(spans=(128, 192), H: int = 2, D: int = 128, n: int = 2) -> None:
    torch.manual_seed(0)
    device = "cuda"
    total, num_spans = sum(spans), len(spans)
    scale = 1.0 / math.sqrt(D)

    q = torch.randn(total, H, D, device=device).bfloat16()
    k = torch.nn.functional.normalize(torch.randn(total * n, H, D, device=device), dim=-1).bfloat16()
    v = torch.randn(total * n, H, D, device=device).bfloat16()
    gate = torch.empty(total, H, device=device).uniform_(0.98, 1.0).log().contiguous()
    beta = torch.rand(total * n, H, device=device).contiguous()
    do = torch.randn(total, H, D, device=device).bfloat16()
    cu = torch.tensor([0, *torch.tensor(spans).cumsum(0).tolist()], dtype=torch.int32, device=device)

    g = cudnn.pygraph()
    q_t = g.tensor([total, H, D], data_type=cudnn.data_type.BFLOAT16, name="q")
    k_t = g.tensor([total * n, H, D], data_type=cudnn.data_type.BFLOAT16, name="k")
    g_t = g.tensor([total, H], data_type=cudnn.data_type.FLOAT, name="g")
    beta_t = g.tensor([total * n, H], data_type=cudnn.data_type.FLOAT, name="beta")
    cu_t = g.tensor([num_spans + 1], data_type=cudnn.data_type.INT32, name="cu_seqlens")
    do_t = g.tensor([total, H, D], data_type=cudnn.data_type.BFLOAT16, name="dO")
    dS0_t, transition_t = g.gdp_summary_bwd(
        q=q_t,
        k=k_t,
        g=g_t,
        beta=beta_t,
        cu_seqlens=cu_t,
        dO=do_t,
        num_householder=n,
        scale=scale,
        output_transition=True,
        name="gdp_summary_bwd",
    )
    dS0_t.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    transition_t.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    _build_plans(g)

    d_initial_state = torch.empty(num_spans, H, D, D, dtype=torch.float32, device=device)
    transition = torch.empty(num_spans, H, D, D, dtype=torch.float32, device=device)
    pack = {q_t: q, k_t: k, g_t: gate, beta_t: beta, cu_t: cu, do_t: do, dS0_t: d_initial_state, transition_t: transition}
    g.execute(pack, torch.empty(max(g.get_workspace_size(), 1), dtype=torch.uint8, device=device))
    torch.cuda.synchronize()

    qd, kd, vd, gd, bd, dod = (x.double() for x in (q, k, v, gate, beta, do))
    s0 = torch.zeros(num_spans, H, D, D, dtype=torch.float64, device=device, requires_grad=True)
    o_ref, _ = _reference(qd, kd, vd, gd, bd, cu, n, scale, s0)
    (dS0_ref,) = torch.autograd.grad((o_ref * dod).sum(), [s0])
    eye = torch.eye(D, dtype=torch.float64, device=device).expand(num_spans, H, D, D)
    transition_ref = _reference(qd, kd, torch.zeros_like(vd), gd, bd, cu, n, scale, eye)[1].transpose(-2, -1)
    cu_uncut = torch.tensor([0, total], dtype=torch.int32, device=device)
    s0_uncut = torch.zeros(1, H, D, D, dtype=torch.float64, device=device, requires_grad=True)
    o_uncut, _ = _reference(qd, kd, vd, gd, bd, cu_uncut, n, scale, s0_uncut)
    (composed_ref,) = torch.autograd.grad((o_uncut * dod).sum(), [s0_uncut])
    composed = d_initial_state[1].double() @ transition[0].double() + d_initial_state[0].double()
    r_g, r_m, r_x = _rms_ratio(d_initial_state, dS0_ref), _rms_ratio(transition, transition_ref), _rms_ratio(composed, composed_ref[0])
    assert r_g < 5e-2, f"d_initial_state rms ratio {r_g:.4g}"
    assert r_m < 2e-2, f"transition rms ratio {r_m:.4g}"
    assert r_x < 5e-2, f"composed d_initial_state rms ratio {r_x:.4g}"
    print(f"[16] PASS  gdp summary backward        spans={list(spans)} H={H} D={D} n={n} (dS0 rms {r_g:.2e}, transition rms {r_m:.2e}, composed rms {r_x:.2e})")


if __name__ == "__main__":
    main()
