# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example 20: Mamba-2 SSD backward summary (pure cuDNN frontend API).

The MAMBA2_SUMMARY_BWD node is the reverse state-gradient pass behind context
parallelism: the Mamba-2 state-gradient recurrence over every packed sequence
without the input gradients. It reads ``q`` = C, ``g`` = the raw ``dt``
(``a`` = A) and ``dO``; B and x play no part in the recurrence, and
the node's ``k`` port is bound to C like the ``mamba2_summary_bwd`` op does.
For every packed sequence it returns ``d_initial_state`` ``[N, H, P, N_state]``,
the gradient of ``sum(o * dO)`` with respect to the span's incoming state from
a zero outgoing state gradient (the optional ``d_final_state`` seeds it
instead), and with ``output_transition=True`` the span ``transition``
``[N, H]``, the log2 of the span's decay product, so that state gradients
compose in reverse with one multiply::

    dX_j = exp2(transition_j) dX_{j+1} + d_initial_state_j

Here one logical sequence is cut into two spans that are packed as two
sequences. Each span's ``d_initial_state`` is checked against fp64 autograd
through the recurrence, its ``transition`` against the fp64 log2 decay sum,
and the reverse host-side composition against the fp64 gradient of the uncut
sequence with respect to its initial state.
"""

from __future__ import annotations

import cudnn
import torch


def _build_plans(g) -> None:
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    names = [g.get_plan_name_at_index(i) for i in range(len(g.plans))]
    g.select_plan(names.index("mamba2_summary_frost"))
    g.check_support()
    g.build_plans()


def _rms_ratio(out, ref):
    out, ref = out.detach().double(), ref.detach().double()
    return ((out - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt().clamp_min(1e-12)).item()


def _reference(q, k, v, dt_raw, A, dt_bias, cu, s0):
    """Differentiable fp64 per-token recurrence over the packed batch from the V-major seed ``s0``.
    Returns (o, log2 decay sum per span and head)."""
    dt = torch.nn.functional.softplus(dt_raw + dt_bias)
    log_alpha = A * dt
    outs, transition = [], []
    for i in range(cu.numel() - 1):
        S = s0[i].transpose(-2, -1)
        for t in range(int(cu[i]), int(cu[i + 1])):
            S = log_alpha[t].exp()[:, None, None] * S + dt[t][:, None, None] * torch.einsum("hn,hp->hnp", k[t], v[t])
            outs.append(torch.einsum("hn,hnp->hp", q[t], S))
        transition.append(log_alpha[int(cu[i]) : int(cu[i + 1])].sum(0) / torch.log(torch.tensor(2.0, dtype=torch.float64)))
    return torch.stack(outs, dim=0), torch.stack(transition, dim=0)


def main(spans=(192, 320), H: int = 2, N_state: int = 128, P: int = 64) -> None:
    torch.manual_seed(0)
    device = "cuda"
    total, num_spans = sum(spans), len(spans)

    q = (torch.randn(total, H, N_state, device=device) * 0.1).bfloat16()  # C
    k = torch.nn.functional.normalize(torch.randn(total, H, N_state, device=device), dim=-1).bfloat16()  # B
    v = torch.randn(total, H, P, device=device).bfloat16()  # x
    A = -(torch.rand(H, device=device) * 0.02 + 0.01)
    dt_bias = torch.rand(H, device=device) * 0.5
    dt_raw = (torch.rand(total, H, device=device) * 4.0 - 4.0).contiguous()
    do = torch.randn(total, H, P, device=device).bfloat16()
    cu = torch.tensor([0, *torch.tensor(spans).cumsum(0).tolist()], dtype=torch.int32, device=device)

    g = cudnn.pygraph()
    q_t = g.tensor([total, H, N_state], data_type=cudnn.data_type.BFLOAT16, name="q")
    k_t = g.tensor([total, H, N_state], data_type=cudnn.data_type.BFLOAT16, name="k")
    g_t = g.tensor([total, H], data_type=cudnn.data_type.FLOAT, name="g")
    a_t = g.tensor([H], data_type=cudnn.data_type.FLOAT, name="a")
    dt_bias_t = g.tensor([H], data_type=cudnn.data_type.FLOAT, name="dt_bias")
    cu_t = g.tensor([num_spans + 1], data_type=cudnn.data_type.INT32, name="cu_seqlens")
    do_t = g.tensor([total, H, P], data_type=cudnn.data_type.BFLOAT16, name="dO")
    dS0_t, transition_t = g.mamba2_summary_bwd(
        q=q_t,
        k=k_t,
        g=g_t,
        a=a_t,
        cu_seqlens=cu_t,
        dO=do_t,
        dt_bias=dt_bias_t,
        output_transition=True,
        name="mamba2_summary_bwd",
    )
    dS0_t.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    transition_t.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    _build_plans(g)

    d_initial_state = torch.empty(num_spans, H, P, N_state, dtype=torch.float32, device=device)
    transition = torch.empty(num_spans, H, dtype=torch.float32, device=device)
    pack = {q_t: q, k_t: q, g_t: dt_raw, a_t: A, dt_bias_t: dt_bias, cu_t: cu, do_t: do, dS0_t: d_initial_state, transition_t: transition}
    g.execute(pack, torch.empty(max(g.get_workspace_size(), 1), dtype=torch.uint8, device=device))
    torch.cuda.synchronize()

    qd, kd, vd, dtd, Ad, bd, dod = (x.double() for x in (q, k, v, dt_raw, A, dt_bias, do))
    s0 = torch.zeros(num_spans, H, P, N_state, dtype=torch.float64, device=device, requires_grad=True)
    o_ref, transition_ref = _reference(qd, kd, vd, dtd, Ad, bd, cu, s0)
    (dS0_ref,) = torch.autograd.grad((o_ref * dod).sum(), [s0])
    cu_uncut = torch.tensor([0, total], dtype=torch.int32, device=device)
    s0_uncut = torch.zeros(1, H, P, N_state, dtype=torch.float64, device=device, requires_grad=True)
    o_uncut, _ = _reference(qd, kd, vd, dtd, Ad, bd, cu_uncut, s0_uncut)
    (composed_ref,) = torch.autograd.grad((o_uncut * dod).sum(), [s0_uncut])
    composed = torch.exp2(transition[0].double())[:, None, None] * d_initial_state[1].double() + d_initial_state[0].double()
    r_g, r_m, r_x = _rms_ratio(d_initial_state, dS0_ref), _rms_ratio(transition, transition_ref), _rms_ratio(composed, composed_ref[0])
    assert r_g < 2e-2, f"d_initial_state rms ratio {r_g:.4g}"
    assert r_m < 1e-3, f"transition rms ratio {r_m:.4g}"
    assert r_x < 2e-2, f"composed d_initial_state rms ratio {r_x:.4g}"
    print(
        f"[20] PASS  mamba2 summary backward     spans={list(spans)} H={H} N={N_state} P={P} (dS0 rms {r_g:.2e}, transition rms {r_m:.2e}, composed rms {r_x:.2e})"
    )


if __name__ == "__main__":
    main()
