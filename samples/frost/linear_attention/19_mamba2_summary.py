# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example 19: Mamba-2 SSD forward summary (pure cuDNN frontend API).

The MAMBA2_SUMMARY node is the state-only pass behind context parallelism: the
Mamba-2 state recurrence over every packed sequence without the output GEMMs.
In the graph API's delta-rule vocabulary ``k`` = B, ``v`` = x and ``g`` = the
raw ``dt`` (and ``a`` = A), exactly as the MAMBA2 node of example 17. For
every packed sequence it returns ``final_state`` ``[N, H, P, N_state]``, the
state after the span from a zero seed (or from the optional
``initial_state``), and with ``output_transition=True`` the span
``transition`` ``[N, H]``: Mamba-2's transition is a scalar per head, the log2
of the span's decay product ``exp(A sum_t dt_t)``, so that consecutive spans
compose with one multiply::

    X_{j+1} = exp2(transition_j) X_j + final_state_j

Here one logical sequence is cut into two spans that are packed as two
sequences. Each span's ``final_state`` is checked against the fp64 recurrence,
its ``transition`` against the fp64 log2 decay sum, and the host-side
composition against the fp64 final state of the uncut sequence. Decays near
one keep the transition well away from zero, so the cross-span term of the
composition is exercised.
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


def _reference_state(k, v, dt_raw, A, dt_bias, cu, s0):
    """fp64 per-token recurrence over the packed batch from the V-major seed ``s0``.
    Returns (final_state V-major, log2 decay sum per span and head)."""
    H, P = v.shape[1], v.shape[2]
    N_state = k.shape[2]
    k, v, dt_raw, A, dt_bias = (x.double() for x in (k, v, dt_raw, A, dt_bias))
    dt = torch.nn.functional.softplus(dt_raw + dt_bias)
    log_alpha = A * dt
    fs = torch.zeros(cu.numel() - 1, H, P, N_state, dtype=torch.float64, device=k.device)
    transition = torch.zeros(cu.numel() - 1, H, dtype=torch.float64, device=k.device)
    for i in range(cu.numel() - 1):
        S = s0[i].double().transpose(-2, -1)
        for t in range(int(cu[i]), int(cu[i + 1])):
            S = log_alpha[t].exp()[:, None, None] * S + dt[t][:, None, None] * torch.einsum("hn,hp->hnp", k[t], v[t])
        fs[i] = S.transpose(-2, -1)
        transition[i] = log_alpha[int(cu[i]) : int(cu[i + 1])].sum(0) / torch.log(torch.tensor(2.0, dtype=torch.float64))
    return fs, transition


def main(spans=(192, 320), H: int = 2, N_state: int = 128, P: int = 64) -> None:
    torch.manual_seed(0)
    device = "cuda"
    total, num_spans = sum(spans), len(spans)

    k = torch.nn.functional.normalize(torch.randn(total, H, N_state, device=device), dim=-1).bfloat16()  # B
    v = torch.randn(total, H, P, device=device).bfloat16()  # x
    A = -(torch.rand(H, device=device) * 0.02 + 0.01)
    dt_bias = torch.rand(H, device=device) * 0.5
    dt_raw = (torch.rand(total, H, device=device) * 4.0 - 4.0).contiguous()
    cu = torch.tensor([0, *torch.tensor(spans).cumsum(0).tolist()], dtype=torch.int32, device=device)

    g = cudnn.pygraph()
    k_t = g.tensor([total, H, N_state], data_type=cudnn.data_type.BFLOAT16, name="k")
    v_t = g.tensor([total, H, P], data_type=cudnn.data_type.BFLOAT16, name="v")
    g_t = g.tensor([total, H], data_type=cudnn.data_type.FLOAT, name="g")
    a_t = g.tensor([H], data_type=cudnn.data_type.FLOAT, name="a")
    dt_bias_t = g.tensor([H], data_type=cudnn.data_type.FLOAT, name="dt_bias")
    cu_t = g.tensor([num_spans + 1], data_type=cudnn.data_type.INT32, name="cu_seqlens")
    fs_t, transition_t = g.mamba2_summary(
        k=k_t,
        v=v_t,
        g=g_t,
        a=a_t,
        cu_seqlens=cu_t,
        dt_bias=dt_bias_t,
        output_transition=True,
        name="mamba2_summary",
    )
    fs_t.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    transition_t.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    _build_plans(g)

    final_state = torch.empty(num_spans, H, P, N_state, dtype=torch.float32, device=device)
    transition = torch.empty(num_spans, H, dtype=torch.float32, device=device)
    pack = {k_t: k, v_t: v, g_t: dt_raw, a_t: A, dt_bias_t: dt_bias, cu_t: cu, fs_t: final_state, transition_t: transition}
    g.execute(pack, torch.empty(max(g.get_workspace_size(), 1), dtype=torch.uint8, device=device))
    torch.cuda.synchronize()

    zero = torch.zeros(num_spans, H, P, N_state, dtype=torch.float64, device=device)
    fs_ref, transition_ref = _reference_state(k, v, dt_raw, A, dt_bias, cu, zero)
    cu_uncut = torch.tensor([0, total], dtype=torch.int32, device=device)
    composed_ref = _reference_state(k, v, dt_raw, A, dt_bias, cu_uncut, zero[:1])[0][0]
    composed = torch.exp2(transition[1].double())[:, None, None] * final_state[0].double() + final_state[1].double()
    r_h, r_m, r_x = _rms_ratio(final_state, fs_ref), _rms_ratio(transition, transition_ref), _rms_ratio(composed, composed_ref)
    assert r_h < 2e-2, f"final_state rms ratio {r_h:.4g}"
    assert r_m < 1e-3, f"transition rms ratio {r_m:.4g}"
    assert r_x < 2e-2, f"composed final state rms ratio {r_x:.4g}"
    print(
        f"[19] PASS  mamba2 summary              spans={list(spans)} H={H} N={N_state} P={P} (fs rms {r_h:.2e}, transition rms {r_m:.2e}, composed rms {r_x:.2e})"
    )


if __name__ == "__main__":
    main()
