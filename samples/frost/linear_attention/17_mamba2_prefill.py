# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example 17: Mamba-2 SSD prefill (pure cuDNN frontend API).

Mamba-2 is the gated delta rule without the erase term: one per-token scalar ``dt`` is both the
decay exponent and the write strength::

    dt_t = softplus(dt_t + dt_bias)            (clamped to [dt_min, dt_max] when given)
    S_t  = exp(A dt_t) S_{t-1} + dt_t k_t^T v_t
    o_t  = q_t S_t  (+ D x_t) (* z_t sigmoid(z_t))

In the graph API's delta-rule vocabulary ``q`` = C, ``k`` = B, ``v`` = x and ``g`` = the raw
``dt``; the node takes the decay rate ``a`` (A) itself.
THD layout: ``[total, heads, dim]`` with ``cu_seqlens`` sequence boundaries; B / C may be
grouped over the x heads.  The final state comes back V-major ``[N, H, P, N_state]``.
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


def _reference(q, k, v, dt_raw, A, dt_bias, cu):
    """fp64 per-token recurrence over the packed batch. Returns (o, final_state) with a V-major state."""
    total, H, P = v.shape
    N_state = q.shape[2]
    q, k, v, dt_raw, A, dt_bias = (x.double() for x in (q, k, v, dt_raw, A, dt_bias))
    dt = torch.nn.functional.softplus(dt_raw + dt_bias)
    alpha = torch.exp(A * dt)
    o = torch.zeros(total, H, P, dtype=torch.float64, device=q.device)
    fs = torch.zeros(cu.numel() - 1, H, P, N_state, dtype=torch.float64, device=q.device)
    for i in range(cu.numel() - 1):
        S = torch.zeros(H, N_state, P, dtype=torch.float64, device=q.device)
        for t in range(int(cu[i]), int(cu[i + 1])):
            S = alpha[t][:, None, None] * S + dt[t][:, None, None] * torch.einsum("hn,hp->hnp", k[t], v[t])
            o[t] = torch.einsum("hn,hnp->hp", q[t], S)
        fs[i] = S.transpose(-2, -1)
    return o, fs


def main(seq_lens=(192, 320), H: int = 2, N_state: int = 128, P: int = 64) -> None:
    torch.manual_seed(0)
    device = "cuda"
    total, num_seqs = sum(seq_lens), len(seq_lens)

    q = (torch.randn(total, H, N_state, device=device) * 0.1).bfloat16()  # C
    k = torch.nn.functional.normalize(torch.randn(total, H, N_state, device=device), dim=-1).bfloat16()  # B
    v = torch.randn(total, H, P, device=device).bfloat16()  # x
    A = -(torch.rand(H, device=device) * 7.0 + 1.0)
    dt_bias = torch.rand(H, device=device) * 0.5
    dt_raw = (torch.rand(total, H, device=device) * 4.0 - 4.0).contiguous()
    cu = torch.tensor([0, *torch.tensor(seq_lens).cumsum(0).tolist()], dtype=torch.int32, device=device)

    g = cudnn.pygraph()
    q_t = g.tensor([total, H, N_state], data_type=cudnn.data_type.BFLOAT16, name="q")
    k_t = g.tensor([total, H, N_state], data_type=cudnn.data_type.BFLOAT16, name="k")
    v_t = g.tensor([total, H, P], data_type=cudnn.data_type.BFLOAT16, name="v")
    g_t = g.tensor([total, H], data_type=cudnn.data_type.FLOAT, name="g")
    a_t = g.tensor([H], data_type=cudnn.data_type.FLOAT, name="a")
    dt_bias_t = g.tensor([H], data_type=cudnn.data_type.FLOAT, name="dt_bias")
    cu_t = g.tensor([num_seqs + 1], data_type=cudnn.data_type.INT32, name="cu_seqlens")
    O_t, fs_t, _ckpt_t, _pre_t = g.mamba2(
        q=q_t,
        k=k_t,
        v=v_t,
        g=g_t,
        a=a_t,
        cu_seqlens=cu_t,
        dt_bias=dt_bias_t,
        output_final_state=True,
        name="mamba2",
    )
    O_t.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    fs_t.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    _build_plans(g)

    o = torch.empty(total, H, P, dtype=torch.bfloat16, device=device)
    fs = torch.empty(num_seqs, H, P, N_state, dtype=torch.float32, device=device)
    pack = {q_t: q, k_t: k, v_t: v, g_t: dt_raw, a_t: A, dt_bias_t: dt_bias, cu_t: cu, O_t: o, fs_t: fs}
    g.execute(pack, torch.empty(max(g.get_workspace_size(), 1), dtype=torch.uint8, device=device))
    torch.cuda.synchronize()

    o_ref, fs_ref = _reference(q, k, v, dt_raw, A, dt_bias, cu)
    r_o, r_s = _rms_ratio(o, o_ref), _rms_ratio(fs, fs_ref)
    assert r_o < 2e-2, f"o rms ratio {r_o:.4g}"
    assert r_s < 2e-2, f"final_state rms ratio {r_s:.4g}"
    print(f"[17] PASS  mamba2 prefill              seq_lens={list(seq_lens)} H={H} N={N_state} P={P} (o rms {r_o:.2e}, fs rms {r_s:.2e})")


if __name__ == "__main__":
    main()
