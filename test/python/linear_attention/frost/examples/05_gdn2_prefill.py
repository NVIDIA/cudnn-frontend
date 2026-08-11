# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example 05: GDN-2 (Gated DeltaNet v2) prefill (pure cuDNN frontend API).

GDN-2 gates every channel: per-key decay ``a_t = exp(g_t) in R^K``, per-key
erase gate ``beta_t in R^K``, and per-value write gate ``w_t in R^V``::

    S'  = Diag(a_t) S_{t-1}
    S_t = S' + k_t^T (w_t . v_t - (beta_t . k_t) S')
    o_t = q_t S_t

``beta``/``w`` are io-dtype post-sigmoid tensors; ``use_qk_l2norm=False``
passes q/k through as given, so this example feeds pre-normalized rows.
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
    g.select_plan(names.index("gdn2_frost"))  # pin the FROST entry
    g.check_support()
    g.build_plans()


def _rms_ratio(out, ref):
    out, ref = out.detach().double(), ref.detach().double()
    return ((out - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt().clamp_min(1e-12)).item()


def _randu(rows, dim, device):
    """Per-row uniform [-0.25, 0.25) with normally-distributed means: mildly
    heterogeneous data that keeps the recurrence stable."""
    means = torch.randn(rows, 1, device=device) * 0.05
    return means + torch.rand(rows, dim, device=device) * 0.5 - 0.25


def _reference(q, k, v, g, beta, w, cu, scale):
    """fp64 per-token recurrence over the packed batch. Returns (o, final_state)."""
    total, H, D = q.shape
    V = v.shape[2]
    q, k, v, g, beta, w = (x.double() for x in (q, k, v, g, beta, w))
    o = torch.zeros(total, H, V, dtype=torch.float64, device=q.device)
    fs = torch.zeros(cu.numel() - 1, H, D, V, dtype=torch.float64, device=q.device)
    for n in range(cu.numel() - 1):
        S = torch.zeros(H, D, V, dtype=torch.float64, device=q.device)
        for t in range(int(cu[n]), int(cu[n + 1])):
            S = g[t].exp()[..., None] * S  # per-key-channel decay first
            erase = torch.einsum("hd,hdv->hv", beta[t] * k[t], S)
            v_new = w[t] * v[t] - erase
            S = S + torch.einsum("hd,hv->hdv", k[t], v_new)
            o[t] = torch.einsum("hd,hdv->hv", q[t] * scale, S)
        fs[n] = S
    return o, fs


def main(seq_lens=(192, 320), H: int = 2, D: int = 128) -> None:
    torch.manual_seed(0)
    device = "cuda"
    total, num_seqs = sum(seq_lens), len(seq_lens)
    scale = 1.0 / math.sqrt(D)

    q = torch.nn.functional.normalize(_randu(total * H, D, device), dim=-1).reshape(total, H, D).bfloat16()
    k = torch.nn.functional.normalize(_randu(total * H, D, device), dim=-1).reshape(total, H, D).bfloat16()
    v = _randu(total * H, D, device).reshape(total, H, D).bfloat16()
    gate = torch.empty(total, H, D, device=device).uniform_(0.5, 1.0).log().contiguous()
    beta = (torch.rand(total, H, D, device=device).sigmoid() * 2.0).bfloat16().contiguous()
    w = torch.rand(total, H, D, device=device).sigmoid().bfloat16().contiguous()
    cu = torch.tensor([0, *torch.tensor(seq_lens).cumsum(0).tolist()], dtype=torch.int32, device=device)

    g = cudnn.pygraph()
    q_t = g.tensor([total, H, D], data_type=cudnn.data_type.BFLOAT16, name="q")
    k_t = g.tensor([total, H, D], data_type=cudnn.data_type.BFLOAT16, name="k")
    v_t = g.tensor([total, H, D], data_type=cudnn.data_type.BFLOAT16, name="v")
    g_t = g.tensor([total, H, D], data_type=cudnn.data_type.FLOAT, name="g")
    beta_t = g.tensor([total, H, D], data_type=cudnn.data_type.BFLOAT16, name="beta")
    w_t = g.tensor([total, H, D], data_type=cudnn.data_type.BFLOAT16, name="w")
    cu_t = g.tensor([num_seqs + 1], data_type=cudnn.data_type.INT32, name="cu_seqlens")
    O_t, fs_t, _h_t = g.gdn2(
        q=q_t,
        k=k_t,
        v=v_t,
        g=g_t,
        beta=beta_t,
        w=w_t,
        cu_seqlens=cu_t,
        scale=scale,
        output_final_state=True,
        use_qk_l2norm=False,
        name="gdn2",
    )
    O_t.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    fs_t.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    _build_plans(g)

    o = torch.empty(total, H, D, dtype=torch.bfloat16, device=device)
    fs = torch.empty(num_seqs, H, D, D, dtype=torch.float32, device=device)
    pack = {q_t: q, k_t: k, v_t: v, g_t: gate, beta_t: beta, w_t: w, cu_t: cu, O_t: o, fs_t: fs}
    g.execute(pack, torch.empty(max(g.get_workspace_size(), 1), dtype=torch.uint8, device=device))
    torch.cuda.synchronize()

    o_ref, fs_ref = _reference(q, k, v, gate, beta, w, cu, scale)
    r_o = _rms_ratio(o, o_ref)
    assert r_o < 5e-2, f"o rms ratio {r_o:.4g}"
    r_s = _rms_ratio(fs, fs_ref)
    assert r_s < 5e-2, f"final_state rms ratio {r_s:.4g}"
    print(f"[05] PASS  gdn2 prefill                seq_lens={list(seq_lens)} H={H} D={D} (fs rms {r_s:.2e})")


if __name__ == "__main__":
    main()
