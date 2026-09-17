# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example 05: GDN-2 (Gated DeltaNet v2) prefill (pure cuDNN frontend API).

GDN-2 gates every channel: per-key decay ``a_t = exp(g_t) in R^K``, per-key
erase gate ``beta_t in R^K``, and per-value write gate ``w_t in R^V``::

    S'  = Diag(a_t) S_{t-1}
    S_t = S' + k_t^T (w_t . v_t - (beta_t . k_t) S')
    o_t = q_t S_t

``beta``/``w`` are io-dtype post-sigmoid tensors. This example runs the
in-kernel q/k L2 norm, the bounded safe gate, and the erase-side beta
safeguard (``beta_guard``). The decay input is a raw pre-activation; with
``safe_gate=True`` the kernel computes the bounded gate

    g = -5 * sigmoid(exp(A_log) * (a + dt_bias))

here with ``A_log = 0`` and ``dt_bias = 0``. The guard shrinks tokens whose
per-channel beta contrast would make the decayed erase step expansive toward
the key-weighted mean beta. Even tokens keep real decay headroom (guard
stays quiet); odd tokens carry planted near-zero-decay channels (guard
fires); the fp64 reference applies the same activation and projection.
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


def _randu(rows, dim, device):
    """Per-row uniform [-0.25, 0.25) with normally-distributed means: mildly
    heterogeneous data that keeps the recurrence stable."""
    means = torch.randn(rows, 1, device=device) * 0.05
    return means + torch.rand(rows, dim, device=device) * 0.5 - 0.25


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


def _reference(q, k, v, g, beta, w, a_log, dt_bias, cu, scale):
    """fp64 per-token recurrence over the packed batch, with the safe-gate
    activation, in-kernel L2 norm, and beta guard applied. Returns
    (o, final_state)."""
    total, H, D = q.shape
    V = v.shape[2]
    q, k, v, g, beta, w = (x.double() for x in (q, k, v, g, beta, w))
    g = -5.0 * torch.sigmoid(a_log.double().exp()[None, :, None] * (g + dt_bias.double()[None]))
    q = torch.nn.functional.normalize(q, dim=-1)
    k = torch.nn.functional.normalize(k, dim=-1)
    beta = _beta_guard(k, beta, g.exp(), torch.bfloat16)
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
        fs[n] = S.transpose(-2, -1)
    return o, fs


def main(seq_lens=(192, 320), H: int = 2, D: int = 128) -> None:
    torch.manual_seed(0)
    device = "cuda"
    total, num_seqs = sum(seq_lens), len(seq_lens)
    scale = 1.0 / math.sqrt(D)

    q = _randu(total * H, D, device).reshape(total, H, D).bfloat16()
    k = _randu(total * H, D, device).reshape(total, H, D).bfloat16()
    v = _randu(total * H, D, device).reshape(total, H, D).bfloat16()
    gate = -2.5 + 0.7 * torch.randn(total, H, D, device=device)
    gate[1::2, :, :4] = -8.0
    gate = gate.contiguous()
    a_log = torch.zeros(H, device=device, dtype=torch.float32)
    dt_bias = torch.zeros(H, D, device=device, dtype=torch.float32)
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
    a_log_t = g.tensor([H], data_type=cudnn.data_type.FLOAT, name="a_log")
    dt_bias_t = g.tensor([H, D], data_type=cudnn.data_type.FLOAT, name="dt_bias")
    O_t, fs_t, _h_t = g.gdn2(
        q=q_t,
        k=k_t,
        v=v_t,
        g=g_t,
        beta=beta_t,
        w=w_t,
        cu_seqlens=cu_t,
        a_log=a_log_t,
        dt_bias=dt_bias_t,
        scale=scale,
        output_final_state=True,
        use_qk_l2norm=True,
        safe_gate=True,
        beta_guard=True,
        name="gdn2",
    )
    O_t.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    fs_t.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    _build_plans(g)

    o = torch.empty(total, H, D, dtype=torch.bfloat16, device=device)
    fs = torch.empty(num_seqs, H, D, D, dtype=torch.float32, device=device)
    pack = {q_t: q, k_t: k, v_t: v, g_t: gate, beta_t: beta, w_t: w, cu_t: cu, a_log_t: a_log, dt_bias_t: dt_bias, O_t: o, fs_t: fs}
    g.execute(pack, torch.empty(max(g.get_workspace_size(), 1), dtype=torch.uint8, device=device))
    torch.cuda.synchronize()

    o_ref, fs_ref = _reference(q, k, v, gate, beta, w, a_log, dt_bias, cu, scale)
    r_o = _rms_ratio(o, o_ref)
    assert r_o < 5e-2, f"o rms ratio {r_o:.4g}"
    r_s = _rms_ratio(fs, fs_ref)
    assert r_s < 5e-2, f"final_state rms ratio {r_s:.4g}"
    print(f"[05] PASS  gdn2 prefill (safe gate + beta guard)  seq_lens={list(seq_lens)} H={H} D={D} (fs rms {r_s:.2e})")


if __name__ == "__main__":
    main()
