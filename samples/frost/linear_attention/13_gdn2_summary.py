# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example 13: GDN-2 (Gated DeltaNet v2) forward summary (pure cuDNN frontend API).

The GDN2_SUMMARY node is the state-only pass behind context parallelism,
under GDN-2's per-key decay ``a_t = exp(g_t) in R^K``, per-key erase gate
``beta_t in R^K`` and per-value write gate ``w_t in R^V``. For every packed
sequence it returns ``final_state`` ``[N, H, V, K]``, the state after the
span from a zero seed (or from the optional ``initial_state``), and with
``output_transition=True`` the span ``transition`` ``[N, H, K, K]`` in the
stored V-major domain, so that consecutive spans compose as::

    X_{j+1} = X_j @ transition_j + final_state_j

A summary must read its gates exactly as the main op does, so this example
runs the same in-kernel k L2 norm, bounded safe gate and erase-side beta
safeguard (``beta_guard``) as example 05: with ``safe_gate=True`` the kernel
computes

    g = -5 * sigmoid(exp(A_log) * (a + dt_bias))

here with ``A_log = 0`` and ``dt_bias = 0``, and the fp64 reference applies
the same activation and projection. Here one logical sequence is cut into two
spans that are packed as two sequences. Each span's ``final_state`` is
checked against the fp64 recurrence, its ``transition`` against the same
recurrence run from the identity with ``v = 0``, and the host-side
composition against the fp64 final state of the uncut sequence.
Pre-activations far below zero keep the decays near one and the transition
well away from zero; odd tokens carry planted near-zero-decay channels, so
the guard fires on them and stays quiet on the even tokens.
"""

from __future__ import annotations

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


def _reference_state(k, v, g, beta, w, a_log, dt_bias, cu, s0):
    """fp64 per-token recurrence over the packed batch from the V-major seed ``s0``, with the safe-gate
    activation, in-kernel L2 norm, and beta guard applied. Returns the V-major final state."""
    total, H, D = k.shape
    V = v.shape[2]
    k, v, g, beta, w = (x.double() for x in (k, v, g, beta, w))
    g = -5.0 * torch.sigmoid(a_log.double().exp()[None, :, None] * (g + dt_bias.double()[None]))
    k = torch.nn.functional.normalize(k, dim=-1)
    beta = _beta_guard(k, beta, g.exp(), torch.bfloat16)
    fs = torch.zeros(cu.numel() - 1, H, V, D, dtype=torch.float64, device=k.device)
    for i in range(cu.numel() - 1):
        S = s0[i].double().transpose(-2, -1)
        for t in range(int(cu[i]), int(cu[i + 1])):
            S = g[t].exp()[..., None] * S  # per-key-channel decay first
            erase = torch.einsum("hd,hdv->hv", beta[t] * k[t], S)
            v_new = w[t] * v[t] - erase
            S = S + torch.einsum("hd,hv->hdv", k[t], v_new)
        fs[i] = S.transpose(-2, -1)
    return fs


def main(spans=(128, 192), H: int = 2, D: int = 128) -> None:
    torch.manual_seed(0)
    device = "cuda"
    total, num_spans = sum(spans), len(spans)

    k = _randu(total * H, D, device).reshape(total, H, D).bfloat16()
    v = _randu(total * H, D, device).reshape(total, H, D).bfloat16()
    gate = -6.5 + 0.7 * torch.randn(total, H, D, device=device)
    gate[1::2, :, :4] = -12.0
    gate = gate.contiguous()
    a_log = torch.zeros(H, device=device, dtype=torch.float32)
    dt_bias = torch.zeros(H, D, device=device, dtype=torch.float32)
    beta = (torch.rand(total, H, D, device=device).sigmoid() * 2.0).bfloat16().contiguous()
    w = torch.rand(total, H, D, device=device).sigmoid().bfloat16().contiguous()
    cu = torch.tensor([0, *torch.tensor(spans).cumsum(0).tolist()], dtype=torch.int32, device=device)

    g = cudnn.pygraph()
    k_t = g.tensor([total, H, D], data_type=cudnn.data_type.BFLOAT16, name="k")
    v_t = g.tensor([total, H, D], data_type=cudnn.data_type.BFLOAT16, name="v")
    g_t = g.tensor([total, H, D], data_type=cudnn.data_type.FLOAT, name="g")
    beta_t = g.tensor([total, H, D], data_type=cudnn.data_type.BFLOAT16, name="beta")
    w_t = g.tensor([total, H, D], data_type=cudnn.data_type.BFLOAT16, name="w")
    cu_t = g.tensor([num_spans + 1], data_type=cudnn.data_type.INT32, name="cu_seqlens")
    a_log_t = g.tensor([H], data_type=cudnn.data_type.FLOAT, name="a_log")
    dt_bias_t = g.tensor([H, D], data_type=cudnn.data_type.FLOAT, name="dt_bias")
    fs_t, transition_t = g.gdn2_summary(
        k=k_t,
        v=v_t,
        g=g_t,
        beta=beta_t,
        w=w_t,
        cu_seqlens=cu_t,
        a_log=a_log_t,
        dt_bias=dt_bias_t,
        output_transition=True,
        use_qk_l2norm=True,
        safe_gate=True,
        beta_guard=True,
        name="gdn2_summary",
    )
    fs_t.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    transition_t.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    _build_plans(g)

    final_state = torch.empty(num_spans, H, D, D, dtype=torch.float32, device=device)
    transition = torch.empty(num_spans, H, D, D, dtype=torch.float32, device=device)
    pack = {k_t: k, v_t: v, g_t: gate, beta_t: beta, w_t: w, cu_t: cu, a_log_t: a_log, dt_bias_t: dt_bias, fs_t: final_state, transition_t: transition}
    g.execute(pack, torch.empty(max(g.get_workspace_size(), 1), dtype=torch.uint8, device=device))
    torch.cuda.synchronize()

    zero = torch.zeros(num_spans, H, D, D, dtype=torch.float64, device=device)
    eye = torch.eye(D, dtype=torch.float64, device=device).expand(num_spans, H, D, D)
    fs_ref = _reference_state(k, v, gate, beta, w, a_log, dt_bias, cu, zero)
    transition_ref = _reference_state(k, torch.zeros_like(v), gate, beta, w, a_log, dt_bias, cu, eye)
    cu_uncut = torch.tensor([0, total], dtype=torch.int32, device=device)
    composed_ref = _reference_state(k, v, gate, beta, w, a_log, dt_bias, cu_uncut, zero[:1])[0]
    composed = final_state[0].double() @ transition[1].double() + final_state[1].double()
    r_h, r_m, r_x = _rms_ratio(final_state, fs_ref), _rms_ratio(transition, transition_ref), _rms_ratio(composed, composed_ref)
    assert r_h < 5e-2, f"final_state rms ratio {r_h:.4g}"
    assert r_m < 2e-2, f"transition rms ratio {r_m:.4g}"
    assert r_x < 5e-2, f"composed final state rms ratio {r_x:.4g}"
    print(
        f"[13] PASS  gdn2 summary (safe gate + beta guard)  spans={list(spans)} H={H} D={D} (fs rms {r_h:.2e}, transition rms {r_m:.2e}, composed rms {r_x:.2e})"
    )


if __name__ == "__main__":
    main()
