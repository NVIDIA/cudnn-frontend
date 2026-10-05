# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example 11: KDA (Kimi Delta Attention) forward summary (pure cuDNN frontend API).

The KDA_SUMMARY node is the state-only pass behind context parallelism, under
KDA's per-key-channel decay ``a_t = exp(g_t) in R^K``. For every packed
sequence it returns ``final_state`` ``[N, H, V, K]``, the state after the span
from a zero seed (or from the optional ``initial_state``), and with
``output_transition=True`` the span ``transition`` ``[N, H, K, K]`` in the
stored V-major domain, so that consecutive spans compose as::

    X_{j+1} = X_j @ transition_j + final_state_j

Here one logical sequence is cut into two spans that are packed as two
sequences. Each span's ``final_state`` is checked against the fp64
recurrence, its ``transition`` against the same recurrence run from the
identity with ``v = 0``, and the host-side composition against the fp64 final
state of the uncut sequence. ``use_qk_l2norm=False`` passes ``k`` through as
given, so this example feeds pre-normalized rows; decays near one keep the
transition well away from zero, so the cross-span term of the composition is
exercised.
"""

from __future__ import annotations

import cudnn
import torch


def _build_plans(g) -> None:
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    names = [g.get_plan_name_at_index(i) for i in range(len(g.plans))]
    g.select_plan(names.index("kda_summary_frost"))
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


def _reference_state(k, v, g, beta, cu, s0):
    """fp64 per-token recurrence over the packed batch from the V-major seed ``s0``. Returns the V-major final state."""
    total, H, D = k.shape
    V = v.shape[2]
    k, v, g, beta = (x.double() for x in (k, v, g, beta))
    fs = torch.zeros(cu.numel() - 1, H, V, D, dtype=torch.float64, device=k.device)
    for i in range(cu.numel() - 1):
        S = s0[i].double().transpose(-2, -1)
        for t in range(int(cu[i]), int(cu[i + 1])):
            S = g[t].exp()[..., None] * S  # per-key-channel decay first
            residual = v[t] - torch.einsum("hd,hdv->hv", k[t], S)
            S = S + beta[t][:, None, None] * torch.einsum("hd,hv->hdv", k[t], residual)
        fs[i] = S.transpose(-2, -1)
    return fs


def main(spans=(128, 192), H: int = 2, D: int = 128) -> None:
    torch.manual_seed(0)
    device = "cuda"
    total, num_spans = sum(spans), len(spans)

    k = torch.nn.functional.normalize(_randu(total * H, D, device), dim=-1).reshape(total, H, D).bfloat16()
    v = _randu(total * H, D, device).reshape(total, H, D).bfloat16()
    gate = torch.empty(total, H, D, device=device).uniform_(0.98, 1.0).log().contiguous()
    beta = torch.rand(total, H, device=device).sigmoid().contiguous()
    cu = torch.tensor([0, *torch.tensor(spans).cumsum(0).tolist()], dtype=torch.int32, device=device)

    g = cudnn.pygraph()
    k_t = g.tensor([total, H, D], data_type=cudnn.data_type.BFLOAT16, name="k")
    v_t = g.tensor([total, H, D], data_type=cudnn.data_type.BFLOAT16, name="v")
    g_t = g.tensor([total, H, D], data_type=cudnn.data_type.FLOAT, name="g")
    beta_t = g.tensor([total, H], data_type=cudnn.data_type.FLOAT, name="beta")
    cu_t = g.tensor([num_spans + 1], data_type=cudnn.data_type.INT32, name="cu_seqlens")
    fs_t, transition_t = g.kda_summary(
        k=k_t,
        v=v_t,
        g=g_t,
        beta=beta_t,
        cu_seqlens=cu_t,
        output_transition=True,
        use_qk_l2norm=False,
        name="kda_summary",
    )
    fs_t.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    transition_t.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    _build_plans(g)

    final_state = torch.empty(num_spans, H, D, D, dtype=torch.float32, device=device)
    transition = torch.empty(num_spans, H, D, D, dtype=torch.float32, device=device)
    pack = {k_t: k, v_t: v, g_t: gate, beta_t: beta, cu_t: cu, fs_t: final_state, transition_t: transition}
    g.execute(pack, torch.empty(max(g.get_workspace_size(), 1), dtype=torch.uint8, device=device))
    torch.cuda.synchronize()

    zero = torch.zeros(num_spans, H, D, D, dtype=torch.float64, device=device)
    eye = torch.eye(D, dtype=torch.float64, device=device).expand(num_spans, H, D, D)
    fs_ref = _reference_state(k, v, gate, beta, cu, zero)
    transition_ref = _reference_state(k, torch.zeros_like(v), gate, beta, cu, eye)
    cu_uncut = torch.tensor([0, total], dtype=torch.int32, device=device)
    composed_ref = _reference_state(k, v, gate, beta, cu_uncut, zero[:1])[0]
    composed = final_state[0].double() @ transition[1].double() + final_state[1].double()
    r_h, r_m, r_x = _rms_ratio(final_state, fs_ref), _rms_ratio(transition, transition_ref), _rms_ratio(composed, composed_ref)
    assert r_h < 5e-2, f"final_state rms ratio {r_h:.4g}"
    assert r_m < 2e-2, f"transition rms ratio {r_m:.4g}"
    assert r_x < 5e-2, f"composed final state rms ratio {r_x:.4g}"
    print(f"[11] PASS  kda summary                 spans={list(spans)} H={H} D={D} (fs rms {r_h:.2e}, transition rms {r_m:.2e}, composed rms {r_x:.2e})")


if __name__ == "__main__":
    main()
