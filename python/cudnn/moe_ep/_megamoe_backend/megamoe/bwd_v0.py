# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""M1 pool-reuse backward (BWD_DESIGN.md): the megakernel's dataflow, in torch.

Same gradient semantics as ``fp8_bwd.fp8_backward`` (mxfp8 on every GEMM
operand) but organized the way the backward megakernel will run, consuming
the forward kernel's persistent pools instead of rebuilding EP state:

- NO routing plan / dispatch of x: pool row -> token mapping comes from
  ``token_src_metadata``; the pool layout (128-aligned expert slots) IS the
  grouped-GEMM ``offs``.
- NO fc1/act/y recompute GEMMs: raw gate+up comes from the ``generate_c``
  stash ``fc1_c``; d(topk_weights) = <gemm1_out_pre_tw, silu(g)*u> needs no y.
- dout reaches pool order by one gather (world=1) or one allgather + gather
  (world>1) — the kernel replaces this with metadata-driven peer pulls.
- backward GEMM chain:  dA = tw * (doutg @ W2^T)   [gmm 2d3d, K=H]
                        du,dg = SwiGLU'(dA; g,u)   [elemwise]
                        dxg  = [du,dg] @ W13^T     [gmm 2d3d, K=2I]
  wgrads on the same pool-ordered operands [gmm 2d2d, K=tokens].

Weight quantizations (W2^T along H, W13^T along 2I) are cached on the layer
until ``refresh_weights``; with turboquant they are built in the rotated
basis and dX/dW13 are rotated back through Q^T (same contract as fp8_bwd).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

import torch.distributed as dist

import megamoe.repo_path  # noqa: F401

from pt.quant import rotate_trailing

from megamoe.fp8_bwd import (
    _col_atom_order,
    _phase,
    _round_up,
    gmm_2d3d,
    gmm_wgrad_2d2d,
    quant_weights_3d,
)
from megamoe.pools import decode_token_src_metadata, local_pool_views
from megamoe.training import dequant_mxfp8_pool


def _weight_cache(layer):
    """Only the two dgrad operands (no recompute weights needed)."""
    wc = getattr(layer, "_pool_bwd_wcache", None)
    if wc is not None:
        return wc
    with _phase("weight-quant"):
        w13 = layer.w13.detach().to(torch.bfloat16)
        if layer.qcfg.turboquant:
            w13 = rotate_trailing(w13, layer.q_rot)
        w2 = layer.w2.detach().to(torch.bfloat16)
        wc = {
            "w2t_kh": quant_weights_3d(w2.transpose(1, 2).contiguous()),      # (E,I,H)  K=H
            "w13t_k2i": quant_weights_3d(w13.transpose(1, 2).contiguous()),   # (E,H,2I) K=2I
        }
    layer._pool_bwd_wcache = wc
    return wc


def pool_backward(layer, topk_ids, topk_weights, dout, T):
    """Metadata-driven mxfp8 backward; returns (dx, dtw, dw13, dw2)."""
    ep = layer.ep_cfg
    H, I, K = ep.hidden_size, ep.intermediate_size, ep.top_k
    E_local = ep.num_experts // ep.ep_size
    expert_start = ep.ep_rank * E_local
    world = ep.ep_size
    fwd = layer._fwd
    device = dout.device
    group = ep.process_group

    with _phase("prep"):
        # global routing counts -> this rank's pool layout (host-side; the
        # kernel version inherits these from the forward launch instead).
        if world > 1:
            ids_all = torch.empty(
                (world, T, K), dtype=topk_ids.dtype, device=device
            )
            dist.all_gather_into_tensor(ids_all, topk_ids.contiguous(), group=group)
        else:
            ids_all = topk_ids.view(1, T, K)
        local = ids_all.reshape(-1) - expert_start
        counts = torch.bincount(
            local[(local >= 0) & (local < E_local)], minlength=E_local
        ).tolist()

        padded = [_round_up(n, 128) for n in counts]
        Mp = max(sum(padded), 128)
        offs = torch.tensor(
            [sum(padded[: i + 1]) for i in range(E_local)],
            device=device, dtype=torch.int32,
        )
        doffs = [sum(padded[:i]) for i in range(E_local)]
        valid = torch.cat(
            [torch.arange(o, o + n, device=device) for o, n in zip(doffs, counts)]
        ) if sum(counts) else torch.empty(0, dtype=torch.long, device=device)

        lv = local_pool_views(fwd)
        src_rank, src_token, src_topk, _, _ = decode_token_src_metadata(
            lv["token_src_metadata"][:Mp]
        )
        sr = src_rank[valid].long()
        st = src_token[valid].long()
        sk = src_topk[valid].long()
        tw_rows = lv["l1_topk_weights_buffer"][:Mp][valid].float()

        wc = _weight_cache(layer)
        w2tq_kh, sw2t_kh = wc["w2t_kh"]
        w13tq_k2i, sw13t_k2i = wc["w13t_k2i"]

        # wgrad scale atom permutations for the pool layout
        cbt = Mp // 128
        order_h = _col_atom_order(offs, H // 128, cbt)
        order_2i = _col_atom_order(offs, (2 * I) // 128, cbt)
        order_i = _col_atom_order(offs, I // 128, cbt)

    with torch.no_grad():
        with _phase("dispatch"):
            # dout (and, for wgrad, the quantized x) in pool order by gather
            dout_b = dout.to(torch.bfloat16)
            x_q = dequant_mxfp8_pool(
                fwd.my_activation[:T], fwd.my_activation_sf[:T], H
            ).to(torch.bfloat16)
            if world > 1:
                dout_all = torch.empty((world * T, H), dtype=dout_b.dtype, device=device)
                dist.all_gather_into_tensor(dout_all, dout_b.contiguous(), group=group)
                x_all = torch.empty((world * T, H), dtype=x_q.dtype, device=device)
                dist.all_gather_into_tensor(x_all, x_q, group=group)
                flat = sr * T + st
            else:
                dout_all, x_all, flat = dout_b, x_q, st
            DOUTG = dout_b.new_zeros((Mp, H))
            DOUTG.index_copy_(0, valid, dout_all[flat])
            XG = x_q.new_zeros((Mp, H))
            XG.index_copy_(0, valid, x_all[flat])

        # gate/up from the forward stash (raw, pre-tw; kernel interleave)
        with _phase("elemwise"):
            pair = fwd.fc1_c[:Mp].view(Mp, I // 32, 2, 32)
            gate = pair[:, :, 0].reshape(Mp, I).float()
            up = pair[:, :, 1].reshape(Mp, I).float()
            act_raw = F.silu(gate) * up                       # silu(g)*u, no tw
            ACTW = act_raw.new_zeros((Mp, I), dtype=torch.bfloat16)
            ACTW.index_copy_(
                0, valid, (act_raw[valid] * tw_rows[:, None]).to(torch.bfloat16)
            )

        # dgrad chain
        G1 = gmm_2d3d(DOUTG, offs, w2tq_kh, sw2t_kh)          # doutg @ W2^T (pre-tw)
        with _phase("elemwise"):
            dtw_rows = (G1[valid].float() * act_raw[valid]).sum(dim=-1)
            dA = G1.float()
            dA[valid] *= tw_rows[:, None]
            sg = torch.sigmoid(gate)
            dgate = dA * up * sg * (1 + gate * (1 - sg))
            dup = dA * F.silu(gate)
            # pad rows: dA==0 there (zero DOUTG rows, zero fc1_c rows — probe
            # C5), and every term multiplies dA, so DFC1 pads are zero.
            DFC1 = torch.cat([dup, dgate], dim=-1).to(torch.bfloat16)  # pt [lin|gate]
        DXG = gmm_2d3d(DFC1, offs, w13tq_k2i, sw13t_k2i)      # (Mp, H)

        # wgrads (tokens on K); pool rows are zero-padded by construction
        dw13 = gmm_wgrad_2d2d(DFC1, XG, offs, order_2i, order_h)
        if layer.qcfg.turboquant:
            dw13 = rotate_trailing(dw13.float(), layer.q_rot.t())
        dw13 = dw13.to(layer.w13.dtype)
        dw2 = gmm_wgrad_2d2d(DOUTG, ACTW, offs, order_h, order_i).to(layer.w2.dtype)

        with _phase("comm-adj"):
            # dx: sum pool contributions back to (src_rank, src_token)
            dx_all = torch.zeros((world * T, H), device=device, dtype=torch.float32)
            dx_all.index_add_(0, flat, DXG[valid].float())
            # dtw: scatter scalars back to (src_rank, src_token, src_topk)
            dtw_all = torch.zeros((world * T, K), device=device, dtype=torch.float32)
            dtw_all[flat, sk] = dtw_rows
            if world > 1:
                dist.all_reduce(dx_all, group=group)
                dist.all_reduce(dtw_all, group=group)
            r0 = ep.ep_rank * T
            dx = dx_all[r0 : r0 + T]
            dtw = dtw_all[r0 : r0 + T]
            if layer.qcfg.turboquant:
                dx = rotate_trailing(dx, layer.q_rot.t())

    return dx.to(torch.bfloat16), dtw, dw13, dw2
