# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Manual MXFP8 backward for the hybrid layer — real fp8 grouped GEMMs.

Replaces the autograd fake-quant replay with explicit phases running on
``torch._scaled_grouped_mm`` (validated contracts:
pt/tests/test_scaled_grouped_mm_mxfp8.py for 2d-2d wgrad, torch's
test_scaled_matmul_cuda.py recipe for 2d-3d):

    combine-adjoint a2a -> [2d-3d] fc1/act recompute -> [2d-3d] y for dTW
    -> [2d-3d] dgrad chain (SwiGLU-bwd in torch) -> [2d-2d] wgrads
    -> dispatch-adjoint a2a

Token groups are zero-padded to 128-row alignment; pad rows quantize to
zero blocks and contribute nothing. 128 (= 4 scale cols = 1 swizzle atom
col) makes the per-group to_blocked concatenation expressible as pure
tensor views: row-group packing IS the whole-tensor swizzle, and K-group
packing is one atom-permutation index_select — no per-group python loops.
All feature dims (H, 2I, I) must be multiples of 128. Weight
quantizations (both contraction axes) are cached on the layer until
``refresh_weights`` invalidates them.

Gradient semantics match the replay with ``quant_bprop=True`` (mxfp8 on
every GEMM operand, same scale convention); small deviations come from the
grouped GEMM's bf16 output rounding vs the replay's fp32 simulation.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn.functional as F

# Optional phase instrumentation: a bench installs a list here and phases
# append (name, start_event, end_event); None (default) is zero-overhead.
TIMER = None


@contextmanager
def _phase(name):
    if TIMER is None:
        yield
        return
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    try:
        yield
    finally:
        e.record()
        TIMER.append((name, s, e))

import megamoe.repo_path  # noqa: F401

from moe_nvfp4_swapab.runner_common import to_blocked

from pt.quant import MXFP8_BLOCK, quant_mxfp8_tensors, rotate_trailing

from megamoe.quant_kernels import mxfp8_rowquant, mxfp8_transquant


def _round_up(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


class GroupedPad:
    """Zero-pad a grouped-by-expert row tensor to align-multiple group sizes."""

    def __init__(self, tokens_per_expert, align: int, device):
        counts = [_round_up(n, align) for n in tokens_per_expert]
        self.total = sum(counts)
        self.offs = torch.tensor(
            [sum(counts[: i + 1]) for i in range(len(counts))],
            device=device, dtype=torch.int32,
        )
        idx, row = [], 0
        for n, c in zip(tokens_per_expert, counts):
            idx.append(torch.arange(row, row + n, device=device))
            row += c
        self.src_rows = (
            torch.cat(idx) if idx else torch.empty(0, device=device, dtype=torch.long)
        )

    def pack(self, t: torch.Tensor) -> torch.Tensor:
        out = t.new_zeros((self.total, t.shape[1]))
        out.index_copy_(0, self.src_rows, t)
        return out

    def unpack(self, t: torch.Tensor) -> torch.Tensor:
        return t.index_select(0, self.src_rows)


def _pack_scales_rowgroups_ref(s: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    """Loop reference: per-row-group to_blocked of (M, Kc) scales -> (-1, Kc)."""
    kc = s.shape[1]
    parts, lo = [], 0
    for hi in offs.tolist():
        if hi > lo:
            parts.append(to_blocked(s[lo:hi]))
        lo = hi
    return torch.cat(parts).reshape(-1, kc)


def _atoms(s: torch.Tensor) -> torch.Tensor:
    """(M, Kc) e8m0 (M%128==0, Kc%4==0) -> (M/128 * Kc/4, 512) swizzled atoms
    in row-block-major order (the 32x4x4 within-atom rearrange applied)."""
    rb, cb = s.shape[0] // 128, s.shape[1] // 4
    return (
        s.view(rb, 128, cb, 4).permute(0, 2, 1, 3)
        .reshape(-1, 4, 32, 4).transpose(1, 2).reshape(rb * cb, 512)
    )


def _pack_scales_rowgroups(s: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    """Vectorized per-row-group swizzle. With every group's row count a
    multiple of 128, the concatenation of per-group to_blocked layouts IS the
    whole-tensor swizzle (group row-blocks are contiguous), so no loop."""
    kc = s.shape[1]
    if kc % 4:
        raise ValueError(f"scale cols ({kc}) must be a multiple of 4 (K % 128 == 0)")
    if s.shape[0] % 128:
        raise ValueError("row groups must be padded to 128")
    return _atoms(s).reshape(-1, kc)


def quant_weights_3d(w: torch.Tensor):
    """(E, N, K) bf16 -> (fp8 data, stacked per-expert blocked scales)."""
    wq, sw = quant_mxfp8_tensors(w, dim=-1)
    sw_b = torch.stack([to_blocked(sw[e]) for e in range(w.shape[0])])
    return wq, sw_b


def gmm_2d3d(
    x: torch.Tensor,           # (Mtot, K) bf16, row groups 32-aligned
    offs: torch.Tensor,        # int32 group end offsets over M
    wq: torch.Tensor,          # (E, N, K) fp8 (from quant_weights_3d)
    sw_b: torch.Tensor,        # stacked blocked scales
) -> torch.Tensor:
    """out[rows g] = x[rows g] @ w[g].T -> (Mtot, N) bf16."""
    with _phase("act-quant"):
        xq, sx = mxfp8_rowquant(x)
        sx_b = _pack_scales_rowgroups(sx, offs)
    with _phase("gemm"):
        return torch._scaled_grouped_mm(
            xq, wq.transpose(-2, -1), sx_b, sw_b, offs=offs, out_dtype=torch.bfloat16
        )


def _pack_scales_colgroups_ref(
    s: torch.Tensor, offs: torch.Tensor, mn: int
) -> torch.Tensor:
    """Loop reference: per-K-group to_blocked of (MN, Ktot/32) scales -> 2D."""
    parts, lo = [], 0
    for hi in offs.tolist():
        if hi > lo:
            parts.append(to_blocked(s[:, lo // MXFP8_BLOCK : hi // MXFP8_BLOCK]))
        lo = hi
    return torch.cat(parts).reshape(_round_up(mn, 128), -1)


def _col_atom_order(offs: torch.Tensor, rb: int, cbt: int) -> torch.Tensor:
    """Atom permutation: row-block-major atoms -> per-K-group (group outer,
    row-block, col-block) order matching cat-of-per-group to_blocked."""
    device = offs.device
    rows = torch.arange(rb, device=device) * cbt
    parts, lo = [], 0
    for hi in offs.tolist():
        cb0, cb1 = lo // 128, hi // 128  # 128 tokens = 4 scale cols = 1 atom col
        if cb1 > cb0:
            cols = torch.arange(cb0, cb1, device=device)
            parts.append(
                (rows[:, None] + cols[None, :]).reshape(-1)
            )
        lo = hi
    return torch.cat(parts)


def _pack_scales_colgroups(
    s: torch.Tensor, offs: torch.Tensor, mn: int, atom_order: torch.Tensor
) -> torch.Tensor:
    """Vectorized per-K-group swizzle: whole-tensor atomization + one cached
    index_select. Requires MN%128==0 and 128-aligned K groups."""
    if mn % 128 or s.shape[1] % 4:
        raise ValueError("colgroup swizzle needs MN%128==0 and 128-aligned groups")
    return _atoms(s)[atom_order].reshape(mn, -1)


def gmm_wgrad_2d2d(
    dy: torch.Tensor,   # (Ktot, M) bf16, token groups 128-aligned
    act: torch.Tensor,  # (Ktot, N) bf16
    offs: torch.Tensor,
    atom_order_m: torch.Tensor,
    atom_order_n: torch.Tensor,
) -> torch.Tensor:
    """dW[e] = dy[ke].T @ act[ke] -> (E, M, N) bf16, tokens on K."""
    with _phase("act-quant"):
        a, sa = mxfp8_transquant(dy)
        bt, sb = mxfp8_transquant(act)
        sa_b = _pack_scales_colgroups(sa, offs, dy.shape[1], atom_order_m)
        sb_b = _pack_scales_colgroups(sb, offs, act.shape[1], atom_order_n)
    with _phase("gemm"):
        return torch._scaled_grouped_mm(
            a, bt.t(), sa_b, sb_b, offs=offs, out_dtype=torch.bfloat16
        )


def fp8_backward(layer, topk_ids, topk_weights, dout, T):
    """Manual mxfp8 backward; returns (dx, dtw, dw13, dw2)."""
    from pt.dispatch_combine import dispatch
    from pt.routing import build_routing_plan

    from megamoe.training import dequant_mxfp8_pool

    ep = layer.ep_cfg
    H, K = ep.hidden_size, ep.top_k
    fwd = layer._fwd
    comm = layer.comm

    with _phase("prep"):
        x_q = dequant_mxfp8_pool(
            fwd.my_activation[:T], fwd.my_activation_sf[:T], H
        ).to(torch.bfloat16)
        plan = build_routing_plan(topk_ids, ep, comm)
        pad = GroupedPad(plan.tokens_per_expert, 128, dout.device)

    with torch.no_grad():
        with _phase("dispatch"):
            # arrivals (reuse of the quantized pool payload) + dispatched dout
            xg = dispatch(x_q, plan, comm)
            doutg = dispatch(dout.to(torch.bfloat16), plan, comm)
            XG, DOUTG = pad.pack(xg), pad.pack(doutg)

            # wgrad scale atom permutations (routing-dependent, cheap)
            cbt = pad.total // 128
            order_h = _col_atom_order(pad.offs, H // 128, cbt)
            order_2i = _col_atom_order(pad.offs, (2 * ep.intermediate_size) // 128, cbt)
            order_i = _col_atom_order(pad.offs, ep.intermediate_size // 128, cbt)

        # weight quantizations (both contraction axes), cached until
        # refresh_weights() invalidates. With turboquant the pool activation
        # is in the ROTATED basis, so fc1-side weights are rotated to match
        # (exactly what the kernel's load_weights folded in); dW13 and dX are
        # rotated back through Q^T at the end.
        tq = layer.qcfg.turboquant
        wc = layer._bwd_wcache
        if wc is None:
            with _phase("weight-quant"):
                w13 = layer.w13.detach().to(torch.bfloat16)
                if tq:
                    w13 = rotate_trailing(w13, layer.q_rot)
                w2 = layer.w2.detach().to(torch.bfloat16)
                wc = layer._bwd_wcache = {
                    "w13_kh": quant_weights_3d(w13),                                # (E,2I,H) K=H
                    "w2_ki": quant_weights_3d(w2),                                  # (E,H,I)  K=I
                    "w2t_kh": quant_weights_3d(w2.transpose(1, 2).contiguous()),    # (E,I,H)  K=H
                    "w13t_k2i": quant_weights_3d(w13.transpose(1, 2).contiguous()),  # (E,H,2I) K=2I
                }
        w13q_kh, sw13_kh = wc["w13_kh"]
        w2q_ki, sw2_ki = wc["w2_ki"]
        w2tq_kh, sw2t_kh = wc["w2t_kh"]
        w13tq_k2i, sw13t_k2i = wc["w13t_k2i"]

        # recompute fc1 / act / y (fp8 grouped)
        FC1 = gmm_2d3d(XG, pad.offs, w13q_kh, sw13_kh)              # (Mp, 2I)
        with _phase("elemwise"):
            lin, gate = FC1.chunk(2, dim=-1)
            ACT = (F.silu(gate.float()) * lin.float()).to(torch.bfloat16)
        Y = gmm_2d3d(ACT, pad.offs, w2q_ki, sw2_ki)                 # (Mp, H)

        with _phase("comm-adj"):
            # d(topk_weights): per-copy <dout, y> on owner side, scalars back
            dtw_g = (Y.float() * DOUTG.float()).sum(dim=-1, keepdim=True)
            dtw_recv = pad.unpack(dtw_g)[plan.inv_recv_sort_idx]
            dtw_sorted = comm.all_to_all_no_grad(
                dtw_recv, plan.input_splits, plan.output_splits
            )
            dtw = dtw_sorted[plan.inv_sort_idx].view(T, K).float()

            # combine adjoint: dY to owners
            dy_flat = (
                dout.float().unsqueeze(1) * topk_weights.float().unsqueeze(-1)
            ).reshape(T * K, H).to(torch.bfloat16)
            dy_recv = comm.all_to_all_no_grad(
                dy_flat[plan.sort_idx], plan.output_splits, plan.input_splits
            )
            DYG = pad.pack(dy_recv[plan.recv_sort_idx])

        # dgrad chain
        DACT = gmm_2d3d(DYG, pad.offs, w2tq_kh, sw2t_kh)            # (Mp, I)
        with _phase("elemwise"):
            g32, l32 = gate.float(), lin.float()
            sg = torch.sigmoid(g32)
            dgate = DACT.float() * l32 * sg * (1 + g32 * (1 - sg))
            dlin = DACT.float() * F.silu(g32)
            DFC1 = torch.cat([dlin, dgate], dim=-1).to(torch.bfloat16)  # (Mp, 2I)
        DXG = gmm_2d3d(DFC1, pad.offs, w13tq_k2i, sw13t_k2i)        # (Mp, H)

        # wgrads (tokens on K); dW13 lands in the rotated basis under tq
        dw13 = gmm_wgrad_2d2d(DFC1, XG, pad.offs, order_2i, order_h)
        if tq:
            dw13 = rotate_trailing(dw13.float(), layer.q_rot.t())
        dw13 = dw13.to(layer.w13.dtype)
        dw2 = gmm_wgrad_2d2d(DYG, ACT, pad.offs, order_h, order_i).to(layer.w2.dtype)

        with _phase("comm-adj"):
            # dispatch adjoint: dX back to sources, sum over top-k copies
            dx_sorted = comm.all_to_all_no_grad(
                pad.unpack(DXG)[plan.inv_recv_sort_idx],
                plan.input_splits, plan.output_splits,
            )
            dx = torch.zeros((T, H), device=dout.device, dtype=torch.float32)
            dx.index_add_(0, plan.copy_token_idx, dx_sorted.float())
            if tq:
                dx = rotate_trailing(dx, layer.q_rot.t())

    return dx.to(torch.bfloat16), dtw, dw13, dw2
