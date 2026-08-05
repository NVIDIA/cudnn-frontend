# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Hybrid training layer validation: kernel fprop + torch bprop, single GPU.

Launch:

    MEGA_NO_DIST=1 python -m megamoe.tests.test_hybrid_training

Checks:

1. **Pool conformance** — dequant(my_activation/_sf) must reproduce
   pt's ``fake_quant_mxfp8(x)``: the device DataPreprocess quant and the
   host/pt quantizer must agree, or the backward replay consumes different
   bytes than the kernel multiplied.
2. **Forward numerics** — hybrid (kernel) output vs the pt simulated-mxfp8
   reference and vs the exact fp32 oracle.
3. **Backward numerics** — hybrid grads (dX, dTW, dW13, dW2) vs
   ``ReferenceMoEFp4(mxfp8)`` grads. With check 1 holding, the replay sees
   the same quantized operands, so these should agree tightly.
4. **Generation hazard** — two forwards then backward on the first must
   raise (the pools were overwritten).
5. **Train-step smoke** — Adam step + refresh_weights + another fwd/bwd.
"""

import os
import sys

os.environ.setdefault("MEGA_NO_DIST", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.distributed as dist

import megamoe.repo_path  # noqa: F401

from megamoe.forward import MegaMoeForwardConfig
from megamoe.training import MegaMoeHybridMxfp8Layer, dequant_mxfp8_pool

from pt import EpConfig, QuantConfig, ReferenceMoE, ReferenceMoEFp4
from pt.quant import fake_quant_mxfp8

TOKENS = 128
HIDDEN = 1024
INTERMEDIATE = 512
NUM_EXPERTS = 32
TOPK = 4
SEED = 4242
OUTLIER_FRAC = 0.01
OUTLIER_SCALE = 30.0


def rel_err(a, b):
    denom = b.float().norm().item()
    return (a.float() - b.float()).norm().item() / max(denom, 1e-30)


def gen_problem(device):
    gen = torch.Generator(device=device).manual_seed(SEED)
    x = torch.randn((TOKENS, HIDDEN), device=device, generator=gen) / 10.0
    hot = torch.randperm(HIDDEN, generator=gen, device=device)[
        : max(1, int(HIDDEN * OUTLIER_FRAC))
    ]
    x[:, hot] *= OUTLIER_SCALE
    scores = torch.rand((TOKENS, NUM_EXPERTS), device=device, generator=gen)
    _, ids = scores.topk(TOPK, dim=-1)
    w = torch.rand((TOKENS, TOPK), device=device, generator=gen) + 0.1
    tw = (w / w.sum(-1, keepdim=True)).float()
    w13 = (torch.randn((NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN),
                       device=device, generator=gen) * 0.05).bfloat16()
    w2 = (torch.randn((NUM_EXPERTS, HIDDEN, INTERMEDIATE),
                      device=device, generator=gen) * 0.05).bfloat16()
    gout = (torch.randn((TOKENS, HIDDEN), device=device, generator=gen)).bfloat16()
    return x.bfloat16(), ids.long(), tw, w13, w2, gout


def run_ref(cls, x, ids, tw, gout, w13, w2, **kw):
    x_r = x.detach().clone().requires_grad_()
    tw_r = tw.detach().clone().requires_grad_()
    m = cls(w13.detach().clone(), w2.detach().clone(), **kw)
    out = m(x_r, ids, tw_r)
    out.backward(gout)
    return out.detach(), x_r.grad, tw_r.grad, m.w13.grad, m.w2.grad


def main():
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    if not dist.is_initialized():
        dist.init_process_group(
            "nccl", init_method="tcp://127.0.0.1:29517", world_size=1, rank=0
        )

    x, ids, tw, w13, w2, gout = gen_problem(device)
    qcfg = QuantConfig(fprop_fmt="mxfp8", quant_bprop=True)

    ep_cfg = EpConfig(
        num_experts=NUM_EXPERTS, top_k=TOPK, hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE, ep_size=1, ep_rank=0,
    )
    mm_cfg = MegaMoeForwardConfig(
        max_tokens_per_rank=TOKENS, hidden=HIDDEN, intermediate=INTERMEDIATE,
        num_total_experts=NUM_EXPERTS, num_topk=TOPK,
    )
    hybrid = MegaMoeHybridMxfp8Layer(
        ep_cfg, mm_cfg, w13.detach().clone(), w2.detach().clone(), qcfg
    )

    # ---- fwd + bwd through the hybrid ----
    x_h = x.detach().clone().requires_grad_()
    tw_h = tw.detach().clone().requires_grad_()
    out_h = hybrid(x_h, ids, tw_h)
    out_h.backward(gout)

    # ---- [1] pool conformance: device preproc quant vs pt quantizer ----
    pool = dequant_mxfp8_pool(
        hybrid._fwd.my_activation[:TOKENS],
        hybrid._fwd.my_activation_sf[:TOKENS],
        HIDDEN,
    )
    host = fake_quant_mxfp8(x.float(), dim=-1)
    n_diff = (pool != host).sum().item()
    print(f"[1] pool vs pt quantizer: {n_diff}/{pool.numel()} mismatched")
    assert n_diff <= pool.numel() * 1e-4, "DataPreprocess and pt quantizer diverge"

    # ---- [2] forward numerics ----
    ref_fp4 = run_ref(ReferenceMoEFp4, x, ids, tw, gout, w13, w2, qcfg=qcfg)
    exact = run_ref(
        ReferenceMoE, x.float(), ids, tw, gout.float(), w13.float(), w2.float()
    )
    e_vs_sim = rel_err(out_h, ref_fp4[0])
    e_vs_fp32 = rel_err(out_h, exact[0])
    e_sim_fp32 = rel_err(ref_fp4[0], exact[0])
    print(f"[2] fwd rel_err: hybrid-vs-sim={e_vs_sim:.4f}  "
          f"hybrid-vs-fp32={e_vs_fp32:.4f}  (sim-vs-fp32={e_sim_fp32:.4f})")
    assert e_vs_fp32 < 0.15, "kernel fwd far from fp32 oracle"
    assert e_vs_sim < 0.10, "kernel fwd far from pt simulation"

    # ---- [3] backward numerics vs the pt simulated reference ----
    names = ("dX", "dTW", "dW13", "dW2")
    got = (x_h.grad, tw_h.grad, hybrid.w13.grad, hybrid.w2.grad)
    errs = {n: rel_err(g, r) for n, g, r in zip(names, got, ref_fp4[1:])}
    print("[3] bwd rel_err vs ReferenceMoEFp4: "
          + "  ".join(f"{n}={e:.4f}" for n, e in errs.items()))
    for n, e in errs.items():
        assert e < 0.05, (n, e, "hybrid replay grads diverge from simulation")

    # ---- [3a] vectorized scale swizzle == per-group to_blocked loop ----
    from megamoe.fp8_bwd import (
        _col_atom_order,
        _pack_scales_colgroups,
        _pack_scales_colgroups_ref,
        _pack_scales_rowgroups,
        _pack_scales_rowgroups_ref,
    )
    gen = torch.Generator(device=device).manual_seed(SEED + 3)
    tpe = [256, 0, 768, 1024]  # 128-aligned token groups, one empty
    offs = torch.tensor([256, 256, 1024, 2048], device=device, dtype=torch.int32)
    s_col = torch.randint(
        0, 255, (1024, 2048 // 32), device=device, dtype=torch.uint8, generator=gen
    ).view(torch.float8_e8m0fnu)
    order = _col_atom_order(offs, 1024 // 128, 2048 // 128)
    v = _pack_scales_colgroups(s_col, offs, 1024, order)
    r = _pack_scales_colgroups_ref(s_col, offs, 1024)
    assert torch.equal(v.view(torch.uint8), r.view(torch.uint8)), "colgroup swizzle"
    s_row = torch.randint(
        0, 255, (2048, 1024 // 32), device=device, dtype=torch.uint8, generator=gen
    ).view(torch.float8_e8m0fnu)
    v2 = _pack_scales_rowgroups(s_row, offs)
    r2 = _pack_scales_rowgroups_ref(s_row, offs)
    assert torch.equal(v2.view(torch.uint8), r2.view(torch.uint8)), "rowgroup swizzle"
    print("[3a] vectorized swizzle == per-group to_blocked (col + row groups)")

    # Triton quantizers must be bit-identical to the torch reference.
    from megamoe.quant_kernels import HAVE_TRITON, mxfp8_rowquant, mxfp8_transquant
    from pt.quant import quant_mxfp8_tensors
    if HAVE_TRITON:
        t = torch.randn((640, 1024), device=device, generator=gen).bfloat16()
        t[:, :4] *= 30.0
        q_t, s_t = mxfp8_rowquant(t)
        q_r, s_r = quant_mxfp8_tensors(t, dim=-1)
        assert torch.equal(q_t.view(torch.uint8), q_r.view(torch.uint8)), "rowquant data"
        assert torch.equal(s_t.view(torch.uint8), s_r.view(torch.uint8)), "rowquant scales"
        q_t2, s_t2 = mxfp8_transquant(t)
        q_r2, s_r2 = quant_mxfp8_tensors(t.t().contiguous(), dim=-1)
        assert torch.equal(q_t2.view(torch.uint8), q_r2.view(torch.uint8)), "transquant data"
        assert torch.equal(s_t2.view(torch.uint8), s_r2.view(torch.uint8)), "transquant scales"
        print("[3a'] triton rowquant/transquant bit-identical to torch reference")
    else:
        print("[3a'] triton unavailable — torch fallback in use")

    # ---- [3b] fp8 manual backward vs the replay ----
    hybrid8 = MegaMoeHybridMxfp8Layer(
        ep_cfg, mm_cfg, w13.detach().clone(), w2.detach().clone(), qcfg,
        bwd_impl="fp8",
    )
    x_8 = x.detach().clone().requires_grad_()
    tw_8 = tw.detach().clone().requires_grad_()
    out_8 = hybrid8(x_8, ids, tw_8)
    out_8.backward(gout)
    got8 = (x_8.grad, tw_8.grad, hybrid8.w13.grad, hybrid8.w2.grad)
    errs8 = {n: rel_err(g, r) for n, g, r in zip(names, got8, got)}
    print("[3b] fp8-bwd rel_err vs replay-bwd: "
          + "  ".join(f"{n}={e:.4f}" for n, e in errs8.items()))
    for n, e in errs8.items():
        assert e < 0.05, (n, e, "fp8 manual backward diverges from replay")
    hybrid8.finalize()

    # ---- [4] generation hazard ----
    x_a = x.detach().clone().requires_grad_()
    out_a = hybrid(x_a, ids, tw.detach().clone().requires_grad_())
    _ = hybrid(x.detach().clone(), ids, tw.detach().clone())
    try:
        out_a.backward(gout)
        raise AssertionError("stale-generation backward did not raise")
    except RuntimeError as e:
        assert "generation" in str(e)
        print("[4] generation hazard raises as expected")

    # ---- [4b] turboquant hybrid: fwd + grads vs the tq reference ----
    qcfg_tq = QuantConfig(fprop_fmt="mxfp8", quant_bprop=True, turboquant=True)
    ref_tq = run_ref(ReferenceMoEFp4, x, ids, tw, gout, w13, w2, qcfg=qcfg_tq)
    for impl, tol in (("replay", 0.02), ("fp8", 0.05)):
        h_tq = MegaMoeHybridMxfp8Layer(
            ep_cfg, mm_cfg, w13.detach().clone(), w2.detach().clone(),
            qcfg_tq, bwd_impl=impl,
        )
        x_t = x.detach().clone().requires_grad_()
        tw_t = tw.detach().clone().requires_grad_()
        out_t = h_tq(x_t, ids, tw_t)
        out_t.backward(gout)
        e_fwd = rel_err(out_t, ref_tq[0])
        got_t = (x_t.grad, tw_t.grad, h_tq.w13.grad, h_tq.w2.grad)
        errs_t = {n: rel_err(g, r) for n, g, r in zip(names, got_t, ref_tq[1:])}
        print(f"[4b] tq {impl:<6}: fwd={e_fwd:.4f}  "
              + "  ".join(f"{n}={e:.4f}" for n, e in errs_t.items()))
        assert e_fwd < 0.10, ("tq fwd", impl, e_fwd)
        for n, e in errs_t.items():
            assert e < tol, ("tq bwd", impl, n, e)
        h_tq.finalize()

    # ---- [5] train-step smoke: perturbed start, must descend toward the
    # fp32 teacher output (grads flow -> optimizer -> refresh_weights -> next
    # fwd sees updated kernel weight buffers) ----
    gen = torch.Generator(device=device).manual_seed(SEED + 7)
    student = MegaMoeHybridMxfp8Layer(
        ep_cfg, mm_cfg,
        (w13.float() * (1 + 0.2 * torch.randn(w13.shape, device=device, generator=gen))).bfloat16(),
        (w2.float() * (1 + 0.2 * torch.randn(w2.shape, device=device, generator=gen))).bfloat16(),
        qcfg,
    )
    opt = torch.optim.Adam(student.parameters(), lr=2e-3)
    losses = []
    for _ in range(20):
        x_s = x.detach().clone().requires_grad_()
        out = student(x_s, ids, tw.detach().clone())
        loss = torch.nn.functional.mse_loss(out.float(), exact[0].float())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        student.refresh_weights()
        losses.append(loss.item())
    print(f"[5] 20-step train smoke, loss {losses[0]:.5f} -> {losses[-1]:.5f}")
    assert losses[-1] < losses[0] * 0.5, ("training did not descend", losses)
    student.finalize()

    hybrid.finalize()
    dist.destroy_process_group()
    print("HYBRID TRAINING PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
