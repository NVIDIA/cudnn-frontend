# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""NVFP4(+TurboQuant) training numerics for the pt path — the accuracy
experiment behind "would fp4 training work, and does rotation help?"

Launch (single GPU, no dist):

    python -m pt.tests.test_fp4_qat_numerics

Part 1 — one-step accuracy: against an fp32 oracle (plain ReferenceMoE in
fp32), measure relative error of the forward output AND every training
gradient (dX, d topk_weights, dW13, dW2) for:

    bf16            — the current pt path (precision floor)
    mxfp8[+tq/+qb]  — 8-bit GEMMs; the regime where turboquant helps
    nvfp4[+tq/+qb/+sr] — 4-bit GEMMs; tq expected ~neutral
    nvfp4/8bw[+tq]  — the mixed recipe: nvfp4 fprop, mxfp8 quantized bprop

on activations with realistic per-channel outliers (1% hot channels x30 —
the case incoherence processing exists for).

Part 2 — convergence: short Adam distillation (student regresses a frozen
fp32 teacher's outputs from perturbed weights) comparing loss trajectories
across the same variants.

Expectations encoded in the asserts (from megamoe's forward measurement,
test_turboquant_numerics.py): nvfp4's per-16 fp8 scales already absorb
outliers, so turboquant is neutral-to-slightly-negative here — the table
quantifies it for GRADS, which is the new information.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from pt import QuantConfig, ReferenceMoE, ReferenceMoEFp4  # noqa: E402

TOKENS = 256
HIDDEN = 1024
INTERMEDIATE = 512
NUM_EXPERTS = 32
TOPK = 4
SEED = 4242
OUTLIER_FRAC = 0.01
OUTLIER_SCALE = 30.0

TRAIN_STEPS = 200
TRAIN_BATCHES = 8
TRAIN_TOKENS = 128
LR = 2e-3
WEIGHT_NOISE = 0.3  # relative perturbation of the student's start point


def gen_problem(device):
    gen = torch.Generator(device=device).manual_seed(SEED)
    x = torch.randn((TOKENS, HIDDEN), device=device, generator=gen) / 10.0
    n_out = max(1, int(HIDDEN * OUTLIER_FRAC))
    hot = torch.randperm(HIDDEN, generator=gen, device=device)[:n_out]
    x[:, hot] *= OUTLIER_SCALE

    scores = torch.rand((TOKENS, NUM_EXPERTS), device=device, generator=gen)
    _, topk_ids = scores.topk(TOPK, dim=-1)
    w = torch.rand((TOKENS, TOPK), device=device, generator=gen) + 0.1
    topk_w = (w / w.sum(-1, keepdim=True)).float()

    w13 = torch.randn((NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN), device=device, generator=gen) * 0.05
    w2 = torch.randn((NUM_EXPERTS, HIDDEN, INTERMEDIATE), device=device, generator=gen) * 0.05
    gout = torch.randn((TOKENS, HIDDEN), device=device, generator=gen)
    return x, topk_ids.long(), topk_w, w13, w2, gout


def rel_err(a, b):
    denom = b.float().norm().item()
    return (a.float() - b.float()).norm().item() / max(denom, 1e-30)


def run_variant(ctor, x, ids, tw, gout):
    """Forward+backward one step; return (out, dX, dTW, dW13, dW2)."""
    torch.manual_seed(SEED)  # fixes SR draws so variants are reproducible
    x_v = x.detach().clone().requires_grad_()
    tw_v = tw.detach().clone().requires_grad_()
    m = ctor()
    out = m(x_v, ids, tw_v)
    out.backward(gout)
    return out.detach(), x_v.grad, tw_v.grad, m.w13.grad, m.w2.grad


def make_variants(w13b, w2b):
    def fp4(**kw):
        return lambda: ReferenceMoEFp4(
            w13b.detach().clone(), w2b.detach().clone(), QuantConfig(**kw)
        )

    return {
        "bf16": lambda: ReferenceMoE(w13b.detach().clone(), w2b.detach().clone()),
        "mxfp8": fp4(fprop_fmt="mxfp8"),
        "mxfp8+tq": fp4(fprop_fmt="mxfp8", turboquant=True),
        "mxfp8+qb": fp4(fprop_fmt="mxfp8", quant_bprop=True),
        "mxfp8+tq+qb": fp4(fprop_fmt="mxfp8", turboquant=True, quant_bprop=True),
        "nvfp4": fp4(),
        "nvfp4+tq": fp4(turboquant=True),
        "nvfp4+qb": fp4(quant_bprop=True),
        "nvfp4+tq+qb": fp4(turboquant=True, quant_bprop=True),
        "nvfp4+qb+sr": fp4(quant_bprop=True, stochastic_rounding_grads=True),
        "nvfp4/8bw": fp4(bprop_fmt="mxfp8", quant_bprop=True),
        "nvfp4/8bw+tq": fp4(bprop_fmt="mxfp8", quant_bprop=True, turboquant=True),
    }


def part1_accuracy(device):
    x, ids, tw, w13, w2, gout = gen_problem(device)

    # fp32 oracle
    ref = run_variant(
        lambda: ReferenceMoE(w13.detach().clone(), w2.detach().clone()),
        x, ids, tw, gout,
    )

    x_b, w13_b, w2_b, gout_b = x.bfloat16(), w13.bfloat16(), w2.bfloat16(), gout.bfloat16()
    variants = make_variants(w13_b, w2_b)

    names = ("fwd", "dX", "dTW", "dW13", "dW2")
    results = {}
    print(f"\n== Part 1: one-step rel_err vs fp32 oracle "
          f"(T={TOKENS} H={HIDDEN} I={INTERMEDIATE} E={NUM_EXPERTS} K={TOPK}, "
          f"{OUTLIER_FRAC:.0%} channels x{OUTLIER_SCALE:.0f}) ==")
    print(f"{'variant':<14}" + "".join(f"{n:>12}" for n in names))
    for name, ctor in variants.items():
        got = run_variant(ctor, x_b, ids, tw, gout_b)
        errs = tuple(rel_err(g, r) for g, r in zip(got, ref))
        results[name] = dict(zip(names, errs))
        print(f"{name:<14}" + "".join(f"{e:>12.4f}" for e in errs))

    r = results
    # Sanity: bf16 floor is tight; everything stays in a usable regime.
    assert r["bf16"]["fwd"] < 0.02, r["bf16"]
    for name, e in r.items():
        assert e["fwd"] < 0.35, (name, e)
        assert max(e["dX"], e["dW13"], e["dW2"]) < 0.60, (name, e)
    # mxfp8 is the low-loss regime (~0.07 fwd rel_err under these 1%x30
    # outliers), and the one where turboquant must not hurt (megamoe fwd
    # measurement: rotation recovers outlier damage).
    assert r["mxfp8"]["fwd"] < 0.10, r["mxfp8"]
    assert r["mxfp8+tq"]["fwd"] <= r["mxfp8"]["fwd"] * 1.05, (r["mxfp8"], r["mxfp8+tq"])
    # Known result (megamoe fwd measurement): turboquant is ~neutral on
    # nvfp4. Bound the regression loosely.
    assert r["nvfp4+tq"]["fwd"] <= r["nvfp4"]["fwd"] * 1.25, (r["nvfp4"], r["nvfp4+tq"])
    # The mixed recipe's grads must sit at-or-below full-fp4 bprop grads.
    assert r["nvfp4/8bw"]["dW13"] <= r["nvfp4+qb"]["dW13"] * 1.05, (
        r["nvfp4/8bw"], r["nvfp4+qb"]
    )
    return results


def part2_convergence(device):
    gen = torch.Generator(device=device).manual_seed(SEED + 1)
    w13 = torch.randn((NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN), device=device, generator=gen) * 0.05
    w2 = torch.randn((NUM_EXPERTS, HIDDEN, INTERMEDIATE), device=device, generator=gen) * 0.05
    teacher = ReferenceMoE(w13.detach().clone(), w2.detach().clone())

    n_hot = max(1, int(HIDDEN * OUTLIER_FRAC))
    batches = []
    for _ in range(TRAIN_BATCHES):
        x = torch.randn((TRAIN_TOKENS, HIDDEN), device=device, generator=gen) / 10.0
        hot = torch.randperm(HIDDEN, generator=gen, device=device)[:n_hot]
        x[:, hot] *= OUTLIER_SCALE
        scores = torch.rand((TRAIN_TOKENS, NUM_EXPERTS), device=device, generator=gen)
        _, ids = scores.topk(TOPK, dim=-1)
        w = torch.rand((TRAIN_TOKENS, TOPK), device=device, generator=gen) + 0.1
        tw = (w / w.sum(-1, keepdim=True)).float()
        with torch.no_grad():
            y = teacher(x, ids.long(), tw)
        batches.append((x.bfloat16(), ids.long(), tw, y))

    # Perturbed student start, shared by every variant.
    noise13 = torch.randn(w13.shape, device=device, generator=gen)
    noise2 = torch.randn(w2.shape, device=device, generator=gen)
    s13 = (w13 * (1 + WEIGHT_NOISE * noise13)).bfloat16()
    s2 = (w2 * (1 + WEIGHT_NOISE * noise2)).bfloat16()

    variants = make_variants(s13, s2)
    marks = [0, TRAIN_STEPS // 4, TRAIN_STEPS // 2, 3 * TRAIN_STEPS // 4, TRAIN_STEPS - 1]

    print(f"\n== Part 2: {TRAIN_STEPS}-step Adam distillation vs fp32 teacher "
          f"(MSE, lr={LR}, {WEIGHT_NOISE:.0%} weight perturbation) ==")
    print(f"{'variant':<14}" + "".join(f"{('step ' + str(s)):>12}" for s in marks))
    finals = {}
    for name, ctor in variants.items():
        torch.manual_seed(SEED)  # same SR/optimizer determinism per variant
        m = ctor()
        opt = torch.optim.Adam(m.parameters(), lr=LR)
        curve = {}
        for step in range(TRAIN_STEPS):
            x, ids, tw, y = batches[step % TRAIN_BATCHES]
            out = m(x, ids, tw)
            loss = torch.nn.functional.mse_loss(out.float(), y.float())
            if step in marks:
                curve[step] = loss.item()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        finals[name] = curve
        print(f"{name:<14}" + "".join(f"{curve[s]:>12.5f}" for s in marks))

    # Every variant must actually train (loss falls substantially).
    for name, curve in finals.items():
        assert curve[marks[-1]] < curve[0] * 0.5, (name, curve)
    return finals


def main():
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    part1_accuracy(device)
    part2_convergence(device)
    print("\nFP4 QAT NUMERICS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
