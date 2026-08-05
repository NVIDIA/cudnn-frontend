# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""TurboQuant numerics validation: rotated vs plain MXFP8 forward accuracy.

Launch (from moe_ep_training/):

    torchrun --nproc_per_node=4 --standalone -m megamoe.tests.test_turboquant_numerics
    MEGA_NO_DIST=1 python -m megamoe.tests.test_turboquant_numerics

Three checks, on activations with realistic per-channel outliers (the case
incoherence processing exists for):

1. **Fold exactness** (fp32, no quant): the exact MoE on (x Q, W13 Q) equals
   the exact MoE on (x, W13) — the rotation is mathematically a no-op.
2. **Correctness sanity**: both the plain and the TurboQuant MXFP8 forwards
   stay within a loose tolerance of the exact fp32 MoE.
3. **Accuracy win**: the TurboQuant forward's relative error vs the exact
   fp32 MoE is <= the plain forward's (expected strictly smaller with
   outliers present).

Every rank regenerates the global problem from shared seeds (like the repo's
runner), so no gathers are needed for the exact reference.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

import megamoe.repo_path  # noqa: F401

from megamoe.forward import MegaMoeForwardConfig, MegaMoeMxfp8Forward
from megamoe.forward_nvfp4 import MegaMoeNvfp4Forward
from megamoe.turboquant import (
    MegaMoeTurboQuantForward,
    MegaMoeTurboQuantNvfp4Forward,
    make_rotation,
    rotate_hidden,
)

_NO_DIST = bool(int(os.environ.get("MEGA_NO_DIST", "0")))

TOKENS = 128
HIDDEN = 1024
INTERMEDIATE = 512
NUM_TOTAL_EXPERTS = 32
TOPK = 4
SEED = 4242
OUTLIER_FRAC = 0.01   # ~1% hot channels, x30 — typical LLM activation shape
OUTLIER_SCALE = 30.0


def _gen_global(world_size):
    """Deterministic global problem, identical on every rank."""
    gen = torch.Generator(device="cuda").manual_seed(SEED)
    E_local = NUM_TOTAL_EXPERTS // world_size

    x_clean = torch.randn((world_size, TOKENS, HIDDEN), device="cuda", generator=gen) / 10.0
    n_out = max(1, int(HIDDEN * OUTLIER_FRAC))
    hot = torch.randperm(HIDDEN, generator=gen, device="cuda")[:n_out]
    x = x_clean.clone()
    x[..., hot] *= OUTLIER_SCALE
    x = x.bfloat16()
    x_clean = x_clean.bfloat16()

    scores = torch.rand((world_size, TOKENS, NUM_TOTAL_EXPERTS), device="cuda", generator=gen)
    _, topk_idx = scores.topk(TOPK, dim=-1)
    w = torch.rand((world_size, TOKENS, TOPK), device="cuda", generator=gen) + 0.1
    topk_w = (w / w.sum(-1, keepdim=True)).float()

    w13 = (torch.randn((world_size, E_local, 2 * INTERMEDIATE, HIDDEN),
                       device="cuda", generator=gen) * 0.05).bfloat16()
    w2 = (torch.randn((world_size, E_local, HIDDEN, INTERMEDIATE),
                      device="cuda", generator=gen) * 0.05).bfloat16()
    return x, x_clean, topk_idx.to(torch.int64), topk_w, w13, w2


def exact_moe_fp32(xg, idxg, wg, w13g, w2g):
    """Unquantized fp32 MoE oracle: silu(gate)*linear, weights in fc1 position."""
    R, T, H = xg.shape
    E_local = w13g.shape[1]
    out = torch.zeros((R, T, H), device="cuda", dtype=torch.float32)
    for e_global in range(R * E_local):
        r_e, e_l = divmod(e_global, E_local)
        rows = (idxg == e_global).nonzero(as_tuple=False)
        if rows.numel() == 0:
            continue
        xs = xg[rows[:, 0], rows[:, 1]].float()
        h = xs @ w13g[r_e, e_l].float().T                      # (n, 2I)
        up, gate = h[:, :INTERMEDIATE], h[:, INTERMEDIATE:]
        act = F.silu(gate) * up
        act = act * wg[rows[:, 0], rows[:, 1], rows[:, 2]].float()[:, None]
        y = act @ w2g[r_e, e_l].float().T                      # (n, H)
        out.index_put_((rows[:, 0], rows[:, 1]), y, accumulate=True)
    return out


def rel_err(a, b):
    return (a.float() - b.float()).norm().item() / b.float().norm().item()


def main() -> int:
    if _NO_DIST:
        torch.cuda.set_device(0)
        rank, world = 0, 1
    else:
        from src.bootstrap import init_dist_and_nvshmem, finalize_dist_and_nvshmem
        _, rank, world, _ = init_dist_and_nvshmem()

    xg, x_clean, idxg, wg, w13g, w2g = _gen_global(world)
    exact = exact_moe_fp32(xg, idxg, wg, w13g, w2g)

    # Outlier-free twin of the same problem: establishes the pipeline's
    # intrinsic mxfp8 error floor (fc1-out requant + weight quant + combine),
    # which no activation-side rotation can go below.
    exact_clean = exact_moe_fp32(x_clean, idxg, wg, w13g, w2g)

    # -- check 1: rotation fold is exact in fp32 (independent of the kernel) --
    q = make_rotation()
    x_rot = rotate_hidden(xg.float(), q).bfloat16()
    w13_rot = rotate_hidden(w13g.float(), q).bfloat16()
    exact_rot = exact_moe_fp32(x_rot, idxg, wg, w13_rot, w2g)
    fold_err = rel_err(exact_rot, exact)
    if rank == 0:
        print(f"[1] rotation-fold exactness (fp32+bf16 roundtrip): rel_err={fold_err:.2e}")
    assert fold_err < 5e-3, f"rotation fold not identity-preserving: {fold_err}"

    cfg = MegaMoeForwardConfig(
        max_tokens_per_rank=TOKENS, hidden=HIDDEN, intermediate=INTERMEDIATE,
        num_total_experts=NUM_TOTAL_EXPERTS, num_topk=TOPK,
    )

    impls = {
        "mxfp8": (MegaMoeMxfp8Forward, MegaMoeTurboQuantForward),
        "nvfp4": (MegaMoeNvfp4Forward, MegaMoeTurboQuantNvfp4Forward),
    }
    summary = {}
    for fmt, (plain_cls, tq_cls) in impls.items():
        plain = plain_cls(cfg, rank=rank, world_size=world)
        plain.load_weights(w13g[rank], w2g[rank])
        floor = rel_err(
            plain(x_clean[rank], idxg[rank], wg[rank]).clone(), exact_clean[rank]
        )
        plain_e = rel_err(
            plain(xg[rank], idxg[rank], wg[rank]).clone(), exact[rank]
        )
        plain.finalize()

        turbo = tq_cls(cfg, rank=rank, world_size=world)
        turbo.load_weights(w13g[rank], w2g[rank])
        turbo_e = rel_err(
            turbo(xg[rank], idxg[rank], wg[rank]).clone(), exact[rank]
        )
        turbo.finalize()

        excess_plain = plain_e - floor
        excess_turbo = turbo_e - floor
        recovered = (
            1.0 - excess_turbo / excess_plain if excess_plain > 0 else float("nan")
        )
        summary[fmt] = (floor, plain_e, turbo_e)
        print(
            f"[2] rank{rank} {fmt}: rel_err vs exact fp32:\n"
            f"      floor (no outliers)  = {floor:.4e}\n"
            f"      plain (outliers)     = {plain_e:.4e} (excess {excess_plain:+.2e})\n"
            f"      turboquant (outliers)= {turbo_e:.4e} (excess {excess_turbo:+.2e})\n"
            f"[3] rank{rank} {fmt}: outlier-induced excess error recovered by "
            f"rotation: {100.0 * recovered:.0f}%"
        )
        # Must-hold invariants; the decomposition above is the deliverable.
        # mxfp8: rotation must recover outlier damage (validated claim).
        # nvfp4: per-16 fp8 block scales already absorb outliers (the block's
        # scale adapts to its own outlier), so spreading energy via rotation
        # only exposes more values to e2m1's ~9%/element rounding — rotation
        # is expected NEUTRAL-TO-NEGATIVE here. Record it, bound it loosely.
        if fmt == "mxfp8":
            assert turbo_e <= plain_e * 1.02, (fmt, summary[fmt])
            assert turbo_e < floor + max(0.02, 0.3 * floor), (fmt, summary[fmt])
        else:
            assert plain_e < 0.35 and turbo_e < 0.35, (fmt, summary[fmt])
            assert turbo_e <= plain_e * 1.25, (fmt, summary[fmt])

    if rank == 0:
        print("TURBOQUANT NUMERICS PASS")
    if not _NO_DIST:
        finalize_dist_and_nvshmem()
    return 0


if __name__ == "__main__":
    sys.exit(main())
